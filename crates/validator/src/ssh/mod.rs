//! SSH Connection Management for Validator
//!
//! This module provides SSH connection functionality specifically for
//! validator operations, built on top of the common SSH infrastructure.
//! Implements SCP file transfer, remote command execution with timeout handling,
//! result file download, cleanup, SSH session management, and authentication retries.
//!
//! ## Security Architecture
//!
//! The SSH-based validation approach is fundamental to Basilica's security model:
//!
//! ### Why SSH Instead of gRPC?
//!
//! 1. **Binary Control**: Validators upload and execute their own attestation binaries,
//!    preventing executors from running modified versions that could fake results.
//!
//! 2. **Direct Execution**: Commands run directly on the hardware without intermediary
//!    services that could intercept or modify results.
//!
//! 3. **Atomic Operations**: The entire attestation process happens in a single remote
//!    execution, reducing attack surface compared to multi-round protocols.
//!
//! 4. **Audit Trail**: All SSH operations are logged with the "ssh_audit" target,
//!    providing forensic capabilities for security analysis.
//!
//! ### Security Features
//!
//! - **Key-based Authentication**: No password authentication allowed
//! - **Ephemeral Execution**: Binaries and files are cleaned up after use
//! - **Connection Pooling**: Reuses connections while maintaining security
//! - **Comprehensive Logging**: All operations logged for audit purposes
//! - **Retry Logic**: Handles transient failures without compromising security

#[cfg(test)]
mod tests;

use anyhow::Result;
use common::identity::ExecutorId;
use common::ssh::{
    SshConnectionConfig, SshConnectionDetails, SshConnectionManager, SshFileTransferManager,
    StandardSshClient,
};
use std::collections::HashMap;
use std::path::Path;
use std::sync::{Arc, Mutex};
use std::time::{Duration, Instant};
use tracing::{debug, error, info, warn};

/// SSH connection pool entry
#[derive(Debug, Clone)]
struct ConnectionPoolEntry {
    #[allow(dead_code)]
    details: SshConnectionDetails,
    #[allow(dead_code)]
    last_used: Instant,
    #[allow(dead_code)]
    connection_count: u32,
    #[allow(dead_code)]
    success_count: u32,
    #[allow(dead_code)]
    failure_count: u32,
}

/// SSH session statistics
#[derive(Debug, Clone)]
pub struct SshSessionStats {
    pub total_connections: u32,
    pub successful_connections: u32,
    pub failed_connections: u32,
    pub total_transfers: u32,
    pub successful_transfers: u32,
    pub failed_transfers: u32,
    pub total_commands: u32,
    pub successful_commands: u32,
    pub failed_commands: u32,
    pub average_response_time_ms: f64,
}

/// SSH client wrapper for validator operations with enhanced functionality
pub struct ValidatorSshClient {
    client: StandardSshClient,
    connection_pool: Arc<Mutex<HashMap<String, ConnectionPoolEntry>>>,
    session_stats: Arc<Mutex<SshSessionStats>>,
    #[allow(dead_code)]
    max_pool_size: usize,
    #[allow(dead_code)]
    pool_timeout: Duration,
    retry_config: RetryConfig,
}

/// Retry configuration for SSH operations
#[derive(Debug, Clone)]
pub struct RetryConfig {
    pub max_attempts: u32,
    pub initial_delay: Duration,
    pub max_delay: Duration,
    pub backoff_multiplier: f64,
    pub retry_on_timeout: bool,
    pub retry_on_connection_error: bool,
}

impl Default for RetryConfig {
    fn default() -> Self {
        Self {
            max_attempts: 3,
            initial_delay: Duration::from_millis(500),
            max_delay: Duration::from_secs(30),
            backoff_multiplier: 2.0,
            retry_on_timeout: true,
            retry_on_connection_error: true,
        }
    }
}

impl Default for SshSessionStats {
    fn default() -> Self {
        Self {
            total_connections: 0,
            successful_connections: 0,
            failed_connections: 0,
            total_transfers: 0,
            successful_transfers: 0,
            failed_transfers: 0,
            total_commands: 0,
            successful_commands: 0,
            failed_commands: 0,
            average_response_time_ms: 0.0,
        }
    }
}

impl ValidatorSshClient {
    /// Create a new validator SSH client with default configuration
    pub fn new() -> Self {
        Self {
            client: StandardSshClient::new(),
            connection_pool: Arc::new(Mutex::new(HashMap::new())),
            session_stats: Arc::new(Mutex::new(SshSessionStats::default())),
            max_pool_size: 100,
            pool_timeout: Duration::from_secs(300), // 5 minutes
            retry_config: RetryConfig::default(),
        }
    }

    /// Create a new validator SSH client with custom configuration
    pub fn with_config(config: SshConnectionConfig) -> Self {
        Self {
            client: StandardSshClient::with_config(config),
            connection_pool: Arc::new(Mutex::new(HashMap::new())),
            session_stats: Arc::new(Mutex::new(SshSessionStats::default())),
            max_pool_size: 100,
            pool_timeout: Duration::from_secs(300),
            retry_config: RetryConfig::default(),
        }
    }

    /// Create a new validator SSH client with custom retry configuration
    pub fn with_retry_config(config: SshConnectionConfig, retry_config: RetryConfig) -> Self {
        Self {
            client: StandardSshClient::with_config(config),
            connection_pool: Arc::new(Mutex::new(HashMap::new())),
            session_stats: Arc::new(Mutex::new(SshSessionStats::default())),
            max_pool_size: 100,
            pool_timeout: Duration::from_secs(300),
            retry_config,
        }
    }

    /// Get connection pool key for caching
    #[allow(dead_code)]
    fn get_pool_key(&self, details: &SshConnectionDetails) -> String {
        format!("{}@{}:{}", details.username, details.host, details.port)
    }

    /// Add connection to pool or update existing entry
    #[allow(dead_code)]
    fn update_connection_pool(&self, details: &SshConnectionDetails, success: bool) {
        let key = self.get_pool_key(details);
        let mut pool = self.connection_pool.lock().unwrap();

        // Clean up expired entries if pool is getting large
        if pool.len() >= self.max_pool_size {
            let cutoff = Instant::now() - self.pool_timeout;
            pool.retain(|_, entry| entry.last_used > cutoff);
        }

        let entry = pool.entry(key).or_insert_with(|| ConnectionPoolEntry {
            details: details.clone(),
            last_used: Instant::now(),
            connection_count: 0,
            success_count: 0,
            failure_count: 0,
        });

        entry.last_used = Instant::now();
        entry.connection_count += 1;

        if success {
            entry.success_count += 1;
        } else {
            entry.failure_count += 1;
        }
    }

    /// Update session statistics
    #[allow(dead_code)]
    fn update_stats<F>(&self, operation: F)
    where
        F: FnOnce(&mut SshSessionStats),
    {
        if let Ok(mut stats) = self.session_stats.lock() {
            operation(&mut stats);
        }
    }

    /// Get current session statistics
    pub fn get_session_stats(&self) -> SshSessionStats {
        self.session_stats.lock().unwrap().clone()
    }

    /// Get connection pool information
    pub fn get_pool_info(&self) -> (usize, Vec<String>) {
        let pool = self.connection_pool.lock().unwrap();
        let keys: Vec<String> = pool.keys().cloned().collect();
        (pool.len(), keys)
    }

    /// Clear connection pool
    pub fn clear_pool(&self) {
        self.connection_pool.lock().unwrap().clear();
    }

    /// Test SSH connection to executor
    pub async fn test_connection(&self, details: &SshConnectionDetails) -> Result<()> {
        info!(
            "Testing SSH connection to {}@{}",
            details.username, details.host
        );
        self.client.test_connection(details).await
    }

    /// Execute command on executor with audit logging
    pub async fn execute_command(
        &self,
        details: &SshConnectionDetails,
        command: &str,
        capture_output: bool,
    ) -> Result<String> {
        let start_time = std::time::Instant::now();
        let command_summary = if command.len() > 100 {
            format!("{}...", &command[..100])
        } else {
            command.to_string()
        };

        info!(
            target: "ssh_audit",
            host = %details.host,
            username = %details.username,
            port = details.port,
            command = %command_summary,
            "SSH command execution started"
        );

        match self
            .client
            .execute_command(details, command, capture_output)
            .await
        {
            Ok(output) => {
                let duration = start_time.elapsed();
                info!(
                    target: "ssh_audit",
                    host = %details.host,
                    username = %details.username,
                    command = %command_summary,
                    duration_ms = duration.as_millis(),
                    output_bytes = output.len(),
                    "SSH command execution succeeded"
                );
                Ok(output)
            }
            Err(e) => {
                let duration = start_time.elapsed();
                error!(
                    target: "ssh_audit",
                    host = %details.host,
                    username = %details.username,
                    command = %command_summary,
                    duration_ms = duration.as_millis(),
                    error = %e,
                    "SSH command execution failed"
                );
                Err(e)
            }
        }
    }

    /// Execute command with retry logic
    pub async fn execute_command_with_retry(
        &self,
        details: &SshConnectionDetails,
        command: &str,
        capture_output: bool,
    ) -> Result<String> {
        self.client
            .execute_command_with_retry(details, command, capture_output)
            .await
    }

    /// Upload file to executor with audit logging
    pub async fn upload_file(
        &self,
        details: &SshConnectionDetails,
        local_path: &Path,
        remote_path: &str,
    ) -> Result<()> {
        let start_time = std::time::Instant::now();
        let file_size = std::fs::metadata(local_path).map(|m| m.len()).unwrap_or(0);

        info!(
            target: "ssh_audit",
            host = %details.host,
            username = %details.username,
            local_path = %local_path.display(),
            remote_path = %remote_path,
            file_size_bytes = file_size,
            "SSH file upload started"
        );

        match self
            .client
            .upload_file(details, local_path, remote_path)
            .await
        {
            Ok(()) => {
                let duration = start_time.elapsed();
                info!(
                    target: "ssh_audit",
                    host = %details.host,
                    username = %details.username,
                    local_path = %local_path.display(),
                    remote_path = %remote_path,
                    file_size_bytes = file_size,
                    duration_ms = duration.as_millis(),
                    transfer_rate_mbps = (file_size as f64 / 1024.0 / 1024.0) / duration.as_secs_f64(),
                    "SSH file upload succeeded"
                );
                Ok(())
            }
            Err(e) => {
                let duration = start_time.elapsed();
                error!(
                    target: "ssh_audit",
                    host = %details.host,
                    username = %details.username,
                    local_path = %local_path.display(),
                    remote_path = %remote_path,
                    file_size_bytes = file_size,
                    duration_ms = duration.as_millis(),
                    error = %e,
                    "SSH file upload failed"
                );
                Err(e)
            }
        }
    }

    /// Download file from executor with audit logging
    pub async fn download_file(
        &self,
        details: &SshConnectionDetails,
        remote_path: &str,
        local_path: &Path,
    ) -> Result<()> {
        let start_time = std::time::Instant::now();

        info!(
            target: "ssh_audit",
            host = %details.host,
            username = %details.username,
            remote_path = %remote_path,
            local_path = %local_path.display(),
            "SSH file download started"
        );

        match self
            .client
            .download_file(details, remote_path, local_path)
            .await
        {
            Ok(()) => {
                let duration = start_time.elapsed();
                let file_size = std::fs::metadata(local_path).map(|m| m.len()).unwrap_or(0);

                info!(
                    target: "ssh_audit",
                    host = %details.host,
                    username = %details.username,
                    remote_path = %remote_path,
                    local_path = %local_path.display(),
                    file_size_bytes = file_size,
                    duration_ms = duration.as_millis(),
                    transfer_rate_mbps = (file_size as f64 / 1024.0 / 1024.0) / duration.as_secs_f64(),
                    "SSH file download succeeded"
                );
                Ok(())
            }
            Err(e) => {
                let duration = start_time.elapsed();
                error!(
                    target: "ssh_audit",
                    host = %details.host,
                    username = %details.username,
                    remote_path = %remote_path,
                    local_path = %local_path.display(),
                    duration_ms = duration.as_millis(),
                    error = %e,
                    "SSH file download failed"
                );
                Err(e)
            }
        }
    }

    /// Cleanup remote files
    pub async fn cleanup_remote_files(
        &self,
        details: &SshConnectionDetails,
        file_paths: &[String],
    ) -> Result<()> {
        self.client.cleanup_remote_files(details, file_paths).await
    }

    /// Create SSH connection details for executor
    pub fn create_executor_connection(
        _executor_id: ExecutorId,
        host: String,
        username: String,
        port: Option<u16>,
        private_key_path: std::path::PathBuf,
        timeout: Option<Duration>,
    ) -> SshConnectionDetails {
        SshConnectionDetails {
            host,
            username,
            port: port.unwrap_or(22),
            private_key_path,
            timeout: timeout.unwrap_or(Duration::from_secs(30)),
        }
    }
}

impl Default for ValidatorSshClient {
    fn default() -> Self {
        Self::new()
    }
}

/// SSH connection details specifically for executor validation
#[derive(Debug, Clone)]
pub struct ExecutorSshDetails {
    /// Executor ID
    pub executor_id: ExecutorId,
    /// SSH connection details
    pub connection: SshConnectionDetails,
}

impl ExecutorSshDetails {
    /// Create new executor SSH details
    pub fn new(
        executor_id: ExecutorId,
        host: String,
        username: String,
        port: Option<u16>,
        private_key_path: std::path::PathBuf,
        timeout: Option<Duration>,
    ) -> Self {
        Self {
            executor_id,
            connection: SshConnectionDetails {
                host,
                username,
                port: port.unwrap_or(22),
                private_key_path,
                timeout: timeout.unwrap_or(Duration::from_secs(30)),
            },
        }
    }

    /// Get the underlying SSH connection details
    pub fn connection(&self) -> &SshConnectionDetails {
        &self.connection
    }

    /// Get executor ID
    pub fn executor_id(&self) -> &ExecutorId {
        &self.executor_id
    }
}

/// Bulk SSH operations for handling multiple targets
impl ValidatorSshClient {
    /// Upload file to multiple executors concurrently
    pub async fn upload_file_to_multiple(
        &self,
        targets: &[(SshConnectionDetails, String)], // (connection, remote_path)
        local_path: &Path,
        max_concurrent: usize,
    ) -> Result<Vec<Result<()>>> {
        use futures::stream::{self, StreamExt};

        let semaphore = Arc::new(tokio::sync::Semaphore::new(max_concurrent));

        let tasks: Vec<_> = targets
            .iter()
            .map(|(details, remote_path)| {
                let details = details.clone();
                let remote_path = remote_path.clone();
                let local_path = local_path.to_path_buf();
                let semaphore = semaphore.clone();

                async move {
                    let _permit = semaphore.acquire().await.unwrap();
                    self.upload_file(&details, &local_path, &remote_path).await
                }
            })
            .collect();

        let results = stream::iter(tasks)
            .buffer_unordered(max_concurrent)
            .collect::<Vec<_>>()
            .await;

        Ok(results)
    }

    /// Execute command on multiple executors concurrently
    pub async fn execute_command_on_multiple(
        &self,
        targets: &[SshConnectionDetails],
        command: &str,
        capture_output: bool,
        max_concurrent: usize,
    ) -> Result<Vec<Result<String>>> {
        use futures::stream::{self, StreamExt};

        let semaphore = Arc::new(tokio::sync::Semaphore::new(max_concurrent));
        let command = command.to_string();

        let tasks: Vec<_> = targets
            .iter()
            .map(|details| {
                let details = details.clone();
                let command = command.clone();
                let semaphore = semaphore.clone();

                async move {
                    let _permit = semaphore.acquire().await.unwrap();
                    self.execute_command(&details, &command, capture_output)
                        .await
                }
            })
            .collect();

        let results = stream::iter(tasks)
            .buffer_unordered(max_concurrent)
            .collect::<Vec<_>>()
            .await;

        Ok(results)
    }

    /// Generic retry execution with exponential backoff
    #[allow(dead_code)]
    async fn execute_with_retry<F, Fut, T>(&self, operation: F, operation_name: &str) -> Result<T>
    where
        F: Fn(&StandardSshClient) -> Fut,
        Fut: std::future::Future<Output = Result<T>>,
    {
        let mut delay = self.retry_config.initial_delay;
        let mut last_error = None;

        for attempt in 1..=self.retry_config.max_attempts {
            debug!(
                "Executing {} - attempt {} of {}",
                operation_name, attempt, self.retry_config.max_attempts
            );

            match operation(&self.client).await {
                Ok(result) => {
                    if attempt > 1 {
                        info!(
                            "Operation '{}' succeeded on attempt {}",
                            operation_name, attempt
                        );
                    }
                    return Ok(result);
                }
                Err(e) => {
                    warn!(
                        "Operation '{}' failed on attempt {}: {}",
                        operation_name, attempt, e
                    );

                    last_error = Some(e);

                    // Check if we should retry based on error type
                    let should_retry = self.should_retry_error(last_error.as_ref().unwrap());

                    if attempt < self.retry_config.max_attempts && should_retry {
                        debug!(
                            "Retrying operation '{}' in {:.2}s",
                            operation_name,
                            delay.as_secs_f64()
                        );
                        tokio::time::sleep(delay).await;

                        // Exponential backoff with jitter
                        delay = std::cmp::min(
                            Duration::from_millis(
                                (delay.as_millis() as f64 * self.retry_config.backoff_multiplier)
                                    as u64,
                            ),
                            self.retry_config.max_delay,
                        );

                        // Add some jitter to prevent thundering herd
                        let jitter = Duration::from_millis(fastrand::u64(0..=100));
                        delay += jitter;
                    } else if !should_retry {
                        debug!(
                            "Not retrying operation '{}' due to error type",
                            operation_name
                        );
                        break;
                    }
                }
            }
        }

        error!(
            "Operation '{}' failed after {} attempts",
            operation_name, self.retry_config.max_attempts
        );

        Err(last_error.unwrap_or_else(|| {
            anyhow::anyhow!(
                "Operation '{}' failed after {} attempts",
                operation_name,
                self.retry_config.max_attempts
            )
        }))
    }

    /// Determine if an error should trigger a retry
    pub fn should_retry_error(&self, error: &anyhow::Error) -> bool {
        let error_string = error.to_string().to_lowercase();

        // Don't retry authentication errors
        if error_string.contains("permission denied")
            || error_string.contains("authentication failed")
            || error_string.contains("invalid private key")
        {
            return false;
        }

        // Don't retry file not found errors
        if error_string.contains("no such file or directory") {
            return false;
        }

        // Retry connection errors if configured
        if error_string.contains("connection refused")
            || error_string.contains("network unreachable")
            || error_string.contains("host unreachable")
        {
            return self.retry_config.retry_on_connection_error;
        }

        // Retry timeout errors if configured
        if error_string.contains("timed out") || error_string.contains("timeout") {
            return self.retry_config.retry_on_timeout;
        }

        // Retry temporary failures
        if error_string.contains("temporary failure")
            || error_string.contains("resource temporarily unavailable")
            || error_string.contains("try again")
        {
            return true;
        }

        // Default to not retrying unknown errors
        false
    }
}
