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
//!    preventing nodes from running modified versions that could fake results.
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

pub mod automation_components;
pub mod dynamic_discovery_controller;
pub mod key_manager;
pub mod key_manager_builder;
pub mod session;

pub use automation_components::SshAutomationComponents;
pub use key_manager::ValidatorSshKeyManager;
pub use session::DirectSshSessionManager;
pub type SshSessionManager = DirectSshSessionManager; // Alias for backward compatibility

#[cfg(test)]
mod tests;

use anyhow::Result;
use basilica_common::identity::NodeId;
use basilica_common::ssh::{
    PackageManager, SshConnectionConfig, SshConnectionDetails, SshConnectionManager,
    SshFileTransferManager, StandardSshClient,
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
    /// Ensure host key is available
    pub async fn ensure_host_key_available(&self, details: &SshConnectionDetails) -> Result<()> {
        self.client.ensure_host_key_available(details).await
    }

    /// Create a new validator SSH client with default configuration
    pub fn new() -> Self {
        // Use permissive SSH config to avoid host key verification issues
        let config = SshConnectionConfig {
            strict_host_key_checking: false,
            known_hosts_file: None,
            ..Default::default()
        };
        Self {
            client: StandardSshClient::with_config(config),
            connection_pool: Arc::new(Mutex::new(HashMap::new())),
            session_stats: Arc::new(Mutex::new(SshSessionStats::default())),
            max_pool_size: 100,
            pool_timeout: Duration::from_secs(900),
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
            pool_timeout: Duration::from_secs(900),
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
            pool_timeout: Duration::from_secs(900),
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
        let mut pool = match self.connection_pool.lock() {
            Ok(pool) => pool,
            Err(e) => {
                tracing::error!("Connection pool mutex poisoned: {}", e);
                return;
            }
        };

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
        match self.session_stats.lock() {
            Ok(stats) => stats.clone(),
            Err(e) => {
                tracing::error!("Session stats mutex poisoned: {}", e);
                SshSessionStats::default()
            }
        }
    }

    /// Get connection pool information
    pub fn get_pool_info(&self) -> (usize, Vec<String>) {
        match self.connection_pool.lock() {
            Ok(pool) => {
                let keys: Vec<String> = pool.keys().cloned().collect();
                (pool.len(), keys)
            }
            Err(e) => {
                tracing::error!("Connection pool mutex poisoned: {}", e);
                (0, Vec::new())
            }
        }
    }

    /// Clear connection pool
    pub fn clear_pool(&self) {
        match self.connection_pool.lock() {
            Ok(mut pool) => pool.clear(),
            Err(e) => tracing::error!("Connection pool mutex poisoned: {}", e),
        }
    }

    /// Refresh host key for node
    pub async fn refresh_host_key(&self, details: &SshConnectionDetails) -> Result<()> {
        info!(
            target: "ssh_audit",
            host = %details.host,
            port = details.port,
            "Refreshing host key"
        );
        self.client.refresh_host_key(details).await
    }

    /// Test SSH connection to node
    pub async fn test_connection(&self, details: &SshConnectionDetails) -> Result<()> {
        info!(
            "Testing SSH connection to {}@{}",
            details.username, details.host
        );
        self.client.test_connection(details).await
    }

    /// Execute command on node with audit logging
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

    /// Upload file to node with audit logging
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

    /// Download file from node with audit logging
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

    /// Create SSH connection details for node
    pub fn create_node_connection(
        _node_id: NodeId,
        host: String,
        username: String,
        port: u16,
        private_key_path: std::path::PathBuf,
        timeout: Option<Duration>,
    ) -> SshConnectionDetails {
        SshConnectionDetails {
            host,
            username,
            port,
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

/// SSH connection details specifically for node validation
#[derive(Debug, Clone)]
pub struct NodeSshDetails {
    /// Node ID
    pub node_id: NodeId,
    /// SSH connection details
    pub connection: SshConnectionDetails,
}

impl NodeSshDetails {
    /// Create new node SSH details
    pub fn new(
        node_id: NodeId,
        host: String,
        username: String,
        port: u16,
        private_key_path: std::path::PathBuf,
        timeout: Option<Duration>,
    ) -> Self {
        Self {
            node_id,
            connection: SshConnectionDetails {
                host,
                username,
                port,
                private_key_path,
                timeout: timeout.unwrap_or(Duration::from_secs(30)),
            },
        }
    }

    /// Get the underlying SSH connection details
    pub fn connection(&self) -> &SshConnectionDetails {
        &self.connection
    }

    /// Get node ID
    pub fn node_id(&self) -> &NodeId {
        &self.node_id
    }
}

/// Bulk SSH operations for handling multiple targets
impl ValidatorSshClient {
    /// Upload file to multiple nodes concurrently
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
                    let _permit = match semaphore.acquire().await {
                        Ok(permit) => permit,
                        Err(e) => {
                            tracing::error!(
                                "Failed to acquire semaphore permit for file upload: {}",
                                e
                            );
                            return Err(anyhow::anyhow!("Semaphore acquisition failed: {}", e));
                        }
                    };
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

    /// Execute command on multiple nodes concurrently
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
                    let _permit = match semaphore.acquire().await {
                        Ok(permit) => permit,
                        Err(e) => {
                            tracing::error!(
                                "Failed to acquire semaphore permit for command execution: {}",
                                e
                            );
                            return Err(anyhow::anyhow!("Semaphore acquisition failed: {}", e));
                        }
                    };
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

    /// Ensures a command is installed on the remote system, installing it if necessary
    ///
    /// # Arguments
    /// * `ssh_details` - SSH connection details for the remote system
    /// * `command_name` - The name of the command to check (e.g., "lshw", "curl")
    /// * `package_name` - The package to install if the command is not found (may differ from command_name, e.g., "openssh-server" for "sshd")
    ///
    /// # Returns
    /// * `Ok(())` if the command is already installed or successfully installed
    /// * `Err` if installation fails or package manager is not supported
    ///
    /// # Example
    /// ```ignore
    /// // Ensure lshw is installed (command and package have same name)
    /// client.ensure_installed(ssh_details, "lshw", "lshw").await?;
    ///
    /// // Ensure sshd is installed (command and package have different names)
    /// client.ensure_installed(ssh_details, "sshd", "openssh-server").await?;
    /// ```
    pub async fn ensure_installed(
        &self,
        ssh_details: &SshConnectionDetails,
        command_name: &str,
        package_name: &str,
    ) -> Result<()> {
        debug!(
            command = command_name,
            package = package_name,
            "Checking if command is installed"
        );

        // Check if command exists using `command -v` (more portable than `which`)
        let check_cmd = format!("command -v {}", command_name);
        match self.execute_command(ssh_details, &check_cmd, true).await {
            Ok(output) if !output.trim().is_empty() => {
                debug!(
                    command = command_name,
                    path = output.trim(),
                    "Command is already installed"
                );
                return Ok(());
            }
            _ => {
                info!(
                    command = command_name,
                    package = package_name,
                    "Command not found, attempting to install"
                );
            }
        }

        // Check if we're running as root or if sudo is available
        let is_root = self
            .execute_command(ssh_details, "test \"$(id -u)\" -eq 0", false)
            .await
            .is_ok();
        let has_sudo = if is_root {
            false // If we're root, we don't need sudo
        } else {
            self.execute_command(ssh_details, "command -v sudo", false)
                .await
                .is_ok()
        };

        debug!(
            is_root = is_root,
            has_sudo = has_sudo,
            "Detected privilege level for package installation"
        );

        // Detect package manager
        let package_manager = self.detect_package_manager(ssh_details).await?;

        // Install package based on detected package manager
        info!(
            package = package_name,
            package_manager = %package_manager,
            is_root = is_root,
            has_sudo = has_sudo,
            "Installing package"
        );

        // Determine if we should use sudo based on root status and sudo availability
        let use_sudo = !is_root && has_sudo;

        if !is_root && !has_sudo {
            warn!("Neither root access nor sudo is available - package installation may fail");
        }

        let install_cmd = package_manager.install_command(package_name, use_sudo);

        // Use a longer timeout for package installation operations (5 minutes)
        let mut install_details = ssh_details.clone();
        install_details.timeout = Duration::from_secs(900);

        self.execute_command(&install_details, &install_cmd, true)
            .await
            .map_err(|e| {
                anyhow::anyhow!(
                    "Failed to install {} via {}: {}",
                    package_name,
                    package_manager,
                    e
                )
            })?;

        // Verify installation was successful
        let verify_cmd = format!("command -v {}", command_name);
        match self.execute_command(ssh_details, &verify_cmd, true).await {
            Ok(output) if !output.trim().is_empty() => {
                info!(
                    command = command_name,
                    package = package_name,
                    path = output.trim(),
                    "Successfully installed command"
                );
                Ok(())
            }
            _ => Err(anyhow::anyhow!(
                "Package {} was installed but command {} is still not available",
                package_name,
                command_name
            )),
        }
    }

    /// Detects the package manager available on the remote system
    async fn detect_package_manager(
        &self,
        ssh_details: &SshConnectionDetails,
    ) -> Result<PackageManager> {
        // Check for apt-get (Debian/Ubuntu)
        if self
            .execute_command(ssh_details, PackageManager::Apt.check_command(), true)
            .await
            .is_ok()
        {
            return Ok(PackageManager::Apt);
        }

        // Check for dnf first (newer Fedora/RHEL 8+)
        if self
            .execute_command(ssh_details, PackageManager::Dnf.check_command(), true)
            .await
            .is_ok()
        {
            return Ok(PackageManager::Dnf);
        }

        // Check for yum (older RHEL/CentOS/Fedora)
        if self
            .execute_command(ssh_details, PackageManager::Yum.check_command(), true)
            .await
            .is_ok()
        {
            return Ok(PackageManager::Yum);
        }

        // Check for apk (Alpine)
        if self
            .execute_command(ssh_details, PackageManager::Apk.check_command(), true)
            .await
            .is_ok()
        {
            return Ok(PackageManager::Apk);
        }

        Err(anyhow::anyhow!(
            "Could not detect a supported package manager (apt, dnf, yum, or apk)"
        ))
    }
}
