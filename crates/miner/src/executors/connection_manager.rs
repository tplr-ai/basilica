//! Executor Connection Manager
//!
//! Manages SSH connections to executors for the miner, providing connection pooling,
//! health checks, and command execution capabilities.

use anyhow::{Context, Result};
use common::identity::ExecutorId;
use common::ssh::{
    SshConnectionConfig, SshConnectionDetails, SshConnectionManager, StandardSshClient,
};
use protocol::miner_discovery::SshSessionStatus;
use std::collections::HashMap;
use std::path::PathBuf;
use std::sync::Arc;
use std::time::{Duration, Instant};
use tokio::sync::RwLock;
use tracing::{debug, error, info, warn};

/// Information about an executor
#[derive(Debug, Clone)]
pub struct ExecutorInfo {
    pub id: ExecutorId,
    pub host: String,
    pub ssh_port: u16,
    pub ssh_username: String,
    pub grpc_endpoint: Option<String>,
    pub last_health_check: Option<Instant>,
    pub is_healthy: bool,
}

/// SSH connection to an executor
#[derive(Clone)]
pub struct ExecutorConnection {
    pub executor_id: ExecutorId,
    ssh_client: Arc<StandardSshClient>,
    connection_details: SshConnectionDetails,
    last_used: Arc<RwLock<Instant>>,
}

impl ExecutorConnection {
    /// Get SSH credentials string
    pub fn get_ssh_credentials(&self) -> String {
        format!(
            "{}@{}:{}",
            self.connection_details.username,
            self.connection_details.host,
            self.connection_details.port
        )
    }

    /// Execute command on executor
    pub async fn execute_command(&self, command: &str) -> Result<String> {
        debug!(
            "Executing command on executor {}: {}",
            self.executor_id,
            if command.len() > 100 {
                format!("{}...", &command[..100])
            } else {
                command.to_string()
            }
        );

        // Update last used time
        {
            let mut last_used = self.last_used.write().await;
            *last_used = Instant::now();
        }

        self.ssh_client
            .execute_command(&self.connection_details, command, true)
            .await
            .with_context(|| format!("Failed to execute command on executor {}", self.executor_id))
    }

    /// Test connection health
    pub async fn test_connection(&self) -> Result<()> {
        self.ssh_client
            .test_connection(&self.connection_details)
            .await
    }
}

/// Configuration for executor connections
#[derive(Debug, Clone)]
pub struct ExecutorConnectionConfig {
    /// Path to miner's SSH key for executor access
    pub miner_executor_key_path: PathBuf,
    /// Default username for executor SSH
    pub default_executor_username: String,
    /// Connection timeout
    pub connection_timeout: Duration,
    /// Maximum idle time before connection is closed
    pub max_idle_time: Duration,
    /// Health check interval
    pub health_check_interval: Duration,
}

impl Default for ExecutorConnectionConfig {
    fn default() -> Self {
        Self {
            miner_executor_key_path: PathBuf::from("~/.ssh/miner_executor_key"),
            default_executor_username: "executor".to_string(),
            connection_timeout: Duration::from_secs(30),
            max_idle_time: Duration::from_secs(300),
            health_check_interval: Duration::from_secs(60),
        }
    }
}

/// Manages connections to multiple executors
pub struct ExecutorConnectionManager {
    /// Configuration
    config: ExecutorConnectionConfig,
    /// SSH client configuration
    ssh_config: SshConnectionConfig,
    /// Executor information
    executors: Arc<RwLock<HashMap<ExecutorId, ExecutorInfo>>>,
    /// Active connections
    connections: Arc<RwLock<HashMap<ExecutorId, Arc<ExecutorConnection>>>>,
}

impl ExecutorConnectionManager {
    /// Create a new connection manager
    pub fn new(config: ExecutorConnectionConfig) -> Self {
        let ssh_config = SshConnectionConfig {
            connection_timeout: config.connection_timeout,
            ..Default::default()
        };

        Self {
            config,
            ssh_config,
            executors: Arc::new(RwLock::new(HashMap::new())),
            connections: Arc::new(RwLock::new(HashMap::new())),
        }
    }

    /// Create a test instance
    #[cfg(test)]
    pub fn new_test() -> Self {
        Self::new(ExecutorConnectionConfig::default())
    }

    /// Register an executor
    pub async fn register_executor(&self, info: ExecutorInfo) -> Result<()> {
        info!("Registering executor {}", info.id);

        let mut executors = self.executors.write().await;
        executors.insert(info.id.clone(), info);

        Ok(())
    }

    /// Unregister an executor
    pub async fn unregister_executor(&self, executor_id: &ExecutorId) -> Result<()> {
        info!("Unregistering executor {}", executor_id);

        // Remove executor info
        {
            let mut executors = self.executors.write().await;
            executors.remove(executor_id);
        }

        // Close any active connection
        {
            let mut connections = self.connections.write().await;
            connections.remove(executor_id);
        }

        Ok(())
    }

    /// Get or establish connection to executor
    pub async fn get_executor_connection(
        &self,
        executor_id: &str,
    ) -> Result<Arc<ExecutorConnection>> {
        use std::str::FromStr;
        let executor_id = ExecutorId::from_str(executor_id)
            .map_err(|e| anyhow::anyhow!("Invalid executor ID: {}", e))?;

        // Check if we have an active connection
        {
            let connections = self.connections.read().await;
            if let Some(conn) = connections.get(&executor_id) {
                // Test if connection is still alive
                if conn.test_connection().await.is_ok() {
                    debug!("Reusing existing connection to executor {}", executor_id);
                    return Ok(conn.clone());
                } else {
                    debug!("Existing connection to executor {} is dead", executor_id);
                }
            }
        }

        // Get executor info
        let executor_info = {
            let executors = self.executors.read().await;
            executors
                .get(&executor_id)
                .cloned()
                .ok_or_else(|| anyhow::anyhow!("Executor {} not registered", executor_id))?
        };

        // Establish new connection
        info!("Establishing new connection to executor {}", executor_id);

        let connection_details = SshConnectionDetails {
            host: executor_info.host.clone(),
            port: executor_info.ssh_port,
            username: executor_info.ssh_username.clone(),
            private_key_path: self.config.miner_executor_key_path.clone(),
            timeout: self.config.connection_timeout,
        };

        let ssh_client = Arc::new(StandardSshClient::with_config(self.ssh_config.clone()));

        // Test connection
        ssh_client
            .test_connection(&connection_details)
            .await
            .with_context(|| format!("Failed to connect to executor {executor_id}"))?;

        let connection = Arc::new(ExecutorConnection {
            executor_id: executor_id.clone(),
            ssh_client,
            connection_details,
            last_used: Arc::new(RwLock::new(Instant::now())),
        });

        // Cache connection
        {
            let mut connections = self.connections.write().await;
            connections.insert(executor_id, connection.clone());
        }

        Ok(connection)
    }

    /// Get executor information
    pub async fn get_executor_info(&self, executor_id: &str) -> Result<ExecutorInfo> {
        use std::str::FromStr;
        let executor_id = ExecutorId::from_str(executor_id)
            .map_err(|e| anyhow::anyhow!("Invalid executor ID: {}", e))?;
        let executors = self.executors.read().await;
        executors
            .get(&executor_id)
            .cloned()
            .ok_or_else(|| anyhow::anyhow!("Executor {} not found", executor_id))
    }

    /// List all registered executors
    pub async fn list_executors(&self) -> Vec<ExecutorInfo> {
        let executors = self.executors.read().await;
        executors.values().cloned().collect()
    }

    /// Perform health check on an executor
    pub async fn health_check_executor(&self, executor_id: &ExecutorId) -> Result<bool> {
        debug!("Performing health check on executor {}", executor_id);

        match self.get_executor_connection(&executor_id.to_string()).await {
            Ok(conn) => {
                // Try to execute a simple command
                match conn.execute_command("echo 'health_check'").await {
                    Ok(output) => {
                        let is_healthy = output.trim() == "health_check";

                        // Update executor info
                        let mut executors = self.executors.write().await;
                        if let Some(info) = executors.get_mut(executor_id) {
                            info.last_health_check = Some(Instant::now());
                            info.is_healthy = is_healthy;
                        }

                        Ok(is_healthy)
                    }
                    Err(e) => {
                        warn!("Health check failed for executor {}: {}", executor_id, e);

                        // Update executor info
                        let mut executors = self.executors.write().await;
                        if let Some(info) = executors.get_mut(executor_id) {
                            info.last_health_check = Some(Instant::now());
                            info.is_healthy = false;
                        }

                        Ok(false)
                    }
                }
            }
            Err(e) => {
                error!(
                    "Failed to connect to executor {} for health check: {}",
                    executor_id, e
                );

                // Update executor info
                let mut executors = self.executors.write().await;
                if let Some(info) = executors.get_mut(executor_id) {
                    info.last_health_check = Some(Instant::now());
                    info.is_healthy = false;
                }

                Ok(false)
            }
        }
    }

    /// Run periodic health checks
    pub async fn run_health_check_task(&self) {
        let mut interval = tokio::time::interval(self.config.health_check_interval);

        loop {
            interval.tick().await;

            let executor_ids: Vec<ExecutorId> = {
                let executors = self.executors.read().await;
                executors.keys().cloned().collect()
            };

            for executor_id in executor_ids {
                if let Err(e) = self.health_check_executor(&executor_id).await {
                    error!("Health check error for executor {}: {}", executor_id, e);
                }
            }

            // Clean up idle connections
            self.cleanup_idle_connections().await;
        }
    }

    /// Clean up idle connections
    async fn cleanup_idle_connections(&self) {
        let mut connections = self.connections.write().await;
        let now = Instant::now();

        connections.retain(|executor_id, conn| {
            let last_used = conn.last_used.try_read();
            if let Ok(last_used) = last_used {
                let idle_time = now.duration_since(*last_used);
                if idle_time > self.config.max_idle_time {
                    info!("Closing idle connection to executor {}", executor_id);
                    return false;
                }
            }
            true
        });
    }

    /// Execute command on multiple executors
    pub async fn execute_on_all(&self, command: &str) -> HashMap<ExecutorId, Result<String>> {
        let executor_ids: Vec<ExecutorId> = {
            let executors = self.executors.read().await;
            executors.keys().cloned().collect()
        };

        let mut results = HashMap::new();

        for executor_id in executor_ids {
            let result = match self.get_executor_connection(&executor_id.to_string()).await {
                Ok(conn) => conn.execute_command(command).await,
                Err(e) => Err(e),
            };
            results.insert(executor_id, result);
        }

        results
    }

    /// Add validator SSH public key to executor's authorized_keys
    pub async fn add_validator_ssh_key(
        &self,
        executor_id: &str,
        public_key: &str,
        session_id: &str,
        expires_at: chrono::DateTime<chrono::Utc>,
    ) -> Result<()> {
        use std::str::FromStr;
        let executor_id = ExecutorId::from_str(executor_id)
            .map_err(|e| anyhow::anyhow!("Invalid executor ID: {}", e))?;

        let connection = self
            .get_executor_connection(&executor_id.to_string())
            .await
            .context("Failed to get executor connection for SSH key addition")?;

        // Create authorized_keys entry with session metadata and expiration
        let key_entry = format!(
            "{} validator-session-{} expires={} miner-managed",
            public_key.trim(),
            session_id,
            expires_at.to_rfc3339()
        );

        // Ensure .ssh directory exists with proper permissions
        let setup_ssh_cmd = r#"
            mkdir -p ~/.ssh && \
            chmod 700 ~/.ssh && \
            touch ~/.ssh/authorized_keys && \
            chmod 600 ~/.ssh/authorized_keys
        "#;

        connection
            .execute_command(setup_ssh_cmd)
            .await
            .context("Failed to setup SSH directory on executor")?;

        // Add the key to authorized_keys
        let add_key_cmd = format!(
            "echo '{}' >> ~/.ssh/authorized_keys",
            key_entry.replace('\'', "'\"'\"'")
        );

        connection
            .execute_command(&add_key_cmd)
            .await
            .context("Failed to add validator SSH key to executor")?;

        info!(
            "Added validator SSH key for session {} to executor {}",
            session_id, executor_id
        );

        Ok(())
    }

    /// Remove validator SSH key from executor's authorized_keys
    pub async fn remove_validator_ssh_key(
        &self,
        executor_id: &str,
        session_id: &str,
    ) -> Result<()> {
        use std::str::FromStr;
        let executor_id = ExecutorId::from_str(executor_id)
            .map_err(|e| anyhow::anyhow!("Invalid executor ID: {}", e))?;

        let connection = self
            .get_executor_connection(&executor_id.to_string())
            .await
            .context("Failed to get executor connection for SSH key removal")?;

        // Remove key by session ID comment
        let remove_key_cmd = format!(
            "sed -i '/validator-session-{session_id}/d' ~/.ssh/authorized_keys 2>/dev/null || true"
        );

        connection
            .execute_command(&remove_key_cmd)
            .await
            .context("Failed to remove validator SSH key from executor")?;

        info!(
            "Removed validator SSH key for session {} from executor {}",
            session_id, executor_id
        );

        Ok(())
    }

    /// Clean up expired SSH keys from all executors
    pub async fn cleanup_expired_ssh_keys(&self) -> Result<usize> {
        let now = chrono::Utc::now();
        let executor_ids: Vec<ExecutorId> = {
            let executors = self.executors.read().await;
            executors.keys().cloned().collect()
        };

        let mut cleaned_count = 0;

        for executor_id in executor_ids {
            match self
                .cleanup_expired_keys_for_executor(&executor_id, now)
                .await
            {
                Ok(count) => {
                    cleaned_count += count;
                    if count > 0 {
                        info!(
                            "Cleaned up {} expired SSH keys from executor {}",
                            count, executor_id
                        );
                    }
                }
                Err(e) => {
                    warn!(
                        "Failed to cleanup expired SSH keys for executor {}: {}",
                        executor_id, e
                    );
                }
            }
        }

        if cleaned_count > 0 {
            info!(
                "Total cleaned up {} expired SSH keys across all executors",
                cleaned_count
            );
        }

        Ok(cleaned_count)
    }

    /// Clean up expired SSH keys for a specific executor
    async fn cleanup_expired_keys_for_executor(
        &self,
        executor_id: &ExecutorId,
        now: chrono::DateTime<chrono::Utc>,
    ) -> Result<usize> {
        let connection = match self.get_executor_connection(&executor_id.to_string()).await {
            Ok(conn) => conn,
            Err(_) => return Ok(0), // Skip if executor is not accessible
        };

        // Get all validator session keys and check expiration
        let list_keys_cmd =
            "grep -n 'validator-session-.*expires=' ~/.ssh/authorized_keys 2>/dev/null || true";

        let output = connection
            .execute_command(list_keys_cmd)
            .await
            .unwrap_or_default();

        if output.trim().is_empty() {
            return Ok(0);
        }

        let mut lines_to_remove = Vec::new();
        let mut expired_count = 0;

        for line in output.lines() {
            if let Some(line_num_sep) = line.find(':') {
                let line_num = &line[..line_num_sep];
                let key_content = &line[line_num_sep + 1..];

                // Extract expiration time
                if let Some(expires_start) = key_content.find("expires=") {
                    let expires_str = &key_content[expires_start + 8..];
                    if let Some(expires_end) = expires_str.find(' ') {
                        let expires_str = &expires_str[..expires_end];

                        if let Ok(expires_at) = chrono::DateTime::parse_from_rfc3339(expires_str) {
                            if expires_at.with_timezone(&chrono::Utc) < now {
                                lines_to_remove.push(line_num.to_string());
                                expired_count += 1;
                            }
                        }
                    }
                }
            }
        }

        // Remove expired keys (in reverse order to maintain line numbers)
        for line_num in lines_to_remove.into_iter().rev() {
            let remove_cmd =
                format!("sed -i '{line_num}d' ~/.ssh/authorized_keys 2>/dev/null || true");

            if let Err(e) = connection.execute_command(&remove_cmd).await {
                warn!(
                    "Failed to remove expired SSH key at line {} from executor {}: {}",
                    line_num, executor_id, e
                );
            }
        }

        Ok(expired_count)
    }

    /// List active validator SSH sessions for an executor
    pub async fn list_validator_ssh_sessions(
        &self,
        executor_id: &str,
    ) -> Result<Vec<ValidatorSshSession>> {
        use std::str::FromStr;
        let executor_id = ExecutorId::from_str(executor_id)
            .map_err(|e| anyhow::anyhow!("Invalid executor ID: {}", e))?;

        let connection = self
            .get_executor_connection(&executor_id.to_string())
            .await
            .context("Failed to get executor connection for session listing")?;

        let list_cmd =
            "grep 'validator-session-.*expires=' ~/.ssh/authorized_keys 2>/dev/null || true";

        let output = connection
            .execute_command(list_cmd)
            .await
            .unwrap_or_default();

        let mut sessions = Vec::new();
        let now = chrono::Utc::now();

        for line in output.lines() {
            if let Some(session_info) = self.parse_ssh_session_line(line, now) {
                sessions.push(session_info);
            }
        }

        Ok(sessions)
    }

    /// Parse SSH session information from authorized_keys line
    fn parse_ssh_session_line(
        &self,
        line: &str,
        now: chrono::DateTime<chrono::Utc>,
    ) -> Option<ValidatorSshSession> {
        // Extract session ID
        let session_start = line.find("validator-session-")?;
        let session_part = &line[session_start + 18..];
        let session_end = session_part.find(' ')?;
        let session_id = session_part[..session_end].to_string();

        // Extract expiration time
        let expires_start = line.find("expires=")?;
        let expires_str = &line[expires_start + 8..];
        let expires_end = expires_str.find(' ')?;
        let expires_str = &expires_str[..expires_end];

        let expires_at = chrono::DateTime::parse_from_rfc3339(expires_str)
            .ok()?
            .with_timezone(&chrono::Utc);

        let status = if expires_at < now {
            SshSessionStatus::Expired
        } else {
            SshSessionStatus::Active
        };

        Some(ValidatorSshSession {
            session_id,
            expires_at,
            status,
            key_type: "validator".to_string(),
        })
    }
}

/// Information about an active validator SSH session
#[derive(Debug, Clone)]
pub struct ValidatorSshSession {
    pub session_id: String,
    pub expires_at: chrono::DateTime<chrono::Utc>,
    pub status: SshSessionStatus,
    pub key_type: String,
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_ssh_credentials_format() {
        use uuid::Uuid;
        let conn = ExecutorConnection {
            executor_id: ExecutorId::from_uuid(
                Uuid::parse_str("00000000-0000-0000-0000-000000000001").unwrap(),
            ),
            ssh_client: Arc::new(StandardSshClient::new()),
            connection_details: SshConnectionDetails {
                host: "192.168.1.100".to_string(),
                username: "executor".to_string(),
                port: 22,
                private_key_path: PathBuf::from("/tmp/key"),
                timeout: Duration::from_secs(30),
            },
            last_used: Arc::new(RwLock::new(Instant::now())),
        };

        assert_eq!(conn.get_ssh_credentials(), "executor@192.168.1.100:22");
    }

    #[tokio::test]
    async fn test_executor_registration() {
        use uuid::Uuid;
        let manager = ExecutorConnectionManager::new_test();

        let info = ExecutorInfo {
            id: ExecutorId::from_uuid(
                Uuid::parse_str("00000000-0000-0000-0000-000000000002").unwrap(),
            ),
            host: "192.168.1.100".to_string(),
            ssh_port: 22,
            ssh_username: "executor".to_string(),
            grpc_endpoint: Some("http://192.168.1.100:50051".to_string()),
            last_health_check: None,
            is_healthy: true,
        };

        manager.register_executor(info.clone()).await.unwrap();

        let registered = manager
            .get_executor_info("00000000-0000-0000-0000-000000000002")
            .await
            .unwrap();
        assert_eq!(registered.id, info.id);
        assert_eq!(registered.host, info.host);

        let all_executors = manager.list_executors().await;
        assert_eq!(all_executors.len(), 1);
        assert_eq!(all_executors[0].id, info.id);
    }

    #[tokio::test]
    async fn test_executor_unregistration() {
        use uuid::Uuid;
        let manager = ExecutorConnectionManager::new_test();

        let info = ExecutorInfo {
            id: ExecutorId::from_uuid(
                Uuid::parse_str("00000000-0000-0000-0000-000000000002").unwrap(),
            ),
            host: "192.168.1.100".to_string(),
            ssh_port: 22,
            ssh_username: "executor".to_string(),
            grpc_endpoint: None,
            last_health_check: None,
            is_healthy: true,
        };

        manager.register_executor(info).await.unwrap();
        assert_eq!(manager.list_executors().await.len(), 1);

        manager
            .unregister_executor(&ExecutorId::from_uuid(
                Uuid::parse_str("00000000-0000-0000-0000-000000000002").unwrap(),
            ))
            .await
            .unwrap();
        assert_eq!(manager.list_executors().await.len(), 0);

        let result = manager
            .get_executor_info("00000000-0000-0000-0000-000000000002")
            .await;
        assert!(result.is_err());
    }
}
