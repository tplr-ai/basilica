//! Executor Connection Manager
//!
//! Manages SSH connections to executors for the miner, providing connection pooling,
//! health checks, and command execution capabilities.

use anyhow::{Context, Result};
use common::identity::ExecutorId;
use common::ssh::{
    SshConnectionConfig, SshConnectionDetails, SshConnectionManager, StandardSshClient,
};
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
