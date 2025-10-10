//! SSH Session Management for Direct Node Access
//!
//! Manages direct SSH connections to nodes without intermediary session management.
//! Validators connect directly to nodes using SSH credentials provided during discovery.

use anyhow::{Context, Result};
use basilica_common::ssh::SshConnectionDetails;
use std::collections::HashSet;
use std::path::PathBuf;
use std::sync::Arc;
use std::time::Duration;
use tokio::sync::Mutex;
use tracing::{info, warn};

/// Trait for SSH key providers
pub trait SshKeyProvider {
    /// Get SSH key pair
    fn get_key_pair(&self) -> Result<(String, PathBuf)>;
}

/// SSH session configuration for establishing sessions
#[derive(Debug, Clone)]
pub struct SshSessionConfig {
    pub purpose: String,
    pub session_duration_secs: i64,
    pub session_metadata: String,
    pub rental_mode: bool,
    pub timeout: Duration,
}

impl Default for SshSessionConfig {
    fn default() -> Self {
        Self {
            purpose: "binary_validation".to_string(),
            session_duration_secs: 300,
            session_metadata: "binary_validation_session".to_string(),
            rental_mode: false,
            timeout: Duration::from_secs(30),
        }
    }
}

/// Direct node connection details
#[derive(Debug, Clone)]
pub struct NodeConnectionDetails {
    pub node_id: String,
    pub host: String,
    pub port: u16,
    pub username: String,
    pub ssh_key_path: Option<PathBuf>,
}

impl NodeConnectionDetails {
    /// Convert to SshConnectionDetails for SSH operations
    pub fn to_ssh_details(&self, private_key_path: PathBuf) -> SshConnectionDetails {
        SshConnectionDetails {
            host: self.host.clone(),
            port: self.port,
            username: self.username.clone(),
            private_key_path,
            timeout: std::time::Duration::from_secs(30),
        }
    }
}

/// Adapter for ValidatorSshKeyManager to implement SshKeyProvider
pub struct ValidatorSshKeyProvider<'a> {
    key_manager: &'a crate::ssh::ValidatorSshKeyManager,
}

impl<'a> ValidatorSshKeyProvider<'a> {
    pub fn new(key_manager: &'a crate::ssh::ValidatorSshKeyManager) -> Self {
        Self { key_manager }
    }
}

impl<'a> SshKeyProvider for ValidatorSshKeyProvider<'a> {
    fn get_key_pair(&self) -> Result<(String, PathBuf)> {
        if let Some((public_key, private_key_path)) = self.key_manager.get_persistent_key() {
            Ok((public_key.clone(), private_key_path.clone()))
        } else {
            Err(anyhow::anyhow!("No persistent SSH key available"))
        }
    }
}

/// SSH session manager for direct node connections
pub struct DirectSshSessionManager {
    /// Currently active sessions
    active_sessions: Arc<Mutex<HashSet<String>>>,
    /// SSH key provider
    key_provider: Box<dyn SshKeyProvider + Send + Sync>,
}

impl DirectSshSessionManager {
    /// Create a new session manager
    pub fn new(key_provider: Box<dyn SshKeyProvider + Send + Sync>) -> Self {
        Self {
            active_sessions: Arc::new(Mutex::new(HashSet::new())),
            key_provider,
        }
    }

    /// Establish direct SSH connection to a node
    pub async fn connect_to_node(
        &self,
        node_details: &NodeConnectionDetails,
    ) -> Result<SshConnectionDetails> {
        let (_, private_key_path) = self
            .key_provider
            .get_key_pair()
            .context("Failed to get SSH key pair")?;

        // Mark session as active
        {
            let mut sessions = self.active_sessions.lock().await;
            sessions.insert(node_details.node_id.clone());
        }

        info!(
            "Establishing direct SSH connection to node {} at {}:{}",
            node_details.node_id, node_details.host, node_details.port
        );

        Ok(node_details.to_ssh_details(private_key_path))
    }

    /// Release SSH connection for a node
    pub async fn release_node_connection(&self, node_id: &str) -> Result<()> {
        let mut sessions = self.active_sessions.lock().await;
        if sessions.remove(node_id) {
            info!("Released SSH connection for node {}", node_id);
        } else {
            warn!(
                "Attempted to release non-existent session for node {}",
                node_id
            );
        }
        Ok(())
    }

    /// Check if a node has an active session
    pub async fn is_node_active(&self, node_id: &str) -> bool {
        let sessions = self.active_sessions.lock().await;
        sessions.contains(node_id)
    }

    /// Get all active node IDs
    pub async fn get_active_nodes(&self) -> Vec<String> {
        let sessions = self.active_sessions.lock().await;
        sessions.iter().cloned().collect()
    }

    /// Clean up all active sessions
    pub async fn cleanup_all_sessions(&self) -> Result<()> {
        let mut sessions = self.active_sessions.lock().await;
        let count = sessions.len();
        sessions.clear();
        info!("Cleaned up {} active SSH sessions", count);
        Ok(())
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use std::path::PathBuf;

    struct MockKeyProvider;

    impl SshKeyProvider for MockKeyProvider {
        fn get_key_pair(&self) -> Result<(String, PathBuf)> {
            Ok((
                "ssh-rsa AAAAB3NzaC1yc2E...".to_string(),
                PathBuf::from("/tmp/test_key"),
            ))
        }
    }

    #[tokio::test]
    async fn test_direct_connection() {
        let manager = DirectSshSessionManager::new(Box::new(MockKeyProvider));

        let node_details = NodeConnectionDetails {
            node_id: "test-node-1".to_string(),
            host: "192.168.1.100".to_string(),
            port: 22,
            username: "basilica".to_string(),
            ssh_key_path: None,
        };

        let ssh_details = manager.connect_to_node(&node_details).await.unwrap();
        assert_eq!(ssh_details.host, "192.168.1.100");
        assert_eq!(ssh_details.port, 22);
        assert_eq!(ssh_details.username, "basilica");

        assert!(manager.is_node_active("test-node-1").await);

        manager
            .release_node_connection("test-node-1")
            .await
            .unwrap();
        assert!(!manager.is_node_active("test-node-1").await);
    }

    #[tokio::test]
    async fn test_multiple_sessions() {
        let manager = DirectSshSessionManager::new(Box::new(MockKeyProvider));

        for i in 1..=3 {
            let node_details = NodeConnectionDetails {
                node_id: format!("test-node-{}", i),
                host: format!("192.168.1.{}", 100 + i),
                port: 22,
                username: "basilica".to_string(),
                ssh_key_path: None,
            };
            manager.connect_to_node(&node_details).await.unwrap();
        }

        let active_nodes = manager.get_active_nodes().await;
        assert_eq!(active_nodes.len(), 3);

        manager.cleanup_all_sessions().await.unwrap();
        let active_nodes = manager.get_active_nodes().await;
        assert_eq!(active_nodes.len(), 0);
    }
}
