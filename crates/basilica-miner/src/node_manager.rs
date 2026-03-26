//! Node management module for direct node access
//!
//! This module manages the nodes that the miner offers to validators.
//! Nodes are compute resources with SSH access that validators can use directly.

use anyhow::Result;
use basilica_common::ssh::{
    SshConnectionConfig, SshConnectionDetails, SshConnectionManager, StandardSshClient,
};
use basilica_common::types::GpuCategory;
use serde::{Deserialize, Serialize};
use ssh_key::PublicKey;
use std::collections::HashMap;
use std::net::IpAddr;
use std::path::PathBuf;
use std::sync::Arc;
use std::time::Duration;
use tokio::sync::RwLock;
use tracing::{debug, info, warn};

use crate::config::{expand_tilde_in_path, NodeSshConfig};

/// Shell command template for securely rewriting authorized_keys
/// Creates a secure temp file, filters out validator keys, and atomically replaces authorized_keys
const SSH_REWRITE_AUTHORIZED_KEYS_BASE: &str = r#"umask 077; mkdir -p ~/.ssh && chmod 700 ~/.ssh && tmp="$(mktemp ~/.ssh/authorized_keys.XXXXXX)" && (grep -v 'validator-' ~/.ssh/authorized_keys 2>/dev/null || true) > "$tmp""#;

/// Shell command suffix to atomically move temp file to authorized_keys
const SSH_MOVE_TO_AUTHORIZED_KEYS: &str =
    r#"mv -f "$tmp" ~/.ssh/authorized_keys && chmod 600 ~/.ssh/authorized_keys"#;

/// Configuration for a single node
#[derive(Clone, Debug, Serialize)]
pub struct NodeConfig {
    /// SSH host IP literal (IPv4 or IPv6)
    pub host: String,
    /// SSH port (typically 22)
    pub port: u16,
    /// SSH username for validator access
    pub username: String,
    /// GPU category for this node (e.g., "H100", "A100", "RTX4090")
    pub gpu_category: String,
    /// Number of GPUs on this node
    pub gpu_count: u32,
    /// Additional SSH options
    pub additional_opts: Option<String>,
}

/// Intermediate struct for deserializing NodeConfig
#[derive(Deserialize)]
struct NodeConfigRaw {
    host: String,
    port: u16,
    username: String,
    #[serde(default = "default_gpu_category")]
    gpu_category: String,
    #[serde(default = "default_gpu_count")]
    gpu_count: u32,
    additional_opts: Option<String>,
}

impl<'de> Deserialize<'de> for NodeConfig {
    fn deserialize<D>(deserializer: D) -> Result<Self, D::Error>
    where
        D: serde::Deserializer<'de>,
    {
        let raw = NodeConfigRaw::deserialize(deserializer)?;

        // Validate gpu_category is a known GPU type
        let gpu_cat: GpuCategory = raw.gpu_category.parse().unwrap(); // Infallible
        if matches!(&gpu_cat, GpuCategory::Other(_)) {
            return Err(serde::de::Error::custom(format!(
                "GPU type '{}' is not supported",
                raw.gpu_category
            )));
        }

        Ok(NodeConfig {
            host: raw.host,
            port: raw.port,
            username: raw.username,
            gpu_category: gpu_cat.to_string(),
            gpu_count: raw.gpu_count,
            additional_opts: raw.additional_opts,
        })
    }
}

fn default_gpu_category() -> String {
    "UNKNOWN".to_string()
}

fn default_gpu_count() -> u32 {
    1
}

impl NodeConfig {
    /// Convert to SSH connection details for remote command execution
    fn to_ssh_connection_details(&self, private_key_path: PathBuf) -> SshConnectionDetails {
        SshConnectionDetails {
            host: self.host.clone(),
            username: self.username.clone(),
            port: self.port,
            private_key_path,
            timeout: Duration::from_secs(30),
        }
    }
}

/// Manages nodes available for rental
pub struct NodeManager {
    /// Map of host to node configuration
    nodes: Arc<RwLock<HashMap<String, NodeConfig>>>,
    /// Currently assigned validator hotkey (single-assignment model)
    current_assigned_validator: Arc<RwLock<Option<String>>>,
    /// SSH client for executing remote commands
    ssh_client: Arc<StandardSshClient>,
    /// SSH configuration
    ssh_config: NodeSshConfig,
}

/// Node with its configuration (identified by host)
#[derive(Clone, Debug)]
pub struct RegisteredNode {
    pub config: NodeConfig,
}

impl std::fmt::Debug for NodeManager {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        f.debug_struct("NodeManager")
            .field("nodes", &"<Arc<RwLock<HashMap>>>")
            .field(
                "current_assigned_validator",
                &"<Arc<RwLock<Option<String>>>>",
            )
            .field("ssh_client", &"<StandardSshClient>")
            .finish()
    }
}

impl Default for NodeManager {
    fn default() -> Self {
        Self::new(NodeSshConfig::default())
    }
}

impl NodeManager {
    /// Create a new node manager
    pub fn new(ssh_config: NodeSshConfig) -> Self {
        // Use permissive SSH config to avoid host key verification issues
        let config = SshConnectionConfig {
            strict_host_key_checking: false,
            known_hosts_file: None,
            ..Default::default()
        };
        Self {
            nodes: Arc::new(RwLock::new(HashMap::new())),
            current_assigned_validator: Arc::new(RwLock::new(None)),
            ssh_client: Arc::new(StandardSshClient::with_config(config)),
            ssh_config,
        }
    }

    /// Register a node for availability (keyed by host IP)
    pub async fn register_node(&self, config: NodeConfig) -> Result<()> {
        let new_node_ip = config.host.parse::<IpAddr>().map_err(|_| {
            anyhow::anyhow!(
                "Invalid node host '{}': must be a valid IPv4 or IPv6 literal",
                config.host
            )
        })?;

        let mut nodes = self.nodes.write().await;

        for existing_host in nodes.keys() {
            let existing_ip = existing_host.parse::<IpAddr>().map_err(|_| {
                anyhow::anyhow!(
                    "Invalid existing node host '{}': expected IPv4/IPv6 literal",
                    existing_host
                )
            })?;
            if existing_ip == new_node_ip {
                anyhow::bail!(
                    "Node with IP {} is already registered (existing host: '{}')",
                    new_node_ip,
                    existing_host
                );
            }
        }

        info!("Registering node at {}:{}", config.host, config.port);
        nodes.insert(config.host.clone(), config);
        Ok(())
    }

    /// Remove a node from availability by host
    pub async fn unregister_node(&self, host: &str) -> Result<()> {
        let mut nodes = self.nodes.write().await;
        if nodes.remove(host).is_some() {
            info!("Unregistered node {}", host);
        } else {
            warn!("Attempted to unregister unknown node {}", host);
        }
        Ok(())
    }

    /// Get all available nodes
    pub async fn list_nodes(&self) -> Result<Vec<RegisteredNode>> {
        let nodes = self.nodes.read().await;
        Ok(nodes
            .values()
            .map(|config| RegisteredNode {
                config: config.clone(),
            })
            .collect())
    }

    /// Get a specific node by host
    pub async fn get_node(&self, host: &str) -> Result<Option<NodeConfig>> {
        let nodes = self.nodes.read().await;
        Ok(nodes.get(host).cloned())
    }

    /// Normalize SSH public key by extracting algorithm + key and adding our identifier
    pub fn normalize_ssh_key(ssh_public_key: &str, validator_hotkey: &str) -> String {
        let parts: Vec<&str> = ssh_public_key.split_whitespace().collect();

        if parts.len() >= 2 {
            // Keep only algorithm and base64 key, add our identifier as comment
            format!("{} {} validator-{}", parts[0], parts[1], validator_hotkey)
        } else {
            // Fallback if format is unexpected
            format!("{} validator-{}", ssh_public_key.trim(), validator_hotkey)
        }
    }

    /// Extract core key (algorithm + base64) for comparison
    pub fn extract_key_core(ssh_public_key: &str) -> String {
        let parts: Vec<&str> = ssh_public_key.split_whitespace().collect();

        if parts.len() >= 2 {
            format!("{} {}", parts[0], parts[1])
        } else {
            ssh_public_key.trim().to_string()
        }
    }

    /// Set the currently assigned validator (no SSH operations, just updates state)
    pub async fn set_assigned_validator(&self, validator_hotkey: &str) {
        let mut current = self.current_assigned_validator.write().await;
        *current = Some(validator_hotkey.to_string());
        info!("Assigned validator: {}", validator_hotkey);
    }

    /// Get the currently assigned validator hotkey
    pub async fn get_assigned_validator(&self) -> Option<String> {
        self.current_assigned_validator.read().await.clone()
    }

    /// Deploy validator SSH keys to all managed nodes
    /// This method validates the SSH key, normalizes it, and deploys it to all nodes
    /// using exclusive access (removes old validator keys, adds new one)
    pub async fn deploy_validator_keys(
        &self,
        validator_hotkey: &str,
        ssh_public_key: &str,
    ) -> Result<()> {
        // Validate SSH public key format
        let trimmed_key = ssh_public_key.trim();
        if PublicKey::from_openssh(trimmed_key).is_err() {
            return Err(anyhow::anyhow!("Invalid SSH public key format"));
        }

        // Normalize the key with our identifier
        let normalized_key = Self::normalize_ssh_key(trimmed_key, validator_hotkey);

        // Get all nodes
        let nodes = self.list_nodes().await?;
        let node_count = nodes.len();

        if node_count == 0 {
            info!(
                "Validator {} has no available nodes; skipping SSH key deployment",
                validator_hotkey
            );
            return Ok(());
        }

        // Get the miner's SSH private key path from config
        let private_key_path = self.get_ssh_key_path();

        // Deploy the SSH key to each node, ensuring exclusive access
        for registered_node in &nodes {
            info!(
                "Setting exclusive SSH access for validator {} on node {}",
                validator_hotkey, registered_node.config.host
            );

            // Build SSH connection details
            let connection_details = registered_node
                .config
                .to_ssh_connection_details(private_key_path.clone());

            // Set exclusive validator key (removes all other validators, adds current one)
            self.set_exclusive_validator_key_on_node(
                &connection_details,
                &registered_node.config.host,
                validator_hotkey,
                &normalized_key,
            )
            .await?;
        }

        info!(
            "Deployed SSH keys for validator {} on {} nodes",
            validator_hotkey, node_count
        );

        Ok(())
    }

    /// Revoke a validator's authorization and remove their SSH key from all nodes
    pub async fn revoke_validator(&self, validator_hotkey: &str) -> Result<()> {
        info!("Revoking validator {} authorization", validator_hotkey);

        // Check if this validator is currently assigned
        let should_revoke = {
            let current = self.current_assigned_validator.read().await;
            current.as_deref() == Some(validator_hotkey)
        };

        if !should_revoke {
            info!(
                "Validator {} is not the current assignment; skipping revoke",
                validator_hotkey
            );
            return Ok(());
        }

        let nodes = self.list_nodes().await?;

        // Get the miner's SSH private key path from config
        let private_key_path = self.get_ssh_key_path();

        // Remove all validator keys from each node
        for registered_node in &nodes {
            info!(
                "Removing all validator keys from node {} (revoking validator {})",
                registered_node.config.host, validator_hotkey
            );

            // Build SSH connection details
            let connection_details = registered_node
                .config
                .to_ssh_connection_details(private_key_path.clone());

            // Remove all validator keys from the node
            if let Err(e) = self
                .remove_all_validator_keys_on_node(
                    &connection_details,
                    &registered_node.config.host,
                )
                .await
            {
                warn!(
                    "Failed to remove validator keys from node {}: {}",
                    registered_node.config.host, e
                );
            }
        }

        // Remove from assignment
        let mut current = self.current_assigned_validator.write().await;
        current.take();

        info!(
            "Revoked validator {} authorization from {} nodes",
            validator_hotkey,
            nodes.len()
        );

        Ok(())
    }

    /// Check if a validator is authorized
    pub async fn is_validator_authorized(&self, validator_hotkey: &str) -> bool {
        let current = self.current_assigned_validator.read().await;
        current.as_deref() == Some(validator_hotkey)
    }

    /// Get the SSH private key path from config with tilde expansion
    /// Note: Path existence is validated during config loading, so this method
    /// assumes the path is valid.
    fn get_ssh_key_path(&self) -> PathBuf {
        expand_tilde_in_path(&self.ssh_config.miner_node_key_path)
    }

    /// Remove all validator keys from a node's authorized_keys file
    async fn remove_all_validator_keys_on_node(
        &self,
        connection_details: &SshConnectionDetails,
        host: &str,
    ) -> Result<()> {
        debug!("Removing all validator keys from node {}", host);

        let ssh_command = format!(
            "{} && {}",
            SSH_REWRITE_AUTHORIZED_KEYS_BASE, SSH_MOVE_TO_AUTHORIZED_KEYS
        );

        self.ssh_client
            .execute_command(connection_details, &ssh_command, true)
            .await
            .map_err(|e| {
                anyhow::anyhow!("Failed to remove validator keys from node {}: {}", host, e)
            })?;

        debug!("Successfully removed all validator keys from node {}", host);
        Ok(())
    }

    /// Set exclusive validator key on a node (removes all other validators, adds current one)
    async fn set_exclusive_validator_key_on_node(
        &self,
        connection_details: &SshConnectionDetails,
        host: &str,
        validator_hotkey: &str,
        normalized_key: &str,
    ) -> Result<()> {
        // Atomic operation: remove all validator keys and add the new one
        // Escape single quotes in the SSH key for safe shell interpolation
        let escaped_key = normalized_key.replace("'", "'\\''");
        let ssh_command = format!(
            r#"{} && printf '%s\n' '{}' >> "$tmp" && {}"#,
            SSH_REWRITE_AUTHORIZED_KEYS_BASE, escaped_key, SSH_MOVE_TO_AUTHORIZED_KEYS
        );

        self.ssh_client
            .execute_command(connection_details, &ssh_command, true)
            .await
            .map_err(|e| {
                anyhow::anyhow!(
                    "Failed to set exclusive validator key on node {}: {}",
                    host,
                    e
                )
            })?;

        info!(
            "Successfully set exclusive access for validator {} on node {}",
            validator_hotkey, host
        );
        Ok(())
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[tokio::test]
    async fn test_node_registration() {
        let manager = NodeManager::new(NodeSshConfig::default());

        let config = NodeConfig {
            host: "192.168.1.100".to_string(),
            port: 22,
            username: "basilica".to_string(),
            gpu_category: "H100".to_string(),
            gpu_count: 8,
            additional_opts: None,
        };

        manager.register_node(config.clone()).await.unwrap();

        let nodes = manager.list_nodes().await.unwrap();
        assert_eq!(nodes.len(), 1);
        assert_eq!(nodes[0].config.host, "192.168.1.100");

        let node = manager.get_node("192.168.1.100").await.unwrap();
        assert!(node.is_some());
        assert_eq!(node.unwrap().host, "192.168.1.100");
    }

    #[tokio::test]
    async fn test_validator_assignment() {
        let manager = NodeManager::new(NodeSshConfig::default());

        let validator_key = "validator-hotkey-123";

        // Set assigned validator
        manager.set_assigned_validator(validator_key).await;

        assert!(manager.is_validator_authorized(validator_key).await);
        assert!(!manager.is_validator_authorized("unknown-validator").await);
    }

    #[tokio::test]
    async fn test_deploy_validator_keys_without_nodes() {
        let manager = NodeManager::new(NodeSshConfig::default());

        let validator_key = "validator-123";
        // Valid ed25519 test key
        let ssh_key = "ssh-ed25519 AAAAC3NzaC1lZDI1NTE5AAAAIOMqqnkVzrm0SdG6UOoqKLsabgH5C9okWi0dh2l9GKJl test@example";

        // Without any nodes, this should succeed but not deploy anywhere
        let result = manager.deploy_validator_keys(validator_key, ssh_key).await;
        assert!(result.is_ok());
    }
}
