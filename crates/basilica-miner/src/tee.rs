//! TEE (Trusted Execution Environment) Status Module
//!
//! Tracks TEE capabilities of managed nodes for offering TEE-enabled compute.

use std::collections::HashMap;
use std::sync::Arc;
use tokio::sync::RwLock;
use tracing::{debug, info, warn};

use basilica_common::ssh::{SshConnectionDetails, SshConnectionManager, StandardSshClient};

/// TEE capabilities of a node
#[derive(Debug, Clone, Default)]
pub struct NodeTeeStatus {
    /// Whether the node supports Intel TDX
    pub tdx_available: bool,
    /// TDX device path if available
    pub tdx_device: Option<String>,
    /// Whether GPU CC mode is available
    pub gpu_cc_available: bool,
    /// GPU models with CC capability
    pub cc_gpu_models: Vec<String>,
    /// Last check timestamp
    pub last_checked: Option<chrono::DateTime<chrono::Utc>>,
    /// Error message from last check
    pub error: Option<String>,
}

/// Manages TEE status of nodes
pub struct TeeStatusManager {
    /// Cached TEE status per node
    status_cache: Arc<RwLock<HashMap<String, NodeTeeStatus>>>,
    /// SSH client for checking node capabilities
    ssh_client: Arc<StandardSshClient>,
}

impl TeeStatusManager {
    /// Create a new TEE status manager
    pub fn new(ssh_client: Arc<StandardSshClient>) -> Self {
        Self {
            status_cache: Arc::new(RwLock::new(HashMap::new())),
            ssh_client,
        }
    }

    /// Check TEE capabilities of a node
    pub async fn check_node_tee_status(
        &self,
        node_id: &str,
        connection: &SshConnectionDetails,
    ) -> NodeTeeStatus {
        info!("[TEE] Checking TEE status for node {}", node_id);

        let mut status = NodeTeeStatus::default();
        status.last_checked = Some(chrono::Utc::now());

        // Check TDX availability
        match self.check_tdx_available(connection).await {
            Ok((available, device)) => {
                status.tdx_available = available;
                status.tdx_device = device;
                if available {
                    debug!("[TEE] Node {} has TDX support", node_id);
                }
            }
            Err(e) => {
                warn!("[TEE] Failed to check TDX on node {}: {}", node_id, e);
                status.error = Some(format!("TDX check failed: {}", e));
            }
        }

        // Check GPU CC mode availability
        match self.check_gpu_cc_available(connection).await {
            Ok((available, models)) => {
                status.gpu_cc_available = available;
                status.cc_gpu_models = models;
                if available {
                    debug!("[TEE] Node {} has GPU CC mode support", node_id);
                }
            }
            Err(e) => {
                warn!("[TEE] Failed to check GPU CC on node {}: {}", node_id, e);
                if status.error.is_none() {
                    status.error = Some(format!("GPU CC check failed: {}", e));
                }
            }
        }

        // Cache the status
        {
            let mut cache = self.status_cache.write().await;
            cache.insert(node_id.to_string(), status.clone());
        }

        info!(
            "[TEE] Node {} TEE status: tdx={}, gpu_cc={}",
            node_id, status.tdx_available, status.gpu_cc_available
        );

        status
    }

    /// Get cached TEE status for a node
    pub async fn get_cached_status(&self, node_id: &str) -> Option<NodeTeeStatus> {
        let cache = self.status_cache.read().await;
        cache.get(node_id).cloned()
    }

    /// Get all TEE-enabled nodes
    pub async fn get_tee_enabled_nodes(&self) -> Vec<String> {
        let cache = self.status_cache.read().await;
        cache
            .iter()
            .filter(|(_, status)| status.tdx_available || status.gpu_cc_available)
            .map(|(node_id, _)| node_id.clone())
            .collect()
    }

    /// Check if node has full TEE (TDX + GPU CC)
    pub async fn is_full_tee_node(&self, node_id: &str) -> bool {
        let cache = self.status_cache.read().await;
        cache
            .get(node_id)
            .map(|s| s.tdx_available && s.gpu_cc_available)
            .unwrap_or(false)
    }

    /// Check TDX availability via SSH
    async fn check_tdx_available(
        &self,
        connection: &SshConnectionDetails,
    ) -> anyhow::Result<(bool, Option<String>)> {
        let check_cmd = r#"
            if [ -c /dev/tdx_guest ]; then
                echo "TDX_AVAILABLE:/dev/tdx_guest"
            elif [ -c /dev/tdx-guest ]; then
                echo "TDX_AVAILABLE:/dev/tdx-guest"
            elif [ -d /sys/firmware/tdx ]; then
                echo "TDX_FIRMWARE_PRESENT"
            else
                echo "TDX_NOT_AVAILABLE"
            fi
        "#;

        let output = self
            .ssh_client
            .execute_command(connection, check_cmd, true)
            .await?;

        let output = output.trim();
        if output.starts_with("TDX_AVAILABLE:") {
            let device = output.strip_prefix("TDX_AVAILABLE:").unwrap_or("");
            Ok((true, Some(device.to_string())))
        } else if output == "TDX_FIRMWARE_PRESENT" {
            // TDX firmware present but guest device not loaded
            Ok((false, None))
        } else {
            Ok((false, None))
        }
    }

    /// Check GPU CC mode availability via SSH
    async fn check_gpu_cc_available(
        &self,
        connection: &SshConnectionDetails,
    ) -> anyhow::Result<(bool, Vec<String>)> {
        let check_cmd = r#"
            if command -v nvidia-smi &>/dev/null; then
                nvidia-smi --query-gpu=name,gpu_uuid --format=csv,noheader 2>/dev/null | while read line; do
                    gpu_name=$(echo "$line" | cut -d',' -f1 | xargs)
                    # Check if GPU supports CC mode (H100, H200)
                    if echo "$gpu_name" | grep -qE 'H100|H200'; then
                        echo "CC_GPU:$gpu_name"
                    fi
                done
                
                # Check actual CC mode status
                cc_status=$(nvidia-smi -q 2>/dev/null | grep -i 'Conf Compute Mode' | head -1)
                if echo "$cc_status" | grep -qi 'enabled'; then
                    echo "CC_MODE_ENABLED"
                fi
            else
                echo "NO_NVIDIA_SMI"
            fi
        "#;

        let output = self
            .ssh_client
            .execute_command(connection, check_cmd, true)
            .await?;

        let mut cc_enabled = false;
        let mut models = Vec::new();

        for line in output.lines() {
            let line = line.trim();
            if line.starts_with("CC_GPU:") {
                if let Some(model) = line.strip_prefix("CC_GPU:") {
                    models.push(model.to_string());
                }
            } else if line == "CC_MODE_ENABLED" {
                cc_enabled = true;
            }
        }

        // CC available if we have CC-capable GPUs and CC mode is enabled
        let available = cc_enabled && !models.is_empty();
        Ok((available, models))
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use basilica_common::ssh::SshConnectionConfig;

    #[tokio::test]
    async fn test_status_caching() {
        let ssh_client = Arc::new(StandardSshClient::with_config(
            SshConnectionConfig::default(),
        ));
        let manager = TeeStatusManager::new(ssh_client);

        // No cached status initially
        assert!(manager.get_cached_status("node-1").await.is_none());

        // Manually insert a status for testing
        {
            let mut cache = manager.status_cache.write().await;
            cache.insert(
                "node-1".to_string(),
                NodeTeeStatus {
                    tdx_available: true,
                    gpu_cc_available: true,
                    ..Default::default()
                },
            );
        }

        // Should be cached now
        let status = manager.get_cached_status("node-1").await;
        assert!(status.is_some());
        assert!(status.unwrap().tdx_available);
    }

    #[tokio::test]
    async fn test_get_tee_enabled_nodes() {
        let ssh_client = Arc::new(StandardSshClient::with_config(
            SshConnectionConfig::default(),
        ));
        let manager = TeeStatusManager::new(ssh_client);

        // Insert test data
        {
            let mut cache = manager.status_cache.write().await;
            cache.insert(
                "node-1".to_string(),
                NodeTeeStatus {
                    tdx_available: true,
                    gpu_cc_available: true,
                    ..Default::default()
                },
            );
            cache.insert(
                "node-2".to_string(),
                NodeTeeStatus {
                    tdx_available: false,
                    gpu_cc_available: false,
                    ..Default::default()
                },
            );
            cache.insert(
                "node-3".to_string(),
                NodeTeeStatus {
                    tdx_available: true,
                    gpu_cc_available: false,
                    ..Default::default()
                },
            );
        }

        let tee_nodes = manager.get_tee_enabled_nodes().await;
        assert_eq!(tee_nodes.len(), 2);
        assert!(tee_nodes.contains(&"node-1".to_string()));
        assert!(tee_nodes.contains(&"node-3".to_string()));
    }
}
