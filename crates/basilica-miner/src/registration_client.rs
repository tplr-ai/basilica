//! Miner Registration Client
//!
//! Low-level gRPC client for miner→validator communication:
//! - RegisterBid: Register nodes with SSH details + pricing
//! - HealthCheck: Periodic heartbeat to keep registrations active
//! - UpdateBid: Update pricing when it changes
//! - RemoveBid: Unregister nodes on shutdown
//! - SSH key deployment to nodes
//!
//! Note: The registration lifecycle (registration, health checks, price updates)
//! is orchestrated by BidManager. This module provides the underlying gRPC calls.

use std::sync::Arc;
use std::time::Duration;

use anyhow::{Context, Result};
use basilica_protocol::miner_discovery::{
    miner_registration_client::MinerRegistrationClient, HealthCheckRequest, NodeRegistration,
    RegisterBidRequest, RemoveBidRequest, UpdateBidRequest,
};
use chrono::Utc;
use tokio::sync::RwLock;
use tonic::transport::Channel;
use tracing::{debug, info, warn};

use crate::node_manager::NodeManager;

/// Trait for Bittensor service operations needed for registration
pub trait BittensorServiceApi: Send + Sync {
    fn get_account_id(&self) -> String;
    fn sign_data(&self, data: &[u8]) -> Result<String>;
}

impl BittensorServiceApi for bittensor::Service {
    fn get_account_id(&self) -> String {
        self.get_account_id().to_string()
    }

    fn sign_data(&self, data: &[u8]) -> Result<String> {
        self.sign_data(data).map_err(|e| anyhow::anyhow!(e))
    }
}

/// Registration state tracking
#[derive(Debug, Clone)]
pub struct RegistrationState {
    /// Whether registration was successful
    pub registered: bool,
    /// Registration ID from validator
    pub registration_id: Option<String>,
    /// Validator's SSH public key to deploy
    pub validator_ssh_public_key: Option<String>,
    /// Health check interval from validator
    pub health_check_interval_secs: u32,
}

impl Default for RegistrationState {
    fn default() -> Self {
        Self {
            registered: false,
            registration_id: None,
            validator_ssh_public_key: None,
            health_check_interval_secs: 60,
        }
    }
}

/// Miner Registration Client
pub struct RegistrationClient {
    request_timeout: Duration,
    node_manager: Arc<NodeManager>,
    bittensor_service: Arc<dyn BittensorServiceApi>,
    miner_hotkey: String,
    state: Arc<RwLock<RegistrationState>>,
}

impl RegistrationClient {
    pub fn new(
        request_timeout: Duration,
        node_manager: Arc<NodeManager>,
        bittensor_service: Arc<dyn BittensorServiceApi>,
    ) -> Self {
        let miner_hotkey = bittensor_service.get_account_id();
        Self {
            request_timeout,
            node_manager,
            bittensor_service,
            miner_hotkey,
            state: Arc::new(RwLock::new(RegistrationState::default())),
        }
    }

    /// Connect to the validator's registration service at the given endpoint
    async fn connect(&self, endpoint: &str) -> Result<MinerRegistrationClient<Channel>> {
        let channel = Channel::from_shared(endpoint.to_string())
            .context("invalid endpoint URL")?
            .timeout(self.request_timeout)
            .connect()
            .await
            .context("failed to connect to validator")?;

        Ok(MinerRegistrationClient::new(channel))
    }

    /// Build and sign the message for RegisterBid
    fn build_register_bid_message(&self, timestamp: i64) -> String {
        format!("{}|{}", self.miner_hotkey.trim(), timestamp)
    }

    /// Build and sign the message for UpdateBid
    fn build_update_bid_message(
        &self,
        host: &str,
        hourly_rate_cents: u32,
        timestamp: i64,
    ) -> String {
        format!(
            "{}|{}|{}|{}",
            self.miner_hotkey.trim(),
            host.trim(),
            hourly_rate_cents,
            timestamp,
        )
    }

    /// Build and sign the message for RemoveBid
    fn build_remove_bid_message(&self, hosts: &[String], timestamp: i64) -> String {
        let hosts_str = hosts.join(",");
        format!("{}|{}|{}", self.miner_hotkey.trim(), hosts_str, timestamp,)
    }

    /// Build and sign the message for HealthCheck
    fn build_health_check_message(&self, hosts: &[String], timestamp: i64) -> String {
        let hosts_str = hosts.join(",");
        format!("{}|{}|{}", self.miner_hotkey.trim(), hosts_str, timestamp)
    }

    /// Sign a message using the miner's hotkey
    fn sign_message(&self, message: &str) -> Result<Vec<u8>> {
        let signature_hex = self
            .bittensor_service
            .sign_data(message.as_bytes())
            .context("failed to sign message")?;
        let signature_bytes =
            hex::decode(signature_hex).context("failed to decode signature hex")?;
        Ok(signature_bytes)
    }

    /// Register nodes with the validator using pre-built registrations.
    /// This is the primary registration method - BidManager builds the registrations
    /// with prices from BiddingConfig.
    pub async fn register_nodes_with_registrations(
        &self,
        grpc_endpoint: &str,
        node_registrations: Vec<NodeRegistration>,
    ) -> Result<RegistrationState> {
        let mut client = self.connect(grpc_endpoint).await?;

        // Build and sign request
        let timestamp = Utc::now().timestamp();
        let message = self.build_register_bid_message(timestamp);
        let signature = self.sign_message(&message)?;

        let node_count = node_registrations.len();
        let request = RegisterBidRequest {
            miner_hotkey: self.miner_hotkey.clone(),
            nodes: node_registrations,
            timestamp,
            signature,
        };

        info!(
            miner_hotkey = %self.miner_hotkey,
            node_count = node_count,
            "Registering nodes with validator"
        );
        if node_count == 0 {
            warn!("Sending zero-node RegisterBid to deactivate all existing bids");
        }

        let response = client
            .register_bid(request)
            .await
            .map_err(|status| {
                anyhow::anyhow!(
                    "RegisterBid RPC failed (code: {}, message: {})",
                    status.code(),
                    status.message(),
                )
            })?
            .into_inner();

        if !response.accepted {
            return Err(anyhow::anyhow!(
                "Registration rejected: {}",
                response.error_message
            ));
        }

        let state = RegistrationState {
            registered: true,
            registration_id: Some(response.registration_id.clone()),
            validator_ssh_public_key: if response.validator_ssh_public_key.is_empty() {
                None
            } else {
                Some(response.validator_ssh_public_key.clone())
            },
            health_check_interval_secs: response.health_check_interval_secs,
        };

        // Update internal state
        *self.state.write().await = state.clone();

        info!(
            registration_id = %response.registration_id,
            health_check_interval_secs = response.health_check_interval_secs,
            has_ssh_key = !response.validator_ssh_public_key.is_empty(),
            "Successfully registered with validator"
        );

        // Log collateral status if present
        if let Some(status) = &response.collateral_status {
            match status.status.as_str() {
                "warning" | "undercollateralized" | "excluded" => {
                    warn!(
                        status = %status.status,
                        current_usd = status.current_usd_value,
                        min_required = status.minimum_usd_required,
                        action = %status.action_required,
                        "Collateral status requires attention"
                    );
                }
                _ => {
                    info!(
                        status = %status.status,
                        current_usd = status.current_usd_value,
                        "Collateral status OK"
                    );
                }
            }
        }

        Ok(state)
    }

    /// Deploy validator's SSH key to all nodes.
    /// Called after successful registration.
    pub async fn deploy_validator_ssh_key(&self) -> Result<()> {
        let state = self.state.read().await;
        let ssh_key = match &state.validator_ssh_public_key {
            Some(key) if !key.is_empty() => key.clone(),
            _ => {
                info!("No validator SSH key to deploy");
                return Ok(());
            }
        };
        drop(state); // Release lock before async operation

        info!("Deploying validator SSH key to nodes");

        // Use node manager to deploy SSH keys
        // Note: This assumes node_manager has a method to deploy SSH keys
        // The validator_hotkey is used as identifier for the key
        self.node_manager
            .deploy_validator_keys("validator", &ssh_key)
            .await
            .context("failed to deploy validator SSH key to nodes")?;

        info!("Successfully deployed validator SSH key to all nodes");
        Ok(())
    }

    /// Send periodic health check to validator.
    pub async fn send_health_check(&self, grpc_endpoint: &str) -> Result<u32> {
        let state = self.state.read().await;
        if !state.registered {
            return Err(anyhow::anyhow!("not registered yet"));
        }
        drop(state);

        let mut client = self.connect(grpc_endpoint).await?;

        let timestamp = Utc::now().timestamp();
        let hosts: Vec<String> = vec![]; // Empty = all nodes
        let message = self.build_health_check_message(&hosts, timestamp);
        let signature = self.sign_message(&message)?;

        let request = HealthCheckRequest {
            miner_hotkey: self.miner_hotkey.clone(),
            hosts,
            timestamp,
            signature,
        };

        let response = client
            .health_check(request)
            .await
            .map_err(|status| {
                anyhow::anyhow!(
                    "HealthCheck RPC failed (code: {}, message: {})",
                    status.code(),
                    status.message(),
                )
            })?
            .into_inner();

        if !response.accepted {
            return Err(anyhow::anyhow!(
                "Health check rejected: {}",
                response.error_message
            ));
        }

        debug!(
            nodes_active = response.nodes_active,
            "Health check successful"
        );
        Ok(response.nodes_active)
    }

    /// Update price for a specific node.
    pub async fn update_node_price(
        &self,
        grpc_endpoint: &str,
        host: &str,
        hourly_rate_cents: u32,
    ) -> Result<()> {
        let state = self.state.read().await;
        if !state.registered {
            return Err(anyhow::anyhow!("not registered yet"));
        }
        drop(state);

        let mut client = self.connect(grpc_endpoint).await?;

        let timestamp = Utc::now().timestamp();
        let message = self.build_update_bid_message(host, hourly_rate_cents, timestamp);
        let signature = self.sign_message(&message)?;

        let request = UpdateBidRequest {
            miner_hotkey: self.miner_hotkey.clone(),
            hourly_rate_cents,
            timestamp,
            signature,
            host: host.to_string(),
        };

        let response = client
            .update_bid(request)
            .await
            .map_err(|status| {
                anyhow::anyhow!(
                    "UpdateBid RPC failed (code: {}, message: {})",
                    status.code(),
                    status.message(),
                )
            })?
            .into_inner();

        if !response.accepted {
            return Err(anyhow::anyhow!(
                "Price update rejected: {}",
                response.error_message
            ));
        }

        info!(
            host = host,
            hourly_rate_cents = hourly_rate_cents,
            "Updated node price"
        );
        Ok(())
    }

    /// Remove nodes from availability.
    /// If hosts is empty, removes all nodes.
    pub async fn remove_nodes(&self, grpc_endpoint: &str, hosts: Vec<String>) -> Result<u32> {
        let state = self.state.read().await;
        if !state.registered {
            return Err(anyhow::anyhow!("not registered yet"));
        }
        drop(state);

        let mut client = self.connect(grpc_endpoint).await?;

        let timestamp = Utc::now().timestamp();
        let message = self.build_remove_bid_message(&hosts, timestamp);
        let signature = self.sign_message(&message)?;

        let request = RemoveBidRequest {
            miner_hotkey: self.miner_hotkey.clone(),
            timestamp,
            signature,
            hosts,
        };

        let response = client
            .remove_bid(request)
            .await
            .map_err(|status| {
                anyhow::anyhow!(
                    "RemoveBid RPC failed (code: {}, message: {})",
                    status.code(),
                    status.message(),
                )
            })?
            .into_inner();

        if !response.accepted {
            return Err(anyhow::anyhow!(
                "Node removal rejected: {}",
                response.error_message
            ));
        }

        info!(nodes_removed = response.nodes_removed, "Removed nodes");
        Ok(response.nodes_removed)
    }

    /// Get current registration state
    pub async fn get_state(&self) -> RegistrationState {
        self.state.read().await.clone()
    }
}
#[cfg(test)]
mod tests {
    use super::*;

    struct MockBittensorService {
        hotkey: String,
    }

    impl BittensorServiceApi for MockBittensorService {
        fn get_account_id(&self) -> String {
            self.hotkey.clone()
        }

        fn sign_data(&self, _data: &[u8]) -> Result<String> {
            // Return a fake signature for testing
            Ok("aabbccdd".to_string())
        }
    }

    #[test]
    fn test_build_register_bid_message() {
        let node_manager = Arc::new(NodeManager::new(crate::config::NodeSshConfig::default()));
        let bittensor = Arc::new(MockBittensorService {
            hotkey: "5GrwvaEF".to_string(),
        });

        let client = RegistrationClient::new(Duration::from_secs(30), node_manager, bittensor);
        let message = client.build_register_bid_message(1234567890);
        assert_eq!(message, "5GrwvaEF|1234567890");
    }

    #[test]
    fn test_build_health_check_message() {
        let node_manager = Arc::new(NodeManager::new(crate::config::NodeSshConfig::default()));
        let bittensor = Arc::new(MockBittensorService {
            hotkey: "5GrwvaEF".to_string(),
        });

        let client = RegistrationClient::new(Duration::from_secs(30), node_manager, bittensor);
        // Test with empty hosts (all nodes)
        let message = client.build_health_check_message(&[], 1234567890);
        assert_eq!(message, "5GrwvaEF||1234567890");

        // Test with specific hosts
        let message = client.build_health_check_message(
            &["192.168.1.1".to_string(), "192.168.1.2".to_string()],
            1234567890,
        );
        assert_eq!(message, "5GrwvaEF|192.168.1.1,192.168.1.2|1234567890");
    }

    #[test]
    fn test_build_update_bid_message() {
        let node_manager = Arc::new(NodeManager::new(crate::config::NodeSshConfig::default()));
        let bittensor = Arc::new(MockBittensorService {
            hotkey: "5GrwvaEF".to_string(),
        });

        let client = RegistrationClient::new(Duration::from_secs(30), node_manager, bittensor);
        let message = client.build_update_bid_message("192.168.1.1", 250, 1234567890);
        assert_eq!(message, "5GrwvaEF|192.168.1.1|250|1234567890");
    }

    #[test]
    fn test_build_remove_bid_message() {
        let node_manager = Arc::new(NodeManager::new(crate::config::NodeSshConfig::default()));
        let bittensor = Arc::new(MockBittensorService {
            hotkey: "5GrwvaEF".to_string(),
        });

        let client = RegistrationClient::new(Duration::from_secs(30), node_manager, bittensor);
        let message = client.build_remove_bid_message(
            &["192.168.1.1".to_string(), "192.168.1.2".to_string()],
            1234567890,
        );
        assert_eq!(message, "5GrwvaEF|192.168.1.1,192.168.1.2|1234567890");
    }
}
