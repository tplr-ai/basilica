//! Validator discovery and assignment module
//!
//! This module handles discovering active validators and selecting the single
//! validator that should receive all managed nodes. The gRPC endpoint for bid
//! registration is constructed using the validator's axon IP and a configured
//! bidding gRPC port (default: 50052).

use anyhow::Result;
use async_trait::async_trait;
use std::sync::Arc;
use tracing::{debug, info, warn};

use crate::node_manager::{NodeManager, RegisteredNode};

fn require_selected_validator(selected: Option<ValidatorInfo>) -> Result<ValidatorInfo> {
    selected.ok_or_else(|| anyhow::anyhow!("No eligible validators found during discovery"))
}

/// Information about a validator
#[derive(Debug, Clone)]
pub struct ValidatorInfo {
    pub uid: u16,
    pub hotkey: String,
    pub coldkey: String,
    pub stake: u64,
    /// Axon endpoint from metagraph (e.g., "http://1.2.3.4:8080")
    pub axon_endpoint: Option<String>,
}

/// A fully discovered validator with a known gRPC endpoint
#[derive(Debug, Clone)]
pub struct DiscoveredValidator {
    pub hotkey: String,
    pub grpc_endpoint: String,
}

/// Validator discovery service
pub struct ValidatorDiscovery {
    bittensor_service: Arc<bittensor::Service>,
    node_manager: Arc<NodeManager>,
    assignment_strategy: Box<dyn AssignmentStrategy>,
    netuid: u16,
    bid_grpc_port: u16,
}

/// Strategy for assigning nodes to validators
#[async_trait]
pub trait AssignmentStrategy: Send + Sync {
    /// Select validator to assign nodes to (all nodes are assigned to the selected validator)
    async fn select_assignment(
        &self,
        validators: Vec<ValidatorInfo>,
        nodes: Vec<RegisteredNode>,
    ) -> Result<Option<ValidatorInfo>>;
}

impl ValidatorDiscovery {
    /// Create new validator discovery service
    pub fn new(
        bittensor_service: Arc<bittensor::Service>,
        node_manager: Arc<NodeManager>,
        assignment_strategy: Box<dyn AssignmentStrategy>,
        netuid: u16,
        bid_grpc_port: u16,
    ) -> Self {
        Self {
            bittensor_service,
            node_manager,
            assignment_strategy,
            netuid,
            bid_grpc_port,
        }
    }

    /// Get list of active validators from the metagraph
    pub async fn get_active_validators(&self) -> Result<Vec<ValidatorInfo>> {
        let metagraph = self.bittensor_service.get_metagraph(self.netuid).await?;
        let discovery = bittensor::NeuronDiscovery::new(&metagraph);

        let neurons = discovery.get_validators()?;
        let validators: Vec<ValidatorInfo> = neurons
            .into_iter()
            .map(|n| {
                let axon_endpoint = n.axon_info.map(|a| format!("http://{}:{}", a.ip, a.port));

                ValidatorInfo {
                    uid: n.uid,
                    hotkey: n.hotkey,
                    coldkey: n.coldkey,
                    stake: n.stake,
                    axon_endpoint,
                }
            })
            .collect();

        debug!("Found {} active validators", validators.len());
        Ok(validators)
    }

    /// Perform discovery and assignment, returning the discovered validator.
    pub async fn run_discovery(&self) -> Result<DiscoveredValidator> {
        info!("Starting validator discovery");

        // Get active validators
        let validators = self.get_active_validators().await?;
        info!("Found {} validators with permits", validators.len());

        // Get available nodes
        let nodes = self.node_manager.list_nodes().await?;
        info!("Found {} available nodes", nodes.len());
        let has_nodes = !nodes.is_empty();

        if !has_nodes {
            warn!(
                "No nodes available for assignment; continuing discovery to enable zero-node bid deactivation"
            );
        }

        // Run assignment strategy to select the validator (all nodes go to selected validator)
        let selected_validator = self
            .assignment_strategy
            .select_assignment(validators, nodes)
            .await?;

        let validator = require_selected_validator(selected_validator)?;

        if !has_nodes {
            info!(
                "Selected validator {} (uid: {}) for zero-node deactivation flow",
                validator.hotkey, validator.uid
            );
        } else {
            info!(
                "Assigning all nodes to validator {} (uid: {})",
                validator.hotkey, validator.uid
            );
        }

        // Update assignment in NodeManager (single source of truth)
        self.node_manager
            .set_assigned_validator(&validator.hotkey)
            .await;

        // Construct gRPC endpoint from the validator's axon IP and configured bid_grpc_port
        let axon_endpoint = validator.axon_endpoint.as_ref().ok_or_else(|| {
            anyhow::anyhow!(
                "Validator {} has no axon endpoint; cannot construct gRPC endpoint",
                validator.hotkey
            )
        })?;

        let axon_url = url::Url::parse(axon_endpoint)
            .map_err(|e| anyhow::anyhow!("Invalid axon endpoint URL: {}", e))?;
        let host = axon_url
            .host_str()
            .ok_or_else(|| anyhow::anyhow!("No host in axon endpoint"))?;
        let grpc_endpoint = format!("http://{}:{}", host, self.bid_grpc_port);

        let discovered = DiscoveredValidator {
            hotkey: validator.hotkey.clone(),
            grpc_endpoint,
        };

        info!(
            grpc_endpoint = %discovered.grpc_endpoint,
            "Constructed validator gRPC endpoint"
        );

        Ok(discovered)
    }
}

/// Fixed assignment strategy - assigns all nodes to a specific validator
pub struct FixedAssignment {
    validator_hotkey: String,
}

impl FixedAssignment {
    pub fn new(validator_hotkey: String) -> Self {
        Self { validator_hotkey }
    }
}

#[async_trait]
impl AssignmentStrategy for FixedAssignment {
    async fn select_assignment(
        &self,
        validators: Vec<ValidatorInfo>,
        _nodes: Vec<RegisteredNode>,
    ) -> Result<Option<ValidatorInfo>> {
        // Find the validator with the specified hotkey
        if let Some(validator) = validators
            .into_iter()
            .find(|v| v.hotkey == self.validator_hotkey)
        {
            Ok(Some(validator))
        } else {
            warn!(
                "Validator with hotkey {} not found in active validators",
                self.validator_hotkey
            );
            Ok(None)
        }
    }
}

/// Highest stake assignment strategy - assigns all nodes to the validator with highest stake
pub struct HighestStakeAssignment;

#[async_trait]
impl AssignmentStrategy for HighestStakeAssignment {
    async fn select_assignment(
        &self,
        mut validators: Vec<ValidatorInfo>,
        _nodes: Vec<RegisteredNode>,
    ) -> Result<Option<ValidatorInfo>> {
        // Sort by stake (highest first)
        validators.sort_by(|a, b| b.stake.cmp(&a.stake));

        // Take the highest staked validator
        Ok(validators.into_iter().next())
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    fn validator(uid: u16, hotkey: &str, stake: u64) -> ValidatorInfo {
        ValidatorInfo {
            uid,
            hotkey: hotkey.to_string(),
            coldkey: format!("coldkey-{hotkey}"),
            stake,
            axon_endpoint: Some("http://127.0.0.1:8080".to_string()),
        }
    }

    #[tokio::test]
    async fn highest_stake_selects_validator_with_empty_nodes() {
        let strategy = HighestStakeAssignment;
        let validators = vec![
            validator(10, "validator-a", 100),
            validator(11, "validator-b", 250),
        ];

        let selected = strategy
            .select_assignment(validators, vec![])
            .await
            .expect("strategy selection should succeed");

        let selected = require_selected_validator(selected).expect("validator should be selected");
        assert_eq!(selected.hotkey, "validator-b");
    }

    #[tokio::test]
    async fn no_validators_returns_error() {
        let strategy = HighestStakeAssignment;
        let selected = strategy
            .select_assignment(vec![], vec![])
            .await
            .expect("strategy selection should succeed");

        let err = require_selected_validator(selected)
            .expect_err("selection should fail when no validators are available");
        assert_eq!(
            err.to_string(),
            "No eligible validators found during discovery"
        );
    }

    #[tokio::test]
    async fn fixed_assignment_not_found_returns_error() {
        let strategy = FixedAssignment::new("validator-z".to_string());
        let validators = vec![validator(10, "validator-a", 100)];

        let selected = strategy
            .select_assignment(validators, vec![])
            .await
            .expect("strategy selection should succeed");

        let err = require_selected_validator(selected)
            .expect_err("selection should fail when fixed validator is not present");
        assert_eq!(
            err.to_string(),
            "No eligible validators found during discovery"
        );
    }
}
