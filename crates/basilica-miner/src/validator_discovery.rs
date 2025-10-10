//! Validator discovery and assignment module
//!
//! This module handles discovering active validators and selecting the single
//! validator that should receive all managed nodes.

use anyhow::Result;
use async_trait::async_trait;
use std::sync::Arc;
use tracing::{debug, info, warn};

use crate::node_manager::{NodeManager, RegisteredNode};

/// Information about a validator
#[derive(Debug, Clone)]
pub struct ValidatorInfo {
    pub uid: u16,
    pub hotkey: String,
    pub coldkey: String,
    pub stake: u128,
    pub trust: f64,
    pub consensus: f64,
    pub validator_permit: bool,
}

/// Validator discovery service
pub struct ValidatorDiscovery {
    bittensor_service: Arc<bittensor::Service>,
    node_manager: Arc<NodeManager>,
    assignment_strategy: Box<dyn AssignmentStrategy>,
    netuid: u16,
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
    ) -> Self {
        Self {
            bittensor_service,
            node_manager,
            assignment_strategy,
            netuid,
        }
    }

    /// Get list of active validators from the metagraph
    pub async fn get_active_validators(&self) -> Result<Vec<ValidatorInfo>> {
        let metagraph = self.bittensor_service.get_metagraph(self.netuid).await?;

        let validators: Vec<ValidatorInfo> = (0..metagraph.hotkeys.len())
            .filter_map(|idx| {
                let uid = idx as u16;
                let is_validator = metagraph
                    .validator_permit
                    .get(idx)
                    .copied()
                    .unwrap_or(false);

                if is_validator {
                    let hotkey = metagraph.hotkeys.get(idx)?.to_string();
                    let coldkey = metagraph
                        .coldkeys
                        .get(idx)
                        .map(|c| c.to_string())
                        .unwrap_or_default();
                    let stake = metagraph
                        .total_stake
                        .get(idx)
                        .map(|s| s.0 as u128)
                        .unwrap_or(0);
                    let trust = metagraph
                        .trust
                        .get(idx)
                        .map(|t| t.0 as f64 / 65535.0)
                        .unwrap_or(0.0);
                    let consensus = metagraph
                        .consensus
                        .get(idx)
                        .map(|c| c.0 as f64 / 65535.0)
                        .unwrap_or(0.0);

                    Some(ValidatorInfo {
                        uid,
                        hotkey,
                        coldkey,
                        stake,
                        trust,
                        consensus,
                        validator_permit: true,
                    })
                } else {
                    None
                }
            })
            .collect();

        debug!("Found {} active validators", validators.len());
        Ok(validators)
    }

    /// Perform discovery and assignment
    pub async fn run_discovery(&self) -> Result<()> {
        info!("Starting validator discovery");

        // Get active validators
        let validators = self.get_active_validators().await?;
        info!("Found {} validators with permits", validators.len());

        // Get available nodes
        let nodes = self.node_manager.list_nodes().await?;
        info!("Found {} available nodes", nodes.len());

        if nodes.is_empty() {
            warn!("No nodes available for assignment");
            return Ok(());
        }

        // Run assignment strategy to select the validator (all nodes go to selected validator)
        let selected_validator = self
            .assignment_strategy
            .select_assignment(validators, nodes)
            .await?;

        if let Some(validator) = selected_validator {
            info!(
                "Assigning all nodes to validator {} (uid: {})",
                validator.hotkey, validator.uid
            );

            // Update assignment in NodeManager (single source of truth)
            self.node_manager
                .set_assigned_validator(&validator.hotkey)
                .await;
        } else {
            info!("No eligible validators found during discovery; no assignment made");
        }

        Ok(())
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
