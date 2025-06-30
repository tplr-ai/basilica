//! Validator discovery service using Bittensor metagraph

use crate::{config::Config, error::Result};
use bittensor::{Metagraph, Service as BittensorService};
use dashmap::DashMap;
use serde::{Deserialize, Serialize};
use std::sync::Arc;
use tokio::time::interval;
use tracing::{debug, error, info, warn};

/// Validator information structure
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct ValidatorInfo {
    /// Validator unique identifier (UID in the subnet)
    pub uid: u16,

    /// Validator hotkey (SS58 address)
    pub hotkey: String,

    /// Validator API endpoint URL
    pub endpoint: String,

    /// Validator score/weight in the network
    pub score: f64,

    /// Is the validator currently active
    pub is_active: bool,

    /// Last health check timestamp
    pub last_health_check: Option<chrono::DateTime<chrono::Utc>>,

    /// Health status
    pub is_healthy: bool,

    /// Number of consecutive failures
    pub failure_count: u32,
}

/// Validator discovery service
pub struct ValidatorDiscovery {
    /// Bittensor service for metagraph queries
    bittensor_service: Arc<BittensorService>,

    /// Configuration
    config: Arc<Config>,

    /// Discovered validators (UID -> ValidatorInfo)
    validators: Arc<DashMap<u16, ValidatorInfo>>,

    /// Health monitor
    health_monitor: Arc<super::HealthMonitor>,
}

impl ValidatorDiscovery {
    /// Create a new validator discovery service
    pub fn new(bittensor_service: Arc<BittensorService>, config: Arc<Config>) -> Self {
        let validators = Arc::new(DashMap::new());
        let health_monitor = Arc::new(super::HealthMonitor::new(
            validators.clone(),
            config.clone(),
        ));

        Self {
            bittensor_service,
            config,
            validators,
            health_monitor,
        }
    }

    /// Start the discovery loop
    pub async fn start_discovery_loop(&self) {
        info!("Starting validator discovery loop");

        let mut discovery_interval = interval(self.config.discovery_interval());

        // Start health monitoring in background
        let health_monitor = self.health_monitor.clone();
        tokio::spawn(async move {
            health_monitor.start_monitoring().await;
        });

        loop {
            discovery_interval.tick().await;

            match self.discover_validators().await {
                Ok(count) => {
                    info!("Discovered {} validators", count);
                }
                Err(e) => {
                    error!("Validator discovery failed: {}", e);
                }
            }
        }
    }

    /// Discover validators from the Bittensor metagraph
    async fn discover_validators(&self) -> Result<usize> {
        debug!("Fetching metagraph for validator discovery");

        let metagraph = self
            .bittensor_service
            .get_metagraph(self.config.bittensor.netuid)
            .await?;

        let mut discovered_count = 0;

        // Process neurons from the metagraph
        // The metagraph has various fields like axons, neurons_info, etc.
        // We need to iterate through active validators
        let n_neurons = metagraph.active.len();
        for uid in 0..n_neurons {
            // Check if this neuron is active and a validator
            if let (Some(active), Some(validator_permit)) = (
                metagraph.active.get(uid),
                metagraph.validator_permit.get(uid),
            ) {
                if *active && *validator_permit {
                    // Get axon info for this UID
                    if let Some(axon_info) = metagraph.axons.get(uid) {
                        match self
                            .process_validator_axon(uid as u16, axon_info, &metagraph)
                            .await
                        {
                            Ok(true) => discovered_count += 1,
                            Ok(false) => {
                                debug!("Skipped validator {} (below score threshold)", uid);
                            }
                            Err(e) => {
                                warn!("Failed to process validator {}: {}", uid, e);
                            }
                        }
                    }
                }
            }
        }

        // Clean up inactive validators
        self.cleanup_inactive_validators();

        Ok(discovered_count)
    }

    /// Process a validator axon from the metagraph
    async fn process_validator_axon(
        &self,
        uid: u16,
        axon_info: &bittensor::AxonInfo,
        metagraph: &Metagraph<bittensor::AccountId>,
    ) -> Result<bool> {
        // Calculate validator score (could be based on stake, trust, or other metrics)
        let score = self.calculate_validator_score(uid, metagraph);

        // Check minimum score threshold
        if score < self.config.bittensor.min_validator_score {
            return Ok(false);
        }

        // Get validator endpoint from axon info
        let endpoint = format!("http://{}:{}", axon_info.ip, axon_info.port);

        // Get hotkey for this UID
        let hotkey = if let Some(key) = metagraph.hotkeys.get(uid as usize) {
            key.to_string()
        } else {
            warn!("No hotkey found for validator {}", uid);
            return Ok(false);
        };

        // Create or update validator info
        let validator_info = ValidatorInfo {
            uid,
            hotkey,
            endpoint,
            score,
            is_active: true,
            last_health_check: None,
            is_healthy: false, // Will be updated by health monitor
            failure_count: 0,
        };

        debug!("Discovered validator {}: {}", uid, validator_info.endpoint);
        self.validators.insert(uid, validator_info);

        Ok(true)
    }

    /// Calculate validator score based on various metrics
    fn calculate_validator_score(
        &self,
        uid: u16,
        metagraph: &Metagraph<bittensor::AccountId>,
    ) -> f64 {
        // For now, use a simple scoring mechanism
        // In production, this would use stake, trust, and other metrics from the metagraph
        // Since we don't have access to the exact metagraph fields, we'll use a placeholder

        // Check if validator has permit
        let has_permit = metagraph
            .validator_permit
            .get(uid as usize)
            .copied()
            .unwrap_or(false);

        if has_permit {
            // Return a score between 0.5 and 1.0 for validators with permits
            0.75
        } else {
            0.0
        }
    }

    /// Clean up validators that are no longer in the metagraph
    fn cleanup_inactive_validators(&self) {
        let mut to_remove = Vec::new();

        for entry in self.validators.iter() {
            if !entry.is_active {
                to_remove.push(*entry.key());
            }
        }

        for uid in to_remove {
            info!("Removing inactive validator {}", uid);
            self.validators.remove(&uid);
        }
    }

    /// Get all healthy validators
    pub fn get_healthy_validators(&self) -> Vec<ValidatorInfo> {
        self.validators
            .iter()
            .filter(|entry| entry.is_healthy)
            .map(|entry| entry.value().clone())
            .collect()
    }

    /// Get a specific validator by UID
    pub fn get_validator(&self, uid: u16) -> Option<ValidatorInfo> {
        self.validators.get(&uid).map(|entry| entry.value().clone())
    }

    /// Get all validators (including unhealthy ones)
    pub fn get_all_validators(&self) -> Vec<ValidatorInfo> {
        self.validators
            .iter()
            .map(|entry| entry.value().clone())
            .collect()
    }

    /// Get validator count
    pub fn validator_count(&self) -> usize {
        self.validators.len()
    }

    /// Get healthy validator count
    pub fn healthy_validator_count(&self) -> usize {
        self.validators
            .iter()
            .filter(|entry| entry.is_healthy)
            .count()
    }

    /// Update validator health status
    pub fn update_validator_health(&self, uid: u16, is_healthy: bool) {
        if let Some(mut validator) = self.validators.get_mut(&uid) {
            validator.is_healthy = is_healthy;
            validator.last_health_check = Some(chrono::Utc::now());

            if !is_healthy {
                validator.failure_count += 1;
            } else {
                validator.failure_count = 0;
            }
        }
    }
}
