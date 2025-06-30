//! # Validator Discovery
//!
//! Discovers validators from the Bittensor metagraph and manages executor assignments
//! based on arbitrary miner-defined strategies.

use anyhow::Result;
use async_trait::async_trait;
use std::collections::HashMap;
use std::sync::Arc;
use tokio::sync::RwLock;
use tracing::{debug, info, warn};

use crate::executor_manager::{AvailableExecutor, ExecutorManager};

/// Information about a discovered validator
#[derive(Debug, Clone)]
pub struct ValidatorInfo {
    /// The validator's unique identifier
    pub uid: u16,
    /// The validator's hotkey (SS58 address)
    pub hotkey: String,
    /// The validator's stake amount
    pub stake: u64,
    /// The validator's axon endpoint
    pub axon_endpoint: Option<String>,
}

/// Manages validator discovery and executor assignments
pub struct ValidatorDiscovery {
    bittensor_service: Arc<bittensor::Service>,
    executor_manager: Arc<ExecutorManager>,
    assignment_strategy: Box<dyn AssignmentStrategy>,
    assignments: Arc<RwLock<HashMap<String, Vec<String>>>>, // validator_hotkey -> executor_ids
    netuid: u16,
}

/// Strategy for assigning executors to validators
#[async_trait]
pub trait AssignmentStrategy: Send + Sync {
    /// Assign executors to validators based on custom logic
    async fn assign_executors(
        &self,
        validators: Vec<ValidatorInfo>,
        executors: Vec<AvailableExecutor>,
        current_assignments: &HashMap<String, Vec<String>>,
    ) -> HashMap<String, Vec<String>>;
}

impl ValidatorDiscovery {
    /// Create a new validator discovery service
    pub fn new(
        bittensor_service: Arc<bittensor::Service>,
        executor_manager: Arc<ExecutorManager>,
        assignment_strategy: Box<dyn AssignmentStrategy>,
        netuid: u16,
    ) -> Self {
        Self {
            bittensor_service,
            executor_manager,
            assignment_strategy,
            assignments: Arc::new(RwLock::new(HashMap::new())),
            netuid,
        }
    }

    /// Discover validators and update executor assignments
    pub async fn discover_and_assign(&self) -> Result<()> {
        info!("Starting validator discovery and executor assignment");

        // Fetch metagraph
        let metagraph = self
            .bittensor_service
            .get_metagraph(self.netuid)
            .await
            .map_err(|e| anyhow::anyhow!("Failed to get metagraph: {}", e))?;

        // Use discovery module to find validators
        let discovery = bittensor::NeuronDiscovery::new(&metagraph);
        let validators = discovery.get_validators()?;

        info!("Discovered {} validators from metagraph", validators.len());

        // Convert to ValidatorInfo
        let validator_infos: Vec<ValidatorInfo> = validators
            .into_iter()
            .map(|v| {
                let axon_endpoint = v.axon_info.map(|a| format!("{}:{}", a.ip, a.port));
                ValidatorInfo {
                    uid: v.uid,
                    hotkey: v.hotkey,
                    stake: v.stake,
                    axon_endpoint,
                }
            })
            .collect();

        // Log validators with axon endpoints
        let with_endpoints = validator_infos
            .iter()
            .filter(|v| v.axon_endpoint.is_some())
            .count();
        info!(
            "Found {} validators with axon endpoints out of {} total",
            with_endpoints,
            validator_infos.len()
        );

        // Get available executors
        let executors = self.executor_manager.list_available().await?;
        info!("Found {} available executors", executors.len());

        // Get current assignments
        let current_assignments = self.assignments.read().await.clone();

        // Apply assignment strategy
        let new_assignments = self
            .assignment_strategy
            .assign_executors(validator_infos, executors, &current_assignments)
            .await;

        // Log assignment changes
        for (validator_hotkey, executor_ids) in &new_assignments {
            if let Some(old_ids) = current_assignments.get(validator_hotkey) {
                if old_ids != executor_ids {
                    info!(
                        "Updated assignment for validator {}: {:?} -> {:?}",
                        validator_hotkey, old_ids, executor_ids
                    );
                }
            } else {
                info!(
                    "New assignment for validator {}: {:?}",
                    validator_hotkey, executor_ids
                );
            }
        }

        // Store new assignments
        *self.assignments.write().await = new_assignments;

        info!("Completed validator discovery and executor assignment");
        Ok(())
    }

    /// Get current assignments for a specific validator
    pub async fn get_validator_assignments(&self, validator_hotkey: &str) -> Option<Vec<String>> {
        self.assignments.read().await.get(validator_hotkey).cloned()
    }

    /// Get all current assignments
    pub async fn get_all_assignments(&self) -> HashMap<String, Vec<String>> {
        self.assignments.read().await.clone()
    }

    /// Check if a validator has any assigned executors
    pub async fn has_assignments(&self, validator_hotkey: &str) -> bool {
        self.assignments
            .read()
            .await
            .get(validator_hotkey)
            .map(|ids| !ids.is_empty())
            .unwrap_or(false)
    }
}

/// Business logic based assignment strategy
pub struct BusinessLogicAssignment {
    /// Validators that should be prioritized
    pub preferred_validators: Vec<String>,
    /// Maximum executors to assign per validator
    pub max_executors_per_validator: usize,
}

#[async_trait]
impl AssignmentStrategy for BusinessLogicAssignment {
    async fn assign_executors(
        &self,
        validators: Vec<ValidatorInfo>,
        executors: Vec<AvailableExecutor>,
        _current_assignments: &HashMap<String, Vec<String>>,
    ) -> HashMap<String, Vec<String>> {
        let mut assignments = HashMap::new();
        let mut executor_iter = executors.into_iter();

        // First, assign to preferred validators
        for validator in validators.iter() {
            if self.preferred_validators.contains(&validator.hotkey) {
                let assigned: Vec<String> = executor_iter
                    .by_ref()
                    .take(self.max_executors_per_validator)
                    .map(|e| e.id)
                    .collect();

                if !assigned.is_empty() {
                    debug!(
                        "Assigned {} executors to preferred validator {}",
                        assigned.len(),
                        validator.hotkey
                    );
                    assignments.insert(validator.hotkey.clone(), assigned);
                }
            }
        }

        // Then, assign remaining executors to other validators
        for validator in validators.iter() {
            if !assignments.contains_key(&validator.hotkey) && validator.axon_endpoint.is_some() {
                if let Some(executor) = executor_iter.next() {
                    debug!(
                        "Assigned executor {} to validator {}",
                        executor.id, validator.hotkey
                    );
                    assignments.insert(validator.hotkey.clone(), vec![executor.id]);
                }
            }
        }

        assignments
    }
}

/// Random assignment strategy
pub struct RandomAssignment {
    /// Maximum executors to assign per validator
    pub max_executors_per_validator: usize,
}

#[async_trait]
impl AssignmentStrategy for RandomAssignment {
    async fn assign_executors(
        &self,
        validators: Vec<ValidatorInfo>,
        executors: Vec<AvailableExecutor>,
        _current_assignments: &HashMap<String, Vec<String>>,
    ) -> HashMap<String, Vec<String>> {
        use rand::seq::SliceRandom;
        let mut rng = rand::thread_rng();

        let mut assignments = HashMap::new();
        let mut available_executors = executors;
        available_executors.shuffle(&mut rng);

        let mut validators_with_endpoints: Vec<_> = validators
            .into_iter()
            .filter(|v| v.axon_endpoint.is_some())
            .collect();
        validators_with_endpoints.shuffle(&mut rng);

        let mut executor_iter = available_executors.into_iter();

        for validator in validators_with_endpoints {
            let assigned: Vec<String> = executor_iter
                .by_ref()
                .take(self.max_executors_per_validator)
                .map(|e| e.id)
                .collect();

            if !assigned.is_empty() {
                assignments.insert(validator.hotkey, assigned);
            }
        }

        assignments
    }
}

/// Stake-based assignment strategy
pub struct StakeBasedAssignment {
    /// Minimum stake required for any assignment
    pub min_stake_threshold: u64,
    /// Scaling factor for executors per TAO of stake
    pub executors_per_tao: f64,
}

#[async_trait]
impl AssignmentStrategy for StakeBasedAssignment {
    async fn assign_executors(
        &self,
        validators: Vec<ValidatorInfo>,
        executors: Vec<AvailableExecutor>,
        _current_assignments: &HashMap<String, Vec<String>>,
    ) -> HashMap<String, Vec<String>> {
        let mut assignments = HashMap::new();

        // Filter and sort validators by stake (descending)
        let mut eligible_validators: Vec<_> = validators
            .into_iter()
            .filter(|v| v.stake >= self.min_stake_threshold && v.axon_endpoint.is_some())
            .collect();
        eligible_validators.sort_by(|a, b| b.stake.cmp(&a.stake));

        let mut executor_iter = executors.into_iter();

        for validator in eligible_validators {
            // Calculate executors based on stake
            let tao_stake = validator.stake as f64 / 1e9; // Convert from RAO to TAO
            let executor_count = (tao_stake * self.executors_per_tao).round() as usize;
            let executor_count = executor_count.max(1); // At least 1 executor if eligible

            let assigned: Vec<String> = executor_iter
                .by_ref()
                .take(executor_count)
                .map(|e| e.id)
                .collect();

            if !assigned.is_empty() {
                debug!(
                    "Assigned {} executors to validator {} with {:.2} TAO stake",
                    assigned.len(),
                    validator.hotkey,
                    tao_stake
                );
                assignments.insert(validator.hotkey, assigned);
            }
        }

        assignments
    }
}

/// Round-robin assignment strategy
pub struct RoundRobinAssignment;

#[async_trait]
impl AssignmentStrategy for RoundRobinAssignment {
    async fn assign_executors(
        &self,
        validators: Vec<ValidatorInfo>,
        executors: Vec<AvailableExecutor>,
        _current_assignments: &HashMap<String, Vec<String>>,
    ) -> HashMap<String, Vec<String>> {
        let mut assignments: HashMap<String, Vec<String>> = HashMap::new();

        let eligible_validators: Vec<_> = validators
            .into_iter()
            .filter(|v| v.axon_endpoint.is_some())
            .collect();

        if eligible_validators.is_empty() {
            warn!("No validators with axon endpoints found for assignment");
            return assignments;
        }

        // Initialize empty vectors for each validator
        for validator in &eligible_validators {
            assignments.insert(validator.hotkey.clone(), Vec::new());
        }

        // Distribute executors round-robin
        for (idx, executor) in executors.into_iter().enumerate() {
            let validator_idx = idx % eligible_validators.len();
            let validator_hotkey = &eligible_validators[validator_idx].hotkey;

            assignments
                .get_mut(validator_hotkey)
                .unwrap()
                .push(executor.id);
        }

        // Remove validators with no executors
        assignments.retain(|_, v| !v.is_empty());

        assignments
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[tokio::test]
    async fn test_round_robin_assignment() {
        let validators = vec![
            ValidatorInfo {
                uid: 1,
                hotkey: "validator1".to_string(),
                stake: 1000,
                axon_endpoint: Some("1.1.1.1:8080".to_string()),
            },
            ValidatorInfo {
                uid: 2,
                hotkey: "validator2".to_string(),
                stake: 2000,
                axon_endpoint: Some("2.2.2.2:8080".to_string()),
            },
        ];

        let executors = vec![
            AvailableExecutor {
                id: "exec1".to_string(),
                name: "Executor 1".to_string(),
                grpc_address: "".to_string(),
                resources: None,
                gpu_count: 1,
            },
            AvailableExecutor {
                id: "exec2".to_string(),
                name: "Executor 2".to_string(),
                grpc_address: "".to_string(),
                resources: None,
                gpu_count: 1,
            },
            AvailableExecutor {
                id: "exec3".to_string(),
                name: "Executor 3".to_string(),
                grpc_address: "".to_string(),
                resources: None,
                gpu_count: 1,
            },
        ];

        let strategy = RoundRobinAssignment;
        let assignments = strategy
            .assign_executors(validators, executors, &HashMap::new())
            .await;

        assert_eq!(assignments.len(), 2);
        assert_eq!(
            assignments.get("validator1").unwrap(),
            &vec!["exec1", "exec3"]
        );
        assert_eq!(assignments.get("validator2").unwrap(), &vec!["exec2"]);
    }
}
