//! Load balancer module for distributing requests across validators

mod strategy;
mod validator_pool;

pub use strategy::Strategy;
pub use validator_pool::ValidatorPool;

use crate::{config::LoadBalancerStrategy as ConfigStrategy, discovery::ValidatorInfo, Result};
use std::sync::Arc;
use tracing::{debug, warn};

/// Load balancer for distributing requests across validators
pub struct LoadBalancer {
    /// Current strategy
    strategy: Box<dyn Strategy + Send + Sync>,

    /// Validator pool
    pool: Arc<ValidatorPool>,
}

impl LoadBalancer {
    /// Create a new load balancer with the specified strategy
    pub fn new(strategy: ConfigStrategy) -> Self {
        let pool = Arc::new(ValidatorPool::new());
        let strategy = Self::create_strategy(strategy, pool.clone());

        Self { strategy, pool }
    }

    /// Create strategy implementation based on configuration
    fn create_strategy(
        config_strategy: ConfigStrategy,
        pool: Arc<ValidatorPool>,
    ) -> Box<dyn Strategy + Send + Sync> {
        match config_strategy {
            ConfigStrategy::RoundRobin => Box::new(strategy::RoundRobinStrategy::new(pool)),
            ConfigStrategy::LeastConnections => {
                Box::new(strategy::LeastConnectionsStrategy::new(pool))
            }
            ConfigStrategy::WeightedScore => Box::new(strategy::WeightedScoreStrategy::new(pool)),
            ConfigStrategy::Random => Box::new(strategy::RandomStrategy::new(pool)),
            ConfigStrategy::ConsistentHash => Box::new(strategy::ConsistentHashStrategy::new(pool)),
        }
    }

    /// Update the validator pool with current validators
    pub fn update_validators(&mut self, validators: Vec<ValidatorInfo>) {
        self.pool.update(validators);
    }

    /// Select a validator for the next request
    pub async fn select_validator(&self) -> Result<ValidatorInfo> {
        self.strategy.select().await
    }

    /// Select a validator using a specific key (for consistent hashing)
    pub async fn select_validator_with_key(&self, key: &str) -> Result<ValidatorInfo> {
        self.strategy.select_with_key(key).await
    }

    /// Report a successful request to a validator
    pub fn report_success(&self, validator_uid: u16) {
        self.pool.decrement_connections(validator_uid);
        debug!("Reported successful request to validator {}", validator_uid);
    }

    /// Report a failed request to a validator
    pub fn report_failure(&self, validator_uid: u16) {
        self.pool.decrement_connections(validator_uid);
        self.pool.increment_failures(validator_uid);
        warn!("Reported failed request to validator {}", validator_uid);
    }

    /// Get current pool statistics
    pub fn get_stats(&self) -> PoolStats {
        PoolStats {
            total_validators: self.pool.total_count(),
            healthy_validators: self.pool.healthy_count(),
            total_connections: self.pool.total_connections(),
        }
    }

    /// Get total number of active connections
    pub fn get_total_connections(&self) -> usize {
        self.pool.total_connections()
    }
}

/// Pool statistics
#[derive(Debug, Clone)]
pub struct PoolStats {
    /// Total number of validators in the pool
    pub total_validators: usize,

    /// Number of healthy validators
    pub healthy_validators: usize,

    /// Total number of active connections
    pub total_connections: usize,
}
