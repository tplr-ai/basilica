//! Load balancer configuration

use serde::{Deserialize, Serialize};

/// Load balancer configuration
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct LoadBalancerConfig {
    /// Load balancing strategy
    pub strategy: LoadBalancerStrategy,

    /// Health check interval in seconds
    pub health_check_interval: u64,

    /// Connection timeout in seconds
    pub connection_timeout: u64,

    /// Maximum retries per request
    pub max_retries: u32,

    /// Retry backoff multiplier
    pub retry_backoff_multiplier: f64,

    /// Maximum number of validators in the pool
    pub max_pool_size: usize,
}

/// Load balancing strategies
#[derive(Debug, Clone, Serialize, Deserialize)]
#[serde(rename_all = "snake_case")]
pub enum LoadBalancerStrategy {
    /// Round-robin distribution
    RoundRobin,

    /// Least connections
    LeastConnections,

    /// Weighted by validator score
    WeightedScore,

    /// Random selection
    Random,

    /// Consistent hashing
    ConsistentHash,
}

impl Default for LoadBalancerConfig {
    fn default() -> Self {
        Self {
            strategy: LoadBalancerStrategy::WeightedScore,
            health_check_interval: 30,
            connection_timeout: 10,
            max_retries: 3,
            retry_backoff_multiplier: 2.0,
            max_pool_size: 100,
        }
    }
}
