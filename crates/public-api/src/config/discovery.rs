//! Validator discovery configuration

use serde::{Deserialize, Serialize};

/// Validator discovery configuration
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct DiscoveryConfig {
    /// Maximum number of validators to track
    pub max_validators: usize,

    /// Validator timeout in seconds
    pub validator_timeout: u64,

    /// Enable automatic failover
    pub enable_failover: bool,

    /// Failover threshold (consecutive failures)
    pub failover_threshold: u32,
}

impl Default for DiscoveryConfig {
    fn default() -> Self {
        Self {
            max_validators: 100,
            validator_timeout: 30,
            enable_failover: true,
            failover_threshold: 3,
        }
    }
}
