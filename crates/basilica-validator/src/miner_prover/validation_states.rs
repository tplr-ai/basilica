//! Centralized validation states for node verification
//!
//! This module implements a state tracking system for the node validation pipeline.
//! Each node progresses through validation states with metrics tracking:
//!
//! **Metric Values:**
//! - `0.0`: Not in this state
//! - `1.0`: Currently in this state
//! - `2.0`: Failed at this state
//!
//! **State Flow:**
//! - **Lightweight**: InQueue → Connecting → Connected → ConnectivityChecking → NatValidating → Completed
//! - **Full**: InQueue → Connecting → Connected → DockerValidating → NatValidating → BinaryValidating → Completed
//!
//! **Example** (node failed at NAT validation):
//! ```text
//! basilica_validator_node_validation_state{node_id="e1",state="in_queue"} 0.0
//! basilica_validator_node_validation_state{node_id="e1",state="connecting"} 0.0
//! basilica_validator_node_validation_state{node_id="e1",state="connected"} 0.0
//! basilica_validator_node_validation_state{node_id="e1",state="nat_validating"} 2.0
//! basilica_validator_node_validation_state{node_id="e1",state="completed"} 0.0
//! ```

use super::types::ValidationType;

/// Executor validation pipeline states
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum ValidationState {
    // Common states
    InQueue,
    Connecting,
    Connected,

    // Lightweight-specific states
    ConnectivityChecking,

    // Full-specific states
    DockerValidating,

    // Shared validation states
    NatValidating,
    BinaryValidating,

    // Final state
    Completed,
}

impl ValidationState {
    /// Returns the ordered state sequence for a validation type
    pub fn states_for_type(validation_type: ValidationType) -> &'static [ValidationState] {
        use ValidationState::*;

        match validation_type {
            ValidationType::Lightweight => &[
                InQueue,
                Connecting,
                Connected,
                ConnectivityChecking,
                NatValidating,
                Completed,
            ],
            ValidationType::Full => &[
                InQueue,
                Connecting,
                Connected,
                DockerValidating,
                NatValidating,
                BinaryValidating,
                Completed,
            ],
        }
    }

    /// Returns metric label string for this state
    pub fn as_str(&self) -> &'static str {
        match self {
            Self::InQueue => "in_queue",
            Self::Connecting => "connecting",
            Self::Connected => "connected",
            Self::ConnectivityChecking => "connectivity_checking",
            Self::DockerValidating => "docker_validating",
            Self::NatValidating => "nat_validating",
            Self::BinaryValidating => "binary_validating",
            Self::Completed => "completed",
        }
    }

    /// Returns all metric label strings for a validation type
    pub fn all_state_strings(validation_type: ValidationType) -> Vec<&'static str> {
        Self::states_for_type(validation_type)
            .iter()
            .map(|s| s.as_str())
            .collect()
    }
}

/// Validation state outcome for metrics
pub enum StateResult {
    Current, // Currently in this state (value: 1.0)
    Failed,  // Failed at this state (value: 2.0)
}

impl StateResult {
    /// Converts to Prometheus metric value (1.0 for Current, 2.0 for Failed)
    pub fn to_metric_value(&self) -> f64 {
        match self {
            Self::Current => 1.0,
            Self::Failed => 2.0,
        }
    }
}
