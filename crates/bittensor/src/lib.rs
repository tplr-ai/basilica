//! # Bittensor Integration
//!
//! Centralized service for all Bittensor blockchain interactions using crabtensor.
//! Provides high-level interface for wallet management, transaction submission,
//! and chain state queries.

pub mod discovery;
pub mod error;
pub mod registration;
pub mod retry;
pub mod service;
pub mod utils;

// Include our generated API module when building for testnet
#[cfg(feature = "testnet")]
pub mod api;

#[cfg(test)]
mod error_tests;

pub use discovery::NeuronDiscovery;
pub use error::{BittensorError, ErrorCategory, RetryConfig};
pub use registration::{
    ChainRegistration, RegistrationConfig, RegistrationConfigBuilder, RegistrationStateSnapshot,
};
pub use retry::{retry_operation, retry_operation_with_timeout, CircuitBreaker, RetryExecutor};
pub use service::Service;
pub use utils::{
    account_id_to_hotkey, create_signature, hotkey_to_account_id, normalize_weights, rao_to_tao,
    set_weights_payload, tao_to_rao, verify_bittensor_signature, NormalizedWeight,
};

// Re-export the entire crabtensor api module for full access to types
pub use crabtensor::api;

// Re-export key crabtensor types for convenience
pub use crabtensor::api::runtime_types::pallet_subtensor::{
    pallet::{AxonInfo, PrometheusInfo},
    rpc_info::{
        metagraph::{Metagraph, SelectiveMetagraph},
        neuron_info::{NeuronInfo, NeuronInfoLite},
    },
};
pub use crabtensor::AccountId;
