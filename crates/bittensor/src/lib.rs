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

// Include our generated API module
// We always use our own metadata to ensure compatibility
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
    set_weights_payload, sign_message_hex, sign_with_keypair, tao_to_rao,
    verify_bittensor_signature, BittensorSignature, NormalizedWeight,
};

// Re-export key types from our generated API
pub use crate::api::api::runtime_types::pallet_subtensor::{
    pallet::{AxonInfo, PrometheusInfo},
    rpc_info::{
        metagraph::{Metagraph, SelectiveMetagraph},
        neuron_info::{NeuronInfo, NeuronInfoLite},
    },
};

// Type alias for AccountId
pub type AccountId = subxt::config::polkadot::AccountId32;
