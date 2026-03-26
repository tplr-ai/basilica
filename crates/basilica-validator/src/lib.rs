//! # Validator Library
//!
//! Core library for the Basilica validator component that performs verification,
//! scoring, and participates in the Bittensor network.

pub mod agent_installer;
pub mod api;
pub mod ban_system;
pub mod basilica_api;
pub mod billing;
pub mod bittensor_core;
pub mod cli;
pub mod collateral;
pub mod config;
pub mod gpu;
pub mod grpc;
pub mod journal;
pub mod k8s_profile_publisher;
pub mod metrics;
pub mod miner_prover;
pub mod node_profile;
pub mod os_process;
pub mod persistence;
pub mod rental;
pub mod rental_adapter;
pub mod service;
pub mod ssh;

// Main public API exports
#[cfg(feature = "client")]
pub use api::client::ValidatorClient;
pub use api::types::{RentCapacityRequest, RentCapacityResponse};
pub use api::ApiHandler;
pub use bittensor_core::weight_setter::WeightSetter;
pub use cli::{Args, Command};
pub use config::{ValidatorConfig, VerificationConfig};
pub use metrics::{
    ValidatorApiMetrics, ValidatorBusinessMetrics, ValidatorMetrics, ValidatorPrometheusMetrics,
};
// Journal functionality temporarily disabled for testing
pub use miner_prover::types::ValidationError;
pub use miner_prover::{
    types::{MinerInfo, NodeInfo},
    MinerProver,
};
pub use persistence::entities::{
    challenge_result::ChallengeResult, environment_validation::EnvironmentValidation,
    VerificationLog,
};
pub use persistence::SimplePersistence;
pub use rental::{RentalInfo, RentalManager, RentalRequest, RentalResponse};
pub use service::{ServiceStatus, ValidatorService};
pub use ssh::{NodeSshDetails, ValidatorSshClient};

/// Validator library version
pub const VERSION: &str = env!("CARGO_PKG_VERSION");
