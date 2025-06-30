//! # Validator Library
//!
//! Core library for the Basilica validator component that performs verification,
//! scoring, and participates in the Bittensor network.

pub mod api;
pub mod bittensor_core;
pub mod cli;
pub mod config;
pub mod journal;
pub mod metrics;
pub mod miner_prover;
pub mod persistence;
pub mod ssh;
pub mod validation;

// Main public API exports
pub use api::types::{RentCapacityRequest, RentCapacityResponse};
pub use api::ApiHandler;
pub use bittensor_core::weight_setter::WeightSetter;
pub use cli::{Args, Cli, Command, CommandHandler};
pub use config::{ValidatorConfig, VerificationConfig};
pub use metrics::{
    ValidatorApiMetrics, ValidatorBusinessMetrics, ValidatorMetrics, ValidatorPrometheusMetrics,
};
// Journal functionality temporarily disabled for testing
pub use miner_prover::{
    types::{ExecutorInfo, MinerInfo},
    MinerProver,
};
pub use persistence::entities::{
    challenge_result::ChallengeResult, environment_validation::EnvironmentValidation,
    VerificationLog,
};
pub use persistence::SimplePersistence;
pub use ssh::{ExecutorSshDetails, ValidatorSshClient};
pub use validation::{
    AttestationResult, HardwareSpecs, HardwareValidator, HardwareValidatorFactory,
    ValidationConfig, ValidationError,
};

/// Re-export common error types
pub use common::error::{BasilcaError, BasilcaResult};

/// Validator library version
pub const VERSION: &str = env!("CARGO_PKG_VERSION");
