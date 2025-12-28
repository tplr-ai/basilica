//! # basilica-validator
//!
//! Basilica Validator - Bittensor neuron for GPU hardware verification and miner scoring.
//!
//! This crate provides the core validator functionality for the Basilica network.
//! Validators verify GPU hardware capabilities, score miners, and participate in
//! the Bittensor consensus mechanism.
//!
//! ## Overview
//!
//! The validator uses SSH-based direct verification where it connects directly to
//! miners' GPU nodes, eliminating intermediary trust requirements while maintaining
//! security through cryptographic verification.
//!
//! ## Key Features
//!
//! - **Hardware Verification**: Binary validation system for secure GPU verification
//! - **SSH-Based Verification**: Direct SSH access to miner nodes for trustless validation
//! - **Bittensor Integration**: Native participation in Bittensor consensus with weight allocation
//! - **GPU Profiling**: Automatic detection and profiling of GPU capabilities
//! - **REST API**: External access to validator data and status
//!
//! ## Quick Start
//!
//! ```rust,ignore
//! use basilica_validator::{ValidatorConfig, ValidatorService};
//!
//! #[tokio::main]
//! async fn main() -> Result<(), Box<dyn std::error::Error>> {
//!     // Load configuration
//!     let config = ValidatorConfig::load("validator.toml")?;
//!     
//!     // Create and start the validator service
//!     let service = ValidatorService::new(config).await?;
//!     service.run().await?;
//!     
//!     Ok(())
//! }
//! ```
//!
//! ## Feature Flags
//!
//! - `client` - Enable HTTP client for external services (default)
//! - `test-utils` - Enable test utilities
//! - `cli` - Enable CLI support with clap derives
//!
//! ## Architecture
//!
//! ```text
//! ┌─────────────────┐     gRPC      ┌─────────────┐     SSH      ┌──────────┐
//! │    Validator    │──────────────▶│    Miner    │─────────────▶│ GPU Node │
//! │                 │◀──────────────│             │              │          │
//! └─────────────────┘               └─────────────┘              └──────────┘
//!         │
//!         ▼
//! ┌─────────────────┐
//! │   Bittensor     │
//! │   (Weights)     │
//! └─────────────────┘
//! ```
//!
//! ## Related Crates
//!
//! - [`basilica-common`](https://crates.io/crates/basilica-common) - Core shared types
//! - [`basilica-protocol`](https://crates.io/crates/basilica-protocol) - gRPC definitions
//! - [`basilica-miner`](https://crates.io/crates/basilica-miner) - Miner implementation

pub mod agent_installer;
pub mod api;
pub mod ban_system;
pub mod billing;
pub mod bittensor_core;
pub mod cli;
pub mod collateral;
pub mod config;
pub mod gpu;
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
