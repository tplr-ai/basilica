//! # Basilica
//!
//! **Decentralized GPU marketplace on the Bittensor network.**
//!
//! Basilica enables GPU compute rental through a decentralized marketplace where:
//! - **Miners** provide GPU hardware and earn TAO rewards
//! - **Validators** verify GPU availability and set weights
//! - **Users** rent GPU compute via a simple SDK or CLI
//!
//! ## Quick Start
//!
//! Add to your `Cargo.toml`:
//!
//! ```toml
//! [dependencies]
//! basilica = "0.1"
//! ```
//!
//! ### Deploy a GPU workload
//!
//! ```rust,ignore
//! use basilica::prelude::*;
//!
//! #[tokio::main]
//! async fn main() -> Result<(), Box<dyn std::error::Error>> {
//!     // Connect to Basilica
//!     let client = BasilicaClient::builder()
//!         .api_key("your-api-key")
//!         .build()
//!         .await?;
//!
//!     // Deploy a GPU workload
//!     let deployment = client.create_deployment(CreateDeploymentRequest {
//!         image: "pytorch/pytorch:2.0-cuda11.8".into(),
//!         gpu_count: Some(1),
//!         ..Default::default()
//!     }).await?;
//!
//!     println!("Deployed: {}", deployment.id);
//!     Ok(())
//! }
//! ```
//!
//! ## Feature Flags
//!
//! Enable only what you need:
//!
//! | Feature | Description | Default |
//! |---------|-------------|---------|
//! | `sdk` | High-level client SDK | ✅ |
//! | `cli` | Command-line interface | ❌ |
//! | `validator` | Run a validator node | ❌ |
//! | `miner` | Run a miner node | ❌ |
//! | `api` | REST API server | ❌ |
//! | `full` | Everything | ❌ |
//!
//! ```toml
//! # Just the SDK (default)
//! basilica = "0.1"
//!
//! # SDK + CLI
//! basilica = { version = "0.1", features = ["cli"] }
//!
//! # Everything
//! basilica = { version = "0.1", features = ["full"] }
//! ```
//!
//! ## Crate Structure
//!
//! This is an umbrella crate that re-exports:
//!
//! - [`basilica_common`] - Core types, crypto, and utilities
//! - [`basilica_protocol`] - gRPC protocol definitions
//! - [`basilica_sdk`] - High-level client SDK (feature: `sdk`)
//! - [`basilica_cli`] - Command-line interface (feature: `cli`)
//! - [`basilica_validator`] - Validator node (feature: `validator`)
//! - [`basilica_miner`] - Miner node (feature: `miner`)
//! - [`basilica_api`] - REST API server (feature: `api`)
//!
//! ## Architecture
//!
//! ```text
//! ┌─────────────────────────────────────────────────────────────┐
//! │                      Bittensor Network                       │
//! │                    (Weights & Incentives)                    │
//! └─────────────────────────────────────────────────────────────┘
//!                              ▲
//!                              │
//!          ┌───────────────────┼───────────────────┐
//!          │                   │                   │
//!    ┌─────┴─────┐       ┌─────┴─────┐       ┌─────┴─────┐
//!    │ Validator │◄─────►│   Miner   │       │    SDK    │
//!    │   Node    │       │   Node    │       │  Client   │
//!    └───────────┘       └───────────┘       └───────────┘
//!          │                   │                   │
//!          │              ┌────┴────┐              │
//!          │              │   GPU   │              │
//!          │              │ Workers │              │
//!          │              └─────────┘              │
//!          │                                       │
//!          └───────────────────┬───────────────────┘
//!                              │
//!                        ┌─────┴─────┐
//!                        │  REST API │
//!                        └───────────┘
//! ```
//!
//! ## Links
//!
//! - [GitHub Repository](https://github.com/one-covenant/basilica)
//! - [Documentation](https://docs.rs/basilica)
//! - [Website](https://basilica.ai)
//! - [Bittensor](https://bittensor.com)

#![cfg_attr(docsrs, feature(doc_cfg))]

// Always re-export core types
pub use basilica_common as common;
pub use basilica_protocol as protocol;

// Re-export common types at crate root for convenience
pub use basilica_common::identity::{Hotkey, MinerUid, NodeId, ValidatorUid};

// SDK (default feature)
#[cfg(feature = "sdk")]
#[cfg_attr(docsrs, doc(cfg(feature = "sdk")))]
pub use basilica_sdk as sdk;

#[cfg(feature = "sdk")]
#[cfg_attr(docsrs, doc(cfg(feature = "sdk")))]
pub use basilica_sdk::{BasilicaClient, CreateDeploymentRequest, DeploymentResponse};

// CLI
#[cfg(feature = "cli")]
#[cfg_attr(docsrs, doc(cfg(feature = "cli")))]
pub use basilica_cli as cli;

// Validator
#[cfg(feature = "validator")]
#[cfg_attr(docsrs, doc(cfg(feature = "validator")))]
pub use basilica_validator as validator;

// Miner
#[cfg(feature = "miner")]
#[cfg_attr(docsrs, doc(cfg(feature = "miner")))]
pub use basilica_miner as miner;

// API
#[cfg(feature = "api")]
#[cfg_attr(docsrs, doc(cfg(feature = "api")))]
pub use basilica_api as api;

/// Commonly used types and traits.
///
/// ```rust
/// use basilica::prelude::*;
/// ```
pub mod prelude {
    pub use crate::{Hotkey, MinerUid, NodeId, ValidatorUid};

    #[cfg(feature = "sdk")]
    pub use crate::sdk::{BasilicaClient, CreateDeploymentRequest, DeploymentResponse};
}

