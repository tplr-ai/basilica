//! # basilica-miner
//!
//! Basilica Miner - Bittensor neuron that manages GPU node fleets.
//!
//! This crate provides the miner implementation for the Basilica network.
//! Miners manage fleets of GPU nodes, handle validator authentication,
//! and route verification requests.
//!
//! ## Overview
//!
//! The miner acts as an intermediary between validators and GPU nodes:
//! - Registers with the Bittensor network via an Axon server
//! - Authenticates validators using gRPC with Sr25519 signatures  
//! - Deploys ephemeral SSH keys to managed nodes
//! - Routes validator verification requests to appropriate nodes
//!
//! ## Quick Start
//!
//! ```rust,ignore
//! use basilica_miner::{MinerConfig, NodeManager};
//!
//! #[tokio::main]
//! async fn main() -> Result<(), Box<dyn std::error::Error>> {
//!     // Load configuration
//!     let config = MinerConfig::load("miner.toml")?;
//!     
//!     // Initialize node manager
//!     let node_manager = NodeManager::new(&config).await?;
//!     
//!     // Start serving validators
//!     node_manager.run().await?;
//!     
//!     Ok(())
//! }
//! ```
//!
//! ## GPU Node Requirements
//!
//! Nodes managed by the miner need:
//! - Standard SSH server
//! - Docker with NVIDIA Container Toolkit
//! - CUDA drivers ≥12.8
//! - NVIDIA GPUs (A100, H100, B200, etc.)
//!
//! ## Security Model
//!
//! - **Ephemeral SSH keys**: Validators generate ed25519 keys per session
//! - **Key tagging**: Keys are tagged with validator hotkey for identification
//! - **Auto-cleanup**: Miner removes expired keys after session timeout (~1 hour)
//! - **Sr25519 signatures**: All validator requests are cryptographically signed
//!
//! ## Related Crates
//!
//! - [`basilica-common`](https://crates.io/crates/basilica-common) - Core shared types
//! - [`basilica-protocol`](https://crates.io/crates/basilica-protocol) - gRPC definitions
//! - [`basilica-validator`](https://crates.io/crates/basilica-validator) - Validator implementation

pub mod bittensor_core;
pub mod cli;
pub mod config;
pub mod node_manager;
pub mod persistence;
// pub mod request_verification;
pub mod validator_comms;
pub mod validator_discovery;

// Re-export commonly used types
pub use config::{MinerConfig, SecurityConfig};
pub use node_manager::NodeManager;
