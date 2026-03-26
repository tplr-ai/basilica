//! Basilca Miner Library
//!
//! This module exposes the miner's functionality for testing and external use.

pub mod bidding;
pub mod bittensor_core;
pub mod cli;
pub mod config;
pub mod node_manager;
pub mod persistence;
pub mod registration_client;
pub mod validator_discovery;

// Re-export commonly used types
pub use bidding::BidManager;
pub use config::{BiddingConfig, MinerConfig, SecurityConfig};
pub use node_manager::NodeManager;
pub use registration_client::{RegistrationClient, RegistrationState};
