//! Basilca Miner Library
//!
//! This module exposes the miner's functionality for testing and external use.

pub mod bittensor_core;
pub mod cli;
pub mod config;
pub mod node_manager;
pub mod persistence;
// pub mod request_verification;
pub mod services;
pub mod validator_comms;
pub mod validator_discovery;

// Re-export commonly used types
pub use config::{MinerConfig, SecurityConfig};
pub use node_manager::NodeManager;
