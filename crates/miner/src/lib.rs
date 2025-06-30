//! Basilca Miner Library
//!
//! This module exposes the miner's functionality for testing and external use.

pub mod auth;
pub mod bittensor_core;
pub mod cli;
pub mod config;
pub mod executor_manager;
pub mod persistence;
pub mod request_verification;
pub mod session_cleanup;
pub mod ssh;
pub mod validator_comms;
pub mod validator_discovery;

// Re-export commonly used types
pub use config::{ExecutorConfig, MinerConfig, SecurityConfig};
pub use executor_manager::ExecutorManager;
pub use ssh::{MinerSshConfig, SshSessionManager, ValidatorAccessService};
