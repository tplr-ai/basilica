//! # Bittensor Core
//!
//! Core Bittensor network integration for the Validator.
//! Handles weight setting, metagraph operations, and network communication.

pub mod chain_registration;
pub mod weight_allocation;
pub mod weight_setter;

pub use chain_registration::ChainRegistration;
pub use weight_setter::WeightSetter;
