//! Hardware Validation Module
//!
//! Provides hardware validation functionality for executor verification.
//! This module contains validator-specific logic for attestation and validation.

pub mod attestor;
pub mod challenge_converter;
pub mod challenge_generator;
pub mod factory;
pub mod gpu_profile_query;
pub mod gpu_validator;
pub mod gpu_validator_v2;
pub mod integrity;
pub mod key_manager;
pub mod secure_validator;
pub mod signature_verifier;
pub mod types;
pub mod validator;

pub use factory::*;
pub use types::*;
pub use validator::*;

#[cfg(test)]
mod challenge_generator_tests;
