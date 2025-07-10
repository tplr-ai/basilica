//! Hardware Validation Module
//!
//! Provides hardware validation functionality for executor verification.
//! This module contains validator-specific logic for binary validation.

pub mod challenge_converter;
pub mod challenge_generator;
pub mod errors;
pub mod gpu_profile_query;
pub mod secure_validator;
pub mod types;

#[cfg(test)]
mod challenge_generator_tests;
