//! Shared Data Types for TEE Operations
//!
//! This module contains the core data types used throughout the TEE crate.
//!
//! ## Module Organization
//!
//! - [`evidence`]: GPU device info and attestation evidence types
//! - [`measurements`]: TDX measurement types (MRTD, RTMRs)
//! - [`verification`]: Verification result types
//! - [`serde_utils`]: Serialization helpers

pub mod evidence;
pub mod measurements;
pub mod serde_utils;
pub mod verification;

// Re-export all types for convenience
pub use evidence::{GpuAttestationEvidence, GpuDeviceInfo};
pub use measurements::ExpectedMeasurements;
pub use verification::{GpuCcVerificationResult, TdxVerificationResult, TeeVerificationResult};
