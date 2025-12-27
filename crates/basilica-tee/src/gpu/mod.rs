//! NVIDIA GPU support for Confidential Computing
//!
//! This module provides functionality for:
//! - GPU device information via NVML
//! - GPU attestation evidence generation and parsing
//! - CC (Confidential Compute) mode verification
//! - Remote attestation via NVIDIA NRAS
//! - GPU ID utilities

pub mod device;
pub mod evidence;
pub mod evidence_parser;
pub mod nvtrust;
pub mod remote_verification;
pub mod utils;
pub mod verifier;

pub use device::GpuDeviceProvider;
#[allow(deprecated)]
pub use evidence::GpuEvidenceParser;
pub use evidence_parser::{parse_evidence, JsonEvidenceParser};
pub use nvtrust::NvEvidenceProvider;
pub use remote_verification::{NrasConfig, RemoteGpuVerifier};
pub use verifier::{verify_evidence, LocalGpuVerifier};
