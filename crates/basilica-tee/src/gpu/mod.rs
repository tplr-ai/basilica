//! NVIDIA GPU support for Confidential Computing
//!
//! This module provides functionality for:
//! - GPU device information via NVML
//! - GPU attestation evidence generation
//! - CC (Confidential Compute) mode verification

pub mod device;
pub mod evidence;
pub mod nvtrust;

pub use device::GpuDeviceProvider;
pub use evidence::GpuEvidenceParser;
pub use nvtrust::NvEvidenceProvider;

