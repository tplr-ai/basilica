//! # Basilica TEE
//!
//! TEE (Trusted Execution Environment) support for Basilica, providing Intel TDX
//! quote verification and NVIDIA GPU Confidential Computing attestation.
//!
//! ## Features
//!
//! - `server` (default): Enables the axum-based attestation HTTP server
//! - `nvml`: Enables NVIDIA NVML bindings for GPU device info
//!
//! ## Modules
//!
//! - [`tdx`]: Intel TDX quote parsing, generation, and verification
//! - [`gpu`]: NVIDIA GPU device info and CC attestation
//! - [`server`]: HTTP attestation server (requires `server` feature)
//! - [`bootstrap`]: Remote TEE setup for executor nodes
//! - [`config`]: Configuration types for TEE settings
//! - [`error`]: Error types for TEE operations
//! - [`types`]: Shared data types
//! - [`traits`]: Core trait abstractions for providers and verifiers

pub mod bootstrap;
pub mod config;
pub mod crypto;
pub mod error;
pub mod gpu;
#[cfg(feature = "server")]
pub mod server;
pub mod service;
pub mod tdx;
pub mod traits;
pub mod types;

// Re-export commonly used types
pub use config::{GpuCcConfig, TdxConfig, TeeConfig};
pub use error::{TeeError, TeeResult};

// Re-export core traits
pub use traits::{
    CertificateHasher, EvidenceParser, EvidenceProvider, GpuVerifier, QuoteProvider, TdxVerifier,
};

// Re-export service
pub use service::{TeeAttestationResult, TeeService, TeeServiceBuilder, TeeServiceConfig};

/// Crate version
pub const VERSION: &str = env!("CARGO_PKG_VERSION");
