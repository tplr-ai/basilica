//! # basilica-common
//!
//! Core shared types, cryptographic utilities, and infrastructure for the Basilica GPU marketplace.
//!
//! This crate provides the foundational building blocks that all other Basilica crates depend on.
//! It is designed to be lightweight while providing essential functionality for building
//! decentralized compute applications on Bittensor.
//!
//! ## Overview
//!
//! `basilica-common` provides:
//!
//! - **Identity Types**: `Hotkey`, `NodeId`, `ValidatorUid`, `MinerUid` with SS58 validation
//! - **Cryptography**: Blake3 hashing, Ed25519/Sr25519 signature verification
//! - **Configuration**: Unified config loading with TOML files and environment overrides
//! - **Persistence**: Repository traits and database abstractions (SQLite/PostgreSQL)
//! - **SSH**: Trait abstractions for SSH key management
//! - **Metrics**: Standardized metrics collection interfaces
//!
//! ## Quick Start
//!
//! ```rust,ignore
//! use basilica_common::{Hotkey, Config, CryptoProvider};
//!
//! // Parse a Bittensor hotkey from SS58 format
//! let hotkey = Hotkey::from_ss58("5GrwvaEF5zXb26Fz9rcQpDWS57CtERHpNehXCPcNoHGKutQY")?;
//!
//! // Load configuration with environment overrides
//! let config = Config::builder()
//!     .file("config.toml")
//!     .env_prefix("BASILICA")
//!     .build()?;
//! ```
//!
//! ## Feature Flags
//!
//! - `sqlite` - Enable SQLite persistence backend
//! - `postgres` - Enable PostgreSQL persistence backend  
//! - `crypto-extra` - Additional cryptographic utilities
//!
//! ## Module Organization
//!
//! | Module | Description |
//! |--------|-------------|
//! | [`crypto`] | Cryptographic primitives (Blake3, Ed25519, Sr25519, P256) |
//! | [`identity`] | Network identity types with validation |
//! | [`config`] | Configuration loading and management |
//! | [`persistence`] | Database traits and repository patterns |
//! | [`ssh`] | SSH key management abstractions |
//! | [`metrics`] | Metrics collection traits |
//! | [`compute`] | Compute resource definitions |
//! | [`rental`] | GPU rental types and states |
//!
//! ## Design Principles
//!
//! - Minimal dependencies to avoid bloat in dependent crates
//! - Strong typing with validation logic
//! - Serde support for serialization across network boundaries
//! - Memory safety and security by design
//! - Trait-based abstractions for dependency injection
//!
//! ## Related Crates
//!
//! - [`basilica-sdk`](https://crates.io/crates/basilica-sdk) - High-level client SDK
//! - [`basilica-protocol`](https://crates.io/crates/basilica-protocol) - gRPC protocol definitions
//! - [`basilica-validator`](https://crates.io/crates/basilica-validator) - Validator node implementation

pub mod auth_constants;
pub mod compute;
pub mod config;
pub mod convert;
pub mod crypto;
pub mod distributed;
pub mod error;
pub mod identity;
pub mod journal;
pub mod logging;
pub mod metrics;
pub mod network;
pub mod node_identity;
pub mod persistence;
pub mod rental;
pub mod ssh;
pub mod storage;
pub mod types;
pub mod utils;
pub mod validation;

// Re-export commonly used types at the crate root for convenience
pub use auth_constants::*;
pub use config::*;
pub use crypto::*;
pub use error::*;
pub use identity::*;
pub use types::{ApiKeyName, ApiKeyNameError, LocationProfile};

// Re-export from specific modules to avoid ambiguity
pub use metrics::labels;
pub use metrics::traits as metrics_traits;
pub use persistence::traits as persistence_traits;
pub use ssh::traits as ssh_traits;
pub use storage::{KeyValueStorage, MemoryStorage};

// Re-export the main types directly
pub use compute::*;
pub use metrics::traits::*;
pub use persistence::traits::*;
pub use rental::*;

/// Version of the common crate
pub const VERSION: &str = env!("CARGO_PKG_VERSION");

/// Protocol version for compatibility checking between components
pub const PROTOCOL_VERSION: &str = "1.0.0";

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_version_constants() {
        assert!(VERSION.chars().any(|c| c.is_ascii_digit()));
        assert!(PROTOCOL_VERSION.chars().any(|c| c.is_ascii_digit()));
    }
}
