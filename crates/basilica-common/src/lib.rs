//! # Common Basilica
//!
//! Core shared types, error definitions, and cryptographic utilities for the Basilica .
//! This crate provides the fundamental building blocks that all other Basilica crates depend on.
//!
//! ## Key Features
//! - Identity types (Hotkey, NodeId, ValidatorUid, MinerUid)
//! - Comprehensive error handling with BasilicaError trait
//! - Cryptographic utilities for hashing and signature verification
//! - Shared persistence abstractions and repository patterns
//! - Common service lifecycle management
//! - Standardized metrics collection interfaces
//!
//! ## Design Principles
//! - Minimal dependencies to avoid bloat in dependent crates
//! - Strong typing with validation logic
//! - Serde support for serialization across network boundaries
//! - Memory safety and security by design
//! - Trait-based abstractions for dependency injection

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
