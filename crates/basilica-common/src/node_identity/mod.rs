//! Node identity system with UUID support
//!
//! This module provides node identification using:
//! - UUID: Primary unique identifier for data integrity
//!
//! # Example
//! ```
//! use basilica_common::node_identity::{NodeId, NodeIdentity};
//!
//! let id = NodeId::new("my-seed").unwrap();
//! println!("UUID: {}", id.uuid());
//! ```

pub mod constants;
pub mod display;
pub mod examples;
pub mod identity_store;
pub mod integration;
// Temporarily disabled - contains HUID-specific tests that need updating
// pub mod integration_tests;
pub mod interfaces;
pub mod matching;
pub mod migration;
pub mod node_id;
pub mod validation;
pub mod word_provider;
pub mod words;

pub use constants::*;
pub use display::{NodeIdentityDisplay, NodeIdentityDisplayExt};
pub use identity_store::SqliteIdentityStore;
#[cfg(feature = "sqlite")]
pub use integration::IdentityTransaction;
pub use integration::{IdentityConfig, IdentityDbFactory, IdentityPoolExt};
pub use interfaces::*;
pub use matching::*;
pub use migration::{IdentityMigrationManager, MigrationConfig, MigrationStats};
pub use node_id::NodeId;
pub use validation::*;
pub use word_provider::StaticWordProvider;
