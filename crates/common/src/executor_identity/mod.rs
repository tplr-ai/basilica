//! Executor identity system with UUID + HUID support
//!
//! This module provides a dual identifier system for executors:
//! - UUID: Primary unique identifier for data integrity
//! - HUID: Human-Unique Identifier for user-friendly interaction
//!
//! # Example
//! ```
//! use basilica_common::executor_identity::ExecutorId;
//!
//! let id = ExecutorId::new();
//! println!("UUID: {}", id.uuid());
//! println!("HUID: {}", id.huid()); // e.g., "swift-falcon-a3f2"
//! ```

pub mod constants;
pub mod display;
pub mod examples;
pub mod executor_id;
pub mod identity_store;
pub mod integration;
pub mod integration_tests;
pub mod interfaces;
pub mod matching;
pub mod migration;
pub mod validation;
pub mod word_provider;
pub mod words;

pub use constants::*;
pub use display::{ExecutorIdentityDisplay, ExecutorIdentityDisplayExt};
pub use executor_id::ExecutorId;
pub use identity_store::SqliteIdentityStore;
#[cfg(feature = "sqlite")]
pub use integration::IdentityTransaction;
pub use integration::{IdentityConfig, IdentityDbFactory, IdentityPoolExt};
pub use interfaces::*;
pub use matching::*;
pub use migration::{IdentityMigrationManager, MigrationConfig, MigrationStats};
pub use validation::*;
pub use word_provider::StaticWordProvider;
