//! # Configuration Abstractions
//!
//! Common configuration patterns and database configuration shared across
//! all Basilca components.

pub mod loader;
pub mod traits;
pub mod types;

// Re-export commonly used types
pub use loader::*;
pub use traits::*;
pub use types::*;
