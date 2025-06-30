//! # Persistence Abstractions
//!
//! Common traits and patterns for database operations across all Basilca components.
//! Provides repository pattern, health checks, migrations, and cleanup interfaces.

pub mod connection;
pub mod pagination;
pub mod sqlite;
pub mod traits;

// Re-export commonly used types
pub use connection::*;
pub use pagination::*;
pub use sqlite::*;
pub use traits::*;
