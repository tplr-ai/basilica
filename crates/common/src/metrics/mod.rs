//! # Metrics Abstractions
//!
//! Common metrics collection and reporting interfaces for monitoring system performance
//! and behavior across all Basilca components.

pub mod labels;
pub mod traits;

// Re-export commonly used types
pub use labels::*;
pub use traits::*;
