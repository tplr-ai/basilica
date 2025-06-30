//! # Service Abstractions
//!
//! Common service lifecycle management patterns and dependency injection interfaces.
//! Provides standardized ways to start, stop, and monitor services across all components.

pub mod traits;

// Re-export commonly used types
pub use traits::*;
