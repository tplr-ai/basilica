//! Configuration management for the Basilica Executor
//!
//! This module provides a modular configuration system following SOLID principles:
//! - Single Responsibility: Each module handles one aspect of configuration
//! - Open/Closed: Easy to extend with new configuration types
//! - Interface Segregation: Specific traits for different concerns
//! - Dependency Inversion: Abstractions over concrete implementations

pub mod docker;
pub mod system;
pub mod types;
pub mod validation;

// Re-exports for convenience
pub use docker::*;
pub use system::*;
pub use types::*;
