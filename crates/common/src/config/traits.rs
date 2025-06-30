//! # Configuration Traits
//!
//! Core traits for configuration loading and management.

use serde::de::DeserializeOwned;
use std::path::{Path, PathBuf};

use crate::error::ConfigurationError;

/// Configuration loader trait
///
/// Provides a standardized interface for loading configuration from various sources
/// with layered configuration support (defaults, files, environment variables).
pub trait ConfigLoader<C: DeserializeOwned + Send + Sync> {
    /// Load configuration with optional path override
    ///
    /// # Arguments
    /// * `path_override` - Optional path to configuration file
    ///
    /// # Returns
    /// * Loaded and validated configuration
    ///
    /// # Implementation Notes
    /// - Should support layered configuration (defaults -> file -> env vars)
    /// - Should validate configuration after loading
    /// - Should apply environment variable overrides
    fn load(path_override: Option<PathBuf>) -> Result<C, ConfigurationError>;

    /// Load configuration from specific file
    ///
    /// # Arguments
    /// * `path` - Path to configuration file
    ///
    /// # Returns
    /// * Loaded configuration from file
    fn load_from_file(path: &Path) -> Result<C, ConfigurationError>;

    /// Apply environment variable overrides to configuration
    ///
    /// # Arguments
    /// * `config` - Mutable reference to configuration
    /// * `prefix` - Environment variable prefix (e.g., "BASILCA")
    ///
    /// # Returns
    /// * Result indicating success or failure
    fn apply_env_overrides(config: &mut C, prefix: &str) -> Result<(), ConfigurationError>;
}
