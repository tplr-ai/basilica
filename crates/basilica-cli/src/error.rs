//! Error types for the Basilica CLI

use color_eyre::eyre::Report;
use thiserror::Error;

/// CLI error type with minimal variants
#[derive(Debug, Error)]
pub enum CliError {
    /// Configuration file issues
    #[error("Configuration error")]
    Config(#[from] basilica_common::ConfigurationError),

    /// API communication errors
    #[error("API error: {0}")]
    Api(#[from] basilica_sdk::error::ApiError),

    /// Authentication/authorization issues
    #[error(transparent)]
    Auth(#[from] crate::auth::AuthError),

    /// Everything else (using color-eyre's Report for rich errors)
    #[error(transparent)]
    Internal(#[from] Report),
}

/// Result type alias for CLI operations
pub type Result<T> = std::result::Result<T, CliError>;
