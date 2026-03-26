//! Error types for the Basilica CLI

use color_eyre::eyre::Report;
use std::path::PathBuf;
use thiserror::Error;

/// Errors related to source file handling
#[derive(Debug, Error)]
pub enum SourceError {
    #[error("File not found: {path}")]
    FileNotFound { path: PathBuf },

    #[error("File is empty: {path}")]
    EmptyFile { path: PathBuf },

    #[error("File too large ({size} bytes, max {max} bytes): {path}")]
    FileTooLarge {
        path: PathBuf,
        size: usize,
        max: usize,
    },

    #[error("Failed to read file: {0}")]
    ReadError(#[from] std::io::Error),

    #[error("Invalid source format: {0}")]
    InvalidFormat(String),

    #[error("Cannot determine source type for: {input}")]
    UnknownSourceType { input: String },
}

/// Errors related to deployment operations
#[derive(Debug, Error)]
pub enum DeployError {
    #[error("Validation failed: {message}")]
    Validation { message: String },

    #[error("Deployment '{name}' not found")]
    NotFound { name: String },

    #[error("Deployment '{name}' failed: {reason}")]
    DeploymentFailed { name: String, reason: String },

    #[error("Deployment '{name}' timed out after {timeout_secs}s")]
    Timeout { name: String, timeout_secs: u32 },

    #[error("Resource quota exceeded: {message}")]
    QuotaExceeded { message: String },

    #[error("GPU resource validation failed: {message}")]
    GpuResourceMismatch { message: String },

    #[error("No private deployments found")]
    NoPrivateDeployments,

    #[error("Share token operation failed: {message}")]
    ShareTokenError { message: String },

    #[error("Source error: {0}")]
    Source(#[from] SourceError),

    #[error("API error: {0}")]
    Api(#[from] basilica_sdk::error::ApiError),
}

/// CLI error type with minimal variants
/// Note: SourceError converts through DeployError::Source, not directly to CliError
/// This avoids ambiguity in error conversion chains
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

    /// Deployment errors
    #[error(transparent)]
    Deploy(#[from] DeployError),

    /// Invalid volume provider
    #[error("Invalid provider: {0}")]
    InvalidProvider(String),

    /// Everything else (using color-eyre's Report for rich errors)
    #[error(transparent)]
    Internal(#[from] Report),
}

/// Result type alias for CLI operations
pub type Result<T> = std::result::Result<T, CliError>;
