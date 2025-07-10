//! Binary Validation Error Types
//!
//! Error types specific to binary validation operations.

/// Binary validation error types
#[derive(Debug, thiserror::Error)]
pub enum BinaryValidationError {
    #[error("Executor binary not found at path: {path}")]
    ExecutorBinaryNotFound { path: String },

    #[error("Validator binary not found at path: {path}")]
    ValidatorBinaryNotFound { path: String },

    #[error("Binary upload failed: {reason}")]
    BinaryUploadFailed { reason: String },

    #[error("Binary execution failed: {reason}")]
    BinaryExecutionFailed { reason: String },

    #[error("Binary execution timeout after {timeout_secs} seconds")]
    BinaryExecutionTimeout { timeout_secs: u64 },

    #[error("Binary output parsing failed: {reason}")]
    BinaryOutputParsingFailed { reason: String },

    #[error("SSH connection failed: {reason}")]
    SshConnectionFailed { reason: String },

    #[error("Invalid binary output format: {details}")]
    InvalidBinaryOutputFormat { details: String },

    #[error("Binary validation configuration error: {reason}")]
    ConfigurationError { reason: String },
}
