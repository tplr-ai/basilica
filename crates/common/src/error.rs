//! Error handling for Basilca
//!
//! This module defines the core error handling infrastructure used throughout
//! the Basilca system. It provides:
//! - `BasilcaError` trait for consistent error handling
//! - Specific error types for different domains (Network, Crypto, Config, etc.)
//! - Integration with `thiserror` for ergonomic error handling
//!
//! # Design Principles
//! - All errors implement Send + Sync for async compatibility
//! - Use thiserror for library errors, anyhow for application errors
//! - Provide clear, actionable error messages
//! - Support error chaining and context

use thiserror::Error;

/// Base trait for all Basilca-specific errors
///
/// This trait ensures all Basilca errors are:
/// - Thread-safe (Send + Sync)
/// - Static lifetime (no borrowed data)
/// - Implement standard Error trait
///
/// # Implementation Notes for Developers
/// When creating new error types:
/// 1. Derive from thiserror::Error
/// 2. Implement BasilcaError trait
/// 3. Use `#[from]` for automatic conversions from underlying errors
/// 4. Provide clear, user-facing error messages
/// 5. Include context information where helpful
pub trait BasilcaError: std::error::Error + Send + Sync + 'static {}

/// Network-related errors
///
/// These errors occur during network operations like gRPC calls,
/// HTTP requests, or WebSocket connections.
#[derive(Error, Debug)]
pub enum NetworkError {
    /// Connection failed to establish
    #[error("Failed to connect to {endpoint}: {source}")]
    ConnectionFailed {
        endpoint: String,
        #[source]
        source: Box<dyn std::error::Error + Send + Sync>,
    },

    /// Connection was lost during operation
    #[error("Connection lost to {endpoint}")]
    ConnectionLost { endpoint: String },

    /// Request timed out
    #[error("Request timed out after {timeout_secs} seconds")]
    Timeout { timeout_secs: u64 },

    /// gRPC specific error
    #[error("gRPC error: {message}")]
    GrpcError { message: String },

    /// HTTP specific error
    #[error("HTTP error {status_code}: {message}")]
    HttpError { status_code: u16, message: String },

    /// Authentication failed
    #[error("Authentication failed: {reason}")]
    AuthenticationFailed { reason: String },

    /// Authorization failed
    #[error("Authorization failed: {reason}")]
    AuthorizationFailed { reason: String },

    /// Invalid response format
    #[error("Invalid response format: {details}")]
    InvalidResponse { details: String },

    /// Network interface or configuration error
    #[error("Network configuration error: {details}")]
    ConfigurationError { details: String },
}

impl BasilcaError for NetworkError {}

/// Cryptographic operation errors
///
/// These errors occur during cryptographic operations like hashing,
/// signature verification, encryption, or key management.
#[derive(Error, Debug)]
pub enum CryptoError {
    /// Hash computation failed
    #[error("Hash computation failed: {algorithm}")]
    HashFailed { algorithm: String },

    /// Signature verification failed
    #[error("Signature verification failed for hotkey {hotkey}")]
    SignatureVerificationFailed { hotkey: String },

    /// Invalid signature format
    #[error("Invalid signature format: {details}")]
    InvalidSignature { details: String },

    /// Invalid public key format
    #[error("Invalid public key format: {details}")]
    InvalidPublicKey { details: String },

    /// Encryption failed
    #[error("Encryption failed: {details}")]
    EncryptionFailed { details: String },

    /// Decryption failed
    #[error("Decryption failed: {details}")]
    DecryptionFailed { details: String },

    /// Key generation failed
    #[error("Key generation failed: {details}")]
    KeyGenerationFailed { details: String },

    /// Key derivation failed
    #[error("Key derivation failed: {details}")]
    KeyDerivationFailed { details: String },

    /// Random number generation failed
    #[error("Random number generation failed")]
    RandomGenerationFailed,

    /// Generic cryptographic error
    #[error("Cryptographic error: {message}")]
    Generic { message: String },
}

impl BasilcaError for CryptoError {}

/// Configuration-related errors
///
/// These errors occur during configuration loading, parsing, or validation.
#[derive(Error, Debug)]
pub enum ConfigurationError {
    /// Configuration file not found
    #[error("Configuration file not found: {path}")]
    FileNotFound { path: String },

    /// Configuration file cannot be read
    #[error("Cannot read configuration file {path}: {source}")]
    ReadError {
        path: String,
        #[source]
        source: Box<dyn std::error::Error + Send + Sync>,
    },

    /// Configuration parsing failed
    #[error("Failed to parse configuration: {details}")]
    ParseError { details: String },

    /// Invalid configuration value
    #[error("Invalid configuration value for {key}: {value} ({reason})")]
    InvalidValue {
        key: String,
        value: String,
        reason: String,
    },

    /// Missing required configuration
    #[error("Missing required configuration: {key}")]
    MissingRequired { key: String },

    /// Environment variable error
    #[error("Environment variable error for {var}: {details}")]
    EnvironmentError { var: String, details: String },

    /// Configuration validation failed
    #[error("Configuration validation failed: {details}")]
    ValidationFailed { details: String },
}

impl BasilcaError for ConfigurationError {}

/// Database and persistence-related errors
///
/// These errors occur during database operations, file I/O, or other
/// persistence operations.
#[derive(Error, Debug)]
pub enum PersistenceError {
    /// Database connection failed
    #[error("Database connection failed: {source}")]
    ConnectionFailed {
        #[source]
        source: Box<dyn std::error::Error + Send + Sync>,
    },

    /// Database query failed
    #[error("Database query failed: {query}")]
    QueryFailed { query: String },

    /// Database transaction failed
    #[error("Database transaction failed: {details}")]
    TransactionFailed { details: String },

    /// Database migration failed
    #[error("Database migration failed: {details}")]
    MigrationFailed { details: String },

    /// File I/O error
    #[error("File I/O error for {path}: {source}")]
    FileError {
        path: String,
        #[source]
        source: Box<dyn std::error::Error + Send + Sync>,
    },

    /// Serialization failed
    #[error("Serialization failed: {details}")]
    SerializationFailed { details: String },

    /// Deserialization failed
    #[error("Deserialization failed: {details}")]
    DeserializationFailed { details: String },

    /// Data corruption detected
    #[error("Data corruption detected in {location}: {details}")]
    DataCorruption { location: String, details: String },

    /// Constraint violation
    #[error("Database constraint violation: {constraint}")]
    ConstraintViolation { constraint: String },

    /// Record not found
    #[error("Record not found: {details}")]
    NotFound { details: String },
}

impl BasilcaError for PersistenceError {}

/// System-level errors
///
/// These errors occur during system operations like process management,
/// resource access, or hardware interaction.
#[derive(Error, Debug)]
pub enum SystemError {
    /// Process execution failed
    #[error("Process execution failed: {command}")]
    ProcessFailed { command: String },

    /// Resource unavailable
    #[error("Resource unavailable: {resource}")]
    ResourceUnavailable { resource: String },

    /// Permission denied
    #[error("Permission denied for operation: {operation}")]
    PermissionDenied { operation: String },

    /// Hardware error
    #[error("Hardware error: {component} - {details}")]
    HardwareError { component: String, details: String },

    /// System configuration error
    #[error("System configuration error: {details}")]
    ConfigurationError { details: String },

    /// Resource limit exceeded
    #[error("Resource limit exceeded: {resource} ({limit})")]
    ResourceLimitExceeded { resource: String, limit: String },

    /// Validation error for system specs
    #[error("System validation failed for {component}: {message}")]
    ValidationError { component: String, message: String },
}

impl BasilcaError for SystemError {}

/// Validation errors
///
/// These errors occur during input validation or data format checking.
#[derive(Error, Debug)]
pub enum ValidationError {
    /// Invalid input format
    #[error("Invalid {field} format: {value}")]
    InvalidFormat { field: String, value: String },

    /// Value out of range
    #[error("{field} value {value} is out of range [{min}, {max}]")]
    OutOfRange {
        field: String,
        value: String,
        min: String,
        max: String,
    },

    /// Required field missing
    #[error("Required field missing: {field}")]
    MissingField { field: String },

    /// Field constraint violation
    #[error("Field constraint violation for {field}: {constraint}")]
    ConstraintViolation { field: String, constraint: String },

    /// Invalid enum value
    #[error("Invalid {enum_name} value: {value}")]
    InvalidEnum { enum_name: String, value: String },
}

impl BasilcaError for ValidationError {}

/// Verification-related errors
///
/// These errors occur during verification processes like executor validation,
/// challenge verification, or system integrity checks.
#[derive(Error, Debug)]
pub enum VerificationError {
    /// Invalid verification data format
    #[error("Invalid verification data: {details}")]
    InvalidData { details: String },

    /// Verification challenge failed
    #[error("Challenge verification failed: {challenge_type}")]
    ChallengeFailed { challenge_type: String },

    /// System profile verification failed
    #[error("System profile verification failed: {details}")]
    ProfileVerificationFailed { details: String },

    /// Executor integrity check failed
    #[error("Executor integrity check failed: {details}")]
    IntegrityCheckFailed { details: String },

    /// Verification timeout
    #[error("Verification timeout after {timeout_secs} seconds")]
    Timeout { timeout_secs: u64 },

    /// Generic verification error
    #[error("Verification error: {details}")]
    Generic { details: String },
}

impl BasilcaError for VerificationError {}

/// Result type alias for common Basilca operations
pub type BasilcaResult<T, E = Box<dyn BasilcaError>> = Result<T, E>;

/// Utility functions for error handling
impl NetworkError {
    /// Create a connection failed error from any error type
    pub fn connection_failed(
        endpoint: impl Into<String>,
        source: impl std::error::Error + Send + Sync + 'static,
    ) -> Self {
        Self::ConnectionFailed {
            endpoint: endpoint.into(),
            source: Box::new(source),
        }
    }

    /// Create a gRPC error from tonic status
    pub fn from_grpc_status(status: &str) -> Self {
        Self::GrpcError {
            message: status.to_string(),
        }
    }
}

impl CryptoError {
    /// Create a generic crypto error
    pub fn generic(message: impl Into<String>) -> Self {
        Self::Generic {
            message: message.into(),
        }
    }
}

impl ConfigurationError {
    /// Create a validation failed error
    pub fn validation_failed(details: impl Into<String>) -> Self {
        Self::ValidationFailed {
            details: details.into(),
        }
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use std::error::Error;

    #[test]
    fn test_error_display() {
        let network_err = NetworkError::ConnectionFailed {
            endpoint: "localhost:8080".to_string(),
            source: Box::new(std::io::Error::new(
                std::io::ErrorKind::ConnectionRefused,
                "Connection refused",
            )),
        };

        let display = format!("{network_err}");
        assert!(display.contains("localhost:8080"));
        assert!(display.contains("Failed to connect"));
    }

    #[test]
    fn test_error_source_chain() {
        let io_error = std::io::Error::new(std::io::ErrorKind::PermissionDenied, "Access denied");
        let config_err = ConfigurationError::ReadError {
            path: "/etc/config.toml".to_string(),
            source: Box::new(io_error),
        };

        assert!(config_err.source().is_some());
    }

    #[test]
    fn test_basilca_error_trait() {
        fn test_basilca_error(_: impl BasilcaError) {}

        // These should compile, proving they implement BasilcaError
        test_basilca_error(NetworkError::ConnectionLost {
            endpoint: "test".to_string(),
        });
        test_basilca_error(CryptoError::RandomGenerationFailed);
        test_basilca_error(ConfigurationError::ValidationFailed {
            details: "test".to_string(),
        });
    }

    #[test]
    fn test_utility_functions() {
        let io_error = std::io::Error::new(std::io::ErrorKind::TimedOut, "Timeout");
        let network_err = NetworkError::connection_failed("example.com:443", io_error);

        match network_err {
            NetworkError::ConnectionFailed { endpoint, .. } => {
                assert_eq!(endpoint, "example.com:443");
            }
            _ => panic!("Expected ConnectionFailed variant"),
        }
    }
}
