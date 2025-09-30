//! Error types for the Basilica SDK

use serde::{Deserialize, Serialize};
use thiserror::Error;

/// Main error type for the Basilica SDK
#[derive(Debug, Error)]
pub enum ApiError {
    /// HTTP client error
    #[error("HTTP client error: {0}")]
    HttpClient(#[from] reqwest::Error),

    /// Missing authentication (no token provided)
    #[error("Authentication required: {message}")]
    MissingAuthentication { message: String },

    /// Authentication error (expired/invalid token)
    #[error("Authentication error: {message}")]
    Authentication { message: String },

    /// Authorization error
    #[error("Authorization error: {message}")]
    Authorization { message: String },

    /// Rate limit exceeded
    #[error("Rate limit exceeded")]
    RateLimitExceeded,

    /// Invalid request
    #[error("Invalid request: {message}")]
    InvalidRequest { message: String },

    /// Not found
    #[error("Resource not found: {resource}")]
    NotFound { resource: String },

    /// Bad request with message
    #[error("Bad request: {message}")]
    BadRequest { message: String },

    /// Conflict error (e.g., duplicate resource)
    #[error("Conflict: {message}")]
    Conflict { message: String },

    /// Internal server error
    #[error("Internal server error: {message}")]
    Internal { message: String },

    /// Service unavailable
    #[error("Service temporarily unavailable")]
    ServiceUnavailable,

    /// Timeout error
    #[error("Request timeout")]
    Timeout,

    /// Validator communication error
    #[error("Validator communication error: {message}")]
    ValidatorCommunication { message: String },
}

/// Result type alias
pub type Result<T> = std::result::Result<T, ApiError>;

impl ApiError {
    /// Get error code for this error
    pub fn error_code(&self) -> &'static str {
        match self {
            ApiError::HttpClient(_) => "BASILICA_API_HTTP_CLIENT_ERROR",
            ApiError::MissingAuthentication { .. } => "BASILICA_API_AUTH_MISSING",
            ApiError::Authentication { .. } => "BASILICA_API_AUTH_ERROR",
            ApiError::Authorization { .. } => "BASILICA_API_AUTHZ_ERROR",
            ApiError::RateLimitExceeded => "BASILICA_API_RATE_LIMIT",
            ApiError::InvalidRequest { .. } => "BASILICA_API_INVALID_REQUEST",
            ApiError::NotFound { .. } => "BASILICA_API_NOT_FOUND",
            ApiError::BadRequest { .. } => "BASILICA_API_BAD_REQUEST",
            ApiError::Conflict { .. } => "BASILICA_API_CONFLICT",
            ApiError::Internal { .. } => "BASILICA_API_INTERNAL_ERROR",
            ApiError::ServiceUnavailable => "BASILICA_API_SERVICE_UNAVAILABLE",
            ApiError::Timeout => "BASILICA_API_TIMEOUT",
            ApiError::ValidatorCommunication { .. } => "BASILICA_API_VALIDATOR_COMM_ERROR",
        }
    }

    /// Check if error is retryable
    pub fn is_retryable(&self) -> bool {
        matches!(
            self,
            ApiError::HttpClient(_)
                | ApiError::ValidatorCommunication { .. }
                | ApiError::Timeout
                | ApiError::ServiceUnavailable
        )
    }

    /// Check if error is a client error
    pub fn is_client_error(&self) -> bool {
        matches!(
            self,
            ApiError::MissingAuthentication { .. }
                | ApiError::Authentication { .. }
                | ApiError::Authorization { .. }
                | ApiError::RateLimitExceeded
                | ApiError::InvalidRequest { .. }
                | ApiError::NotFound { .. }
                | ApiError::BadRequest { .. }
                | ApiError::Conflict { .. }
        )
    }
}

/// Error response structure from API
#[derive(Debug, Serialize, Deserialize)]
pub struct ErrorResponse {
    /// Error details
    pub error: ErrorDetails,
}

/// Error details structure
#[derive(Debug, Serialize, Deserialize)]
pub struct ErrorDetails {
    /// Error code
    pub code: String,

    /// Human-readable error message
    pub message: String,

    /// ISO 8601 timestamp
    pub timestamp: chrono::DateTime<chrono::Utc>,

    /// Whether the error is retryable
    pub retryable: bool,
}
