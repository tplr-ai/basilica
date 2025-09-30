//! Error types for the Basilica API gateway

use axum::{
    http::StatusCode,
    response::{IntoResponse, Response},
    Json,
};
use basilica_common::BasilicaError;
use serde_json::json;
use thiserror::Error;

/// Main error type for the Basilica API
#[derive(Debug, Error)]
pub enum ApiError {
    /// Configuration error
    #[error("Configuration error: {0}")]
    Config(#[from] basilica_common::ConfigurationError),

    /// Configuration error string
    #[error("Configuration error: {0}")]
    ConfigError(String),

    /// Bittensor integration error
    #[error("Bittensor error: {0}")]
    Bittensor(#[from] bittensor::BittensorError),

    /// HTTP client error
    #[error("HTTP client error: {0}")]
    HttpClient(#[from] reqwest::Error),

    /// Validator communication error
    #[error("Validator communication error: {message}")]
    ValidatorCommunication { message: String },

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

    /// Aggregation error
    #[error("Aggregation error: {message}")]
    Aggregation { message: String },

    /// Cache error
    #[error("Cache error: {message}")]
    Cache { message: String },

    /// Timeout error
    #[error("Request timeout")]
    Timeout,

    /// Internal server error
    #[error("Internal server error: {message}")]
    Internal { message: String },

    /// Service unavailable
    #[error("Service temporarily unavailable")]
    ServiceUnavailable,

    /// Not found
    #[error("{message}")]
    NotFound { message: String },

    /// Bad request with message
    #[error("Bad request: {message}")]
    BadRequest { message: String },

    /// Conflict error
    #[error("Conflict: {message}")]
    Conflict { message: String },

    /// Serialization error
    #[error("Serialization error: {0}")]
    Serialization(#[from] serde_json::Error),

    /// Other errors
    #[error("{0}")]
    Other(#[from] anyhow::Error),
}

/// Result type alias
pub type Result<T> = std::result::Result<T, ApiError>;

impl BasilicaError for ApiError {}

impl ApiError {
    /// Get error code for this error
    pub fn error_code(&self) -> &'static str {
        match self {
            ApiError::Config(_) => "BASILICA_API_CONFIG_ERROR",
            ApiError::Bittensor(_) => "BASILICA_API_BITTENSOR_ERROR",
            ApiError::HttpClient(_) => "BASILICA_API_HTTP_CLIENT_ERROR",
            ApiError::ValidatorCommunication { .. } => "BASILICA_API_VALIDATOR_COMM_ERROR",
            ApiError::ConfigError(_) => "BASILICA_API_CONFIG_ERROR",
            ApiError::MissingAuthentication { .. } => "BASILICA_API_AUTH_MISSING",
            ApiError::Authentication { .. } => "BASILICA_API_AUTH_ERROR",
            ApiError::Authorization { .. } => "BASILICA_API_AUTHZ_ERROR",
            ApiError::RateLimitExceeded => "BASILICA_API_RATE_LIMIT",
            ApiError::InvalidRequest { .. } => "BASILICA_API_INVALID_REQUEST",
            ApiError::Aggregation { .. } => "BASILICA_API_AGGREGATION_ERROR",
            ApiError::Cache { .. } => "BASILICA_API_CACHE_ERROR",
            ApiError::Timeout => "BASILICA_API_TIMEOUT",
            ApiError::Internal { .. } => "BASILICA_API_INTERNAL_ERROR",
            ApiError::ServiceUnavailable => "BASILICA_API_SERVICE_UNAVAILABLE",
            ApiError::NotFound { .. } => "BASILICA_API_NOT_FOUND",
            ApiError::BadRequest { .. } => "BASILICA_API_BAD_REQUEST",
            ApiError::Conflict { .. } => "BASILICA_API_CONFLICT",
            ApiError::Serialization(_) => "BASILICA_API_SERIALIZATION_ERROR",
            ApiError::Other(_) => "BASILICA_API_OTHER_ERROR",
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

impl IntoResponse for ApiError {
    fn into_response(self) -> Response {
        let (status, error_message) = match &self {
            ApiError::Config(_) => (StatusCode::INTERNAL_SERVER_ERROR, self.to_string()),
            ApiError::Bittensor(_) => (StatusCode::SERVICE_UNAVAILABLE, self.to_string()),
            ApiError::HttpClient(_) => (StatusCode::BAD_GATEWAY, self.to_string()),
            ApiError::ValidatorCommunication { .. } => (StatusCode::BAD_GATEWAY, self.to_string()),
            ApiError::ConfigError(_) => (StatusCode::INTERNAL_SERVER_ERROR, self.to_string()),
            ApiError::MissingAuthentication { .. } => (StatusCode::UNAUTHORIZED, self.to_string()),
            ApiError::Authentication { .. } => (StatusCode::UNAUTHORIZED, self.to_string()),
            ApiError::Authorization { .. } => (StatusCode::FORBIDDEN, self.to_string()),
            ApiError::RateLimitExceeded => (
                StatusCode::TOO_MANY_REQUESTS,
                "Too many requests. Please try again later.".to_string(),
            ),
            ApiError::InvalidRequest { .. } => (StatusCode::BAD_REQUEST, self.to_string()),
            ApiError::Aggregation { .. } => (StatusCode::INTERNAL_SERVER_ERROR, self.to_string()),
            ApiError::Cache { .. } => (StatusCode::INTERNAL_SERVER_ERROR, self.to_string()),
            ApiError::Timeout => (StatusCode::REQUEST_TIMEOUT, self.to_string()),
            ApiError::Internal { .. } => (StatusCode::INTERNAL_SERVER_ERROR, self.to_string()),
            ApiError::ServiceUnavailable => (StatusCode::SERVICE_UNAVAILABLE, self.to_string()),
            ApiError::NotFound { .. } => (StatusCode::NOT_FOUND, self.to_string()),
            ApiError::BadRequest { .. } => (StatusCode::BAD_REQUEST, self.to_string()),
            ApiError::Conflict { .. } => (StatusCode::CONFLICT, self.to_string()),
            ApiError::Serialization(_) => (StatusCode::INTERNAL_SERVER_ERROR, self.to_string()),
            ApiError::Other(_) => (StatusCode::INTERNAL_SERVER_ERROR, self.to_string()),
        };

        let body = Json(json!({
            "error": {
                "code": self.error_code(),
                "message": error_message,
                "timestamp": chrono::Utc::now(),
                "retryable": self.is_retryable(),
            }
        }));

        (status, body).into_response()
    }
}

/// Error response structure for API documentation
#[derive(Debug, serde::Serialize, serde::Deserialize)]
pub struct ErrorResponse {
    /// Error details
    pub error: ErrorDetails,
}

/// Error details structure
#[derive(Debug, serde::Serialize, serde::Deserialize)]
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

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_error_codes() {
        assert_eq!(
            ApiError::RateLimitExceeded.error_code(),
            "BASILICA_API_RATE_LIMIT"
        );
        assert_eq!(ApiError::Timeout.error_code(), "BASILICA_API_TIMEOUT");
    }

    #[test]
    fn test_retryable_errors() {
        assert!(ApiError::Timeout.is_retryable());
        assert!(ApiError::ServiceUnavailable.is_retryable());
        assert!(!ApiError::Authentication {
            message: "test".to_string()
        }
        .is_retryable());
    }

    #[test]
    fn test_client_errors() {
        assert!(ApiError::MissingAuthentication {
            message: "test".to_string()
        }
        .is_client_error());
        assert!(ApiError::Authentication {
            message: "test".to_string()
        }
        .is_client_error());
        assert!(ApiError::RateLimitExceeded.is_client_error());
        assert!(!ApiError::Timeout.is_client_error());
    }
}
