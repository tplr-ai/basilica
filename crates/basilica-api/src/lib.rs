//! # Basilica API Gateway
//!
//! A smart HTTP gateway that provides centralized access to a single Basilica validator.
//!
//! ## Features
//!
//! - **Direct Connection**: Direct connection to a specific validator by hotkey configured in settings
//! - **Health Monitoring**: Continuous health checking of the connected validator
//! - **Authentication**: API key and JWT-based authentication
//! - **Rate Limiting**: Configurable rate limits with different tiers
//! - **Caching**: Response caching with in-memory or Redis backends
//! - **OpenAPI Documentation**: Auto-generated API documentation

// Server modules (always available for backward compatibility)
pub mod api;
pub mod config;
pub mod country_mapping;
pub mod error;
pub mod server;

// Re-export commonly used types
pub use config::Config;
pub use error::{ApiError, Result};
pub use server::Server;

/// Version of the basilica-api crate
pub const VERSION: &str = env!("CARGO_PKG_VERSION");

/// Protocol version for API compatibility
pub const API_VERSION: &str = "v1";

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_version_constants() {
        assert_eq!(API_VERSION, "v1");
    }
}
