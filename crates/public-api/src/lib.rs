//! # Basilica Public API Gateway
//!
//! A smart HTTP gateway that provides centralized access to the Basilica validator network.
//!
//! ## Features
//!
//! - **Validator Discovery**: Automatic discovery of validators using Bittensor metagraph
//! - **Load Balancing**: Multiple strategies for distributing requests across validators
//! - **Request Aggregation**: Combine responses from multiple validators
//! - **Authentication**: API key and JWT-based authentication
//! - **Rate Limiting**: Configurable rate limits with different tiers
//! - **Caching**: Response caching with in-memory or Redis backends
//! - **OpenAPI Documentation**: Auto-generated API documentation
//! - **Monitoring**: Prometheus metrics and distributed tracing

pub mod aggregator;
pub mod api;
pub mod config;
pub mod discovery;
pub mod error;
pub mod load_balancer;
pub mod server;

// Re-export commonly used types
pub use config::Config;
pub use error::{Error, Result};
pub use server::Server;

/// Version of the public-api crate
pub const VERSION: &str = env!("CARGO_PKG_VERSION");

/// Protocol version for API compatibility
pub const API_VERSION: &str = "v1";

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_version_constants() {
        assert!(!VERSION.is_empty());
        assert_eq!(API_VERSION, "v1");
    }
}
