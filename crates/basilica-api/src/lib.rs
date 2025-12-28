//! # basilica-api
//!
//! Smart HTTP gateway for Basilica validator network with load balancing and caching.
//!
//! This crate provides an HTTP gateway that enables easy access to the Basilica network.
//! It handles authentication, load balancing, caching, and request aggregation for
//! optimal performance and reliability.
//!
//! ## Overview
//!
//! `basilica-api` provides:
//!
//! - **Load Balancing**: Smart distribution of requests across validators
//! - **Request Aggregation**: Combine similar requests for efficiency
//! - **Authentication**: API key and JWT-based authentication
//! - **Rate Limiting**: Protect backends from overload with configurable limits
//! - **Caching**: Response caching for improved latency
//! - **WebSocket Support**: Real-time streaming capabilities
//!
//! ## Quick Start
//!
//! ```rust,ignore
//! use basilica_api::{Config, Server};
//!
//! #[tokio::main]
//! async fn main() -> Result<(), Box<dyn std::error::Error>> {
//!     // Load configuration
//!     let config = Config::load("api.toml")?;
//!     
//!     // Create and start the API server
//!     let server = Server::new(config).await?;
//!     server.run().await?;
//!     
//!     Ok(())
//! }
//! ```
//!
//! ## Feature Flags
//!
//! - `server` - Enable HTTP server functionality (default)
//! - `client` - Enable HTTP client functionality
//! - `utoipa` - Enable OpenAPI documentation generation
//! - `full` - Enable all features
//!
//! ## API Endpoints
//!
//! | Endpoint | Method | Description |
//! |----------|--------|-------------|
//! | `/health` | GET | Health check |
//! | `/miners` | GET | List available miners |
//! | `/rentals` | POST | Create a GPU rental |
//! | `/rentals/{id}` | GET | Get rental status |
//! | `/ws` | WS | WebSocket for streaming |
//!
//! ## Related Crates
//!
//! - [`basilica-sdk`](https://crates.io/crates/basilica-sdk) - Client SDK
//! - [`basilica-validator`](https://crates.io/crates/basilica-validator) - Validator implementation
//! - [`basilica-billing`](https://crates.io/crates/basilica-billing) - Billing service

// Server modules (always available for backward compatibility)
pub mod api;
pub mod apimetrics;
pub mod config;
pub mod country_mapping;
pub mod db;
pub mod dns;
pub mod envoy;
pub mod error;
pub mod gateway;
pub mod k8s;
pub mod metrics;
pub mod server;
pub mod ssh;
pub mod wireguard;

// Backward compatibility alias
pub use k8s as k8s_client;

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
