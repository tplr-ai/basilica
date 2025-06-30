//! Server configuration

use serde::{Deserialize, Serialize};
use std::net::SocketAddr;

/// Server configuration
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct ServerConfig {
    /// Server bind address
    pub bind_address: SocketAddr,

    /// Maximum number of concurrent connections
    pub max_connections: usize,

    /// Request timeout in seconds
    pub request_timeout: u64,

    /// Enable compression
    pub enable_compression: bool,

    /// CORS allowed origins
    pub cors_origins: Vec<String>,
}

impl Default for ServerConfig {
    fn default() -> Self {
        Self {
            bind_address: "0.0.0.0:8000".parse().unwrap(),
            max_connections: 10000,
            request_timeout: 30,
            enable_compression: true,
            cors_origins: vec!["*".to_string()],
        }
    }
}
