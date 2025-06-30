//! Rate limiting configuration

use serde::{Deserialize, Serialize};

/// Rate limiting configuration
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct RateLimitConfig {
    /// Default requests per minute
    pub default_requests_per_minute: u32,

    /// Burst size
    pub burst_size: u32,

    /// Enable per-IP rate limiting
    pub per_ip_limiting: bool,

    /// Premium tier requests per minute
    pub premium_requests_per_minute: u32,

    /// Rate limit storage backend
    pub storage_backend: RateLimitBackend,
}

/// Rate limit storage backends
#[derive(Debug, Clone, Serialize, Deserialize)]
#[serde(rename_all = "snake_case")]
pub enum RateLimitBackend {
    /// In-memory storage
    InMemory,

    /// Redis storage
    Redis,
}

impl Default for RateLimitConfig {
    fn default() -> Self {
        Self {
            default_requests_per_minute: 60,
            burst_size: 100,
            per_ip_limiting: true,
            premium_requests_per_minute: 600,
            storage_backend: RateLimitBackend::InMemory,
        }
    }
}
