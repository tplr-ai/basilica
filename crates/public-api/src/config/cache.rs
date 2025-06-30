//! Cache configuration

use serde::{Deserialize, Serialize};

/// Cache configuration
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct CacheConfig {
    /// Cache backend type
    pub backend: CacheBackend,

    /// Default TTL in seconds
    pub default_ttl: u64,

    /// Maximum cache size (in-memory only)
    pub max_size: usize,

    /// Redis connection URL (if using Redis backend)
    pub redis_url: Option<String>,

    /// Cache key prefix
    pub key_prefix: String,
}

/// Cache backend types
#[derive(Debug, Clone, Serialize, Deserialize)]
#[serde(rename_all = "snake_case")]
pub enum CacheBackend {
    /// In-memory cache
    InMemory,

    /// Redis cache
    Redis,
}

impl Default for CacheConfig {
    fn default() -> Self {
        Self {
            backend: CacheBackend::InMemory,
            default_ttl: 300,
            max_size: 10000,
            redis_url: None,
            key_prefix: "basilica:public-api:".to_string(),
        }
    }
}
