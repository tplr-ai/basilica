//! Caching utilities for federation

use moka::future::Cache;
use std::hash::Hash;
use std::time::Duration;

/// Cache builder for federation components
pub struct CacheBuilder<K, V> {
    max_capacity: Option<u64>,
    time_to_live: Option<Duration>,
    time_to_idle: Option<Duration>,
}

impl<K, V> CacheBuilder<K, V>
where
    K: Hash + Eq + Send + Sync + 'static,
    V: Clone + Send + Sync + 'static,
{
    /// Create a new cache builder
    pub fn new() -> Self {
        Self {
            max_capacity: None,
            time_to_live: None,
            time_to_idle: None,
        }
    }
    
    /// Set maximum capacity
    pub fn max_capacity(mut self, capacity: u64) -> Self {
        self.max_capacity = Some(capacity);
        self
    }
    
    /// Set time to live
    pub fn time_to_live(mut self, ttl: Duration) -> Self {
        self.time_to_live = Some(ttl);
        self
    }
    
    /// Set time to idle
    pub fn time_to_idle(mut self, tti: Duration) -> Self {
        self.time_to_idle = Some(tti);
        self
    }
    
    /// Build the cache
    pub fn build(self) -> Cache<K, V> {
        let mut builder = Cache::builder();
        
        if let Some(capacity) = self.max_capacity {
            builder = builder.max_capacity(capacity);
        }
        
        if let Some(ttl) = self.time_to_live {
            builder = builder.time_to_live(ttl);
        }
        
        if let Some(tti) = self.time_to_idle {
            builder = builder.time_to_idle(tti);
        }
        
        builder.build()
    }
}

impl<K, V> Default for CacheBuilder<K, V>
where
    K: Hash + Eq + Send + Sync + 'static,
    V: Clone + Send + Sync + 'static,
{
    fn default() -> Self {
        Self::new()
    }
}

/// Service cache for federation
pub type ServiceCache = Cache<String, Vec<crate::discovery::ServiceInfo>>;

/// Health cache for federation
pub type HealthCache = Cache<String, crate::health::ClusterHealth>;

/// Create a service cache with default settings
pub fn create_service_cache(ttl: Duration) -> ServiceCache {
    CacheBuilder::new()
        .max_capacity(10_000)
        .time_to_live(ttl)
        .build()
}

/// Create a health cache with default settings
pub fn create_health_cache(ttl: Duration) -> HealthCache {
    CacheBuilder::new()
        .max_capacity(100)
        .time_to_live(ttl)
        .build()
}

