//! Rate limiting middleware

use crate::{error::Error, server::AppState};
use axum::{
    extract::{ConnectInfo, Request},
    http::StatusCode,
    middleware::Next,
    response::Response,
};
use dashmap::DashMap;
use governor::{
    clock::DefaultClock,
    state::{InMemoryState, NotKeyed},
    Quota, RateLimiter,
};
use std::{net::SocketAddr, sync::Arc, time::Duration};

/// Rate limit key type
#[derive(Debug, Clone, Hash, PartialEq, Eq)]
pub(super) enum RateLimitKey {
    /// IP-based rate limiting
    Ip(String),
    /// API key-based rate limiting
    ApiKey(String),
}

/// Type alias for rate limiter
type RateLimiterType = Arc<RateLimiter<NotKeyed, InMemoryState, DefaultClock>>;

/// Rate limiter storage
pub struct RateLimitStorage {
    /// Default limiter for anonymous requests
    default_limiter: RateLimiterType,
    /// Per-IP limiters
    ip_limiters: Arc<DashMap<String, RateLimiterType>>,
    /// Per-API key limiters
    api_key_limiters: Arc<DashMap<String, RateLimiterType>>,
    /// Configuration
    config: Arc<crate::config::RateLimitConfig>,
}

impl RateLimitStorage {
    /// Create new rate limit storage
    pub fn new(config: Arc<crate::config::RateLimitConfig>) -> Self {
        let default_quota = Quota::per_minute(
            std::num::NonZeroU32::new(config.default_requests_per_minute)
                .unwrap_or(std::num::NonZeroU32::new(60).unwrap()),
        );

        Self {
            default_limiter: Arc::new(RateLimiter::direct(default_quota)),
            ip_limiters: Arc::new(DashMap::new()),
            api_key_limiters: Arc::new(DashMap::new()),
            config,
        }
    }

    /// Get or create limiter for IP
    fn get_ip_limiter(&self, ip: &str) -> Arc<RateLimiter<NotKeyed, InMemoryState, DefaultClock>> {
        self.ip_limiters
            .entry(ip.to_string())
            .or_insert_with(|| {
                let quota = Quota::per_minute(
                    std::num::NonZeroU32::new(self.config.default_requests_per_minute)
                        .unwrap_or(std::num::NonZeroU32::new(60).unwrap()),
                );
                Arc::new(RateLimiter::direct(quota))
            })
            .clone()
    }

    /// Get or create limiter for API key
    fn get_api_key_limiter(
        &self,
        api_key: &str,
    ) -> Arc<RateLimiter<NotKeyed, InMemoryState, DefaultClock>> {
        self.api_key_limiters
            .entry(api_key.to_string())
            .or_insert_with(|| {
                // Determine rate limit based on API key pattern
                let requests_per_minute = if api_key.starts_with("sk_enterprise_") {
                    6000 // Enterprise tier: 100 requests per second
                } else if api_key.starts_with("sk_premium_") || api_key.starts_with("sk_live_") {
                    self.config.premium_requests_per_minute
                } else if api_key.starts_with("sk_test_") {
                    300 // Test tier: 5 requests per second
                } else {
                    self.config.default_requests_per_minute
                };

                let quota = Quota::per_minute(
                    std::num::NonZeroU32::new(requests_per_minute)
                        .unwrap_or(std::num::NonZeroU32::new(60).unwrap()),
                );
                Arc::new(RateLimiter::direct(quota))
            })
            .clone()
    }

    /// Check rate limit
    pub async fn check_limit(&self, key: RateLimitKey) -> Result<(), Error> {
        let limiter = match &key {
            RateLimitKey::Ip(ip) if self.config.per_ip_limiting => self.get_ip_limiter(ip),
            RateLimitKey::ApiKey(api_key) => self.get_api_key_limiter(api_key),
            _ => self.default_limiter.clone(),
        };

        match limiter.check() {
            Ok(_) => Ok(()),
            Err(_) => Err(Error::RateLimitExceeded),
        }
    }

    /// Clean up old entries periodically
    pub async fn cleanup(&self) {
        // For now, we'll keep all entries
        // In production, you'd track last access time for each limiter
        // and remove entries that haven't been used in the last hour

        // Optionally clear very old IP limiters if the map grows too large
        if self.ip_limiters.len() > 10000 {
            // Remove random entries to keep size manageable
            let to_remove: Vec<String> = self
                .ip_limiters
                .iter()
                .take(1000)
                .map(|entry| entry.key().clone())
                .collect();

            for key in to_remove {
                self.ip_limiters.remove(&key);
            }
        }
    }
}

/// Rate limit middleware
#[derive(Clone)]
pub struct RateLimitMiddleware {
    #[allow(dead_code)]
    storage: Arc<RateLimitStorage>,
    #[allow(dead_code)]
    config: Arc<crate::config::Config>,
}

impl RateLimitMiddleware {
    /// Create new rate limit middleware
    pub fn new(state: AppState) -> Self {
        let storage = Arc::new(RateLimitStorage::new(Arc::new(
            state.config.rate_limit.clone(),
        )));

        // Start cleanup task
        let storage_clone = storage.clone();
        tokio::spawn(async move {
            let mut interval = tokio::time::interval(Duration::from_secs(300)); // 5 minutes
            loop {
                interval.tick().await;
                storage_clone.cleanup().await;
            }
        });

        Self {
            storage,
            config: state.config.clone(),
        }
    }

    /// Extract rate limit key from request
    #[allow(dead_code)]
    fn extract_key(req: &Request) -> RateLimitKey {
        // First check for API key
        if let Some(api_key) = req.headers().get("X-API-Key").and_then(|h| h.to_str().ok()) {
            return RateLimitKey::ApiKey(api_key.to_string());
        }

        // Fall back to IP address
        if let Some(ConnectInfo(addr)) = req.extensions().get::<ConnectInfo<SocketAddr>>() {
            return RateLimitKey::Ip(addr.ip().to_string());
        }

        // Default to anonymous
        RateLimitKey::Ip("anonymous".to_string())
    }
}

/// Rate limit handler for axum middleware
pub async fn rate_limit_middleware(
    storage: Arc<RateLimitStorage>,
    req: Request,
    next: Next,
) -> Result<Response, StatusCode> {
    // Extract rate limit key
    let key = match req.headers().get("X-API-Key").and_then(|h| h.to_str().ok()) {
        Some(api_key) => RateLimitKey::ApiKey(api_key.to_string()),
        None => {
            // Try to get IP from connection info
            if let Some(ConnectInfo(addr)) = req.extensions().get::<ConnectInfo<SocketAddr>>() {
                RateLimitKey::Ip(addr.ip().to_string())
            } else {
                RateLimitKey::Ip("anonymous".to_string())
            }
        }
    };

    // Check rate limit
    match storage.check_limit(key).await {
        Ok(_) => Ok(next.run(req).await),
        Err(_) => Err(StatusCode::TOO_MANY_REQUESTS),
    }
}
