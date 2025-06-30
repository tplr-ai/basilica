//! Caching middleware

use crate::server::AppState;
use axum::{
    body::Body,
    extract::Request,
    http::{Method, StatusCode},
    middleware::Next,
    response::Response,
};
use moka::future::Cache;
use serde::{Deserialize, Serialize};
use sha2::{Digest, Sha256};
use std::{
    sync::Arc,
    time::{Duration, SystemTime},
};

/// Cache key components
#[derive(Debug, Hash, PartialEq, Eq)]
struct CacheKey {
    /// HTTP method
    method: String,
    /// Request path
    path: String,
    /// Query string (sorted)
    query: String,
    /// API key (for user-specific caching)
    api_key: Option<String>,
}

/// Cached response data
#[derive(Debug, Clone, Serialize, Deserialize)]
struct CachedResponse {
    /// Response status code
    status: u16,
    /// Response headers
    headers: Vec<(String, String)>,
    /// Response body
    body: Vec<u8>,
    /// Cache expiration time
    expires_at: SystemTime,
}

/// Cache storage backend
pub struct CacheStorage {
    /// In-memory cache using moka
    memory_cache: Cache<String, CachedResponse>,
    /// Configuration
    config: Arc<crate::config::CacheConfig>,
}

impl CacheStorage {
    /// Create new cache storage
    pub fn new(config: Arc<crate::config::CacheConfig>) -> Self {
        let memory_cache = Cache::builder()
            .max_capacity(config.max_size as u64)
            .time_to_live(Duration::from_secs(config.default_ttl))
            .build();

        Self {
            memory_cache,
            config,
        }
    }

    /// Generate cache key from request
    fn generate_key(req: &Request) -> CacheKey {
        let method = req.method().to_string();
        let path = req.uri().path().to_string();

        // Sort query parameters for consistent cache keys
        let query = if let Some(q) = req.uri().query() {
            let mut params: Vec<_> = url::form_urlencoded::parse(q.as_bytes())
                .map(|(k, v)| format!("{k}={v}"))
                .collect();
            params.sort();
            params.join("&")
        } else {
            String::new()
        };

        // Extract API key from extensions (set by auth middleware)
        let api_key = req
            .extensions()
            .get::<Arc<crate::api::types::ApiKeyInfo>>()
            .map(|info| info.key_id.clone());

        CacheKey {
            method,
            path,
            query,
            api_key,
        }
    }

    /// Generate hash key from cache key
    fn hash_key(&self, key: &CacheKey) -> String {
        let mut hasher = Sha256::new();
        hasher.update(&key.method);
        hasher.update(&key.path);
        hasher.update(&key.query);
        if let Some(api_key) = &key.api_key {
            hasher.update(api_key);
        }
        format!("{}{:x}", self.config.key_prefix, hasher.finalize())
    }

    /// Check if request is cacheable
    fn is_cacheable(req: &Request) -> bool {
        // Only cache GET requests
        req.method() == Method::GET &&
        // Don't cache if no-cache header is present
        !req.headers()
            .get("cache-control")
            .and_then(|h| h.to_str().ok())
            .map(|s| s.contains("no-cache"))
            .unwrap_or(false)
    }

    /// Check if response is cacheable
    fn is_response_cacheable(status: StatusCode) -> bool {
        // Only cache successful responses
        status.is_success() || status == StatusCode::NOT_MODIFIED
    }

    /// Get cached response
    pub async fn get(&self, key: &CacheKey) -> Option<CachedResponse> {
        let hash_key = self.hash_key(key);

        if let Some(cached) = self.memory_cache.get(&hash_key).await {
            // Check if still valid
            if cached.expires_at > SystemTime::now() {
                return Some(cached);
            }
            // Remove expired entry
            self.memory_cache.remove(&hash_key).await;
        }

        None
    }

    /// Store response in cache
    pub async fn set(&self, key: &CacheKey, response: CachedResponse) {
        let hash_key = self.hash_key(key);
        self.memory_cache.insert(hash_key, response).await;
    }
}

/// Cache middleware
#[derive(Clone)]
pub struct CacheMiddleware {
    storage: Arc<CacheStorage>,
    config: Arc<crate::config::Config>,
}

impl CacheMiddleware {
    /// Create new cache middleware
    pub fn new(state: AppState) -> Self {
        let storage = Arc::new(CacheStorage::new(Arc::new(state.config.cache.clone())));

        Self {
            storage,
            config: state.config.clone(),
        }
    }
}

/// Cache handler for axum middleware
pub async fn cache_middleware(
    storage: Arc<CacheStorage>,
    config: Arc<crate::config::Config>,
    req: Request,
    next: Next,
) -> Result<Response, StatusCode> {
    // Check if request is cacheable
    if !CacheStorage::is_cacheable(&req) {
        return Ok(next.run(req).await);
    }

    // Generate cache key
    let cache_key = CacheStorage::generate_key(&req);

    // Check cache
    if let Some(cached) = storage.get(&cache_key).await {
        // Build response from cached data
        let mut response = Response::builder().status(cached.status);

        for (name, value) in &cached.headers {
            response = response.header(name, value);
        }

        // Add cache hit header
        response = response.header("X-Cache", "HIT");

        return response
            .body(Body::from(cached.body))
            .map_err(|_| StatusCode::INTERNAL_SERVER_ERROR);
    }

    // Execute request
    let response = next.run(req).await;

    // Check if response is cacheable
    if CacheStorage::is_response_cacheable(response.status()) {
        // Extract response data for caching
        let (parts, body) = response.into_parts();

        // Collect body bytes
        let body_bytes = match axum::body::to_bytes(body, usize::MAX).await {
            Ok(bytes) => bytes,
            Err(_) => return Err(StatusCode::INTERNAL_SERVER_ERROR),
        };

        // Prepare cached response
        let cached_response = CachedResponse {
            status: parts.status.as_u16(),
            headers: parts
                .headers
                .iter()
                .filter_map(|(name, value)| {
                    // Don't cache certain headers
                    if !matches!(name.as_str(), "date" | "x-request-id") {
                        Some((name.to_string(), value.to_str().ok()?.to_string()))
                    } else {
                        None
                    }
                })
                .collect(),
            body: body_bytes.to_vec(),
            expires_at: SystemTime::now() + Duration::from_secs(config.cache.default_ttl),
        };

        // Store in cache
        storage.set(&cache_key, cached_response).await;

        // Rebuild response with original headers
        let mut response_builder = Response::builder().status(parts.status);

        // Copy all original headers
        for (name, value) in &parts.headers {
            let name_str = name.as_str();
            if let Ok(value_str) = value.to_str() {
                response_builder = response_builder.header(name_str, value_str);
            }
        }

        // Add cache miss header
        response_builder = response_builder.header("X-Cache", "MISS");

        response_builder
            .body(Body::from(body_bytes))
            .map_err(|_| StatusCode::INTERNAL_SERVER_ERROR)
    } else {
        Ok(response)
    }
}
