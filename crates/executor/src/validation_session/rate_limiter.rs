//! Rate limiting system for validator access control
//!
//! Implements a token bucket algorithm for rate limiting validator requests,
//! providing per-validator rate limits and burst allowances to prevent abuse.

use crate::validation_session::types::{RateLimitConfig, ValidatorId};
use anyhow::Result;
use serde::{Deserialize, Serialize};
use std::collections::HashMap;
use std::time::{Duration, SystemTime};
use tokio::sync::RwLock;
use tracing::debug;

/// Token bucket for rate limiting
#[derive(Debug, Clone, Serialize, Deserialize)]
struct TokenBucket {
    tokens: f64,
    last_refill: SystemTime,
    capacity: f64,
    refill_rate: f64, // tokens per second
}

impl TokenBucket {
    /// Create a new token bucket
    fn new(capacity: f64, refill_rate: f64) -> Self {
        Self {
            tokens: capacity,
            last_refill: SystemTime::now(),
            capacity,
            refill_rate,
        }
    }

    /// Try to consume tokens from the bucket
    fn try_consume(&mut self, tokens: f64) -> bool {
        self.refill();

        if self.tokens >= tokens {
            self.tokens -= tokens;
            true
        } else {
            false
        }
    }

    /// Refill tokens based on elapsed time
    fn refill(&mut self) {
        let now = SystemTime::now();
        let elapsed = now
            .duration_since(self.last_refill)
            .unwrap_or_default()
            .as_secs_f64();

        self.tokens = (self.tokens + elapsed * self.refill_rate).min(self.capacity);
        self.last_refill = now;
    }

    /// Get current token count
    fn current_tokens(&mut self) -> f64 {
        self.refill();
        self.tokens
    }
}

/// Rate limiter for validator requests
pub struct ValidatorRateLimiter {
    config: RateLimitConfig,
    ssh_buckets: RwLock<HashMap<String, TokenBucket>>,
    api_buckets: RwLock<HashMap<String, TokenBucket>>,
}

impl ValidatorRateLimiter {
    /// Create a new rate limiter
    pub fn new(config: RateLimitConfig) -> Self {
        Self {
            config,
            ssh_buckets: RwLock::new(HashMap::new()),
            api_buckets: RwLock::new(HashMap::new()),
        }
    }

    /// Check if SSH request is allowed for validator
    pub async fn check_ssh_request(&self, validator_id: &ValidatorId) -> Result<bool> {
        self.check_request_internal(&validator_id.hotkey, RequestType::Ssh, 1.0)
            .await
    }

    /// Check if API request is allowed for validator
    pub async fn check_api_request(&self, validator_id: &ValidatorId) -> Result<bool> {
        self.check_request_internal(&validator_id.hotkey, RequestType::Api, 1.0)
            .await
    }

    /// Check if multiple requests are allowed (for burst handling)
    pub async fn check_burst_request(
        &self,
        validator_id: &ValidatorId,
        request_type: RequestType,
        count: u32,
    ) -> Result<bool> {
        let tokens = count as f64;
        self.check_request_internal(&validator_id.hotkey, request_type, tokens)
            .await
    }

    /// Internal method to check rate limits
    async fn check_request_internal(
        &self,
        hotkey: &str,
        request_type: RequestType,
        tokens: f64,
    ) -> Result<bool> {
        match request_type {
            RequestType::Ssh => {
                let mut buckets = self.ssh_buckets.write().await;
                let bucket = buckets.entry(hotkey.to_string()).or_insert_with(|| {
                    let rate = self.config.ssh_requests_per_minute as f64 / 60.0;
                    let capacity =
                        (self.config.ssh_requests_per_minute + self.config.burst_allowance) as f64;
                    TokenBucket::new(capacity, rate)
                });

                let allowed = bucket.try_consume(tokens);
                if !allowed {
                    debug!(
                        "SSH rate limit exceeded for validator: {} (tokens: {}, available: {})",
                        hotkey,
                        tokens,
                        bucket.current_tokens()
                    );
                }
                Ok(allowed)
            }
            RequestType::Api => {
                let mut buckets = self.api_buckets.write().await;
                let bucket = buckets.entry(hotkey.to_string()).or_insert_with(|| {
                    let rate = self.config.api_requests_per_minute as f64 / 60.0;
                    let capacity =
                        (self.config.api_requests_per_minute + self.config.burst_allowance) as f64;
                    TokenBucket::new(capacity, rate)
                });

                let allowed = bucket.try_consume(tokens);
                if !allowed {
                    debug!(
                        "API rate limit exceeded for validator: {} (tokens: {}, available: {})",
                        hotkey,
                        tokens,
                        bucket.current_tokens()
                    );
                }
                Ok(allowed)
            }
        }
    }

    /// Get rate limit status for a validator
    pub async fn get_status(&self, validator_id: &ValidatorId) -> RateLimitStatus {
        let ssh_tokens = {
            let mut buckets = self.ssh_buckets.write().await;
            buckets
                .get_mut(&validator_id.hotkey)
                .map(|bucket| bucket.current_tokens())
                .unwrap_or(self.config.ssh_requests_per_minute as f64)
        };

        let api_tokens = {
            let mut buckets = self.api_buckets.write().await;
            buckets
                .get_mut(&validator_id.hotkey)
                .map(|bucket| bucket.current_tokens())
                .unwrap_or(self.config.api_requests_per_minute as f64)
        };

        RateLimitStatus {
            ssh_tokens_remaining: ssh_tokens as u32,
            api_tokens_remaining: api_tokens as u32,
            ssh_limit: self.config.ssh_requests_per_minute,
            api_limit: self.config.api_requests_per_minute,
            window_seconds: self.config.rate_limit_window_seconds,
        }
    }

    /// Reset rate limits for a validator (on successful authentication)
    pub async fn reset_limits(&self, validator_id: &ValidatorId) {
        let mut ssh_buckets = self.ssh_buckets.write().await;
        let mut api_buckets = self.api_buckets.write().await;

        // Reset to full capacity
        if let Some(bucket) = ssh_buckets.get_mut(&validator_id.hotkey) {
            bucket.tokens = bucket.capacity;
        }

        if let Some(bucket) = api_buckets.get_mut(&validator_id.hotkey) {
            bucket.tokens = bucket.capacity;
        }

        debug!("Reset rate limits for validator: {}", validator_id.hotkey);
    }

    /// Clean up old rate limit entries
    pub async fn cleanup_old_entries(&self) -> Result<u32> {
        let cutoff = SystemTime::now() - Duration::from_secs(3600); // 1 hour
        let mut cleaned = 0;

        // Clean SSH buckets
        {
            let mut buckets = self.ssh_buckets.write().await;
            buckets.retain(|_, bucket| {
                if bucket.last_refill < cutoff {
                    cleaned += 1;
                    false
                } else {
                    true
                }
            });
        }

        // Clean API buckets
        {
            let mut buckets = self.api_buckets.write().await;
            buckets.retain(|_, bucket| {
                if bucket.last_refill < cutoff {
                    cleaned += 1;
                    false
                } else {
                    true
                }
            });
        }

        if cleaned > 0 {
            debug!("Cleaned up {} old rate limit entries", cleaned);
        }

        Ok(cleaned)
    }

    /// Get statistics about rate limiting
    pub async fn get_stats(&self) -> RateLimitStats {
        let ssh_buckets = self.ssh_buckets.read().await;
        let api_buckets = self.api_buckets.read().await;

        RateLimitStats {
            active_ssh_limiters: ssh_buckets.len(),
            active_api_limiters: api_buckets.len(),
            config: self.config.clone(),
        }
    }
}

/// Type of request for rate limiting
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum RequestType {
    Ssh,
    Api,
}

/// Rate limit status for a validator
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct RateLimitStatus {
    pub ssh_tokens_remaining: u32,
    pub api_tokens_remaining: u32,
    pub ssh_limit: u32,
    pub api_limit: u32,
    pub window_seconds: u64,
}

/// Rate limiter statistics
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct RateLimitStats {
    pub active_ssh_limiters: usize,
    pub active_api_limiters: usize,
    pub config: RateLimitConfig,
}

#[cfg(test)]
mod tests {
    use super::*;
    use tokio::time::{sleep, Duration as TokioDuration};

    #[tokio::test]
    async fn test_token_bucket_basic() {
        let mut bucket = TokenBucket::new(10.0, 1.0); // 10 capacity, 1 token/sec

        // Should be able to consume initial tokens
        assert!(bucket.try_consume(5.0));
        // The refill method might add fractional tokens due to elapsed time
        let remaining = bucket.tokens;
        assert!(
            (4.9..=5.1).contains(&remaining),
            "Expected ~5.0, got {remaining}"
        );

        // Should not be able to consume more than available
        assert!(!bucket.try_consume(6.0));
        // Tokens should be approximately the same (maybe slightly more due to time)
        let final_tokens = bucket.tokens;
        assert!(
            (4.9..=5.1).contains(&final_tokens),
            "Expected ~5.0, got {final_tokens}"
        );
    }

    #[tokio::test]
    async fn test_token_bucket_refill() {
        let mut bucket = TokenBucket::new(10.0, 10.0); // 10 capacity, 10 tokens/sec

        // Consume all tokens
        assert!(bucket.try_consume(10.0));
        assert_eq!(bucket.tokens, 0.0);

        // Wait for refill
        sleep(TokioDuration::from_millis(500)).await; // 0.5 seconds

        // Should have ~5 tokens refilled
        let tokens = bucket.current_tokens();
        assert!((4.0..=6.0).contains(&tokens));
    }

    #[tokio::test]
    async fn test_rate_limiter_ssh() {
        let config = RateLimitConfig {
            ssh_requests_per_minute: 60, // 1 per second
            burst_allowance: 10,
            ..Default::default()
        };

        let limiter = ValidatorRateLimiter::new(config);
        let validator_id = ValidatorId::new("test_validator".to_string());

        // Should allow initial requests
        assert!(limiter.check_ssh_request(&validator_id).await.unwrap());

        // Check burst handling
        assert!(limiter
            .check_burst_request(&validator_id, RequestType::Ssh, 5)
            .await
            .unwrap());
    }

    #[tokio::test]
    async fn test_rate_limiter_exceeded() {
        let config = RateLimitConfig {
            ssh_requests_per_minute: 2, // Very low limit
            burst_allowance: 1,
            ..Default::default()
        };

        let limiter = ValidatorRateLimiter::new(config);
        let validator_id = ValidatorId::new("test_validator".to_string());

        // First three requests should pass (2 base + 1 burst)
        assert!(limiter.check_ssh_request(&validator_id).await.unwrap());
        assert!(limiter.check_ssh_request(&validator_id).await.unwrap());
        assert!(limiter.check_ssh_request(&validator_id).await.unwrap());

        // Fourth request should fail
        assert!(!limiter.check_ssh_request(&validator_id).await.unwrap());
    }

    #[tokio::test]
    async fn test_rate_limit_reset() {
        let config = RateLimitConfig {
            ssh_requests_per_minute: 1,
            burst_allowance: 0,
            ..Default::default()
        };

        let limiter = ValidatorRateLimiter::new(config);
        let validator_id = ValidatorId::new("test_validator".to_string());

        // Consume the only allowed request
        assert!(limiter.check_ssh_request(&validator_id).await.unwrap());
        assert!(!limiter.check_ssh_request(&validator_id).await.unwrap());

        // Reset and try again
        limiter.reset_limits(&validator_id).await;
        assert!(limiter.check_ssh_request(&validator_id).await.unwrap());
    }
}
