//! Retry logic for federation operations

use crate::error::{FederationError, Result};
use std::time::Duration;
use tracing::{debug, warn};

/// Retry configuration
#[derive(Debug, Clone)]
pub struct RetryConfig {
    /// Maximum number of retry attempts
    pub max_attempts: u32,
    
    /// Initial delay between retries
    pub initial_delay: Duration,
    
    /// Maximum delay between retries
    pub max_delay: Duration,
    
    /// Exponential backoff multiplier
    pub backoff_multiplier: f64,
    
    /// Whether to retry on timeout errors
    pub retry_on_timeout: bool,
}

impl Default for RetryConfig {
    fn default() -> Self {
        Self {
            max_attempts: 3,
            initial_delay: Duration::from_millis(100),
            max_delay: Duration::from_secs(5),
            backoff_multiplier: 2.0,
            retry_on_timeout: true,
        }
    }
}

/// Retry executor
pub struct RetryExecutor {
    config: RetryConfig,
}

impl RetryExecutor {
    /// Create a new retry executor
    pub fn new(config: RetryConfig) -> Self {
        Self { config }
    }
    
    /// Execute a function with retry logic
    pub async fn execute<F, Fut, T>(&self, mut f: F) -> Result<T>
    where
        F: FnMut() -> Fut,
        Fut: std::future::Future<Output = Result<T>>,
    {
        let mut attempt = 0;
        let mut delay = self.config.initial_delay;
        
        loop {
            attempt += 1;
            
            match f().await {
                Ok(result) => {
                    if attempt > 1 {
                        debug!(attempts = attempt, "Operation succeeded after retries");
                    }
                    return Ok(result);
                }
                Err(e) => {
                    if attempt >= self.config.max_attempts {
                        warn!(
                            attempts = attempt,
                            error = %e,
                            "Max retry attempts reached"
                        );
                        return Err(e);
                    }
                    
                    // Check if error is retryable
                    if !self.is_retryable(&e) {
                        return Err(e);
                    }
                    
                    debug!(
                        attempt = attempt,
                        max_attempts = self.config.max_attempts,
                        delay_ms = delay.as_millis(),
                        "Retrying operation"
                    );
                    
                    tokio::time::sleep(delay).await;
                    
                    // Calculate next delay with exponential backoff
                    delay = Duration::from_millis(
                        (delay.as_millis() as f64 * self.config.backoff_multiplier) as u64
                    ).min(self.config.max_delay);
                }
            }
        }
    }
    
    /// Check if an error is retryable
    fn is_retryable(&self, error: &FederationError) -> bool {
        match error {
            FederationError::Timeout(_) => self.config.retry_on_timeout,
            FederationError::Request(_) => true,
            FederationError::Kube(_) => true,
            FederationError::Health(_) => true,
            FederationError::Discovery(_) => true,
            FederationError::LoadBalancing(_) => true,
            FederationError::ResourceManagement(_) => true,
            _ => false,
        }
    }
}

/// Retry helper function
pub async fn retry<F, Fut, T>(config: RetryConfig, f: F) -> Result<T>
where
    F: FnMut() -> Fut,
    Fut: std::future::Future<Output = Result<T>>,
{
    let executor = RetryExecutor::new(config);
    executor.execute(f).await
}

