//! # Retry Logic with Exponential Backoff
//!
//! Production-ready retry mechanisms for Bittensor operations with configurable
//! exponential backoff, jitter, and error-specific retry strategies.

use crate::error::{BittensorError, RetryConfig};
use std::future::Future;
use std::time::Duration;
use tokio::time::sleep;
use tracing::{debug, info, warn};

/// Exponential backoff calculator with jitter support
#[derive(Debug, Clone)]
pub struct ExponentialBackoff {
    config: RetryConfig,
    current_attempt: u32,
}

impl ExponentialBackoff {
    /// Creates a new exponential backoff instance
    pub fn new(config: RetryConfig) -> Self {
        Self {
            config,
            current_attempt: 0,
        }
    }

    /// Calculates the next delay duration
    pub fn next_delay(&mut self) -> Option<Duration> {
        if self.current_attempt >= self.config.max_attempts {
            return None;
        }

        let base_delay = self.config.initial_delay.as_millis() as f64;
        let multiplier = self
            .config
            .backoff_multiplier
            .powi(self.current_attempt as i32);
        let calculated_delay = Duration::from_millis((base_delay * multiplier) as u64);

        // Cap at max_delay
        let mut delay = if calculated_delay > self.config.max_delay {
            self.config.max_delay
        } else {
            calculated_delay
        };

        // Add jitter if enabled
        if self.config.jitter {
            delay = Self::add_jitter(delay);
        }

        self.current_attempt += 1;
        Some(delay)
    }

    /// Adds random jitter to prevent thundering herd
    fn add_jitter(delay: Duration) -> Duration {
        use rand::Rng;
        let jitter_ms = rand::thread_rng().gen_range(0..=delay.as_millis() as u64 / 4);
        delay + Duration::from_millis(jitter_ms)
    }

    /// Resets the backoff state
    pub fn reset(&mut self) {
        self.current_attempt = 0;
    }

    /// Gets the current attempt number
    pub fn attempts(&self) -> u32 {
        self.current_attempt
    }
}

/// Retry executor with comprehensive error handling
pub struct RetryExecutor {
    total_timeout: Option<Duration>,
}

impl RetryExecutor {
    /// Creates a new retry executor
    pub fn new() -> Self {
        Self {
            total_timeout: None,
        }
    }

    /// Sets a total timeout for all retry attempts
    pub fn with_timeout(mut self, timeout: Duration) -> Self {
        self.total_timeout = Some(timeout);
        self
    }

    /// Executes an operation with retry logic based on error types
    pub async fn execute<F, Fut, T>(&self, operation: F) -> Result<T, BittensorError>
    where
        F: Fn() -> Fut,
        Fut: Future<Output = Result<T, BittensorError>>,
    {
        let start_time = tokio::time::Instant::now();

        // First attempt without delay
        match operation().await {
            Ok(result) => Ok(result),
            Err(error) => {
                if !error.is_retryable() {
                    debug!("Error is not retryable: {:?}", error);
                    return Err(error);
                }

                let config = match error.retry_config() {
                    Some(config) => config,
                    None => {
                        debug!("No retry config for error: {:?}", error);
                        return Err(error);
                    }
                };

                info!(
                    "Starting retry for error category: {:?}, max_attempts: {}",
                    error.category(),
                    config.max_attempts
                );

                let mut backoff = ExponentialBackoff::new(config);
                let mut _last_error = error;

                // Retry loop
                while let Some(delay) = backoff.next_delay() {
                    // Check total timeout
                    if let Some(total_timeout) = self.total_timeout {
                        if start_time.elapsed() + delay >= total_timeout {
                            warn!(
                                "Total timeout reached after {} attempts",
                                backoff.attempts()
                            );
                            return Err(BittensorError::backoff_timeout(start_time.elapsed()));
                        }
                    }

                    debug!(
                        "Retry attempt {} after delay {:?}",
                        backoff.attempts(),
                        delay
                    );
                    sleep(delay).await;

                    match operation().await {
                        Ok(result) => {
                            info!("Operation succeeded after {} attempts", backoff.attempts());
                            return Ok(result);
                        }
                        Err(error) => {
                            _last_error = error;

                            // If error category changed, stop retrying
                            if !_last_error.is_retryable() {
                                debug!("Error became non-retryable: {:?}", _last_error);
                                return Err(_last_error);
                            }

                            warn!(
                                "Retry attempt {} failed: {}",
                                backoff.attempts(),
                                _last_error
                            );
                        }
                    }
                }

                warn!(
                    "All {} retry attempts exhausted, last error: {}",
                    backoff.config.max_attempts, _last_error
                );
                Err(BittensorError::max_retries_exceeded(
                    backoff.config.max_attempts,
                ))
            }
        }
    }

    /// Executes an operation with custom retry configuration
    pub async fn execute_with_config<F, Fut, T>(
        &self,
        operation: F,
        config: RetryConfig,
    ) -> Result<T, BittensorError>
    where
        F: Fn() -> Fut,
        Fut: Future<Output = Result<T, BittensorError>>,
    {
        let start_time = tokio::time::Instant::now();
        let mut backoff = ExponentialBackoff::new(config);

        // First attempt
        match operation().await {
            Ok(result) => Ok(result),
            Err(mut _last_error) => {
                info!(
                    "Starting custom retry, max_attempts: {}",
                    backoff.config.max_attempts
                );

                // Retry loop
                while let Some(delay) = backoff.next_delay() {
                    // Check total timeout
                    if let Some(total_timeout) = self.total_timeout {
                        if start_time.elapsed() + delay >= total_timeout {
                            warn!(
                                "Total timeout reached after {} attempts",
                                backoff.attempts()
                            );
                            return Err(BittensorError::backoff_timeout(start_time.elapsed()));
                        }
                    }

                    debug!(
                        "Custom retry attempt {} after delay {:?}",
                        backoff.attempts(),
                        delay
                    );
                    sleep(delay).await;

                    match operation().await {
                        Ok(result) => {
                            info!(
                                "Custom retry succeeded after {} attempts",
                                backoff.attempts()
                            );
                            return Ok(result);
                        }
                        Err(error) => {
                            _last_error = error;
                            warn!(
                                "Custom retry attempt {} failed: {}",
                                backoff.attempts(),
                                _last_error
                            );
                        }
                    }
                }

                Err(BittensorError::max_retries_exceeded(
                    backoff.config.max_attempts,
                ))
            }
        }
    }
}

impl Default for RetryExecutor {
    fn default() -> Self {
        Self::new()
    }
}

/// Convenience function for retrying operations with default settings
pub async fn retry_operation<F, Fut, T>(operation: F) -> Result<T, BittensorError>
where
    F: Fn() -> Fut,
    Fut: Future<Output = Result<T, BittensorError>>,
{
    RetryExecutor::new().execute(operation).await
}

/// Convenience function for retrying operations with timeout
pub async fn retry_operation_with_timeout<F, Fut, T>(
    operation: F,
    timeout: Duration,
) -> Result<T, BittensorError>
where
    F: Fn() -> Fut,
    Fut: Future<Output = Result<T, BittensorError>>,
{
    RetryExecutor::new()
        .with_timeout(timeout)
        .execute(operation)
        .await
}

/// Circuit breaker for preventing cascade failures
#[derive(Debug, Clone)]
pub struct CircuitBreaker {
    failure_threshold: u32,
    recovery_timeout: Duration,
    current_failures: u32,
    state: CircuitState,
    last_failure_time: Option<tokio::time::Instant>,
}

#[derive(Debug, Clone, Copy, PartialEq, Eq)]
enum CircuitState {
    Closed,   // Normal operation
    Open,     // Failing fast
    HalfOpen, // Testing recovery
}

impl CircuitBreaker {
    /// Creates a new circuit breaker
    pub fn new(failure_threshold: u32, recovery_timeout: Duration) -> Self {
        Self {
            failure_threshold,
            recovery_timeout,
            current_failures: 0,
            state: CircuitState::Closed,
            last_failure_time: None,
        }
    }

    /// Executes an operation through the circuit breaker
    pub async fn execute<F, Fut, T>(&mut self, operation: F) -> Result<T, BittensorError>
    where
        F: Fn() -> Fut,
        Fut: Future<Output = Result<T, BittensorError>>,
    {
        match self.state {
            CircuitState::Open => {
                if let Some(last_failure) = self.last_failure_time {
                    if last_failure.elapsed() >= self.recovery_timeout {
                        debug!("Circuit breaker transitioning to half-open");
                        self.state = CircuitState::HalfOpen;
                    } else {
                        return Err(BittensorError::ServiceUnavailable {
                            message: "Circuit breaker is open".to_string(),
                        });
                    }
                } else {
                    return Err(BittensorError::ServiceUnavailable {
                        message: "Circuit breaker is open".to_string(),
                    });
                }
            }
            CircuitState::Closed | CircuitState::HalfOpen => {}
        }

        match operation().await {
            Ok(result) => {
                // Success - reset circuit breaker
                if self.state == CircuitState::HalfOpen {
                    debug!("Circuit breaker recovering - closing circuit");
                    self.state = CircuitState::Closed;
                }
                self.current_failures = 0;
                self.last_failure_time = None;
                Ok(result)
            }
            Err(error) => {
                // Failure - update circuit breaker state
                self.current_failures += 1;
                self.last_failure_time = Some(tokio::time::Instant::now());

                if self.current_failures >= self.failure_threshold {
                    warn!(
                        "Circuit breaker opening after {} failures",
                        self.current_failures
                    );
                    self.state = CircuitState::Open;
                }

                Err(error)
            }
        }
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use std::sync::atomic::{AtomicU32, Ordering};
    use std::sync::Arc;

    #[test]
    fn test_exponential_backoff() {
        let config = RetryConfig {
            max_attempts: 3,
            initial_delay: Duration::from_millis(100),
            max_delay: Duration::from_secs(5),
            backoff_multiplier: 2.0,
            jitter: false,
        };

        let mut backoff = ExponentialBackoff::new(config);

        // First delay
        let delay1 = backoff.next_delay().unwrap();
        assert_eq!(delay1, Duration::from_millis(100));
        assert_eq!(backoff.attempts(), 1);

        // Second delay
        let delay2 = backoff.next_delay().unwrap();
        assert_eq!(delay2, Duration::from_millis(200));
        assert_eq!(backoff.attempts(), 2);

        // Third delay
        let delay3 = backoff.next_delay().unwrap();
        assert_eq!(delay3, Duration::from_millis(400));
        assert_eq!(backoff.attempts(), 3);

        // Should return None after max attempts
        assert!(backoff.next_delay().is_none());
    }

    #[tokio::test]
    async fn test_retry_executor_success_after_failure() {
        let counter = Arc::new(AtomicU32::new(0));
        let counter_clone = counter.clone();

        let operation = move || {
            let counter = counter_clone.clone();
            async move {
                let count = counter.fetch_add(1, Ordering::SeqCst);
                if count < 2 {
                    Err(BittensorError::RpcConnectionError {
                        message: "Connection failed".to_string(),
                    })
                } else {
                    Ok("success")
                }
            }
        };

        let executor = RetryExecutor::new();
        let result: Result<&str, BittensorError> = executor.execute(operation).await;

        assert!(result.is_ok());
        assert_eq!(result.unwrap(), "success");
        assert_eq!(counter.load(Ordering::SeqCst), 3);
    }

    #[tokio::test]
    async fn test_retry_executor_non_retryable_error() {
        let operation = || async {
            Err(BittensorError::InvalidHotkey {
                hotkey: "invalid".to_string(),
            })
        };

        let executor = RetryExecutor::new();
        let result: Result<&str, BittensorError> = executor.execute(operation).await;

        assert!(result.is_err());
        match result.unwrap_err() {
            BittensorError::InvalidHotkey { .. } => {}
            other => panic!("Expected InvalidHotkey, got {other:?}"),
        }
    }

    #[tokio::test]
    async fn test_circuit_breaker() {
        let mut circuit_breaker = CircuitBreaker::new(2, Duration::from_millis(100));
        let counter = Arc::new(AtomicU32::new(0));

        // First failure
        let counter_clone = counter.clone();
        let result: Result<(), BittensorError> = circuit_breaker
            .execute(|| {
                let counter = counter_clone.clone();
                async move {
                    counter.fetch_add(1, Ordering::SeqCst);
                    Err(BittensorError::RpcConnectionError {
                        message: "Connection failed".to_string(),
                    })
                }
            })
            .await;
        assert!(result.is_err());

        // Second failure - should open circuit
        let counter_clone = counter.clone();
        let result: Result<(), BittensorError> = circuit_breaker
            .execute(|| {
                let counter = counter_clone.clone();
                async move {
                    counter.fetch_add(1, Ordering::SeqCst);
                    Err(BittensorError::RpcConnectionError {
                        message: "Connection failed".to_string(),
                    })
                }
            })
            .await;
        assert!(result.is_err());

        // Third call should fail fast without calling operation
        let counter_before = counter.load(Ordering::SeqCst);
        let result: Result<&str, BittensorError> = circuit_breaker
            .execute(|| {
                let counter = counter.clone();
                async move {
                    counter.fetch_add(1, Ordering::SeqCst);
                    Ok("should not reach here")
                }
            })
            .await;
        assert!(result.is_err());
        assert_eq!(counter.load(Ordering::SeqCst), counter_before); // No increment

        match result.unwrap_err() {
            BittensorError::ServiceUnavailable { .. } => {}
            other => panic!("Expected ServiceUnavailable, got {other:?}"),
        }
    }
}
