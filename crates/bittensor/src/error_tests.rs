//! # Comprehensive Error Handling Tests
//!
//! Production-ready tests for Bittensor error handling and retry mechanisms.

#[cfg(test)]
mod tests {
    use super::super::error::{BittensorError, ErrorCategory, RetryConfig};
    use super::super::retry::{CircuitBreaker, ExponentialBackoff, RetryExecutor};
    use std::sync::atomic::{AtomicU32, Ordering};
    use std::sync::Arc;
    use std::time::Duration;
    use tokio::time::sleep;

    #[test]
    fn test_error_categorization() {
        // Test transient errors
        let rpc_timeout = BittensorError::RpcTimeoutError {
            message: "Connection timed out".to_string(),
            timeout: Duration::from_secs(30),
        };
        assert_eq!(rpc_timeout.category(), ErrorCategory::Transient);
        assert!(rpc_timeout.is_retryable());

        let tx_timeout = BittensorError::TxTimeoutError {
            message: "Transaction timed out".to_string(),
            timeout: Duration::from_secs(60),
        };
        assert_eq!(tx_timeout.category(), ErrorCategory::Transient);
        assert!(tx_timeout.is_retryable());

        // Test network errors
        let network_error = BittensorError::NetworkConnectivityError {
            message: "Network unavailable".to_string(),
        };
        assert_eq!(network_error.category(), ErrorCategory::Network);
        assert!(network_error.is_retryable());

        // Test rate limit errors
        let rate_limit = BittensorError::RateLimitExceeded {
            message: "Too many requests".to_string(),
        };
        assert_eq!(rate_limit.category(), ErrorCategory::RateLimit);
        assert!(rate_limit.is_retryable());

        // Test authentication errors
        let auth_error = BittensorError::SignatureError {
            message: "Invalid signature".to_string(),
        };
        assert_eq!(auth_error.category(), ErrorCategory::Auth);
        assert!(auth_error.is_retryable());

        // Test configuration errors (non-retryable)
        let config_error = BittensorError::InvalidHotkey {
            hotkey: "invalid_hotkey".to_string(),
        };
        assert_eq!(config_error.category(), ErrorCategory::Config);
        assert!(!config_error.is_retryable());

        // Test permanent errors (non-retryable)
        let permanent_error = BittensorError::NeuronNotFound {
            uid: 123,
            netuid: 1,
        };
        assert_eq!(permanent_error.category(), ErrorCategory::Permanent);
        assert!(!permanent_error.is_retryable());
    }

    #[test]
    fn test_retry_config_generation() {
        // Test transient error retry config
        let transient_error = BittensorError::RpcConnectionError {
            message: "Connection failed".to_string(),
        };
        let config = transient_error.retry_config().unwrap();
        assert_eq!(config.max_attempts, 5);
        assert_eq!(config.initial_delay, Duration::from_millis(200));
        assert_eq!(config.backoff_multiplier, 1.5);
        assert!(config.jitter);

        // Test rate limit error retry config
        let rate_limit_error = BittensorError::RateLimitExceeded {
            message: "Rate limited".to_string(),
        };
        let config = rate_limit_error.retry_config().unwrap();
        assert_eq!(config.max_attempts, 3);
        assert_eq!(config.initial_delay, Duration::from_secs(1));
        assert_eq!(config.backoff_multiplier, 2.0);
        assert!(!config.jitter); // No jitter for rate limits

        // Test non-retryable error
        let config_error = BittensorError::InvalidHotkey {
            hotkey: "invalid".to_string(),
        };
        assert!(config_error.retry_config().is_none());
    }

    #[test]
    fn test_legacy_error_categorization() {
        // Test legacy RPC error with timeout content
        let legacy_timeout = BittensorError::RpcError {
            message: "Request timeout occurred".to_string(),
        };
        assert_eq!(legacy_timeout.category(), ErrorCategory::Transient);

        // Test legacy RPC error without timeout content
        let legacy_permanent = BittensorError::RpcError {
            message: "Invalid method".to_string(),
        };
        assert_eq!(legacy_permanent.category(), ErrorCategory::Permanent);

        // Test legacy wallet error with file content
        let legacy_wallet_config = BittensorError::WalletError {
            message: "File not found".to_string(),
        };
        assert_eq!(legacy_wallet_config.category(), ErrorCategory::Config);

        // Test legacy wallet error without file content
        let legacy_wallet_auth = BittensorError::WalletError {
            message: "Authentication failed".to_string(),
        };
        assert_eq!(legacy_wallet_auth.category(), ErrorCategory::Auth);
    }

    #[test]
    fn test_exponential_backoff_calculation() {
        let config = RetryConfig {
            max_attempts: 4,
            initial_delay: Duration::from_millis(100),
            max_delay: Duration::from_secs(5),
            backoff_multiplier: 2.0,
            jitter: false,
        };

        let mut backoff = ExponentialBackoff::new(config);

        // Test delay progression
        assert_eq!(backoff.next_delay().unwrap(), Duration::from_millis(100));
        assert_eq!(backoff.attempts(), 1);

        assert_eq!(backoff.next_delay().unwrap(), Duration::from_millis(200));
        assert_eq!(backoff.attempts(), 2);

        assert_eq!(backoff.next_delay().unwrap(), Duration::from_millis(400));
        assert_eq!(backoff.attempts(), 3);

        assert_eq!(backoff.next_delay().unwrap(), Duration::from_millis(800));
        assert_eq!(backoff.attempts(), 4);

        // Should return None after max attempts
        assert!(backoff.next_delay().is_none());
    }

    #[test]
    fn test_exponential_backoff_max_delay_cap() {
        let config = RetryConfig {
            max_attempts: 5,
            initial_delay: Duration::from_millis(1000),
            max_delay: Duration::from_millis(2000), // Cap at 2 seconds
            backoff_multiplier: 3.0,
            jitter: false,
        };

        let mut backoff = ExponentialBackoff::new(config);

        assert_eq!(backoff.next_delay().unwrap(), Duration::from_millis(1000));
        assert_eq!(backoff.next_delay().unwrap(), Duration::from_millis(2000)); // Capped
        assert_eq!(backoff.next_delay().unwrap(), Duration::from_millis(2000)); // Still capped
    }

    #[test]
    fn test_exponential_backoff_reset() {
        let config = RetryConfig::default();
        let mut backoff = ExponentialBackoff::new(config);

        // Make some attempts
        backoff.next_delay();
        backoff.next_delay();
        assert_eq!(backoff.attempts(), 2);

        // Reset should bring back to zero
        backoff.reset();
        assert_eq!(backoff.attempts(), 0);
    }

    #[tokio::test]
    async fn test_retry_executor_success_on_first_attempt() {
        let operation = || async { Ok::<&str, BittensorError>("success") };

        let executor = RetryExecutor::new();
        let result: Result<&str, BittensorError> = executor.execute(operation).await;

        assert!(result.is_ok());
        assert_eq!(result.unwrap(), "success");
    }

    #[tokio::test]
    async fn test_retry_executor_success_after_retries() {
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
        assert_eq!(counter.load(Ordering::SeqCst), 3); // 1 initial + 2 retries
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
            BittensorError::InvalidHotkey { hotkey } => {
                assert_eq!(hotkey, "invalid");
            }
            other => panic!("Expected InvalidHotkey, got {other:?}"),
        }
    }

    #[tokio::test]
    async fn test_retry_executor_max_retries_exceeded() {
        let operation = || async {
            Err(BittensorError::RpcConnectionError {
                message: "Always fails".to_string(),
            })
        };

        let executor = RetryExecutor::new();
        let result: Result<&str, BittensorError> = executor.execute(operation).await;

        assert!(result.is_err());
        match result.unwrap_err() {
            BittensorError::MaxRetriesExceeded { attempts } => {
                assert_eq!(attempts, 5); // Default max attempts for transient errors
            }
            other => panic!("Expected MaxRetriesExceeded, got {other:?}"),
        }
    }

    #[tokio::test]
    async fn test_retry_executor_with_timeout() {
        let operation = || async {
            sleep(Duration::from_millis(200)).await;
            Err(BittensorError::RpcConnectionError {
                message: "Slow operation".to_string(),
            })
        };

        let executor = RetryExecutor::new().with_timeout(Duration::from_millis(500));
        let result: Result<&str, BittensorError> = executor.execute(operation).await;

        assert!(result.is_err());
        match result.unwrap_err() {
            BittensorError::BackoffTimeoutReached { duration } => {
                assert!(duration >= Duration::from_millis(400)); // Allow some variance
            }
            other => panic!("Expected BackoffTimeoutReached, got {other:?}"),
        }
    }

    #[tokio::test]
    async fn test_retry_executor_custom_config() {
        let custom_config = RetryConfig {
            max_attempts: 2,
            initial_delay: Duration::from_millis(10),
            max_delay: Duration::from_secs(1),
            backoff_multiplier: 1.0,
            jitter: false,
        };

        let counter = Arc::new(AtomicU32::new(0));
        let counter_clone = counter.clone();

        let operation = move || {
            let counter = counter_clone.clone();
            async move {
                counter.fetch_add(1, Ordering::SeqCst);
                Err(BittensorError::RpcConnectionError {
                    message: "Always fails".to_string(),
                })
            }
        };

        let executor = RetryExecutor::new();
        let result: Result<&str, BittensorError> =
            executor.execute_with_config(operation, custom_config).await;

        assert!(result.is_err());
        // For custom config, it attempts the operation first (1 time) + max_attempts retries (2 times) = 3 total
        assert_eq!(counter.load(Ordering::SeqCst), 3);
    }

    #[tokio::test]
    async fn test_circuit_breaker_normal_operation() {
        let mut circuit_breaker = CircuitBreaker::new(3, Duration::from_millis(100));

        let operation = || async { Ok::<&str, BittensorError>("success") };

        let result = circuit_breaker.execute(operation).await;
        assert!(result.is_ok());
        assert_eq!(result.unwrap(), "success");
    }

    #[tokio::test]
    async fn test_circuit_breaker_opens_after_failures() {
        let mut circuit_breaker = CircuitBreaker::new(2, Duration::from_millis(100));

        // First failure
        let result: Result<&str, BittensorError> = circuit_breaker
            .execute(|| async {
                Err(BittensorError::RpcConnectionError {
                    message: "Connection failed".to_string(),
                })
            })
            .await;
        assert!(result.is_err());

        // Second failure - should open circuit
        let result: Result<&str, BittensorError> = circuit_breaker
            .execute(|| async {
                Err(BittensorError::RpcConnectionError {
                    message: "Connection failed".to_string(),
                })
            })
            .await;
        assert!(result.is_err());

        // Third call should fail fast
        let counter = Arc::new(AtomicU32::new(0));
        let counter_clone = counter.clone();
        let result = circuit_breaker
            .execute(move || {
                let counter = counter_clone.clone();
                async move {
                    counter.fetch_add(1, Ordering::SeqCst);
                    Ok("should not reach here")
                }
            })
            .await;

        assert!(result.is_err());
        assert_eq!(counter.load(Ordering::SeqCst), 0); // Operation not called
        match result.unwrap_err() {
            BittensorError::ServiceUnavailable { .. } => {}
            other => panic!("Expected ServiceUnavailable, got {other:?}"),
        }
    }

    #[tokio::test]
    async fn test_circuit_breaker_recovery() {
        let mut circuit_breaker = CircuitBreaker::new(1, Duration::from_millis(50));

        // Cause failure to open circuit
        let result: Result<&str, BittensorError> = circuit_breaker
            .execute(|| async {
                Err(BittensorError::RpcConnectionError {
                    message: "Connection failed".to_string(),
                })
            })
            .await;
        assert!(result.is_err());

        // Should fail fast immediately
        let result = circuit_breaker
            .execute(|| async { Ok::<&str, BittensorError>("success") })
            .await;
        assert!(result.is_err());

        // Wait for recovery timeout
        sleep(Duration::from_millis(60)).await;

        // Should now attempt operation and succeed
        let result = circuit_breaker
            .execute(|| async { Ok::<&str, BittensorError>("success") })
            .await;
        assert!(result.is_ok());
        assert_eq!(result.unwrap(), "success");
    }

    #[test]
    fn test_error_display_formatting() {
        let timeout_error = BittensorError::TxTimeoutError {
            message: "Transaction timed out".to_string(),
            timeout: Duration::from_secs(60),
        };
        let display = format!("{timeout_error}");
        assert!(display.contains("Transaction timeout after 60s"));
        assert!(display.contains("Transaction timed out"));

        let insufficient_fees = BittensorError::InsufficientTxFees {
            required: 1000,
            available: 500,
        };
        let display = format!("{insufficient_fees}");
        assert!(display.contains("required 1000"));
        assert!(display.contains("available 500"));

        let weight_error = BittensorError::WeightSettingFailed {
            netuid: 1,
            reason: "Invalid weights".to_string(),
        };
        let display = format!("{weight_error}");
        assert!(display.contains("subnet 1"));
        assert!(display.contains("Invalid weights"));
    }

    #[test]
    fn test_error_debug_formatting() {
        let error = BittensorError::NeuronNotFound {
            uid: 123,
            netuid: 1,
        };
        let debug = format!("{error:?}");
        assert!(debug.contains("NeuronNotFound"));
        assert!(debug.contains("uid: 123"));
        assert!(debug.contains("netuid: 1"));
    }

    #[test]
    fn test_retry_config_presets() {
        let transient = RetryConfig::transient();
        assert_eq!(transient.max_attempts, 5);
        assert_eq!(transient.backoff_multiplier, 1.5);
        assert!(transient.jitter);

        let rate_limit = RetryConfig::rate_limit();
        assert_eq!(rate_limit.max_attempts, 3);
        assert_eq!(rate_limit.backoff_multiplier, 2.0);
        assert!(!rate_limit.jitter);

        let network = RetryConfig::network();
        assert_eq!(network.max_attempts, 4);
        assert!(network.jitter);

        let auth = RetryConfig::auth();
        assert_eq!(auth.max_attempts, 2);
        assert_eq!(auth.backoff_multiplier, 1.0);
        assert!(!auth.jitter);
    }

    #[test]
    fn test_error_creation_helpers() {
        let max_retries = BittensorError::max_retries_exceeded(5);
        match max_retries {
            BittensorError::MaxRetriesExceeded { attempts } => {
                assert_eq!(attempts, 5);
            }
            _ => panic!("Expected MaxRetriesExceeded"),
        }

        let backoff_timeout = BittensorError::backoff_timeout(Duration::from_secs(30));
        match backoff_timeout {
            BittensorError::BackoffTimeoutReached { duration } => {
                assert_eq!(duration, Duration::from_secs(30));
            }
            _ => panic!("Expected BackoffTimeoutReached"),
        }
    }
}
