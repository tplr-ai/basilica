//! Integration tests for connection reliability improvements

use basilica_common::config::BittensorConfig;
use bittensor::{
    BittensorError, ConnectionManager, ConnectionPool, ConnectionPoolBuilder, ConnectionState,
    HealthChecker, RetryConfig,
};
use std::sync::Arc;
use std::time::Duration;
use tokio::time::{sleep, timeout};

/// Helper function to create test config
fn test_config() -> BittensorConfig {
    BittensorConfig {
        wallet_name: "test".to_string(),
        hotkey_name: "test".to_string(),
        network: "test".to_string(),
        netuid: 1,
        chain_endpoint: Some("wss://test.invalid:443".to_string()),
        fallback_endpoints: vec![
            "wss://test2.invalid:443".to_string(),
            "wss://test3.invalid:443".to_string(),
        ],
        weight_interval_secs: 300,
        connection_pool_size: Some(3),
        health_check_interval: Some(Duration::from_millis(100)),
        circuit_breaker_threshold: Some(3),
        circuit_breaker_recovery: Some(Duration::from_secs(1)),
    }
}

#[tokio::test]
async fn test_connection_pool_initialization_failure_handling() {
    let config = test_config();
    let pool = ConnectionPoolBuilder::new(config.get_chain_endpoints())
        .max_connections(2)
        .retry_config(RetryConfig {
            max_attempts: 2,
            initial_delay: Duration::from_millis(10),
            max_delay: Duration::from_millis(50),
            backoff_multiplier: 2.0,
            jitter: false,
        })
        .build();

    // Initialization should fail with invalid endpoints
    let result = pool.initialize().await;
    assert!(result.is_err());

    match result {
        Err(BittensorError::NetworkError { .. }) => {
            // Expected error type
        }
        _ => panic!("Expected NetworkError"),
    }
}

#[tokio::test]
async fn test_connection_pool_fallback_behavior() {
    let endpoints = vec![
        "wss://invalid1.test:443".to_string(),
        "wss://invalid2.test:443".to_string(),
        "wss://invalid3.test:443".to_string(),
    ];

    let pool = ConnectionPool::new(endpoints, 3);

    // Should try to reconnect with backoff
    let result = timeout(Duration::from_secs(2), pool.get_healthy_client()).await;

    assert!(result.is_err() || result.unwrap().is_err());
}

#[tokio::test]
async fn test_health_checker_monitoring() {
    let pool = Arc::new(ConnectionPool::new(
        vec!["wss://test.invalid:443".to_string()],
        1,
    ));

    let checker = Arc::new(
        HealthChecker::new()
            .with_interval(Duration::from_millis(50))
            .with_timeout(Duration::from_millis(10))
            .with_failure_threshold(2),
    );

    // Start monitoring
    let handle = checker.clone().start_monitoring(pool.clone());

    // Let it run for a bit
    sleep(Duration::from_millis(200)).await;

    // Check metrics
    let metrics = checker.metrics();
    assert!(metrics.total_checks > 0);

    // Stop monitoring
    handle.abort();
}

#[tokio::test]
async fn test_connection_state_transitions() {
    let manager = ConnectionManager::new(test_config());

    // Initial state should be uninitialized
    let state = manager.get_state().await;
    assert!(matches!(state, ConnectionState::Uninitialized));

    // Attempting to get client should trigger connection attempt
    let _ = manager.get_client().await;

    // State should have changed
    let state = manager.get_state().await;
    assert!(!matches!(state, ConnectionState::Uninitialized));
}

#[tokio::test]
async fn test_connection_manager_metrics() {
    let manager = ConnectionManager::new(test_config());

    // Initial metrics
    let metrics = manager.metrics();
    assert_eq!(metrics.success_count, 0);
    assert_eq!(metrics.failure_count, 0);

    // Try to connect (will fail with invalid endpoint)
    let _ = manager.connect().await;

    // Metrics should be updated
    let metrics = manager.metrics();
    assert!(metrics.failure_count > 0 || metrics.success_count > 0);
}

#[tokio::test]
async fn test_retry_configuration() {
    let config = RetryConfig {
        max_attempts: 3,
        initial_delay: Duration::from_millis(10),
        max_delay: Duration::from_millis(100),
        backoff_multiplier: 2.0,
        jitter: false,
    };

    let pool = ConnectionPoolBuilder::new(vec!["wss://test.invalid:443".to_string()])
        .retry_config(config.clone())
        .max_connections(1)
        .build();

    let start = tokio::time::Instant::now();
    let _ = pool.reconnect_with_backoff().await;
    let elapsed = start.elapsed();

    // Should have attempted retries with backoff
    // 10ms + 20ms + 40ms = 70ms minimum
    assert!(elapsed >= Duration::from_millis(70));
}

#[tokio::test]
async fn test_concurrent_health_checks() {
    let pool = Arc::new(ConnectionPool::new(
        vec![
            "wss://test1.invalid:443".to_string(),
            "wss://test2.invalid:443".to_string(),
        ],
        2,
    ));

    let checker = HealthChecker::new().with_timeout(Duration::from_millis(10));

    // Run multiple health checks concurrently
    let handles: Vec<_> = (0..5)
        .map(|_| {
            let pool_clone = pool.clone();
            let _checker_clone = checker.clone();
            tokio::spawn(async move { pool_clone.healthy_connection_count().await })
        })
        .collect();

    let results = futures::future::join_all(handles).await;

    // All should complete without panic
    for result in results {
        assert!(result.is_ok());
    }
}

#[tokio::test]
async fn test_connection_state_status_messages() {
    let states = vec![
        ConnectionState::Uninitialized,
        ConnectionState::Connected {
            since: tokio::time::Instant::now(),
            endpoint: "wss://test:443".to_string(),
        },
        ConnectionState::Reconnecting {
            attempts: 3,
            since: tokio::time::Instant::now(),
            last_error: Some("timeout".to_string()),
        },
        ConnectionState::Failed {
            error: "connection refused".to_string(),
            at: tokio::time::Instant::now(),
            consecutive_failures: 5,
        },
    ];

    for state in states {
        let msg = state.status_message();
        assert!(!msg.is_empty());

        match &state {
            ConnectionState::Connected { .. } => assert!(msg.contains("Connected")),
            ConnectionState::Reconnecting { .. } => assert!(msg.contains("Reconnecting")),
            ConnectionState::Failed { .. } => assert!(msg.contains("Failed")),
            ConnectionState::Uninitialized => assert!(msg.contains("Not initialized")),
        }
    }
}

#[tokio::test]
async fn test_connection_pool_refresh() {
    let pool = ConnectionPool::new(vec!["wss://test.invalid:443".to_string()], 1);

    // Refresh should attempt re-initialization
    let result = pool.refresh_connections().await;
    assert!(result.is_err()); // Will fail with invalid endpoint

    // Total connections should still be 0
    assert_eq!(pool.total_connections().await, 0);
}

#[tokio::test]
async fn test_health_checker_failure_threshold() {
    let pool = Arc::new(ConnectionPool::new(
        vec!["wss://test.invalid:443".to_string()],
        1,
    ));

    let checker = Arc::new(
        HealthChecker::new()
            .with_interval(Duration::from_millis(50))
            .with_failure_threshold(2),
    );

    let handle = checker.clone().start_monitoring(pool);

    // Wait for multiple check cycles
    sleep(Duration::from_millis(200)).await;

    handle.abort();

    // Should have accumulated failures
    let metrics = checker.metrics();
    assert!(metrics.failed_checks > 0 || metrics.total_checks > 0);
}

#[tokio::test]
async fn test_connection_manager_force_reconnect() {
    let manager = ConnectionManager::new(test_config());

    // Force reconnect
    let result = manager.force_reconnect().await;
    assert!(result.is_err()); // Will fail with invalid endpoints

    // State should reflect attempt
    let state = manager.get_state().await;
    assert!(!matches!(state, ConnectionState::Uninitialized));
}

#[tokio::test]
async fn test_configuration_endpoint_deduplication() {
    let mut config = test_config();
    config.chain_endpoint = Some("wss://duplicate.test:443".to_string());
    config.fallback_endpoints = vec![
        "wss://duplicate.test:443".to_string(), // Duplicate
        "wss://unique.test:443".to_string(),
    ];

    let endpoints = config.get_chain_endpoints();

    // Should deduplicate
    assert_eq!(endpoints.len(), 2);
    assert!(endpoints.contains(&"wss://duplicate.test:443".to_string()));
    assert!(endpoints.contains(&"wss://unique.test:443".to_string()));
}

#[tokio::test]
async fn test_health_checker_reset_metrics() {
    let checker = HealthChecker::new();

    // Simulate some metrics
    for _ in 0..5 {
        checker
            .metrics
            .total_checks
            .fetch_add(1, std::sync::atomic::Ordering::Relaxed);
        checker
            .metrics
            .successful_checks
            .fetch_add(1, std::sync::atomic::Ordering::Relaxed);
    }

    let metrics = checker.metrics();
    assert_eq!(metrics.total_checks, 5);
    assert_eq!(metrics.successful_checks, 5);

    // Reset
    checker.reset_metrics();

    let metrics = checker.metrics();
    assert_eq!(metrics.total_checks, 0);
    assert_eq!(metrics.successful_checks, 0);
}

#[tokio::test]
async fn test_connection_pool_builder_configuration() {
    let custom_checker = HealthChecker::new()
        .with_interval(Duration::from_secs(30))
        .with_timeout(Duration::from_secs(10));

    let custom_retry = RetryConfig {
        max_attempts: 10,
        initial_delay: Duration::from_secs(1),
        max_delay: Duration::from_secs(60),
        backoff_multiplier: 3.0,
        jitter: true,
    };

    let pool = ConnectionPoolBuilder::new(vec!["wss://test:443".to_string()])
        .max_connections(5)
        .health_checker(custom_checker)
        .retry_config(custom_retry.clone())
        .build();

    assert_eq!(pool.max_connections, 5);
    assert_eq!(pool.retry_config.max_attempts, 10);
}

#[tokio::test]
async fn test_edge_case_empty_endpoints() {
    let pool = ConnectionPool::new(vec![], 3);
    let result = pool.initialize().await;

    assert!(result.is_err());
    if let Err(BittensorError::ConfigError { field, .. }) = result {
        assert_eq!(field, "endpoints");
    } else {
        panic!("Expected ConfigError for empty endpoints");
    }
}

#[tokio::test]
async fn test_edge_case_zero_max_connections() {
    let pool = ConnectionPool::new(vec!["wss://test:443".to_string()], 0);

    // Should handle gracefully
    let count = pool.healthy_connection_count().await;
    assert_eq!(count, 0);
}

#[tokio::test]
async fn test_concurrent_connection_attempts() {
    let manager = Arc::new(ConnectionManager::new(test_config()));

    // Multiple threads trying to connect simultaneously
    let handles: Vec<_> = (0..5)
        .map(|_| {
            let manager_clone = manager.clone();
            tokio::spawn(async move { manager_clone.get_client().await })
        })
        .collect();

    let results = futures::future::join_all(handles).await;

    // All should complete without panic
    for result in results {
        assert!(result.is_ok()); // Task completed
                                 // The actual connection will fail due to invalid endpoints
    }
}

#[tokio::test]
async fn test_connection_manager_consecutive_failures() {
    let mut manager = ConnectionManager::new(test_config());
    manager.max_consecutive_failures = 2;

    // Simulate failures
    manager
        .update_state(ConnectionState::Failed {
            error: "test".to_string(),
            at: tokio::time::Instant::now(),
            consecutive_failures: 3,
        })
        .await;

    // Should fail due to exceeding max consecutive failures
    let result = manager.reconnect_with_backoff().await;
    assert!(result.is_err());
}

// Performance test - ensure connection pool doesn't leak memory
#[tokio::test]
#[ignore] // Run with --ignored for performance tests
async fn test_connection_pool_memory_stability() {
    let pool = Arc::new(ConnectionPool::new(
        vec!["wss://test.invalid:443".to_string()],
        1,
    ));

    // Run many health checks
    for _ in 0..1000 {
        let _ = pool.healthy_connection_count().await;
        // Small delay to avoid overwhelming
        sleep(Duration::from_micros(100)).await;
    }

    // Memory should be stable (manual verification needed)
}
