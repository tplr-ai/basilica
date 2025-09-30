//! Connection state management for observability and debugging

use crate::error::BittensorError;
use basilica_common::config::BittensorConfig;
use std::sync::atomic::{AtomicU64, Ordering};
use std::sync::Arc;
use std::time::Duration;
use subxt::{OnlineClient, PolkadotConfig};
use tokio::sync::RwLock;
use tokio::time::Instant;
use tracing::{debug, info, warn};

type ChainClient = OnlineClient<PolkadotConfig>;

/// Connection state for monitoring and debugging
#[derive(Debug, Clone)]
pub enum ConnectionState {
    /// Successfully connected
    Connected { since: Instant, endpoint: String },
    /// Currently attempting to reconnect
    Reconnecting {
        attempts: u32,
        since: Instant,
        last_error: Option<String>,
    },
    /// Connection failed
    Failed {
        error: String,
        at: Instant,
        consecutive_failures: u32,
    },
    /// Not yet initialized
    Uninitialized,
}

impl ConnectionState {
    /// Check if the connection is healthy
    pub fn is_healthy(&self) -> bool {
        matches!(self, ConnectionState::Connected { .. })
    }

    /// Get a human-readable status message
    pub fn status_message(&self) -> String {
        match self {
            ConnectionState::Connected { since, endpoint } => {
                format!("Connected to {} (uptime: {:?})", endpoint, since.elapsed())
            }
            ConnectionState::Reconnecting {
                attempts,
                since,
                last_error,
            } => {
                let error_msg = last_error.as_deref().unwrap_or("unknown");
                format!(
                    "Reconnecting (attempt {}, elapsed: {:?}, last error: {})",
                    attempts,
                    since.elapsed(),
                    error_msg
                )
            }
            ConnectionState::Failed {
                error,
                at,
                consecutive_failures,
            } => {
                format!(
                    "Failed {} times (last: {:?} ago): {}",
                    consecutive_failures,
                    at.elapsed(),
                    error
                )
            }
            ConnectionState::Uninitialized => "Not initialized".to_string(),
        }
    }
}

/// Manages connection lifecycle and state transitions
pub struct ConnectionManager {
    state: Arc<RwLock<ConnectionState>>,
    client: Arc<RwLock<Option<Arc<ChainClient>>>>,
    config: BittensorConfig,
    metrics: Arc<ConnectionMetrics>,
    #[doc(hidden)]
    pub max_consecutive_failures: u32,
}

impl ConnectionManager {
    pub fn new(config: BittensorConfig) -> Self {
        Self {
            state: Arc::new(RwLock::new(ConnectionState::Uninitialized)),
            client: Arc::new(RwLock::new(None)),
            config,
            metrics: Arc::new(ConnectionMetrics::new()),
            max_consecutive_failures: 10,
        }
    }

    /// Establish initial connection
    pub async fn connect(&self) -> Result<(), BittensorError> {
        self.update_state(ConnectionState::Reconnecting {
            attempts: 1,
            since: Instant::now(),
            last_error: None,
        })
        .await;

        match self.establish_connection().await {
            Ok((client, endpoint)) => {
                *self.client.write().await = Some(Arc::new(client));

                self.update_state(ConnectionState::Connected {
                    since: Instant::now(),
                    endpoint: endpoint.clone(),
                })
                .await;

                self.metrics.record_connection_success();
                info!("Successfully connected to {}", endpoint);
                Ok(())
            }
            Err(e) => {
                let error_msg = e.to_string();

                self.update_state(ConnectionState::Failed {
                    error: error_msg.clone(),
                    at: Instant::now(),
                    consecutive_failures: 1,
                })
                .await;

                self.metrics.record_connection_failure();
                Err(e)
            }
        }
    }

    /// Get client with automatic reconnection
    pub async fn get_client(&self) -> Result<Arc<ChainClient>, BittensorError> {
        let state = self.state.read().await.clone();

        match state {
            ConnectionState::Connected { .. } => {
                // Fast path: already connected
                self.client.read().await.as_ref().cloned().ok_or_else(|| {
                    BittensorError::ServiceUnavailable {
                        message: "Client not initialized despite connected state".to_string(),
                    }
                })
            }
            ConnectionState::Reconnecting {
                attempts, since, ..
            } => {
                // Wait for ongoing reconnection or trigger new one
                if since.elapsed() > Duration::from_secs(30) {
                    drop(state);
                    self.reconnect_with_backoff().await
                } else {
                    Err(BittensorError::ServiceUnavailable {
                        message: format!("Reconnecting (attempt {})", attempts),
                    })
                }
            }
            ConnectionState::Failed {
                at,
                consecutive_failures,
                ..
            } => {
                // Retry after a delay
                let retry_delay = self.calculate_retry_delay(consecutive_failures);

                if at.elapsed() > retry_delay {
                    drop(state);
                    self.reconnect_with_backoff().await
                } else {
                    Err(BittensorError::ServiceUnavailable {
                        message: format!(
                            "Connection failed, retry in {:?}",
                            retry_delay.saturating_sub(at.elapsed())
                        ),
                    })
                }
            }
            ConnectionState::Uninitialized => {
                drop(state);
                self.connect().await?;
                Box::pin(self.get_client()).await
            }
        }
    }

    /// Reconnect with exponential backoff
    #[doc(hidden)]
    pub async fn reconnect_with_backoff(&self) -> Result<Arc<ChainClient>, BittensorError> {
        let mut attempts = 0u32;
        let mut consecutive_failures = self.get_consecutive_failures().await;

        loop {
            attempts += 1;
            consecutive_failures += 1;

            if consecutive_failures > self.max_consecutive_failures {
                return Err(BittensorError::NetworkError {
                    message: format!(
                        "Maximum consecutive failures ({}) exceeded",
                        self.max_consecutive_failures
                    ),
                });
            }

            self.update_state(ConnectionState::Reconnecting {
                attempts,
                since: Instant::now(),
                last_error: None,
            })
            .await;

            match self.establish_connection().await {
                Ok((client, endpoint)) => {
                    let client_arc = Arc::new(client);
                    *self.client.write().await = Some(Arc::clone(&client_arc));

                    self.update_state(ConnectionState::Connected {
                        since: Instant::now(),
                        endpoint,
                    })
                    .await;

                    self.metrics.record_connection_success();
                    return Ok(client_arc);
                }
                Err(e) => {
                    let error_msg = e.to_string();
                    warn!("Reconnection attempt {} failed: {}", attempts, error_msg);

                    self.update_state(ConnectionState::Failed {
                        error: error_msg,
                        at: Instant::now(),
                        consecutive_failures,
                    })
                    .await;

                    self.metrics.record_connection_failure();

                    if attempts >= 3 {
                        return Err(e);
                    }

                    let delay = self.calculate_retry_delay(attempts);
                    tokio::time::sleep(delay).await;
                }
            }
        }
    }

    /// Establish connection to any available endpoint
    async fn establish_connection(&self) -> Result<(ChainClient, String), BittensorError> {
        let endpoints = self.config.get_chain_endpoints();

        for (idx, endpoint) in endpoints.iter().enumerate() {
            debug!(
                "Trying endpoint {}/{}: {}",
                idx + 1,
                endpoints.len(),
                endpoint
            );

            let timeout_duration = Duration::from_secs(30);

            match tokio::time::timeout(
                timeout_duration,
                OnlineClient::<PolkadotConfig>::from_url(endpoint),
            )
            .await
            {
                Ok(Ok(client)) => {
                    info!("Successfully connected to {}", endpoint);
                    return Ok((client, endpoint.to_string()));
                }
                Ok(Err(e)) => {
                    warn!("Failed to connect to {}: {}", endpoint, e);
                }
                Err(_) => {
                    warn!(
                        "Connection to {} timed out after {:?}",
                        endpoint, timeout_duration
                    );
                }
            }

            // Small delay between endpoint attempts
            if idx < endpoints.len() - 1 {
                tokio::time::sleep(Duration::from_millis(500)).await;
            }
        }

        Err(BittensorError::NetworkError {
            message: "Failed to connect to any endpoint".to_string(),
        })
    }

    /// Update connection state
    #[doc(hidden)]
    pub async fn update_state(&self, new_state: ConnectionState) {
        *self.state.write().await = new_state;
    }

    /// Get current consecutive failures count
    async fn get_consecutive_failures(&self) -> u32 {
        match &*self.state.read().await {
            ConnectionState::Failed {
                consecutive_failures,
                ..
            } => *consecutive_failures,
            _ => 0,
        }
    }

    /// Calculate retry delay based on attempt number
    fn calculate_retry_delay(&self, attempt: u32) -> Duration {
        let base_delay = Duration::from_secs(1);
        let max_delay = Duration::from_secs(60);

        let exponential_delay = base_delay * 2u32.pow(attempt.saturating_sub(1));
        exponential_delay.min(max_delay)
    }

    /// Get current connection state
    pub async fn get_state(&self) -> ConnectionState {
        self.state.read().await.clone()
    }

    /// Get connection metrics
    pub fn metrics(&self) -> ConnectionMetricsSnapshot {
        self.metrics.snapshot()
    }

    /// Force reconnection
    pub async fn force_reconnect(&self) -> Result<(), BittensorError> {
        info!("Forcing reconnection");
        self.update_state(ConnectionState::Uninitialized).await;
        self.connect().await
    }

    /// Check if connected
    pub async fn is_connected(&self) -> bool {
        self.state.read().await.is_healthy()
    }
}

/// Connection metrics for monitoring
struct ConnectionMetrics {
    success_count: AtomicU64,
    failure_count: AtomicU64,
    total_reconnects: AtomicU64,
    last_success: Arc<RwLock<Option<Instant>>>,
    last_failure: Arc<RwLock<Option<Instant>>>,
}

impl ConnectionMetrics {
    fn new() -> Self {
        Self {
            success_count: AtomicU64::new(0),
            failure_count: AtomicU64::new(0),
            total_reconnects: AtomicU64::new(0),
            last_success: Arc::new(RwLock::new(None)),
            last_failure: Arc::new(RwLock::new(None)),
        }
    }

    fn record_connection_success(&self) {
        self.success_count.fetch_add(1, Ordering::Relaxed);
        let last_success = Arc::clone(&self.last_success);
        tokio::spawn(async move {
            *last_success.write().await = Some(Instant::now());
        });
    }

    fn record_connection_failure(&self) {
        self.failure_count.fetch_add(1, Ordering::Relaxed);
        self.total_reconnects.fetch_add(1, Ordering::Relaxed);
        let last_failure = Arc::clone(&self.last_failure);
        tokio::spawn(async move {
            *last_failure.write().await = Some(Instant::now());
        });
    }

    fn snapshot(&self) -> ConnectionMetricsSnapshot {
        ConnectionMetricsSnapshot {
            success_count: self.success_count.load(Ordering::Relaxed),
            failure_count: self.failure_count.load(Ordering::Relaxed),
            total_reconnects: self.total_reconnects.load(Ordering::Relaxed),
            success_rate: self.calculate_success_rate(),
        }
    }

    fn calculate_success_rate(&self) -> f64 {
        let successes = self.success_count.load(Ordering::Relaxed) as f64;
        let failures = self.failure_count.load(Ordering::Relaxed) as f64;
        let total = successes + failures;

        if total == 0.0 {
            100.0
        } else {
            (successes / total) * 100.0
        }
    }
}

/// Snapshot of connection metrics
#[derive(Debug, Clone)]
pub struct ConnectionMetricsSnapshot {
    pub success_count: u64,
    pub failure_count: u64,
    pub total_reconnects: u64,
    pub success_rate: f64,
}

#[cfg(test)]
mod tests {
    use super::*;

    fn test_config() -> BittensorConfig {
        BittensorConfig {
            network: "test".to_string(),
            chain_endpoint: Some("wss://test.endpoint:443".to_string()),
            wallet_name: "test_wallet".to_string(),
            hotkey_name: "test_hotkey".to_string(),
            netuid: 1,
            ..Default::default()
        }
    }

    #[tokio::test]
    async fn test_connection_state_initialization() {
        let manager = ConnectionManager::new(test_config());
        let state = manager.get_state().await;
        assert!(matches!(state, ConnectionState::Uninitialized));
    }

    #[tokio::test]
    async fn test_connection_state_is_healthy() {
        let state = ConnectionState::Connected {
            since: Instant::now(),
            endpoint: "test".to_string(),
        };
        assert!(state.is_healthy());

        let state = ConnectionState::Failed {
            error: "error".to_string(),
            at: Instant::now(),
            consecutive_failures: 1,
        };
        assert!(!state.is_healthy());

        let state = ConnectionState::Reconnecting {
            attempts: 1,
            since: Instant::now(),
            last_error: None,
        };
        assert!(!state.is_healthy());

        let state = ConnectionState::Uninitialized;
        assert!(!state.is_healthy());
    }

    #[tokio::test]
    async fn test_status_message() {
        let state = ConnectionState::Connected {
            since: Instant::now(),
            endpoint: "wss://test:443".to_string(),
        };
        let msg = state.status_message();
        assert!(msg.contains("Connected to wss://test:443"));

        let state = ConnectionState::Failed {
            error: "connection refused".to_string(),
            at: Instant::now(),
            consecutive_failures: 3,
        };
        let msg = state.status_message();
        assert!(msg.contains("Failed 3 times"));
        assert!(msg.contains("connection refused"));

        let state = ConnectionState::Reconnecting {
            attempts: 2,
            since: Instant::now(),
            last_error: Some("timeout".to_string()),
        };
        let msg = state.status_message();
        assert!(msg.contains("attempt 2"));
        assert!(msg.contains("timeout"));

        let state = ConnectionState::Uninitialized;
        assert_eq!(state.status_message(), "Not initialized");
    }

    #[tokio::test]
    async fn test_calculate_retry_delay() {
        let manager = ConnectionManager::new(test_config());

        let delay1 = manager.calculate_retry_delay(1);
        assert_eq!(delay1, Duration::from_secs(1));

        let delay2 = manager.calculate_retry_delay(2);
        assert_eq!(delay2, Duration::from_secs(2));

        let delay3 = manager.calculate_retry_delay(3);
        assert_eq!(delay3, Duration::from_secs(4));

        let delay4 = manager.calculate_retry_delay(4);
        assert_eq!(delay4, Duration::from_secs(8));

        // Test max delay cap
        let delay_max = manager.calculate_retry_delay(10);
        assert_eq!(delay_max, Duration::from_secs(60));
    }

    #[tokio::test]
    async fn test_get_consecutive_failures() {
        let manager = ConnectionManager::new(test_config());

        // Initially should be 0
        let failures = manager.get_consecutive_failures().await;
        assert_eq!(failures, 0);

        // Update to failed state
        manager
            .update_state(ConnectionState::Failed {
                error: "test".to_string(),
                at: Instant::now(),
                consecutive_failures: 5,
            })
            .await;

        let failures = manager.get_consecutive_failures().await;
        assert_eq!(failures, 5);

        // Update to connected state
        manager
            .update_state(ConnectionState::Connected {
                since: Instant::now(),
                endpoint: "test".to_string(),
            })
            .await;

        let failures = manager.get_consecutive_failures().await;
        assert_eq!(failures, 0);
    }

    #[tokio::test]
    async fn test_is_connected() {
        let manager = ConnectionManager::new(test_config());

        // Initially not connected
        assert!(!manager.is_connected().await);

        // Update to connected state
        manager
            .update_state(ConnectionState::Connected {
                since: Instant::now(),
                endpoint: "test".to_string(),
            })
            .await;

        assert!(manager.is_connected().await);
    }

    #[tokio::test]
    async fn test_metrics_calculation() {
        let metrics = ConnectionMetrics::new();

        // Initial state
        let snapshot = metrics.snapshot();
        assert_eq!(snapshot.success_count, 0);
        assert_eq!(snapshot.failure_count, 0);
        assert_eq!(snapshot.total_reconnects, 0);
        assert_eq!(snapshot.success_rate, 100.0);

        // Record some successes and failures
        metrics.success_count.store(7, Ordering::Relaxed);
        metrics.failure_count.store(3, Ordering::Relaxed);
        metrics.total_reconnects.store(3, Ordering::Relaxed);

        let snapshot = metrics.snapshot();
        assert_eq!(snapshot.success_count, 7);
        assert_eq!(snapshot.failure_count, 3);
        assert_eq!(snapshot.total_reconnects, 3);
        assert!((snapshot.success_rate - 70.0).abs() < 0.01);
    }

    #[tokio::test]
    async fn test_connection_manager_get_client_uninitialized() {
        let manager = ConnectionManager::new(test_config());

        // Getting client when uninitialized should try to connect
        let result = manager.get_client().await;
        assert!(result.is_err()); // Will fail as we don't have a real endpoint
    }

    #[tokio::test]
    async fn test_max_consecutive_failures() {
        let mut manager = ConnectionManager::new(test_config());
        manager.max_consecutive_failures = 2;

        // Set high consecutive failures
        manager
            .update_state(ConnectionState::Failed {
                error: "test".to_string(),
                at: Instant::now(),
                consecutive_failures: 3,
            })
            .await;

        // Should fail due to max consecutive failures
        let result = manager.reconnect_with_backoff().await;
        assert!(result.is_err());

        if let Err(BittensorError::NetworkError { message }) = result {
            assert!(message.contains("Maximum consecutive failures"));
        } else {
            panic!("Expected NetworkError with max failures message");
        }
    }

    #[tokio::test]
    async fn test_state_transitions() {
        let manager = ConnectionManager::new(test_config());

        // Uninitialized -> Reconnecting
        manager
            .update_state(ConnectionState::Reconnecting {
                attempts: 1,
                since: Instant::now(),
                last_error: None,
            })
            .await;

        let state = manager.get_state().await;
        assert!(matches!(state, ConnectionState::Reconnecting { .. }));

        // Reconnecting -> Failed
        manager
            .update_state(ConnectionState::Failed {
                error: "error".to_string(),
                at: Instant::now(),
                consecutive_failures: 1,
            })
            .await;

        let state = manager.get_state().await;
        assert!(matches!(state, ConnectionState::Failed { .. }));

        // Failed -> Connected
        manager
            .update_state(ConnectionState::Connected {
                since: Instant::now(),
                endpoint: "endpoint".to_string(),
            })
            .await;

        let state = manager.get_state().await;
        assert!(matches!(state, ConnectionState::Connected { .. }));
    }
}
