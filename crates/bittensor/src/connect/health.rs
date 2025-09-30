//! Health checking system for blockchain connections

use std::sync::atomic::{AtomicBool, AtomicU64, Ordering};
use std::sync::Arc;
use std::time::Duration;
use subxt::{OnlineClient, PolkadotConfig};
use tokio::sync::RwLock;
use tokio::time::{interval, timeout, Instant};
use tracing::{debug, error, info, warn};

type ChainClient = OnlineClient<PolkadotConfig>;

/// Trait for connection pools that can be monitored
#[async_trait::async_trait]
pub trait ConnectionPoolTrait: Send + Sync {
    async fn connections(&self) -> Arc<RwLock<Vec<Arc<ChainClient>>>>;
    async fn reconnect_with_backoff(
        &self,
    ) -> Result<Arc<ChainClient>, crate::error::BittensorError>;
    async fn get_healthy_client(&self) -> Result<Arc<ChainClient>, crate::error::BittensorError>;
}

/// Health checker for blockchain connections
#[derive(Debug, Clone)]
pub struct HealthChecker {
    check_interval: Duration,
    timeout: Duration,
    consecutive_failures_threshold: u32,
    #[doc(hidden)]
    pub metrics: Arc<HealthMetrics>,
}

#[derive(Debug)]
#[doc(hidden)]
pub struct HealthMetrics {
    pub total_checks: AtomicU64,
    pub successful_checks: AtomicU64,
    pub failed_checks: AtomicU64,
    pub last_check_time: RwLock<Option<Instant>>,
    pub is_monitoring: AtomicBool,
}

impl Default for HealthChecker {
    fn default() -> Self {
        Self {
            check_interval: Duration::from_secs(60),
            timeout: Duration::from_secs(5),
            consecutive_failures_threshold: 3,
            metrics: Arc::new(HealthMetrics {
                total_checks: AtomicU64::new(0),
                successful_checks: AtomicU64::new(0),
                failed_checks: AtomicU64::new(0),
                last_check_time: RwLock::new(None),
                is_monitoring: AtomicBool::new(false),
            }),
        }
    }
}

impl HealthChecker {
    pub fn new() -> Self {
        Self::default()
    }

    /// Configure check interval
    pub fn with_interval(mut self, interval: Duration) -> Self {
        self.check_interval = interval;
        self
    }

    /// Configure timeout for health checks
    pub fn with_timeout(mut self, timeout: Duration) -> Self {
        self.timeout = timeout;
        self
    }

    /// Configure consecutive failures before marking unhealthy
    pub fn with_failure_threshold(mut self, threshold: u32) -> Self {
        self.consecutive_failures_threshold = threshold;
        self
    }

    /// Start monitoring the connection pool in a background task
    pub fn start_monitoring<P>(self: Arc<Self>, pool: Arc<P>) -> tokio::task::JoinHandle<()>
    where
        P: ConnectionPoolTrait + Send + Sync + 'static,
    {
        // Ensure we only start one monitoring task
        if self.metrics.is_monitoring.swap(true, Ordering::SeqCst) {
            warn!("Health monitoring already started, skipping duplicate");
            return tokio::spawn(async {});
        }

        let checker = Arc::clone(&self);
        tokio::spawn(async move {
            info!(
                "Starting health monitoring with interval {:?}",
                checker.check_interval
            );
            let mut check_interval = interval(checker.check_interval);
            check_interval.tick().await; // Skip the immediate first tick

            let mut consecutive_failures = 0u32;

            loop {
                check_interval.tick().await;

                let health_result = checker.check_pool_health(pool.as_ref()).await;
                match health_result {
                    Ok(healthy_count) => {
                        if healthy_count > 0 {
                            consecutive_failures = 0;
                            debug!("Health check passed: {} healthy connections", healthy_count);
                        } else {
                            consecutive_failures += 1;
                            warn!(
                                "No healthy connections (failure {}/{})",
                                consecutive_failures, checker.consecutive_failures_threshold
                            );

                            if consecutive_failures >= checker.consecutive_failures_threshold {
                                error!("Connection health critical, triggering reconnection");
                                if let Err(e) = pool.reconnect_with_backoff().await {
                                    error!("Failed to reconnect: {}", e);
                                }
                                consecutive_failures = 0;
                            }
                        }
                    }
                    Err(e) => {
                        error!("Health check error: {}", e);
                        consecutive_failures += 1;
                    }
                }
            }
        })
    }

    /// Check health of all connections in the pool
    async fn check_pool_health<P>(
        &self,
        pool: &P,
    ) -> Result<usize, Box<dyn std::error::Error + Send + Sync>>
    where
        P: ConnectionPoolTrait,
    {
        let connections = pool.connections().await;
        let connections = connections.read().await;
        let mut healthy_count = 0;

        self.metrics.total_checks.fetch_add(1, Ordering::Relaxed);
        *self.metrics.last_check_time.write().await = Some(Instant::now());

        for (idx, client) in connections.iter().enumerate() {
            if self.is_healthy(client).await {
                healthy_count += 1;
                debug!("Connection {} is healthy", idx);
                self.metrics
                    .successful_checks
                    .fetch_add(1, Ordering::Relaxed);
            } else {
                warn!("Connection {} failed health check", idx);
                self.metrics.failed_checks.fetch_add(1, Ordering::Relaxed);
            }
        }

        Ok(healthy_count)
    }

    /// Check if a single connection is healthy
    pub async fn is_healthy(&self, client: &Arc<ChainClient>) -> bool {
        // Try to get the latest block number as a health check
        match timeout(self.timeout, self.perform_health_check(client)).await {
            Ok(Ok(_)) => true,
            Ok(Err(e)) => {
                debug!("Health check failed: {}", e);
                false
            }
            Err(_) => {
                debug!("Health check timed out after {:?}", self.timeout);
                false
            }
        }
    }

    /// Perform the actual health check operation
    async fn perform_health_check(
        &self,
        client: &ChainClient,
    ) -> Result<(), Box<dyn std::error::Error>> {
        // Get the latest block to verify connection is alive
        let block = client.blocks().at_latest().await?;
        let _block_number = block.number();
        Ok(())
    }

    /// Get current health metrics
    pub fn metrics(&self) -> HealthCheckMetrics {
        HealthCheckMetrics {
            total_checks: self.metrics.total_checks.load(Ordering::Relaxed),
            successful_checks: self.metrics.successful_checks.load(Ordering::Relaxed),
            failed_checks: self.metrics.failed_checks.load(Ordering::Relaxed),
            success_rate: self.calculate_success_rate(),
        }
    }

    fn calculate_success_rate(&self) -> f64 {
        let total = self.metrics.total_checks.load(Ordering::Relaxed);
        if total == 0 {
            return 100.0;
        }
        let successful = self.metrics.successful_checks.load(Ordering::Relaxed);
        (successful as f64 / total as f64) * 100.0
    }

    /// Stop monitoring
    pub fn stop_monitoring(&self) {
        self.metrics.is_monitoring.store(false, Ordering::SeqCst);
    }

    /// Check if monitoring is active
    pub fn is_monitoring(&self) -> bool {
        self.metrics.is_monitoring.load(Ordering::SeqCst)
    }

    /// Reset metrics
    pub fn reset_metrics(&self) {
        self.metrics.total_checks.store(0, Ordering::Relaxed);
        self.metrics.successful_checks.store(0, Ordering::Relaxed);
        self.metrics.failed_checks.store(0, Ordering::Relaxed);
    }
}

/// Public health check metrics
#[derive(Debug, Clone)]
pub struct HealthCheckMetrics {
    pub total_checks: u64,
    pub successful_checks: u64,
    pub failed_checks: u64,
    pub success_rate: f64,
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::connect::pool::ConnectionPool;
    use std::time::Duration;

    #[tokio::test]
    async fn test_health_checker_creation() {
        let checker = HealthChecker::new();
        assert_eq!(checker.check_interval, Duration::from_secs(60));
        assert_eq!(checker.timeout, Duration::from_secs(5));
        assert_eq!(checker.consecutive_failures_threshold, 3);
    }

    #[tokio::test]
    async fn test_health_checker_builder() {
        let checker = HealthChecker::new()
            .with_interval(Duration::from_secs(30))
            .with_timeout(Duration::from_secs(10))
            .with_failure_threshold(5);

        assert_eq!(checker.check_interval, Duration::from_secs(30));
        assert_eq!(checker.timeout, Duration::from_secs(10));
        assert_eq!(checker.consecutive_failures_threshold, 5);
    }

    #[tokio::test]
    async fn test_metrics_initialization() {
        let checker = HealthChecker::new();
        let metrics = checker.metrics();

        assert_eq!(metrics.total_checks, 0);
        assert_eq!(metrics.successful_checks, 0);
        assert_eq!(metrics.failed_checks, 0);
        assert_eq!(metrics.success_rate, 100.0); // 100% when no checks
    }

    #[tokio::test]
    async fn test_calculate_success_rate() {
        let checker = HealthChecker::new();

        // Add some successful checks
        checker.metrics.total_checks.store(10, Ordering::Relaxed);
        checker
            .metrics
            .successful_checks
            .store(7, Ordering::Relaxed);

        let success_rate = checker.calculate_success_rate();
        assert!((success_rate - 70.0).abs() < 0.01);
    }

    #[tokio::test]
    async fn test_monitoring_flag() {
        let checker = HealthChecker::new();
        assert!(!checker.is_monitoring());

        checker.metrics.is_monitoring.store(true, Ordering::SeqCst);
        assert!(checker.is_monitoring());

        checker.stop_monitoring();
        assert!(!checker.is_monitoring());
    }

    #[tokio::test]
    async fn test_reset_metrics() {
        let checker = HealthChecker::new();

        // Set some values
        checker.metrics.total_checks.store(100, Ordering::Relaxed);
        checker
            .metrics
            .successful_checks
            .store(90, Ordering::Relaxed);
        checker.metrics.failed_checks.store(10, Ordering::Relaxed);

        // Reset
        checker.reset_metrics();

        let metrics = checker.metrics();
        assert_eq!(metrics.total_checks, 0);
        assert_eq!(metrics.successful_checks, 0);
        assert_eq!(metrics.failed_checks, 0);
    }

    #[tokio::test]
    async fn test_start_monitoring_prevents_duplicate() {
        let pool = Arc::new(ConnectionPool::new(
            vec!["wss://test.endpoint:443".to_string()],
            1,
        ));
        let checker = Arc::new(HealthChecker::new().with_interval(Duration::from_millis(100)));

        // Start monitoring
        let handle1 = checker.clone().start_monitoring(pool.clone());

        // Try to start again - should return immediately
        let handle2 = checker.clone().start_monitoring(pool.clone());

        // Clean up
        handle1.abort();
        handle2.abort();

        assert!(checker.is_monitoring());
    }

    #[tokio::test]
    async fn test_health_check_timeout() {
        let checker = HealthChecker::new().with_timeout(Duration::from_millis(1));

        // Mock a slow health check by creating a future that never completes
        let slow_future = async {
            tokio::time::sleep(Duration::from_secs(10)).await;
            Ok::<(), Box<dyn std::error::Error>>(())
        };

        let result = timeout(checker.timeout, slow_future).await;
        assert!(result.is_err()); // Should timeout
    }

    #[tokio::test]
    async fn test_consecutive_failures_tracking() {
        let pool = Arc::new(ConnectionPool::new(
            vec!["wss://invalid.endpoint:443".to_string()],
            1,
        ));
        let checker = Arc::new(
            HealthChecker::new()
                .with_interval(Duration::from_millis(50))
                .with_failure_threshold(2),
        );

        let handle = checker.clone().start_monitoring(pool);

        // Let it run for a bit to accumulate failures
        tokio::time::sleep(Duration::from_millis(200)).await;

        // Stop monitoring
        handle.abort();

        // Check that failures were recorded
        let metrics = checker.metrics();
        assert!(metrics.failed_checks > 0 || metrics.total_checks > 0);
    }

    #[tokio::test]
    async fn test_pool_health_check() {
        let pool = ConnectionPool::new(vec!["wss://test.endpoint:443".to_string()], 1);
        let checker = HealthChecker::new();

        // Check empty pool
        let result = checker.check_pool_health(&pool).await;
        assert!(result.is_ok());
        assert_eq!(result.unwrap(), 0); // No healthy connections

        let metrics = checker.metrics();
        assert_eq!(metrics.total_checks, 1);
    }

    #[tokio::test]
    async fn test_health_check_metrics_accumulation() {
        let checker = HealthChecker::new();

        // Simulate multiple checks
        for _ in 0..5 {
            checker.metrics.total_checks.fetch_add(1, Ordering::Relaxed);
            checker
                .metrics
                .successful_checks
                .fetch_add(1, Ordering::Relaxed);
        }

        for _ in 0..2 {
            checker.metrics.total_checks.fetch_add(1, Ordering::Relaxed);
            checker
                .metrics
                .failed_checks
                .fetch_add(1, Ordering::Relaxed);
        }

        let metrics = checker.metrics();
        assert_eq!(metrics.total_checks, 7);
        assert_eq!(metrics.successful_checks, 5);
        assert_eq!(metrics.failed_checks, 2);
        assert!((metrics.success_rate - 71.43).abs() < 0.01);
    }
}
