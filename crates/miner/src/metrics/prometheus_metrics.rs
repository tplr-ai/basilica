//! Core Prometheus metrics implementation for Miner

use anyhow::Result;
use metrics::{counter, describe_counter, describe_gauge, describe_histogram, gauge, histogram};
use std::sync::Arc;
use std::time::{Duration, Instant, SystemTime};
use tokio::sync::RwLock;
use tracing::{debug, error, warn};

/// Core Prometheus metrics collector for Miner
pub struct MinerPrometheusMetrics {
    // Performance tracking
    start_time: Instant,
    last_collection: Arc<RwLock<SystemTime>>,
}

impl MinerPrometheusMetrics {
    /// Create new Prometheus metrics collector
    pub fn new() -> Result<Self> {
        // Register and describe all metrics

        // Executor management metrics
        describe_gauge!(
            "basilica_miner_executors_total",
            "Total number of managed executors"
        );
        describe_gauge!(
            "basilica_miner_executors_healthy",
            "Number of healthy executors"
        );
        describe_gauge!(
            "basilica_miner_executors_unhealthy",
            "Number of unhealthy executors"
        );
        describe_counter!(
            "basilica_miner_executor_health_checks_total",
            "Total executor health checks performed"
        );
        describe_histogram!(
            "basilica_miner_executor_health_check_duration_seconds",
            "Duration of executor health checks"
        );

        // Validator interaction metrics
        describe_counter!(
            "basilica_miner_validator_requests_total",
            "Total requests from validators"
        );
        describe_counter!(
            "basilica_miner_validator_auth_failures_total",
            "Total validator authentication failures"
        );
        describe_gauge!(
            "basilica_miner_validator_sessions_active",
            "Currently active validator sessions"
        );
        describe_histogram!(
            "basilica_miner_validator_session_duration_seconds",
            "Duration of validator sessions"
        );
        describe_counter!(
            "basilica_miner_validator_executor_discoveries_total",
            "Total executor discovery requests"
        );

        // SSH management metrics
        describe_counter!(
            "basilica_miner_ssh_sessions_created_total",
            "Total SSH sessions created"
        );
        describe_counter!(
            "basilica_miner_ssh_sessions_closed_total",
            "Total SSH sessions closed"
        );
        describe_gauge!(
            "basilica_miner_ssh_sessions_active",
            "Currently active SSH sessions"
        );
        describe_histogram!(
            "basilica_miner_ssh_session_duration_seconds",
            "Duration of SSH sessions"
        );
        describe_counter!(
            "basilica_miner_ssh_key_deployments_total",
            "Total SSH key deployments"
        );
        describe_counter!(
            "basilica_miner_ssh_failures_total",
            "Total SSH operation failures"
        );

        // Deployment metrics
        describe_counter!(
            "basilica_miner_deployment_attempts_total",
            "Total deployment attempts"
        );
        describe_counter!(
            "basilica_miner_deployment_failures_total",
            "Total deployment failures"
        );
        describe_histogram!(
            "basilica_miner_deployment_duration_seconds",
            "Duration of deployment operations"
        );
        describe_gauge!(
            "basilica_miner_remote_executors_deployed",
            "Number of remotely deployed executors"
        );

        // Database metrics
        describe_counter!(
            "basilica_miner_database_operations_total",
            "Total database operations"
        );
        describe_counter!(
            "basilica_miner_database_errors_total",
            "Total database errors"
        );
        describe_histogram!(
            "basilica_miner_database_query_duration_seconds",
            "Database query duration"
        );
        describe_gauge!(
            "basilica_miner_database_connections_active",
            "Active database connections"
        );

        // Bittensor integration metrics
        describe_counter!(
            "basilica_miner_bittensor_registrations_total",
            "Total Bittensor network registrations"
        );
        describe_counter!(
            "basilica_miner_bittensor_errors_total",
            "Total Bittensor operation errors"
        );
        describe_gauge!("basilica_miner_bittensor_uid", "Current Bittensor UID");
        describe_gauge!("basilica_miner_bittensor_stake", "Current stake amount");
        describe_counter!(
            "basilica_miner_axon_requests_total",
            "Total axon server requests"
        );

        // System metrics
        describe_gauge!("basilica_miner_uptime_seconds", "Miner uptime in seconds");
        describe_gauge!("basilica_miner_memory_usage_bytes", "Memory usage in bytes");
        describe_gauge!("basilica_miner_cpu_usage_percent", "CPU usage percentage");

        Ok(Self {
            start_time: Instant::now(),
            last_collection: Arc::new(RwLock::new(SystemTime::now())),
        })
    }

    /// Record executor health check
    pub fn record_executor_health_check(
        &self,
        _executor_id: &str,
        success: bool,
        duration: Duration,
        healthy: bool,
    ) {
        counter!("basilica_miner_executor_health_checks_total").increment(1);
        histogram!("basilica_miner_executor_health_check_duration_seconds")
            .record(duration.as_secs_f64());

        debug!(
            "Recorded executor health check: success={}, healthy={}, duration={:?}",
            success, healthy, duration
        );
    }

    /// Update executor counts
    pub fn update_executor_counts(&self, total: u64, healthy: u64, unhealthy: u64) {
        gauge!("basilica_miner_executors_total").set(total as f64);
        gauge!("basilica_miner_executors_healthy").set(healthy as f64);
        gauge!("basilica_miner_executors_unhealthy").set(unhealthy as f64);
    }

    /// Record validator request
    pub fn record_validator_request(
        &self,
        _validator_hotkey: &str,
        _request_type: &str,
        success: bool,
        duration: Duration,
    ) {
        counter!("basilica_miner_validator_requests_total").increment(1);

        if !success {
            counter!("basilica_miner_validator_auth_failures_total").increment(1);
        }

        debug!(
            "Recorded validator request: success={}, duration={:?}",
            success, duration
        );
    }

    /// Update active validator sessions
    pub fn set_active_validator_sessions(&self, count: u64) {
        gauge!("basilica_miner_validator_sessions_active").set(count as f64);
    }

    /// Record validator session
    pub fn record_validator_session(&self, _validator_hotkey: &str, duration: Duration) {
        histogram!("basilica_miner_validator_session_duration_seconds")
            .record(duration.as_secs_f64());
    }

    /// Record executor discovery request
    pub fn record_executor_discovery(&self, _validator_hotkey: &str, executors_returned: u32) {
        counter!("basilica_miner_validator_executor_discoveries_total").increment(1);
        debug!(
            "Recorded executor discovery: executors_returned={}",
            executors_returned
        );
    }

    /// Record SSH session creation
    pub fn record_ssh_session_created(&self, _executor_id: &str, _validator_hotkey: &str) {
        counter!("basilica_miner_ssh_sessions_created_total").increment(1);
    }

    /// Record SSH session closure
    pub fn record_ssh_session_closed(
        &self,
        _executor_id: &str,
        _validator_hotkey: &str,
        duration: Duration,
    ) {
        counter!("basilica_miner_ssh_sessions_closed_total").increment(1);
        histogram!("basilica_miner_ssh_session_duration_seconds").record(duration.as_secs_f64());
    }

    /// Update active SSH sessions count
    pub fn set_active_ssh_sessions(&self, count: u64) {
        gauge!("basilica_miner_ssh_sessions_active").set(count as f64);
    }

    /// Record SSH key deployment
    pub fn record_ssh_key_deployment(
        &self,
        _executor_id: &str,
        success: bool,
        _operation_type: &str,
    ) {
        counter!("basilica_miner_ssh_key_deployments_total").increment(1);

        if !success {
            counter!("basilica_miner_ssh_failures_total").increment(1);
        }
    }

    /// Record deployment operation
    pub fn record_deployment(
        &self,
        _executor_id: &str,
        success: bool,
        duration: Duration,
        _deployment_type: &str,
    ) {
        counter!("basilica_miner_deployment_attempts_total").increment(1);
        histogram!("basilica_miner_deployment_duration_seconds").record(duration.as_secs_f64());

        if !success {
            counter!("basilica_miner_deployment_failures_total").increment(1);
        }

        debug!(
            "Recorded deployment: success={}, duration={:?}",
            success, duration
        );
    }

    /// Update remote executors deployed count
    pub fn set_remote_executors_deployed(&self, count: u64) {
        gauge!("basilica_miner_remote_executors_deployed").set(count as f64);
    }

    /// Record database operation
    pub fn record_database_operation(&self, _operation: &str, success: bool, duration: Duration) {
        counter!("basilica_miner_database_operations_total").increment(1);
        histogram!("basilica_miner_database_query_duration_seconds").record(duration.as_secs_f64());

        if !success {
            counter!("basilica_miner_database_errors_total").increment(1);
        }
    }

    /// Set active database connections
    pub fn set_database_connections(&self, count: u64) {
        gauge!("basilica_miner_database_connections_active").set(count as f64);
    }

    /// Record Bittensor registration
    pub fn record_bittensor_registration(&self, success: bool, uid: Option<u16>) {
        counter!("basilica_miner_bittensor_registrations_total").increment(1);

        if let Some(uid_value) = uid {
            gauge!("basilica_miner_bittensor_uid").set(uid_value as f64);
        }

        if !success {
            counter!("basilica_miner_bittensor_errors_total").increment(1);
        }
    }

    /// Update Bittensor stake
    pub fn set_bittensor_stake(&self, stake: u64) {
        gauge!("basilica_miner_bittensor_stake").set(stake as f64);
    }

    /// Record axon request
    pub fn record_axon_request(&self, _method: &str, success: bool) {
        counter!("basilica_miner_axon_requests_total").increment(1);
        debug!("Recorded axon request: success={}", success);
    }

    /// Record Bittensor error
    pub fn record_bittensor_error(&self, operation: &str, error_type: &str) {
        counter!("basilica_miner_bittensor_errors_total").increment(1);
        warn!(
            "Bittensor error: operation={}, error_type={}",
            operation, error_type
        );
    }

    /// Collect periodic metrics
    pub async fn collect_periodic_metrics(&self) {
        if let Err(e) = self.try_collect_periodic_metrics().await {
            error!("Failed to collect periodic metrics: {}", e);
        }
    }

    async fn try_collect_periodic_metrics(&self) -> Result<()> {
        // Update collection timestamp
        {
            let mut last_collection = self.last_collection.write().await;
            *last_collection = SystemTime::now();
        }

        // Update uptime
        let uptime = self.start_time.elapsed().as_secs_f64();
        gauge!("basilica_miner_uptime_seconds").set(uptime);

        // Collect system metrics
        if let Ok(memory_usage) = self.get_memory_usage().await {
            gauge!("basilica_miner_memory_usage_bytes").set(memory_usage as f64);
        }

        if let Ok(cpu_usage) = self.get_cpu_usage().await {
            gauge!("basilica_miner_cpu_usage_percent").set(cpu_usage);
        }

        Ok(())
    }

    async fn get_memory_usage(&self) -> Result<u64> {
        let meminfo = tokio::fs::read_to_string("/proc/meminfo").await?;
        let mut total = 0u64;
        let mut available = 0u64;

        for line in meminfo.lines() {
            if line.starts_with("MemTotal:") {
                total = line
                    .split_whitespace()
                    .nth(1)
                    .ok_or_else(|| anyhow::anyhow!("Invalid MemTotal format"))?
                    .parse::<u64>()?
                    * 1024; // Convert KB to bytes
            } else if line.starts_with("MemAvailable:") {
                available = line
                    .split_whitespace()
                    .nth(1)
                    .ok_or_else(|| anyhow::anyhow!("Invalid MemAvailable format"))?
                    .parse::<u64>()?
                    * 1024; // Convert KB to bytes
            }
        }

        Ok(total.saturating_sub(available))
    }

    async fn get_cpu_usage(&self) -> Result<f64> {
        // Read from /proc/loadavg for CPU load average
        let loadavg = tokio::fs::read_to_string("/proc/loadavg").await?;
        let load_1min: f64 = loadavg
            .split_whitespace()
            .next()
            .ok_or_else(|| anyhow::anyhow!("Invalid loadavg format"))?
            .parse()?;

        // Convert load average to percentage (approximate)
        Ok((load_1min * 100.0).min(100.0))
    }

    /// Get uptime in seconds
    pub fn uptime_seconds(&self) -> f64 {
        self.start_time.elapsed().as_secs_f64()
    }

    /// Get last collection timestamp
    pub async fn last_collection_timestamp(&self) -> SystemTime {
        *self.last_collection.read().await
    }
}
