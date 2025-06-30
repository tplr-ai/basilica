//! Core Prometheus metrics implementation for Validator

use anyhow::Result;
use metrics::{counter, describe_counter, describe_gauge, describe_histogram, gauge, histogram};
use std::sync::Arc;
use std::time::{Duration, Instant, SystemTime};
use tokio::sync::RwLock;
use tracing::{debug, error};

/// Core Prometheus metrics collector for Validator
pub struct ValidatorPrometheusMetrics {
    // Performance tracking
    start_time: Instant,
    last_collection: Arc<RwLock<SystemTime>>,
}

impl ValidatorPrometheusMetrics {
    /// Create new Prometheus metrics collector
    pub fn new() -> Result<Self> {
        // Register and describe all metrics

        // Validation metrics
        describe_counter!(
            "basilica_validator_validations_total",
            "Total number of validations performed"
        );
        describe_histogram!(
            "basilica_validator_validation_duration_seconds",
            "Duration of validation operations"
        );
        describe_histogram!(
            "basilica_validator_validation_score",
            "Validation scores assigned to executors"
        );
        describe_counter!(
            "basilica_validator_validation_errors_total",
            "Total validation errors"
        );

        // SSH metrics
        describe_counter!(
            "basilica_validator_ssh_connections_total",
            "Total SSH connections made"
        );
        describe_histogram!(
            "basilica_validator_ssh_connection_duration_seconds",
            "SSH connection duration"
        );
        describe_counter!(
            "basilica_validator_ssh_failures_total",
            "Total SSH connection failures"
        );
        describe_gauge!(
            "basilica_validator_ssh_active_connections",
            "Currently active SSH connections"
        );

        // Database metrics
        describe_gauge!(
            "basilica_validator_database_connections_total",
            "Active database connections"
        );
        describe_histogram!(
            "basilica_validator_database_query_duration_seconds",
            "Database query duration"
        );
        describe_counter!(
            "basilica_validator_database_errors_total",
            "Total database errors"
        );
        describe_counter!(
            "basilica_validator_database_operations_total",
            "Total database operations"
        );

        // API metrics
        describe_counter!(
            "basilica_validator_http_requests_total",
            "Total HTTP requests processed"
        );
        describe_histogram!(
            "basilica_validator_http_request_duration_seconds",
            "HTTP request duration"
        );
        describe_histogram!(
            "basilica_validator_http_response_size_bytes",
            "HTTP response size"
        );

        // System metrics
        describe_gauge!(
            "basilica_validator_cpu_usage_percent",
            "CPU usage percentage"
        );
        describe_gauge!(
            "basilica_validator_memory_usage_bytes",
            "Memory usage in bytes"
        );
        describe_gauge!(
            "basilica_validator_memory_total_bytes",
            "Total memory in bytes"
        );
        describe_gauge!("basilica_validator_disk_usage_bytes", "Disk usage in bytes");
        describe_gauge!(
            "basilica_validator_disk_total_bytes",
            "Total disk space in bytes"
        );

        // Business metrics
        describe_gauge!(
            "basilica_validator_executor_health_status",
            "Executor health status (1=healthy, 0=unhealthy)"
        );
        describe_counter!(
            "basilica_validator_consensus_weight_sets_total",
            "Total consensus weight sets"
        );
        describe_histogram!(
            "basilica_validator_verification_session_duration_seconds",
            "Verification session duration"
        );
        describe_counter!(
            "basilica_validator_attestation_verification_total",
            "Total attestation verifications"
        );

        Ok(Self {
            start_time: Instant::now(),
            last_collection: Arc::new(RwLock::new(SystemTime::now())),
        })
    }

    /// Record validation operation
    pub fn record_validation(
        &self,
        _executor_id: &str,
        success: bool,
        duration: Duration,
        score: Option<f64>,
    ) {
        counter!("basilica_validator_validations_total").increment(1);
        histogram!("basilica_validator_validation_duration_seconds").record(duration.as_secs_f64());

        if let Some(score_value) = score {
            histogram!("basilica_validator_validation_score").record(score_value);
        }

        if !success {
            counter!("basilica_validator_validation_errors_total").increment(1);
        }

        debug!(
            "Recorded validation: success={}, duration={:?}, score={:?}",
            success, duration, score
        );
    }

    /// Record SSH connection operation
    pub fn record_ssh_connection(&self, _host: &str, success: bool, duration: Duration) {
        counter!("basilica_validator_ssh_connections_total").increment(1);
        histogram!("basilica_validator_ssh_connection_duration_seconds")
            .record(duration.as_secs_f64());

        if !success {
            counter!("basilica_validator_ssh_failures_total").increment(1);
        }
    }

    /// Update active SSH connections count
    pub fn set_active_ssh_connections(&self, count: i64) {
        gauge!("basilica_validator_ssh_active_connections").set(count as f64);
    }

    /// Record database operation
    pub fn record_database_operation(&self, _operation: &str, success: bool, duration: Duration) {
        counter!("basilica_validator_database_operations_total").increment(1);
        histogram!("basilica_validator_database_query_duration_seconds")
            .record(duration.as_secs_f64());

        if !success {
            counter!("basilica_validator_database_errors_total").increment(1);
        }
    }

    /// Set database connections count
    pub fn set_database_connections(&self, count: i64) {
        gauge!("basilica_validator_database_connections_total").set(count as f64);
    }

    /// Record HTTP request
    pub fn record_http_request(
        &self,
        _method: &str,
        _path: &str,
        _status_code: u16,
        duration: Duration,
        response_size: usize,
    ) {
        counter!("basilica_validator_http_requests_total").increment(1);
        histogram!("basilica_validator_http_request_duration_seconds")
            .record(duration.as_secs_f64());
        histogram!("basilica_validator_http_response_size_bytes").record(response_size as f64);
    }

    /// Update system metrics
    pub fn update_system_metrics(
        &self,
        cpu_percent: f64,
        memory_used: u64,
        memory_total: u64,
        disk_used: u64,
        disk_total: u64,
        _net_sent: u64,
        _net_received: u64,
    ) {
        gauge!("basilica_validator_cpu_usage_percent").set(cpu_percent);
        gauge!("basilica_validator_memory_usage_bytes").set(memory_used as f64);
        gauge!("basilica_validator_memory_total_bytes").set(memory_total as f64);
        gauge!("basilica_validator_disk_usage_bytes").set(disk_used as f64);
        gauge!("basilica_validator_disk_total_bytes").set(disk_total as f64);
    }

    /// Set executor health status
    pub fn set_executor_health(&self, _executor_id: &str, healthy: bool) {
        gauge!("basilica_validator_executor_health_status").set(if healthy { 1.0 } else { 0.0 });
    }

    /// Record consensus weight set operation
    pub fn record_consensus_weight_set(&self, _success: bool) {
        counter!("basilica_validator_consensus_weight_sets_total").increment(1);
    }

    /// Record verification session
    pub fn record_verification_session(&self, _session_type: &str, duration: Duration) {
        histogram!("basilica_validator_verification_session_duration_seconds")
            .record(duration.as_secs_f64());
    }

    /// Record attestation verification
    pub fn record_attestation_verification(&self, _success: bool, _attestation_type: &str) {
        counter!("basilica_validator_attestation_verification_total").increment(1);
    }

    /// Collect system metrics periodically
    pub async fn collect_system_metrics(&self) {
        if let Err(e) = self.try_collect_system_metrics().await {
            error!("Failed to collect system metrics: {}", e);
        }
    }

    async fn try_collect_system_metrics(&self) -> Result<()> {
        // Update collection timestamp
        {
            let mut last_collection = self.last_collection.write().await;
            *last_collection = SystemTime::now();
        }

        // Collect CPU usage
        if let Ok(cpu_info) = self.get_cpu_usage().await {
            gauge!("basilica_validator_cpu_usage_percent").set(cpu_info);
        }

        // Collect memory usage
        if let Ok((used, total)) = self.get_memory_usage().await {
            gauge!("basilica_validator_memory_usage_bytes").set(used as f64);
            gauge!("basilica_validator_memory_total_bytes").set(total as f64);
        }

        // Collect disk usage
        if let Ok((used, total)) = self.get_disk_usage().await {
            gauge!("basilica_validator_disk_usage_bytes").set(used as f64);
            gauge!("basilica_validator_disk_total_bytes").set(total as f64);
        }

        Ok(())
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

    async fn get_memory_usage(&self) -> Result<(u64, u64)> {
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

        let used = total.saturating_sub(available);
        Ok((used, total))
    }

    async fn get_disk_usage(&self) -> Result<(u64, u64)> {
        // Use statvfs-like approach via df command
        let output = tokio::process::Command::new("df")
            .arg("/")
            .arg("--output=used,size")
            .arg("--block-size=1")
            .output()
            .await?;

        let stdout = String::from_utf8(output.stdout)?;
        let lines: Vec<&str> = stdout.lines().collect();

        if lines.len() >= 2 {
            let data_line = lines[1];
            let parts: Vec<&str> = data_line.split_whitespace().collect();
            if parts.len() >= 2 {
                let used: u64 = parts[0].parse()?;
                let total: u64 = parts[1].parse()?;
                return Ok((used, total));
            }
        }

        Err(anyhow::anyhow!("Failed to parse df output"))
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
