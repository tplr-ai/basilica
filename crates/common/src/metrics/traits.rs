//! # Metrics Traits
//!
//! Core traits and interfaces for metrics collection and reporting.

use async_trait::async_trait;
use serde::{Deserialize, Serialize};
use std::collections::HashMap;
use std::fmt::Debug;
use std::time::{Duration, SystemTime};

/// Renamed from MetricsCollector to match SPEC
#[async_trait]
pub trait MetricsRecorder: Send + Sync {
    /// Record a counter metric (monotonically increasing value)
    async fn record_counter(&self, name: &str, value: u64, labels: &[(&str, &str)]);

    /// Record a histogram metric (distribution of values over time)
    async fn record_histogram(&self, name: &str, value: f64, labels: &[(&str, &str)]);

    /// Record a gauge metric (current value that can go up or down)
    async fn record_gauge(&self, name: &str, value: f64, labels: &[(&str, &str)]);

    /// Increment a counter by 1
    async fn increment_counter(&self, name: &str, labels: &[(&str, &str)]) {
        self.record_counter(name, 1, labels).await;
    }

    /// Record timing information
    async fn record_timing(&self, name: &str, duration: Duration, labels: &[(&str, &str)]) {
        self.record_histogram(name, duration.as_secs_f64(), labels)
            .await;
    }

    /// Create a timer for measuring operation duration
    fn start_timer(&self, name: &str, labels: Vec<(&str, &str)>) -> MetricTimer;
}

/// Renamed from SystemMetrics to match SPEC
#[async_trait]
pub trait SystemMetricsProvider: Send + Sync {
    /// Collect current CPU usage percentage (0.0 - 100.0)
    async fn cpu_usage(&self) -> Result<f64, anyhow::Error>;

    /// Collect memory usage (used_bytes, total_bytes)
    async fn memory_usage(&self) -> Result<(u64, u64), anyhow::Error>;

    /// Collect disk usage (used_bytes, total_bytes)
    async fn disk_usage(&self) -> Result<(u64, u64), anyhow::Error>;

    /// Collect network statistics (bytes_sent, bytes_received)
    async fn network_stats(&self) -> Result<(u64, u64), anyhow::Error>;

    /// Collect GPU metrics if available
    async fn collect_gpu_metrics(&self) -> Result<Option<GpuMetrics>, anyhow::Error>;

    /// Collect all system metrics in one call
    async fn collect_all(&self) -> Result<SystemMetricsSnapshot, anyhow::Error> {
        let cpu = self.cpu_usage().await?;
        let (memory_used, memory_total) = self.memory_usage().await?;
        let (disk_used, disk_total) = self.disk_usage().await?;
        let (net_sent, net_recv) = self.network_stats().await?;
        let gpu = self.collect_gpu_metrics().await?;

        Ok(SystemMetricsSnapshot {
            timestamp: SystemTime::now(),
            cpu_usage_percent: cpu,
            memory_used_bytes: memory_used,
            memory_total_bytes: memory_total,
            disk_used_bytes: disk_used,
            disk_total_bytes: disk_total,
            network_bytes_sent: net_sent,
            network_bytes_received: net_recv,
            gpu_metrics: gpu,
        })
    }
}

/// Business metrics specific to Basilca operations
#[async_trait]
pub trait BasilcaMetrics: Send + Sync {
    /// Record task execution metrics
    async fn record_task_execution(
        &self,
        task_type: &str,
        duration: Duration,
        success: bool,
        labels: &[(&str, &str)],
    );

    /// Record verification attempt
    async fn record_verification_attempt(
        &self,
        executor_id: &str,
        verification_type: &str,
        success: bool,
        score: Option<f64>,
    );

    /// Record mining operation
    async fn record_mining_operation(
        &self,
        operation: &str,
        miner_hotkey: &str,
        success: bool,
        duration: Duration,
    );

    /// Record validator operation
    async fn record_validator_operation(
        &self,
        operation: &str,
        validator_hotkey: &str,
        success: bool,
        duration: Duration,
    );

    /// Record executor health status
    async fn record_executor_health(&self, executor_id: &str, healthy: bool);

    /// Record network consensus metrics
    async fn record_consensus_metrics(&self, weights_set: bool, stake_amount: u64);
}

/// Timer for measuring operation duration
pub struct MetricTimer {
    name: String,
    labels: Vec<(String, String)>,
    start_time: SystemTime,
}

impl MetricTimer {
    pub fn new(name: String, labels: Vec<(&str, &str)>) -> Self {
        Self {
            name,
            labels: labels
                .into_iter()
                .map(|(k, v)| (k.to_string(), v.to_string()))
                .collect(),
            start_time: SystemTime::now(),
        }
    }

    /// Finish timing and record the metric
    pub async fn finish(self, recorder: &dyn MetricsRecorder) {
        if let Ok(duration) = self.start_time.elapsed() {
            let labels: Vec<(&str, &str)> = self
                .labels
                .iter()
                .map(|(k, v)| (k.as_str(), v.as_str()))
                .collect();
            recorder.record_timing(&self.name, duration, &labels).await;
        }
    }
}

/// GPU metrics information
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct GpuMetrics {
    pub gpu_count: u32,
    pub devices: Vec<GpuDevice>,
}

/// Individual GPU device metrics
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct GpuDevice {
    pub device_id: u32,
    pub name: String,
    pub utilization_percent: f64,
    pub memory_used_bytes: u64,
    pub memory_total_bytes: u64,
    pub temperature_celsius: Option<f64>,
    pub power_usage_watts: Option<f64>,
}

/// Complete system metrics snapshot
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct SystemMetricsSnapshot {
    pub timestamp: SystemTime,
    pub cpu_usage_percent: f64,
    pub memory_used_bytes: u64,
    pub memory_total_bytes: u64,
    pub disk_used_bytes: u64,
    pub disk_total_bytes: u64,
    pub network_bytes_sent: u64,
    pub network_bytes_received: u64,
    pub gpu_metrics: Option<GpuMetrics>,
}

/// Metrics aggregation for reporting
#[async_trait]
pub trait MetricsAggregator: Send + Sync {
    /// Get aggregated metrics for a time period
    async fn get_aggregated_metrics(
        &self,
        start_time: SystemTime,
        end_time: SystemTime,
        interval: Duration,
    ) -> Result<Vec<AggregatedMetric>, anyhow::Error>;

    /// Get top N metrics by value
    async fn get_top_metrics(
        &self,
        metric_name: &str,
        limit: usize,
        time_window: Duration,
    ) -> Result<Vec<MetricPoint>, anyhow::Error>;
}

/// Aggregated metric data point
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct AggregatedMetric {
    pub name: String,
    pub timestamp: SystemTime,
    pub labels: HashMap<String, String>,
    pub value: MetricValue,
}

/// Metric value types
#[derive(Debug, Clone, Serialize, Deserialize)]
pub enum MetricValue {
    Counter(u64),
    Gauge(f64),
    Histogram {
        count: u64,
        sum: f64,
        min: f64,
        max: f64,
        mean: f64,
        percentiles: HashMap<String, f64>, // "p50", "p95", "p99", etc.
    },
}

/// Individual metric data point
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct MetricPoint {
    pub timestamp: SystemTime,
    pub value: f64,
    pub labels: HashMap<String, String>,
}

/// Metrics export interface for external systems
#[async_trait]
pub trait MetricsExporter: Send + Sync {
    /// Export metrics in Prometheus format
    async fn export_prometheus(&self) -> Result<String, anyhow::Error>;

    /// Export metrics as JSON
    async fn export_json(&self) -> Result<String, anyhow::Error>;

    /// Export specific metric by name
    async fn export_metric(
        &self,
        name: &str,
        format: ExportFormat,
    ) -> Result<String, anyhow::Error>;
}

/// Supported export formats
#[derive(Debug, Clone, Copy)]
pub enum ExportFormat {
    Prometheus,
    Json,
    Csv,
}

/// Service metrics collection
pub trait ServiceMetrics {
    /// Record service start event
    fn record_start(&self, service_name: &str);

    /// Record service stop event
    fn record_stop(&self, service_name: &str);

    /// Record service error
    fn record_error(&self, service_name: &str, error: &str);

    /// Record health check result
    fn record_health_check(
        &self,
        service_name: &str,
        status: crate::services::HealthState,
        duration: Duration,
    );

    /// Record service restart
    fn record_restart(&self, service_name: &str);
}

/// Common metric names used across Basilca components
pub mod metric_names {
    // System metrics
    pub const CPU_USAGE: &str = "basilca_cpu_usage_percent";
    pub const MEMORY_USAGE: &str = "basilca_memory_usage_bytes";
    pub const DISK_USAGE: &str = "basilca_disk_usage_bytes";
    pub const NETWORK_IO: &str = "basilca_network_io_bytes";
    pub const GPU_UTILIZATION: &str = "basilca_gpu_utilization_percent";

    // Task metrics
    pub const TASK_DURATION: &str = "basilca_task_duration_seconds";
    pub const TASK_COUNT: &str = "basilca_task_count_total";
    pub const TASK_ERRORS: &str = "basilca_task_errors_total";

    // Verification metrics
    pub const VERIFICATION_DURATION: &str = "basilca_verification_duration_seconds";
    pub const VERIFICATION_SCORE: &str = "basilca_verification_score";
    pub const VERIFICATION_SUCCESS: &str = "basilca_verification_success_total";

    // Network metrics
    pub const CONSENSUS_WEIGHT_SETS: &str = "basilca_consensus_weight_sets_total";
    pub const NETWORK_STAKE: &str = "basilca_network_stake_amount";
    pub const PEER_CONNECTIONS: &str = "basilca_peer_connections_total";

    // Health metrics
    pub const SERVICE_HEALTH: &str = "basilca_service_health_status";
    pub const EXECUTOR_HEALTH: &str = "basilca_executor_health_status";
}
