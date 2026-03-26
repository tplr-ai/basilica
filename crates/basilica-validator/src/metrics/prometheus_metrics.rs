//! Core Prometheus metrics implementation for Validator

use anyhow::Result;
use chrono::{DateTime, Utc};
use metrics::{counter, describe_counter, describe_gauge, describe_histogram, gauge, histogram};
use std::sync::Arc;
use std::time::{Duration, SystemTime};
use tokio::sync::RwLock;
use tracing::{debug, error, info};

use crate::miner_prover::types::ValidationType;
use crate::miner_prover::validation_states::{StateResult, ValidationState};
use crate::persistence::SimplePersistence;

/// Core Prometheus metrics collector for Validator
pub struct ValidatorPrometheusMetrics {
    last_collection: Arc<RwLock<SystemTime>>,
    persistence: Arc<SimplePersistence>,
}

impl ValidatorPrometheusMetrics {
    /// Create new Prometheus metrics collector
    pub fn new(persistence: Arc<SimplePersistence>) -> Result<Self> {
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
            "Validation scores assigned to nodes"
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
            "basilica_validator_node_health_status",
            "Node health status (1=healthy, 0=unhealthy)"
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

        // GPU metrics
        describe_gauge!(
            "basilica_validator_miner_gpu_count",
            "Total GPU count per miner"
        );
        describe_histogram!(
            "basilica_validator_miner_gpu_weighted_score",
            "GPU count weighted scores for miners"
        );
        describe_gauge!("basilica_validator_node_gpu_count", "GPU count per node");

        // Weight metrics
        describe_gauge!(
            "basilica_validator_miner_weight",
            "Weight assigned to each miner"
        );

        // Validation metrics
        describe_counter!(
            "basilica_validator_miner_successful_validations",
            "Count of successful validations per miner"
        );

        // GPU profile metrics
        describe_gauge!(
            "basilica_validator_miner_gpu_profiles",
            "GPU profiles for miners"
        );

        // Node uptime metrics
        describe_gauge!(
            "basilica_node_uptime_minutes",
            "Node uptime in minutes for ramp-up calculation"
        );
        describe_gauge!(
            "basilica_node_uptime_multiplier",
            "Node uptime-based reward multiplier (0.0-1.0)"
        );

        // Rental metrics
        describe_gauge!(
            "basilica_validator_node_rental_status",
            "Node rental status (1=rented, 0=available)"
        );
        describe_counter!(
            "basilica_validator_rentals_created_total",
            "Total number of rentals created"
        );
        describe_gauge!(
            "validator_node_ban_till",
            "Unix timestamp (seconds) when a node ban expires; 0 indicates no active ban"
        );

        // RPC failure metrics
        describe_counter!(
            "basilica_validator_rpc_critical_failures_total",
            "Total RPC failures after all retry attempts exhausted"
        );

        // Discovered miners metrics
        describe_gauge!(
            "basilica_validator_discovered_miners_total",
            "Total number of miners currently discovered from metagraph"
        );

        // Validation state tracking metrics
        describe_gauge!(
            "basilica_validator_node_validation_state",
            "Current validation state of nodes (0=not in state, 1=current, 2=failed)"
        );

        // Billing telemetry metrics
        describe_counter!(
            "basilica_validator_billing_telemetry_collected_total",
            "Total billing telemetry records collected"
        );
        describe_counter!(
            "basilica_validator_billing_telemetry_sent_total",
            "Total billing telemetry records successfully sent"
        );
        describe_counter!(
            "basilica_validator_billing_telemetry_dropped_total",
            "Total billing telemetry records dropped"
        );
        describe_gauge!(
            "basilica_validator_billing_queue_depth",
            "Current depth of billing telemetry queue"
        );
        describe_gauge!(
            "basilica_validator_billing_channel_utilization_percent",
            "Channel buffer utilization percentage"
        );
        describe_gauge!(
            "basilica_validator_billing_circuit_breaker_state",
            "Circuit breaker state (0=closed, 1=half_open, 2=open)"
        );
        describe_histogram!(
            "basilica_validator_billing_retry_attempts",
            "Number of retry attempts before success or final failure"
        );
        describe_histogram!(
            "basilica_validator_billing_telemetry_latency_seconds",
            "End-to-end latency from collection to successful send"
        );

        // Collateral metrics
        describe_gauge!(
            "basilica_validator_collateral_alpha_usd_price",
            "Current Alpha/USD price used for collateral evaluation"
        );
        describe_gauge!(
            "basilica_validator_collateral_node_status",
            "Node collateral status (-1=unknown,0=excluded,1=undercollateralized,2=warning,3=sufficient)"
        );
        describe_counter!(
            "basilica_validator_collateral_slash_triggered_total",
            "Total collateral slash triggers (shadow + real)"
        );
        describe_counter!(
            "basilica_validator_collateral_slash_shadow_total",
            "Total collateral slashes logged in shadow mode"
        );
        describe_counter!(
            "basilica_validator_collateral_slash_executed_total",
            "Total collateral slashes executed on-chain"
        );

        Ok(Self {
            last_collection: Arc::new(RwLock::new(SystemTime::now())),
            persistence,
        })
    }

    /// Record validation operation
    pub fn record_validation(
        &self,
        _node_id: &str,
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

    /// Set node health status
    pub fn set_node_health(&self, _node_id: &str, healthy: bool) {
        gauge!("basilica_validator_node_health_status").set(if healthy { 1.0 } else { 0.0 });
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

    /// Record GPU profile metrics for a miner
    pub fn record_miner_gpu_count_and_score(
        &self,
        miner_uid: u16,
        gpu_count: u32,
        weighted_score: f64,
    ) {
        gauge!("basilica_validator_miner_gpu_count", "miner_uid" => miner_uid.to_string())
            .set(gpu_count as f64);
        histogram!("basilica_validator_miner_gpu_weighted_score", "miner_uid" => miner_uid.to_string())
            .record(weighted_score);
    }

    /// Record GPU count for an node
    pub fn record_node_gpu_count(
        &self,
        miner_uid: u16,
        node_id: &str,
        gpu_model: &str,
        gpu_count: usize,
    ) {
        gauge!("basilica_validator_node_gpu_count",
            "miner_uid" => miner_uid.to_string(),
            "node_id" => node_id.to_string(),
            "gpu_model" => gpu_model.to_string()
        )
        .set(gpu_count as f64);
    }

    /// Record weight assigned to a miner
    pub fn record_miner_weight(&self, miner_uid: u16, weight: u16) {
        gauge!("basilica_validator_miner_weight",
            "miner_uid" => miner_uid.to_string()
        )
        .set(weight as f64);
    }

    /// Record successful validation for a miner
    pub fn record_miner_successful_validation(&self, miner_uid: u16, node_id: &str) {
        counter!("basilica_validator_miner_successful_validations",
            "miner_uid" => miner_uid.to_string(),
            "node_id" => node_id.to_string()
        )
        .increment(1);
    }

    /// Record GPU profile for a miner
    pub fn record_miner_gpu_profile(
        &self,
        miner_uid: u16,
        gpu_profile: &str,
        node_id: &str,
        count: u32,
    ) {
        gauge!("basilica_validator_miner_gpu_profiles",
            "miner_uid" => miner_uid.to_string(),
            "gpu_profile" => gpu_profile.to_string(),
            "node_id" => node_id.to_string()
        )
        .set(count as f64);
    }

    /// Record node uptime metrics for ramp-up tracking
    pub fn record_node_uptime_metrics(
        &self,
        miner_uid: u16,
        node_id: &str,
        uptime_minutes: f64,
        multiplier: f64,
    ) {
        gauge!("basilica_node_uptime_minutes",
            "node_id" => node_id.to_string(),
            "miner_uid" => miner_uid.to_string()
        )
        .set(uptime_minutes);

        gauge!("basilica_node_uptime_multiplier",
            "node_id" => node_id.to_string(),
            "miner_uid" => miner_uid.to_string()
        )
        .set(multiplier);
    }

    /// Reset node uptime metrics for a node that has been removed
    pub fn reset_node_uptime_metrics(&self, miner_uid: u16, node_id: &str) {
        gauge!("basilica_node_uptime_minutes",
            "node_id" => node_id.to_string(),
            "miner_uid" => miner_uid.to_string()
        )
        .set(0.0);

        gauge!("basilica_node_uptime_multiplier",
            "node_id" => node_id.to_string(),
            "miner_uid" => miner_uid.to_string()
        )
        .set(0.0);

        debug!(
            miner_uid = miner_uid,
            node_id = node_id,
            "Reset node uptime metrics after node removal"
        );
    }

    /// Record node rental status
    pub fn record_node_rental_status(
        &self,
        node_id: &str,
        miner_uid: u16,
        gpu_type: &str,
        is_rented: bool,
    ) {
        gauge!("basilica_validator_node_rental_status",
            "node_id" => node_id.to_string(),
            "miner_uid" => miner_uid.to_string(),
            "gpu_type" => gpu_type.to_string()
        )
        .set(if is_rented { 1.0 } else { 0.0 });
    }

    /// Record rental creation
    pub fn record_rental_created(&self, miner_uid: u16, gpu_type: &str) {
        counter!("basilica_validator_rentals_created_total",
            "miner_uid" => miner_uid.to_string(),
            "gpu_type" => gpu_type.to_string()
        )
        .increment(1);
    }

    /// Record node ban expiry timestamp (seconds since epoch, 0 if not banned)
    pub fn record_node_ban_till(
        &self,
        node_id: &str,
        miner_uid: u16,
        ban_expiry: Option<DateTime<Utc>>,
    ) {
        let expiry_ts = ban_expiry.map(|ts| ts.timestamp() as f64).unwrap_or(0.0);

        gauge!("validator_node_ban_till",
            "node_id" => node_id.to_string(),
            "miner_uid" => miner_uid.to_string()
        )
        .set(expiry_ts);
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

    /// Collect GPU metrics from database
    pub async fn collect_gpu_metrics_from_database(&self) {
        let miners = self.persistence.get_all_registered_miners().await.unwrap();

        for miner in miners {
            let miner_uid = miner
                .miner_id
                .strip_prefix("miner_")
                .and_then(|uid_str| uid_str.parse::<u16>().ok())
                .unwrap_or(0);

            let node_gpu_counts = self
                .persistence
                .get_miner_gpu_uuid_assignments(&miner.miner_id)
                .await
                .unwrap();

            debug!(
                "Miner {} (UID: {}) has {} nodes with GPU assignments",
                miner.miner_id,
                miner_uid,
                node_gpu_counts.len()
            );

            // Only set metrics for nodes that have GPU assignments
            for (node_id, gpu_count, gpu_model, gpu_memory_gb) in &node_gpu_counts {
                let node_uuid = node_id.as_str();

                debug!(
                    "Setting node GPU count: miner_uid={}, node_id={}, gpu_model={}, gpu_count={}, gpu_memory_gb={}",
                    miner_uid, node_uuid, gpu_model, gpu_count, gpu_memory_gb
                );

                gauge!("basilica_validator_node_gpu_count",
                    "miner_uid" => miner_uid.to_string(),
                    "node_id" => node_uuid.to_string(),
                    "gpu_model" => gpu_model.to_string()
                )
                .set(*gpu_count as f64);
            }

            let total_count = self
                .persistence
                .get_miner_total_gpu_count_from_assignments(&miner.miner_id)
                .await
                .unwrap();

            debug!(
                "Setting miner total GPU count: miner_uid={}, total_count={}",
                miner_uid, total_count
            );

            gauge!("basilica_validator_miner_gpu_count",
                "miner_uid" => miner_uid.to_string()
            )
            .set(total_count as f64);
        }

        info!("Completed GPU metrics collection from database");
    }

    /// Record RPC critical failure
    pub fn record_rpc_critical_failure(&self, method: &str, error_type: &str) {
        counter!("basilica_validator_rpc_critical_failures_total",
            "method" => method.to_string(),
            "error_type" => error_type.to_string()
        )
        .increment(1);

        error!(
            "RPC critical failure recorded: method={}, error_type={}",
            method, error_type
        );
    }

    /// Set total discovered miners count
    pub fn set_discovered_miners_total(&self, count: u64) {
        gauge!("basilica_validator_discovered_miners_total").set(count as f64);
    }

    /// Sets node validation state atomically, clearing all other states for the validation type
    pub fn set_node_validation_state(
        &self,
        node_id: &str,
        miner_uid: u16,
        validation_type: ValidationType,
        current_state: ValidationState,
        result: StateResult,
    ) {
        let validation_type_str = match validation_type {
            ValidationType::Full => "full",
            ValidationType::Lightweight => "lightweight",
        };

        // Get all possible states for this validation type
        let all_states = ValidationState::states_for_type(validation_type);

        // Set metrics for all states
        for state in all_states {
            let value = if *state == current_state {
                result.to_metric_value()
            } else {
                0.0
            };

            gauge!("basilica_validator_node_validation_state",
                "node_id" => node_id.to_string(),
                "miner_uid" => miner_uid.to_string(),
                "validation_type" => validation_type_str.to_string(),
                "state" => state.as_str().to_string()
            )
            .set(value);
        }
    }

    /// Clears all validation states for an node (sets all to 0.0)
    pub fn clear_node_validation_states(
        &self,
        node_id: &str,
        miner_uid: u16,
        validation_type: ValidationType,
    ) {
        let validation_type_str = match validation_type {
            ValidationType::Full => "full",
            ValidationType::Lightweight => "lightweight",
        };

        let all_states = ValidationState::states_for_type(validation_type);

        for state in all_states {
            gauge!("basilica_validator_node_validation_state",
                "node_id" => node_id.to_string(),
                "miner_uid" => miner_uid.to_string(),
                "validation_type" => validation_type_str.to_string(),
                "state" => state.as_str().to_string()
            )
            .set(0.0);
        }
    }

    pub fn record_billing_telemetry_collected(&self, rental_id: &str) {
        counter!("basilica_validator_billing_telemetry_collected_total",
            "rental_id" => rental_id.to_string()
        )
        .increment(1);
    }

    pub fn record_billing_telemetry_sent(&self, count: usize) {
        counter!("basilica_validator_billing_telemetry_sent_total").increment(count as u64);
    }

    pub fn record_billing_telemetry_dropped(&self, reason: &str, count: usize) {
        counter!("basilica_validator_billing_telemetry_dropped_total",
            "reason" => reason.to_string()
        )
        .increment(count as u64);
    }

    pub fn set_billing_queue_depth(&self, depth: usize) {
        gauge!("basilica_validator_billing_queue_depth").set(depth as f64);
    }

    pub fn set_billing_channel_utilization(&self, utilization_percent: f64) {
        gauge!("basilica_validator_billing_channel_utilization_percent").set(utilization_percent);
    }

    pub fn set_billing_circuit_breaker_state(&self, state: &str) {
        let state_value = match state {
            "closed" => 0.0,
            "half_open" => 1.0,
            "open" => 2.0,
            _ => -1.0,
        };
        gauge!("basilica_validator_billing_circuit_breaker_state").set(state_value);
    }

    pub fn record_billing_retry_attempts(&self, attempts: u32) {
        histogram!("basilica_validator_billing_retry_attempts").record(attempts as f64);
    }

    pub fn record_billing_telemetry_latency(&self, latency: Duration) {
        histogram!("basilica_validator_billing_telemetry_latency_seconds")
            .record(latency.as_secs_f64());
    }

    pub fn record_collateral_price(&self, alpha_usd: f64) {
        gauge!("basilica_validator_collateral_alpha_usd_price").set(alpha_usd);
    }

    pub fn record_collateral_node_status(
        &self,
        hotkey: &str,
        node_id: &str,
        gpu_category: &str,
        status: &str,
    ) {
        // TODO: Consider reducing label cardinality if metrics backend can't handle per-node series.
        let value = match status {
            "sufficient" => 3.0,
            "warning" => 2.0,
            "undercollateralized" => 1.0,
            "excluded" => 0.0,
            "unknown" => -1.0,
            _ => -2.0,
        };
        gauge!(
            "basilica_validator_collateral_node_status",
            "hotkey" => hotkey.to_string(),
            "node_id" => node_id.to_string(),
            "gpu_category" => gpu_category.to_string()
        )
        .set(value);
    }

    pub fn record_collateral_slash_triggered(&self, misbehaviour_type: &str) {
        counter!(
            "basilica_validator_collateral_slash_triggered_total",
            "type" => misbehaviour_type.to_string()
        )
        .increment(1);
    }

    pub fn record_collateral_slash_shadow(&self) {
        counter!("basilica_validator_collateral_slash_shadow_total").increment(1);
    }

    pub fn record_collateral_slash_executed(&self) {
        counter!("basilica_validator_collateral_slash_executed_total").increment(1);
    }
}
