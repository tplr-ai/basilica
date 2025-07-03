//! Executor-specific metrics for Miner fleet management

use anyhow::Result;
use std::collections::HashMap;
use std::sync::Arc;
use std::time::{Duration, SystemTime};
use tokio::sync::RwLock;
use tracing::{debug, warn};

use crate::metrics::MinerPrometheusMetrics;

/// Executor-specific metrics collector for Miner
pub struct MinerExecutorMetrics {
    prometheus: Arc<MinerPrometheusMetrics>,
    // Per-executor tracking
    executor_stats: Arc<RwLock<HashMap<String, ExecutorStats>>>,
    fleet_health: Arc<RwLock<FleetHealthStats>>,
}

#[derive(Debug, Default, Clone)]
struct ExecutorStats {
    executor_id: String,
    is_healthy: bool,
    total_health_checks: u64,
    failed_health_checks: u64,
    last_health_check: Option<SystemTime>,
    average_response_time: Duration,
    total_deployments: u64,
    successful_deployments: u64,
    is_remote: bool,
    ssh_sessions_count: u64,
    last_deployment: Option<SystemTime>,
}

#[derive(Debug, Default, Clone)]
struct FleetHealthStats {
    _total_executors: u64,
    _healthy_count: u64,
    _unhealthy_count: u64,
    total_checks_last_hour: u64,
    failed_checks_last_hour: u64,
    average_fleet_response_time: Duration,
    last_health_sweep: Option<SystemTime>,
}

impl MinerExecutorMetrics {
    /// Create new executor metrics tracker
    pub fn new(prometheus: Arc<MinerPrometheusMetrics>) -> Result<Self> {
        Ok(Self {
            prometheus,
            executor_stats: Arc::new(RwLock::new(HashMap::new())),
            fleet_health: Arc::new(RwLock::new(FleetHealthStats::default())),
        })
    }

    /// Record executor health check with detailed tracking
    pub async fn record_executor_health_check(
        &self,
        executor_id: &str,
        check_success: bool,
        executor_healthy: bool,
        response_time: Duration,
        check_details: &ExecutorHealthCheckDetails,
    ) {
        // Record in Prometheus
        self.prometheus.record_executor_health_check(
            executor_id,
            check_success,
            response_time,
            executor_healthy,
        );

        // Update per-executor stats
        self.update_executor_health_stats(
            executor_id,
            check_success,
            executor_healthy,
            response_time,
        )
        .await;

        // Update fleet health
        self.update_fleet_health_stats(check_success, response_time)
            .await;

        debug!(
            "Recorded detailed executor health check: executor={}, success={}, healthy={}, response_time={:?}, details={:?}",
            executor_id, check_success, executor_healthy, response_time, check_details
        );
    }

    /// Record executor deployment with comprehensive tracking
    pub async fn record_executor_deployment(
        &self,
        executor_id: &str,
        deployment_details: &ExecutorDeploymentDetails,
        success: bool,
        duration: Duration,
    ) {
        // Record in Prometheus
        self.prometheus.record_deployment(
            executor_id,
            success,
            duration,
            &deployment_details.deployment_type,
        );

        // Update per-executor deployment stats
        self.update_executor_deployment_stats(
            executor_id,
            success,
            duration,
            deployment_details.is_remote,
        )
        .await;

        // Update remote executors count if applicable
        if deployment_details.is_remote && success {
            let deployed_count = self.count_remote_deployed_executors().await;
            self.prometheus
                .set_remote_executors_deployed(deployed_count);
        }

        debug!(
            "Recorded executor deployment: executor={}, type={}, success={}, duration={:?}, remote={}",
            executor_id, deployment_details.deployment_type, success, duration, deployment_details.is_remote
        );
    }

    /// Track SSH session for executor
    pub async fn track_executor_ssh_session(
        &self,
        executor_id: &str,
        validator_hotkey: &str,
        operation: &str, // "start", "end"
        session_duration: Option<Duration>,
    ) {
        match operation {
            "start" => {
                self.prometheus
                    .record_ssh_session_created(executor_id, validator_hotkey);
                self.increment_executor_ssh_sessions(executor_id).await;
            }
            "end" => {
                if let Some(duration) = session_duration {
                    self.prometheus.record_ssh_session_closed(
                        executor_id,
                        validator_hotkey,
                        duration,
                    );
                }
            }
            _ => {
                warn!("Unknown SSH operation: {}", operation);
            }
        }

        // Update active SSH sessions count
        let active_count = self.count_active_ssh_sessions().await;
        self.prometheus.set_active_ssh_sessions(active_count);
    }

    /// Update executor availability status
    pub async fn update_executor_availability(
        &self,
        executor_id: &str,
        available: bool,
        reason: &str,
    ) {
        // Update executor stats
        {
            let mut stats = self.executor_stats.write().await;
            stats
                .entry(executor_id.to_string())
                .or_insert_with(|| ExecutorStats {
                    executor_id: executor_id.to_string(),
                    ..Default::default()
                })
                .is_healthy = available;
        }

        // Update fleet health counts
        let (total, healthy, unhealthy) = self.calculate_fleet_health().await;
        self.prometheus
            .update_executor_counts(total, healthy, unhealthy);

        debug!(
            "Updated executor availability: executor={}, available={}, reason={}",
            executor_id, available, reason
        );
    }

    /// Perform fleet health sweep
    pub async fn perform_fleet_health_sweep(&self) -> FleetHealthSweepResult {
        let start_time = std::time::Instant::now();

        // Update fleet health timestamp
        {
            let mut fleet_health = self.fleet_health.write().await;
            fleet_health.last_health_sweep = Some(SystemTime::now());
        }

        // Calculate current fleet health
        let (total, healthy, unhealthy) = self.calculate_fleet_health().await;
        let average_response_time = self.calculate_average_response_time().await;

        // Update Prometheus metrics
        self.prometheus
            .update_executor_counts(total, healthy, unhealthy);

        let sweep_duration = start_time.elapsed();

        FleetHealthSweepResult {
            total_executors: total,
            healthy_executors: healthy,
            unhealthy_executors: unhealthy,
            average_response_time,
            sweep_duration,
        }
    }

    /// Get executor performance summary
    pub async fn get_executor_performance_summary(
        &self,
        executor_id: &str,
    ) -> Option<ExecutorPerformanceSummary> {
        let stats = self.executor_stats.read().await;

        stats
            .get(executor_id)
            .map(|executor_stats| ExecutorPerformanceSummary {
                executor_id: executor_stats.executor_id.clone(),
                is_healthy: executor_stats.is_healthy,
                health_check_success_rate: if executor_stats.total_health_checks > 0 {
                    (executor_stats.total_health_checks - executor_stats.failed_health_checks)
                        as f64
                        / executor_stats.total_health_checks as f64
                } else {
                    0.0
                },
                average_response_time: executor_stats.average_response_time,
                deployment_success_rate: if executor_stats.total_deployments > 0 {
                    executor_stats.successful_deployments as f64
                        / executor_stats.total_deployments as f64
                } else {
                    0.0
                },
                is_remote: executor_stats.is_remote,
                ssh_sessions_count: executor_stats.ssh_sessions_count,
                last_health_check: executor_stats.last_health_check,
                last_deployment: executor_stats.last_deployment,
            })
    }

    /// Get fleet health overview
    pub async fn get_fleet_health_overview(&self) -> FleetHealthOverview {
        let fleet_health = self.fleet_health.read().await;
        let (total, healthy, unhealthy) = self.calculate_fleet_health().await;

        FleetHealthOverview {
            total_executors: total,
            healthy_executors: healthy,
            unhealthy_executors: unhealthy,
            health_percentage: if total > 0 {
                healthy as f64 / total as f64 * 100.0
            } else {
                0.0
            },
            average_response_time: fleet_health.average_fleet_response_time,
            last_health_sweep: fleet_health.last_health_sweep,
            checks_last_hour: fleet_health.total_checks_last_hour,
            failed_checks_last_hour: fleet_health.failed_checks_last_hour,
        }
    }

    async fn update_executor_health_stats(
        &self,
        executor_id: &str,
        check_success: bool,
        executor_healthy: bool,
        response_time: Duration,
    ) {
        let mut stats = self.executor_stats.write().await;

        let executor_stats =
            stats
                .entry(executor_id.to_string())
                .or_insert_with(|| ExecutorStats {
                    executor_id: executor_id.to_string(),
                    ..Default::default()
                });

        executor_stats.total_health_checks += 1;
        executor_stats.is_healthy = executor_healthy;
        executor_stats.last_health_check = Some(SystemTime::now());

        if !check_success {
            executor_stats.failed_health_checks += 1;
        }

        // Update running average for response time
        let total_checks = executor_stats.total_health_checks;
        executor_stats.average_response_time = Duration::from_secs_f64(
            (executor_stats.average_response_time.as_secs_f64() * (total_checks - 1) as f64
                + response_time.as_secs_f64())
                / total_checks as f64,
        );
    }

    async fn update_executor_deployment_stats(
        &self,
        executor_id: &str,
        success: bool,
        _duration: Duration,
        is_remote: bool,
    ) {
        let mut stats = self.executor_stats.write().await;

        let executor_stats =
            stats
                .entry(executor_id.to_string())
                .or_insert_with(|| ExecutorStats {
                    executor_id: executor_id.to_string(),
                    ..Default::default()
                });

        executor_stats.total_deployments += 1;
        executor_stats.is_remote = is_remote;
        executor_stats.last_deployment = Some(SystemTime::now());

        if success {
            executor_stats.successful_deployments += 1;
        }
    }

    async fn increment_executor_ssh_sessions(&self, executor_id: &str) {
        let mut stats = self.executor_stats.write().await;

        stats
            .entry(executor_id.to_string())
            .or_insert_with(|| ExecutorStats {
                executor_id: executor_id.to_string(),
                ..Default::default()
            })
            .ssh_sessions_count += 1;
    }

    async fn update_fleet_health_stats(&self, check_success: bool, response_time: Duration) {
        let mut fleet_health = self.fleet_health.write().await;

        fleet_health.total_checks_last_hour += 1;
        if !check_success {
            fleet_health.failed_checks_last_hour += 1;
        }

        // Update average response time
        let total_checks = fleet_health.total_checks_last_hour;
        fleet_health.average_fleet_response_time = Duration::from_secs_f64(
            (fleet_health.average_fleet_response_time.as_secs_f64() * (total_checks - 1) as f64
                + response_time.as_secs_f64())
                / total_checks as f64,
        );
    }

    async fn calculate_fleet_health(&self) -> (u64, u64, u64) {
        let stats = self.executor_stats.read().await;

        let total = stats.len() as u64;
        let healthy = stats.values().filter(|s| s.is_healthy).count() as u64;
        let unhealthy = total - healthy;

        (total, healthy, unhealthy)
    }

    async fn calculate_average_response_time(&self) -> Duration {
        let stats = self.executor_stats.read().await;

        if stats.is_empty() {
            return Duration::from_secs(0);
        }

        let total_response_time: f64 = stats
            .values()
            .map(|s| s.average_response_time.as_secs_f64())
            .sum();

        Duration::from_secs_f64(total_response_time / stats.len() as f64)
    }

    async fn count_remote_deployed_executors(&self) -> u64 {
        let stats = self.executor_stats.read().await;
        stats
            .values()
            .filter(|s| s.is_remote && s.successful_deployments > 0)
            .count() as u64
    }

    async fn count_active_ssh_sessions(&self) -> u64 {
        let stats = self.executor_stats.read().await;
        stats.values().map(|s| s.ssh_sessions_count).sum()
    }
}

/// Details for executor health check
#[derive(Debug, Clone)]
pub struct ExecutorHealthCheckDetails {
    pub check_type: String, // "grpc", "http", "ping"
    pub endpoint: String,
    pub timeout: Duration,
    pub grpc_status: Option<String>,
    pub error_message: Option<String>,
}

/// Details for executor deployment
#[derive(Debug, Clone)]
pub struct ExecutorDeploymentDetails {
    pub deployment_type: String, // "remote", "local", "docker"
    pub is_remote: bool,
    pub target_host: Option<String>,
    pub binary_path: String,
    pub config_template: Option<String>,
    pub systemd_service: bool,
}

/// Result of fleet health sweep
#[derive(Debug, Clone)]
pub struct FleetHealthSweepResult {
    pub total_executors: u64,
    pub healthy_executors: u64,
    pub unhealthy_executors: u64,
    pub average_response_time: Duration,
    pub sweep_duration: Duration,
}

/// Performance summary for individual executor
#[derive(Debug, Clone)]
pub struct ExecutorPerformanceSummary {
    pub executor_id: String,
    pub is_healthy: bool,
    pub health_check_success_rate: f64,
    pub average_response_time: Duration,
    pub deployment_success_rate: f64,
    pub is_remote: bool,
    pub ssh_sessions_count: u64,
    pub last_health_check: Option<SystemTime>,
    pub last_deployment: Option<SystemTime>,
}

/// Fleet health overview
#[derive(Debug, Clone)]
pub struct FleetHealthOverview {
    pub total_executors: u64,
    pub healthy_executors: u64,
    pub unhealthy_executors: u64,
    pub health_percentage: f64,
    pub average_response_time: Duration,
    pub last_health_sweep: Option<SystemTime>,
    pub checks_last_hour: u64,
    pub failed_checks_last_hour: u64,
}
