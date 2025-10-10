//! Node-specific metrics for Miner fleet management

use anyhow::Result;
use std::collections::HashMap;
use std::sync::Arc;
use std::time::{Duration, SystemTime};
use tokio::sync::RwLock;
use tracing::{debug, warn};

use crate::metrics::MinerPrometheusMetrics;

/// Node-specific metrics collector for Miner
pub struct MinerNodeMetrics {
    prometheus: Arc<MinerPrometheusMetrics>,
    // Per-node tracking
    node_stats: Arc<RwLock<HashMap<String, NodeStats>>>,
    fleet_health: Arc<RwLock<FleetHealthStats>>,
}

#[derive(Debug, Default, Clone)]
struct NodeStats {
    node_id: String,
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
    _total_nodes: u64,
    _healthy_count: u64,
    _unhealthy_count: u64,
    total_checks_last_hour: u64,
    failed_checks_last_hour: u64,
    average_fleet_response_time: Duration,
    last_health_sweep: Option<SystemTime>,
}

impl MinerNodeMetrics {
    /// Create new node metrics tracker
    pub fn new(prometheus: Arc<MinerPrometheusMetrics>) -> Result<Self> {
        Ok(Self {
            prometheus,
            node_stats: Arc::new(RwLock::new(HashMap::new())),
            fleet_health: Arc::new(RwLock::new(FleetHealthStats::default())),
        })
    }

    /// Record node health check with detailed tracking
    pub async fn record_node_health_check(
        &self,
        node_id: &str,
        check_success: bool,
        node_healthy: bool,
        response_time: Duration,
        check_details: &NodeHealthCheckDetails,
    ) {
        // Record in Prometheus
        self.prometheus.record_node_health_check(
            node_id,
            check_success,
            response_time,
            node_healthy,
        );

        // Update per-node stats
        self.update_node_health_stats(node_id, check_success, node_healthy, response_time)
            .await;

        // Update fleet health
        self.update_fleet_health_stats(check_success, response_time)
            .await;

        debug!(
            "Recorded detailed node health check: node={}, success={}, healthy={}, response_time={:?}, details={:?}",
            node_id, check_success, node_healthy, response_time, check_details
        );
    }

    /// Record node deployment with comprehensive tracking
    pub async fn record_node_deployment(
        &self,
        node_id: &str,
        deployment_details: &NodeDeploymentDetails,
        success: bool,
        duration: Duration,
    ) {
        // Record in Prometheus
        self.prometheus.record_deployment(
            node_id,
            success,
            duration,
            &deployment_details.deployment_type,
        );

        // Update per-node deployment stats
        self.update_node_deployment_stats(node_id, success, duration, deployment_details.is_remote)
            .await;

        // Update remote nodes count if applicable
        if deployment_details.is_remote && success {
            let deployed_count = self.count_remote_deployed_nodes().await;
            self.prometheus.set_remote_nodes_deployed(deployed_count);
        }

        debug!(
            "Recorded node deployment: node={}, type={}, success={}, duration={:?}, remote={}",
            node_id,
            deployment_details.deployment_type,
            success,
            duration,
            deployment_details.is_remote
        );
    }

    /// Track SSH session for node
    pub async fn track_node_ssh_session(
        &self,
        node_id: &str,
        validator_hotkey: &str,
        operation: &str, // "start", "end"
        session_duration: Option<Duration>,
    ) {
        match operation {
            "start" => {
                self.prometheus
                    .record_ssh_session_created(node_id, validator_hotkey);
                self.increment_node_ssh_sessions(node_id).await;
            }
            "end" => {
                if let Some(duration) = session_duration {
                    self.prometheus
                        .record_ssh_session_closed(node_id, validator_hotkey, duration);
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

    /// Update node availability status
    pub async fn update_node_availability(&self, node_id: &str, available: bool, reason: &str) {
        // Update node stats
        {
            let mut stats = self.node_stats.write().await;
            stats
                .entry(node_id.to_string())
                .or_insert_with(|| NodeStats {
                    node_id: node_id.to_string(),
                    ..Default::default()
                })
                .is_healthy = available;
        }

        // Update fleet health counts
        let (total, healthy, unhealthy) = self.calculate_fleet_health().await;
        self.prometheus
            .update_node_counts(total, healthy, unhealthy);

        debug!(
            "Updated node availability: node={}, available={}, reason={}",
            node_id, available, reason
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
            .update_node_counts(total, healthy, unhealthy);

        let sweep_duration = start_time.elapsed();

        FleetHealthSweepResult {
            total_nodes: total,
            healthy_nodes: healthy,
            unhealthy_nodes: unhealthy,
            average_response_time,
            sweep_duration,
        }
    }

    /// Get node performance summary
    pub async fn get_node_performance_summary(
        &self,
        node_id: &str,
    ) -> Option<NodePerformanceSummary> {
        let stats = self.node_stats.read().await;

        stats.get(node_id).map(|node_stats| NodePerformanceSummary {
            node_id: node_stats.node_id.clone(),
            is_healthy: node_stats.is_healthy,
            health_check_success_rate: if node_stats.total_health_checks > 0 {
                (node_stats.total_health_checks - node_stats.failed_health_checks) as f64
                    / node_stats.total_health_checks as f64
            } else {
                0.0
            },
            average_response_time: node_stats.average_response_time,
            deployment_success_rate: if node_stats.total_deployments > 0 {
                node_stats.successful_deployments as f64 / node_stats.total_deployments as f64
            } else {
                0.0
            },
            is_remote: node_stats.is_remote,
            ssh_sessions_count: node_stats.ssh_sessions_count,
            last_health_check: node_stats.last_health_check,
            last_deployment: node_stats.last_deployment,
        })
    }

    /// Get fleet health overview
    pub async fn get_fleet_health_overview(&self) -> FleetHealthOverview {
        let fleet_health = self.fleet_health.read().await;
        let (total, healthy, unhealthy) = self.calculate_fleet_health().await;

        FleetHealthOverview {
            total_nodes: total,
            healthy_nodes: healthy,
            unhealthy_nodes: unhealthy,
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

    async fn update_node_health_stats(
        &self,
        node_id: &str,
        check_success: bool,
        node_healthy: bool,
        response_time: Duration,
    ) {
        let mut stats = self.node_stats.write().await;

        let node_stats = stats
            .entry(node_id.to_string())
            .or_insert_with(|| NodeStats {
                node_id: node_id.to_string(),
                ..Default::default()
            });

        node_stats.total_health_checks += 1;
        node_stats.is_healthy = node_healthy;
        node_stats.last_health_check = Some(SystemTime::now());

        if !check_success {
            node_stats.failed_health_checks += 1;
        }

        // Update running average for response time
        let total_checks = node_stats.total_health_checks;
        node_stats.average_response_time = Duration::from_secs_f64(
            (node_stats.average_response_time.as_secs_f64() * (total_checks - 1) as f64
                + response_time.as_secs_f64())
                / total_checks as f64,
        );
    }

    async fn update_node_deployment_stats(
        &self,
        node_id: &str,
        success: bool,
        _duration: Duration,
        is_remote: bool,
    ) {
        let mut stats = self.node_stats.write().await;

        let node_stats = stats
            .entry(node_id.to_string())
            .or_insert_with(|| NodeStats {
                node_id: node_id.to_string(),
                ..Default::default()
            });

        node_stats.total_deployments += 1;
        node_stats.is_remote = is_remote;
        node_stats.last_deployment = Some(SystemTime::now());

        if success {
            node_stats.successful_deployments += 1;
        }
    }

    async fn increment_node_ssh_sessions(&self, node_id: &str) {
        let mut stats = self.node_stats.write().await;

        stats
            .entry(node_id.to_string())
            .or_insert_with(|| NodeStats {
                node_id: node_id.to_string(),
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
        let stats = self.node_stats.read().await;

        let total = stats.len() as u64;
        let healthy = stats.values().filter(|s| s.is_healthy).count() as u64;
        let unhealthy = total - healthy;

        (total, healthy, unhealthy)
    }

    async fn calculate_average_response_time(&self) -> Duration {
        let stats = self.node_stats.read().await;

        if stats.is_empty() {
            return Duration::from_secs(0);
        }

        let total_response_time: f64 = stats
            .values()
            .map(|s| s.average_response_time.as_secs_f64())
            .sum();

        Duration::from_secs_f64(total_response_time / stats.len() as f64)
    }

    async fn count_remote_deployed_nodes(&self) -> u64 {
        let stats = self.node_stats.read().await;
        stats
            .values()
            .filter(|s| s.is_remote && s.successful_deployments > 0)
            .count() as u64
    }

    async fn count_active_ssh_sessions(&self) -> u64 {
        let stats = self.node_stats.read().await;
        stats.values().map(|s| s.ssh_sessions_count).sum()
    }
}

/// Details for node health check
#[derive(Debug, Clone)]
pub struct NodeHealthCheckDetails {
    pub check_type: String, // "grpc", "http", "ping"
    pub endpoint: String,
    pub timeout: Duration,
    pub grpc_status: Option<String>,
    pub error_message: Option<String>,
}

/// Details for node deployment
#[derive(Debug, Clone)]
pub struct NodeDeploymentDetails {
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
    pub total_nodes: u64,
    pub healthy_nodes: u64,
    pub unhealthy_nodes: u64,
    pub average_response_time: Duration,
    pub sweep_duration: Duration,
}

/// Performance summary for individual node
#[derive(Debug, Clone)]
pub struct NodePerformanceSummary {
    pub node_id: String,
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
    pub total_nodes: u64,
    pub healthy_nodes: u64,
    pub unhealthy_nodes: u64,
    pub health_percentage: f64,
    pub average_response_time: Duration,
    pub last_health_sweep: Option<SystemTime>,
    pub checks_last_hour: u64,
    pub failed_checks_last_hour: u64,
}
