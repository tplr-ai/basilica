//! Business-specific metrics for Miner operations

use anyhow::Result;
use async_trait::async_trait;
use std::collections::HashMap;
use std::sync::Arc;
use std::time::Duration;
use tokio::sync::RwLock;
use tracing::{debug, warn};

use crate::metrics::MinerPrometheusMetrics;
use common::metrics::traits::BasilcaMetrics;

/// Business metrics specific to Miner operations
pub struct MinerBusinessMetrics {
    prometheus: Arc<MinerPrometheusMetrics>,
    // Cache for aggregated business metrics
    executor_fleet_stats: Arc<RwLock<ExecutorFleetStats>>,
    validator_interaction_stats: Arc<RwLock<HashMap<String, ValidatorInteractionStats>>>,
    deployment_stats: Arc<RwLock<DeploymentStats>>,
}

#[derive(Debug, Default, Clone)]
struct ExecutorFleetStats {
    total_executors: u64,
    healthy_executors: u64,
    total_health_checks: u64,
    failed_health_checks: u64,
    average_health_check_duration: Duration,
}

#[derive(Debug, Default, Clone)]
struct ValidatorInteractionStats {
    validator_hotkey: String,
    total_requests: u64,
    successful_requests: u64,
    auth_failures: u64,
    active_sessions: u64,
    total_executor_discoveries: u64,
    last_interaction: Option<std::time::SystemTime>,
}

#[derive(Debug, Default, Clone)]
struct DeploymentStats {
    total_deployments: u64,
    successful_deployments: u64,
    failed_deployments: u64,
    remote_executors_deployed: u64,
    ssh_sessions_created: u64,
    ssh_sessions_active: u64,
}

impl MinerBusinessMetrics {
    /// Create new business metrics tracker
    pub fn new(prometheus: Arc<MinerPrometheusMetrics>) -> Result<Self> {
        Ok(Self {
            prometheus,
            executor_fleet_stats: Arc::new(RwLock::new(ExecutorFleetStats::default())),
            validator_interaction_stats: Arc::new(RwLock::new(HashMap::new())),
            deployment_stats: Arc::new(RwLock::new(DeploymentStats::default())),
        })
    }

    /// Record complete executor health check workflow
    pub async fn record_executor_health_workflow(
        &self,
        executor_id: &str,
        check_success: bool,
        executor_healthy: bool,
        duration: Duration,
        check_type: &str,
    ) {
        // Record in Prometheus
        self.prometheus.record_executor_health_check(
            executor_id,
            check_success,
            duration,
            executor_healthy,
        );

        // Update aggregated stats
        self.update_executor_fleet_stats(check_success, executor_healthy, duration)
            .await;

        debug!(
            "Recorded executor health workflow: executor={}, check_type={}, check_success={}, healthy={}, duration={:?}",
            executor_id, check_type, check_success, executor_healthy, duration
        );
    }

    /// Record validator interaction session
    pub async fn record_validator_interaction(
        &self,
        validator_hotkey: &str,
        request_type: &str,
        success: bool,
        duration: Duration,
        session_duration: Option<Duration>,
    ) {
        // Record in Prometheus
        self.prometheus
            .record_validator_request(validator_hotkey, request_type, success, duration);

        if let Some(session_dur) = session_duration {
            self.prometheus
                .record_validator_session(validator_hotkey, session_dur);
        }

        // Update interaction stats
        self.update_validator_interaction_stats(validator_hotkey, success, request_type == "auth")
            .await;

        debug!(
            "Recorded validator interaction: validator={}, type={}, success={}, duration={:?}",
            validator_hotkey, request_type, success, duration
        );
    }

    /// Record SSH session management
    pub async fn record_ssh_session_management(
        &self,
        executor_id: &str,
        validator_hotkey: &str,
        operation: &str, // "create", "close", "key_deployment"
        success: bool,
        duration: Option<Duration>,
    ) {
        match operation {
            "create" => {
                self.prometheus
                    .record_ssh_session_created(executor_id, validator_hotkey);
                self.update_deployment_stats_ssh_created().await;
            }
            "close" => {
                if let Some(dur) = duration {
                    self.prometheus
                        .record_ssh_session_closed(executor_id, validator_hotkey, dur);
                }
            }
            "key_deployment" => {
                self.prometheus
                    .record_ssh_key_deployment(executor_id, success, "deployment");
            }
            _ => {
                warn!("Unknown SSH operation: {}", operation);
            }
        }

        debug!(
            "Recorded SSH session management: executor={}, validator={}, operation={}, success={}",
            executor_id, validator_hotkey, operation, success
        );
    }

    /// Record executor deployment operation
    pub async fn record_executor_deployment(
        &self,
        executor_id: &str,
        deployment_type: &str, // "remote", "local", "docker"
        success: bool,
        duration: Duration,
        is_remote: bool,
    ) {
        // Record in Prometheus
        self.prometheus
            .record_deployment(executor_id, success, duration, deployment_type);

        // Update deployment stats
        self.update_deployment_stats(success, is_remote).await;

        debug!(
            "Recorded executor deployment: executor={}, type={}, success={}, duration={:?}, remote={}",
            executor_id, deployment_type, success, duration, is_remote
        );
    }

    /// Record executor discovery for validator
    pub async fn record_executor_discovery(
        &self,
        validator_hotkey: &str,
        executors_returned: u32,
        discovery_type: &str,
    ) {
        // Record in Prometheus
        self.prometheus
            .record_executor_discovery(validator_hotkey, executors_returned);

        // Update validator interaction stats
        {
            let mut stats = self.validator_interaction_stats.write().await;
            stats
                .entry(validator_hotkey.to_string())
                .or_insert_with(|| ValidatorInteractionStats {
                    validator_hotkey: validator_hotkey.to_string(),
                    ..Default::default()
                })
                .total_executor_discoveries += 1;
        }

        debug!(
            "Recorded executor discovery: validator={}, type={}, executors_returned={}",
            validator_hotkey, discovery_type, executors_returned
        );
    }

    /// Update executor fleet status
    pub async fn update_executor_fleet_status(&self, total: u64, healthy: u64, unhealthy: u64) {
        // Record in Prometheus
        self.prometheus
            .update_executor_counts(total, healthy, unhealthy);

        // Update fleet stats
        {
            let mut stats = self.executor_fleet_stats.write().await;
            stats.total_executors = total;
            stats.healthy_executors = healthy;
        }

        debug!(
            "Updated executor fleet status: total={}, healthy={}, unhealthy={}",
            total, healthy, unhealthy
        );
    }

    /// Update active session counts
    pub async fn update_active_sessions(&self, validator_sessions: u64, ssh_sessions: u64) {
        // Record in Prometheus
        self.prometheus
            .set_active_validator_sessions(validator_sessions);
        self.prometheus.set_active_ssh_sessions(ssh_sessions);

        // Update deployment stats
        {
            let mut stats = self.deployment_stats.write().await;
            stats.ssh_sessions_active = ssh_sessions;
        }
    }

    /// Get executor fleet summary
    pub async fn get_executor_fleet_summary(&self) -> ExecutorFleetSummary {
        let stats = self.executor_fleet_stats.read().await;

        ExecutorFleetSummary {
            total_executors: stats.total_executors,
            healthy_executors: stats.healthy_executors,
            health_rate: if stats.total_health_checks > 0 {
                (stats.total_health_checks - stats.failed_health_checks) as f64
                    / stats.total_health_checks as f64
            } else {
                0.0
            },
            average_health_check_duration: stats.average_health_check_duration,
        }
    }

    /// Get validator interaction summary
    pub async fn get_validator_interaction_summary(
        &self,
        validator_hotkey: &str,
    ) -> Option<ValidatorInteractionSummary> {
        let stats = self.validator_interaction_stats.read().await;

        stats
            .get(validator_hotkey)
            .map(|validator_stats| ValidatorInteractionSummary {
                validator_hotkey: validator_stats.validator_hotkey.clone(),
                total_requests: validator_stats.total_requests,
                success_rate: if validator_stats.total_requests > 0 {
                    validator_stats.successful_requests as f64
                        / validator_stats.total_requests as f64
                } else {
                    0.0
                },
                auth_failure_rate: if validator_stats.total_requests > 0 {
                    validator_stats.auth_failures as f64 / validator_stats.total_requests as f64
                } else {
                    0.0
                },
                active_sessions: validator_stats.active_sessions,
                total_executor_discoveries: validator_stats.total_executor_discoveries,
                last_interaction: validator_stats.last_interaction,
            })
    }

    /// Get deployment summary
    pub async fn get_deployment_summary(&self) -> DeploymentSummary {
        let stats = self.deployment_stats.read().await;

        DeploymentSummary {
            total_deployments: stats.total_deployments,
            success_rate: if stats.total_deployments > 0 {
                stats.successful_deployments as f64 / stats.total_deployments as f64
            } else {
                0.0
            },
            remote_executors_deployed: stats.remote_executors_deployed,
            ssh_sessions_created: stats.ssh_sessions_created,
            ssh_sessions_active: stats.ssh_sessions_active,
        }
    }

    async fn update_executor_fleet_stats(
        &self,
        check_success: bool,
        _executor_healthy: bool,
        duration: Duration,
    ) {
        let mut stats = self.executor_fleet_stats.write().await;

        stats.total_health_checks += 1;
        if !check_success {
            stats.failed_health_checks += 1;
        }

        // Update running average for health check duration
        let total_checks = stats.total_health_checks;
        stats.average_health_check_duration = Duration::from_secs_f64(
            (stats.average_health_check_duration.as_secs_f64() * (total_checks - 1) as f64
                + duration.as_secs_f64())
                / total_checks as f64,
        );
    }

    async fn update_validator_interaction_stats(
        &self,
        validator_hotkey: &str,
        success: bool,
        is_auth_failure: bool,
    ) {
        let mut stats = self.validator_interaction_stats.write().await;

        let validator_stats = stats
            .entry(validator_hotkey.to_string())
            .or_insert_with(|| ValidatorInteractionStats {
                validator_hotkey: validator_hotkey.to_string(),
                ..Default::default()
            });

        validator_stats.total_requests += 1;
        if success {
            validator_stats.successful_requests += 1;
        }
        if is_auth_failure {
            validator_stats.auth_failures += 1;
        }

        validator_stats.last_interaction = Some(std::time::SystemTime::now());
    }

    async fn update_deployment_stats(&self, success: bool, is_remote: bool) {
        let mut stats = self.deployment_stats.write().await;

        stats.total_deployments += 1;
        if success {
            stats.successful_deployments += 1;
            if is_remote {
                stats.remote_executors_deployed += 1;
            }
        } else {
            stats.failed_deployments += 1;
        }
    }

    async fn update_deployment_stats_ssh_created(&self) {
        let mut stats = self.deployment_stats.write().await;
        stats.ssh_sessions_created += 1;
    }
}

/// Summary of executor fleet operations
#[derive(Debug, Clone)]
pub struct ExecutorFleetSummary {
    pub total_executors: u64,
    pub healthy_executors: u64,
    pub health_rate: f64,
    pub average_health_check_duration: Duration,
}

/// Summary of validator interactions
#[derive(Debug, Clone)]
pub struct ValidatorInteractionSummary {
    pub validator_hotkey: String,
    pub total_requests: u64,
    pub success_rate: f64,
    pub auth_failure_rate: f64,
    pub active_sessions: u64,
    pub total_executor_discoveries: u64,
    pub last_interaction: Option<std::time::SystemTime>,
}

/// Summary of deployment operations
#[derive(Debug, Clone)]
pub struct DeploymentSummary {
    pub total_deployments: u64,
    pub success_rate: f64,
    pub remote_executors_deployed: u64,
    pub ssh_sessions_created: u64,
    pub ssh_sessions_active: u64,
}

#[async_trait]
impl BasilcaMetrics for MinerBusinessMetrics {
    /// Record task execution metrics
    async fn record_task_execution(
        &self,
        task_type: &str,
        duration: Duration,
        success: bool,
        labels: &[(&str, &str)],
    ) {
        // Map to deployment operations for miners
        let executor_id = labels
            .iter()
            .find(|(k, _)| *k == "executor_id")
            .map(|(_, v)| *v)
            .unwrap_or("unknown");

        let is_remote = labels
            .iter()
            .any(|(k, v)| *k == "deployment_type" && *v == "remote");

        self.record_executor_deployment(executor_id, task_type, success, duration, is_remote)
            .await;
    }

    /// Record verification attempt (miners track as validator interactions)
    async fn record_verification_attempt(
        &self,
        executor_id: &str,
        verification_type: &str,
        success: bool,
        score: Option<f64>,
    ) {
        debug!(
            "Verification attempt recorded in miner: executor={}, type={}, success={}, score={:?}",
            executor_id, verification_type, success, score
        );
        // Miners don't perform verifications directly, but can track related metrics
    }

    /// Record mining operation
    async fn record_mining_operation(
        &self,
        operation: &str,
        miner_hotkey: &str,
        success: bool,
        duration: Duration,
    ) {
        debug!(
            "Mining operation: operation={}, miner={}, success={}, duration={:?}",
            operation, miner_hotkey, success, duration
        );

        // Map common mining operations to existing metrics
        match operation {
            "executor_discovery" => {
                // Record as executor discovery without specific validator
                self.record_executor_discovery("unknown_validator", 1, operation)
                    .await;
            }
            "fleet_health_check" => {
                let executor_id = "fleet"; // Generic identifier for fleet operations
                self.record_executor_health_workflow(
                    executor_id,
                    success,
                    success, // Assume healthy if operation succeeded
                    duration,
                    operation,
                )
                .await;
            }
            "ssh_management" => {
                self.record_ssh_session_management(
                    "unknown_executor",
                    "unknown_validator",
                    "management",
                    success,
                    Some(duration),
                )
                .await;
            }
            _ => {
                debug!("Generic mining operation: {}", operation);
            }
        }
    }

    /// Record validator operation (track as validator interactions)
    async fn record_validator_operation(
        &self,
        operation: &str,
        validator_hotkey: &str,
        success: bool,
        duration: Duration,
    ) {
        self.record_validator_interaction(
            validator_hotkey,
            operation,
            success,
            duration,
            None, // Session duration not tracked here
        )
        .await;
    }

    /// Record executor health status
    async fn record_executor_health(&self, executor_id: &str, healthy: bool) {
        self.record_executor_health_workflow(
            executor_id,
            true, // Health check succeeded if we got a result
            healthy,
            Duration::from_millis(0), // Duration not tracked here
            "health_status_update",
        )
        .await;
    }

    /// Record network consensus metrics (not directly applicable for miners)
    async fn record_consensus_metrics(&self, weights_set: bool, stake_amount: u64) {
        debug!(
            "Consensus metrics recorded in miner: weights_set={}, stake={}",
            weights_set, stake_amount
        );
        // Miners don't set consensus weights but may track related interactions
    }
}
