//! Business-specific metrics for Validator operations

use anyhow::Result;
use async_trait::async_trait;
use std::collections::HashMap;
use std::sync::Arc;
use std::time::Duration;
use tokio::sync::RwLock;
use tracing::{debug, warn};

use crate::metrics::ValidatorPrometheusMetrics;
use common::metrics::traits::BasilcaMetrics;

/// Business metrics specific to Validator operations
pub struct ValidatorBusinessMetrics {
    prometheus: Arc<ValidatorPrometheusMetrics>,
    // Cache for aggregated business metrics
    verification_stats: Arc<RwLock<VerificationStats>>,
    executor_stats: Arc<RwLock<HashMap<String, ExecutorStats>>>,
    consensus_stats: Arc<RwLock<ConsensusStats>>,
}

#[derive(Debug, Default, Clone)]
struct VerificationStats {
    total_verifications: u64,
    successful_verifications: u64,
    failed_verifications: u64,
    average_score: f64,
    total_duration: Duration,
}

#[derive(Debug, Default, Clone)]
struct ExecutorStats {
    executor_id: String,
    total_validations: u64,
    successful_validations: u64,
    current_health_status: bool,
    average_score: f64,
    last_validation: Option<std::time::SystemTime>,
}

#[derive(Debug, Default, Clone)]
struct ConsensusStats {
    total_weight_sets: u64,
    successful_weight_sets: u64,
    last_weight_set: Option<std::time::SystemTime>,
    current_stake: u64,
}

impl ValidatorBusinessMetrics {
    /// Create new business metrics tracker
    pub fn new(prometheus: Arc<ValidatorPrometheusMetrics>) -> Result<Self> {
        Ok(Self {
            prometheus,
            verification_stats: Arc::new(RwLock::new(VerificationStats::default())),
            executor_stats: Arc::new(RwLock::new(HashMap::new())),
            consensus_stats: Arc::new(RwLock::new(ConsensusStats::default())),
        })
    }

    /// Record complete validation workflow
    pub async fn record_validation_workflow(
        &self,
        executor_id: &str,
        success: bool,
        duration: Duration,
        score: Option<f64>,
        validation_type: &str,
    ) {
        // Record in Prometheus
        self.prometheus
            .record_validation(executor_id, success, duration, score);

        // Update aggregated stats
        self.update_verification_stats(success, duration, score)
            .await;
        self.update_executor_stats(executor_id, success, score)
            .await;

        debug!(
            "Recorded validation workflow: executor={}, type={}, success={}, duration={:?}, score={:?}",
            executor_id, validation_type, success, duration, score
        );
    }

    /// Record SSH validation session
    pub async fn record_ssh_validation_session(
        &self,
        executor_id: &str,
        host: &str,
        success: bool,
        duration: Duration,
        session_type: &str,
    ) {
        // Record SSH connection metrics
        self.prometheus
            .record_ssh_connection(host, success, duration);

        // Record verification session metrics
        self.prometheus
            .record_verification_session(session_type, duration);

        if !success {
            warn!(
                "SSH validation session failed: executor={}, host={}, duration={:?}",
                executor_id, host, duration
            );
        }
    }

    /// Record attestation verification
    pub async fn record_attestation_verification(
        &self,
        executor_id: &str,
        attestation_type: &str,
        success: bool,
        signature_valid: bool,
        hardware_verified: bool,
    ) {
        // Record attestation verification
        self.prometheus
            .record_attestation_verification(success, attestation_type);

        // Record detailed metrics with labels
        let overall_success = success && signature_valid && hardware_verified;

        if !overall_success {
            warn!(
                "Attestation verification issues: executor={}, type={}, success={}, sig_valid={}, hw_verified={}",
                executor_id, attestation_type, success, signature_valid, hardware_verified
            );
        }

        debug!(
            "Attestation verification: executor={}, type={}, success={}, sig_valid={}, hw_verified={}",
            executor_id, attestation_type, success, signature_valid, hardware_verified
        );
    }

    /// Record consensus weight setting operation
    pub async fn record_consensus_operation(
        &self,
        success: bool,
        weight_count: usize,
        total_stake: u64,
        operation_duration: Duration,
    ) {
        // Record consensus metrics
        self.prometheus.record_consensus_weight_set(success);

        // Update consensus stats
        {
            let mut stats = self.consensus_stats.write().await;
            stats.total_weight_sets += 1;
            if success {
                stats.successful_weight_sets += 1;
                stats.last_weight_set = Some(std::time::SystemTime::now());
                stats.current_stake = total_stake;
            }
        }

        debug!(
            "Consensus operation: success={}, weights={}, stake={}, duration={:?}",
            success, weight_count, total_stake, operation_duration
        );
    }

    /// Update executor health status
    pub async fn update_executor_health(&self, executor_id: &str, healthy: bool) {
        // Record health status
        self.prometheus.set_executor_health(executor_id, healthy);

        // Update executor stats
        {
            let mut stats = self.executor_stats.write().await;
            stats
                .entry(executor_id.to_string())
                .or_insert_with(|| ExecutorStats {
                    executor_id: executor_id.to_string(),
                    ..Default::default()
                })
                .current_health_status = healthy;
        }

        debug!("Updated executor health: {}={}", executor_id, healthy);
    }

    /// Get verification statistics summary
    pub async fn get_verification_summary(&self) -> VerificationSummary {
        let stats = self.verification_stats.read().await;

        VerificationSummary {
            total_verifications: stats.total_verifications,
            success_rate: if stats.total_verifications > 0 {
                stats.successful_verifications as f64 / stats.total_verifications as f64
            } else {
                0.0
            },
            average_score: stats.average_score,
            average_duration: if stats.total_verifications > 0 {
                stats.total_duration / stats.total_verifications as u32
            } else {
                Duration::from_secs(0)
            },
        }
    }

    /// Get executor performance summary
    pub async fn get_executor_summary(&self, executor_id: &str) -> Option<ExecutorSummary> {
        let stats = self.executor_stats.read().await;

        stats
            .get(executor_id)
            .map(|executor_stats| ExecutorSummary {
                executor_id: executor_stats.executor_id.clone(),
                total_validations: executor_stats.total_validations,
                success_rate: if executor_stats.total_validations > 0 {
                    executor_stats.successful_validations as f64
                        / executor_stats.total_validations as f64
                } else {
                    0.0
                },
                average_score: executor_stats.average_score,
                current_health: executor_stats.current_health_status,
                last_validation: executor_stats.last_validation,
            })
    }

    /// Get consensus summary
    pub async fn get_consensus_summary(&self) -> ConsensusSummary {
        let stats = self.consensus_stats.read().await;

        ConsensusSummary {
            total_weight_sets: stats.total_weight_sets,
            success_rate: if stats.total_weight_sets > 0 {
                stats.successful_weight_sets as f64 / stats.total_weight_sets as f64
            } else {
                0.0
            },
            current_stake: stats.current_stake,
            last_weight_set: stats.last_weight_set,
        }
    }

    async fn update_verification_stats(
        &self,
        success: bool,
        duration: Duration,
        score: Option<f64>,
    ) {
        let mut stats = self.verification_stats.write().await;

        stats.total_verifications += 1;
        if success {
            stats.successful_verifications += 1;
        } else {
            stats.failed_verifications += 1;
        }

        stats.total_duration += duration;

        if let Some(score_value) = score {
            // Update running average
            let total_with_scores = stats.total_verifications;
            stats.average_score = (stats.average_score * (total_with_scores - 1) as f64
                + score_value)
                / total_with_scores as f64;
        }
    }

    async fn update_executor_stats(&self, executor_id: &str, success: bool, score: Option<f64>) {
        let mut stats = self.executor_stats.write().await;

        let executor_stats =
            stats
                .entry(executor_id.to_string())
                .or_insert_with(|| ExecutorStats {
                    executor_id: executor_id.to_string(),
                    ..Default::default()
                });

        executor_stats.total_validations += 1;
        if success {
            executor_stats.successful_validations += 1;
        }

        executor_stats.last_validation = Some(std::time::SystemTime::now());

        if let Some(score_value) = score {
            // Update running average
            let total = executor_stats.total_validations;
            executor_stats.average_score =
                (executor_stats.average_score * (total - 1) as f64 + score_value) / total as f64;
        }
    }
}

/// Summary of verification operations
#[derive(Debug, Clone)]
pub struct VerificationSummary {
    pub total_verifications: u64,
    pub success_rate: f64,
    pub average_score: f64,
    pub average_duration: Duration,
}

/// Summary of executor performance
#[derive(Debug, Clone)]
pub struct ExecutorSummary {
    pub executor_id: String,
    pub total_validations: u64,
    pub success_rate: f64,
    pub average_score: f64,
    pub current_health: bool,
    pub last_validation: Option<std::time::SystemTime>,
}

/// Summary of consensus operations
#[derive(Debug, Clone)]
pub struct ConsensusSummary {
    pub total_weight_sets: u64,
    pub success_rate: f64,
    pub current_stake: u64,
    pub last_weight_set: Option<std::time::SystemTime>,
}

#[async_trait]
impl BasilcaMetrics for ValidatorBusinessMetrics {
    /// Record task execution metrics
    async fn record_task_execution(
        &self,
        task_type: &str,
        duration: Duration,
        success: bool,
        labels: &[(&str, &str)],
    ) {
        // Map to validation workflow for validators
        let executor_id = labels
            .iter()
            .find(|(k, _)| *k == "executor_id")
            .map(|(_, v)| *v)
            .unwrap_or("unknown");

        self.record_validation_workflow(executor_id, success, duration, None, task_type)
            .await;
    }

    /// Record verification attempt
    async fn record_verification_attempt(
        &self,
        executor_id: &str,
        verification_type: &str,
        success: bool,
        score: Option<f64>,
    ) {
        self.record_validation_workflow(
            executor_id,
            success,
            Duration::from_millis(0), // Duration not tracked here
            score,
            verification_type,
        )
        .await;
    }

    /// Record mining operation (not applicable for validator, log for awareness)
    async fn record_mining_operation(
        &self,
        operation: &str,
        miner_hotkey: &str,
        success: bool,
        duration: Duration,
    ) {
        debug!(
            "Mining operation recorded in validator: operation={}, miner={}, success={}, duration={:?}",
            operation, miner_hotkey, success, duration
        );
        // Validators don't perform mining operations directly, but may track miner interactions
    }

    /// Record validator operation
    async fn record_validator_operation(
        &self,
        operation: &str,
        validator_hotkey: &str,
        success: bool,
        duration: Duration,
    ) {
        debug!(
            "Validator operation: operation={}, validator={}, success={}, duration={:?}",
            operation, validator_hotkey, success, duration
        );

        // Map common validator operations to existing metrics
        match operation {
            "consensus_weight_set" | "set_weights" => {
                self.record_consensus_operation(success, 0, 0, duration)
                    .await;
            }
            "ssh_validation" => {
                let executor_id = "unknown"; // Would need to be passed in labels
                self.record_ssh_validation_session(
                    executor_id,
                    "unknown_host",
                    success,
                    duration,
                    "validator_operation",
                )
                .await;
            }
            _ => {
                // Generic operation tracking could be added to Prometheus metrics
                debug!("Generic validator operation: {}", operation);
            }
        }
    }

    /// Record executor health status
    async fn record_executor_health(&self, executor_id: &str, healthy: bool) {
        self.update_executor_health(executor_id, healthy).await;
    }

    /// Record network consensus metrics
    async fn record_consensus_metrics(&self, weights_set: bool, stake_amount: u64) {
        self.record_consensus_operation(
            weights_set,
            0, // weight_count not available here
            stake_amount,
            Duration::from_millis(0), // duration not tracked here
        )
        .await;
    }
}
