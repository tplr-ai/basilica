//! Business-specific metrics for Validator operations

use anyhow::Result;
use async_trait::async_trait;
use std::collections::HashMap;
use std::sync::Arc;
use std::time::Duration;
use tokio::sync::RwLock;
use tracing::{debug, warn};

use crate::metrics::ValidatorPrometheusMetrics;
use basilica_common::metrics::traits::BasilcaMetrics;

/// Business metrics specific to Validator operations
pub struct ValidatorBusinessMetrics {
    prometheus: Arc<ValidatorPrometheusMetrics>,
    // Cache for aggregated business metrics
    verification_stats: Arc<RwLock<VerificationStats>>,
    node_stats: Arc<RwLock<HashMap<String, NodeStats>>>,
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
struct NodeStats {
    node_id: String,
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
            node_stats: Arc::new(RwLock::new(HashMap::new())),
            consensus_stats: Arc::new(RwLock::new(ConsensusStats::default())),
        })
    }

    /// Record complete validation workflow
    pub async fn record_validation_workflow(
        &self,
        node_id: &str,
        success: bool,
        duration: Duration,
        score: Option<f64>,
        validation_type: &str,
    ) {
        // Record in Prometheus
        self.prometheus
            .record_validation(node_id, success, duration, score);

        // Update aggregated stats
        self.update_verification_stats(success, duration, score)
            .await;
        self.update_node_stats(node_id, success, score).await;

        debug!(
            "Recorded validation workflow: node={}, type={}, success={}, duration={:?}, score={:?}",
            node_id, validation_type, success, duration, score
        );
    }

    /// Record GPU profile validation with actual metrics
    pub async fn record_gpu_profile_validation(
        &self,
        miner_uid: u16,
        node_id: &str,
        gpu_model: &str,
        gpu_count: usize,
        success: bool,
        weighted_score: f64,
    ) {
        // Record node GPU count in Prometheus
        self.prometheus
            .record_node_gpu_count(miner_uid, node_id, gpu_model, gpu_count);

        // Record miner GPU metrics if validation successful
        if success {
            // Get total GPU count for miner from aggregated stats
            let total_gpu_count = {
                let stats = self.node_stats.read().await;
                let mut miner_total = 0u32;
                for (exec_id, _exec_stats) in stats.iter() {
                    if exec_id.contains(&miner_uid.to_string()) {
                        // In real implementation, would track GPU counts per node
                        miner_total += gpu_count as u32;
                    }
                }
                miner_total.max(gpu_count as u32)
            };

            // Record miner GPU profile metrics
            self.prometheus.record_miner_gpu_count_and_score(
                miner_uid,
                total_gpu_count,
                weighted_score,
            );
        }
    }

    /// Record SSH validation session
    pub async fn record_ssh_validation_session(
        &self,
        node_id: &str,
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
                "SSH validation session failed: node={}, host={}, duration={:?}",
                node_id, host, duration
            );
        }
    }

    /// Record attestation verification
    pub async fn record_attestation_verification(
        &self,
        node_id: &str,
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
                "Attestation verification issues: node={}, type={}, success={}, sig_valid={}, hw_verified={}",
                node_id, attestation_type, success, signature_valid, hardware_verified
            );
        }

        debug!(
            "Attestation verification: node={}, type={}, success={}, sig_valid={}, hw_verified={}",
            node_id, attestation_type, success, signature_valid, hardware_verified
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

    /// Update node health status
    pub async fn update_node_health(&self, node_id: &str, healthy: bool) {
        // Record health status
        self.prometheus.set_node_health(node_id, healthy);

        // Update node stats
        {
            let mut stats = self.node_stats.write().await;
            stats
                .entry(node_id.to_string())
                .or_insert_with(|| NodeStats {
                    node_id: node_id.to_string(),
                    ..Default::default()
                })
                .current_health_status = healthy;
        }

        debug!("Updated node health: {}={}", node_id, healthy);
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

    /// Get node performance summary
    pub async fn get_node_summary(&self, node_id: &str) -> Option<NodeSummary> {
        let stats = self.node_stats.read().await;

        stats.get(node_id).map(|node_stats| NodeSummary {
            node_id: node_stats.node_id.clone(),
            total_validations: node_stats.total_validations,
            success_rate: if node_stats.total_validations > 0 {
                node_stats.successful_validations as f64 / node_stats.total_validations as f64
            } else {
                0.0
            },
            average_score: node_stats.average_score,
            current_health: node_stats.current_health_status,
            last_validation: node_stats.last_validation,
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

    async fn update_node_stats(&self, node_id: &str, success: bool, score: Option<f64>) {
        let mut stats = self.node_stats.write().await;

        let node_stats = stats
            .entry(node_id.to_string())
            .or_insert_with(|| NodeStats {
                node_id: node_id.to_string(),
                ..Default::default()
            });

        node_stats.total_validations += 1;
        if success {
            node_stats.successful_validations += 1;
        }

        node_stats.last_validation = Some(std::time::SystemTime::now());

        if let Some(score_value) = score {
            // Update running average
            let total = node_stats.total_validations;
            node_stats.average_score =
                (node_stats.average_score * (total - 1) as f64 + score_value) / total as f64;
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

/// Summary of node performance
#[derive(Debug, Clone)]
pub struct NodeSummary {
    pub node_id: String,
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
        let node_id = labels
            .iter()
            .find(|(k, _)| *k == "node_id")
            .map(|(_, v)| *v)
            .unwrap_or("unknown");

        self.record_validation_workflow(node_id, success, duration, None, task_type)
            .await;
    }

    /// Record verification attempt
    async fn record_verification_attempt(
        &self,
        node_id: &str,
        verification_type: &str,
        success: bool,
        score: Option<f64>,
    ) {
        self.record_validation_workflow(
            node_id,
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
                let node_id = "unknown"; // Would need to be passed in labels
                self.record_ssh_validation_session(
                    node_id,
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

    /// Record node health status
    async fn record_node_health(&self, node_id: &str, healthy: bool) {
        self.update_node_health(node_id, healthy).await;
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
