//! # Verification Engine
//!
//! Handles the actual verification of miners and their nodes.
//! Implements Single Responsibility Principle by focusing only on verification logic.

use super::types::MinerInfo;
use super::types::{NodeInfoDetailed, NodeVerificationResult, ValidationType};
use super::validation_states::{StateResult, ValidationState};
use super::validation_strategy::{ValidationNode, ValidationStrategy, ValidationStrategySelector};
use super::validation_worker::{ValidationWorkerQueue, WorkerQueueConfig};
use crate::agent_installer::{build_install_commands, build_uninstall_commands, K3sAgentConfig};
use crate::ban_system::BanManager;
use crate::config::VerificationConfig;
use crate::gpu::MinerGpuProfile;
use crate::k8s_profile_publisher::NodeProfilePublisher;
use crate::metrics::ValidatorMetrics;
use crate::node_profile::{labels_from_validation, to_node_profile_spec, NodeProfileInput};
use crate::persistence::{
    entities::{MisbehaviourType, VerificationLog},
    gpu_profile_repository::GpuProfileRepository,
    SimplePersistence,
};
use crate::ssh::{ValidatorSshClient, ValidatorSshKeyManager};
use anyhow::Result;
use basilica_common::identity::{Hotkey, MinerUid, NodeId};
use basilica_common::types::GpuCategory;
use chrono::Utc;
use futures::future::join_all;
use serde_json;
use std::collections::HashMap;
use std::path::PathBuf;
use std::str::FromStr;
use std::sync::Arc;
use std::time::Duration;
use tracing::{debug, error, info, warn};

fn normalize_gpu_category(input: &str) -> String {
    match GpuCategory::from_str(input) {
        Ok(category) => category.to_string(),
        Err(infallible) => match infallible {},
    }
}

struct GpuDeclarationMismatchContext {
    miner_uid: u16,
    node_id: String,
    mismatch_reason: String,
    declared_gpu_category: Option<String>,
    declared_gpu_count: Option<u32>,
    detected_gpu_category: Option<String>,
    detected_gpu_count: u32,
}

#[derive(Clone)]
pub struct VerificationEngine {
    config: VerificationConfig,
    validator_hotkey: Hotkey,
    /// Database persistence for storing verification results
    persistence: Arc<SimplePersistence>,
    /// Whether to use dynamic discovery or fall back to static config
    use_dynamic_discovery: bool,
    /// SSH key path for node access (fallback)
    ssh_key_path: Option<PathBuf>,
    /// Optional Bittensor service for signing
    bittensor_service: Option<Arc<bittensor::Service>>,
    /// SSH key manager for session keys
    ssh_key_manager: Option<Arc<ValidatorSshKeyManager>>,
    /// Metrics for tracking rental status and other validator metrics
    metrics: Option<Arc<ValidatorMetrics>>,
    /// Validation strategy selector for determining validation approach
    validation_strategy_selector: Arc<ValidationStrategySelector>,
    /// Validation node for running validation strategies
    validation_node: Arc<tokio::sync::RwLock<ValidationNode>>,
    /// Optional worker queue for decoupled execution
    worker_queue: Option<Arc<ValidationWorkerQueue>>,
    /// Optional NodeProfile publisher (DI)
    node_profile_publisher: Option<Arc<dyn NodeProfilePublisher + Send + Sync>>,
}

impl VerificationEngine {
    /// Extract miner UID from `miner_###` identifiers
    fn miner_uid_from_miner_id(miner_id: &str) -> Option<u16> {
        miner_id
            .strip_prefix("miner_")
            .and_then(|uid_str| uid_str.parse::<u16>().ok())
    }

    /// Execute complete automated verification workflow with SSH session management (specs-compliant)
    pub async fn execute_verification_workflow(
        &self,
        task: &super::scheduler::VerificationTask,
    ) -> Result<VerificationResult> {
        info!(
            miner_uid = task.miner_uid,
            intended_strategy = ?task.intended_validation_strategy,
            "[EVAL_FLOW] Executing verification workflow for miner {} (intended strategy: {:?})",
            task.miner_uid, task.intended_validation_strategy
        );

        let workflow_start = std::time::Instant::now();
        let mut verification_steps = Vec::new();

        // Step 1: Get nodes from database (populated by miner's RegisterBid call)
        let known_node_data = self
            .persistence
            .get_known_nodes_for_miner(task.miner_uid)
            .await?;
        let node_list = self.convert_db_data_to_node_info(known_node_data, task.miner_uid)?;

        info!(
            miner_uid = task.miner_uid,
            node_count = node_list.len(),
            "[EVAL_FLOW] Found {} registered nodes for miner",
            node_list.len()
        );

        verification_steps.push(VerificationStep {
            step_name: "node_discovery".to_string(),
            status: StepStatus::Completed,
            duration: workflow_start.elapsed(),
            details: format!("Found {} nodes for verification", node_list.len()),
        });

        if node_list.is_empty() {
            info!(
                miner_uid = task.miner_uid,
                intended_strategy = ?task.intended_validation_strategy,
                "[EVAL_FLOW] No nodes found for miner {}", task.miner_uid
            );

            return Ok(VerificationResult {
                miner_uid: task.miner_uid,
                overall_score: 0.0,
                verification_steps,
                completed_at: chrono::Utc::now(),
                error: Some("No nodes found for miner".to_string()),
            });
        }

        // Route to worker queue if enabled
        if let Some(ref worker_queue) = self.worker_queue {
            info!(
                miner_uid = task.miner_uid,
                node_count = node_list.len(),
                intended_strategy = ?task.intended_validation_strategy,
                "[EVAL_FLOW] Routing {} nodes to worker queue for miner {}",
                node_list.len(),
                task.miner_uid
            );
            return self
                .route_to_worker_queue(
                    node_list,
                    task,
                    worker_queue,
                    &mut verification_steps,
                    workflow_start,
                )
                .await;
        }

        // Step 2: Execute SSH-based verification for each node
        let mut node_results = Vec::new();
        let mut nodes_skipped_for_strategy = 0;
        let total_nodes = node_list.len();

        info!(
            miner_uid = task.miner_uid,
            node_count = total_nodes,
            intended_strategy = ?task.intended_validation_strategy,
            "[EVAL_FLOW] Starting nodes verification"
        );

        // Create futures for all node validations
        let validation_futures: Vec<_> = node_list
            .into_iter()
            .map(|node_info| {
                let self_clone = self.clone();
                let miner_endpoint = task.miner_endpoint.clone();
                let miner_uid = task.miner_uid;
                let miner_hotkey = task.miner_hotkey.clone();
                let intended_strategy = task.intended_validation_strategy;

                async move {
                    info!(
                        miner_uid = miner_uid,
                        node_id = %node_info.id,
                        intended_strategy = ?intended_strategy,
                        "[EVAL_FLOW] Starting verification for node"
                    );

                    // Set in-queue state for this specific node being validated
                    if let Some(ref metrics) = self_clone.validation_node.read().await.metrics() {
                        metrics.prometheus().set_node_validation_state(
                            &node_info.id.to_string(),
                            miner_uid,
                            intended_strategy,
                            ValidationState::InQueue,
                            StateResult::Current,
                        );
                    }

                    let result = self_clone
                        .verify_node(
                            &miner_endpoint,
                            &node_info,
                            miner_uid,
                            &miner_hotkey,
                            intended_strategy,
                        )
                        .await;

                    (node_info, result)
                }
            })
            .collect();

        // Execute all validations concurrently
        let validation_results = join_all(validation_futures).await;

        // Process all validation results
        let mut success_count = 0usize;
        let mut failure_count = 0usize;
        for (node_info, result) in validation_results {
            match result {
                Ok(result) => {
                    let score = result.verification_score;
                    info!(
                        miner_uid = task.miner_uid,
                        node_id = %node_info.id,
                        verification_score = score,
                        intended_strategy = ?task.intended_validation_strategy,
                        "[EVAL_FLOW] SSH verification completed"
                    );
                    node_results.push(result);
                    success_count += 1;
                    verification_steps.push(VerificationStep {
                        step_name: format!("ssh_verification_{}", node_info.id),
                        status: StepStatus::Completed,
                        duration: workflow_start.elapsed(),
                        details: format!("SSH verification completed, score: {score}"),
                    });
                }
                Err(e) if e.to_string().contains("Strategy mismatch") => {
                    nodes_skipped_for_strategy += 1;
                    debug!(
                        miner_uid = task.miner_uid,
                        node_id = %node_info.id,
                        pipeline_type = ?task.intended_validation_strategy,
                        intended_strategy = ?task.intended_validation_strategy,
                        "[EVAL_FLOW] Node requires different validation type, will be handled by other pipeline"
                    );
                    verification_steps.push(VerificationStep {
                        step_name: format!("ssh_verification_{}", node_info.id),
                        status: StepStatus::Completed,
                        duration: workflow_start.elapsed(),
                        details: "Skipped - handled by other validation pipeline".to_string(),
                    });
                }
                Err(e) => {
                    error!(
                        miner_uid = task.miner_uid,
                        node_id = %node_info.id,
                        error = %e,
                        intended_strategy = ?task.intended_validation_strategy,
                        "[EVAL_FLOW] verification failed"
                    );
                    failure_count += 1;
                    verification_steps.push(VerificationStep {
                        step_name: format!("ssh_verification_{}", node_info.id),
                        status: StepStatus::Failed,
                        duration: workflow_start.elapsed(),
                        details: format!("SSH verification error: {e}"),
                    });
                }
            }
        }
        info!(
            miner_uid = task.miner_uid,
            intended_strategy = ?task.intended_validation_strategy,
            total_nodes = total_nodes,
            completed = success_count,
            failed = failure_count,
            skipped = nodes_skipped_for_strategy,
            "[EVAL_FLOW] Node validation results collected"
        );

        // Step 3: Calculate overall verification score
        let overall_score = if node_results.is_empty() {
            // Only return 0 if ALL nodes were skipped for strategy mismatch
            // If we have no results and all were skipped, this pipeline isn't responsible for this miner
            if nodes_skipped_for_strategy == total_nodes && total_nodes > 0 {
                debug!(
                    miner_uid = task.miner_uid,
                    intended_strategy = ?task.intended_validation_strategy,
                    skipped_count = nodes_skipped_for_strategy,
                    pipeline_type = ?task.intended_validation_strategy,
                    "[EVAL_FLOW] All nodes require different validation type, score will come from other pipeline"
                );
            }
            0.0
        } else {
            let avg_score = node_results
                .iter()
                .map(|r| r.verification_score)
                .sum::<f64>()
                / node_results.len() as f64;

            info!(
                miner_uid = task.miner_uid,
                intended_strategy = ?task.intended_validation_strategy,
                validated_count = node_results.len(),
                skipped_count = nodes_skipped_for_strategy,
                total_nodes = total_nodes,
                average_score = avg_score,
                pipeline_type = ?task.intended_validation_strategy,
                "[EVAL_FLOW] Validation completed for miner"
            );

            avg_score
        };

        // Step 4: Store individual node verification results
        // Construct MinerInfo from task data
        let hotkey = Hotkey::new(task.miner_hotkey.clone())
            .map_err(|e| anyhow::anyhow!("Invalid miner hotkey '{}': {}", task.miner_hotkey, e))?;

        let miner_info = MinerInfo {
            uid: MinerUid::new(task.miner_uid),
            hotkey,
            endpoint: task.miner_endpoint.clone(),
            is_validator: task.is_validator,
            stake_tao: task.stake_tao,
            last_verified: None,
            verification_score: overall_score,
        };

        for result in &node_results {
            self.store_node_verification_result_with_miner_info(
                task.miner_uid,
                result,
                &miner_info,
            )
            .await?;
        }

        verification_steps.push(VerificationStep {
            step_name: "result_storage".to_string(),
            status: StepStatus::Completed,
            duration: workflow_start.elapsed(),
            details: format!("Stored verification result with score: {overall_score:.2}"),
        });

        info!(
            miner_uid = task.miner_uid,
            intended_strategy = ?task.intended_validation_strategy,
            validated_nodes = node_results.len(),
            skipped_nodes = nodes_skipped_for_strategy,
            total_nodes = total_nodes,
            pipeline_type = ?task.intended_validation_strategy,
            overall_score = overall_score,
            "[EVAL_FLOW] Verification workflow completed for miner {} in {:?}, score: {:.2} ({} of {} nodes validated in {} pipeline)",
            task.miner_uid,
            workflow_start.elapsed(),
            overall_score,
            node_results.len(),
            total_nodes,
            match task.intended_validation_strategy {
                ValidationType::Full => "full",
                ValidationType::Lightweight => "lightweight",
            }
        );

        Ok(VerificationResult {
            miner_uid: task.miner_uid,
            overall_score,
            verification_steps,
            completed_at: chrono::Utc::now(),
            error: None,
        })
    }

    /// Route nodes to worker queue for parallel processing
    async fn route_to_worker_queue(
        &self,
        nodes: Vec<NodeInfoDetailed>,
        task: &super::scheduler::VerificationTask,
        worker_queue: &Arc<ValidationWorkerQueue>,
        verification_steps: &mut Vec<VerificationStep>,
        workflow_start: std::time::Instant,
    ) -> Result<VerificationResult> {
        // Publish all nodes to the appropriate queue
        let mut published_count = 0;
        let mut failed_count = 0;

        for node in nodes {
            match worker_queue.publish(node.clone(), task.clone()).await {
                Ok(_) => {
                    published_count += 1;
                    // Set in-queue state only for nodes successfully published to queue
                    if let Some(ref metrics) = self.validation_node.read().await.metrics() {
                        metrics.prometheus().set_node_validation_state(
                            &node.id.to_string(),
                            task.miner_uid,
                            task.intended_validation_strategy,
                            ValidationState::InQueue,
                            StateResult::Current,
                        );
                    }
                }
                Err(e) => {
                    warn!("Failed to publish node to queue: {}", e);
                    failed_count += 1;
                }
            }
        }

        verification_steps.push(VerificationStep {
            step_name: "queue_dispatch".to_string(),
            status: if failed_count == 0 {
                StepStatus::Completed
            } else {
                StepStatus::PartialSuccess
            },
            duration: workflow_start.elapsed(),
            details: format!(
                "Published {} nodes to queue ({} failed)",
                published_count, failed_count
            ),
        });

        // Return result indicating work was queued
        // Actual scores will be updated asynchronously by workers
        Ok(VerificationResult {
            miner_uid: task.miner_uid,
            overall_score: 0.0,
            verification_steps: verification_steps.clone(),
            completed_at: chrono::Utc::now(),
            error: if published_count == 0 {
                Some("Failed to publish any nodes to queue".to_string())
            } else {
                None
            },
        })
    }

    /// Store node verification result with actual miner information
    pub async fn store_node_verification_result_with_miner_info(
        &self,
        miner_uid: u16,
        node_result: &NodeVerificationResult,
        miner_info: &super::types::MinerInfo,
    ) -> Result<()> {
        info!(
            miner_uid = miner_uid,
            node_id = %node_result.node_id,
            verification_score = node_result.verification_score,
            validation_type = %node_result.validation_type,
            "Storing node verification result to database for miner {}, node {}: score={:.2}",
            miner_uid, node_result.node_id, node_result.verification_score
        );

        // Create verification log entry for database storage
        let success = match node_result.validation_type {
            ValidationType::Lightweight => node_result.ssh_connection_successful,
            ValidationType::Full => {
                node_result.ssh_connection_successful && node_result.binary_validation_successful
            }
        };

        let verification_log = VerificationLog::new(
            node_result.node_id.to_string(),
            self.validator_hotkey.to_string(),
            "ssh_automation".to_string(),
            node_result.verification_score,
            success,
            serde_json::json!({
                "miner_uid": miner_uid,
                "node_id": node_result.node_id.to_string(),
                "ssh_connection_successful": node_result.ssh_connection_successful,
                "binary_validation_successful": node_result.binary_validation_successful,
                "failure_reasons": node_result.failure_reasons,
                "verification_method": "ssh_automation",
                "node_result": node_result.node_result,
                "gpu_count": node_result.gpu_count,
                "score_details": {
                    "verification_score": node_result.verification_score,
                    "ssh_score": if node_result.ssh_connection_successful { 0.5 } else { 0.0 },
                    "binary_score": if node_result.binary_validation_successful { 0.5 } else { 0.0 }
                }
            }),
            node_result.execution_time.as_millis() as i64,
            if !node_result.ssh_connection_successful {
                Some("SSH connection failed".to_string())
            } else if node_result.validation_type == ValidationType::Full
                && !node_result.binary_validation_successful
            {
                Some(if node_result.failure_reasons.is_empty() {
                    "Binary validation failed".to_string()
                } else {
                    node_result.failure_reasons.join("; ")
                })
            } else {
                None
            },
        );

        // Store directly to database to avoid repository trait issues
        let query = r#"
            INSERT INTO verification_logs (
                id, node_id, validator_hotkey, verification_type, timestamp,
                score, success, details, duration_ms, error_message, created_at, updated_at,
                last_binary_validation, last_binary_validation_score
            ) VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?)
        "#;

        let now = chrono::Utc::now().to_rfc3339();
        let success = verification_log.success;

        // Set binary validation timestamp and score if this was a successful binary validation
        let (binary_validation_time, binary_validation_score) =
            if success && node_result.binary_validation_successful {
                (Some(now.clone()), Some(node_result.verification_score))
            } else {
                (None, None)
            };

        if let Err(e) = sqlx::query(query)
            .bind(verification_log.id.to_string())
            .bind(&verification_log.node_id)
            .bind(&verification_log.validator_hotkey)
            .bind(&verification_log.verification_type)
            .bind(verification_log.timestamp.to_rfc3339())
            .bind(verification_log.score)
            .bind(if success { 1 } else { 0 })
            .bind(
                serde_json::to_string(&verification_log.details)
                    .unwrap_or_else(|_| "{}".to_string()),
            )
            .bind(verification_log.duration_ms)
            .bind(&verification_log.error_message)
            .bind(verification_log.created_at.to_rfc3339())
            .bind(verification_log.updated_at.to_rfc3339())
            .bind(binary_validation_time)
            .bind(binary_validation_score)
            .execute(self.persistence.pool())
            .await
        {
            error!("Failed to store verification log: {}", e);
            return Err(anyhow::anyhow!("Database storage failed: {}", e));
        }

        let miner_id = format!("miner_{miner_uid}");
        let status = match (success, &node_result.validation_type) {
            (false, _) => "offline".to_string(),
            (true, ValidationType::Full) => "online".to_string(),
            (true, ValidationType::Lightweight) => {
                match self
                    .persistence
                    .has_active_rental(&node_result.node_id.to_string(), &miner_id)
                    .await
                {
                    Ok(true) => "online".to_string(),
                    _ => sqlx::query_scalar::<_, String>(
                        "SELECT status FROM miner_nodes WHERE miner_id = ? AND node_id = ?",
                    )
                    .bind(&miner_id)
                    .bind(&verification_log.node_id)
                    .fetch_optional(self.persistence.pool())
                    .await
                    .ok()
                    .flatten()
                    .unwrap_or_else(|| "verified".to_string()),
                }
            }
        };

        info!(
            security = true,
            miner_uid = miner_uid,
            node_id = %node_result.node_id,
            validation_type = %node_result.validation_type,
            new_status = %status,
            "Status update based on validation type"
        );

        // Use transaction to ensure atomic updates
        let mut tx = self.persistence.pool().begin().await?;

        // Update node status
        if let Err(e) = sqlx::query(
            "UPDATE miner_nodes
             SET status = ?, last_node_check = datetime('now')
             WHERE node_id = ? AND miner_id = ?",
        )
        .bind(&status)
        .bind(&verification_log.node_id)
        .bind(&miner_id)
        .execute(&mut *tx)
        .await
        {
            warn!("Failed to update node health status: {}", e);
            tx.rollback().await?;
            return Err(anyhow::anyhow!("Failed to update node status: {}", e));
        }

        // escape plan, if verification failed, clean up GPU assignments
        if !(success
            || node_result.validation_type == ValidationType::Lightweight
                && node_result.ssh_connection_successful)
        {
            // Mark NodeProfile health=Invalid (best-effort)
            if let Some(publisher) = &self.node_profile_publisher {
                let ns =
                    std::env::var("BASILICA_NAMESPACE").unwrap_or_else(|_| "default".to_string());
                let node_name = node_result.node_id.to_string();
                let _ = publisher
                    .set_node_profile_health(&ns, &node_name, "Invalid")
                    .await;
            }
            // Best-effort uninstall k3s agent
            let _ = self
                .maybe_uninstall_k3s(miner_uid, &node_result.node_id.to_string())
                .await;
            self.persistence
                .cleanup_gpu_assignments(&verification_log.node_id, &miner_id, Some(&mut tx))
                .await?;
            tx.commit().await?;
            return Ok(());
        }

        tx.commit().await?;

        let gpu_infos = node_result
            .node_result
            .as_ref()
            .map(|er| er.gpu_infos.clone())
            .unwrap_or_default();

        match node_result.validation_type {
            ValidationType::Full => {
                info!(
                    security = true,
                    miner_uid = miner_uid,
                    node_id = %node_result.node_id,
                    validation_type = "full",
                    gpu_count = gpu_infos.len(),
                    action = "processing_full_validation",
                    "Processing full validation for miner {}, node {}",
                    miner_uid, node_result.node_id
                );

                self.persistence
                    .ensure_miner_node_relationship(
                        miner_uid,
                        &node_result.node_id.to_string(),
                        &node_result.node_ssh_endpoint,
                        &node_result.node_ip,
                        miner_info,
                        node_result.hourly_rate_cents,
                    )
                    .await?;

                self.persistence
                    .store_gpu_uuid_assignments(
                        miner_uid,
                        &node_result.node_id.to_string(),
                        &gpu_infos,
                    )
                    .await?;

                // Node pricing is automatically stored via ensure_miner_node_relationship above,
                // which receives hourly_rate_cents from NodeConnectionDetails during discovery

                // Create/update GPU profile for this miner after successful verification
                let gpu_repo = GpuProfileRepository::new(self.persistence.pool().clone());

                // Get actual GPU counts from the just-stored assignments
                let miner_id = format!("miner_{}", miner_uid);
                let gpu_counts = self
                    .persistence
                    .get_miner_gpu_uuid_assignments(&miner_id)
                    .await?;
                let mut gpu_map: HashMap<String, u32> = HashMap::new();
                for (_, count, gpu_name, _) in gpu_counts {
                    let category = GpuCategory::from_str(&gpu_name).unwrap();
                    let model = category.to_string();
                    *gpu_map.entry(model).or_insert(0) += count;
                }

                let existing_count = self
                    .persistence
                    .get_miner_verification_count(&miner_id, 3)
                    .await?;
                let total_verification_count = existing_count + 1;

                let profile = MinerGpuProfile {
                    miner_uid: MinerUid::new(miner_uid),
                    gpu_counts: gpu_map,
                    total_score: node_result.verification_score,
                    verification_count: total_verification_count,
                    last_updated: Utc::now(),
                    last_successful_validation: Some(Utc::now()),
                };

                if let Err(e) = gpu_repo.upsert_gpu_profile(&profile).await {
                    warn!(
                            "Failed to update GPU profile for miner {} after successful verification: {}",
                            miner_uid, e
                        );
                } else {
                    info!(
                        "Successfully updated GPU profile for miner {}: {} GPUs",
                        miner_uid,
                        profile.gpu_counts.values().sum::<u32>()
                    );
                }

                // Set rental metrics for successfully validated nodes
                // Marking them as available (not rented)
                if success {
                    if let Some(ref metrics) = self.metrics {
                        // Extract GPU type from the first GPU found
                        let gpu_type = gpu_infos
                            .first()
                            .map(|gpu| {
                                let category = GpuCategory::from_str(&gpu.gpu_name).unwrap();
                                category.to_string()
                            })
                            .unwrap_or_else(|| "unknown".to_string());

                        metrics.prometheus().record_node_rental_status(
                            &node_result.node_id.to_string(),
                            miner_uid,
                            &gpu_type,
                            false, // is_rented = false (available for rental)
                        );

                        debug!(
                            "Set rental metric for validated node {} (miner_uid: {}, gpu_type: {}, rented: false)",
                            node_result.node_id, miner_uid, gpu_type
                        );
                    }

                    // Publish BasilicaNodeProfile CR and apply Node labels
                    if let Some(ref nr) = node_result.node_result {
                        if let Err(e) = self
                            .publish_node_profile_and_labels(
                                miner_uid,
                                &node_result.node_id.to_string(),
                                nr,
                                &node_result.failure_reasons,
                            )
                            .await
                        {
                            warn!(
                                security = true,
                                miner_uid = miner_uid,
                                node_id = %node_result.node_id,
                                error = %e,
                                "Failed to publish node profile or apply labels (non-fatal)"
                            );
                        }
                    }

                    // Join k3s cluster (optional, gated)
                    self.maybe_join_k3s(miner_uid, &node_result.node_id.to_string())
                        .await;
                }
            }
            ValidationType::Lightweight => {
                info!(
                    security = true,
                    miner_uid = miner_uid,
                    node_id = %node_result.node_id,
                    validation_type = "lightweight",
                    gpu_count = gpu_infos.len(),
                    action = "processing_lightweight_validation",
                    "Processing lightweight validation for miner {}, node {}",
                    miner_uid, node_result.node_id
                );

                self.persistence
                    .update_gpu_assignment_timestamps(
                        miner_uid,
                        &node_result.node_id.to_string(),
                        &gpu_infos,
                    )
                    .await?;
            }
        }

        info!(
            miner_uid = miner_uid,
            node_id = %node_result.node_id,
            verification_score = node_result.verification_score,
            validation_type = %node_result.validation_type,
            "Node verification result successfully stored to database for miner {}, node {}: score={:.2}",
            miner_uid, node_result.node_id, node_result.verification_score
        );

        Ok(())
    }

    async fn publish_node_profile_and_labels(
        &self,
        miner_uid: u16,
        node_id: &str,
        nr: &super::types::NodeResult,
        failure_reasons: &[String],
    ) -> Result<()> {
        let (namespace, cr, maybe_labels) = self
            .prepare_node_profile_cr_and_labels(miner_uid, node_id, nr, failure_reasons)
            .await?;

        if let Some(publisher) = &self.node_profile_publisher {
            publisher.upsert_node_profile(&namespace, &cr).await?;
            if let Some((node_name, labels)) = maybe_labels {
                let _ = publisher.apply_node_labels(&node_name, &labels).await;
            }
        }

        Ok(())
    }

    fn shell_quote_for_bash(s: &str) -> String {
        // Wrap in single quotes and escape existing single quotes
        let escaped = s.replace('\'', "'\\''");
        format!("'{}'", escaped)
    }

    async fn get_node_ssh_details(
        &self,
        miner_uid: u16,
        node_id: &str,
    ) -> Result<basilica_common::ssh::SshConnectionDetails> {
        let miner_id = format!("miner_{}", miner_uid);
        let endpoint = self
            .persistence
            .get_node_ssh_endpoint(node_id, &miner_id)
            .await?
            .ok_or_else(|| anyhow::anyhow!("SSH endpoint not found for node {}", node_id))?;
        let key_manager = self
            .ssh_key_manager
            .as_ref()
            .ok_or_else(|| anyhow::anyhow!("SSH key manager not available"))?;
        key_manager
            .get_ssh_connection_details(&endpoint)
            .map_err(|e| anyhow::anyhow!("{}", e))
    }

    async fn maybe_join_k3s(&self, miner_uid: u16, node_id: &str) {
        if std::env::var("BASILICA_ENABLE_K3S_JOIN").ok().as_deref() != Some("true") {
            return;
        }
        let url = match std::env::var("BASILICA_K3S_URL") {
            Ok(v) if !v.is_empty() => v,
            _ => return,
        };
        let token = match std::env::var("BASILICA_K3S_TOKEN") {
            Ok(v) if !v.is_empty() => v,
            _ => return,
        };
        let channel = std::env::var("BASILICA_K3S_CHANNEL").ok();
        let exclusive = std::env::var("BASILICA_TAINT_EXCLUSIVE")
            .ok()
            .map(|v| v == "1" || v.eq_ignore_ascii_case("true"))
            .unwrap_or(false);

        let mut cfg = K3sAgentConfig::new(url, token);
        cfg.node_name = Some(node_id.to_string());
        cfg.extra_args
            .push("--node-taint basilica.ai/workloads-only=true:NoSchedule".into());
        if exclusive {
            cfg.extra_args
                .push("--node-taint basilica.ai/rental-exclusive=true:NoSchedule".into());
        }
        cfg.channel = channel;

        let cmds = build_install_commands(&cfg);
        let ssh = ValidatorSshClient::new();
        let details = match self.get_node_ssh_details(miner_uid, node_id).await {
            Ok(d) => d,
            Err(_) => return,
        };
        for cmd in cmds {
            let wrapped = format!("bash -lc {}", Self::shell_quote_for_bash(&cmd));
            let _ = ssh.execute_command(&details, &wrapped, false).await;
        }
    }

    async fn maybe_uninstall_k3s(&self, miner_uid: u16, node_id: &str) {
        if std::env::var("BASILICA_ENABLE_K3S_JOIN").ok().as_deref() != Some("true") {
            return;
        }
        let ssh = ValidatorSshClient::new();
        let details = match self.get_node_ssh_details(miner_uid, node_id).await {
            Ok(d) => d,
            Err(_) => return,
        };
        for cmd in build_uninstall_commands() {
            let wrapped = format!("bash -lc {}", Self::shell_quote_for_bash(&cmd));
            let _ = ssh.execute_command(&details, &wrapped, false).await;
        }
    }

    async fn prepare_node_profile_cr_and_labels(
        &self,
        miner_uid: u16,
        node_id: &str,
        nr: &super::types::NodeResult,
        failure_reasons: &[String],
    ) -> Result<(
        String,
        kube::core::DynamicObject,
        Option<(String, std::collections::BTreeMap<String, String>)>,
    )> {
        // Resolve namespace for CR publishing
        let namespace =
            std::env::var("BASILICA_NAMESPACE").unwrap_or_else(|_| "default".to_string());

        // Attempt to derive hostname and region from stored network profile
        let (hostname_opt, region_opt, org_opt) = match self
            .persistence
            .get_node_network_profile(miner_uid, node_id)
            .await?
        {
            Some((
                _full,
                _ip,
                hostname,
                _city,
                region,
                _country,
                _loc,
                organization,
                _postal,
                _tz,
                _ts,
            )) => (hostname, region, organization),
            None => (None, None, None),
        };

        let region = region_opt.as_deref().unwrap_or("unknown");
        let provider = infer_provider_from_org(org_opt.as_deref());

        // Build spec and CR
        let spec = to_node_profile_spec(&NodeProfileInput {
            provider,
            region,
            node_result: nr,
        });
        // Prefer hostname from network profile; fallback to node_id which is used
        // as the k3s node name during join (see maybe_join_k3s).
        let kube_node_name = hostname_opt.as_deref().or(Some(node_id));
        let last_validated = Some(chrono::Utc::now().to_rfc3339());
        let cr = crate::k8s_profile_publisher::K8sNodeProfilePublisher::build_node_profile_cr(
            node_id,
            &namespace,
            &spec,
            kube_node_name,
            last_validated.as_deref(),
            Some("Valid"),
            Some(failure_reasons),
        )?;

        // Base labels from validation
        let node_group = crate::node_profile::assign_node_group(node_id, &self.config.node_groups);
        let mut labels = labels_from_validation(nr, provider, region, Some(node_group));
        // Enrich labels with Docker profile if available
        if let Ok(Some((
            _full_json,
            service_active,
            docker_version,
            _images,
            dind_supported,
            validation_error,
        ))) = self
            .persistence
            .get_node_docker_profile(miner_uid, node_id)
            .await
        {
            labels.insert(
                "basilica.ai/docker-active".into(),
                service_active.to_string(),
            );
            if let Some(ver) = docker_version {
                labels.insert("basilica.ai/docker-version".into(), ver);
            }
            labels.insert("basilica.ai/dind".into(), dind_supported.to_string());
            if let Some(err) = validation_error {
                if !err.is_empty() {
                    labels.insert("basilica.ai/docker-error".into(), err);
                }
            }
        }

        let maybe_labels = kube_node_name.map(|name| (name.to_string(), labels));

        Ok((namespace, cr, maybe_labels))
    }

    /// Sync miners from metagraph to database
    pub async fn sync_miners_from_metagraph(&self, miners: &[MinerInfo]) -> Result<()> {
        self.persistence.sync_miners_from_metagraph(miners).await
    }

    /// Get whether dynamic discovery is enabled
    pub fn use_dynamic_discovery(&self) -> bool {
        self.use_dynamic_discovery
    }

    /// Get SSH key manager reference
    pub fn ssh_key_manager(&self) -> &Option<Arc<ValidatorSshKeyManager>> {
        &self.ssh_key_manager
    }

    /// Get bittensor service reference
    pub fn bittensor_service(&self) -> &Option<Arc<bittensor::Service>> {
        &self.bittensor_service
    }

    /// Get SSH key path reference
    pub fn ssh_key_path(&self) -> &Option<PathBuf> {
        &self.ssh_key_path
    }

    /// Create VerificationEngine with SSH automation components (new preferred method)
    #[allow(clippy::too_many_arguments)]
    pub fn with_ssh_automation(
        config: VerificationConfig,
        validator_hotkey: Hotkey,
        ssh_client: Arc<ValidatorSshClient>,
        persistence: Arc<SimplePersistence>,
        use_dynamic_discovery: bool,
        ssh_key_manager: Option<Arc<ValidatorSshKeyManager>>,
        bittensor_service: Option<Arc<bittensor::Service>>,
        metrics: Option<Arc<ValidatorMetrics>>,
        node_profile_publisher: Option<Arc<dyn NodeProfilePublisher + Send + Sync>>,
    ) -> Result<Self> {
        // Validate required components for dynamic discovery
        if use_dynamic_discovery && ssh_key_manager.is_none() {
            return Err(anyhow::anyhow!(
                "SSH key manager is required when dynamic discovery is enabled"
            ));
        }

        Ok(Self {
            config: config.clone(),
            validator_hotkey,
            persistence: persistence.clone(),
            use_dynamic_discovery,
            ssh_key_path: None, // Not used when SSH key manager is available
            bittensor_service,
            ssh_key_manager: ssh_key_manager.clone(),
            metrics: metrics.clone(),
            validation_strategy_selector: Arc::new(ValidationStrategySelector::new(
                config.clone(),
                persistence.clone(),
            )),
            validation_node: Arc::new(tokio::sync::RwLock::new(ValidationNode::new(
                config.clone(),
                ssh_client.clone(),
                metrics,
                persistence.clone(),
            ))),
            worker_queue: None, // Will be set separately if needed
            node_profile_publisher,
        })
    }

    /// Initialize validation server mode
    pub async fn initialize_validation_server(&mut self) -> Result<()> {
        let Some(ref binary_config) = self.config.binary_validation else {
            info!("Binary validation not configured - skipping validation server initialization");
            return Ok(());
        };
        info!("Initializing validation server mode for VerificationEngine");
        let mut node = self.validation_node.write().await;
        node.initialize_server_mode(binary_config).await?;
        info!("Validation server mode initialized successfully");
        Ok(())
    }

    /// Shutdown validation server cleanly
    pub async fn shutdown_validation_server(&self) -> Result<()> {
        let node = self.validation_node.read().await;
        node.shutdown_server_mode().await
    }

    /// Check if SSH automation is properly configured
    pub fn is_ssh_automation_ready(&self) -> bool {
        if self.use_dynamic_discovery() {
            self.ssh_key_manager().is_some()
        } else {
            // Static configuration requires either key manager or fallback key path
            self.ssh_key_manager().is_some() || self.ssh_key_path().is_some()
        }
    }

    /// Get SSH automation status
    pub fn get_ssh_automation_status(&self) -> SshAutomationStatus {
        SshAutomationStatus {
            dynamic_discovery_enabled: self.use_dynamic_discovery(),
            ssh_key_manager_available: self.ssh_key_manager().is_some(),
            bittensor_service_available: self.bittensor_service().is_some(),
            fallback_key_path: self.ssh_key_path().clone(),
        }
    }

    /// Get configuration summary for debugging
    pub fn get_config_summary(&self) -> String {
        format!(
            "VerificationEngine[dynamic_discovery={}, ssh_key_manager={}, bittensor_service={}, worker_queue={}]",
            self.use_dynamic_discovery(),
            self.ssh_key_manager().is_some(),
            self.bittensor_service().is_some(),
            self.worker_queue.is_some()
        )
    }

    /// Get access to validation metrics
    pub async fn get_metrics(&self) -> Option<Arc<ValidatorMetrics>> {
        self.validation_node.read().await.metrics().clone()
    }

    /// Set worker queue for decoupled execution
    pub fn set_worker_queue(&mut self, queue: Arc<ValidationWorkerQueue>) {
        self.worker_queue = Some(queue);
    }

    /// Check if worker queue is enabled
    pub fn has_worker_queue(&self) -> bool {
        self.worker_queue.is_some()
    }

    /// Initialize and start worker queue
    pub async fn init_worker_queue(&mut self) -> Result<()> {
        let config = WorkerQueueConfig::default();
        let queue = Arc::new(ValidationWorkerQueue::new(config, Arc::new(self.clone())));

        queue.start().await?;
        self.worker_queue = Some(queue);

        info!("Worker queue initialized and started");
        Ok(())
    }

    /// Clean up nodes that have consecutive failed validations
    /// This is called periodically (every 15 minutes) to remove nodes that:
    /// 1. Are offline and still have GPU assignments (immediate cleanup)
    /// 2. Have had 2+ consecutive failed validations with no successes (delete)
    /// 3. Have been offline for 30+ minutes (stale cleanup)
    pub async fn cleanup_failed_nodes_after_failures(
        &self,
        consecutive_failures_threshold: i32,
    ) -> Result<()> {
        let removed_nodes = self
            .persistence
            .cleanup_failed_nodes_after_failures(
                consecutive_failures_threshold,
                self.config.gpu_assignment_cleanup_ttl,
            )
            .await?;

        if let Some(ref metrics) = self.metrics {
            let prometheus = metrics.prometheus();
            for (miner_id, node_id) in removed_nodes {
                if let Some(miner_uid) = Self::miner_uid_from_miner_id(&miner_id) {
                    prometheus.reset_node_uptime_metrics(miner_uid, &node_id);
                } else {
                    debug!(
                        miner_id = %miner_id,
                        node_id = %node_id,
                        "Unable to parse miner UID while resetting uptime metrics after cleanup"
                    );
                }
            }
        }

        Ok(())
    }

    /// Enhanced verify node with SSH automation and binary validation
    pub async fn verify_node(
        &self,
        miner_endpoint: &str,
        node_info: &NodeInfoDetailed,
        miner_uid: u16,
        _miner_hotkey: &str,
        intended_strategy: ValidationType,
    ) -> Result<NodeVerificationResult> {
        info!(
            miner_uid = miner_uid,
            node_id = %node_info.id,
            miner_endpoint = %miner_endpoint,
            "[EVAL_FLOW] Starting node verification"
        );

        // Step 1: Determine validation strategy
        let strategy = match self
            .validation_strategy_selector
            .determine_validation_strategy(&node_info.id.to_string(), miner_uid)
            .await
        {
            Ok(s) => s,
            Err(e) => {
                error!(
                    miner_uid = miner_uid,
                    node_id = %node_info.id,
                    error = %e,
                    "[EVAL_FLOW] Failed to determine validation strategy, defaulting to full"
                );
                super::validation_strategy::ValidationStrategy::Full
            }
        };
        debug!(
            miner_uid = miner_uid,
            node_id = %node_info.id,
            determined_strategy = ?strategy,
            intended_strategy = ?intended_strategy,
            "[EVAL_FLOW] Validation strategy resolved"
        );

        // Strategy filtering: skip if strategy doesn't match pipeline
        let strategy_matches = matches!(
            (&strategy, &intended_strategy),
            (ValidationStrategy::Full, ValidationType::Full)
                | (
                    ValidationStrategy::Lightweight { .. },
                    ValidationType::Lightweight
                )
        );

        if !strategy_matches {
            debug!(
                node_id = %node_info.id,
                determined_strategy = ?strategy,
                intended = ?intended_strategy,
                "[EVAL_FLOW] Strategy mismatch - node needs different validation type"
            );

            return Err(anyhow::anyhow!(
                "Strategy mismatch: node needs different validation type"
            ));
        }

        // Step 2: Direct SSH connections to node

        // Get SSH connection details for direct node connection
        let ssh_details = if let Some(ref key_manager) = self.ssh_key_manager {
            key_manager
                .get_ssh_connection_details(&node_info.node_ssh_endpoint)
                .unwrap()
        } else {
            return Err(anyhow::anyhow!("SSH key manager not available"));
        };

        // Step 3: Execute validation based on strategy
        let mut result = match strategy {
            ValidationStrategy::Lightweight {
                previous_score,
                node_result,
                gpu_count,
                binary_validation_successful,
            } => {
                self.validation_node
                    .read()
                    .await
                    .execute_lightweight_validation(
                        miner_uid,
                        node_info,
                        &ssh_details,
                        previous_score,
                        node_result,
                        gpu_count,
                        binary_validation_successful,
                        &self.validator_hotkey,
                        &self.config,
                    )
                    .await
            }
            ValidationStrategy::Full => {
                self.validation_node
                    .read()
                    .await
                    .execute_full_validation(
                        node_info,
                        &ssh_details,
                        self.config.binary_validation.as_ref(),
                        &self.validator_hotkey,
                        miner_uid,
                    )
                    .await
            }
        }?;

        if result.validation_type == ValidationType::Full {
            self.enforce_declared_gpu_claims_for_full_validation(miner_uid, &mut result)
                .await?;
        }

        Ok(result)
    }

    async fn enforce_declared_gpu_claims_for_full_validation(
        &self,
        miner_uid: u16,
        node_result: &mut NodeVerificationResult,
    ) -> Result<()> {
        if node_result.validation_type != ValidationType::Full {
            return Ok(());
        }

        let Some(validation_result) = node_result.node_result.as_ref() else {
            return Ok(());
        };

        let node_id = node_result.node_id.to_string();
        let miner_id = format!("miner_{miner_uid}");

        let detected_gpu_count = validation_result.gpu_infos.len() as u32;
        let detected_gpu_category = validation_result
            .gpu_infos
            .first()
            .map(|gpu| normalize_gpu_category(&gpu.gpu_name));

        let declared_metadata = self
            .persistence
            .get_node_bid_metadata(&miner_id, &node_id)
            .await?;

        let mut declared_gpu_count: Option<u32> = None;
        let mut declared_gpu_category: Option<String> = None;

        let mismatch_reason = if let Some(metadata) = declared_metadata {
            declared_gpu_count = Some(metadata.gpu_count);

            let declared_category_trimmed = metadata.gpu_category.trim();
            if declared_category_trimmed.is_empty() {
                Some("declared gpu_category is empty".to_string())
            } else {
                let normalized_declared_category =
                    normalize_gpu_category(declared_category_trimmed);
                declared_gpu_category = Some(normalized_declared_category.clone());

                if normalized_declared_category == "OTHER" {
                    Some(format!(
                        "declared gpu_category '{}' is invalid or unsupported",
                        declared_category_trimmed
                    ))
                } else if metadata.gpu_count == 0 {
                    Some("declared gpu_count must be greater than 0".to_string())
                } else if detected_gpu_count != metadata.gpu_count {
                    Some(format!(
                        "detected gpu_count {} does not match declared {}",
                        detected_gpu_count, metadata.gpu_count
                    ))
                } else {
                    match detected_gpu_category.as_deref() {
                        Some(detected_category)
                            if detected_category == normalized_declared_category =>
                        {
                            None
                        }
                        Some(detected_category) => Some(format!(
                            "detected gpu_category '{}' does not match declared '{}'",
                            detected_category, normalized_declared_category
                        )),
                        None => Some(
                            "detected gpu_category is missing from validation result".to_string(),
                        ),
                    }
                }
            }
        } else {
            Some("node missing declared bid metadata in miner_nodes".to_string())
        };

        if let Some(reason) = mismatch_reason {
            self.mark_gpu_declaration_mismatch(
                node_result,
                GpuDeclarationMismatchContext {
                    miner_uid,
                    node_id,
                    mismatch_reason: reason,
                    declared_gpu_category,
                    declared_gpu_count,
                    detected_gpu_category,
                    detected_gpu_count,
                },
            )
            .await;
        }

        Ok(())
    }

    async fn mark_gpu_declaration_mismatch(
        &self,
        node_result: &mut NodeVerificationResult,
        context: GpuDeclarationMismatchContext,
    ) {
        let GpuDeclarationMismatchContext {
            miner_uid,
            node_id,
            mismatch_reason,
            declared_gpu_category,
            declared_gpu_count,
            detected_gpu_category,
            detected_gpu_count,
        } = context;

        node_result.binary_validation_successful = false;
        node_result.verification_score = 0.0;
        node_result.validation_details.binary_score = 0.0;
        node_result.validation_details.combined_score = 0.0;
        let mismatch_reason_tag = MisbehaviourType::GpuDeclarationMismatch.as_str();
        if !node_result
            .failure_reasons
            .iter()
            .any(|reason| reason == mismatch_reason_tag)
        {
            node_result
                .failure_reasons
                .push(mismatch_reason_tag.to_string());
        }
        node_result.error = Some(format!("GPU declaration mismatch: {}", mismatch_reason));

        let details = serde_json::json!({
            "reason": mismatch_reason,
            "node_id": node_id,
            "miner_uid": miner_uid,
            "declared_gpu_category": declared_gpu_category,
            "declared_gpu_count": declared_gpu_count,
            "detected_gpu_count": detected_gpu_count,
            "detected_gpu_category": detected_gpu_category,
        })
        .to_string();

        let ban_manager = BanManager::new(
            self.persistence.clone(),
            self.metrics.as_ref().map(|metrics| metrics.prometheus()),
            None,
            None,
        );

        if let Err(log_error) = ban_manager
            .log_misbehaviour(
                miner_uid,
                &node_id,
                MisbehaviourType::GpuDeclarationMismatch,
                &details,
            )
            .await
        {
            warn!(
                miner_uid = miner_uid,
                node_id = &node_id,
                error = %log_error,
                "Failed to log GPU declaration mismatch misbehaviour"
            );
        }

        warn!(
            security = true,
            miner_uid = miner_uid,
            node_id = &node_id,
            mismatch_reason = &mismatch_reason,
            declared_gpu_count = declared_gpu_count.unwrap_or(0),
            detected_gpu_count = detected_gpu_count,
            detected_gpu_category = detected_gpu_category.as_deref().unwrap_or("unknown"),
            "Full validation failed due to GPU declaration mismatch"
        );
    }

    /// Convert database node data to NodeInfoDetailed
    fn convert_db_data_to_node_info(
        &self,
        db_data: Vec<(String, String, String, i32, String, u32)>,
        miner_uid: u16,
    ) -> Result<Vec<NodeInfoDetailed>> {
        let mut nodes = Vec::new();

        for (node_id, ssh_endpoint, node_ip, gpu_count, status, hourly_rate_cents) in db_data {
            let node_id_parsed = NodeId::from_str(&node_id)
                .map_err(|e| anyhow::anyhow!("Invalid node ID '{}': {}", node_id, e))?;

            nodes.push(NodeInfoDetailed {
                id: node_id_parsed,
                miner_uid: MinerUid::new(miner_uid),
                status,
                capabilities: if gpu_count > 0 {
                    vec!["gpu".to_string()]
                } else {
                    vec![]
                },
                node_ssh_endpoint: ssh_endpoint,
                node_ip,
                hourly_rate_cents,
            });
        }

        Ok(nodes)
    }

    pub fn get_ssh_public_key(&self) -> Option<String> {
        if let Some(ref key_manager) = self.ssh_key_manager {
            key_manager.get_ssh_public_key()
        } else {
            None
        }
    }
}

fn infer_provider_from_org(org: Option<&str>) -> &'static str {
    if let Some(o) = org {
        let low = o.to_ascii_lowercase();
        if low.contains("amazon") || low.contains("aws") {
            return "aws";
        }
        if low.contains("google") || low.contains("gcp") {
            return "gcp";
        }
        if low.contains("microsoft") || low.contains("azure") {
            return "azure";
        }
    }
    "onprem"
}

#[cfg(test)]
mod node_profile_wiring_tests {
    use super::*;
    use crate::config::{AutomaticVerificationConfig, SshSessionConfig};
    use crate::miner_prover::types::{
        BinaryCpuInfo, BinaryMemoryInfo, BinaryNetworkInfo, GpuInfo, NetworkInterface,
        SmUtilizationStats, ValidationDetails,
    };
    use crate::miner_prover::verification_engine_builder::VerificationEngineBuilder;
    use crate::persistence::SimplePersistence;
    use std::str::FromStr;

    fn sample_node_result() -> crate::miner_prover::types::NodeResult {
        use crate::miner_prover::types::*;
        NodeResult {
            gpu_name: "NVIDIA A100".into(),
            gpu_uuid: "GPU-XYZ".into(),
            gpu_infos: vec![GpuInfo {
                index: 0,
                gpu_name: "NVIDIA A100".into(),
                gpu_uuid: "GPU-XYZ".into(),
                gpu_memory_gb: 80.0,
                computation_time_ns: 0,
                memory_bandwidth_gbps: 0.0,
                sm_utilization: SmUtilizationStats {
                    min_utilization: 0.0,
                    max_utilization: 0.0,
                    avg_utilization: 0.0,
                    per_sm_stats: vec![],
                },
                active_sms: 0,
                total_sms: 0,
                anti_debug_passed: true,
            }],
            cpu_info: BinaryCpuInfo {
                model: "AMD EPYC".into(),
                cores: 64,
                threads: 128,
                frequency_mhz: 0,
            },
            memory_info: BinaryMemoryInfo {
                total_gb: 256.0,
                available_gb: 0.0,
            },
            network_info: BinaryNetworkInfo {
                interfaces: vec![NetworkInterface {
                    name: "eth0".into(),
                    mac_address: "aa:bb".into(),
                    ip_addresses: vec!["10.0.0.2".into()],
                }],
            },
            cpu_pow: None,
            storage_pow: None,
            matrix_c: CompressedMatrix {
                rows: 0,
                cols: 0,
                data: vec![],
            },
            computation_time_ns: 0,
            checksum: [0u8; 32],
            sm_utilization: SmUtilizationStats {
                min_utilization: 0.0,
                max_utilization: 0.0,
                avg_utilization: 0.0,
                per_sm_stats: vec![],
            },
            active_sms: 0,
            total_sms: 0,
            memory_bandwidth_gbps: 0.0,
            anti_debug_passed: true,
            timing_fingerprint: 0,
        }
    }

    async fn create_test_engine() -> Result<(VerificationEngine, Arc<SimplePersistence>)> {
        let persistence = Arc::new(SimplePersistence::for_testing().await?);
        let config = VerificationConfig::test_default();
        let builder = VerificationEngineBuilder::new(
            config,
            AutomaticVerificationConfig::test_default(),
            SshSessionConfig::test_default(),
            Hotkey::new("5DAAnrj7VHTznn2AWBemMuyBwZWs6FNFjdyVXUeYum3PTXFy".to_string())
                .expect("valid hotkey"),
            persistence.clone(),
            None,
        );
        let engine = builder.build_for_testing().await?;
        Ok((engine, persistence))
    }

    async fn register_declared_node(
        persistence: &SimplePersistence,
        miner_uid: u16,
        host: &str,
        gpu_category: &str,
        gpu_count: u32,
    ) -> Result<String> {
        let miner_info = MinerInfo {
            uid: MinerUid::new(miner_uid),
            hotkey: Hotkey::new("5DAAnrj7VHTznn2AWBemMuyBwZWs6FNFjdyVXUeYum3PTXFy".to_string())
                .expect("valid hotkey"),
            endpoint: "http://127.0.0.1:9090".to_string(),
            is_validator: false,
            stake_tao: 100.0,
            last_verified: None,
            verification_score: 0.0,
        };
        persistence
            .ensure_miner_exists_with_info(&miner_info)
            .await?;

        let miner_id = format!("miner_{miner_uid}");
        persistence
            .upsert_registered_node(&miner_id, host, 22, "root", gpu_category, gpu_count, 1200)
            .await?;

        Ok(NodeId::new(host)?.uuid.to_string())
    }

    async fn insert_legacy_node_without_gpu_category(
        persistence: &SimplePersistence,
        miner_uid: u16,
        host: &str,
        gpu_count: u32,
    ) -> Result<String> {
        let miner_info = MinerInfo {
            uid: MinerUid::new(miner_uid),
            hotkey: Hotkey::new("5DAAnrj7VHTznn2AWBemMuyBwZWs6FNFjdyVXUeYum3PTXFy".to_string())
                .expect("valid hotkey"),
            endpoint: "http://127.0.0.1:9090".to_string(),
            is_validator: false,
            stake_tao: 100.0,
            last_verified: None,
            verification_score: 0.0,
        };
        persistence
            .ensure_miner_exists_with_info(&miner_info)
            .await?;

        let node_id = NodeId::new(host)?.uuid.to_string();
        let miner_id = format!("miner_{miner_uid}");
        let relationship_id = format!("{miner_id}_{node_id}");
        let ssh_endpoint = format!("root@{host}:22");

        sqlx::query(
            "INSERT INTO miner_nodes (
                id, miner_id, node_id, ssh_endpoint, node_ip, gpu_count, hourly_rate_cents,
                status, bid_active, last_node_check, created_at
             ) VALUES (?, ?, ?, ?, ?, ?, ?, 'online', 1, datetime('now'), datetime('now'))",
        )
        .bind(&relationship_id)
        .bind(&miner_id)
        .bind(&node_id)
        .bind(&ssh_endpoint)
        .bind(host)
        .bind(gpu_count as i64)
        .bind(1200i64)
        .execute(persistence.pool())
        .await?;

        Ok(node_id)
    }

    fn build_node_result_with_gpu_names(
        gpu_names: &[&str],
    ) -> crate::miner_prover::types::NodeResult {
        let mut node_result = sample_node_result();
        node_result.gpu_infos = gpu_names
            .iter()
            .enumerate()
            .map(|(index, gpu_name)| GpuInfo {
                index: index as u32,
                gpu_name: (*gpu_name).to_string(),
                gpu_uuid: format!("GPU-TEST-{index}"),
                gpu_memory_gb: 80.0,
                computation_time_ns: 0,
                memory_bandwidth_gbps: 0.0,
                sm_utilization: SmUtilizationStats {
                    min_utilization: 0.0,
                    max_utilization: 0.0,
                    avg_utilization: 0.0,
                    per_sm_stats: vec![],
                },
                active_sms: 0,
                total_sms: 0,
                anti_debug_passed: true,
            })
            .collect();

        if let Some(first_gpu) = node_result.gpu_infos.first() {
            node_result.gpu_name = first_gpu.gpu_name.clone();
            node_result.gpu_uuid = first_gpu.gpu_uuid.clone();
        } else {
            node_result.gpu_name = String::new();
            node_result.gpu_uuid = String::new();
        }

        node_result.cpu_info = BinaryCpuInfo {
            model: "AMD EPYC".to_string(),
            cores: 64,
            threads: 128,
            frequency_mhz: 0,
        };
        node_result.memory_info = BinaryMemoryInfo {
            total_gb: 256.0,
            available_gb: 128.0,
        };
        node_result.network_info = BinaryNetworkInfo {
            interfaces: vec![NetworkInterface {
                name: "eth0".to_string(),
                mac_address: "aa:bb".to_string(),
                ip_addresses: vec!["10.0.0.2".to_string()],
            }],
        };

        node_result
    }

    fn build_full_verification_result(
        node_id: &str,
        gpu_names: &[&str],
        validation_type: ValidationType,
    ) -> NodeVerificationResult {
        NodeVerificationResult {
            node_id: NodeId::from_str(node_id).expect("valid node id"),
            node_ssh_endpoint: "root@127.0.0.1:22".to_string(),
            node_ip: "127.0.0.1".to_string(),
            verification_score: 1.0,
            ssh_connection_successful: true,
            binary_validation_successful: true,
            node_result: Some(build_node_result_with_gpu_names(gpu_names)),
            failure_reasons: vec![],
            error: None,
            execution_time: Duration::from_secs(1),
            validation_details: ValidationDetails {
                ssh_test_duration: Duration::from_millis(10),
                binary_upload_duration: Duration::from_millis(10),
                binary_execution_duration: Duration::from_millis(10),
                total_validation_duration: Duration::from_millis(30),
                ssh_score: 1.0,
                binary_score: 1.0,
                combined_score: 1.0,
            },
            gpu_count: gpu_names.len() as u64,
            validation_type,
            hourly_rate_cents: 1200,
        }
    }

    #[tokio::test]
    async fn prepares_cr_and_labels_from_validation_and_network_profile() -> Result<()> {
        // Arrange: persistence with network profile
        let persistence = Arc::new(SimplePersistence::for_testing().await?);

        // Seed network profile with hostname, region, and organization
        persistence
            .store_node_network_profile(
                100u16,
                "node-abc",
                Some("192.0.2.10".into()),
                Some("kube-node-1".into()),
                Some("City".into()),
                Some("us-east-1".into()),
                Some("US".into()),
                None,
                Some("Amazon Web Services".into()),
                None,
                None,
                &chrono::Utc::now().to_rfc3339(),
                "{}",
            )
            .await?;

        // Build a minimal engine for calling the helper
        let config = VerificationConfig::test_default();
        let builder = VerificationEngineBuilder::new(
            config.clone(),
            AutomaticVerificationConfig::test_default(),
            SshSessionConfig::test_default(),
            Hotkey::new("5DAAnrj7VHTznn2AWBemMuyBwZWs6FNFjdyVXUeYum3PTXFy".to_string()).unwrap(),
            persistence.clone(),
            None,
        );
        let engine = builder.build_for_testing().await?;

        // Namespace influence
        std::env::set_var("BASILICA_NAMESPACE", "testns");

        let nr = sample_node_result();
        let (ns, cr, maybe_labels) = engine
            .prepare_node_profile_cr_and_labels(100, "node-abc", &nr, &[])
            .await?;

        // Assert namespace
        assert_eq!(ns, "testns");

        // Assert CR metadata and spec
        assert_eq!(cr.metadata.name.as_deref(), Some("node-abc"));
        assert_eq!(cr.metadata.namespace.as_deref(), Some("testns"));
        let spec = cr.data.get("spec").expect("spec present");
        assert_eq!(
            spec.get("provider").and_then(|v| v.as_str()).unwrap(),
            "aws"
        );
        assert_eq!(
            spec.get("region").and_then(|v| v.as_str()).unwrap(),
            "us-east-1"
        );

        // Assert status
        let status = cr.data.get("status").expect("status present");
        assert_eq!(
            status.get("kubeNodeName").and_then(|v| v.as_str()).unwrap(),
            "kube-node-1"
        );

        // Assert labels
        let (node_name, labels) = maybe_labels.expect("labels present");
        assert_eq!(node_name, "kube-node-1");
        assert_eq!(labels.get("basilica.ai/region").unwrap(), "us-east-1");
        assert_eq!(labels.get("basilica.ai/provider").unwrap(), "aws");
        assert_eq!(labels.get("basilica.ai/gpu-model").unwrap(), "NVIDIA A100");

        // Seed docker profile then re-run to assert docker labels
        persistence
            .store_node_docker_profile(
                100u16,
                "node-abc",
                true,
                Some("24.0.7".into()),
                vec![],
                true,
                None,
                "{}",
            )
            .await?;
        let (_ns2, _cr2, maybe_labels2) = engine
            .prepare_node_profile_cr_and_labels(100, "node-abc", &nr, &[])
            .await?;
        let (_node2, labels2) = maybe_labels2.expect("labels present");
        assert_eq!(labels2.get("basilica.ai/docker-active").unwrap(), "true");
        assert_eq!(labels2.get("basilica.ai/docker-version").unwrap(), "24.0.7");
        assert_eq!(labels2.get("basilica.ai/dind").unwrap(), "true");

        Ok(())
    }

    #[tokio::test]
    async fn full_validation_gpu_claim_check_passes_for_exact_match() -> Result<()> {
        let (engine, persistence) = create_test_engine().await?;
        let miner_uid = 201u16;
        let node_id =
            register_declared_node(&persistence, miner_uid, "10.0.20.1", "A100", 2).await?;
        let mut verification_result = build_full_verification_result(
            &node_id,
            &["NVIDIA A100", "NVIDIA A100"],
            ValidationType::Full,
        );

        engine
            .enforce_declared_gpu_claims_for_full_validation(miner_uid, &mut verification_result)
            .await?;

        assert!(verification_result.binary_validation_successful);
        assert_eq!(verification_result.verification_score, 1.0);
        assert!(!verification_result
            .failure_reasons
            .iter()
            .any(|reason| reason == MisbehaviourType::GpuDeclarationMismatch.as_str()));

        let logs = persistence
            .get_misbehaviour_logs(miner_uid, &node_id, chrono::Duration::days(7))
            .await?;
        assert!(logs.is_empty());

        Ok(())
    }

    #[tokio::test]
    async fn full_validation_gpu_claim_check_fails_on_count_mismatch() -> Result<()> {
        let (engine, persistence) = create_test_engine().await?;
        let miner_uid = 202u16;
        let node_id =
            register_declared_node(&persistence, miner_uid, "10.0.20.2", "A100", 4).await?;
        let mut verification_result = build_full_verification_result(
            &node_id,
            &["NVIDIA A100", "NVIDIA A100"],
            ValidationType::Full,
        );

        engine
            .enforce_declared_gpu_claims_for_full_validation(miner_uid, &mut verification_result)
            .await?;

        assert!(!verification_result.binary_validation_successful);
        assert_eq!(verification_result.verification_score, 0.0);
        assert_eq!(verification_result.validation_details.binary_score, 0.0);
        assert_eq!(verification_result.validation_details.combined_score, 0.0);
        assert!(verification_result
            .failure_reasons
            .iter()
            .any(|reason| reason == MisbehaviourType::GpuDeclarationMismatch.as_str()));
        assert!(verification_result
            .error
            .as_deref()
            .unwrap_or_default()
            .contains("GPU declaration mismatch"));

        let logs = persistence
            .get_misbehaviour_logs(miner_uid, &node_id, chrono::Duration::days(7))
            .await?;
        assert_eq!(logs.len(), 1);
        assert_eq!(
            logs[0].type_of_misbehaviour,
            MisbehaviourType::GpuDeclarationMismatch
        );

        Ok(())
    }

    #[tokio::test]
    async fn full_validation_gpu_claim_check_fails_on_category_mismatch() -> Result<()> {
        let (engine, persistence) = create_test_engine().await?;
        let miner_uid = 203u16;
        let node_id =
            register_declared_node(&persistence, miner_uid, "10.0.20.3", "A100", 1).await?;
        let mut verification_result =
            build_full_verification_result(&node_id, &["NVIDIA H100"], ValidationType::Full);

        engine
            .enforce_declared_gpu_claims_for_full_validation(miner_uid, &mut verification_result)
            .await?;

        assert!(!verification_result.binary_validation_successful);
        assert!(verification_result
            .failure_reasons
            .iter()
            .any(|reason| reason == MisbehaviourType::GpuDeclarationMismatch.as_str()));

        let logs = persistence
            .get_misbehaviour_logs(miner_uid, &node_id, chrono::Duration::days(7))
            .await?;
        assert_eq!(logs.len(), 1);
        assert_eq!(
            logs[0].type_of_misbehaviour,
            MisbehaviourType::GpuDeclarationMismatch
        );

        Ok(())
    }

    #[tokio::test]
    async fn full_validation_gpu_claim_check_uses_first_detected_gpu_for_category() -> Result<()> {
        let (engine, persistence) = create_test_engine().await?;
        let miner_uid = 204u16;
        let node_id =
            register_declared_node(&persistence, miner_uid, "10.0.20.4", "A100", 2).await?;
        let mut verification_result = build_full_verification_result(
            &node_id,
            &["NVIDIA H100", "NVIDIA A100"],
            ValidationType::Full,
        );

        engine
            .enforce_declared_gpu_claims_for_full_validation(miner_uid, &mut verification_result)
            .await?;

        assert!(!verification_result.binary_validation_successful);
        assert!(verification_result
            .failure_reasons
            .iter()
            .any(|reason| reason == MisbehaviourType::GpuDeclarationMismatch.as_str()));

        let logs = persistence
            .get_misbehaviour_logs(miner_uid, &node_id, chrono::Duration::days(7))
            .await?;
        assert_eq!(logs.len(), 1);
        assert_eq!(
            logs[0].type_of_misbehaviour,
            MisbehaviourType::GpuDeclarationMismatch
        );
        assert!(logs[0].details.contains("detected gpu_category"));

        Ok(())
    }

    #[tokio::test]
    async fn full_validation_gpu_claim_check_fails_when_declared_metadata_missing() -> Result<()> {
        let (engine, persistence) = create_test_engine().await?;
        let miner_uid = 205u16;
        let node_id =
            insert_legacy_node_without_gpu_category(&persistence, miner_uid, "10.0.20.5", 1)
                .await?;
        let mut verification_result =
            build_full_verification_result(&node_id, &["NVIDIA A100"], ValidationType::Full);

        engine
            .enforce_declared_gpu_claims_for_full_validation(miner_uid, &mut verification_result)
            .await?;

        assert!(!verification_result.binary_validation_successful);
        assert!(verification_result
            .failure_reasons
            .iter()
            .any(|reason| reason == MisbehaviourType::GpuDeclarationMismatch.as_str()));

        let logs = persistence
            .get_misbehaviour_logs(miner_uid, &node_id, chrono::Duration::days(7))
            .await?;
        assert_eq!(logs.len(), 1);
        assert_eq!(
            logs[0].type_of_misbehaviour,
            MisbehaviourType::GpuDeclarationMismatch
        );

        Ok(())
    }

    #[tokio::test]
    async fn lightweight_validation_does_not_trigger_gpu_claim_enforcement() -> Result<()> {
        let (engine, persistence) = create_test_engine().await?;
        let miner_uid = 206u16;
        let node_id =
            register_declared_node(&persistence, miner_uid, "10.0.20.6", "A100", 4).await?;
        let mut verification_result =
            build_full_verification_result(&node_id, &["NVIDIA H100"], ValidationType::Lightweight);

        engine
            .enforce_declared_gpu_claims_for_full_validation(miner_uid, &mut verification_result)
            .await?;

        assert!(verification_result.binary_validation_successful);
        assert_eq!(verification_result.verification_score, 1.0);
        assert!(!verification_result
            .failure_reasons
            .iter()
            .any(|reason| reason == MisbehaviourType::GpuDeclarationMismatch.as_str()));

        let logs = persistence
            .get_misbehaviour_logs(miner_uid, &node_id, chrono::Duration::days(7))
            .await?;
        assert!(logs.is_empty());

        Ok(())
    }

    #[tokio::test]
    async fn store_node_verification_result_sets_last_node_check_in_sqlite_format() -> Result<()> {
        let (engine, persistence) = create_test_engine().await?;
        let miner_uid = 207u16;
        let node_id =
            register_declared_node(&persistence, miner_uid, "10.0.20.7", "A100", 1).await?;

        let verification_result =
            build_full_verification_result(&node_id, &["NVIDIA A100"], ValidationType::Full);
        let miner_info = MinerInfo {
            uid: MinerUid::new(miner_uid),
            hotkey: Hotkey::new("5DAAnrj7VHTznn2AWBemMuyBwZWs6FNFjdyVXUeYum3PTXFy".to_string())
                .expect("valid hotkey"),
            endpoint: "http://127.0.0.1:9090".to_string(),
            is_validator: false,
            stake_tao: 100.0,
            last_verified: None,
            verification_score: 0.0,
        };

        engine
            .store_node_verification_result_with_miner_info(
                miner_uid,
                &verification_result,
                &miner_info,
            )
            .await?;

        let miner_id = format!("miner_{miner_uid}");
        let last_node_check: Option<String> = sqlx::query_scalar(
            "SELECT last_node_check
             FROM miner_nodes
             WHERE miner_id = ? AND node_id = ?",
        )
        .bind(&miner_id)
        .bind(&node_id)
        .fetch_one(persistence.pool())
        .await?;

        let timestamp = last_node_check.expect("last_node_check should be set");
        assert!(!timestamp.contains('T'));
        chrono::NaiveDateTime::parse_from_str(&timestamp, "%Y-%m-%d %H:%M:%S")
            .expect("timestamp should use SQLite datetime format");

        Ok(())
    }
}

/// SSH automation status information
#[derive(Debug, Clone)]
pub struct SshAutomationStatus {
    pub dynamic_discovery_enabled: bool,
    pub ssh_key_manager_available: bool,
    pub bittensor_service_available: bool,
    pub fallback_key_path: Option<PathBuf>,
}

impl std::fmt::Display for SshAutomationStatus {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        write!(
            f,
            "SSH Automation Status[dynamic={}, key_manager={}, bittensor={}, fallback_key={}]",
            self.dynamic_discovery_enabled,
            self.ssh_key_manager_available,
            self.bittensor_service_available,
            self.fallback_key_path
                .as_ref()
                .map(|p| p.display().to_string())
                .unwrap_or("none".to_string())
        )
    }
}

/// Verification step tracking
#[derive(Debug, Clone)]
pub struct VerificationStep {
    pub step_name: String,
    pub status: StepStatus,
    pub duration: Duration,
    pub details: String,
}

/// Step status tracking
#[derive(Debug, Clone)]
pub enum StepStatus {
    Pending,
    InProgress,
    Completed,
    Failed,
    PartialSuccess,
}

/// Enhanced verification result structure
#[derive(Debug, Clone)]
pub struct VerificationResult {
    pub miner_uid: u16,
    pub overall_score: f64,
    pub verification_steps: Vec<VerificationStep>,
    pub completed_at: chrono::DateTime<chrono::Utc>,
    pub error: Option<String>,
}
