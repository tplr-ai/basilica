//! # Verification Engine
//!
//! Handles the actual verification of miners and their nodes.
//! Implements Single Responsibility Principle by focusing only on verification logic.

use super::miner_client::{MinerClient, MinerClientConfig};
use super::types::MinerInfo;
use super::types::{NodeInfoDetailed, NodeVerificationResult, ValidationType};
use super::validation_states::{StateResult, ValidationState};
use super::validation_strategy::{ValidationNode, ValidationStrategy, ValidationStrategySelector};
use super::validation_worker::{ValidationWorkerQueue, WorkerQueueConfig};
use crate::config::VerificationConfig;
use crate::gpu::{categorization::GpuCategory, MinerGpuProfile};
use crate::metrics::ValidatorMetrics;
use crate::persistence::{
    entities::VerificationLog, gpu_profile_repository::GpuProfileRepository, SimplePersistence,
};
use crate::ssh::{ValidatorSshClient, ValidatorSshKeyManager};
use anyhow::{Context, Result};
use basilica_common::identity::{Hotkey, MinerUid, NodeId};
use chrono::Utc;
use futures::future::join_all;
use sqlx::Row;
use std::collections::HashMap;
use std::path::PathBuf;
use std::str::FromStr;
use std::sync::Arc;
use std::time::Duration;
use tracing::{debug, error, info, warn};

#[derive(Clone)]
pub struct VerificationEngine {
    config: VerificationConfig,
    miner_client_config: MinerClientConfig,
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
}

impl VerificationEngine {
    /// Check if an endpoint is invalid
    fn is_invalid_endpoint(&self, endpoint: &str) -> bool {
        // Check for common invalid patterns
        if endpoint.contains("0:0:0:0:0:0:0:0")
            || endpoint.contains("0.0.0.0")
            || endpoint.is_empty()
            || !endpoint.starts_with("http")
        {
            debug!("Invalid endpoint detected: {}", endpoint);
            return true;
        }

        // Validate URL parsing
        if let Ok(url) = url::Url::parse(endpoint) {
            if let Some(host) = url.host_str() {
                // Check for zero or loopback addresses that indicate invalid configuration
                if host == "0.0.0.0" || host == "::" || host == "localhost" || host == "127.0.0.1" {
                    debug!("Invalid host in endpoint: {}", endpoint);
                    return true;
                }
            } else {
                debug!("No host found in endpoint: {}", endpoint);
                return true;
            }
        } else {
            debug!("Failed to parse endpoint as URL: {}", endpoint);
            return true;
        }

        false
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

        // Step 1: Get nodes from discovery + database fallback
        let discovered_nodes = self
            .discover_miner_nodes(task.miner_uid, &task.miner_endpoint, &task.miner_hotkey)
            .await
            .unwrap_or_else(|e| {
                warn!(
                    "Failed to discover nodes for miner {} via gRPC: {}. Using database fallback.",
                    task.miner_uid, e
                );
                Vec::new()
            });

        let known_node_data = self
            .persistence
            .get_known_nodes_for_miner(task.miner_uid)
            .await?;
        let known_nodes = self.convert_db_data_to_node_info(known_node_data, task.miner_uid)?;
        let node_list = self.combine_node_lists(discovered_nodes, known_nodes);

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
                    verification_steps.push(VerificationStep {
                        step_name: format!("ssh_verification_{}", node_info.id),
                        status: StepStatus::Failed,
                        duration: workflow_start.elapsed(),
                        details: format!("SSH verification error: {e}"),
                    });
                }
            }
        }

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

    /// Discover nodes from miner via gRPC
    async fn discover_miner_nodes(
        &self,
        miner_uid: u16,
        miner_endpoint: &str,
        miner_hotkey: &str,
    ) -> Result<Vec<NodeInfoDetailed>> {
        info!(
            miner_uid = miner_uid,
            "[EVAL_FLOW] Starting node discovery from miner at: {}", miner_endpoint
        );
        debug!(
            miner_uid = miner_uid,
            "[EVAL_FLOW] Using config: timeout={:?}, grpc_port_offset={:?}, use_dynamic_discovery={}",
            self.config.discovery_timeout,
            self.config.grpc_port_offset,
            self.use_dynamic_discovery
        );

        // Validate endpoint before attempting connection
        if self.is_invalid_endpoint(miner_endpoint) {
            error!(
                miner_uid = miner_uid,
                "[EVAL_FLOW] Invalid miner endpoint detected: {}", miner_endpoint
            );
            return Err(anyhow::anyhow!(
                "Invalid miner endpoint: {}. Skipping discovery.",
                miner_endpoint
            ));
        }
        info!(
            miner_uid = miner_uid,
            "[EVAL_FLOW] Endpoint validation passed for: {}", miner_endpoint
        );

        // Create authenticated miner client
        info!(
            miner_uid = miner_uid,
            "[EVAL_FLOW] Creating authenticated miner client with validator hotkey: {}",
            self.validator_hotkey
                .to_string()
                .chars()
                .take(8)
                .collect::<String>()
                + "..."
        );
        let client = self.create_authenticated_client()?;

        // Connect and authenticate to miner
        info!(
            miner_uid = miner_uid,
            "[EVAL_FLOW] Attempting gRPC connection to miner at: {}", miner_endpoint
        );
        let connection_start = std::time::Instant::now();
        let mut connection = match client
            .connect_and_authenticate(miner_uid, miner_endpoint, miner_hotkey)
            .await
        {
            Ok(conn) => {
                info!(
                    miner_uid = miner_uid,
                    "[EVAL_FLOW] Successfully connected and authenticated to miner in {:?}",
                    connection_start.elapsed()
                );
                conn
            }
            Err(e) => {
                error!(
                    miner_uid = miner_uid,
                    "[EVAL_FLOW] Failed to connect to miner at {} after {:?}: {}",
                    miner_endpoint,
                    connection_start.elapsed(),
                    e
                );
                return Err(e).context("Failed to connect to miner for node discovery");
            }
        };

        info!(miner_uid = miner_uid, "[EVAL_FLOW] Requesting nodes");
        let request_start = std::time::Instant::now();
        let node_details = match connection.request_nodes().await {
            Ok(details) => {
                info!(
                    miner_uid = miner_uid,
                    "[EVAL_FLOW] Successfully received node details in {:?}, count={}",
                    request_start.elapsed(),
                    details.len()
                );
                for (i, detail) in details.iter().enumerate() {
                    info!(
                        miner_uid = miner_uid,
                        node_id = %detail.node_id,
                        "[EVAL_FLOW] Node {}: id={}, endpoint={}:{}",
                        i,
                        detail.node_id,
                        detail.host,
                        detail.port
                    );
                }
                details
            }
            Err(e) => {
                error!(
                    miner_uid = miner_uid,
                    "[EVAL_FLOW] Failed to request nodes from miner after {:?}: {}",
                    request_start.elapsed(),
                    e
                );
                return Ok(vec![]);
            }
        };

        let node_count = node_details.len();
        let nodes: Vec<NodeInfoDetailed> = node_details
            .into_iter()
            .map(|details| -> Result<NodeInfoDetailed> {
                debug!(
                    miner_uid = miner_uid,
                    node_id = %details.node_id,
                    "[EVAL_FLOW] Discovered node details from miner: miner_uid={}, node_id={}",
                    miner_uid, details.node_id
                );

                Ok(NodeInfoDetailed {
                    id: NodeId::from_str(&details.node_id).map_err(|e| {
                        anyhow::anyhow!("Invalid node ID '{}': {}", details.node_id, e)
                    })?,
                    miner_uid: MinerUid::new(miner_uid),
                    status: "available".to_string(),
                    capabilities: vec!["gpu".to_string()],
                    node_ssh_endpoint: format!(
                        "{}@{}:{}",
                        details.username, details.host, details.port
                    ),
                })
            })
            .collect::<Result<Vec<_>, _>>()?;

        info!(
            miner_uid = miner_uid,
            "[EVAL_FLOW] Node discovery completed: {} nodes mapped from {} details",
            nodes.len(),
            node_count
        );

        Ok(nodes)
    }

    /// Helper function to clean up active SSH session for a node
    async fn cleanup_active_session(&self, node_id: &str) {
        // Direct SSH sessions are now managed at connection level
        // No explicit release needed
        let _ = node_id; // Avoid unused parameter warning
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
                Some("Binary validation failed".to_string())
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
             SET status = ?, last_health_check = ?, updated_at = ?
             WHERE node_id = ?",
        )
        .bind(&status)
        .bind(&now)
        .bind(&now)
        .bind(&verification_log.node_id)
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
                        miner_info,
                    )
                    .await?;

                self.persistence
                    .store_gpu_uuid_assignments(
                        miner_uid,
                        &node_result.node_id.to_string(),
                        &gpu_infos,
                    )
                    .await?;

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

    /// Sync miners from metagraph to database
    pub async fn sync_miners_from_metagraph(&self, miners: &[MinerInfo]) -> Result<()> {
        self.persistence.sync_miners_from_metagraph(miners).await
    }

    /// Create authenticated miner client
    pub fn create_authenticated_client(&self) -> Result<MinerClient> {
        let mut client = if let Some(ref bittensor_service) = self.bittensor_service {
            let signer = Arc::new(super::miner_client::BittensorServiceSigner::new(
                bittensor_service.clone(),
            ));
            MinerClient::with_signer(
                self.miner_client_config.clone(),
                self.validator_hotkey.clone(),
                signer,
            )
        } else {
            MinerClient::new(
                self.miner_client_config.clone(),
                self.validator_hotkey.clone(),
            )
        };

        // Add SSH public key if available
        if let Some(public_key) = self.get_ssh_public_key() {
            client = client.with_ssh_public_key(public_key);
        }

        Ok(client)
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
        miner_client_config: MinerClientConfig,
        validator_hotkey: Hotkey,
        ssh_client: Arc<ValidatorSshClient>,
        persistence: Arc<SimplePersistence>,
        use_dynamic_discovery: bool,
        ssh_key_manager: Option<Arc<ValidatorSshKeyManager>>,
        bittensor_service: Option<Arc<bittensor::Service>>,
        metrics: Option<Arc<ValidatorMetrics>>,
    ) -> Result<Self> {
        // Validate required components for dynamic discovery
        if use_dynamic_discovery && ssh_key_manager.is_none() {
            return Err(anyhow::anyhow!(
                "SSH key manager is required when dynamic discovery is enabled"
            ));
        }

        Ok(Self {
            config: config.clone(),
            miner_client_config,
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
        })
    }

    /// Initialize validation server mode
    pub async fn initialize_validation_server(&mut self) -> Result<()> {
        info!("Initializing validation server mode for VerificationEngine");
        let mut node = self.validation_node.write().await;
        node.initialize_server_mode(&self.config.binary_validation)
            .await?;
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
        self.persistence
            .cleanup_failed_nodes_after_failures(
                consecutive_failures_threshold,
                self.config.gpu_assignment_cleanup_ttl,
            )
            .await?;

        Ok(())
    }

    /// Original cleanup implementation (now moved to persistence layer)
    #[allow(dead_code)]
    async fn cleanup_failed_nodes_after_failures_original(
        &self,
        consecutive_failures_threshold: i32,
    ) -> Result<()> {
        info!(
            "Running node cleanup - checking for {} consecutive failures",
            consecutive_failures_threshold
        );

        // Step 1: Clean up any GPU assignments for offline nodes (immediate fix)
        let offline_with_gpus_query = r#"
            SELECT DISTINCT me.node_id, me.miner_id, COUNT(ga.gpu_uuid) as gpu_count
            FROM miner_nodes me
            INNER JOIN gpu_uuid_assignments ga ON me.node_id = ga.node_id AND me.miner_id = ga.miner_id
            WHERE me.status = 'offline'
            GROUP BY me.node_id, me.miner_id
        "#;

        let offline_with_gpus = sqlx::query(offline_with_gpus_query)
            .fetch_all(self.persistence.pool())
            .await?;

        let mut gpu_assignments_cleaned = 0;
        for row in offline_with_gpus {
            let node_id: String = row.try_get("node_id")?;
            let miner_id: String = row.try_get("miner_id")?;
            let gpu_count: i64 = row.try_get("gpu_count")?;

            info!(
                "Cleaning up {} GPU assignments for offline node {} (miner: {})",
                gpu_count, node_id, miner_id
            );

            let rows_cleaned = self
                .persistence
                .cleanup_gpu_assignments(&node_id, &miner_id, None)
                .await?;
            gpu_assignments_cleaned += rows_cleaned;
        }

        // Step 1b: Clean up nodes with mismatched GPU counts
        let mismatched_gpu_query = r#"
            SELECT me.node_id, me.miner_id, me.gpu_count, me.status
            FROM miner_nodes me
            WHERE me.gpu_count > 0
            AND NOT EXISTS (
                SELECT 1 FROM gpu_uuid_assignments ga
                WHERE ga.node_id = me.node_id AND ga.miner_id = me.miner_id
            )
        "#;

        let mismatched_nodes = sqlx::query(mismatched_gpu_query)
            .fetch_all(self.persistence.pool())
            .await?;

        for row in mismatched_nodes {
            let node_id: String = row.try_get("node_id")?;
            let miner_id: String = row.try_get("miner_id")?;
            let gpu_count: i32 = row.try_get("gpu_count")?;
            let status: String = row.try_get("status")?;

            warn!(
                "Node {} (miner: {}) claims {} GPUs but has no assignments, status: {}. Resetting GPU count to 0",
                node_id, miner_id, gpu_count, status
            );

            // Reset GPU count to 0 to reflect reality
            sqlx::query(
                "UPDATE miner_nodes SET gpu_count = 0, updated_at = datetime('now')
                 WHERE node_id = ? AND miner_id = ?",
            )
            .bind(&node_id)
            .bind(&miner_id)
            .execute(self.persistence.pool())
            .await?;

            // Mark offline if they claim GPUs but have none
            if status == "online" || status == "verified" {
                sqlx::query(
                    "UPDATE miner_nodes SET status = 'offline', updated_at = datetime('now')
                     WHERE node_id = ? AND miner_id = ?",
                )
                .bind(&node_id)
                .bind(&miner_id)
                .execute(self.persistence.pool())
                .await?;

                info!(
                    "Marked node {} as offline (claimed {} GPUs but has 0 assignments)",
                    node_id, gpu_count
                );
            }
        }

        // Step 1c: Clean up stale GPU assignments (GPUs that haven't been verified recently)
        // Increased threshold from 1 hour to 6 hours to reduce aggressive cleanup
        let stale_gpu_cleanup_query = r#"
            DELETE FROM gpu_uuid_assignments
            WHERE last_verified < datetime('now', '-6 hours')
            OR (
                EXISTS (
                    SELECT 1 FROM miner_nodes me
                    WHERE me.node_id = gpu_uuid_assignments.node_id
                    AND me.miner_id = gpu_uuid_assignments.miner_id
                    AND me.status = 'offline'
                    AND (
                        me.last_health_check < datetime('now', '-2 hours')
                        OR (me.last_health_check IS NULL AND me.updated_at < datetime('now', '-2 hours'))
                    )
                )
            )
        "#;

        let stale_gpu_result = sqlx::query(stale_gpu_cleanup_query)
            .execute(self.persistence.pool())
            .await?;

        if stale_gpu_result.rows_affected() > 0 {
            info!(
                security = true,
                cleaned_count = stale_gpu_result.rows_affected(),
                cleanup_reason = "stale_timeout",
                threshold_hours = 6,
                "Cleaned up {} stale GPU assignments (not verified in last 6 hours or belonging to offline nodes >2h)",
                stale_gpu_result.rows_affected()
            );
        }

        // Step 1d: Clean up GPU assignments from nodes offline
        // Increased minimum cleanup time from 30 minutes to 2 hours
        let cleanup_minutes = self
            .config
            .gpu_assignment_cleanup_ttl
            .map(|d| d.as_secs() / 60)
            .unwrap_or(120)
            .max(120); // Ensure minimum 2 hours to reduce aggressive cleanup

        info!(
            "Cleaning GPU assignments from nodes offline >{} minutes",
            cleanup_minutes
        );
        let stale_offline_query = format!(
            r#"
            SELECT DISTINCT me.node_id, me.miner_id, COUNT(ga.gpu_uuid) as gpu_count
            FROM miner_nodes me
            INNER JOIN gpu_uuid_assignments ga ON me.node_id = ga.node_id AND me.miner_id = ga.miner_id
            WHERE me.status = 'offline'
            AND (
                me.last_health_check < datetime('now', '-{cleanup_minutes} minutes')
                OR (me.last_health_check IS NULL AND me.updated_at < datetime('now', '-{cleanup_minutes} minutes'))
            )
            GROUP BY me.node_id, me.miner_id
            "#
        );

        let stale_offline = sqlx::query(&stale_offline_query)
            .fetch_all(self.persistence.pool())
            .await?;

        let mut stale_gpu_cleaned = 0;
        for row in stale_offline {
            let node_id: String = row.try_get("node_id")?;
            let miner_id: String = row.try_get("miner_id")?;
            let gpu_count: i64 = row.try_get("gpu_count")?;

            info!(
                security = true,
                node_id = %node_id,
                miner_id = %miner_id,
                gpu_count = gpu_count,
                cleanup_minutes = cleanup_minutes,
                "Cleaning GPU assignments from node offline >{}min", cleanup_minutes
            );

            let cleaned = self
                .persistence
                .cleanup_gpu_assignments(&node_id, &miner_id, None)
                .await?;
            stale_gpu_cleaned += cleaned;
        }

        if stale_gpu_cleaned > 0 {
            info!(
                security = true,
                cleaned_count = stale_gpu_cleaned,
                cleanup_minutes = cleanup_minutes,
                "Cleaned {} GPU assignments from nodes offline >{}min",
                stale_gpu_cleaned,
                cleanup_minutes
            );
        }

        // Step 2: Find and delete nodes with consecutive failures
        let delete_nodes_query = r#"
            WITH recent_verifications AS (
                SELECT
                    vl.node_id,
                    vl.success,
                    vl.timestamp,
                    ROW_NUMBER() OVER (PARTITION BY vl.node_id ORDER BY vl.timestamp DESC) as rn
                FROM verification_logs vl
                WHERE vl.timestamp > datetime('now', '-1 hour')
            )
            SELECT
                me.node_id,
                me.miner_id,
                me.status,
                COALESCE(SUM(CASE WHEN rv.success = 0 AND rv.rn <= ? THEN 1 ELSE 0 END), 0) as consecutive_fails,
                COALESCE(SUM(CASE WHEN rv.success = 1 AND rv.rn <= ? THEN 1 ELSE 0 END), 0) as recent_successes,
                MAX(rv.timestamp) as last_verification
            FROM miner_nodes me
            LEFT JOIN recent_verifications rv ON me.node_id = rv.node_id
            WHERE me.status = 'offline'
            GROUP BY me.node_id, me.miner_id, me.status
            HAVING consecutive_fails >= ? AND recent_successes = 0
        "#;

        let nodes_to_delete = sqlx::query(delete_nodes_query)
            .bind(consecutive_failures_threshold)
            .bind(consecutive_failures_threshold)
            .bind(consecutive_failures_threshold)
            .fetch_all(self.persistence.pool())
            .await?;

        let mut deleted = 0;
        for row in nodes_to_delete {
            let node_id: String = row.try_get("node_id")?;
            let miner_id: String = row.try_get("miner_id")?;
            let consecutive_fails: i64 = row.try_get("consecutive_fails")?;
            let last_verification: Option<String> = row.try_get("last_verification").ok();

            info!(
                "Permanently deleting node {} (miner: {}) after {} consecutive failures, last seen: {}",
                node_id, miner_id, consecutive_fails,
                last_verification.as_deref().unwrap_or("never")
            );

            // Use transaction to ensure atomic deletion
            let mut tx = self.persistence.pool().begin().await?;

            // Clean up any remaining GPU assignments
            self.persistence
                .cleanup_gpu_assignments(&node_id, &miner_id, Some(&mut tx))
                .await?;

            // Delete the node record
            sqlx::query("DELETE FROM miner_nodes WHERE node_id = ? AND miner_id = ?")
                .bind(&node_id)
                .bind(&miner_id)
                .execute(&mut *tx)
                .await?;

            tx.commit().await?;
            deleted += 1;

            // Clean up any active SSH sessions
            self.cleanup_active_session(&node_id).await;
        }

        // Step 3: Delete stale offline nodes
        let cleanup_minutes = self
            .config
            .gpu_assignment_cleanup_ttl
            .map(|d| d.as_secs() / 60)
            .unwrap_or(120)
            .max(120);

        let stale_delete_query = format!(
            r#"
            DELETE FROM miner_nodes
            WHERE status = 'offline'
            AND (
                last_health_check < datetime('now', '-{} minutes')
                OR (last_health_check IS NULL AND updated_at < datetime('now', '-{} minutes'))
            )
            "#,
            cleanup_minutes, cleanup_minutes
        );

        info!(
            "Deleting stale offline nodes using {}min timeout (configurable via gpu_assignment_cleanup_ttl)",
            cleanup_minutes
        );

        let stale_result = sqlx::query(&stale_delete_query)
            .execute(self.persistence.pool())
            .await?;

        let stale_deleted = stale_result.rows_affected();

        // Step 4: Update GPU profiles for all miners with wrong gpu count profile
        let affected_miners_query = r#"
            SELECT DISTINCT miner_uid
            FROM miner_gpu_profiles
            WHERE miner_uid IN (
                -- Miners with offline nodes
                SELECT DISTINCT CAST(SUBSTR(miner_id, 7) AS INTEGER)
                FROM miner_nodes
                WHERE status = 'offline'

                UNION

                -- Miners with non-empty GPU profiles but no active nodes
                SELECT miner_uid
                FROM miner_gpu_profiles
                WHERE gpu_counts_json <> '{}'
                AND NOT EXISTS (
                    SELECT 1 FROM miner_nodes
                    WHERE miner_id = 'miner_' || miner_gpu_profiles.miner_uid
                    AND status NOT IN ('offline', 'failed', 'stale')
                )
            )
        "#;

        let affected_miners = sqlx::query(affected_miners_query)
            .fetch_all(self.persistence.pool())
            .await?;

        for row in affected_miners {
            let miner_uid: i64 = row.try_get("miner_uid")?;
            let miner_id = format!("miner_{}", miner_uid);

            let gpu_counts = self
                .persistence
                .get_miner_gpu_uuid_assignments(&miner_id)
                .await?;

            let mut gpu_map: std::collections::HashMap<String, u32> =
                std::collections::HashMap::new();
            for (_, count, gpu_name, _) in gpu_counts {
                let category =
                    crate::gpu::categorization::GpuCategory::from_str(&gpu_name).unwrap();
                let model = category.to_string();
                *gpu_map.entry(model).or_insert(0) += count;
            }

            let update_query = if gpu_map.is_empty() {
                r#"
                UPDATE miner_gpu_profiles
                SET gpu_counts_json = ?,
                    total_score = 0.0,
                    verification_count = 0,
                    last_successful_validation = NULL,
                    last_updated = datetime('now')
                WHERE miner_uid = ?
                "#
            } else {
                r#"
                UPDATE miner_gpu_profiles
                SET gpu_counts_json = ?,
                    last_updated = datetime('now')
                WHERE miner_uid = ?
                "#
            };

            let gpu_json = serde_json::to_string(&gpu_map)?;
            let result = sqlx::query(update_query)
                .bind(&gpu_json)
                .bind(miner_uid)
                .execute(self.persistence.pool())
                .await?;

            if result.rows_affected() > 0 {
                info!(
                    "Updated GPU profile for miner {} after cleanup: {}",
                    miner_uid, gpu_json
                );
            }
        }

        // Log summary
        if gpu_assignments_cleaned > 0 {
            info!(
                "Deleted {} GPU assignments from offline nodes",
                gpu_assignments_cleaned
            );
        }

        if deleted > 0 {
            info!(
                "Deleted {} nodes with {} or more consecutive failures",
                deleted, consecutive_failures_threshold
            );
        }

        if stale_deleted > 0 {
            info!("Deleted {} stale offline nodes", stale_deleted);
        }

        if gpu_assignments_cleaned == 0 && deleted == 0 && stale_deleted == 0 {
            debug!("No nodes needed cleanup in this cycle");
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
        let result = match strategy {
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
                let binary_config = &self.config.binary_validation;
                self.validation_node
                    .read()
                    .await
                    .execute_full_validation(
                        node_info,
                        &ssh_details,
                        binary_config,
                        &self.validator_hotkey,
                        miner_uid,
                    )
                    .await
            }
        };

        result
    }

    /// Convert database node data to NodeInfoDetailed
    fn convert_db_data_to_node_info(
        &self,
        db_data: Vec<(String, String, i32, String)>,
        miner_uid: u16,
    ) -> Result<Vec<NodeInfoDetailed>> {
        let mut nodes = Vec::new();

        for (node_id, ssh_endpoint, gpu_count, status) in db_data {
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
            });
        }

        Ok(nodes)
    }

    /// Combine discovered and known node lists
    fn combine_node_lists(
        &self,
        discovered: Vec<NodeInfoDetailed>,
        known: Vec<NodeInfoDetailed>,
    ) -> Vec<NodeInfoDetailed> {
        let mut combined = Vec::new();
        let mut seen_ids = std::collections::HashSet::new();

        for node in discovered {
            if seen_ids.insert(node.id.to_string()) {
                combined.push(node);
            }
        }

        for node in known {
            if seen_ids.insert(node.id.to_string()) {
                combined.push(node);
            }
        }

        combined
    }

    pub fn get_ssh_public_key(&self) -> Option<String> {
        if let Some(ref key_manager) = self.ssh_key_manager {
            key_manager.get_ssh_public_key()
        } else {
            None
        }
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
