//! # Verification Engine
//!
//! Handles the actual verification of miners and their executors.
//! Implements Single Responsibility Principle by focusing only on verification logic.

use super::miner_client::{MinerClient, MinerClientConfig};
use super::types::MinerInfo;
use super::types::{ExecutorInfoDetailed, ExecutorVerificationResult, GpuInfo, ValidationType};
use super::validation_states::{StateResult, ValidationState};
use super::validation_strategy::{
    ValidationExecutor, ValidationStrategy, ValidationStrategySelector,
};
use super::validation_worker::{ValidationWorkerQueue, WorkerQueueConfig};
use crate::config::VerificationConfig;
use crate::gpu::{categorization::GpuCategory, MinerGpuProfile};
use crate::metrics::ValidatorMetrics;
use crate::persistence::{
    entities::VerificationLog, gpu_profile_repository::GpuProfileRepository, SimplePersistence,
};
use crate::ssh::{SshSessionManager, ValidatorSshClient, ValidatorSshKeyManager};
use anyhow::{Context, Result};
use basilica_common::identity::{ExecutorId, Hotkey, MinerUid};
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
    /// SSH key path for executor access (fallback)
    ssh_key_path: Option<PathBuf>,
    /// Optional Bittensor service for signing
    bittensor_service: Option<Arc<bittensor::Service>>,
    /// SSH key manager for session keys
    ssh_key_manager: Option<Arc<ValidatorSshKeyManager>>,
    /// SSH session manager for preventing concurrent sessions
    ssh_session_manager: Arc<SshSessionManager>,
    /// Metrics for tracking rental status and other validator metrics
    metrics: Option<Arc<ValidatorMetrics>>,
    /// Validation strategy selector for determining validation approach
    validation_strategy_selector: Arc<ValidationStrategySelector>,
    /// Validation executor for running validation strategies
    validation_executor: Arc<tokio::sync::RwLock<ValidationExecutor>>,
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

        // Step 1: Get executors from discovery + database fallback
        let discovered_executors = self.discover_miner_executors(&task.miner_endpoint, &task.miner_hotkey).await
            .unwrap_or_else(|e| {
                warn!("Failed to discover executors for miner {} via gRPC: {}. Using database fallback.", task.miner_uid, e);
                Vec::new()
            });

        let known_executor_data = self
            .persistence
            .get_known_executors_for_miner(task.miner_uid)
            .await?;
        let known_executors =
            self.convert_db_data_to_executor_info(known_executor_data, task.miner_uid)?;
        let executor_list = self.combine_executor_lists(discovered_executors, known_executors);

        verification_steps.push(VerificationStep {
            step_name: "executor_discovery".to_string(),
            status: StepStatus::Completed,
            duration: workflow_start.elapsed(),
            details: format!("Found {} executors for verification", executor_list.len()),
        });

        if executor_list.is_empty() {
            info!(
                miner_uid = task.miner_uid,
                intended_strategy = ?task.intended_validation_strategy,
                "[EVAL_FLOW] No executors found for miner {}", task.miner_uid
            );

            return Ok(VerificationResult {
                miner_uid: task.miner_uid,
                overall_score: 0.0,
                verification_steps,
                completed_at: chrono::Utc::now(),
                error: Some("No executors found for miner".to_string()),
            });
        }

        // Route to worker queue if enabled
        if let Some(ref worker_queue) = self.worker_queue {
            info!(
                miner_uid = task.miner_uid,
                executor_count = executor_list.len(),
                intended_strategy = ?task.intended_validation_strategy,
                "[EVAL_FLOW] Routing {} executors to worker queue for miner {}",
                executor_list.len(),
                task.miner_uid
            );
            return self
                .route_to_worker_queue(
                    executor_list,
                    task,
                    worker_queue,
                    &mut verification_steps,
                    workflow_start,
                )
                .await;
        }

        // Step 2: Execute SSH-based verification for each executor
        let mut executor_results = Vec::new();
        let mut executors_skipped_for_strategy = 0;
        let total_executors = executor_list.len();

        info!(
            miner_uid = task.miner_uid,
            executor_count = total_executors,
            intended_strategy = ?task.intended_validation_strategy,
            "[EVAL_FLOW] Starting executors verification"
        );

        // Create futures for all executor validations
        let validation_futures: Vec<_> = executor_list
            .into_iter()
            .map(|executor_info| {
                let self_clone = self.clone();
                let miner_endpoint = task.miner_endpoint.clone();
                let miner_uid = task.miner_uid;
                let miner_hotkey = task.miner_hotkey.clone();
                let intended_strategy = task.intended_validation_strategy;

                async move {
                    info!(
                        miner_uid = miner_uid,
                        executor_id = %executor_info.id,
                        intended_strategy = ?intended_strategy,
                        "[EVAL_FLOW] Starting verification for executor"
                    );

                    // Set in-queue state for this specific executor being validated
                    if let Some(ref metrics) = self_clone.validation_executor.read().await.metrics()
                    {
                        metrics.prometheus().set_executor_validation_state(
                            &executor_info.id.to_string(),
                            miner_uid,
                            intended_strategy,
                            ValidationState::InQueue,
                            StateResult::Current,
                        );
                    }

                    let result = self_clone
                        .verify_executor(
                            &miner_endpoint,
                            &executor_info,
                            miner_uid,
                            &miner_hotkey,
                            intended_strategy,
                        )
                        .await;

                    (executor_info, result)
                }
            })
            .collect();

        // Execute all validations concurrently
        let validation_results = join_all(validation_futures).await;

        // Process all validation results
        for (executor_info, result) in validation_results {
            match result {
                Ok(result) => {
                    let score = result.verification_score;
                    info!(
                        miner_uid = task.miner_uid,
                        executor_id = %executor_info.id,
                        verification_score = score,
                        intended_strategy = ?task.intended_validation_strategy,
                        "[EVAL_FLOW] SSH verification completed"
                    );
                    executor_results.push(result);
                    verification_steps.push(VerificationStep {
                        step_name: format!("ssh_verification_{}", executor_info.id),
                        status: StepStatus::Completed,
                        duration: workflow_start.elapsed(),
                        details: format!("SSH verification completed, score: {score}"),
                    });
                }
                Err(e) if e.to_string().contains("Strategy mismatch") => {
                    executors_skipped_for_strategy += 1;
                    debug!(
                        miner_uid = task.miner_uid,
                        executor_id = %executor_info.id,
                        pipeline_type = ?task.intended_validation_strategy,
                        intended_strategy = ?task.intended_validation_strategy,
                        "[EVAL_FLOW] Executor requires different validation type, will be handled by other pipeline"
                    );
                    verification_steps.push(VerificationStep {
                        step_name: format!("ssh_verification_{}", executor_info.id),
                        status: StepStatus::Completed,
                        duration: workflow_start.elapsed(),
                        details: "Skipped - handled by other validation pipeline".to_string(),
                    });
                }
                Err(e) => {
                    error!(
                        miner_uid = task.miner_uid,
                        executor_id = %executor_info.id,
                        error = %e,
                        intended_strategy = ?task.intended_validation_strategy,
                        "[EVAL_FLOW] verification failed"
                    );
                    verification_steps.push(VerificationStep {
                        step_name: format!("ssh_verification_{}", executor_info.id),
                        status: StepStatus::Failed,
                        duration: workflow_start.elapsed(),
                        details: format!("SSH verification error: {e}"),
                    });
                }
            }
        }

        // Step 3: Calculate overall verification score
        let overall_score = if executor_results.is_empty() {
            // Only return 0 if ALL executors were skipped for strategy mismatch
            // If we have no results and all were skipped, this pipeline isn't responsible for this miner
            if executors_skipped_for_strategy == total_executors && total_executors > 0 {
                debug!(
                    miner_uid = task.miner_uid,
                    intended_strategy = ?task.intended_validation_strategy,
                    skipped_count = executors_skipped_for_strategy,
                    pipeline_type = ?task.intended_validation_strategy,
                    "[EVAL_FLOW] All executors require different validation type, score will come from other pipeline"
                );
            }
            0.0
        } else {
            let avg_score = executor_results
                .iter()
                .map(|r| r.verification_score)
                .sum::<f64>()
                / executor_results.len() as f64;

            info!(
                miner_uid = task.miner_uid,
                intended_strategy = ?task.intended_validation_strategy,
                validated_count = executor_results.len(),
                skipped_count = executors_skipped_for_strategy,
                total_executors = total_executors,
                average_score = avg_score,
                pipeline_type = ?task.intended_validation_strategy,
                "[EVAL_FLOW] Validation completed for miner"
            );

            avg_score
        };

        // Step 4: Store individual executor verification results
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

        for result in &executor_results {
            self.store_executor_verification_result_with_miner_info(
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
            validated_executors = executor_results.len(),
            skipped_executors = executors_skipped_for_strategy,
            total_executors = total_executors,
            pipeline_type = ?task.intended_validation_strategy,
            overall_score = overall_score,
            "[EVAL_FLOW] Verification workflow completed for miner {} in {:?}, score: {:.2} ({} of {} executors validated in {} pipeline)",
            task.miner_uid,
            workflow_start.elapsed(),
            overall_score,
            executor_results.len(),
            total_executors,
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

    /// Route executors to worker queue for parallel processing
    async fn route_to_worker_queue(
        &self,
        executors: Vec<ExecutorInfoDetailed>,
        task: &super::scheduler::VerificationTask,
        worker_queue: &Arc<ValidationWorkerQueue>,
        verification_steps: &mut Vec<VerificationStep>,
        workflow_start: std::time::Instant,
    ) -> Result<VerificationResult> {
        // Publish all executors to the appropriate queue
        let mut published_count = 0;
        let mut failed_count = 0;

        for executor in executors {
            match worker_queue.publish(executor.clone(), task.clone()).await {
                Ok(_) => {
                    // Set in-queue state only for executors successfully published to queue
                    if let Some(ref metrics) = self.validation_executor.read().await.metrics() {
                        metrics.prometheus().set_executor_validation_state(
                            &executor.id.to_string(),
                            task.miner_uid,
                            task.intended_validation_strategy,
                            ValidationState::InQueue,
                            StateResult::Current,
                        );
                    }
                    published_count += 1;
                }
                Err(e) => {
                    warn!("Failed to publish executor to queue: {}", e);
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
                "Published {} executors to queue ({} failed)",
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
                Some("Failed to publish any executors to queue".to_string())
            } else {
                None
            },
        })
    }

    /// Discover executors from miner via gRPC
    async fn discover_miner_executors(
        &self,
        miner_endpoint: &str,
        miner_hotkey: &str,
    ) -> Result<Vec<ExecutorInfoDetailed>> {
        info!(
            "[EVAL_FLOW] Starting executor discovery from miner at: {}",
            miner_endpoint
        );
        debug!("[EVAL_FLOW] Using config: timeout={:?}, grpc_port_offset={:?}, use_dynamic_discovery={}",
               self.config.discovery_timeout, self.config.grpc_port_offset, self.use_dynamic_discovery);

        // Validate endpoint before attempting connection
        if self.is_invalid_endpoint(miner_endpoint) {
            error!(
                "[EVAL_FLOW] Invalid miner endpoint detected: {}",
                miner_endpoint
            );
            return Err(anyhow::anyhow!(
                "Invalid miner endpoint: {}. Skipping discovery.",
                miner_endpoint
            ));
        }
        info!(
            "[EVAL_FLOW] Endpoint validation passed for: {}",
            miner_endpoint
        );

        // Create authenticated miner client
        info!(
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
            "[EVAL_FLOW] Attempting gRPC connection to miner at: {}",
            miner_endpoint
        );
        let connection_start = std::time::Instant::now();
        let mut connection = match client
            .connect_and_authenticate(miner_endpoint, miner_hotkey)
            .await
        {
            Ok(conn) => {
                info!(
                    "[EVAL_FLOW] Successfully connected and authenticated to miner in {:?}",
                    connection_start.elapsed()
                );
                conn
            }
            Err(e) => {
                error!(
                    "[EVAL_FLOW] Failed to connect to miner at {} after {:?}: {}",
                    miner_endpoint,
                    connection_start.elapsed(),
                    e
                );
                return Err(e).context("Failed to connect to miner for executor discovery");
            }
        };

        // Request executors with requirements
        let requirements = basilica_protocol::common::ResourceLimits {
            max_cpu_cores: 4,
            max_memory_mb: 8192,
            max_storage_mb: 10240,
            max_containers: 1,
            max_bandwidth_mbps: 100.0,
            max_gpus: 1,
        };

        let lease_duration = Duration::from_secs(3600); // 1 hour lease

        info!("[EVAL_FLOW] Requesting executors with requirements: cpu_cores={}, memory_mb={}, storage_mb={}, max_gpus={}, lease_duration={:?}",
              requirements.max_cpu_cores, requirements.max_memory_mb, requirements.max_storage_mb,
              requirements.max_gpus, lease_duration);

        let request_start = std::time::Instant::now();
        let executor_details = match connection
            .request_executors(Some(requirements), lease_duration)
            .await
        {
            Ok(details) => {
                info!(
                    "[EVAL_FLOW] Successfully received executor details in {:?}, count={}",
                    request_start.elapsed(),
                    details.len()
                );
                for (i, detail) in details.iter().enumerate() {
                    info!(
                        "[EVAL_FLOW] Executor {}: id={}, grpc_endpoint={}",
                        i, detail.executor_id, detail.grpc_endpoint
                    );
                }
                details
            }
            Err(e) => {
                error!(
                    "[EVAL_FLOW] Failed to request executors from miner after {:?}: {}",
                    request_start.elapsed(),
                    e
                );
                return Ok(vec![]);
            }
        };

        let executor_count = executor_details.len();
        let executors: Vec<ExecutorInfoDetailed> = executor_details
            .into_iter()
            .map(|details| -> Result<ExecutorInfoDetailed> {
                Ok(ExecutorInfoDetailed {
                    id: ExecutorId::from_str(&details.executor_id).map_err(|e| {
                        anyhow::anyhow!("Invalid executor ID '{}': {}", details.executor_id, e)
                    })?,
                    status: "available".to_string(),
                    capabilities: vec!["gpu".to_string()],
                    grpc_endpoint: details.grpc_endpoint,
                })
            })
            .collect::<Result<Vec<_>, _>>()?;

        info!(
            "[EVAL_FLOW] Executor discovery completed: {} executors mapped from {} details",
            executors.len(),
            executor_count
        );

        Ok(executors)
    }

    /// Clean up GPU assignments for an executor
    async fn cleanup_gpu_assignments(
        &self,
        executor_id: &str,
        miner_id: &str,
        tx: Option<&mut sqlx::Transaction<'_, sqlx::Sqlite>>,
    ) -> Result<u64> {
        let query = "DELETE FROM gpu_uuid_assignments WHERE executor_id = ? AND miner_id = ?";

        let rows_affected = if let Some(transaction) = tx {
            sqlx::query(query)
                .bind(executor_id)
                .bind(miner_id)
                .execute(&mut **transaction)
                .await?
                .rows_affected()
        } else {
            sqlx::query(query)
                .bind(executor_id)
                .bind(miner_id)
                .execute(self.persistence.pool())
                .await?
                .rows_affected()
        };

        if rows_affected > 0 {
            info!(
                "Cleaned up {} GPU assignments for executor {} (miner: {})",
                rows_affected, executor_id, miner_id
            );
        }

        Ok(rows_affected)
    }

    /// Helper function to clean up active SSH session for an executor (legacy method)
    async fn cleanup_active_session(&self, executor_id: &str) {
        self.ssh_session_manager.release_session(executor_id).await;
    }

    /// Store executor verification result with actual miner information
    pub async fn store_executor_verification_result_with_miner_info(
        &self,
        miner_uid: u16,
        executor_result: &ExecutorVerificationResult,
        miner_info: &super::types::MinerInfo,
    ) -> Result<()> {
        info!(
            miner_uid = miner_uid,
            executor_id = %executor_result.executor_id,
            verification_score = executor_result.verification_score,
            validation_type = %executor_result.validation_type,
            "Storing executor verification result to database for miner {}, executor {}: score={:.2}",
            miner_uid, executor_result.executor_id, executor_result.verification_score
        );

        // Create verification log entry for database storage
        let success = match executor_result.validation_type {
            ValidationType::Lightweight => executor_result.ssh_connection_successful,
            ValidationType::Full => {
                executor_result.ssh_connection_successful
                    && executor_result.binary_validation_successful
            }
        };

        let verification_log = VerificationLog::new(
            executor_result.executor_id.to_string(),
            self.validator_hotkey.to_string(),
            "ssh_automation".to_string(),
            executor_result.verification_score,
            success,
            serde_json::json!({
                "miner_uid": miner_uid,
                "executor_id": executor_result.executor_id.to_string(),
                "ssh_connection_successful": executor_result.ssh_connection_successful,
                "binary_validation_successful": executor_result.binary_validation_successful,
                "verification_method": "ssh_automation",
                "executor_result": executor_result.executor_result,
                "gpu_count": executor_result.gpu_count,
                "score_details": {
                    "verification_score": executor_result.verification_score,
                    "ssh_score": if executor_result.ssh_connection_successful { 0.5 } else { 0.0 },
                    "binary_score": if executor_result.binary_validation_successful { 0.5 } else { 0.0 }
                }
            }),
            executor_result.execution_time.as_millis() as i64,
            if !executor_result.ssh_connection_successful {
                Some("SSH connection failed".to_string())
            } else if executor_result.validation_type == ValidationType::Full
                && !executor_result.binary_validation_successful
            {
                Some("Binary validation failed".to_string())
            } else {
                None
            },
        );

        // Store directly to database to avoid repository trait issues
        let query = r#"
            INSERT INTO verification_logs (
                id, executor_id, validator_hotkey, verification_type, timestamp,
                score, success, details, duration_ms, error_message, created_at, updated_at,
                last_binary_validation, last_binary_validation_score
            ) VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?)
        "#;

        let now = chrono::Utc::now().to_rfc3339();
        let success = verification_log.success;

        // Set binary validation timestamp and score if this was a successful binary validation
        let (binary_validation_time, binary_validation_score) =
            if success && executor_result.binary_validation_successful {
                (Some(now.clone()), Some(executor_result.verification_score))
            } else {
                (None, None)
            };

        if let Err(e) = sqlx::query(query)
            .bind(verification_log.id.to_string())
            .bind(&verification_log.executor_id)
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
        let status = match (success, &executor_result.validation_type) {
            (false, _) => "offline".to_string(),
            (true, ValidationType::Full) => "online".to_string(),
            (true, ValidationType::Lightweight) => {
                match self
                    .persistence
                    .has_active_rental(&executor_result.executor_id.to_string(), &miner_id)
                    .await
                {
                    Ok(true) => "online".to_string(),
                    _ => sqlx::query_scalar::<_, String>(
                        "SELECT status FROM miner_executors WHERE miner_id = ? AND executor_id = ?",
                    )
                    .bind(&miner_id)
                    .bind(&verification_log.executor_id)
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
            executor_id = %executor_result.executor_id,
            validation_type = %executor_result.validation_type,
            new_status = %status,
            "Status update based on validation type"
        );

        // Use transaction to ensure atomic updates
        let mut tx = self.persistence.pool().begin().await?;

        // Update executor status
        if let Err(e) = sqlx::query(
            "UPDATE miner_executors
             SET status = ?, last_health_check = ?, updated_at = ?
             WHERE executor_id = ?",
        )
        .bind(&status)
        .bind(&now)
        .bind(&now)
        .bind(&verification_log.executor_id)
        .execute(&mut *tx)
        .await
        {
            warn!("Failed to update executor health status: {}", e);
            tx.rollback().await?;
            return Err(anyhow::anyhow!("Failed to update executor status: {}", e));
        }

        // escape plan, if verification failed, clean up GPU assignments
        if !(success
            || executor_result.validation_type == ValidationType::Lightweight
                && executor_result.ssh_connection_successful)
        {
            self.cleanup_gpu_assignments(&verification_log.executor_id, &miner_id, Some(&mut tx))
                .await?;
            tx.commit().await?;
            return Ok(());
        }

        tx.commit().await?;

        let gpu_infos = executor_result
            .executor_result
            .as_ref()
            .map(|er| er.gpu_infos.clone())
            .unwrap_or_default();

        match executor_result.validation_type {
            ValidationType::Full => {
                info!(
                    security = true,
                    miner_uid = miner_uid,
                    executor_id = %executor_result.executor_id,
                    validation_type = "full",
                    gpu_count = gpu_infos.len(),
                    action = "processing_full_validation",
                    "Processing full validation for miner {}, executor {}",
                    miner_uid, executor_result.executor_id
                );

                self.ensure_miner_executor_relationship(
                    miner_uid,
                    &executor_result.executor_id.to_string(),
                    &executor_result.grpc_endpoint,
                    miner_info,
                )
                .await?;

                self.store_gpu_uuid_assignments(
                    miner_uid,
                    &executor_result.executor_id.to_string(),
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
                    total_score: executor_result.verification_score,
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

                // Set rental metrics for successfully validated executors
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

                        metrics.prometheus().record_executor_rental_status(
                            &executor_result.executor_id.to_string(),
                            miner_uid,
                            &gpu_type,
                            false, // is_rented = false (available for rental)
                        );

                        debug!(
                            "Set rental metric for validated executor {} (miner_uid: {}, gpu_type: {}, rented: false)",
                            executor_result.executor_id, miner_uid, gpu_type
                        );
                    }
                }
            }
            ValidationType::Lightweight => {
                info!(
                    security = true,
                    miner_uid = miner_uid,
                    executor_id = %executor_result.executor_id,
                    validation_type = "lightweight",
                    gpu_count = gpu_infos.len(),
                    action = "processing_lightweight_validation",
                    "Processing lightweight validation for miner {}, executor {}",
                    miner_uid, executor_result.executor_id
                );

                self.update_gpu_assignment_timestamps(
                    miner_uid,
                    &executor_result.executor_id.to_string(),
                    &gpu_infos,
                )
                .await?;
            }
        }

        info!(
            miner_uid = miner_uid,
            executor_id = %executor_result.executor_id,
            verification_score = executor_result.verification_score,
            validation_type = %executor_result.validation_type,
            "Executor verification result successfully stored to database for miner {}, executor {}: score={:.2}",
            miner_uid, executor_result.executor_id, executor_result.verification_score
        );

        Ok(())
    }

    /// Ensure miner-executor relationship exists
    async fn ensure_miner_executor_relationship(
        &self,
        miner_uid: u16,
        executor_id: &str,
        executor_grpc_endpoint: &str,
        miner_info: &super::types::MinerInfo,
    ) -> Result<()> {
        info!(
            miner_uid = miner_uid,
            executor_id = executor_id,
            "Ensuring miner-executor relationship for miner {} and executor {} with real data",
            miner_uid,
            executor_id
        );

        let miner_id = format!("miner_{miner_uid}");

        // First ensure the miner exists in miners table with real data
        self.ensure_miner_exists_with_info(miner_info).await?;

        // Check if relationship already exists
        let query =
            "SELECT COUNT(*) as count FROM miner_executors WHERE miner_id = ? AND executor_id = ?";
        let row = sqlx::query(query)
            .bind(&miner_id)
            .bind(executor_id)
            .fetch_one(self.persistence.pool())
            .await
            .map_err(|e| anyhow::anyhow!("Failed to check miner-executor relationship: {}", e))?;

        let count: i64 = row.get("count");

        if count == 0 {
            // Check if this grpc_address is already used by a different miner
            let existing_miner: Option<String> = sqlx::query_scalar(
                "SELECT miner_id FROM miner_executors WHERE grpc_address = ? AND miner_id != ? LIMIT 1"
            )
            .bind(executor_grpc_endpoint)
            .bind(&miner_id)
            .fetch_optional(self.persistence.pool())
            .await
            .map_err(|e| anyhow::anyhow!("Failed to check grpc_address uniqueness: {}", e))?;

            if let Some(other_miner) = existing_miner {
                return Err(anyhow::anyhow!(
                    "Cannot create executor relationship: grpc_address {} is already registered to {}",
                    executor_grpc_endpoint, other_miner
                ));
            }

            // Check if this is an executor ID change for the same miner
            let old_executor_id: Option<String> = sqlx::query_scalar(
                "SELECT executor_id FROM miner_executors WHERE grpc_address = ? AND miner_id = ?",
            )
            .bind(executor_grpc_endpoint)
            .bind(&miner_id)
            .fetch_optional(self.persistence.pool())
            .await
            .map_err(|e| anyhow::anyhow!("Failed to check for existing executor: {}", e))?;

            if let Some(old_id) = old_executor_id {
                info!(
                    "Miner {} is changing executor ID from {} to {} for endpoint {}",
                    miner_id, old_id, executor_id, executor_grpc_endpoint
                );

                let mut tx = self.persistence.pool().begin().await?;

                sqlx::query(
                    "UPDATE gpu_uuid_assignments SET executor_id = ? WHERE executor_id = ? AND miner_id = ?"
                )
                .bind(executor_id)
                .bind(&old_id)
                .bind(&miner_id)
                .execute(&mut *tx)
                .await?;

                sqlx::query("DELETE FROM miner_executors WHERE executor_id = ? AND miner_id = ?")
                    .bind(&old_id)
                    .bind(&miner_id)
                    .execute(&mut *tx)
                    .await?;

                tx.commit().await?;

                info!(
                    "Successfully migrated GPU assignments from executor {} to {}",
                    old_id, executor_id
                );
            }

            // Insert new relationship with required fields
            let insert_query = r#"
                INSERT OR IGNORE INTO miner_executors (
                    id, miner_id, executor_id, grpc_address, gpu_count,
                    status, created_at, updated_at
                ) VALUES (?, ?, ?, ?, ?, ?, datetime('now'), datetime('now'))
            "#;

            let relationship_id = format!("{miner_id}_{executor_id}");

            sqlx::query(insert_query)
                .bind(&relationship_id)
                .bind(&miner_id)
                .bind(executor_id)
                .bind(executor_grpc_endpoint)
                // -- these will be updated from verification details
                .bind(0) // gpu_count
                //---------
                .bind("online") // status - online until verification completes
                .execute(self.persistence.pool())
                .await
                .map_err(|e| {
                    anyhow::anyhow!("Failed to insert miner-executor relationship: {}", e)
                })?;

            info!(
                miner_uid = miner_uid,
                executor_id = executor_id,
                "Created miner-executor relationship: {} -> {} with endpoint {}",
                miner_id,
                executor_id,
                executor_grpc_endpoint
            );
        } else {
            debug!(
                miner_uid = miner_uid,
                executor_id = executor_id,
                "Miner-executor relationship already exists: {} -> {}",
                miner_id,
                executor_id
            );

            // Even if relationship exists, check for duplicates with same grpc_address
            let duplicate_check_query: &'static str =
                "SELECT id, executor_id FROM miner_executors WHERE grpc_address = ? AND id != ?";
            let relationship_id = format!("{miner_id}_{executor_id}");

            let duplicates = sqlx::query(duplicate_check_query)
                .bind(executor_grpc_endpoint)
                .bind(&relationship_id)
                .fetch_all(self.persistence.pool())
                .await
                .map_err(|e| anyhow::anyhow!("Failed to check for duplicate executors: {}", e))?;

            if !duplicates.is_empty() {
                let duplicate_count = duplicates.len();
                warn!(
                    "Found {} duplicate executors with same grpc_address {} for miner {}",
                    duplicate_count, executor_grpc_endpoint, miner_id
                );

                // Delete the duplicates to clean up fraudulent registrations
                for duplicate in duplicates {
                    let dup_id: String = duplicate.get("id");
                    let dup_executor_id: String = duplicate.get("executor_id");

                    warn!(
                        "Marking duplicate executor {} (id: {}) as offline with same grpc_address as {} for miner {}",
                        dup_executor_id, dup_id, executor_id, miner_id
                    );

                    sqlx::query("UPDATE miner_executors SET status = 'offline', last_health_check = datetime('now'), updated_at = datetime('now') WHERE id = ?")
                        .bind(&dup_id)
                        .execute(self.persistence.pool())
                        .await
                        .map_err(|e| {
                            anyhow::anyhow!("Failed to update duplicate executor status: {}", e)
                        })?;

                    // Also clean up associated GPU assignments for the duplicate
                    self.cleanup_gpu_assignments(&dup_executor_id, &miner_id, None)
                        .await
                        .map_err(|e| {
                            anyhow::anyhow!(
                                "Failed to clean up GPU assignments for duplicate: {}",
                                e
                            )
                        })?;
                }

                info!(
                    "Cleaned up {} duplicate executors for miner {} with grpc_address {}",
                    duplicate_count, miner_id, executor_grpc_endpoint
                );
            }
        }

        Ok(())
    }

    /// Store GPU UUID assignments for an executor
    async fn store_gpu_uuid_assignments(
        &self,
        miner_uid: u16,
        executor_id: &str,
        gpu_infos: &[GpuInfo],
    ) -> Result<()> {
        let miner_id = format!("miner_{miner_uid}");
        let now = chrono::Utc::now().to_rfc3339();

        // Collect all valid GPU UUIDs being reported
        let reported_gpu_uuids: Vec<String> = gpu_infos
            .iter()
            .filter(|g| !g.gpu_uuid.is_empty() && g.gpu_uuid != "Unknown UUID")
            .map(|g| g.gpu_uuid.clone())
            .collect();

        // Clean up GPU assignments based on what's reported
        if !reported_gpu_uuids.is_empty() {
            // Some GPUs reported - clean up any that are no longer reported
            let placeholders = reported_gpu_uuids
                .iter()
                .map(|_| "?")
                .collect::<Vec<_>>()
                .join(", ");
            let query = format!(
                "DELETE FROM gpu_uuid_assignments
                 WHERE miner_id = ? AND executor_id = ?
                 AND gpu_uuid NOT IN ({placeholders})"
            );

            let mut q = sqlx::query(&query).bind(&miner_id).bind(executor_id);

            for uuid in &reported_gpu_uuids {
                q = q.bind(uuid);
            }

            let deleted = q.execute(self.persistence.pool()).await?;

            if deleted.rows_affected() > 0 {
                info!(
                    "Cleaned up {} stale GPU assignments for {}/{}",
                    deleted.rows_affected(),
                    miner_id,
                    executor_id
                );
            }
        } else {
            // No GPUs reported - clean up all assignments for this executor
            let deleted_rows = self
                .cleanup_gpu_assignments(executor_id, &miner_id, None)
                .await?;

            if deleted_rows > 0 {
                info!(
                    "Cleaned up {} GPU assignments for {}/{} (no GPUs reported)",
                    deleted_rows, miner_id, executor_id
                );
            }
        }

        for gpu_info in gpu_infos {
            // Skip invalid UUIDs
            if gpu_info.gpu_uuid.is_empty() || gpu_info.gpu_uuid == "Unknown UUID" {
                continue;
            }

            // Check if this GPU UUID already exists
            let existing = sqlx::query(
                "SELECT miner_id, executor_id FROM gpu_uuid_assignments WHERE gpu_uuid = ?",
            )
            .bind(&gpu_info.gpu_uuid)
            .fetch_optional(self.persistence.pool())
            .await?;

            if let Some(row) = existing {
                let existing_miner_id: String = row.get("miner_id");
                let existing_executor_id: String = row.get("executor_id");

                if existing_miner_id != miner_id || existing_executor_id != executor_id {
                    // Check if the existing executor is still active
                    let executor_status_query =
                        "SELECT status FROM miner_executors WHERE executor_id = ? AND miner_id = ?";
                    let status_row = sqlx::query(executor_status_query)
                        .bind(&existing_executor_id)
                        .bind(&existing_miner_id)
                        .fetch_optional(self.persistence.pool())
                        .await?;

                    let can_reassign = if let Some(row) = status_row {
                        let status: String = row.get("status");
                        // Allow reassignment if executor is offline, failed, or stale
                        status == "offline" || status == "failed" || status == "stale"
                    } else {
                        // Executor doesn't exist in miner_executors table - allow reassignment
                        true
                    };

                    if can_reassign {
                        // GPU reassignment allowed - previous executor is inactive
                        info!(
                            security = true,
                            gpu_uuid = %gpu_info.gpu_uuid,
                            previous_miner_id = %existing_miner_id,
                            previous_executor_id = %existing_executor_id,
                            new_miner_id = %miner_id,
                            new_executor_id = %executor_id,
                            gpu_memory_gb = %gpu_info.gpu_memory_gb,
                            action = "gpu_assignment_reassigned",
                            reassignment_reason = "previous_executor_inactive",
                            "GPU {} reassigned from {}/{} to {}/{} (previous executor inactive)",
                            gpu_info.gpu_uuid,
                            existing_miner_id,
                            existing_executor_id,
                            miner_id,
                            executor_id
                        );

                        sqlx::query(
                            "UPDATE gpu_uuid_assignments
                             SET miner_id = ?, executor_id = ?, gpu_index = ?, gpu_name = ?,
                                 gpu_memory_gb = ?, last_verified = ?, updated_at = ?
                             WHERE gpu_uuid = ?",
                        )
                        .bind(&miner_id)
                        .bind(executor_id)
                        .bind(gpu_info.index as i32)
                        .bind(&gpu_info.gpu_name)
                        .bind(gpu_info.gpu_memory_gb)
                        .bind(&now)
                        .bind(&now)
                        .bind(&gpu_info.gpu_uuid)
                        .execute(self.persistence.pool())
                        .await?;
                    } else {
                        // Executor is still active - reject the reassignment
                        warn!(
                            security = true,
                            gpu_uuid = %gpu_info.gpu_uuid,
                            existing_miner_id = %existing_miner_id,
                            existing_executor_id = %existing_executor_id,
                            attempting_miner_id = %miner_id,
                            attempting_executor_id = %executor_id,
                            action = "gpu_assignment_rejected",
                            rejection_reason = "already_owned_by_active_executor",
                            "GPU UUID {} still owned by active executor {}/{}, rejecting claim from {}/{}",
                            gpu_info.gpu_uuid,
                            existing_miner_id,
                            existing_executor_id,
                            miner_id,
                            executor_id
                        );
                        // Skip this GPU - don't store it for the new claimant
                        continue;
                    }
                } else {
                    // Same owner - just update last_verified
                    sqlx::query(
                        "UPDATE gpu_uuid_assignments
                         SET gpu_memory_gb = ?, last_verified = ?, updated_at = ?
                         WHERE gpu_uuid = ?",
                    )
                    .bind(gpu_info.gpu_memory_gb)
                    .bind(&now)
                    .bind(&now)
                    .bind(&gpu_info.gpu_uuid)
                    .execute(self.persistence.pool())
                    .await?;
                }
            } else {
                // New GPU UUID - insert
                sqlx::query(
                    "INSERT INTO gpu_uuid_assignments
                     (gpu_uuid, gpu_index, executor_id, miner_id, gpu_name, gpu_memory_gb, last_verified, created_at, updated_at)
                     VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?)"
                )
                .bind(&gpu_info.gpu_uuid)
                .bind(gpu_info.index as i32)
                .bind(executor_id)
                .bind(&miner_id)
                .bind(&gpu_info.gpu_name)
                .bind(gpu_info.gpu_memory_gb)
                .bind(&now)
                .bind(&now)
                .bind(&now)
                .execute(self.persistence.pool())
                .await?;

                info!(
                    security = true,
                    gpu_uuid = %gpu_info.gpu_uuid,
                    gpu_index = gpu_info.index,
                    executor_id = %executor_id,
                    miner_id = %miner_id,
                    gpu_name = %gpu_info.gpu_name,
                    gpu_memory_gb = %gpu_info.gpu_memory_gb,
                    action = "gpu_assignment_created",
                    "Registered new GPU {} (index {}) for {}/{}",
                    gpu_info.gpu_uuid, gpu_info.index, miner_id, executor_id
                );
            }
        }

        // Update gpu_count in miner_executors based on actual GPU assignments
        let gpu_count = sqlx::query_scalar::<_, i64>(
            "SELECT COUNT(*) FROM gpu_uuid_assignments WHERE miner_id = ? AND executor_id = ?",
        )
        .bind(&miner_id)
        .bind(executor_id)
        .fetch_one(self.persistence.pool())
        .await?;

        // Status hierarchy: "online" > "verified" > "offline"
        let current_status = sqlx::query_scalar::<_, String>(
            "SELECT status FROM miner_executors WHERE miner_id = ? AND executor_id = ?",
        )
        .bind(&miner_id)
        .bind(executor_id)
        .fetch_one(self.persistence.pool())
        .await?;

        let new_status = match (current_status.as_str(), gpu_count > 0) {
            ("online", true) => "online",   // Keep online status if GPUs present
            ("verified", true) => "online", // Promote verified back to online if GPUs present
            ("online", false) => "offline", // Downgrade to offline if no GPUs
            (_, true) => "verified",        // Set verified if GPUs present and not online
            (_, false) => "offline",        // Set offline if no GPUs
        };

        sqlx::query(
            "UPDATE miner_executors SET gpu_count = ?, status = ?, updated_at = datetime('now')
             WHERE miner_id = ? AND executor_id = ?",
        )
        .bind(gpu_count as i32)
        .bind(new_status)
        .bind(&miner_id)
        .bind(executor_id)
        .execute(self.persistence.pool())
        .await?;

        if gpu_count > 0 {
            info!(
                security = true,
                executor_id = %executor_id,
                miner_id = %miner_id,
                gpu_count = gpu_count,
                new_status = %new_status,
                action = "executor_gpu_verification_success",
                "Executor {}/{} verified with {} GPUs, status: {}",
                miner_id, executor_id, gpu_count, new_status
            );
        } else {
            warn!(
                security = true,
                executor_id = %executor_id,
                miner_id = %miner_id,
                gpu_count = 0,
                new_status = %new_status,
                action = "executor_gpu_verification_failure",
                "Executor {}/{} has no GPUs, marking as {}",
                miner_id, executor_id, new_status
            );
        }

        // Validate that the GPU count matches the expected count
        let expected_gpu_count = gpu_infos
            .iter()
            .filter(|g| !g.gpu_uuid.is_empty() && g.gpu_uuid != "Unknown UUID")
            .count() as i64;

        if gpu_count != expected_gpu_count {
            warn!(
                "GPU assignment mismatch for {}/{}: stored {} GPUs but expected {}",
                miner_id, executor_id, gpu_count, expected_gpu_count
            );
        }

        // Fail verification if executor claims GPUs but none were stored
        if expected_gpu_count > 0 && gpu_count == 0 {
            error!(
                "Failed to store GPU assignments for {}/{}: expected {} GPUs but stored 0",
                miner_id, executor_id, expected_gpu_count
            );
            return Err(anyhow::anyhow!(
                "GPU assignment validation failed: no valid GPU UUIDs stored despite {} GPUs reported",
                expected_gpu_count
            ));
        }

        Ok(())
    }

    /// Update last_verified timestamp for existing GPU assignments
    async fn update_gpu_assignment_timestamps(
        &self,
        miner_uid: u16,
        executor_id: &str,
        gpu_infos: &[GpuInfo],
    ) -> Result<()> {
        let miner_id = format!("miner_{miner_uid}");
        let now = chrono::Utc::now().to_rfc3339();

        let reported_gpu_uuids: Vec<String> = gpu_infos
            .iter()
            .filter(|g| !g.gpu_uuid.is_empty() && g.gpu_uuid != "Unknown UUID")
            .map(|g| g.gpu_uuid.clone())
            .collect();

        if reported_gpu_uuids.is_empty() {
            debug!(
                "No valid GPU UUIDs reported for {}/{} in lightweight validation",
                miner_id, executor_id
            );
            return Ok(());
        }

        let placeholders = reported_gpu_uuids
            .iter()
            .map(|_| "?")
            .collect::<Vec<_>>()
            .join(", ");

        let query = format!(
            "UPDATE gpu_uuid_assignments
             SET last_verified = ?, updated_at = ?
             WHERE miner_id = ? AND executor_id = ? AND gpu_uuid IN ({placeholders})"
        );

        let mut q = sqlx::query(&query)
            .bind(&now)
            .bind(&now)
            .bind(&miner_id)
            .bind(executor_id);

        for uuid in &reported_gpu_uuids {
            q = q.bind(uuid);
        }

        let result = q.execute(self.persistence.pool()).await?;
        let updated_count = result.rows_affected();

        if updated_count > 0 {
            info!(
                security = true,
                miner_uid = miner_uid,
                executor_id = %executor_id,
                validation_type = "lightweight",
                updated_assignments = updated_count,
                action = "gpu_assignment_timestamp_updated",
                "Updated {} GPU assignment timestamps for {}/{} (lightweight validation)",
                updated_count, miner_id, executor_id
            );
        } else {
            debug!(
                security = true,
                miner_uid = miner_uid,
                executor_id = %executor_id,
                validation_type = "lightweight",
                "No GPU assignments found to update for {}/{} with {} reported UUIDs",
                miner_id,
                executor_id,
                reported_gpu_uuids.len()
            );
            if self
                .persistence
                .has_active_rental(executor_id, &miner_id)
                .await
                .unwrap_or(false)
            {
                self.store_gpu_uuid_assignments(miner_uid, executor_id, gpu_infos)
                    .await?;
            } else {
                debug!(
                    security = true,
                    miner_uid = miner_uid,
                    executor_id = %executor_id,
                    validation_type = "lightweight",
                    "Skipping GPU assignment creation in lightweight (no active rental)"
                );
            }
        }

        Ok(())
    }

    /// Ensure miner exists in miners table
    ///
    /// This function handles three scenarios:
    /// 1. if UID already exists with same hotkey -> Update data
    /// 2. if UID already exists with different hotkey -> Update to new hotkey (recycled UID)
    /// 3. if UID doesn't exist but hotkey does -> on re-registration, migrate the UID
    /// 4. if neither UID nor hotkey exist -> Create new miner
    async fn ensure_miner_exists_with_info(
        &self,
        miner_info: &super::types::MinerInfo,
    ) -> Result<()> {
        let new_miner_uid = format!("miner_{}", miner_info.uid.as_u16());
        let hotkey = miner_info.hotkey.to_string();

        // Step 1: handle recycled UIDs
        let existing_by_uid = self.check_miner_by_uid(&new_miner_uid).await?;

        if let Some((_, existing_hotkey)) = existing_by_uid {
            return self
                .handle_recycled_miner_uid(&new_miner_uid, &hotkey, &existing_hotkey, miner_info)
                .await;
        }

        // Step 2: handle UID changes when a hotkey moves to a new UID (re-registration)
        let existing_by_hotkey = self.check_miner_by_hotkey(&hotkey).await?;

        if let Some(old_miner_uid) = existing_by_hotkey {
            return self
                .handle_uid_change(&old_miner_uid, &new_miner_uid, &hotkey, miner_info)
                .await;
        }

        // Step 3: handle new miners when neither UID nor hotkey exist - create new miner
        self.create_new_miner(&new_miner_uid, &hotkey, miner_info)
            .await
    }

    /// Check if a miner with the given UID exists
    async fn check_miner_by_uid(&self, miner_uid: &str) -> Result<Option<(String, String)>> {
        let query = "SELECT id, hotkey FROM miners WHERE id = ?";
        let result = sqlx::query(query)
            .bind(miner_uid)
            .fetch_optional(self.persistence.pool())
            .await
            .map_err(|e| anyhow::anyhow!("Failed to check miner by uid: {}", e))?;

        Ok(result.map(|row| {
            let id: String = row.get("id");
            let hotkey: String = row.get("hotkey");
            (id, hotkey)
        }))
    }

    /// Check if a miner with the given hotkey exists
    async fn check_miner_by_hotkey(&self, hotkey: &str) -> Result<Option<String>> {
        let query = "SELECT id FROM miners WHERE hotkey = ?";
        let result = sqlx::query(query)
            .bind(hotkey)
            .fetch_optional(self.persistence.pool())
            .await
            .map_err(|e| anyhow::anyhow!("Failed to check miner by hotkey: {}", e))?;

        Ok(result.map(|row| row.get("id")))
    }

    /// Handle case where miner UID already exists
    async fn handle_recycled_miner_uid(
        &self,
        miner_uid: &str,
        new_hotkey: &str,
        existing_hotkey: &str,
        miner_info: &super::types::MinerInfo,
    ) -> Result<()> {
        if existing_hotkey != new_hotkey {
            // Case: Recycled UID - same UID but different hotkey
            info!(
                "Miner {} exists with old hotkey {}, updating to new hotkey {}",
                miner_uid, existing_hotkey, new_hotkey
            );

            let update_query = r#"
                UPDATE miners SET
                    hotkey = ?, endpoint = ?, verification_score = ?,
                    last_seen = datetime('now'), updated_at = datetime('now')
                WHERE id = ?
            "#;

            sqlx::query(update_query)
                .bind(new_hotkey)
                .bind(&miner_info.endpoint)
                .bind(miner_info.verification_score)
                .bind(miner_uid)
                .execute(self.persistence.pool())
                .await
                .map_err(|e| anyhow::anyhow!("Failed to update miner with new hotkey: {}", e))?;

            debug!("Updated miner {} with new hotkey and data", miner_uid);
        } else {
            // Case: Same miner, same hotkey - just update the data
            self.update_miner_data(miner_uid, miner_info).await?;
        }

        Ok(())
    }

    /// Handle case where hotkey exists but with different ID (UID change)
    async fn handle_uid_change(
        &self,
        old_miner_id: &str,
        new_miner_id: &str,
        hotkey: &str,
        miner_info: &super::types::MinerInfo,
    ) -> Result<()> {
        info!(
            "Detected UID change for hotkey {}: {} -> {}",
            hotkey, old_miner_id, new_miner_id
        );

        // Migrate the miner UID
        if let Err(e) = self
            .migrate_miner_uid(old_miner_id, new_miner_id, miner_info)
            .await
        {
            error!(
                "Failed to migrate miner UID from {} to {}: {}",
                old_miner_id, new_miner_id, e
            );
            return Err(e);
        }

        Ok(())
    }

    /// Update existing miner data
    async fn update_miner_data(
        &self,
        miner_id: &str,
        miner_info: &super::types::MinerInfo,
    ) -> Result<()> {
        let update_query = r#"
            UPDATE miners SET
                endpoint = ?, verification_score = ?,
                last_seen = datetime('now'), updated_at = datetime('now')
            WHERE id = ?
        "#;

        sqlx::query(update_query)
            .bind(&miner_info.endpoint)
            .bind(miner_info.verification_score)
            .bind(miner_id)
            .execute(self.persistence.pool())
            .await
            .map_err(|e| anyhow::anyhow!("Failed to update miner: {}", e))?;

        debug!("Updated miner record: {} with latest data", miner_id);
        Ok(())
    }

    /// Create a new miner record
    async fn create_new_miner(
        &self,
        miner_uid: &str,
        hotkey: &str,
        miner_info: &super::types::MinerInfo,
    ) -> Result<()> {
        let insert_query = r#"
            INSERT INTO miners (
                id, hotkey, endpoint, verification_score, uptime_percentage,
                last_seen, registered_at, updated_at, executor_info
            ) VALUES (?, ?, ?, ?, ?, datetime('now'), datetime('now'), datetime('now'), ?)
        "#;

        sqlx::query(insert_query)
            .bind(miner_uid)
            .bind(hotkey)
            .bind(&miner_info.endpoint)
            .bind(miner_info.verification_score)
            .bind(100.0) // uptime_percentage
            .bind("{}") // executor_info
            .execute(self.persistence.pool())
            .await
            .map_err(|e| anyhow::anyhow!("Failed to insert miner: {}", e))?;

        info!(
            "Created miner record: {} with hotkey {} and endpoint {}",
            miner_uid, hotkey, miner_info.endpoint
        );

        Ok(())
    }

    /// Migrate miner UID when it changes in the network
    async fn migrate_miner_uid(
        &self,
        old_miner_uid: &str,
        new_miner_uid: &str,
        miner_info: &super::types::MinerInfo,
    ) -> Result<()> {
        info!(
            "Starting UID migration: {} -> {} for hotkey {}",
            old_miner_uid, new_miner_uid, miner_info.hotkey
        );

        // Use a transaction to ensure atomicity
        let mut tx = self
            .persistence
            .pool()
            .begin()
            .await
            .map_err(|e| anyhow::anyhow!("Failed to begin transaction: {}", e))?;

        // 1. First, get the old miner data
        debug!("Fetching old miner record: {}", old_miner_uid);
        let get_old_miner = "SELECT * FROM miners WHERE id = ?";
        let old_miner_row = sqlx::query(get_old_miner)
            .bind(old_miner_uid)
            .fetch_optional(&mut *tx)
            .await
            .map_err(|e| anyhow::anyhow!("Failed to fetch old miner record: {}", e))?;

        if old_miner_row.is_none() {
            return Err(anyhow::anyhow!(
                "Old miner record not found: {}",
                old_miner_uid
            ));
        }

        let old_row = old_miner_row.unwrap();
        debug!("Found old miner record for migration");

        // 2. Check if any miner with this hotkey exists (including the target)
        debug!(
            "Checking for existing miners with hotkey: {}",
            miner_info.hotkey
        );
        let check_hotkey = "SELECT id FROM miners WHERE hotkey = ?";
        let all_with_hotkey = sqlx::query(check_hotkey)
            .bind(miner_info.hotkey.to_string())
            .fetch_all(&mut *tx)
            .await
            .map_err(|e| anyhow::anyhow!("Failed to check hotkey existence: {}", e))?;

        // Find if any of them is NOT the old miner
        let existing_with_hotkey = all_with_hotkey.into_iter().find(|row| {
            let id: String = row.get("id");
            id != old_miner_uid
        });

        let should_create_new = if let Some(row) = existing_with_hotkey {
            let existing_id: String = row.get("id");
            debug!(
                "Found existing miner with hotkey {}: id={}",
                miner_info.hotkey, existing_id
            );
            if existing_id == new_miner_uid {
                // The new miner record already exists, just need to delete old
                debug!("New miner record already exists with correct ID");
                false
            } else {
                // Another miner exists with this hotkey but different ID
                warn!(
                    "Cannot migrate: Another miner {} already exists with hotkey {} (trying to create {})",
                    existing_id, miner_info.hotkey, new_miner_uid
                );
                return Err(anyhow::anyhow!(
                    "Cannot migrate: Another miner {} already exists with hotkey {}",
                    existing_id,
                    miner_info.hotkey
                ));
            }
        } else {
            debug!(
                "No existing miner with hotkey {}, will create new record",
                miner_info.hotkey
            );
            true
        };

        // Extract old miner data we'll need
        let verification_score = old_row
            .try_get::<f64, _>("verification_score")
            .unwrap_or(0.0);
        let uptime_percentage = old_row
            .try_get::<f64, _>("uptime_percentage")
            .unwrap_or(100.0);
        let registered_at = old_row
            .try_get::<String, _>("registered_at")
            .unwrap_or_else(|_| chrono::Utc::now().to_rfc3339());
        let executor_info = old_row
            .try_get::<String, _>("executor_info")
            .unwrap_or_else(|_| "{}".to_string());

        // 3. Get all related data before deletion
        debug!("Fetching related executor data");
        let get_executors = "SELECT * FROM miner_executors WHERE miner_id = ?";
        let executors = sqlx::query(get_executors)
            .bind(old_miner_uid)
            .fetch_all(&mut *tx)
            .await
            .map_err(|e| anyhow::anyhow!("Failed to fetch executors: {}", e))?;

        debug!("Found {} executors to migrate", executors.len());

        // 4. Delete old miner record (this will CASCADE delete miner_executors and verification_requests)
        debug!("Deleting old miner record: {}", old_miner_uid);
        let delete_old_miner = "DELETE FROM miners WHERE id = ?";
        sqlx::query(delete_old_miner)
            .bind(old_miner_uid)
            .execute(&mut *tx)
            .await
            .map_err(|e| anyhow::anyhow!("Failed to delete old miner record: {}", e))?;

        debug!("Deleted old miner record and related data");

        // 5. Create new miner record if needed
        if should_create_new {
            debug!("Creating new miner record: {}", new_miner_uid);
            let insert_new_miner = r#"
                INSERT INTO miners (
                    id, hotkey, endpoint, verification_score, uptime_percentage,
                    last_seen, registered_at, updated_at, executor_info
                ) VALUES (?, ?, ?, ?, ?, datetime('now'), ?, datetime('now'), ?)
            "#;

            sqlx::query(insert_new_miner)
                .bind(new_miner_uid)
                .bind(miner_info.hotkey.to_string())
                .bind(&miner_info.endpoint)
                .bind(verification_score)
                .bind(uptime_percentage)
                .bind(registered_at)
                .bind(executor_info)
                .execute(&mut *tx)
                .await
                .map_err(|e| anyhow::anyhow!("Failed to create new miner record: {}", e))?;

            debug!("Successfully created new miner record");
        }

        // 6. Re-create executor relationships
        let mut executor_count = 0;
        for executor_row in executors {
            let executor_id: String = executor_row.get("executor_id");
            let grpc_address: String = executor_row.get("grpc_address");
            let gpu_count: i32 = executor_row.get("gpu_count");
            let status: String = executor_row
                .try_get("status")
                .unwrap_or_else(|_| "unknown".to_string());
            // Check if this grpc_address is already in use by another miner
            let existing_check = sqlx::query(
                "SELECT COUNT(*) as count FROM miner_executors WHERE grpc_address = ? AND miner_id != ?"
            )
            .bind(&grpc_address)
            .bind(new_miner_uid)
            .fetch_one(&mut *tx)
            .await?;

            let existing_count: i64 = existing_check.get("count");
            if existing_count > 0 {
                warn!(
                    "Skipping executor {} during UID migration: grpc_address {} already in use by another miner",
                    executor_id, grpc_address
                );
                continue;
            }

            let new_id = format!("{new_miner_uid}_{executor_id}");

            let insert_executor = r#"
                INSERT INTO miner_executors (
                    id, miner_id, executor_id, grpc_address, gpu_count,
                    status, last_health_check,
                    created_at, updated_at
                ) VALUES (?, ?, ?, ?, ?, ?, NULL, datetime('now'), datetime('now'))
            "#;

            sqlx::query(insert_executor)
                .bind(&new_id)
                .bind(new_miner_uid)
                .bind(&executor_id)
                .bind(&grpc_address)
                .bind(gpu_count)
                .bind(&status)
                .execute(&mut *tx)
                .await
                .map_err(|e| anyhow::anyhow!("Failed to recreate executor relationship: {}", e))?;

            executor_count += 1;
        }

        debug!("Recreated {} executor relationships", executor_count);

        // 7. Migrate GPU UUID assignments
        debug!(
            "Migrating GPU UUID assignments from {} to {}",
            old_miner_uid, new_miner_uid
        );
        let update_gpu_assignments = r#"
            UPDATE gpu_uuid_assignments
            SET miner_id = ?
            WHERE miner_id = ?
        "#;

        let gpu_result = sqlx::query(update_gpu_assignments)
            .bind(new_miner_uid)
            .bind(old_miner_uid)
            .execute(&mut *tx)
            .await?;

        debug!(
            "Migrated {} GPU UUID assignments",
            gpu_result.rows_affected()
        );

        // Commit the transaction
        debug!("Committing transaction");
        tx.commit()
            .await
            .map_err(|e| anyhow::anyhow!("Failed to commit transaction: {}", e))?;

        info!(
            "Successfully migrated miner UID: {} -> {}. Migrated {} executors",
            old_miner_uid, new_miner_uid, executor_count
        );

        Ok(())
    }

    /// Sync miners from metagraph to database
    pub async fn sync_miners_from_metagraph(&self, miners: &[MinerInfo]) -> Result<()> {
        info!("Syncing {} miners from metagraph to database", miners.len());

        for miner in miners {
            // Discovery already filters out miners without valid axon endpoints
            if let Err(e) = self.ensure_miner_exists_with_info(miner).await {
                warn!(
                    "Failed to sync miner {} to database: {}",
                    miner.uid.as_u16(),
                    e
                );
            } else {
                debug!(
                    "Successfully synced miner {} with endpoint {} to database",
                    miner.uid.as_u16(),
                    miner.endpoint
                );
            }
        }

        info!("Completed syncing miners from metagraph");
        Ok(())
    }

    /// Create authenticated miner client
    fn create_authenticated_client(&self) -> Result<MinerClient> {
        Ok(
            if let Some(ref bittensor_service) = self.bittensor_service {
                let signer = Box::new(super::miner_client::BittensorServiceSigner::new(
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
            },
        )
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
            ssh_key_manager,
            ssh_session_manager: Arc::new(SshSessionManager::new()),
            metrics: metrics.clone(),
            validation_strategy_selector: Arc::new(ValidationStrategySelector::new(
                config.clone(),
                persistence.clone(),
            )),
            validation_executor: Arc::new(tokio::sync::RwLock::new(ValidationExecutor::new(
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
        let mut executor = self.validation_executor.write().await;
        executor
            .initialize_server_mode(&self.config.binary_validation)
            .await?;
        info!("Validation server mode initialized successfully");
        Ok(())
    }

    /// Shutdown validation server cleanly
    pub async fn shutdown_validation_server(&self) -> Result<()> {
        let executor = self.validation_executor.read().await;
        executor.shutdown_server_mode().await
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
        self.validation_executor.read().await.metrics().clone()
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

    /// Clean up executors that have consecutive failed validations
    /// This is called periodically (every 15 minutes) to remove executors that:
    /// 1. Are offline and still have GPU assignments (immediate cleanup)
    /// 2. Have had 2+ consecutive failed validations with no successes (delete)
    /// 3. Have been offline for 30+ minutes (stale cleanup)
    pub async fn cleanup_failed_executors_after_failures(
        &self,
        consecutive_failures_threshold: i32,
    ) -> Result<()> {
        info!(
            "Running executor cleanup - checking for {} consecutive failures",
            consecutive_failures_threshold
        );

        // Step 1: Clean up any GPU assignments for offline executors (immediate fix)
        let offline_with_gpus_query = r#"
            SELECT DISTINCT me.executor_id, me.miner_id, COUNT(ga.gpu_uuid) as gpu_count
            FROM miner_executors me
            INNER JOIN gpu_uuid_assignments ga ON me.executor_id = ga.executor_id AND me.miner_id = ga.miner_id
            WHERE me.status = 'offline'
            GROUP BY me.executor_id, me.miner_id
        "#;

        let offline_with_gpus = sqlx::query(offline_with_gpus_query)
            .fetch_all(self.persistence.pool())
            .await?;

        let mut gpu_assignments_cleaned = 0;
        for row in offline_with_gpus {
            let executor_id: String = row.try_get("executor_id")?;
            let miner_id: String = row.try_get("miner_id")?;
            let gpu_count: i64 = row.try_get("gpu_count")?;

            info!(
                "Cleaning up {} GPU assignments for offline executor {} (miner: {})",
                gpu_count, executor_id, miner_id
            );

            let rows_cleaned = self
                .cleanup_gpu_assignments(&executor_id, &miner_id, None)
                .await?;
            gpu_assignments_cleaned += rows_cleaned;
        }

        // Step 1b: Clean up executors with mismatched GPU counts
        let mismatched_gpu_query = r#"
            SELECT me.executor_id, me.miner_id, me.gpu_count, me.status
            FROM miner_executors me
            WHERE me.gpu_count > 0
            AND NOT EXISTS (
                SELECT 1 FROM gpu_uuid_assignments ga
                WHERE ga.executor_id = me.executor_id AND ga.miner_id = me.miner_id
            )
        "#;

        let mismatched_executors = sqlx::query(mismatched_gpu_query)
            .fetch_all(self.persistence.pool())
            .await?;

        for row in mismatched_executors {
            let executor_id: String = row.try_get("executor_id")?;
            let miner_id: String = row.try_get("miner_id")?;
            let gpu_count: i32 = row.try_get("gpu_count")?;
            let status: String = row.try_get("status")?;

            warn!(
                "Executor {} (miner: {}) claims {} GPUs but has no assignments, status: {}. Resetting GPU count to 0",
                executor_id, miner_id, gpu_count, status
            );

            // Reset GPU count to 0 to reflect reality
            sqlx::query(
                "UPDATE miner_executors SET gpu_count = 0, updated_at = datetime('now')
                 WHERE executor_id = ? AND miner_id = ?",
            )
            .bind(&executor_id)
            .bind(&miner_id)
            .execute(self.persistence.pool())
            .await?;

            // Mark offline if they claim GPUs but have none
            if status == "online" || status == "verified" {
                sqlx::query(
                    "UPDATE miner_executors SET status = 'offline', updated_at = datetime('now')
                     WHERE executor_id = ? AND miner_id = ?",
                )
                .bind(&executor_id)
                .bind(&miner_id)
                .execute(self.persistence.pool())
                .await?;

                info!(
                    "Marked executor {} as offline (claimed {} GPUs but has 0 assignments)",
                    executor_id, gpu_count
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
                    SELECT 1 FROM miner_executors me
                    WHERE me.executor_id = gpu_uuid_assignments.executor_id
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
                "Cleaned up {} stale GPU assignments (not verified in last 6 hours or belonging to offline executors >2h)",
                stale_gpu_result.rows_affected()
            );
        }

        // Step 1d: Clean up GPU assignments from executors offline
        // Increased minimum cleanup time from 30 minutes to 2 hours
        let cleanup_minutes = self
            .config
            .gpu_assignment_cleanup_ttl
            .map(|d| d.as_secs() / 60)
            .unwrap_or(120)
            .max(120); // Ensure minimum 2 hours to reduce aggressive cleanup

        info!(
            "Cleaning GPU assignments from executors offline >{} minutes",
            cleanup_minutes
        );
        let stale_offline_query = format!(
            r#"
            SELECT DISTINCT me.executor_id, me.miner_id, COUNT(ga.gpu_uuid) as gpu_count
            FROM miner_executors me
            INNER JOIN gpu_uuid_assignments ga ON me.executor_id = ga.executor_id AND me.miner_id = ga.miner_id
            WHERE me.status = 'offline'
            AND (
                me.last_health_check < datetime('now', '-{cleanup_minutes} minutes')
                OR (me.last_health_check IS NULL AND me.updated_at < datetime('now', '-{cleanup_minutes} minutes'))
            )
            GROUP BY me.executor_id, me.miner_id
            "#
        );

        let stale_offline = sqlx::query(&stale_offline_query)
            .fetch_all(self.persistence.pool())
            .await?;

        let mut stale_gpu_cleaned = 0;
        for row in stale_offline {
            let executor_id: String = row.try_get("executor_id")?;
            let miner_id: String = row.try_get("miner_id")?;
            let gpu_count: i64 = row.try_get("gpu_count")?;

            info!(
                security = true,
                executor_id = %executor_id,
                miner_id = %miner_id,
                gpu_count = gpu_count,
                cleanup_minutes = cleanup_minutes,
                "Cleaning GPU assignments from executor offline >{}min", cleanup_minutes
            );

            let cleaned = self
                .cleanup_gpu_assignments(&executor_id, &miner_id, None)
                .await?;
            stale_gpu_cleaned += cleaned;
        }

        if stale_gpu_cleaned > 0 {
            info!(
                security = true,
                cleaned_count = stale_gpu_cleaned,
                cleanup_minutes = cleanup_minutes,
                "Cleaned {} GPU assignments from executors offline >{}min",
                stale_gpu_cleaned,
                cleanup_minutes
            );
        }

        // Step 2: Find and delete executors with consecutive failures
        let delete_executors_query = r#"
            WITH recent_verifications AS (
                SELECT
                    vl.executor_id,
                    vl.success,
                    vl.timestamp,
                    ROW_NUMBER() OVER (PARTITION BY vl.executor_id ORDER BY vl.timestamp DESC) as rn
                FROM verification_logs vl
                WHERE vl.timestamp > datetime('now', '-1 hour')
            )
            SELECT
                me.executor_id,
                me.miner_id,
                me.status,
                COALESCE(SUM(CASE WHEN rv.success = 0 AND rv.rn <= ? THEN 1 ELSE 0 END), 0) as consecutive_fails,
                COALESCE(SUM(CASE WHEN rv.success = 1 AND rv.rn <= ? THEN 1 ELSE 0 END), 0) as recent_successes,
                MAX(rv.timestamp) as last_verification
            FROM miner_executors me
            LEFT JOIN recent_verifications rv ON me.executor_id = rv.executor_id
            WHERE me.status = 'offline'
            GROUP BY me.executor_id, me.miner_id, me.status
            HAVING consecutive_fails >= ? AND recent_successes = 0
        "#;

        let executors_to_delete = sqlx::query(delete_executors_query)
            .bind(consecutive_failures_threshold)
            .bind(consecutive_failures_threshold)
            .bind(consecutive_failures_threshold)
            .fetch_all(self.persistence.pool())
            .await?;

        let mut deleted = 0;
        for row in executors_to_delete {
            let executor_id: String = row.try_get("executor_id")?;
            let miner_id: String = row.try_get("miner_id")?;
            let consecutive_fails: i64 = row.try_get("consecutive_fails")?;
            let last_verification: Option<String> = row.try_get("last_verification").ok();

            info!(
                "Permanently deleting executor {} (miner: {}) after {} consecutive failures, last seen: {}",
                executor_id, miner_id, consecutive_fails,
                last_verification.as_deref().unwrap_or("never")
            );

            // Use transaction to ensure atomic deletion
            let mut tx = self.persistence.pool().begin().await?;

            // Clean up any remaining GPU assignments
            self.cleanup_gpu_assignments(&executor_id, &miner_id, Some(&mut tx))
                .await?;

            // Delete the executor record
            sqlx::query("DELETE FROM miner_executors WHERE executor_id = ? AND miner_id = ?")
                .bind(&executor_id)
                .bind(&miner_id)
                .execute(&mut *tx)
                .await?;

            tx.commit().await?;
            deleted += 1;

            // Clean up any active SSH sessions
            self.cleanup_active_session(&executor_id).await;
        }

        // Step 3: Delete stale offline executors
        let cleanup_minutes = self
            .config
            .gpu_assignment_cleanup_ttl
            .map(|d| d.as_secs() / 60)
            .unwrap_or(120)
            .max(120);

        let stale_delete_query = format!(
            r#"
            DELETE FROM miner_executors
            WHERE status = 'offline'
            AND (
                last_health_check < datetime('now', '-{} minutes')
                OR (last_health_check IS NULL AND updated_at < datetime('now', '-{} minutes'))
            )
            "#,
            cleanup_minutes, cleanup_minutes
        );

        info!(
            "Deleting stale offline executors using {}min timeout (configurable via gpu_assignment_cleanup_ttl)",
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
                -- Miners with offline executors
                SELECT DISTINCT CAST(SUBSTR(miner_id, 7) AS INTEGER)
                FROM miner_executors
                WHERE status = 'offline'

                UNION

                -- Miners with non-empty GPU profiles but no active executors
                SELECT miner_uid
                FROM miner_gpu_profiles
                WHERE gpu_counts_json <> '{}'
                AND NOT EXISTS (
                    SELECT 1 FROM miner_executors
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
                "Deleted {} GPU assignments from offline executors",
                gpu_assignments_cleaned
            );
        }

        if deleted > 0 {
            info!(
                "Deleted {} executors with {} or more consecutive failures",
                deleted, consecutive_failures_threshold
            );
        }

        if stale_deleted > 0 {
            info!("Deleted {} stale offline executors", stale_deleted);
        }

        if gpu_assignments_cleaned == 0 && deleted == 0 && stale_deleted == 0 {
            debug!("No executors needed cleanup in this cycle");
        }

        Ok(())
    }

    /// Enhanced verify executor with SSH automation and binary validation
    pub async fn verify_executor(
        &self,
        miner_endpoint: &str,
        executor_info: &ExecutorInfoDetailed,
        miner_uid: u16,
        miner_hotkey: &str,
        intended_strategy: ValidationType,
    ) -> Result<ExecutorVerificationResult> {
        info!(
            miner_uid = miner_uid,
            executor_id = %executor_info.id,
            miner_endpoint = %miner_endpoint,
            "[EVAL_FLOW] Starting executor verification"
        );

        // Step 1: Determine validation strategy
        let strategy = match self
            .validation_strategy_selector
            .determine_validation_strategy(&executor_info.id.to_string(), miner_uid)
            .await
        {
            Ok(s) => s,
            Err(e) => {
                error!(
                    miner_uid = miner_uid,
                    executor_id = %executor_info.id,
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
                executor_id = %executor_info.id,
                determined_strategy = ?strategy,
                intended = ?intended_strategy,
                "[EVAL_FLOW] Strategy mismatch - executor needs different validation type"
            );

            return Err(anyhow::anyhow!(
                "Strategy mismatch: executor needs different validation type"
            ));
        }

        // Step 2: Establish miner connection first
        let client = self.create_authenticated_client()?;
        let mut connection = client
            .connect_and_authenticate(miner_endpoint, miner_hotkey)
            .await?;

        // Step 3: Acquire session lock
        self.ssh_session_manager
            .acquire_session(&executor_info.id.to_string())
            .await
            .map_err(|e| {
                anyhow::anyhow!(
                    "Failed to acquire SSH session for executor {}: {}",
                    executor_info.id,
                    e
                )
            })?;

        let (ssh_details, session_info) = if let Some(ref key_manager) = self.ssh_key_manager {
            let key_provider = crate::ssh::session::ValidatorSshKeyProvider::new(key_manager);
            let ssh_session = crate::ssh::session::SshSessionHelper::establish_ssh_session(
                &mut connection,
                &executor_info.id.to_string(),
                &self.validator_hotkey,
                &key_provider,
                None,
            )
            .await;
            match ssh_session {
                Ok(v) => v,
                Err(e) => {
                    // Release the session lock on failure to prevent deadlocks for this executor
                    self.ssh_session_manager
                        .release_session(&executor_info.id.to_string())
                        .await;
                    return Err(e);
                }
            }
        } else {
            self.ssh_session_manager
                .release_session(&executor_info.id.to_string())
                .await;
            return Err(anyhow::anyhow!("SSH key manager not available"));
        };

        // Step 4: Execute validation based on strategy
        let result = match strategy {
            ValidationStrategy::Lightweight {
                previous_score,
                executor_result,
                gpu_count,
                binary_validation_successful,
            } => {
                self.validation_executor
                    .read()
                    .await
                    .execute_lightweight_validation(
                        miner_uid,
                        executor_info,
                        &ssh_details,
                        &session_info,
                        previous_score,
                        executor_result,
                        gpu_count,
                        binary_validation_successful,
                        &self.validator_hotkey,
                        &self.config,
                    )
                    .await
            }
            ValidationStrategy::Full => {
                let binary_config = &self.config.binary_validation;
                self.validation_executor
                    .read()
                    .await
                    .execute_full_validation(
                        executor_info,
                        &ssh_details,
                        &session_info,
                        binary_config,
                        &self.validator_hotkey,
                        miner_uid,
                    )
                    .await
            }
        };

        // Step 5: Cleanup SSH session
        crate::ssh::session::SshSessionHelper::cleanup_ssh_session(
            &mut connection,
            &session_info,
            &self.validator_hotkey,
        )
        .await;

        // Step 6: Release session lock
        self.ssh_session_manager
            .release_session(&executor_info.id.to_string())
            .await;

        result
    }

    /// Convert database executor data to ExecutorInfoDetailed
    fn convert_db_data_to_executor_info(
        &self,
        db_data: Vec<(String, String, i32, String)>,
        _miner_uid: u16,
    ) -> Result<Vec<ExecutorInfoDetailed>> {
        let mut executors = Vec::new();

        for (executor_id, grpc_address, gpu_count, status) in db_data {
            let executor_id_parsed = ExecutorId::from_str(&executor_id)
                .map_err(|e| anyhow::anyhow!("Invalid executor ID '{}': {}", executor_id, e))?;

            executors.push(ExecutorInfoDetailed {
                id: executor_id_parsed,
                status,
                capabilities: if gpu_count > 0 {
                    vec!["gpu".to_string()]
                } else {
                    vec![]
                },
                grpc_endpoint: grpc_address,
            });
        }

        Ok(executors)
    }

    /// Combine discovered and known executor lists
    fn combine_executor_lists(
        &self,
        discovered: Vec<ExecutorInfoDetailed>,
        known: Vec<ExecutorInfoDetailed>,
    ) -> Vec<ExecutorInfoDetailed> {
        let mut combined = Vec::new();
        let mut seen_ids = std::collections::HashSet::new();

        for executor in discovered {
            if seen_ids.insert(executor.id.to_string()) {
                combined.push(executor);
            }
        }

        for executor in known {
            if seen_ids.insert(executor.id.to_string()) {
                combined.push(executor);
            }
        }

        combined
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
