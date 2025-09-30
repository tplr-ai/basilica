//! Validation Strategy Module
//!
//! Determines the appropriate validation strategy based on executor status,
//! validation history, and configuration settings. Also handles the execution
//! of different validation strategies (lightweight vs full validation).

use super::types::{
    ExecutorInfoDetailed, ExecutorResult, ExecutorVerificationResult, ValidationDetails,
    ValidationType,
};
use super::validation_binary::BinaryValidator;
use super::validation_docker::DockerCollector;
use super::validation_hardware::HardwareCollector;
use super::validation_nat::NatCollector;
use super::validation_network::NetworkProfileCollector;
use super::validation_speedtest::NetworkSpeedCollector;
use super::validation_states::{StateResult, ValidationState};
use super::validation_storage::StorageCollector;
use crate::config::VerificationConfig;
use crate::metrics::ValidatorMetrics;
use crate::persistence::SimplePersistence;
use crate::ssh::ValidatorSshClient;
use anyhow::Result;
use basilica_common::identity::Hotkey;
use basilica_common::ssh::SshConnectionDetails;
use sqlx::Row;
use std::sync::Arc;
use std::time::{Duration, Instant};
use tracing::{debug, error, info, warn};

/// Validation strategy to determine execution path
#[allow(clippy::large_enum_variant)]
#[derive(Debug, Clone)]
pub enum ValidationStrategy {
    /// Full binary validation required
    Full,
    /// Lightweight connectivity check only
    Lightweight {
        previous_score: f64,
        executor_result: Option<ExecutorResult>,
        gpu_count: u64,
        binary_validation_successful: bool,
    },
}

/// Validation strategy selector for determining appropriate validation approach
pub struct ValidationStrategySelector {
    config: VerificationConfig,
    persistence: Arc<SimplePersistence>,
}

/// Validation executor for running different validation strategies
pub struct ValidationExecutor {
    ssh_client: Arc<ValidatorSshClient>,
    binary_validator: BinaryValidator,
    hardware_collector: HardwareCollector,
    network_collector: NetworkProfileCollector,
    speedtest_collector: NetworkSpeedCollector,
    docker_collector: DockerCollector,
    nat_collector: NatCollector,
    storage_collector: StorageCollector,
    metrics: Option<Arc<ValidatorMetrics>>,
}

impl ValidationStrategySelector {
    /// Create new validation strategy selector
    pub fn new(config: VerificationConfig, persistence: Arc<SimplePersistence>) -> Self {
        Self {
            config,
            persistence,
        }
    }

    /// Determine validation strategy based on executor status and validation history
    pub async fn determine_validation_strategy(
        &self,
        executor_id: &str,
        miner_uid: u16,
    ) -> Result<ValidationStrategy> {
        let miner_id = format!("miner_{}", miner_uid);

        debug!(
            executor_id = executor_id,
            miner_uid = miner_uid,
            "[EVAL_FLOW] Determining validation strategy"
        );

        // Check if executor has an active rental
        let has_active_rental = self
            .persistence
            .has_active_rental(executor_id, &miner_id)
            .await
            .unwrap_or_else(|e| {
                warn!(
                    executor_id = executor_id,
                    miner_uid = miner_uid,
                    error = %e,
                    "[EVAL_FLOW] Failed to check for active rental, assuming no rental"
                );
                false
            });

        // If there's an active rental, skip binary validation check and go straight to lightweight
        if !has_active_rental {
            let needs_binary_validation = self
                .is_binary_validation_needed(executor_id, &miner_id, miner_uid)
                .await
                .unwrap_or_else(|e| {
                    error!(
                        executor_id = executor_id,
                        miner_uid = miner_uid,
                        error = %e,
                        "[EVAL_FLOW] Failed to determine if binary validation needed, defaulting to full"
                    );
                    true
                });

            if needs_binary_validation {
                info!(
                    security = true,
                    executor_id = executor_id,
                    miner_uid = miner_uid,
                    validation_strategy = "Full",
                    "[EVAL_FLOW] Strategy: Full validation required"
                );
                return Ok(ValidationStrategy::Full);
            }
        }

        let (previous_score, executor_result, gpu_count, binary_validation_successful) = match self
            .persistence
            .get_last_full_validation_data(executor_id, &miner_id)
            .await
        {
            Ok(Some((score, exec_result, gpu_cnt, binary_success))) => {
                (score, exec_result, gpu_cnt, binary_success)
            }
            Ok(None) => {
                // If no previous validation data and no active rental, require full validation
                if !has_active_rental {
                    debug!(
                        executor_id = executor_id,
                        miner_uid = miner_uid,
                        "[EVAL_FLOW] No previous validation data found - requiring full validation"
                    );
                    return Ok(ValidationStrategy::Full);
                }
                // For active rentals without previous data, use default values
                debug!(
                    executor_id = executor_id,
                    miner_uid = miner_uid,
                    "[EVAL_FLOW] Active rental with no previous validation data - using defaults"
                );
                (0.8, None, 0, false)
            }
            Err(e) => {
                // If we can't get previous data and no active rental, require full validation
                if !has_active_rental {
                    error!(
                        executor_id = executor_id,
                        miner_uid = miner_uid,
                        error = %e,
                        "[EVAL_FLOW] Failed to get previous validation data - requiring full validation"
                    );
                    return Ok(ValidationStrategy::Full);
                }
                // For active rentals, use defaults even if we can't get previous data
                warn!(
                    executor_id = executor_id,
                    miner_uid = miner_uid,
                    error = %e,
                    "[EVAL_FLOW] Active rental, failed to get previous data - using defaults"
                );
                (0.8, None, 0, false)
            }
        };

        info!(
            security = true,
            executor_id = executor_id,
            miner_uid = miner_uid,
            validation_strategy = "Lightweight",
            has_active_rental = has_active_rental,
            previous_score = previous_score,
            gpu_count = gpu_count,
            binary_validation_successful = binary_validation_successful,
            "[EVAL_FLOW] Strategy: Lightweight validation with previous validation data (has_active_rental: {})", has_active_rental
        );

        Ok(ValidationStrategy::Lightweight {
            previous_score,
            executor_result,
            gpu_count,
            binary_validation_successful,
        })
    }

    /// Check if binary validation is needed for an executor
    async fn is_binary_validation_needed(
        &self,
        executor_id: &str,
        miner_id: &str,
        miner_uid: u16,
    ) -> Result<bool> {
        let status_query =
            "SELECT status FROM miner_executors WHERE executor_id = ? AND miner_id = ?";
        let status_row = sqlx::query(status_query)
            .bind(executor_id)
            .bind(miner_id)
            .fetch_optional(self.persistence.pool())
            .await?;

        if let Some(row) = status_row {
            let status: String = row.get("status");
            if status != "online" && status != "verified" {
                debug!(
                    executor_id = executor_id,
                    miner_id = miner_id,
                    status = status,
                    "Binary validation needed - executor not in online/verified status"
                );
                return Ok(true);
            }
        } else {
            debug!(
                executor_id = executor_id,
                miner_id = miner_id,
                "Binary validation needed - executor not found in database"
            );
            return Ok(true);
        }

        let last_validation = self
            .get_last_binary_validation(executor_id, miner_uid)
            .await?;

        match last_validation {
            None => {
                debug!(
                    executor_id = executor_id,
                    miner_id = miner_id,
                    "Binary validation needed - no previous successful validation found"
                );
                Ok(true)
            }
            Some((timestamp, _score)) => {
                let elapsed = chrono::Utc::now() - timestamp;
                let validation_interval =
                    chrono::Duration::from_std(self.config.executor_validation_interval)
                        .map_err(|e| anyhow::anyhow!("Invalid validation interval: {}", e))?;

                let needs_validation = elapsed > validation_interval;
                debug!(
                    executor_id = executor_id,
                    miner_id = miner_id,
                    elapsed_secs = elapsed.num_seconds(),
                    interval_secs = validation_interval.num_seconds(),
                    needs_validation = needs_validation,
                    "Binary validation check - last validation was {} seconds ago",
                    elapsed.num_seconds()
                );
                Ok(needs_validation)
            }
        }
    }

    /// Get last successful binary validation for an executor
    async fn get_last_binary_validation(
        &self,
        executor_id: &str,
        miner_uid: u16,
    ) -> Result<Option<(chrono::DateTime<chrono::Utc>, f64)>> {
        debug!(
            executor_id = executor_id,
            miner_uid = miner_uid,
            "Attempting to find last binary validation for executor_id"
        );

        let query = r#"
            SELECT timestamp, score
            FROM verification_logs
            WHERE executor_id = ?
              AND success = 1
              AND verification_type = 'ssh_automation'
              AND (
                json_extract(details, '$.binary_validation_successful') = 1
                OR json_extract(details, '$.binary_validation_successful') = 'true'
              )
            ORDER BY timestamp DESC
            LIMIT 1
        "#;

        let row = sqlx::query(query)
            .bind(executor_id)
            .fetch_optional(self.persistence.pool())
            .await?;

        let row = if row.is_none() {
            debug!(
                executor_id = executor_id,
                miner_uid = miner_uid,
                "No validation found with composite executor_id, trying plain executor_id as fallback"
            );

            sqlx::query(query)
                .bind(executor_id)
                .fetch_optional(self.persistence.pool())
                .await?
        } else {
            debug!(
                executor_id = executor_id,
                miner_uid = miner_uid,
                "Found validation with composite executor_id"
            );
            row
        };

        if let Some(row) = row {
            let timestamp_str: String = row.get("timestamp");
            let score: f64 = row.get("score");

            let timestamp = chrono::DateTime::parse_from_rfc3339(&timestamp_str)
                .map_err(|e| anyhow::anyhow!("Invalid timestamp format: {}", e))?
                .with_timezone(&chrono::Utc);

            Ok(Some((timestamp, score)))
        } else {
            Ok(None)
        }
    }
}

impl ValidationExecutor {
    /// Create a new validation executor
    pub fn new(
        config: VerificationConfig,
        ssh_client: Arc<ValidatorSshClient>,
        metrics: Option<Arc<ValidatorMetrics>>,
        persistence: Arc<SimplePersistence>,
    ) -> Self {
        let binary_validator = BinaryValidator::new(ssh_client.clone());
        let hardware_collector = HardwareCollector::new(ssh_client.clone(), persistence.clone());
        let network_collector =
            NetworkProfileCollector::new(ssh_client.clone(), persistence.clone());
        let speedtest_collector =
            NetworkSpeedCollector::new(ssh_client.clone(), persistence.clone());
        let docker_collector = DockerCollector::new(
            ssh_client.clone(),
            persistence.clone(),
            config.docker_validation.docker_image.clone(),
            config.docker_validation.pull_timeout_secs,
        );
        let nat_collector = NatCollector::new(ssh_client.clone(), persistence.clone());
        let storage_collector = StorageCollector::new(
            ssh_client.clone(),
            persistence,
            config.storage_validation.min_required_storage_bytes,
        );

        Self {
            ssh_client,
            binary_validator,
            hardware_collector,
            network_collector,
            speedtest_collector,
            docker_collector,
            nat_collector,
            storage_collector,
            metrics,
        }
    }

    /// Initialize server mode for binary validator if configured
    pub async fn initialize_server_mode(
        &mut self,
        binary_config: &crate::config::BinaryValidationConfig,
    ) -> Result<()> {
        self.binary_validator
            .initialize_server_mode(binary_config)
            .await
    }

    /// Shutdown server mode cleanly
    pub async fn shutdown_server_mode(&self) -> Result<()> {
        self.binary_validator.shutdown().await
    }

    /// Get access to metrics for state tracking
    pub fn metrics(&self) -> &Option<Arc<ValidatorMetrics>> {
        &self.metrics
    }

    /// Execute lightweight validation (connectivity check only)
    #[allow(clippy::too_many_arguments)]
    pub async fn execute_lightweight_validation(
        &self,
        miner_uid: u16,
        executor_info: &ExecutorInfoDetailed,
        ssh_details: &SshConnectionDetails,
        _session_info: &basilica_protocol::miner_discovery::InitiateSshSessionResponse,
        previous_score: f64,
        executor_result: Option<ExecutorResult>,
        gpu_count: u64,
        _binary_validation_successful: bool,
        _validator_hotkey: &Hotkey,
        _config: &crate::config::VerificationConfig,
    ) -> Result<ExecutorVerificationResult> {
        info!(
            miner_uid = miner_uid,
            executor_id = %executor_info.id,
            previous_score = previous_score,
            "[EVAL_FLOW] Executing lightweight validation"
        );

        let total_start = Instant::now();
        let executor_id = executor_info.id.to_string();

        // Track state: Connecting
        if let Some(ref metrics) = self.metrics {
            metrics.prometheus().set_executor_validation_state(
                &executor_id,
                miner_uid,
                ValidationType::Lightweight,
                ValidationState::Connecting,
                StateResult::Current,
            );
        }

        let connectivity_successful = match self
            .ssh_client
            .execute_command(
                ssh_details,
                "nvidia-smi --query-gpu=uuid --format=csv,noheader",
                true,
            )
            .await
        {
            Ok(output) => {
                // Move to Connected state
                if let Some(ref metrics) = self.metrics {
                    metrics.prometheus().set_executor_validation_state(
                        &executor_id,
                        miner_uid,
                        ValidationType::Lightweight,
                        ValidationState::Connected,
                        StateResult::Current,
                    );
                }

                // Move to ConnectivityChecking state
                if let Some(ref metrics) = self.metrics {
                    metrics.prometheus().set_executor_validation_state(
                        &executor_id,
                        miner_uid,
                        ValidationType::Lightweight,
                        ValidationState::ConnectivityChecking,
                        StateResult::Current,
                    );
                }

                let lines: Vec<&str> = output
                    .lines()
                    .map(|l| l.trim())
                    .filter(|l| !l.is_empty())
                    .collect();
                let gpus_detected = lines.iter().filter(|l| l.starts_with("GPU-")).count();
                let gpu_present = gpus_detected > 0;
                info!(
                    miner_uid = miner_uid,
                    executor_id = %executor_info.id,
                    gpu_present = gpu_present,
                    gpus_detected = gpus_detected,
                    "[EVAL_FLOW] GPU availability check completed"
                );

                if !gpu_present {
                    // Failed at connectivity check
                    if let Some(ref metrics) = self.metrics {
                        metrics.prometheus().set_executor_validation_state(
                            &executor_id,
                            miner_uid,
                            ValidationType::Lightweight,
                            ValidationState::ConnectivityChecking,
                            StateResult::Failed,
                        );
                    }
                }
                gpu_present
            }
            Err(e) => {
                warn!(
                    miner_uid = miner_uid,
                    executor_id = %executor_info.id,
                    error = %e,
                    "[EVAL_FLOW] Lightweight connectivity check failed"
                );

                // Failed at Connecting stage
                if let Some(ref metrics) = self.metrics {
                    metrics.prometheus().set_executor_validation_state(
                        &executor_id,
                        miner_uid,
                        ValidationType::Lightweight,
                        ValidationState::Connecting,
                        StateResult::Failed,
                    );
                }

                false
            }
        };

        let nat_validation_successful = if !connectivity_successful {
            false
        } else {
            // Move to NatValidating state
            if let Some(ref metrics) = self.metrics {
                metrics.prometheus().set_executor_validation_state(
                    &executor_id,
                    miner_uid,
                    ValidationType::Lightweight,
                    ValidationState::NatValidating,
                    StateResult::Current,
                );
            }

            let nat_collector = self.nat_collector.clone();

            match nat_collector
                .collect_with_fallback(&executor_id, miner_uid, ssh_details)
                .await
            {
                Some(result) if result.is_accessible => {
                    info!(
                        miner_uid = miner_uid,
                        executor_id = %executor_info.id,
                        "[EVAL_FLOW] NAT validation successful"
                    );
                    true
                }
                _ => {
                    warn!(
                        miner_uid = miner_uid,
                        executor_id = %executor_info.id,
                        "[EVAL_FLOW] NAT validation failed"
                    );

                    // Failed at NAT stage
                    if let Some(ref metrics) = self.metrics {
                        metrics.prometheus().set_executor_validation_state(
                            &executor_id,
                            miner_uid,
                            ValidationType::Lightweight,
                            ValidationState::NatValidating,
                            StateResult::Failed,
                        );
                    }

                    false
                }
            }
        };

        let validation_successful = connectivity_successful && nat_validation_successful;
        if !validation_successful {
            error!(
                miner_uid = miner_uid,
                executor_id = executor_id,
                connectivity_successful = connectivity_successful,
                nat_validation_successful = nat_validation_successful,
                "[EVAL_FLOW] Critical validation failed during lightweight validation"
            );
        }

        let total_duration = total_start.elapsed();

        let verification_score = if validation_successful {
            previous_score
        } else {
            0.0
        };

        let details = ValidationDetails {
            ssh_test_duration: total_duration,
            binary_upload_duration: Duration::from_secs(0),
            binary_execution_duration: Duration::from_secs(0),
            total_validation_duration: total_duration,
            ssh_score: if validation_successful { 1.0 } else { 0.0 },
            binary_score: 0.0,
            combined_score: verification_score,
        };

        info!(
            miner_uid = miner_uid,
            executor_id = %executor_info.id,
            score = verification_score,
            duration_ms = total_duration.as_millis(),
            node_available = connectivity_successful,
            nat_validation_successful = nat_validation_successful,
            validation_successful = validation_successful,
            "[EVAL_FLOW] Lightweight validation completed"
        );

        // Record lightweight validation metrics
        if let Some(ref metrics) = self.metrics {
            metrics
                .business()
                .record_attestation_verification(
                    &executor_info.id.to_string(),
                    "connectivity_check",
                    validation_successful,
                    validation_successful,
                    false,
                )
                .await;
        }

        // Final state - only set if validation was successful
        if validation_successful {
            if let Some(ref metrics) = self.metrics {
                metrics.prometheus().set_executor_validation_state(
                    &executor_id,
                    miner_uid,
                    ValidationType::Lightweight,
                    ValidationState::Completed,
                    StateResult::Current,
                );
            }
        }
        // Note: If failed, we keep the state where it failed (don't move to Completed)

        Ok(ExecutorVerificationResult {
            executor_id: executor_info.id.clone(),
            grpc_endpoint: executor_info.grpc_endpoint.clone(),
            verification_score,
            ssh_connection_successful: validation_successful,
            binary_validation_successful: false,
            executor_result,
            error: if validation_successful {
                None
            } else if !connectivity_successful {
                Some("Connectivity check failed".to_string())
            } else {
                Some("NAT validation failed".to_string())
            },
            execution_time: total_duration,
            validation_details: details,
            gpu_count,
            validation_type: ValidationType::Lightweight,
        })
    }

    /// Execute full validation
    pub async fn execute_full_validation(
        &self,
        executor_info: &ExecutorInfoDetailed,
        ssh_details: &SshConnectionDetails,
        session_info: &basilica_protocol::miner_discovery::InitiateSshSessionResponse,
        binary_config: &crate::config::BinaryValidationConfig,
        _validator_hotkey: &Hotkey,
        miner_uid: u16,
    ) -> Result<ExecutorVerificationResult> {
        info!(
            miner_uid = miner_uid,
            executor_id = %executor_info.id,
            "[EVAL_FLOW] Executing full validation"
        );

        let total_start = Instant::now();
        let executor_id = executor_info.id.to_string();
        let mut validation_details = ValidationDetails {
            ssh_test_duration: Duration::from_secs(0),
            binary_upload_duration: Duration::from_secs(0),
            binary_execution_duration: Duration::from_secs(0),
            total_validation_duration: Duration::from_secs(0),
            ssh_score: 0.0,
            binary_score: 0.0,
            combined_score: 0.0,
        };

        // Track state: Connecting
        if let Some(ref metrics) = self.metrics {
            metrics.prometheus().set_executor_validation_state(
                &executor_id,
                miner_uid,
                ValidationType::Full,
                ValidationState::Connecting,
                StateResult::Current,
            );
        }

        // Phase 1: SSH Connection Test
        let ssh_test_start = Instant::now();
        let ssh_connection_successful: bool =
            match self.ssh_client.test_connection(ssh_details).await {
                Ok(_) => {
                    // Move to Connected state
                    if let Some(ref metrics) = self.metrics {
                        metrics.prometheus().set_executor_validation_state(
                            &executor_id,
                            miner_uid,
                            ValidationType::Full,
                            ValidationState::Connected,
                            StateResult::Current,
                        );
                    }

                    info!(
                        miner_uid = miner_uid,
                        executor_id = %executor_info.id,
                        "[EVAL_FLOW] SSH connection test successful"
                    );
                    true
                }
                Err(e) => {
                    // Failed at Connecting stage
                    if let Some(ref metrics) = self.metrics {
                        metrics.prometheus().set_executor_validation_state(
                            &executor_id,
                            miner_uid,
                            ValidationType::Full,
                            ValidationState::Connecting,
                            StateResult::Failed,
                        );
                    }

                    error!(
                        miner_uid = miner_uid,
                        executor_id = %executor_info.id,
                        error = %e,
                        "[EVAL_FLOW] SSH connection test failed"
                    );

                    false
                }
            };

        validation_details.ssh_test_duration = ssh_test_start.elapsed();
        validation_details.ssh_score = if ssh_connection_successful { 0.8 } else { 0.0 };

        // Phase 1.5: Node Profiling Collection
        let mut quality_validations_successful = false;
        if ssh_connection_successful {
            let executor_id = executor_info.id.to_string();

            // Move to DockerValidating state
            if let Some(ref metrics) = self.metrics {
                metrics.prometheus().set_executor_validation_state(
                    &executor_id,
                    miner_uid,
                    ValidationType::Full,
                    ValidationState::DockerValidating,
                    StateResult::Current,
                );
            }
            let hardware_collector = self.hardware_collector.clone();
            let network_collector = self.network_collector.clone();
            let speedtest_collector = self.speedtest_collector.clone();
            let docker_collector = self.docker_collector.clone();
            let nat_collector = self.nat_collector.clone();
            let storage_collector = self.storage_collector.clone();

            let hardware_future =
                hardware_collector.collect_with_fallback(&executor_id, miner_uid, ssh_details);
            let network_future =
                network_collector.collect_with_fallback(&executor_id, miner_uid, ssh_details);
            let speedtest_future =
                speedtest_collector.collect_with_fallback(&executor_id, miner_uid, ssh_details);
            let docker_future =
                docker_collector.collect_with_fallback(&executor_id, miner_uid, ssh_details);
            let nat_future =
                nat_collector.collect_with_fallback(&executor_id, miner_uid, ssh_details);
            let storage_future =
                storage_collector.collect_with_fallback(&executor_id, miner_uid, ssh_details);

            let (_hardware, _network, _speedtest, docker_result, nat_result, storage_result) = tokio::join!(
                hardware_future,
                network_future,
                speedtest_future,
                docker_future,
                nat_future,
                storage_future
            );

            // For now, I'm disabling storage validation from affecting the overall quality validation result
            // as we may have valid executors with <1TB storage that we want to allow
            // once we have a better understanding of the ecosystem we can re-enable this
            // quality_validations_successful = docker_result.is_some() && nat_result.is_some() && storage_result.is_some();
            let nat_successful = nat_result
                .as_ref()
                .map(|n| n.is_accessible)
                .unwrap_or(false);
            quality_validations_successful = docker_result.is_some() && nat_successful;

            // Track state transitions based on validation results
            if docker_result.is_none() {
                // Failed at DockerValidating stage
                if let Some(ref metrics) = self.metrics {
                    metrics.prometheus().set_executor_validation_state(
                        &executor_id,
                        miner_uid,
                        ValidationType::Full,
                        ValidationState::DockerValidating,
                        StateResult::Failed,
                    );
                }
            } else if !nat_successful {
                // Docker passed, move to NatValidating and mark as failed
                if let Some(ref metrics) = self.metrics {
                    metrics.prometheus().set_executor_validation_state(
                        &executor_id,
                        miner_uid,
                        ValidationType::Full,
                        ValidationState::NatValidating,
                        StateResult::Failed,
                    );
                }
            } else {
                // Both Docker and NAT passed, move to NatValidating as current
                if let Some(ref metrics) = self.metrics {
                    metrics.prometheus().set_executor_validation_state(
                        &executor_id,
                        miner_uid,
                        ValidationType::Full,
                        ValidationState::NatValidating,
                        StateResult::Current,
                    );
                }
            }

            if !quality_validations_successful {
                error!(
                    miner_uid = miner_uid,
                    executor_id = %executor_info.id,
                    docker_successful = docker_result.is_some(),
                    nat_successful = nat_result.is_some(),
                    storage_successful = storage_result.is_some(),
                    storage_available_tb = storage_result.as_ref().map(|s|
                        format!("{:.2}", s.available_bytes as f64 / 1024_f64.powi(4))
                    ).unwrap_or_else(|| "N/A".to_string()),
                    "[EVAL_FLOW] Critical pre-validations failed"
                );
            }
        }

        // Phase 2: Binary Validation
        let mut binary_validation_successful = false;
        let mut executor_result = None;
        let mut binary_score = 0.0;
        let mut gpu_count = 0u64;
        let pre_validations_successful =
            ssh_connection_successful && quality_validations_successful;

        if pre_validations_successful && binary_config.enabled {
            // Move to BinaryValidating state
            if let Some(ref metrics) = self.metrics {
                metrics.prometheus().set_executor_validation_state(
                    &executor_id,
                    miner_uid,
                    ValidationType::Full,
                    ValidationState::BinaryValidating,
                    StateResult::Current,
                );
            }

            match self
                .binary_validator
                .execute_binary_validation(
                    &executor_info.id.to_string(),
                    miner_uid,
                    ssh_details,
                    session_info,
                    binary_config,
                )
                .await
            {
                Ok(binary_result) => {
                    binary_validation_successful = binary_result.success;
                    executor_result = binary_result.executor_result;
                    binary_score = binary_result.validation_score;
                    gpu_count = binary_result.gpu_count;
                    validation_details.binary_execution_duration =
                        Duration::from_millis(binary_result.execution_time_ms);

                    if let Some(ref metrics) = self.metrics {
                        metrics
                            .business()
                            .record_attestation_verification(
                                &executor_info.id.to_string(),
                                "hardware_attestation",
                                binary_validation_successful,
                                true, // signature_valid - binary executed successfully
                                binary_validation_successful,
                            )
                            .await;
                    }
                }
                Err(e) => {
                    // Failed at BinaryValidating stage
                    if let Some(ref metrics) = self.metrics {
                        metrics.prometheus().set_executor_validation_state(
                            &executor_id,
                            miner_uid,
                            ValidationType::Full,
                            ValidationState::BinaryValidating,
                            StateResult::Failed,
                        );
                    }

                    error!(
                        miner_uid = miner_uid,
                        executor_id = %executor_info.id,
                        error = %e,
                        "[EVAL_FLOW] Binary validation failed"
                    );

                    if let Some(ref metrics) = self.metrics {
                        metrics
                            .business()
                            .record_attestation_verification(
                                &executor_info.id.to_string(),
                                "hardware_attestation",
                                false,
                                false,
                                false,
                            )
                            .await;
                    }
                }
            }
        } else if !binary_config.enabled && quality_validations_successful {
            binary_validation_successful = true;
            binary_score = 0.8;
        }

        let combined_score = self.calculate_combined_verification_score(
            validation_details.ssh_score,
            binary_score,
            pre_validations_successful,
            binary_validation_successful,
            binary_config,
        );

        validation_details.combined_score = combined_score;
        validation_details.binary_score = binary_score;
        validation_details.total_validation_duration = total_start.elapsed();

        // Record overall validation status
        let overall_success = pre_validations_successful && binary_validation_successful;

        // Final state - only set if validation was successful
        if overall_success {
            if let Some(ref metrics) = self.metrics {
                metrics.prometheus().set_executor_validation_state(
                    &executor_id,
                    miner_uid,
                    ValidationType::Full,
                    ValidationState::Completed,
                    StateResult::Current,
                );
            }
        }
        // Note: If failed, we keep the state where it failed (don't move to Completed)

        Ok(ExecutorVerificationResult {
            executor_id: executor_info.id.clone(),
            grpc_endpoint: executor_info.grpc_endpoint.clone(),
            verification_score: combined_score,
            ssh_connection_successful,
            binary_validation_successful: pre_validations_successful
                && binary_validation_successful,
            executor_result,
            error: None,
            execution_time: total_start.elapsed(),
            validation_details,
            gpu_count,
            validation_type: ValidationType::Full,
        })
    }

    /// Calculate validation score from raw GPU results
    pub fn calculate_validation_score_from_raw_results(
        &self,
        raw_json: &serde_json::Value,
    ) -> Result<f64> {
        let gpu_results = raw_json
            .get("gpu_results")
            .and_then(|v| v.as_array())
            .ok_or_else(|| anyhow::anyhow!("No gpu_results found in output"))?;

        if gpu_results.is_empty() {
            return Ok(0.0);
        }

        let mut total_score = 0.0;
        let gpu_count = gpu_results.len();

        for gpu_result in gpu_results {
            let mut gpu_score: f64 = 0.0;

            // Base score for successful execution
            gpu_score += 0.3;

            // Anti-debug check
            if gpu_result
                .get("anti_debug_passed")
                .and_then(|v| v.as_bool())
                .unwrap_or(false)
            {
                gpu_score += 0.2;
            }

            // SM utilization scoring
            if let Some(sm_util) = gpu_result.get("sm_utilization") {
                let avg_utilization = sm_util.get("avg").and_then(|v| v.as_f64()).unwrap_or(0.0);
                let sm_score = if avg_utilization > 0.8 {
                    0.2
                } else if avg_utilization > 0.6 {
                    0.1
                } else {
                    0.0
                };
                gpu_score += sm_score;
            }

            // Memory bandwidth scoring
            let bandwidth = gpu_result
                .get("memory_bandwidth_gbps")
                .and_then(|v| v.as_f64())
                .unwrap_or(0.0);
            let bandwidth_score = if bandwidth > 15000.0 {
                0.15
            } else if bandwidth > 10000.0 {
                0.1
            } else if bandwidth > 5000.0 {
                0.05
            } else {
                0.0
            };
            gpu_score += bandwidth_score;

            // Computation timing score
            let computation_time_ns = gpu_result
                .get("computation_time_ns")
                .and_then(|v| v.as_u64())
                .unwrap_or(0);
            let computation_time_ms = computation_time_ns / 1_000_000;
            let timing_score = if computation_time_ms > 10 && computation_time_ms < 5000 {
                0.05
            } else {
                0.0
            };
            gpu_score += timing_score;

            total_score += gpu_score.clamp(0.0, 1.0);
        }

        let average_score = total_score / gpu_count as f64;
        info!(
            "[EVAL_FLOW] Calculated validation score from {} GPUs: {:.3}",
            gpu_count, average_score
        );

        Ok(average_score)
    }

    /// Calculate combined verification score from SSH and binary validation
    pub fn calculate_combined_verification_score(
        &self,
        ssh_score: f64,
        binary_score: f64,
        pre_validations_successful: bool,
        binary_successful: bool,
        binary_config: &crate::config::BinaryValidationConfig,
    ) -> f64 {
        info!(
            "[EVAL_FLOW] Starting combined score calculation - pre-val: {:.3} (success: {}), Binary: {:.3} (success: {})",
            ssh_score, pre_validations_successful, binary_score, binary_successful
        );

        // If SSH fails, total score is 0
        if !pre_validations_successful {
            error!("[EVAL_FLOW] pre validations failed, returning combined score: 0.0");
            return 0.0;
        }

        // If binary validation is disabled, use SSH score only
        if !binary_config.enabled {
            info!(
                "[EVAL_FLOW] Binary validation disabled, using SSH score only: {:.3}",
                ssh_score
            );
            return ssh_score;
        }

        // If binary validation is enabled but failed, penalize but don't zero
        if !binary_successful {
            let penalized_score = ssh_score * 0.5;
            warn!(
                "[EVAL_FLOW] Binary validation failed, applying 50% penalty to SSH score: {:.3} -> {:.3}",
                ssh_score, penalized_score
            );
            return penalized_score;
        }

        // Calculate weighted combination
        let ssh_weight = 1.0 - binary_config.score_weight;
        let binary_weight = binary_config.score_weight;

        let combined_score = (ssh_score * ssh_weight) + (binary_score * binary_weight);

        info!(
            "[EVAL_FLOW] Combined score calculation: ({:.3} × {:.3}) + ({:.3} × {:.3}) = {:.3}",
            ssh_score, ssh_weight, binary_score, binary_weight, combined_score
        );

        // Ensure score is within bounds
        combined_score.clamp(0.0, 1.0)
    }
}
