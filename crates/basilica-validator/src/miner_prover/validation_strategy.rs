//! Validation Strategy Module
//!
//! Determines the appropriate validation strategy based on node status,
//! validation history, and configuration settings. Also handles the execution
//! of different validation strategies (lightweight vs full validation).

use super::types::{
    NodeInfoDetailed, NodeResult, NodeVerificationResult, ValidationDetails, ValidationType,
};
use super::validation_binary::BinaryValidator;
use super::validation_docker::DockerCollector;
use super::validation_hardware::HardwareCollector;
use super::validation_misbehaviour::Misbehaviour;
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
use sha2::{Digest, Sha256};
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
        node_result: Option<NodeResult>,
        gpu_count: u64,
        binary_validation_successful: bool,
    },
}

/// Validation strategy selector for determining appropriate validation approach
pub struct ValidationStrategySelector {
    config: VerificationConfig,
    persistence: Arc<SimplePersistence>,
}

/// Validation node for running different validation strategies
pub struct ValidationNode {
    ssh_client: Arc<ValidatorSshClient>,
    binary_validator: BinaryValidator,
    hardware_collector: HardwareCollector,
    network_collector: NetworkProfileCollector,
    speedtest_collector: NetworkSpeedCollector,
    docker_collector: DockerCollector,
    nat_collector: NatCollector,
    storage_collector: StorageCollector,
    misbehaviour_collector: Misbehaviour,
    metrics: Option<Arc<ValidatorMetrics>>,
    persistence: Arc<SimplePersistence>,
}

impl ValidationStrategySelector {
    /// Create new validation strategy selector
    pub fn new(config: VerificationConfig, persistence: Arc<SimplePersistence>) -> Self {
        Self {
            config,
            persistence,
        }
    }

    /// Determine validation strategy based on node status and validation history
    pub async fn determine_validation_strategy(
        &self,
        node_id: &str,
        miner_uid: u16,
    ) -> Result<ValidationStrategy> {
        let miner_id = format!("miner_{}", miner_uid);

        debug!(
            node_id = node_id,
            miner_uid = miner_uid,
            "[EVAL_FLOW] Determining validation strategy"
        );

        // Check if node has an active rental
        let has_active_rental = self
            .persistence
            .has_active_rental(node_id, &miner_id)
            .await
            .unwrap_or_else(|e| {
                warn!(
                    node_id = node_id,
                    miner_uid = miner_uid,
                    error = %e,
                    "[EVAL_FLOW] Failed to check for active rental, assuming no rental"
                );
                false
            });

        // If there's an active rental, skip binary validation check and go straight to lightweight
        if has_active_rental {
            // TODO: Consider deferring lightweight checks if a recent validation already succeeded.
        } else {
            let last_terminated_at = self
                .persistence
                .get_last_rental_terminated_at(node_id, &miner_id)
                .await
                .unwrap_or_else(|e| {
                    warn!(
                        node_id = node_id,
                        miner_uid = miner_uid,
                        error = %e,
                        "[EVAL_FLOW] Failed to read last rental termination time"
                    );
                    None
                });

            let last_full_validation_at = self
                .persistence
                .get_last_full_validation_timestamp(node_id, &miner_id)
                .await
                .unwrap_or_else(|e| {
                    warn!(
                        node_id = node_id,
                        miner_uid = miner_uid,
                        error = %e,
                        "[EVAL_FLOW] Failed to read last full validation time"
                    );
                    None
                });

            if let Some(terminated_at) = last_terminated_at {
                let should_force_full = last_full_validation_at
                    .map(|ts| ts < terminated_at)
                    .unwrap_or(true);
                if should_force_full {
                    info!(
                        security = true,
                        node_id = node_id,
                        miner_uid = miner_uid,
                        termination_time = %terminated_at,
                        "[EVAL_FLOW] Forcing full validation after rental termination"
                    );
                    return Ok(ValidationStrategy::Full);
                }
            }

            let needs_binary_validation = self
                .is_binary_validation_needed(node_id, &miner_id, miner_uid)
                .await
                .unwrap_or_else(|e| {
                    error!(
                        node_id = node_id,
                        miner_uid = miner_uid,
                        error = %e,
                        "[EVAL_FLOW] Failed to determine if binary validation needed, defaulting to full"
                    );
                    true
                });

            if needs_binary_validation {
                info!(
                    security = true,
                    node_id = node_id,
                    miner_uid = miner_uid,
                    validation_strategy = "Full",
                    "[EVAL_FLOW] Strategy: Full validation required"
                );
                return Ok(ValidationStrategy::Full);
            }
        }

        let (previous_score, node_result, gpu_count, binary_validation_successful) = match self
            .persistence
            .get_last_full_validation_data(node_id, &miner_id)
            .await
        {
            Ok(Some((score, exec_result, gpu_cnt, binary_success))) => {
                (score, exec_result, gpu_cnt, binary_success)
            }
            Ok(None) => {
                // If no previous validation data and no active rental, require full validation
                if !has_active_rental {
                    debug!(
                        node_id = node_id,
                        miner_uid = miner_uid,
                        "[EVAL_FLOW] No previous validation data found - requiring full validation"
                    );
                    return Ok(ValidationStrategy::Full);
                }
                // For active rentals without previous data, use default values
                debug!(
                    node_id = node_id,
                    miner_uid = miner_uid,
                    "[EVAL_FLOW] Active rental with no previous validation data - using defaults"
                );
                (0.8, None, 0, false)
            }
            Err(e) => {
                // If we can't get previous data and no active rental, require full validation
                if !has_active_rental {
                    error!(
                        node_id = node_id,
                        miner_uid = miner_uid,
                        error = %e,
                        "[EVAL_FLOW] Failed to get previous validation data - requiring full validation"
                    );
                    return Ok(ValidationStrategy::Full);
                }
                // For active rentals, use defaults even if we can't get previous data
                warn!(
                    node_id = node_id,
                    miner_uid = miner_uid,
                    error = %e,
                    "[EVAL_FLOW] Active rental, failed to get previous data - using defaults"
                );
                (0.8, None, 0, false)
            }
        };

        info!(
            security = true,
            node_id = node_id,
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
            node_result,
            gpu_count,
            binary_validation_successful,
        })
    }

    /// Check if binary validation is needed for an node
    async fn is_binary_validation_needed(
        &self,
        node_id: &str,
        miner_id: &str,
        miner_uid: u16,
    ) -> Result<bool> {
        let status_query = "SELECT status FROM miner_nodes WHERE node_id = ? AND miner_id = ?";
        let status_row = sqlx::query(status_query)
            .bind(node_id)
            .bind(miner_id)
            .fetch_optional(self.persistence.pool())
            .await?;

        if let Some(row) = status_row {
            let status: String = row.get("status");
            if status != "online" && status != "verified" {
                debug!(
                    node_id = node_id,
                    miner_id = miner_id,
                    status = status,
                    "[EVAL_FLOW] Binary validation needed - node not in online/verified status"
                );
                return Ok(true);
            }
        } else {
            debug!(
                node_id = node_id,
                miner_id = miner_id,
                "[EVAL_FLOW] Binary validation needed - node not found in database"
            );
            return Ok(true);
        }

        // If node has no GPU assignments, it needs full validation
        if !self
            .persistence
            .node_has_gpu_assignments(miner_id, node_id)
            .await
            .unwrap_or(false)
        {
            info!(
                node_id = node_id,
                miner_id = miner_id,
                "[EVAL_FLOW] Binary validation needed - node has no GPU UUID assignments"
            );
            return Ok(true);
        }

        let last_validation = self.get_last_binary_validation(node_id, miner_uid).await?;

        match last_validation {
            None => {
                debug!(
                    node_id = node_id,
                    miner_id = miner_id,
                    "[EVAL_FLOW] Binary validation needed - no previous successful validation found"
                );
                Ok(true)
            }
            Some((timestamp, _score)) => {
                let elapsed = chrono::Utc::now() - timestamp;
                let validation_interval =
                    chrono::Duration::from_std(self.config.node_validation_interval)
                        .map_err(|e| anyhow::anyhow!("Invalid validation interval: {}", e))?;

                let needs_validation = elapsed > validation_interval;
                debug!(
                    node_id = node_id,
                    miner_id = miner_id,
                    elapsed_secs = elapsed.num_seconds(),
                    interval_secs = validation_interval.num_seconds(),
                    needs_validation = needs_validation,
                    "[EVAL_FLOW] Binary validation check - last validation was {} seconds ago",
                    elapsed.num_seconds()
                );
                Ok(needs_validation)
            }
        }
    }

    /// Get last successful binary validation for an node
    async fn get_last_binary_validation(
        &self,
        node_id: &str,
        miner_uid: u16,
    ) -> Result<Option<(chrono::DateTime<chrono::Utc>, f64)>> {
        debug!(
            node_id = node_id,
            miner_uid = miner_uid,
            "Attempting to find last binary validation for node_id"
        );

        let query = r#"
            SELECT timestamp, score
            FROM verification_logs
            WHERE node_id = ?
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
            .bind(node_id)
            .fetch_optional(self.persistence.pool())
            .await?;

        let row = if row.is_none() {
            debug!(
                node_id = node_id,
                miner_uid = miner_uid,
                "No validation found with composite node_id, trying plain node_id as fallback"
            );

            sqlx::query(query)
                .bind(node_id)
                .fetch_optional(self.persistence.pool())
                .await?
        } else {
            debug!(
                node_id = node_id,
                miner_uid = miner_uid,
                "Found validation with composite node_id"
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

impl ValidationNode {
    /// Create a new validation node
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
            persistence.clone(),
            config.storage_validation.min_required_storage_bytes,
        );
        let prometheus_metrics = metrics.as_ref().map(|m| m.prometheus());
        let misbehaviour_collector = Misbehaviour::new(persistence.clone(), prometheus_metrics);

        Self {
            ssh_client,
            binary_validator,
            hardware_collector,
            network_collector,
            speedtest_collector,
            docker_collector,
            nat_collector,
            storage_collector,
            misbehaviour_collector,
            metrics,
            persistence,
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
        node_info: &NodeInfoDetailed,
        ssh_details: &SshConnectionDetails,
        previous_score: f64,
        node_result: Option<NodeResult>,
        gpu_count: u64,
        _binary_validation_successful: bool,
        _validator_hotkey: &Hotkey,
        _config: &crate::config::VerificationConfig,
    ) -> Result<NodeVerificationResult> {
        info!(
            miner_uid = miner_uid,
            node_id = %node_info.id,
            previous_score = previous_score,
            "[EVAL_FLOW] Executing lightweight validation"
        );

        let total_start = Instant::now();
        let node_id = node_info.id.to_string();

        // Track state: Connecting
        if let Some(ref metrics) = self.metrics {
            metrics.prometheus().set_node_validation_state(
                &node_id,
                miner_uid,
                ValidationType::Lightweight,
                ValidationState::Connecting,
                StateResult::Current,
            );
        }

        let mut nvidia_smi_output: Option<String> = None;
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
                nvidia_smi_output = Some(output.clone());
                // Move to Connected state
                if let Some(ref metrics) = self.metrics {
                    metrics.prometheus().set_node_validation_state(
                        &node_id,
                        miner_uid,
                        ValidationType::Lightweight,
                        ValidationState::Connected,
                        StateResult::Current,
                    );
                }

                // Move to ConnectivityChecking state
                if let Some(ref metrics) = self.metrics {
                    metrics.prometheus().set_node_validation_state(
                        &node_id,
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
                    node_id = %node_info.id,
                    gpu_present = gpu_present,
                    gpus_detected = gpus_detected,
                    "[EVAL_FLOW] GPU availability check completed"
                );

                if !gpu_present {
                    // Failed at connectivity check
                    if let Some(ref metrics) = self.metrics {
                        metrics.prometheus().set_node_validation_state(
                            &node_id,
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
                    node_id = %node_info.id,
                    error = %e,
                    "[EVAL_FLOW] Lightweight connectivity check failed"
                );

                // Failed at Connecting stage
                if let Some(ref metrics) = self.metrics {
                    metrics.prometheus().set_node_validation_state(
                        &node_id,
                        miner_uid,
                        ValidationType::Lightweight,
                        ValidationState::Connecting,
                        StateResult::Failed,
                    );
                }

                false
            }
        };

        let nonce_challenge_ok = if connectivity_successful {
            let nonce = rand::random::<u64>();
            let mut hasher = Sha256::new();
            hasher.update(nonce.to_string().as_bytes());
            let expected = hex::encode(hasher.finalize());
            let nonce_cmd = format!("printf '{}' | sha256sum", nonce);
            match self
                .ssh_client
                .execute_command(ssh_details, &nonce_cmd, true)
                .await
            {
                Ok(output) => output
                    .split_whitespace()
                    .next()
                    .map(|v| v == expected)
                    .unwrap_or(false),
                Err(_) => false,
            }
        } else {
            false
        };

        let gpu_uuid_matches = if connectivity_successful {
            let miner_id = format!("miner_{miner_uid}");
            let known_uuids = self
                .persistence
                .get_node_gpu_uuids(&miner_id, &node_id)
                .await
                .unwrap_or_default();
            if known_uuids.is_empty() {
                // TODO: Decide whether to fail closed when no GPU UUIDs are on record.
                true
            } else {
                // nvidia-smi returns UUIDs with dashes (e.g., GPU-84ccface-663f-f5fd-8e8e-109d0f78bd2f)
                // but the CUDA API stores them without dashes (e.g., GPU-84ccface663ff5fd8e8e109d0f78bd2f)
                // Normalize by removing dashes for comparison
                let detected_uuids: Vec<String> = nvidia_smi_output
                    .as_deref()
                    .unwrap_or("")
                    .lines()
                    .map(|l| l.trim())
                    .filter(|l| l.starts_with("GPU-"))
                    .map(|l| l.replace('-', "").replacen("GPU", "GPU-", 1))
                    .collect();
                detected_uuids.iter().any(|u| known_uuids.contains(u))
            }
        } else {
            false
        };

        // Check if node is banned
        let misbehaviour_check_passed = match self
            .misbehaviour_collector
            .collect_with_fallback(&node_id, miner_uid)
            .await
        {
            Some(profile) if !profile.is_banned => {
                debug!(
                    miner_uid = miner_uid,
                    node_id = %node_info.id,
                    "[EVAL_FLOW] Misbehaviour: node is legit"
                );
                true
            }
            Some(profile) if profile.is_banned => {
                warn!(
                    miner_uid = miner_uid,
                    node_id = %node_info.id,
                    ban_expiry = ?profile.ban_expiry,
                    "[EVAL_FLOW] Misbehaviour: node is banned"
                );
                false
            }
            _ => {
                debug!(
                    miner_uid = miner_uid,
                    node_id = %node_info.id,
                    "[EVAL_FLOW] Misbehaviour check defaulted to pass"
                );
                true
            }
        };

        let nat_validation_successful = if !connectivity_successful
            || !misbehaviour_check_passed
            || !nonce_challenge_ok
            || !gpu_uuid_matches
        {
            false
        } else {
            // Move to NatValidating state
            if let Some(ref metrics) = self.metrics {
                metrics.prometheus().set_node_validation_state(
                    &node_id,
                    miner_uid,
                    ValidationType::Lightweight,
                    ValidationState::NatValidating,
                    StateResult::Current,
                );
            }

            let nat_collector = self.nat_collector.clone();

            match nat_collector
                .collect_with_fallback(&node_id, miner_uid, ssh_details)
                .await
            {
                Some(result) if result.is_accessible => {
                    info!(
                        miner_uid = miner_uid,
                        node_id = %node_info.id,
                        "[EVAL_FLOW] NAT validation successful"
                    );
                    true
                }
                _ => {
                    warn!(
                        miner_uid = miner_uid,
                        node_id = %node_info.id,
                        "[EVAL_FLOW] NAT validation failed"
                    );

                    // Failed at NAT stage
                    if let Some(ref metrics) = self.metrics {
                        metrics.prometheus().set_node_validation_state(
                            &node_id,
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

        let validation_successful = connectivity_successful
            && misbehaviour_check_passed
            && nonce_challenge_ok
            && gpu_uuid_matches
            && nat_validation_successful;
        if !validation_successful {
            error!(
                miner_uid = miner_uid,
                node_id = node_id,
                connectivity_successful = connectivity_successful,
                nat_validation_successful = nat_validation_successful,
                nonce_challenge_ok = nonce_challenge_ok,
                gpu_uuid_matches = gpu_uuid_matches,
                "[EVAL_FLOW] Critical validation failed during lightweight validation"
            );
        }

        let total_duration = total_start.elapsed();

        let verification_score = if validation_successful { 1.0 } else { 0.0 };
        let mut failure_reasons = Vec::new();
        if !validation_successful {
            if !connectivity_successful {
                failure_reasons.push("connectivity_failed".to_string());
            } else if !nonce_challenge_ok {
                failure_reasons.push("nonce_challenge_failed".to_string());
            } else if !gpu_uuid_matches {
                failure_reasons.push("gpu_uuid_mismatch".to_string());
            } else {
                failure_reasons.push("nat_validation_failed".to_string());
            }
        }

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
            node_id = %node_info.id,
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
                    &node_info.id.to_string(),
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
                metrics.prometheus().set_node_validation_state(
                    &node_id,
                    miner_uid,
                    ValidationType::Lightweight,
                    ValidationState::Completed,
                    StateResult::Current,
                );
            }
        }
        // Note: If failed, we keep the state where it failed (don't move to Completed)

        Ok(NodeVerificationResult {
            node_id: node_info.id.clone(),
            node_ssh_endpoint: node_info.node_ssh_endpoint.clone(),
            node_ip: node_info.node_ip.clone(),
            verification_score,
            ssh_connection_successful: validation_successful,
            binary_validation_successful: false,
            node_result,
            failure_reasons,
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
            hourly_rate_cents: node_info.hourly_rate_cents,
        })
    }

    /// Execute full validation
    pub async fn execute_full_validation(
        &self,
        node_info: &NodeInfoDetailed,
        ssh_details: &SshConnectionDetails,
        binary_config: Option<&crate::config::BinaryValidationConfig>,
        _validator_hotkey: &Hotkey,
        miner_uid: u16,
    ) -> Result<NodeVerificationResult> {
        info!(
            miner_uid = miner_uid,
            node_id = %node_info.id,
            binary_validation_enabled = binary_config.is_some(),
            binary_timeout_secs = binary_config.map(|c| c.execution_timeout_secs).unwrap_or(0),
            "[EVAL_FLOW] Executing full validation"
        );

        let total_start = Instant::now();
        let node_id = node_info.id.to_string();
        let mut validation_details = ValidationDetails {
            ssh_test_duration: Duration::from_secs(0),
            binary_upload_duration: Duration::from_secs(0),
            binary_execution_duration: Duration::from_secs(0),
            total_validation_duration: Duration::from_secs(0),
            ssh_score: 0.0,
            binary_score: 0.0,
            combined_score: 0.0,
        };
        let mut failure_reasons: Vec<String> = Vec::new();

        // Track state: Connecting
        if let Some(ref metrics) = self.metrics {
            metrics.prometheus().set_node_validation_state(
                &node_id,
                miner_uid,
                ValidationType::Full,
                ValidationState::Connecting,
                StateResult::Current,
            );
        }

        // Phase 1: SSH Connection Test
        let ssh_test_start = Instant::now();

        // Refresh host key to prevent stale key issues with validator binary
        if let Err(e) = self.ssh_client.refresh_host_key(ssh_details).await {
            warn!(
                miner_uid = miner_uid,
                node_id = %node_info.id,
                error = %e,
                "[EVAL_FLOW] Failed to refresh host key, continuing with validation"
            );
        }

        let ssh_connection_successful: bool =
            match self.ssh_client.test_connection(ssh_details).await {
                Ok(_) => {
                    // Move to Connected state
                    if let Some(ref metrics) = self.metrics {
                        metrics.prometheus().set_node_validation_state(
                            &node_id,
                            miner_uid,
                            ValidationType::Full,
                            ValidationState::Connected,
                            StateResult::Current,
                        );
                    }

                    info!(
                        miner_uid = miner_uid,
                        node_id = %node_info.id,
                        "[EVAL_FLOW] SSH connection test successful"
                    );
                    true
                }
                Err(e) => {
                    // Failed at Connecting stage
                    if let Some(ref metrics) = self.metrics {
                        metrics.prometheus().set_node_validation_state(
                            &node_id,
                            miner_uid,
                            ValidationType::Full,
                            ValidationState::Connecting,
                            StateResult::Failed,
                        );
                    }

                    error!(
                        miner_uid = miner_uid,
                        node_id = %node_info.id,
                        error = %e,
                        "[EVAL_FLOW] SSH connection test failed"
                    );

                    false
                }
            };

        validation_details.ssh_test_duration = ssh_test_start.elapsed();
        validation_details.ssh_score = if ssh_connection_successful { 1.0 } else { 0.0 };

        // Phase 1.5: Node Profiling Collection
        let mut quality_validations_successful = false;
        if ssh_connection_successful {
            // Move to DockerValidating state
            if let Some(ref metrics) = self.metrics {
                metrics.prometheus().set_node_validation_state(
                    &node_id,
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
            let misbehaviour_collector = self.misbehaviour_collector.clone();

            let hardware_future =
                hardware_collector.collect_with_fallback(&node_id, miner_uid, ssh_details);
            let network_future =
                network_collector.collect_with_fallback(&node_id, miner_uid, ssh_details);
            let speedtest_future =
                speedtest_collector.collect_with_fallback(&node_id, miner_uid, ssh_details);
            let docker_future =
                docker_collector.collect_with_fallback(&node_id, miner_uid, ssh_details);
            let nat_future = nat_collector.collect_with_fallback(&node_id, miner_uid, ssh_details);
            let storage_future =
                storage_collector.collect_with_fallback(&node_id, miner_uid, ssh_details);
            let misbehaviour_future =
                misbehaviour_collector.collect_with_fallback(&node_id, miner_uid);

            let (
                _hardware,
                _network,
                _speedtest,
                docker_result,
                nat_result,
                storage_result,
                misbehaviour_result,
            ) = tokio::join!(
                hardware_future,
                network_future,
                speedtest_future,
                docker_future,
                nat_future,
                storage_future,
                misbehaviour_future
            );

            // For now, I'm disabling storage validation from affecting the overall quality validation result
            // as we may have valid nodes with <1TB storage that we want to allow
            // once we have a better understanding of the ecosystem we can re-enable this
            // quality_validations_successful = docker_result.is_some() && nat_result.is_some() && storage_result.is_some();
            let nat_successful = nat_result
                .as_ref()
                .map(|n| n.is_accessible)
                .unwrap_or(false);
            let not_banned = misbehaviour_result
                .as_ref()
                .map(|m| !m.is_banned)
                .unwrap_or(true);
            quality_validations_successful =
                docker_result.is_some() && nat_successful && not_banned;

            if docker_result.is_none() {
                failure_reasons.push("docker_validation_failed".to_string());
            }
            if nat_result.is_none() || !nat_successful {
                failure_reasons.push("nat_validation_failed".to_string());
            }
            if storage_result.is_none() {
                failure_reasons.push("storage_validation_failed".to_string());
            }

            // Track state transitions based on validation results
            if docker_result.is_none() {
                // Failed at DockerValidating stage
                if let Some(ref metrics) = self.metrics {
                    metrics.prometheus().set_node_validation_state(
                        &node_id,
                        miner_uid,
                        ValidationType::Full,
                        ValidationState::DockerValidating,
                        StateResult::Failed,
                    );
                }
            } else if !nat_successful {
                // Docker passed, move to NatValidating and mark as failed
                if let Some(ref metrics) = self.metrics {
                    metrics.prometheus().set_node_validation_state(
                        &node_id,
                        miner_uid,
                        ValidationType::Full,
                        ValidationState::NatValidating,
                        StateResult::Failed,
                    );
                }
            } else if !not_banned {
                // Docker and NAT passed but node is banned
                if let Some(ref metrics) = self.metrics {
                    metrics.prometheus().set_node_validation_state(
                        &node_id,
                        miner_uid,
                        ValidationType::Full,
                        ValidationState::NatValidating,
                        StateResult::Failed,
                    );
                }
            } else {
                // All validations passed (Docker, NAT, and not banned)
                if let Some(ref metrics) = self.metrics {
                    metrics.prometheus().set_node_validation_state(
                        &node_id,
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
                    node_id = %node_info.id,
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
        let mut node_result = None;
        let mut binary_score = 0.0;
        let mut gpu_count = 0u64;
        if !ssh_connection_successful {
            failure_reasons.push("ssh_connection_failed".to_string());
        }
        let pre_validations_successful =
            ssh_connection_successful && quality_validations_successful;

        // Binary validation is enabled if config is present (Some = enabled)
        let binary_validation_enabled = binary_config.is_some();

        if pre_validations_successful && binary_validation_enabled {
            let binary_cfg = binary_config.unwrap(); // Safe because we checked is_some()
                                                     // Move to BinaryValidating state
            if let Some(ref metrics) = self.metrics {
                metrics.prometheus().set_node_validation_state(
                    &node_id,
                    miner_uid,
                    ValidationType::Full,
                    ValidationState::BinaryValidating,
                    StateResult::Current,
                );
            }

            match self
                .binary_validator
                .execute_binary_validation(
                    &node_info.id.to_string(),
                    miner_uid,
                    ssh_details,
                    binary_cfg,
                )
                .await
            {
                Ok(binary_result) => {
                    binary_validation_successful = binary_result.success;
                    node_result = binary_result.node_result;
                    binary_score = if binary_result.success { 1.0 } else { 0.0 };
                    gpu_count = binary_result.gpu_count;
                    failure_reasons = binary_result.failure_reasons.clone();
                    validation_details.binary_execution_duration =
                        Duration::from_millis(binary_result.execution_time_ms);
                    info!(
                        miner_uid = miner_uid,
                        node_id = %node_info.id,
                        binary_success = binary_result.success,
                        failure_reasons_count = failure_reasons.len(),
                        execution_ms = binary_result.execution_time_ms,
                        gpu_count = binary_result.gpu_count,
                        "[EVAL_FLOW] Binary validation completed"
                    );

                    if let Some(ref metrics) = self.metrics {
                        metrics
                            .business()
                            .record_attestation_verification(
                                &node_info.id.to_string(),
                                "hardware_attestation",
                                binary_validation_successful,
                                true, // signature_valid - binary executed successfully
                                binary_validation_successful,
                            )
                            .await;
                    }

                    // Set failed state if binary validation reported failure
                    if !binary_validation_successful {
                        if let Some(ref metrics) = self.metrics {
                            metrics.prometheus().set_node_validation_state(
                                &node_id,
                                miner_uid,
                                ValidationType::Full,
                                ValidationState::BinaryValidating,
                                StateResult::Failed,
                            );
                        }
                    }
                }
                Err(e) => {
                    // Failed at BinaryValidating stage
                    if let Some(ref metrics) = self.metrics {
                        metrics.prometheus().set_node_validation_state(
                            &node_id,
                            miner_uid,
                            ValidationType::Full,
                            ValidationState::BinaryValidating,
                            StateResult::Failed,
                        );
                    }

                    error!(
                        miner_uid = miner_uid,
                        node_id = %node_info.id,
                        error = %e,
                        "[EVAL_FLOW] Binary validation failed"
                    );
                    failure_reasons.push("binary_execution_failed".to_string());

                    if let Some(ref metrics) = self.metrics {
                        metrics
                            .business()
                            .record_attestation_verification(
                                &node_info.id.to_string(),
                                "hardware_attestation",
                                false,
                                false,
                                false,
                            )
                            .await;
                    }
                }
            }
        } else if !binary_validation_enabled && quality_validations_successful {
            // Binary validation is disabled - use SSH score only (0.8 max)
            binary_validation_successful = true;
            binary_score = 1.0;
        }

        let combined_score = self.calculate_combined_verification_score(
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
                metrics.prometheus().set_node_validation_state(
                    &node_id,
                    miner_uid,
                    ValidationType::Full,
                    ValidationState::Completed,
                    StateResult::Current,
                );
            }
        }
        // Note: If failed, we keep the state where it failed (don't move to Completed)

        let error_message = if pre_validations_successful && binary_validation_successful {
            None
        } else if !failure_reasons.is_empty() {
            Some(failure_reasons.join("; "))
        } else if !pre_validations_successful {
            Some("pre_validations_failed".to_string())
        } else {
            Some("binary_validation_failed".to_string())
        };

        Ok(NodeVerificationResult {
            node_id: node_info.id.clone(),
            node_ssh_endpoint: node_info.node_ssh_endpoint.clone(),
            node_ip: node_info.node_ip.clone(),
            verification_score: combined_score,
            ssh_connection_successful,
            binary_validation_successful: pre_validations_successful
                && binary_validation_successful,
            node_result,
            failure_reasons,
            error: error_message,
            execution_time: total_start.elapsed(),
            validation_details,
            gpu_count,
            validation_type: ValidationType::Full,
            hourly_rate_cents: node_info.hourly_rate_cents,
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
        pre_validations_successful: bool,
        binary_successful: bool,
        binary_config: Option<&crate::config::BinaryValidationConfig>,
    ) -> f64 {
        if !pre_validations_successful {
            return 0.0;
        }
        // Binary validation disabled if None
        if binary_config.is_none() {
            return 1.0;
        }
        if !binary_successful {
            return 0.0;
        }
        1.0
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use chrono::Utc;

    #[tokio::test]
    async fn forces_full_after_rental_termination() -> Result<()> {
        let persistence = Arc::new(SimplePersistence::for_testing().await?);
        let config = VerificationConfig::test_default();
        let selector = ValidationStrategySelector::new(config, persistence.clone());

        let miner_uid = 101u16;
        let miner_id = format!("miner_{}", miner_uid);
        let node_id = "node-test-1";

        sqlx::query(
            "INSERT INTO miners (id, hotkey, endpoint, updated_at)
             VALUES (?, ?, ?, ?)",
        )
        .bind(&miner_id)
        .bind("hotkey")
        .bind("http://127.0.0.1:8091")
        .bind(Utc::now().to_rfc3339())
        .execute(persistence.pool())
        .await?;

        // Seed miner_nodes so status check doesn't force full validation by itself.
        sqlx::query(
            "INSERT INTO miner_nodes (id, miner_id, node_id, ssh_endpoint, node_ip, gpu_count, hourly_rate_cents, status, created_at)
             VALUES (?, ?, ?, ?, ?, ?, ?, ?, datetime('now'))",
        )
        .bind(format!("{miner_id}_{node_id}"))
        .bind(&miner_id)
        .bind(node_id)
        .bind("root@127.0.0.1:22")
        .bind("127.0.0.1")
        .bind(1)
        .bind(100)
        .bind("online")
        .execute(persistence.pool())
        .await?;

        let last_validation = Utc::now() - chrono::Duration::hours(1);
        let details = serde_json::json!({
            "binary_validation_successful": true
        });
        sqlx::query(
            "INSERT INTO verification_logs (id, node_id, validator_hotkey, verification_type, timestamp, score, success, details, duration_ms, error_message, created_at, updated_at)
             VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?)",
        )
        .bind(uuid::Uuid::new_v4().to_string())
        .bind(node_id)
        .bind("validator_hotkey")
        .bind("ssh_automation")
        .bind(last_validation.to_rfc3339())
        .bind(1.0)
        .bind(1)
        .bind(details.to_string())
        .bind(1000)
        .bind(Option::<String>::None)
        .bind(last_validation.to_rfc3339())
        .bind(last_validation.to_rfc3339())
        .execute(persistence.pool())
        .await?;

        let terminated_at = Utc::now() - chrono::Duration::minutes(30);
        sqlx::query(
            "INSERT INTO rentals (id, validator_hotkey, node_id, miner_id, container_id, ssh_session_id, ssh_credentials, state, created_at, container_spec, metadata, terminated_at)
             VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?)",
        )
        .bind("rental-test-1")
        .bind("validator_hotkey")
        .bind(node_id)
        .bind(&miner_id)
        .bind("container")
        .bind("session")
        .bind("root@127.0.0.1:22")
        .bind("stopped")
        .bind(Utc::now().to_rfc3339())
        .bind("{}")
        .bind("{}")
        .bind(terminated_at.to_rfc3339())
        .execute(persistence.pool())
        .await?;

        let strategy = selector
            .determine_validation_strategy(node_id, miner_uid)
            .await?;

        assert!(matches!(strategy, ValidationStrategy::Full));
        Ok(())
    }
}
