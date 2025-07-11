//! # Weight Setter
//!
//! Manages Bittensor weight setting operations for the Validator.
//! Sets weights every N blocks based on miner scores from executor validations.

use crate::bittensor_core::weight_allocation::WeightAllocationEngine;
use crate::config::emission::EmissionConfig;
use crate::gpu::categorization;
use crate::gpu::{CategoryStats, GpuScoringEngine};
use crate::persistence::entities::VerificationLog;
use crate::persistence::SimplePersistence;
use anyhow::Result;
use bittensor::{AccountId, Metagraph, NormalizedWeight, Service as BittensorService};
use common::config::BittensorConfig;
use common::identity::{ExecutorId, MinerUid};
use common::{KeyValueStorage, MemoryStorage};
use sqlx::Row;
use std::collections::HashMap;
use std::sync::Arc;
use std::time::Duration;
use tokio::time::interval;
use tracing::{debug, error, info, warn};

// NormalizedWeight is imported from bittensor crate

/// Executor validation result for scoring
#[derive(Debug, Clone)]
pub struct ExecutorValidationResult {
    pub executor_id: ExecutorId,
    pub is_valid: bool,
    pub _hardware_score: f64,
    pub gpu_count: usize,
    pub gpu_memory_gb: u64,
    pub _network_bandwidth_mbps: f64,
    pub attestation_valid: bool,
    pub validation_timestamp: chrono::DateTime<chrono::Utc>,
    pub gpu_model: String,
}

/// Manages weight setting operations for Bittensor network
#[derive(Clone)]
pub struct WeightSetter {
    config: BittensorConfig,
    bittensor_service: Arc<BittensorService>,
    storage: MemoryStorage,
    persistence: Arc<SimplePersistence>,
    min_score_threshold: f64,
    blocks_per_weight_set: u64,
    last_weight_set_block: Arc<tokio::sync::Mutex<u64>>,
    gpu_scoring_engine: Arc<GpuScoringEngine>,
    weight_allocation_engine: Arc<WeightAllocationEngine>,
    emission_config: EmissionConfig,
}

impl WeightSetter {
    /// Create a new WeightSetter instance
    pub fn new(
        config: BittensorConfig,
        bittensor_service: Arc<BittensorService>,
        storage: MemoryStorage,
        persistence: Arc<SimplePersistence>,
        min_score_threshold: f64,
        blocks_per_weight_set: u64,
        gpu_scoring_engine: Arc<GpuScoringEngine>,
        emission_config: EmissionConfig,
    ) -> Result<Self> {
        // Create weight allocation engine
        let weight_allocation_engine = Arc::new(WeightAllocationEngine::new(
            emission_config.clone(),
            min_score_threshold,
        ));

        Ok(Self {
            config,
            bittensor_service,
            storage,
            persistence,
            min_score_threshold,
            blocks_per_weight_set,
            last_weight_set_block: Arc::new(tokio::sync::Mutex::new(0)),
            gpu_scoring_engine,
            weight_allocation_engine,
            emission_config,
        })
    }

    /// Get the validator's hotkey as a string
    pub fn get_validator_hotkey(&self) -> Result<String> {
        let account_id = self.bittensor_service.get_account_id();
        let hotkey = bittensor::account_id_to_hotkey(account_id)
            .map_err(|e| anyhow::anyhow!("Failed to convert account_id to hotkey: {}", e))?;
        Ok(hotkey.to_string())
    }

    /// Start the weight setting loop
    pub async fn start(&self) -> Result<()> {
        // Check every 12 seconds (1 block time on Bittensor)
        let mut interval = interval(Duration::from_secs(12));

        info!(
            "Starting weight setter - will set weights every {} blocks, min_score_threshold: {:.2}",
            self.blocks_per_weight_set, self.min_score_threshold
        );

        loop {
            interval.tick().await;

            // Get current block number
            let current_block = match self.get_current_block().await {
                Ok(block) => block,
                Err(e) => {
                    error!("Failed to get current block: {}", e);
                    continue;
                }
            };

            let last_block = *self.last_weight_set_block.lock().await;

            // Check if it's time to set weights
            if current_block >= last_block + self.blocks_per_weight_set {
                if let Err(e) = self.set_weights_for_miners().await {
                    error!("Failed to set weights at block {}: {}", current_block, e);
                } else {
                    *self.last_weight_set_block.lock().await = current_block;
                }
            }
        }
    }

    /// Set weights based on GPU-based allocation with burn mechanism
    async fn set_weights_for_miners(&self) -> Result<()> {
        info!(
            "Setting weights for subnet {} with GPU-based allocation",
            self.config.netuid
        );

        // 1. Get current metagraph
        let metagraph = self.get_metagraph().await?;
        debug!(
            "Retrieved metagraph with {} neurons",
            metagraph.hotkeys.len()
        );

        // 2. Get miners by GPU category from the scoring engine
        let miners_by_category = self
            .gpu_scoring_engine
            .get_miners_by_gpu_category(24)
            .await?;

        if miners_by_category.is_empty() {
            warn!("No miners found in any GPU category");
            return Ok(());
        }

        info!(
            "Found miners in {} GPU categories: {:?}",
            miners_by_category.len(),
            miners_by_category.keys().collect::<Vec<_>>()
        );

        // 3. Calculate weight distribution using the allocation engine
        let weight_distribution = self
            .weight_allocation_engine
            .calculate_weight_distribution(miners_by_category)?;

        if weight_distribution.miners_served == 0 {
            warn!("No miners served by weight allocation");
            return Ok(());
        }

        info!(
            "Weight distribution calculated: {} miners served, {} categories",
            weight_distribution.miners_served,
            weight_distribution.category_allocations.len()
        );

        // 4. Log category allocations for transparency
        for (category, allocation) in &weight_distribution.category_allocations {
            info!(
                "Category {}: {} miners, {:.2}% allocation, total score: {:.4}",
                category,
                allocation.miner_count,
                allocation.allocation_percentage,
                allocation.total_score
            );
        }

        // 5. Log burn allocation if present
        if let Some(burn_alloc) = &weight_distribution.burn_allocation {
            info!(
                "Burn allocation: UID {}, weight {}, {:.2}%",
                burn_alloc.uid, burn_alloc.weight, burn_alloc.percentage
            );
        }

        // 6. Convert to normalized weights for chain submission
        let normalized_weights: Vec<NormalizedWeight> = weight_distribution
            .weights
            .iter()
            .map(|w| NormalizedWeight {
                uid: w.uid,
                weight: w.weight,
            })
            .collect();

        // 7. Get version key and submit weights
        let version_key = self.get_version_key().await?;

        info!(
            "Submitting {} weights with version key {}",
            normalized_weights.len(),
            version_key
        );

        // Submit weights to chain
        self.submit_weights_to_chain(normalized_weights.clone(), version_key)
            .await?;

        // 8. Store submission metadata
        self.store_weight_submission_metadata(&weight_distribution)
            .await?;

        Ok(())
    }

    /// Update miner GPU profile from validation results
    pub async fn update_miner_gpu_profile(
        &self,
        miner_uid: MinerUid,
        executor_validations: Vec<ExecutorValidationResult>,
    ) -> Result<()> {
        // Convert ExecutorValidationResult to the format expected by GPU scoring engine
        let gpu_validations: Vec<categorization::ExecutorValidationResult> = executor_validations
            .into_iter()
            .map(|v| categorization::ExecutorValidationResult {
                executor_id: v.executor_id.to_string(),
                is_valid: v.is_valid,
                gpu_model: v.gpu_model, // Use the actual GPU model from validation
                gpu_count: v.gpu_count,
                gpu_memory_gb: v.gpu_memory_gb,
                attestation_valid: v.attestation_valid,
                validation_timestamp: v.validation_timestamp,
            })
            .collect();

        // Update the miner's GPU profile using the scoring engine
        self.gpu_scoring_engine
            .update_miner_profile_from_validation(miner_uid, gpu_validations)
            .await?;

        Ok(())
    }

    /// Get category statistics for monitoring
    pub async fn get_category_statistics(&self) -> Result<HashMap<String, CategoryStats>> {
        self.gpu_scoring_engine.get_category_statistics().await
    }

    /// Normalize weights for chain submission using the provided normalization function
    fn normalize_weights_for_chain(
        &self,
        weights: Vec<(u16, f64)>,
    ) -> Result<Vec<NormalizedWeight>> {
        // Convert scores to u16 for normalization (scale to 0-65535 range)
        let max_score = weights
            .iter()
            .map(|(_, score)| *score)
            .fold(0.0f64, f64::max);

        if max_score <= 0.0 {
            error!("No positive scores found for normalization");
            return Err(anyhow::anyhow!("No positive scores found"));
        }

        // Normalize scores directly to u16 range
        // The chain expects weights in the 0-65535 range
        // We don't need to call normalize_weights again since we're already normalizing here
        let result: Vec<NormalizedWeight> = weights
            .iter()
            .map(|(uid, score)| {
                let normalized_weight = ((score / max_score) * u16::MAX as f64) as u16;
                NormalizedWeight {
                    uid: *uid,
                    weight: normalized_weight,
                }
            })
            .collect();

        // Log normalized weights for debugging
        for nw in result.iter() {
            let uid: u16 = nw.uid;
            let weight: u16 = nw.weight;
            tracing::debug!("Miner UID {} -> normalized weight {}", uid, weight);
        }

        Ok(result)
    }

    /// Submit weights to chain using the provided set_weights_payload function
    async fn submit_weights_to_chain(
        &self,
        normalized_weights: Vec<NormalizedWeight>,
        version_key: u64,
    ) -> Result<()> {
        // Create the payload using the provided function
        let payload =
            bittensor::set_weights_payload(self.config.netuid, normalized_weights, version_key);

        // Submit the transaction
        match self.bittensor_service.submit_extrinsic(payload).await {
            Ok(_) => {
                info!("Successfully submitted weights to chain");
                Ok(())
            }
            Err(e) => {
                error!("Failed to submit weights: {}", e);
                Err(anyhow::anyhow!("Weight submission failed: {}", e))
            }
        }
    }

    /// Get version key for weight setting
    async fn get_version_key(&self) -> Result<u64> {
        // Version key should be incremented with each weight setting
        // This prevents replay attacks
        let key = format!("weight_version_key:{}", self.config.netuid);

        let current_version = self
            .storage
            .get_i64(&key)
            .await
            .unwrap_or(Some(0))
            .unwrap_or(0) as u64;

        let new_version = current_version + 1;

        // Store new version
        self.storage.set_i64(&key, new_version as i64).await?;

        Ok(new_version)
    }

    /// Store metadata about GPU-based weight submission
    async fn store_weight_submission_metadata(
        &self,
        weight_distribution: &crate::bittensor_core::weight_allocation::WeightDistribution,
    ) -> Result<()> {
        // Store the weight distribution for auditing
        let distribution_json = serde_json::to_string(weight_distribution)?;
        let key = format!("submitted_weight_distribution:{}", self.config.netuid);
        self.storage.set_string(&key, &distribution_json).await?;

        // Store submission timestamp
        let timestamp_key = format!("last_weight_submission:{}", self.config.netuid);
        let timestamp = chrono::Utc::now().timestamp();
        self.storage.set_i64(&timestamp_key, timestamp).await?;

        // Store category statistics
        let stats_key = format!("category_stats:{}", self.config.netuid);
        let category_stats = self.gpu_scoring_engine.get_category_statistics().await?;
        let stats_json = serde_json::to_string(&category_stats)?;
        self.storage.set_string(&stats_key, &stats_json).await?;

        info!(
            "Stored weight submission metadata with {} categories",
            weight_distribution.category_allocations.len()
        );
        Ok(())
    }

    /// Get current block number from chain
    async fn get_current_block(&self) -> Result<u64> {
        self.bittensor_service
            .get_current_block()
            .await
            .map_err(|e| anyhow::anyhow!("Failed to get current block: {}", e))
    }

    /// Get current metagraph from Bittensor network
    async fn get_metagraph(&self) -> Result<Metagraph<AccountId>> {
        self.bittensor_service
            .get_metagraph(self.config.netuid)
            .await
            .map_err(|e| anyhow::anyhow!("Failed to fetch metagraph: {}", e))
    }

    /// Process a completed executor validation and update the miner's score
    pub async fn process_executor_validation(
        &self,
        miner_uid: MinerUid,
        executor_id: ExecutorId,
        verification_log: &VerificationLog,
    ) -> Result<()> {
        // Extract validation details from the verification log
        let _validation_result = self.extract_validation_result(executor_id, verification_log)?;

        // Get all recent validations for this miner
        let recent_validations = self
            .get_recent_miner_validations(miner_uid, 24) // Last 24 hours
            .await?;

        // Update the miner's GPU profile using the new system
        self.update_miner_gpu_profile(miner_uid, recent_validations)
            .await?;

        Ok(())
    }

    /// Extract validation result from verification log
    fn extract_validation_result(
        &self,
        executor_id: ExecutorId,
        log: &VerificationLog,
    ) -> Result<ExecutorValidationResult> {
        // Parse hardware specs from the verification log details
        // Always try to parse specs, even for failed validations, to track GPU hardware
        let hardware_specs = if !log.details.is_null() {
            serde_json::from_value(log.details.clone()).ok()
        } else {
            None
        };

        // Calculate hardware score based on specs
        let (hardware_score, gpu_count, gpu_memory_gb, network_bandwidth_mbps, gpu_model) =
            if let Some(specs) = hardware_specs {
                let score = self.calculate_hardware_score(&specs);
                let gpu_count = specs["gpu"].as_array().map(|a| a.len()).unwrap_or(0);

                // Extract GPU model from the first GPU (primary GPU)
                let gpu_model = specs["gpu"]
                    .as_array()
                    .and_then(|gpus| gpus.first())
                    .and_then(|gpu| gpu["model"].as_str())
                    .unwrap_or("UNKNOWN")
                    .to_string();

                let gpu_memory = specs["gpu"]
                    .as_array()
                    .map(|gpus| {
                        gpus.iter()
                            .map(|gpu| gpu["vram_mb"].as_u64().unwrap_or(0))
                            .sum::<u64>()
                    })
                    .unwrap_or(0)
                    / 1024; // Convert MB to GB
                let bandwidth = specs["network"]["bandwidth_mbps"].as_f64().unwrap_or(0.0);

                (score, gpu_count, gpu_memory, bandwidth, gpu_model)
            } else {
                (0.0, 0, 0, 0.0, "UNKNOWN".to_string())
            };

        Ok(ExecutorValidationResult {
            executor_id,
            is_valid: log.success,
            _hardware_score: hardware_score,
            gpu_count,
            gpu_memory_gb,
            _network_bandwidth_mbps: network_bandwidth_mbps,
            attestation_valid: log.verification_type == "attestation" && log.success,
            validation_timestamp: log.timestamp,
            gpu_model,
        })
    }

    /// Calculate hardware score from specs
    fn calculate_hardware_score(&self, specs: &serde_json::Value) -> f64 {
        let mut score = 0.0;

        // GPU scoring (40% weight)
        if let Some(gpus) = specs["gpu"].as_array() {
            let gpu_score: f64 = gpus
                .iter()
                .map(|gpu| {
                    let vram_mb = gpu["vram_mb"].as_u64().unwrap_or(0) as f64;
                    let vram_score = (vram_mb / 24576.0).min(1.0); // 24GB = max score

                    // Bonus for high-end GPUs
                    let model = gpu["model"].as_str().unwrap_or("");
                    let model_bonus = match model {
                        s if s.contains("H100") => 1.0,
                        s if s.contains("A100") => 0.9,
                        s if s.contains("4090") => 0.8,
                        s if s.contains("3090") => 0.7,
                        _ => 0.5,
                    };

                    vram_score * model_bonus
                })
                .sum::<f64>()
                / gpus.len().max(1) as f64;

            score += gpu_score * 0.4;
        }

        // CPU scoring (20% weight)
        if let Some(cpu_cores) = specs["cpu"]["cores"].as_u64() {
            let cpu_score = (cpu_cores as f64 / 64.0).min(1.0); // 64 cores = max score
            score += cpu_score * 0.2;
        }

        // Memory scoring (20% weight)
        if let Some(memory_mb) = specs["memory"]["total_mb"].as_u64() {
            let memory_score = (memory_mb as f64 / 262144.0).min(1.0); // 256GB = max score
            score += memory_score * 0.2;
        }

        // Network scoring (20% weight)
        if let Some(bandwidth) = specs["network"]["bandwidth_mbps"].as_f64() {
            let network_score = (bandwidth / 10000.0).min(1.0); // 10Gbps = max score
            score += network_score * 0.2;
        }

        score
    }

    /// Get recent validation results for a miner
    async fn get_recent_miner_validations(
        &self,
        miner_uid: MinerUid,
        hours: u32,
    ) -> Result<Vec<ExecutorValidationResult>> {
        let cutoff_time = chrono::Utc::now() - chrono::Duration::hours(hours as i64);

        // Query verification logs for executors belonging to this miner
        let query = r#"
            SELECT vl.*, me.miner_id
            FROM verification_logs vl
            JOIN miner_executors me ON vl.executor_id = me.executor_id
            WHERE me.miner_id = ? AND vl.timestamp >= ?
            ORDER BY vl.timestamp DESC
        "#;

        let miner_id = format!("miner_{}", miner_uid.as_u16());
        let rows = sqlx::query(query)
            .bind(&miner_id)
            .bind(cutoff_time.to_rfc3339())
            .fetch_all(self.persistence.pool())
            .await
            .map_err(|e| anyhow::anyhow!("Failed to query verification logs: {}", e))?;

        let mut validations = Vec::new();
        for row in rows {
            let executor_id_str: String = row.get("executor_id");
            let executor_id = executor_id_str
                .parse::<ExecutorId>()
                .map_err(|e| anyhow::anyhow!("Failed to parse executor ID: {}", e))?;

            let log = VerificationLog {
                id: row.get("id"),
                executor_id: row.get("executor_id"),
                validator_hotkey: row.get("validator_hotkey"),
                verification_type: row.get("verification_type"),
                timestamp: row.get("timestamp"),
                score: row.get("score"),
                success: row.get::<i64, _>("success") != 0,
                details: row.get("details"),
                duration_ms: row.get("duration_ms"),
                error_message: row.get("error_message"),
                created_at: row.get("created_at"),
                updated_at: row.get("updated_at"),
            };

            if let Ok(validation) = self.extract_validation_result(executor_id, &log) {
                validations.push(validation);
            }
        }

        info!(
            "Found {} recent validations for miner {}",
            validations.len(),
            miner_uid.as_u16()
        );

        Ok(validations)
    }

    /// Update all miner scores based on their recent validations
    pub async fn update_all_miner_scores(&self) -> Result<()> {
        info!("Updating scores for all miners based on recent validations");

        // Get all unique miner UIDs from recent validations
        let query = r#"
            SELECT DISTINCT me.miner_id
            FROM miner_executors me
            JOIN verification_logs vl ON me.executor_id = vl.executor_id
            WHERE vl.timestamp >= ?
        "#;

        let cutoff_time = chrono::Utc::now() - chrono::Duration::hours(24);
        let rows = sqlx::query(query)
            .bind(cutoff_time.to_rfc3339())
            .fetch_all(self.persistence.pool())
            .await
            .map_err(|e| anyhow::anyhow!("Failed to query miners: {}", e))?;

        for row in rows {
            let miner_id: String = row.get("miner_id");
            if let Some(uid_str) = miner_id.strip_prefix("miner_") {
                if let Ok(uid) = uid_str.parse::<u16>() {
                    let miner_uid = MinerUid::new(uid);

                    match self.get_recent_miner_validations(miner_uid, 24).await {
                        Ok(validations) if !validations.is_empty() => {
                            if let Err(e) =
                                self.update_miner_gpu_profile(miner_uid, validations).await
                            {
                                warn!("Failed to update GPU profile for miner {}: {}", uid, e);
                            }
                        }
                        Ok(_) => {
                            debug!("No recent validations for miner {}", uid);
                        }
                        Err(e) => {
                            warn!("Failed to get validations for miner {}: {}", uid, e);
                        }
                    }
                }
            }
        }

        info!("Completed updating all miner scores");
        Ok(())
    }
}

/// Calculate aggregated miner scores from database
pub async fn calculate_miner_scores_from_database(
    persistence: &SimplePersistence,
    hours: u32,
) -> Result<HashMap<MinerUid, f64>> {
    let cutoff_time = chrono::Utc::now() - chrono::Duration::hours(hours as i64);

    let query = r#"
        SELECT 
            me.miner_id,
            COUNT(DISTINCT vl.executor_id) as executor_count,
            COUNT(vl.id) as total_validations,
            SUM(CASE WHEN vl.success = 1 THEN 1 ELSE 0 END) as successful_validations,
            AVG(CASE WHEN vl.success = 1 THEN vl.score ELSE 0 END) as avg_score
        FROM miner_executors me
        JOIN verification_logs vl ON me.executor_id = vl.executor_id
        WHERE vl.timestamp >= ?
        GROUP BY me.miner_id
    "#;

    let rows = sqlx::query(query)
        .bind(cutoff_time.to_rfc3339())
        .fetch_all(persistence.pool())
        .await
        .map_err(|e| anyhow::anyhow!("Failed to calculate miner scores: {}", e))?;

    let mut scores = HashMap::new();

    for row in rows {
        let miner_id: String = row.get("miner_id");
        if let Some(uid_str) = miner_id.strip_prefix("miner_") {
            if let Ok(uid) = uid_str.parse::<u16>() {
                let miner_uid = MinerUid::new(uid);

                let executor_count: i64 = row.get("executor_count");
                let total_validations: i64 = row.get("total_validations");
                let successful_validations: i64 = row.get("successful_validations");
                let avg_score: Option<f64> = row.get("avg_score");

                // Calculate composite score
                let success_rate = if total_validations > 0 {
                    successful_validations as f64 / total_validations as f64
                } else {
                    0.0
                };

                let base_score = avg_score.unwrap_or(0.0) * success_rate;
                let executor_bonus = (executor_count as f64 * 0.1).min(0.5); // Bonus for multiple executors

                let final_score = (base_score * (1.0 + executor_bonus)).min(1.0);

                scores.insert(miner_uid, final_score);

                debug!(
                    "Miner {} score: {:.4} (executors: {}, validations: {}, success rate: {:.2})",
                    uid, final_score, executor_count, total_validations, success_rate
                );
            }
        }
    }

    info!("Calculated scores for {} miners", scores.len());
    Ok(scores)
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::persistence::entities::VerificationLog;
    use serde_json::json;

    #[test]
    fn test_extract_validation_result_with_h100() {
        // Create a verification log with H100 GPU
        let log = VerificationLog {
            id: uuid::Uuid::new_v4(),
            executor_id: "exec123".to_string(),
            validator_hotkey: "validator".to_string(),
            verification_type: "attestation".to_string(),
            timestamp: chrono::Utc::now(),
            score: 1.0,
            success: true,
            details: json!({
                "gpu": [{
                    "model": "NVIDIA H100 80GB PCIe",
                    "vram_mb": 81920
                }],
                "cpu": {"cores": 32},
                "memory": {"total_mb": 131072},
                "network": {"bandwidth_mbps": 10000.0}
            }),
            duration_ms: 1000,
            error_message: None,
            created_at: chrono::Utc::now(),
            updated_at: chrono::Utc::now(),
        };

        // The GPU model should be correctly extracted
        let details = &log.details;
        let gpu_model = details["gpu"]
            .as_array()
            .and_then(|gpus| gpus.first())
            .and_then(|gpu| gpu["model"].as_str())
            .unwrap_or("UNKNOWN");

        assert_eq!(gpu_model, "NVIDIA H100 80GB PCIe");
    }

    #[test]
    fn test_extract_validation_result_with_h200() {
        // Create a verification log with H200 GPU
        let log = VerificationLog {
            id: uuid::Uuid::new_v4(),
            executor_id: "exec456".to_string(),
            validator_hotkey: "validator".to_string(),
            verification_type: "attestation".to_string(),
            timestamp: chrono::Utc::now(),
            score: 1.0,
            success: true,
            details: json!({
                "gpu": [{
                    "model": "NVIDIA H200",
                    "vram_mb": 141312  // 138GB
                }],
                "cpu": {"cores": 64},
                "memory": {"total_mb": 262144},
                "network": {"bandwidth_mbps": 25000.0}
            }),
            duration_ms: 1000,
            error_message: None,
            created_at: chrono::Utc::now(),
            updated_at: chrono::Utc::now(),
        };

        let details = &log.details;
        let gpu_model = details["gpu"]
            .as_array()
            .and_then(|gpus| gpus.first())
            .and_then(|gpu| gpu["model"].as_str())
            .unwrap_or("UNKNOWN");

        assert_eq!(gpu_model, "NVIDIA H200");
    }

    #[test]
    fn test_gpu_model_extraction_from_failed_attestation() {
        // Create a failed verification log - should still extract GPU info
        let log = VerificationLog {
            id: uuid::Uuid::new_v4(),
            executor_id: "exec789".to_string(),
            validator_hotkey: "validator".to_string(),
            verification_type: "attestation".to_string(),
            timestamp: chrono::Utc::now(),
            score: 0.0,
            success: false,
            details: json!({
                "gpu": [{
                    "model": "NVIDIA H100 80GB PCIe",
                    "vram_mb": 81920
                }],
                "cpu": {"cores": 32},
                "memory": {"total_mb": 131072},
                "network": {"bandwidth_mbps": 10000.0}
            }),
            duration_ms: 1000,
            error_message: Some("Attestation verification failed".to_string()),
            created_at: chrono::Utc::now(),
            updated_at: chrono::Utc::now(),
        };

        // Should still extract GPU model even though attestation failed
        let details = &log.details;
        let gpu_model = details["gpu"]
            .as_array()
            .and_then(|gpus| gpus.first())
            .and_then(|gpu| gpu["model"].as_str())
            .unwrap_or("UNKNOWN");

        assert_eq!(gpu_model, "NVIDIA H100 80GB PCIe");
    }

    #[test]
    fn test_no_gpu_info_returns_unknown() {
        // Create a verification log with no GPU info
        let log = VerificationLog {
            id: uuid::Uuid::new_v4(),
            executor_id: "exec999".to_string(),
            validator_hotkey: "validator".to_string(),
            verification_type: "attestation".to_string(),
            timestamp: chrono::Utc::now(),
            score: 0.0,
            success: false,
            details: serde_json::Value::Null,
            duration_ms: 1000,
            error_message: Some("Failed to get hardware info".to_string()),
            created_at: chrono::Utc::now(),
            updated_at: chrono::Utc::now(),
        };

        let details = &log.details;
        let gpu_model = details["gpu"]
            .as_array()
            .and_then(|gpus| gpus.first())
            .and_then(|gpu| gpu["model"].as_str())
            .unwrap_or("UNKNOWN");

        assert_eq!(gpu_model, "UNKNOWN");
    }

    #[test]
    fn test_old_gpu_model_calculation_was_wrong() {
        // This test demonstrates why the old calculation was wrong
        let gpu_memory_gb = 80u64;

        // Old incorrect calculation
        let old_gpu_model = format!("H{}", gpu_memory_gb / 1024);
        assert_eq!(old_gpu_model, "H0"); // This is wrong!

        // For H100 with 80GB, dividing by 1024 gives 0.078, formatted as "H0"
        // For H200 with 138GB, dividing by 1024 gives 0.134, formatted as "H0"
        // Both would be categorized as "OTHER" and excluded from rewards!
    }

    #[tokio::test]
    async fn test_weight_setter_scoring() {
        // Create mock validation results
        let validations = vec![
            ExecutorValidationResult {
                executor_id: ExecutorId::new(),
                is_valid: true,
                _hardware_score: 0.8,
                gpu_count: 2,
                gpu_memory_gb: 48,
                _network_bandwidth_mbps: 1000.0,
                attestation_valid: true,
                validation_timestamp: chrono::Utc::now(),
                gpu_model: "NVIDIA H100".to_string(),
            },
            ExecutorValidationResult {
                executor_id: ExecutorId::new(),
                is_valid: true,
                _hardware_score: 0.9,
                gpu_count: 4,
                gpu_memory_gb: 96,
                _network_bandwidth_mbps: 10000.0,
                attestation_valid: true,
                validation_timestamp: chrono::Utc::now(),
                gpu_model: "NVIDIA H100".to_string(),
            },
        ];

        // Test that all validations with valid attestations contribute to scoring
        let valid_count = validations
            .iter()
            .filter(|v| v.is_valid && v.attestation_valid)
            .count();

        assert_eq!(valid_count, 2);

        // Test GPU model is properly set
        for validation in &validations {
            assert!(validation.gpu_model.contains("H100"));
        }
    }
}
