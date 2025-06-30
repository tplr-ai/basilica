//! # Weight Setter
//!
//! Manages Bittensor weight setting operations for the Validator.
//! Sets weights every N blocks based on miner scores from executor validations.

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
    pub hardware_score: f64,
    pub gpu_count: usize,
    pub gpu_memory_gb: u64,
    pub network_bandwidth_mbps: f64,
    pub attestation_valid: bool,
    pub validation_timestamp: chrono::DateTime<chrono::Utc>,
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
    ) -> Result<Self> {
        Ok(Self {
            config,
            bittensor_service,
            storage,
            persistence,
            min_score_threshold,
            blocks_per_weight_set,
            last_weight_set_block: Arc::new(tokio::sync::Mutex::new(0)),
        })
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

    /// Set weights based on miner scores from executor validations
    async fn set_weights_for_miners(&self) -> Result<()> {
        info!("Setting weights for subnet {}", self.config.netuid);

        // 1. Get current metagraph
        let metagraph = self.get_metagraph().await?;
        debug!(
            "Retrieved metagraph with {} neurons",
            metagraph.hotkeys.len()
        );

        // 2. Calculate miner scores based on their executor validations
        let miner_scores = self.calculate_miner_scores(&metagraph).await?;

        if miner_scores.is_empty() {
            warn!("No miner scores available");
            return Ok(());
        }

        // 3. Filter by minimum threshold and prepare weights
        let weights: Vec<(u16, f64)> = miner_scores
            .into_iter()
            .filter(|(_, score)| *score >= self.min_score_threshold)
            .map(|(uid, score)| (uid.as_u16(), score))
            .collect();

        if weights.is_empty() {
            warn!(
                "No miners meet minimum score threshold of {}",
                self.min_score_threshold
            );
            return Ok(());
        }

        info!("Found {} miners meeting threshold", weights.len());

        // 4. Normalize weights using chain normalization
        let normalized_weights = self.normalize_weights_for_chain(weights.clone())?;

        // 5. Get version key and submit weights
        let version_key = self.get_version_key().await?;

        info!(
            "Submitting {} weights with version key {}",
            normalized_weights.len(),
            version_key
        );

        // Submit weights to chain
        self.submit_weights_to_chain(normalized_weights, version_key)
            .await?;

        // Store submission metadata
        self.store_weight_submission_metadata(&weights).await?;

        Ok(())
    }

    /// Calculate scores for each miner based on their executor validations
    async fn calculate_miner_scores(
        &self,
        metagraph: &Metagraph<AccountId>,
    ) -> Result<HashMap<MinerUid, f64>> {
        let mut miner_scores = HashMap::new();

        for (uid, _hotkey) in metagraph.hotkeys.iter().enumerate() {
            let uid = uid as u16;

            // Skip validators
            if metagraph
                .validator_permit
                .get(uid as usize)
                .copied()
                .unwrap_or(false)
            {
                continue;
            }

            let miner_uid = MinerUid::new(uid);

            // Get miner's score based on executor validations
            let score = self.get_miner_validation_score(miner_uid).await;

            if score > 0.0 {
                miner_scores.insert(miner_uid, score);
                debug!("Miner {} score: {:.4}", uid, score);
            }
        }

        info!("Calculated scores for {} miners", miner_scores.len());
        Ok(miner_scores)
    }

    /// Get validation score for a miner based on their executors' performance
    async fn get_miner_validation_score(&self, miner_uid: MinerUid) -> f64 {
        let key = format!("miner_validation_score:{}", miner_uid.as_u16());

        // Try to get the aggregated validation score
        match self.storage.get_f64(&key).await {
            Ok(Some(score)) => score,
            Ok(None) => {
                // No score yet - return 0.0
                // Scores are updated by update_miner_score_from_validation
                0.0
            }
            Err(e) => {
                warn!(
                    "Failed to get score for miner {}: {}",
                    miner_uid.as_u16(),
                    e
                );
                0.0
            }
        }
    }

    /// Update miner score based on executor validation results
    pub async fn update_miner_score_from_validation(
        &self,
        miner_uid: MinerUid,
        executor_validations: Vec<ExecutorValidationResult>,
    ) -> Result<()> {
        if executor_validations.is_empty() {
            return Ok(());
        }

        // Calculate aggregate score
        let total_score: f64 = executor_validations
            .iter()
            .map(|v| {
                if !v.is_valid || !v.attestation_valid {
                    return 0.0;
                }

                let base_score = v.hardware_score;

                // Weight by GPU quality (more GPUs and memory = higher weight)
                let gpu_weight =
                    (v.gpu_count as f64 * 0.1).min(0.5) + (v.gpu_memory_gb as f64 / 100.0).min(0.5);

                // Weight by network performance
                let network_weight = (v.network_bandwidth_mbps / 10000.0).clamp(0.5, 1.0);

                base_score * (1.0 + gpu_weight) * network_weight
            })
            .sum();

        let average_score = total_score / executor_validations.len() as f64;

        // Apply exponential moving average for stability
        let key = format!("miner_validation_score:{}", miner_uid.as_u16());
        let previous_score = self
            .storage
            .get_f64(&key)
            .await
            .unwrap_or(None)
            .unwrap_or(0.0);
        let smoothed_score = if previous_score > 0.0 {
            0.7 * previous_score + 0.3 * average_score // EMA with alpha=0.3
        } else {
            average_score
        };

        // Store the score
        self.storage.set_f64(&key, smoothed_score).await?;

        info!(
            "Updated score for miner {} to {:.4} based on {} executor validations",
            miner_uid.as_u16(),
            smoothed_score,
            executor_validations.len()
        );

        Ok(())
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

    /// Store metadata about weight submission
    async fn store_weight_submission_metadata(&self, weights: &[(u16, f64)]) -> Result<()> {
        // Store the weights we submitted for auditing
        let weights_json = serde_json::to_string(weights)?;
        let key = format!("submitted_weights:{}", self.config.netuid);
        self.storage.set_string(&key, &weights_json).await?;

        // Store submission timestamp
        let timestamp_key = format!("last_weight_submission:{}", self.config.netuid);
        let timestamp = chrono::Utc::now().timestamp();
        self.storage.set_i64(&timestamp_key, timestamp).await?;

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

        // Update the miner's score
        self.update_miner_score_from_validation(miner_uid, recent_validations)
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
        let hardware_specs = if log.success && !log.details.is_null() {
            serde_json::from_value(log.details.clone()).ok()
        } else {
            None
        };

        // Calculate hardware score based on specs
        let (hardware_score, gpu_count, gpu_memory_gb, network_bandwidth_mbps) =
            if let Some(specs) = hardware_specs {
                let score = self.calculate_hardware_score(&specs);
                let gpu_count = specs["gpu"].as_array().map(|a| a.len()).unwrap_or(0);
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

                (score, gpu_count, gpu_memory, bandwidth)
            } else {
                (0.0, 0, 0, 0.0)
            };

        Ok(ExecutorValidationResult {
            executor_id,
            is_valid: log.success,
            hardware_score,
            gpu_count,
            gpu_memory_gb,
            network_bandwidth_mbps,
            attestation_valid: log.verification_type == "attestation" && log.success,
            validation_timestamp: log.timestamp,
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
                            if let Err(e) = self
                                .update_miner_score_from_validation(miner_uid, validations)
                                .await
                            {
                                warn!("Failed to update score for miner {}: {}", uid, e);
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
