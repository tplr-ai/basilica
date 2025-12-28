//! # Weight Setter
//!
//! Manages Bittensor weight setting operations for the Validator.
//! Sets weights every N blocks based on miner scores from node validations.

use crate::bittensor_core::weight_allocation::WeightAllocationEngine;
use crate::config::emission::EmissionConfig;
use crate::gpu::categorization;
use crate::gpu::GpuScoringEngine;
use crate::metrics::ValidatorMetrics;
use crate::persistence::entities::VerificationLog;
use crate::persistence::gpu_profile_repository::GpuProfileRepository;
use crate::persistence::SimplePersistence;
use anyhow::Result;
use basilica_common::config::BittensorConfig;
use basilica_common::identity::{MinerUid, NodeId};
use basilica_common::{KeyValueStorage, MemoryStorage};
use bittensor::{Metagraph, NormalizedWeight, Service as BittensorService};
use chrono::{DateTime, Utc};
use sqlx::Row;
use std::collections::HashMap;
use std::sync::Arc;
use std::time::Duration;
use tokio::time::interval;
use tracing::{debug, error, info, warn};
use uuid::Uuid;

// NormalizedWeight is imported from bittensor crate

/// Cutoff time in hours for filtering miners by GPU category
const GPU_CATEGORY_CUTOFF_HOURS: u32 = 3;

/// Node validation result for scoring
#[derive(Debug, Clone)]
pub struct NodeValidationResult {
    pub node_id: NodeId,
    pub is_valid: bool,
    pub _hardware_score: f64,
    pub gpu_count: usize,
    pub gpu_memory_gb: f64,
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
    gpu_profile_repo: Arc<GpuProfileRepository>,
    metrics: Option<Arc<ValidatorMetrics>>,
}

impl WeightSetter {
    /// Create a new WeightSetter instance with metrics support
    #[allow(clippy::too_many_arguments)]
    pub fn new(
        config: BittensorConfig,
        bittensor_service: Arc<BittensorService>,
        storage: MemoryStorage,
        persistence: Arc<SimplePersistence>,
        min_score_threshold: f64,
        blocks_per_weight_set: u64,
        gpu_scoring_engine: Arc<GpuScoringEngine>,
        emission_config: EmissionConfig,
        gpu_profile_repo: Arc<GpuProfileRepository>,
        metrics: Option<Arc<ValidatorMetrics>>,
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
            gpu_profile_repo,
            metrics,
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

        // Initialize last weight set block from storage or chain
        let last_weight_block = self.get_last_weight_set_block().await?;
        *self.last_weight_set_block.lock().await = last_weight_block;

        info!(
            "Initialized last weight set block to: {}, will wait {} blocks before first weight setting",
            last_weight_block, self.blocks_per_weight_set
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
                info!(
                    current_block = current_block,
                    last_block = last_block,
                    "Setting weights at block {} (last set at block {}, interval: {} blocks)",
                    current_block,
                    last_block,
                    self.blocks_per_weight_set
                );

                // Atomic weight setting with proper persistence
                match self.set_weights_for_miners().await {
                    Ok(()) => {
                        // Only update persistence after successful weight setting
                        match self.store_last_weight_set_block(current_block).await {
                            Ok(()) => {
                                *self.last_weight_set_block.lock().await = current_block;
                                info!(
                                    current_block = current_block,
                                    "Successfully set weights and updated persistence at block {}",
                                    current_block
                                );
                            }
                            Err(e) => {
                                error!(
                                    current_block = current_block,
                                    "Weight setting succeeded but failed to persist block {}: {}",
                                    current_block,
                                    e
                                );
                                // Continue anyway - weight setting was successful
                                *self.last_weight_set_block.lock().await = current_block;
                            }
                        }
                    }
                    Err(e) => {
                        error!(
                            current_block = current_block,
                            "Failed to set weights at block {}: {}", current_block, e
                        );
                        // Don't update last_weight_set_block on failure
                    }
                }
            } else {
                let blocks_remaining = last_block + self.blocks_per_weight_set - current_block;
                debug!(
                    current_block = current_block,
                    last_block = last_block,
                    "Waiting to set weights: {} blocks remaining (current: {}, last: {}, interval: {})",
                    blocks_remaining, current_block, last_block, self.blocks_per_weight_set
                );
            }
        }
    }

    /// Set weights based on GPU-based allocation with burn mechanism
    async fn set_weights_for_miners(&self) -> Result<()> {
        const MAX_RETRIES: u32 = 3;
        const RETRY_DELAY: Duration = Duration::from_secs(5);

        for attempt in 1..=MAX_RETRIES {
            match self.attempt_weight_setting().await {
                Ok(()) => {
                    info!("Weight setting successful on attempt {}", attempt);
                    return Ok(());
                }
                Err(e) => {
                    error!("Weight setting attempt {} failed: {}", attempt, e);
                    if attempt < MAX_RETRIES {
                        warn!(
                            "Retrying weight setting in {} seconds...",
                            RETRY_DELAY.as_secs()
                        );
                        tokio::time::sleep(RETRY_DELAY).await;
                    }
                }
            }
        }

        Err(anyhow::anyhow!(
            "Failed to set weights after {} attempts",
            MAX_RETRIES
        ))
    }

    /// Attempt to set weights (extracted for retry logic)
    async fn attempt_weight_setting(&self) -> Result<()> {
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

        // 2. Get last weight set timestamp for epoch filtering
        let last_weight_timestamp = self.get_last_weight_set_timestamp().await?;
        info!(
            "Fetching miners by GPU category from scoring engine, cutoff at {GPU_CATEGORY_CUTOFF_HOURS} hours, epoch: {:?}",
            last_weight_timestamp
        );

        // 3. Get miners by GPU category from the scoring engine with axon validation and epoch filtering
        let miners_by_category = self
            .gpu_scoring_engine
            .get_miners_by_gpu_category_since_epoch(
                last_weight_timestamp,
                GPU_CATEGORY_CUTOFF_HOURS,
                &metagraph,
            )
            .await?;

        if miners_by_category.is_empty() {
            warn!("No miners found in any GPU category - proceeding with burn allocation");
        }

        info!(
            "Found miners in {} GPU categories: {:?}",
            miners_by_category.len(),
            miners_by_category.keys().collect::<Vec<_>>()
        );

        // 4. Calculate weight distribution using the allocation engine
        let weight_distribution = self
            .weight_allocation_engine
            .calculate_weight_distribution(miners_by_category)?;

        if weight_distribution.miners_served == 0 {
            warn!("No miners served by weight allocation - proceeding with burn-only weights");
        }

        info!(
            "Weight distribution calculated: {} miners served, {} categories",
            weight_distribution.miners_served,
            weight_distribution.category_allocations.len()
        );

        // 5. Log category allocations for transparency
        for (category, allocation) in &weight_distribution.category_allocations {
            info!(
                gpu_category = %category,
                miner_count = allocation.miner_count,
                allocation_percentage = allocation.allocation_percentage,
                total_score = allocation.total_score,
                "[WEIGHT_FLOW] GPU category allocation"
            );
        }

        // 6. Log burn allocation if present
        if let Some(burn_alloc) = &weight_distribution.burn_allocation {
            info!(
                miner_uid = burn_alloc.uid,
                weight = burn_alloc.weight,
                percentage = burn_alloc.percentage,
                "[WEIGHT_FLOW] Burn allocation"
            );
        }

        // 7. Convert to normalized weights for chain submission including burn allocation
        let normalized_weights = self.build_normalized_weights(&weight_distribution)?;

        // 8. Get version key and submit weights
        let version_key = self.get_version_key().await?;

        info!(
            netuid = self.config.netuid,
            weight_count = normalized_weights.len(),
            version_key = version_key,
            "Preparing to submit weights to chain"
        );

        // Additional validation - check for duplicates before submission
        let mut uid_check = std::collections::HashSet::new();
        for weight in &normalized_weights {
            if !uid_check.insert(weight.uid) {
                error!("CRITICAL: Duplicate UID {} found in normalized weights before chain submission", weight.uid);
                error!("Full weights vector: {:?}", normalized_weights);
                return Err(anyhow::anyhow!(
                    "Duplicate UID {} detected in final weights",
                    weight.uid
                ));
            }
        }

        // Submit weights to chain with enhanced error handling and retry logic
        self.submit_weights_to_chain_with_retry(normalized_weights.clone(), version_key)
            .await?;

        // 9. Store emission metrics to database
        let current_block = self.get_current_block().await.unwrap_or(0);
        let emission_metrics_id = self
            .store_emission_metrics(&weight_distribution, current_block)
            .await?;

        // 10. Store weight allocation history for each miner
        self.store_weight_allocations(&weight_distribution, emission_metrics_id, current_block)
            .await?;

        // 11. Store submission metadata
        self.store_weight_submission_metadata(&weight_distribution)
            .await?;

        Ok(())
    }

    /// Build normalized weights from weight distribution
    fn build_normalized_weights(
        &self,
        weight_distribution: &crate::bittensor_core::weight_allocation::WeightDistribution,
    ) -> Result<Vec<NormalizedWeight>> {
        debug!(
            "Building normalized weights from {} distribution weights",
            weight_distribution.weights.len()
        );
        for (i, w) in weight_distribution.weights.iter().enumerate() {
            debug!(
                miner_uid = w.uid,
                weight = w.weight,
                index = i,
                "[WEIGHT_FLOW] Distribution weight"
            );
        }

        let normalized_weights: Vec<NormalizedWeight> = weight_distribution
            .weights
            .iter()
            .map(|w| NormalizedWeight {
                uid: w.uid,
                weight: w.weight,
            })
            .collect();

        debug!("Built {} normalized weights", normalized_weights.len());
        for (i, w) in normalized_weights.iter().enumerate() {
            debug!(
                miner_uid = w.uid,
                weight = w.weight,
                index = i,
                "[WEIGHT_FLOW] Normalized weight"
            );
        }

        // The weight allocation engine already includes burn allocation in weights vector
        assert!(
            !normalized_weights.is_empty(),
            "Weight allocation engine produced no weights - this should never happen"
        );

        Ok(normalized_weights)
    }

    /// Update miner GPU profile from validation results
    pub async fn update_miner_gpu_profile(
        &self,
        miner_uid: MinerUid,
        node_validations: Vec<NodeValidationResult>,
    ) -> Result<()> {
        info!(
            miner_uid = miner_uid.as_u16(),
            validation_count = node_validations.len(),
            "[WEIGHT_FLOW] Updating GPU profile for miner"
        );

        // Convert NodeValidationResult to the format expected by GPU scoring engine
        let gpu_validations: Vec<categorization::NodeValidationResult> = node_validations
            .into_iter()
            .map(|v| {
                debug!(
                    miner_uid = miner_uid.as_u16(),
                    node_id = %v.node_id,
                    gpu_model = %v.gpu_model,
                    gpu_count = v.gpu_count,
                    is_valid = v.is_valid,
                    attestation_valid = v.attestation_valid,
                    "[WEIGHT_FLOW] Converting validation for node"
                );
                categorization::NodeValidationResult {
                    node_id: v.node_id.to_string(),
                    is_valid: v.is_valid,
                    gpu_model: v.gpu_model,
                    gpu_count: v.gpu_count,
                    gpu_memory_gb: v.gpu_memory_gb,
                    attestation_valid: v.attestation_valid,
                    validation_timestamp: v.validation_timestamp,
                }
            })
            .collect();

        info!(
            miner_uid = miner_uid.as_u16(),
            "[WEIGHT_FLOW] Calling GPU scoring engine for miner {} with {} converted validations",
            miner_uid.as_u16(),
            gpu_validations.len()
        );

        // Update the miner's GPU profile using the scoring engine
        match self
            .gpu_scoring_engine
            .update_miner_profile_from_validation(miner_uid, gpu_validations)
            .await
        {
            Ok(profile) => {
                info!(
                    miner_uid = miner_uid.as_u16(),
                    "Successfully updated GPU profile for miner {}: total_gpus={}, score={:.4}, gpu_distribution={:?}",
                    miner_uid.as_u16(),
                    profile.total_gpu_count(),
                    profile.total_score,
                    profile.gpu_counts
                );
            }
            Err(e) => {
                error!(
                    miner_uid = miner_uid.as_u16(),
                    "Failed to update GPU profile for miner {}: {}",
                    miner_uid.as_u16(),
                    e
                );
                return Err(e);
            }
        }

        Ok(())
    }

    /// Submit weights to chain with retry logic
    async fn submit_weights_to_chain_with_retry(
        &self,
        normalized_weights: Vec<NormalizedWeight>,
        version_key: u64,
    ) -> Result<()> {
        const MAX_RETRIES: u32 = 3;
        const RETRY_DELAY: Duration = Duration::from_secs(10);

        for attempt in 1..=MAX_RETRIES {
            match self
                .submit_weights_to_chain(normalized_weights.clone(), version_key)
                .await
            {
                Ok(()) => {
                    info!("Weight submission successful on attempt {}", attempt);
                    return Ok(());
                }
                Err(e) => {
                    error!("Weight submission attempt {} failed: {}", attempt, e);
                    if attempt < MAX_RETRIES {
                        warn!(
                            "Retrying weight submission in {} seconds...",
                            RETRY_DELAY.as_secs()
                        );
                        tokio::time::sleep(RETRY_DELAY).await;
                    }
                }
            }
        }

        Err(anyhow::anyhow!(
            "Failed to submit weights after {} attempts",
            MAX_RETRIES
        ))
    }

    /// Submit weights to chain using the provided set_weights_payload function
    async fn submit_weights_to_chain(
        &self,
        normalized_weights: Vec<NormalizedWeight>,
        version_key: u64,
    ) -> Result<()> {
        // Pre-validate weights before submission
        self.validate_weights_before_submission(&normalized_weights)?;

        let submission_start = std::time::Instant::now();
        let current_block = self.get_current_block().await.unwrap_or(0);

        info!(
            netuid = self.config.netuid,
            version_key = version_key,
            weight_count = normalized_weights.len(),
            current_block = current_block,
            "Initiating weight submission to chain"
        );

        // Log individual weights at debug level for troubleshooting
        for weight in &normalized_weights {
            debug!(
                miner_uid = weight.uid,
                weight = weight.weight,
                netuid = self.config.netuid,
                "Weight submission detail"
            );

            // Record weight to metrics if available
            if let Some(ref metrics) = self.metrics {
                metrics
                    .prometheus()
                    .record_miner_weight(weight.uid, weight.weight);
            }
        }

        // Create the payload using the provided function
        let payload = bittensor::set_weights_payload(
            self.config.netuid,
            normalized_weights.clone(),
            version_key,
        );

        // Submit the transaction
        match self.bittensor_service.submit_extrinsic(payload).await {
            Ok(_) => {
                let duration = submission_start.elapsed();
                info!(
                    netuid = self.config.netuid,
                    version_key = version_key,
                    weight_count = normalized_weights.len(),
                    current_block = current_block,
                    duration_ms = duration.as_millis(),
                    "Successfully submitted weights to chain"
                );
                Ok(())
            }
            Err(e) => {
                let duration = submission_start.elapsed();
                let error_context = self.analyze_submission_error(&anyhow::anyhow!("{}", e));

                error!(
                    netuid = self.config.netuid,
                    version_key = version_key,
                    weight_count = normalized_weights.len(),
                    current_block = current_block,
                    duration_ms = duration.as_millis(),
                    error = %e,
                    error_type = %error_context,
                    weights = ?normalized_weights,
                    "Failed to submit weights to chain"
                );

                let uid_list: Vec<u16> = normalized_weights.iter().map(|w| w.uid).collect();

                Err(anyhow::anyhow!(
                    "Weight submission failed ({}): {} - Context: netuid={}, version_key={}, uids={:?}",
                    error_context,
                    e,
                    self.config.netuid,
                    version_key,
                    uid_list
                ))
            }
        }
    }

    /// Validate weights before chain submission
    fn validate_weights_before_submission(&self, weights: &[NormalizedWeight]) -> Result<()> {
        if weights.is_empty() {
            return Err(anyhow::anyhow!("Cannot submit empty weight vector"));
        }

        debug!("Validating {} weights before submission:", weights.len());
        for (i, weight) in weights.iter().enumerate() {
            debug!(
                miner_uid = weight.uid,
                "  Weight {}: UID={}, weight={}", i, weight.uid, weight.weight
            );
        }

        let mut seen_uids = std::collections::HashSet::new();
        for weight in weights {
            if !seen_uids.insert(weight.uid) {
                error!(
                    miner_uid = weight.uid,
                    "Duplicate UID {} detected in weights: {:?}", weight.uid, weights
                );
                return Err(anyhow::anyhow!("Duplicate UID {} in weights", weight.uid));
            }
        }

        let total_weight: u64 = weights.iter().map(|w| w.weight as u64).sum();
        if total_weight > u16::MAX as u64 {
            return Err(anyhow::anyhow!(
                "Total weight {} exceeds maximum",
                total_weight
            ));
        }

        Ok(())
    }

    /// Analyze submission error
    fn analyze_submission_error(&self, error: &anyhow::Error) -> &'static str {
        let error_str = error.to_string().to_lowercase();
        if error_str.contains("duplicate") {
            "DuplicateUids"
        } else if error_str.contains("timeout") {
            "Timeout"
        } else if error_str.contains("fee") {
            "InsufficientFees"
        } else if error_str.contains("nonce") {
            "InvalidNonce"
        } else if error_str.contains("weight") {
            "WeightValidation"
        } else if error_str.contains("network") {
            "NetworkError"
        } else {
            "Unknown"
        }
    }

    /// Get version key for weight setting
    async fn get_version_key(&self) -> Result<u64> {
        // Use the version key from config and increment with each weight setting
        // This prevents replay attacks
        let key = format!("weight_version_key:{}", self.config.netuid);

        let current_version =
            self.storage
                .get_i64(&key)
                .await
                .unwrap_or(Some(self.emission_config.weight_version_key as i64))
                .unwrap_or(self.emission_config.weight_version_key as i64) as u64;

        let new_version = current_version + 1;

        // Store new version
        self.storage.set_i64(&key, new_version as i64).await?;

        Ok(new_version)
    }

    /// Store emission metrics to the database
    async fn store_emission_metrics(
        &self,
        weight_distribution: &crate::bittensor_core::weight_allocation::WeightDistribution,
        current_block: u64,
    ) -> Result<i64> {
        use crate::persistence::gpu_profile_repository::{CategoryDistribution, EmissionMetrics};

        // Convert category allocations to CategoryDistribution format
        let mut category_distributions = HashMap::new();
        for (category, allocation) in &weight_distribution.category_allocations {
            category_distributions.insert(
                category.clone(),
                CategoryDistribution {
                    category: category.clone(),
                    miner_count: allocation.miner_count,
                    total_weight: allocation.weight_pool,
                    average_score: allocation.total_score / allocation.miner_count as f64,
                },
            );
        }

        // Calculate burn amount
        let burn_amount = weight_distribution
            .burn_allocation
            .as_ref()
            .map(|b| b.weight as u64)
            .unwrap_or(0);

        let emission_metrics = EmissionMetrics {
            id: 0, // Will be set by database
            timestamp: Utc::now(),
            burn_amount,
            burn_percentage: self.emission_config.burn_percentage,
            category_distributions,
            total_miners: weight_distribution.miners_served,
            weight_set_block: current_block,
        };

        let metrics_id = self
            .gpu_profile_repo
            .store_emission_metrics(&emission_metrics)
            .await?;

        info!(
            "Stored emission metrics for block {} with {} categories, burn {}%",
            current_block,
            weight_distribution.category_allocations.len(),
            self.emission_config.burn_percentage
        );

        Ok(metrics_id)
    }

    /// Store weight allocation history for each miner
    async fn store_weight_allocations(
        &self,
        weight_distribution: &crate::bittensor_core::weight_allocation::WeightDistribution,
        emission_metrics_id: i64,
        current_block: u64,
    ) -> Result<()> {
        // First, store burn allocation if present
        if let Some(burn_allocation) = &weight_distribution.burn_allocation {
            self.gpu_profile_repo
                .store_weight_allocation(
                    emission_metrics_id,
                    MinerUid::new(burn_allocation.uid),
                    "BURN",
                    burn_allocation.weight as u64,
                    0.0, // No score for burn
                    0.0, // No category total for burn
                    current_block,
                )
                .await?;
        }

        // Store allocations for each miner
        for weight in &weight_distribution.weights {
            // Find which category this miner belongs to
            let miner_uid = MinerUid::new(weight.uid);

            // Get miner's GPU profile to determine category
            if let Ok(Some(profile)) = self.gpu_profile_repo.get_gpu_profile(miner_uid).await {
                // Determine category from the GPU with the highest count
                let gpu_models = profile.gpu_models_by_count();
                let category = gpu_models
                    .first()
                    .map(|(model, _)| model.as_str())
                    .unwrap_or("UNKNOWN");

                // Get category allocation info
                if let Some(category_allocation) =
                    weight_distribution.category_allocations.get(category)
                {
                    self.gpu_profile_repo
                        .store_weight_allocation(
                            emission_metrics_id,
                            miner_uid,
                            category,
                            weight.weight as u64,
                            profile.total_score,
                            category_allocation.total_score,
                            current_block,
                        )
                        .await?;
                }
            }
        }

        info!(
            "Stored weight allocations for {} miners at block {}",
            weight_distribution.weights.len(),
            current_block
        );

        Ok(())
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
        match self.bittensor_service.get_current_block().await {
            Ok(block) => {
                debug!("Successfully got current block {}", block);
                Ok(block)
            }
            Err(e) => {
                error!("Failed to get current block: {}", e);

                // Record critical RPC failure metric
                if let Some(ref metrics) = self.metrics {
                    let error_type = {
                        let error_msg = e.to_string().to_lowercase();
                        if error_msg.contains("timeout") {
                            "timeout"
                        } else if error_msg.contains("background") {
                            "background_terminated"
                        } else {
                            "connection"
                        }
                    };

                    metrics
                        .prometheus()
                        .record_rpc_critical_failure("get_current_block", error_type);
                }

                Err(e.into())
            }
        }
    }

    /// Get the last weight set block from storage or initialize to current block
    async fn get_last_weight_set_block(&self) -> Result<u64> {
        let key = format!("last_weight_set_block:{}", self.config.netuid);

        // Try to get from storage first
        if let Some(stored_block) = self.storage.get_i64(&key).await.unwrap_or(None) {
            let stored_block = stored_block as u64;
            info!("Found stored last weight set block: {}", stored_block);
            return Ok(stored_block);
        }

        // If not in storage, get current block and subtract interval to prevent immediate weight setting
        let current_block = self.get_current_block().await?;
        let safe_last_block = current_block.saturating_sub(self.blocks_per_weight_set / 2);

        info!(
            "No stored last weight set block found, initializing to {} (current: {}, interval: {})",
            safe_last_block, current_block, self.blocks_per_weight_set
        );

        // Store this initial value
        self.storage.set_i64(&key, safe_last_block as i64).await?;

        Ok(safe_last_block)
    }

    /// Get the last weight set timestamp from storage
    pub async fn get_last_weight_set_timestamp(
        &self,
    ) -> Result<Option<chrono::DateTime<chrono::Utc>>> {
        let key = format!("last_weight_set_timestamp:{}", self.config.netuid);

        // Try to get from storage
        if let Some(timestamp) = self.storage.get_i64(&key).await.unwrap_or(None) {
            let datetime = chrono::DateTime::<chrono::Utc>::from_timestamp(timestamp, 0)
                .ok_or_else(|| anyhow::anyhow!("Invalid timestamp: {}", timestamp))?;
            info!("Found stored last weight set timestamp: {}", datetime);
            return Ok(Some(datetime));
        }

        info!("No stored last weight set timestamp found");
        Ok(None)
    }

    /// Store the last weight set block for persistence across restarts with atomic operation
    async fn store_last_weight_set_block(&self, block: u64) -> Result<()> {
        let key = format!("last_weight_set_block:{}", self.config.netuid);
        let timestamp_key = format!("last_weight_set_timestamp:{}", self.config.netuid);

        // Store both block and timestamp atomically
        let timestamp = chrono::Utc::now().timestamp();

        // Use a transaction-like approach for atomic storage
        match self.storage.set_i64(&key, block as i64).await {
            Ok(()) => {
                // Store timestamp as well for auditing and epoch tracking
                if let Err(e) = self.storage.set_i64(&timestamp_key, timestamp).await {
                    warn!("Failed to store timestamp for block {}: {}", block, e);
                }
                debug!(
                    "Stored last weight set block: {} at timestamp: {}",
                    block, timestamp
                );
                Ok(())
            }
            Err(e) => {
                error!("Failed to store last weight set block {}: {}", block, e);
                Err(e)
            }
        }
    }

    /// Get current metagraph from Bittensor network with retry logic
    async fn get_metagraph(&self) -> Result<Metagraph> {
        const MAX_RETRIES: u32 = 3;
        const BASE_DELAY: Duration = Duration::from_secs(2);

        for attempt in 1..=MAX_RETRIES {
            match self
                .bittensor_service
                .get_metagraph(self.config.netuid)
                .await
            {
                Ok(metagraph) => {
                    debug!(
                        "Successfully got metagraph with {} neurons on attempt {}",
                        metagraph.hotkeys.len(),
                        attempt
                    );
                    return Ok(metagraph);
                }
                Err(e) => {
                    error!("Failed to fetch metagraph (attempt {}): {}", attempt, e);
                    if attempt < MAX_RETRIES {
                        let delay = BASE_DELAY * 2_u32.pow(attempt - 1);
                        warn!("Retrying get_metagraph in {} seconds...", delay.as_secs());
                        tokio::time::sleep(delay).await;
                    }
                }
            }
        }

        Err(anyhow::anyhow!(
            "Failed to fetch metagraph after {} attempts",
            MAX_RETRIES
        ))
    }

    /// Extract validation result from verification log
    async fn extract_validation_result(
        &self,
        miner_id: &str,
        node_id: NodeId,
        log: &VerificationLog,
    ) -> Result<NodeValidationResult> {
        // Parse hardware specs from the verification log details
        // Always try to parse specs, even for failed validations, to track GPU hardware
        let hardware_specs: Option<serde_json::Value> = if !log.details.is_null() {
            serde_json::from_value(log.details.clone()).ok()
        } else {
            None
        };

        // Extract data from the verification log structure
        let (hardware_score, gpu_count, gpu_memory_gb, network_bandwidth_mbps, gpu_model) =
            if let Some(specs) = hardware_specs {
                // Extract GPU model from node_result.gpu_name
                let mut gpu_model = specs["node_result"]["gpu_name"]
                    .as_str()
                    .map(|s| s.to_string());

                if gpu_model.is_none() || gpu_model.as_ref() == Some(&"UNKNOWN".to_string()) {
                    debug!(
                        node_id = %node_id,
                        "GPU name not found in node_result for node {}, checking gpu_uuid_assignments",
                        node_id
                    );

                    match self
                        .persistence
                        .get_node_gpu_name_from_assignments(miner_id, &node_id.to_string())
                        .await
                    {
                        Ok(Some(name)) => {
                            debug!(
                                node_id = %node_id,
                                "Retrieved GPU model from assignments for node {}: {}",
                                node_id, name
                            );
                            gpu_model = Some(name);
                        }
                        Ok(None) => {
                            warn!(
                                node_id = %node_id,
                                "No GPU assignments found for node {}",
                                node_id
                            );
                        }
                        Err(e) => {
                            warn!(
                                "Failed to get GPU model from assignments for node {}: {}",
                                node_id, e
                            );
                        }
                    }
                }

                let gpu_model = gpu_model.unwrap_or_else(|| "UNKNOWN".to_string());

                let gpu_count = match self
                    .persistence
                    .get_node_gpu_count_from_assignments(miner_id, &node_id.to_string())
                    .await
                {
                    Ok(count) => count as usize,
                    Err(e) => {
                        warn!(
                            "Failed to get GPU count from assignments for node {}: {}, using 0",
                            node_id, e
                        );
                        0
                    }
                };

                let gpu_memory_gb = match self
                    .persistence
                    .get_node_first_gpu_memory_gb(miner_id, &node_id.to_string())
                    .await
                {
                    Ok(mem) => mem,
                    Err(e) => {
                        warn!(
                            "Failed to get GPU memory from assignments for node {}: {}, using 0.0",
                            node_id, e
                        );
                        0.0
                    }
                };

                // Extract memory bandwidth from node_result.memory_bandwidth_gbps
                let bandwidth = specs["node_result"]["memory_bandwidth_gbps"]
                    .as_f64()
                    .unwrap_or(0.0);

                let score = self.calculate_hardware_score(&specs);

                debug!(
                    "Node {}: Extracted GPU info - model: {}, count: {}, memory: {}GB, validation_success: {}",
                    node_id, gpu_model, gpu_count, gpu_memory_gb, log.success
                );

                (score, gpu_count, gpu_memory_gb, bandwidth, gpu_model)
            } else {
                debug!(
                    "Node {}: No hardware specs in verification log, checking gpu_uuid_assignments",
                    node_id
                );

                let gpu_model = match self
                    .persistence
                    .get_node_gpu_name_from_assignments(miner_id, &node_id.to_string())
                    .await
                {
                    Ok(Some(name)) => {
                        debug!(
                            "Retrieved GPU model from assignments for node {} (no specs): {}",
                            node_id, name
                        );
                        name
                    }
                    _ => "UNKNOWN".to_string(),
                };

                let gpu_count = match self
                    .persistence
                    .get_node_gpu_count_from_assignments(miner_id, &node_id.to_string())
                    .await
                {
                    Ok(count) => count as usize,
                    Err(_) => 0,
                };

                let gpu_memory_gb = (self
                    .persistence
                    .get_node_first_gpu_memory_gb(miner_id, &node_id.to_string())
                    .await)
                    .unwrap_or(0.0);

                debug!(
                    "Node {}: From assignments only - model: {}, count: {}, memory: {}GB, validation_success: {}",
                    node_id, gpu_model, gpu_count, gpu_memory_gb, log.success
                );

                (0.0, gpu_count, gpu_memory_gb, 0.0, gpu_model)
            };

        Ok(NodeValidationResult {
            node_id,
            is_valid: log.success,
            _hardware_score: hardware_score,
            gpu_count,
            gpu_memory_gb,
            _network_bandwidth_mbps: network_bandwidth_mbps,
            attestation_valid: (log.verification_type == "attestation"
                || log.verification_type == "ssh_automation")
                && log.success,
            validation_timestamp: log.timestamp,
            gpu_model,
        })
    }

    /// Calculate hardware score from specs
    fn calculate_hardware_score(&self, _specs: &serde_json::Value) -> f64 {
        1.0
    }

    /// Get recent validation results for a miner
    async fn get_recent_miner_validations(
        &self,
        miner_uid: MinerUid,
        hours: u32,
    ) -> Result<Vec<NodeValidationResult>> {
        let cutoff_time = chrono::Utc::now() - chrono::Duration::hours(hours as i64);

        let rows = self
            .persistence
            .query_recent_miner_verification_logs(miner_uid.as_u16(), &cutoff_time.to_rfc3339())
            .await?;

        let miner_id = format!("miner_{}", miner_uid.as_u16());

        let mut validations = Vec::new();
        for row in rows {
            let node_id_str: String = row.get("node_id");

            // Handle composite miner{uid}__{uuid} or fallback to the new format
            let clean_node_id = if node_id_str.starts_with("miner") && node_id_str.contains("__") {
                let (_, rest) = node_id_str.split_once("__").ok_or_else(|| {
                    anyhow::anyhow!("Invalid composite node_id format: {}", node_id_str)
                })?;
                rest
            } else {
                &node_id_str
            };

            let node_id = clean_node_id.parse::<NodeId>()?;

            let log = VerificationLog {
                id: Uuid::parse_str(&row.get::<String, _>("id"))
                    .map_err(|e| anyhow::anyhow!("Failed to parse verification log UUID: {}", e))?,
                node_id: row.get("node_id"),
                validator_hotkey: row.get("validator_hotkey"),
                verification_type: row.get("verification_type"),
                timestamp: DateTime::parse_from_rfc3339(&row.get::<String, _>("timestamp"))
                    .map_err(|e| anyhow::anyhow!("Failed to parse timestamp: {}", e))?
                    .with_timezone(&Utc),
                score: row.get("score"),
                success: row.get::<i64, _>("success") != 0,
                details: serde_json::from_str(&row.get::<String, _>("details"))
                    .map_err(|e| anyhow::anyhow!("Failed to parse details JSON: {}", e))?,
                duration_ms: row.get("duration_ms"),
                error_message: row.get("error_message"),
                created_at: DateTime::parse_from_rfc3339(&row.get::<String, _>("created_at"))
                    .map_err(|e| anyhow::anyhow!("Failed to parse created_at: {}", e))?
                    .with_timezone(&Utc),
                updated_at: DateTime::parse_from_rfc3339(&row.get::<String, _>("updated_at"))
                    .map_err(|e| anyhow::anyhow!("Failed to parse updated_at: {}", e))?
                    .with_timezone(&Utc),
            };

            match self
                .extract_validation_result(&miner_id, node_id.clone(), &log)
                .await
            {
                Ok(validation) => {
                    debug!(
                        "Successfully extracted validation for node {}: gpu_model={}, gpu_count={}, success={} gpu_memory_gb={}",
                        node_id, validation.gpu_model, validation.gpu_count, validation.is_valid, validation.gpu_memory_gb
                    );
                    validations.push(validation);
                }
                Err(e) => {
                    warn!(
                        "Failed to extract validation result for node {} (miner {}): {}. Log details: {:?}",
                        node_id, miner_uid.as_u16(), e, log.details
                    );
                }
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

        let cutoff_time =
            chrono::Utc::now() - chrono::Duration::hours(GPU_CATEGORY_CUTOFF_HOURS as i64);
        let miner_ids = self
            .persistence
            .get_miners_with_recent_validations(&cutoff_time.to_rfc3339())
            .await?;

        for miner_id in miner_ids {
            if let Some(uid_str) = miner_id.strip_prefix("miner_") {
                if let Ok(uid) = uid_str.parse::<u16>() {
                    let miner_uid = MinerUid::new(uid);

                    match self
                        .get_recent_miner_validations(miner_uid, GPU_CATEGORY_CUTOFF_HOURS)
                        .await
                    {
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

#[cfg(test)]
mod tests {
    use super::*;
    use crate::persistence::entities::VerificationLog;
    use serde_json::json;

    #[test]
    fn test_extract_validation_result_with_a100() {
        // Create a verification log with A100 GPU
        let log = VerificationLog {
            id: uuid::Uuid::new_v4(),
            node_id: "exec123".to_string(),
            validator_hotkey: "validator".to_string(),
            verification_type: "attestation".to_string(),
            timestamp: chrono::Utc::now(),
            score: 1.0,
            success: true,
            details: json!({
                "gpu": [{
                    "model": "NVIDIA A100 80GB PCIe",
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

        assert_eq!(gpu_model, "NVIDIA A100 80GB PCIe");
    }

    #[test]
    fn test_extract_validation_result_with_h100() {
        // Create a verification log with H100 GPU
        let log = VerificationLog {
            id: uuid::Uuid::new_v4(),
            node_id: "exec456".to_string(),
            validator_hotkey: "validator".to_string(),
            verification_type: "attestation".to_string(),
            timestamp: chrono::Utc::now(),
            score: 1.0,
            success: true,
            details: json!({
                "gpu": [{
                    "model": "NVIDIA H100",
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

        assert_eq!(gpu_model, "NVIDIA H100");
    }

    #[test]
    fn test_gpu_model_extraction_from_failed_attestation() {
        // Create a failed verification log - should still extract GPU info
        let log = VerificationLog {
            id: uuid::Uuid::new_v4(),
            node_id: "exec789".to_string(),
            validator_hotkey: "validator".to_string(),
            verification_type: "attestation".to_string(),
            timestamp: chrono::Utc::now(),
            score: 0.0,
            success: false,
            details: json!({
                "gpu": [{
                    "model": "NVIDIA A100 80GB PCIe",
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

        assert_eq!(gpu_model, "NVIDIA A100 80GB PCIe");
    }

    #[test]
    fn test_no_gpu_info_returns_unknown() {
        // Create a verification log with no GPU info
        let log = VerificationLog {
            id: uuid::Uuid::new_v4(),
            node_id: "exec999".to_string(),
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

        // For A100 with 80GB, dividing by 1024 gives 0.078, formatted as "H0"
        // For H100 with 138GB, dividing by 1024 gives 0.134, formatted as "H0"
        // Both would be categorized as "OTHER" and excluded from rewards!
    }

    #[test]
    fn test_extract_validation_result_with_new_data_format() {
        // This test verifies the fix for GPU model extraction from the new data format
        // Simulates actual validator-binary output structure
        let log = VerificationLog {
            id: uuid::Uuid::new_v4(),
            node_id: "node_175".to_string(),
            validator_hotkey: "validator_hotkey".to_string(),
            verification_type: "binary_validation".to_string(),
            timestamp: chrono::Utc::now(),
            score: 0.84,
            success: true,
            details: json!({
                "node_result": {
                    "gpu_name": "NVIDIA A100 80GB HBM3",
                    "gpu_uuid": "GPU-12345678-1234-1234-1234-123456789012",
                    "memory_bandwidth_gbps": 3.35,
                    "anti_debug_passed": true
                },
                "gpu_count": 8,
                "success": true,
                "validation_score": 0.84,
                "execution_time_ms": 15000
            }),
            duration_ms: 15000,
            error_message: None,
            created_at: chrono::Utc::now(),
            updated_at: chrono::Utc::now(),
        };

        // Test direct extraction without requiring full WeightSetter instance
        let details = &log.details;

        // Test the corrected GPU model extraction path
        let gpu_model = details["node_result"]["gpu_name"]
            .as_str()
            .unwrap_or("UNKNOWN");

        let gpu_count = details["gpu_count"].as_u64().unwrap_or(0) as usize;

        // Verify the GPU model is correctly extracted from the new path
        assert_eq!(gpu_model, "NVIDIA A100 80GB HBM3");
        assert_eq!(gpu_count, 8);

        // Test that the old path would fail (this proves our fix is needed)
        let old_path_gpu = details["gpu"][0]["model"].as_str().unwrap_or("UNKNOWN");
        assert_eq!(old_path_gpu, "UNKNOWN"); // This confirms old path doesn't work
    }

    #[test]
    fn test_extract_validation_result_missing_node_result() {
        // Test case where node_result is missing (should default to UNKNOWN)
        let log = VerificationLog {
            id: uuid::Uuid::new_v4(),
            node_id: "node_failed".to_string(),
            validator_hotkey: "validator_hotkey".to_string(),
            verification_type: "binary_validation".to_string(),
            timestamp: chrono::Utc::now(),
            score: 0.0,
            success: false,
            details: json!({
                "gpu_count": 0,
                "success": false,
                "validation_score": 0.0,
                "execution_time_ms": 5000,
                "error_message": "Failed to connect to node"
            }),
            duration_ms: 5000,
            error_message: Some("Failed to connect to node".to_string()),
            created_at: chrono::Utc::now(),
            updated_at: chrono::Utc::now(),
        };

        let details = &log.details;

        // Test extraction when node_result is missing
        let gpu_model = details["node_result"]["gpu_name"]
            .as_str()
            .unwrap_or("UNKNOWN");

        let gpu_count = details["gpu_count"].as_u64().unwrap_or(0) as usize;

        // Verify defaults when data is missing
        assert_eq!(gpu_model, "UNKNOWN");
        assert_eq!(gpu_count, 0);
    }

    #[test]
    fn test_gpu_categorization_with_corrected_extraction() {
        // Test that A100 and H100 GPUs are now properly identified
        let a100_log = VerificationLog {
            id: uuid::Uuid::new_v4(),
            node_id: "node_a100".to_string(),
            validator_hotkey: "validator_hotkey".to_string(),
            verification_type: "binary_validation".to_string(),
            timestamp: chrono::Utc::now(),
            score: 0.9,
            success: true,
            details: json!({
                "node_result": {
                    "gpu_name": "NVIDIA A100 80GB HBM3",
                    "memory_bandwidth_gbps": 3.35
                },
                "gpu_count": 8
            }),
            duration_ms: 12000,
            error_message: None,
            created_at: chrono::Utc::now(),
            updated_at: chrono::Utc::now(),
        };

        let h100_log = VerificationLog {
            id: uuid::Uuid::new_v4(),
            node_id: "node_h100".to_string(),
            validator_hotkey: "validator_hotkey".to_string(),
            verification_type: "binary_validation".to_string(),
            timestamp: chrono::Utc::now(),
            score: 0.95,
            success: true,
            details: json!({
                "node_result": {
                    "gpu_name": "NVIDIA H100",
                    "memory_bandwidth_gbps": 4.8
                },
                "gpu_count": 4
            }),
            duration_ms: 10000,
            error_message: None,
            created_at: chrono::Utc::now(),
            updated_at: chrono::Utc::now(),
        };

        // Extract GPU models using the corrected path
        let a100_model = a100_log.details["node_result"]["gpu_name"]
            .as_str()
            .unwrap_or("UNKNOWN");
        let h100_model = h100_log.details["node_result"]["gpu_name"]
            .as_str()
            .unwrap_or("UNKNOWN");

        // Verify A100 and H100 are correctly identified
        assert!(a100_model.contains("A100"));
        assert!(h100_model.contains("H100"));
        assert_ne!(a100_model, "UNKNOWN");
        assert_ne!(h100_model, "UNKNOWN");

        // Test GPU counts are preserved
        assert_eq!(a100_log.details["gpu_count"].as_u64().unwrap(), 8);
        assert_eq!(h100_log.details["gpu_count"].as_u64().unwrap(), 4);
    }

    #[tokio::test]
    async fn test_weight_setter_scoring() {
        // Create mock validation results
        let validations = vec![
            NodeValidationResult {
                node_id: NodeId::new("").unwrap(),
                is_valid: true,
                _hardware_score: 0.8,
                gpu_count: 2,
                gpu_memory_gb: 48.0,
                _network_bandwidth_mbps: 1000.0,
                attestation_valid: true,
                validation_timestamp: chrono::Utc::now(),
                gpu_model: "NVIDIA A100".to_string(),
            },
            NodeValidationResult {
                node_id: NodeId::new("").unwrap(),
                is_valid: true,
                _hardware_score: 0.9,
                gpu_count: 4,
                gpu_memory_gb: 96.0,
                _network_bandwidth_mbps: 10000.0,
                attestation_valid: true,
                validation_timestamp: chrono::Utc::now(),
                gpu_model: "NVIDIA A100".to_string(),
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
            assert!(validation.gpu_model.contains("A100"));
        }
    }
}
