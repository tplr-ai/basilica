use anyhow::Result;
use chrono::{DateTime, Utc};
use std::collections::HashMap;
use std::sync::Arc;
use tracing::{debug, error, info, warn};

use super::categorization::{MinerGpuProfile, NodeValidationResult};
use crate::config::emission::EmissionConfig;
use crate::metrics::ValidatorMetrics;
use crate::persistence::gpu_profile_repository::GpuProfileRepository;
use crate::persistence::SimplePersistence;
use basilica_common::identity::MinerUid;
use basilica_common::types::GpuCategory;
use std::str::FromStr;

pub struct GpuScoringEngine {
    gpu_profile_repo: Arc<GpuProfileRepository>,
    persistence: Arc<SimplePersistence>,
    metrics: Option<Arc<ValidatorMetrics>>,
    emission_config: EmissionConfig,
}

impl GpuScoringEngine {
    pub fn new(
        gpu_profile_repo: Arc<GpuProfileRepository>,
        persistence: Arc<SimplePersistence>,
        emission_config: EmissionConfig,
    ) -> Self {
        Self {
            gpu_profile_repo,
            persistence,
            metrics: None,
            emission_config,
        }
    }

    /// Create new engine with metrics support
    pub fn with_metrics(
        gpu_profile_repo: Arc<GpuProfileRepository>,
        persistence: Arc<SimplePersistence>,
        metrics: Arc<ValidatorMetrics>,
        emission_config: EmissionConfig,
    ) -> Self {
        Self {
            gpu_profile_repo,
            persistence,
            metrics: Some(metrics),
            emission_config,
        }
    }

    /// Update miner profile from validation results
    pub async fn update_miner_profile_from_validation(
        &self,
        miner_uid: MinerUid,
        node_validations: Vec<NodeValidationResult>,
    ) -> Result<MinerGpuProfile> {
        // Calculate verification score from node results
        let new_score = self.calculate_verification_score(&node_validations);

        // Check if there are any successful validations
        let has_successful_validation = node_validations
            .iter()
            .any(|v| v.is_valid && v.attestation_valid);

        // Create or update the profile with the calculated score
        let mut profile = MinerGpuProfile::new(miner_uid, &node_validations, new_score);

        // If there's a successful validation, update the timestamp
        if has_successful_validation {
            profile.last_successful_validation = Some(Utc::now());
        }

        // Store the profile
        self.gpu_profile_repo.upsert_gpu_profile(&profile).await?;

        info!(
            miner_uid = miner_uid.as_u16(),
            score = new_score,
            total_gpus = profile.total_gpu_count(),
            validations = node_validations.len(),
            gpu_distribution = ?profile.gpu_counts,
            "Updated miner GPU profile with GPU count weighting"
        );

        // Record metrics if available
        if let Some(metrics) = &self.metrics {
            // Record miner GPU profile metrics
            metrics.prometheus().record_miner_gpu_count_and_score(
                miner_uid.as_u16(),
                profile.total_gpu_count(),
                new_score,
            );

            // Record individual node GPU counts
            for validation in &node_validations {
                if validation.is_valid && validation.attestation_valid {
                    metrics.prometheus().record_node_gpu_count(
                        miner_uid.as_u16(),
                        &validation.node_id,
                        &validation.gpu_model,
                        validation.gpu_count,
                    );

                    // Record successful validation
                    metrics.prometheus().record_miner_successful_validation(
                        miner_uid.as_u16(),
                        &validation.node_id,
                    );

                    // Record GPU profile
                    metrics.prometheus().record_miner_gpu_profile(
                        miner_uid.as_u16(),
                        &validation.gpu_model,
                        &validation.node_id,
                        validation.gpu_count as u32,
                    );

                    // Also record through business metrics for complete tracking
                    metrics
                        .business()
                        .record_gpu_profile_validation(
                            miner_uid.as_u16(),
                            &validation.node_id,
                            &validation.gpu_model,
                            validation.gpu_count,
                            validation.is_valid && validation.attestation_valid,
                            new_score,
                        )
                        .await;
                }
            }
        }

        Ok(profile)
    }

    /// Check if a GPU model is configured for rewards based on emission config
    fn is_gpu_model_rewardable(&self, gpu_model: &str) -> bool {
        let category = GpuCategory::from_str(gpu_model).unwrap();
        let normalized_model = category.to_string();
        self.emission_config
            .gpu_allocations
            .contains_key(&normalized_model)
    }

    /// Calculate verification score from node results
    fn calculate_verification_score(&self, node_validations: &[NodeValidationResult]) -> f64 {
        if node_validations.is_empty() {
            return 0.0;
        }

        let mut valid_count = 0;
        let mut total_count = 0;
        let mut total_gpu_count = 0;
        let mut unique_nodes = std::collections::HashSet::new();

        // count unique nodes and their GPU counts
        for validation in node_validations {
            unique_nodes.insert(&validation.node_id);
            total_count += 1;

            // Count valid attestations and accumulate GPU counts
            if validation.is_valid && validation.attestation_valid {
                valid_count += 1;
            }
        }

        // sum GPU counts from unique nodes only
        let mut seen_nodes = std::collections::HashSet::new();
        for validation in node_validations {
            if validation.is_valid
                && validation.attestation_valid
                && seen_nodes.insert(&validation.node_id)
            {
                total_gpu_count += validation.gpu_count;
            }
        }

        if total_count > 0 {
            // Calculate base pass/fail ratio
            let final_score = valid_count as f64 / total_count as f64;

            // Log the actual GPU-weighted score for transparency
            let gpu_weighted_score = final_score * total_gpu_count as f64;

            debug!(
                validations = node_validations.len(),
                valid_count = valid_count,
                total_count = total_count,
                unique_nodes = unique_nodes.len(),
                total_gpu_count = total_gpu_count,
                final_score = final_score,
                gpu_weighted_score = gpu_weighted_score,
                "Calculated verification score (normalized for DB, GPU count tracked separately)"
            );
            final_score
        } else {
            warn!(
                validations = node_validations.len(),
                "No validations found for score calculation"
            );
            0.0
        }
    }

    /// Calculate uptime-based multiplication factor for a specific node
    /// Uses 14-day linear ramp-up based on actual uptime from verification logs
    async fn calculate_uptime_multiplication_factor(
        &self,
        miner_uid: MinerUid,
        node_id: &str,
    ) -> f64 {
        let miner_id = format!("miner_{}", miner_uid.as_u16());

        match self
            .persistence
            .calculate_node_uptime_multiplier(&miner_id, node_id)
            .await
        {
            Ok((uptime_minutes, multiplier)) => {
                debug!(
                    miner_uid = miner_uid.as_u16(),
                    node_id = %node_id,
                    uptime_minutes = uptime_minutes,
                    multiplier = multiplier,
                    "Applied uptime multiplication factor"
                );

                // Emit Prometheus metrics
                if let Some(metrics) = &self.metrics {
                    metrics.prometheus().record_node_uptime_metrics(
                        miner_uid.as_u16(),
                        node_id,
                        uptime_minutes,
                        multiplier,
                    );
                }

                multiplier
            }
            Err(e) => {
                error!(
                    miner_uid = miner_uid.as_u16(),
                    node_id = %node_id,
                    error = %e,
                    "Failed to calculate node uptime multiplier, using default 1.0"
                );
                1.0 // Fallback to no penalty on error
            }
        }
    }

    /// Get all miners grouped by GPU category with multi-category support
    /// A single miner can appear in multiple categories if they have multiple GPU types
    /// Only includes GPU categories configured in emission config for rewards
    /// Filters out miners without active axons on the chain
    /// Only includes miners with successful validations since the given timestamp
    pub async fn get_miners_by_gpu_category_since_epoch(
        &self,
        epoch_timestamp: Option<DateTime<Utc>>,
        cutoff_hours: u32,
        metagraph: &bittensor::Metagraph,
    ) -> Result<HashMap<String, Vec<(MinerUid, f64)>>> {
        let all_profiles = self.gpu_profile_repo.get_all_gpu_profiles().await?;
        let cutoff_time = Utc::now() - chrono::Duration::hours(cutoff_hours as i64);

        let mut miners_by_category = HashMap::new();

        for profile in all_profiles {
            // Filter by cutoff time
            if profile.last_updated < cutoff_time {
                continue;
            }

            // Filter by last successful validation epoch if provided
            if let Some(epoch) = epoch_timestamp {
                // Skip miners who haven't had successful validations since the last epoch
                match profile.last_successful_validation {
                    Some(last_validation) if last_validation >= epoch => {
                        // Miner has successful validation since epoch, include them
                    }
                    _ => {
                        debug!(
                            miner_uid = profile.miner_uid.as_u16(),
                            last_validation = ?profile.last_successful_validation,
                            epoch = ?epoch,
                            "Skipping miner: No successful validation since last epoch"
                        );
                        continue;
                    }
                }
            }

            // Check if miner has active axon on chain
            let uid_index = profile.miner_uid.as_u16() as usize;
            if uid_index >= metagraph.hotkeys.len() {
                debug!(
                    miner_uid = profile.miner_uid.as_u16(),
                    "Skipping miner: UID exceeds metagraph size"
                );
                continue;
            }

            // Check if the UID has an active axon (non-zero IP and port)
            let Some(axon) = metagraph.axons.get(uid_index) else {
                debug!(
                    miner_uid = profile.miner_uid.as_u16(),
                    "Skipping miner: No axon found for UID"
                );
                continue;
            };

            if axon.port == 0 || axon.ip == 0 {
                debug!(
                    miner_uid = profile.miner_uid.as_u16(),
                    "Skipping miner: Inactive axon (zero IP or port)"
                );
                continue;
            }

            let rewardable_gpus: Vec<(String, GpuCategory, u32)> = self
                .gpu_profile_repo
                .get_miner_gpu_assignments(profile.miner_uid)
                .await?.iter().filter_map(|(node_id, (gpu_count, gpu_name, gpu_memory_gb))| {
                    if *gpu_count > 0 {
                        let category = GpuCategory::from_str(gpu_name).unwrap();
                        let normalized_model = category.to_string();
                        // Only include GPUs configured in emission config for rewards
                        if self.is_gpu_model_rewardable(gpu_name) {
                            // Check if miner meets minimum GPU count and VRAM requirements
                            if let Some(allocation) = self.emission_config.get_gpu_allocation(&normalized_model) {
                                let meets_gpu_count = *gpu_count >= allocation.min_gpu_count;
                                let meets_vram = if let Some(min_vram) = allocation.min_gpu_vram {
                                    // Check if the miner's GPU has enough VRAM
                                    min_vram == 1 || min_vram == 0 || *gpu_memory_gb >= min_vram as f64
                                } else {
                                    // No VRAM requirement
                                    true
                                };

                                if meets_gpu_count && meets_vram {
                                    info!(
                                        miner_uid = profile.miner_uid.as_u16(),
                                        node_id = %node_id,
                                        gpu_model = %gpu_name,
                                        gpu_count = *gpu_count,
                                        min_required = allocation.min_gpu_count,
                                        "Miner meets all emission requirements"
                                    );
                                    Some((node_id.clone(), category, *gpu_count))
                                } else {
                                    if !meets_gpu_count {
                                        info!(
                                            miner_uid = profile.miner_uid.as_u16(),
                                            node_id = %node_id,
                                            gpu_model = %gpu_name,
                                            gpu_count = *gpu_count,
                                            min_required = allocation.min_gpu_count,
                                            "Skipping miner: Does not meet minimum GPU count requirement"
                                        );
                                    }
                                    if !meets_vram {
                                        info!(
                                            miner_uid = profile.miner_uid.as_u16(),
                                            node_id = %node_id,
                                            gpu_model = %gpu_name,
                                            gpu_vram = *gpu_memory_gb,
                                            min_required = allocation.min_gpu_vram,
                                            "Skipping miner: Does not meet minimum GPU VRAM requirement"
                                        );
                                    }
                                    None
                                }
                            } else {
                                None
                            }
                        } else {
                            None
                        }
                    } else {
                        None
                    }
                })
                .collect();

            let mut rewardable_gpu_counts: HashMap<String, f64> = HashMap::new();
            for (node_id, category, count) in rewardable_gpus {
                let normalized_model = category.to_string();
                let uptime_factor = self
                    .calculate_uptime_multiplication_factor(profile.miner_uid, &node_id)
                    .await;
                let weighted_count = count as f64 * uptime_factor;
                *rewardable_gpu_counts.entry(normalized_model).or_insert(0.0) += weighted_count;
            }

            // Skip miners with no rewardable GPUs
            if rewardable_gpu_counts.is_empty() {
                continue;
            }

            // Add the miner to each rewardable category they have GPUs in
            for (normalized_model, weighted_gpu_count) in rewardable_gpu_counts {
                // Multiply by weighted_gpu_count to get the actual linear score
                let category_score = profile.total_score * weighted_gpu_count;

                miners_by_category
                    .entry(normalized_model)
                    .or_insert_with(Vec::new)
                    .push((profile.miner_uid, category_score));
            }
        }

        // Sort miners within each category by score (descending)
        for miners in miners_by_category.values_mut() {
            miners.sort_by(|a, b| b.1.partial_cmp(&a.1).unwrap_or(std::cmp::Ordering::Equal));
        }

        info!(
            categories = miners_by_category.len(),
            total_entries = miners_by_category.values().map(|v| v.len()).sum::<usize>(),
            cutoff_hours = cutoff_hours,
            metagraph_size = metagraph.hotkeys.len(),
            "Retrieved miners by GPU category (configured models only for rewards, with active axon validation)"
        );

        Ok(miners_by_category)
    }

    /// Get category statistics with multi-category support
    /// Statistics are calculated per category based on proportional scores
    /// Only includes GPU categories configured in emission config for rewards
    pub async fn get_category_statistics(&self) -> Result<HashMap<String, CategoryStats>> {
        let all_profiles = self.gpu_profile_repo.get_all_gpu_profiles().await?;
        let mut category_stats = HashMap::new();

        for profile in all_profiles {
            // Only consider GPUs listed in emission config for rewards
            let rewardable_gpu_counts: HashMap<String, u32> = profile
                .gpu_counts
                .iter()
                .filter_map(|(gpu_model, &gpu_count)| {
                    if gpu_count > 0 {
                        let category = GpuCategory::from_str(gpu_model).unwrap();
                        let normalized_model = category.to_string();
                        // Only include GPUs configured in emission config for rewards
                        if self.is_gpu_model_rewardable(gpu_model) {
                            // Check if miner meets minimum GPU count requirement
                            if let Some(allocation) =
                                self.emission_config.get_gpu_allocation(&normalized_model)
                            {
                                if gpu_count >= allocation.min_gpu_count {
                                    Some((normalized_model, gpu_count))
                                } else {
                                    None
                                }
                            } else {
                                None
                            }
                        } else {
                            None
                        }
                    } else {
                        None
                    }
                })
                .collect();

            // Skip miners with no rewardable GPUs
            if rewardable_gpu_counts.is_empty() {
                continue;
            }

            let total_rewardable_gpus: u32 = rewardable_gpu_counts.values().sum();

            // Add stats for each rewardable category the miner has GPUs in
            for (normalized_model, gpu_count) in rewardable_gpu_counts {
                // Calculate proportional score based on rewardable GPU count
                let category_score = if total_rewardable_gpus > 0 {
                    profile.total_score * (gpu_count as f64 / total_rewardable_gpus as f64)
                } else {
                    0.0
                };

                let stats =
                    category_stats
                        .entry(normalized_model)
                        .or_insert_with(|| CategoryStats {
                            miner_count: 0,
                            total_score: 0.0,
                            min_score: f64::MAX,
                            max_score: f64::MIN,
                            average_score: 0.0,
                        });

                stats.miner_count += 1;
                stats.total_score += category_score;
                stats.min_score = stats.min_score.min(category_score);
                stats.max_score = stats.max_score.max(category_score);
            }
        }

        // Calculate averages
        for stats in category_stats.values_mut() {
            if stats.miner_count > 0 {
                stats.average_score = stats.total_score / stats.miner_count as f64;
            }

            // Fix edge case where no miners exist
            if stats.min_score == f64::MAX {
                stats.min_score = 0.0;
            }
            if stats.max_score == f64::MIN {
                stats.max_score = 0.0;
            }
        }

        Ok(category_stats)
    }
}

#[derive(Debug, Clone, serde::Serialize, serde::Deserialize)]
pub struct CategoryStats {
    pub miner_count: u32,
    pub average_score: f64,
    pub total_score: f64,
    pub min_score: f64,
    pub max_score: f64,
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::persistence::gpu_profile_repository::GpuProfileRepository;
    use crate::persistence::SimplePersistence;
    use basilica_common::identity::MinerUid;
    use chrono::Utc;
    use std::collections::HashMap;
    use std::sync::Arc;

    /// Helper function to create a test MinerGpuProfile without specific memory requirements
    fn create_test_profile(
        miner_uid: u16,
        gpu_counts: HashMap<String, u32>,
        total_score: f64,
        now: DateTime<Utc>,
    ) -> MinerGpuProfile {
        MinerGpuProfile {
            miner_uid: MinerUid::new(miner_uid),
            gpu_counts,
            total_score,
            verification_count: 1,
            last_updated: now,
            last_successful_validation: Some(now - chrono::Duration::hours(1)),
        }
    }

    /// Helper function to insert a test miner
    async fn insert_test_miner(
        persistence: &SimplePersistence,
        miner_id: &str,
        hotkey: &str,
        registered_at: DateTime<Utc>,
    ) -> anyhow::Result<()> {
        let now = Utc::now();
        sqlx::query(
            "INSERT INTO miners (id, hotkey, endpoint, last_seen, registered_at, updated_at, node_info)
             VALUES (?, ?, ?, ?, ?, ?, ?)"
        )
        .bind(miner_id)
        .bind(hotkey)
        .bind("127.0.0.1:8080")
        .bind(now.to_rfc3339())
        .bind(registered_at.to_rfc3339())
        .bind(now.to_rfc3339())
        .bind("{}")
        .execute(persistence.pool())
        .await?;
        Ok(())
    }

    /// Helper function to insert a test miner node
    async fn insert_test_miner_node(
        persistence: &SimplePersistence,
        miner_id: &str,
        node_id: &str,
        gpu_count: i64,
        created_at: DateTime<Utc>,
    ) -> anyhow::Result<()> {
        let now = Utc::now();
        sqlx::query(
            "INSERT INTO miner_nodes (id, miner_id, node_id, ssh_endpoint, gpu_count, status, created_at, updated_at)
             VALUES (?, ?, ?, ?, ?, ?, ?, ?)"
        )
        .bind(format!("{}:{}", miner_id, node_id))
        .bind(miner_id)
        .bind(node_id)
        .bind("127.0.0.1:8080")
        .bind(gpu_count)
        .bind("online")
        .bind(created_at.to_rfc3339())
        .bind(now.to_rfc3339())
        .execute(persistence.pool())
        .await?;
        Ok(())
    }

    /// Helper function to insert a test GPU UUID assignment
    async fn insert_test_gpu_uuid(
        persistence: &SimplePersistence,
        miner_id: &str,
        node_id: &str,
        gpu_name: &str,
    ) -> anyhow::Result<()> {
        let now = Utc::now();
        let gpu_uuid = format!("test-gpu-uuid-{}-{}", miner_id, node_id);
        sqlx::query(
            "INSERT INTO gpu_uuid_assignments (gpu_uuid, gpu_index, node_id, miner_id, gpu_name, last_verified)
             VALUES (?, ?, ?, ?, ?, ?)"
        )
        .bind(gpu_uuid)
        .bind(0i32)
        .bind(node_id)
        .bind(miner_id)
        .bind(gpu_name)
        .bind(now.to_rfc3339())
        .execute(persistence.pool())
        .await?;
        Ok(())
    }

    /// Helper function to insert a test verification log
    async fn insert_test_verification_log(
        persistence: &SimplePersistence,
        node_id: &str,
        timestamp: DateTime<Utc>,
        success: bool,
        with_binary_validation: bool,
    ) -> anyhow::Result<()> {
        let now = Utc::now();
        let log_id = uuid::Uuid::new_v4().to_string();
        let score = if success { 1.0 } else { 0.0 };
        let success_int = if success { 1i32 } else { 0i32 };
        let binary_validation = if with_binary_validation {
            Some("binary_validation_data")
        } else {
            None
        };

        sqlx::query(
            "INSERT INTO verification_logs (id, node_id, validator_hotkey, verification_type, timestamp, score, success, details, duration_ms, last_binary_validation, created_at, updated_at)
             VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?)"
        )
        .bind(log_id)
        .bind(node_id)
        .bind("test_validator_hotkey")
        .bind("gpu_validation")
        .bind(timestamp.to_rfc3339())
        .bind(score)
        .bind(success_int)
        .bind("{}")
        .bind(1000i64)
        .bind(binary_validation)
        .bind(now.to_rfc3339())
        .bind(now.to_rfc3339())
        .execute(persistence.pool())
        .await?;
        Ok(())
    }

    /// Helper function to seed all required data for GPU profile tests
    async fn seed_test_data(
        persistence: &SimplePersistence,
        gpu_repo: &GpuProfileRepository,
        profiles: &[MinerGpuProfile],
    ) -> anyhow::Result<()> {
        let now = Utc::now();

        for profile in profiles {
            // Store basic profile data
            gpu_repo.upsert_gpu_profile(profile).await?;

            let miner_id = format!("miner_{}", profile.miner_uid.as_u16());
            let node_id = format!(
                "miner{}__test-node-{}",
                profile.miner_uid.as_u16(),
                profile.miner_uid.as_u16()
            );

            // Seed miners table first (required for foreign key constraint)
            sqlx::query(
                "INSERT OR REPLACE INTO miners (id, hotkey, endpoint, last_seen, registered_at, updated_at, node_info)
                 VALUES (?, ?, ?, ?, ?, ?, ?)"
            )
            .bind(&miner_id)
            .bind(format!("hotkey_{}", profile.miner_uid.as_u16()))
            .bind("127.0.0.1:8080")
            .bind(now.to_rfc3339())
            .bind(now.to_rfc3339())
            .bind(now.to_rfc3339())
            .bind("{}")
            .execute(persistence.pool())
            .await?;

            // Seed gpu_uuid_assignments table
            for (gpu_model, count) in &profile.gpu_counts {
                for i in 0..*count {
                    let gpu_uuid =
                        format!("gpu-{}-{}-{}", profile.miner_uid.as_u16(), gpu_model, i);
                    sqlx::query(
                        "INSERT INTO gpu_uuid_assignments (gpu_uuid, gpu_index, node_id, miner_id, gpu_name, gpu_memory_gb, last_verified)
                         VALUES (?, ?, ?, ?, ?, ?, ?)"
                    )
                    .bind(&gpu_uuid)
                    .bind(i as i32)
                    .bind(&node_id)
                    .bind(&miner_id)
                    .bind(gpu_model)
                    .bind(80i64) // Default 80GB for test data
                    .bind(now.to_rfc3339())
                    .execute(persistence.pool())
                    .await?;
                }
            }

            // Seed miner_nodes table
            sqlx::query(
                "INSERT INTO miner_nodes (id, miner_id, node_id, ssh_endpoint, gpu_count, status, created_at, updated_at)
                 VALUES (?, ?, ?, ?, ?, ?, ?, ?)"
            )
            .bind(&node_id)
            .bind(&miner_id)
            .bind(&node_id)
            .bind("127.0.0.1:8080")
            .bind(profile.gpu_counts.values().sum::<u32>() as i64)
            .bind("online")
            .bind(now.to_rfc3339())
            .bind(now.to_rfc3339())
            .execute(persistence.pool())
            .await?;

            // Seed verification_logs table if there's a successful validation
            if let Some(last_successful) = profile.last_successful_validation {
                let log_id = uuid::Uuid::new_v4().to_string();
                sqlx::query(
                    "INSERT INTO verification_logs (id, node_id, validator_hotkey, verification_type, timestamp, score, success, details, duration_ms, error_message, created_at, updated_at)
                     VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?)"
                )
                .bind(&log_id)
                .bind(&node_id)
                .bind("test_validator_hotkey")
                .bind("gpu_validation")
                .bind(last_successful.to_rfc3339())
                .bind(profile.total_score)
                .bind(1)
                .bind("{}")
                .bind(1000i64)
                .bind(Option::<String>::None)
                .bind(now.to_rfc3339())
                .bind(now.to_rfc3339())
                .execute(persistence.pool())
                .await?;
            }
        }

        Ok(())
    }

    async fn create_test_gpu_profile_repo(
    ) -> Result<(Arc<GpuProfileRepository>, Arc<SimplePersistence>)> {
        let persistence = Arc::new(crate::persistence::SimplePersistence::for_testing().await?);
        let repo = Arc::new(GpuProfileRepository::new(persistence.pool().clone()));
        Ok((repo, persistence))
    }

    #[tokio::test]
    async fn test_verification_score_calculation() {
        let (repo, persistence) = create_test_gpu_profile_repo().await.unwrap();
        let engine = GpuScoringEngine::new(repo, persistence, EmissionConfig::for_testing());

        // Test with valid attestations
        let validations = vec![
            NodeValidationResult {
                node_id: "exec1".to_string(),
                is_valid: true,
                gpu_model: "A100".to_string(),
                gpu_count: 2,
                gpu_memory_gb: 80.0,
                attestation_valid: true,
                validation_timestamp: Utc::now(),
            },
            NodeValidationResult {
                node_id: "exec2".to_string(),
                is_valid: true,
                gpu_model: "A100".to_string(),
                gpu_count: 1,
                gpu_memory_gb: 80.0,
                attestation_valid: true,
                validation_timestamp: Utc::now(),
            },
        ];

        let score = engine.calculate_verification_score(&validations);
        // 2 valid validations: validation_ratio = 1.0
        // Actual GPU weight = 1.0 * 3 = 3.0
        let expected = 1.0;
        assert!((score - expected).abs() < 0.001);

        // Test with invalid attestations
        let invalid_validations = vec![NodeValidationResult {
            node_id: "exec1".to_string(),
            is_valid: false,
            gpu_model: "A100".to_string(),
            gpu_count: 2,
            gpu_memory_gb: 80.0,
            attestation_valid: false,
            validation_timestamp: Utc::now(),
        }];

        let score = engine.calculate_verification_score(&invalid_validations);
        assert_eq!(score, 0.0);

        // Test with mixed results
        let mixed_validations = vec![
            NodeValidationResult {
                node_id: "exec1".to_string(),
                is_valid: true,
                gpu_model: "A100".to_string(),
                gpu_count: 2,
                gpu_memory_gb: 80.0,
                attestation_valid: true,
                validation_timestamp: Utc::now(),
            },
            NodeValidationResult {
                node_id: "exec2".to_string(),
                is_valid: false,
                gpu_model: "A100".to_string(),
                gpu_count: 1,
                gpu_memory_gb: 80.0,
                attestation_valid: false,
                validation_timestamp: Utc::now(),
            },
        ];

        let score = engine.calculate_verification_score(&mixed_validations);
        // 1 valid out of 2 = 0.5 validation ratio
        // Actual GPU weight = 0.5 * 2 = 1.0
        let expected = 0.5;
        assert!((score - expected).abs() < 0.001);

        // Test with empty validations
        let empty_validations = vec![];
        let score = engine.calculate_verification_score(&empty_validations);
        assert_eq!(score, 0.0);

        // Test that pass/fail scoring gives 1.0 for valid attestations regardless of memory
        let high_memory_validations = vec![NodeValidationResult {
            node_id: "exec1".to_string(),
            is_valid: true,
            gpu_model: "A100".to_string(),
            gpu_count: 1,
            gpu_memory_gb: 80.0,
            attestation_valid: true,
            validation_timestamp: Utc::now(),
        }];

        let low_memory_validations = vec![NodeValidationResult {
            node_id: "exec1".to_string(),
            is_valid: true,
            gpu_model: "A100".to_string(),
            gpu_count: 1,
            gpu_memory_gb: 16.0,
            attestation_valid: true,
            validation_timestamp: Utc::now(),
        }];

        let high_score = engine.calculate_verification_score(&high_memory_validations);
        let low_score = engine.calculate_verification_score(&low_memory_validations);
        // Actual GPU weight = 1.0 * 1 = 1.0
        assert_eq!(high_score, 1.0);
        assert_eq!(low_score, 1.0);
    }

    #[tokio::test]
    async fn test_gpu_count_weighting() {
        let (repo, persistence) = create_test_gpu_profile_repo().await.unwrap();
        let engine = GpuScoringEngine::new(repo, persistence, EmissionConfig::for_testing());

        // Test different GPU counts
        for gpu_count in 1..=8 {
            let validations = vec![NodeValidationResult {
                node_id: format!("exec_{gpu_count}"),
                is_valid: true,
                gpu_model: "A100".to_string(),
                gpu_count,
                gpu_memory_gb: 80.0,
                attestation_valid: true,
                validation_timestamp: Utc::now(),
            }];

            let score = engine.calculate_verification_score(&validations);
            let expected_score = 1.0;
            assert!(
                (score - expected_score).abs() < 0.001,
                "GPU count {gpu_count} should give score {expected_score}, got {score}"
            );
        }

        // Test with many GPUs (no cap, linear scaling)
        let many_gpu_validations = vec![NodeValidationResult {
            node_id: "exec_many".to_string(),
            is_valid: true,
            gpu_model: "A100".to_string(),
            gpu_count: 128,
            gpu_memory_gb: 80.0,
            attestation_valid: true,
            validation_timestamp: Utc::now(),
        }];

        let score = engine.calculate_verification_score(&many_gpu_validations);
        assert_eq!(score, 1.0);
    }

    #[tokio::test]
    async fn test_miner_profile_update() {
        let (repo, persistence) = create_test_gpu_profile_repo().await.unwrap();
        let engine = GpuScoringEngine::new(repo, persistence, EmissionConfig::for_testing());

        let miner_uid = MinerUid::new(1);
        let validations = vec![NodeValidationResult {
            node_id: "exec1".to_string(),
            is_valid: true,
            gpu_model: "A100".to_string(),
            gpu_count: 2,
            gpu_memory_gb: 80.0,
            attestation_valid: true,
            validation_timestamp: Utc::now(),
        }];

        // Test new profile creation
        let profile = engine
            .update_miner_profile_from_validation(miner_uid, validations)
            .await
            .unwrap();
        assert_eq!(profile.miner_uid, miner_uid);
        assert!(profile.total_score > 0.0);

        // Test existing profile update with different memory
        let new_validations = vec![NodeValidationResult {
            node_id: "exec2".to_string(),
            is_valid: true,
            gpu_model: "A100".to_string(),
            gpu_count: 1,
            gpu_memory_gb: 40.0, // Different memory than first validation (80GB)
            attestation_valid: true,
            validation_timestamp: Utc::now(),
        }];

        let updated_profile = engine
            .update_miner_profile_from_validation(miner_uid, new_validations)
            .await
            .unwrap();
        assert_eq!(updated_profile.miner_uid, miner_uid);
        assert_eq!(updated_profile.total_score, 1.0);
    }

    #[tokio::test]
    async fn test_category_statistics() {
        let (repo, persistence) = create_test_gpu_profile_repo().await.unwrap();
        let engine = GpuScoringEngine::new(
            repo.clone(),
            persistence.clone(),
            EmissionConfig::for_testing(),
        );

        // Create test profiles
        let mut a100_counts_1 = HashMap::new();
        a100_counts_1.insert("A100".to_string(), 2);

        let mut a100_counts_2 = HashMap::new();
        a100_counts_2.insert("A100".to_string(), 1);

        let mut h100_counts = HashMap::new();
        h100_counts.insert("H100".to_string(), 1);

        let now = Utc::now();
        let profiles = vec![
            create_test_profile(1, a100_counts_1, 0.8, now),
            create_test_profile(2, a100_counts_2, 0.6, now),
            create_test_profile(3, h100_counts, 0.9, now),
        ];

        // Seed all required data
        seed_test_data(&persistence, &repo, &profiles)
            .await
            .unwrap();

        let stats = engine.get_category_statistics().await.unwrap();

        assert_eq!(stats.len(), 2);

        let a100_stats = stats.get("A100").unwrap();
        assert_eq!(a100_stats.miner_count, 2);
        assert_eq!(a100_stats.average_score, 0.7);
        assert_eq!(a100_stats.total_score, 1.4);
        assert_eq!(a100_stats.min_score, 0.6);
        assert_eq!(a100_stats.max_score, 0.8);

        let h100_stats = stats.get("H100").unwrap();
        assert_eq!(h100_stats.miner_count, 1);
        assert_eq!(h100_stats.average_score, 0.9);
        assert_eq!(h100_stats.total_score, 0.9);
        assert_eq!(h100_stats.min_score, 0.9);
        assert_eq!(h100_stats.max_score, 0.9);
    }

    #[tokio::test]
    async fn test_pass_fail_scoring_edge_cases() {
        let (repo, persistence) = create_test_gpu_profile_repo().await.unwrap();
        let engine = GpuScoringEngine::new(repo, persistence, EmissionConfig::for_testing());

        // Test all invalid validations
        let all_invalid = vec![
            NodeValidationResult {
                node_id: "exec1".to_string(),
                is_valid: false,
                gpu_model: "A100".to_string(),
                gpu_count: 1,
                gpu_memory_gb: 80.0,
                attestation_valid: false,
                validation_timestamp: Utc::now(),
            },
            NodeValidationResult {
                node_id: "exec2".to_string(),
                is_valid: true,
                gpu_model: "A100".to_string(),
                gpu_count: 1,
                gpu_memory_gb: 80.0,
                attestation_valid: false, // Attestation invalid
                validation_timestamp: Utc::now(),
            },
        ];

        let score = engine.calculate_verification_score(&all_invalid);
        assert_eq!(score, 0.0); // All failed

        // Test partial success
        let partial_success = vec![
            NodeValidationResult {
                node_id: "exec1".to_string(),
                is_valid: true,
                gpu_model: "A100".to_string(),
                gpu_count: 1,
                gpu_memory_gb: 80.0,
                attestation_valid: true,
                validation_timestamp: Utc::now(),
            },
            NodeValidationResult {
                node_id: "exec2".to_string(),
                is_valid: false,
                gpu_model: "A100".to_string(),
                gpu_count: 1,
                gpu_memory_gb: 80.0,
                attestation_valid: false,
                validation_timestamp: Utc::now(),
            },
            NodeValidationResult {
                node_id: "exec3".to_string(),
                is_valid: true,
                gpu_model: "A100".to_string(),
                gpu_count: 1,
                gpu_memory_gb: 40.0,
                attestation_valid: true,
                validation_timestamp: Utc::now(),
            },
        ];

        let score = engine.calculate_verification_score(&partial_success);
        let expected = 2.0 / 3.0; // Stored score is validation ratio
        assert!((score - expected).abs() < 0.001);
    }

    #[tokio::test]
    async fn test_direct_score_update() {
        let (repo, persistence) = create_test_gpu_profile_repo().await.unwrap();
        let engine =
            GpuScoringEngine::new(repo.clone(), persistence, EmissionConfig::for_testing());

        let miner_uid = MinerUid::new(100);

        // Create initial profile with score 0.2
        let mut gpu_counts = HashMap::new();
        gpu_counts.insert("A100".to_string(), 1);
        let mut initial_profile = create_test_profile(100, gpu_counts, 0.2, Utc::now());
        initial_profile.last_successful_validation = None;
        repo.upsert_gpu_profile(&initial_profile).await.unwrap();

        // Update with new validations that would give score 1.0
        let validations = vec![NodeValidationResult {
            node_id: "exec1".to_string(),
            is_valid: true,
            gpu_model: "A100".to_string(),
            gpu_count: 1,
            gpu_memory_gb: 80.0,
            attestation_valid: true,
            validation_timestamp: Utc::now(),
        }];

        let profile = engine
            .update_miner_profile_from_validation(miner_uid, validations)
            .await
            .unwrap();

        assert_eq!(profile.total_score, 1.0);
    }

    #[tokio::test]
    async fn test_scoring_ignores_gpu_memory() {
        let (repo, persistence) = create_test_gpu_profile_repo().await.unwrap();
        let engine = GpuScoringEngine::new(repo, persistence, EmissionConfig::for_testing());

        // Test various memory sizes all get same score
        let memory_sizes = vec![16, 24, 40, 80, 100];

        for memory in memory_sizes {
            let validations = vec![NodeValidationResult {
                node_id: format!("exec_{memory}"),
                is_valid: true,
                gpu_model: "A100".to_string(),
                gpu_count: 1,
                gpu_memory_gb: memory as f64,
                attestation_valid: true,
                validation_timestamp: Utc::now(),
            }];

            let score = engine.calculate_verification_score(&validations);
            assert_eq!(score, 1.0, "Memory {memory} should give score 1.0");
        }
    }

    #[tokio::test]
    async fn test_b200_gpu_support() {
        let (repo, persistence) = create_test_gpu_profile_repo().await.unwrap();
        let engine = GpuScoringEngine::new(
            repo.clone(),
            persistence.clone(),
            EmissionConfig::for_testing(),
        );

        // Test that B200 is considered rewardable
        assert!(engine.is_gpu_model_rewardable("B200"));
        assert!(engine.is_gpu_model_rewardable("NVIDIA B200"));
        assert!(engine.is_gpu_model_rewardable("Tesla B200"));

        // Test that A100 and H100 are still rewardable
        assert!(engine.is_gpu_model_rewardable("A100"));
        assert!(engine.is_gpu_model_rewardable("H100"));

        // Test that unconfigured GPUs are not rewardable
        assert!(!engine.is_gpu_model_rewardable("V100"));

        // Create B200 profile
        let mut b200_counts = HashMap::new();
        b200_counts.insert("B200".to_string(), 4);

        let now = Utc::now();
        let b200_profile = create_test_profile(100, b200_counts, 1.0, now);

        // Seed B200 data
        seed_test_data(&persistence, &repo, &[b200_profile])
            .await
            .unwrap();

        // Test category statistics include B200
        let stats = engine.get_category_statistics().await.unwrap();
        assert!(
            stats.contains_key("B200"),
            "B200 should be included in category statistics"
        );

        let b200_stats = stats.get("B200").unwrap();
        assert_eq!(b200_stats.miner_count, 1);
        assert_eq!(b200_stats.total_score, 1.0);
    }

    #[tokio::test]
    async fn test_emission_config_filtering() {
        let (repo, persistence) = create_test_gpu_profile_repo().await.unwrap();

        // Create custom emission config with only A100 and B200 (exclude H100)
        let mut custom_gpu_allocations = HashMap::new();
        custom_gpu_allocations.insert(
            "A100".to_string(),
            crate::config::emission::GpuAllocation::new(20.0),
        );
        custom_gpu_allocations.insert(
            "B200".to_string(),
            crate::config::emission::GpuAllocation::new(80.0),
        );

        let custom_emission_config = EmissionConfig {
            burn_percentage: 10.0,
            burn_uid: 999,
            gpu_allocations: custom_gpu_allocations,
            min_miners_per_category: 1,
            weight_set_interval_blocks: 360,
            weight_version_key: 0,
        };

        let engine =
            GpuScoringEngine::new(repo.clone(), persistence.clone(), custom_emission_config);

        // Test filtering matches custom config
        assert!(engine.is_gpu_model_rewardable("A100"));
        assert!(engine.is_gpu_model_rewardable("B200"));
        assert!(!engine.is_gpu_model_rewardable("H100"));

        // Create profiles with all GPU types
        let mut a100_counts = HashMap::new();
        a100_counts.insert("A100".to_string(), 2);

        let mut h100_counts = HashMap::new();
        h100_counts.insert("H100".to_string(), 1);

        let mut b200_counts = HashMap::new();
        b200_counts.insert("B200".to_string(), 3);

        let now = Utc::now();
        let profiles = vec![
            create_test_profile(1, a100_counts, 0.8, now),
            create_test_profile(2, h100_counts, 0.9, now),
            create_test_profile(3, b200_counts, 1.0, now),
        ];

        // Seed all data
        seed_test_data(&persistence, &repo, &profiles)
            .await
            .unwrap();

        // Test category statistics only include configured GPUs
        let stats = engine.get_category_statistics().await.unwrap();

        // Should have A100 and B200 but NOT H100
        assert_eq!(stats.len(), 2, "Should only have 2 categories (A100, B200)");
        assert!(stats.contains_key("A100"), "Should include A100");
        assert!(stats.contains_key("B200"), "Should include B200");
        assert!(
            !stats.contains_key("H100"),
            "Should NOT include H100 (not in emission config)"
        );

        // Verify correct stats
        assert_eq!(stats.get("A100").unwrap().miner_count, 1);
        assert_eq!(stats.get("B200").unwrap().miner_count, 1);
    }

    #[tokio::test]
    async fn test_multi_gpu_category_with_b200() {
        let (repo, persistence) = create_test_gpu_profile_repo().await.unwrap();
        let engine = GpuScoringEngine::new(
            repo.clone(),
            persistence.clone(),
            EmissionConfig::for_testing(),
        );

        // Create a miner with multiple GPU types including B200
        let mut multi_gpu_counts = HashMap::new();
        multi_gpu_counts.insert("A100".to_string(), 1);
        multi_gpu_counts.insert("B200".to_string(), 2);

        let now = Utc::now();
        let multi_gpu_profile = create_test_profile(42, multi_gpu_counts, 0.9, now);

        // Seed data
        seed_test_data(&persistence, &repo, &[multi_gpu_profile])
            .await
            .unwrap();

        // Test category statistics account for both GPU types
        let stats = engine.get_category_statistics().await.unwrap();

        // Should have both A100 and B200 categories
        assert!(stats.contains_key("A100"));
        assert!(stats.contains_key("B200"));

        // Both should show the same miner (miner can be in multiple categories)
        assert_eq!(stats.get("A100").unwrap().miner_count, 1);
        assert_eq!(stats.get("B200").unwrap().miner_count, 1);
    }

    #[test]
    fn test_is_gpu_model_rewardable_normalization() {
        // Create test emission config
        let mut gpu_allocations = HashMap::new();
        gpu_allocations.insert(
            "A100".to_string(),
            crate::config::emission::GpuAllocation::new(20.0),
        );
        gpu_allocations.insert(
            "B200".to_string(),
            crate::config::emission::GpuAllocation::new(80.0),
        );
        let emission_config = EmissionConfig {
            burn_percentage: 10.0,
            burn_uid: 999,
            gpu_allocations,
            min_miners_per_category: 1,
            weight_set_interval_blocks: 360,
            weight_version_key: 0,
        };

        // This test doesn't need async functionality, just the is_gpu_model_rewardable method

        // Test that various GPU model strings are normalized correctly
        let test_cases = vec![
            ("A100", true),
            ("NVIDIA A100", true),
            ("Tesla A100", true),
            ("a100", true),
            ("B200", true),
            ("NVIDIA B200", true),
            ("b200", true),
            ("H100", false), // Not in our custom config
            ("V100", false),
            ("A100", true),
            ("GTX1080", false),
        ];

        // Test the underlying logic through GpuCategory::from_str
        for (model, should_be_rewardable) in test_cases {
            let category = GpuCategory::from_str(model).unwrap();
            let normalized = category.to_string();
            let is_rewardable = emission_config.gpu_allocations.contains_key(&normalized);
            assert_eq!(
                is_rewardable, should_be_rewardable,
                "GPU model '{}' normalized to '{}', expected rewardable: {}, got: {}",
                model, normalized, should_be_rewardable, is_rewardable
            );
        }
    }

    #[tokio::test]
    async fn test_min_gpu_count_filtering() {
        let (repo, persistence) = create_test_gpu_profile_repo().await.unwrap();

        // Create custom emission config with min_gpu_count requirements
        let mut gpu_allocations = HashMap::new();
        gpu_allocations.insert(
            "A100".to_string(),
            crate::config::emission::GpuAllocation::with_min_count(25.0, 4),
        );
        gpu_allocations.insert(
            "H100".to_string(),
            crate::config::emission::GpuAllocation::with_min_count(25.0, 2),
        );
        gpu_allocations.insert(
            "B200".to_string(),
            crate::config::emission::GpuAllocation::with_min_count(50.0, 8),
        );

        let emission_config = EmissionConfig {
            burn_percentage: 10.0,
            burn_uid: 999,
            gpu_allocations,
            min_miners_per_category: 1,
            weight_set_interval_blocks: 360,
            weight_version_key: 0,
        };

        let engine = GpuScoringEngine::new(repo.clone(), persistence.clone(), emission_config);

        // Create profiles with different GPU counts
        let now = Utc::now();

        // Helper to create single GPU type profile
        let create_single_gpu_profile = |uid: u16, gpu_model: &str, count: u32, score: f64| {
            let mut gpu_counts = HashMap::new();
            gpu_counts.insert(gpu_model.to_string(), count);
            create_test_profile(uid, gpu_counts, score, now)
        };

        let profiles = vec![
            // Miner 1: Has 3x A100 (below min of 4) - should be excluded
            create_single_gpu_profile(1, "A100", 3, 0.9),
            // Miner 2: Has 4x A100 (meets min of 4) - should be included
            create_single_gpu_profile(2, "A100", 4, 0.8),
            // Miner 3: Has 1x H100 (below min of 2) - should be excluded
            create_single_gpu_profile(3, "H100", 1, 0.7),
            // Miner 4: Has 2x H100 (meets min of 2) - should be included
            create_single_gpu_profile(4, "H100", 2, 0.8),
            // Miner 5: Has 7x B200 (below min of 8) - should be excluded
            create_single_gpu_profile(5, "B200", 7, 1.0),
            // Miner 6: Has 8x B200 (meets min of 8) - should be included
            create_single_gpu_profile(6, "B200", 8, 1.0),
        ];

        // Seed all required data
        seed_test_data(&persistence, &repo, &profiles)
            .await
            .unwrap();

        // Test category statistics respect min_gpu_count
        let stats = engine.get_category_statistics().await.unwrap();

        // Check A100 category - should only have miner 2
        assert_eq!(
            stats.get("A100").unwrap().miner_count,
            1,
            "A100 should have 1 miner (miner 2)"
        );
        assert_eq!(stats.get("A100").unwrap().total_score, 0.8);

        // Check H100 category - should only have miner 4
        assert_eq!(
            stats.get("H100").unwrap().miner_count,
            1,
            "H100 should have 1 miner (miner 4)"
        );
        assert_eq!(stats.get("H100").unwrap().total_score, 0.8);

        // Check B200 category - should only have miner 6
        assert_eq!(
            stats.get("B200").unwrap().miner_count,
            1,
            "B200 should have 1 miner (miner 6)"
        );
        assert_eq!(stats.get("B200").unwrap().total_score, 1.0);

        // Test get_miners_by_gpu_category_since_epoch is skipped
        // The metagraph type requires complex initialization that comes from the chain
        // The important min_gpu_count filtering logic is already tested in get_category_statistics above
    }

    #[tokio::test]
    async fn test_uptime_ramp_up_calculation() {
        // Test cases for different uptime durations
        let test_cases = vec![
            (0.0, 0.0),       // 0 minutes = 0%
            (1440.0, 0.0714), // 1 day = 7.14%
            (4320.0, 0.2143), // 3 days = 21.43%
            (10080.0, 0.5),   // 7 days = 50%
            (20160.0, 1.0),   // 14 days = 100%
            (43200.0, 1.0),   // 30 days = 100% (capped)
        ];

        for (uptime_minutes, expected_multiplier) in test_cases {
            const FULL_WEIGHT_MINUTES: f64 = 20_160.0;
            let multiplier = uptime_minutes / FULL_WEIGHT_MINUTES;
            let actual = multiplier.min(1.0);
            assert!(
                (actual - expected_multiplier).abs() < 0.0001,
                "For {uptime_minutes} minutes, expected {expected_multiplier}, got {actual}"
            );
        }
    }

    #[tokio::test]
    async fn test_new_node_with_no_verifications() {
        let (_, persistence) = create_test_gpu_profile_repo().await.unwrap();

        let miner_id = "miner_999";
        let node_id = "test_node_new";
        let now = Utc::now();

        // Create miner and node without any verification logs
        insert_test_miner(&persistence, miner_id, "hotkey_999", now)
            .await
            .unwrap();
        insert_test_miner_node(&persistence, miner_id, node_id, 1, now)
            .await
            .unwrap();

        // Should get (0.0, 0.0) - no uptime and no multiplier (no GPU UUID assigned)
        let (uptime_minutes, multiplier) = persistence
            .calculate_node_uptime_multiplier(miner_id, node_id)
            .await
            .unwrap();

        assert_eq!(
            uptime_minutes, 0.0,
            "New node without GPU UUID should get 0.0 uptime minutes"
        );
        assert_eq!(
            multiplier, 0.0,
            "New node without GPU UUID should get 0.0 multiplier"
        );
    }

    #[tokio::test]
    async fn test_node_with_continuous_success() {
        let (_, persistence) = create_test_gpu_profile_repo().await.unwrap();

        let miner_id = "miner_1000";
        let node_id = "test_node_success";
        let now = Utc::now();
        let seven_days_ago = now - chrono::Duration::days(7);

        // Create miner and node
        insert_test_miner(&persistence, miner_id, "hotkey_1000", seven_days_ago)
            .await
            .unwrap();
        insert_test_miner_node(&persistence, miner_id, node_id, 1, seven_days_ago)
            .await
            .unwrap();

        // Add GPU UUID
        insert_test_gpu_uuid(&persistence, miner_id, node_id, "A100")
            .await
            .unwrap();

        // Add verification logs showing 7 days of continuous success
        insert_test_verification_log(&persistence, node_id, seven_days_ago, true, true)
            .await
            .unwrap();

        // Should get ~0.5 multiplier (7 days out of 14)
        let (_uptime_minutes, multiplier) = persistence
            .calculate_node_uptime_multiplier(miner_id, node_id)
            .await
            .unwrap();

        assert!(
            (multiplier - 0.5).abs() < 0.01,
            "Node with 7 days uptime should get ~0.5 multiplier, got {multiplier}"
        );
    }

    #[tokio::test]
    async fn test_node_with_failure_resets_uptime() {
        let (_, persistence) = create_test_gpu_profile_repo().await.unwrap();

        let miner_id = "miner_1001";
        let node_id = "test_node_failure";
        let now = Utc::now();
        let seven_days_ago = now - chrono::Duration::days(7);
        let two_days_ago = now - chrono::Duration::days(2);
        let one_day_ago = now - chrono::Duration::days(1);

        // Create miner and node
        insert_test_miner(&persistence, miner_id, "hotkey_1001", seven_days_ago)
            .await
            .unwrap();
        insert_test_miner_node(&persistence, miner_id, node_id, 1, seven_days_ago)
            .await
            .unwrap();

        // Add GPU UUID
        insert_test_gpu_uuid(&persistence, miner_id, node_id, "A100")
            .await
            .unwrap();

        // Add verification logs: success 7 days ago, failure 2 days ago, success 1 day ago
        // Only the last success period (1 day) should count
        insert_test_verification_log(&persistence, node_id, seven_days_ago, true, true)
            .await
            .unwrap();

        // Failure 2 days ago - this resets uptime
        insert_test_verification_log(&persistence, node_id, two_days_ago, false, true)
            .await
            .unwrap();

        // Success 1 day ago - starts new uptime period
        insert_test_verification_log(&persistence, node_id, one_day_ago, true, true)
            .await
            .unwrap();

        // Should get ~0.071 multiplier (1 day out of 14, ~7.14%)
        let (_uptime_minutes, multiplier) = persistence
            .calculate_node_uptime_multiplier(miner_id, node_id)
            .await
            .unwrap();

        assert!(
            multiplier < 0.1,
            "Node with failure should only count uptime from last success, got {multiplier}"
        );
    }
}
