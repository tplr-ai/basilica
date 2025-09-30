use anyhow::Result;
use chrono::{DateTime, Utc};
use std::collections::HashMap;
use std::sync::Arc;
use tracing::{debug, info, warn};

use super::categorization::{ExecutorValidationResult, GpuCategory, MinerGpuProfile};
use crate::config::emission::EmissionConfig;
use crate::metrics::ValidatorMetrics;
use crate::persistence::gpu_profile_repository::GpuProfileRepository;
use basilica_common::identity::MinerUid;
use std::str::FromStr;

pub struct GpuScoringEngine {
    gpu_profile_repo: Arc<GpuProfileRepository>,
    metrics: Option<Arc<ValidatorMetrics>>,
    emission_config: EmissionConfig,
}

impl GpuScoringEngine {
    pub fn new(
        gpu_profile_repo: Arc<GpuProfileRepository>,
        emission_config: EmissionConfig,
    ) -> Self {
        Self {
            gpu_profile_repo,
            metrics: None,
            emission_config,
        }
    }

    /// Create new engine with metrics support
    pub fn with_metrics(
        gpu_profile_repo: Arc<GpuProfileRepository>,
        metrics: Arc<ValidatorMetrics>,
        emission_config: EmissionConfig,
    ) -> Self {
        Self {
            gpu_profile_repo,
            metrics: Some(metrics),
            emission_config,
        }
    }

    /// Update miner profile from validation results
    pub async fn update_miner_profile_from_validation(
        &self,
        miner_uid: MinerUid,
        executor_validations: Vec<ExecutorValidationResult>,
    ) -> Result<MinerGpuProfile> {
        // Calculate verification score from executor results
        let new_score = self.calculate_verification_score(&executor_validations);

        // Check if there are any successful validations
        let has_successful_validation = executor_validations
            .iter()
            .any(|v| v.is_valid && v.attestation_valid);

        // Create or update the profile with the calculated score
        let mut profile = MinerGpuProfile::new(miner_uid, &executor_validations, new_score);

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
            validations = executor_validations.len(),
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

            // Record individual executor GPU counts
            for validation in &executor_validations {
                if validation.is_valid && validation.attestation_valid {
                    metrics.prometheus().record_executor_gpu_count(
                        miner_uid.as_u16(),
                        &validation.executor_id,
                        &validation.gpu_model,
                        validation.gpu_count,
                    );

                    // Record successful validation
                    metrics.prometheus().record_miner_successful_validation(
                        miner_uid.as_u16(),
                        &validation.executor_id,
                    );

                    // Record GPU profile
                    metrics.prometheus().record_miner_gpu_profile(
                        miner_uid.as_u16(),
                        &validation.gpu_model,
                        &validation.executor_id,
                        validation.gpu_count as u32,
                    );

                    // Also record through business metrics for complete tracking
                    metrics
                        .business()
                        .record_gpu_profile_validation(
                            miner_uid.as_u16(),
                            &validation.executor_id,
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

    /// Calculate verification score from executor results
    fn calculate_verification_score(
        &self,
        executor_validations: &[ExecutorValidationResult],
    ) -> f64 {
        if executor_validations.is_empty() {
            return 0.0;
        }

        let mut valid_count = 0;
        let mut total_count = 0;
        let mut total_gpu_count = 0;
        let mut unique_executors = std::collections::HashSet::new();

        // count unique executors and their GPU counts
        for validation in executor_validations {
            unique_executors.insert(&validation.executor_id);
            total_count += 1;

            // Count valid attestations and accumulate GPU counts
            if validation.is_valid && validation.attestation_valid {
                valid_count += 1;
            }
        }

        // sum GPU counts from unique executors only
        let mut seen_executors = std::collections::HashSet::new();
        for validation in executor_validations {
            if validation.is_valid
                && validation.attestation_valid
                && seen_executors.insert(&validation.executor_id)
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
                validations = executor_validations.len(),
                valid_count = valid_count,
                total_count = total_count,
                unique_executors = unique_executors.len(),
                total_gpu_count = total_gpu_count,
                final_score = final_score,
                gpu_weighted_score = gpu_weighted_score,
                "Calculated verification score (normalized for DB, GPU count tracked separately)"
            );
            final_score
        } else {
            warn!(
                validations = executor_validations.len(),
                "No validations found for score calculation"
            );
            0.0
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
        metagraph: &bittensor::Metagraph<bittensor::AccountId>,
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

            let rewardable_gpus: Vec<(GpuCategory, u32)> = self
                .gpu_profile_repo
                .get_miner_gpu_assignments(profile.miner_uid)
                .await?.iter().filter_map(|(executor_id, (gpu_count, gpu_name, gpu_memory_gb))| {
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
                                        executor_id = %executor_id,
                                        gpu_model = %gpu_name,
                                        gpu_count = *gpu_count,
                                        min_required = allocation.min_gpu_count,
                                        "Miner meets all emission requirements"
                                    );
                                    Some((category, *gpu_count))
                                } else {
                                    if !meets_gpu_count {
                                        info!(
                                            miner_uid = profile.miner_uid.as_u16(),
                                            executor_id = %executor_id,
                                            gpu_model = %gpu_name,
                                            gpu_count = *gpu_count,
                                            min_required = allocation.min_gpu_count,
                                            "Skipping miner: Does not meet minimum GPU count requirement"
                                        );
                                    }
                                    if !meets_vram {
                                        info!(
                                            miner_uid = profile.miner_uid.as_u16(),
                                            executor_id = %executor_id,
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

            let rewardable_gpu_counts: HashMap<String, u32> =
                rewardable_gpus
                    .into_iter()
                    .fold(HashMap::new(), |mut acc, (category, count)| {
                        let normalized_model = category.to_string();
                        *acc.entry(normalized_model).or_insert(0) += count;
                        acc
                    });

            // Skip miners with no rewardable GPUs
            if rewardable_gpu_counts.is_empty() {
                continue;
            }

            // Add the miner to each rewardable category they have GPUs in
            for (normalized_model, gpu_count) in rewardable_gpu_counts {
                // Multiply by gpu_count to get the actual linear score
                let category_score = profile.total_score * gpu_count as f64;

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
    use tempfile::NamedTempFile;

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
            let executor_id = format!(
                "miner{}__test-executor-{}",
                profile.miner_uid.as_u16(),
                profile.miner_uid.as_u16()
            );

            // Seed miners table first (required for foreign key constraint)
            sqlx::query(
                "INSERT OR REPLACE INTO miners (id, hotkey, endpoint, last_seen, registered_at, updated_at, executor_info)
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
                        "INSERT INTO gpu_uuid_assignments (gpu_uuid, gpu_index, executor_id, miner_id, gpu_name, gpu_memory_gb, last_verified)
                         VALUES (?, ?, ?, ?, ?, ?, ?)"
                    )
                    .bind(&gpu_uuid)
                    .bind(i as i32)
                    .bind(&executor_id)
                    .bind(&miner_id)
                    .bind(gpu_model)
                    .bind(80i64) // Default 80GB for test data
                    .bind(now.to_rfc3339())
                    .execute(persistence.pool())
                    .await?;
                }
            }

            // Seed miner_executors table
            sqlx::query(
                "INSERT INTO miner_executors (id, miner_id, executor_id, grpc_address, gpu_count, status, created_at, updated_at)
                 VALUES (?, ?, ?, ?, ?, ?, ?, ?)"
            )
            .bind(&executor_id)
            .bind(&miner_id)
            .bind(&executor_id)
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
                    "INSERT INTO verification_logs (id, executor_id, validator_hotkey, verification_type, timestamp, score, success, details, duration_ms, error_message, created_at, updated_at)
                     VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?)"
                )
                .bind(&log_id)
                .bind(&executor_id)
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

    async fn create_test_gpu_profile_repo() -> Result<(Arc<GpuProfileRepository>, NamedTempFile)> {
        let temp_file = NamedTempFile::new()?;
        let db_path = temp_file.path().to_str().unwrap();

        let persistence =
            crate::persistence::SimplePersistence::new(db_path, "test".to_string()).await?;
        let repo = Arc::new(GpuProfileRepository::new(persistence.pool().clone()));

        Ok((repo, temp_file))
    }

    #[tokio::test]
    async fn test_verification_score_calculation() {
        let (repo, _temp_file) = create_test_gpu_profile_repo().await.unwrap();
        let engine = GpuScoringEngine::new(repo, EmissionConfig::for_testing());

        // Test with valid attestations
        let validations = vec![
            ExecutorValidationResult {
                executor_id: "exec1".to_string(),
                is_valid: true,
                gpu_model: "A100".to_string(),
                gpu_count: 2,
                gpu_memory_gb: 80.0,
                attestation_valid: true,
                validation_timestamp: Utc::now(),
            },
            ExecutorValidationResult {
                executor_id: "exec2".to_string(),
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
        let invalid_validations = vec![ExecutorValidationResult {
            executor_id: "exec1".to_string(),
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
            ExecutorValidationResult {
                executor_id: "exec1".to_string(),
                is_valid: true,
                gpu_model: "A100".to_string(),
                gpu_count: 2,
                gpu_memory_gb: 80.0,
                attestation_valid: true,
                validation_timestamp: Utc::now(),
            },
            ExecutorValidationResult {
                executor_id: "exec2".to_string(),
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
        let high_memory_validations = vec![ExecutorValidationResult {
            executor_id: "exec1".to_string(),
            is_valid: true,
            gpu_model: "A100".to_string(),
            gpu_count: 1,
            gpu_memory_gb: 80.0,
            attestation_valid: true,
            validation_timestamp: Utc::now(),
        }];

        let low_memory_validations = vec![ExecutorValidationResult {
            executor_id: "exec1".to_string(),
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
        let (repo, _temp_file) = create_test_gpu_profile_repo().await.unwrap();
        let engine = GpuScoringEngine::new(repo, EmissionConfig::for_testing());

        // Test different GPU counts
        for gpu_count in 1..=8 {
            let validations = vec![ExecutorValidationResult {
                executor_id: format!("exec_{gpu_count}"),
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
        let many_gpu_validations = vec![ExecutorValidationResult {
            executor_id: "exec_many".to_string(),
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
        let (repo, _temp_file) = create_test_gpu_profile_repo().await.unwrap();
        let engine = GpuScoringEngine::new(repo, EmissionConfig::for_testing());

        let miner_uid = MinerUid::new(1);
        let validations = vec![ExecutorValidationResult {
            executor_id: "exec1".to_string(),
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
        let new_validations = vec![ExecutorValidationResult {
            executor_id: "exec2".to_string(),
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
        let (repo, _temp_file) = create_test_gpu_profile_repo().await.unwrap();
        let engine = GpuScoringEngine::new(repo.clone(), EmissionConfig::for_testing());

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
        let persistence = crate::persistence::SimplePersistence::with_pool(repo.pool().clone());
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
        let (repo, _temp_file) = create_test_gpu_profile_repo().await.unwrap();
        let engine = GpuScoringEngine::new(repo, EmissionConfig::for_testing());

        // Test all invalid validations
        let all_invalid = vec![
            ExecutorValidationResult {
                executor_id: "exec1".to_string(),
                is_valid: false,
                gpu_model: "A100".to_string(),
                gpu_count: 1,
                gpu_memory_gb: 80.0,
                attestation_valid: false,
                validation_timestamp: Utc::now(),
            },
            ExecutorValidationResult {
                executor_id: "exec2".to_string(),
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
            ExecutorValidationResult {
                executor_id: "exec1".to_string(),
                is_valid: true,
                gpu_model: "A100".to_string(),
                gpu_count: 1,
                gpu_memory_gb: 80.0,
                attestation_valid: true,
                validation_timestamp: Utc::now(),
            },
            ExecutorValidationResult {
                executor_id: "exec2".to_string(),
                is_valid: false,
                gpu_model: "A100".to_string(),
                gpu_count: 1,
                gpu_memory_gb: 80.0,
                attestation_valid: false,
                validation_timestamp: Utc::now(),
            },
            ExecutorValidationResult {
                executor_id: "exec3".to_string(),
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
        let (repo, _temp_file) = create_test_gpu_profile_repo().await.unwrap();
        let engine = GpuScoringEngine::new(repo.clone(), EmissionConfig::for_testing());

        let miner_uid = MinerUid::new(100);

        // Create initial profile with score 0.2
        let mut gpu_counts = HashMap::new();
        gpu_counts.insert("A100".to_string(), 1);
        let mut initial_profile = create_test_profile(100, gpu_counts, 0.2, Utc::now());
        initial_profile.last_successful_validation = None;
        repo.upsert_gpu_profile(&initial_profile).await.unwrap();

        // Update with new validations that would give score 1.0
        let validations = vec![ExecutorValidationResult {
            executor_id: "exec1".to_string(),
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
        let (repo, _temp_file) = create_test_gpu_profile_repo().await.unwrap();
        let engine = GpuScoringEngine::new(repo, EmissionConfig::for_testing());

        // Test various memory sizes all get same score
        let memory_sizes = vec![16, 24, 40, 80, 100];

        for memory in memory_sizes {
            let validations = vec![ExecutorValidationResult {
                executor_id: format!("exec_{memory}"),
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
        let (repo, _temp_file) = create_test_gpu_profile_repo().await.unwrap();
        let engine = GpuScoringEngine::new(repo.clone(), EmissionConfig::for_testing());

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
        let persistence = crate::persistence::SimplePersistence::with_pool(repo.pool().clone());
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
        let (repo, _temp_file) = create_test_gpu_profile_repo().await.unwrap();

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

        let engine = GpuScoringEngine::new(repo.clone(), custom_emission_config);

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
        let persistence = crate::persistence::SimplePersistence::with_pool(repo.pool().clone());
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
        let (repo, _temp_file) = create_test_gpu_profile_repo().await.unwrap();
        let engine = GpuScoringEngine::new(repo.clone(), EmissionConfig::for_testing());

        // Create a miner with multiple GPU types including B200
        let mut multi_gpu_counts = HashMap::new();
        multi_gpu_counts.insert("A100".to_string(), 1);
        multi_gpu_counts.insert("B200".to_string(), 2);

        let now = Utc::now();
        let multi_gpu_profile = create_test_profile(42, multi_gpu_counts, 0.9, now);

        // Seed data
        let persistence = crate::persistence::SimplePersistence::with_pool(repo.pool().clone());
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

        // Create a minimal repo for testing (we only need the method, not async functionality)
        let _temp_file = tempfile::NamedTempFile::new().unwrap();

        // This test doesn't need async functionality, just the is_gpu_model_rewardable method
        // So we'll test the underlying logic directly

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
        use crate::gpu::categorization::GpuCategory;
        use std::str::FromStr;

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
        let (repo, _temp_file) = create_test_gpu_profile_repo().await.unwrap();

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

        let engine = GpuScoringEngine::new(repo.clone(), emission_config);

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
        let persistence = crate::persistence::SimplePersistence::with_pool(repo.pool().clone());
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
}
