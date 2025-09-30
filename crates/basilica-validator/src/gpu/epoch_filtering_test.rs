#[cfg(test)]
mod tests {
    use crate::config::emission::EmissionConfig;
    use crate::gpu::{GpuScoringEngine, MinerGpuProfile};
    use crate::persistence::{gpu_profile_repository::GpuProfileRepository, SimplePersistence};
    use basilica_common::identity::MinerUid;
    use chrono::Utc;

    use std::collections::HashMap;
    use std::sync::Arc;

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

            // Seed miner_executors table with online status
            let executor_key = format!("{}:{}", &miner_id, &executor_id);
            sqlx::query(
                "INSERT OR REPLACE INTO miner_executors (id, miner_id, executor_id, grpc_address, gpu_count, status, gpu_uuids, created_at, updated_at)
                 VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?)"
            )
            .bind(&executor_key)
            .bind(&miner_id)
            .bind(&executor_id)
            .bind("http://127.0.0.1:50051")
            .bind(profile.gpu_counts.values().sum::<u32>() as i64)
            .bind("online")
            .bind("") // Empty gpu_uuids, we'll use gpu_uuid_assignments instead
            .bind(now.to_rfc3339())
            .bind(now.to_rfc3339())
            .execute(persistence.pool())
            .await?;

            // Seed gpu_uuid_assignments table
            for (gpu_model, count) in &profile.gpu_counts {
                for i in 0..*count {
                    let gpu_uuid =
                        format!("gpu-{}-{}-{}", profile.miner_uid.as_u16(), gpu_model, i);
                    sqlx::query(
                        "INSERT INTO gpu_uuid_assignments (gpu_uuid, gpu_index, executor_id, miner_id, gpu_name, last_verified)
                         VALUES (?, ?, ?, ?, ?, ?)"
                    )
                    .bind(&gpu_uuid)
                    .bind(i as i32)
                    .bind(&executor_id)
                    .bind(&miner_id)
                    .bind(gpu_model)
                    .bind(now.to_rfc3339())
                    .execute(persistence.pool())
                    .await?;
                }
            }

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

    #[tokio::test]
    async fn test_epoch_based_filtering() -> anyhow::Result<()> {
        // Create test database
        let db_path = format!("/tmp/test_epoch_filtering_{}.db", uuid::Uuid::new_v4());
        let persistence =
            Arc::new(SimplePersistence::new(&db_path, "test_validator".to_string()).await?);
        let gpu_repo = Arc::new(GpuProfileRepository::new(persistence.pool().clone()));

        let now = Utc::now();
        let one_hour_ago = now - chrono::Duration::hours(1);
        let three_hours_ago = now - chrono::Duration::hours(3);
        let five_hours_ago = now - chrono::Duration::hours(5);

        // Create profiles with different validation times
        let profiles = vec![
            // Miner 1: Recent validation (should always be included)
            MinerGpuProfile {
                miner_uid: MinerUid::new(1),
                gpu_counts: HashMap::from([("A100".to_string(), 2)]),
                total_score: 0.9,
                verification_count: 10,
                last_updated: now,
                last_successful_validation: Some(one_hour_ago),
            },
            // Miner 2: Older validation (included only without epoch filtering)
            MinerGpuProfile {
                miner_uid: MinerUid::new(2),
                gpu_counts: HashMap::from([("A100".to_string(), 1)]),
                total_score: 0.8,
                verification_count: 8,
                last_updated: now,
                last_successful_validation: Some(three_hours_ago),
            },
            // Miner 3: Very old validation
            MinerGpuProfile {
                miner_uid: MinerUid::new(3),
                gpu_counts: HashMap::from([("A100".to_string(), 3)]),
                total_score: 0.7,
                verification_count: 5,
                last_updated: now,
                last_successful_validation: Some(five_hours_ago),
            },
            // Miner 4: No successful validation ever
            MinerGpuProfile {
                miner_uid: MinerUid::new(4),
                gpu_counts: HashMap::from([("A100".to_string(), 1)]),
                total_score: 0.5,
                verification_count: 2,
                last_updated: now,
                last_successful_validation: None,
            },
            // Miner 5: Different GPU type with recent validation
            MinerGpuProfile {
                miner_uid: MinerUid::new(5),
                gpu_counts: HashMap::from([("H100".to_string(), 2)]),
                total_score: 0.95,
                verification_count: 12,
                last_updated: now,
                last_successful_validation: Some(one_hour_ago),
            },
        ];

        // Store all profiles with complete data
        seed_test_data(&persistence, &gpu_repo, &profiles).await?;

        // Test 1: Get all profiles without epoch filtering
        let all_profiles = gpu_repo.get_all_gpu_profiles().await?;
        assert_eq!(all_profiles.len(), 5, "Should have all 5 profiles");

        // Test 2: Filter profiles with validation after 2 hours ago
        let two_hours_ago = now - chrono::Duration::hours(2);
        let recent_profiles: Vec<_> = all_profiles
            .iter()
            .filter(|p| {
                p.last_successful_validation
                    .map(|ts| ts >= two_hours_ago)
                    .unwrap_or(false)
            })
            .collect();

        assert_eq!(recent_profiles.len(), 2, "Should have 2 recent profiles");
        assert!(recent_profiles.iter().any(|p| p.miner_uid.as_u16() == 1));
        assert!(recent_profiles.iter().any(|p| p.miner_uid.as_u16() == 5));

        // Test 3: Filter profiles with validation after 4 hours ago
        let four_hours_ago = now - chrono::Duration::hours(4);
        let semi_recent_profiles: Vec<_> = all_profiles
            .iter()
            .filter(|p| {
                p.last_successful_validation
                    .map(|ts| ts >= four_hours_ago)
                    .unwrap_or(false)
            })
            .collect();

        assert_eq!(
            semi_recent_profiles.len(),
            3,
            "Should have 3 semi-recent profiles"
        );
        assert!(semi_recent_profiles
            .iter()
            .any(|p| p.miner_uid.as_u16() == 1));
        assert!(semi_recent_profiles
            .iter()
            .any(|p| p.miner_uid.as_u16() == 2));
        assert!(semi_recent_profiles
            .iter()
            .any(|p| p.miner_uid.as_u16() == 5));

        // Test 4: Verify miners without successful validation are always excluded
        let profiles_with_validation: Vec<_> = all_profiles
            .iter()
            .filter(|p| p.last_successful_validation.is_some())
            .collect();

        assert_eq!(profiles_with_validation.len(), 4);
        assert!(!profiles_with_validation
            .iter()
            .any(|p| p.miner_uid.as_u16() == 4));

        // Test 5: Update last successful validation for a miner by re-upserting
        let new_timestamp = now;
        let mut miner3_profile = gpu_repo.get_gpu_profile(MinerUid::new(3)).await?.unwrap();
        miner3_profile.last_successful_validation = Some(new_timestamp);
        gpu_repo.upsert_gpu_profile(&miner3_profile).await?;

        // Retrieve and verify update
        let updated_profile = gpu_repo.get_gpu_profile(MinerUid::new(3)).await?.unwrap();
        assert_eq!(
            updated_profile.last_successful_validation,
            Some(new_timestamp)
        );

        // Test 6: Verify GPU category distribution
        let a100_count = all_profiles
            .iter()
            .filter(|p| p.has_gpu_model("A100"))
            .count();
        let h100_count = all_profiles
            .iter()
            .filter(|p| p.has_gpu_model("H100"))
            .count();

        assert_eq!(a100_count, 4, "Should have 4 A100 miners");
        assert_eq!(h100_count, 1, "Should have 1 H100 miner");

        // Clean up
        std::fs::remove_file(&db_path).ok();

        Ok(())
    }

    #[tokio::test]
    async fn test_scoring_engine_epoch_filtering_logic() -> anyhow::Result<()> {
        // Create test database
        let db_path = format!("/tmp/test_scoring_engine_epoch_{}.db", uuid::Uuid::new_v4());
        let persistence =
            Arc::new(SimplePersistence::new(&db_path, "test_validator".to_string()).await?);
        let gpu_repo = Arc::new(GpuProfileRepository::new(persistence.pool().clone()));
        let scoring_engine = GpuScoringEngine::new(gpu_repo.clone(), EmissionConfig::for_testing());

        let now = Utc::now();
        let one_hour_ago = now - chrono::Duration::hours(1);
        let three_hours_ago = now - chrono::Duration::hours(3);

        // Create test profiles
        let profiles = vec![
            MinerGpuProfile {
                miner_uid: MinerUid::new(10),
                gpu_counts: HashMap::from([("A100".to_string(), 4)]),
                total_score: 0.9,
                verification_count: 20,
                last_updated: now,
                last_successful_validation: Some(one_hour_ago),
            },
            MinerGpuProfile {
                miner_uid: MinerUid::new(11),
                gpu_counts: HashMap::from([("A100".to_string(), 2)]),
                total_score: 0.8,
                verification_count: 15,
                last_updated: now,
                last_successful_validation: Some(three_hours_ago),
            },
            MinerGpuProfile {
                miner_uid: MinerUid::new(12),
                gpu_counts: HashMap::from([("H100".to_string(), 1)]),
                total_score: 0.85,
                verification_count: 18,
                last_updated: now,
                last_successful_validation: None, // Never validated
            },
        ];

        // Store profiles with complete data
        seed_test_data(&persistence, &gpu_repo, &profiles).await?;

        // Test category statistics
        let stats = scoring_engine.get_category_statistics().await?;

        assert_eq!(stats.len(), 2, "Should have A100 and H100 categories");
        assert!(stats.contains_key("A100"));
        assert!(stats.contains_key("H100"));

        let a100_stats = stats.get("A100").unwrap();
        assert_eq!(a100_stats.miner_count, 2);
        assert!(a100_stats.average_score > 0.0);

        let h100_stats = stats.get("H100").unwrap();
        assert_eq!(h100_stats.miner_count, 1);

        // Clean up
        std::fs::remove_file(&db_path).ok();

        Ok(())
    }

    #[tokio::test]
    async fn test_multi_gpu_profile_with_epoch() -> anyhow::Result<()> {
        // Create test database
        let db_path = format!("/tmp/test_multi_gpu_epoch_{}.db", uuid::Uuid::new_v4());
        let persistence =
            Arc::new(SimplePersistence::new(&db_path, "test_validator".to_string()).await?);
        let gpu_repo = Arc::new(GpuProfileRepository::new(persistence.pool().clone()));

        let now = Utc::now();
        let recent = now - chrono::Duration::minutes(30);

        // Create a miner with multiple GPU types
        let multi_gpu_profile = MinerGpuProfile {
            miner_uid: MinerUid::new(100),
            gpu_counts: HashMap::from([("A100".to_string(), 4), ("H100".to_string(), 2)]),
            total_score: 0.92,
            verification_count: 50,
            last_updated: now,
            last_successful_validation: Some(recent),
        };

        // Store profile with complete data
        seed_test_data(
            &persistence,
            &gpu_repo,
            std::slice::from_ref(&multi_gpu_profile),
        )
        .await?;

        // Retrieve and verify
        let retrieved = gpu_repo.get_gpu_profile(MinerUid::new(100)).await?.unwrap();

        assert_eq!(retrieved.gpu_counts.len(), 2, "Should have 2 GPU types");
        assert_eq!(retrieved.gpu_counts.get("A100"), Some(&4));
        assert_eq!(retrieved.gpu_counts.get("H100"), Some(&2));
        assert_eq!(retrieved.last_successful_validation, Some(recent));

        // Clean up
        std::fs::remove_file(&db_path).ok();

        Ok(())
    }
}
