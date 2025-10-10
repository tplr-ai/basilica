#[cfg(test)]
mod tests {
    use crate::gpu::MinerGpuProfile;
    use crate::persistence::gpu_profile_repository::GpuProfileRepository;
    use crate::persistence::SimplePersistence;
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
            .bind("verified") // Set status to 'verified' for tests
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

    #[tokio::test]
    async fn test_last_successful_validation_field() -> anyhow::Result<()> {
        let persistence = Arc::new(SimplePersistence::for_testing().await?);
        let gpu_repo = Arc::new(GpuProfileRepository::new(persistence.pool().clone()));

        let now = Utc::now();
        let one_hour_ago = now - chrono::Duration::hours(1);

        let profile = MinerGpuProfile {
            miner_uid: MinerUid::new(1),
            gpu_counts: HashMap::from([("A100".to_string(), 2)]),
            total_score: 0.8,
            verification_count: 5,
            last_updated: now,
            last_successful_validation: Some(one_hour_ago),
        };

        // Store and retrieve
        seed_test_data(&persistence, &gpu_repo, std::slice::from_ref(&profile)).await?;
        let retrieved = gpu_repo.get_gpu_profile(MinerUid::new(1)).await?;

        assert!(retrieved.is_some());
        let retrieved_profile = retrieved.unwrap();
        assert_eq!(
            retrieved_profile.last_successful_validation,
            Some(one_hour_ago)
        );

        // Test update by modifying and re-upserting the profile
        let mut updated_profile = retrieved_profile;
        updated_profile.last_successful_validation = Some(now);
        gpu_repo.upsert_gpu_profile(&updated_profile).await?;

        let updated = gpu_repo.get_gpu_profile(MinerUid::new(1)).await?;
        assert!(updated.is_some());
        let final_profile = updated.unwrap();
        assert_eq!(final_profile.last_successful_validation, Some(now));

        Ok(())
    }

    #[tokio::test]
    async fn test_profile_filtering_by_epoch() -> anyhow::Result<()> {
        let persistence = Arc::new(SimplePersistence::for_testing().await?);
        let gpu_repo = Arc::new(GpuProfileRepository::new(persistence.pool().clone()));

        let now = Utc::now();
        let one_hour_ago = now - chrono::Duration::hours(1);
        let three_hours_ago = now - chrono::Duration::hours(3);

        // Create profiles with different validation times
        let profiles = vec![
            MinerGpuProfile {
                miner_uid: MinerUid::new(1),
                gpu_counts: std::collections::HashMap::from([("A100".to_string(), 2)]),
                total_score: 0.8,
                verification_count: 5,
                last_updated: now,
                last_successful_validation: Some(one_hour_ago), // Recent
            },
            MinerGpuProfile {
                miner_uid: MinerUid::new(2),
                gpu_counts: std::collections::HashMap::from([("A100".to_string(), 1)]),
                total_score: 0.7,
                verification_count: 3,
                last_updated: now,
                last_successful_validation: Some(three_hours_ago), // Old
            },
            MinerGpuProfile {
                miner_uid: MinerUid::new(3),
                gpu_counts: std::collections::HashMap::from([("H100".to_string(), 1)]),
                total_score: 0.6,
                verification_count: 2,
                last_updated: now,
                last_successful_validation: None, // Never validated
            },
        ];

        // Store all profiles with complete data
        seed_test_data(&persistence, &gpu_repo, &profiles).await?;

        // Retrieve all profiles
        let all_profiles = gpu_repo.get_all_gpu_profiles().await?;
        assert_eq!(all_profiles.len(), 3);

        // Filter by epoch (only profiles with validation after 2 hours ago)
        let two_hours_ago = now - chrono::Duration::hours(2);
        let recent_profiles: Vec<_> = all_profiles
            .into_iter()
            .filter(|p| {
                p.last_successful_validation
                    .map(|ts| ts >= two_hours_ago)
                    .unwrap_or(false)
            })
            .collect();

        // Only miner 1 should pass the filter
        assert_eq!(recent_profiles.len(), 1);
        assert_eq!(recent_profiles[0].miner_uid.as_u16(), 1);

        Ok(())
    }
}
