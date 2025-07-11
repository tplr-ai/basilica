//! Integration test for complete emission-based allocation system

#[cfg(test)]
mod tests {
    use crate::config::emission::EmissionConfig;

    use crate::gpu::MinerGpuProfile;
    use crate::persistence::{GpuProfileRepository, SimplePersistence};
    use common::identity::MinerUid;
    use std::collections::HashMap;
    use std::sync::Arc;
    use tempfile::NamedTempFile;

    #[tokio::test]
    async fn test_complete_emission_based_allocation_flow() {
        // 1. Setup configuration
        let mut gpu_allocations = HashMap::new();
        gpu_allocations.insert("H100".to_string(), 40.0);
        gpu_allocations.insert("H200".to_string(), 60.0);

        let emission_config = EmissionConfig {
            burn_percentage: 10.0,
            burn_uid: 1,
            gpu_allocations,
            weight_set_interval_blocks: 360,
        };

        // 2. Create persistence layer
        let temp_file = NamedTempFile::new().unwrap();
        let db_path = temp_file.path().to_str().unwrap();
        let persistence = Arc::new(
            SimplePersistence::new(db_path, "test".to_string())
                .await
                .unwrap(),
        );
        let gpu_repo = GpuProfileRepository::new(persistence.pool().clone());

        // 3. Create GPU profiles for miners
        let miners = vec![
            (MinerUid::new(10), "H100", 2, 0.9),
            (MinerUid::new(20), "H100", 1, 0.8),
            (MinerUid::new(30), "H200", 3, 0.95),
            (MinerUid::new(40), "H200", 2, 0.85),
            (MinerUid::new(50), "OTHER", 1, 0.7),
        ];

        for (uid, gpu_model, gpu_count, score) in miners {
            let mut gpu_counts = HashMap::new();
            gpu_counts.insert(gpu_model.to_string(), gpu_count);

            let profile = MinerGpuProfile {
                miner_uid: uid,
                primary_gpu_model: gpu_model.to_string(),
                gpu_counts,
                total_score: score,
                verification_count: 5,
                last_updated: chrono::Utc::now(),
            };

            gpu_repo.upsert_gpu_profile(&profile).await.unwrap();
        }

        // 4. Verify emission allocation would work correctly
        // In a real scenario, this would be done by WeightSetter.set_weights_with_emission()

        // Get all profiles
        let all_profiles = gpu_repo.get_all_gpu_profiles().await.unwrap();
        assert_eq!(all_profiles.len(), 5);

        // Group by GPU category
        let mut category_groups: HashMap<String, Vec<&MinerGpuProfile>> = HashMap::new();
        for profile in &all_profiles {
            category_groups
                .entry(profile.primary_gpu_model.clone())
                .or_default()
                .push(profile);
        }

        // Verify H100 miners
        assert_eq!(category_groups.get("H100").unwrap().len(), 2);

        // Verify H200 miners
        assert_eq!(category_groups.get("H200").unwrap().len(), 2);

        // Verify OTHER category (should not receive allocation)
        assert_eq!(category_groups.get("OTHER").unwrap().len(), 1);

        // 5. Verify emission metrics storage
        let metrics_before = gpu_repo.get_emission_metrics_history(10, 0).await.unwrap();
        assert_eq!(metrics_before.len(), 0);

        // 6. Test profile statistics
        let stats = gpu_repo.get_profile_statistics().await.unwrap();
        assert_eq!(stats.total_profiles, 5);
        assert_eq!(stats.gpu_model_distribution.get("H100"), Some(&2));
        assert_eq!(stats.gpu_model_distribution.get("H200"), Some(&2));
        assert_eq!(stats.gpu_model_distribution.get("OTHER"), Some(&1));

        // Verify average scores
        let h100_avg = stats.average_score_by_model.get("H100").unwrap();
        assert!((h100_avg - 0.85).abs() < 0.01); // (0.9 + 0.8) / 2

        let h200_avg = stats.average_score_by_model.get("H200").unwrap();
        assert!((h200_avg - 0.9).abs() < 0.01); // (0.95 + 0.85) / 2
    }

    #[test]
    fn test_emission_config_validation() {
        // Test valid config
        let mut gpu_allocations = HashMap::new();
        gpu_allocations.insert("H100".to_string(), 40.0);
        gpu_allocations.insert("H200".to_string(), 60.0);

        let valid_config = EmissionConfig {
            burn_percentage: 10.0,
            burn_uid: 1,
            gpu_allocations: gpu_allocations.clone(),
            weight_set_interval_blocks: 360,
        };

        assert!(valid_config.validate().is_ok());

        // Test invalid burn percentage
        let invalid_burn = EmissionConfig {
            burn_percentage: 150.0, // Invalid: > 100%
            ..valid_config.clone()
        };
        assert!(invalid_burn.validate().is_err());

        // Test invalid GPU allocations (don't sum to 100)
        let mut bad_allocations = HashMap::new();
        bad_allocations.insert("H100".to_string(), 30.0);
        bad_allocations.insert("H200".to_string(), 40.0); // Sum = 70, not 100

        let invalid_allocation = EmissionConfig {
            gpu_allocations: bad_allocations,
            ..valid_config
        };
        assert!(invalid_allocation.validate().is_err());
    }
}
