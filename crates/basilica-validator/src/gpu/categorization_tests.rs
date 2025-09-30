#[cfg(test)]
mod tests {
    use crate::gpu::categorization::*;
    use basilica_common::identity::MinerUid;
    use chrono::Utc;

    #[test]
    fn test_gpu_model_normalization() {
        use std::str::FromStr;

        // Test A100 variants
        assert_eq!(
            GpuCategory::from_str("NVIDIA A100 PCIe")
                .unwrap()
                .to_string(),
            "A100"
        );
        assert_eq!(
            GpuCategory::from_str("A100 SXM5").unwrap().to_string(),
            "A100"
        );
        assert_eq!(GpuCategory::from_str("a100").unwrap().to_string(), "A100");
        assert_eq!(
            GpuCategory::from_str("NVIDIA A100-80GB")
                .unwrap()
                .to_string(),
            "A100"
        );

        // Test H100 variants
        assert_eq!(
            GpuCategory::from_str("NVIDIA H100").unwrap().to_string(),
            "H100"
        );
        assert_eq!(
            GpuCategory::from_str("H100 SXM").unwrap().to_string(),
            "H100"
        );
        assert_eq!(GpuCategory::from_str("h100").unwrap().to_string(), "H100");
        assert_eq!(
            GpuCategory::from_str("Tesla H100").unwrap().to_string(),
            "H100"
        );

        // Test B200 variants
        assert_eq!(
            GpuCategory::from_str("HGX B200").unwrap().to_string(),
            "B200"
        );
        assert_eq!(
            GpuCategory::from_str("B200 HGX ").unwrap().to_string(),
            "B200"
        );
        assert_eq!(
            GpuCategory::from_str("DGX B200").unwrap().to_string(),
            "B200"
        );
        assert_eq!(
            GpuCategory::from_str("B200 DGX").unwrap().to_string(),
            "B200"
        );
        assert_eq!(
            GpuCategory::from_str("NVIDIA B200").unwrap().to_string(),
            "B200"
        );
        assert_eq!(GpuCategory::from_str("b200").unwrap().to_string(), "B200");

        // Test other GPU variants (should all return OTHER)
        assert_eq!(
            GpuCategory::from_str("GeForce RTX 4090")
                .unwrap()
                .to_string(),
            "OTHER"
        );
        assert_eq!(
            GpuCategory::from_str("RTX 4090").unwrap().to_string(),
            "OTHER"
        );
        assert_eq!(
            GpuCategory::from_str("NVIDIA GeForce RTX 4090")
                .unwrap()
                .to_string(),
            "OTHER"
        );
        assert_eq!(
            GpuCategory::from_str("rtx4090").unwrap().to_string(),
            "OTHER"
        );
        assert_eq!(
            GpuCategory::from_str("RTX 3090 Ti").unwrap().to_string(),
            "OTHER"
        );
        assert_eq!(
            GpuCategory::from_str("GeForce RTX 3090")
                .unwrap()
                .to_string(),
            "OTHER"
        );
        assert_eq!(
            GpuCategory::from_str("RTX 3080").unwrap().to_string(),
            "OTHER"
        );
        assert_eq!(
            GpuCategory::from_str("RTX 4080").unwrap().to_string(),
            "OTHER"
        );

        // Test unknown models
        assert_eq!(
            GpuCategory::from_str("Unknown GPU").unwrap().to_string(),
            "OTHER"
        );
        assert_eq!(GpuCategory::from_str("").unwrap().to_string(), "OTHER");
        assert_eq!(GpuCategory::from_str("V100").unwrap().to_string(), "OTHER");
        assert_eq!(
            GpuCategory::from_str("GTX 1080").unwrap().to_string(),
            "OTHER"
        );

        // Test edge cases
        assert_eq!(
            GpuCategory::from_str("   A100   ").unwrap().to_string(),
            "A100"
        );
        assert_eq!(
            GpuCategory::from_str("NVIDIA NVIDIA A100")
                .unwrap()
                .to_string(),
            "A100"
        );
    }

    #[test]
    fn test_model_to_category_conversion() {
        // Test all known categories
        assert_eq!(GpuCategorizer::model_to_category("A100"), GpuCategory::A100);
        assert_eq!(GpuCategorizer::model_to_category("H100"), GpuCategory::H100);
        assert_eq!(GpuCategorizer::model_to_category("B200"), GpuCategory::B200);
        // These should return Other now
        match GpuCategorizer::model_to_category("RTX4090") {
            GpuCategory::Other(model) => assert_eq!(model, "RTX4090"),
            _ => panic!("Expected Other category"),
        }

        // Test case sensitivity
        assert_eq!(GpuCategorizer::model_to_category("a100"), GpuCategory::A100);
        assert_eq!(GpuCategorizer::model_to_category("h100"), GpuCategory::H100);
        assert_eq!(GpuCategorizer::model_to_category("b200"), GpuCategory::B200);

        // Test unknown models
        match GpuCategorizer::model_to_category("V100") {
            GpuCategory::Other(model) => assert_eq!(model, "V100"),
            _ => panic!("Expected Other category"),
        }

        match GpuCategorizer::model_to_category("GTX1080") {
            GpuCategory::Other(model) => assert_eq!(model, "GTX1080"),
            _ => panic!("Expected Other category"),
        }
    }

    #[test]
    fn test_gpu_distribution_calculation() {
        // Test single GPU type
        let validations = vec![
            ExecutorValidationResult::new_for_testing(
                "exec1".to_string(),
                "NVIDIA A100".to_string(),
                2,
                true,
                true,
            ),
            ExecutorValidationResult::new_for_testing(
                "exec2".to_string(),
                "A100 SXM".to_string(),
                1,
                true,
                true,
            ),
        ];

        let gpu_counts = GpuCategorizer::calculate_gpu_distribution(&validations);
        assert_eq!(gpu_counts.get("A100"), Some(&3));

        // Test multiple GPU types
        let validations = vec![
            ExecutorValidationResult::new_for_testing(
                "exec1".to_string(),
                "NVIDIA A100".to_string(),
                1,
                true,
                true,
            ),
            ExecutorValidationResult::new_for_testing(
                "exec2".to_string(),
                "H100".to_string(),
                2,
                true,
                true,
            ),
            ExecutorValidationResult::new_for_testing(
                "exec3".to_string(),
                "NVIDIA H100".to_string(),
                1,
                true,
                true,
            ),
        ];

        let gpu_counts = GpuCategorizer::calculate_gpu_distribution(&validations);
        assert_eq!(gpu_counts.get("A100"), Some(&1));
        assert_eq!(gpu_counts.get("H100"), Some(&3));

        // Test tie scenarios
        let validations = vec![
            ExecutorValidationResult::new_for_testing(
                "exec1".to_string(),
                "A100".to_string(),
                2,
                true,
                true,
            ),
            ExecutorValidationResult::new_for_testing(
                "exec2".to_string(),
                "H100".to_string(),
                2,
                true,
                true,
            ),
        ];

        let gpu_counts = GpuCategorizer::calculate_gpu_distribution(&validations);
        assert_eq!(gpu_counts.get("A100"), Some(&2));
        assert_eq!(gpu_counts.get("H100"), Some(&2));

        // Test empty validation results
        let validations = vec![];
        let gpu_counts = GpuCategorizer::calculate_gpu_distribution(&validations);
        assert!(gpu_counts.is_empty());

        // Test all invalid validations
        let validations = vec![
            ExecutorValidationResult::new_for_testing(
                "exec1".to_string(),
                "A100".to_string(),
                1,
                false,
                false,
            ),
            ExecutorValidationResult::new_for_testing(
                "exec2".to_string(),
                "A100".to_string(),
                1,
                true,
                false,
            ),
        ];

        let gpu_counts = GpuCategorizer::calculate_gpu_distribution(&validations);
        assert!(gpu_counts.is_empty());
    }

    #[test]
    fn test_miner_gpu_profile_creation() {
        let miner_uid = MinerUid::new(123);
        let validations = vec![
            ExecutorValidationResult::new_for_testing(
                "exec1".to_string(),
                "A100".to_string(),
                2,
                true,
                true,
            ),
            ExecutorValidationResult::new_for_testing(
                "exec2".to_string(),
                "H100".to_string(),
                1,
                true,
                true,
            ),
        ];

        let profile = MinerGpuProfile::new(miner_uid, &validations, 0.85);

        assert_eq!(profile.miner_uid, miner_uid);
        assert_eq!(profile.total_score, 0.85);
        assert_eq!(profile.verification_count, 2);
        assert_eq!(profile.total_gpu_count(), 3);
        assert_eq!(profile.get_gpu_count("A100"), 2);
        assert_eq!(profile.get_gpu_count("H100"), 1);
        assert!(profile.has_gpu_model("A100"));
        assert!(profile.has_gpu_model("H100"));
        assert!(!profile.has_gpu_model("B200"));

        // Test profile updates
        let mut profile = profile;
        let new_validations = vec![ExecutorValidationResult::new_for_testing(
            "exec3".to_string(),
            "H100".to_string(),
            4,
            true,
            true,
        )];

        profile.update_with_validations(&new_validations, 0.92);

        assert_eq!(profile.total_score, 0.92);
        assert_eq!(profile.verification_count, 1);
        assert_eq!(profile.total_gpu_count(), 4);
        assert_eq!(profile.get_gpu_count("H100"), 4);
        assert_eq!(profile.get_gpu_count("A100"), 0); // Replaced

        // Test timestamp handling
        let old_timestamp = profile.last_updated;
        std::thread::sleep(std::time::Duration::from_millis(10));
        profile.update_with_validations(&new_validations, 0.95);
        assert!(profile.last_updated > old_timestamp);
    }

    #[test]
    fn test_gpu_models_by_count() {
        let miner_uid = MinerUid::new(456);
        let validations = vec![
            ExecutorValidationResult::new_for_testing(
                "exec1".to_string(),
                "A100".to_string(),
                1,
                true,
                true,
            ),
            ExecutorValidationResult::new_for_testing(
                "exec2".to_string(),
                "A100".to_string(),
                4,
                true,
                true,
            ),
            ExecutorValidationResult::new_for_testing(
                "exec3".to_string(),
                "RTX4090".to_string(),
                2,
                true,
                true,
            ),
        ];

        let profile = MinerGpuProfile::new(miner_uid, &validations, 0.8);
        let models_by_count = profile.gpu_models_by_count();

        // Should be sorted by count descending
        assert_eq!(models_by_count.len(), 2);
        assert_eq!(models_by_count[0], ("A100".to_string(), 5)); // A100(1) + A100(4) = 5
        assert_eq!(models_by_count[1], ("OTHER".to_string(), 2)); // RTX4090(2) = OTHER(2)
    }

    #[test]
    fn test_edge_cases() {
        // Test unicode GPU names
        let validations = vec![ExecutorValidationResult::new_for_testing(
            "exec1".to_string(),
            "NVIDIA A100-新".to_string(),
            1,
            true,
            true,
        )];
        let gpu_counts = GpuCategorizer::calculate_gpu_distribution(&validations);
        assert_eq!(gpu_counts.get("A100"), Some(&1));

        // Test very long GPU names
        let long_name = "A".repeat(1000) + " A100";
        let validations = vec![ExecutorValidationResult::new_for_testing(
            "exec1".to_string(),
            long_name,
            1,
            true,
            true,
        )];
        let gpu_counts = GpuCategorizer::calculate_gpu_distribution(&validations);
        assert_eq!(gpu_counts.get("A100"), Some(&1));

        // Test special characters
        let validations = vec![ExecutorValidationResult::new_for_testing(
            "exec1".to_string(),
            "NVIDIA@@A100##PCIe".to_string(),
            1,
            true,
            true,
        )];
        let gpu_counts = GpuCategorizer::calculate_gpu_distribution(&validations);
        assert_eq!(gpu_counts.get("A100"), Some(&1));

        // Test null/empty strings
        let validations = vec![ExecutorValidationResult::new_for_testing(
            "exec1".to_string(),
            "".to_string(),
            1,
            true,
            true,
        )];
        let gpu_counts = GpuCategorizer::calculate_gpu_distribution(&validations);
        assert_eq!(gpu_counts.get("OTHER"), Some(&1));

        // Test whitespace-only strings
        let validations = vec![ExecutorValidationResult::new_for_testing(
            "exec1".to_string(),
            "   ".to_string(),
            1,
            true,
            true,
        )];
        let gpu_counts = GpuCategorizer::calculate_gpu_distribution(&validations);
        assert_eq!(gpu_counts.get("OTHER"), Some(&1));
    }

    #[test]
    fn test_gpu_category_enum() {
        // Test enum variants
        let a100 = GpuCategory::A100;
        let h100 = GpuCategory::H100;
        let b200 = GpuCategory::B200;
        let other = GpuCategory::Other("CustomGPU".to_string());

        assert_eq!(a100, GpuCategory::A100);
        assert_ne!(h100, a100);
        assert_eq!(b200, GpuCategory::B200);
        assert_ne!(b200, h100);

        match other {
            GpuCategory::Other(name) => assert_eq!(name, "CustomGPU"),
            _ => panic!("Expected Other variant"),
        }

        // Test Debug trait
        let debug_str = format!("{a100:?}");
        assert!(debug_str.contains("A100"));

        // Test Clone trait
        let a100_clone = a100.clone();
        assert_eq!(a100, a100_clone);
    }

    #[test]
    fn test_executor_validation_result() {
        let result = ExecutorValidationResult::new_for_testing(
            "test_executor".to_string(),
            "A100".to_string(),
            4,
            true,
            true,
        );

        assert_eq!(result.executor_id, "test_executor");
        assert_eq!(result.gpu_model, "A100");
        assert_eq!(result.gpu_count, 4);
        assert!(result.is_valid);
        assert!(result.attestation_valid);
        assert_eq!(result.gpu_memory_gb, 80.0);

        // Test validation timestamp is recent
        let now = Utc::now();
        let diff = now.signed_duration_since(result.validation_timestamp);
        assert!(diff.num_seconds() < 1);
    }

    #[test]
    fn test_complex_gpu_normalization_scenarios() {
        use std::str::FromStr;

        // Test multiple NVIDIA prefixes
        assert_eq!(
            GpuCategory::from_str("NVIDIA NVIDIA GeForce RTX 4090")
                .unwrap()
                .to_string(),
            "OTHER"
        );

        // Test mixed case with numbers
        assert_eq!(
            GpuCategory::from_str("nvidia a100-80gb-pcie")
                .unwrap()
                .to_string(),
            "A100"
        );

        // Test Tesla prefix variations
        assert_eq!(
            GpuCategory::from_str("Tesla V100").unwrap().to_string(),
            "OTHER"
        );

        // Test partial matches
        assert_eq!(
            GpuCategory::from_str("Some A100 GPU").unwrap().to_string(),
            "A100"
        );

        // Test RTX variants with spaces
        assert_eq!(
            GpuCategory::from_str("RTX   4090   Ti")
                .unwrap()
                .to_string(),
            "OTHER"
        );
    }

    #[test]
    fn test_profile_edge_cases_with_zero_gpus() {
        let miner_uid = MinerUid::new(789);
        let validations = vec![ExecutorValidationResult::new_for_testing(
            "exec1".to_string(),
            "A100".to_string(),
            0, // Zero GPUs
            true,
            true,
        )];

        let profile = MinerGpuProfile::new(miner_uid, &validations, 0.0);

        assert_eq!(profile.total_gpu_count(), 0);
        assert!(profile.has_gpu_model("A100"));
        assert_eq!(profile.get_gpu_count("A100"), 0);
    }

    #[test]
    fn test_large_gpu_counts() {
        let miner_uid = MinerUid::new(999);
        let validations = vec![ExecutorValidationResult::new_for_testing(
            "exec1".to_string(),
            "A100".to_string(),
            u32::MAX as usize,
            true,
            true,
        )];

        let profile = MinerGpuProfile::new(miner_uid, &validations, 1.0);

        assert_eq!(profile.total_gpu_count(), u32::MAX);
        assert_eq!(profile.get_gpu_count("A100"), u32::MAX);
    }
}
