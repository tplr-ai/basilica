#[cfg(test)]
mod tests {
    use crate::gpu::categorization::*;
    use chrono::Utc;
    use common::identity::MinerUid;

    #[test]
    fn test_gpu_model_normalization() {
        // Test H100 variants
        assert_eq!(
            GpuCategorizer::normalize_gpu_model("NVIDIA H100 PCIe"),
            "H100"
        );
        assert_eq!(GpuCategorizer::normalize_gpu_model("H100 SXM5"), "H100");
        assert_eq!(GpuCategorizer::normalize_gpu_model("h100"), "H100");
        assert_eq!(GpuCategorizer::normalize_gpu_model("Tesla H100"), "H100");
        assert_eq!(
            GpuCategorizer::normalize_gpu_model("NVIDIA H100-80GB"),
            "H100"
        );

        // Test H200 variants
        assert_eq!(GpuCategorizer::normalize_gpu_model("NVIDIA H200"), "H200");
        assert_eq!(GpuCategorizer::normalize_gpu_model("H200 SXM"), "H200");
        assert_eq!(GpuCategorizer::normalize_gpu_model("h200"), "H200");
        assert_eq!(GpuCategorizer::normalize_gpu_model("Tesla H200"), "H200");

        // Test other GPU variants (should all return OTHER)
        assert_eq!(GpuCategorizer::normalize_gpu_model("A100 80GB"), "OTHER");
        assert_eq!(GpuCategorizer::normalize_gpu_model("Tesla A100"), "OTHER");
        assert_eq!(
            GpuCategorizer::normalize_gpu_model("NVIDIA A100-SXM4-40GB"),
            "OTHER"
        );
        assert_eq!(GpuCategorizer::normalize_gpu_model("a100"), "OTHER");
        assert_eq!(
            GpuCategorizer::normalize_gpu_model("GeForce RTX 4090"),
            "OTHER"
        );
        assert_eq!(GpuCategorizer::normalize_gpu_model("RTX 4090"), "OTHER");
        assert_eq!(
            GpuCategorizer::normalize_gpu_model("NVIDIA GeForce RTX 4090"),
            "OTHER"
        );
        assert_eq!(GpuCategorizer::normalize_gpu_model("rtx4090"), "OTHER");
        assert_eq!(GpuCategorizer::normalize_gpu_model("RTX 3090 Ti"), "OTHER");
        assert_eq!(
            GpuCategorizer::normalize_gpu_model("GeForce RTX 3090"),
            "OTHER"
        );
        assert_eq!(GpuCategorizer::normalize_gpu_model("RTX 3080"), "OTHER");
        assert_eq!(GpuCategorizer::normalize_gpu_model("RTX 4080"), "OTHER");

        // Test unknown models
        assert_eq!(GpuCategorizer::normalize_gpu_model("Unknown GPU"), "OTHER");
        assert_eq!(GpuCategorizer::normalize_gpu_model(""), "OTHER");
        assert_eq!(GpuCategorizer::normalize_gpu_model("V100"), "OTHER");
        assert_eq!(GpuCategorizer::normalize_gpu_model("GTX 1080"), "OTHER");

        // Test edge cases
        assert_eq!(GpuCategorizer::normalize_gpu_model("   H100   "), "H100");
        assert_eq!(
            GpuCategorizer::normalize_gpu_model("NVIDIA NVIDIA H100"),
            "H100"
        );
    }

    #[test]
    fn test_model_to_category_conversion() {
        // Test all known categories
        assert_eq!(GpuCategorizer::model_to_category("H100"), GpuCategory::H100);
        assert_eq!(GpuCategorizer::model_to_category("H200"), GpuCategory::H200);
        // These should return Other now
        match GpuCategorizer::model_to_category("A100") {
            GpuCategory::Other(model) => assert_eq!(model, "A100"),
            _ => panic!("Expected Other category"),
        }
        match GpuCategorizer::model_to_category("RTX4090") {
            GpuCategory::Other(model) => assert_eq!(model, "RTX4090"),
            _ => panic!("Expected Other category"),
        }

        // Test case sensitivity
        assert_eq!(GpuCategorizer::model_to_category("h100"), GpuCategory::H100);
        assert_eq!(GpuCategorizer::model_to_category("h200"), GpuCategory::H200);

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
    fn test_primary_gpu_determination() {
        // Test single GPU type
        let validations = vec![
            ExecutorValidationResult::new_for_testing(
                "exec1".to_string(),
                "NVIDIA H100".to_string(),
                2,
                true,
                true,
            ),
            ExecutorValidationResult::new_for_testing(
                "exec2".to_string(),
                "H100 SXM".to_string(),
                1,
                true,
                true,
            ),
        ];

        let primary = GpuCategorizer::determine_primary_gpu_model(&validations);
        assert_eq!(primary, "H100");

        // Test multiple GPU types (should pick most common by count)
        let validations = vec![
            ExecutorValidationResult::new_for_testing(
                "exec1".to_string(),
                "NVIDIA H100".to_string(),
                1,
                true,
                true,
            ),
            ExecutorValidationResult::new_for_testing(
                "exec2".to_string(),
                "H200".to_string(),
                2,
                true,
                true,
            ),
            ExecutorValidationResult::new_for_testing(
                "exec3".to_string(),
                "NVIDIA H200".to_string(),
                1,
                true,
                true,
            ),
        ];

        let primary = GpuCategorizer::determine_primary_gpu_model(&validations);
        assert_eq!(primary, "H200"); // 3 H200 vs 1 H100

        // Test tie scenarios - should return the first one found
        let validations = vec![
            ExecutorValidationResult::new_for_testing(
                "exec1".to_string(),
                "H100".to_string(),
                2,
                true,
                true,
            ),
            ExecutorValidationResult::new_for_testing(
                "exec2".to_string(),
                "H200".to_string(),
                2,
                true,
                true,
            ),
        ];

        let primary = GpuCategorizer::determine_primary_gpu_model(&validations);
        // Should be either H100 or H200 (both have count 2)
        assert!(primary == "H100" || primary == "H200");

        // Test empty validation results
        let validations = vec![];
        let primary = GpuCategorizer::determine_primary_gpu_model(&validations);
        assert_eq!(primary, "OTHER");

        // Test all invalid validations
        let validations = vec![
            ExecutorValidationResult::new_for_testing(
                "exec1".to_string(),
                "H100".to_string(),
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

        let primary = GpuCategorizer::determine_primary_gpu_model(&validations);
        assert_eq!(primary, "OTHER");
    }

    #[test]
    fn test_gpu_distribution_calculation() {
        // Test single GPU model
        let validations = vec![
            ExecutorValidationResult::new_for_testing(
                "exec1".to_string(),
                "H100".to_string(),
                2,
                true,
                true,
            ),
            ExecutorValidationResult::new_for_testing(
                "exec2".to_string(),
                "NVIDIA H100".to_string(),
                1,
                true,
                true,
            ),
        ];

        let distribution = GpuCategorizer::calculate_gpu_distribution(&validations);
        assert_eq!(distribution.get("H100"), Some(&3));
        assert_eq!(distribution.len(), 1);

        // Test multiple GPU models
        let validations = vec![
            ExecutorValidationResult::new_for_testing(
                "exec1".to_string(),
                "H100".to_string(),
                1,
                true,
                true,
            ),
            ExecutorValidationResult::new_for_testing(
                "exec2".to_string(),
                "H200".to_string(),
                2,
                true,
                true,
            ),
            ExecutorValidationResult::new_for_testing(
                "exec3".to_string(),
                "UNKNOWN".to_string(),
                1,
                true,
                true,
            ),
        ];

        let distribution = GpuCategorizer::calculate_gpu_distribution(&validations);
        assert_eq!(distribution.get("H100"), Some(&1));
        assert_eq!(distribution.get("H200"), Some(&2));
        assert_eq!(distribution.get("OTHER"), Some(&1));
        assert_eq!(distribution.len(), 3);

        // Test mixed valid/invalid validations
        let validations = vec![
            ExecutorValidationResult::new_for_testing(
                "exec1".to_string(),
                "H100".to_string(),
                1,
                true,
                true,
            ),
            ExecutorValidationResult::new_for_testing(
                "exec2".to_string(),
                "H200".to_string(),
                1,
                false,
                true,
            ),
            ExecutorValidationResult::new_for_testing(
                "exec3".to_string(),
                "OTHER".to_string(),
                1,
                true,
                false,
            ),
        ];

        let distribution = GpuCategorizer::calculate_gpu_distribution(&validations);
        assert_eq!(distribution.get("H100"), Some(&1));
        assert_eq!(distribution.get("H200"), None);
        assert_eq!(distribution.get("OTHER"), None);
        assert_eq!(distribution.len(), 1);

        // Test zero GPU counts
        let validations = vec![ExecutorValidationResult::new_for_testing(
            "exec1".to_string(),
            "H100".to_string(),
            0,
            true,
            true,
        )];

        let distribution = GpuCategorizer::calculate_gpu_distribution(&validations);
        assert_eq!(distribution.get("H100"), Some(&0));
    }

    #[test]
    fn test_miner_gpu_profile_creation() {
        let miner_uid = MinerUid::new(123);
        let validations = vec![
            ExecutorValidationResult::new_for_testing(
                "exec1".to_string(),
                "H100".to_string(),
                2,
                true,
                true,
            ),
            ExecutorValidationResult::new_for_testing(
                "exec2".to_string(),
                "H200".to_string(),
                1,
                true,
                true,
            ),
        ];

        let profile = MinerGpuProfile::new(miner_uid, &validations, 0.85);

        assert_eq!(profile.miner_uid, miner_uid);
        assert_eq!(profile.primary_gpu_model, "H100"); // More GPUs
        assert_eq!(profile.total_score, 0.85);
        assert_eq!(profile.verification_count, 2);
        assert_eq!(profile.total_gpu_count(), 3);
        assert_eq!(profile.get_gpu_count("H100"), 2);
        assert_eq!(profile.get_gpu_count("H200"), 1);
        assert!(profile.has_gpu_model("H100"));
        assert!(profile.has_gpu_model("H200"));
        assert!(!profile.has_gpu_model("A100"));

        // Test profile updates
        let mut profile = profile;
        let new_validations = vec![ExecutorValidationResult::new_for_testing(
            "exec3".to_string(),
            "H200".to_string(),
            4,
            true,
            true,
        )];

        profile.update_with_validations(&new_validations, 0.92);

        assert_eq!(profile.primary_gpu_model, "H200");
        assert_eq!(profile.total_score, 0.92);
        assert_eq!(profile.verification_count, 1);
        assert_eq!(profile.total_gpu_count(), 4);
        assert_eq!(profile.get_gpu_count("H200"), 4);
        assert_eq!(profile.get_gpu_count("H100"), 0); // Replaced

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
                "H100".to_string(),
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
        assert_eq!(models_by_count[0], ("H100".to_string(), 4));
        assert_eq!(models_by_count[1], ("OTHER".to_string(), 3)); // A100(1) + RTX4090(2) = OTHER(3)
    }

    #[test]
    fn test_edge_cases() {
        // Test unicode GPU names
        let validations = vec![ExecutorValidationResult::new_for_testing(
            "exec1".to_string(),
            "NVIDIA H100-新".to_string(),
            1,
            true,
            true,
        )];
        let primary = GpuCategorizer::determine_primary_gpu_model(&validations);
        assert_eq!(primary, "H100");

        // Test very long GPU names
        let long_name = "A".repeat(1000) + " H100";
        let validations = vec![ExecutorValidationResult::new_for_testing(
            "exec1".to_string(),
            long_name,
            1,
            true,
            true,
        )];
        let primary = GpuCategorizer::determine_primary_gpu_model(&validations);
        assert_eq!(primary, "H100");

        // Test special characters
        let validations = vec![ExecutorValidationResult::new_for_testing(
            "exec1".to_string(),
            "NVIDIA@@H100##PCIe".to_string(),
            1,
            true,
            true,
        )];
        let primary = GpuCategorizer::determine_primary_gpu_model(&validations);
        assert_eq!(primary, "H100");

        // Test null/empty strings
        let validations = vec![ExecutorValidationResult::new_for_testing(
            "exec1".to_string(),
            "".to_string(),
            1,
            true,
            true,
        )];
        let primary = GpuCategorizer::determine_primary_gpu_model(&validations);
        assert_eq!(primary, "OTHER");

        // Test whitespace-only strings
        let validations = vec![ExecutorValidationResult::new_for_testing(
            "exec1".to_string(),
            "   ".to_string(),
            1,
            true,
            true,
        )];
        let primary = GpuCategorizer::determine_primary_gpu_model(&validations);
        assert_eq!(primary, "OTHER");
    }

    #[test]
    fn test_gpu_category_enum() {
        // Test enum variants
        let h100 = GpuCategory::H100;
        let h200 = GpuCategory::H200;
        let other = GpuCategory::Other("CustomGPU".to_string());

        assert_eq!(h100, GpuCategory::H100);
        assert_ne!(h100, h200);

        match other {
            GpuCategory::Other(name) => assert_eq!(name, "CustomGPU"),
            _ => panic!("Expected Other variant"),
        }

        // Test Debug trait
        let debug_str = format!("{h100:?}");
        assert!(debug_str.contains("H100"));

        // Test Clone trait
        let h100_clone = h100.clone();
        assert_eq!(h100, h100_clone);
    }

    #[test]
    fn test_executor_validation_result() {
        let result = ExecutorValidationResult::new_for_testing(
            "test_executor".to_string(),
            "H100".to_string(),
            4,
            true,
            true,
        );

        assert_eq!(result.executor_id, "test_executor");
        assert_eq!(result.gpu_model, "H100");
        assert_eq!(result.gpu_count, 4);
        assert!(result.is_valid);
        assert!(result.attestation_valid);
        assert_eq!(result.gpu_memory_gb, 80);

        // Test validation timestamp is recent
        let now = Utc::now();
        let diff = now.signed_duration_since(result.validation_timestamp);
        assert!(diff.num_seconds() < 1);
    }

    #[test]
    fn test_complex_gpu_normalization_scenarios() {
        // Test multiple NVIDIA prefixes
        assert_eq!(
            GpuCategorizer::normalize_gpu_model("NVIDIA NVIDIA GeForce RTX 4090"),
            "OTHER"
        );

        // Test mixed case with numbers
        assert_eq!(
            GpuCategorizer::normalize_gpu_model("nvidia h100-80gb-pcie"),
            "H100"
        );

        // Test Tesla prefix variations
        assert_eq!(GpuCategorizer::normalize_gpu_model("Tesla V100"), "OTHER");

        // Test partial matches
        assert_eq!(GpuCategorizer::normalize_gpu_model("Some H100 GPU"), "H100");

        // Test RTX variants with spaces
        assert_eq!(
            GpuCategorizer::normalize_gpu_model("RTX   4090   Ti"),
            "OTHER"
        );
    }

    #[test]
    fn test_profile_edge_cases_with_zero_gpus() {
        let miner_uid = MinerUid::new(789);
        let validations = vec![ExecutorValidationResult::new_for_testing(
            "exec1".to_string(),
            "H100".to_string(),
            0, // Zero GPUs
            true,
            true,
        )];

        let profile = MinerGpuProfile::new(miner_uid, &validations, 0.0);

        assert_eq!(profile.total_gpu_count(), 0);
        assert_eq!(profile.primary_gpu_model, "H100");
        assert!(profile.has_gpu_model("H100"));
        assert_eq!(profile.get_gpu_count("H100"), 0);
    }

    #[test]
    fn test_large_gpu_counts() {
        let miner_uid = MinerUid::new(999);
        let validations = vec![ExecutorValidationResult::new_for_testing(
            "exec1".to_string(),
            "H100".to_string(),
            u32::MAX as usize,
            true,
            true,
        )];

        let profile = MinerGpuProfile::new(miner_uid, &validations, 1.0);

        assert_eq!(profile.total_gpu_count(), u32::MAX);
        assert_eq!(profile.get_gpu_count("H100"), u32::MAX);
    }
}
