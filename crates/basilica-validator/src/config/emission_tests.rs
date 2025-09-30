#[cfg(test)]
mod tests {
    use crate::config::emission::{EmissionConfig, GpuAllocation, DEFAULT_BURN_UID};
    use std::collections::HashMap;
    use std::io::Write;
    use std::path::Path;
    use tempfile::NamedTempFile;

    #[test]
    fn test_default_config_is_invalid() {
        let config = EmissionConfig::default();
        assert!(config.validate().is_err()); // Default is intentionally empty

        // Verify default values
        assert_eq!(config.burn_percentage, 0.0);
        assert_eq!(config.burn_uid, DEFAULT_BURN_UID);
        assert_eq!(config.weight_set_interval_blocks, 360);

        // Verify default GPU allocations are empty (requires explicit config)
        assert_eq!(config.gpu_allocations.len(), 0);
    }

    #[test]
    fn test_burn_percentage_validation() {
        // Test valid ranges - use for_testing which has valid GPU allocations
        let mut config = EmissionConfig::for_testing();
        config.burn_percentage = 0.0;
        assert!(config.validate().is_ok());

        config.burn_percentage = 50.0;
        assert!(config.validate().is_ok());

        config.burn_percentage = 100.0;
        assert!(config.validate().is_ok());

        // Test invalid ranges
        config.burn_percentage = -0.1;
        assert!(config.validate().is_err());

        config.burn_percentage = 100.1;
        assert!(config.validate().is_err());

        config.burn_percentage = -50.0;
        assert!(config.validate().is_err());

        config.burn_percentage = 150.0;
        assert!(config.validate().is_err());
    }

    #[test]
    fn test_gpu_allocations_sum_to_100() {
        let mut config = EmissionConfig::default();

        // Test valid allocation sums
        let mut allocations = HashMap::new();
        allocations.insert("A100".to_string(), GpuAllocation::new(30.0));
        allocations.insert("H100".to_string(), GpuAllocation::new(70.0));
        config.gpu_allocations = allocations;
        assert!(config.validate().is_ok());

        // Test three-way split
        let mut allocations = HashMap::new();
        allocations.insert("A100".to_string(), GpuAllocation::new(33.33));
        allocations.insert("H100".to_string(), GpuAllocation::new(33.33));
        allocations.insert("B200".to_string(), GpuAllocation::new(33.34));
        config.gpu_allocations = allocations;
        assert!(config.validate().is_ok());

        // Test invalid allocation sums
        let mut allocations = HashMap::new();
        allocations.insert("A100".to_string(), GpuAllocation::new(40.0));
        allocations.insert("H100".to_string(), GpuAllocation::new(40.0));
        config.gpu_allocations = allocations;
        assert!(config.validate().is_err()); // Sum = 80

        let mut allocations = HashMap::new();
        allocations.insert("A100".to_string(), GpuAllocation::new(60.0));
        allocations.insert("H100".to_string(), GpuAllocation::new(60.0));
        config.gpu_allocations = allocations;
        assert!(config.validate().is_err()); // Sum = 120

        // Test empty allocations
        config.gpu_allocations = HashMap::new();
        assert!(config.validate().is_err());

        // Test negative allocations
        let mut allocations = HashMap::new();
        allocations.insert("A100".to_string(), GpuAllocation::new(-10.0));
        allocations.insert("H100".to_string(), GpuAllocation::new(110.0));
        config.gpu_allocations = allocations;
        assert!(config.validate().is_err());

        // Test min_gpu_count validation
        let mut allocations = HashMap::new();
        allocations.insert("A100".to_string(), GpuAllocation::with_min_count(50.0, 0));
        allocations.insert("H100".to_string(), GpuAllocation::with_min_count(50.0, 1));
        config.gpu_allocations = allocations;
        assert!(config.validate().is_err()); // min_gpu_count can't be 0

        // Test valid min_gpu_count
        let mut allocations = HashMap::new();
        allocations.insert("A100".to_string(), GpuAllocation::with_min_count(25.0, 4));
        allocations.insert("H100".to_string(), GpuAllocation::with_min_count(25.0, 2));
        allocations.insert("B200".to_string(), GpuAllocation::with_min_count(50.0, 8));
        config.gpu_allocations = allocations;
        assert!(config.validate().is_ok());
    }

    #[test]
    fn test_weight_interval_validation() {
        // Test valid intervals - use for_testing which has valid GPU allocations
        let mut config = EmissionConfig::for_testing();
        config.weight_set_interval_blocks = 1;
        assert!(config.validate().is_ok());

        config.weight_set_interval_blocks = 360;
        assert!(config.validate().is_ok());

        config.weight_set_interval_blocks = 1000;
        assert!(config.validate().is_ok());

        // Test zero interval (should fail)
        config.weight_set_interval_blocks = 0;
        assert!(config.validate().is_err());
    }

    #[test]
    fn test_config_serialization() {
        let config = EmissionConfig::for_testing(); // Use valid config for serialization test

        // Test TOML serialization/deserialization
        let toml_str = toml::to_string(&config).expect("Failed to serialize to TOML");
        let deserialized: EmissionConfig =
            toml::from_str(&toml_str).expect("Failed to deserialize from TOML");
        assert_eq!(config, deserialized);

        // Test JSON serialization/deserialization
        let json_str = serde_json::to_string(&config).expect("Failed to serialize to JSON");
        let deserialized: EmissionConfig =
            serde_json::from_str(&json_str).expect("Failed to deserialize from JSON");
        assert_eq!(config, deserialized);

        // Test that serialized config is valid
        assert!(deserialized.validate().is_ok());
    }

    #[test]
    fn test_config_from_toml_file() {
        // Test loading from valid TOML file
        let toml_content = r#"
burn_percentage = 15.0
burn_uid = 123
weight_set_interval_blocks = 720
weight_version_key = 0

[gpu_allocations]
A100 = 25.0
H100 = 50.0
B200 = 25.0
"#;

        let mut temp_file = NamedTempFile::new().expect("Failed to create temp file");
        temp_file
            .write_all(toml_content.as_bytes())
            .expect("Failed to write temp file");

        let config = EmissionConfig::from_toml_file(temp_file.path())
            .expect("Failed to load from TOML file");

        assert_eq!(config.burn_percentage, 15.0);
        assert_eq!(config.burn_uid, 123);
        assert_eq!(config.weight_set_interval_blocks, 720);
        assert_eq!(config.gpu_allocations.len(), 3);
        assert_eq!(config.get_gpu_allocation_weight("A100"), Some(25.0));
        assert_eq!(config.get_gpu_allocation_weight("H100"), Some(50.0));
        assert_eq!(config.get_gpu_allocation_weight("B200"), Some(25.0));

        // Test loading from invalid TOML file (allocations don't sum to 100)
        let invalid_toml = r#"
burn_percentage = 10.0
burn_uid = 0
weight_set_interval_blocks = 360

[gpu_allocations]
A100 = 30.0
H100 = 30.0
"#;

        let mut temp_file = NamedTempFile::new().expect("Failed to create temp file");
        temp_file
            .write_all(invalid_toml.as_bytes())
            .expect("Failed to write temp file");

        let result = EmissionConfig::from_toml_file(temp_file.path());
        assert!(result.is_err());

        // Test loading from non-existent file
        let result = EmissionConfig::from_toml_file(Path::new("/non/existent/file.toml"));
        assert!(result.is_err());
    }

    #[test]
    fn test_config_merge_with_defaults() {
        // Test partial config merging
        let partial_config = EmissionConfig {
            burn_percentage: 20.0,
            burn_uid: 456,
            gpu_allocations: HashMap::new(), // Empty - should use default
            min_miners_per_category: 1,
            weight_set_interval_blocks: 0, // Invalid - should use default
            weight_version_key: 0,
        };

        let merged = partial_config.merge_with_defaults();

        assert_eq!(merged.burn_percentage, 20.0); // Preserved
        assert_eq!(merged.burn_uid, 456); // Preserved
        assert_eq!(merged.weight_set_interval_blocks, 360); // Default
        assert_eq!(merged.min_miners_per_category, 1); // Default
        assert_eq!(merged.gpu_allocations.len(), 0); // Default GPU allocations (empty)

        // Test complete config override (no merging needed)
        let complete_config = EmissionConfig::for_testing();
        let merged = complete_config.clone().merge_with_defaults();
        assert_eq!(merged, complete_config);
    }

    #[test]
    fn test_edge_cases() {
        // Test extreme values - maximum values
        let mut config = EmissionConfig::for_testing();
        config.burn_percentage = 100.0;
        config.burn_uid = u16::MAX;
        config.weight_set_interval_blocks = u64::MAX;
        assert!(config.validate().is_ok());

        // Test unicode in GPU model names
        let mut allocations = HashMap::new();
        allocations.insert("A100-新".to_string(), GpuAllocation::new(50.0));
        allocations.insert("H100-α".to_string(), GpuAllocation::new(50.0));
        config.gpu_allocations = allocations;
        assert!(config.validate().is_ok());

        // Test very long GPU model names
        let mut allocations = HashMap::new();
        let long_name = "A".repeat(1000);
        allocations.insert(long_name, GpuAllocation::new(100.0));
        config.gpu_allocations = allocations;
        assert!(config.validate().is_ok());

        // Test very small positive allocations
        let mut allocations = HashMap::new();
        allocations.insert("A100".to_string(), GpuAllocation::new(0.001));
        allocations.insert("H100".to_string(), GpuAllocation::new(99.999));
        config.gpu_allocations = allocations;
        assert!(config.validate().is_ok());
    }

    #[test]
    fn test_gpu_allocation_methods() {
        let mut config = EmissionConfig::for_testing(); // Use testing config which has allocations

        // Test has_gpu_model
        assert!(config.has_gpu_model("A100"));
        assert!(config.has_gpu_model("H100"));
        assert!(config.has_gpu_model("B200"));
        assert!(!config.has_gpu_model("RTX4090"));

        // Test get_gpu_allocation
        assert_eq!(config.get_gpu_allocation_weight("A100"), Some(8.0));
        assert_eq!(config.get_gpu_allocation_weight("H100"), Some(12.0));
        assert_eq!(config.get_gpu_allocation_weight("B200"), Some(80.0));
        assert_eq!(config.get_gpu_allocation_weight("RTX4090"), None);

        // Test set_gpu_allocation with valid values
        // Clear and set allocations that sum to 100%
        config.gpu_allocations.clear();
        config
            .set_gpu_allocation("A100".to_string(), 30.0, 1)
            .unwrap();
        config
            .set_gpu_allocation("H100".to_string(), 50.0, 1)
            .unwrap();
        assert!(config
            .set_gpu_allocation("B200".to_string(), 20.0, 1)
            .is_ok());

        // Validate the final configuration sums to 100%
        assert!(config.validate().is_ok());

        assert_eq!(config.get_gpu_allocation_weight("B200"), Some(20.0));
        assert_eq!(config.get_gpu_allocation_weight("A100"), Some(30.0));
        assert_eq!(config.get_gpu_allocation_weight("H100"), Some(50.0));

        // Test set_gpu_allocation with invalid values (negative)
        assert!(config
            .set_gpu_allocation("Test".to_string(), -10.0, 1)
            .is_err());

        // Test set_gpu_allocation with invalid min_gpu_count
        assert!(config
            .set_gpu_allocation("Test".to_string(), 10.0, 0)
            .is_err());

        // Test set_gpu_allocation that would make total != 100
        config
            .set_gpu_allocation("H300".to_string(), 50.0, 1)
            .unwrap(); // Should succeed but make config invalid
        assert!(config.validate().is_err()); // Should fail validation

        // Remove the invalid H300 allocation to restore valid state
        config.remove_gpu_allocation("H300");
        assert!(config.validate().is_ok());

        // Test remove_gpu_allocation
        let removed = config.remove_gpu_allocation("A100");
        assert!(removed.is_some());
        assert_eq!(removed.unwrap().weight, 30.0);
        assert!(!config.has_gpu_model("A100"));

        // Test remove non-existent allocation
        let removed = config.remove_gpu_allocation("NonExistent");
        assert_eq!(removed, None);

        // Test gpu_models_by_allocation (should be sorted by percentage desc)
        let models = config.gpu_models_by_allocation();
        assert_eq!(models.len(), 2);
        assert_eq!(models[0].0, "H100".to_string());
        assert_eq!(models[0].1.weight, 50.0);
        assert_eq!(models[1].0, "B200".to_string());
        assert_eq!(models[1].1.weight, 20.0);

        // Test gpu_model_count
        assert_eq!(config.gpu_model_count(), 2);
    }

    #[test]
    fn test_for_testing_config() {
        let config = EmissionConfig::for_testing();
        assert!(config.validate().is_ok());

        assert_eq!(config.burn_percentage, 10.0);
        assert_eq!(config.burn_uid, 999);
        assert_eq!(config.weight_set_interval_blocks, 360);
        assert_eq!(config.gpu_allocations.len(), 3);

        let total: f64 = config.gpu_allocations.values().map(|a| a.weight).sum();
        assert!((total - 100.0).abs() < 0.01);
    }

    #[test]
    fn test_precision_handling() {
        // Test that small floating point differences are handled correctly
        let mut allocations = HashMap::new();
        allocations.insert("A100".to_string(), GpuAllocation::new(33.333333));
        allocations.insert("H100".to_string(), GpuAllocation::new(33.333333));
        allocations.insert("B200".to_string(), GpuAllocation::new(33.333334));

        let config = EmissionConfig {
            burn_percentage: 0.0,
            burn_uid: 0,
            gpu_allocations: allocations,
            min_miners_per_category: 1,
            weight_set_interval_blocks: 360,
            weight_version_key: 0,
        };

        // Should be valid because total is very close to 100.0
        assert!(config.validate().is_ok());

        // Test that larger differences are caught
        let mut allocations = HashMap::new();
        allocations.insert("A100".to_string(), GpuAllocation::new(33.0));
        allocations.insert("H100".to_string(), GpuAllocation::new(33.0));
        allocations.insert("B200".to_string(), GpuAllocation::new(33.0));

        let config = EmissionConfig {
            burn_percentage: 0.0,
            burn_uid: 0,
            gpu_allocations: allocations,
            min_miners_per_category: 1,
            weight_set_interval_blocks: 360,
            weight_version_key: 0,
        };

        // Should be invalid because total is 99.0 (difference > 0.01)
        assert!(config.validate().is_err());
    }

    #[test]
    fn test_config_legacy_format_compatibility() {
        // Test loading from legacy TOML format (just floats)
        let toml_content = r#"
burn_percentage = 15.0
burn_uid = 123
weight_set_interval_blocks = 720
weight_version_key = 0
min_miners_per_category = 2

[gpu_allocations]
A100 = 25.0
H100 = 50.0
B200 = 25.0
"#;

        let mut temp_file = NamedTempFile::new().expect("Failed to create temp file");
        temp_file
            .write_all(toml_content.as_bytes())
            .expect("Failed to write temp file");

        let config = EmissionConfig::from_toml_file(temp_file.path())
            .expect("Failed to load from TOML file");

        assert_eq!(config.burn_percentage, 15.0);
        assert_eq!(config.burn_uid, 123);
        assert_eq!(config.min_miners_per_category, 2);
        assert_eq!(config.gpu_allocations.len(), 3);

        // Check that legacy format defaults to min_gpu_count = 1
        let a100 = config.get_gpu_allocation("A100").unwrap();
        assert_eq!(a100.weight, 25.0);
        assert_eq!(a100.min_gpu_count, 1);

        // Test loading from new TOML format with explicit min_gpu_count
        let toml_content = r#"
burn_percentage = 15.0
burn_uid = 123
weight_set_interval_blocks = 720
weight_version_key = 0

[gpu_allocations]
A100 = { weight = 8.0, min_gpu_count = 4 }
H100 = { weight = 12.0, min_gpu_count = 2 }
B200 = { weight = 80.0, min_gpu_count = 8 }
"#;

        let mut temp_file = NamedTempFile::new().expect("Failed to create temp file");
        temp_file
            .write_all(toml_content.as_bytes())
            .expect("Failed to write temp file");

        let config = EmissionConfig::from_toml_file(temp_file.path())
            .expect("Failed to load from TOML file");

        let a100 = config.get_gpu_allocation("A100").unwrap();
        assert_eq!(a100.weight, 8.0);
        assert_eq!(a100.min_gpu_count, 4);

        let h100 = config.get_gpu_allocation("H100").unwrap();
        assert_eq!(h100.weight, 12.0);
        assert_eq!(h100.min_gpu_count, 2);

        let b200 = config.get_gpu_allocation("B200").unwrap();
        assert_eq!(b200.weight, 80.0);
        assert_eq!(b200.min_gpu_count, 8);
    }

    #[test]
    fn test_min_gpu_vram_configuration() {
        // Test loading TOML with min_gpu_vram
        let toml_content = r#"
burn_percentage = 10.0
burn_uid = 204
weight_set_interval_blocks = 360
weight_version_key = 0

[gpu_allocations]
A100 = { weight = 8.0, min_gpu_count = 1, min_gpu_vram = 80 }
H100 = { weight = 12.0, min_gpu_count = 1, min_gpu_vram = 80 }
B200 = { weight = 80.0, min_gpu_count = 8, min_gpu_vram = 192 }
"#;

        let mut temp_file = NamedTempFile::new().expect("Failed to create temp file");
        temp_file
            .write_all(toml_content.as_bytes())
            .expect("Failed to write temp file");

        let config = EmissionConfig::from_toml_file(temp_file.path())
            .expect("Failed to load from TOML file");

        // Verify min_gpu_vram is loaded correctly
        let a100 = config.get_gpu_allocation("A100").unwrap();
        assert_eq!(a100.weight, 8.0);
        assert_eq!(a100.min_gpu_count, 1);
        assert_eq!(a100.min_gpu_vram, Some(80));

        let h100 = config.get_gpu_allocation("H100").unwrap();
        assert_eq!(h100.weight, 12.0);
        assert_eq!(h100.min_gpu_count, 1);
        assert_eq!(h100.min_gpu_vram, Some(80));

        let b200 = config.get_gpu_allocation("B200").unwrap();
        assert_eq!(b200.weight, 80.0);
        assert_eq!(b200.min_gpu_count, 8);
        assert_eq!(b200.min_gpu_vram, Some(192));
    }

    #[test]
    fn test_min_gpu_vram_validation() {
        let mut config = EmissionConfig::for_testing();

        // Test valid min_gpu_vram
        config.gpu_allocations.clear();
        config
            .set_gpu_allocation_with_vram("A100".to_string(), 50.0, 1, Some(80))
            .unwrap();
        config
            .set_gpu_allocation_with_vram("H100".to_string(), 50.0, 1, Some(80))
            .unwrap();
        assert!(config.validate().is_ok());

        // Test invalid min_gpu_vram (0)
        assert!(config
            .set_gpu_allocation_with_vram("B200".to_string(), 50.0, 1, Some(0))
            .is_err());

        // Test mix of with and without min_gpu_vram
        config.gpu_allocations.clear();
        config
            .set_gpu_allocation_with_vram("A100".to_string(), 25.0, 1, Some(80))
            .unwrap();
        config
            .set_gpu_allocation_with_vram("H100".to_string(), 25.0, 1, None)
            .unwrap();
        config
            .set_gpu_allocation_with_vram("B200".to_string(), 50.0, 1, Some(192))
            .unwrap();
        assert!(config.validate().is_ok());
    }

    #[test]
    fn test_gpu_allocation_with_requirements() {
        // Test creating allocations with full requirements
        let alloc = GpuAllocation::with_requirements(50.0, 4, Some(80));
        assert_eq!(alloc.weight, 50.0);
        assert_eq!(alloc.min_gpu_count, 4);
        assert_eq!(alloc.min_gpu_vram, Some(80));

        // Test creating allocations without VRAM requirement
        let alloc = GpuAllocation::with_requirements(30.0, 2, None);
        assert_eq!(alloc.weight, 30.0);
        assert_eq!(alloc.min_gpu_count, 2);
        assert_eq!(alloc.min_gpu_vram, None);

        // Test backward compatibility constructors
        let alloc = GpuAllocation::new(20.0);
        assert_eq!(alloc.weight, 20.0);
        assert_eq!(alloc.min_gpu_count, 1);
        assert_eq!(alloc.min_gpu_vram, None);

        let alloc = GpuAllocation::with_min_count(40.0, 8);
        assert_eq!(alloc.weight, 40.0);
        assert_eq!(alloc.min_gpu_count, 8);
        assert_eq!(alloc.min_gpu_vram, None);
    }
}
