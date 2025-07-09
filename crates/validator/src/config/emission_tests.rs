#[cfg(test)]
mod tests {
    use crate::config::emission::EmissionConfig;
    use std::collections::HashMap;
    use std::io::Write;
    use std::path::Path;
    use tempfile::NamedTempFile;

    #[test]
    fn test_default_config_is_valid() {
        let config = EmissionConfig::default();
        assert!(config.validate().is_ok());

        // Verify default values
        assert_eq!(config.burn_percentage, 0.0);
        assert_eq!(config.burn_uid, 0);
        assert_eq!(config.weight_set_interval_blocks, 360);
        assert_eq!(config.min_miners_per_category, 1);

        // Verify default GPU allocations
        assert_eq!(config.gpu_allocations.len(), 2);
        assert_eq!(config.gpu_allocations.get("H100"), Some(&40.0));
        assert_eq!(config.gpu_allocations.get("H200"), Some(&60.0));

        // Verify they sum to 100
        let total: f64 = config.gpu_allocations.values().sum();
        assert!((total - 100.0).abs() < 0.01);
    }

    #[test]
    fn test_burn_percentage_validation() {
        // Test valid ranges
        let mut config = EmissionConfig {
            burn_percentage: 0.0,
            ..Default::default()
        };
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
        allocations.insert("H100".to_string(), 30.0);
        allocations.insert("H200".to_string(), 70.0);
        config.gpu_allocations = allocations;
        assert!(config.validate().is_ok());

        // Test three-way split
        let mut allocations = HashMap::new();
        allocations.insert("H100".to_string(), 33.33);
        allocations.insert("H200".to_string(), 33.33);
        allocations.insert("A100".to_string(), 33.34);
        config.gpu_allocations = allocations;
        assert!(config.validate().is_ok());

        // Test invalid allocation sums
        let mut allocations = HashMap::new();
        allocations.insert("H100".to_string(), 40.0);
        allocations.insert("H200".to_string(), 40.0);
        config.gpu_allocations = allocations;
        assert!(config.validate().is_err()); // Sum = 80

        let mut allocations = HashMap::new();
        allocations.insert("H100".to_string(), 60.0);
        allocations.insert("H200".to_string(), 60.0);
        config.gpu_allocations = allocations;
        assert!(config.validate().is_err()); // Sum = 120

        // Test empty allocations
        config.gpu_allocations = HashMap::new();
        assert!(config.validate().is_err());

        // Test negative allocations
        let mut allocations = HashMap::new();
        allocations.insert("H100".to_string(), -10.0);
        allocations.insert("H200".to_string(), 110.0);
        config.gpu_allocations = allocations;
        assert!(config.validate().is_err());
    }

    #[test]
    fn test_weight_interval_validation() {
        // Test valid intervals
        let mut config = EmissionConfig {
            weight_set_interval_blocks: 1,
            ..Default::default()
        };
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
        let config = EmissionConfig::default();

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
min_miners_per_category = 2

[gpu_allocations]
H100 = 25.0
H200 = 50.0
A100 = 25.0
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
        assert_eq!(config.min_miners_per_category, 2);
        assert_eq!(config.gpu_allocations.len(), 3);
        assert_eq!(config.gpu_allocations.get("H100"), Some(&25.0));
        assert_eq!(config.gpu_allocations.get("H200"), Some(&50.0));
        assert_eq!(config.gpu_allocations.get("A100"), Some(&25.0));

        // Test loading from invalid TOML file (allocations don't sum to 100)
        let invalid_toml = r#"
burn_percentage = 10.0
burn_uid = 0
weight_set_interval_blocks = 360
min_miners_per_category = 1

[gpu_allocations]
H100 = 30.0
H200 = 30.0
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
            weight_set_interval_blocks: 0,   // Invalid - should use default
            min_miners_per_category: 0,      // Invalid - should use default
        };

        let merged = partial_config.merge_with_defaults();

        assert_eq!(merged.burn_percentage, 20.0); // Preserved
        assert_eq!(merged.burn_uid, 456); // Preserved
        assert_eq!(merged.weight_set_interval_blocks, 360); // Default
        assert_eq!(merged.min_miners_per_category, 1); // Default
        assert_eq!(merged.gpu_allocations.len(), 2); // Default GPU allocations

        // Test complete config override (no merging needed)
        let complete_config = EmissionConfig::for_testing();
        let merged = complete_config.clone().merge_with_defaults();
        assert_eq!(merged, complete_config);
    }

    #[test]
    fn test_edge_cases() {
        // Test extreme values - maximum values
        let mut config = EmissionConfig {
            burn_percentage: 100.0,
            burn_uid: u16::MAX,
            weight_set_interval_blocks: u64::MAX,
            min_miners_per_category: u32::MAX,
            ..Default::default()
        };
        assert!(config.validate().is_ok());

        // Test unicode in GPU model names
        let mut allocations = HashMap::new();
        allocations.insert("H100-新".to_string(), 50.0);
        allocations.insert("H200-α".to_string(), 50.0);
        config.gpu_allocations = allocations;
        assert!(config.validate().is_ok());

        // Test very long GPU model names
        let mut allocations = HashMap::new();
        let long_name = "A".repeat(1000);
        allocations.insert(long_name, 100.0);
        config.gpu_allocations = allocations;
        assert!(config.validate().is_ok());

        // Test very small positive allocations
        let mut allocations = HashMap::new();
        allocations.insert("H100".to_string(), 0.001);
        allocations.insert("H200".to_string(), 99.999);
        config.gpu_allocations = allocations;
        assert!(config.validate().is_ok());
    }

    #[test]
    fn test_gpu_allocation_methods() {
        let mut config = EmissionConfig::default();

        // Test has_gpu_model
        assert!(config.has_gpu_model("H100"));
        assert!(config.has_gpu_model("H200"));
        assert!(!config.has_gpu_model("A100"));

        // Test get_gpu_allocation
        assert_eq!(config.get_gpu_allocation("H100"), Some(40.0));
        assert_eq!(config.get_gpu_allocation("H200"), Some(60.0));
        assert_eq!(config.get_gpu_allocation("A100"), None);

        // Test set_gpu_allocation with valid values
        // First adjust existing allocations to make room
        config.set_gpu_allocation("H100".to_string(), 30.0).unwrap();
        config.set_gpu_allocation("H200".to_string(), 50.0).unwrap();
        assert!(config.set_gpu_allocation("A100".to_string(), 20.0).is_ok());

        // Validate the final configuration sums to 100%
        assert!(config.validate().is_ok());

        assert_eq!(config.get_gpu_allocation("A100"), Some(20.0));
        assert_eq!(config.get_gpu_allocation("H100"), Some(30.0));
        assert_eq!(config.get_gpu_allocation("H200"), Some(50.0));

        // Test set_gpu_allocation with invalid values (negative)
        assert!(config
            .set_gpu_allocation("Test".to_string(), -10.0)
            .is_err());

        // Test set_gpu_allocation that would make total != 100
        config.set_gpu_allocation("H300".to_string(), 50.0).unwrap(); // Should succeed but make config invalid
        assert!(config.validate().is_err()); // Should fail validation

        // Remove the invalid H300 allocation to restore valid state
        config.remove_gpu_allocation("H300");
        assert!(config.validate().is_ok());

        // Test remove_gpu_allocation
        let removed = config.remove_gpu_allocation("A100");
        assert_eq!(removed, Some(20.0));
        assert!(!config.has_gpu_model("A100"));

        // Test remove non-existent allocation
        let removed = config.remove_gpu_allocation("NonExistent");
        assert_eq!(removed, None);

        // Test gpu_models_by_allocation (should be sorted by percentage desc)
        let models = config.gpu_models_by_allocation();
        assert_eq!(models.len(), 2);
        assert_eq!(models[0], ("H200".to_string(), 50.0));
        assert_eq!(models[1], ("H100".to_string(), 30.0));

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
        assert_eq!(config.min_miners_per_category, 1);
        assert_eq!(config.gpu_allocations.len(), 2);

        let total: f64 = config.gpu_allocations.values().sum();
        assert!((total - 100.0).abs() < 0.01);
    }

    #[test]
    fn test_min_miners_per_category_validation() {
        // Test valid values
        let mut config = EmissionConfig {
            min_miners_per_category: 1,
            ..Default::default()
        };
        assert!(config.validate().is_ok());

        config.min_miners_per_category = 10;
        assert!(config.validate().is_ok());

        config.min_miners_per_category = u32::MAX;
        assert!(config.validate().is_ok());

        // Test invalid value
        config.min_miners_per_category = 0;
        assert!(config.validate().is_err());
    }

    #[test]
    fn test_precision_handling() {
        // Test that small floating point differences are handled correctly
        let mut allocations = HashMap::new();
        allocations.insert("H100".to_string(), 33.333333);
        allocations.insert("H200".to_string(), 33.333333);
        allocations.insert("A100".to_string(), 33.333334);

        let config = EmissionConfig {
            burn_percentage: 0.0,
            burn_uid: 0,
            gpu_allocations: allocations,
            weight_set_interval_blocks: 360,
            min_miners_per_category: 1,
        };

        // Should be valid because total is very close to 100.0
        assert!(config.validate().is_ok());

        // Test that larger differences are caught
        let mut allocations = HashMap::new();
        allocations.insert("H100".to_string(), 33.0);
        allocations.insert("H200".to_string(), 33.0);
        allocations.insert("A100".to_string(), 33.0);

        let config = EmissionConfig {
            burn_percentage: 0.0,
            burn_uid: 0,
            gpu_allocations: allocations,
            weight_set_interval_blocks: 360,
            min_miners_per_category: 1,
        };

        // Should be invalid because total is 99.0 (difference > 0.01)
        assert!(config.validate().is_err());
    }
}
