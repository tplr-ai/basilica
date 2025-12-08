#[cfg(test)]
mod tests {
    use crate::config::{
        emission::{GpuAllocation, DEFAULT_BURN_UID},
        ValidatorConfig,
    };
    use basilica_common::config::ConfigValidation;

    #[test]
    fn test_validator_config_includes_emission_config() {
        let config = ValidatorConfig::default();

        // Verify emission config is included
        assert_eq!(config.emission.burn_percentage, 0.0);
        assert_eq!(config.emission.burn_uid, DEFAULT_BURN_UID);
        assert_eq!(config.emission.weight_set_interval_blocks, 360);
        assert_eq!(config.emission.gpu_allocations.len(), 0);

        // Default config should fail validation because GPU allocations are empty
        assert!(config.validate().is_err());
    }

    #[test]
    fn test_validator_config_emission_validation() {
        let mut config = ValidatorConfig::default();

        // Modify emission config to be invalid
        config.emission.burn_percentage = 150.0; // Invalid

        // Should fail validation
        assert!(config.validate().is_err());

        // Set valid burn percentage and GPU allocations
        config.emission.burn_percentage = 10.0;
        config
            .emission
            .gpu_allocations
            .insert("A100".to_string(), GpuAllocation::new(50.0));
        config
            .emission
            .gpu_allocations
            .insert("H100".to_string(), GpuAllocation::new(50.0));
        // Disable binary validation since we don't have the binaries in test
        config.verification.binary_validation.enabled = false;
        assert!(config.validate().is_ok());
    }

    #[test]
    fn test_validator_config_serialization_with_emission() {
        let mut config = ValidatorConfig::default();
        // Disable binary validation since we don't have the binaries in test
        config.verification.binary_validation.enabled = false;

        // Add valid GPU allocations for testing
        config.emission.burn_percentage = 10.0;
        config
            .emission
            .gpu_allocations
            .insert("A100".to_string(), GpuAllocation::new(8.0));
        config
            .emission
            .gpu_allocations
            .insert("H100".to_string(), GpuAllocation::new(12.0));
        config
            .emission
            .gpu_allocations
            .insert("B200".to_string(), GpuAllocation::new(80.0));

        // Test TOML serialization includes emission config
        let toml_str = toml::to_string(&config).expect("Failed to serialize to TOML");
        assert!(toml_str.contains("[emission]"));
        assert!(toml_str.contains("burn_percentage"));
        // Check that GPU allocations are present in the TOML
        assert!(toml_str.contains("gpu_allocations"));

        // Test deserialization
        let deserialized: ValidatorConfig =
            toml::from_str(&toml_str).expect("Failed to deserialize from TOML");

        assert_eq!(
            config.emission.burn_percentage,
            deserialized.emission.burn_percentage
        );
        assert_eq!(
            config.emission.gpu_allocations,
            deserialized.emission.gpu_allocations
        );

        // Verify deserialized config is valid
        assert!(deserialized.validate().is_ok());
    }
}
