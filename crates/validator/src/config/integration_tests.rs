#[cfg(test)]
mod tests {
    use crate::config::ValidatorConfig;
    use common::config::ConfigValidation;

    #[test]
    fn test_validator_config_includes_emission_config() {
        let config = ValidatorConfig::default();

        // Verify emission config is included
        assert_eq!(config.emission.burn_percentage, 0.0);
        assert_eq!(config.emission.burn_uid, 0);
        assert_eq!(config.emission.weight_set_interval_blocks, 360);
        assert_eq!(config.emission.min_miners_per_category, 1);
        assert_eq!(config.emission.gpu_allocations.len(), 2);

        // Verify the config validates
        assert!(config.validate().is_ok());
    }

    #[test]
    fn test_validator_config_emission_validation() {
        let mut config = ValidatorConfig::default();

        // Modify emission config to be invalid
        config.emission.burn_percentage = 150.0; // Invalid

        // Should fail validation
        assert!(config.validate().is_err());

        // Fix the config
        config.emission.burn_percentage = 10.0;
        assert!(config.validate().is_ok());
    }

    #[test]
    fn test_validator_config_serialization_with_emission() {
        let config = ValidatorConfig::default();

        // Test TOML serialization includes emission config
        let toml_str = toml::to_string(&config).expect("Failed to serialize to TOML");
        assert!(toml_str.contains("[emission]"));
        assert!(toml_str.contains("burn_percentage"));
        assert!(toml_str.contains("[emission.gpu_allocations]"));

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
