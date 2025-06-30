//! Configuration validation implementation

use common::config::ConfigValidation;
use common::error::ConfigurationError;

use super::{DockerConfigValidation, ExecutorConfig, SystemConfigValidation};

impl ConfigValidation for ExecutorConfig {
    type Error = ConfigurationError;

    fn validate(&self) -> Result<(), Self::Error> {
        // Validate common configs using their validation
        self.server.validate()?;

        // Validate system config
        self.system
            .validate_usage_limits()
            .map_err(|msg| ConfigurationError::InvalidValue {
                key: "system".to_string(),
                value: "usage_limits".to_string(),
                reason: msg,
            })?;

        self.system.validate_monitoring_settings().map_err(|msg| {
            ConfigurationError::InvalidValue {
                key: "system".to_string(),
                value: "monitoring_settings".to_string(),
                reason: msg,
            }
        })?;

        // Validate Docker config
        self.docker
            .validate_resource_limits()
            .map_err(|msg| ConfigurationError::InvalidValue {
                key: "docker".to_string(),
                value: "resource_limits".to_string(),
                reason: msg,
            })?;

        self.docker.validate_network_settings().map_err(|msg| {
            ConfigurationError::InvalidValue {
                key: "docker".to_string(),
                value: "network_settings".to_string(),
                reason: msg,
            }
        })?;

        self.docker.validate_registry_settings().map_err(|msg| {
            ConfigurationError::InvalidValue {
                key: "docker".to_string(),
                value: "registry_settings".to_string(),
                reason: msg,
            }
        })?;

        // Validate validator configuration if enabled
        if self.validator.enabled {
            // Basic validation for validator config
            // SSH restrictions are now managed via the validation_session service
        }

        Ok(())
    }

    fn warnings(&self) -> Vec<String> {
        let mut warnings = Vec::new();

        // Collect warnings from all config sections
        warnings.extend(self.system.usage_warnings());
        warnings.extend(self.docker.docker_warnings());

        // Add validator warnings if enabled
        if self.validator.enabled {
            // Warn about IP whitelist
            if self.validator.access_config.ip_whitelist.is_empty() {
                warnings.push(
                    "No IP whitelist configured for validator access - all IPs will be allowed"
                        .to_string(),
                );
            }
        }

        warnings
    }
}
