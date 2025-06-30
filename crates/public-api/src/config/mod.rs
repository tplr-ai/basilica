//! Configuration module for the Public API gateway

mod auth;
mod cache;
mod discovery;
mod load_balancer;
mod rate_limit;
mod server;
mod telemetry;

pub use auth::AuthConfig;
pub use cache::{CacheBackend, CacheConfig};
pub use discovery::DiscoveryConfig;
pub use load_balancer::{LoadBalancerConfig, LoadBalancerStrategy};
pub use rate_limit::{RateLimitBackend, RateLimitConfig};
pub use server::ServerConfig;
pub use telemetry::TelemetryConfig;

use common::config::{BittensorConfig, ConfigLoader};
use common::ConfigurationError as ConfigError;
use figment::{
    providers::{Env, Format, Serialized, Toml},
    Figment,
};
use serde::{Deserialize, Serialize};
use std::path::{Path, PathBuf};
use std::time::Duration;

/// Bittensor integration configuration
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct BittensorIntegrationConfig {
    /// Network name (e.g., "finney", "test", "local")
    pub network: String,

    /// Subnet UID for validator discovery
    pub netuid: u16,

    /// Chain endpoint URL (optional, uses default for network if not specified)
    pub chain_endpoint: Option<String>,

    /// Validator discovery interval in seconds
    pub discovery_interval: u64,

    /// Minimum validator score to consider
    pub min_validator_score: f64,
}

impl Default for BittensorIntegrationConfig {
    fn default() -> Self {
        Self {
            network: "finney".to_string(),
            netuid: 42,
            chain_endpoint: None,
            discovery_interval: 60,
            min_validator_score: 0.5,
        }
    }
}

/// Main configuration structure for the Public API
#[derive(Debug, Clone, Serialize, Deserialize, Default)]
pub struct Config {
    /// Server configuration
    pub server: ServerConfig,

    /// Bittensor network configuration
    pub bittensor: BittensorIntegrationConfig,

    /// Load balancer configuration
    pub load_balancer: LoadBalancerConfig,

    /// Cache configuration
    pub cache: CacheConfig,

    /// Rate limiting configuration
    pub rate_limit: RateLimitConfig,

    /// Authentication configuration
    pub auth: AuthConfig,

    /// Telemetry configuration
    pub telemetry: TelemetryConfig,

    /// Validator discovery configuration
    pub discovery: DiscoveryConfig,
}

impl Config {
    /// Load configuration from file and environment
    pub fn load(config_path: Option<&Path>) -> Result<Self, ConfigError> {
        match config_path {
            Some(path) => <Config as ConfigLoader<Config>>::load_from_file(path),
            None => <Config as ConfigLoader<Config>>::load(None),
        }
    }

    /// Generate example configuration file
    pub fn generate_example() -> Result<String, ConfigError> {
        let config = Self::default();
        toml::to_string_pretty(&config).map_err(|e| ConfigError::ParseError {
            details: format!("Failed to serialize config: {e}"),
        })
    }

    /// Get request timeout as Duration
    pub fn request_timeout(&self) -> Duration {
        Duration::from_secs(self.server.request_timeout)
    }

    /// Get health check interval as Duration
    pub fn health_check_interval(&self) -> Duration {
        Duration::from_secs(self.load_balancer.health_check_interval)
    }

    /// Get discovery interval as Duration
    pub fn discovery_interval(&self) -> Duration {
        Duration::from_secs(self.bittensor.discovery_interval)
    }

    /// Get connection timeout as Duration
    pub fn connection_timeout(&self) -> Duration {
        Duration::from_secs(self.load_balancer.connection_timeout)
    }

    /// Get validator timeout as Duration
    pub fn validator_timeout(&self) -> Duration {
        Duration::from_secs(self.discovery.validator_timeout)
    }

    /// Create BittensorConfig from our configuration
    pub fn to_bittensor_config(&self) -> BittensorConfig {
        BittensorConfig {
            network: self.bittensor.network.clone(),
            netuid: self.bittensor.netuid,
            chain_endpoint: self.bittensor.chain_endpoint.clone(),
            wallet_name: "public-api".to_string(),
            hotkey_name: "default".to_string(),
            weight_interval_secs: 300, // 5 minutes default
        }
    }
}

impl ConfigLoader<Config> for Config {
    fn load(path: Option<PathBuf>) -> Result<Config, ConfigError> {
        let figment = match path {
            Some(p) => Figment::from(Serialized::defaults(Config::default()))
                .merge(Toml::file(p))
                .merge(Env::prefixed("PUBLIC_API_").split("__")),
            None => Figment::from(Serialized::defaults(Config::default()))
                .merge(Toml::file("public-api.toml"))
                .merge(Env::prefixed("PUBLIC_API_").split("__")),
        };

        figment.extract().map_err(|e| ConfigError::ParseError {
            details: e.to_string(),
        })
    }

    fn load_from_file(path: &Path) -> Result<Config, ConfigError> {
        let figment = Figment::from(Serialized::defaults(Config::default()))
            .merge(Toml::file(path))
            .merge(Env::prefixed("PUBLIC_API_").split("__"));

        figment.extract().map_err(|e| ConfigError::ParseError {
            details: e.to_string(),
        })
    }

    fn apply_env_overrides(config: &mut Config, prefix: &str) -> Result<(), ConfigError> {
        let figment = Figment::from(Serialized::defaults(config.clone()))
            .merge(Env::prefixed(prefix).split("__"));

        *config = figment.extract().map_err(|e| ConfigError::ParseError {
            details: e.to_string(),
        })?;

        Ok(())
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_default_config() {
        let config = Config::default();
        assert_eq!(config.server.bind_address.port(), 8000);
        assert_eq!(config.bittensor.network, "finney");
        assert_eq!(config.bittensor.netuid, 42);
    }

    #[test]
    fn test_config_serialization() {
        let config = Config::default();
        let serialized = toml::to_string(&config).unwrap();
        let deserialized: Config = toml::from_str(&serialized).unwrap();

        assert_eq!(config.server.bind_address, deserialized.server.bind_address);
        assert_eq!(config.bittensor.network, deserialized.bittensor.network);
    }

    #[test]
    fn test_duration_conversions() {
        let config = Config::default();
        assert_eq!(config.request_timeout(), Duration::from_secs(30));
        assert_eq!(config.health_check_interval(), Duration::from_secs(30));
        assert_eq!(config.discovery_interval(), Duration::from_secs(60));
    }

    #[test]
    fn test_bittensor_config_conversion() {
        let config = Config::default();
        let bt_config = config.to_bittensor_config();

        assert_eq!(bt_config.network, config.bittensor.network);
        assert_eq!(bt_config.netuid, config.bittensor.netuid);
        assert_eq!(bt_config.wallet_name, "public-api");
    }
}
