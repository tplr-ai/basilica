//! Configuration module for the Basilica API gateway

mod cache;
mod rate_limit;
mod server;

pub use cache::{CacheBackend, CacheConfig};
pub use rate_limit::{RateLimitBackend, RateLimitConfig};
pub use server::ServerConfig;

use basilica_common::config::BittensorConfig;
use basilica_common::ConfigurationError as ConfigError;
use figment::{
    providers::{Env, Format, Serialized, Toml},
    Figment,
};
use serde::{Deserialize, Serialize};
use std::path::PathBuf;
use std::time::Duration;

/// Rental health check interval in seconds
const RENTAL_HEALTH_CHECK_INTERVAL_SECS: u64 = 5;

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

    /// Validator hotkey to connect to (SS58 address) - REQUIRED
    pub validator_hotkey: String,
}

impl Default for BittensorIntegrationConfig {
    fn default() -> Self {
        Self {
            network: "finney".to_string(),
            netuid: 42,
            chain_endpoint: None,
            discovery_interval: 60,
            validator_hotkey: String::new(), // Must be provided in config
        }
    }
}

/// Database configuration
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct DatabaseConfig {
    /// Database URL (e.g., "sqlite:basilica-api.db" or "postgres://user:pass@host/db")
    pub url: String,

    /// Maximum number of connections in the pool
    pub max_connections: u32,
}

impl Default for DatabaseConfig {
    fn default() -> Self {
        Self {
            url: "postgres://basilica:dev@localhost:5432/basilica".to_string(),
            max_connections: 5,
        }
    }
}

/// Main configuration structure for the Basilica API
#[derive(Debug, Clone, Serialize, Deserialize, Default)]
pub struct Config {
    /// Server configuration
    pub server: ServerConfig,

    /// Bittensor network configuration
    pub bittensor: BittensorIntegrationConfig,

    /// Cache configuration
    pub cache: CacheConfig,

    /// Rate limiting configuration
    pub rate_limit: RateLimitConfig,

    /// Database configuration
    pub database: DatabaseConfig,
}

impl Config {
    /// Load configuration from file and environment
    pub fn load(path_override: Option<PathBuf>) -> Result<Self, ConfigError> {
        let default_config = Config::default();
        let mut figment = Figment::from(Serialized::defaults(default_config));

        if let Some(path) = path_override {
            if path.exists() {
                figment = figment.merge(Toml::file(&path));
            }
        } else {
            let default_path = PathBuf::from("basilica-api.toml");
            if default_path.exists() {
                figment = figment.merge(Toml::file(default_path));
            }
        }

        figment = figment.merge(Env::prefixed("BASILICA_API_").split("__"));

        figment.extract().map_err(|e| ConfigError::ParseError {
            details: e.to_string(),
        })
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
        Duration::from_secs(30) // Default 30 seconds
    }

    /// Get discovery interval as Duration
    pub fn discovery_interval(&self) -> Duration {
        Duration::from_secs(self.bittensor.discovery_interval)
    }

    /// Get connection timeout as Duration
    pub fn connection_timeout(&self) -> Duration {
        Duration::from_secs(10) // Default 10 seconds
    }

    /// Get validator timeout as Duration
    pub fn validator_timeout(&self) -> Duration {
        Duration::from_secs(30) // Default 30 seconds
    }

    /// Get rental health check interval as Duration
    pub fn rental_health_check_interval(&self) -> Duration {
        Duration::from_secs(RENTAL_HEALTH_CHECK_INTERVAL_SECS)
    }

    /// Create BittensorConfig from our configuration
    pub fn to_bittensor_config(&self) -> BittensorConfig {
        BittensorConfig {
            network: self.bittensor.network.clone(),
            netuid: self.bittensor.netuid,
            chain_endpoint: self.bittensor.chain_endpoint.clone(),
            wallet_name: "default".to_string(),
            hotkey_name: "default".to_string(),
            weight_interval_secs: 300, // 5 minutes default
            ..Default::default()
        }
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
    fn test_bittensor_config_conversion() {
        let config = Config::default();
        let bt_config = config.to_bittensor_config();

        assert_eq!(bt_config.network, config.bittensor.network);
        assert_eq!(bt_config.netuid, config.bittensor.netuid);
        assert_eq!(bt_config.wallet_name, "default");
    }
}
