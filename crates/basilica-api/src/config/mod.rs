//! Configuration module for the Basilica API gateway

mod cache;
mod deployment;
mod dns;
mod rate_limit;
mod server;

pub use cache::{CacheBackend, CacheConfig};
pub use deployment::DeploymentConfig;
pub use dns::DnsConfig;
pub use rate_limit::{RateLimitBackend, RateLimitConfig};
pub use server::ServerConfig;

use crate::ssh::K3sSshConfig;
use crate::wireguard::WireGuardServerConfig;

use basilica_common::config::{types::MetricsConfig, BittensorConfig};
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

/// Node token cleanup interval in seconds
const NODE_TOKEN_CLEANUP_INTERVAL_SECS: u64 = 3600;

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

/// Payments service configuration
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct PaymentsServiceConfig {
    /// Enable payments service integration
    pub enabled: bool,

    /// Payments service gRPC endpoint
    pub endpoint: String,
}

impl Default for PaymentsServiceConfig {
    fn default() -> Self {
        Self {
            enabled: true,
            endpoint: "http://localhost:50061".to_string(),
        }
    }
}

/// Billing service configuration
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct BillingServiceConfig {
    /// Enable billing service integration
    pub enabled: bool,

    /// Billing service gRPC endpoint
    pub endpoint: String,
}

impl Default for BillingServiceConfig {
    fn default() -> Self {
        Self {
            enabled: true,
            endpoint: "http://localhost:50051".to_string(),
        }
    }
}

/// GPU Aggregator configuration (for secure cloud)
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct AggregatorConfig {
    /// Cache TTL for GPU offerings in seconds
    #[serde(default = "default_aggregator_ttl")]
    pub ttl_seconds: u64,

    /// GPU provider configurations
    pub providers: AggregatorProvidersConfig,
}

fn default_aggregator_ttl() -> u64 {
    45
}

impl Default for AggregatorConfig {
    fn default() -> Self {
        Self {
            ttl_seconds: default_aggregator_ttl(),
            providers: AggregatorProvidersConfig::default(),
        }
    }
}

/// GPU providers configuration
#[derive(Debug, Clone, Serialize, Deserialize, Default)]
pub struct AggregatorProvidersConfig {
    #[serde(default)]
    pub datacrunch: basilica_aggregator::config::ProviderConfig,
    #[serde(default)]
    pub hyperstack: basilica_aggregator::config::ProviderConfig,
    #[serde(default)]
    pub lambda: basilica_aggregator::config::ProviderConfig,
    #[serde(default)]
    pub hydrahost: basilica_aggregator::config::ProviderConfig,
}

/// Pricing configuration for marketplace markups
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct PricingConfig {
    /// Markup percentage for community cloud rentals
    #[serde(default = "default_community_markup")]
    pub community_markup_percent: f64,

    /// Markup percentage for secure cloud rentals
    #[serde(default = "default_secure_cloud_markup")]
    pub secure_cloud_markup_percent: f64,
}

fn default_community_markup() -> f64 {
    10.0
}

fn default_secure_cloud_markup() -> f64 {
    10.0
}

impl Default for PricingConfig {
    fn default() -> Self {
        Self {
            community_markup_percent: default_community_markup(),
            secure_cloud_markup_percent: default_secure_cloud_markup(),
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

    /// Payments service configuration
    pub payments: PaymentsServiceConfig,

    /// Billing service configuration
    pub billing: BillingServiceConfig,

    /// Deployment configuration
    #[serde(default)]
    pub deployment: DeploymentConfig,

    /// DNS configuration for public deployments
    #[serde(default)]
    pub dns: DnsConfig,

    /// Metrics configuration
    #[serde(default)]
    pub metrics: MetricsConfig,

    /// GPU Aggregator configuration (secure cloud)
    #[serde(default)]
    pub aggregator: AggregatorConfig,

    /// Pricing configuration (marketplace markups)
    #[serde(default)]
    pub pricing: PricingConfig,

    /// K3s SSH configuration for token generation
    #[serde(default)]
    pub k3s_ssh: K3sSshConfig,

    /// WireGuard VPN configuration for remote GPU nodes
    #[serde(default)]
    pub wireguard: WireGuardServerConfig,
}

impl Config {
    /// Load configuration from file and environment
    pub fn load(path_override: Option<PathBuf>) -> Result<Self, ConfigError> {
        let default_config = Config::default();
        let mut figment = Figment::from(Serialized::defaults(default_config));

        if let Some(path) = path_override {
            if path.exists() {
                figment = figment.merge(Toml::file(&path));
            } else {
                return Err(ConfigError::FileNotFound {
                    path: path.display().to_string(),
                });
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

    /// Get node token cleanup interval as Duration
    pub fn node_token_cleanup_interval(&self) -> Duration {
        Duration::from_secs(NODE_TOKEN_CLEANUP_INTERVAL_SECS)
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
            read_only: true,           // API only needs read-only access for metagraph queries
            ..Default::default()
        }
    }

    /// Create aggregator config from API config
    pub fn to_aggregator_config(&self) -> basilica_aggregator::AggregatorConfig {
        basilica_aggregator::AggregatorConfig {
            server: basilica_aggregator::config::ServerConfig {
                host: "0.0.0.0".to_string(), // Not used when embedded in API
                port: 0,                     // Not used when embedded in API
            },
            cache: basilica_aggregator::config::CacheConfig {
                ttl_seconds: self.aggregator.ttl_seconds,
            },
            providers: basilica_aggregator::config::ProvidersConfig {
                datacrunch: self.aggregator.providers.datacrunch.clone(),
                hyperstack: self.aggregator.providers.hyperstack.clone(),
                lambda: self.aggregator.providers.lambda.clone(),
                hydrahost: self.aggregator.providers.hydrahost.clone(),
            },
            database: basilica_aggregator::config::DatabaseConfig {
                path: self.database.url.clone(), // Uses PostgreSQL URL from API config
            },
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

    #[test]
    fn test_billing_config_defaults() {
        let config = BillingServiceConfig::default();
        assert!(config.enabled);
        assert_eq!(config.endpoint, "http://localhost:50051");
    }

    #[test]
    fn test_billing_config_custom_endpoint() {
        let toml_str = r#"
            enabled = true
            endpoint = "https://billing.basilica.ai:50051"
        "#;
        let config: BillingServiceConfig = toml::from_str(toml_str).unwrap();
        assert!(config.enabled);
        assert_eq!(config.endpoint, "https://billing.basilica.ai:50051");
    }

    #[test]
    fn test_billing_config_disabled() {
        let toml_str = r#"
            enabled = false
            endpoint = "http://localhost:50051"
        "#;
        let config: BillingServiceConfig = toml::from_str(toml_str).unwrap();
        assert!(!config.enabled);
    }
}
