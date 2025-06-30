//! # Validator Configuration
//!
//! Layered configuration management for the Basilca Validator.

use anyhow::Result;
use serde::{Deserialize, Serialize};
use std::time::Duration;

use common::config::{
    loader, BittensorConfig, ConfigValidation, DatabaseConfig, LoggingConfig, MetricsConfig,
    ServerConfig,
};
use common::error::ConfigurationError;

/// Validator-specific Bittensor configuration
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct ValidatorBittensorConfig {
    /// Common Bittensor configuration
    #[serde(flatten)]
    pub common: BittensorConfig,

    /// Axon server port for Bittensor network
    pub axon_port: u16,

    /// External IP address for the axon
    pub external_ip: Option<String>,
}

#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct ValidatorConfig {
    /// Database configuration
    pub database: DatabaseConfig,

    /// Server configuration for API
    pub server: ServerConfig,

    /// Logging configuration
    pub logging: LoggingConfig,

    /// Metrics configuration
    pub metrics: MetricsConfig,

    /// Bittensor configuration
    pub bittensor: ValidatorBittensorConfig,

    /// Verification configuration
    pub verification: VerificationConfig,

    /// Storage configuration
    pub storage: StorageConfig,

    /// API-specific configuration
    pub api: ApiConfig,
}

#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct VerificationConfig {
    /// How often to verify miners
    pub verification_interval: Duration,
    /// Maximum concurrent verifications
    pub max_concurrent_verifications: usize,
    /// Challenge timeout
    pub challenge_timeout: Duration,
    /// Minimum score threshold for miners
    pub min_score_threshold: f64,
    /// Minimum stake threshold in TAO for miners to be verified
    pub min_stake_threshold: f64,
    /// Maximum number of miners to verify per round
    pub max_miners_per_round: usize,
    /// Minimum interval between verifications of the same miner
    pub min_verification_interval: Duration,
    /// Network ID for the subnet
    pub netuid: u16,
}

#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct StorageConfig {
    /// Storage directory path
    pub data_dir: String,
}

#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct ApiConfig {
    /// API key for external authentication
    pub api_key: Option<String>,
    /// Maximum request body size in bytes
    pub max_body_size: usize,
    /// Bind address for the API server
    pub bind_address: String,
}

impl Default for ValidatorConfig {
    fn default() -> Self {
        Self {
            database: DatabaseConfig {
                url: "sqlite:./data/validator.db".to_string(),
                ..Default::default()
            },
            server: ServerConfig {
                host: "0.0.0.0".to_string(),
                port: 8080,
                ..Default::default()
            },
            logging: LoggingConfig::default(),
            metrics: MetricsConfig::default(),
            bittensor: ValidatorBittensorConfig {
                common: BittensorConfig {
                    wallet_name: "validator".to_string(),
                    hotkey_name: "default".to_string(),
                    network: "finney".to_string(),
                    netuid: 1,
                    chain_endpoint: Some("wss://entrypoint-finney.opentensor.ai:443".to_string()),
                    weight_interval_secs: 300,
                },
                axon_port: 9090,
                external_ip: None,
            },
            verification: VerificationConfig {
                verification_interval: Duration::from_secs(600),
                max_concurrent_verifications: 50,
                challenge_timeout: Duration::from_secs(120),
                min_score_threshold: 0.1,
                min_stake_threshold: 1.0, // 1 TAO minimum
                max_miners_per_round: 20,
                min_verification_interval: Duration::from_secs(1800), // 30 minutes
                netuid: 1,                                            // Default subnet
            },
            storage: StorageConfig {
                data_dir: "./data".to_string(),
            },
            api: ApiConfig {
                api_key: None,
                max_body_size: 1024 * 1024, // 1MB
                bind_address: "0.0.0.0:8080".to_string(),
            },
        }
    }
}

impl ConfigValidation for ValidatorConfig {
    type Error = ConfigurationError;

    fn validate(&self) -> Result<(), Self::Error> {
        // Validate common configs using their validation
        self.database.validate()?;
        self.server.validate()?;

        // Validate validator-specific config
        if self.bittensor.common.netuid == 0 {
            return Err(ConfigurationError::InvalidValue {
                key: "bittensor.netuid".to_string(),
                value: self.bittensor.common.netuid.to_string(),
                reason: "Netuid must be greater than 0".to_string(),
            });
        }

        if self.bittensor.axon_port == 0 {
            return Err(ConfigurationError::InvalidValue {
                key: "bittensor.axon_port".to_string(),
                value: self.bittensor.axon_port.to_string(),
                reason: "Axon port must be greater than 0".to_string(),
            });
        }

        if self.verification.max_concurrent_verifications == 0 {
            return Err(ConfigurationError::InvalidValue {
                key: "verification.max_concurrent_verifications".to_string(),
                value: self.verification.max_concurrent_verifications.to_string(),
                reason: "Must allow at least 1 concurrent verification".to_string(),
            });
        }

        if self.storage.data_dir.is_empty() {
            return Err(ConfigurationError::InvalidValue {
                key: "storage.data_dir".to_string(),
                value: self.storage.data_dir.clone(),
                reason: "Storage data directory cannot be empty".to_string(),
            });
        }

        Ok(())
    }

    fn warnings(&self) -> Vec<String> {
        let mut warnings = Vec::new();

        if self.api.api_key.is_none() {
            warnings
                .push("No API key configured - external API access will be disabled".to_string());
        }

        if self.verification.min_score_threshold < 0.1 {
            warnings.push("Very low minimum score threshold may allow poor performers".to_string());
        }

        warnings
    }
}

impl ValidatorConfig {
    /// Load configuration using common loader with environment prefix
    pub fn load() -> Result<Self> {
        Ok(loader::load_config::<Self>()?)
    }

    /// Load configuration from specific file
    pub fn load_from_file(path: &std::path::Path) -> Result<Self> {
        Ok(loader::load_from_file::<Self>(path)?)
    }
}

// TODO: Add configuration hot-reloading capabilities
// TODO: Add configuration encryption for sensitive values
// TODO: Add configuration templating with variable substitution
// TODO: Add configuration schema validation
// TODO: Add configuration migration between versions
