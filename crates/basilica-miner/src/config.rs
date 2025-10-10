//! # Miner Configuration
//!
//! Configuration structures and validation for the Basilca Miner.

use anyhow::Result;
use serde::{Deserialize, Serialize};
use serde_with::{serde_as, DurationSeconds};
use std::fs;
use std::path::{Path, PathBuf};
use std::time::Duration;

use basilica_common::config::{
    loader, BittensorConfig, ConfigValidation, DatabaseConfig, MetricsConfig,
};
use basilica_common::error::ConfigurationError;

use crate::node_manager::NodeConfig;

/// Main miner configuration
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct MinerConfig {
    /// Bittensor network configuration
    pub bittensor: MinerBittensorConfig,

    /// Database configuration
    pub database: DatabaseConfig,

    /// Metrics configuration
    pub metrics: MetricsConfig,

    /// Validator communications configuration
    pub validator_comms: ValidatorCommsConfig,

    /// Node management configuration
    pub node_management: NodeManagementConfig,

    /// Security configuration
    pub security: SecurityConfig,

    /// SSH session configuration for validator access
    pub ssh_session: NodeSshConfig,

    /// Advertised address configuration
    #[serde(default)]
    pub advertised_addresses: MinerAdvertisedAddresses,

    /// Validator assignment configuration
    #[serde(default)]
    pub validator_assignment: ValidatorAssignmentConfig,
}

/// Miner-specific Bittensor configuration
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct MinerBittensorConfig {
    /// Common Bittensor configuration
    #[serde(flatten)]
    pub common: BittensorConfig,

    /// Coldkey name for wallet operations
    pub coldkey_name: String,

    /// Axon server port for Bittensor network
    pub axon_port: u16,

    /// External IP address for the axon
    pub external_ip: Option<String>,

    /// Maximum number of UIDs to set weights for
    pub max_weight_uids: u16,
}

/// Validator communications configuration
#[serde_as]
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct ValidatorCommsConfig {
    /// Host to bind the gRPC server to
    pub host: String,

    /// Port to bind the gRPC server to
    pub port: u16,

    /// Request timeout for validator calls
    #[serde_as(as = "DurationSeconds<u64>")]
    pub request_timeout: Duration,
}

/// Node management configuration
#[serde_as]
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct NodeManagementConfig {
    /// Static list of nodes managed by this miner
    pub nodes: Vec<NodeConfig>,

    /// Health check interval for nodes
    #[serde_as(as = "DurationSeconds<u64>")]
    pub health_check_interval: Duration,

    /// Timeout for node health checks
    #[serde_as(as = "DurationSeconds<u64>")]
    pub health_check_timeout: Duration,

    /// Maximum retry attempts for failed operations
    pub max_retry_attempts: u32,

    /// Enable automatic status recovery
    pub auto_recovery: bool,
}

/// Security configuration
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct SecurityConfig {
    /// Enable request signing verification
    pub verify_signatures: bool,

    /// Ethereum private key for collateral contract
    pub private_key_file: Option<PathBuf>,
}

/// SSH configuration for node access by validators
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct NodeSshConfig {
    /// Path to miner's SSH key for node access
    pub miner_node_key_path: PathBuf,

    /// Default username for node SSH
    pub default_node_username: String,
}

/// Validator assignment configuration
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct ValidatorAssignmentConfig {
    /// Assignment strategy to use ("highest_stake", "fixed_assignment")
    #[serde(default = "default_strategy")]
    pub strategy: String,

    /// Specific validator hotkey to assign nodes to (required for "fixed_assignment" strategy)
    pub validator_hotkey: Option<String>,
}

impl Default for ValidatorAssignmentConfig {
    fn default() -> Self {
        Self {
            strategy: default_strategy(),
            validator_hotkey: None,
        }
    }
}

/// Advertised address configuration for miner services
#[derive(Debug, Clone, Serialize, Deserialize, Default)]
pub struct MinerAdvertisedAddresses {
    /// Advertised gRPC endpoint for validator communication
    pub grpc_endpoint: Option<String>,
    /// Advertised discovery endpoint for miner-to-miner communication
    pub discovery_endpoint: Option<String>,
    /// Override axon endpoint for Bittensor chain registration
    pub axon_endpoint: Option<String>,
    /// Advertised metrics endpoint
    pub metrics_endpoint: Option<String>,
}

impl Default for MinerConfig {
    fn default() -> Self {
        Self {
            bittensor: MinerBittensorConfig::default(),
            database: DatabaseConfig {
                url: "sqlite:./data/miner.db".to_string(),
                ..Default::default()
            },
            metrics: MetricsConfig::default(),
            validator_comms: ValidatorCommsConfig::default(),
            node_management: NodeManagementConfig::default(),
            security: SecurityConfig::default(),
            ssh_session: NodeSshConfig::default(),
            advertised_addresses: MinerAdvertisedAddresses::default(),
            validator_assignment: ValidatorAssignmentConfig::default(),
        }
    }
}

fn default_strategy() -> String {
    "highest_stake".to_string()
}

impl Default for NodeSshConfig {
    fn default() -> Self {
        Self {
            miner_node_key_path: PathBuf::from("~/.ssh/miner_node_key"),
            default_node_username: "node".to_string(),
        }
    }
}

impl Default for MinerBittensorConfig {
    fn default() -> Self {
        Self {
            common: BittensorConfig {
                wallet_name: "miner".to_string(),
                hotkey_name: "default".to_string(),
                network: "finney".to_string(),
                netuid: 39,                // Basilca subnet ID
                chain_endpoint: None,      // Will be auto-detected based on network
                weight_interval_secs: 300, // 5 minutes
                ..Default::default()
            },
            coldkey_name: "default".to_string(),
            axon_port: 8091,
            external_ip: None,
            max_weight_uids: 256,
        }
    }
}

impl Default for ValidatorCommsConfig {
    fn default() -> Self {
        Self {
            host: "0.0.0.0".to_string(),
            port: 50051,
            request_timeout: Duration::from_secs(30),
        }
    }
}

impl Default for NodeManagementConfig {
    fn default() -> Self {
        Self {
            nodes: vec![],
            health_check_interval: Duration::from_secs(60),
            health_check_timeout: Duration::from_secs(10),
            max_retry_attempts: 3,
            auto_recovery: true,
        }
    }
}

impl Default for SecurityConfig {
    fn default() -> Self {
        Self {
            verify_signatures: true,
            private_key_file: None,
        }
    }
}

impl ConfigValidation for MinerConfig {
    type Error = ConfigurationError;

    fn validate(&self) -> Result<(), Self::Error> {
        // Validate common configs using their validation
        self.database.validate()?;

        // Validate Bittensor configuration - delegate to common validation
        self.bittensor.common.validate()?;

        // Validate miner-specific fields
        if self.bittensor.common.netuid == 0 {
            return Err(ConfigurationError::InvalidValue {
                key: "bittensor.common.netuid".to_string(),
                value: self.bittensor.common.netuid.to_string(),
                reason: "Invalid netuid: must be greater than 0".to_string(),
            });
        }

        if self.bittensor.axon_port == 0 {
            return Err(ConfigurationError::InvalidValue {
                key: "bittensor.axon_port".to_string(),
                value: self.bittensor.axon_port.to_string(),
                reason: "Invalid axon port: must be greater than 0".to_string(),
            });
        }

        // Validate each node config
        for (idx, node) in self.node_management.nodes.iter().enumerate() {
            if node.host.is_empty() {
                return Err(ConfigurationError::InvalidValue {
                    key: format!("node_management.nodes[{}].host", idx),
                    value: node.host.clone(),
                    reason: "Node host cannot be empty".to_string(),
                });
            }
            if node.username.is_empty() {
                return Err(ConfigurationError::InvalidValue {
                    key: format!("node_management.nodes[{}].username", idx),
                    value: node.username.clone(),
                    reason: "Node username cannot be empty".to_string(),
                });
            }
        }

        // Validate validator assignment configuration
        if self.validator_assignment.strategy == "fixed_assignment"
            && self.validator_assignment.validator_hotkey.is_none()
        {
            return Err(ConfigurationError::InvalidValue {
                key: "validator_assignment.validator_hotkey".to_string(),
                value: "None".to_string(),
                reason: "validator_hotkey is required when using 'fixed_assignment' strategy"
                    .to_string(),
            });
        }

        Ok(())
    }

    fn warnings(&self) -> Vec<String> {
        vec![]
    }
}

impl MinerConfig {
    /// Load configuration using common loader
    pub fn load() -> Result<Self> {
        Ok(loader::load_config::<Self>()?)
    }

    /// Load configuration from specific file
    pub fn load_from_file(path: &Path) -> Result<Self> {
        Ok(loader::load_from_file::<Self>(path)?)
    }

    /// Get the advertised gRPC endpoint for validators
    pub fn get_advertised_grpc_endpoint(&self) -> String {
        self.advertised_addresses
            .grpc_endpoint
            .as_ref()
            .cloned()
            .unwrap_or_else(|| {
                format!(
                    "http://{}:{}",
                    self.validator_comms.host, self.validator_comms.port
                )
            })
    }

    /// Get the advertised axon endpoint for Bittensor registration
    pub fn get_advertised_axon_endpoint(&self) -> String {
        if let Some(endpoint) = &self.advertised_addresses.axon_endpoint {
            endpoint.clone()
        } else if let Some(external_ip) = &self.bittensor.external_ip {
            format!("http://{}:{}", external_ip, self.bittensor.axon_port)
        } else {
            format!(
                "http://{}:{}",
                self.validator_comms.host, self.bittensor.axon_port
            )
        }
    }

    /// Get the advertised metrics endpoint
    pub fn get_advertised_metrics_endpoint(&self) -> String {
        self.advertised_addresses
            .metrics_endpoint
            .as_ref()
            .cloned()
            .unwrap_or_else(|| {
                format!(
                    "http://{}:{}",
                    self.validator_comms.host,
                    self.metrics
                        .prometheus
                        .as_ref()
                        .map(|p| p.port)
                        .unwrap_or(9090)
                )
            })
    }

    /// Validate all advertised address configurations
    pub fn validate_advertised_addresses(&self) -> Result<()> {
        if let Some(ref grpc_endpoint) = self.advertised_addresses.grpc_endpoint {
            if !grpc_endpoint.starts_with("http://") && !grpc_endpoint.starts_with("https://") {
                return Err(anyhow::anyhow!(
                    "gRPC endpoint must start with http:// or https://"
                ));
            }
        }

        if let Some(ref axon_endpoint) = self.advertised_addresses.axon_endpoint {
            if !axon_endpoint.starts_with("http://") && !axon_endpoint.starts_with("https://") {
                return Err(anyhow::anyhow!(
                    "Axon endpoint must start with http:// or https://"
                ));
            }
        }

        Ok(())
    }
}

impl SecurityConfig {
    pub fn get_private_key(&self) -> Result<String, anyhow::Error> {
        match self.private_key_file {
            Some(ref path) => {
                if !Path::new(path).exists() {
                    Err(anyhow::anyhow!("private key file does not exist"))
                } else {
                    match fs::read_to_string(path) {
                        Ok(private_key) => Ok(private_key.trim().to_string()),
                        Err(e) => Err(anyhow::anyhow!("Failed to read private key file: {}", e)),
                    }
                }
            }
            None => Err(anyhow::anyhow!("private_key_file config is required")),
        }
    }
}

// TODO: Implement the following for production readiness:
// 1. Configuration hot-reloading without service restart
// 2. Environment-specific configuration overlays
// 3. Secret management integration (e.g., HashiCorp Vault)
// 4. Configuration validation with detailed error messages
// 5. Configuration migration utilities for version upgrades
// 6. Dynamic reconfiguration of rate limits and timeouts
// 7. Configuration backup and restore functionality
// 8. Encrypted configuration values for sensitive data
