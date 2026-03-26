//! # Miner Configuration
//!
//! Configuration structures and validation for the Basilca Miner.

use anyhow::Result;
use serde::{Deserialize, Serialize};
use serde_with::{serde_as, DurationSeconds};
use std::collections::BTreeMap;
use std::fs;
use std::net::IpAddr;
use std::path::{Path, PathBuf};
use std::time::Duration;

use basilica_common::config::{
    loader, BittensorConfig, ConfigValidation, DatabaseConfig, MetricsConfig, DEFAULT_BID_GRPC_PORT,
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

    /// Automatic bidding configuration
    #[serde(default)]
    pub bidding: BiddingConfig,

    /// Port for the validator's bidding gRPC service (default: 50052)
    #[serde(default = "default_bid_grpc_port")]
    pub bid_grpc_port: u16,
}

/// Miner-specific Bittensor configuration
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct MinerBittensorConfig {
    /// Common Bittensor configuration
    #[serde(flatten)]
    pub common: BittensorConfig,

    /// Axon server port for Bittensor network
    pub axon_port: u16,

    /// External IP address for the axon
    pub external_ip: Option<String>,

    /// Maximum number of UIDs to set weights for
    pub max_weight_uids: u16,
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

/// Automatic bidding configuration
///
/// Note: Config files accept prices in dollars (e.g., 2.50 for $2.50/hour),
/// which are converted to cents internally on load.
///
/// BidManager always runs and waits for validator discovery to provide the gRPC endpoint.
/// All GPU categories in your nodes MUST have prices defined in the static strategy.
///
/// TODO: Add floor_prices when dynamic bidding strategies are implemented.
#[derive(Debug, Clone, Serialize, Deserialize, Default)]
pub struct BiddingConfig {
    /// Active bidding strategy (single enum variant)
    #[serde(default)]
    pub strategy: BiddingStrategy,
}

/// Bidding strategy configuration (one active variant)
#[derive(Debug, Clone, Serialize, Deserialize)]
#[serde(rename_all = "snake_case")]
pub enum BiddingStrategy {
    /// Fixed prices by GPU category (in cents)
    Static {
        /// Static prices by GPU category in CENTS (converted from dollars in config)
        /// Config accepts dollars (e.g., 2.50), stored as cents (250)
        /// Every GPU category in your nodes MUST have a price here.
        #[serde(
            default,
            rename = "static_prices",
            deserialize_with = "deserialize_dollars_to_cents"
        )]
        static_prices_cents: std::collections::HashMap<String, u32>,
    },
}

/// Convert dollars (f64) to cents (u32)
fn dollars_to_cents(dollars: f64) -> u32 {
    (dollars * 100.0).round() as u32
}

/// Deserialize a HashMap of dollar values (f64) to cents (u32)
fn deserialize_dollars_to_cents<'de, D>(
    deserializer: D,
) -> Result<std::collections::HashMap<String, u32>, D::Error>
where
    D: serde::Deserializer<'de>,
{
    let dollars: std::collections::HashMap<String, f64> =
        std::collections::HashMap::deserialize(deserializer)?;
    Ok(dollars
        .into_iter()
        .map(|(k, v)| (k, dollars_to_cents(v)))
        .collect())
}

impl Default for BiddingStrategy {
    fn default() -> Self {
        Self::Static {
            static_prices_cents: std::collections::HashMap::new(),
        }
    }
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
            node_management: NodeManagementConfig::default(),
            security: SecurityConfig::default(),
            ssh_session: NodeSshConfig::default(),
            advertised_addresses: MinerAdvertisedAddresses::default(),
            validator_assignment: ValidatorAssignmentConfig::default(),
            bidding: BiddingConfig::default(),
            bid_grpc_port: DEFAULT_BID_GRPC_PORT,
        }
    }
}

fn default_strategy() -> String {
    "highest_stake".to_string()
}

fn default_bid_grpc_port() -> u16 {
    DEFAULT_BID_GRPC_PORT
}

impl Default for NodeSshConfig {
    fn default() -> Self {
        Self {
            miner_node_key_path: PathBuf::from("~/.ssh/miner_node_key"),
            default_node_username: "node".to_string(),
        }
    }
}

/// Expand tilde (~) in path to HOME environment variable
pub fn expand_tilde_in_path(path: &Path) -> PathBuf {
    if path.starts_with("~") {
        if let Ok(home) = std::env::var("HOME") {
            PathBuf::from(path.to_string_lossy().replacen('~', &home, 1))
        } else {
            path.to_path_buf()
        }
    } else {
        path.to_path_buf()
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
            axon_port: 8091,
            external_ip: None,
            max_weight_uids: 256,
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
        self.bittensor
            .common
            .validate()
            .map_err(|e| ConfigurationError::InvalidValue {
                key: "bittensor".to_string(),
                value: "".to_string(),
                reason: e,
            })?;

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
        validate_unique_node_ips(&self.node_management.nodes)?;

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

        // Validate SSH key path
        let ssh_key_path = expand_tilde_in_path(&self.ssh_session.miner_node_key_path);
        if !ssh_key_path.exists() {
            return Err(ConfigurationError::InvalidValue {
                key: "ssh_session.miner_node_key_path".to_string(),
                value: self.ssh_session.miner_node_key_path.display().to_string(),
                reason: format!(
                    "SSH private key not found at path: {} (expanded to: {}). \
                     The miner requires a valid SSH key to access GPU nodes. \
                     Please ensure the key file exists and is readable.",
                    self.ssh_session.miner_node_key_path.display(),
                    ssh_key_path.display()
                ),
            });
        }

        if !ssh_key_path.is_file() {
            return Err(ConfigurationError::InvalidValue {
                key: "ssh_session.miner_node_key_path".to_string(),
                value: self.ssh_session.miner_node_key_path.display().to_string(),
                reason: format!(
                    "SSH key path exists but is not a file: {} (expanded to: {})",
                    self.ssh_session.miner_node_key_path.display(),
                    ssh_key_path.display()
                ),
            });
        }

        Ok(())
    }

    fn warnings(&self) -> Vec<String> {
        vec![]
    }
}

fn validate_unique_node_ips(nodes: &[NodeConfig]) -> Result<(), ConfigurationError> {
    let mut node_indices_by_ip: BTreeMap<IpAddr, Vec<usize>> = BTreeMap::new();

    for (idx, node) in nodes.iter().enumerate() {
        let parsed_ip =
            node.host
                .parse::<IpAddr>()
                .map_err(|_| ConfigurationError::InvalidValue {
                    key: format!("node_management.nodes[{}].host", idx),
                    value: node.host.clone(),
                    reason: "Node host must be a valid IPv4 or IPv6 literal".to_string(),
                })?;
        node_indices_by_ip.entry(parsed_ip).or_default().push(idx);
    }

    let duplicate_ip_entries: Vec<(IpAddr, Vec<usize>)> = node_indices_by_ip
        .into_iter()
        .filter(|(_, indices)| indices.len() > 1)
        .collect();

    if duplicate_ip_entries.is_empty() {
        return Ok(());
    }

    let duplicate_reason = duplicate_ip_entries
        .iter()
        .map(|(ip, indices)| {
            let index_list = indices
                .iter()
                .map(usize::to_string)
                .collect::<Vec<_>>()
                .join(", ");
            format!("{ip} at indices [{index_list}]")
        })
        .collect::<Vec<_>>()
        .join("; ");

    Err(ConfigurationError::InvalidValue {
        key: "node_management.nodes".to_string(),
        value: "contains duplicate node IPs".to_string(),
        reason: format!("Duplicate node IPs are not allowed: {duplicate_reason}"),
    })
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
    pub fn get_advertised_grpc_endpoint(&self) -> Option<String> {
        self.advertised_addresses.grpc_endpoint.clone()
    }

    /// Get the advertised axon endpoint for Bittensor registration
    pub fn get_advertised_axon_endpoint(&self) -> String {
        if let Some(endpoint) = &self.advertised_addresses.axon_endpoint {
            endpoint.clone()
        } else if let Some(external_ip) = &self.bittensor.external_ip {
            format!("http://{}:{}", external_ip, self.bittensor.axon_port)
        } else {
            format!("http://0.0.0.0:{}", self.bittensor.axon_port)
        }
    }

    /// Get the advertised metrics endpoint
    pub fn get_advertised_metrics_endpoint(&self) -> Option<String> {
        self.advertised_addresses.metrics_endpoint.clone()
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

#[cfg(test)]
mod tests {
    use super::*;

    fn test_node(host: &str) -> NodeConfig {
        NodeConfig {
            host: host.to_string(),
            port: 22,
            username: "basilica".to_string(),
            gpu_category: "H100".to_string(),
            gpu_count: 8,
            additional_opts: None,
        }
    }

    #[test]
    fn validate_unique_node_ips_accepts_unique_ipv4() {
        let nodes = vec![test_node("192.168.1.10"), test_node("192.168.1.11")];
        assert!(validate_unique_node_ips(&nodes).is_ok());
    }

    #[test]
    fn validate_unique_node_ips_accepts_unique_ipv6() {
        let nodes = vec![test_node("2001:db8::1"), test_node("2001:db8::2")];
        assert!(validate_unique_node_ips(&nodes).is_ok());
    }

    #[test]
    fn validate_unique_node_ips_rejects_non_ip_host() {
        let nodes = vec![test_node("gpu-node.local")];
        let err = validate_unique_node_ips(&nodes).unwrap_err();

        match err {
            ConfigurationError::InvalidValue { key, reason, .. } => {
                assert_eq!(key, "node_management.nodes[0].host");
                assert!(reason.contains("IPv4 or IPv6 literal"));
            }
            _ => panic!("expected InvalidValue"),
        }
    }

    #[test]
    fn validate_unique_node_ips_rejects_duplicate_ipv4() {
        let nodes = vec![test_node("192.168.1.10"), test_node("192.168.1.10")];
        let err = validate_unique_node_ips(&nodes).unwrap_err();

        match err {
            ConfigurationError::InvalidValue { key, reason, .. } => {
                assert_eq!(key, "node_management.nodes");
                assert!(reason.contains("192.168.1.10 at indices [0, 1]"));
            }
            _ => panic!("expected InvalidValue"),
        }
    }

    #[test]
    fn validate_unique_node_ips_rejects_equivalent_ipv6_representations() {
        let nodes = vec![test_node("2001:db8::1"), test_node("2001:0db8:0:0:0:0:0:1")];
        let err = validate_unique_node_ips(&nodes).unwrap_err();

        match err {
            ConfigurationError::InvalidValue { key, reason, .. } => {
                assert_eq!(key, "node_management.nodes");
                assert!(reason.contains("at indices [0, 1]"));
            }
            _ => panic!("expected InvalidValue"),
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
