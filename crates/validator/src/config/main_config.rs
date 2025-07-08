//! # Validator Configuration
//!
//! Layered configuration management for the Basilca Validator.

use anyhow::Result;
use serde::{Deserialize, Serialize};
use std::path::PathBuf;
use std::time::Duration;

use common::config::{
    loader, BittensorConfig, ConfigValidation, DatabaseConfig, LoggingConfig, MetricsConfig,
    ServerConfig,
};
use common::error::ConfigurationError;

/// Enhanced validator Bittensor configuration with advertised address support
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct ValidatorBittensorConfig {
    /// Common Bittensor configuration
    #[serde(flatten)]
    pub common: BittensorConfig,

    /// Axon server port for Bittensor network
    pub axon_port: u16,

    /// External IP address for the axon
    pub external_ip: Option<String>,

    /// Advertised axon endpoint override (full URL)
    pub advertised_axon_endpoint: Option<String>,

    /// Enable TLS for advertised axon endpoint
    #[serde(default)]
    pub advertised_axon_tls: bool,
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

    /// Automatic verification configuration
    #[serde(default)]
    pub automatic_verification: AutomaticVerificationConfig,

    /// Storage configuration
    pub storage: StorageConfig,

    /// API-specific configuration
    pub api: ApiConfig,

    /// SSH session configuration
    pub ssh_session: SshSessionConfig,

    /// Emission allocation configuration
    #[serde(default)]
    pub emission: super::emission::EmissionConfig,
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
    /// Maximum number of miners to verify per round
    pub max_miners_per_round: usize,
    /// Minimum interval between verifications of the same miner
    pub min_verification_interval: Duration,
    /// Network ID for the subnet
    pub netuid: u16,
    /// Enable dynamic discovery of executor SSH details from miners
    #[serde(default = "default_use_dynamic_discovery")]
    pub use_dynamic_discovery: bool,
    /// Timeout for miner discovery operations
    #[serde(default = "default_discovery_timeout")]
    pub discovery_timeout: Duration,
    /// Fall back to static SSH config if dynamic discovery fails
    #[serde(default = "default_fallback_to_static")]
    pub fallback_to_static: bool,
    /// Cache miner endpoint info TTL
    #[serde(default = "default_cache_miner_info_ttl")]
    pub cache_miner_info_ttl: Duration,
    /// gRPC port offset from axon port (if not using default 50061)
    #[serde(default)]
    pub grpc_port_offset: Option<u16>,
}

fn default_use_dynamic_discovery() -> bool {
    true
}

fn default_discovery_timeout() -> Duration {
    Duration::from_secs(30)
}

fn default_fallback_to_static() -> bool {
    true
}

fn default_cache_miner_info_ttl() -> Duration {
    Duration::from_secs(300) // 5 minutes
}

/// Configuration for automatic verification during discovery
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct AutomaticVerificationConfig {
    /// Enable automatic verification during discovery
    #[serde(default = "default_enable_automatic")]
    pub enabled: bool,

    /// Discovery verification interval (seconds)
    #[serde(default = "default_discovery_interval")]
    pub discovery_interval: u64,

    /// Minimum time between verifications for the same miner (hours)
    #[serde(default = "default_min_verification_interval_hours")]
    pub min_verification_interval_hours: u64,

    /// Maximum concurrent verifications
    #[serde(default = "default_max_concurrent_verifications")]
    pub max_concurrent_verifications: usize,

    /// Enable SSH session automation
    #[serde(default = "default_enable_ssh_automation")]
    pub enable_ssh_automation: bool,
}

fn default_enable_automatic() -> bool {
    true
}

fn default_discovery_interval() -> u64 {
    300 // 5 minutes
}

fn default_min_verification_interval_hours() -> u64 {
    1
}

fn default_max_concurrent_verifications() -> usize {
    5
}

fn default_enable_ssh_automation() -> bool {
    true
}

impl Default for AutomaticVerificationConfig {
    fn default() -> Self {
        Self {
            enabled: default_enable_automatic(),
            discovery_interval: default_discovery_interval(),
            min_verification_interval_hours: default_min_verification_interval_hours(),
            max_concurrent_verifications: default_max_concurrent_verifications(),
            enable_ssh_automation: default_enable_ssh_automation(),
        }
    }
}

fn default_enable_automated_sessions() -> bool {
    true
}

fn default_max_concurrent_sessions() -> usize {
    5
}

fn default_session_rate_limit() -> usize {
    20
}

fn default_enable_audit_logging() -> bool {
    true
}

fn default_audit_log_path() -> PathBuf {
    PathBuf::from("/var/log/basilica/ssh_audit.log")
}

fn default_ssh_connection_timeout() -> Duration {
    Duration::from_secs(30)
}

fn default_ssh_command_timeout() -> Duration {
    Duration::from_secs(60)
}

fn default_ssh_retry_attempts() -> u32 {
    3
}

fn default_ssh_retry_delay() -> Duration {
    Duration::from_secs(2)
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

#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct SshSessionConfig {
    /// Directory for storing ephemeral SSH keys
    pub ssh_key_directory: PathBuf,

    /// SSH key algorithm (ed25519, rsa, ecdsa)
    pub key_algorithm: String,

    /// Path to persistent SSH private key file (optional)
    pub persistent_ssh_key_path: Option<PathBuf>,

    /// Default session duration in seconds
    pub default_session_duration: u64,

    /// Maximum session duration in seconds
    pub max_session_duration: u64,

    /// Cleanup interval for expired keys
    pub key_cleanup_interval: Duration,

    /// Enable automated SSH session management during discovery
    #[serde(default = "default_enable_automated_sessions")]
    pub enable_automated_sessions: bool,

    /// Maximum concurrent SSH sessions per validator
    #[serde(default = "default_max_concurrent_sessions")]
    pub max_concurrent_sessions: usize,

    /// Session rate limit (sessions per hour)
    #[serde(default = "default_session_rate_limit")]
    pub session_rate_limit: usize,

    /// Enable audit logging for SSH sessions
    #[serde(default = "default_enable_audit_logging")]
    pub enable_audit_logging: bool,

    /// Audit log file path
    #[serde(default = "default_audit_log_path")]
    pub audit_log_path: PathBuf,

    /// SSH connection timeout
    #[serde(default = "default_ssh_connection_timeout")]
    pub ssh_connection_timeout: Duration,

    /// SSH command execution timeout
    #[serde(default = "default_ssh_command_timeout")]
    pub ssh_command_timeout: Duration,

    /// Retry SSH connection attempts
    #[serde(default = "default_ssh_retry_attempts")]
    pub ssh_retry_attempts: u32,

    /// Delay between SSH retry attempts
    #[serde(default = "default_ssh_retry_delay")]
    pub ssh_retry_delay: Duration,
}

impl Default for SshSessionConfig {
    fn default() -> Self {
        Self {
            ssh_key_directory: PathBuf::from("/tmp/validator_ssh_keys"),
            key_algorithm: "ed25519".to_string(),
            persistent_ssh_key_path: None,
            default_session_duration: 300, // 5 minutes
            max_session_duration: 3600,    // 1 hour
            key_cleanup_interval: Duration::from_secs(60),
            enable_automated_sessions: default_enable_automated_sessions(),
            max_concurrent_sessions: default_max_concurrent_sessions(),
            session_rate_limit: default_session_rate_limit(),
            enable_audit_logging: default_enable_audit_logging(),
            audit_log_path: default_audit_log_path(),
            ssh_connection_timeout: default_ssh_connection_timeout(),
            ssh_command_timeout: default_ssh_command_timeout(),
            ssh_retry_attempts: default_ssh_retry_attempts(),
            ssh_retry_delay: default_ssh_retry_delay(),
        }
    }
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
                advertised_axon_endpoint: None,
                advertised_axon_tls: false,
            },
            verification: VerificationConfig {
                verification_interval: Duration::from_secs(600),
                max_concurrent_verifications: 50,
                challenge_timeout: Duration::from_secs(120),
                min_score_threshold: 0.1,
                max_miners_per_round: 20,
                min_verification_interval: Duration::from_secs(1800), // 30 minutes
                netuid: 1,                                            // Default subnet
                use_dynamic_discovery: default_use_dynamic_discovery(),
                discovery_timeout: default_discovery_timeout(),
                fallback_to_static: default_fallback_to_static(),
                cache_miner_info_ttl: default_cache_miner_info_ttl(),
                grpc_port_offset: None,
            },
            automatic_verification: AutomaticVerificationConfig::default(),
            storage: StorageConfig {
                data_dir: "./data".to_string(),
            },
            api: ApiConfig {
                api_key: None,
                max_body_size: 1024 * 1024, // 1MB
                bind_address: "0.0.0.0:8080".to_string(),
            },
            ssh_session: SshSessionConfig::default(),
            emission: super::emission::EmissionConfig::default(),
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

        // Validate advertised axon configuration
        if let Err(msg) = self.bittensor.validate_advertised_axon() {
            return Err(ConfigurationError::InvalidValue {
                key: "bittensor.advertised_axon".to_string(),
                value: "advertised_endpoint".to_string(),
                reason: msg,
            });
        }

        // Validate emission configuration
        if let Err(e) = self.emission.validate() {
            return Err(ConfigurationError::InvalidValue {
                key: "emission".to_string(),
                value: "emission_config".to_string(),
                reason: e.to_string(),
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

impl ValidatorBittensorConfig {
    /// Get the advertised axon endpoint for chain registration
    pub fn get_advertised_axon_endpoint(&self) -> String {
        if let Some(endpoint) = &self.advertised_axon_endpoint {
            endpoint.clone()
        } else if let Some(external_ip) = &self.external_ip {
            let protocol = if self.advertised_axon_tls {
                "https"
            } else {
                "http"
            };
            format!("{}://{}:{}", protocol, external_ip, self.axon_port)
        } else {
            format!("http://0.0.0.0:{}", self.axon_port)
        }
    }

    /// Extract host and port from advertised axon endpoint
    pub fn get_advertised_axon_host_port(&self) -> Result<(String, u16), String> {
        let endpoint = self.get_advertised_axon_endpoint();

        let url = url::Url::parse(&endpoint)
            .map_err(|e| format!("Invalid advertised axon endpoint: {e}"))?;

        let host = url
            .host_str()
            .ok_or_else(|| "No host in advertised axon endpoint".to_string())?
            .to_string();

        let port = url
            .port()
            .ok_or_else(|| "No port in advertised axon endpoint".to_string())?;

        Ok((host, port))
    }

    /// Validate advertised axon configuration
    pub fn validate_advertised_axon(&self) -> Result<(), String> {
        if let Some(ref endpoint) = self.advertised_axon_endpoint {
            if !endpoint.starts_with("http://") && !endpoint.starts_with("https://") {
                return Err(
                    "Advertised axon endpoint must start with http:// or https://".to_string(),
                );
            }

            let _ = self.get_advertised_axon_host_port()?;
        }

        if self.axon_port == 0 {
            return Err("Axon port cannot be zero".to_string());
        }

        Ok(())
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
