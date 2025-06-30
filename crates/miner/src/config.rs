//! # Miner Configuration
//!
//! Configuration structures and validation for the Basilca Miner.

use anyhow::Result;
use serde::{Deserialize, Serialize};
use std::path::{Path, PathBuf};
use std::time::Duration;

use common::config::{
    loader, BittensorConfig, ConfigValidation, DatabaseConfig, LoggingConfig, MetricsConfig,
    ServerConfig,
};
use common::error::ConfigurationError;
use common::identity::{Hotkey, MinerUid};

/// Remote machine configuration for executor deployment
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct RemoteMachineConfig {
    pub id: String,
    pub name: String,
    pub ssh: SshConnectionConfig,
    pub gpu_count: Option<u32>,
    pub executor_binary_path: Option<String>,
    pub executor_data_dir: Option<String>,
    pub executor_port: u16,
}

/// SSH connection configuration
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct SshConnectionConfig {
    pub host: String,
    pub port: u16,
    pub username: String,
    pub private_key_path: PathBuf,
    pub jump_host: Option<String>,
    pub ssh_options: Vec<String>,
}

/// Remote executor deployment configuration
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct RemoteExecutorDeploymentConfig {
    pub remote_machines: Vec<RemoteMachineConfig>,
    pub local_executor_binary: PathBuf,
    pub executor_config_template: String,
    pub auto_deploy: bool,
    pub auto_start: bool,
    pub health_check_interval: Duration,
}

/// Main miner configuration
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct MinerConfig {
    /// Bittensor network configuration
    pub bittensor: MinerBittensorConfig,

    /// Database configuration
    pub database: DatabaseConfig,

    /// Server configuration for validator communications
    pub server: ServerConfig,

    /// Logging configuration
    pub logging: LoggingConfig,

    /// Metrics configuration
    pub metrics: MetricsConfig,

    /// Validator communications configuration
    pub validator_comms: ValidatorCommsConfig,

    /// Executor management configuration
    pub executor_management: ExecutorManagementConfig,

    /// Remote executor deployment configuration (optional)
    pub remote_executor_deployment: Option<RemoteExecutorDeploymentConfig>,

    /// Security configuration
    pub security: SecurityConfig,
}

/// Miner-specific Bittensor configuration
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct MinerBittensorConfig {
    /// Common Bittensor configuration
    #[serde(flatten)]
    pub common: BittensorConfig,

    /// Miner's unique identifier on the network
    pub uid: MinerUid,

    /// Coldkey name for wallet operations
    pub coldkey_name: String,

    /// Axon server port for Bittensor network
    pub axon_port: u16,

    /// External IP address for the axon
    pub external_ip: Option<String>,

    /// Maximum number of UIDs to set weights for
    pub max_weight_uids: u16,

    /// Skip chain registration check (for local testing only)
    #[serde(default)]
    pub skip_registration: bool,
}

/// Validator communications configuration
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct ValidatorCommsConfig {
    /// TLS configuration for secure communications
    pub tls: Option<TlsConfig>,

    /// Authentication configuration
    pub auth: AuthConfig,

    /// Request timeout for validator calls
    pub request_timeout: Duration,

    /// Maximum concurrent validator sessions
    pub max_concurrent_sessions: u32,

    /// Rate limiting configuration
    pub rate_limit: RateLimitConfig,
}

/// Executor management configuration
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct ExecutorManagementConfig {
    /// Static list of executors managed by this miner
    pub executors: Vec<ExecutorConfig>,

    /// Health check interval for executors
    pub health_check_interval: Duration,

    /// Timeout for executor health checks
    pub health_check_timeout: Duration,

    /// Maximum retry attempts for failed operations
    pub max_retry_attempts: u32,

    /// Enable automatic status recovery
    pub auto_recovery: bool,
}

/// Static executor configuration
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct ExecutorConfig {
    /// Unique identifier for the executor
    pub id: String,

    /// gRPC address of the executor (host:port)
    pub grpc_address: String,

    /// Optional display name for the executor
    pub name: Option<String>,

    /// Optional metadata about the executor
    pub metadata: Option<serde_json::Value>,
}

/// Security configuration
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct SecurityConfig {
    /// Enable mTLS for all gRPC communications
    pub enable_mtls: bool,

    /// Certificate paths for mTLS
    pub cert_path: Option<PathBuf>,
    pub key_path: Option<PathBuf>,
    pub ca_cert_path: Option<PathBuf>,

    /// JWT secret for authentication tokens
    pub jwt_secret: String,

    /// Token expiration time
    pub token_expiration: Duration,

    /// Allowed validator hotkeys
    pub allowed_validators: Vec<Hotkey>,

    /// Enable request signing verification
    pub verify_signatures: bool,
}

/// TLS configuration
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct TlsConfig {
    /// Certificate file path
    pub cert_file: PathBuf,

    /// Private key file path
    pub key_file: PathBuf,

    /// CA certificate file path (for client cert verification)
    pub ca_cert_file: Option<PathBuf>,

    /// Require client certificates
    pub require_client_cert: bool,
}

/// Authentication configuration
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct AuthConfig {
    /// Enable authentication for validator requests
    pub enabled: bool,

    /// Authentication method
    pub method: AuthMethod,

    /// Token validation settings
    pub token_validation: TokenValidationConfig,
}

/// Authentication methods
#[derive(Debug, Clone, Serialize, Deserialize)]
#[serde(rename_all = "snake_case")]
pub enum AuthMethod {
    /// JWT token based authentication
    Jwt,
    /// Bittensor signature based authentication
    BittensorSignature,
    /// mTLS certificate based authentication
    MutualTls,
}

/// Token validation configuration
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct TokenValidationConfig {
    /// Issuer validation
    pub validate_issuer: bool,

    /// Audience validation
    pub validate_audience: bool,

    /// Expiration validation
    pub validate_expiration: bool,

    /// Clock skew tolerance
    pub clock_skew_tolerance: Duration,
}

/// Rate limiting configuration
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct RateLimitConfig {
    /// Enable rate limiting
    pub enabled: bool,

    /// Requests per second limit
    pub requests_per_second: u32,

    /// Burst capacity
    pub burst_capacity: u32,

    /// Rate limit window duration
    pub window_duration: Duration,
}

impl Default for MinerConfig {
    fn default() -> Self {
        Self {
            bittensor: MinerBittensorConfig::default(),
            database: DatabaseConfig {
                url: "sqlite:./data/miner.db".to_string(),
                ..Default::default()
            },
            server: ServerConfig {
                host: "0.0.0.0".to_string(),
                port: 8092,
                ..Default::default()
            },
            logging: LoggingConfig::default(),
            metrics: MetricsConfig::default(),
            validator_comms: ValidatorCommsConfig::default(),
            executor_management: ExecutorManagementConfig::default(),
            remote_executor_deployment: None,
            security: SecurityConfig::default(),
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
                netuid: 27, // Basilca subnet ID
                chain_endpoint: Some("wss://entrypoint-finney.opentensor.ai:443".to_string()),
                weight_interval_secs: 300, // 5 minutes
            },
            uid: MinerUid::from(0),
            coldkey_name: "default".to_string(),
            axon_port: 8091,
            external_ip: None,
            max_weight_uids: 256,
            skip_registration: false,
        }
    }
}

impl Default for TlsConfig {
    fn default() -> Self {
        Self {
            cert_file: PathBuf::from("cert.pem"),
            key_file: PathBuf::from("key.pem"),
            ca_cert_file: None,
            require_client_cert: false,
        }
    }
}

impl Default for ValidatorCommsConfig {
    fn default() -> Self {
        Self {
            tls: None,
            auth: AuthConfig::default(),
            request_timeout: Duration::from_secs(30),
            max_concurrent_sessions: 100,
            rate_limit: RateLimitConfig::default(),
        }
    }
}

impl Default for ExecutorManagementConfig {
    fn default() -> Self {
        Self {
            executors: vec![],
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
            enable_mtls: false,
            cert_path: None,
            key_path: None,
            ca_cert_path: None,
            jwt_secret: "change-me-in-production".to_string(),
            token_expiration: Duration::from_secs(3600),
            allowed_validators: vec![],
            verify_signatures: true,
        }
    }
}

impl Default for AuthConfig {
    fn default() -> Self {
        Self {
            enabled: true,
            method: AuthMethod::BittensorSignature,
            token_validation: TokenValidationConfig::default(),
        }
    }
}

impl Default for TokenValidationConfig {
    fn default() -> Self {
        Self {
            validate_issuer: true,
            validate_audience: true,
            validate_expiration: true,
            clock_skew_tolerance: Duration::from_secs(60),
        }
    }
}

impl Default for RateLimitConfig {
    fn default() -> Self {
        Self {
            enabled: true,
            requests_per_second: 10,
            burst_capacity: 20,
            window_duration: Duration::from_secs(60),
        }
    }
}

impl ConfigValidation for MinerConfig {
    type Error = ConfigurationError;

    fn validate(&self) -> Result<(), Self::Error> {
        // Validate common configs using their validation
        self.database.validate()?;
        self.server.validate()?;

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

        // Validate executor management - allow empty if using remote deployment
        if self.executor_management.executors.is_empty()
            && self.remote_executor_deployment.is_none()
        {
            return Err(ConfigurationError::InvalidValue {
                key: "executor_management.executors".to_string(),
                value: "[]".to_string(),
                reason:
                    "At least one executor must be configured or remote deployment must be enabled"
                        .to_string(),
            });
        }

        // Validate each executor config
        for (idx, executor) in self.executor_management.executors.iter().enumerate() {
            // Allow empty ID as it will be auto-generated
            // if executor.id.is_empty() {
            //     return Err(ConfigurationError::InvalidValue {
            //         key: format!("executor_management.executors[{}].id", idx),
            //         value: executor.id.clone(),
            //         reason: "Executor ID cannot be empty".to_string(),
            //     });
            // }

            if executor.grpc_address.is_empty() {
                return Err(ConfigurationError::InvalidValue {
                    key: format!("executor_management.executors[{idx}].grpc_address"),
                    value: executor.grpc_address.clone(),
                    reason: "Executor gRPC address cannot be empty".to_string(),
                });
            }
        }

        // Validate security configuration
        if self.security.jwt_secret == "change-me-in-production" {
            return Err(ConfigurationError::InvalidValue {
                key: "security.jwt_secret".to_string(),
                value: "***".to_string(),
                reason: "JWT secret must be changed from default value in production".to_string(),
            });
        }

        // Validate TLS configuration if enabled
        if let Some(ref tls) = self.validator_comms.tls {
            if !tls.cert_file.exists() {
                return Err(ConfigurationError::InvalidValue {
                    key: "validator_comms.tls.cert_file".to_string(),
                    value: format!("{:?}", tls.cert_file),
                    reason: "TLS certificate file does not exist".to_string(),
                });
            }

            if !tls.key_file.exists() {
                return Err(ConfigurationError::InvalidValue {
                    key: "validator_comms.tls.key_file".to_string(),
                    value: format!("{:?}", tls.key_file),
                    reason: "TLS key file does not exist".to_string(),
                });
            }
        }

        Ok(())
    }

    fn warnings(&self) -> Vec<String> {
        let mut warnings = Vec::new();

        if !self.security.enable_mtls {
            warnings.push("mTLS is disabled - consider enabling for production".to_string());
        }

        if self.security.allowed_validators.is_empty() {
            warnings
                .push("No validators in allowlist - all validators will be accepted".to_string());
        }

        if !self.validator_comms.rate_limit.enabled {
            warnings.push("Rate limiting is disabled - may be vulnerable to DoS".to_string());
        }

        warnings
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
