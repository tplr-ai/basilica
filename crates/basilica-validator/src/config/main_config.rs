//! # Validator Configuration
//!
//! Layered configuration management for the Basilca Validator.

use anyhow::Result;
use serde::{Deserialize, Serialize};
use std::path::{Path, PathBuf};
use std::time::Duration;

use basilica_common::config::{
    loader, BittensorConfig, ConfigValidation, DatabaseConfig, LoggingConfig, MetricsConfig,
    ServerConfig,
};
use basilica_common::error::ConfigurationError;

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

    /// Database cleanup configuration
    #[serde(default)]
    pub cleanup: crate::persistence::cleanup_task::CleanupConfig,

    /// Billing telemetry streaming configuration
    #[serde(default)]
    pub billing: BillingConfig,
}

#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct VerificationConfig {
    /// How often to verify miners
    pub verification_interval: Duration,
    /// Maximum concurrent verifications
    pub max_concurrent_verifications: usize,
    /// Maximum concurrent full validations (resource-intensive binary validations)
    #[serde(default = "default_max_concurrent_full_validations")]
    pub max_concurrent_full_validations: usize,
    /// Challenge timeout
    pub challenge_timeout: Duration,
    /// Minimum score threshold for miners
    pub min_score_threshold: f64,
    /// Maximum number of miners to verify per round
    pub max_miners_per_round: usize,
    /// Minimum interval between verifications of the same miner
    pub min_verification_interval: Duration,
    /// Enable dynamic discovery of node SSH details from miners
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
    /// Binary validation configuration
    #[serde(default)]
    pub binary_validation: BinaryValidationConfig,
    /// Docker validation configuration
    #[serde(default)]
    pub docker_validation: DockerValidationConfig,
    /// Storage validation configuration
    #[serde(default)]
    pub storage_validation: StorageValidationConfig,
    /// Collateral event scan interval
    #[serde(default = "default_collateral_event_scan_interval")]
    pub collateral_event_scan_interval: Duration,
    /// Interval between full binary validations per node
    #[serde(default = "default_node_validation_interval")]
    pub node_validation_interval: Duration,
    /// Time period for cleaning up GPU assignments from offline nodes (min 2 hours)
    #[serde(default = "default_gpu_assignment_cleanup_ttl")]
    pub gpu_assignment_cleanup_ttl: Option<Duration>,
    /// Enable worker queue for decoupled validation execution
    #[serde(default = "default_enable_worker_queue")]
    pub enable_worker_queue: bool,
    /// Node group assignment configuration
    #[serde(default)]
    pub node_groups: NodeGroupConfig,
}

/// Configuration for node group assignment strategy
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct NodeGroupConfig {
    /// Strategy for assigning nodes to groups: "round-robin", "all-jobs", "all-rentals"
    #[serde(default = "default_node_group_strategy")]
    pub strategy: String,
    /// Percentage of nodes to assign to jobs group (0-100) when using round-robin
    #[serde(default = "default_jobs_percentage")]
    pub jobs_percentage: u64,
    /// Force all nodes to specific group (overrides strategy): "jobs", "rentals", or None
    #[serde(default)]
    pub force_group: Option<String>,
}

impl Default for NodeGroupConfig {
    fn default() -> Self {
        Self {
            strategy: default_node_group_strategy(),
            jobs_percentage: default_jobs_percentage(),
            force_group: None,
        }
    }
}

fn default_node_group_strategy() -> String {
    "round-robin".to_string()
}

fn default_jobs_percentage() -> u64 {
    50 // Default 50/50 split
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

fn default_collateral_event_scan_interval() -> Duration {
    Duration::from_secs(12) // one block time
}

fn default_node_validation_interval() -> Duration {
    Duration::from_secs(6 * 3600) // 6 hours
}

fn default_gpu_assignment_cleanup_ttl() -> Option<Duration> {
    Some(Duration::from_secs(120 * 60)) // 2 hours
}

fn default_enable_worker_queue() -> bool {
    false // Disabled by default until fully tested
}

fn default_max_concurrent_full_validations() -> usize {
    1024 // Allow up to 1024 concurrent validation requests to the server
}

impl VerificationConfig {
    #[cfg(test)]
    pub fn test_default() -> Self {
        Self {
            verification_interval: Duration::from_secs(60),
            max_concurrent_verifications: 50,
            max_concurrent_full_validations: 1,
            challenge_timeout: Duration::from_secs(120),
            min_score_threshold: 0.1,
            max_miners_per_round: 10,
            min_verification_interval: Duration::from_secs(300),
            use_dynamic_discovery: true,
            discovery_timeout: Duration::from_secs(30),
            fallback_to_static: true,
            cache_miner_info_ttl: Duration::from_secs(300),
            grpc_port_offset: None,
            binary_validation: BinaryValidationConfig::default(),
            docker_validation: DockerValidationConfig::default(),
            collateral_event_scan_interval: Duration::from_secs(12),
            node_validation_interval: Duration::from_secs(3600),
            gpu_assignment_cleanup_ttl: Some(Duration::from_secs(7200)),
            enable_worker_queue: false,
            storage_validation: StorageValidationConfig::default(),
            node_groups: NodeGroupConfig::default(),
        }
    }
}

/// Configuration for binary validation using validator-binary and executor-binary
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct BinaryValidationConfig {
    /// Path to validator-binary executable
    pub validator_binary_path: PathBuf,
    /// Path to executor-binary for upload
    pub executor_binary_path: PathBuf,
    /// Binary execution timeout in seconds
    pub execution_timeout_secs: u64,
    /// Output format for binary execution
    pub output_format: String,
    /// Enable binary validation (fallback to SSH test only)
    pub enabled: bool,
    /// Binary validation weight in final score calculation
    pub score_weight: f64,
    /// Default node port for SSH tunnel cleanup
    #[serde(default = "default_node_port")]
    pub node_port: u16,
    /// Server mode configuration
    #[serde(default)]
    pub server_mode: ValidationServerConfig,
}

/// Configuration for validation server mode
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct ValidationServerConfig {
    /// Server bind address
    #[serde(default = "default_server_bind_address")]
    pub bind_address: String,
    /// Maximum concurrent remote operations
    #[serde(default = "default_remote_concurrency")]
    pub remote_concurrency: usize,
    /// Maximum concurrent verification operations
    #[serde(default = "default_verify_concurrency")]
    pub verify_concurrency: usize,
    /// Queue capacity for pending jobs
    #[serde(default = "default_queue_capacity")]
    pub queue_capacity: usize,
    /// Health check interval in seconds
    #[serde(default = "default_health_check_interval")]
    pub health_check_interval_secs: u64,
    /// Job polling interval in milliseconds
    #[serde(default = "default_job_poll_interval_ms")]
    pub job_poll_interval_ms: u64,
    /// Maximum polling attempts before timing out
    #[serde(default = "default_max_poll_attempts")]
    pub max_poll_attempts: usize,
    /// Maximum time to wait for server to become ready (seconds)
    #[serde(default = "default_server_ready_timeout_secs")]
    pub server_ready_timeout_secs: u64,
    /// Interval between server readiness checks (milliseconds)
    #[serde(default = "default_server_ready_check_interval_ms")]
    pub server_ready_check_interval_ms: u64,
    /// Maximum workflow retry attempts on persistent 404 errors
    #[serde(default = "default_max_workflow_retry_attempts")]
    pub max_workflow_retry_attempts: u32,
    /// Base delay for workflow retry backoff (milliseconds)
    #[serde(default = "default_workflow_retry_base_delay_ms")]
    pub workflow_retry_base_delay_ms: u64,
}

impl Default for ValidationServerConfig {
    fn default() -> Self {
        Self {
            bind_address: default_server_bind_address(),
            remote_concurrency: default_remote_concurrency(),
            verify_concurrency: default_verify_concurrency(),
            queue_capacity: default_queue_capacity(),
            health_check_interval_secs: default_health_check_interval(),
            job_poll_interval_ms: default_job_poll_interval_ms(),
            max_poll_attempts: default_max_poll_attempts(),
            server_ready_timeout_secs: default_server_ready_timeout_secs(),
            server_ready_check_interval_ms: default_server_ready_check_interval_ms(),
            max_workflow_retry_attempts: default_max_workflow_retry_attempts(),
            workflow_retry_base_delay_ms: default_workflow_retry_base_delay_ms(),
        }
    }
}
/// Configuration for Docker validation
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct DockerValidationConfig {
    /// Docker image to pull during validation
    #[serde(default = "default_docker_image")]
    pub docker_image: String,
    /// Timeout for pulling Docker image in seconds
    #[serde(default = "default_docker_pull_timeout")]
    pub pull_timeout_secs: u64,
}

impl Default for DockerValidationConfig {
    fn default() -> Self {
        Self {
            docker_image: default_docker_image(),
            pull_timeout_secs: default_docker_pull_timeout(),
        }
    }
}

/// Configuration for storage validation
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct StorageValidationConfig {
    /// Minimum required storage in bytes (default: 1TB)
    #[serde(default = "default_min_required_storage_bytes")]
    pub min_required_storage_bytes: u64,
}

impl Default for StorageValidationConfig {
    fn default() -> Self {
        Self {
            min_required_storage_bytes: default_min_required_storage_bytes(),
        }
    }
}

fn default_min_required_storage_bytes() -> u64 {
    1_099_511_627_776 // 1TB in bytes
}

fn default_docker_image() -> String {
    "nvidia/cuda:12.8.0-runtime-ubuntu22.04".to_string()
}

fn default_docker_pull_timeout() -> u64 {
    1800 // 30 minutes
}

fn default_server_bind_address() -> String {
    "127.0.0.1:4010".to_string()
}

fn default_remote_concurrency() -> usize {
    1024
}

fn default_verify_concurrency() -> usize {
    1 // Only one GPU validation at a time on the server
}

fn default_queue_capacity() -> usize {
    4096
}

fn default_health_check_interval() -> u64 {
    30 // 30 seconds
}

fn default_job_poll_interval_ms() -> u64 {
    500 // Poll every 500ms initially
}

fn default_max_poll_attempts() -> usize {
    2400 // 20 minutes with 500ms intervals
}

fn default_server_ready_timeout_secs() -> u64 {
    30 // 30 seconds to wait for server to become ready
}

fn default_server_ready_check_interval_ms() -> u64 {
    500 // Check every 500ms
}

fn default_max_workflow_retry_attempts() -> u32 {
    3 // Retry entire workflow up to 3 times
}

fn default_workflow_retry_base_delay_ms() -> u64 {
    2000 // 2 seconds base delay, with exponential backoff
}

fn default_node_port() -> u16 {
    3000
}

impl Default for BinaryValidationConfig {
    fn default() -> Self {
        Self {
            validator_binary_path: PathBuf::from("./validator-binary"),
            executor_binary_path: PathBuf::from("./executor-binary"),
            execution_timeout_secs: 1200,
            output_format: "json".to_string(),
            enabled: true,
            score_weight: 0.8,
            node_port: default_node_port(),
            server_mode: ValidationServerConfig::default(),
        }
    }
}

impl BinaryValidationConfig {
    #[cfg(test)]
    pub fn test_default() -> Self {
        Self::default()
    }
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
    50
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

impl AutomaticVerificationConfig {
    #[cfg(test)]
    pub fn test_default() -> Self {
        Self::default()
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

fn default_strict_host_key_checking() -> bool {
    false
}

fn default_known_hosts_file() -> Option<PathBuf> {
    None
}

fn default_rental_session_duration() -> u64 {
    0
}

#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct BillingConfig {
    /// Enable billing telemetry streaming
    #[serde(default = "default_billing_enabled")]
    pub enabled: bool,

    /// Billing service gRPC endpoint
    #[serde(default = "default_billing_endpoint")]
    pub billing_endpoint: String,

    /// Request timeout in seconds
    #[serde(default = "default_billing_timeout_secs")]
    pub timeout_secs: u64,

    /// Use TLS for billing connection
    #[serde(default = "default_billing_use_tls")]
    pub use_tls: bool,

    /// Maximum telemetry batch size
    #[serde(default = "default_billing_batch_size")]
    pub batch_size: usize,

    /// Billing telemetry collection interval in seconds
    #[serde(default = "default_billing_collection_interval_secs")]
    pub collection_interval_secs: u64,

    /// Telemetry flush interval in seconds
    #[serde(default = "default_billing_flush_interval_secs")]
    pub flush_interval_secs: u64,

    /// Maximum retry attempts for failed telemetry
    #[serde(default = "default_billing_max_retries")]
    pub max_retries: u32,

    /// Circuit breaker failure threshold
    #[serde(default = "default_billing_circuit_failure_threshold")]
    pub circuit_failure_threshold: u32,

    /// Circuit breaker recovery timeout in seconds
    #[serde(default = "default_billing_circuit_recovery_timeout_secs")]
    pub circuit_recovery_timeout_secs: u64,

    /// Circuit breaker failure window duration in seconds
    #[serde(default = "default_billing_circuit_window_duration_secs")]
    pub circuit_window_duration_secs: u64,
}

fn default_billing_enabled() -> bool {
    false
}

fn default_billing_endpoint() -> String {
    "http://127.0.0.1:50051".to_string()
}

fn default_billing_timeout_secs() -> u64 {
    30
}

fn default_billing_use_tls() -> bool {
    false
}

fn default_billing_batch_size() -> usize {
    100
}

fn default_billing_collection_interval_secs() -> u64 {
    30
}

fn default_billing_flush_interval_secs() -> u64 {
    30
}

fn default_billing_max_retries() -> u32 {
    10
}

fn default_billing_circuit_failure_threshold() -> u32 {
    5
}

fn default_billing_circuit_recovery_timeout_secs() -> u64 {
    30
}

fn default_billing_circuit_window_duration_secs() -> u64 {
    60
}

impl Default for BillingConfig {
    fn default() -> Self {
        Self {
            enabled: default_billing_enabled(),
            billing_endpoint: default_billing_endpoint(),
            timeout_secs: default_billing_timeout_secs(),
            use_tls: default_billing_use_tls(),
            batch_size: default_billing_batch_size(),
            collection_interval_secs: default_billing_collection_interval_secs(),
            flush_interval_secs: default_billing_flush_interval_secs(),
            max_retries: default_billing_max_retries(),
            circuit_failure_threshold: default_billing_circuit_failure_threshold(),
            circuit_recovery_timeout_secs: default_billing_circuit_recovery_timeout_secs(),
            circuit_window_duration_secs: default_billing_circuit_window_duration_secs(),
        }
    }
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
    /// Default port for miner connections
    #[serde(default = "default_miner_port")]
    pub miner_port: u16,
}

fn default_miner_port() -> u16 {
    8091
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

    /// Rental session duration in seconds (0 = no predetermined duration)
    #[serde(default = "default_rental_session_duration")]
    pub rental_session_duration: u64,

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

    /// Enable strict host key checking for SSH connections
    #[serde(default = "default_strict_host_key_checking")]
    pub strict_host_key_checking: bool,

    /// Path to known hosts file for SSH host key verification
    #[serde(default = "default_known_hosts_file")]
    pub known_hosts_file: Option<PathBuf>,
}

impl Default for SshSessionConfig {
    fn default() -> Self {
        Self {
            ssh_key_directory: PathBuf::from("/tmp/validator_ssh_keys"),
            key_algorithm: "ed25519".to_string(),
            persistent_ssh_key_path: None,
            default_session_duration: 300, // 5 minutes
            max_session_duration: 3600,    // 1 hour
            rental_session_duration: default_rental_session_duration(),
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
            strict_host_key_checking: default_strict_host_key_checking(),
            known_hosts_file: default_known_hosts_file(),
        }
    }
}

impl SshSessionConfig {
    #[cfg(test)]
    pub fn test_default() -> Self {
        Self::default()
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
                    ..Default::default()
                },
                axon_port: 9090,
                external_ip: None,
                advertised_axon_endpoint: None,
                advertised_axon_tls: false,
            },
            verification: VerificationConfig {
                verification_interval: Duration::from_secs(600),
                max_concurrent_verifications: 50,
                max_concurrent_full_validations: default_max_concurrent_full_validations(),
                challenge_timeout: Duration::from_secs(120),
                min_score_threshold: 0.1,
                max_miners_per_round: 20,
                min_verification_interval: Duration::from_secs(1800), // 30 minutes
                use_dynamic_discovery: default_use_dynamic_discovery(),
                discovery_timeout: default_discovery_timeout(),
                fallback_to_static: default_fallback_to_static(),
                cache_miner_info_ttl: default_cache_miner_info_ttl(),
                grpc_port_offset: None,
                binary_validation: BinaryValidationConfig::default(),
                docker_validation: DockerValidationConfig::default(),
                storage_validation: StorageValidationConfig::default(),
                collateral_event_scan_interval: default_collateral_event_scan_interval(),
                node_validation_interval: default_node_validation_interval(),
                gpu_assignment_cleanup_ttl: default_gpu_assignment_cleanup_ttl(),
                enable_worker_queue: default_enable_worker_queue(),
                node_groups: NodeGroupConfig::default(),
            },
            automatic_verification: AutomaticVerificationConfig::default(),
            storage: StorageConfig {
                data_dir: "./data".to_string(),
            },
            api: ApiConfig {
                api_key: None,
                max_body_size: 1024 * 1024, // 1MB
                bind_address: "0.0.0.0:8080".to_string(),
                miner_port: default_miner_port(),
            },
            ssh_session: SshSessionConfig::default(),
            emission: super::emission::EmissionConfig::default(),
            cleanup: crate::persistence::cleanup_task::CleanupConfig::default(),
            billing: BillingConfig::default(),
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

        if self.verification.max_concurrent_full_validations == 0 {
            return Err(ConfigurationError::InvalidValue {
                key: "verification.max_concurrent_full_validations".to_string(),
                value: self
                    .verification
                    .max_concurrent_full_validations
                    .to_string(),
                reason: "Must allow at least 1 concurrent full validation".to_string(),
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

        // Validate billing configuration
        if self.billing.enabled {
            if self.billing.billing_endpoint.is_empty() {
                return Err(ConfigurationError::InvalidValue {
                    key: "billing.billing_endpoint".to_string(),
                    value: self.billing.billing_endpoint.clone(),
                    reason: "Billing endpoint cannot be empty when billing is enabled".to_string(),
                });
            }

            if self.billing.timeout_secs == 0 {
                return Err(ConfigurationError::InvalidValue {
                    key: "billing.timeout_secs".to_string(),
                    value: self.billing.timeout_secs.to_string(),
                    reason: "Timeout must be greater than 0".to_string(),
                });
            }

            if self.billing.batch_size == 0 {
                return Err(ConfigurationError::InvalidValue {
                    key: "billing.batch_size".to_string(),
                    value: self.billing.batch_size.to_string(),
                    reason: "Batch size must be greater than 0".to_string(),
                });
            }

            if self.billing.collection_interval_secs == 0 {
                return Err(ConfigurationError::InvalidValue {
                    key: "billing.collection_interval_secs".to_string(),
                    value: self.billing.collection_interval_secs.to_string(),
                    reason: "Collection interval must be greater than 0".to_string(),
                });
            }

            if self.billing.flush_interval_secs == 0 {
                return Err(ConfigurationError::InvalidValue {
                    key: "billing.flush_interval_secs".to_string(),
                    value: self.billing.flush_interval_secs.to_string(),
                    reason: "Flush interval must be greater than 0".to_string(),
                });
            }
        }

        // Validate binary validation configuration
        if self.verification.binary_validation.enabled {
            let validator_path = &self.verification.binary_validation.validator_binary_path;
            if !validator_path.exists() {
                return Err(ConfigurationError::InvalidValue {
                    key: "verification.binary_validation.validator_binary_path".to_string(),
                    value: validator_path.display().to_string(),
                    reason: "Validator binary path does not exist".to_string(),
                });
            }

            let executor_path = &self.verification.binary_validation.executor_binary_path;
            if !executor_path.exists() {
                return Err(ConfigurationError::InvalidValue {
                    key: "verification.binary_validation.executor_binary_path".to_string(),
                    value: executor_path.display().to_string(),
                    reason: "Executor binary path does not exist".to_string(),
                });
            }
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
    pub fn load_from_file(path: &Path) -> Result<Self> {
        Ok(loader::load_from_file::<Self>(path)?)
    }
}

// TODO: Add configuration hot-reloading capabilities
// TODO: Add configuration encryption for sensitive values
// TODO: Add configuration templating with variable substitution
// TODO: Add configuration schema validation
// TODO: Add configuration migration between versions
