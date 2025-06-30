//! Validation Types
//!
//! Data types and structures for hardware validation.

use common::identity::ExecutorId;
use serde::{Deserialize, Serialize};
use std::path::PathBuf;
use std::time::{Duration, SystemTime};

/// GPU attestation result from executor validation
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct AttestationResult {
    /// Executor that was validated
    pub executor_id: ExecutorId,
    /// Validation timestamp
    pub validated_at: SystemTime,
    /// Whether validation was successful
    pub is_valid: bool,
    /// Hardware specifications from attestation
    pub hardware_specs: Option<HardwareSpecs>,
    /// Attestation signature
    pub signature: Option<String>,
    /// Error message if validation failed
    pub error_message: Option<String>,
    /// Time taken for validation
    pub validation_duration: Duration,
}

/// Hardware specifications extracted from attestation
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct HardwareSpecs {
    /// CPU information
    pub cpu: CpuInfo,
    /// GPU information
    pub gpu: Vec<GpuInfo>,
    /// Memory information
    pub memory: MemoryInfo,
    /// Storage information
    pub storage: StorageInfo,
    /// Network performance
    pub network: NetworkInfo,
    /// Docker environment status
    pub docker_status: DockerStatus,
}

/// CPU information
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct CpuInfo {
    /// CPU model
    pub model: String,
    /// Number of cores
    pub cores: u32,
    /// CPU frequency in MHz
    pub frequency_mhz: u32,
    /// CPU architecture
    pub architecture: String,
}

/// GPU information
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct GpuInfo {
    /// GPU vendor (NVIDIA, AMD, Intel)
    pub vendor: String,
    /// GPU model
    pub model: String,
    /// VRAM in MB
    pub vram_mb: u64,
    /// GPU driver version
    pub driver_version: String,
    /// CUDA compute capability (if applicable)
    pub compute_capability: Option<String>,
    /// GPU utilization percentage
    pub utilization_percent: Option<f32>,
}

/// Memory information
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct MemoryInfo {
    /// Total RAM in MB
    pub total_mb: u64,
    /// Available RAM in MB
    pub available_mb: u64,
    /// Memory type (DDR4, DDR5, etc.)
    pub memory_type: String,
}

/// Storage information
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct StorageInfo {
    /// Total storage in GB
    pub total_gb: u64,
    /// Available storage in GB
    pub available_gb: u64,
    /// Storage type (SSD, HDD, NVMe)
    pub storage_type: String,
    /// Disk I/O performance metrics
    pub io_performance: Option<IoPerformance>,
}

/// Disk I/O performance metrics
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct IoPerformance {
    /// Read speed in MB/s
    pub read_speed_mbps: f64,
    /// Write speed in MB/s
    pub write_speed_mbps: f64,
    /// IOPS (Input/Output Operations Per Second)
    pub iops: u32,
}

/// Network performance information
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct NetworkInfo {
    /// Bandwidth in Mbps
    pub bandwidth_mbps: f64,
    /// Network latency in milliseconds
    pub latency_ms: f64,
    /// Packet loss percentage
    pub packet_loss_percent: f32,
}

/// Docker environment status
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct DockerStatus {
    /// Whether Docker is installed
    pub is_installed: bool,
    /// Docker version
    pub version: String,
    /// Whether Docker daemon is running
    pub daemon_running: bool,
    /// Whether NVIDIA Docker runtime is available
    pub nvidia_runtime_available: bool,
    /// Available Docker images
    pub available_images: Vec<String>,
}

/// Validation configuration
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct ValidationConfig {
    /// Path to gpu-attestor binary
    pub gpu_attestor_binary_path: PathBuf,
    /// Remote working directory for validation
    pub remote_work_dir: String,
    /// Command execution timeout
    pub execution_timeout: Duration,
    /// Maximum file transfer size in bytes
    pub max_transfer_size: u64,
    /// Whether to keep remote files after validation
    pub cleanup_remote_files: bool,
    /// Default SSH private key path for executor connections
    pub default_ssh_key_path: Option<PathBuf>,
}

impl Default for ValidationConfig {
    fn default() -> Self {
        Self {
            gpu_attestor_binary_path: PathBuf::from("./gpu-attestor"),
            remote_work_dir: "/tmp/basilica_validation".to_string(),
            execution_timeout: Duration::from_secs(300),
            max_transfer_size: 100 * 1024 * 1024, // 100MB
            cleanup_remote_files: true,
            default_ssh_key_path: None,
        }
    }
}

/// Validation error types
#[derive(Debug, thiserror::Error)]
pub enum ValidationError {
    /// Attestation validation failed
    #[error("Attestation validation failed: {0}")]
    AttestationValidationFailed(String),

    /// IO error
    #[error("IO error: {0}")]
    IoError(#[from] std::io::Error),

    /// Serialization error
    #[error("Serialization error: {0}")]
    SerializationError(#[from] serde_json::Error),

    /// Configuration error
    #[error("Configuration error: {0}")]
    ConfigError(String),

    /// Binary not found
    #[error("GPU attestor binary not found at: {0}")]
    BinaryNotFound(PathBuf),

    /// SSH connection error
    #[error("SSH connection error: {0}")]
    SshError(#[from] anyhow::Error),

    /// Signature verification failed
    #[error("Signature verification failed: {0}")]
    SignatureVerificationFailed(String),

    /// Integrity check failed
    #[error("Integrity check failed: {0}")]
    IntegrityCheckFailed(String),

    /// Remote command execution failed
    #[error("Remote command execution failed: {0}")]
    ExecutionFailed(String),

    /// Database operation failed
    #[error("Database error: {0}")]
    DatabaseError(String),

    /// Parse error
    #[error("Parse error: {0}")]
    ParseError(String),
}

pub type ValidationResult<T> = Result<T, ValidationError>;

// ============================================================================
// GPU Attestor Integration Types
// ============================================================================

/// Attestation report from gpu-attestor
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct AttestationReport {
    /// Report version
    pub version: String,
    /// Timestamp when report was generated
    pub timestamp: String,
    /// Executor ID
    pub executor_id: String,
    /// Binary information
    pub binary_info: Option<BinaryInfo>,
    /// GPU information array
    pub gpu_info: Vec<GpuAttestorGpu>,
    /// System information
    pub system_info: SystemInfo,
    /// Network benchmark results
    pub network_benchmark: Option<NetworkBenchmark>,
    /// VDF proof results
    pub vdf_proof: Option<VdfProof>,
    /// Validator-provided nonce for replay protection
    #[serde(skip_serializing_if = "Option::is_none")]
    pub validator_nonce: Option<String>,
}

/// Binary information from attestation
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct BinaryInfo {
    /// Binary path
    pub path: String,
    /// Whether signature was verified
    pub signature_verified: bool,
    /// Validator public key fingerprint
    pub validator_public_key_fingerprint: String,
}

/// GPU information from gpu-attestor
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct GpuAttestorGpu {
    /// GPU vendor
    pub vendor: String,
    /// GPU name/model
    pub name: String,
    /// Total memory in bytes
    pub memory_total: u64,
    /// Driver version
    pub driver_version: String,
    /// Temperature in Celsius
    pub temperature: Option<f32>,
    /// GPU utilization percentage
    pub utilization: Option<f32>,
}

/// System information from gpu-attestor
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct SystemInfo {
    /// CPU information
    pub cpu: CpuAttestorInfo,
    /// Memory information
    pub memory: MemoryAttestorInfo,
    /// Docker information
    pub docker: DockerAttestorInfo,
}

/// CPU information from gpu-attestor
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct CpuAttestorInfo {
    /// Number of cores
    pub cores: u32,
    /// Number of threads
    pub threads: u32,
    /// CPU brand/model
    pub brand: String,
}

/// Memory information from gpu-attestor
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct MemoryAttestorInfo {
    /// Total memory in bytes
    pub total_bytes: u64,
}

/// Docker information from gpu-attestor
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct DockerAttestorInfo {
    /// Whether Docker is running
    pub is_running: bool,
    /// Docker version
    pub version: String,
}

/// Network benchmark results
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct NetworkBenchmark {
    /// Latency test results
    pub latency_tests: Vec<LatencyTest>,
    /// Throughput test results
    pub throughput_tests: Vec<ThroughputTest>,
    /// DNS resolution test
    pub dns_resolution_test: DnsTest,
}

/// Latency test result
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct LatencyTest {
    /// Target host
    pub target: String,
    /// Average latency in milliseconds
    pub avg_latency_ms: f64,
    /// Packet loss percentage
    pub packet_loss_percent: f32,
}

/// Throughput test result
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct ThroughputTest {
    /// Test direction (upload/download)
    pub direction: String,
    /// Throughput in Mbps
    pub throughput_mbps: f64,
}

/// DNS resolution test
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct DnsTest {
    /// Whether DNS resolution succeeded
    pub success: bool,
    /// Resolution time in milliseconds
    pub resolution_time_ms: f64,
}

/// VDF proof from gpu-attestor
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct VdfProof {
    /// Computation time in milliseconds
    pub computation_time_ms: u64,
    /// VDF algorithm used
    pub algorithm: String,
}

// ============================================================================
// Key Management Types
// ============================================================================

/// Ephemeral key information
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct EphemeralKey {
    /// Key identifier
    pub key_id: String,
    /// Public key in hex format
    pub public_key_hex: String,
    /// Key creation timestamp
    pub created_at: SystemTime,
    /// Key expiration timestamp
    pub expires_at: SystemTime,
    /// Whether this key is currently active
    pub is_active: bool,
}

/// Key rotation configuration
#[derive(Debug, Clone)]
pub struct KeyRotationConfig {
    /// How often to rotate keys
    pub rotation_interval: Duration,
    /// How long to keep old keys for verification
    pub key_retention_period: Duration,
    /// Maximum number of keys to keep
    pub max_keys: usize,
    /// Directory to store key metadata
    pub key_storage_dir: PathBuf,
}

impl Default for KeyRotationConfig {
    fn default() -> Self {
        Self {
            rotation_interval: Duration::from_secs(3600), // 1 hour
            key_retention_period: Duration::from_secs(7200), // 2 hours
            max_keys: 10,
            key_storage_dir: PathBuf::from("./keys"),
        }
    }
}

// ============================================================================
// Signature Verification Types
// ============================================================================

/// P256 public key information
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct P256PublicKey {
    /// Key identifier
    pub key_id: String,
    /// Compressed public key (33 bytes) in hex format
    pub compressed_key_hex: String,
    /// Key creation timestamp
    pub created_at: SystemTime,
    /// Whether this key is trusted for verification
    pub is_trusted: bool,
}

/// Signature verification result
#[derive(Debug, Clone)]
pub struct SignatureVerificationResult {
    /// Whether the signature is valid
    pub is_valid: bool,
    /// Key ID used for verification
    pub key_id: String,
    /// Timestamp of verification
    pub verified_at: SystemTime,
    /// Error message if verification failed
    pub error_message: Option<String>,
}
