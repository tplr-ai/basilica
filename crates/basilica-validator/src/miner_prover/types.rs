//! # Types for Miner Verification
//!
//! Shared data structures used across the miner verification system.

use basilica_common::identity::{Hotkey, MinerUid, NodeId};
use serde::{Deserialize, Serialize};
use std::time::Duration;

/// Information about a miner being verified
#[derive(Debug, Clone)]
pub struct MinerInfo {
    pub uid: MinerUid,
    pub hotkey: Hotkey,
    pub endpoint: String,
    pub is_validator: bool,
    pub stake_tao: f64,
    pub last_verified: Option<chrono::DateTime<chrono::Utc>>,
    pub verification_score: f64,
}

/// Information about a node available for verification
#[derive(Debug, Clone)]
pub struct NodeInfo {
    pub id: NodeId, // Using NodeId alias which maps to NodeId
    pub miner_uid: MinerUid,
    pub node_ssh_endpoint: String,
}

#[derive(Debug, Clone)]
pub struct VerificationStats {
    pub active_verifications: usize,
    pub max_concurrent: usize,
}

/// Validation error types
#[derive(Debug, thiserror::Error)]
pub enum ValidationError {
    #[error("Attestation validation failed: {0}")]
    AttestationValidationFailed(String),
    #[error("IO error: {0}")]
    IoError(#[from] std::io::Error),
    #[error("Serialization error: {0}")]
    SerializationError(#[from] serde_json::Error),
    #[error("Configuration error: {0}")]
    ConfigError(String),
    #[error("Cryptographic error: {0}")]
    CryptoError(String),
    #[error("SSH connection error: {0}")]
    SshError(#[from] anyhow::Error),
    #[error("Signature verification failed: {0}")]
    SignatureVerificationFailed(String),
    #[error("Integrity check failed: {0}")]
    IntegrityCheckFailed(String),
    #[error("Remote command execution failed: {0}")]
    ExecutionFailed(String),
    #[error("Database error: {0}")]
    DatabaseError(String),
    #[error("Parse error: {0}")]
    ParseError(String),
}

pub type ValidationResult<T> = Result<T, ValidationError>;

/// Validation type enum to distinguish between full and lightweight validation
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum ValidationType {
    Full,
    Lightweight,
}

impl std::fmt::Display for ValidationType {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        match self {
            ValidationType::Full => write!(f, "full"),
            ValidationType::Lightweight => write!(f, "lightweight"),
        }
    }
}

/// Enhanced node verification result
#[derive(Debug, Clone)]
pub struct NodeVerificationResult {
    pub node_id: NodeId, // Using NodeId alias which maps to NodeId
    pub node_ssh_endpoint: String,
    pub verification_score: f64,
    pub ssh_connection_successful: bool,
    pub binary_validation_successful: bool,
    pub node_result: Option<NodeResult>,
    pub error: Option<String>,
    pub execution_time: Duration,
    pub validation_details: ValidationDetails,
    pub gpu_count: u64,
    pub validation_type: ValidationType,
}

/// Detailed validation timing and scoring information
#[derive(Debug, Clone)]
pub struct ValidationDetails {
    pub ssh_test_duration: Duration,
    pub binary_upload_duration: Duration,
    pub binary_execution_duration: Duration,
    pub total_validation_duration: Duration,
    pub ssh_score: f64,
    pub binary_score: f64,
    pub combined_score: f64,
}

/// GPU node result from binary validation
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct NodeResult {
    pub gpu_name: String,
    pub gpu_uuid: String,
    pub gpu_infos: Vec<GpuInfo>,
    pub cpu_info: BinaryCpuInfo,
    pub memory_info: BinaryMemoryInfo,
    pub network_info: BinaryNetworkInfo,
    pub matrix_c: CompressedMatrix,
    pub computation_time_ns: u64,
    pub checksum: [u8; 32],
    pub sm_utilization: SmUtilizationStats,
    pub active_sms: u32,
    pub total_sms: u32,
    pub memory_bandwidth_gbps: f64,
    pub anti_debug_passed: bool,
    pub timing_fingerprint: u64,
}

/// GPU information from verification
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct GpuInfo {
    pub index: u32,
    pub gpu_name: String,
    pub gpu_uuid: String,
    pub gpu_memory_gb: f64,
    pub computation_time_ns: u64,
    pub memory_bandwidth_gbps: f64,
    pub sm_utilization: SmUtilizationStats,
    pub active_sms: u32,
    pub total_sms: u32,
    pub anti_debug_passed: bool,
}

/// CPU information for binary validation
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct BinaryCpuInfo {
    pub model: String,
    pub cores: u32,
    pub threads: u32,
    pub frequency_mhz: u32,
}

/// Memory information for binary validation
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct BinaryMemoryInfo {
    pub total_gb: f64,
    pub available_gb: f64,
}

/// Network information for binary validation
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct BinaryNetworkInfo {
    pub interfaces: Vec<NetworkInterface>,
}

/// Network interface information
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct NetworkInterface {
    pub name: String,
    pub mac_address: String,
    pub ip_addresses: Vec<String>,
}

/// Compressed matrix data
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct CompressedMatrix {
    pub rows: u32,
    pub cols: u32,
    pub data: Vec<f64>,
}

/// SM (Streaming Multiprocessor) utilization statistics
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct SmUtilizationStats {
    pub min_utilization: f64,
    pub max_utilization: f64,
    pub avg_utilization: f64,
    pub per_sm_stats: Vec<SmStat>,
}

/// Individual SM statistics
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct SmStat {
    pub sm_id: u32,
    pub utilization: f64,
    pub active_warps: u32,
    pub max_warps: u32,
}

/// Detailed node information for verification processes
#[derive(Debug, Clone)]
pub struct NodeInfoDetailed {
    pub id: NodeId,
    pub miner_uid: MinerUid,
    pub status: String,
    pub capabilities: Vec<String>,
    pub node_ssh_endpoint: String,
}

/// Output from validator binary execution
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct ValidatorBinaryOutput {
    pub success: bool,
    pub node_result: Option<NodeResult>,
    pub error_message: Option<String>,
    pub execution_time_ms: u64,
    pub validation_score: f64,
    pub gpu_count: u64,
}
