//! API types for the Public API Gateway

use serde::{Deserialize, Serialize};
use std::collections::HashMap;
use utoipa::ToSchema;

// Define types that mirror validator API types

/// GPU requirements
#[derive(Debug, Clone, Deserialize, Serialize, ToSchema)]
pub struct GpuRequirements {
    /// Minimum memory in GB
    pub min_memory_gb: u32,

    /// GPU type (optional)
    pub gpu_type: Option<String>,

    /// Number of GPUs required
    pub gpu_count: u32,
}

/// GPU specifications
#[derive(Debug, Clone, Serialize, Deserialize, ToSchema)]
pub struct GpuSpec {
    /// GPU name
    pub name: String,

    /// Memory in GB
    pub memory_gb: u32,

    /// Compute capability
    pub compute_capability: String,
}

/// CPU specifications
#[derive(Debug, Clone, Serialize, Deserialize, ToSchema)]
pub struct CpuSpec {
    /// Number of cores
    pub cores: u32,

    /// CPU model
    pub model: String,

    /// Memory in GB
    pub memory_gb: u32,
}

/// SSH access information
#[derive(Debug, Clone, Serialize, Deserialize, ToSchema)]
pub struct SshAccess {
    /// Host address
    pub host: String,

    /// SSH port
    pub port: u16,

    /// Username
    pub username: String,
}

/// Rental status
#[derive(Debug, Clone, Serialize, Deserialize, ToSchema)]
#[serde(rename_all = "lowercase")]
pub enum RentalStatus {
    /// Rental is pending
    Pending,

    /// Rental is active
    Active,

    /// Rental is terminated
    Terminated,

    /// Rental failed
    Failed,
}

/// Request to rent GPU capacity
#[derive(Debug, Deserialize, Serialize, ToSchema)]
pub struct RentCapacityRequest {
    /// GPU requirements
    pub gpu_requirements: GpuRequirements,

    /// SSH public key for access
    pub ssh_public_key: String,

    /// Docker image to run
    pub docker_image: String,

    /// Environment variables
    pub env_vars: Option<HashMap<String, String>>,

    /// Maximum rental duration in hours
    pub max_duration_hours: u32,
}

/// Response for capacity rental request
#[derive(Debug, Serialize, Deserialize, ToSchema)]
pub struct RentCapacityResponse {
    /// Rental ID
    pub rental_id: String,

    /// Executor details
    pub executor: ExecutorDetails,

    /// SSH access information
    pub ssh_access: SshAccess,

    /// Cost per hour
    pub cost_per_hour: f64,
}

/// Executor details
#[derive(Debug, Clone, Serialize, Deserialize, ToSchema)]
pub struct ExecutorDetails {
    /// Executor ID
    pub id: String,

    /// GPU specifications
    pub gpu_specs: Vec<GpuSpec>,

    /// CPU specifications
    pub cpu_specs: CpuSpec,

    /// Location (optional)
    pub location: Option<String>,
}

/// Rental status response
#[derive(Debug, Serialize, Deserialize, ToSchema)]
pub struct RentalStatusResponse {
    /// Rental ID
    pub rental_id: String,

    /// Current status
    pub status: RentalStatus,

    /// Executor details
    pub executor: ExecutorDetails,

    /// Creation timestamp
    pub created_at: chrono::DateTime<chrono::Utc>,

    /// Last update timestamp
    pub updated_at: chrono::DateTime<chrono::Utc>,

    /// Cost incurred so far
    pub cost_incurred: f64,
}

/// Request to terminate a rental
#[derive(Debug, Serialize, Deserialize, ToSchema)]
pub struct TerminateRentalRequest {
    /// Reason for termination (optional)
    pub reason: Option<String>,
}

/// Response for rental termination
#[derive(Debug, Serialize, Deserialize, ToSchema)]
pub struct TerminateRentalResponse {
    /// Success flag
    pub success: bool,

    /// Message
    pub message: String,
}

/// Log query parameters
#[derive(Debug, Deserialize)]
pub struct LogQuery {
    /// Follow logs
    pub follow: Option<bool>,

    /// Number of lines to tail
    pub tail: Option<u32>,
}

/// List executors query parameters
#[derive(Debug, Deserialize)]
pub struct ListExecutorsQuery {
    /// Minimum GPU count
    pub min_gpu_count: Option<u32>,

    /// GPU type filter
    pub gpu_type: Option<String>,

    /// Page number
    pub page: Option<u32>,

    /// Page size
    pub page_size: Option<u32>,
}

/// List executors response
#[derive(Debug, Serialize, Deserialize, ToSchema)]
pub struct ListExecutorsResponse {
    /// List of executors
    pub executors: Vec<ExecutorDetails>,

    /// Total count
    pub total_count: usize,

    /// Current page
    pub page: u32,

    /// Page size
    pub page_size: u32,
}

/// Validator details
#[derive(Debug, Clone, Serialize, Deserialize, ToSchema)]
pub struct ValidatorDetails {
    /// Validator UID
    pub uid: u16,

    /// Validator hotkey
    pub hotkey: String,

    /// Validator endpoint
    pub endpoint: String,

    /// Validator score
    pub score: f64,

    /// Is healthy
    pub is_healthy: bool,

    /// Last health check
    pub last_health_check: Option<chrono::DateTime<chrono::Utc>>,
}

/// List validators response
#[derive(Debug, Serialize, Deserialize, ToSchema)]
pub struct ListValidatorsResponse {
    /// List of validators
    pub validators: Vec<ValidatorDetails>,

    /// Total count
    pub total_count: usize,
}

/// Miner details
#[derive(Debug, Clone, Serialize, Deserialize, ToSchema)]
pub struct MinerDetails {
    /// Miner ID
    pub miner_id: String,

    /// Miner hotkey
    pub hotkey: String,

    /// Miner endpoint
    pub endpoint: String,

    /// Number of executors
    pub executor_count: u32,

    /// Total GPU count
    pub total_gpu_count: u32,

    /// Verification score
    pub verification_score: f64,

    /// Last seen timestamp
    pub last_seen: chrono::DateTime<chrono::Utc>,
}

/// List miners query parameters
#[derive(Debug, Deserialize)]
pub struct ListMinersQuery {
    /// Minimum GPU count filter
    pub min_gpu_count: Option<u32>,

    /// Minimum score filter
    pub min_score: Option<f64>,

    /// Page number
    pub page: Option<u32>,

    /// Page size
    pub page_size: Option<u32>,
}

/// List miners response
#[derive(Debug, Serialize, Deserialize, ToSchema)]
pub struct ListMinersResponse {
    /// List of miners
    pub miners: Vec<MinerDetails>,

    /// Total count
    pub total_count: usize,

    /// Current page
    pub page: u32,

    /// Page size
    pub page_size: u32,
}

/// Health check response
#[derive(Debug, Serialize, ToSchema)]
pub struct HealthCheckResponse {
    /// Service status
    pub status: String,

    /// Service version
    pub version: String,

    /// Timestamp
    pub timestamp: chrono::DateTime<chrono::Utc>,

    /// Healthy validators count
    pub healthy_validators: usize,

    /// Total validators count
    pub total_validators: usize,
}

/// Telemetry response
#[derive(Debug, Serialize, ToSchema)]
pub struct TelemetryResponse {
    /// Request count
    pub request_count: u64,

    /// Average response time (ms)
    pub avg_response_time_ms: f64,

    /// Success rate
    pub success_rate: f64,

    /// Active connections
    pub active_connections: usize,

    /// Cache hit rate
    pub cache_hit_rate: f64,

    /// Timestamp
    pub timestamp: chrono::DateTime<chrono::Utc>,
}

/// API key info (for authenticated requests)
#[derive(Debug, Clone)]
pub struct ApiKeyInfo {
    /// API key ID
    pub key_id: String,

    /// API key tier
    pub tier: ApiKeyTier,

    /// Rate limit override
    pub rate_limit_override: Option<u32>,
}

/// API key tiers
#[derive(Debug, Clone, PartialEq)]
pub enum ApiKeyTier {
    /// Free tier
    Free,

    /// Premium tier
    Premium,

    /// Enterprise tier
    Enterprise,
}
