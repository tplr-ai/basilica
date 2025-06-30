//! API Types and Data Transfer Objects
//!
//! All request/response types, enums, and shared data structures for the validator API

use serde::{Deserialize, Serialize};
use std::collections::HashMap;

/// Request to rent GPU capacity
#[derive(Debug, Deserialize)]
pub struct RentCapacityRequest {
    pub gpu_requirements: GpuRequirements,
    pub ssh_public_key: String,
    pub docker_image: String,
    pub env_vars: Option<HashMap<String, String>>,
    pub max_duration_hours: u32,
}

#[derive(Debug, Deserialize)]
pub struct GpuRequirements {
    pub min_memory_gb: u32,
    pub gpu_type: Option<String>,
    pub gpu_count: u32,
}

/// Response for capacity rental request
#[derive(Debug, Serialize)]
pub struct RentCapacityResponse {
    pub rental_id: String,
    pub executor: ExecutorDetails,
    pub ssh_access: SshAccess,
    pub cost_per_hour: f64,
}

#[derive(Debug, Serialize)]
pub struct ExecutorDetails {
    pub id: String,
    pub gpu_specs: Vec<GpuSpec>,
    pub cpu_specs: CpuSpec,
    pub location: Option<String>,
}

#[derive(Debug, Serialize, Deserialize, Clone)]
pub struct GpuSpec {
    pub name: String,
    pub memory_gb: u32,
    pub compute_capability: String,
}

#[derive(Debug, Serialize, Deserialize, Clone)]
pub struct CpuSpec {
    pub cores: u32,
    pub model: String,
    pub memory_gb: u32,
}

#[derive(Debug, Serialize)]
pub struct SshAccess {
    pub host: String,
    pub port: u16,
    pub username: String,
}

/// Request to terminate a rental
#[derive(Debug, Deserialize)]
pub struct TerminateRentalRequest {
    pub reason: Option<String>,
}

/// Response for rental termination
#[derive(Debug, Serialize)]
pub struct TerminateRentalResponse {
    pub success: bool,
    pub message: String,
}

/// Rental status information
#[derive(Debug, Serialize)]
pub struct RentalStatusResponse {
    pub rental_id: String,
    pub status: RentalStatus,
    pub executor: ExecutorDetails,
    pub created_at: chrono::DateTime<chrono::Utc>,
    pub updated_at: chrono::DateTime<chrono::Utc>,
    pub cost_incurred: f64,
}

#[derive(Debug, Serialize)]
pub enum RentalStatus {
    Pending,
    Active,
    Terminated,
    Failed,
}

/// Available capacity listing
#[derive(Debug, Serialize)]
pub struct ListCapacityResponse {
    pub available_executors: Vec<AvailableExecutor>,
    pub total_count: usize,
}

#[derive(Debug, Serialize)]
pub struct AvailableExecutor {
    pub executor: ExecutorDetails,
    pub availability: AvailabilityInfo,
    pub cost_per_hour: f64,
}

#[derive(Debug, Serialize)]
pub struct AvailabilityInfo {
    pub available_until: Option<chrono::DateTime<chrono::Utc>>,
    pub verification_score: f64,
    pub uptime_percentage: f64,
}

/// Query parameters for capacity listing
#[derive(Debug, Deserialize)]
pub struct ListCapacityQuery {
    pub min_gpu_memory: Option<u32>,
    pub gpu_type: Option<String>,
    pub min_gpu_count: Option<u32>,
    pub max_cost_per_hour: Option<f64>,
}

/// Log streaming query parameters
#[derive(Debug, Deserialize)]
pub struct LogQuery {
    pub follow: Option<bool>,
    pub tail: Option<u32>,
}

/// Miner registration request
#[derive(Debug, Deserialize)]
pub struct RegisterMinerRequest {
    pub miner_id: String,
    pub hotkey: String,
    pub endpoint: String,
    pub signature: String,
    pub executors: Vec<ExecutorRegistration>,
}

#[derive(Debug, Deserialize, Serialize)]
pub struct ExecutorRegistration {
    pub executor_id: String,
    pub grpc_address: String,
    pub gpu_count: u32,
    pub gpu_specs: Vec<GpuSpec>,
    pub cpu_specs: CpuSpec,
}

/// Miner registration response
#[derive(Debug, Serialize)]
pub struct RegisterMinerResponse {
    pub success: bool,
    pub miner_id: String,
    pub message: String,
}

/// Miner details for listing
#[derive(Debug, Serialize)]
pub struct MinerDetails {
    pub miner_id: String,
    pub hotkey: String,
    pub endpoint: String,
    pub status: MinerStatus,
    pub executor_count: u32,
    pub total_gpu_count: u32,
    pub verification_score: f64,
    pub uptime_percentage: f64,
    pub last_seen: chrono::DateTime<chrono::Utc>,
    pub registered_at: chrono::DateTime<chrono::Utc>,
}

/// Miner status enumeration
#[derive(Debug, Serialize)]
pub enum MinerStatus {
    Active,
    Inactive,
    Offline,
    Verifying,
    Suspended,
}

/// List miners response
#[derive(Debug, Serialize)]
pub struct ListMinersResponse {
    pub miners: Vec<MinerDetails>,
    pub total_count: usize,
    pub page: u32,
    pub page_size: u32,
}

/// Query parameters for miner listing
#[derive(Debug, Deserialize)]
pub struct ListMinersQuery {
    pub status: Option<String>,
    pub min_gpu_count: Option<u32>,
    pub min_score: Option<f64>,
    pub page: Option<u32>,
    pub page_size: Option<u32>,
}

/// Miner update request
#[derive(Debug, Deserialize)]
pub struct UpdateMinerRequest {
    pub endpoint: Option<String>,
    pub signature: String,
    pub executors: Option<Vec<ExecutorRegistration>>,
}

/// Miner health status response
#[derive(Debug, Serialize)]
pub struct MinerHealthResponse {
    pub miner_id: String,
    pub overall_status: MinerStatus,
    pub last_health_check: chrono::DateTime<chrono::Utc>,
    pub executor_health: Vec<ExecutorHealthStatus>,
    pub response_time_ms: u64,
}

#[derive(Debug, Serialize)]
pub struct ExecutorHealthStatus {
    pub executor_id: String,
    pub status: String,
    pub last_seen: chrono::DateTime<chrono::Utc>,
    pub gpu_utilization: f64,
    pub memory_usage: f64,
}

/// Verification trigger request
#[derive(Debug, Deserialize)]
pub struct TriggerVerificationRequest {
    pub verification_type: String,
    pub executor_id: Option<String>,
}

/// Verification trigger response
#[derive(Debug, Serialize)]
pub struct TriggerVerificationResponse {
    pub verification_id: String,
    pub status: String,
    pub estimated_completion: chrono::DateTime<chrono::Utc>,
}

/// API error type
#[derive(Debug)]
pub enum ApiError {
    NotFound(String),
    BadRequest(String),
    Unauthorized,
    InternalError(String),
}

impl axum::response::IntoResponse for ApiError {
    fn into_response(self) -> axum::response::Response {
        use axum::http::StatusCode;
        use axum::Json;

        let (status, message) = match self {
            ApiError::NotFound(msg) => (StatusCode::NOT_FOUND, msg),
            ApiError::BadRequest(msg) => (StatusCode::BAD_REQUEST, msg),
            ApiError::Unauthorized => (StatusCode::UNAUTHORIZED, "Unauthorized".to_string()),
            ApiError::InternalError(msg) => (StatusCode::INTERNAL_SERVER_ERROR, msg),
        };

        let body = Json(serde_json::json!({
            "error": message,
            "timestamp": chrono::Utc::now()
        }));

        (status, body).into_response()
    }
}
