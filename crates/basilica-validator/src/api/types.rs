//! API Types and Data Transfer Objects
//!
//! All request/response types, enums, and shared data structures for the validator API

pub use crate::node_types::{CpuSpec, GpuSpec, NetworkSpeedInfo, NodeDetails};
#[allow(unused_imports)]
pub use basilica_common::validator_api::{
    AvailabilityInfo, AvailableNode, ContainerInfo as ApiContainerInfo, GpuRequirements,
    ListAvailableNodesQuery, ListAvailableNodesResponse, ListRentalsQuery, ListRentalsResponse,
    LogQuery, PortMapping as ApiPortMapping, PortMappingRequest, RentCapacityRequest,
    RentCapacityResponse, RentalListItem, RentalResponse as ApiRentalResponse,
    RentalRestartResponse as ApiRentalRestartResponse, RentalState as ApiRentalState, RentalStatus,
    RentalStatusQuery, RentalStatusResponse, ResourceRequirementsRequest, SshAccess,
    StartRentalRequest, TerminateRentalRequest, VolumeMountRequest,
};
use serde::{Deserialize, Serialize};
use serde_json::Value;
use std::collections::HashMap;

/// Log streaming query parameters
#[derive(Debug, Deserialize)]
pub struct LogStreamQuery {
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
    pub nodes: Vec<NodeRegistration>,
}

#[derive(Debug, Deserialize, Serialize)]
pub struct NodeRegistration {
    pub node_id: String,
    pub ssh_endpoint: String,
    pub node_ip: String,
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
    pub node_count: u32,
    pub updated_at: chrono::DateTime<chrono::Utc>,
}

/// Miner status enumeration
#[derive(Debug, Serialize)]
pub enum MinerStatus {
    Active,
    Inactive,
    Offline,
    Verifying,
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
    pub page: Option<u32>,
    pub page_size: Option<u32>,
}

/// Miner update request
#[derive(Debug, Deserialize)]
pub struct UpdateMinerRequest {
    pub endpoint: Option<String>,
    pub signature: String,
    pub nodes: Option<Vec<NodeRegistration>>,
}

/// Miner health status response
#[derive(Debug, Serialize)]
pub struct MinerHealthResponse {
    pub miner_id: String,
    pub overall_status: MinerStatus,
    pub last_health_check: chrono::DateTime<chrono::Utc>,
    pub node_health: Vec<NodeHealthStatus>,
    pub response_time_ms: u64,
}

#[derive(Debug, Serialize)]
pub struct NodeHealthStatus {
    pub node_id: String,
    pub status: String,
    pub last_health_check: chrono::DateTime<chrono::Utc>,
}

/// Verification trigger request
#[derive(Debug, Deserialize)]
pub struct TriggerVerificationRequest {
    pub verification_type: String,
    pub node_id: Option<String>,
}

/// Verification trigger response
#[derive(Debug, Serialize)]
pub struct TriggerVerificationResponse {
    pub verification_id: String,
    pub status: String,
    pub estimated_completion: chrono::DateTime<chrono::Utc>,
}

/// Verification log item returned by the API.
#[derive(Debug, Serialize, Deserialize)]
pub struct VerificationLogResponse {
    pub id: String,
    pub node_id: String,
    pub validator_hotkey: String,
    pub verification_type: String,
    pub timestamp: chrono::DateTime<chrono::Utc>,
    pub score: f64,
    pub success: bool,
    pub details: Value,
    pub duration_ms: i64,
    pub error_message: Option<String>,
    pub created_at: chrono::DateTime<chrono::Utc>,
    pub updated_at: chrono::DateTime<chrono::Utc>,
}

/// Verification log listing response.
#[derive(Debug, Serialize, Deserialize)]
pub struct VerificationLogsResponse {
    pub logs: Vec<VerificationLogResponse>,
    pub total_count: usize,
}

/// Emission metrics response
#[derive(Debug, Serialize)]
pub struct EmissionMetricsResponse {
    pub id: i64,
    pub timestamp: chrono::DateTime<chrono::Utc>,
    pub burn_amount: u64,
    pub burn_percentage: f64,
    pub category_distributions: HashMap<String, CategoryDistributionResponse>,
    pub total_miners: u32,
    pub weight_set_block: u64,
}

#[derive(Debug, Serialize)]
pub struct CategoryDistributionResponse {
    pub category: String,
    pub miner_count: u32,
    pub total_weight: u64,
    pub average_score: f64,
}

#[derive(Debug, Serialize)]
pub struct MinerWeightAllocation {
    pub miner_uid: u16,
    pub gpu_category: String,
    pub allocated_weight: u64,
    pub miner_score: f64,
    pub percentage_of_category: f64,
}

#[derive(Debug, Serialize)]
pub struct CategoryWeightSummary {
    pub category: String,
    pub total_weight: u64,
    pub miner_count: u32,
    pub average_score: f64,
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
