//! TEE (Trusted Execution Environment) API routes for the validator
//!
//! Provides endpoints for querying TEE verification status.

use axum::{
    extract::{Path, State},
    Json,
};
use serde::{Deserialize, Serialize};
use tracing::{debug, info, warn};

use crate::api::{types::ApiError, ApiState};

/// TEE status summary response
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct TeeStatusSummaryResponse {
    /// Total number of nodes with TEE status recorded
    pub total_nodes: u64,
    /// Number of TEE-verified nodes
    pub tee_verified_count: u64,
    /// Number of TDX-verified nodes
    pub tdx_verified_count: u64,
    /// Number of GPU CC enabled nodes
    pub gpu_cc_enabled_count: u64,
    /// Timestamp of summary
    pub timestamp: chrono::DateTime<chrono::Utc>,
}

/// TEE status for a specific node
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct NodeTeeStatusResponse {
    /// Node identifier
    pub node_id: String,
    /// Miner UID that owns this node
    pub miner_uid: u16,
    /// Whether the node has been TEE verified
    pub tee_verified: bool,
    /// TDX verification passed
    pub tdx_verified: bool,
    /// TDX quote valid (signature verified)
    pub tdx_quote_valid: Option<bool>,
    /// TDX MRTD matches expected
    pub tdx_mrtd_matches: Option<bool>,
    /// MRTD measurement (hex encoded)
    pub mrtd_hex: Option<String>,
    /// GPU Confidential Compute mode enabled
    pub gpu_cc_enabled: bool,
    /// GPU CC attestation valid
    pub gpu_cc_attestation_valid: Option<bool>,
    /// GPU model
    pub gpu_model: Option<String>,
    /// GPU UUID
    pub gpu_uuid: Option<String>,
    /// Last verification timestamp
    pub last_verified_at: String,
    /// Error message from last verification attempt
    pub error: Option<String>,
}

/// TEE requirements for availability check
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct TeeRequirements {
    /// Require TDX verification
    pub require_tdx: bool,
    /// Require GPU CC mode
    pub require_gpu_cc: bool,
    /// Expected MRTD (optional, for strict matching)
    pub expected_mrtd_hex: Option<String>,
}

/// TEE availability response
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct TeeAvailabilityResponse {
    /// Whether TEE requirements can be satisfied
    pub available: bool,
    /// Number of nodes matching requirements
    pub matching_nodes: u64,
    /// Message about availability
    pub message: String,
}

/// Get TEE status summary across all nodes
pub async fn get_tee_status_summary(
    State(state): State<ApiState>,
) -> Result<Json<TeeStatusSummaryResponse>, ApiError> {
    debug!("[TEE] Fetching TEE status summary");

    let summary = state
        .persistence
        .get_tee_status_summary()
        .await
        .map_err(|e| {
            warn!("[TEE] Failed to get TEE status summary: {}", e);
            ApiError::InternalError(format!("Failed to get TEE status: {}", e))
        })?;

    Ok(Json(TeeStatusSummaryResponse {
        total_nodes: summary.total_nodes,
        tee_verified_count: summary.tee_verified_count,
        tdx_verified_count: summary.tdx_verified_count,
        gpu_cc_enabled_count: summary.gpu_cc_enabled_count,
        timestamp: chrono::Utc::now(),
    }))
}

/// Get TEE status for a specific node
pub async fn get_node_tee_status(
    State(state): State<ApiState>,
    Path(node_id): Path<String>,
) -> Result<Json<NodeTeeStatusResponse>, ApiError> {
    info!("[TEE] Fetching TEE status for node: {}", node_id);

    let status = state
        .persistence
        .get_node_tee_status_by_node_id(&node_id)
        .await
        .map_err(|e| {
            warn!("[TEE] Failed to get node TEE status: {}", e);
            ApiError::InternalError(format!("Failed to get node TEE status: {}", e))
        })?;

    match status {
        Some(row) => Ok(Json(NodeTeeStatusResponse {
            node_id: row.node_id,
            miner_uid: row.miner_uid,
            tee_verified: row.tee_verified,
            tdx_verified: row.tdx_verified,
            tdx_quote_valid: row.tdx_quote_valid,
            tdx_mrtd_matches: row.tdx_mrtd_matches,
            mrtd_hex: row.tdx_mrtd_hex,
            gpu_cc_enabled: row.gpu_cc_enabled,
            gpu_cc_attestation_valid: row.gpu_cc_attestation_valid,
            gpu_model: row.gpu_cc_model,
            gpu_uuid: row.gpu_cc_uuid,
            last_verified_at: row.last_verification_at,
            error: row.verification_error,
        })),
        None => Err(ApiError::NotFound(format!(
            "No TEE status found for node: {}",
            node_id
        ))),
    }
}

/// List all TEE-verified nodes
pub async fn list_tee_verified_nodes(
    State(state): State<ApiState>,
) -> Result<Json<Vec<String>>, ApiError> {
    debug!("[TEE] Listing TEE-verified nodes");

    let nodes = state
        .persistence
        .get_all_tee_verified_nodes()
        .await
        .map_err(|e| {
            warn!("[TEE] Failed to list TEE-verified nodes: {}", e);
            ApiError::InternalError(format!("Failed to list TEE-verified nodes: {}", e))
        })?;

    Ok(Json(nodes))
}

/// Check if TEE requirements can be satisfied
pub async fn check_tee_availability(
    State(state): State<ApiState>,
    Json(requirements): Json<TeeRequirements>,
) -> Result<Json<TeeAvailabilityResponse>, ApiError> {
    info!("[TEE] Checking TEE availability: {:?}", requirements);

    let count = state
        .persistence
        .count_nodes_matching_tee_requirements(
            requirements.require_tdx,
            requirements.require_gpu_cc,
            requirements.expected_mrtd_hex.as_deref(),
        )
        .await
        .map_err(|e| {
            warn!("[TEE] Failed to check TEE availability: {}", e);
            ApiError::InternalError(format!("Failed to check TEE availability: {}", e))
        })?;

    let available = count > 0;
    let message = if available {
        format!("{} nodes match the TEE requirements", count)
    } else {
        "No nodes match the TEE requirements".to_string()
    };

    Ok(Json(TeeAvailabilityResponse {
        available,
        matching_nodes: count,
        message,
    }))
}
