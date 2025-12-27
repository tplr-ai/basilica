//! TEE (Trusted Execution Environment) status routes
//!
//! Provides endpoints for querying TEE verification status of nodes.
//! These endpoints proxy to the validator API which maintains TEE state.

use axum::{
    extract::{Path, State},
    Json,
};
use serde::{Deserialize, Serialize};
use tracing::{debug, info, warn};

use crate::{
    error::{ApiError, Result},
    server::AppState,
};

/// TEE status summary for a node
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct NodeTeeStatus {
    /// Node identifier
    pub node_id: String,
    /// Whether the node has been TEE verified
    pub tee_verified: bool,
    /// TDX verification passed
    pub tdx_verified: bool,
    /// GPU Confidential Compute mode enabled
    pub gpu_cc_enabled: bool,
    /// GPU model (if GPU CC was checked)
    pub gpu_model: Option<String>,
    /// MRTD measurement (hex encoded) - build-time measurement
    pub mrtd_hex: Option<String>,
    /// Last verification timestamp
    pub last_verified_at: Option<chrono::DateTime<chrono::Utc>>,
    /// Error message from last verification attempt
    pub error: Option<String>,
}

/// TEE status summary response
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct TeeStatusSummary {
    /// Total number of nodes
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

/// Get TEE status summary across all nodes
///
/// Returns aggregate TEE verification statistics from the validator.
pub async fn get_tee_status_summary(
    State(state): State<AppState>,
) -> Result<Json<TeeStatusSummary>> {
    debug!("[TEE] Fetching TEE status summary from validator");

    let validator_response = state
        .validator_client
        .get_tee_status_summary()
        .await
        .map_err(|e| {
            warn!(
                "[TEE] Failed to fetch TEE status summary from validator: {}",
                e
            );
            ApiError::ValidatorCommunication {
                message: format!("Failed to fetch TEE status: {}", e),
            }
        })?;

    Ok(Json(TeeStatusSummary {
        total_nodes: validator_response.total_nodes,
        tee_verified_count: validator_response.tee_verified_count,
        tdx_verified_count: validator_response.tdx_verified_count,
        gpu_cc_enabled_count: validator_response.gpu_cc_enabled_count,
        timestamp: validator_response.timestamp,
    }))
}

/// Get TEE status for a specific node
///
/// Returns detailed TEE verification status for a single node.
pub async fn get_node_tee_status(
    State(state): State<AppState>,
    Path(node_id): Path<String>,
) -> Result<Json<NodeTeeStatus>> {
    info!("[TEE] Fetching TEE status for node: {}", node_id);

    let validator_response = state
        .validator_client
        .get_node_tee_status(&node_id)
        .await
        .map_err(|e| {
            warn!(
                "[TEE] Failed to fetch node TEE status from validator: {}",
                e
            );
            ApiError::ValidatorCommunication {
                message: format!("Failed to fetch node TEE status: {}", e),
            }
        })?;

    // Parse the last_verified_at timestamp
    let last_verified_at =
        chrono::DateTime::parse_from_rfc3339(&validator_response.last_verified_at)
            .ok()
            .map(|dt| dt.with_timezone(&chrono::Utc));

    Ok(Json(NodeTeeStatus {
        node_id: validator_response.node_id,
        tee_verified: validator_response.tee_verified,
        tdx_verified: validator_response.tdx_verified,
        gpu_cc_enabled: validator_response.gpu_cc_enabled,
        gpu_model: validator_response.gpu_model,
        mrtd_hex: validator_response.mrtd_hex,
        last_verified_at,
        error: validator_response.error,
    }))
}

/// List all TEE-verified nodes
///
/// Returns a list of node IDs that have passed TEE verification.
pub async fn list_tee_verified_nodes(State(state): State<AppState>) -> Result<Json<Vec<String>>> {
    debug!("[TEE] Listing TEE-verified nodes");

    let nodes = state
        .validator_client
        .list_tee_verified_nodes()
        .await
        .map_err(|e| {
            warn!(
                "[TEE] Failed to list TEE-verified nodes from validator: {}",
                e
            );
            ApiError::ValidatorCommunication {
                message: format!("Failed to list TEE-verified nodes: {}", e),
            }
        })?;

    Ok(Json(nodes))
}

/// TEE requirements for a rental request
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct TeeRequirements {
    /// Require TDX verification
    pub require_tdx: bool,
    /// Require GPU CC mode
    pub require_gpu_cc: bool,
    /// Expected MRTD (optional, for strict matching)
    pub expected_mrtd_hex: Option<String>,
    /// Allowed GPU models for CC mode
    pub allowed_gpu_models: Option<Vec<String>>,
}

/// Check if TEE requirements can be satisfied
///
/// Given TEE requirements, returns whether there are nodes available
/// that can satisfy those requirements.
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct TeeAvailabilityResponse {
    /// Whether TEE requirements can be satisfied
    pub available: bool,
    /// Number of nodes matching requirements
    pub matching_nodes: u64,
    /// Message about availability
    pub message: String,
}

pub async fn check_tee_availability(
    State(state): State<AppState>,
    Json(requirements): Json<TeeRequirements>,
) -> Result<Json<TeeAvailabilityResponse>> {
    info!("[TEE] Checking TEE availability: {:?}", requirements);

    // Convert to validator's TeeRequirements format
    let validator_requirements = basilica_validator::api::routes::tee::TeeRequirements {
        require_tdx: requirements.require_tdx,
        require_gpu_cc: requirements.require_gpu_cc,
        expected_mrtd_hex: requirements.expected_mrtd_hex,
    };

    let validator_response = state
        .validator_client
        .check_tee_availability(validator_requirements)
        .await
        .map_err(|e| {
            warn!(
                "[TEE] Failed to check TEE availability from validator: {}",
                e
            );
            ApiError::ValidatorCommunication {
                message: format!("Failed to check TEE availability: {}", e),
            }
        })?;

    Ok(Json(TeeAvailabilityResponse {
        available: validator_response.available,
        matching_nodes: validator_response.matching_nodes,
        message: validator_response.message,
    }))
}
