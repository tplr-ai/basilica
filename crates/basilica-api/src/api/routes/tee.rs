//! TEE (Trusted Execution Environment) status routes
//!
//! Provides endpoints for querying TEE verification status of nodes.

use axum::{
    extract::{Path, State},
    Json,
};
use serde::{Deserialize, Serialize};
use tracing::info;

use crate::{error::Result, server::AppState};

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
/// Returns aggregate TEE verification statistics.
pub async fn get_tee_status_summary(
    State(_state): State<AppState>,
) -> Result<Json<TeeStatusSummary>> {
    info!("[TEE] Fetching TEE status summary from validator");

    // Query validator for TEE status summary
    // TODO: Implement actual validator endpoint call when available
    // The validator needs to expose a TEE summary endpoint

    let summary = TeeStatusSummary {
        total_nodes: 0,
        tee_verified_count: 0,
        tdx_verified_count: 0,
        gpu_cc_enabled_count: 0,
        timestamp: chrono::Utc::now(),
    };

    Ok(Json(summary))
}

/// Get TEE status for a specific node
///
/// Returns detailed TEE verification status for a single node.
pub async fn get_node_tee_status(
    State(_state): State<AppState>,
    Path(node_id): Path<String>,
) -> Result<Json<NodeTeeStatus>> {
    info!("[TEE] Fetching TEE status for node: {}", node_id);

    // Query validator for node TEE status
    // TODO: Implement actual validator endpoint call when available
    // The validator needs to expose a per-node TEE status endpoint

    let status = NodeTeeStatus {
        node_id: node_id.clone(),
        tee_verified: false,
        tdx_verified: false,
        gpu_cc_enabled: false,
        gpu_model: None,
        mrtd_hex: None,
        last_verified_at: None,
        error: Some("TEE status not yet implemented".to_string()),
    };

    Ok(Json(status))
}

/// List all TEE-verified nodes
///
/// Returns a list of node IDs that have passed TEE verification.
pub async fn list_tee_verified_nodes(State(_state): State<AppState>) -> Result<Json<Vec<String>>> {
    info!("[TEE] Listing TEE-verified nodes");

    // Query validator for TEE-verified nodes
    // TODO: Implement actual validator endpoint call when available
    // The validator needs to expose a TEE-verified nodes list endpoint

    Ok(Json(vec![]))
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
    State(_state): State<AppState>,
    Json(requirements): Json<TeeRequirements>,
) -> Result<Json<TeeAvailabilityResponse>> {
    info!("[TEE] Checking TEE availability: {:?}", requirements);

    // Query validator for matching nodes
    // TODO: Implement actual validator endpoint call when available
    // The validator needs to expose a TEE availability check endpoint

    let response = TeeAvailabilityResponse {
        available: false,
        matching_nodes: 0,
        message: "TEE availability check not yet implemented".to_string(),
    };

    Ok(Json(response))
}
