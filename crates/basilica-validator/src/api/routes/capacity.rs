//! Capacity management routes

use crate::api::types::*;
use crate::api::ApiState;
use axum::{
    extract::{Query, State},
    http::Uri,
    Json,
};
use tracing::{error, info};

/// List available nodes for rental
pub async fn list_available_nodes(
    State(state): State<ApiState>,
    Query(mut query): Query<ListAvailableNodesQuery>,
    uri: Uri,
) -> Result<Json<ListAvailableNodesResponse>, ApiError> {
    // Default to available=true for /nodes endpoint
    if query.available.is_none() && uri.path() == "/nodes" {
        query.available = Some(true);
    }

    info!("Listing nodes with filters: {:?}", query);

    // Get available nodes from the database
    // Note: The persistence layer currently treats all queries as "available=true"
    // The 'available' parameter is handled by our endpoint logic above
    match state
        .persistence
        .get_available_nodes(
            query.min_gpu_memory,
            query.gpu_type.clone(),
            query.min_gpu_count,
            query.location.clone(),
        )
        .await
    {
        Ok(node_data) => {
            let mut available_nodes = Vec::new();

            for node in node_data {
                // Convert to API response format
                let network_speed = if node.download_mbps.is_some() || node.upload_mbps.is_some() {
                    Some(crate::api::types::NetworkSpeedInfo {
                        download_mbps: node.download_mbps,
                        upload_mbps: node.upload_mbps,
                        test_timestamp: node.speed_test_timestamp,
                    })
                } else {
                    None
                };

                let node_details = NodeDetails {
                    id: node.node_id,
                    gpu_specs: node.gpu_specs,
                    cpu_specs: node.cpu_specs,
                    location: node.location,
                    network_speed,
                };

                available_nodes.push(AvailableNode {
                    node: node_details,
                    availability: AvailabilityInfo {
                        available_until: None, // Could be calculated based on rental patterns
                        verification_score: node.verification_score,
                        uptime_percentage: node.uptime_percentage,
                    },
                });
            }

            Ok(Json(ListAvailableNodesResponse {
                total_count: available_nodes.len(),
                available_nodes,
            }))
        }
        Err(e) => {
            error!("Failed to query available nodes: {}", e);
            Err(ApiError::InternalError(
                "Failed to retrieve available nodes".to_string(),
            ))
        }
    }
}
