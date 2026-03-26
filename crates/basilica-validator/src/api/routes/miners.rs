//! Miner management routes

use crate::api::types::*;
use crate::api::ApiState;
use axum::{
    extract::{Path, Query, State},
    Json,
};
use chrono::Utc;
use tracing::{error, info};

/// List all registered miners with filtering and pagination
pub async fn list_miners(
    State(state): State<ApiState>,
    Query(query): Query<ListMinersQuery>,
) -> Result<Json<ListMinersResponse>, ApiError> {
    info!("Listing miners with filters: {:?}", query);

    let page = query.page.unwrap_or(1);
    let page_size = query.page_size.unwrap_or(20).min(100);
    let offset = (page.saturating_sub(1)) * page_size;

    match state
        .persistence
        .get_registered_miners(offset, page_size)
        .await
    {
        Ok(miners_data) => {
            let mut miners = Vec::new();

            for miner_data in miners_data {
                let status = determine_miner_status(&miner_data);

                if let Some(status_filter) = &query.status {
                    if !status_matches_filter(&status, status_filter) {
                        continue;
                    }
                }

                if let Some(min_gpu_count) = query.min_gpu_count {
                    if miner_data.node_count < min_gpu_count {
                        continue;
                    }
                }

                miners.push(MinerDetails {
                    miner_id: miner_data.miner_id,
                    hotkey: miner_data.hotkey,
                    endpoint: miner_data.endpoint,
                    status,
                    node_count: miner_data.node_count,
                    updated_at: miner_data.updated_at,
                });
            }

            let total_count = miners.len();

            Ok(Json(ListMinersResponse {
                miners,
                total_count,
                page,
                page_size,
            }))
        }
        Err(e) => {
            error!("Failed to list miners: {}", e);
            Err(ApiError::InternalError(
                "Failed to retrieve miners".to_string(),
            ))
        }
    }
}

/// Get details for a specific miner
pub async fn get_miner(
    State(state): State<ApiState>,
    Path(miner_id): Path<String>,
) -> Result<Json<MinerDetails>, ApiError> {
    info!("Getting miner details: {}", miner_id);

    match state.persistence.get_miner_by_id(&miner_id).await {
        Ok(Some(miner_data)) => {
            let status = determine_miner_status(&miner_data);

            Ok(Json(MinerDetails {
                miner_id: miner_data.miner_id,
                hotkey: miner_data.hotkey,
                endpoint: miner_data.endpoint,
                status,
                node_count: miner_data.node_count,
                updated_at: miner_data.updated_at,
            }))
        }
        Ok(None) => Err(ApiError::NotFound("Miner not found".to_string())),
        Err(e) => {
            error!("Failed to get miner {}: {}", miner_id, e);
            Err(ApiError::InternalError(
                "Failed to retrieve miner".to_string(),
            ))
        }
    }
}

/// Get health status for a specific miner
pub async fn get_miner_health(
    State(state): State<ApiState>,
    Path(miner_id): Path<String>,
) -> Result<Json<MinerHealthResponse>, ApiError> {
    info!("Getting miner health: {}", miner_id);

    let start_time = std::time::Instant::now();

    match state.persistence.get_miner_health(&miner_id).await {
        Ok(Some(health_data)) => {
            let response_time_ms = start_time.elapsed().as_millis() as u64;
            let status = determine_miner_status_from_health(&health_data);

            let node_health = health_data
                .node_health
                .into_iter()
                .map(|eh| NodeHealthStatus {
                    node_id: eh.node_id,
                    status: eh.status,
                    last_health_check: eh.last_health_check,
                })
                .collect();

            Ok(Json(MinerHealthResponse {
                miner_id,
                overall_status: status,
                last_health_check: health_data.last_health_check,
                node_health,
                response_time_ms,
            }))
        }
        Ok(None) => Err(ApiError::NotFound("Miner not found".to_string())),
        Err(e) => {
            error!("Failed to get miner health {}: {}", miner_id, e);
            Err(ApiError::InternalError(
                "Failed to retrieve miner health".to_string(),
            ))
        }
    }
}

/// List nodes for a specific miner
pub async fn list_miner_nodes(
    State(state): State<ApiState>,
    Path(miner_id): Path<String>,
) -> Result<Json<Vec<NodeDetails>>, ApiError> {
    info!("Listing nodes for miner: {}", miner_id);

    match state.persistence.get_miner_nodes(&miner_id).await {
        Ok(nodes) => {
            let node_details = nodes
                .into_iter()
                .map(|node| NodeDetails {
                    id: node.node_id,
                    gpu_specs: node.gpu_specs,
                    cpu_specs: node.cpu_specs,
                    location: node.location,
                    network_speed: None,
                    hourly_rate_cents: None,
                })
                .collect();

            Ok(Json(node_details))
        }
        Err(e) => {
            error!("Failed to list nodes for miner {}: {}", miner_id, e);
            if e.to_string().contains("not found") {
                Err(ApiError::NotFound("Miner not found".to_string()))
            } else {
                Err(ApiError::InternalError(
                    "Failed to retrieve nodes".to_string(),
                ))
            }
        }
    }
}

fn determine_miner_status(miner_data: &crate::persistence::MinerData) -> MinerStatus {
    let now = Utc::now();
    let time_since_update = now.signed_duration_since(miner_data.updated_at);

    if time_since_update.num_minutes() > 10 {
        MinerStatus::Offline
    } else {
        MinerStatus::Active
    }
}

fn determine_miner_status_from_health(
    health_data: &crate::persistence::MinerHealthData,
) -> MinerStatus {
    let now = Utc::now();
    let time_since_check = now.signed_duration_since(health_data.last_health_check);

    if time_since_check.num_minutes() > 5 {
        MinerStatus::Offline
    } else if health_data
        .node_health
        .iter()
        .any(|eh| eh.status == "verifying")
    {
        MinerStatus::Verifying
    } else if health_data
        .node_health
        .iter()
        .all(|eh| eh.status == "healthy")
    {
        MinerStatus::Active
    } else {
        MinerStatus::Inactive
    }
}

fn status_matches_filter(status: &MinerStatus, filter: &str) -> bool {
    match filter.to_lowercase().as_str() {
        "active" => matches!(status, MinerStatus::Active),
        "inactive" => matches!(status, MinerStatus::Inactive),
        "offline" => matches!(status, MinerStatus::Offline),
        "verifying" => matches!(status, MinerStatus::Verifying),
        _ => true,
    }
}
