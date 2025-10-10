//! Miner management routes

use crate::api::types::*;
use crate::api::ApiState;
use axum::{
    extract::{Path, Query, State},
    Json,
};
use chrono::Utc;
use serde_json::Value;
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

                let total_gpu_count = calculate_total_gpu_count(&miner_data.node_info);

                if let Some(min_gpu_count) = query.min_gpu_count {
                    if total_gpu_count < min_gpu_count {
                        continue;
                    }
                }

                if let Some(min_score) = query.min_score {
                    if miner_data.verification_score < min_score {
                        continue;
                    }
                }

                miners.push(MinerDetails {
                    miner_id: miner_data.miner_id,
                    hotkey: miner_data.hotkey,
                    endpoint: miner_data.endpoint,
                    status,
                    node_count: miner_data.node_count,
                    total_gpu_count,
                    verification_score: miner_data.verification_score,
                    uptime_percentage: miner_data.uptime_percentage,
                    last_seen: miner_data.last_seen,
                    registered_at: miner_data.registered_at,
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
            let total_gpu_count = calculate_total_gpu_count(&miner_data.node_info);

            Ok(Json(MinerDetails {
                miner_id: miner_data.miner_id,
                hotkey: miner_data.hotkey,
                endpoint: miner_data.endpoint,
                status,
                node_count: miner_data.node_count,
                total_gpu_count,
                verification_score: miner_data.verification_score,
                uptime_percentage: miner_data.uptime_percentage,
                last_seen: miner_data.last_seen,
                registered_at: miner_data.registered_at,
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
                    last_seen: eh.last_seen,
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
    let time_since_last_seen = now.signed_duration_since(miner_data.last_seen);

    if time_since_last_seen.num_minutes() > 10 {
        MinerStatus::Offline
    } else if miner_data.verification_score < 0.5 {
        MinerStatus::Suspended
    } else if miner_data.uptime_percentage < 80.0 {
        MinerStatus::Inactive
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
        "suspended" => matches!(status, MinerStatus::Suspended),
        _ => true,
    }
}

fn calculate_total_gpu_count(node_info: &Value) -> u32 {
    if let Some(nodes) = node_info.as_array() {
        nodes
            .iter()
            .filter_map(|node| node.get("gpu_count").and_then(|gc| gc.as_u64()))
            .sum::<u64>() as u32
    } else {
        0
    }
}
