//! Miner management routes

use crate::api::types::*;
use crate::api::ApiState;
use axum::{
    extract::{Path, Query, State},
    Json,
};
use chrono::Utc;
use serde_json::Value;
use tracing::{error, info, warn};
use uuid::Uuid;

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

                let total_gpu_count = calculate_total_gpu_count(&miner_data.executor_info);

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
                    executor_count: miner_data.executor_count,
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

/// Register a new miner with the validator
pub async fn register_miner(
    State(state): State<ApiState>,
    Json(request): Json<RegisterMinerRequest>,
) -> Result<Json<RegisterMinerResponse>, ApiError> {
    info!("Registering miner: {}", request.miner_id);

    if request.miner_id.is_empty() || request.hotkey.is_empty() {
        return Err(ApiError::BadRequest(
            "Miner ID and hotkey are required".to_string(),
        ));
    }

    if request.executors.is_empty() {
        return Err(ApiError::BadRequest(
            "At least one executor must be registered".to_string(),
        ));
    }

    match verify_miner_signature(&request).await {
        Ok(false) => {
            warn!(
                "Invalid signature for miner registration: {}",
                request.miner_id
            );
            return Err(ApiError::Unauthorized);
        }
        Err(e) => {
            error!("Failed to verify miner signature: {}", e);
            return Err(ApiError::InternalError(
                "Signature verification failed".to_string(),
            ));
        }
        Ok(true) => {}
    }

    let registration_result = state
        .persistence
        .register_miner(
            &request.miner_id,
            &request.hotkey,
            &request.endpoint,
            &request.executors,
        )
        .await;

    match registration_result {
        Ok(()) => {
            info!("Successfully registered miner: {}", request.miner_id);
            Ok(Json(RegisterMinerResponse {
                success: true,
                miner_id: request.miner_id,
                message: "Miner registered successfully".to_string(),
            }))
        }
        Err(e) => {
            error!("Failed to register miner {}: {}", request.miner_id, e);
            if e.to_string().contains("already exists") {
                Err(ApiError::BadRequest("Miner already registered".to_string()))
            } else {
                Err(ApiError::InternalError("Registration failed".to_string()))
            }
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
            let total_gpu_count = calculate_total_gpu_count(&miner_data.executor_info);

            Ok(Json(MinerDetails {
                miner_id: miner_data.miner_id,
                hotkey: miner_data.hotkey,
                endpoint: miner_data.endpoint,
                status,
                executor_count: miner_data.executor_count,
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

/// Update miner information
pub async fn update_miner(
    State(state): State<ApiState>,
    Path(miner_id): Path<String>,
    Json(request): Json<UpdateMinerRequest>,
) -> Result<Json<RegisterMinerResponse>, ApiError> {
    info!("Updating miner: {}", miner_id);

    match verify_miner_update_signature(&miner_id, &request).await {
        Ok(false) => {
            warn!("Invalid signature for miner update: {}", miner_id);
            return Err(ApiError::Unauthorized);
        }
        Err(e) => {
            error!("Failed to verify miner update signature: {}", e);
            return Err(ApiError::InternalError(
                "Signature verification failed".to_string(),
            ));
        }
        Ok(true) => {}
    }

    match state.persistence.update_miner(&miner_id, &request).await {
        Ok(()) => {
            info!("Successfully updated miner: {}", miner_id);
            Ok(Json(RegisterMinerResponse {
                success: true,
                miner_id,
                message: "Miner updated successfully".to_string(),
            }))
        }
        Err(e) => {
            error!("Failed to update miner {}: {}", miner_id, e);
            if e.to_string().contains("not found") {
                Err(ApiError::NotFound("Miner not found".to_string()))
            } else {
                Err(ApiError::InternalError("Update failed".to_string()))
            }
        }
    }
}

/// Remove a miner from the registry
pub async fn remove_miner(
    State(state): State<ApiState>,
    Path(miner_id): Path<String>,
) -> Result<Json<RegisterMinerResponse>, ApiError> {
    info!("Removing miner: {}", miner_id);

    match state.persistence.remove_miner(&miner_id).await {
        Ok(()) => {
            info!("Successfully removed miner: {}", miner_id);
            Ok(Json(RegisterMinerResponse {
                success: true,
                miner_id,
                message: "Miner removed successfully".to_string(),
            }))
        }
        Err(e) => {
            error!("Failed to remove miner {}: {}", miner_id, e);
            if e.to_string().contains("not found") {
                Err(ApiError::NotFound("Miner not found".to_string()))
            } else {
                Err(ApiError::InternalError("Removal failed".to_string()))
            }
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

            let executor_health = health_data
                .executor_health
                .into_iter()
                .map(|eh| ExecutorHealthStatus {
                    executor_id: eh.executor_id,
                    status: eh.status,
                    last_seen: eh.last_seen,
                    gpu_utilization: eh.gpu_utilization,
                    memory_usage: eh.memory_usage,
                })
                .collect();

            Ok(Json(MinerHealthResponse {
                miner_id,
                overall_status: status,
                last_health_check: health_data.last_health_check,
                executor_health,
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

/// Trigger verification process for a miner
pub async fn trigger_miner_verification(
    State(state): State<ApiState>,
    Path(miner_id): Path<String>,
    Json(request): Json<TriggerVerificationRequest>,
) -> Result<Json<TriggerVerificationResponse>, ApiError> {
    info!("Triggering verification for miner: {}", miner_id);

    let verification_id = Uuid::new_v4().to_string();
    let estimated_completion = Utc::now() + chrono::Duration::minutes(10);

    match state
        .persistence
        .schedule_verification(
            &miner_id,
            &verification_id,
            &request.verification_type,
            request.executor_id.as_deref(),
        )
        .await
    {
        Ok(()) => {
            info!(
                "Scheduled verification {} for miner {}",
                verification_id, miner_id
            );
            Ok(Json(TriggerVerificationResponse {
                verification_id,
                status: "scheduled".to_string(),
                estimated_completion,
            }))
        }
        Err(e) => {
            error!(
                "Failed to schedule verification for miner {}: {}",
                miner_id, e
            );
            if e.to_string().contains("not found") {
                Err(ApiError::NotFound("Miner not found".to_string()))
            } else {
                Err(ApiError::InternalError(
                    "Verification scheduling failed".to_string(),
                ))
            }
        }
    }
}

/// List executors for a specific miner
pub async fn list_miner_executors(
    State(state): State<ApiState>,
    Path(miner_id): Path<String>,
) -> Result<Json<Vec<ExecutorDetails>>, ApiError> {
    info!("Listing executors for miner: {}", miner_id);

    match state.persistence.get_miner_executors(&miner_id).await {
        Ok(executors) => {
            let executor_details = executors
                .into_iter()
                .map(|exec| ExecutorDetails {
                    id: exec.executor_id,
                    gpu_specs: exec.gpu_specs,
                    cpu_specs: exec.cpu_specs,
                    location: exec.location,
                })
                .collect();

            Ok(Json(executor_details))
        }
        Err(e) => {
            error!("Failed to list executors for miner {}: {}", miner_id, e);
            if e.to_string().contains("not found") {
                Err(ApiError::NotFound("Miner not found".to_string()))
            } else {
                Err(ApiError::InternalError(
                    "Failed to retrieve executors".to_string(),
                ))
            }
        }
    }
}

// Helper functions

async fn verify_miner_signature(request: &RegisterMinerRequest) -> Result<bool, anyhow::Error> {
    let message = format!(
        "{}:{}:{}",
        request.miner_id, request.hotkey, request.endpoint
    );
    common::crypto::verify_signature(&request.signature, &message, &request.hotkey).await
}

async fn verify_miner_update_signature(
    miner_id: &str,
    request: &UpdateMinerRequest,
) -> Result<bool, anyhow::Error> {
    let endpoint = request.endpoint.as_deref().unwrap_or("");
    let message = format!("{miner_id}:{endpoint}:update");
    common::crypto::verify_signature(&request.signature, &message, miner_id).await
}

fn determine_miner_status(
    miner_data: &crate::persistence::simple_persistence::MinerData,
) -> MinerStatus {
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
    health_data: &crate::persistence::simple_persistence::MinerHealthData,
) -> MinerStatus {
    let now = Utc::now();
    let time_since_check = now.signed_duration_since(health_data.last_health_check);

    if time_since_check.num_minutes() > 5 {
        MinerStatus::Offline
    } else if health_data
        .executor_health
        .iter()
        .any(|eh| eh.status == "verifying")
    {
        MinerStatus::Verifying
    } else if health_data
        .executor_health
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

fn calculate_total_gpu_count(executor_info: &Value) -> u32 {
    if let Some(executors) = executor_info.as_array() {
        executors
            .iter()
            .filter_map(|exec| exec.get("gpu_count").and_then(|gc| gc.as_u64()))
            .sum::<u64>() as u32
    } else {
        0
    }
}
