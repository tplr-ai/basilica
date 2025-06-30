//! Rental management routes

use crate::api::types::*;
use crate::api::ApiState;
use crate::persistence::entities::{Rental, RentalStatus as EntityRentalStatus};
use axum::{
    extract::{Path, State},
    Json,
};
use serde_json::json;
use tracing::{error, info};
use uuid::Uuid;

/// Rent GPU capacity
pub async fn rent_capacity(
    State(state): State<ApiState>,
    Json(request): Json<RentCapacityRequest>,
) -> Result<Json<RentCapacityResponse>, ApiError> {
    info!("Processing capacity rental request: {:?}", request);

    // Validate request
    if request.max_duration_hours == 0 {
        return Err(ApiError::BadRequest(
            "max_duration_hours must be greater than 0".to_string(),
        ));
    }

    if request.gpu_requirements.gpu_count == 0 {
        return Err(ApiError::BadRequest(
            "gpu_count must be greater than 0".to_string(),
        ));
    }

    // Find suitable executor
    let min_score = Some(0.7); // Require good verification score for rentals
    let min_success_rate = Some(0.8);

    match state
        .persistence
        .get_available_capacity(min_score, min_success_rate, 10, 0)
        .await
    {
        Ok(capacity_entries) => {
            // Filter executors by requirements
            let suitable_executor = capacity_entries
                .into_iter()
                .find(|entry| executor_meets_requirements(entry, &request.gpu_requirements));

            match suitable_executor {
                Some(executor) => {
                    let cost_per_hour = calculate_rental_cost(&executor);

                    // Generate SSH access info (simplified for now)
                    let ssh_access = SshAccess {
                        host: format!("{}.executor.basilica.ai", executor.executor_id),
                        port: 22,
                        username: "basilica".to_string(),
                    };

                    // Create GPU requirements JSON
                    let gpu_requirements_json = json!({
                        "min_memory_gb": request.gpu_requirements.min_memory_gb,
                        "gpu_type": request.gpu_requirements.gpu_type,
                        "gpu_count": request.gpu_requirements.gpu_count
                    });

                    // Create SSH access JSON
                    let ssh_access_json = json!({
                        "host": ssh_access.host,
                        "port": ssh_access.port,
                        "username": ssh_access.username
                    });

                    // Convert env_vars to JSON
                    let env_vars_json = request.env_vars.map(|vars| json!(vars));

                    // Create rental record
                    let rental = Rental::new(
                        executor.executor_id.clone(),
                        request.ssh_public_key,
                        request.docker_image,
                        env_vars_json,
                        gpu_requirements_json,
                        ssh_access_json,
                        request.max_duration_hours,
                        cost_per_hour,
                    );

                    // Store rental in database
                    match state.persistence.create_rental(&rental).await {
                        Ok(()) => {
                            let executor_details =
                                extract_executor_details_from_capacity(&executor)?;

                            Ok(Json(RentCapacityResponse {
                                rental_id: rental.id.to_string(),
                                executor: executor_details,
                                ssh_access,
                                cost_per_hour,
                            }))
                        }
                        Err(e) => {
                            error!("Failed to create rental: {}", e);
                            Err(ApiError::InternalError(
                                "Failed to create rental".to_string(),
                            ))
                        }
                    }
                }
                None => Err(ApiError::NotFound(
                    "No suitable executor found for requirements".to_string(),
                )),
            }
        }
        Err(e) => {
            error!("Failed to query available capacity: {}", e);
            Err(ApiError::InternalError(
                "Failed to find available capacity".to_string(),
            ))
        }
    }
}

/// Terminate a rental
pub async fn terminate_rental(
    State(state): State<ApiState>,
    Path(rental_id): Path<String>,
    Json(request): Json<TerminateRentalRequest>,
) -> Result<Json<TerminateRentalResponse>, ApiError> {
    info!("Terminating rental {}: {:?}", rental_id, request);

    let rental_uuid = Uuid::parse_str(&rental_id)
        .map_err(|_| ApiError::BadRequest("Invalid rental ID format".to_string()))?;

    match state.persistence.get_rental(&rental_uuid).await {
        Ok(Some(mut rental)) => {
            if rental.is_terminated() {
                return Err(ApiError::BadRequest(
                    "Rental is already terminated".to_string(),
                ));
            }

            // Calculate final cost
            let total_cost = rental.current_cost();

            // Terminate rental
            rental.terminate(request.reason, total_cost);

            // Update in database
            match state.persistence.update_rental(&rental).await {
                Ok(()) => {
                    info!(
                        "Rental {} terminated successfully, total cost: ${:.2}",
                        rental_id, total_cost
                    );

                    Ok(Json(TerminateRentalResponse {
                        success: true,
                        message: format!(
                            "Rental terminated successfully. Total cost: ${total_cost:.2}"
                        ),
                    }))
                }
                Err(e) => {
                    error!("Failed to update rental termination: {}", e);
                    Err(ApiError::InternalError(
                        "Failed to terminate rental".to_string(),
                    ))
                }
            }
        }
        Ok(None) => Err(ApiError::NotFound(format!("Rental {rental_id} not found"))),
        Err(e) => {
            error!("Failed to query rental: {}", e);
            Err(ApiError::InternalError("Database error".to_string()))
        }
    }
}

/// Get rental status
pub async fn get_rental_status(
    State(state): State<ApiState>,
    Path(rental_id): Path<String>,
) -> Result<Json<RentalStatusResponse>, ApiError> {
    info!("Getting status for rental {}", rental_id);

    let rental_uuid = Uuid::parse_str(&rental_id)
        .map_err(|_| ApiError::BadRequest("Invalid rental ID format".to_string()))?;

    match state.persistence.get_rental(&rental_uuid).await {
        Ok(Some(rental)) => {
            // Convert entity status to API status
            let status = match rental.status {
                EntityRentalStatus::Pending => RentalStatus::Pending,
                EntityRentalStatus::Active => RentalStatus::Active,
                EntityRentalStatus::Terminated => RentalStatus::Terminated,
                EntityRentalStatus::Failed => RentalStatus::Failed,
            };

            // Get executor stats for additional details
            let executor_details = match state
                .persistence
                .get_executor_stats(&rental.executor_id)
                .await
            {
                Ok(Some(_stats)) => {
                    // Extract details from rental's hardware info in ssh_access_info
                    create_executor_details_from_rental(&rental)
                }
                _ => {
                    // Fallback executor details
                    ExecutorDetails {
                        id: rental.executor_id.clone(),
                        gpu_specs: vec![],
                        cpu_specs: CpuSpec {
                            cores: 0,
                            model: "Unknown".to_string(),
                            memory_gb: 0,
                        },
                        location: None,
                    }
                }
            };

            Ok(Json(RentalStatusResponse {
                rental_id: rental.id.to_string(),
                status,
                executor: executor_details,
                created_at: rental.created_at,
                updated_at: rental.updated_at,
                cost_incurred: rental.current_cost(),
            }))
        }
        Ok(None) => Err(ApiError::NotFound(format!("Rental {rental_id} not found"))),
        Err(e) => {
            error!("Failed to query rental: {}", e);
            Err(ApiError::InternalError("Database error".to_string()))
        }
    }
}

/// Check if executor meets GPU requirements
fn executor_meets_requirements(
    executor: &crate::persistence::simple_persistence::CapacityEntry,
    requirements: &GpuRequirements,
) -> bool {
    let hardware_info = &executor.hardware_info;

    // Check GPU count
    if let Some(gpu_info) = hardware_info.get("gpu") {
        if let Some(gpus) = gpu_info.as_array() {
            if gpus.len() < requirements.gpu_count as usize {
                return false;
            }

            // Check memory requirement
            let has_sufficient_memory = gpus.iter().any(|gpu| {
                gpu.get("memory_gb").and_then(|m| m.as_u64()).unwrap_or(0)
                    >= requirements.min_memory_gb as u64
            });

            if !has_sufficient_memory {
                return false;
            }

            // Check GPU type if specified
            if let Some(required_type) = &requirements.gpu_type {
                let matches_type = gpus.iter().any(|gpu| {
                    gpu.get("name")
                        .and_then(|n| n.as_str())
                        .unwrap_or("")
                        .to_lowercase()
                        .contains(&required_type.to_lowercase())
                });

                if !matches_type {
                    return false;
                }
            }

            return true;
        }
    }
    false
}

/// Calculate rental cost per hour
fn calculate_rental_cost(executor: &crate::persistence::simple_persistence::CapacityEntry) -> f64 {
    let base_cost = 2.0; // Base $2/hour for rentals (higher than capacity listing)
    let score_multiplier = executor.verification_score.max(0.1);
    let premium_multiplier = 1.5; // 50% premium for actual rentals

    base_cost * score_multiplier * premium_multiplier
}

/// Extract executor details from capacity entry
fn extract_executor_details_from_capacity(
    entry: &crate::persistence::simple_persistence::CapacityEntry,
) -> Result<ExecutorDetails, ApiError> {
    let gpu_specs = if let Some(gpu_info) = entry.hardware_info.get("gpu") {
        if let Some(gpus) = gpu_info.as_array() {
            gpus.iter()
                .map(|gpu| GpuSpec {
                    name: gpu
                        .get("name")
                        .and_then(|n| n.as_str())
                        .unwrap_or("Unknown")
                        .to_string(),
                    memory_gb: gpu.get("memory_gb").and_then(|m| m.as_u64()).unwrap_or(0) as u32,
                    compute_capability: gpu
                        .get("compute_capability")
                        .and_then(|c| c.as_str())
                        .unwrap_or("Unknown")
                        .to_string(),
                })
                .collect()
        } else {
            vec![]
        }
    } else {
        vec![]
    };

    let cpu_specs = if let Some(cpu_info) = entry.hardware_info.get("cpu") {
        CpuSpec {
            cores: cpu_info.get("cores").and_then(|c| c.as_u64()).unwrap_or(0) as u32,
            model: cpu_info
                .get("model")
                .and_then(|m| m.as_str())
                .unwrap_or("Unknown")
                .to_string(),
            memory_gb: cpu_info
                .get("memory_gb")
                .and_then(|m| m.as_u64())
                .unwrap_or(0) as u32,
        }
    } else {
        CpuSpec {
            cores: 0,
            model: "Unknown".to_string(),
            memory_gb: 0,
        }
    };

    Ok(ExecutorDetails {
        id: entry.executor_id.clone(),
        gpu_specs,
        cpu_specs,
        location: entry
            .hardware_info
            .get("location")
            .and_then(|l| l.as_str())
            .map(|s| s.to_string()),
    })
}

/// Create executor details from rental record
fn create_executor_details_from_rental(rental: &Rental) -> ExecutorDetails {
    // Extract GPU specs from rental requirements
    let gpu_specs = if let Some(req) = rental.gpu_requirements.as_object() {
        let memory_gb = req
            .get("min_memory_gb")
            .and_then(|v| v.as_u64())
            .unwrap_or(0) as u32;
        let gpu_type = req
            .get("gpu_type")
            .and_then(|v| v.as_str())
            .unwrap_or("Unknown");
        let gpu_count = req.get("gpu_count").and_then(|v| v.as_u64()).unwrap_or(1) as usize;

        (0..gpu_count)
            .map(|_| GpuSpec {
                name: gpu_type.to_string(),
                memory_gb,
                compute_capability: "Unknown".to_string(),
            })
            .collect()
    } else {
        vec![]
    };

    ExecutorDetails {
        id: rental.executor_id.clone(),
        gpu_specs,
        cpu_specs: CpuSpec {
            cores: 0,
            model: "Unknown".to_string(),
            memory_gb: 0,
        },
        location: None,
    }
}
