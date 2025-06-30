//! Capacity management routes

use crate::api::types::*;
use crate::api::ApiState;
use axum::{
    extract::{Query, State},
    Json,
};
use serde_json::Value;
use tracing::{error, info};

/// List available GPU capacity
pub async fn list_available_capacity(
    State(state): State<ApiState>,
    Query(query): Query<ListCapacityQuery>,
) -> Result<Json<ListCapacityResponse>, ApiError> {
    info!("Listing available capacity with filters: {:?}", query);

    let min_score = query.max_cost_per_hour.map(|cost| {
        // Simple scoring: higher cost tolerance = lower min score required
        if cost > 10.0 {
            0.5
        } else if cost > 5.0 {
            0.7
        } else {
            0.9
        }
    });
    let min_success_rate = Some(0.8); // Require 80% success rate minimum

    let limit = 50; // Default page size
    let offset = 0; // TODO: Add pagination support

    match state
        .persistence
        .get_available_capacity(min_score, min_success_rate, limit, offset)
        .await
    {
        Ok(capacity_entries) => {
            let mut available_executors = Vec::new();

            for entry in capacity_entries {
                // Filter by query parameters
                if let Some(min_gpu_memory) = query.min_gpu_memory {
                    if !hardware_meets_memory_requirement(&entry.hardware_info, min_gpu_memory) {
                        continue;
                    }
                }

                if let Some(gpu_type) = &query.gpu_type {
                    if !hardware_matches_gpu_type(&entry.hardware_info, gpu_type) {
                        continue;
                    }
                }

                if let Some(min_gpu_count) = query.min_gpu_count {
                    if !hardware_meets_gpu_count(&entry.hardware_info, min_gpu_count) {
                        continue;
                    }
                }

                let cost_per_hour = calculate_cost_per_hour(&entry);

                if let Some(max_cost) = query.max_cost_per_hour {
                    if cost_per_hour > max_cost {
                        continue;
                    }
                }

                let executor_details = extract_executor_details(&entry)?;

                available_executors.push(AvailableExecutor {
                    executor: executor_details,
                    availability: AvailabilityInfo {
                        available_until: None, // TODO: Calculate based on current rentals
                        verification_score: entry.verification_score,
                        uptime_percentage: entry.success_rate * 100.0,
                    },
                    cost_per_hour,
                });
            }

            Ok(Json(ListCapacityResponse {
                total_count: available_executors.len(),
                available_executors,
            }))
        }
        Err(e) => {
            error!("Failed to query available capacity: {}", e);
            Err(ApiError::InternalError(
                "Failed to retrieve capacity data".to_string(),
            ))
        }
    }
}

/// Check if hardware meets memory requirement
fn hardware_meets_memory_requirement(hardware_info: &Value, min_memory_gb: u32) -> bool {
    if let Some(gpu_info) = hardware_info.get("gpu") {
        if let Some(gpus) = gpu_info.as_array() {
            return gpus.iter().any(|gpu| {
                gpu.get("memory_gb").and_then(|m| m.as_u64()).unwrap_or(0) >= min_memory_gb as u64
            });
        }
    }
    false
}

/// Check if hardware matches GPU type
fn hardware_matches_gpu_type(hardware_info: &Value, gpu_type: &str) -> bool {
    if let Some(gpu_info) = hardware_info.get("gpu") {
        if let Some(gpus) = gpu_info.as_array() {
            return gpus.iter().any(|gpu| {
                gpu.get("name")
                    .and_then(|n| n.as_str())
                    .unwrap_or("")
                    .to_lowercase()
                    .contains(&gpu_type.to_lowercase())
            });
        }
    }
    false
}

/// Check if hardware meets GPU count requirement
fn hardware_meets_gpu_count(hardware_info: &Value, min_count: u32) -> bool {
    if let Some(gpu_info) = hardware_info.get("gpu") {
        if let Some(gpus) = gpu_info.as_array() {
            return gpus.len() >= min_count as usize;
        }
    }
    false
}

/// Calculate cost per hour based on executor performance and specs
fn calculate_cost_per_hour(entry: &crate::persistence::simple_persistence::CapacityEntry) -> f64 {
    let base_cost = 1.0; // Base $1/hour
    let score_multiplier = entry.verification_score.max(0.1); // Minimum 0.1x multiplier
    let demand_multiplier = 1.2; // 20% markup for availability

    base_cost * score_multiplier * demand_multiplier
}

/// Extract executor details from capacity entry
fn extract_executor_details(
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
