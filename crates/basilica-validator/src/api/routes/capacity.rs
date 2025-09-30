//! Capacity management routes

use crate::api::types::*;
use crate::api::ApiState;
use axum::{
    extract::{Query, State},
    http::Uri,
    Json,
};
use tracing::{error, info};

/// List available executors for rental
pub async fn list_available_executors(
    State(state): State<ApiState>,
    Query(mut query): Query<ListAvailableExecutorsQuery>,
    uri: Uri,
) -> Result<Json<ListAvailableExecutorsResponse>, ApiError> {
    // Default to available=true for /executors endpoint
    if query.available.is_none() && uri.path() == "/executors" {
        query.available = Some(true);
    }

    info!("Listing executors with filters: {:?}", query);

    // Get available executors from the database
    // Note: The persistence layer currently treats all queries as "available=true"
    // The 'available' parameter is handled by our endpoint logic above
    match state
        .persistence
        .get_available_executors(
            query.min_gpu_memory,
            query.gpu_type.clone(),
            query.min_gpu_count,
            query.location.clone(),
        )
        .await
    {
        Ok(executor_data) => {
            let mut available_executors = Vec::new();

            for executor in executor_data {
                // Convert to API response format
                let network_speed =
                    if executor.download_mbps.is_some() || executor.upload_mbps.is_some() {
                        Some(crate::api::types::NetworkSpeedInfo {
                            download_mbps: executor.download_mbps,
                            upload_mbps: executor.upload_mbps,
                            test_timestamp: executor.speed_test_timestamp,
                        })
                    } else {
                        None
                    };

                let executor_details = ExecutorDetails {
                    id: executor.executor_id,
                    gpu_specs: executor.gpu_specs,
                    cpu_specs: executor.cpu_specs,
                    location: executor.location,
                    network_speed,
                };

                available_executors.push(AvailableExecutor {
                    executor: executor_details,
                    availability: AvailabilityInfo {
                        available_until: None, // Could be calculated based on rental patterns
                        verification_score: executor.verification_score,
                        uptime_percentage: executor.uptime_percentage,
                    },
                });
            }

            Ok(Json(ListAvailableExecutorsResponse {
                total_count: available_executors.len(),
                available_executors,
            }))
        }
        Err(e) => {
            error!("Failed to query available executors: {}", e);
            Err(ApiError::InternalError(
                "Failed to retrieve available executors".to_string(),
            ))
        }
    }
}
