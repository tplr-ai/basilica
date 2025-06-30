//! Rental management route handlers

use crate::{
    aggregator::{RequestDistributor, ResponseAggregator},
    api::types::{
        RentCapacityRequest, RentCapacityResponse, RentalStatusResponse, TerminateRentalRequest,
        TerminateRentalResponse,
    },
    error::{Error, Result},
    server::AppState,
};
use axum::{
    extract::{Path, State},
    Json,
};
use tracing::{debug, info};

/// Rent GPU capacity
#[utoipa::path(
    post,
    path = "/api/v1/rentals",
    request_body = RentCapacityRequest,
    responses(
        (status = 201, description = "Rental created successfully", body = RentCapacityResponse),
        (status = 400, description = "Invalid request", body = crate::error::ErrorResponse),
        (status = 503, description = "No validators available", body = crate::error::ErrorResponse),
    ),
    tag = "rentals",
)]
pub async fn rent_capacity(
    State(state): State<AppState>,
    Json(request): Json<RentCapacityRequest>,
) -> Result<Json<RentCapacityResponse>> {
    info!(
        "Processing rental request for {} GPUs",
        request.gpu_requirements.gpu_count
    );

    // Create request distributor
    let distributor =
        RequestDistributor::new(state.http_client.clone(), state.load_balancer.clone());

    // Build request to forward to validator
    let validator_request = state
        .http_client
        .post("/api/v1/rentals")
        .json(&request)
        .build()
        .map_err(|e| Error::Internal {
            message: format!("Failed to build request: {e}"),
        })?;

    // Send to a single validator (could be extended to try multiple)
    let response = distributor.send_to_single(validator_request).await?;

    // Parse response
    let rental_response: RentCapacityResponse =
        response
            .json()
            .await
            .map_err(|e| Error::ValidatorCommunication {
                message: format!("Failed to parse validator response: {e}"),
            })?;

    info!("Successfully created rental: {}", rental_response.rental_id);

    Ok(Json(rental_response))
}

/// Get rental status
#[utoipa::path(
    get,
    path = "/api/v1/rentals/{rental_id}",
    params(
        ("rental_id" = String, Path, description = "Rental ID"),
    ),
    responses(
        (status = 200, description = "Rental status", body = RentalStatusResponse),
        (status = 404, description = "Rental not found", body = crate::error::ErrorResponse),
        (status = 503, description = "Service unavailable", body = crate::error::ErrorResponse),
    ),
    tag = "rentals",
)]
pub async fn get_rental_status(
    State(state): State<AppState>,
    Path(rental_id): Path<String>,
) -> Result<Json<RentalStatusResponse>> {
    debug!("Getting status for rental: {}", rental_id);

    // Create request distributor
    let distributor =
        RequestDistributor::new(state.http_client.clone(), state.load_balancer.clone());

    // Build request to forward to validators
    let validator_request = state
        .http_client
        .get(format!("/api/v1/rentals/{rental_id}"))
        .build()
        .map_err(|e| Error::Internal {
            message: format!("Failed to build request: {e}"),
        })?;

    // Try multiple validators for better availability
    match distributor.send_to_multiple(validator_request, 3).await {
        Ok(responses) => {
            // Take the first successful response
            let status = ResponseAggregator::take_first::<RentalStatusResponse>(responses).await?;
            Ok(Json(status))
        }
        Err(_) => {
            // If multiple requests fail, try a single validator
            let single_request = state
                .http_client
                .get(format!("/api/v1/rentals/{rental_id}"))
                .build()
                .map_err(|e| Error::Internal {
                    message: format!("Failed to build request: {e}"),
                })?;

            let response = distributor.send_to_single(single_request).await?;

            let status: RentalStatusResponse =
                response
                    .json()
                    .await
                    .map_err(|e| Error::ValidatorCommunication {
                        message: format!("Failed to parse validator response: {e}"),
                    })?;

            Ok(Json(status))
        }
    }
}

/// Terminate a rental
#[utoipa::path(
    post,
    path = "/api/v1/rentals/{rental_id}/terminate",
    params(
        ("rental_id" = String, Path, description = "Rental ID"),
    ),
    request_body = TerminateRentalRequest,
    responses(
        (status = 200, description = "Rental terminated", body = TerminateRentalResponse),
        (status = 404, description = "Rental not found", body = crate::error::ErrorResponse),
        (status = 503, description = "Service unavailable", body = crate::error::ErrorResponse),
    ),
    tag = "rentals",
)]
pub async fn terminate_rental(
    State(state): State<AppState>,
    Path(rental_id): Path<String>,
    Json(request): Json<TerminateRentalRequest>,
) -> Result<Json<TerminateRentalResponse>> {
    info!("Terminating rental: {}", rental_id);

    // Create request distributor
    let distributor =
        RequestDistributor::new(state.http_client.clone(), state.load_balancer.clone());

    // Build request to forward to validator
    let validator_request = state
        .http_client
        .post(format!("/api/v1/rentals/{rental_id}/terminate"))
        .json(&request)
        .build()
        .map_err(|e| Error::Internal {
            message: format!("Failed to build request: {e}"),
        })?;

    // Send to validator
    let response = distributor.send_to_single(validator_request).await?;

    // Parse response
    let terminate_response: TerminateRentalResponse =
        response
            .json()
            .await
            .map_err(|e| Error::ValidatorCommunication {
                message: format!("Failed to parse validator response: {e}"),
            })?;

    info!("Successfully terminated rental: {}", rental_id);

    Ok(Json(terminate_response))
}
