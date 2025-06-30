//! Health check route handler

use crate::{api::types::HealthCheckResponse, server::AppState};
use axum::{extract::State, Json};

/// Health check endpoint
#[utoipa::path(
    get,
    path = "/api/v1/health",
    responses(
        (status = 200, description = "Service is healthy", body = HealthCheckResponse),
    ),
    tag = "health",
)]
pub async fn health_check(State(state): State<AppState>) -> Json<HealthCheckResponse> {
    let discovery = &state.discovery;

    Json(HealthCheckResponse {
        status: "healthy".to_string(),
        version: crate::VERSION.to_string(),
        timestamp: chrono::Utc::now(),
        healthy_validators: discovery.healthy_validator_count(),
        total_validators: discovery.validator_count(),
    })
}
