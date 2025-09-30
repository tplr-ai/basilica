//! Health check route handler

use crate::server::AppState;
use axum::{extract::State, Json};
use basilica_sdk::types::HealthCheckResponse;

/// Health check endpoint
pub async fn health_check(State(_state): State<AppState>) -> Json<HealthCheckResponse> {
    // We always have one configured validator
    // Health status is monitored in background but doesn't affect API availability
    Json(HealthCheckResponse {
        status: "healthy".to_string(),
        version: crate::VERSION.to_string(),
        timestamp: chrono::Utc::now(),
        healthy_validators: 1,
        total_validators: 1,
    })
}
