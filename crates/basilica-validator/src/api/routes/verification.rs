use axum::{
    extract::{Path, State},
    Json,
};
use serde_json::Value;
use tracing::error;

use crate::api::{types::ApiError, ApiState};
// Verification Workflow
pub async fn list_active_verifications(
    State(state): State<ApiState>,
) -> Result<Json<Value>, ApiError> {
    match state
        .persistence
        .query_verification_logs(None, Some(false), 10, 0)
        .await
    {
        Ok(logs) => {
            let response = serde_json::to_value(logs).unwrap();
            Ok(Json(response))
        }
        Err(e) => {
            error!("Failed to query verification_logs: {}", e);
            Err(ApiError::InternalError("Database error".to_string()))
        }
    }
}
pub async fn get_verification_results(
    State(state): State<ApiState>,
    Path(node_id): Path<String>,
) -> Result<Json<Value>, ApiError> {
    match state
        .persistence
        .query_verification_logs(Some(&node_id), None, 10, 0)
        .await
    {
        Ok(logs) => {
            let response = serde_json::to_value(logs).unwrap();
            Ok(Json(response))
        }
        Err(e) => {
            error!("Failed to query verification_logs: {}", e);
            Err(ApiError::InternalError("Database error".to_string()))
        }
    }
}
// pub async fn trigger_verification(State(_state): State<ApiState>) -> StatusCode {
//     StatusCode::NOT_IMPLEMENTED
// }
