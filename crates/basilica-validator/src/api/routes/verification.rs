use axum::{
    extract::{Path, State},
    Json,
};
use tracing::error;

use crate::api::{
    types::{ApiError, VerificationLogResponse, VerificationLogsResponse},
    ApiState,
};

fn build_verification_logs_response(
    logs: Vec<crate::persistence::entities::VerificationLog>,
) -> VerificationLogsResponse {
    VerificationLogsResponse {
        total_count: logs.len(),
        logs: logs
            .into_iter()
            .map(|log| VerificationLogResponse {
                id: log.id.to_string(),
                node_id: log.node_id,
                validator_hotkey: log.validator_hotkey,
                verification_type: log.verification_type,
                timestamp: log.timestamp,
                score: log.score,
                success: log.success,
                details: log.details,
                duration_ms: log.duration_ms,
                error_message: log.error_message,
                created_at: log.created_at,
                updated_at: log.updated_at,
            })
            .collect(),
    }
}

// Verification Workflow
pub async fn list_active_verifications(
    State(state): State<ApiState>,
) -> Result<Json<VerificationLogsResponse>, ApiError> {
    match state
        .persistence
        .query_verification_logs(None, Some(false), 10, 0)
        .await
    {
        Ok(logs) => Ok(Json(build_verification_logs_response(logs))),
        Err(e) => {
            error!("Failed to query verification_logs: {}", e);
            Err(ApiError::InternalError("Database error".to_string()))
        }
    }
}
pub async fn get_verification_results(
    State(state): State<ApiState>,
    Path(node_id): Path<String>,
) -> Result<Json<VerificationLogsResponse>, ApiError> {
    match state
        .persistence
        .query_verification_logs(Some(&node_id), None, 10, 0)
        .await
    {
        Ok(logs) => Ok(Json(build_verification_logs_response(logs))),
        Err(e) => {
            error!("Failed to query verification_logs: {}", e);
            Err(ApiError::InternalError("Database error".to_string()))
        }
    }
}
// pub async fn trigger_verification(State(_state): State<ApiState>) -> StatusCode {
//     StatusCode::NOT_IMPLEMENTED
// }

#[cfg(test)]
mod tests {
    use super::build_verification_logs_response;
    use crate::persistence::entities::VerificationLog;
    use serde_json::json;

    #[test]
    fn wraps_verification_logs_in_typed_response() {
        let response = build_verification_logs_response(vec![VerificationLog::new(
            "node-1".to_string(),
            "validator-hotkey".to_string(),
            "ssh_automation".to_string(),
            0.9,
            true,
            json!({ "step": "ping" }),
            125,
            None,
        )]);

        assert_eq!(response.total_count, 1);
        assert_eq!(response.logs[0].node_id, "node-1");
        assert_eq!(response.logs[0].verification_type, "ssh_automation");
        assert_eq!(response.logs[0].details["step"], "ping");
    }
}
