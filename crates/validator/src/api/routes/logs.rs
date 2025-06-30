//! Log streaming routes

use crate::api::types::*;
use crate::api::ApiState;
use axum::{
    extract::{Path, Query, State},
    response::sse::Event,
};
use futures::{stream, Stream};
use std::convert::Infallible;
use std::time::Duration;
use tracing::{error, info};
use uuid::Uuid;

/// Stream rental logs
pub async fn stream_rental_logs(
    State(state): State<ApiState>,
    Path(rental_id): Path<String>,
    Query(query): Query<LogQuery>,
) -> Result<axum::response::sse::Sse<impl Stream<Item = Result<Event, Infallible>>>, ApiError> {
    info!("Streaming logs for rental {}", rental_id);

    // Parse rental ID
    let rental_uuid = Uuid::parse_str(&rental_id)
        .map_err(|_| ApiError::BadRequest("Invalid rental ID format".to_string()))?;

    // Validate rental exists
    match state.persistence.get_rental(&rental_uuid).await {
        Ok(Some(rental)) => {
            if !rental.is_active() {
                return Err(ApiError::BadRequest("Rental is not active".to_string()));
            }

            // Get verification logs related to this executor
            let logs_result = state
                .persistence
                .query_verification_logs(
                    Some(&rental.executor_id),
                    None, // Include both success and failure logs
                    query.tail.unwrap_or(100),
                    0,
                )
                .await;

            match logs_result {
                Ok(logs) => {
                    // Create log stream from verification logs
                    let log_events: Vec<Result<Event, Infallible>> = logs
                        .into_iter()
                        .map(|log| {
                            let log_data = serde_json::json!({
                                "timestamp": log.timestamp,
                                "level": if log.success { "INFO" } else { "ERROR" },
                                "message": format!("Verification {}: score={:.2}, duration={}ms",
                                                 log.verification_type, log.score, log.duration_ms),
                                "executor_id": log.executor_id,
                                "details": log.details,
                                "error": log.error_message
                            });

                            Ok(Event::default()
                                .json_data(&log_data)
                                .unwrap_or_else(|_| Event::default().data("Invalid log data")))
                        })
                        .collect();

                    // Add a final status message
                    let mut all_events = log_events;
                    all_events.push(Ok(Event::default().data(format!(
                        "End of logs for rental {} (showing last {} entries)",
                        rental_id,
                        query.tail.unwrap_or(100)
                    ))));

                    // If follow is enabled, add heartbeat message
                    if query.follow.unwrap_or(false) {
                        all_events.push(Ok(Event::default()
                            .event("heartbeat")
                            .data(format!("Rental {rental_id} monitoring active"))));

                        Ok(
                            axum::response::sse::Sse::new(stream::iter(all_events)).keep_alive(
                                axum::response::sse::KeepAlive::new()
                                    .interval(Duration::from_secs(30))
                                    .text("keep-alive"),
                            ),
                        )
                    } else {
                        Ok(axum::response::sse::Sse::new(stream::iter(all_events)))
                    }
                }
                Err(e) => {
                    error!("Failed to query verification logs: {}", e);
                    Err(ApiError::InternalError(
                        "Failed to retrieve logs".to_string(),
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
