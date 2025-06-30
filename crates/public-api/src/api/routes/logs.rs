//! Log streaming route handlers

use crate::{
    api::types::LogQuery,
    error::{Error, Result},
    server::AppState,
};
use axum::{
    extract::{Path, Query, State},
    response::sse::{Event, KeepAlive, Sse},
};
use futures::{Stream, StreamExt};
use std::convert::Infallible;
use std::time::Duration;
use tracing::{debug, error, warn};

/// Stream rental logs
#[utoipa::path(
    get,
    path = "/api/v1/rentals/{rental_id}/logs",
    params(
        ("rental_id" = String, Path, description = "Rental ID"),
        ("follow" = Option<bool>, Query, description = "Follow logs"),
        ("tail" = Option<u32>, Query, description = "Number of lines to tail"),
    ),
    responses(
        (status = 200, description = "Log stream", content_type = "text/event-stream"),
        (status = 404, description = "Rental not found", body = crate::error::ErrorResponse),
    ),
    tag = "logs",
)]
pub async fn stream_rental_logs(
    State(state): State<AppState>,
    Path(rental_id): Path<String>,
    Query(query): Query<LogQuery>,
) -> Result<Sse<impl Stream<Item = std::result::Result<Event, Infallible>>>> {
    debug!("Starting log stream for rental: {}", rental_id);

    // Get healthy validators
    let validators = state.discovery.get_healthy_validators();
    if validators.is_empty() {
        return Err(Error::NoValidatorsAvailable);
    }

    // Find which validator has this rental
    let mut rental_validator = None;
    for validator in &validators {
        let url = format!("{}/api/v1/rentals/{}", validator.endpoint, rental_id);

        match state
            .http_client
            .get(&url)
            .timeout(Duration::from_secs(5))
            .send()
            .await
        {
            Ok(response) => {
                if response.status().is_success() {
                    rental_validator = Some(validator.clone());
                    break;
                }
            }
            Err(e) => {
                warn!(
                    "Failed to check rental on validator {}: {}",
                    validator.endpoint, e
                );
            }
        }
    }

    let validator = rental_validator.ok_or_else(|| Error::NotFound {
        resource: format!("Rental {rental_id}"),
    })?;

    // Build SSE URL for the validator
    let sse_url = format!(
        "{}/api/v1/rentals/{}/logs?follow={}&tail={}",
        validator.endpoint,
        rental_id,
        query.follow.unwrap_or(true),
        query.tail.unwrap_or(100)
    );

    // Create the stream
    let stream = create_log_stream(state.http_client.clone(), sse_url, rental_id);

    Ok(Sse::new(stream).keep_alive(KeepAlive::default()))
}

/// Create a log stream from a validator
fn create_log_stream(
    client: reqwest::Client,
    url: String,
    rental_id: String,
) -> impl Stream<Item = std::result::Result<Event, Infallible>> {
    async_stream::stream! {
        debug!("Connecting to log stream: {}", url);

        // Send initial connection event
        yield Ok(Event::default()
            .event("connected")
            .data(format!("Connected to log stream for rental {rental_id}")));

        // Connect to the validator's SSE endpoint
        match client.get(&url).send().await {
            Ok(response) => {
                if !response.status().is_success() {
                    yield Ok(Event::default()
                        .event("error")
                        .data(format!("Validator returned error: {}", response.status())));
                    return;
                }

                // Stream the response bytes
                let mut stream = response.bytes_stream();
                let mut buffer = String::new();

                while let Some(chunk_result) = stream.next().await {
                    match chunk_result {
                        Ok(chunk) => {
                            // Convert bytes to string
                            if let Ok(text) = std::str::from_utf8(&chunk) {
                                buffer.push_str(text);

                                // Process complete lines
                                while let Some(line_end) = buffer.find('\n') {
                                    let line = buffer[..line_end].to_string();
                                    buffer = buffer[line_end + 1..].to_string();

                                    // Parse SSE format
                                    if line.starts_with("data: ") {
                                        let data = line[6..].to_string();
                                        yield Ok(Event::default().data(data));
                                    } else if line.starts_with("event: ") {
                                        let event_type = line[7..].to_string();
                                        if event_type == "done" {
                                            yield Ok(Event::default().event("done").data(""));
                                            return;
                                        }
                                    }
                                }
                            }
                        }
                        Err(e) => {
                            error!("Error reading log stream: {}", e);
                            yield Ok(Event::default()
                                .event("error")
                                .data(format!("Stream error: {e}")));
                            return;
                        }
                    }
                }

                // Send completion event
                yield Ok(Event::default()
                    .event("complete")
                    .data("Log stream completed"));
            }
            Err(e) => {
                error!("Failed to connect to log stream: {}", e);
                yield Ok(Event::default()
                    .event("error")
                    .data(format!("Failed to connect to validator: {e}")));
            }
        }
    }
}
