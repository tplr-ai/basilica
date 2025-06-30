//! Executor management route handlers

use crate::{
    aggregator::ResponseAggregator,
    api::types::{ExecutorDetails, ListExecutorsQuery, ListExecutorsResponse},
    error::{Error, Result},
    server::AppState,
};
use axum::{
    extract::{Path, Query, State},
    Json,
};
use std::collections::HashMap;
use tracing::{debug, warn};

/// List executors
#[utoipa::path(
    get,
    path = "/api/v1/executors",
    params(
        ("min_gpu_count" = Option<u32>, Query, description = "Minimum GPU count"),
        ("gpu_type" = Option<String>, Query, description = "GPU type filter"),
        ("page" = Option<u32>, Query, description = "Page number"),
        ("page_size" = Option<u32>, Query, description = "Page size"),
    ),
    responses(
        (status = 200, description = "List of executors", body = ListExecutorsResponse),
        (status = 503, description = "Service unavailable", body = crate::error::ErrorResponse),
    ),
    tag = "executors",
)]
pub async fn list_executors(
    State(state): State<AppState>,
    Query(query): Query<ListExecutorsQuery>,
) -> Result<Json<ListExecutorsResponse>> {
    let page = query.page.unwrap_or(1);
    let page_size = query.page_size.unwrap_or(20).min(100);

    // Get healthy validators
    let validators = state.discovery.get_healthy_validators();
    if validators.is_empty() {
        return Err(Error::NoValidatorsAvailable);
    }

    // Prepare request for validators
    let request_path = format!("/api/v1/executors?page={page}&page_size={page_size}");

    // Add query parameters
    let mut query_params = vec![];
    if let Some(min_gpu) = query.min_gpu_count {
        query_params.push(format!("min_gpu_count={min_gpu}"));
    }
    if let Some(gpu_type) = &query.gpu_type {
        query_params.push(format!("gpu_type={gpu_type}"));
    }

    let full_path = if query_params.is_empty() {
        request_path
    } else {
        format!("{}&{}", request_path, query_params.join("&"))
    };

    // Create aggregator
    let aggregator = ResponseAggregator::new(
        state.http_client.clone(),
        validators.clone(),
        state.config.request_timeout(),
    );

    // Aggregate responses from validators
    let responses = aggregator.aggregate_get_requests(&full_path, None).await?;

    // Merge executor lists from all validators
    let mut all_executors: HashMap<String, ExecutorDetails> = HashMap::new();
    let mut total_count = 0;

    for response in responses {
        if let Ok(data) = serde_json::from_value::<ListExecutorsResponse>(response.data) {
            total_count = total_count.max(data.total_count);

            for executor in data.executors {
                // Use executor ID as key to deduplicate
                all_executors.insert(executor.id.clone(), executor);
            }
        }
    }

    // Convert to vector and apply filters
    let mut executors: Vec<ExecutorDetails> = all_executors.into_values().collect();

    // Apply GPU count filter if not already applied by validators
    if let Some(min_gpu) = query.min_gpu_count {
        executors.retain(|e| e.gpu_specs.len() >= min_gpu as usize);
    }

    // Apply GPU type filter if not already applied by validators
    if let Some(gpu_type) = &query.gpu_type {
        executors.retain(|e| e.gpu_specs.iter().any(|gpu| gpu.name.contains(gpu_type)));
    }

    // Sort by ID for consistent ordering
    executors.sort_by(|a, b| a.id.cmp(&b.id));

    // Apply pagination
    let start = ((page - 1) * page_size) as usize;
    let end = (start + page_size as usize).min(executors.len());
    let paginated_executors = executors[start..end].to_vec();

    Ok(Json(ListExecutorsResponse {
        executors: paginated_executors,
        total_count: executors.len(),
        page,
        page_size,
    }))
}

/// Get executor by ID
#[utoipa::path(
    get,
    path = "/api/v1/executors/{executor_id}",
    params(
        ("executor_id" = String, Path, description = "Executor ID"),
    ),
    responses(
        (status = 200, description = "Executor details", body = ExecutorDetails),
        (status = 404, description = "Executor not found", body = crate::error::ErrorResponse),
    ),
    tag = "executors",
)]
pub async fn get_executor(
    State(state): State<AppState>,
    Path(executor_id): Path<String>,
) -> Result<Json<ExecutorDetails>> {
    debug!("Getting executor details for ID: {}", executor_id);

    // Get healthy validators
    let validators = state.discovery.get_healthy_validators();
    if validators.is_empty() {
        return Err(Error::NoValidatorsAvailable);
    }

    // Request path
    let request_path = format!("/api/v1/executors/{executor_id}");

    // Try each validator until we find the executor
    for validator in validators {
        let url = format!("{}{}", validator.endpoint, request_path);

        match state
            .http_client
            .get(&url)
            .timeout(state.config.request_timeout())
            .send()
            .await
        {
            Ok(response) => {
                if response.status().is_success() {
                    match response.json::<ExecutorDetails>().await {
                        Ok(executor) => return Ok(Json(executor)),
                        Err(e) => {
                            warn!(
                                "Failed to parse executor response from {}: {}",
                                validator.endpoint, e
                            );
                        }
                    }
                } else if response.status().as_u16() != 404 {
                    warn!(
                        "Validator {} returned error status: {}",
                        validator.endpoint,
                        response.status()
                    );
                }
            }
            Err(e) => {
                warn!("Failed to query validator {}: {}", validator.endpoint, e);
            }
        }
    }

    Err(Error::NotFound {
        resource: format!("Executor {executor_id}"),
    })
}
