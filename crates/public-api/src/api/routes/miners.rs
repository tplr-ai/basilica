//! Miner information route handlers

use crate::{
    aggregator::ResponseAggregator,
    api::types::{ListMinersQuery, ListMinersResponse, MinerDetails},
    error::{Error, Result},
    server::AppState,
};
use axum::{
    extract::{Path, Query, State},
    Json,
};
use std::collections::HashMap;
use tracing::{debug, warn};

/// List miners
#[utoipa::path(
    get,
    path = "/api/v1/miners",
    params(
        ("min_gpu_count" = Option<u32>, Query, description = "Minimum GPU count"),
        ("min_score" = Option<f64>, Query, description = "Minimum score"),
        ("page" = Option<u32>, Query, description = "Page number"),
        ("page_size" = Option<u32>, Query, description = "Page size"),
    ),
    responses(
        (status = 200, description = "List of miners", body = ListMinersResponse),
        (status = 503, description = "Service unavailable", body = crate::error::ErrorResponse),
    ),
    tag = "miners",
)]
pub async fn list_miners(
    State(state): State<AppState>,
    Query(query): Query<ListMinersQuery>,
) -> Result<Json<ListMinersResponse>> {
    let page = query.page.unwrap_or(1);
    let page_size = query.page_size.unwrap_or(20).min(100);

    // Get healthy validators
    let validators = state.discovery.get_healthy_validators();
    if validators.is_empty() {
        return Err(Error::NoValidatorsAvailable);
    }

    // Prepare request for validators
    let request_path = format!("/api/v1/miners?page={page}&page_size={page_size}");

    // Add query parameters
    let mut query_params = vec![];
    if let Some(min_gpu) = query.min_gpu_count {
        query_params.push(format!("min_gpu_count={min_gpu}"));
    }
    if let Some(min_score) = query.min_score {
        query_params.push(format!("min_score={min_score}"));
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

    // Merge miner lists from all validators
    let mut all_miners: HashMap<String, MinerDetails> = HashMap::new();
    let mut total_count = 0;

    for response in responses {
        if let Ok(data) = serde_json::from_value::<ListMinersResponse>(response.data) {
            total_count = total_count.max(data.total_count);

            for miner in data.miners {
                // Use miner ID (hotkey) as key to deduplicate
                all_miners.insert(miner.miner_id.clone(), miner);
            }
        }
    }

    // Convert to vector and apply filters
    let mut miners: Vec<MinerDetails> = all_miners.into_values().collect();

    // Apply GPU count filter if not already applied by validators
    if let Some(min_gpu) = query.min_gpu_count {
        miners.retain(|m| m.total_gpu_count >= min_gpu);
    }

    // Apply score filter if not already applied by validators
    if let Some(min_score) = query.min_score {
        miners.retain(|m| m.verification_score >= min_score);
    }

    // Sort by verification score (descending) for consistent ordering
    miners.sort_by(|a, b| {
        b.verification_score
            .partial_cmp(&a.verification_score)
            .unwrap()
    });

    // Apply pagination
    let start = ((page - 1) * page_size) as usize;
    let end = (start + page_size as usize).min(miners.len());
    let paginated_miners = miners[start..end].to_vec();

    Ok(Json(ListMinersResponse {
        miners: paginated_miners,
        total_count: miners.len(),
        page,
        page_size,
    }))
}

/// Get miner by ID
#[utoipa::path(
    get,
    path = "/api/v1/miners/{miner_id}",
    params(
        ("miner_id" = String, Path, description = "Miner ID"),
    ),
    responses(
        (status = 200, description = "Miner details", body = MinerDetails),
        (status = 404, description = "Miner not found", body = crate::error::ErrorResponse),
    ),
    tag = "miners",
)]
pub async fn get_miner(
    State(state): State<AppState>,
    Path(miner_id): Path<String>,
) -> Result<Json<MinerDetails>> {
    debug!("Getting miner details for ID: {}", miner_id);

    // Get healthy validators
    let validators = state.discovery.get_healthy_validators();
    if validators.is_empty() {
        return Err(Error::NoValidatorsAvailable);
    }

    // Request path
    let request_path = format!("/api/v1/miners/{miner_id}");

    // Try each validator until we find the miner
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
                    match response.json::<MinerDetails>().await {
                        Ok(miner) => return Ok(Json(miner)),
                        Err(e) => {
                            warn!(
                                "Failed to parse miner response from {}: {}",
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
        resource: format!("Miner {miner_id}"),
    })
}
