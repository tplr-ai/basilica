//! Rental management route handlers

use crate::{
    api::{
        extractors::ownership::{
            archive_rental_ownership, get_user_rentals_with_details, store_rental_ownership,
            OwnedRental,
        },
        middleware::AuthContext,
    },
    country_mapping::normalize_country_code,
    error::Result,
    server::AppState,
};
use axum::{
    extract::{Query, State},
    http::Uri,
    response::{sse::Event, IntoResponse, Response, Sse},
    Json,
};
use basilica_common::utils::validate_docker_image;
use basilica_sdk::types::{
    ApiListRentalsResponse, ApiRentalListItem, ListRentalsQuery, LogStreamQuery, NodeSelection,
    RentalStatusWithSshResponse, StartRentalApiRequest, TerminateRentalRequest,
};
use basilica_validator::{
    api::{
        routes::rentals::StartRentalRequest,
        types::{AvailableNode, ListAvailableNodesQuery, ListAvailableNodesResponse},
    },
    RentalResponse,
};
use futures::stream::Stream;
use rand::seq::SliceRandom;
use tracing::{debug, error, info};

/// Get detailed rental status (with ownership validation)
pub async fn get_rental_status(
    State(state): State<AppState>,
    owned_rental: OwnedRental,
) -> Result<Json<RentalStatusWithSshResponse>> {
    debug!("Getting status for rental: {}", owned_rental.rental_id);

    let client = &state.validator_client;
    let validator_response = client.get_rental_status(&owned_rental.rental_id).await?;

    // Deserialize port mappings from JSON
    let port_mappings = owned_rental.port_mappings.and_then(|json| {
        serde_json::from_value::<Vec<basilica_validator::rental::PortMapping>>(json).ok()
    });

    // Create extended response with SSH credentials and port mappings from database
    let response_with_ssh = RentalStatusWithSshResponse::from_validator_response(
        validator_response,
        owned_rental.ssh_credentials,
        port_mappings,
    );

    Ok(Json(response_with_ssh))
}

// ===== New Validator-Compatible Endpoints =====

/// Start a new rental (validator-compatible endpoint)
pub async fn start_rental(
    State(state): State<AppState>,
    axum::Extension(auth_context): axum::Extension<AuthContext>,
    Json(request): Json<StartRentalApiRequest>,
) -> Result<Json<RentalResponse>> {
    // Get user ID from auth context (already extracted via Extension)
    let user_id = &auth_context.user_id;

    // Validate SSH public key
    if !is_valid_ssh_public_key(&request.ssh_public_key) {
        error!("Invalid SSH public key provided");
        return Err(crate::error::ApiError::BadRequest {
            message: "Invalid SSH public key".into(),
        });
    }

    // Validate container image using OCI specification
    if let Err(e) = validate_docker_image(&request.container_image) {
        error!("Invalid container image provided: {}", e);
        return Err(crate::error::ApiError::BadRequest {
            message: format!("Invalid container image: {}", e),
        });
    }

    // Determine node_id based on the selection strategy
    let node_id = match &request.node_selection {
        NodeSelection::NodeId { node_id } => {
            info!("Starting rental with specified node: {}", node_id);
            node_id.clone()
        }
        NodeSelection::ExactGpuConfiguration { gpu_requirements } => {
            info!(
                "Selecting node based on GPU requirements (exact): {:?}",
                gpu_requirements
            );

            // Query available nodes with filters based on requirements
            let query = ListAvailableNodesQuery {
                available: Some(true),
                min_gpu_memory: Some(gpu_requirements.min_memory_gb),
                gpu_type: gpu_requirements.gpu_type.clone(),
                min_gpu_count: Some(gpu_requirements.gpu_count),
                location: None,
            };

            let nodes_response = state
                .validator_client
                .list_available_nodes(Some(query))
                .await
                .map_err(|e| crate::error::ApiError::Internal {
                    message: format!("Failed to query available nodes: {}", e),
                })?;

            // Filter for exact GPU count
            let exact_count = gpu_requirements.gpu_count as usize;
            let nodes: Vec<_> = nodes_response
                .available_nodes
                .into_iter()
                .filter(|exec| exec.node.gpu_specs.len() == exact_count)
                .collect();

            if nodes.is_empty() {
                error!("No nodes with exactly {} GPU(s) available", exact_count);
                return Err(crate::error::ApiError::NotFound {
                    message: format!(
                        "No nodes with exactly {} GPU(s) matching requirements",
                        exact_count
                    ),
                });
            }

            // Randomly select an node from the filtered list
            let selected_id =
                select_best_node(nodes).ok_or_else(|| crate::error::ApiError::Internal {
                    message: "Failed to select node".into(),
                })?;

            info!(
                "Selected node {} with exactly {} GPU(s)",
                selected_id, exact_count
            );
            selected_id
        }
    };

    // Convert to validator's StartRentalRequest format
    let validator_request = StartRentalRequest {
        node_id,
        container_image: request.container_image,
        ssh_public_key: request.ssh_public_key,
        environment: request.environment,
        ports: request.ports,
        resources: request.resources,
        command: request.command,
        volumes: request.volumes,
        no_ssh: request.no_ssh,
    };
    debug!("Starting rental with request: {:?}", validator_request);

    let validator_response = state
        .validator_client
        .start_rental(validator_request)
        .await?;

    // Serialize port mappings from validator response
    let port_mappings_json = if !validator_response.container_info.mapped_ports.is_empty() {
        Some(serde_json::to_value(&validator_response.container_info.mapped_ports).ok())
    } else {
        None
    };

    // Store ownership record in database with SSH credentials and port mappings
    if let Err(e) = store_rental_ownership(
        &state.db,
        &validator_response.rental_id,
        user_id,
        validator_response.ssh_credentials.as_deref(),
        port_mappings_json.flatten(),
    )
    .await
    {
        error!(
            "Failed to store rental ownership for rental {}: {}. Rolling back rental creation.",
            validator_response.rental_id, e
        );

        // Rollback: terminate the rental on the validator since we can't track ownership
        let rollback_request = TerminateRentalRequest {
            reason: Some("Failed to store ownership record - automatic rollback".to_string()),
        };

        if let Err(rollback_err) = state
            .validator_client
            .terminate_rental(&validator_response.rental_id, rollback_request)
            .await
        {
            error!(
                "CRITICAL: Failed to rollback rental {} after ownership storage failure: {}. Manual cleanup required.",
                validator_response.rental_id, rollback_err
            );
        } else {
            info!(
                "Successfully rolled back rental {} after ownership storage failure",
                validator_response.rental_id
            );
        }

        // Return error to the user
        return Err(crate::error::ApiError::Internal {
            message: "Failed to create rental: unable to store ownership record".into(),
        });
    }

    info!(
        "User {} started rental {}",
        user_id, validator_response.rental_id
    );

    Ok(Json(validator_response))
}

/// Stop a rental (with ownership validation)
pub async fn stop_rental(
    State(state): State<AppState>,
    owned_rental: OwnedRental,
) -> Result<Response> {
    info!(
        "User {} stopping rental {}",
        owned_rental.user_id, owned_rental.rental_id
    );

    // Use terminate_rental API from validator
    let request = TerminateRentalRequest {
        reason: Some("User requested stop".to_string()),
    };

    state
        .validator_client
        .terminate_rental(&owned_rental.rental_id, request.clone())
        .await?;

    // Archive ownership record to terminated_user_rentals table
    if let Err(e) = archive_rental_ownership(
        &state.db,
        &owned_rental.rental_id,
        request.reason.as_deref(),
    )
    .await
    {
        error!("Failed to archive rental ownership record: {}", e);
        // Note: We don't fail the request if ownership archiving fails
    }

    Ok(axum::http::StatusCode::NO_CONTENT.into_response())
}

/// Stream rental logs (with ownership validation)
pub async fn stream_rental_logs(
    State(state): State<AppState>,
    owned_rental: OwnedRental,
    Query(query): Query<LogStreamQuery>,
) -> Result<Sse<impl Stream<Item = std::result::Result<Event, std::io::Error>>>> {
    info!(
        "User {} streaming logs for rental {}",
        owned_rental.user_id, owned_rental.rental_id
    );

    let follow = query.follow.unwrap_or(false);
    let tail_lines = query.tail;

    // Create query parameters for validator
    let log_query = basilica_validator::api::types::LogQuery {
        follow: Some(follow),
        tail: tail_lines,
    };

    // Get SSE stream from validator
    let validator_stream = state
        .validator_client
        .stream_rental_logs(&owned_rental.rental_id, log_query)
        .await
        .map_err(|e| {
            error!("Failed to get log stream from validator: {}", e);
            crate::error::ApiError::ValidatorCommunication {
                message: format!("Failed to stream logs: {}", e),
            }
        })?;

    // Convert validator Event stream to axum SSE Events
    let stream = async_stream::stream! {
        use futures_util::StreamExt;
        futures_util::pin_mut!(validator_stream);

        while let Some(result) = validator_stream.next().await {
            match result {
                Ok(event) => {
                    // Convert validator Event to SSE data
                    let data = serde_json::json!({
                        "timestamp": event.timestamp,
                        "stream": event.stream,
                        "message": event.message,
                    });

                    yield Ok(Event::default().data(data.to_string()));
                }
                Err(e) => {
                    error!("Error in log stream: {}", e);
                    // Send error as an SSE event
                    let data = serde_json::json!({
                        "timestamp": chrono::Utc::now(),
                        "stream": "error",
                        "message": format!("Stream error: {}", e),
                    });
                    yield Ok(Event::default().data(data.to_string()));
                    break;
                }
            }
        }
    };

    Ok(Sse::new(stream))
}

/// List rentals with state filter (validator-compatible)
/// Only returns rentals owned by the authenticated user
pub async fn list_rentals_validator(
    State(state): State<AppState>,
    axum::Extension(auth_context): axum::Extension<AuthContext>,
    Query(query): Query<ListRentalsQuery>,
) -> Result<Json<ApiListRentalsResponse>> {
    info!("Listing rentals with state filter: {:?}", query.status);

    // Get user ID from auth context (already extracted via Extension)
    let user_id = &auth_context.user_id;

    // Get user's rental IDs with SSH status and port mappings from database
    let user_rentals_with_details = get_user_rentals_with_details(&state.db, user_id)
        .await
        .map_err(|e| crate::error::ApiError::Internal {
            message: format!("Failed to get user rentals: {}", e),
        })?;

    // Create maps for quick lookup of SSH status and port mappings
    let mut ssh_status_map = std::collections::HashMap::new();
    let mut port_mappings_map = std::collections::HashMap::new();
    for rental in &user_rentals_with_details {
        ssh_status_map.insert(rental.rental_id.clone(), rental.has_ssh);
        port_mappings_map.insert(rental.rental_id.clone(), rental.port_mappings.clone());
    }

    // Get all rentals from validator
    let all_rentals = state
        .validator_client
        .list_rentals(query.status)
        .await
        .map_err(|e| crate::error::ApiError::ValidatorCommunication {
            message: format!("Failed to list rentals: {e}"),
        })?;

    // Filter to only include user's rentals and use node details from validator response
    let mut api_rentals = Vec::new();

    for rental in all_rentals.rentals {
        // Check if user owns this rental and get SSH status
        let has_ssh = match ssh_status_map.get(&rental.rental_id) {
            Some(&has_ssh) => has_ssh,
            None => continue, // User doesn't own this rental
        };

        // Get port mappings from database and deserialize
        let port_mappings = port_mappings_map
            .get(&rental.rental_id)
            .and_then(|json_opt| json_opt.as_ref())
            .and_then(|json| {
                serde_json::from_value::<Vec<basilica_validator::rental::PortMapping>>(json.clone())
                    .ok()
            });

        // Create API rental item with node details from validator response
        api_rentals.push(ApiRentalListItem {
            rental_id: rental.rental_id,
            node_id: rental.node_id,
            container_id: rental.container_id,
            state: rental.state,
            created_at: rental.created_at,
            miner_id: rental.miner_id,
            container_image: rental.container_image,
            gpu_specs: rental.gpu_specs.unwrap_or_default(),
            has_ssh,
            cpu_specs: rental.cpu_specs,
            location: rental.location,
            network_speed: rental.network_speed,
            port_mappings,
        });
    }

    let filtered_count = api_rentals.len();

    let user_rentals = ApiListRentalsResponse {
        rentals: api_rentals,
        total_count: filtered_count,
    };

    info!(
        "User {} has {} rentals",
        user_id,
        user_rentals.rentals.len()
    );

    Ok(Json(user_rentals))
}

// Validation helpers
fn is_valid_ssh_public_key(key: &str) -> bool {
    if key.trim().is_empty() {
        return false;
    }

    // Must start with ssh- prefix
    if !key.starts_with("ssh-") {
        return false;
    }

    // Must have at least 2 parts (algorithm and key data)
    let parts: Vec<&str> = key.split_whitespace().collect();
    if parts.len() < 2 {
        return false;
    }

    true
}

/// List available nodes for rentals
pub async fn list_available_nodes(
    State(state): State<AppState>,
    Query(mut query): Query<ListAvailableNodesQuery>,
    uri: Uri,
) -> Result<Json<ListAvailableNodesResponse>> {
    // Default to available=true for /nodes endpoint
    if query.available.is_none() && uri.path() == "/nodes" {
        query.available = Some(true);
    }

    // Normalize country code if location is provided
    if let Some(ref mut location) = query.location {
        if let Some(ref country) = location.country {
            location.country = Some(normalize_country_code(country));
        }
    }

    info!("Listing nodes with filters: {:?}", query);

    let response = state
        .validator_client
        .list_available_nodes(Some(query))
        .await?;

    Ok(Json(response))
}

/// Select a random node from a list of available nodes to distribute
/// load and allow users to retry with different nodes if issues occur
fn select_best_node(nodes: Vec<AvailableNode>) -> Option<String> {
    if nodes.is_empty() {
        return None;
    }

    // Randomly select an node from the available list
    let mut rng = rand::thread_rng();
    nodes.choose(&mut rng).map(|e| e.node.id.clone())
}
