//! Rental API routes
//!
//! HTTP endpoints for container rental operations

use anyhow::Result;
use axum::{
    extract::{Path, Query, State},
    http::StatusCode,
    response::{sse::Event, IntoResponse, Sse},
    Json,
};
use basilica_common::types::GpuCategory;
use basilica_common::utils::validate_docker_image;
use futures::stream::Stream;
use serde::Deserialize;
use ssh_key::PublicKey;
use tracing::{error, info};

use crate::{
    api::types::{ListRentalsResponse, RentalStatusResponse},
    persistence::validator_persistence::ValidatorPersistence,
    rental::{RentalInfo, RentalRequest, RentalState},
};
use crate::{
    api::{types::RentalListItem, ApiState},
    rental::RentalResponse,
};

/// Start rental request
#[derive(Debug, Deserialize, serde::Serialize)]
pub struct StartRentalRequest {
    // GPU selection (required)
    /// GPU category: "H100", "A100", "B200", etc.
    pub gpu_category: String,
    /// Number of GPUs required
    pub gpu_count: u32,

    // Optional filters
    /// Minimum GPU memory in GB (e.g., 80 for 80GB)
    #[serde(default)]
    pub min_memory_gb: Option<u32>,
    /// Maximum acceptable cents/GPU-hour
    pub max_hourly_rate_cents: u32,

    // Container config
    pub container_image: String,
    pub ssh_public_key: String,
    #[serde(default)]
    pub environment: std::collections::HashMap<String, String>,
    #[serde(default)]
    pub ports: Vec<PortMappingRequest>,
    #[serde(default)]
    pub resources: ResourceRequirementsRequest,
    #[serde(default = "default_command")]
    pub command: Vec<String>,
    #[serde(default)]
    pub volumes: Vec<VolumeMountRequest>,
}

fn default_command() -> Vec<String> {
    vec!["/bin/bash".to_string()]
}

impl Default for StartRentalRequest {
    fn default() -> Self {
        Self {
            gpu_category: String::new(),
            gpu_count: 1,
            min_memory_gb: None,
            max_hourly_rate_cents: 0,
            container_image: "nvidia/cuda:12.2.0-base-ubuntu22.04".to_string(),
            ssh_public_key: String::new(),
            environment: std::collections::HashMap::new(),
            ports: Vec::new(),
            resources: ResourceRequirementsRequest::default(),
            command: default_command(),
            volumes: Vec::new(),
        }
    }
}

/// Port mapping request
#[derive(Debug, Clone, Deserialize, serde::Serialize)]
pub struct PortMappingRequest {
    pub container_port: u32,
    pub host_port: u32,
    #[serde(default = "default_protocol")]
    pub protocol: String,
}

fn default_protocol() -> String {
    "tcp".to_string()
}

impl Default for PortMappingRequest {
    fn default() -> Self {
        Self {
            container_port: 0,
            host_port: 0,
            protocol: "tcp".to_string(),
        }
    }
}

impl From<basilica_common::utils::PortMapping> for PortMappingRequest {
    fn from(mapping: basilica_common::utils::PortMapping) -> Self {
        Self {
            container_port: mapping.container_port,
            host_port: mapping.host_port,
            protocol: mapping.protocol,
        }
    }
}

impl From<PortMappingRequest> for crate::rental::PortMapping {
    fn from(request: PortMappingRequest) -> Self {
        Self {
            container_port: request.container_port,
            host_port: request.host_port,
            protocol: request.protocol,
        }
    }
}

/// Resource requirements request
#[derive(Debug, Deserialize, serde::Serialize)]
pub struct ResourceRequirementsRequest {
    pub cpu_cores: f64,
    pub memory_mb: i64,
    pub storage_mb: i64,
    pub gpu_count: u32,
    #[serde(default)]
    pub gpu_types: Vec<String>,
}

impl Default for ResourceRequirementsRequest {
    fn default() -> Self {
        Self {
            cpu_cores: 0.0,
            memory_mb: 0,
            storage_mb: 0,
            gpu_count: 0,
            gpu_types: Vec::new(),
        }
    }
}

impl From<ResourceRequirementsRequest> for crate::rental::ResourceRequirements {
    fn from(request: ResourceRequirementsRequest) -> Self {
        Self {
            cpu_cores: request.cpu_cores,
            memory_mb: request.memory_mb,
            storage_mb: request.storage_mb,
            gpu_count: request.gpu_count,
            gpu_types: request.gpu_types,
        }
    }
}

/// Volume mount request
#[derive(Debug, Deserialize, serde::Serialize)]
pub struct VolumeMountRequest {
    pub host_path: String,
    pub container_path: String,
    #[serde(default)]
    pub read_only: bool,
}

impl From<VolumeMountRequest> for crate::rental::VolumeMount {
    fn from(request: VolumeMountRequest) -> Self {
        Self {
            host_path: request.host_path,
            container_path: request.container_path,
            read_only: request.read_only,
        }
    }
}

/// Rental status query parameters
#[derive(Debug, Deserialize)]
pub struct RentalStatusQuery {
    #[allow(dead_code)]
    pub include_resource_usage: Option<bool>,
}

/// Log streaming query parameters
#[derive(Debug, Deserialize)]
pub struct LogStreamQuery {
    pub follow: Option<bool>,
    pub tail: Option<u32>,
}

/// List rentals query parameters
#[derive(Debug, Deserialize)]
pub struct ListRentalsQuery {
    pub state: Option<RentalState>,
    /// Type of listing: "rentals" (default) or "available" for available capacity
    pub list_type: Option<String>,
    /// Filters for available capacity queries
    pub min_gpu_memory: Option<u32>,
    pub gpu_type: Option<String>,
    pub min_gpu_count: Option<u32>,
    pub max_cost_per_hour: Option<f64>,
}

/// Start a new rental
pub async fn start_rental(
    State(state): State<ApiState>,
    Json(request): Json<StartRentalRequest>,
) -> Result<Json<RentalResponse>, StatusCode> {
    // Parse and validate gpu_category using GpuCategory enum
    let gpu_category: GpuCategory = request.gpu_category.parse().unwrap(); // Infallible
    if matches!(&gpu_category, GpuCategory::Other(_)) {
        error!(
            gpu_category = %request.gpu_category,
            "[RENTAL_FLOW] GPU type '{}' is not supported", request.gpu_category
        );
        return Err(StatusCode::BAD_REQUEST);
    }

    info!(
        gpu_category = %gpu_category,
        gpu_count = request.gpu_count,
        "[RENTAL_FLOW] Starting rental for {} x {}",
        request.gpu_count,
        gpu_category
    );

    // Validate gpu_count is at least 1
    if request.gpu_count == 0 {
        error!("[RENTAL_FLOW] gpu_count must be at least 1");
        return Err(StatusCode::BAD_REQUEST);
    }

    let ssh_public_key = request.ssh_public_key.trim();
    if PublicKey::from_openssh(ssh_public_key).is_err() {
        error!(
            gpu_category = %request.gpu_category,
            gpu_count = request.gpu_count,
            "[RENTAL_FLOW] Invalid SSH public key provided"
        );
        return Err(StatusCode::BAD_REQUEST);
    }

    if let Err(e) = validate_docker_image(&request.container_image) {
        error!(
            gpu_category = %request.gpu_category,
            gpu_count = request.gpu_count,
            "[RENTAL_FLOW] Invalid container image provided: {}",
            e
        );
        return Err(StatusCode::BAD_REQUEST);
    }

    let rental_manager = state.rental_manager.as_ref().ok_or_else(|| {
        error!("[RENTAL_FLOW] Rental manager not initialized");
        StatusCode::INTERNAL_SERVER_ERROR
    })?;

    // Filter out any user-specified SSH port mappings and prepare port list
    let mut port_mappings: Vec<crate::rental::PortMapping> = request
        .ports
        .into_iter()
        .filter(|p| p.container_port != 22) // Remove any SSH port mappings
        .map(Into::into)
        .collect();

    // Always add SSH port mapping
    port_mappings.push(crate::rental::PortMapping {
        container_port: 22,
        host_port: 0, // Docker will automatically allocate an available port
        protocol: "tcp".to_string(),
    });

    // Convert request to internal rental request
    let rental_request = RentalRequest {
        validator_hotkey: state.validator_hotkey.to_string(),
        gpu_category: gpu_category.to_string(),
        gpu_count: request.gpu_count,
        min_memory_gb: request.min_memory_gb,
        max_hourly_rate_cents: request.max_hourly_rate_cents,
        container_spec: crate::rental::ContainerSpec {
            image: request.container_image,
            environment: request.environment,
            ports: port_mappings,
            resources: request.resources.into(),
            entrypoint: Vec::new(), // API currently doesn't support custom entrypoint
            command: request.command,
            volumes: request
                .volumes
                .into_iter()
                .filter(|v| !v.host_path.contains("..") && !v.container_path.contains(".."))
                .map(Into::into)
                .collect(),
            labels: std::collections::HashMap::new(),
            capabilities: Vec::new(),
            network: crate::rental::NetworkConfig {
                mode: "bridge".to_string(),
                dns: Vec::new(),
                extra_hosts: std::collections::HashMap::new(),
            },
        },
        ssh_public_key: ssh_public_key.to_string(),
        metadata: std::collections::HashMap::new(),
    };

    // Start rental
    let rental_response = rental_manager
        .start_rental(rental_request)
        .await
        .map_err(|e| {
            let error_msg = e.to_string();
            error!(
                gpu_category = %request.gpu_category,
                gpu_count = request.gpu_count,
                "[RENTAL_FLOW] Failed to start rental: {}",
                error_msg
            );
            // Return 503 if no matching capacity available
            if error_msg.contains("No available nodes matching criteria") {
                StatusCode::SERVICE_UNAVAILABLE
            } else {
                StatusCode::INTERNAL_SERVER_ERROR
            }
        })?;

    Ok(Json(rental_response))
}

/// Get rental status
pub async fn get_rental_status(
    State(state): State<ApiState>,
    Path(rental_id): Path<String>,
) -> Result<Json<RentalStatusResponse>, StatusCode> {
    info!("Getting status for rental {}", rental_id);

    let rental_manager = state
        .rental_manager
        .as_ref()
        .ok_or(StatusCode::INTERNAL_SERVER_ERROR)?;

    // Get rental info first to get node details
    let rental_info = state
        .persistence
        .load_rental(&rental_id)
        .await
        .map_err(|e| {
            error!("Failed to load rental info: {}", e);
            StatusCode::INTERNAL_SERVER_ERROR
        })?
        .ok_or_else(|| {
            error!("Rental {} not found", rental_id);
            StatusCode::NOT_FOUND
        })?;

    let status = rental_manager
        .get_rental_status(&rental_id)
        .await
        .map_err(|e| {
            error!("Failed to get rental status: {}", e);
            StatusCode::NOT_FOUND
        })?;

    // Convert RentalStatus to RentalStatusResponse
    use crate::api::types::{RentalStatus as ApiRentalStatus, RentalStatusResponse};

    let mut node = rental_info.node_details.clone();
    if node.hourly_rate_cents.is_none() {
        node.hourly_rate_cents = state
            .persistence
            .get_node_hourly_rate(&rental_info.node_id)
            .await
            .map_err(|e| {
                tracing::error!(
                    "Failed to get node hourly rate for node {}: {}",
                    rental_info.node_id,
                    e
                );
                StatusCode::INTERNAL_SERVER_ERROR
            })?
            .map(|cents| cents as i32);
    }
    if node.hourly_rate_cents.is_none() {
        tracing::error!(
            "Node hourly_rate_cents missing for rental {} on node {}",
            rental_id,
            rental_info.node_id
        );
        return Err(StatusCode::INTERNAL_SERVER_ERROR);
    }

    // Extract miner_uid from miner_id (format: "miner_{uid}")
    let miner_uid = rental_info
        .miner_id
        .strip_prefix("miner_")
        .and_then(|uid_str| uid_str.parse::<u16>().ok())
        .ok_or_else(|| {
            tracing::error!(
                "Invalid miner_id format for node {}: expected 'miner_<uid>', got '{}'",
                rental_info.node_id,
                rental_info.miner_id
            );
            StatusCode::INTERNAL_SERVER_ERROR
        })?;

    // Get miner_hotkey from database
    let miner_hotkey = state
        .persistence
        .get_miner_hotkey_by_id(&rental_info.miner_id)
        .await
        .map_err(|e| {
            tracing::error!(
                "Failed to get miner hotkey for miner {}: {}",
                rental_info.miner_id,
                e
            );
            StatusCode::INTERNAL_SERVER_ERROR
        })?
        .ok_or_else(|| {
            tracing::error!("Miner hotkey not found for miner {}", rental_info.miner_id);
            StatusCode::INTERNAL_SERVER_ERROR
        })?;

    let response = RentalStatusResponse {
        rental_id: status.rental_id,
        status: match status.state {
            RentalState::Provisioning => ApiRentalStatus::Pending,
            RentalState::Active => ApiRentalStatus::Active,
            RentalState::Restarting => ApiRentalStatus::Active, // Treat restarting as active
            RentalState::Stopping | RentalState::Stopped => ApiRentalStatus::Terminated,
            RentalState::Failed => ApiRentalStatus::Failed,
        },
        node,
        miner_uid,
        miner_hotkey,
        created_at: status.created_at,
        updated_at: status.created_at, // Use created_at for now
    };

    Ok(Json(response))
}

/// Stop a rental
pub async fn stop_rental(
    State(state): State<ApiState>,
    Path(rental_id): Path<String>,
) -> Result<axum::response::Response, StatusCode> {
    info!("Stopping rental {}", rental_id);

    let rental_manager = state
        .rental_manager
        .as_ref()
        .ok_or(StatusCode::INTERNAL_SERVER_ERROR)?;

    rental_manager
        .stop_rental(&rental_id, false)
        .await
        .map_err(|e| {
            error!("Failed to stop rental: {}", e);
            StatusCode::INTERNAL_SERVER_ERROR
        })?;

    Ok(StatusCode::NO_CONTENT.into_response())
}

/// Stream rental logs
pub async fn stream_rental_logs(
    State(state): State<ApiState>,
    Path(rental_id): Path<String>,
    Query(query): Query<LogStreamQuery>,
) -> Result<Sse<impl Stream<Item = Result<Event, std::io::Error>>>, StatusCode> {
    info!("Streaming logs for rental {}", rental_id);

    let rental_manager = state
        .rental_manager
        .as_ref()
        .ok_or(StatusCode::INTERNAL_SERVER_ERROR)?;

    let follow = query.follow.unwrap_or(false);
    let tail_lines = query.tail;

    let mut log_receiver = rental_manager
        .stream_logs(&rental_id, follow, tail_lines)
        .await
        .map_err(|e| {
            error!("Failed to stream logs: {}", e);
            StatusCode::NOT_FOUND
        })?;

    // Convert log stream to SSE events
    let stream = async_stream::stream! {
        while let Some(log_entry) = log_receiver.recv().await {
            let data = serde_json::json!({
                "timestamp": log_entry.timestamp,
                "stream": log_entry.stream,
                "message": log_entry.message,
            });

            yield Ok(Event::default().data(data.to_string()));
        }
    };

    Ok(Sse::new(stream))
}

/// List rentals for the validator
pub async fn list_rentals(
    State(state): State<ApiState>,
    Query(query): Query<ListRentalsQuery>,
) -> Result<Json<ListRentalsResponse>, StatusCode> {
    info!("Listing rentals with filter: {:?}", query.state);

    let validator_hotkey = state.validator_hotkey.to_string();

    // Get all rentals for this validator via rental manager
    let rental_manager = state
        .rental_manager
        .as_ref()
        .ok_or(StatusCode::INTERNAL_SERVER_ERROR)?;
    let rentals = rental_manager
        .list_rentals(&validator_hotkey)
        .await
        .map_err(|e| {
            error!("Failed to list rentals: {}", e);
            StatusCode::INTERNAL_SERVER_ERROR
        })?;

    // Filter by state if specified
    let filtered_rentals: Vec<RentalInfo> = if let Some(state_filter) = query.state {
        rentals
            .into_iter()
            .filter(|r| r.state == state_filter)
            .collect()
    } else {
        rentals // No filter shows all rentals
    };

    // Convert to API response format
    let rental_list: Vec<RentalListItem> = filtered_rentals
        .iter()
        .map(|r| RentalListItem {
            rental_id: r.rental_id.clone(),
            node_id: r.node_id.clone(),
            container_id: r.container_id.clone(),
            state: r.state.clone(),
            created_at: r.created_at.to_rfc3339(),
            miner_id: r.miner_id.clone(),
            container_image: r.container_spec.image.clone(),
            gpu_specs: if r.node_details.gpu_specs.is_empty() {
                None
            } else {
                Some(r.node_details.gpu_specs.clone())
            },
            cpu_specs: Some(r.node_details.cpu_specs.clone()),
            location: r.node_details.location.clone(),
            network_speed: r.node_details.network_speed.clone(),
        })
        .collect();

    let total_count = filtered_rentals.len();

    Ok(Json(ListRentalsResponse {
        rentals: rental_list,
        total_count,
    }))
}
