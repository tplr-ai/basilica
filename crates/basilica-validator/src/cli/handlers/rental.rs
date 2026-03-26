//! Rental command handlers using the Validator HTTP API
//!
//! Handles CLI commands for container rental operations via the Validator API

use anyhow::{Context, Result};
use basilica_common::types::GpuCategory;
use std::time::Duration;
use tracing::info;

#[cfg(feature = "client")]
use crate::api::client::ValidatorClient;
use crate::api::routes::rentals::{
    PortMappingRequest, ResourceRequirementsRequest, StartRentalRequest,
};
use crate::api::types::{ListAvailableNodesQuery, LogQuery, TerminateRentalRequest};
use crate::cli::commands::RentalAction;
use crate::config::ValidatorConfig;
use crate::rental::types::RentalState;
use basilica_common::utils::{parse_env_vars, parse_port_mappings};

/// Create a ValidatorClient from configuration
#[cfg(feature = "client")]
fn create_api_client(config: &ValidatorConfig, api_url: Option<String>) -> Result<ValidatorClient> {
    // Use provided API URL or construct from bind address
    let base_url = api_url.unwrap_or_else(|| {
        // Parse the bind address to check for wildcards
        let bind_str = &config.api.bind_address;
        if let Some(colon_pos) = bind_str.rfind(':') {
            let host = &bind_str[..colon_pos];
            let port = &bind_str[colon_pos + 1..];

            // Replace wildcard addresses with loopback
            if host == "0.0.0.0" || host == "*" {
                format!("http://127.0.0.1:{}", port)
            } else {
                format!("http://{}", config.api.bind_address)
            }
        } else {
            format!("http://{}", config.api.bind_address)
        }
    });

    // Create client with 30 second timeout
    ValidatorClient::new(base_url, Duration::from_secs(30)).context("Failed to create API client")
}

/// Handle rental commands via the Validator API
#[cfg(feature = "client")]
pub async fn handle_rental_command(
    action: RentalAction,
    config: &ValidatorConfig,
    api_url: Option<String>,
) -> Result<()> {
    // Create API client
    let client = create_api_client(config, api_url).context(
        "Failed to create API client. Ensure the validator service is running with API enabled.",
    )?;

    match action {
        RentalAction::Start {
            gpu_category,
            gpu_count,
            min_memory_gb,
            max_hourly_rate_cents,
            image,
            ports,
            env,
            ssh_public_key,
            command,
            cpu_cores,
            memory_mb,
            storage_mb,
        } => {
            handle_start_rental(
                client,
                gpu_category,
                gpu_count,
                min_memory_gb,
                max_hourly_rate_cents,
                image,
                ports,
                env,
                ssh_public_key,
                command,
                cpu_cores,
                memory_mb,
                storage_mb,
            )
            .await
        }
        RentalAction::Status { id } => handle_rental_status(client, id).await,
        RentalAction::Logs { id, follow, tail } => {
            handle_rental_logs(client, id, follow, tail).await
        }
        RentalAction::Stop { id, .. } => handle_stop_rental(client, id).await,
        RentalAction::Ls {
            memory_min,
            gpu_type,
            gpu_min,
        } => handle_ls_nodes(client, memory_min, gpu_type, gpu_min).await,
        RentalAction::Ps { state } => handle_ps_rentals(client, state).await,
    }
}

#[cfg(feature = "client")]
#[allow(clippy::too_many_arguments)]
async fn handle_start_rental(
    client: ValidatorClient,
    gpu_category: String,
    gpu_count: u32,
    min_memory_gb: Option<u32>,
    max_hourly_rate_cents: u32,
    image: String,
    ports: Vec<String>,
    env: Vec<String>,
    ssh_public_key: String,
    command: Vec<String>,
    cpu_cores: Option<f64>,
    memory_mb: Option<i64>,
    storage_mb: Option<i64>,
) -> Result<()> {
    // Validate gpu_category is a known GPU type
    let gpu_cat: GpuCategory = gpu_category.parse().unwrap(); // Infallible
    if matches!(&gpu_cat, GpuCategory::Other(_)) {
        anyhow::bail!("GPU type '{}' is not supported", gpu_category);
    }

    info!(
        "Starting rental for {} x {} GPU(s)",
        gpu_count, gpu_category
    );

    // Parse port mappings and environment variables
    let port_mappings: Vec<PortMappingRequest> = parse_port_mappings(&ports)?
        .into_iter()
        .map(Into::into)
        .collect();
    let environment = parse_env_vars(&env)?;

    // Build API request
    let request = StartRentalRequest {
        gpu_category,
        gpu_count,
        min_memory_gb,
        max_hourly_rate_cents,
        container_image: image,
        ssh_public_key,
        environment,
        ports: port_mappings,
        resources: ResourceRequirementsRequest {
            cpu_cores: cpu_cores.unwrap_or(0.0),
            memory_mb: memory_mb.unwrap_or(0),
            storage_mb: storage_mb.unwrap_or(0),
            gpu_count,
            gpu_types: Vec::new(),
        },
        command,
        volumes: Vec::new(),
    };

    // Call API to start rental
    let response = client
        .start_rental(request)
        .await
        .context("Failed to start rental via API")?;

    info!("Rental started successfully!");
    info!("Rental ID: {}", response.rental_id);
    if let Some(ref ssh_creds) = response.ssh_credentials {
        info!("SSH Access: {}", ssh_creds);
    } else {
        info!("SSH Access: Not available (port 22 not mapped)");
    }
    info!(
        "Container: {} ({})",
        response.container_info.container_name, response.container_info.container_id
    );

    Ok(())
}

#[cfg(feature = "client")]
async fn handle_rental_status(client: ValidatorClient, rental_id: String) -> Result<()> {
    info!("Getting status for rental {}", rental_id);

    // Get rental status via API
    let status = client
        .get_rental_status(&rental_id)
        .await
        .context("Failed to get rental status via API")?;

    info!("Rental Status:");
    info!("  ID: {}", status.rental_id);
    info!("  Status: {:?}", status.status);
    info!("  Node: {}", status.node.id);
    if let Some(location) = &status.node.location {
        info!("  Location: {}", location);
    }
    info!("  Created: {}", status.created_at);
    info!("  Updated: {}", status.updated_at);

    // Display GPU specs if available
    if !status.node.gpu_specs.is_empty() {
        info!("GPU Specs:");
        for gpu in &status.node.gpu_specs {
            info!("  - {}", gpu.name);
        }
    }

    // Display CPU specs
    info!("CPU Specs:");
    info!("  Model: {}", status.node.cpu_specs.model);
    info!("  Cores: {}", status.node.cpu_specs.cores);
    info!("  Memory: {} GB", status.node.cpu_specs.memory_gb);

    Ok(())
}

#[cfg(feature = "client")]
async fn handle_rental_logs(
    client: ValidatorClient,
    rental_id: String,
    follow: bool,
    tail: Option<u32>,
) -> Result<()> {
    info!("Streaming logs for rental {}", rental_id);

    // Create log query
    let query = LogQuery {
        follow: Some(follow),
        tail,
    };

    // Stream logs via API
    let mut log_stream = client
        .stream_rental_logs(&rental_id, query)
        .await
        .context("Failed to stream logs via API")?;

    // Print logs as they arrive
    use futures::StreamExt;
    while let Some(log_result) = log_stream.next().await {
        match log_result {
            Ok(log_entry) => {
                println!(
                    "[{}] [{}] {}",
                    log_entry.timestamp, log_entry.stream, log_entry.message
                );
            }
            Err(e) => {
                eprintln!("Error receiving log: {}", e);
                break;
            }
        }
    }

    Ok(())
}

#[cfg(feature = "client")]
async fn handle_stop_rental(client: ValidatorClient, rental_id: String) -> Result<()> {
    info!("Stopping rental {}", rental_id);

    // Stop rental via API
    let request = TerminateRentalRequest {
        reason: Some("User requested stop via CLI".to_string()),
    };

    client
        .terminate_rental(&rental_id, request)
        .await
        .context("Failed to stop rental via API")?;

    info!("Rental {} stopped successfully", rental_id);

    Ok(())
}

#[cfg(feature = "client")]
async fn handle_ls_nodes(
    client: ValidatorClient,
    memory_min: Option<u32>,
    gpu_type: Option<String>,
    gpu_min: Option<u32>,
) -> Result<()> {
    info!("Listing available nodes");

    // Build query from filters
    let query = ListAvailableNodesQuery {
        available: Some(true), // Filter for available nodes only
        min_gpu_memory: memory_min,
        gpu_type,
        min_gpu_count: gpu_min,
        location: None,
    };

    // List available nodes via API
    let response = client
        .list_available_nodes(Some(query))
        .await
        .context("Failed to list available nodes via API")?;

    if response.available_nodes.is_empty() {
        info!("No available nodes found matching the specified criteria.");
        return Ok(());
    }

    info!("Found {} available nodes:", response.total_count);
    info!("");

    // Format output similar to basilica-cli
    info!("GPU                                   | Node ID                              | CPU        | RAM");
    info!("--------------------------------------+--------------------------------------+------------+--------");

    for node in response.available_nodes {
        // Format GPU info
        let gpu_info = if node.node.gpu_specs.is_empty() {
            "No GPU".to_string()
        } else if node.node.gpu_specs.len() == 1 {
            // Single GPU
            let gpu = &node.node.gpu_specs[0];
            gpu.name.clone()
        } else {
            // Multiple GPUs - check if they're all the same model
            let first_gpu = &node.node.gpu_specs[0];
            let all_same = node
                .node
                .gpu_specs
                .iter()
                .all(|g| g.name == first_gpu.name && g.memory_gb == first_gpu.memory_gb);

            if all_same {
                // All GPUs are identical - use count prefix format
                format!("{}x {}", node.node.gpu_specs.len(), first_gpu.name)
            } else {
                // Different GPU models - list them individually
                let gpu_names: Vec<String> =
                    node.node.gpu_specs.iter().map(|g| g.name.clone()).collect();
                gpu_names.join(", ")
            }
        };

        let node_id = node.node.id;

        info!(
            "{:<36} | {:<36} | {:<10} | {:<6}",
            // Truncate GPU info if too long
            if gpu_info.len() > 36 {
                format!("{}...", &gpu_info[..33])
            } else {
                gpu_info
            },
            node_id,
            format!("{} cores", node.node.cpu_specs.cores),
            format!("{}GB", node.node.cpu_specs.memory_gb),
        );
    }

    info!("");
    info!("Total available nodes: {}", response.total_count);

    Ok(())
}

#[cfg(feature = "client")]
async fn handle_ps_rentals(client: ValidatorClient, state_filter: String) -> Result<()> {
    info!("Listing rentals with filter: {}", state_filter);

    // Parse state filter
    let filter = match state_filter.as_str() {
        "active" => Some(RentalState::Active),
        "stopped" => Some(RentalState::Stopped),
        _ => None, // "all" or any other value shows all rentals
    };

    // List rentals via API
    let response = client
        .list_rentals(filter)
        .await
        .context("Failed to list rentals via API")?;

    if response.rentals.is_empty() {
        info!("No rentals found");
        return Ok(());
    }

    info!("Found {} rentals:", response.rentals.len());
    info!("");
    info!("ID                                    | State   | Node                             | Created");
    info!("--------------------------------------+---------+--------------------------------------+-------------------------");

    for rental in response.rentals {
        // Note: container_id is not available in RentalListItem, so we omit it
        info!(
            "{:<36} | {:<7} | {:<36} | {}",
            rental.rental_id,
            format!("{:?}", rental.state),
            rental.node_id,
            rental.created_at
        );
    }

    Ok(())
}

/// Fallback when client feature is not enabled
#[cfg(not(feature = "client"))]
pub async fn handle_rental_command(
    _action: RentalAction,
    _config: &ValidatorConfig,
    _api_url: Option<String>,
) -> Result<()> {
    Err(anyhow::anyhow!("Rental CLI commands require the validator service to be running with API enabled. Please ensure the validator is running before using rental commands."))
}
