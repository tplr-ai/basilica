//! GPU rental command handlers

use crate::cli::commands::{ComputeCategoryArg, ListFilters, LogsOptions, PsFilters, UpOptions};
use crate::cli::handlers::deploy::helpers::stream_logs_to_stdout;
use crate::cli::handlers::gpu_rental_helpers::{
    active_rentals_query, get_ssh_private_key_path, print_cloud_section_header,
    resolve_offering_unified, resolve_rental_by_id, resolve_rental_with_ssh,
    resolve_target_rental_unified, with_validator_timeout, CommunityCloudSelection, RentalWithSsh,
    SelectedOffering,
};
use crate::cli::handlers::region_mapping::region_matches_country;
use crate::cli::handlers::ssh_keys::select_and_read_ssh_key;
use crate::client::create_authenticated_client;
use crate::config::CliConfig;
use crate::output::{
    compress_path, format_usd, json_output, print_error, print_info, print_success, table_output,
};
use crate::progress::{complete_spinner_and_clear, complete_spinner_error, create_spinner};
use crate::ssh::{find_private_key_for_public_key, parse_ssh_credentials, SshClient};
use crate::CliError;
use basilica_common::types::{ComputeCategory, GpuCategory};
use basilica_common::utils::{parse_env_vars, parse_port_mappings};
use basilica_sdk::types::{
    HistoricalRentalItem, HistoricalRentalsResponse, ListAvailableNodesQuery, ListRentalsQuery,
    LocationProfile, RentalState, ResourceRequirementsRequest, SshAccess, StartRentalApiRequest,
};
use basilica_sdk::ApiError;
use color_eyre::eyre::eyre;
use color_eyre::Section;
use console::style;
use reqwest::StatusCode;
use std::collections::HashMap;
use std::path::PathBuf;
use std::str::FromStr;
use std::time::Duration;
use tracing::{debug, warn};

/// Maximum time to wait for rental to become active and SSH to be ready
const RENTAL_READY_TIMEOUT: Duration = Duration::from_secs(300);

fn usd_per_gpu_hour_to_cents(value: f64) -> Result<u32, CliError> {
    if !value.is_finite() {
        return Err(CliError::Internal(eyre!(
            "Invalid --max-hourly-rate: value must be a finite number"
        )));
    }
    if value < 0.0 {
        return Err(CliError::Internal(eyre!(
            "Invalid --max-hourly-rate: value must be non-negative"
        )));
    }

    let cents = (value * 100.0).round();
    if cents < 0.0 || cents > u32::MAX as f64 {
        return Err(CliError::Internal(eyre!(
            "Invalid --max-hourly-rate: value is out of supported range"
        )));
    }

    Ok(cents as u32)
}

#[cfg(test)]
mod conversion_tests {
    use super::usd_per_gpu_hour_to_cents;

    #[test]
    fn rounds_to_nearest_cent() {
        assert_eq!(usd_per_gpu_hour_to_cents(2.50).unwrap(), 250);
        assert_eq!(usd_per_gpu_hour_to_cents(1.234).unwrap(), 123);
        assert_eq!(usd_per_gpu_hour_to_cents(1.235).unwrap(), 124);
    }

    #[test]
    fn rejects_invalid_values() {
        assert!(usd_per_gpu_hour_to_cents(f64::NAN).is_err());
        assert!(usd_per_gpu_hour_to_cents(f64::INFINITY).is_err());
        assert!(usd_per_gpu_hour_to_cents(-0.01).is_err());
    }
}

/// Enum representing the type of rental offering (GPU or CPU-only)
enum RentalOffering {
    SecureCloud(basilica_common::types::GpuOffering),
    CpuOnly(basilica_sdk::types::CpuOffering),
}

impl RentalOffering {
    fn id(&self) -> &str {
        match self {
            RentalOffering::SecureCloud(o) => &o.id,
            RentalOffering::CpuOnly(o) => &o.id,
        }
    }

    fn rental_type_name(&self) -> &'static str {
        match self {
            RentalOffering::SecureCloud(_) => "rental",
            RentalOffering::CpuOnly(_) => "CPU-only rental",
        }
    }

    fn rental_noun(&self) -> &'static str {
        match self {
            RentalOffering::SecureCloud(_) => "rental",
            RentalOffering::CpuOnly(_) => "CPU rental",
        }
    }

    fn instance_noun(&self) -> &'static str {
        match self {
            RentalOffering::SecureCloud(_) => "Instance",
            RentalOffering::CpuOnly(_) => "CPU instance",
        }
    }
}

/// Represents a GPU target for the `up` command
#[derive(Debug, Clone)]
pub struct GpuTarget(pub GpuCategory);

impl FromStr for GpuTarget {
    type Err = std::convert::Infallible;

    fn from_str(s: &str) -> Result<Self, Self::Err> {
        let gpu_category =
            GpuCategory::from_str(s).expect("GpuCategory::from_str returns Infallible");
        Ok(GpuTarget(gpu_category))
    }
}

/// Helper function to fetch and filter secure cloud GPUs
async fn fetch_and_filter_secure_cloud(
    api_client: &basilica_sdk::BasilicaClient,
    gpu_category: Option<GpuCategory>,
    filters: &ListFilters,
) -> Result<Vec<basilica_common::types::GpuOffering>, CliError> {
    let query = basilica_sdk::types::GpuPriceQuery {
        interconnect: filters.interconnect.clone(),
        region: filters.region.clone(),
        spot_only: if filters.spot { Some(true) } else { None },
        exclude_spot: if filters.exclude_spot {
            Some(true)
        } else {
            None
        },
    };

    let gpus = api_client
        .list_secure_cloud_gpus_filtered(&query)
        .await
        .map_err(|e| -> CliError {
            CliError::Internal(
                eyre!(e)
                    .suggestion("Check your internet connection and try again")
                    .note("If this persists, GPUs may be temporarily unavailable"),
            )
        })?;

    // Apply filters
    let mut filtered_gpus: Vec<_> = gpus
        .into_iter()
        .filter(|gpu| {
            // Filter by GPU type if specified
            if let Some(ref category) = gpu_category {
                let category_str = category.as_str().to_uppercase();
                if !gpu.gpu_type.as_str().to_uppercase().contains(&category_str) {
                    return false;
                }
            }

            // Filter by availability
            if !gpu.availability {
                return false;
            }

            // Filter by GPU count
            if let Some(min_count) = filters.gpu_min {
                if gpu.gpu_count < min_count {
                    return false;
                }
            }

            // Filter by minimum GPU memory
            if let Some(min_memory) = filters.memory_min {
                if let Some(mem_per_gpu) = gpu.gpu_memory_gb_per_gpu {
                    let total_memory = mem_per_gpu * gpu.gpu_count;
                    if total_memory < min_memory {
                        return false;
                    }
                }
            }

            // Filter by country using region mapping
            if let Some(ref country) = filters.country {
                if !region_matches_country(&gpu.region, country) {
                    return false;
                }
            }

            // Filter by max price (total hourly cost for all GPUs)
            if let Some(max_price) = filters.price_max {
                use rust_decimal::prelude::ToPrimitive;
                let total_price =
                    gpu.hourly_rate_per_gpu.to_f64().unwrap_or(f64::MAX) * (gpu.gpu_count as f64);
                if total_price > max_price {
                    return false;
                }
            }

            true
        })
        .collect();

    // Sort by total price (ascending) - per-GPU rate × gpu_count
    filtered_gpus.sort_by(|a, b| {
        let a_total = a.hourly_rate_per_gpu * rust_decimal::Decimal::from(a.gpu_count);
        let b_total = b.hourly_rate_per_gpu * rust_decimal::Decimal::from(b.gpu_count);
        a_total.partial_cmp(&b_total).unwrap()
    });

    Ok(filtered_gpus)
}

/// Helper function to fetch and filter community cloud nodes
async fn fetch_and_filter_community_cloud(
    api_client: &basilica_sdk::BasilicaClient,
    gpu_category: Option<GpuCategory>,
    filters: &ListFilters,
) -> Result<(Vec<basilica_sdk::AvailableNode>, HashMap<String, String>), CliError> {
    // Convert GPU category to string if provided
    let gpu_type = gpu_category.map(|gc| gc.as_str());

    // Build query from filters
    let query = ListAvailableNodesQuery {
        available: Some(true), // Filter for available nodes only
        min_gpu_memory: filters.memory_min,
        gpu_type,
        min_gpu_count: Some(filters.gpu_min.unwrap_or(0)),
        location: filters.country.as_ref().map(|country| LocationProfile {
            city: None,
            region: None,
            country: Some(country.clone()),
        }),
    };

    // Fetch available nodes
    let response = api_client
        .list_available_nodes(Some(query))
        .await
        .map_err(|e| -> CliError {
            CliError::Internal(
                eyre!(e)
                    .suggestion("Check your internet connection and try again")
                    .note("If this persists, nodes may be temporarily unavailable"),
            )
        })?;

    // Apply client-side max price filter if specified
    let filtered_nodes: Vec<_> = if let Some(max_price) = filters.price_max {
        response
            .available_nodes
            .into_iter()
            .filter(|node| {
                if let Some(rate_cents) = node.node.hourly_rate_cents {
                    let gpu_count = node.node.gpu_specs.len() as f64;
                    let rate_dollars = rate_cents as f64 / 100.0;
                    let total_rate = rate_dollars * gpu_count;
                    total_rate <= max_price
                } else {
                    // Include nodes without pricing (will show as "Market")
                    true
                }
            })
            .collect()
    } else {
        response.available_nodes
    };

    // Build pricing map from nodes' hourly_rate_cents field
    // Map GPU type -> hourly rate string (e.g., "h100" -> "2.50")
    let pricing_map: HashMap<String, String> = filtered_nodes
        .iter()
        .filter_map(|node| {
            // Get the first GPU spec to determine GPU type
            let gpu_spec = node.node.gpu_specs.first()?;
            let category = GpuCategory::from_str(&gpu_spec.name).ok()?;
            let gpu_type = category.to_string().to_lowercase();

            // Get pricing from the node's hourly_rate_cents field
            let cents = node.node.hourly_rate_cents? as f64;
            let dollars = cents / 100.0;
            let rate_string = format!("{:.2}", dollars);

            Some((gpu_type, rate_string))
        })
        .collect();

    Ok((filtered_nodes, pricing_map))
}

/// Helper function to display secure cloud GPUs
fn display_secure_cloud_table(
    gpus: &[basilica_common::types::GpuOffering],
) -> Result<(), CliError> {
    if gpus.is_empty() {
        print_info("No GPUs available matching your criteria");
        return Ok(());
    }

    table_output::display_secure_cloud_offerings_detailed(gpus)?;

    Ok(())
}

/// Helper function to display community cloud nodes (aggregated by GPU category)
fn display_community_cloud_table(nodes: &[basilica_sdk::AvailableNode]) -> Result<(), CliError> {
    if nodes.is_empty() {
        print_info("No GPUs available matching your criteria");
        return Ok(());
    }

    use crate::cli::handlers::gpu_rental_helpers::aggregate_nodes_by_gpu_category;
    let aggregations = aggregate_nodes_by_gpu_category(nodes);
    table_output::display_community_cloud_categories(&aggregations)?;

    Ok(())
}

/// Filter secure cloud rentals based on PsFilters
///
/// Applies gpu_type and min_gpu_count filters to secure cloud rentals.
fn filter_secure_cloud_rentals<'a>(
    rentals: &'a [basilica_sdk::types::SecureCloudRentalListItem],
    filters: &PsFilters,
) -> Vec<&'a basilica_sdk::types::SecureCloudRentalListItem> {
    rentals
        .iter()
        .filter(|r| {
            // Skip stopped rentals unless showing history
            if !filters.history && r.stopped_at.is_some() {
                return false;
            }

            // Filter by GPU type if specified
            if let Some(ref gpu_type) = filters.gpu_type {
                if !r.gpu_type.to_uppercase().contains(&gpu_type.to_uppercase()) {
                    return false;
                }
            }

            // Filter by min GPU count if specified
            if let Some(min_count) = filters.min_gpu_count {
                if r.gpu_count < min_count {
                    return false;
                }
            }

            true
        })
        .collect()
}

fn is_secure_cpu_history_item(rental: &HistoricalRentalItem) -> bool {
    rental.compute_type.eq_ignore_ascii_case("cpu")
}

/// Filter CPU offerings based on ListFilters
///
/// Applies country, memory, and price filters to CPU offerings.
fn filter_cpu_offerings(
    offerings: Vec<basilica_sdk::types::CpuOffering>,
    filters: &ListFilters,
) -> Vec<basilica_sdk::types::CpuOffering> {
    offerings
        .into_iter()
        .filter(|offering| {
            // Filter by availability
            if !offering.availability {
                return false;
            }

            // Filter by country using region mapping
            if let Some(ref country) = filters.country {
                if !region_matches_country(&offering.region, country) {
                    return false;
                }
            }

            // Filter by memory (system memory for CPU offerings)
            if let Some(min_memory) = filters.memory_min {
                if offering.system_memory_gb < min_memory {
                    return false;
                }
            }

            // Filter by max price
            if let Some(max_price) = filters.price_max {
                let hourly_rate = offering.hourly_rate.parse::<f64>().unwrap_or(f64::MAX);
                if hourly_rate > max_price {
                    return false;
                }
            }

            true
        })
        .collect()
}

/// Handle the `ls` command - list available nodes for rental
pub async fn handle_ls(
    gpu_category: Option<GpuCategory>,
    filters: ListFilters,
    compute: Option<ComputeCategoryArg>,
    json: bool,
    config: &CliConfig,
) -> Result<(), CliError> {
    let api_client = create_authenticated_client(config).await?;

    // Convert compute arg to compute category
    let compute_category = compute.map(|c| match c {
        ComputeCategoryArg::SecureCloud => ComputeCategory::SecureCloud,
        ComputeCategoryArg::CommunityCloud => ComputeCategory::CommunityCloud,
    });

    // Branch based on compute type
    match compute_category {
        Some(ComputeCategory::SecureCloud) => {
            // Only fetch CPU offerings if no GPU type filter is specified
            let show_cpu = gpu_category.is_none();

            let spinner = create_spinner("Fetching available instances...");
            let (gpu_result, cpu_result) = if show_cpu {
                let (gpu, cpu) = tokio::join!(
                    fetch_and_filter_secure_cloud(&api_client, gpu_category, &filters),
                    api_client.list_cpu_offerings()
                );
                (gpu, Some(cpu))
            } else {
                let gpu = fetch_and_filter_secure_cloud(&api_client, gpu_category, &filters).await;
                (gpu, None)
            };
            complete_spinner_and_clear(spinner);

            let filtered_gpus = gpu_result?;
            let filtered_cpu = if let Some(cpu_res) = cpu_result {
                let cpu_offerings = cpu_res.map_err(|e| CliError::Internal(eyre!(e)))?;
                filter_cpu_offerings(cpu_offerings, &filters)
            } else {
                vec![]
            };

            if json {
                #[derive(serde::Serialize)]
                struct CombinedSecureCloudOfferings<'a> {
                    gpu_offerings: &'a [basilica_common::types::GpuOffering],
                    #[serde(skip_serializing_if = "<[_]>::is_empty")]
                    cpu_offerings: &'a [basilica_sdk::types::CpuOffering],
                }
                let response = CombinedSecureCloudOfferings {
                    gpu_offerings: &filtered_gpus,
                    cpu_offerings: &filtered_cpu,
                };
                json_output(&response)?;
            } else {
                // Display GPU offerings section
                println!("{}", style("GPU Offerings").bold().cyan());
                display_secure_cloud_table(&filtered_gpus)?;

                // Only display CPU offerings section if no GPU type filter
                if show_cpu {
                    println!();
                    println!("{}", style("The Citadel (CPU)").bold().cyan());
                    table_output::display_cpu_offerings_detailed(&filtered_cpu)?;
                }
            }
        }
        Some(ComputeCategory::CommunityCloud) => {
            // Fetch and filter community cloud nodes
            let spinner = create_spinner("Fetching available GPUs...");
            let result =
                fetch_and_filter_community_cloud(&api_client, gpu_category, &filters).await;
            complete_spinner_and_clear(spinner);
            let (nodes, _pricing_map) = result?;

            if json {
                // Create a simple response structure for JSON output
                #[derive(serde::Serialize)]
                struct NodesResponse<'a> {
                    available_nodes: &'a [basilica_sdk::AvailableNode],
                }
                let response = NodesResponse {
                    available_nodes: &nodes,
                };
                json_output(&response)?;
            } else {
                display_community_cloud_table(&nodes)?;
            }
        }
        None => {
            // Display all tables when --compute flag is not specified
            use crate::cli::handlers::gpu_rental_helpers::VALIDATOR_REQUEST_TIMEOUT;

            // Only show CPU offerings if no GPU type filter is specified
            let show_cpu = gpu_category.is_none();

            let spinner = create_spinner("Fetching available instances...");

            // Fetch all in parallel with timeout for community cloud
            // Note: fetch_and_filter_community_cloud returns CliError, not ApiError,
            // so we use inline timeout here instead of with_validator_timeout
            let community_future =
                fetch_and_filter_community_cloud(&api_client, gpu_category.clone(), &filters);

            let (secure_result, community_result, cpu_result) = if show_cpu {
                let (secure, community, cpu) = tokio::join!(
                    fetch_and_filter_secure_cloud(&api_client, gpu_category, &filters),
                    async {
                        match tokio::time::timeout(VALIDATOR_REQUEST_TIMEOUT, community_future)
                            .await
                        {
                            Ok(result) => result,
                            Err(_) => {
                                warn!(
                                    "Validator request timed out after {} seconds",
                                    VALIDATOR_REQUEST_TIMEOUT.as_secs()
                                );
                                Ok((vec![], std::collections::HashMap::new()))
                            }
                        }
                    },
                    api_client.list_cpu_offerings()
                );
                (secure, community, Some(cpu))
            } else {
                let (secure, community) = tokio::join!(
                    fetch_and_filter_secure_cloud(&api_client, gpu_category, &filters),
                    async {
                        match tokio::time::timeout(VALIDATOR_REQUEST_TIMEOUT, community_future)
                            .await
                        {
                            Ok(result) => result,
                            Err(_) => {
                                warn!(
                                    "Validator request timed out after {} seconds",
                                    VALIDATOR_REQUEST_TIMEOUT.as_secs()
                                );
                                Ok((vec![], std::collections::HashMap::new()))
                            }
                        }
                    }
                );
                (secure, community, None)
            };

            complete_spinner_and_clear(spinner);

            let secure_gpus = secure_result?;
            let (community_nodes, _pricing_map) = community_result?;
            let filtered_cpu = if let Some(cpu_res) = cpu_result {
                filter_cpu_offerings(cpu_res.unwrap_or_default(), &filters)
            } else {
                vec![]
            };

            if json {
                #[derive(serde::Serialize)]
                struct CombinedResponse<'a> {
                    secure_cloud: &'a [basilica_common::types::GpuOffering],
                    community_cloud: &'a [basilica_sdk::AvailableNode],
                    #[serde(skip_serializing_if = "<[_]>::is_empty")]
                    cpu_offerings: &'a [basilica_sdk::types::CpuOffering],
                }
                let response = CombinedResponse {
                    secure_cloud: &secure_gpus,
                    community_cloud: &community_nodes,
                    cpu_offerings: &filtered_cpu,
                };
                json_output(&response)?;
            } else {
                print_cloud_section_header("The Bourse (GPU)", true);
                display_community_cloud_table(&community_nodes)?;

                println!();

                print_cloud_section_header("The Citadel (GPU)", false);
                display_secure_cloud_table(&secure_gpus)?;

                // Only display CPU offerings section if no GPU type filter
                if show_cpu {
                    println!();

                    print_cloud_section_header("The Citadel (CPU)", false);
                    table_output::display_cpu_offerings_detailed(&filtered_cpu)?;
                }
            }
        }
    }

    Ok(())
}

/// Ensure user has SSH key registered with Basilica API
///
/// Checks if user already has a key registered. If not, discovers SSH keys
/// in ~/.ssh, lets user select one interactively, and registers it.
/// Returns the SSH key ID if successful.
pub async fn ensure_ssh_key_registered(
    api_client: &basilica_sdk::BasilicaClient,
) -> Result<String, CliError> {
    // Check if user already has SSH key registered
    if let Some(key) = api_client.get_user_ssh_key().await? {
        return Ok(key.id);
    }

    // No key registered - run interactive selection flow
    print_info("No SSH key registered.");

    // Use shared key discovery and selection logic
    let selected_key = select_and_read_ssh_key().await?;

    // Prompt user for confirmation
    use dialoguer::Confirm;

    let filename = selected_key
        .path
        .file_name()
        .and_then(|n| n.to_str())
        .unwrap_or("key");

    let confirmed = Confirm::new()
        .with_prompt(format!("Register {}?", filename))
        .default(true)
        .interact()
        .map_err(|e| CliError::Internal(eyre!(e).wrap_err("Failed to show confirmation prompt")))?;

    if !confirmed {
        return Err(CliError::Internal(eyre!(
            "SSH key required for GPU rentals"
        )));
    }

    // Register key
    let spinner = create_spinner("Registering SSH key...");
    let result = api_client
        .register_ssh_key("default", selected_key.content.trim())
        .await;

    match result {
        Ok(key) => {
            complete_spinner_and_clear(spinner);
            print_success("Successfully registered SSH key");
            Ok(key.id)
        }
        Err(e) => {
            complete_spinner_error(spinner, "Failed to register SSH key");
            Err(CliError::Api(e))
        }
    }
}

/// Unified handler for both secure cloud (GPU) and CPU-only rentals
async fn handle_rental_with_offering(
    api_client: basilica_sdk::BasilicaClient,
    offering: RentalOffering,
    options: UpOptions,
    config: &CliConfig,
) -> Result<(), CliError> {
    let rental_type = offering.rental_type_name();
    let rental_noun = offering.rental_noun();
    let instance_noun = offering.instance_noun();

    // Get SSH key ID (SSH key registration already done in handle_up)
    let ssh_key_id = api_client
        .get_ssh_key()
        .await
        .map_err(|e| CliError::Internal(eyre!(e)))?
        .ok_or_else(|| {
            CliError::Internal(
                eyre!("No SSH key registered with Basilica")
                    .suggestion("Run 'basilica ssh-keys add' to register your SSH key"),
            )
        })?
        .id;

    // Start rental
    let spinner = create_spinner(&format!("Starting {}...", rental_type));

    use basilica_sdk::types::StartSecureCloudRentalRequest;

    let request = StartSecureCloudRentalRequest {
        offering_id: offering.id().to_string(),
        ssh_public_key_id: ssh_key_id,
    };

    // Start the rental using the appropriate API method
    let response = match &offering {
        RentalOffering::SecureCloud(_) => api_client
            .start_secure_cloud_rental(request)
            .await
            .map_err(|e| {
                complete_spinner_error(spinner.clone(), "Failed to start rental");
                CliError::Api(e)
            })?,
        RentalOffering::CpuOnly(_) => api_client.start_cpu_rental(request).await.map_err(|e| {
            complete_spinner_error(spinner.clone(), &format!("Failed to start {}", rental_noun));
            CliError::Api(e)
        })?,
    };
    complete_spinner_and_clear(spinner);

    print_success(&format!(
        "Successfully started {} {}",
        rental_type, response.rental_id
    ));

    if options.detach {
        if let Some(ssh_cmd) = &response.ssh_command {
            let private_key_path = get_ssh_private_key_path(&api_client)
                .await
                .map_err(CliError::Internal)?;
            display_secure_cloud_reconnection_instructions(
                &response.rental_id,
                ssh_cmd,
                &private_key_path,
                &format!("To connect to this {}:", rental_noun),
            )?;
        } else {
            println!();
            print_info(&format!(
                "{} is starting up. Use 'basilica ps' to check status.",
                instance_noun
            ));
            print_info(&format!("SSH with: basilica ssh {}", response.rental_id));
        }
        return Ok(());
    }

    // Wait for rental to become active
    print_info(&format!("Waiting for {} to become active...", rental_noun));
    let rental = match &offering {
        RentalOffering::SecureCloud(_) => {
            poll_secure_cloud_rental_status(&response.rental_id, &api_client).await?
        }
        RentalOffering::CpuOnly(_) => {
            poll_cpu_rental_status(&response.rental_id, &api_client).await?
        }
    };

    if let Some(rental) = rental {
        if let Some(ssh_cmd) = &rental.ssh_command {
            print_info(&format!("Connecting to {}...", rental_noun));
            let (host, port, username) = parse_ssh_credentials(ssh_cmd)?;
            let ssh_access = SshAccess {
                host,
                port,
                username,
            };

            let private_key_path = get_ssh_private_key_path(&api_client)
                .await
                .map_err(CliError::Internal)?;

            let ssh_client = SshClient::new(&config.ssh)?;
            match retry_ssh_connection(
                &ssh_client,
                &ssh_access,
                private_key_path.clone(),
                RENTAL_READY_TIMEOUT,
            )
            .await
            {
                Ok(_) => {
                    print_info("SSH session closed");
                    display_secure_cloud_reconnection_instructions(
                        &response.rental_id,
                        ssh_cmd,
                        &private_key_path,
                        &format!("To reconnect to this {}:", rental_noun),
                    )?;
                }
                Err(e) => {
                    print_error(&format!("SSH connection failed: {}", e));
                    display_secure_cloud_reconnection_instructions(
                        &response.rental_id,
                        ssh_cmd,
                        &private_key_path,
                        "Try manually connecting using:",
                    )?;
                }
            }
        } else {
            println!();
            print_info(&format!(
                "{} is active but SSH is not yet available",
                rental_noun
                    .chars()
                    .next()
                    .unwrap()
                    .to_uppercase()
                    .collect::<String>()
                    + &rental_noun[1..]
            ));
            print_info(&format!("SSH with: basilica ssh {}", response.rental_id));
        }
    } else {
        println!();
        print_info(&format!(
            "{} is taking longer than expected to become active",
            rental_noun
                .chars()
                .next()
                .unwrap()
                .to_uppercase()
                .collect::<String>()
                + &rental_noun[1..]
        ));
        print_info("Check status with: basilica ps");
        print_info(&format!("SSH with: basilica ssh {}", response.rental_id));
    }

    Ok(())
}

/// Poll CPU rental status until it becomes active or times out
async fn poll_cpu_rental_status(
    rental_id: &str,
    api_client: &basilica_sdk::BasilicaClient,
) -> Result<Option<basilica_sdk::types::SecureCloudRentalListItem>, CliError> {
    let start_time = std::time::Instant::now();
    let poll_interval = Duration::from_secs(5);

    loop {
        if start_time.elapsed() > RENTAL_READY_TIMEOUT {
            return Ok(None);
        }

        // Fetch CPU rentals and find our rental
        match api_client.list_cpu_rentals().await {
            Ok(list) => {
                if let Some(rental) = list.rentals.iter().find(|r| r.rental_id == rental_id) {
                    // Check if rental is running and has SSH
                    if rental.status == "running" && rental.ssh_command.is_some() {
                        return Ok(Some(rental.clone()));
                    }
                    // Check for failure states
                    if rental.status == "failed" || rental.status == "error" {
                        return Err(CliError::Internal(eyre!(
                            "CPU rental failed to start: {}",
                            rental.status
                        )));
                    }
                } else {
                    // Rental not found in list - fail immediately
                    return Err(CliError::Internal(eyre!(
                        "CPU rental {} not found in rental list",
                        rental_id
                    )));
                }
            }
            Err(e) => {
                debug!("Failed to poll CPU rental status: {}", e);
            }
        }

        tokio::time::sleep(poll_interval).await;
    }
}

/// Handle community cloud rental with a pre-selected GPU category (from unified selector)
async fn handle_community_cloud_rental_with_selection(
    api_client: basilica_sdk::BasilicaClient,
    selection: CommunityCloudSelection,
    options: UpOptions,
    config: &CliConfig,
) -> Result<(), CliError> {
    let spinner = create_spinner("Preparing rental request...");

    let user_max_hourly_rate_cents = options
        .max_hourly_rate
        .map(usd_per_gpu_hour_to_cents)
        .transpose()?;
    let effective_max_hourly_rate_cents = user_max_hourly_rate_cents
        .or(selection.derived_max_hourly_rate_cents)
        .ok_or_else(|| {
            complete_spinner_error(spinner.clone(), "Missing max hourly rate");
            CliError::Internal(
                eyre!("Selected Bourse offering does not include pricing information")
                    .suggestion("Retry with --max-hourly-rate <USD_PER_GPU_HOUR>"),
            )
        })?;

    // Build rental request
    let container_image = options.image.unwrap_or_else(|| config.image.name.clone());

    let env_vars = parse_env_vars(&options.env)
        .map_err(|e| eyre!("Invalid argument: {}", e.to_string()))
        .inspect_err(|_e| {
            complete_spinner_error(spinner.clone(), "Environment variable parsing failed");
        })?;

    let port_mappings: Vec<basilica_sdk::types::PortMappingRequest> =
        parse_port_mappings(&options.ports)
            .map_err(|e| eyre!("Invalid argument: {}", e.to_string()))
            .inspect_err(|_e| {
                complete_spinner_error(spinner.clone(), "Port mapping parsing failed");
            })?
            .into_iter()
            .map(Into::into)
            .collect();

    let command = if options.command.is_empty() {
        vec!["/bin/bash".to_string()]
    } else {
        options.command
    };

    // Get SSH public key for the rental
    let ssh_key = api_client
        .get_ssh_key()
        .await
        .map_err(|e| {
            complete_spinner_error(spinner.clone(), "Failed to get SSH key");
            CliError::Internal(eyre!(e))
        })?
        .ok_or_else(|| {
            complete_spinner_error(spinner.clone(), "No SSH key registered");
            CliError::Internal(
                eyre!("No SSH key registered with Basilica")
                    .suggestion("Run 'basilica ssh-keys add' to register your SSH key"),
            )
        })?;

    let request = StartRentalApiRequest {
        gpu_category: selection.gpu_category,
        gpu_count: selection.gpu_count,
        min_memory_gb: None,
        max_hourly_rate_cents: effective_max_hourly_rate_cents,
        container_image,
        ssh_public_key: ssh_key.public_key,
        environment: env_vars,
        ports: port_mappings,
        resources: ResourceRequirementsRequest {
            cpu_cores: options.cpu_cores.unwrap_or(0.0),
            memory_mb: options.memory_mb.unwrap_or(0),
            storage_mb: options.storage_mb.unwrap_or(0),
            gpu_count: selection.gpu_count,
            gpu_types: vec![],
        },
        command,
        volumes: vec![],
    };

    spinner.set_message("Creating rental...");
    let response = api_client.start_rental(request).await.map_err(|e| {
        complete_spinner_error(spinner.clone(), "Failed to create rental");
        CliError::Internal(
            eyre!(e)
                .note("The selected node is experiencing issues.")
                .suggestion("Try running 'basilica up' again to select a different node."),
        )
    })?;

    complete_spinner_and_clear(spinner);

    print_success(&format!(
        "Successfully started Bourse rental {}",
        response.rental_id
    ));

    let ssh_creds = match response.ssh_credentials {
        Some(ref creds) => creds,
        None => {
            print_info("SSH access not available (unexpected error)");
            return Ok(());
        }
    };

    if options.detach {
        // Look up the private key for display
        let private_key_path = get_ssh_private_key_path(&api_client)
            .await
            .map_err(CliError::Internal)?;
        display_ssh_connection_instructions(
            &response.rental_id,
            ssh_creds,
            &private_key_path,
            "SSH connection options:",
        )?;
    } else {
        print_info("Waiting for rental to become active...");
        let rental_active = poll_rental_status(&response.rental_id, &api_client).await?;

        if rental_active {
            print_info("Connecting to rental...");
            let (host, port, username) = parse_ssh_credentials(ssh_creds)?;
            let ssh_access = SshAccess {
                host,
                port,
                username,
            };

            let private_key_path = get_ssh_private_key_path(&api_client)
                .await
                .map_err(CliError::Internal)?;

            let ssh_client = SshClient::new(&config.ssh)?;
            match retry_ssh_connection(
                &ssh_client,
                &ssh_access,
                private_key_path.clone(),
                RENTAL_READY_TIMEOUT,
            )
            .await
            {
                Ok(_) => {
                    print_info("SSH session closed");
                    display_ssh_connection_instructions(
                        &response.rental_id,
                        ssh_creds,
                        &private_key_path,
                        "To reconnect to this rental:",
                    )?;
                }
                Err(e) => {
                    print_error(&format!("SSH connection failed: {}", e));
                    display_ssh_connection_instructions(
                        &response.rental_id,
                        ssh_creds,
                        &private_key_path,
                        "Try manually connecting using:",
                    )?;
                }
            }
        } else {
            println!();
            print_info("Rental is taking longer than expected to become active");
            print_info("Check status with: basilica ps");
            print_info(&format!("SSH with: basilica ssh {}", response.rental_id));
        }
    }

    Ok(())
}

/// Validate that no community-cloud-only options are provided for secure cloud rentals
fn validate_no_community_cloud_options(options: &UpOptions) -> Result<(), CliError> {
    let mut invalid_args = Vec::new();

    if options.image.is_some() {
        invalid_args.push("--image");
    }
    if !options.env.is_empty() {
        invalid_args.push("--env");
    }
    if !options.ports.is_empty() {
        invalid_args.push("--ports");
    }
    if !options.command.is_empty() {
        invalid_args.push("--command");
    }
    if options.cpu_cores.is_some() {
        invalid_args.push("--cpu-cores");
    }
    if options.memory_mb.is_some() {
        invalid_args.push("--memory-mb");
    }
    if options.storage_mb.is_some() {
        invalid_args.push("--storage-mb");
    }
    if options.max_hourly_rate.is_some() {
        invalid_args.push("--max-hourly-rate");
    }

    if !invalid_args.is_empty() {
        return Err(CliError::Internal(
            eyre!(
                "The following options are only supported for Bourse rentals: {}",
                invalid_args.join(", ")
            )
            .suggestion("Remove these options when using The Citadel, or use --compute bourse")
            .note(
                "The Citadel provides bare metal access; these options configure Docker containers",
            ),
        ));
    }
    Ok(())
}

/// Handle the `up` command - provision GPU instances
///
/// All paths use the unified offering resolver which presents a consistent
/// selection UI across both secure and community clouds.
pub async fn handle_up(
    target: Option<GpuTarget>,
    options: UpOptions,
    compute: Option<ComputeCategoryArg>,
    config: &CliConfig,
) -> Result<(), CliError> {
    let api_client = create_authenticated_client(config).await?;

    // Ensure SSH key is registered before proceeding (needed for both clouds)
    ensure_ssh_key_registered(&api_client).await?;

    // Extract GPU filter from target
    let gpu_filter_owned = target.as_ref().map(|t| t.0.as_str());

    // Convert compute arg to cloud filter
    let cloud_filter = compute.map(|c| match c {
        ComputeCategoryArg::SecureCloud => ComputeCategory::SecureCloud,
        ComputeCategoryArg::CommunityCloud => ComputeCategory::CommunityCloud,
    });

    // Build flavour filters from CLI options
    let flavour = crate::cli::handlers::gpu_rental_helpers::FlavourFilters {
        interconnect: options.interconnect.clone(),
        region: options.region.clone(),
        spot: options.spot,
        exclude_spot: options.exclude_spot,
    };

    // Use unified offering resolver for all paths
    let selected = resolve_offering_unified(
        &api_client,
        gpu_filter_owned.as_deref(),
        options.gpu_count,
        options.country.as_deref(),
        None, // min_gpu_memory - not available in UpOptions
        cloud_filter,
        &flavour,
    )
    .await
    .map_err(CliError::Internal)?;

    match selected {
        SelectedOffering::SecureCloud(offering) => {
            validate_no_community_cloud_options(&options)?;
            handle_rental_with_offering(
                api_client,
                RentalOffering::SecureCloud(offering),
                options,
                config,
            )
            .await
        }
        SelectedOffering::CommunityCloud(selection) => {
            handle_community_cloud_rental_with_selection(api_client, selection, options, config)
                .await
        }
        SelectedOffering::CpuOnly(offering) => {
            validate_no_community_cloud_options(&options)?;
            handle_rental_with_offering(
                api_client,
                RentalOffering::CpuOnly(offering),
                options,
                config,
            )
            .await
        }
    }
}

/// Handle the `ps` command - list active rentals
pub async fn handle_ps(
    filters: PsFilters,
    compute: Option<ComputeCategoryArg>,
    json: bool,
    config: &CliConfig,
) -> Result<(), CliError> {
    use basilica_common::types::ComputeCategory;

    // Convert to Option<ComputeCategory> - None means show both
    let compute_category = compute.map(ComputeCategory::from);

    let api_client = create_authenticated_client(config).await?;

    // Branch based on compute category
    match compute_category {
        Some(ComputeCategory::CommunityCloud) => {
            if filters.history {
                // History mode: fetch from billing service
                let spinner = create_spinner("Fetching rental history...");

                let history_result = api_client.list_rental_history(Some(100)).await;

                let history = history_result.inspect_err(|_| {
                    complete_spinner_error(spinner.clone(), "Failed to load rental history")
                })?;

                complete_spinner_and_clear(spinner);

                if json {
                    json_output(&history)?;
                } else {
                    // Filter to only community cloud rentals and sort by start time (most recent first)
                    let mut community_history: Vec<_> = history
                        .rentals
                        .iter()
                        .filter(|r| r.cloud_type == "community")
                        .collect();
                    community_history.sort_by(|a, b| b.started_at.cmp(&a.started_at));

                    table_output::display_rental_history(&community_history)?;

                    // Calculate total cost for community cloud only
                    let total_cost: rust_decimal::Decimal = community_history
                        .iter()
                        .filter_map(|r| r.total_cost.parse::<rust_decimal::Decimal>().ok())
                        .sum();

                    println!();
                    println!(
                        "{}: {}",
                        style("Total Cost").cyan(),
                        style(format_usd(&total_cost.to_string())).green().bold()
                    );
                    println!("\nTotal: {} historical rentals", community_history.len());

                    display_ps_quick_start_commands();
                }
            } else {
                // Active rentals mode
                let spinner = create_spinner("Fetching active rentals...");

                // Build query from filters - default to "active" if no status specified
                let query = Some(ListRentalsQuery {
                    status: filters.status.or(Some(RentalState::Active)),
                    gpu_type: filters.gpu_type.clone(),
                    min_gpu_count: filters.min_gpu_count,
                });

                // Fetch rentals with timeout to handle unresponsive validator
                let rentals_result = with_validator_timeout(api_client.list_rentals(query)).await;

                let rentals_list = rentals_result.inspect_err(|_| {
                    complete_spinner_error(spinner.clone(), "Failed to load rentals")
                })?;

                complete_spinner_and_clear(spinner);

                if json {
                    json_output(&rentals_list)?;
                } else {
                    table_output::display_rental_items(&rentals_list.rentals[..])?;

                    println!("\nTotal: {} active rentals", rentals_list.rentals.len());

                    display_ps_quick_start_commands();
                }
            }
        }
        Some(ComputeCategory::SecureCloud) => {
            if filters.history {
                // History mode: fetch from billing service (which stores all rental history)
                let spinner = create_spinner("Fetching rental history...");

                let history_result = api_client.list_rental_history(Some(100)).await;

                let history = history_result.inspect_err(|_| {
                    complete_spinner_error(spinner.clone(), "Failed to load rental history")
                })?;

                complete_spinner_and_clear(spinner);

                if json {
                    use serde_json::json;
                    let mut secure_gpu_history: Vec<_> = history
                        .rentals
                        .iter()
                        .filter(|r| r.cloud_type == "secure" && !is_secure_cpu_history_item(r))
                        .cloned()
                        .collect();
                    secure_gpu_history.sort_by(|a, b| b.started_at.cmp(&a.started_at));

                    let mut secure_cpu_history: Vec<_> = history
                        .rentals
                        .iter()
                        .filter(|r| r.cloud_type == "secure" && is_secure_cpu_history_item(r))
                        .cloned()
                        .collect();
                    secure_cpu_history.sort_by(|a, b| b.started_at.cmp(&a.started_at));

                    let output = json!({
                        "secure_cloud_history": secure_gpu_history,
                        "secure_cloud_cpu_history": secure_cpu_history
                    });
                    json_output(&output)?;
                } else {
                    // Filter to only secure cloud rentals and sort by start time (most recent first)
                    let mut secure_gpu_history: Vec<_> = history
                        .rentals
                        .iter()
                        .filter(|r| r.cloud_type == "secure" && !is_secure_cpu_history_item(r))
                        .collect();
                    secure_gpu_history.sort_by(|a, b| b.started_at.cmp(&a.started_at));

                    let mut secure_cpu_history: Vec<_> = history
                        .rentals
                        .iter()
                        .filter(|r| r.cloud_type == "secure" && is_secure_cpu_history_item(r))
                        .collect();
                    secure_cpu_history.sort_by(|a, b| b.started_at.cmp(&a.started_at));

                    print_cloud_section_header("The Citadel (GPU) Rental History", true);
                    table_output::display_rental_history(&secure_gpu_history)?;

                    let secure_gpu_total_cost: rust_decimal::Decimal = secure_gpu_history
                        .iter()
                        .filter_map(|r| r.total_cost.parse::<rust_decimal::Decimal>().ok())
                        .sum();

                    println!();
                    println!(
                        "{}: {}",
                        style("Total Cost").cyan(),
                        style(format_usd(&secure_gpu_total_cost.to_string()))
                            .green()
                            .bold()
                    );
                    println!(
                        "\nTotal: {} historical Citadel (GPU) rentals",
                        secure_gpu_history.len()
                    );

                    println!();

                    print_cloud_section_header("The Citadel (CPU) History", false);
                    table_output::display_cpu_rental_history(&secure_cpu_history)?;

                    let secure_cpu_total_cost: rust_decimal::Decimal = secure_cpu_history
                        .iter()
                        .filter_map(|r| r.total_cost.parse::<rust_decimal::Decimal>().ok())
                        .sum();

                    println!();
                    println!(
                        "{}: {}",
                        style("Total Cost").cyan(),
                        style(format_usd(&secure_cpu_total_cost.to_string()))
                            .green()
                            .bold()
                    );
                    println!(
                        "\nTotal: {} historical Citadel (CPU) rentals",
                        secure_cpu_history.len()
                    );

                    display_ps_quick_start_commands();
                }
            } else {
                // Active rentals mode: fetch GPU and CPU rentals from secure cloud providers
                let spinner = create_spinner("Fetching active rentals...");

                let (gpu_result, cpu_result) = tokio::join!(
                    api_client.list_secure_cloud_rentals(),
                    api_client.list_cpu_rentals()
                );

                let gpu_rentals_list = gpu_result.inspect_err(|_| {
                    complete_spinner_error(
                        spinner.clone(),
                        "Failed to load secure cloud GPU rentals",
                    )
                })?;

                let cpu_rentals_list = cpu_result.inspect_err(|_| {
                    complete_spinner_error(spinner.clone(), "Failed to load CPU-only rentals")
                })?;

                complete_spinner_and_clear(spinner);

                if json {
                    use serde_json::json;
                    let output = json!({
                        "gpu_rentals": gpu_rentals_list,
                        "cpu_rentals": cpu_rentals_list
                    });
                    json_output(&output)?;
                } else {
                    // Show active GPU rentals with filters applied
                    let gpu_rentals_to_display: Vec<_> =
                        filter_secure_cloud_rentals(&gpu_rentals_list.rentals, &filters)
                            .into_iter()
                            .collect();

                    println!("{}", style("The Citadel (GPU)").bold().cyan());
                    table_output::display_secure_cloud_rentals(&gpu_rentals_to_display)?;

                    println!(
                        "\nTotal: {} Citadel (GPU) rentals",
                        gpu_rentals_to_display.len()
                    );

                    println!();

                    // Show active CPU-only rentals (no gpu_type or min_gpu_count filters apply)
                    let cpu_rentals_to_display: Vec<_> = cpu_rentals_list
                        .rentals
                        .iter()
                        .filter(|r| r.stopped_at.is_none() && r.gpu_count == 0)
                        .collect();

                    println!("{}", style("The Citadel (CPU)").bold().cyan());
                    table_output::display_cpu_rentals(&cpu_rentals_to_display)?;

                    println!(
                        "\nTotal: {} Citadel (CPU) rentals",
                        cpu_rentals_to_display.len()
                    );

                    display_ps_quick_start_commands();
                }
            }
        }
        None => {
            // Dual-table display: show both community cloud and secure cloud rentals
            let spinner = if filters.history {
                create_spinner("Fetching rental history...")
            } else {
                create_spinner("Fetching rentals...")
            };

            if filters.history {
                // History mode: fetch from billing service (which stores ALL rental history)
                let history_result = api_client.list_rental_history(Some(100)).await;

                let history = match history_result {
                    Ok(h) => h,
                    Err(e) => {
                        complete_spinner_error(spinner.clone(), "Failed to load rental history");
                        warn!("Failed to load rental history: {}", e);
                        HistoricalRentalsResponse {
                            rentals: vec![],
                            total_count: 0,
                            total_cost: "0.00".to_string(),
                        }
                    }
                };

                complete_spinner_and_clear(spinner);

                if json {
                    use serde_json::json;
                    // Split history by cloud type and sort by start time (most recent first)
                    let mut community_history: Vec<_> = history
                        .rentals
                        .iter()
                        .filter(|r| r.cloud_type == "community")
                        .cloned()
                        .collect();
                    community_history.sort_by(|a, b| b.started_at.cmp(&a.started_at));
                    let mut secure_gpu_history: Vec<_> = history
                        .rentals
                        .iter()
                        .filter(|r| r.cloud_type == "secure" && !is_secure_cpu_history_item(r))
                        .cloned()
                        .collect();
                    secure_gpu_history.sort_by(|a, b| b.started_at.cmp(&a.started_at));
                    let mut secure_cpu_history: Vec<_> = history
                        .rentals
                        .iter()
                        .filter(|r| r.cloud_type == "secure" && is_secure_cpu_history_item(r))
                        .cloned()
                        .collect();
                    secure_cpu_history.sort_by(|a, b| b.started_at.cmp(&a.started_at));
                    let output = json!({
                        "community_cloud_history": community_history,
                        "secure_cloud_history": secure_gpu_history,
                        "secure_cloud_cpu_history": secure_cpu_history
                    });
                    json_output(&output)?;
                } else {
                    // Filter and sort by start time (most recent first)
                    let mut community_history: Vec<_> = history
                        .rentals
                        .iter()
                        .filter(|r| r.cloud_type == "community")
                        .collect();
                    community_history.sort_by(|a, b| b.started_at.cmp(&a.started_at));

                    let mut secure_gpu_history: Vec<_> = history
                        .rentals
                        .iter()
                        .filter(|r| r.cloud_type == "secure" && !is_secure_cpu_history_item(r))
                        .collect();
                    secure_gpu_history.sort_by(|a, b| b.started_at.cmp(&a.started_at));

                    let mut secure_cpu_history: Vec<_> = history
                        .rentals
                        .iter()
                        .filter(|r| r.cloud_type == "secure" && is_secure_cpu_history_item(r))
                        .collect();
                    secure_cpu_history.sort_by(|a, b| b.started_at.cmp(&a.started_at));

                    // Display community cloud history
                    print_cloud_section_header("The Bourse History", true);
                    table_output::display_rental_history(&community_history)?;

                    let community_total_cost: rust_decimal::Decimal = community_history
                        .iter()
                        .filter_map(|r| r.total_cost.parse::<rust_decimal::Decimal>().ok())
                        .sum();

                    println!();
                    println!(
                        "{}: {}",
                        style("Total Cost").cyan(),
                        style(format_usd(&community_total_cost.to_string()))
                            .green()
                            .bold()
                    );
                    println!(
                        "\nTotal: {} historical Bourse rentals",
                        community_history.len()
                    );

                    println!();

                    // Display secure cloud GPU history
                    print_cloud_section_header("The Citadel (GPU) History", false);
                    table_output::display_rental_history(&secure_gpu_history)?;

                    let secure_gpu_total_cost: rust_decimal::Decimal = secure_gpu_history
                        .iter()
                        .filter_map(|r| r.total_cost.parse::<rust_decimal::Decimal>().ok())
                        .sum();

                    println!();
                    println!(
                        "{}: {}",
                        style("Total Cost").cyan(),
                        style(format_usd(&secure_gpu_total_cost.to_string()))
                            .green()
                            .bold()
                    );
                    println!(
                        "\nTotal: {} historical Citadel (GPU) rentals",
                        secure_gpu_history.len()
                    );

                    println!();

                    // Display secure cloud CPU history
                    print_cloud_section_header("The Citadel (CPU) History", false);
                    table_output::display_cpu_rental_history(&secure_cpu_history)?;

                    let secure_cpu_total_cost: rust_decimal::Decimal = secure_cpu_history
                        .iter()
                        .filter_map(|r| r.total_cost.parse::<rust_decimal::Decimal>().ok())
                        .sum();

                    println!();
                    println!(
                        "{}: {}",
                        style("Total Cost").cyan(),
                        style(format_usd(&secure_cpu_total_cost.to_string()))
                            .green()
                            .bold()
                    );
                    println!(
                        "\nTotal: {} historical Citadel (CPU) rentals",
                        secure_cpu_history.len()
                    );

                    display_ps_quick_start_commands();
                }
            } else {
                // Active rentals mode
                let query = Some(ListRentalsQuery {
                    status: filters.status.clone().or(Some(RentalState::Active)),
                    gpu_type: filters.gpu_type.clone(),
                    min_gpu_count: filters.min_gpu_count,
                });

                // Fetch community, secure cloud GPU, and CPU rentals in parallel
                let (community_result, secure_result, cpu_result) = tokio::join!(
                    with_validator_timeout(api_client.list_rentals(query)),
                    api_client.list_secure_cloud_rentals(),
                    api_client.list_cpu_rentals()
                );

                // Graceful degradation: use empty results on community cloud timeout
                let community_rentals_list = community_result.unwrap_or_else(|e| {
                    warn!("Failed to load community cloud rentals: {}", e);
                    basilica_sdk::types::ApiListRentalsResponse {
                        rentals: vec![],
                        total_count: 0,
                    }
                });

                let secure_rentals_list = secure_result.inspect_err(|_| {
                    complete_spinner_error(spinner.clone(), "Failed to load secure cloud rentals")
                })?;

                let cpu_rentals_list = cpu_result.unwrap_or_else(|e| {
                    warn!("Failed to load CPU-only rentals: {}", e);
                    basilica_sdk::types::ListSecureCloudRentalsResponse {
                        rentals: vec![],
                        total_count: 0,
                    }
                });

                complete_spinner_and_clear(spinner);

                if json {
                    use serde_json::json;
                    let output = json!({
                        "community_cloud": community_rentals_list,
                        "secure_cloud": secure_rentals_list,
                        "cpu_only": cpu_rentals_list
                    });
                    json_output(&output)?;
                } else {
                    // Section 1: Community Cloud
                    print_cloud_section_header("The Bourse", true);

                    table_output::display_rental_items(&community_rentals_list.rentals[..])?;

                    println!(
                        "\nTotal: {} Bourse rentals",
                        community_rentals_list.rentals.len()
                    );

                    println!();

                    // Section 2: Secure Cloud (GPU) with filters applied
                    print_cloud_section_header("The Citadel (GPU)", false);

                    let secure_rentals_to_display: Vec<_> =
                        filter_secure_cloud_rentals(&secure_rentals_list.rentals, &filters)
                            .into_iter()
                            .collect();

                    table_output::display_secure_cloud_rentals(&secure_rentals_to_display)?;

                    println!(
                        "\nTotal: {} Citadel (GPU) rentals",
                        secure_rentals_to_display.len()
                    );

                    println!();

                    // Section 3: Secure Cloud (CPU) (no gpu_type or min_gpu_count filters apply)
                    print_cloud_section_header("The Citadel (CPU)", false);

                    let cpu_rentals_to_display: Vec<_> = cpu_rentals_list
                        .rentals
                        .iter()
                        .filter(|r| r.stopped_at.is_none() && r.gpu_count == 0)
                        .collect();

                    table_output::display_cpu_rentals(&cpu_rentals_to_display)?;

                    println!(
                        "\nTotal: {} Citadel (CPU) rentals",
                        cpu_rentals_to_display.len()
                    );

                    display_ps_quick_start_commands();
                }
            }
        }
    }

    Ok(())
}

/// Handle the `status` command - check rental status
pub async fn handle_status(
    target: Option<String>,
    json: bool,
    config: &CliConfig,
) -> Result<(), CliError> {
    let api_client = create_authenticated_client(config).await?;

    // Determine rental ID and compute type
    let (rental_id, compute_type) = if let Some(target_id) = target {
        // Rental ID provided - resolve which cloud it belongs to
        let compute_type = resolve_rental_by_id(&target_id, &api_client).await?;
        (target_id, compute_type)
    } else {
        // No rental ID provided - use unified selector
        // exclude_vip=false: VIP rentals can be viewed
        resolve_target_rental_unified(None, None, &api_client, false).await?
    };

    let spinner = create_spinner("Fetching rental status...");

    match compute_type {
        ComputeCategory::CommunityCloud => {
            // Fetch community cloud status
            let status =
                api_client
                    .get_rental_status(&rental_id)
                    .await
                    .map_err(|e| -> CliError {
                        complete_spinner_error(spinner.clone(), "Failed to get status");
                        let report = match e {
                            ApiError::NotFound { .. } => eyre!("Rental '{}' not found", rental_id)
                                .suggestion("Try 'basilica ps' to see your active rentals")
                                .note("The rental may have expired or been terminated"),
                            _ => {
                                eyre!(e).suggestion("Check your internet connection and try again")
                            }
                        };
                        CliError::Internal(report)
                    })?;

            complete_spinner_and_clear(spinner);

            if json {
                json_output(&status)?;
            } else {
                // Try to find the private key for display
                let private_key_path =
                    api_client
                        .get_ssh_key()
                        .await
                        .ok()
                        .flatten()
                        .and_then(|ssh_key| {
                            crate::ssh::find_private_key_for_public_key(&ssh_key.public_key).ok()
                        });
                display_rental_status_with_details(&status, private_key_path.as_deref());
            }
        }
        ComputeCategory::SecureCloud => {
            // Fetch secure cloud status (GPU + CPU)
            let (gpu_result, cpu_result) = tokio::join!(
                api_client.list_secure_cloud_rentals(),
                api_client.list_cpu_rentals()
            );

            let mut gpu_error: Option<ApiError> = None;
            let mut cpu_error: Option<ApiError> = None;

            let gpu_rentals = match gpu_result {
                Ok(list) => Some(list),
                Err(err) => {
                    gpu_error = Some(err);
                    None
                }
            };

            let cpu_rentals = match cpu_result {
                Ok(list) => Some(list),
                Err(err) => {
                    cpu_error = Some(err);
                    None
                }
            };

            if gpu_error.is_some() && cpu_error.is_some() {
                complete_spinner_error(spinner.clone(), "Failed to get status");
                let err = gpu_error.or(cpu_error).unwrap();
                return Err(CliError::Internal(
                    eyre!(err).suggestion("Check your internet connection and try again"),
                ));
            }

            if let Some(rental) = cpu_rentals
                .as_ref()
                .and_then(|list| list.rentals.iter().find(|r| r.rental_id == rental_id))
            {
                complete_spinner_and_clear(spinner);

                if json {
                    json_output(&rental)?;
                } else {
                    // Display secure cloud CPU rental details
                    println!("Rental Status: {}", rental.rental_id);
                    println!("  Provider: {}", rental.provider);
                    println!("  Status: {}", rental.status);
                    if let Some(vcpu) = rental.vcpu_count {
                        println!("  vCPU: {}", vcpu);
                    } else {
                        println!("  vCPU: N/A");
                    }
                    if let Some(mem) = rental.system_memory_gb {
                        println!("  Memory: {}GB", mem);
                    }
                    if let Some(ip) = &rental.ip_address {
                        println!("  IP Address: {}", ip);
                    }
                    println!("  Hourly Cost: ${:.2}/hr", rental.hourly_cost);
                    println!("  Created: {}", rental.created_at);
                    if let Some(stopped_at) = &rental.stopped_at {
                        println!("  Stopped: {}", stopped_at);
                    }
                    // Show SSH command with private key path if available locally
                    if let Some(ip) = &rental.ip_address {
                        if let Some(ref ssh_public_key) = rental.ssh_public_key {
                            if let Ok(private_key_path) =
                                find_private_key_for_public_key(ssh_public_key)
                            {
                                // Full SSH command with private key
                                println!(
                                    "  SSH: {}",
                                    style(format!(
                                        "ssh -i {} ubuntu@{}",
                                        compress_path(&private_key_path),
                                        ip
                                    ))
                                    .cyan()
                                );
                            } else {
                                // Key not found locally, show basic command
                                println!("  SSH: ssh ubuntu@{}", ip);
                                println!("  SSH Key: {}", style("Not found locally").yellow());
                            }
                        } else {
                            // No public key info, show basic command
                            println!("  SSH: ssh ubuntu@{}", ip);
                        }
                    }
                }

                return Ok(());
            }

            if let Some(rental) = gpu_rentals
                .as_ref()
                .and_then(|list| list.rentals.iter().find(|r| r.rental_id == rental_id))
            {
                complete_spinner_and_clear(spinner);

                if json {
                    json_output(&rental)?;
                } else {
                    // Display secure cloud GPU rental details
                    println!("Rental Status: {}", rental.rental_id);
                    println!("  Provider: {}", rental.provider);
                    println!("  Status: {}", rental.status);
                    let gpu_label = format!("{}x {}", rental.gpu_count, rental.gpu_type);
                    if rental.is_spot {
                        println!("  GPU: {} (Spot)", gpu_label);
                    } else {
                        println!("  GPU: {}", gpu_label);
                    }
                    if let Some(ip) = &rental.ip_address {
                        println!("  IP Address: {}", ip);
                    }
                    println!("  Hourly Cost: ${:.2}/hr", rental.hourly_cost);
                    println!("  Created: {}", rental.created_at);
                    if let Some(stopped_at) = &rental.stopped_at {
                        println!("  Stopped: {}", stopped_at);
                    }
                    // Show SSH command with private key path if available locally
                    if let Some(ip) = &rental.ip_address {
                        if let Some(ref ssh_public_key) = rental.ssh_public_key {
                            if let Ok(private_key_path) =
                                find_private_key_for_public_key(ssh_public_key)
                            {
                                // Full SSH command with private key
                                println!(
                                    "  SSH: {}",
                                    style(format!(
                                        "ssh -i {} ubuntu@{}",
                                        compress_path(&private_key_path),
                                        ip
                                    ))
                                    .cyan()
                                );
                            } else {
                                // Key not found locally, show basic command
                                println!("  SSH: ssh ubuntu@{}", ip);
                                println!("  SSH Key: {}", style("Not found locally").yellow());
                            }
                        } else {
                            // No public key info, show basic command
                            println!("  SSH: ssh ubuntu@{}", ip);
                        }
                    }
                }

                return Ok(());
            }

            complete_spinner_error(spinner.clone(), "Rental not found");
            return Err(CliError::Internal(
                eyre!("Rental '{}' not found", rental_id)
                    .suggestion("Try 'basilica ps' to see your active rentals")
                    .note("The rental may have expired or been terminated"),
            ));
        }
    }

    Ok(())
}

/// Handle the `logs` command - view rental logs
pub async fn handle_logs(
    target: Option<String>,
    options: LogsOptions,
    config: &CliConfig,
) -> Result<(), CliError> {
    // Create API client
    let api_client = create_authenticated_client(config).await?;

    // Determine rental ID and compute type
    let (rental_id, compute_type) = if let Some(target_id) = target {
        // Rental ID provided - resolve which cloud it belongs to
        let compute_type = resolve_rental_by_id(&target_id, &api_client).await?;
        (target_id, compute_type)
    } else {
        // No rental ID provided - use unified selector
        // exclude_vip=false: VIP rentals can be viewed
        resolve_target_rental_unified(None, None, &api_client, false).await?
    };

    // Check if this is a secure cloud rental
    if matches!(compute_type, ComputeCategory::SecureCloud) {
        return Err(CliError::Internal(
            eyre!("Log streaming is not yet available for secure cloud rentals")
                .note("Secure cloud logs support is coming soon")
                .suggestion(format!(
                    "For now, use SSH to access logs manually: basilica ssh {}",
                    rental_id
                )),
        ));
    }

    let target = rental_id;

    let spinner = create_spinner("Connecting to log stream...");

    // Get log stream from API
    let response = api_client
        .get_rental_logs(&target, options.follow, options.tail)
        .await
        .inspect_err(|_| complete_spinner_error(spinner.clone(), "Failed to connect to logs"))?;

    // Check content type
    let content_type = response
        .headers()
        .get("content-type")
        .and_then(|v| v.to_str().ok())
        .unwrap_or("");

    if !content_type.contains("text/event-stream") {
        // Not an SSE stream, try to get error message
        let status = response.status();
        let body = response
            .text()
            .await
            .unwrap_or_else(|_| "Unknown error".to_string());

        complete_spinner_error(spinner, "Failed to get logs");

        if status == StatusCode::NOT_FOUND {
            return Err(eyre!(
                "Rental '{}' not found. Run 'basilica ps' to see active rentals",
                target
            )
            .into());
        } else {
            return Err(eyre!(
                "API request failed for get logs: status {}: {}",
                status,
                body
            )
            .into());
        }
    }

    complete_spinner_and_clear(spinner);

    println!("Streaming logs for rental {}...", target);
    if options.follow {
        println!("Following log output - press Ctrl+C to stop");
    }

    stream_logs_to_stdout(response).await
}

/// Handle the `down` command - terminate rental
pub async fn handle_down(
    target: Option<String>,
    compute: Option<ComputeCategoryArg>,
    all: bool,
    config: &CliConfig,
) -> Result<(), CliError> {
    use super::gpu_rental_helpers::resolve_target_rental_unified;
    use basilica_common::types::ComputeCategory;

    let api_client = create_authenticated_client(config).await?;
    let compute_filter = compute.map(ComputeCategory::from);

    if all {
        // Stop all active rentals based on compute filter
        let spinner = create_spinner("Fetching active rentals...");

        // Determine what to fetch based on filter
        let (community_rentals, secure_rentals, cpu_rentals) = match compute_filter {
            Some(ComputeCategory::CommunityCloud) => {
                // Fetch only community cloud
                let rentals = api_client
                    .list_rentals(active_rentals_query())
                    .await
                    .map_err(|e| {
                        complete_spinner_error(spinner.clone(), "Failed to fetch rentals");
                        CliError::Internal(eyre!(e).wrap_err("Failed to fetch active rentals"))
                    })?;
                (Some(rentals), None, None)
            }
            Some(ComputeCategory::SecureCloud) => {
                // Fetch secure cloud GPU and CPU rentals
                let (gpu_rentals, cpu_rentals) = tokio::join!(
                    api_client.list_secure_cloud_rentals(),
                    api_client.list_cpu_rentals()
                );
                let rentals = gpu_rentals.map_err(|e| {
                    complete_spinner_error(spinner.clone(), "Failed to fetch secure cloud rentals");
                    CliError::Internal(eyre!(e).wrap_err("Failed to fetch secure cloud rentals"))
                })?;
                (None, Some(rentals), cpu_rentals.ok())
            }
            None => {
                // Fetch all types with timeout for community cloud
                let community_future = api_client.list_rentals(active_rentals_query());
                let (community_result, secure_result, cpu_result) = tokio::join!(
                    with_validator_timeout(community_future),
                    api_client.list_secure_cloud_rentals(),
                    api_client.list_cpu_rentals()
                );
                (community_result.ok(), secure_result.ok(), cpu_result.ok())
            }
        };

        complete_spinner_and_clear(spinner);

        let mut total_count = 0;
        let mut success_count = 0;
        let mut failed_rentals = Vec::new();

        // Stop community cloud rentals
        if let Some(community) = community_rentals {
            for rental in community.rentals {
                total_count += 1;
                let rental_id = &rental.rental_id;
                let spinner = create_spinner(&format!("Stopping rental: {}", rental_id));

                match api_client.stop_rental(rental_id).await {
                    Ok(_) => {
                        complete_spinner_and_clear(spinner);
                        print_success(&format!("Successfully stopped Bourse rental {}", rental_id));
                        success_count += 1;
                    }
                    Err(e) => {
                        complete_spinner_error(
                            spinner,
                            &format!("Failed to stop rental: {}", rental_id),
                        );
                        failed_rentals.push((rental_id.clone(), "community".to_string(), e));
                    }
                }
            }
        }

        // Stop secure cloud GPU rentals (only active ones)
        if let Some(secure) = secure_rentals {
            for rental in secure.rentals {
                // Skip stopped rentals
                if rental.stopped_at.is_some() {
                    continue;
                }

                total_count += 1;
                let rental_id = &rental.rental_id;
                let spinner = create_spinner(&format!("Stopping rental: {}", rental_id));

                match api_client.stop_secure_cloud_rental(rental_id).await {
                    Ok(_) => {
                        complete_spinner_and_clear(spinner);
                        print_success(&format!(
                            "Successfully stopped Citadel (GPU) rental {}",
                            rental_id
                        ));
                        success_count += 1;
                    }
                    Err(e) => {
                        complete_spinner_error(
                            spinner,
                            &format!("Failed to stop rental: {}", rental_id),
                        );
                        failed_rentals.push((rental_id.clone(), "secure-gpu".to_string(), e));
                    }
                }
            }
        }

        // Stop CPU-only rentals (only active ones)
        if let Some(cpu) = cpu_rentals {
            for rental in cpu.rentals {
                // Skip stopped rentals
                if rental.stopped_at.is_some() {
                    continue;
                }

                total_count += 1;
                let rental_id = &rental.rental_id;
                let spinner = create_spinner(&format!("Stopping CPU rental: {}", rental_id));

                match api_client.stop_cpu_rental(rental_id).await {
                    Ok(_) => {
                        complete_spinner_and_clear(spinner);
                        print_success(&format!(
                            "Successfully stopped CPU-only rental {}",
                            rental_id
                        ));
                        success_count += 1;
                    }
                    Err(e) => {
                        complete_spinner_error(
                            spinner,
                            &format!("Failed to stop CPU rental: {}", rental_id),
                        );
                        failed_rentals.push((rental_id.clone(), "cpu-only".to_string(), e));
                    }
                }
            }
        }

        if total_count == 0 {
            println!("No active rentals found.");
            return Ok(());
        }

        // Print summary
        println!();
        if failed_rentals.is_empty() {
            print_success(&format!(
                "Successfully stopped all {} rental{}",
                success_count,
                if success_count == 1 { "" } else { "s" }
            ));
        } else {
            print_success(&format!(
                "Successfully stopped {} out of {} rental{}",
                success_count,
                total_count,
                if total_count == 1 { "" } else { "s" }
            ));

            if !failed_rentals.is_empty() {
                println!("\nFailed to stop the following rentals:");
                for (rental_id, rental_type, error) in failed_rentals {
                    println!("  - {} ({}): {}", rental_id, rental_type, error);
                }
            }
        }
    } else {
        // Single rental termination using unified resolution
        // exclude_vip=true: VIP rentals cannot be stopped by users
        let (rental_id, compute_type) =
            resolve_target_rental_unified(target, compute_filter, &api_client, true).await?;

        let spinner = create_spinner(&format!("Stopping rental: {}", rental_id));

        match compute_type {
            ComputeCategory::CommunityCloud => {
                // Stop community cloud rental
                api_client
                    .stop_rental(&rental_id)
                    .await
                    .map_err(|e| -> CliError {
                        complete_spinner_error(spinner.clone(), "Failed to stop rental");
                        let report = match e {
                            ApiError::NotFound { .. } => eyre!("Rental '{}' not found", rental_id)
                                .suggestion("Try 'basilica ps' to see your active rentals")
                                .note("The rental may have already been stopped"),
                            _ => {
                                eyre!(e).suggestion("Check your internet connection and try again")
                            }
                        };
                        CliError::Internal(report)
                    })?;

                complete_spinner_and_clear(spinner);
                print_success(&format!("Successfully stopped Bourse rental {}", rental_id));
            }
            ComputeCategory::SecureCloud => {
                // Check if this is a CPU rental by looking in CPU rentals list first
                let is_cpu_rental = api_client
                    .list_cpu_rentals()
                    .await
                    .map(|list| list.rentals.iter().any(|r| r.rental_id == rental_id))
                    .unwrap_or(false);

                if is_cpu_rental {
                    // Stop CPU rental
                    api_client
                        .stop_cpu_rental(&rental_id)
                        .await
                        .map_err(|e| -> CliError {
                            complete_spinner_error(spinner.clone(), "Failed to stop CPU rental");
                            let report = match e {
                                ApiError::NotFound { .. } => eyre!(
                                    "CPU rental '{}' not found",
                                    rental_id
                                )
                                .suggestion(
                                    "Try 'basilica ps --compute secure-cloud' to see your rentals",
                                )
                                .note("The rental may have already been stopped"),
                                _ => eyre!(e)
                                    .suggestion("Check your internet connection and try again"),
                            };
                            CliError::Internal(report)
                        })?;

                    complete_spinner_and_clear(spinner);
                    print_success(&format!(
                        "Successfully stopped CPU-only rental {}",
                        rental_id
                    ));
                } else {
                    // Stop secure cloud GPU rental
                    api_client
                        .stop_secure_cloud_rental(&rental_id)
                        .await
                        .map_err(|e| -> CliError {
                            complete_spinner_error(spinner.clone(), "Failed to stop rental");
                            let report = match e {
                                ApiError::NotFound { .. } => eyre!(
                                    "Rental '{}' not found",
                                    rental_id
                                )
                                .suggestion(
                                    "Try 'basilica ps --compute secure-cloud' to see your rentals",
                                )
                                .note("The rental may have already been stopped"),
                                _ => eyre!(e)
                                    .suggestion("Check your internet connection and try again"),
                            };
                            CliError::Internal(report)
                        })?;

                    complete_spinner_and_clear(spinner);
                    print_success(&format!(
                        "Successfully stopped Citadel rental {}",
                        rental_id
                    ));
                }
            }
        }
    }

    Ok(())
}

/// Handle the `restart` command - restart rental container
pub async fn handle_restart(target: Option<String>, config: &CliConfig) -> Result<(), CliError> {
    let api_client = create_authenticated_client(config).await?;

    // Single rental restart (no --all flag as per requirements)
    // exclude_vip=false: VIP rentals included in selector (though restart may fail for secure cloud)
    let (rental_id, _compute_type) =
        resolve_target_rental_unified(target, None, &api_client, false).await?;
    let spinner = create_spinner(&format!("Restarting rental: {}", rental_id));

    api_client
        .restart_rental(&rental_id)
        .await
        .map_err(|e| -> CliError {
            complete_spinner_error(spinner.clone(), "Failed to restart rental");
            let report = match e {
                ApiError::NotFound { .. } => eyre!("Rental '{}' not found", rental_id)
                    .suggestion("Try 'basilica ps' to see your active rentals"),
                ApiError::Conflict { message } => {
                    eyre!("Cannot restart rental: {}", message).suggestion(
                        "Only Active rentals can be restarted. Check rental status with 'basilica status'",
                    )
                }
                _ => eyre!(e).suggestion("Check your internet connection and try again"),
            };
            CliError::Internal(report)
        })?;

    complete_spinner_and_clear(spinner);
    print_success(&format!("Successfully restarted rental: {}", rental_id));

    Ok(())
}

/// Handle the `exec` command - execute commands via SSH
pub async fn handle_exec(
    target: Option<String>,
    command: String,
    config: &CliConfig,
) -> Result<(), CliError> {
    // Create API client to verify rental status
    let api_client = create_authenticated_client(config).await?;

    // Resolve rental and SSH credentials
    let RentalWithSsh {
        rental_id,
        compute_type,
        ssh_command,
        ssh_public_key,
    } = resolve_rental_with_ssh(target.as_deref(), &api_client).await?;

    debug!(
        "Executing command on {} rental: {}",
        match compute_type {
            ComputeCategory::CommunityCloud => "The Bourse",
            ComputeCategory::SecureCloud => "The Citadel",
        },
        rental_id
    );

    // Parse SSH credentials (format is the same for both types)
    let (host, port, username) = parse_ssh_credentials(&ssh_command)?;
    let ssh_access = SshAccess {
        host,
        port,
        username,
    };

    // Find matching private key using the rental's stored public key
    let private_key_path = {
        let public_key = ssh_public_key.ok_or_else(|| {
            CliError::Internal(
                eyre!("No SSH public key available for this rental")
                    .suggestion("The required SSH key may not be on this machine")
                    .note("SSH access requires the original key used during rental creation"),
            )
        })?;

        crate::ssh::find_private_key_for_public_key(&public_key).map_err(CliError::Internal)?
    };

    debug!("Using private key for exec: {}", private_key_path.display());

    // Use SSH client to execute command
    let ssh_client = SshClient::new(&config.ssh)?;
    ssh_client
        .execute_command(&ssh_access, &command, private_key_path)
        .await?;
    Ok(())
}

/// Handle the `ssh` command - SSH into instances
pub async fn handle_ssh(
    target: Option<String>,
    options: crate::cli::commands::SshOptions,
    config: &CliConfig,
) -> Result<(), CliError> {
    // Create API client to verify rental status
    let api_client = create_authenticated_client(config).await?;

    // Resolve rental and SSH credentials
    let RentalWithSsh {
        rental_id,
        compute_type,
        ssh_command,
        ssh_public_key,
    } = resolve_rental_with_ssh(target.as_deref(), &api_client).await?;

    debug!(
        "Opening SSH connection to {} rental: {}",
        match compute_type {
            ComputeCategory::CommunityCloud => "The Bourse",
            ComputeCategory::SecureCloud => "The Citadel",
        },
        rental_id
    );

    // Parse SSH credentials (format is the same for both types)
    debug!("Raw ssh_command: {}", ssh_command);
    let (host, port, username) = parse_ssh_credentials(&ssh_command)?;
    debug!(
        "Parsed credentials - host: {}, port: {}, username: {}",
        host, port, username
    );
    let ssh_access = SshAccess {
        host,
        port,
        username,
    };

    // Find matching private key using the rental's stored public key
    let private_key_path = {
        let public_key = ssh_public_key.ok_or_else(|| {
            CliError::Internal(
                eyre!("No SSH public key available for this rental")
                    .suggestion("The required SSH key may not be on this machine")
                    .note("SSH access requires the original key used during rental creation"),
            )
        })?;

        crate::ssh::find_private_key_for_public_key(&public_key).map_err(CliError::Internal)?
    };

    debug!("Using private key: {}", private_key_path.display());

    // Use SSH client to handle connection with options
    let ssh_client = SshClient::new(&config.ssh)?;

    // Open interactive session with port forwarding options
    ssh_client
        .interactive_session_with_options(&ssh_access, &options, private_key_path)
        .await?;
    Ok(())
}

/// Handle the `cp` command - copy files via SSH
pub async fn handle_cp(
    source: String,
    destination: String,
    config: &CliConfig,
) -> Result<(), CliError> {
    debug!("Copying files from {} to {}", source, destination);

    // Create API client
    let api_client = create_authenticated_client(config).await?;

    // Parse source and destination to check if rental ID is provided
    let (source_rental, source_path) = split_remote_path(&source);
    let (dest_rental, dest_path) = split_remote_path(&destination);

    // Determine rental_id, handling interactive selection if needed
    let (rental_id, is_upload, local_path, remote_path) = match (source_rental, dest_rental) {
        (Some(rental), None) => {
            // Download: remote -> local
            (Some(rental), false, dest_path, source_path)
        }
        (None, Some(rental)) => {
            // Upload: local -> remote
            (Some(rental), true, source_path, dest_path)
        }
        (Some(_), Some(_)) => {
            return Err(CliError::Internal(eyre!(
                "Remote-to-remote copy not supported"
            )));
        }
        (None, None) => {
            // No rental ID provided - will use interactive selection
            // Determine direction based on source file existence
            let source_exists = std::path::Path::new(&source).exists();
            if source_exists {
                // Upload: local file exists, so source is local
                (None, true, source, destination)
            } else {
                // Download: assume source is remote path
                (None, false, destination, source)
            }
        }
    };

    // Resolve rental and fetch SSH credentials
    let RentalWithSsh {
        ssh_command: ssh_credentials,
        ssh_public_key,
        ..
    } = resolve_rental_with_ssh(rental_id.as_deref(), &api_client).await?;

    // Parse SSH credentials
    let (host, port, username) = parse_ssh_credentials(&ssh_credentials)?;
    let ssh_access = SshAccess {
        host,
        port,
        username,
    };

    // Find matching private key using the rental's stored public key
    let private_key_path = {
        let public_key = ssh_public_key.ok_or_else(|| {
            CliError::Internal(
                eyre!("No SSH public key available for this rental")
                    .suggestion("The required SSH key may not be on this machine")
                    .note("SSH access requires the original key used during rental creation"),
            )
        })?;

        crate::ssh::find_private_key_for_public_key(&public_key).map_err(CliError::Internal)?
    };

    debug!(
        "Using private key for file transfer: {}",
        private_key_path.display()
    );

    // Use SSH client for file transfer
    let ssh_client = SshClient::new(&config.ssh).map_err(|e| eyre!(e))?;

    if is_upload {
        ssh_client
            .upload_file(&ssh_access, &local_path, &remote_path, private_key_path)
            .await?;
        Ok(())
    } else {
        ssh_client
            .download_file(&ssh_access, &remote_path, &local_path, private_key_path)
            .await?;
        Ok(())
    }
}

// Helper functions

/// Poll rental status until it becomes active or timeout
async fn poll_rental_status(
    rental_id: &str,
    api_client: &basilica_sdk::BasilicaClient,
) -> Result<bool, CliError> {
    const INITIAL_INTERVAL: Duration = Duration::from_secs(2);
    const MAX_INTERVAL: Duration = Duration::from_secs(10);

    let spinner = create_spinner("Waiting for rental to become active...");
    let start_time = std::time::Instant::now();
    let mut interval = INITIAL_INTERVAL;
    let mut attempt = 0;

    loop {
        // Check if we've exceeded the maximum wait time
        if start_time.elapsed() > RENTAL_READY_TIMEOUT {
            complete_spinner_and_clear(spinner);
            println!("The rental is not yet up. Please wait for a while and SSH manually using `basilica ssh`.");
            return Ok(false);
        }

        attempt += 1;
        spinner.set_message(format!("Checking rental status... (attempt {})", attempt));

        // Check rental status
        match api_client.get_rental_status(rental_id).await {
            Ok(status) => {
                use basilica_sdk::types::RentalStatus;
                match status.status {
                    RentalStatus::Active => {
                        complete_spinner_and_clear(spinner);
                        return Ok(true);
                    }
                    RentalStatus::Failed => {
                        complete_spinner_error(spinner, "Rental failed to start");
                        return Err(CliError::Internal(eyre!(
                            "Rental failed: {}",
                            "Rental failed during initialization",
                        )));
                    }
                    RentalStatus::Terminated => {
                        complete_spinner_error(spinner, "Rental was terminated");
                        return Err(CliError::Internal(eyre!(
                            "Rental failed: {}",
                            "Rental was terminated before becoming active",
                        )));
                    }
                    RentalStatus::Pending => {
                        // Still pending, continue polling
                        spinner.set_message("Rental is pending...");
                    }
                }
            }
            Err(e) => {
                // Log the error but continue polling
                debug!("Error checking rental status: {}", e);
                spinner.set_message("Retrying status check...");
            }
        }

        // Wait before next check with exponential backoff
        tokio::time::sleep(interval).await;

        // Increase interval up to maximum
        interval = std::cmp::min(interval * 2, MAX_INTERVAL);
    }
}

/// Check if an IP address is private (RFC 1918)
fn is_private_ip(ip: &str) -> bool {
    if let Ok(addr) = ip.parse::<std::net::IpAddr>() {
        match addr {
            std::net::IpAddr::V4(ipv4) => {
                let octets = ipv4.octets();
                // 10.0.0.0/8
                octets[0] == 10 ||
                // 172.16.0.0/12
                (octets[0] == 172 && (16..=31).contains(&octets[1])) ||
                // 192.168.0.0/16
                (octets[0] == 192 && octets[1] == 168)
            }
            std::net::IpAddr::V6(_) => false, // IPv6 private ranges not relevant here
        }
    } else {
        false // If we can't parse it, assume it's not private
    }
}

/// Poll secure cloud rental status until it becomes running or timeout
async fn poll_secure_cloud_rental_status(
    rental_id: &str,
    api_client: &basilica_sdk::BasilicaClient,
) -> Result<Option<basilica_sdk::types::SecureCloudRentalListItem>, CliError> {
    const INITIAL_INTERVAL: Duration = Duration::from_secs(5);
    const MAX_INTERVAL: Duration = Duration::from_secs(15);

    let spinner = create_spinner("Waiting for rental to become active...");
    let start_time = std::time::Instant::now();
    let mut interval = INITIAL_INTERVAL;
    let mut attempt = 0;

    loop {
        // Check if we've exceeded the maximum wait time
        if start_time.elapsed() > RENTAL_READY_TIMEOUT {
            complete_spinner_and_clear(spinner);
            println!("The rental is not yet up. Please wait for a while and SSH manually using `basilica ssh`.");
            return Ok(None);
        }

        attempt += 1;
        spinner.set_message(format!("Checking rental status... (attempt {})", attempt));

        // Get rental from list
        match api_client.list_secure_cloud_rentals().await {
            Ok(response) => {
                // Find our rental
                if let Some(rental) = response.rentals.iter().find(|r| r.rental_id == rental_id) {
                    match rental.status.as_str() {
                        "running" => {
                            // Check if SSH command has a public IP
                            if let Some(ssh_cmd) = &rental.ssh_command {
                                // Parse the SSH command to extract the host
                                if let Ok((host, _, _)) = parse_ssh_credentials(ssh_cmd) {
                                    if is_private_ip(&host) {
                                        // Still has private IP, continue polling
                                        spinner.set_message(
                                            "Rental running but waiting for public IP...",
                                        );
                                        // Continue to next iteration
                                    } else {
                                        // Has public IP, return success
                                        complete_spinner_and_clear(spinner);
                                        return Ok(Some(rental.clone()));
                                    }
                                } else {
                                    // Can't parse SSH command, return anyway
                                    complete_spinner_and_clear(spinner);
                                    return Ok(Some(rental.clone()));
                                }
                            } else {
                                // No SSH command yet, continue polling
                                spinner.set_message("Rental running but waiting for SSH info...");
                                // Continue to next iteration
                            }
                        }
                        "error" => {
                            complete_spinner_error(spinner, "Rental failed to start");
                            return Err(CliError::Internal(eyre!(
                                "Rental failed during provisioning"
                            )));
                        }
                        "deleted" => {
                            complete_spinner_error(spinner, "Rental was deleted");
                            return Err(CliError::Internal(eyre!(
                                "Rental was deleted before becoming active"
                            )));
                        }
                        "pending" | "provisioning" => {
                            // Still starting, continue polling
                            spinner.set_message(format!("Rental is {}...", rental.status));
                        }
                        _ => {
                            // Unknown status, continue polling
                            spinner.set_message(format!("Rental status: {}...", rental.status));
                        }
                    }
                } else {
                    // Rental not found in list
                    complete_spinner_error(spinner, "Rental not found");
                    return Err(CliError::Internal(eyre!(
                        "Rental {} not found in rental list",
                        rental_id
                    )));
                }
            }
            Err(e) => {
                // Log the error but continue polling
                debug!("Error checking rental status: {}", e);
                spinner.set_message("Retrying status check...");
            }
        }

        // Wait before next check with exponential backoff
        tokio::time::sleep(interval).await;

        // Increase interval up to maximum
        interval = std::cmp::min(interval * 2, MAX_INTERVAL);
    }
}

/// Retry SSH connection with exponential backoff
///
/// SSH services may not be immediately available after a rental becomes active.
/// This function retries the connection for up to `max_wait` duration with exponential backoff.
///
/// Uses try_connect_silently() for non-interactive readiness probes (no passphrase prompts)
/// then interactive_session() once connected.
async fn retry_ssh_connection(
    ssh_client: &SshClient,
    ssh_access: &SshAccess,
    private_key_path: PathBuf,
    max_wait: Duration,
) -> Result<(), CliError> {
    const INITIAL_INTERVAL: Duration = Duration::from_secs(2);
    const MAX_INTERVAL: Duration = Duration::from_secs(10);

    let start_time = std::time::Instant::now();
    let mut interval = INITIAL_INTERVAL;
    let mut attempt = 0;

    loop {
        attempt += 1;

        // Check timeout before attempting
        if start_time.elapsed() >= max_wait {
            return Err(CliError::Internal(
                eyre!(
                    "SSH connection failed after {} attempts over {}s",
                    attempt - 1,
                    start_time.elapsed().as_secs(),
                )
                .suggestion("The SSH service may not be ready yet. Wait a minute and try 'basilica ssh <rental_id>'"),
            ));
        }

        // Show spinner during wait periods
        let spinner = create_spinner(&format!("Waiting for SSH... (attempt {})", attempt));

        // Brief delay between attempts (skip on first attempt)
        if attempt > 1 {
            tokio::time::sleep(interval).await;
            interval = std::cmp::min(interval * 2, MAX_INTERVAL);
        }

        // Clear spinner before SSH attempt (in case passphrase prompt appears)
        complete_spinner_and_clear(spinner);

        // Try silent connection (suppresses errors, allows passphrase)
        match ssh_client
            .try_connect_silently(ssh_access, private_key_path.clone())
            .await
        {
            Ok(crate::ssh::SshProbeStatus::Ready)
            | Ok(crate::ssh::SshProbeStatus::ReadyAuthRequired) => {
                // SSH is reachable; start interactive session (will prompt once if needed).
                return ssh_client
                    .interactive_session(ssh_access, private_key_path)
                    .await
                    .map_err(|e| CliError::Internal(eyre!("SSH session failed: {}", e)));
            }
            Ok(crate::ssh::SshProbeStatus::NotReady(reason)) => {
                debug!(
                    "SSH attempt {} not ready ({}s elapsed): {}. Retrying in {}s...",
                    attempt,
                    start_time.elapsed().as_secs(),
                    reason,
                    interval.as_secs()
                );
                // Continue to next iteration for retry
            }
            Err(e) => {
                debug!(
                    "SSH attempt {} failed ({}s elapsed): {}. Retrying in {}s...",
                    attempt,
                    start_time.elapsed().as_secs(),
                    e,
                    interval.as_secs()
                );
                // Continue to next iteration for retry
            }
        }
    }
}

/// Display SSH connection instructions after rental creation
fn display_ssh_connection_instructions(
    rental_id: &str,
    ssh_credentials: &str,
    private_key_path: &std::path::Path,
    message: &str,
) -> Result<(), CliError> {
    // Parse SSH credentials to get components
    let (host, port, username) = parse_ssh_credentials(ssh_credentials)?;

    println!();
    print_info(message);
    println!();

    // Option 1: Using basilica CLI (simplest)
    println!("  1. Using Basilica CLI:");
    println!(
        "     {}",
        console::style(format!("basilica ssh {}", rental_id))
            .cyan()
            .bold()
    );
    println!();

    // Option 2: Using standard SSH command
    println!("  2. Using standard SSH:");
    println!(
        "     {}",
        console::style(format!(
            "ssh -i {} -p {} {}@{}",
            compress_path(private_key_path),
            port,
            username,
            host
        ))
        .cyan()
        .bold()
    );

    Ok(())
}

/// Display SSH connection instructions for secure cloud rentals
fn display_secure_cloud_reconnection_instructions(
    rental_id: &str,
    ssh_command: &str,
    private_key_path: &std::path::Path,
    message: &str,
) -> Result<(), CliError> {
    // Parse SSH command to get components
    let (host, port, username) = parse_ssh_credentials(ssh_command)?;

    println!();
    print_info(message);
    println!();

    // Option 1: Using basilica CLI (simplest)
    println!("  1. Using Basilica CLI:");
    println!(
        "     {}",
        console::style(format!("basilica ssh {}", rental_id))
            .cyan()
            .bold()
    );
    println!();

    // Option 2: Using standard SSH command
    println!("  2. Using standard SSH:");
    println!(
        "     {}",
        console::style(format!(
            "ssh -i {} -p {} {}@{}",
            compress_path(private_key_path),
            port,
            username,
            host
        ))
        .cyan()
        .bold()
    );

    Ok(())
}

fn split_remote_path(path: &str) -> (Option<String>, String) {
    if let Some((rental_id, remote_path)) = path.split_once(':') {
        (Some(rental_id.to_string()), remote_path.to_string())
    } else {
        (None, path.to_string())
    }
}

fn display_rental_status_with_details(
    status: &basilica_sdk::types::RentalStatusWithSshResponse,
    private_key_path: Option<&std::path::Path>,
) {
    println!("Rental Status: {}", status.rental_id);
    println!("  Status: {:?}", status.status);
    println!("  Node: {}", status.node.id);
    println!(
        "  Created: {}",
        status.created_at.format("%Y-%m-%d %H:%M:%S UTC")
    );
    println!(
        "  Updated: {}",
        status.updated_at.format("%Y-%m-%d %H:%M:%S UTC")
    );

    // Display port mappings if available
    if let Some(ref port_mappings) = status.port_mappings {
        if !port_mappings.is_empty() {
            println!("\nPort Mappings (Host → Container):");
            let port_strings: Vec<String> = port_mappings
                .iter()
                .map(|p| format!("{}→{}", p.host_port, p.container_port))
                .collect();
            println!("  {}", port_strings.join(", "));
        }
    }

    // Display SSH connection instructions if available
    if let Some(ref ssh_credentials) = status.ssh_credentials {
        if let Ok((host, port, username)) = parse_ssh_credentials(ssh_credentials) {
            println!();
            print_info("SSH Connection:");
            println!();

            // Option 1: Using basilica CLI (simplest)
            println!("  1. Using Basilica CLI:");
            println!(
                "     {}",
                console::style(format!("basilica ssh {}", status.rental_id))
                    .cyan()
                    .bold()
            );
            println!();

            // Option 2: Using standard SSH command
            if let Some(key_path) = private_key_path {
                println!("  2. Using standard SSH:");
                println!(
                    "     {}",
                    console::style(format!(
                        "ssh -i {} -p {} {}@{}",
                        compress_path(key_path),
                        port,
                        username,
                        host
                    ))
                    .cyan()
                    .bold()
                );
            } else {
                println!("  2. Using standard SSH:");
                println!(
                    "     {}",
                    console::style(format!("ssh -p {} {}@{}", port, username, host))
                        .cyan()
                        .bold()
                );
                println!("     {}", style("(private key not found locally)").yellow());
            }
        }
    }
}

/// Display quick start commands after ps output
fn display_ps_quick_start_commands() {
    println!();
    println!("{}", style("Quick Commands:").cyan().bold());

    println!(
        "  {} {}",
        style("basilica ssh").yellow().bold(),
        style("- Connect to your rental").dim()
    );

    println!(
        "  {} {}",
        style("basilica exec").yellow().bold(),
        style("- Run commands on your rental").dim()
    );

    println!(
        "  {} {}",
        style("basilica logs").yellow().bold(),
        style("- Stream container logs").dim()
    );

    println!(
        "  {} {}",
        style("basilica status").yellow().bold(),
        style("- Check status of a specific rental").dim()
    );

    println!(
        "  {} {}",
        style("basilica down").yellow().bold(),
        style("- Stop a GPU rental").dim()
    );
}
