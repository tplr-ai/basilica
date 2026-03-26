//! Common helper functions for GPU rental operations

use crate::cli::handlers::region_mapping::{extract_country_code, region_matches_country};
use crate::error::CliError;
use crate::progress::{complete_spinner_and_clear, complete_spinner_error, create_spinner};
use basilica_common::types::ComputeCategory;
use basilica_common::types::{GpuCategory, GpuOffering};
use basilica_sdk::types::{ListAvailableNodesQuery, ListRentalsQuery, RentalState};
use basilica_sdk::{ApiError, BasilicaClient};
use basilica_validator::api::types::AvailableNode;
use color_eyre::eyre::{eyre, Result};
use color_eyre::Help;
use console::{style, Term};
use dialoguer::Select;
use rust_decimal::prelude::ToPrimitive;
use std::collections::HashMap;
use std::str::FromStr;
use std::time::Duration;
use tracing::warn;

/// Timeout for community cloud (validator) API requests.
/// The validator can be slower due to network hops through the Bittensor network.
pub const VALIDATOR_REQUEST_TIMEOUT: Duration = Duration::from_secs(5);

/// Create a default query for listing active rentals (no filters).
pub fn active_rentals_query() -> Option<ListRentalsQuery> {
    Some(ListRentalsQuery {
        status: Some(RentalState::Active),
        gpu_type: None,
        min_gpu_count: None,
    })
}

/// Wrap a community cloud (validator) request with timeout.
/// Returns ApiError::Timeout on timeout with a warning logged.
pub async fn with_validator_timeout<T>(
    future: impl std::future::Future<Output = Result<T, ApiError>>,
) -> Result<T, ApiError> {
    match tokio::time::timeout(VALIDATOR_REQUEST_TIMEOUT, future).await {
        Ok(result) => result,
        Err(_) => {
            warn!(
                "Validator request timed out after {} seconds",
                VALIDATOR_REQUEST_TIMEOUT.as_secs()
            );
            Err(ApiError::Timeout)
        }
    }
}

/// Print a bold section header for dual-cloud display.
/// Use `leading_newline: true` for the first section.
pub fn print_cloud_section_header(title: &str, leading_newline: bool) {
    if leading_newline {
        println!("\n{}", style(title).bold());
    } else {
        println!("{}", style(title).bold());
    }
}

/// Aggregated GPU category information for community cloud display
#[derive(Debug, Clone)]
pub struct GpuCategoryAggregation {
    /// GPU category name (e.g., "H100", "A100")
    pub gpu_category: String,
    /// Number of GPUs per node
    pub gpu_count: u32,
    /// Memory per GPU in GB
    pub min_memory_gb: u32,
    /// Number of available nodes with this configuration
    pub node_count: usize,
    /// Minimum hourly rate in cents across all nodes (per GPU)
    pub min_rate_cents: Option<i32>,
    /// Maximum hourly rate in cents across all nodes (per GPU)
    pub max_rate_cents: Option<i32>,
}

/// Aggregate community cloud nodes by GPU category and count.
///
/// Groups `AvailableNode` entries by `(gpu_category, gpu_count)` and computes
/// per-group statistics (node count, price range, memory).
pub fn aggregate_nodes_by_gpu_category(nodes: &[AvailableNode]) -> Vec<GpuCategoryAggregation> {
    let mut groups: HashMap<(String, u32), Vec<&AvailableNode>> = HashMap::new();

    for node in nodes {
        if node.node.gpu_specs.is_empty() {
            continue;
        }
        let gpu = &node.node.gpu_specs[0];
        let category = GpuCategory::from_str(&gpu.name)
            .map(|c| c.to_string())
            .unwrap_or_else(|_| gpu.name.clone());
        let gpu_count = node.node.gpu_specs.len() as u32;
        groups.entry((category, gpu_count)).or_default().push(node);
    }

    let mut aggregations: Vec<GpuCategoryAggregation> = groups
        .into_iter()
        .map(|((gpu_category, gpu_count), nodes)| {
            let min_memory_gb = nodes
                .iter()
                .filter_map(|n| n.node.gpu_specs.first())
                .map(|g| g.memory_gb)
                .min()
                .unwrap_or(0);

            let rates: Vec<i32> = nodes
                .iter()
                .filter_map(|n| n.node.hourly_rate_cents)
                .collect();

            let min_rate_cents = rates.iter().copied().min();
            let max_rate_cents = rates.iter().copied().max();

            GpuCategoryAggregation {
                gpu_category,
                gpu_count,
                min_memory_gb,
                node_count: nodes.len(),
                min_rate_cents,
                max_rate_cents,
            }
        })
        .collect();

    // Sort by category name, then gpu_count
    aggregations.sort_by(|a, b| {
        a.gpu_category
            .cmp(&b.gpu_category)
            .then(a.gpu_count.cmp(&b.gpu_count))
    });

    aggregations
}

/// Resolve target rental ID - if not provided, fetch active rentals and prompt for selection
///
/// # Arguments
/// * `target` - Optional rental ID provided by user
/// * `api_client` - Authenticated API client
/// * `require_ssh` - If true, only show rentals with SSH access
pub async fn resolve_target_rental(
    target: Option<String>,
    api_client: &BasilicaClient,
    require_ssh: bool,
) -> Result<String> {
    if let Some(t) = target {
        return Ok(t);
    }

    let spinner = if require_ssh {
        create_spinner("Fetching rentals with SSH access...")
    } else {
        create_spinner("Fetching active rentals...")
    };

    // Fetch active rentals
    let rentals_list = api_client
        .list_rentals(active_rentals_query())
        .await
        .inspect_err(|_| complete_spinner_error(spinner.clone(), "Failed to load rentals"))?;

    complete_spinner_and_clear(spinner);

    // Filter for SSH-enabled rentals if required
    let eligible_rentals = if require_ssh {
        rentals_list
            .rentals
            .into_iter()
            .filter(|r| r.has_ssh)
            .collect()
    } else {
        rentals_list.rentals
    };

    if eligible_rentals.is_empty() {
        return if require_ssh {
            Err(
                eyre!("No rentals with SSH access found. SSH credentials are only available for rentals created in this session")
            )
        } else {
            Err(eyre!("No active rentals found"))
        };
    }

    // Use interactive selector to choose a rental (use compact mode for better readability)
    let selector = crate::interactive::InteractiveSelector::new();
    Ok(selector.select_rental(&eligible_rentals, false)?)
}

/// Unified rental item for selection across both compute types
#[derive(Clone)]
struct UnifiedRentalItem {
    rental_id: String,
    compute_type: ComputeCategory,
    provider_or_node: String,
    location: String,
    gpu_info: String,
    status: String,
    created_at: String,
    ip_address: Option<String>,
}

/// Resolve target rental ID with unified selection across compute types
///
/// # Arguments
/// * `target` - Optional rental ID provided by user
/// * `compute_filter` - Optional compute category to filter rentals
/// * `api_client` - Authenticated API client
/// * `exclude_vip` - If true, VIP rentals will be excluded from selection (for commands like `down`)
///
/// # Returns
/// Returns (rental_id, compute_category) tuple
pub async fn resolve_target_rental_unified(
    target: Option<String>,
    compute_filter: Option<ComputeCategory>,
    api_client: &BasilicaClient,
    exclude_vip: bool,
) -> Result<(String, ComputeCategory)> {
    // If target provided, determine type based on filter or default
    if let Some(t) = target {
        let compute_type = compute_filter.unwrap_or(ComputeCategory::SecureCloud);
        return Ok((t, compute_type));
    }

    let spinner = create_spinner("Fetching active rentals...");

    // Fetch rentals based on filter
    let (community_rentals, secure_rentals, cpu_rentals) = match compute_filter {
        Some(ComputeCategory::CommunityCloud) => {
            // Fetch only community cloud
            let rentals = api_client
                .list_rentals(active_rentals_query())
                .await
                .inspect_err(|_| {
                    complete_spinner_error(
                        spinner.clone(),
                        "Failed to load community cloud rentals",
                    )
                })?;
            (Some(rentals), None, None)
        }
        Some(ComputeCategory::SecureCloud) => {
            // Fetch secure cloud GPU + CPU rentals
            let (gpu_result, cpu_result) = tokio::join!(
                api_client.list_secure_cloud_rentals(),
                api_client.list_cpu_rentals()
            );

            let rentals = gpu_result.inspect_err(|_| {
                complete_spinner_error(spinner.clone(), "Failed to load secure cloud rentals")
            })?;

            let cpu_rentals = match cpu_result {
                Ok(list) => Some(list),
                Err(e) => {
                    warn!("Failed to load CPU-only rentals: {}", e);
                    None
                }
            };

            (None, Some(rentals), cpu_rentals)
        }
        None => {
            // Fetch both types in parallel with timeout for community cloud
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

    // Build unified list
    let mut unified_items: Vec<UnifiedRentalItem> = Vec::new();

    // Add community cloud rentals
    if let Some(community) = community_rentals {
        for rental in community.rentals.iter() {
            let gpu_info = if rental.gpu_specs.is_empty() {
                "Unknown GPU".to_string()
            } else if rental.gpu_specs.len() == 1 {
                rental.gpu_specs[0].name.clone()
            } else {
                format!("{}x {}", rental.gpu_specs.len(), rental.gpu_specs[0].name)
            };

            // Extract country code from location
            let location_code = rental
                .location
                .as_ref()
                .and_then(|loc| extract_country_code(loc))
                .unwrap_or("--")
                .to_string();

            unified_items.push(UnifiedRentalItem {
                rental_id: rental.rental_id.clone(),
                compute_type: ComputeCategory::CommunityCloud,
                provider_or_node: rental.node_id.clone(),
                location: location_code,
                gpu_info,
                status: format!("{:?}", rental.state),
                created_at: rental.created_at.clone(),
                ip_address: None, // IP not available in community cloud list response
            });
        }
    }

    // Add secure cloud GPU rentals (only active ones - where stopped_at is None)
    if let Some(secure) = secure_rentals {
        for rental in secure.rentals.iter() {
            // Skip stopped rentals
            if rental.stopped_at.is_some() {
                continue;
            }

            // Skip VIP rentals if exclude_vip is true (e.g., for `down` command)
            if exclude_vip && rental.is_vip {
                continue;
            }

            let base_gpu = if rental.gpu_count > 1 {
                format!("{}x {}", rental.gpu_count, rental.gpu_type.to_uppercase())
            } else {
                rental.gpu_type.to_uppercase()
            };
            let gpu_info = if rental.is_spot {
                format!("{} (Spot)", base_gpu)
            } else {
                base_gpu
            };

            // Extract country code from location_code
            let location_code = rental
                .location_code
                .as_ref()
                .and_then(|loc| extract_country_code(loc))
                .unwrap_or("--")
                .to_string();

            unified_items.push(UnifiedRentalItem {
                rental_id: rental.rental_id.clone(),
                compute_type: ComputeCategory::SecureCloud,
                provider_or_node: rental.provider.clone(),
                location: location_code,
                gpu_info,
                status: rental.status.clone(),
                created_at: rental.created_at.to_rfc3339(),
                ip_address: rental.ip_address.clone(),
            });
        }
    }

    // Add secure cloud CPU rentals (only active ones - where stopped_at is None)
    if let Some(cpu) = cpu_rentals {
        for rental in cpu.rentals.iter() {
            // Skip stopped rentals
            if rental.stopped_at.is_some() {
                continue;
            }

            // Skip GPU rentals if any are returned (CPU endpoint should be CPU-only)
            if rental.gpu_count > 0 {
                continue;
            }

            // Skip VIP rentals if exclude_vip is true (e.g., for `down` command)
            if exclude_vip && rental.is_vip {
                continue;
            }

            let cpu_info = match (rental.vcpu_count, rental.system_memory_gb) {
                (Some(vcpu), Some(mem)) => format!("{} vCPU / {}GB", vcpu, mem),
                (Some(vcpu), None) => format!("{} vCPU", vcpu),
                (None, Some(mem)) => format!("{}GB RAM", mem),
                (None, None) => "CPU-only".to_string(),
            };

            // Extract country code from location_code
            let location_code = rental
                .location_code
                .as_ref()
                .and_then(|loc| extract_country_code(loc))
                .unwrap_or("--")
                .to_string();

            unified_items.push(UnifiedRentalItem {
                rental_id: rental.rental_id.clone(),
                compute_type: ComputeCategory::SecureCloud,
                provider_or_node: rental.provider.clone(),
                location: location_code,
                gpu_info: cpu_info,
                status: rental.status.clone(),
                created_at: rental.created_at.to_rfc3339(),
                ip_address: rental.ip_address.clone(),
            });
        }
    }

    if unified_items.is_empty() {
        return Err(eyre!("No active rentals found"));
    }

    // Helper to truncate strings for column width
    fn truncate(s: &str, max_len: usize) -> String {
        let char_count = s.chars().count();
        if char_count <= max_len {
            s.to_string()
        } else {
            let truncated: String = s.chars().take(max_len - 1).collect();
            format!("{}…", truncated)
        }
    }

    // Format items for selection
    let items: Vec<String> = unified_items
        .iter()
        .map(|item| {
            let type_label = match item.compute_type {
                ComputeCategory::CommunityCloud => "Bourse   ",
                ComputeCategory::SecureCloud => "Citadel  ",
            };

            let ip_display = item
                .ip_address
                .as_ref()
                .map(|ip| truncate(ip, 15))
                .unwrap_or_else(|| "--".to_string());

            format!(
                "{} | {:<15} | {:<15} | {:<4} | {:<30} | {:<12} | {}",
                style(type_label).cyan(),
                truncate(&item.provider_or_node, 15),
                ip_display,
                item.location,
                truncate(&item.gpu_info, 30),
                item.status,
                truncate(&item.created_at, 19)
            )
        })
        .collect();

    // Show header hint
    println!(
        "{}",
        style("  Type      | Provider        | IP              | Loc  | GPU                            | Status       | Created").dim()
    );

    // Use dialoguer to select
    let theme = dialoguer::theme::ColorfulTheme::default();
    let selection = Select::with_theme(&theme)
        .with_prompt("Select rental to stop")
        .items(&items)
        .default(0)
        .interact_opt()
        .map_err(|e| eyre!("Selection failed: {}", e))?;

    let selection = match selection {
        Some(s) => s,
        None => return Err(eyre!("Selection cancelled")),
    };

    // Clear the header and selection prompt lines
    let term = Term::stdout();
    let _ = term.clear_last_lines(2);

    let selected = &unified_items[selection];
    Ok((selected.rental_id.clone(), selected.compute_type))
}

/// Community cloud GPU selection for rental
pub struct CommunityCloudSelection {
    /// GPU category (e.g., "H100", "A100")
    pub gpu_category: String,
    /// Number of GPUs
    pub gpu_count: u32,
    /// Min GPU memory in GB (optional)
    pub min_memory_gb: Option<u32>,
    /// Selected node's per-GPU hourly rate in cents, if available
    pub derived_max_hourly_rate_cents: Option<u32>,
}

/// Represents a selected offering from either cloud type
pub enum SelectedOffering {
    /// Secure cloud GPU offering (from aggregator)
    SecureCloud(GpuOffering),
    /// Community cloud offering (GPU category and count for bid-based selection)
    CommunityCloud(CommunityCloudSelection),
    /// Secure cloud CPU-only offering (no GPU)
    CpuOnly(basilica_sdk::types::CpuOffering),
}

/// Offering type for display
#[derive(Clone, PartialEq)]
enum OfferingType {
    SecureGpu,
    SecureCpu,
    Community,
}

/// Internal struct for unified offering display
#[derive(Clone)]
struct UnifiedOfferingItem {
    offering_type: OfferingType,
    display_gpu: String,
    display_provider: String,
    display_country: String,
    display_memory: String,
    display_price: String,
    // Original data for creating the offering
    secure_offering: Option<GpuOffering>,
    community_nodes: Option<Vec<AvailableNode>>,
    cpu_offering: Option<basilica_sdk::types::CpuOffering>,
}

/// Resolve GPU offering with unified selection across compute types
///
/// Fetches available offerings from one or both clouds and presents a unified
/// selector for the user to choose from.
///
/// # Arguments
/// * `api_client` - Authenticated API client
/// * `gpu_filter` - Optional GPU type filter (e.g., "h100", "a100")
/// * `gpu_count_filter` - Optional GPU count filter
/// * `country_filter` - Optional country filter for location-based filtering
/// * `min_gpu_memory_filter` - Optional minimum GPU memory filter
/// * `cloud_filter` - Optional cloud filter to restrict to a specific cloud type
///
/// # Returns
/// Returns `SelectedOffering` enum containing either secure or community cloud data
/// Flavour preference filters for secure cloud offerings
#[derive(Default)]
pub struct FlavourFilters {
    pub interconnect: Option<String>,
    pub region: Option<String>,
    pub spot: bool,
    pub exclude_spot: bool,
}

pub async fn resolve_offering_unified(
    api_client: &BasilicaClient,
    gpu_filter: Option<&str>,
    gpu_count_filter: Option<u32>,
    country_filter: Option<&str>,
    min_gpu_memory_filter: Option<u32>,
    cloud_filter: Option<ComputeCategory>,
    flavour: &FlavourFilters,
) -> Result<SelectedOffering> {
    let spinner_msg = match cloud_filter {
        Some(ComputeCategory::SecureCloud) => "Fetching available GPUs from The Citadel...",
        Some(ComputeCategory::CommunityCloud) => "Fetching available GPUs from The Bourse...",
        None => "Fetching available GPUs...",
    };
    let spinner = create_spinner(spinner_msg);

    // Build community query for reuse
    let community_query = ListAvailableNodesQuery {
        available: Some(true),
        min_gpu_memory: min_gpu_memory_filter,
        gpu_type: gpu_filter.map(|s| s.to_string()),
        min_gpu_count: gpu_count_filter,
        location: country_filter.map(|c| basilica_common::LocationProfile {
            city: None,
            region: None,
            country: Some(c.to_string()),
        }),
    };

    // Build GPU price query with flavour filters
    let gpu_price_query = basilica_sdk::types::GpuPriceQuery {
        interconnect: flavour.interconnect.clone(),
        region: flavour.region.clone(),
        spot_only: if flavour.spot { Some(true) } else { None },
        exclude_spot: if flavour.exclude_spot {
            Some(true)
        } else {
            None
        },
    };

    // Conditionally fetch based on cloud filter (include CPU offerings for secure cloud)
    let (secure_result, community_result, cpu_result) = match cloud_filter {
        Some(ComputeCategory::SecureCloud) => {
            // Fetch secure cloud GPU and CPU offerings
            let (secure, cpu) = tokio::join!(
                api_client.list_secure_cloud_gpus_filtered(&gpu_price_query),
                api_client.list_cpu_offerings()
            );
            (secure, Err(ApiError::Timeout), cpu) // Dummy error for community - will be ignored
        }
        Some(ComputeCategory::CommunityCloud) => {
            // Only fetch community cloud (no CPU offerings available)
            let community = api_client.list_available_nodes(Some(community_query)).await;
            (Err(ApiError::Timeout), community, Err(ApiError::Timeout))
        }
        None => {
            // Fetch all in parallel
            let community_future = api_client.list_available_nodes(Some(community_query));
            let (secure, community, cpu) = tokio::join!(
                api_client.list_secure_cloud_gpus_filtered(&gpu_price_query),
                with_validator_timeout(community_future),
                api_client.list_cpu_offerings()
            );
            (secure, community, cpu)
        }
    };

    complete_spinner_and_clear(spinner);

    // Build unified list
    let mut unified_items: Vec<UnifiedOfferingItem> = Vec::new();

    // Add secure cloud GPU offerings
    if let Ok(offerings) = secure_result {
        for offering in offerings {
            // Apply GPU type filter if specified
            if let Some(filter) = gpu_filter {
                if !offering
                    .gpu_type
                    .as_str()
                    .to_uppercase()
                    .contains(&filter.to_uppercase())
                {
                    continue;
                }
            }

            // Apply GPU count filter if specified
            if let Some(count) = gpu_count_filter {
                if offering.gpu_count != count {
                    continue;
                }
            }

            // Apply memory filter if specified
            if let Some(min_mem) = min_gpu_memory_filter {
                if let Some(mem_per_gpu) = offering.gpu_memory_gb_per_gpu {
                    let total_memory = mem_per_gpu * offering.gpu_count;
                    if total_memory < min_mem {
                        continue;
                    }
                }
            }

            // Apply country filter if specified
            if let Some(country) = country_filter {
                if !region_matches_country(&offering.region, country) {
                    continue;
                }
            }

            // Calculate total instance price (API already includes markup)
            let price_per_gpu = offering.hourly_rate_per_gpu.to_f64().unwrap_or(0.0);
            let total_price = price_per_gpu * (offering.gpu_count as f64);

            let memory_str = if let Some(mem_per_gpu) = offering.gpu_memory_gb_per_gpu {
                format!("{}GB", mem_per_gpu * offering.gpu_count)
            } else {
                "N/A".to_string()
            };

            let base_gpu = if let Some(ref interconnect) = offering.interconnect {
                format!(
                    "{}x {} ({})",
                    offering.gpu_count,
                    offering.gpu_type.as_str().to_uppercase(),
                    interconnect
                )
            } else {
                format!(
                    "{}x {}",
                    offering.gpu_count,
                    offering.gpu_type.as_str().to_uppercase()
                )
            };

            let display_gpu = if offering.is_spot {
                format!("{} (Spot)", base_gpu)
            } else {
                base_gpu
            };

            unified_items.push(UnifiedOfferingItem {
                offering_type: OfferingType::SecureGpu,
                display_gpu,
                display_provider: format!("{}", offering.provider),
                display_country: extract_country_code(&offering.region)
                    .unwrap_or("--")
                    .to_string(),
                display_memory: memory_str,
                display_price: format!("${:.2}/hr", total_price),
                secure_offering: Some(offering),
                community_nodes: None,
                cpu_offering: None,
            });
        }
    }

    // Add CPU-only offerings (only if no GPU filter is specified)
    if gpu_filter.is_none() && gpu_count_filter.is_none() {
        if let Ok(cpu_offerings) = cpu_result {
            for offering in cpu_offerings {
                // Apply country filter if specified
                if let Some(country) = country_filter {
                    if !region_matches_country(&offering.region, country) {
                        continue;
                    }
                }

                // Apply memory filter (system memory for CPU offerings)
                if let Some(min_mem) = min_gpu_memory_filter {
                    if offering.system_memory_gb < min_mem {
                        continue;
                    }
                }

                // Parse hourly rate with proper error handling
                let hourly_rate: f64 = match offering.hourly_rate.parse() {
                    Ok(rate) => rate,
                    Err(_) => {
                        warn!(
                            "Invalid hourly_rate '{}' for CPU offering {}, skipping",
                            offering.hourly_rate, offering.id
                        );
                        continue;
                    }
                };

                unified_items.push(UnifiedOfferingItem {
                    offering_type: OfferingType::SecureCpu,
                    display_gpu: format!("{} vCPU", offering.vcpu_count),
                    display_provider: offering.provider.to_string(),
                    display_country: extract_country_code(&offering.region)
                        .unwrap_or("--")
                        .to_string(),
                    display_memory: format!("{}GB RAM", offering.system_memory_gb),
                    display_price: format!("${:.2}/hr", hourly_rate),
                    secure_offering: None,
                    community_nodes: None,
                    cpu_offering: Some(offering),
                });
            }
        }
    }

    // Add community cloud offerings (aggregated by GPU category)
    if let Ok(response) = community_result {
        let filtered_nodes: Vec<AvailableNode> = if let Some(count) = gpu_count_filter {
            response
                .available_nodes
                .into_iter()
                .filter(|n| n.node.gpu_specs.len() as u32 == count)
                .collect()
        } else {
            response.available_nodes
        };

        let mut groups: HashMap<(String, u32), Vec<AvailableNode>> = HashMap::new();
        for node in filtered_nodes {
            if node.node.gpu_specs.is_empty() {
                continue;
            }
            let gpu = &node.node.gpu_specs[0];
            let category = GpuCategory::from_str(&gpu.name)
                .map(|c| c.to_string())
                .unwrap_or_else(|_| gpu.name.clone());
            let gpu_count = node.node.gpu_specs.len() as u32;
            groups.entry((category, gpu_count)).or_default().push(node);
        }

        for ((category, gpu_count), nodes) in &groups {
            let min_memory_gb = nodes
                .iter()
                .filter_map(|n| n.node.gpu_specs.first())
                .map(|g| g.memory_gb)
                .min()
                .unwrap_or(0);

            let rates: Vec<i32> = nodes
                .iter()
                .filter_map(|n| n.node.hourly_rate_cents)
                .collect();
            let min_rate = rates.iter().copied().min();
            let max_rate = rates.iter().copied().max();

            let gpu_info = if *gpu_count > 1 {
                format!("{}x {} ({} available)", gpu_count, category, nodes.len())
            } else {
                format!("{} ({} available)", category, nodes.len())
            };

            let memory_str = format!("{}GB", min_memory_gb);

            let multiplier = *gpu_count as f64;
            let price_str = match (min_rate, max_rate) {
                (Some(min), Some(max)) if min == max => {
                    format!("${:.2}/hr", min as f64 / 100.0 * multiplier)
                }
                (Some(min), Some(max)) => {
                    format!(
                        "${:.2}-{:.2}/hr",
                        min as f64 / 100.0 * multiplier,
                        max as f64 / 100.0 * multiplier
                    )
                }
                _ => "Market".to_string(),
            };

            unified_items.push(UnifiedOfferingItem {
                offering_type: OfferingType::Community,
                display_gpu: gpu_info,
                display_provider: "--".to_string(),
                display_country: "--".to_string(),
                display_memory: memory_str,
                display_price: price_str,
                secure_offering: None,
                community_nodes: Some(nodes.clone()),
                cpu_offering: None,
            });
        }
    }

    if unified_items.is_empty() {
        return Err(eyre!(
            "No offerings available. Try different filters or check back later."
        ));
    }

    // Helper to truncate strings to fit column width (unicode-safe)
    fn truncate(s: &str, max_len: usize) -> String {
        let char_count = s.chars().count();
        if char_count <= max_len {
            s.to_string()
        } else {
            let truncated: String = s.chars().take(max_len - 1).collect();
            format!("{}…", truncated)
        }
    }

    // Format items for selection
    let items: Vec<String> = unified_items
        .iter()
        .map(|item| {
            let type_label = match item.offering_type {
                OfferingType::Community => "Bourse   ",
                OfferingType::SecureGpu => "Citadel  ",
                OfferingType::SecureCpu => "Citadel  ",
            };

            format!(
                "{} │ {:<25} │ {:<15} │ {:<4} │ {:<8} │ {}",
                style(type_label).cyan(),
                truncate(&item.display_gpu, 25),
                truncate(&item.display_provider, 15),
                item.display_country,
                item.display_memory,
                style(&item.display_price).green()
            )
        })
        .collect();

    // Show header hint
    println!(
        "{}",
        style(
            "  Type      │ GPU/CPU                   │ Provider        │ Loc  │ Memory   │ Price"
        )
        .dim()
    );

    // Use dialoguer to select
    let theme = dialoguer::theme::ColorfulTheme::default();
    let selection = Select::with_theme(&theme)
        .with_prompt("Select offering")
        .items(&items)
        .default(0)
        .interact_opt()
        .map_err(|e| eyre!("Selection failed: {}", e))?;

    let selection = match selection {
        Some(s) => s,
        None => return Err(eyre!("Selection cancelled")),
    };

    // Clear the header and selection prompt lines
    let term = Term::stdout();
    let _ = term.clear_last_lines(2);

    let selected = &unified_items[selection];

    // Return appropriate offering type
    match selected.offering_type {
        OfferingType::SecureGpu => {
            let offering = selected
                .secure_offering
                .clone()
                .ok_or_else(|| eyre!("Internal error: secure cloud offering data missing"))?;
            Ok(SelectedOffering::SecureCloud(offering))
        }
        OfferingType::SecureCpu => {
            let offering = selected
                .cpu_offering
                .clone()
                .ok_or_else(|| eyre!("Internal error: CPU offering data missing"))?;
            Ok(SelectedOffering::CpuOnly(offering))
        }
        OfferingType::Community => {
            let nodes = selected
                .community_nodes
                .clone()
                .ok_or_else(|| eyre!("Internal error: community cloud node data missing"))?;

            let first_node = nodes
                .first()
                .ok_or_else(|| eyre!("Internal error: community cloud group is empty"))?;

            // Extract GPU category and count from the group
            let gpu_category = first_node
                .node
                .gpu_specs
                .first()
                .map(|gpu| {
                    GpuCategory::from_str(&gpu.name)
                        .map(|c| c.to_string())
                        .unwrap_or_else(|_| gpu.name.clone())
                })
                .ok_or_else(|| eyre!("Selected group has no GPU specs"))?;

            let gpu_count = first_node.node.gpu_specs.len() as u32;
            let min_memory_gb = nodes
                .iter()
                .filter_map(|n| n.node.gpu_specs.first())
                .map(|g| g.memory_gb)
                .min();

            // Use max hourly rate across all nodes in the group as the bid cap
            let derived_max_hourly_rate_cents = nodes
                .iter()
                .filter_map(|n| n.node.hourly_rate_cents)
                .filter_map(|rate_cents| u32::try_from(rate_cents).ok())
                .max();

            Ok(SelectedOffering::CommunityCloud(CommunityCloudSelection {
                gpu_category,
                gpu_count,
                min_memory_gb,
                derived_max_hourly_rate_cents,
            }))
        }
    }
}

/// Resolve a rental ID to its compute category by checking both cloud types.
///
/// Fetches rentals from both community and secure clouds (in parallel with timeout),
/// and determines which cloud the rental belongs to.
pub async fn resolve_rental_by_id(
    target_id: &str,
    api_client: &BasilicaClient,
) -> Result<ComputeCategory, CliError> {
    let spinner = create_spinner("Looking up rental...");

    let community_future = api_client.list_rentals(active_rentals_query());

    let (community_result, secure_result, cpu_result) = tokio::join!(
        with_validator_timeout(community_future),
        api_client.list_secure_cloud_rentals(),
        api_client.list_cpu_rentals()
    );

    complete_spinner_and_clear(spinner);

    // Check community cloud first
    if let Ok(community) = community_result {
        if community.rentals.iter().any(|r| r.rental_id == target_id) {
            return Ok(ComputeCategory::CommunityCloud);
        }
    }

    // Check secure cloud CPU rentals
    if let Ok(cpu) = &cpu_result {
        if cpu.rentals.iter().any(|r| r.rental_id == target_id) {
            return Ok(ComputeCategory::SecureCloud);
        }
    }

    // Check secure cloud GPU rentals
    if let Ok(secure) = &secure_result {
        if secure.rentals.iter().any(|r| r.rental_id == target_id) {
            return Ok(ComputeCategory::SecureCloud);
        }
    }

    // Not found in either - provide helpful error
    Err(CliError::Internal(
        eyre!("Rental '{}' not found", target_id)
            .suggestion("Try 'basilica ps' to see your active rentals")
            .note("The rental may have expired or been terminated"),
    ))
}

/// Result of resolving a rental with SSH access
pub struct RentalWithSsh {
    pub rental_id: String,
    pub compute_type: ComputeCategory,
    pub ssh_command: String,
    pub ssh_public_key: Option<String>,
}

/// Get the user's registered SSH key and find the corresponding private key path.
///
/// This is a common operation needed when displaying SSH connection instructions.
/// Returns an error if no SSH key is registered or the private key cannot be found locally.
pub async fn get_ssh_private_key_path(api_client: &BasilicaClient) -> Result<std::path::PathBuf> {
    let ssh_key = api_client
        .get_ssh_key()
        .await
        .map_err(|e| eyre!(e))?
        .ok_or_else(|| {
            eyre!("No SSH key registered with Basilica")
                .suggestion("Run 'basilica ssh-keys add' to register your SSH key")
        })?;

    crate::ssh::find_private_key_for_public_key(&ssh_key.public_key)
}

/// Resolve a rental ID to its compute category and fetch SSH credentials.
///
/// When `target_id` is provided, locates the rental and fetches SSH credentials.
/// When `target_id` is None, uses interactive selector then fetches SSH credentials.
pub async fn resolve_rental_with_ssh(
    target_id: Option<&str>,
    api_client: &BasilicaClient,
) -> Result<RentalWithSsh, CliError> {
    if let Some(target_id) = target_id {
        // Rental ID provided - find it and get SSH credentials
        let spinner = create_spinner("Looking up rental...");

        let community_future = api_client.list_rentals(active_rentals_query());

        let (community_result, secure_result, cpu_result) = tokio::join!(
            with_validator_timeout(community_future),
            api_client.list_secure_cloud_rentals(),
            api_client.list_cpu_rentals()
        );

        complete_spinner_and_clear(spinner);

        // Check community cloud first
        if let Ok(ref community) = community_result {
            if let Some(rental) = community.rentals.iter().find(|r| r.rental_id == target_id) {
                let (ssh_command, _) = fetch_community_ssh_info(target_id, api_client).await?;
                return Ok(RentalWithSsh {
                    rental_id: target_id.to_string(),
                    compute_type: ComputeCategory::CommunityCloud,
                    ssh_command,
                    ssh_public_key: rental.ssh_public_key.clone(),
                });
            }
        }

        // Check secure cloud CPU rentals
        if let Ok(cpu) = cpu_result {
            if let Some(rental) = cpu.rentals.iter().find(|r| r.rental_id == target_id) {
                let (ssh_command, ssh_public_key) =
                    secure_rental_ssh_info(target_id, rental, api_client).await?;
                return Ok(RentalWithSsh {
                    rental_id: target_id.to_string(),
                    compute_type: ComputeCategory::SecureCloud,
                    ssh_command,
                    ssh_public_key,
                });
            }
        }

        // Check secure cloud GPU rentals
        if let Ok(secure) = secure_result {
            if let Some(rental) = secure.rentals.iter().find(|r| r.rental_id == target_id) {
                let (ssh_command, ssh_public_key) =
                    secure_rental_ssh_info(target_id, rental, api_client).await?;
                return Ok(RentalWithSsh {
                    rental_id: target_id.to_string(),
                    compute_type: ComputeCategory::SecureCloud,
                    ssh_command,
                    ssh_public_key,
                });
            }
        }

        Err(CliError::Internal(
            eyre!("Rental '{}' not found", target_id)
                .suggestion("Try 'basilica ps' to see your active rentals"),
        ))
    } else {
        // No rental ID - use interactive selector
        // exclude_vip=false: VIP rentals can be accessed via SSH
        let (rental_id, compute_type) =
            resolve_target_rental_unified(None, None, api_client, false).await?;

        let (ssh_command, ssh_public_key) = match compute_type {
            ComputeCategory::CommunityCloud => {
                fetch_community_ssh_info(&rental_id, api_client).await?
            }
            ComputeCategory::SecureCloud => fetch_secure_ssh_info(&rental_id, api_client).await?,
        };

        Ok(RentalWithSsh {
            rental_id,
            compute_type,
            ssh_command,
            ssh_public_key,
        })
    }
}

/// Fetch SSH info (credentials and public key) for a community cloud rental
async fn fetch_community_ssh_info(
    rental_id: &str,
    api_client: &BasilicaClient,
) -> Result<(String, Option<String>), CliError> {
    let rental_status = api_client
        .get_rental_status(rental_id)
        .await
        .map_err(|e| CliError::Internal(eyre!(e)))?;

    let ssh_credentials = rental_status.ssh_credentials.ok_or_else(|| {
        CliError::Internal(
            eyre!("SSH credentials not available")
                .wrap_err(format!(
                    "The rental '{}' does not have SSH access",
                    rental_id
                ))
                .suggestion("Create a new rental to enable SSH access"),
        )
    })?;

    Ok((ssh_credentials, rental_status.ssh_public_key))
}

/// Fetch SSH info (command and public key) for a secure cloud rental
async fn fetch_secure_ssh_info(
    rental_id: &str,
    api_client: &BasilicaClient,
) -> Result<(String, Option<String>), CliError> {
    let (cpu_result, secure_result) = tokio::join!(
        api_client.list_cpu_rentals(),
        api_client.list_secure_cloud_rentals()
    );

    let mut cpu_error: Option<ApiError> = None;
    let mut secure_error: Option<ApiError> = None;

    match cpu_result {
        Ok(cpu_rentals) => {
            if let Some(rental) = cpu_rentals
                .rentals
                .iter()
                .find(|r| r.rental_id == rental_id)
            {
                return secure_rental_ssh_info(rental_id, rental, api_client).await;
            }
        }
        Err(err) => {
            cpu_error = Some(err);
        }
    }

    match secure_result {
        Ok(secure_rentals) => {
            if let Some(rental) = secure_rentals
                .rentals
                .iter()
                .find(|r| r.rental_id == rental_id)
            {
                return secure_rental_ssh_info(rental_id, rental, api_client).await;
            }
        }
        Err(err) => {
            secure_error = Some(err);
        }
    }

    if cpu_error.is_some() || secure_error.is_some() {
        let err = secure_error.or(cpu_error).unwrap();
        return Err(CliError::Internal(eyre!(err)));
    }

    Err(CliError::Internal(eyre!(
        "Rental '{}' not found",
        rental_id
    )))
}

/// Common SSH info extraction for secure cloud GPU/CPU rentals
async fn secure_rental_ssh_info(
    rental_id: &str,
    rental: &basilica_sdk::types::SecureCloudRentalListItem,
    api_client: &BasilicaClient,
) -> Result<(String, Option<String>), CliError> {
    let ssh_command = rental.ssh_command.clone().ok_or_else(|| {
        CliError::Internal(
            eyre!("SSH command not available")
                .wrap_err(format!(
                    "The rental '{}' does not have SSH access configured",
                    rental_id
                ))
                .note("The rental may still be provisioning or SSH may not be enabled"),
        )
    })?;

    // For VIP rentals without ssh_public_key, fall back to user's registered SSH key
    let ssh_public_key = match &rental.ssh_public_key {
        Some(key) => Some(key.clone()),
        None if rental.is_vip => {
            // VIP rentals: fetch user's registered SSH key
            match api_client.get_user_ssh_key().await {
                Ok(Some(user_key)) => Some(user_key.public_key),
                Ok(None) => None,
                Err(e) => {
                    tracing::warn!("Failed to fetch user SSH key for VIP rental: {}", e);
                    None
                }
            }
        }
        None => None, // Non-VIP rentals: keep None (original behavior)
    };

    Ok((ssh_command, ssh_public_key))
}
