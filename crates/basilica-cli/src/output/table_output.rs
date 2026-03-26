//! Table formatting for CLI output

use super::format_usd;
use crate::error::Result;
use basilica_common::types::GpuOffering;
use basilica_common::{types::GpuCategory, LocationProfile};
use basilica_sdk::{
    types::{
        ApiKeyInfo, ApiRentalListItem, GpuSpec, HistoricalRentalItem, ListDepositsResponse,
        RentalUsageResponse, UsageHistoryResponse, VolumeResponse,
    },
    AvailableNode,
};
use chrono::{DateTime, Local};
use console::style;
use rust_decimal::Decimal;
use std::{collections::HashMap, str::FromStr};
use tabled::{builder::Builder, settings::Style, Table, Tabled};

/// Format RFC3339 timestamp to YY-MM-DD HH:MM:SS format
pub fn format_timestamp(timestamp: &str) -> String {
    DateTime::parse_from_rfc3339(timestamp)
        .ok()
        .map(|dt| {
            let local_dt = dt.with_timezone(&Local);
            local_dt.format("%y-%m-%d %H:%M:%S").to_string()
        })
        .unwrap_or_else(|| timestamp.to_string())
}

fn format_duration(seconds: i64) -> String {
    let hours = seconds / 3600;
    let minutes = (seconds % 3600) / 60;
    if hours > 0 {
        format!("{}h {}m", hours, minutes)
    } else {
        format!("{}m", minutes)
    }
}

/// Display rental items in table format
pub fn display_rental_items(rentals: &[ApiRentalListItem]) -> Result<()> {
    if rentals.is_empty() {
        println!("{}", style("No Bourse rentals found").yellow());
        return Ok(());
    }

    // Helper to get rate and cost for a rental from the API response fields
    let get_rental_pricing = |rental: &ApiRentalListItem| -> (String, String) {
        let rate = rental
            .hourly_cost
            .map(|r| format!("${:.2}/hr", r))
            .unwrap_or_else(|| "-".to_string());

        let cost = rental
            .accumulated_cost
            .as_deref()
            .map(format_usd)
            .unwrap_or_else(|| "-".to_string());

        (rate, cost)
    };

    #[derive(Tabled)]
    struct RentalRow {
        #[tabled(rename = "GPU")]
        gpu: String,
        #[tabled(rename = "State")]
        state: String,
        #[tabled(rename = "SSH")]
        ssh: String,
        #[tabled(rename = "Ports (Host → Container)")]
        ports: String,
        #[tabled(rename = "Image")]
        image: String,
        #[tabled(rename = "CPU")]
        cpu: String,
        #[tabled(rename = "RAM")]
        ram: String,
        #[tabled(rename = "Location")]
        location: String,
        #[tabled(rename = "Rate/hr")]
        rate_per_hour: String,
        #[tabled(rename = "Total Cost")]
        total_cost: String,
        #[tabled(rename = "Created")]
        created: String,
    }

    let rows: Vec<RentalRow> = rentals
        .iter()
        .map(|rental| {
            // Format GPU info from specs
            let gpu = format_gpu_info(&rental.gpu_specs);

            // Format CPU info
            let cpu = rental
                .cpu_specs
                .as_ref()
                .map(|cpu| format!("{} ({} cores)", cpu.model, cpu.cores))
                .unwrap_or_else(|| "Unknown".to_string());

            // Format RAM info
            let ram = rental
                .cpu_specs
                .as_ref()
                .map(|cpu| format!("{}GB", cpu.memory_gb))
                .unwrap_or_else(|| "Unknown".to_string());

            // Format location
            let location = format_node_location(&rental.location);

            // Format SSH availability
            let ssh = if rental.has_ssh { "✓" } else { "✗" };

            // Format port mappings (show up to 2-3 ports)
            let ports = format_port_mappings(&rental.port_mappings, Some(2));

            // Get pricing data for this rental
            let (rate_per_hour, total_cost) = get_rental_pricing(rental);

            RentalRow {
                gpu,
                state: rental.state.to_string(),
                ssh: ssh.to_string(),
                ports,
                image: rental.container_image.clone(),
                cpu,
                ram,
                location,
                rate_per_hour,
                total_cost,
                created: format_timestamp(&rental.created_at),
            }
        })
        .collect();

    let mut table = Table::new(rows);
    table.with(Style::modern());
    println!("{table}");

    Ok(())
}

/// Helper function to format port mappings
fn format_port_mappings(
    port_mappings: &Option<Vec<basilica_validator::rental::PortMapping>>,
    max_count: Option<usize>,
) -> String {
    match port_mappings {
        None => "-".to_string(),
        Some(ports) if ports.is_empty() => "-".to_string(),
        Some(ports) => {
            let formatted_ports: Vec<String> = ports
                .iter()
                .map(|p| format!("{}→{}", p.host_port, p.container_port))
                .collect();

            match max_count {
                Some(max) if formatted_ports.len() > max => {
                    let shown = &formatted_ports[..max];
                    let remaining = formatted_ports.len() - max;
                    format!("{}, +{} more", shown.join(", "), remaining)
                }
                _ => formatted_ports.join(", "),
            }
        }
    }
}

/// Helper function to format GPU info
fn format_gpu_info(gpu_specs: &[GpuSpec]) -> String {
    if gpu_specs.is_empty() {
        return "Unknown".to_string();
    }

    // Check if all GPUs are the same
    let first_gpu = &gpu_specs[0];
    let all_same = gpu_specs
        .iter()
        .all(|g| g.name == first_gpu.name && g.memory_gb == first_gpu.memory_gb);

    if all_same {
        if gpu_specs.len() > 1 {
            format!("{}x {}", gpu_specs.len(), first_gpu.name)
        } else {
            format!("1x {}", first_gpu.name)
        }
    } else {
        // List each GPU
        gpu_specs
            .iter()
            .map(|g| g.name.clone())
            .collect::<Vec<_>>()
            .join(", ")
    }
}

/// Display configuration in table format
pub fn display_config(config: &HashMap<String, String>) -> Result<()> {
    #[derive(Tabled)]
    struct ConfigRow {
        #[tabled(rename = "Key")]
        key: String,
        #[tabled(rename = "Value")]
        value: String,
    }

    let mut rows: Vec<ConfigRow> = config
        .iter()
        .map(|(key, value)| ConfigRow {
            key: key.clone(),
            value: value.clone(),
        })
        .collect();

    rows.sort_by(|a, b| a.key.cmp(&b.key));

    let mut table = Table::new(rows);
    table.with(Style::modern());
    println!("{table}");

    Ok(())
}

/// Display API keys in table format
pub fn display_api_keys(keys: &[ApiKeyInfo]) -> Result<()> {
    #[derive(Tabled)]
    struct ApiKeyRow {
        #[tabled(rename = "Name")]
        name: String,
        #[tabled(rename = "Created")]
        created: String,
        #[tabled(rename = "Last Used")]
        last_used: String,
    }

    let rows: Vec<ApiKeyRow> = keys
        .iter()
        .map(|key| ApiKeyRow {
            name: key.name.clone(),
            created: format_timestamp(&key.created_at.to_rfc3339()),
            last_used: key
                .last_used_at
                .map(|dt| format_timestamp(&dt.to_rfc3339()))
                .unwrap_or_else(|| "Never".to_string()),
        })
        .collect();

    let mut table = Table::new(rows);
    table.with(Style::modern());
    println!("{table}");

    Ok(())
}

/// Helper function to format GPU info for an available node
fn format_node_gpu_info(node: &AvailableNode) -> String {
    if node.node.gpu_specs.is_empty() {
        "No GPU".to_string()
    } else {
        format_gpu_info(&node.node.gpu_specs)
    }
}

/// Helper function to format location
fn format_node_location(location: &Option<String>) -> String {
    location
        .as_ref()
        .map(|loc| {
            LocationProfile::from_str(loc)
                .ok()
                .map(|profile| profile.to_string())
                .unwrap_or_else(|| loc.clone())
        })
        .unwrap_or_else(|| "Unknown".to_string())
}

/// Display available nodes in detailed format (individual nodes)
pub fn display_available_nodes_detailed(
    nodes: &[AvailableNode],
    pricing_map: &HashMap<String, String>,
) -> Result<()> {
    if nodes.is_empty() {
        println!("No available nodes found matching the specified criteria.");
        return Ok(());
    }

    // Helper function to calculate price for a node
    let get_node_price = |node: &AvailableNode| -> String {
        if let Some(gpu_spec) = node.node.gpu_specs.first() {
            let category = GpuCategory::from_str(&gpu_spec.name).unwrap();
            let gpu_count = node.node.gpu_specs.len();
            // GPU types are lowercase (h100, a100, etc.)
            let lookup_key = category.to_string().to_lowercase();

            pricing_map
                .get(&lookup_key)
                .and_then(|rate| {
                    rate.parse::<Decimal>().ok().map(|r| {
                        let total_rate = r * Decimal::from(gpu_count);
                        format!("${:.2}/hr", total_rate)
                    })
                })
                .unwrap_or_else(|| "-".to_string())
        } else {
            "-".to_string()
        }
    };

    #[derive(Tabled)]
    struct NodeRow {
        #[tabled(rename = "GPU")]
        gpu_info: String,
        #[tabled(rename = "CPU")]
        cpu: String,
        #[tabled(rename = "RAM")]
        ram: String,
        #[tabled(rename = "Location")]
        location: String,
        #[tabled(rename = "PRICE")]
        price: String,
    }

    let rows: Vec<NodeRow> = nodes
        .iter()
        .map(|node| NodeRow {
            gpu_info: format_node_gpu_info(node),
            cpu: format!(
                "{} ({} cores)",
                node.node.cpu_specs.model, node.node.cpu_specs.cores
            ),
            ram: format!("{}GB", node.node.cpu_specs.memory_gb),
            location: format_node_location(&node.node.location),
            price: get_node_price(node),
        })
        .collect();

    let mut table = Table::new(rows);
    table.with(Style::modern());
    println!("{}", table);

    println!("\nTotal available nodes: {}", nodes.len());

    Ok(())
}

/// Display community cloud GPU categories in aggregated format
pub fn display_community_cloud_categories(
    aggregations: &[crate::cli::handlers::gpu_rental_helpers::GpuCategoryAggregation],
) -> Result<()> {
    if aggregations.is_empty() {
        println!("No available GPUs found matching the specified criteria.");
        return Ok(());
    }

    #[derive(Tabled)]
    struct CategoryRow {
        #[tabled(rename = "GPU")]
        gpu: String,
        #[tabled(rename = "Available")]
        available: String,
        #[tabled(rename = "Price/hr")]
        price: String,
    }

    let rows: Vec<CategoryRow> = aggregations
        .iter()
        .map(|agg| {
            let gpu = if agg.gpu_count > 1 {
                format!("{}x {}", agg.gpu_count, agg.gpu_category)
            } else {
                agg.gpu_category.clone()
            };

            let multiplier = agg.gpu_count as f64;
            let price = match (agg.min_rate_cents, agg.max_rate_cents) {
                (Some(min), Some(max)) if min == max => {
                    format!("${:.2}", min as f64 / 100.0 * multiplier)
                }
                (Some(min), Some(max)) => {
                    format!(
                        "${:.2} - ${:.2}",
                        min as f64 / 100.0 * multiplier,
                        max as f64 / 100.0 * multiplier
                    )
                }
                _ => "Market".to_string(),
            };

            CategoryRow {
                gpu,
                available: format!("{} nodes", agg.node_count),
                price,
            }
        })
        .collect();

    let mut table = Table::new(rows);
    table.with(Style::modern());
    println!("{}", table);

    let total_nodes: usize = aggregations.iter().map(|a| a.node_count).sum();
    println!("\nTotal available nodes: {}", total_nodes);

    Ok(())
}

/// Display secure cloud GPU offerings in detailed format (individual offerings)
pub fn display_secure_cloud_offerings_detailed(offerings: &[GpuOffering]) -> Result<()> {
    if offerings.is_empty() {
        println!("No GPUs available matching your criteria.");
        return Ok(());
    }

    #[derive(Tabled)]
    struct OfferingRow {
        #[tabled(rename = "PROVIDER")]
        provider: String,
        #[tabled(rename = "GPU")]
        gpu_info: String,
        #[tabled(rename = "vCPU")]
        vcpu: String,
        #[tabled(rename = "RAM")]
        ram: String,
        #[tabled(rename = "STORAGE")]
        storage: String,
        #[tabled(rename = "INTERCONNECT")]
        interconnect: String,
        #[tabled(rename = "REGION")]
        region: String,
        #[tabled(rename = "PRICE/HR")]
        price: String,
    }

    let rows: Vec<OfferingRow> = offerings
        .iter()
        .map(|offering| {
            let gpu_info = {
                let base = if offering.gpu_count == 1 {
                    offering.gpu_type.to_string()
                } else {
                    format!("{}x {}", offering.gpu_count, offering.gpu_type)
                };
                if offering.is_spot {
                    format!("{} (Spot)", base)
                } else {
                    base
                }
            };

            // Calculate total hourly cost (per-GPU rate × gpu_count)
            let total_hourly_cost =
                offering.hourly_rate_per_gpu * Decimal::from(offering.gpu_count);
            OfferingRow {
                provider: offering.provider.to_string(),
                gpu_info,
                vcpu: offering.vcpu_count.to_string(),
                ram: format!("{}GB", offering.system_memory_gb),
                storage: offering.storage.clone().unwrap_or_else(|| "-".to_string()),
                interconnect: offering
                    .interconnect
                    .clone()
                    .unwrap_or_else(|| "-".to_string()),
                region: offering.region.clone(),
                price: format!("${:.2}/hr", total_hourly_cost),
            }
        })
        .collect();

    let mut table = Table::new(rows);
    table.with(Style::modern());
    println!("{}", table);

    println!("\nTotal offerings: {}", offerings.len());

    Ok(())
}

/// Display deposits history in table format
pub fn display_deposits(response: &ListDepositsResponse) -> Result<()> {
    println!();
    println!("{}", style("# Deposit History").dim());
    println!();

    let mut builder = Builder::default();

    // Add header
    builder.push_record(["Date (UTC)", "TAO", "Tx Hash", "Conf", "Block", "Status"]);

    let mut total_tao = 0.0;

    for deposit in &response.deposits {
        let amount_tao: f64 = deposit.amount_tao.parse().unwrap_or(0.0);
        total_tao += amount_tao;

        // Format date
        let date = deposit.observed_at.format("%Y-%m-%d %H:%M:%S").to_string();

        // Format tx hash (truncate to first 8 and last 3 chars)
        let tx_hash = if deposit.tx_hash.len() > 11 {
            format!(
                "{}...{}",
                &deposit.tx_hash[..8],
                &deposit.tx_hash[deposit.tx_hash.len() - 3..]
            )
        } else {
            deposit.tx_hash.clone()
        };

        // Format confirmations (12+ means finalized)
        let confirmations = if deposit.finalized_at.is_some() {
            "12+".to_string()
        } else {
            "-".to_string()
        };

        // Format status
        let status = if deposit.credited_at.is_some() {
            "Credited"
        } else if deposit.finalized_at.is_some() {
            "Finalized"
        } else {
            "Pending"
        };

        builder.push_record([
            date.as_str(),
            &format!("{:.3}", amount_tao),
            tx_hash.as_str(),
            confirmations.as_str(),
            &deposit.block_number.to_string(),
            status,
        ]);
    }

    let mut table = builder.build();
    table.with(Style::modern());
    println!("{}", table);

    // Display totals
    println!();
    println!("{}:", style("Total Deposits").bold());
    println!("  {} TAO", style(format!("{:.3}", total_tao)).green());

    Ok(())
}

/// Display detailed usage for a specific rental
pub fn display_rental_usage_detail(usage: &RentalUsageResponse) -> Result<()> {
    println!(
        "{}: {}",
        style("Rental ID").cyan(),
        style(&usage.rental_id).bold()
    );
    println!(
        "{}: {}",
        style("Total Cost").cyan(),
        style(&usage.total_cost).green().bold()
    );
    println!();

    if let Some(summary) = &usage.summary {
        println!("{}", style("Resource Usage Summary").bold());
        println!();
        println!(
            "  {}: {:.1}%",
            style("Avg CPU Usage").cyan(),
            summary.avg_cpu_percent
        );
        println!(
            "  {}: {} MB",
            style("Avg Memory Usage").cyan(),
            summary.avg_memory_mb
        );
        println!(
            "  {}: {:.1}%",
            style("Avg GPU Utilization").cyan(),
            summary.avg_gpu_utilization
        );
        println!(
            "  {}: {} bytes",
            style("Total Network I/O").cyan(),
            summary.total_network_bytes
        );
        println!(
            "  {}: {} bytes",
            style("Total Disk I/O").cyan(),
            summary.total_disk_bytes
        );
        println!(
            "  {}: {} seconds ({:.1} hours)",
            style("Duration").cyan(),
            summary.duration_secs,
            summary.duration_secs as f64 / 3600.0
        );
        println!();
    }

    if !usage.data_points.is_empty() {
        #[derive(Tabled)]
        struct UsageDataRow {
            #[tabled(rename = "Timestamp")]
            timestamp: String,
            #[tabled(rename = "CPU %")]
            cpu_percent: String,
            #[tabled(rename = "Memory (MB)")]
            memory_mb: String,
            #[tabled(rename = "Cost")]
            cost: String,
        }

        const MAX_POINTS: usize = 10;
        let total_points = usage.data_points.len();
        let start_index = total_points.saturating_sub(MAX_POINTS);

        let rows: Vec<UsageDataRow> = usage
            .data_points
            .iter()
            .skip(start_index)
            .map(|dp| UsageDataRow {
                timestamp: dp.timestamp.format("%Y-%m-%d %H:%M:%S UTC").to_string(),
                cpu_percent: format!("{:.1}%", dp.cpu_percent),
                memory_mb: dp.memory_mb.to_string(),
                cost: dp.cost.clone(),
            })
            .collect();

        println!("{}", style("Usage Data Points").bold());
        println!();
        let mut table = Table::new(&rows);
        table.with(Style::modern());
        println!("{}", table);
        if total_points > MAX_POINTS {
            println!(
                "{}",
                style(format!(
                    "Showing last {} of {} data points.",
                    MAX_POINTS, total_points
                ))
                .dim()
            );
        }
        println!();
    } else {
        println!("{}", style("No usage data points available").yellow());
        println!();
    }

    println!("{}", style("Quick Commands:").cyan().bold());
    println!(
        "  {} {}",
        style("basilica ps").yellow().bold(),
        style("- List active rentals with pricing and cost information").dim()
    );

    Ok(())
}

/// Display usage history list
pub fn display_usage_history(history: &UsageHistoryResponse) -> Result<()> {
    if history.rentals.is_empty() {
        println!("{}", style("No rental usage history found").yellow());
        return Ok(());
    }

    #[derive(Tabled)]
    struct UsageHistoryRow {
        #[tabled(rename = "Rental ID")]
        rental_id: String,
        #[tabled(rename = "Node ID")]
        node_id: String,
        #[tabled(rename = "Status")]
        status: String,
        #[tabled(rename = "Hourly Rate")]
        hourly_rate: String,
        #[tabled(rename = "Current Cost")]
        current_cost: String,
        #[tabled(rename = "Started")]
        started: String,
        #[tabled(rename = "Last Updated")]
        last_updated: String,
    }

    let mut rows: Vec<UsageHistoryRow> = history
        .rentals
        .iter()
        .map(|rental| {
            let hourly_rate = rental
                .hourly_rate
                .parse::<Decimal>()
                .ok()
                .map(|rate| format!("${:.2}/hr", rate))
                .unwrap_or_else(|| rental.hourly_rate.clone());

            let current_cost = format_usd(&rental.current_cost);

            UsageHistoryRow {
                rental_id: rental.rental_id.clone(),
                node_id: rental.node_id.clone(),
                status: rental.status.clone(),
                hourly_rate,
                current_cost,
                started: rental.start_time.format("%Y-%m-%d %H:%M UTC").to_string(),
                last_updated: rental.last_updated.format("%Y-%m-%d %H:%M UTC").to_string(),
            }
        })
        .collect();

    rows.sort_by(|a, b| b.started.cmp(&a.started));

    println!(
        "{} ({} total)",
        style("Rental Usage History").bold(),
        style(history.total_count).cyan()
    );
    println!();
    let mut table = Table::new(&rows);
    table.with(Style::modern());
    println!("{}", table);
    println!();

    let total_cost: Decimal = history
        .rentals
        .iter()
        .filter_map(|r| r.current_cost.parse::<Decimal>().ok())
        .sum();

    println!(
        "{}: {}",
        style("Total Cost (All Rentals)").cyan(),
        style(format_usd(&total_cost.to_string())).green().bold()
    );
    println!();
    println!("{}", style("Quick Commands:").cyan().bold());
    println!(
        "  {} {}",
        style("basilica balance").yellow().bold(),
        style("- Show your current credit balance").dim()
    );

    Ok(())
}

/// Display historical rental data from billing service
pub fn display_rental_history(rentals: &[&HistoricalRentalItem]) -> Result<()> {
    if rentals.is_empty() {
        println!("{}", style("No rental history found").yellow());
        return Ok(());
    }

    #[derive(Tabled)]
    struct HistoryRow {
        #[tabled(rename = "Rental ID")]
        rental_id: String,
        #[tabled(rename = "GPUs")]
        gpu_count: String,
        #[tabled(rename = "Status")]
        status: String,
        #[tabled(rename = "Total Cost")]
        total_cost: String,
        #[tabled(rename = "Started")]
        started: String,
        #[tabled(rename = "Stopped")]
        stopped: String,
        #[tabled(rename = "Duration")]
        duration: String,
    }

    let mut rows: Vec<HistoryRow> = rentals
        .iter()
        .map(|rental| {
            let total_cost = format_usd(&rental.total_cost);

            HistoryRow {
                rental_id: rental.rental_id.clone(),
                gpu_count: format!("{}x GPU", rental.gpu_count),
                status: rental.status.clone(),
                total_cost,
                started: rental
                    .started_at
                    .with_timezone(&Local)
                    .format("%Y-%m-%d %H:%M")
                    .to_string(),
                stopped: rental
                    .stopped_at
                    .with_timezone(&Local)
                    .format("%Y-%m-%d %H:%M")
                    .to_string(),
                duration: format_duration(rental.duration_seconds),
            }
        })
        .collect();

    rows.sort_by(|a, b| b.started.cmp(&a.started));

    let mut table = Table::new(&rows);
    table.with(Style::modern());
    println!("{table}");

    Ok(())
}

/// Display historical CPU rental data from billing service
pub fn display_cpu_rental_history(rentals: &[&HistoricalRentalItem]) -> Result<()> {
    if rentals.is_empty() {
        println!("{}", style("No CPU rental history found").yellow());
        return Ok(());
    }

    #[derive(Tabled)]
    struct CpuHistoryRow {
        #[tabled(rename = "Provider")]
        provider: String,
        #[tabled(rename = "vCPU")]
        vcpu: String,
        #[tabled(rename = "RAM")]
        ram: String,
        #[tabled(rename = "Status")]
        status: String,
        #[tabled(rename = "Rate/hr")]
        rate: String,
        #[tabled(rename = "Total Cost")]
        total_cost: String,
        #[tabled(rename = "Started")]
        started: String,
        #[tabled(rename = "Stopped")]
        stopped: String,
        #[tabled(rename = "Duration")]
        duration: String,
    }

    let mut rows: Vec<CpuHistoryRow> = rentals
        .iter()
        .map(|rental| {
            let total_cost = format_usd(&rental.total_cost);

            let rate = rental
                .hourly_rate
                .map(|r| format!("${:.2}/hr", r))
                .unwrap_or_else(|| "-".to_string());

            let vcpu = rental
                .vcpu_count
                .map(|cores| format!("{} cores", cores))
                .unwrap_or_else(|| "-".to_string());

            let ram = rental
                .system_memory_gb
                .map(|gb| format!("{}GB", gb))
                .unwrap_or_else(|| "-".to_string());

            CpuHistoryRow {
                provider: rental.provider.clone().unwrap_or_else(|| "-".to_string()),
                vcpu,
                ram,
                status: rental.status.clone(),
                rate,
                total_cost,
                started: rental
                    .started_at
                    .with_timezone(&Local)
                    .format("%Y-%m-%d %H:%M")
                    .to_string(),
                stopped: rental
                    .stopped_at
                    .with_timezone(&Local)
                    .format("%Y-%m-%d %H:%M")
                    .to_string(),
                duration: format_duration(rental.duration_seconds),
            }
        })
        .collect();

    rows.sort_by(|a, b| b.started.cmp(&a.started));

    let mut table = Table::new(&rows);
    table.with(Style::modern());
    println!("{table}");

    Ok(())
}

/// Display secure cloud rentals in table format
pub fn display_secure_cloud_rentals(
    rentals: &[&basilica_sdk::types::SecureCloudRentalListItem],
) -> Result<()> {
    if rentals.is_empty() {
        println!("{}", style("No Citadel rentals found").yellow());
        return Ok(());
    }

    #[derive(Tabled)]
    struct SecureCloudRentalRow {
        #[tabled(rename = "Provider")]
        provider: String,
        #[tabled(rename = "GPU")]
        gpu: String,
        #[tabled(rename = "State")]
        status: String,
        #[tabled(rename = "IP")]
        ip: String,
        #[tabled(rename = "SSH")]
        ssh: String,
        #[tabled(rename = "CPU")]
        cpu: String,
        #[tabled(rename = "RAM")]
        ram: String,
        #[tabled(rename = "Region")]
        region: String,
        #[tabled(rename = "Rate/hr")]
        hourly_cost: String,
        #[tabled(rename = "Total Cost")]
        total_cost: String,
        #[tabled(rename = "Created")]
        created: String,
    }

    let rows: Vec<SecureCloudRentalRow> = rentals
        .iter()
        .map(|rental| {
            let gpu_str = {
                let base = if rental.gpu_count > 1 {
                    format!("{}x {}", rental.gpu_count, rental.gpu_type.to_uppercase())
                } else {
                    rental.gpu_type.to_uppercase()
                };
                if rental.is_spot {
                    format!("{} (Spot)", base)
                } else {
                    base
                }
            };

            let ssh = if rental.ssh_command.is_some() {
                "✓"
            } else {
                "✗"
            };

            // Format CPU
            let cpu = rental
                .vcpu_count
                .map(|cores| format!("{} cores", cores))
                .unwrap_or_else(|| "-".to_string());

            // Format RAM
            let ram = rental
                .system_memory_gb
                .map(|gb| format!("{}GB", gb))
                .unwrap_or_else(|| "-".to_string());

            // Use accumulated cost from billing service - no fallback
            let total_cost = rental
                .accumulated_cost
                .as_deref()
                .map(format_usd)
                .unwrap_or_else(|| "-".to_string());

            SecureCloudRentalRow {
                provider: rental.provider.clone(),
                gpu: gpu_str,
                status: rental.status.clone(),
                ip: rental.ip_address.clone().unwrap_or_else(|| "-".to_string()),
                ssh: ssh.to_string(),
                cpu,
                ram,
                region: rental
                    .location_code
                    .clone()
                    .unwrap_or_else(|| "-".to_string()),
                hourly_cost: format!("${:.2}/hr", rental.hourly_cost),
                total_cost,
                created: format_timestamp(&rental.created_at.to_rfc3339()),
            }
        })
        .collect();

    let mut table = Table::new(&rows);
    table.with(Style::modern());
    println!("{}", table);

    Ok(())
}

/// Display CPU-only offerings in detailed table format
pub fn display_cpu_offerings_detailed(
    offerings: &[basilica_sdk::types::CpuOffering],
) -> Result<()> {
    if offerings.is_empty() {
        println!("{}", style("No CPU instances available").yellow());
        return Ok(());
    }

    #[derive(Tabled)]
    struct CpuOfferingRow {
        #[tabled(rename = "PROVIDER")]
        provider: String,
        #[tabled(rename = "vCPU")]
        vcpu: String,
        #[tabled(rename = "RAM")]
        ram: String,
        #[tabled(rename = "STORAGE")]
        storage: String,
        #[tabled(rename = "REGION")]
        region: String,
        #[tabled(rename = "PRICE/HR")]
        price: String,
    }

    let rows: Vec<CpuOfferingRow> = offerings
        .iter()
        .map(|offering| CpuOfferingRow {
            provider: offering.provider.clone(),
            vcpu: format!("{} cores", offering.vcpu_count),
            ram: format!("{}GB", offering.system_memory_gb),
            storage: if offering.storage_gb > 0 {
                format!("{}GB", offering.storage_gb)
            } else {
                "-".to_string()
            },
            region: offering.region.clone(),
            price: format!(
                "${:.2}/hr",
                offering.hourly_rate.parse::<f64>().unwrap_or(0.0)
            ),
        })
        .collect();

    let mut table = Table::new(&rows);
    table.with(Style::modern());
    println!("{}", table);

    println!("\nTotal Citadel (CPU) offerings: {}", offerings.len());

    Ok(())
}

/// Display CPU-only rentals in table format (no GPU column)
pub fn display_cpu_rentals(
    rentals: &[&basilica_sdk::types::SecureCloudRentalListItem],
) -> Result<()> {
    if rentals.is_empty() {
        println!("{}", style("No CPU-only rentals found").yellow());
        return Ok(());
    }

    #[derive(Tabled)]
    struct CpuRentalRow {
        #[tabled(rename = "Provider")]
        provider: String,
        #[tabled(rename = "vCPU")]
        vcpu: String,
        #[tabled(rename = "RAM")]
        ram: String,
        #[tabled(rename = "State")]
        status: String,
        #[tabled(rename = "IP")]
        ip: String,
        #[tabled(rename = "SSH")]
        ssh: String,
        #[tabled(rename = "Region")]
        region: String,
        #[tabled(rename = "Rate/hr")]
        hourly_cost: String,
        #[tabled(rename = "Total Cost")]
        total_cost: String,
        #[tabled(rename = "Created")]
        created: String,
    }

    let rows: Vec<CpuRentalRow> = rentals
        .iter()
        .map(|rental| {
            let ssh = if rental.ssh_command.is_some() {
                "✓"
            } else {
                "✗"
            };

            // Format vCPU
            let vcpu = rental
                .vcpu_count
                .map(|cores| format!("{} cores", cores))
                .unwrap_or_else(|| "-".to_string());

            // Format RAM
            let ram = rental
                .system_memory_gb
                .map(|gb| format!("{}GB", gb))
                .unwrap_or_else(|| "-".to_string());

            // Use accumulated cost from billing service
            let total_cost = rental
                .accumulated_cost
                .as_deref()
                .map(format_usd)
                .unwrap_or_else(|| "-".to_string());

            CpuRentalRow {
                provider: rental.provider.clone(),
                vcpu,
                ram,
                status: rental.status.clone(),
                ip: rental.ip_address.clone().unwrap_or_else(|| "-".to_string()),
                ssh: ssh.to_string(),
                region: rental
                    .location_code
                    .clone()
                    .unwrap_or_else(|| "-".to_string()),
                hourly_cost: format!("${:.2}/hr", rental.hourly_cost),
                total_cost,
                created: format_timestamp(&rental.created_at.to_rfc3339()),
            }
        })
        .collect();

    let mut table = Table::new(&rows);
    table.with(Style::modern());
    println!("{}", table);

    Ok(())
}

/// Display volumes in table format
pub fn display_volumes(volumes: &[VolumeResponse]) -> Result<()> {
    if volumes.is_empty() {
        println!("{}", style("No volumes found").yellow());
        return Ok(());
    }

    #[derive(Tabled)]
    struct VolumeRow {
        #[tabled(rename = "Name")]
        name: String,
        #[tabled(rename = "Size")]
        size: String,
        #[tabled(rename = "Status")]
        status: String,
        #[tabled(rename = "Provider")]
        provider: String,
        #[tabled(rename = "Region")]
        region: String,
        #[tabled(rename = "Rental")]
        rental: String,
        #[tabled(rename = "Rate/hr")]
        hourly_cost: String,
        #[tabled(rename = "Total Cost")]
        total_cost: String,
        #[tabled(rename = "Created")]
        created: String,
    }

    let rows: Vec<VolumeRow> = volumes
        .iter()
        .map(|volume| {
            // Format status
            let status = match volume.status {
                basilica_sdk::types::VolumeStatus::Available => "Available".to_string(),
                basilica_sdk::types::VolumeStatus::Attached => "Attached".to_string(),
                basilica_sdk::types::VolumeStatus::Pending => "Pending".to_string(),
                basilica_sdk::types::VolumeStatus::Deleting => "Deleting".to_string(),
                basilica_sdk::types::VolumeStatus::Error => "Error".to_string(),
            };

            // Use accumulated cost from billing service
            let total_cost = volume
                .accumulated_cost
                .as_deref()
                .map(format_usd)
                .unwrap_or_else(|| "-".to_string());

            VolumeRow {
                name: volume.name.clone(),
                size: format!("{}GB", volume.size_gb),
                status,
                provider: volume.provider.clone(),
                region: volume.region.clone(),
                rental: volume.rental_id.clone().unwrap_or_else(|| "-".to_string()),
                hourly_cost: volume
                    .estimated_hourly_cost
                    .map(|c| format!("${:.2}/hr", c))
                    .unwrap_or_else(|| "-".to_string()),
                total_cost,
                created: format_timestamp(&volume.created_at.to_rfc3339()),
            }
        })
        .collect();

    let mut table = Table::new(&rows);
    table.with(Style::modern());
    println!("{}", table);

    Ok(())
}
