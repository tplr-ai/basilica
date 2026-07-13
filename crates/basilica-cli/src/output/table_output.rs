//! Table formatting for CLI output

use super::{format_cents, format_usd};
use crate::cli::commands::ResolvedOutput;
use crate::error::Result;
use basilica_common::types::GpuOffering;
use basilica_common::{types::GpuCategory, LocationProfile};
use basilica_sdk::{
    types::{
        ApiKeyInfo, ApiRentalListItem, CardPurchaseStatus, GpuSpec, HistoricalRentalItem,
        ListCardPurchasesResponse, ListDepositsResponse, RentalUsageResponse, UsageHistoryResponse,
        VolumeResponse,
    },
    AvailableNode,
};
use chrono::{DateTime, Local, Utc};
use console::{style, Term};
use rust_decimal::Decimal;
use std::{collections::HashMap, io::IsTerminal, str::FromStr};
use tabled::{
    builder::Builder,
    settings::{object::Columns, Modify, Style, Width},
    Table, Tabled,
};

const MIN_NAME_WIDTH: usize = 12;

#[derive(Debug, Clone, Copy)]
pub(crate) struct RenderContext {
    pub is_tty: bool,
    pub width: Option<usize>,
    pub now: DateTime<Utc>,
}

impl RenderContext {
    pub(crate) fn stdout() -> Self {
        let is_tty = std::io::stdout().is_terminal();
        let width = is_tty
            .then(|| {
                Term::stdout()
                    .size_checked()
                    .map(|(_, columns)| columns as usize)
            })
            .flatten();
        Self {
            is_tty,
            width,
            now: Utc::now(),
        }
    }
}

pub(crate) fn render_table_with_context(
    mut wide: Table,
    compact: Option<Table>,
    output: ResolvedOutput,
    context: RenderContext,
    natural_name_width: Option<usize>,
) -> String {
    wide.with(Style::modern());

    let use_compact = match output {
        ResolvedOutput::Compact => true,
        ResolvedOutput::Wide | ResolvedOutput::Json => false,
        ResolvedOutput::Auto => {
            context.is_tty
                && context
                    .width
                    .is_some_and(|width| wide.total_width() > width)
        }
    };

    let Some(mut table) = use_compact.then_some(compact).flatten() else {
        return wide.to_string();
    };
    table.with(Style::modern());

    if let (Some(width), Some(name_width)) = (context.width, natural_name_width) {
        let overflow = table.total_width().saturating_sub(width);
        if overflow > 0 && name_width > MIN_NAME_WIDTH {
            let target = name_width.saturating_sub(overflow).max(MIN_NAME_WIDTH);
            table.with(Modify::new(Columns::new(0..1)).with(Width::truncate(target).suffix("…")));
        }
    }

    table.to_string()
}

fn print_standard_table(table: Table) {
    let rendered = render_table_with_context(
        table,
        None,
        ResolvedOutput::Auto,
        RenderContext::stdout(),
        None,
    );
    println!("{rendered}");
}

fn format_created(timestamp: DateTime<Utc>) -> String {
    timestamp
        .with_timezone(&Local)
        .format("%y-%m-%d %H:%M")
        .to_string()
}

pub(crate) fn format_created_str(timestamp: &str) -> String {
    DateTime::parse_from_rfc3339(timestamp)
        .map(|timestamp| format_created(timestamp.with_timezone(&Utc)))
        .unwrap_or_else(|_| timestamp.to_string())
}

fn format_age(created: DateTime<Utc>, now: DateTime<Utc>) -> String {
    let elapsed = now.signed_duration_since(created).num_minutes().max(0);
    if elapsed >= 24 * 60 {
        format!("{}d", elapsed / (24 * 60))
    } else if elapsed >= 60 {
        format!("{}h", elapsed / 60)
    } else {
        format!("{}m", elapsed)
    }
}

pub(crate) fn format_age_str(created: &str, now: DateTime<Utc>) -> String {
    DateTime::parse_from_rfc3339(created)
        .map(|created| format_age(created.with_timezone(&Utc), now))
        .unwrap_or_else(|_| "—".to_string())
}

fn compact_image_name(image: &str) -> String {
    let without_digest = image.split('@').next().unwrap_or(image);
    let last_slash = without_digest.rfind('/');
    let without_tag = match without_digest.rfind(':') {
        Some(colon) if last_slash.is_none_or(|slash| colon > slash) => &without_digest[..colon],
        _ => without_digest,
    };
    let mut parts = without_tag.split('/');
    let first = parts.next().unwrap_or(without_tag);
    if first.contains('.') || first.contains(':') || first == "localhost" {
        parts.collect::<Vec<_>>().join("/")
    } else {
        without_tag.to_string()
    }
}

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
pub fn display_rental_items(rentals: &[ApiRentalListItem], output: ResolvedOutput) -> Result<()> {
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
    struct WideRow {
        #[tabled(rename = "Name")]
        name: String,
        #[tabled(rename = "GPU")]
        gpu: String,
        #[tabled(rename = "State")]
        state: String,
        #[tabled(rename = "IP")]
        ip: String,
        #[tabled(rename = "Ports (Host → Container)")]
        ports: String,
        #[tabled(rename = "Image")]
        image: String,
        #[tabled(rename = "CPU/RAM")]
        cpu_ram: String,
        #[tabled(rename = "Location")]
        location: String,
        #[tabled(rename = "Rate/hr")]
        rate_per_hour: String,
        #[tabled(rename = "Total Cost")]
        total_cost: String,
        #[tabled(rename = "Created")]
        created: String,
    }

    let wide_rows: Vec<WideRow> = rentals
        .iter()
        .map(|rental| {
            // Format GPU info from specs
            let gpu = format_gpu_info(&rental.gpu_specs);

            let cpu_ram = rental
                .cpu_specs
                .as_ref()
                .map(|cpu| format!("{} cores / {}GB", cpu.cores, cpu.memory_gb))
                .unwrap_or_else(|| "—".to_string());

            // Format location
            let location = format_node_location(&rental.location);

            // Format port mappings (show up to 2-3 ports)
            let ports = format_port_mappings(&rental.port_mappings, Some(2));

            // Get pricing data for this rental
            let (rate_per_hour, total_cost) = get_rental_pricing(rental);

            WideRow {
                name: rental.name.clone(),
                gpu,
                state: rental.state.to_string(),
                ip: "—".to_string(),
                ports,
                image: compact_image_name(&rental.container_image),
                cpu_ram,
                location,
                rate_per_hour,
                total_cost,
                created: format_created_str(&rental.created_at),
            }
        })
        .collect();

    #[derive(Tabled)]
    struct CompactRow {
        #[tabled(rename = "Name")]
        name: String,
        #[tabled(rename = "GPU")]
        gpu: String,
        #[tabled(rename = "State")]
        state: String,
        #[tabled(rename = "IP")]
        ip: String,
        #[tabled(rename = "Rate")]
        rate: String,
        #[tabled(rename = "Age")]
        age: String,
    }

    let context = RenderContext::stdout();
    let compact_rows: Vec<CompactRow> = rentals
        .iter()
        .map(|rental| CompactRow {
            name: rental.name.clone(),
            gpu: format_gpu_info(&rental.gpu_specs),
            state: rental.state.to_string(),
            // The Bourse list response currently exposes only `has_ssh`, not the host.
            ip: "—".to_string(),
            rate: rental
                .hourly_cost
                .map(|rate| format!("${rate:.2}/h"))
                .unwrap_or_else(|| "—".to_string()),
            age: format_age_str(&rental.created_at, context.now),
        })
        .collect();
    let name_width = rentals
        .iter()
        .map(|rental| console::measure_text_width(&rental.name))
        .max();
    let rendered = render_table_with_context(
        Table::new(wide_rows),
        Some(Table::new(compact_rows)),
        output,
        context,
        name_width,
    );
    println!("{rendered}");

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

    rows.sort_by_key(|r| r.key.clone());

    print_standard_table(Table::new(rows));

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

    print_standard_table(Table::new(rows));

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
            ram: format!("{} GB", node.node.cpu_specs.memory_gb),
            location: format_node_location(&node.node.location),
            price: get_node_price(node),
        })
        .collect();

    print_standard_table(Table::new(rows));

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

    print_standard_table(Table::new(rows));

    let total_nodes: usize = aggregations.iter().map(|a| a.node_count).sum();
    println!("\nTotal available nodes: {}", total_nodes);

    Ok(())
}

/// Display secure cloud GPU offerings in detailed format (individual offerings)
pub fn display_secure_cloud_offerings_detailed(
    offerings: &[GpuOffering],
    output: ResolvedOutput,
) -> Result<()> {
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
        #[tabled(rename = "VRAM")]
        vram: String,
        #[tabled(rename = "CPU/RAM")]
        cpu_ram: String,
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
            let vram = match offering.gpu_memory_gb_per_gpu {
                Some(mem_per_gpu) => format!("{} GB", mem_per_gpu * offering.gpu_count),
                None => "-".to_string(),
            };
            OfferingRow {
                provider: offering.provider.to_string(),
                gpu_info,
                vram,
                cpu_ram: format!(
                    "{} cores / {}GB",
                    offering.vcpu_count, offering.system_memory_gb
                ),
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

    #[derive(Tabled)]
    struct CompactOfferingRow {
        #[tabled(rename = "PROVIDER")]
        provider: String,
        #[tabled(rename = "GPU")]
        gpu: String,
        #[tabled(rename = "VRAM")]
        vram: String,
        #[tabled(rename = "REGION")]
        region: String,
        #[tabled(rename = "RATE")]
        rate: String,
    }

    let compact_rows = offerings.iter().map(|offering| {
        let gpu = if offering.gpu_count == 1 {
            offering.gpu_type.to_string()
        } else {
            format!("{}x {}", offering.gpu_count, offering.gpu_type)
        };
        CompactOfferingRow {
            provider: offering.provider.to_string(),
            gpu: if offering.is_spot {
                format!("{gpu} (Spot)")
            } else {
                gpu
            },
            vram: offering
                .gpu_memory_gb_per_gpu
                .map(|memory| format!("{} GB", memory * offering.gpu_count))
                .unwrap_or_else(|| "—".to_string()),
            region: offering.region.clone(),
            rate: format!(
                "${:.2}/h",
                offering.hourly_rate_per_gpu * Decimal::from(offering.gpu_count)
            ),
        }
    });
    let context = RenderContext::stdout();
    let rendered = render_table_with_context(
        Table::new(rows),
        Some(Table::new(compact_rows)),
        output,
        context,
        None,
    );
    println!("{rendered}");

    println!("\nTotal offerings: {}", offerings.len());

    Ok(())
}

/// Display deposits history in table format
pub fn display_deposits(response: &ListDepositsResponse) -> Result<()> {
    println!();
    println!("{}", style("Deposit History").bold());
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

    print_standard_table(builder.build());

    // Display totals
    println!();
    println!("{}:", style("Total Deposits").bold());
    println!("  {} TAO", style(format!("{:.3}", total_tao)).green());

    Ok(())
}

/// Display card payment history in table format.
pub fn display_card_purchases(response: &ListCardPurchasesResponse) -> Result<()> {
    println!();
    println!("{}", style("Card Payment History").bold());
    println!();

    let mut builder = Builder::default();
    builder.push_record(["Date (UTC)", "Amount", "Status", "Invoice/Receipt"]);

    let mut total_paid_cents: u64 = 0;

    for purchase in &response.purchases {
        let date = purchase
            .created_at
            .map(|dt| dt.format("%Y-%m-%d %H:%M:%S").to_string())
            .unwrap_or_else(|| "-".to_string());

        let amount = format_cents(purchase.requested_amount_cents);

        let status = match purchase.status {
            CardPurchaseStatus::Completed => "Completed",
            CardPurchaseStatus::Pending => "Pending",
            CardPurchaseStatus::Expired => "Expired",
            CardPurchaseStatus::Unspecified => "Unspecified",
        };

        // Invoice/Receipt cell doubles as the per-session link. Modern
        // terminals (iTerm2, Terminal.app, wezterm, Ghostty, VSCode,
        // kitty, GNOME Terminal) render the OSC 8 escape as clickable;
        // unsupported terminals show only the styled label and users can
        // drop to --json for the raw URL. Styling the label blue +
        // underlined (web-browser link convention) signals clickability
        // visually even before hovering. Prefer the hosted invoice page
        // when present — it bundles receipt-style payment detail already,
        // so a separate receipt link would be redundant. `invoice_pdf`
        // stays on the SDK type but isn't surfaced here because the
        // hosted invoice page has a "Download PDF" button.
        let invoice_cell = match purchase_primary_link(purchase) {
            Some((PrimaryLinkKind::Invoice, url)) => {
                let label = purchase.invoice_number.as_deref().unwrap_or("Invoice");
                link_cell(label, url)
            }
            Some((PrimaryLinkKind::Receipt, url)) => link_cell("Receipt", url),
            Some((PrimaryLinkKind::Resume, url)) => link_cell("Resume", url),
            None => "-".to_string(),
        };

        builder.push_record([date.as_str(), amount.as_str(), status, &invoice_cell]);

        if matches!(purchase.status, CardPurchaseStatus::Completed) {
            total_paid_cents += purchase
                .paid_amount_cents
                .unwrap_or(purchase.requested_amount_cents);
        }
    }

    print_standard_table(builder.build());

    println!();
    println!("{}:", style("Total Card Payments").bold());
    println!("  {}", style(format_cents(total_paid_cents)).green());

    Ok(())
}

#[derive(Debug, Clone, Copy)]
enum PrimaryLinkKind {
    Invoice,
    Receipt,
    Resume,
}

/// Choose the single most informative link to surface per session.
///
/// Completed sessions land on the hosted invoice page when present —
/// it renders both the invoice detail and the payment confirmation,
/// so a separate receipt link would be redundant. Pending sessions
/// whose Stripe checkout has not yet expired fall through to a Resume
/// link so users can pick up the existing checkout instead of starting
/// a new session. A `None` `expires_at` is treated as resumable; only
/// a known-past timestamp disqualifies the row.
fn purchase_primary_link(
    purchase: &basilica_sdk::types::CardPurchaseSummary,
) -> Option<(PrimaryLinkKind, &str)> {
    if let Some(url) = purchase.hosted_invoice_url.as_deref() {
        return Some((PrimaryLinkKind::Invoice, url));
    }
    if let Some(url) = purchase.receipt_url.as_deref() {
        return Some((PrimaryLinkKind::Receipt, url));
    }
    if matches!(purchase.status, CardPurchaseStatus::Pending)
        && purchase.expires_at.is_none_or(|t| t > Utc::now())
    {
        return Some((PrimaryLinkKind::Resume, &purchase.checkout_url));
    }
    None
}

/// Build the table cell for a session's per-session link. The visible
/// label is styled blue + underlined (conventional web-link look), and
/// the OSC 8 escape wrapper makes it clickable on supporting terminals.
/// Width is calculated correctly because `tabled` is compiled with the
/// `ansi` feature, which strips both ANSI CSI (color/underline) and OSC
/// 8 escape bytes before measuring.
fn link_cell(label: &str, url: &str) -> String {
    let styled = style(label).blue().underlined().to_string();
    format!("\x1b]8;;{url}\x1b\\{styled}\x1b]8;;\x1b\\")
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
        print_standard_table(Table::new(&rows));
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

    rows.sort_by_key(|r| std::cmp::Reverse(r.started.clone()));

    println!(
        "{} ({} total)",
        style("Rental Usage History").bold(),
        style(history.total_count).cyan()
    );
    println!();
    print_standard_table(Table::new(&rows));
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
        #[tabled(rename = "Name")]
        name: String,
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
                name: rental.name.clone().unwrap_or_else(|| {
                    format!("id:{}", &rental.rental_id[..8.min(rental.rental_id.len())])
                }),
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

    rows.sort_by_key(|r| std::cmp::Reverse(r.started.clone()));

    print_standard_table(Table::new(&rows));

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
        #[tabled(rename = "Name")]
        name: String,
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
                .map(|gb| format!("{} GB", gb))
                .unwrap_or_else(|| "-".to_string());

            CpuHistoryRow {
                name: rental.name.clone().unwrap_or_else(|| {
                    format!("id:{}", &rental.rental_id[..8.min(rental.rental_id.len())])
                }),
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

    rows.sort_by_key(|r| std::cmp::Reverse(r.started.clone()));

    print_standard_table(Table::new(&rows));

    Ok(())
}

/// Display secure cloud rentals in table format
pub fn display_secure_cloud_rentals(
    rentals: &[&basilica_sdk::types::SecureCloudRentalListItem],
    output: ResolvedOutput,
) -> Result<()> {
    if rentals.is_empty() {
        println!("{}", style("No Citadel rentals found").yellow());
        return Ok(());
    }

    #[derive(Tabled)]
    struct SecureCloudRentalRow {
        #[tabled(rename = "Name")]
        name: String,
        #[tabled(rename = "Provider")]
        provider: String,
        #[tabled(rename = "GPU")]
        gpu: String,
        #[tabled(rename = "State")]
        status: String,
        #[tabled(rename = "IP")]
        ip: String,
        #[tabled(rename = "CPU/RAM")]
        cpu_ram: String,
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

            let cpu_ram = match (rental.vcpu_count, rental.system_memory_gb) {
                (Some(cores), Some(memory_gb)) => format!("{cores} cores / {memory_gb}GB"),
                _ => "—".to_string(),
            };

            // Use accumulated cost from billing service - no fallback
            let total_cost = rental
                .accumulated_cost
                .as_deref()
                .map(format_usd)
                .unwrap_or_else(|| "-".to_string());

            SecureCloudRentalRow {
                name: rental.name.clone(),
                provider: rental.provider.clone(),
                gpu: gpu_str,
                status: rental.status.clone(),
                ip: rental.ip_address.clone().unwrap_or_else(|| "-".to_string()),
                cpu_ram,
                region: rental
                    .location_code
                    .clone()
                    .unwrap_or_else(|| "-".to_string()),
                hourly_cost: format!("${:.2}/hr", rental.hourly_cost),
                total_cost,
                created: format_created(rental.created_at),
            }
        })
        .collect();

    #[derive(Tabled)]
    struct CompactRow {
        #[tabled(rename = "Name")]
        name: String,
        #[tabled(rename = "GPU")]
        gpu: String,
        #[tabled(rename = "State")]
        state: String,
        #[tabled(rename = "IP")]
        ip: String,
        #[tabled(rename = "Rate")]
        rate: String,
        #[tabled(rename = "Age")]
        age: String,
    }
    let context = RenderContext::stdout();
    let compact_rows = rentals.iter().map(|rental| {
        let gpu = if rental.gpu_count > 1 {
            format!("{}x {}", rental.gpu_count, rental.gpu_type.to_uppercase())
        } else {
            rental.gpu_type.to_uppercase()
        };
        CompactRow {
            name: rental.name.clone(),
            gpu: if rental.is_spot {
                format!("{gpu} (Spot)")
            } else {
                gpu
            },
            state: rental.status.clone(),
            ip: rental.ip_address.clone().unwrap_or_else(|| "—".to_string()),
            rate: format!("${:.2}/h", rental.hourly_cost),
            age: format_age(rental.created_at, context.now),
        }
    });
    let name_width = rentals
        .iter()
        .map(|rental| console::measure_text_width(&rental.name))
        .max();
    let rendered = render_table_with_context(
        Table::new(&rows),
        Some(Table::new(compact_rows)),
        output,
        context,
        name_width,
    );
    println!("{rendered}");

    Ok(())
}

/// Display CPU-only offerings in detailed table format
pub fn display_cpu_offerings_detailed(
    offerings: &[basilica_sdk::types::CpuOffering],
    output: ResolvedOutput,
) -> Result<()> {
    if offerings.is_empty() {
        println!("{}", style("No CPU instances available").yellow());
        return Ok(());
    }

    #[derive(Tabled)]
    struct CpuOfferingRow {
        #[tabled(rename = "PROVIDER")]
        provider: String,
        #[tabled(rename = "SIZE")]
        size: String,
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
            size: format!(
                "{} cores / {}GB",
                offering.vcpu_count, offering.system_memory_gb
            ),
            storage: if offering.storage_gb > 0 {
                format!("{} GB", offering.storage_gb)
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

    #[derive(Tabled)]
    struct CompactCpuOfferingRow {
        #[tabled(rename = "PROVIDER")]
        provider: String,
        #[tabled(rename = "SIZE")]
        size: String,
        #[tabled(rename = "REGION")]
        region: String,
        #[tabled(rename = "RATE")]
        rate: String,
    }
    let compact_rows = offerings.iter().map(|offering| CompactCpuOfferingRow {
        provider: offering.provider.clone(),
        size: format!(
            "{} cores / {}GB",
            offering.vcpu_count, offering.system_memory_gb
        ),
        region: offering.region.clone(),
        rate: format!(
            "${:.2}/h",
            offering.hourly_rate.parse::<f64>().unwrap_or(0.0)
        ),
    });
    let rendered = render_table_with_context(
        Table::new(&rows),
        Some(Table::new(compact_rows)),
        output,
        RenderContext::stdout(),
        None,
    );
    println!("{rendered}");

    println!("\nTotal Citadel (CPU) offerings: {}", offerings.len());

    Ok(())
}

/// Display CPU-only rentals in table format (no GPU column)
pub fn display_cpu_rentals(
    rentals: &[&basilica_sdk::types::SecureCloudRentalListItem],
    output: ResolvedOutput,
) -> Result<()> {
    if rentals.is_empty() {
        println!("{}", style("No CPU-only rentals found").yellow());
        return Ok(());
    }

    #[derive(Tabled)]
    struct CpuRentalRow {
        #[tabled(rename = "Name")]
        name: String,
        #[tabled(rename = "Provider")]
        provider: String,
        #[tabled(rename = "Size")]
        size: String,
        #[tabled(rename = "State")]
        status: String,
        #[tabled(rename = "IP")]
        ip: String,
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
            let size = match (rental.vcpu_count, rental.system_memory_gb) {
                (Some(cores), Some(memory_gb)) => format!("{cores} cores / {memory_gb}GB"),
                _ => "—".to_string(),
            };

            // Use accumulated cost from billing service
            let total_cost = rental
                .accumulated_cost
                .as_deref()
                .map(format_usd)
                .unwrap_or_else(|| "-".to_string());

            CpuRentalRow {
                name: rental.name.clone(),
                provider: rental.provider.clone(),
                size,
                status: rental.status.clone(),
                ip: rental.ip_address.clone().unwrap_or_else(|| "-".to_string()),
                region: rental
                    .location_code
                    .clone()
                    .unwrap_or_else(|| "-".to_string()),
                hourly_cost: format!("${:.2}/hr", rental.hourly_cost),
                total_cost,
                created: format_created(rental.created_at),
            }
        })
        .collect();

    #[derive(Tabled)]
    struct CompactRow {
        #[tabled(rename = "Name")]
        name: String,
        #[tabled(rename = "Size")]
        size: String,
        #[tabled(rename = "State")]
        state: String,
        #[tabled(rename = "IP")]
        ip: String,
        #[tabled(rename = "Rate")]
        rate: String,
        #[tabled(rename = "Age")]
        age: String,
    }
    let context = RenderContext::stdout();
    let compact_rows = rentals.iter().map(|rental| CompactRow {
        name: rental.name.clone(),
        size: match (rental.vcpu_count, rental.system_memory_gb) {
            (Some(cores), Some(memory_gb)) => format!("{cores} cores / {memory_gb}GB"),
            _ => "—".to_string(),
        },
        state: rental.status.clone(),
        ip: rental.ip_address.clone().unwrap_or_else(|| "—".to_string()),
        rate: format!("${:.2}/h", rental.hourly_cost),
        age: format_age(rental.created_at, context.now),
    });
    let name_width = rentals
        .iter()
        .map(|rental| console::measure_text_width(&rental.name))
        .max();
    let rendered = render_table_with_context(
        Table::new(&rows),
        Some(Table::new(compact_rows)),
        output,
        context,
        name_width,
    );
    println!("{rendered}");

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
                size: format!("{} GB", volume.size_gb),
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

    print_standard_table(Table::new(&rows));

    Ok(())
}

#[cfg(test)]
mod tests {
    use super::*;
    use basilica_sdk::types::{CardPurchaseStatus, CardPurchaseSummary};

    #[derive(Tabled)]
    struct WideTestRow {
        #[tabled(rename = "Name")]
        name: String,
        #[tabled(rename = "Provider")]
        provider: String,
        #[tabled(rename = "State")]
        state: String,
    }

    #[derive(Tabled)]
    struct CompactTestRow {
        #[tabled(rename = "Name")]
        name: String,
        #[tabled(rename = "State")]
        state: String,
    }

    fn render_test_table(output: ResolvedOutput, is_tty: bool, width: usize) -> String {
        let name = "training-rental-with-a-very-long-name".to_string();
        render_table_with_context(
            Table::new([WideTestRow {
                name: name.clone(),
                provider: "hyperstack".to_string(),
                state: "provisioning".to_string(),
            }]),
            Some(Table::new([CompactTestRow {
                name: name.clone(),
                state: "provisioning".to_string(),
            }])),
            output,
            RenderContext {
                is_tty,
                width: Some(width),
                now: DateTime::UNIX_EPOCH,
            },
            Some(console::measure_text_width(&name)),
        )
    }

    #[test]
    fn wide_layout_golden() {
        let rendered = render_test_table(ResolvedOutput::Wide, true, 40);
        assert_eq!(
            rendered,
            "┌───────────────────────────────────────┬────────────┬──────────────┐\n\
             │ Name                                  │ Provider   │ State        │\n\
             ├───────────────────────────────────────┼────────────┼──────────────┤\n\
             │ training-rental-with-a-very-long-name │ hyperstack │ provisioning │\n\
             └───────────────────────────────────────┴────────────┴──────────────┘"
        );
    }

    #[test]
    fn compact_layout_golden() {
        let rendered = render_test_table(ResolvedOutput::Compact, true, 100);
        assert_eq!(
            rendered,
            "┌───────────────────────────────────────┬──────────────┐\n\
             │ Name                                  │ State        │\n\
             ├───────────────────────────────────────┼──────────────┤\n\
             │ training-rental-with-a-very-long-name │ provisioning │\n\
             └───────────────────────────────────────┴──────────────┘"
        );
    }

    #[test]
    fn auto_layout_selects_compact_at_injected_width() {
        let rendered = render_test_table(ResolvedOutput::Auto, true, 55);
        assert!(!rendered.contains("Provider"));
        assert!(rendered.contains("State"));
        assert!(rendered
            .lines()
            .all(|line| console::measure_text_width(line) <= 55));
    }

    #[test]
    fn compact_layout_truncates_only_name_to_floor() {
        let rendered = render_test_table(ResolvedOutput::Compact, true, 20);
        assert_eq!(
            rendered,
            "┌──────────────┬──────────────┐\n\
             │ Name         │ State        │\n\
             ├──────────────┼──────────────┤\n\
             │ training-re… │ provisioning │\n\
             └──────────────┴──────────────┘"
        );
    }

    #[test]
    fn non_tty_auto_output_is_always_wide() {
        let rendered = render_test_table(ResolvedOutput::Auto, false, 20);
        assert!(rendered.contains("Provider"));
        assert!(rendered.contains("training-rental-with-a-very-long-name"));
    }

    #[test]
    fn relative_age_uses_injected_clock() {
        let now = DateTime::parse_from_rfc3339("2026-07-13T12:00:00Z")
            .unwrap()
            .with_timezone(&Utc);
        assert_eq!(format_age_str("2026-07-13T11:13:00Z", now), "47m");
        assert_eq!(format_age_str("2026-07-13T10:00:00Z", now), "2h");
        assert_eq!(format_age_str("2026-07-10T12:00:00Z", now), "3d");
        assert_eq!(format_age_str("2026-07-14T12:00:00Z", now), "0m");
    }

    #[test]
    fn image_names_drop_registry_tag_and_digest() {
        assert_eq!(
            compact_image_name("ghcr.io/one-covenant/trainer:latest"),
            "one-covenant/trainer"
        );
        assert_eq!(compact_image_name("nvidia/cuda:12.4"), "nvidia/cuda");
        assert_eq!(compact_image_name("ubuntu@sha256:abc"), "ubuntu");
    }

    #[test]
    fn ansi_and_osc8_cells_use_visible_width() {
        #[derive(Tabled)]
        struct Row {
            label: String,
        }
        let label =
            "\x1b]8;;https://example.test/invoice\x1b\\\x1b[34;4mINV-1\x1b[0m\x1b]8;;\x1b\\";
        let mut table = Table::new([Row {
            label: label.to_string(),
        }]);
        table.with(Style::modern());
        assert_eq!(table.total_width(), 9);
        assert!(table
            .to_string()
            .lines()
            .all(|line| tabled::grid::util::string::get_line_width(line) == 9));
    }

    fn purchase(
        hosted_invoice_url: Option<&str>,
        receipt_url: Option<&str>,
        invoice_number: Option<&str>,
    ) -> CardPurchaseSummary {
        CardPurchaseSummary {
            id: "cs_test".to_string(),
            status: CardPurchaseStatus::Completed,
            requested_amount_cents: 1000,
            paid_amount_cents: Some(1000),
            checkout_url: "https://checkout.stripe.test/cs_test".to_string(),
            created_at: None,
            completed_at: None,
            expires_at: None,
            receipt_url: receipt_url.map(String::from),
            invoice_id: hosted_invoice_url.map(|_| "in_abc".to_string()),
            invoice_number: invoice_number.map(String::from),
            hosted_invoice_url: hosted_invoice_url.map(String::from),
            // `invoice_pdf` intentionally `None` everywhere to confirm
            // the CLI never reaches for it — the public contract is that
            // the hosted invoice page carries the PDF link.
            invoice_pdf: None,
        }
    }

    #[test]
    fn invoice_is_preferred_when_both_urls_are_present() {
        // Common `invoice_creation = true` steady state: Stripe emits both
        // artifacts. The invoice page already renders the paid-charge
        // detail the receipt would duplicate, so the invoice wins.
        let purchase = purchase(
            Some("https://invoice.stripe.com/i/abc"),
            Some("https://pay.stripe.com/receipts/xyz"),
            Some("INV-0001"),
        );
        let (kind, url) = purchase_primary_link(&purchase).expect("both urls -> some link");
        assert!(matches!(kind, PrimaryLinkKind::Invoice));
        assert_eq!(url, "https://invoice.stripe.com/i/abc");
    }

    #[test]
    fn falls_back_to_receipt_when_invoice_is_absent() {
        // Sessions created before `invoice_creation` was flipped on (or
        // against a backend without it configured) have no invoice URLs
        // but still land `receipt_url` on `charge.succeeded`. The receipt
        // must surface so those rows are never dead-ends in the CLI.
        let purchase = purchase(None, Some("https://pay.stripe.com/receipts/xyz"), None);
        let (kind, url) = purchase_primary_link(&purchase).expect("receipt only -> some link");
        assert!(matches!(kind, PrimaryLinkKind::Receipt));
        assert_eq!(url, "https://pay.stripe.com/receipts/xyz");
    }

    #[test]
    fn invoice_only_is_rendered_as_invoice() {
        // Transient state where `invoice.finalized` has landed but
        // `charge.succeeded` has not yet been captured. The invoice page
        // already covers the user-facing need, so render it.
        let purchase = purchase(
            Some("https://invoice.stripe.com/i/abc"),
            None,
            Some("INV-0001"),
        );
        let (kind, _) = purchase_primary_link(&purchase).expect("invoice only -> some link");
        assert!(matches!(kind, PrimaryLinkKind::Invoice));
    }

    #[test]
    fn pending_purchase_with_no_artifacts_has_no_link() {
        // The shared `purchase()` helper hardcodes `status = Completed`,
        // so a row with no invoice/receipt artifacts collapses to `-`.
        // The Pending-row case is exercised by the Resume tests below.
        let purchase = purchase(None, None, None);
        assert!(purchase_primary_link(&purchase).is_none());
    }

    fn purchase_with(
        status: CardPurchaseStatus,
        expires_at: Option<chrono::DateTime<Utc>>,
        hosted_invoice_url: Option<&str>,
        receipt_url: Option<&str>,
    ) -> CardPurchaseSummary {
        CardPurchaseSummary {
            id: "cs_test".to_string(),
            status,
            requested_amount_cents: 1000,
            paid_amount_cents: None,
            checkout_url: "https://checkout.stripe.test/cs_test".to_string(),
            created_at: None,
            completed_at: None,
            expires_at,
            receipt_url: receipt_url.map(String::from),
            invoice_id: hosted_invoice_url.map(|_| "in_abc".to_string()),
            invoice_number: None,
            hosted_invoice_url: hosted_invoice_url.map(String::from),
            invoice_pdf: None,
        }
    }

    #[test]
    fn pending_with_future_expiry_resumes_to_checkout_url() {
        // Mirrors the website behavior: a pending session whose Stripe
        // checkout has not yet expired surfaces a Resume link pointing
        // at `checkout_url`.
        let future = Utc::now() + chrono::Duration::minutes(15);
        let purchase = purchase_with(CardPurchaseStatus::Pending, Some(future), None, None);
        let (kind, url) = purchase_primary_link(&purchase).expect("future expiry -> resume");
        assert!(matches!(kind, PrimaryLinkKind::Resume));
        assert_eq!(url, "https://checkout.stripe.test/cs_test");
    }

    #[test]
    fn pending_with_no_expiry_is_resumable() {
        // `expires_at == None` should not block resumption — only a
        // known-past timestamp disqualifies the row.
        let purchase = purchase_with(CardPurchaseStatus::Pending, None, None, None);
        let (kind, _) = purchase_primary_link(&purchase).expect("no expiry -> resume");
        assert!(matches!(kind, PrimaryLinkKind::Resume));
    }

    #[test]
    fn pending_with_past_expiry_has_no_link() {
        // The session is technically `Pending` until Stripe's webhook
        // flips it to `Expired`, but the checkout URL no longer works.
        // Render `-` instead of a broken Resume link.
        let past = Utc::now() - chrono::Duration::minutes(1);
        let purchase = purchase_with(CardPurchaseStatus::Pending, Some(past), None, None);
        assert!(purchase_primary_link(&purchase).is_none());
    }

    #[test]
    fn invoice_wins_over_resume_for_pending_with_invoice_url() {
        // Defensive: in the unlikely case a pending row already has an
        // invoice URL attached, the invoice link still wins so users
        // see the most informative artifact.
        let future = Utc::now() + chrono::Duration::minutes(15);
        let purchase = purchase_with(
            CardPurchaseStatus::Pending,
            Some(future),
            Some("https://invoice.stripe.com/i/abc"),
            None,
        );
        let (kind, url) = purchase_primary_link(&purchase).expect("invoice on pending -> invoice");
        assert!(matches!(kind, PrimaryLinkKind::Invoice));
        assert_eq!(url, "https://invoice.stripe.com/i/abc");
    }

    #[test]
    fn expired_status_has_no_link() {
        // Once Stripe (or the backend) has marked the session Expired,
        // the row is terminal and no link is meaningful.
        let purchase = purchase_with(CardPurchaseStatus::Expired, None, None, None);
        assert!(purchase_primary_link(&purchase).is_none());
    }

    #[test]
    fn unspecified_status_has_no_link() {
        let purchase = purchase_with(CardPurchaseStatus::Unspecified, None, None, None);
        assert!(purchase_primary_link(&purchase).is_none());
    }

    #[test]
    fn link_cell_wraps_label_in_osc8_and_ansi_styling() {
        // Guards the two terminal features we depend on:
        //   * OSC 8 hyperlink: `\x1b]8;;URL\x1b\` … `\x1b]8;;\x1b\`
        //   * ANSI blue + underline on the visible label so terminals
        //     without OSC 8 support still see a link-shaped cell.
        //
        // Force the `console` style machinery on — under `cargo test`
        // stdout is not a TTY, so `style(..).blue()` would otherwise
        // no-op and we would not observe the SGR bytes we actually emit
        // in real CLI runs.
        console::set_colors_enabled(true);

        let cell = link_cell("INV-1", "https://example.test/x");

        assert!(cell.starts_with("\x1b]8;;https://example.test/x\x1b\\"));
        assert!(cell.ends_with("\x1b]8;;\x1b\\"));
        assert!(cell.contains("INV-1"));
        // URL must not bleed into the visible portion.
        assert_eq!(cell.matches("https://example.test/x").count(), 1);
        // SGR style bytes (e.g. underline/color) must wrap the label.
        assert!(cell.contains("\x1b["));
    }
}
