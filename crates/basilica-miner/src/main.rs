//! # Basilca Miner
//!
//! Bittensor neuron that manages a fleet of nodes and serves
//! validator requests for GPU rental and computational challenges.

use anyhow::Result;
use basilica_common::identity::{Hotkey, MinerUid};
use basilica_common::node_identity::NodeId;
use basilica_common::types::GpuCategory;
use clap::Parser;
use collateral_contract::collaterals;
use collateral_contract::config::CollateralNetworkConfig;
use serde::Deserialize;
use std::path::Path;
use std::sync::Arc;
use tokio::signal;
use tokio::sync::watch;
use tracing::{error, info, warn};

mod bidding;
mod bittensor_core;
mod cli;
mod config;
mod metrics;
mod node_manager;
mod persistence;
mod registration_client;
mod validator_discovery;

use bidding::BidManager;
use bittensor_core::ChainRegistration;
use config::MinerConfig;
use node_manager::{NodeManager, RegisteredNode};
use persistence::RegistrationDb;
use registration_client::RegistrationClient;

use crate::cli::{Args, Commands};

/// Main miner state
pub struct MinerState {
    pub config: MinerConfig,
    pub miner_uid: MinerUid,
    pub chain_registration: ChainRegistration,
    pub registration_client: Arc<RegistrationClient>,
    pub registration_db: RegistrationDb,
    pub metrics: Option<metrics::MinerMetrics>,
    pub validator_discovery: validator_discovery::ValidatorDiscovery,
    pub bid_manager: Arc<BidManager>,
    pub node_manager: Arc<NodeManager>,
}

#[derive(Debug, Deserialize)]
struct AlphaPriceResponse {
    price: f64,
}

#[derive(Debug)]
struct CollateralRow {
    node: String,
    gpu: String,
    alpha: String,
    usd: String,
    status: String,
}

impl MinerState {
    /// Initialize miner state
    pub async fn new(config: MinerConfig, enable_metrics: bool) -> Result<Self> {
        info!("Initializing miner...");

        // Initialize metrics system if enabled
        let metrics = if enable_metrics && config.metrics.enabled {
            let miner_metrics = metrics::MinerMetrics::new(config.metrics.clone())?;
            Some(miner_metrics)
        } else {
            None
        };

        // Initialize persistence layer
        let registration_db = RegistrationDb::new(&config.database).await?;

        // Initialize node manager with SSH config
        let node_manager = Arc::new(NodeManager::new(config.ssh_session.clone()));

        // Register all configured nodes (keyed by host)
        info!(
            "Registering {} configured nodes",
            config.node_management.nodes.len()
        );
        for node_config in &config.node_management.nodes {
            match node_manager.register_node(node_config.clone()).await {
                Ok(_) => info!(
                    "Registered node at {}@{}:{}",
                    node_config.username, node_config.host, node_config.port
                ),
                Err(e) => error!(
                    "Failed to register node at {}@{}:{}: {}",
                    node_config.username, node_config.host, node_config.port, e
                ),
            }
        }

        // Verify at least one node is registered
        let registered_nodes = node_manager.list_nodes().await?;
        if registered_nodes.is_empty() {
            warn!("No nodes registered - miner will not be able to serve validators");
        } else {
            info!("Successfully registered {} nodes", registered_nodes.len());
        }

        // Initialize Bittensor chain registration
        let chain_registration = ChainRegistration::new(config.bittensor.clone()).await?;

        // Collateral status warning on startup (best-effort)
        let account_id = chain_registration.get_bittensor_service().get_account_id();
        let ss58_address = format!("{account_id}");
        if let Ok(hotkey) = Hotkey::new(ss58_address) {
            if let Err(err) = log_collateral_status(hotkey.as_str(), &registered_nodes).await {
                warn!("Failed to fetch collateral status: {}", err);
            }
        }

        // Initialize validator discovery
        let strategy: Box<dyn validator_discovery::AssignmentStrategy> = match config
            .validator_assignment
            .strategy
            .as_str()
        {
            "highest_stake" => Box::new(validator_discovery::HighestStakeAssignment),
            "fixed_assignment" => {
                let validator_hotkey = config
                    .validator_assignment
                    .validator_hotkey
                    .clone()
                    .expect("validator_hotkey is required for fixed_assignment strategy");
                Box::new(validator_discovery::FixedAssignment::new(validator_hotkey))
            }
            _ => {
                return Err(anyhow::anyhow!(
                    "Unknown assignment strategy '{}'. Valid strategies are: highest_stake, fixed_assignment",
                    config.validator_assignment.strategy
                ));
            }
        };

        let validator_discovery = validator_discovery::ValidatorDiscovery::new(
            chain_registration.get_bittensor_service(),
            node_manager.clone(),
            strategy,
            config.bittensor.common.netuid,
            config.bid_grpc_port,
        );

        // Initialize registration client for miner→validator communication
        let bittensor_service = chain_registration.get_bittensor_service();
        let registration_client = Arc::new(RegistrationClient::new(
            std::time::Duration::from_secs(30),
            node_manager.clone(),
            bittensor_service,
        ));

        // Initialize bid manager (owns registration lifecycle)
        let bid_manager = Arc::new(BidManager::new(
            config.bidding.clone(),
            node_manager.clone(),
            registration_client.clone(),
        ));

        // Use a placeholder UID that will be updated after chain registration
        let miner_uid = MinerUid::from(0);

        Ok(Self {
            config,
            miner_uid,
            chain_registration,
            registration_client,
            registration_db,
            metrics,
            validator_discovery,
            bid_manager,
            node_manager,
        })
    }

    /// Run health check on all components
    pub async fn health_check(&self) -> Result<()> {
        info!("Running miner health check...");

        // Check database connection
        self.registration_db.health_check().await?;

        info!("Miner components healthy");
        Ok(())
    }

    /// Start all miner services
    pub async fn start_services(&self) -> Result<()> {
        info!("Starting miner services...");

        // Start metrics server if enabled
        if let Some(ref metrics) = self.metrics {
            metrics.start_server().await?;
            info!("Miner metrics server started");
        }

        // Perform one-time chain registration (Bittensor network presence)
        self.chain_registration.register_startup().await?;

        // Log the discovered UID
        if let Some(uid) = self.chain_registration.get_discovered_uid().await {
            info!("Miner registered with discovered UID: {}", uid);
        } else {
            warn!("No UID discovered - miner may not be registered on chain");
        }

        let (shutdown_tx, shutdown_rx) = watch::channel(false);

        // One-shot validator discovery at startup
        let discovered = self.validator_discovery.run_discovery().await?;
        info!(
            grpc_endpoint = %discovered.grpc_endpoint,
            "Validator discovered, starting bid manager"
        );

        // Start bid manager (owns registration lifecycle: register, health checks)
        let bid_manager = self.bid_manager.clone();
        let bid_manager_shutdown_rx = shutdown_rx.clone();
        let grpc_endpoint = discovered.grpc_endpoint;
        let bid_manager_handle = tokio::spawn(async move {
            if let Err(e) = bid_manager
                .run(grpc_endpoint, bid_manager_shutdown_rx)
                .await
            {
                error!("BidManager error: {:#}", e);
            }
        });

        info!("All miner services started successfully");

        // Wait for shutdown signal
        tokio::select! {
            _ = signal::ctrl_c() => {
                info!("Received shutdown signal, stopping miner...");
            }
            _ = bid_manager_handle => {
                error!("BidManager stopped unexpectedly");
            }
        }

        let _ = shutdown_tx.send(true);

        Ok(())
    }
}

async fn log_collateral_status(miner_hotkey: &str, nodes: &[RegisteredNode]) -> Result<()> {
    if nodes.is_empty() {
        return Ok(());
    }

    let price = fetch_alpha_price().await.ok();
    if price.is_none() {
        warn!("Alpha price unavailable; collateral USD values will be omitted");
    }

    let hotkey = Hotkey::new(miner_hotkey.to_string())
        .map_err(|e| anyhow::anyhow!("invalid hotkey: {e}"))?;
    let account_id = hotkey
        .to_account_id()
        .map_err(|e| anyhow::anyhow!("hotkey conversion failed: {e}"))?;
    let account_bytes: &[u8] = account_id.as_ref();
    let hotkey_bytes: [u8; 32] = account_bytes
        .try_into()
        .map_err(|_| anyhow::anyhow!("hotkey bytes length mismatch"))?;

    let network_config = CollateralNetworkConfig::default();
    let mut rows = Vec::new();
    let mut actions = Vec::new();

    for node in nodes {
        let node_uuid = NodeId::new(&node.config.host)?.uuid;
        let amount = collaterals(hotkey_bytes, node_uuid.into_bytes(), &network_config).await?;
        let alpha_amount = alpha_from_wei(amount);
        // Parse to GpuCategory for consistent handling (validation done at config load time)
        let gpu_cat: GpuCategory = node.config.gpu_category.parse().unwrap();
        let min_usd = minimum_usd_per_gpu(&gpu_cat) * node.config.gpu_count as f64;
        let warning_threshold = min_usd * 1.5;
        let (status, action) = match price {
            Some(alpha_price) => {
                let usd = alpha_amount * alpha_price;
                if usd >= warning_threshold {
                    ("Sufficient", None)
                } else if usd >= min_usd {
                    let needed_usd = warning_threshold - usd;
                    let needed_alpha = needed_usd / alpha_price;
                    (
                        "Warning",
                        Some(format!(
                            "Deposit {} Alpha (~${}) to reach safe level",
                            format_alpha(needed_alpha),
                            format_usd(needed_usd)
                        )),
                    )
                } else {
                    let needed_usd = min_usd - usd;
                    let needed_alpha = needed_usd / alpha_price;
                    (
                        "Undercollateralized",
                        Some(format!(
                            "URGENT: Deposit {} Alpha (~${}) to reach minimum",
                            format_alpha(needed_alpha),
                            format_usd(needed_usd)
                        )),
                    )
                }
            }
            None => ("Unknown", Some("Alpha price unavailable".to_string())),
        };

        let usd_display = price
            .map(|alpha_price| format_usd(alpha_amount * alpha_price))
            .unwrap_or_else(|| "N/A".to_string());
        let gpu_label = format!("{} ({}x)", node.config.gpu_category, node.config.gpu_count);

        rows.push(CollateralRow {
            node: node.config.host.clone(),
            gpu: gpu_label,
            alpha: format_alpha(alpha_amount),
            usd: usd_display,
            status: status.to_string(),
        });

        if let Some(action) = action {
            actions.push(format!("{}: {}", node.config.host, action));
        }
    }

    let price_label = price.map(format_usd).unwrap_or_else(|| "N/A".to_string());
    let table = build_table(&["Node", "GPU", "Alpha", "USD Value", "Status"], &rows);
    info!(
        "COLLATERAL STATUS (Alpha price: ${})\n{}",
        price_label, table
    );

    if !actions.is_empty() {
        warn!(
            "WARNING: {} nodes have insufficient collateral",
            actions.len()
        );
        for action in actions {
            warn!("  -> {}", action);
        }
        info!("Use 'collateral-cli tx deposit' to add collateral");
    }

    Ok(())
}

async fn fetch_alpha_price() -> Result<f64> {
    let url = "https://api.taostats.io/alpha/price";
    let response = reqwest::get(url).await?.error_for_status()?;
    let payload: AlphaPriceResponse = response.json().await?;
    Ok(payload.price)
}

fn minimum_usd_per_gpu(gpu_category: &GpuCategory) -> f64 {
    match gpu_category {
        GpuCategory::H100 => 50.0,
        GpuCategory::A100 => 25.0,
        GpuCategory::H200 => 75.0,
        GpuCategory::B200 => 75.0,
        GpuCategory::B300 => 100.0,
        GpuCategory::Other(_) => 10.0,
    }
}

fn format_usd(value: f64) -> String {
    format!("{:.2}", value)
}

fn format_alpha(value: f64) -> String {
    format!("{:.2}", value)
}

fn alpha_from_wei(wei: alloy_primitives::U256) -> f64 {
    // TODO: Switch to fixed-point decimal to avoid precision loss for large values.
    let val = wei.to_string().parse::<f64>().unwrap_or(0.0);
    val / 1e18_f64
}

fn build_table(headers: &[&str], rows: &[CollateralRow]) -> String {
    let mut widths = vec![0usize; headers.len()];
    for (i, header) in headers.iter().enumerate() {
        widths[i] = header.len();
    }
    for row in rows {
        widths[0] = widths[0].max(row.node.len());
        widths[1] = widths[1].max(row.gpu.len());
        widths[2] = widths[2].max(row.alpha.len());
        widths[3] = widths[3].max(row.usd.len());
        widths[4] = widths[4].max(row.status.len());
    }

    let border = build_border(&widths);
    let mut out = String::new();
    out.push_str(&border);
    out.push('\n');
    out.push_str(&build_row(
        &[
            headers[0].to_string(),
            headers[1].to_string(),
            headers[2].to_string(),
            headers[3].to_string(),
            headers[4].to_string(),
        ],
        &widths,
    ));
    out.push('\n');
    out.push_str(&border);
    out.push('\n');
    for row in rows {
        out.push_str(&build_row(
            &[
                row.node.clone(),
                row.gpu.clone(),
                row.alpha.clone(),
                row.usd.clone(),
                row.status.clone(),
            ],
            &widths,
        ));
        out.push('\n');
    }
    out.push_str(&border);
    out
}

fn build_border(widths: &[usize]) -> String {
    let mut out = String::new();
    out.push('+');
    for width in widths {
        out.push_str(&"-".repeat(width + 2));
        out.push('+');
    }
    out
}

fn build_row(cells: &[String], widths: &[usize]) -> String {
    let mut out = String::new();
    out.push('|');
    for (idx, cell) in cells.iter().enumerate() {
        let padded = format!("{:width$}", cell, width = widths[idx]);
        out.push(' ');
        out.push_str(&padded);
        out.push(' ');
        out.push('|');
    }
    out
}

#[tokio::main]
async fn main() -> Result<()> {
    let args = Args::parse();

    // Generate config file if requested
    if args.gen_config {
        let config = MinerConfig::default();
        let toml_content = toml::to_string_pretty(&config)?;
        std::fs::write(&args.config, toml_content)?;
        println!("Generated configuration file: {}", args.config.display());
        return Ok(());
    }

    // Initialize logging using the unified system
    let binary_name = env!("CARGO_BIN_NAME").replace("-", "_");
    let default_filter = format!("{}=info", binary_name);
    basilica_common::logging::init_logging(&args.verbosity, &binary_name, &default_filter)?;

    // Load configuration
    let config = load_config(&args.config)?;
    info!("Loaded configuration from: {}", args.config.display());

    // Handle CLI commands if provided
    if let Some(command) = args.command {
        return handle_cli_command(command, &config).await;
    }

    // Initialize miner state
    let state = MinerState::new(config, args.metrics).await?;

    // Run initial health check
    if let Err(e) = state.health_check().await {
        error!("Initial health check failed: {}", e);
        return Err(e);
    }

    info!("Starting Basilca Miner (UID: {})", state.miner_uid.as_u16());

    // Start all services
    state.start_services().await?;

    info!("Basilca Miner stopped");
    Ok(())
}

/// Handle CLI commands
async fn handle_cli_command(command: Commands, config: &MinerConfig) -> Result<()> {
    match command {
        Commands::Config { config_cmd } => cli::handle_config_command(config_cmd, config).await,
        Commands::Status => {
            let db = RegistrationDb::new(&config.database).await?;
            cli::show_miner_status(config, db).await
        }
        Commands::Migrate => {
            let mut db_config = config.database.clone();
            db_config.run_migrations = true;
            let _db = RegistrationDb::new(&db_config).await?;
            println!("Database migrations completed successfully");
            Ok(())
        }
    }
}

/// Load configuration from file and environment
fn load_config(config_path: &Path) -> Result<MinerConfig> {
    use basilica_common::config::ConfigValidation;

    let path = config_path;
    let config = if path.exists() {
        MinerConfig::load_from_file(path)?
    } else {
        MinerConfig::load()?
    };

    // Validate configuration before proceeding
    config.validate()?;

    // Log any warnings
    let warnings = config.warnings();
    if !warnings.is_empty() {
        warn!("Configuration warnings:");
        for warning in warnings {
            warn!("  - {}", warning);
        }
    }

    Ok(config)
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_minimum_usd_per_gpu_defaults() {
        assert_eq!(minimum_usd_per_gpu(&GpuCategory::H100), 50.0);
        assert_eq!(minimum_usd_per_gpu(&GpuCategory::A100), 25.0);
        assert_eq!(minimum_usd_per_gpu(&GpuCategory::H200), 75.0);
        assert_eq!(minimum_usd_per_gpu(&GpuCategory::B200), 75.0);
        assert_eq!(
            minimum_usd_per_gpu(&GpuCategory::Other("unknown".to_string())),
            10.0
        );
    }

    #[test]
    fn test_alpha_from_wei() {
        let amount = alloy_primitives::U256::from(1_000_000_000_000_000_000u128);
        let alpha = alpha_from_wei(amount);
        assert!((alpha - 1.0).abs() < 1e-6);
    }
}
