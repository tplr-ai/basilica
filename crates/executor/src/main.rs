//! # Basilca Executor
//!
//! Agent software that runs on GPU-providing machines.
//! Responsible for secure task execution, system profiling,
//! and container management for the Basilca network.

use anyhow::Result;
use std::net::SocketAddr;
use tokio::signal;
use tracing::{error, info};
use tracing_subscriber::{layer::SubscriberExt, util::SubscriberInitExt, EnvFilter};

use executor::cli::{execute_command, AppConfig, AppConfigResolver, CliContext, ExecutorArgs};
use executor::grpc_server::ExecutorServer;
use executor::{ExecutorConfig, ExecutorState};

#[tokio::main]
async fn main() -> Result<()> {
    let args = ExecutorArgs::parse_args();

    let config = AppConfigResolver::resolve(&args)?;

    run_app(config).await
}

async fn run_app(config: AppConfig) -> Result<()> {
    match config {
        AppConfig::ConfigGeneration(config) => run_config_generation(config).await,
        AppConfig::Server(config) => run_server_mode(config).await,
        AppConfig::Cli(config) => run_cli_mode(config).await,
        AppConfig::Help => run_help_mode().await,
    }
}

async fn run_config_generation(config: executor::cli::args::ConfigGenConfig) -> Result<()> {
    info!("Generating configuration file: {}", config.output_path);

    let executor_config = ExecutorConfig::default();
    let toml_content = toml::to_string_pretty(&executor_config)?;
    std::fs::write(&config.output_path, toml_content)?;

    println!("Generated configuration file: {}", config.output_path);
    Ok(())
}

async fn run_server_mode(config: executor::cli::args::ServerConfig) -> Result<()> {
    init_logging(&config.log_level)?;

    let executor_config = load_config(&config.config_path)?;
    info!("Loaded configuration from: {}", config.config_path);

    if config.metrics_enabled {
        init_metrics(config.metrics_addr).await?;
        info!("Metrics server started on: {}", config.metrics_addr);
    }

    let state = ExecutorState::new(executor_config).await?;

    if let Err(e) = state.health_check().await {
        error!("Initial health check failed: {}", e);
        return Err(e);
    }

    let listen_addr = SocketAddr::new(state.config.server.host.parse()?, state.config.server.port);
    let advertised_grpc_endpoint = state.config.get_advertised_grpc_endpoint();
    let advertised_ssh_endpoint = state.config.get_advertised_ssh_endpoint();
    let advertised_health_endpoint = state.config.get_advertised_health_endpoint();

    info!("Starting Basilca Executor with advertised address support:");
    info!("  Internal binding: {}", listen_addr);
    info!("  Advertised gRPC endpoint: {}", advertised_grpc_endpoint);
    info!("  Advertised SSH endpoint: {}", advertised_ssh_endpoint);
    info!(
        "  Advertised health endpoint: {}",
        advertised_health_endpoint
    );
    info!(
        "  Address separation: {}",
        state.config.server.has_address_separation()
    );

    // Validate advertised endpoint configuration
    if let Err(e) = state.config.validate_advertised_endpoints() {
        error!("Invalid advertised endpoint configuration: {}", e);
        return Err(anyhow::anyhow!("Configuration validation failed: {}", e));
    }

    // In SPEC v1.6, executors are statically configured on the miner side
    // Register with miner for discovery using advertised endpoints
    register_with_miner(&state.config).await?;

    let server = ExecutorServer::new(state);

    info!("Starting Basilca Executor server on {}", listen_addr);

    tokio::select! {
        result = server.serve(listen_addr) => {
            if let Err(e) = result {
                error!("gRPC server error: {}", e);
                return Err(e);
            }
        }
        _ = signal::ctrl_c() => {
            info!("Received shutdown signal, stopping executor...");
        }
    }

    info!("Basilca Executor stopped");
    Ok(())
}

/// Register executor's advertised endpoint with miner
async fn register_with_miner(config: &ExecutorConfig) -> Result<()> {
    let advertised_endpoint = config.get_advertised_grpc_endpoint();

    info!(
        "Registering executor advertised endpoint with miner: {}",
        advertised_endpoint
    );

    // Implementation would depend on miner-executor communication protocol
    // This could involve:
    // 1. gRPC call to miner's registration endpoint
    // 2. Configuration file update
    // 3. Service discovery registration

    // For now, just log the endpoints that would be registered
    info!("Advertised endpoints registered:");
    info!("  gRPC: {}", config.get_advertised_grpc_endpoint());
    info!("  SSH: {}", config.get_advertised_ssh_endpoint());
    info!("  Health: {}", config.get_advertised_health_endpoint());

    Ok(())
}

async fn run_cli_mode(config: executor::cli::args::CliConfig) -> Result<()> {
    init_logging("info")?;

    if let Some(command) = config.command {
        let context = CliContext::new(config.config_path);
        execute_command(command, &context).await
    } else {
        eprintln!("No command provided. Use --help for available commands or --server to start server mode.");
        Ok(())
    }
}

async fn run_help_mode() -> Result<()> {
    eprintln!("Use --help for available commands");
    Ok(())
}

fn init_logging(level: &str) -> Result<()> {
    let filter = EnvFilter::try_from_default_env().unwrap_or_else(|_| EnvFilter::new(level));

    tracing_subscriber::registry()
        .with(filter)
        .with(
            tracing_subscriber::fmt::layer()
                .with_target(false)
                .with_thread_ids(true)
                .with_file(true)
                .with_line_number(true)
                .compact(),
        )
        .init();

    Ok(())
}

fn load_config(config_path: &str) -> Result<ExecutorConfig> {
    use std::path::PathBuf;

    let path = PathBuf::from(config_path);
    let config = if path.exists() {
        ExecutorConfig::load_from_file(&path)?
    } else {
        ExecutorConfig::load()?
    };

    Ok(config)
}

async fn init_metrics(addr: SocketAddr) -> Result<()> {
    use metrics_exporter_prometheus::PrometheusBuilder;

    let builder = PrometheusBuilder::new();
    builder
        .with_http_listener(addr)
        .install()
        .map_err(|e| anyhow::anyhow!("Failed to install metrics exporter: {}", e))?;

    Ok(())
}
