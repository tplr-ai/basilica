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

    let grpc_addr = SocketAddr::new(state.config.server.host.parse()?, state.config.server.port);

    // In SPEC v1.6, executors are statically configured on the miner side
    // No dynamic registration is needed
    info!("Executor configured in static mode - no miner registration required");

    let server = ExecutorServer::new(state);

    info!("Starting Basilca Executor on {}", grpc_addr);

    tokio::select! {
        result = server.serve(grpc_addr) => {
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
