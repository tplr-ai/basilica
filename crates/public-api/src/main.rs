//! Main entry point for the Basilica Public API Gateway

use clap::Parser;
use public_api::{config::Config, server::Server, Result};
use std::path::PathBuf;
use tracing::{error, info};

#[derive(Parser)]
#[command(
    name = "public-api",
    about = "Basilica Public API Gateway",
    version,
    author
)]
struct Args {
    /// Path to configuration file
    #[arg(short, long, value_name = "FILE")]
    config: Option<PathBuf>,

    /// Generate example configuration file
    #[arg(long)]
    gen_config: bool,

    /// Enable debug logging
    #[arg(short, long)]
    debug: bool,
}

#[tokio::main]
async fn main() -> Result<()> {
    let args = Args::parse();

    // Initialize logging
    let log_level = if args.debug { "debug" } else { "info" };
    let filter = tracing_subscriber::EnvFilter::try_from_default_env()
        .unwrap_or_else(|_| tracing_subscriber::EnvFilter::new(log_level));

    tracing_subscriber::fmt()
        .with_env_filter(filter)
        .with_target(true)
        .with_thread_ids(true)
        .with_file(true)
        .with_line_number(true)
        .json()
        .init();

    info!(
        "Starting Basilica Public API Gateway v{}",
        public_api::VERSION
    );

    // Handle config generation
    if args.gen_config {
        let example_config = Config::generate_example()?;
        println!("{example_config}");
        return Ok(());
    }

    // Load configuration
    let config = Config::load(args.config.as_deref())?;
    info!(
        "Configuration loaded, binding to {}",
        config.server.bind_address
    );

    // Create and run server
    let server = Server::new(config).await?;

    info!("Public API Gateway initialized successfully");

    // Run until shutdown signal
    match server.run().await {
        Ok(()) => {
            info!("Public API Gateway shut down gracefully");
            Ok(())
        }
        Err(e) => {
            error!("Public API Gateway error: {}", e);
            Err(e)
        }
    }
}
