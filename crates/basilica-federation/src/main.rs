//! Basilica Federation - Multi-cluster federation system

use basilica_federation::{FederationApi, FederationConfig};
use clap::Parser;
use tracing::{info, error};
use tracing_subscriber::{layer::SubscriberExt, util::SubscriberInitExt};

#[derive(Parser)]
#[command(name = "basilica-federation")]
#[command(about = "Basilica multi-cluster federation system")]
struct Args {
    /// Configuration file path
    #[arg(short, long, default_value = "federation.toml")]
    config: String,
    
    /// Log level
    #[arg(short, long, default_value = "info")]
    log_level: String,
}

#[tokio::main]
async fn main() -> anyhow::Result<()> {
    let args = Args::parse();
    
    // Initialize tracing
    tracing_subscriber::registry()
        .with(
            tracing_subscriber::EnvFilter::try_from_default_env()
                .unwrap_or_else(|_| tracing_subscriber::EnvFilter::new(&args.log_level)),
        )
        .with(tracing_subscriber::fmt::layer())
        .init();
    
    info!("Starting Basilica Federation");
    
    // Load configuration
    let config = if std::path::Path::new(&args.config).exists() {
        FederationConfig::load()
            .map_err(|e| anyhow::anyhow!("Failed to load config: {}", e))?
    } else {
        info!("Config file not found, using defaults");
        FederationConfig::default()
    };
    
    info!(
        federation = %config.name,
        clusters = config.clusters.len(),
        "Loaded federation configuration"
    );
    
    // Create and start federation API
    let federation_api = FederationApi::new(config).await
        .map_err(|e| anyhow::anyhow!("Failed to create federation API: {}", e))?;
    
    // Start the server
    if let Err(e) = federation_api.start().await {
        error!(error = %e, "Federation API failed");
        return Err(anyhow::anyhow!("Federation API error: {}", e));
    }
    
    Ok(())
}

