//! Basilica TUI - Terminal User Interface for GPU compute marketplace
//!
//! A unified TUI for both end-users (GPU rentals, deployments) and miners
//! (fleet management, validator monitoring).

use anyhow::Result;
use clap::Parser;
use clap_verbosity_flag::Verbosity;

mod actions;
mod app;
mod config;
mod data;
mod events;
mod ui;

use app::App;

/// Basilica TUI - Terminal User Interface
#[derive(Parser, Debug)]
#[command(
    name = "basilica-tui",
    author = "Basilica Team",
    version,
    about = "Terminal UI for Basilica GPU compute marketplace"
)]
pub struct Args {
    /// Start in miner mode
    #[arg(short, long)]
    pub miner: bool,

    /// Configuration file path
    #[arg(short, long)]
    pub config: Option<std::path::PathBuf>,

    #[command(flatten)]
    pub verbosity: Verbosity,

    /// Tick rate in milliseconds
    #[arg(long, default_value = "250")]
    pub tick_rate: u64,
}

#[tokio::main]
async fn main() -> Result<()> {
    let args = Args::parse();

    // Initialize logging
    let binary_name = env!("CARGO_BIN_NAME").replace('-', "_");
    let default_filter = format!("{}=warn", binary_name);
    basilica_common::logging::init_logging(&args.verbosity, &binary_name, &default_filter)?;

    // Load configuration
    let config = config::TuiConfig::load(args.config.as_deref())?;

    // Create and run the application
    let mut app = App::new(config, args.miner, args.tick_rate).await?;
    app.run().await?;

    Ok(())
}
