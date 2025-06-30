//! # Basilca Validator
//!
//! Bittensor neuron for verifying and scoring miners/executors.

use anyhow::Result;

mod api;
mod bittensor_core;
mod cli;
mod config;
mod metrics;
mod miner_prover;
mod persistence;
mod ssh;
mod validation;

use cli::{Args, Cli};

#[tokio::main]
async fn main() -> Result<()> {
    tracing_subscriber::fmt::init();

    let args = Args::parse_args();
    let cli = Cli::new();

    cli.run(args).await
}
