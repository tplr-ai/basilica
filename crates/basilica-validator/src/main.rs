#![allow(dead_code)]

//! # Basilca Validator
//!
//! Bittensor neuron for verifying and scoring miners/nodes.

use anyhow::Result;
use clap::Parser;

mod api;
mod bittensor_core;
mod cli;
mod collateral;
mod config;
mod gpu;
mod journal;
mod metrics;
mod miner_prover;
mod os_process;
mod persistence;
mod rental;
mod service;
mod ssh;

use cli::Args;

#[tokio::main]
async fn main() -> Result<()> {
    let args = Args::parse();

    // Initialize logging using the unified system
    let binary_name = env!("CARGO_BIN_NAME").replace("-", "_");
    let default_filter = format!("{}=info", binary_name);
    basilica_common::logging::init_logging(&args.verbosity, &binary_name, &default_filter)?;

    args.run().await
}
