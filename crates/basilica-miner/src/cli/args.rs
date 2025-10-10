use std::path::PathBuf;

use super::Commands;
use clap::Parser;
use clap_verbosity_flag::{InfoLevel, Verbosity};

#[derive(Parser, Debug)]
#[command(author, version, about = "Basilica Miner - Bittensor neuron managing node fleets", long_about = None)]
pub struct Args {
    /// Configuration file path
    #[arg(short, long, default_value = "miner.toml")]
    pub config: PathBuf,

    #[command(flatten)]
    pub verbosity: Verbosity<InfoLevel>,

    /// Enable prometheus metrics endpoint
    #[arg(long)]
    pub metrics: bool,

    /// Generate sample configuration file
    #[arg(long)]
    pub gen_config: bool,

    /// Subcommands for CLI operations
    #[command(subcommand)]
    pub command: Option<Commands>,
}
