use crate::cli::{
    handlers::{rental, service},
    Command,
};
use clap::Parser;
use clap_verbosity_flag::{InfoLevel, Verbosity};
use std::path::PathBuf;

#[derive(Parser, Debug)]
#[command(name = "validator")]
#[command(about = "Basilica Validator - Bittensor neuron for verification and scoring")]
#[command(version)]
pub struct Args {
    #[command(subcommand)]
    pub command: Command,

    #[arg(short, long, global = true, default_value = "validator.toml")]
    pub config: PathBuf,

    #[command(flatten)]
    pub verbosity: Verbosity<InfoLevel>,

    #[arg(long, global = true)]
    pub dry_run: bool,
}

impl Args {
    pub async fn run(self) -> anyhow::Result<()> {
        match self.command {
            Command::Start => service::handle_start(self.config).await,
            Command::Stop => service::handle_stop().await,
            Command::Status => service::handle_status(self.config).await,
            Command::GenConfig { output } => service::handle_gen_config(output).await,

            Command::Rental { action, api_url } => {
                // Load configuration
                let config = crate::config::ValidatorConfig::load_from_file(&self.config)?;

                // Use the simplified API-based handler
                rental::handle_rental_command(action, &config, api_url).await
            }
        }
    }
}
