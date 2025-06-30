//! CLI interface for Basilica Executor

use anyhow::Result;
use async_trait::async_trait;
use clap::{Parser, Subcommand};

pub mod args;
pub mod commands;
pub mod handlers;

use commands::*;

// Re-export key types for convenience
pub use args::{AppConfig, AppConfigResolver, ApplicationMode, ExecutorArgs};

#[async_trait]
pub trait CliCommand {
    async fn execute(&self, context: &CliContext) -> Result<()>;
}

pub struct CliContext {
    pub config_path: String,
}

impl CliContext {
    pub fn new(config_path: String) -> Self {
        Self { config_path }
    }
}

#[derive(Parser, Debug)]
#[command(name = "executor")]
#[command(about = "Basilica Executor CLI - Manage and control executor capabilities")]
pub struct ExecutorCli {
    #[arg(short, long, default_value = "executor.toml")]
    pub config: String,

    #[command(subcommand)]
    pub command: Commands,
}

#[derive(Subcommand, Debug, Clone)]
pub enum Commands {
    #[command(subcommand)]
    Validator(ValidatorCommands),
    #[command(subcommand)]
    System(SystemCommands),
    #[command(subcommand)]
    Container(ContainerCommands),
    #[command(subcommand)]
    Resource(ResourceCommands),
    #[command(subcommand)]
    Network(NetworkCommands),
    #[command(subcommand)]
    Service(ServiceCommands),
}

pub async fn execute_cli() -> Result<()> {
    let cli = ExecutorCli::parse();
    let context = CliContext::new(cli.config);

    execute_command(cli.command, &context).await
}

pub async fn execute_command(command: Commands, context: &CliContext) -> Result<()> {
    match command {
        Commands::Validator(cmd) => cmd.execute(context).await,
        Commands::System(cmd) => cmd.execute(context).await,
        Commands::Container(cmd) => cmd.execute(context).await,
        Commands::Resource(cmd) => cmd.execute(context).await,
        Commands::Network(cmd) => cmd.execute(context).await,
        Commands::Service(cmd) => cmd.execute(context).await,
    }
}

#[async_trait]
impl CliCommand for Commands {
    async fn execute(&self, context: &CliContext) -> Result<()> {
        match self {
            Commands::Validator(cmd) => cmd.execute(context).await,
            Commands::System(cmd) => cmd.execute(context).await,
            Commands::Container(cmd) => cmd.execute(context).await,
            Commands::Resource(cmd) => cmd.execute(context).await,
            Commands::Network(cmd) => cmd.execute(context).await,
            Commands::Service(cmd) => cmd.execute(context).await,
        }
    }
}
