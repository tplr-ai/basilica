//! # CLI Module
//!
//! Complete command-line interface for miner operations with production-ready
//! operational commands for service management, database operations, and configuration.

use anyhow::Result;
use tracing::error;

use crate::config::MinerConfig;
use crate::persistence::RegistrationDb;

mod args;
mod commands;
pub mod handlers;

pub use args::*;
pub use commands::*;

/// Handle service management commands
pub async fn handle_service_command(command: ServiceCommand, config: &MinerConfig) -> Result<()> {
    let operation = match command {
        ServiceCommand::Start => handlers::ServiceOperation::Start,
        ServiceCommand::Stop => handlers::ServiceOperation::Stop,
        ServiceCommand::Restart => handlers::ServiceOperation::Restart,
        ServiceCommand::Status => handlers::ServiceOperation::Status,
        ServiceCommand::Reload => handlers::ServiceOperation::Reload,
    };

    handlers::handle_service_command(operation, config).await
}

/// Handle database management commands
pub async fn handle_database_command(command: DatabaseCommand, config: &MinerConfig) -> Result<()> {
    let operation = match command {
        DatabaseCommand::Backup { path } => handlers::DatabaseOperation::Backup { path },
        DatabaseCommand::Restore { path } => handlers::DatabaseOperation::Restore { path },
        DatabaseCommand::Stats => handlers::DatabaseOperation::Stats,
        DatabaseCommand::Vacuum => handlers::DatabaseOperation::Vacuum,
        DatabaseCommand::Integrity => handlers::DatabaseOperation::Integrity,
    };

    handlers::handle_database_command(operation, config).await
}

/// Handle configuration management commands
pub async fn handle_config_command(command: ConfigCommand, config: &MinerConfig) -> Result<()> {
    let operation = match command {
        ConfigCommand::Validate { path } => handlers::ConfigOperation::Validate { path },
        ConfigCommand::Show { show_sensitive } => {
            handlers::ConfigOperation::Show { show_sensitive }
        }
        ConfigCommand::Reload => handlers::ConfigOperation::Reload,
        ConfigCommand::Diff { other_path } => handlers::ConfigOperation::Diff { other_path },
        ConfigCommand::Export { format, path } => {
            let config_format = match format.as_str() {
                "json" => handlers::ConfigFormat::Json,
                "yaml" => handlers::ConfigFormat::Yaml,
                _ => handlers::ConfigFormat::Toml,
            };
            handlers::ConfigOperation::Export {
                format: config_format,
                path,
            }
        }
    };

    handlers::handle_config_command(operation, config).await
}

/// Show miner status
pub async fn show_miner_status(config: &MinerConfig, _db: RegistrationDb) -> Result<()> {
    println!("=== Basilca Miner Status ===");
    println!("Network: {}", config.bittensor.common.network);
    println!("Hotkey: {}", config.bittensor.common.hotkey_name);
    println!("Netuid: {}", config.bittensor.common.netuid);
    println!("Axon Port: {}", config.bittensor.axon_port);
    println!("Note: UID will be discovered from chain on startup");
    println!();

    // Show configured nodes (IDs will be auto-generated on startup)
    println!("Configured Nodes: {}", config.node_management.nodes.len());
    for node in &config.node_management.nodes {
        let node_endpoint = format!("{}:{}", node.host, node.port);
        println!(
            "  - ssh://{}@{} (ID will be auto-generated from credentials)",
            node.username, node_endpoint
        );
    }
    println!();

    // Connect to database to get stats
    let db = RegistrationDb::new(&config.database).await?;

    match db.health_check().await {
        Ok(_) => println!("Database: ✓ Connected"),
        Err(e) => {
            error!("Database connection failed: {}", e);
            println!("Database: ✗ Failed to connect");
        }
    }

    Ok(())
}
