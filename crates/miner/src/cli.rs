//! # CLI Module
//!
//! Complete command-line interface for miner operations with production-ready
//! operational commands for service management, database operations, and configuration.

use anyhow::Result;
use clap::Subcommand;
use tracing::error;

use crate::config::MinerConfig;
use crate::persistence::RegistrationDb;

pub mod handlers;

/// Executor management subcommands
#[derive(Subcommand, Debug)]
pub enum ExecutorCommand {
    /// List all configured executors and their health status
    List,

    /// Show detailed information about an executor
    Show {
        /// Executor ID to show
        executor_id: String,
    },

    /// Show health status for all executors
    Health,

    /// Restart a specific executor
    Restart {
        /// Executor ID to restart
        executor_id: String,
    },

    /// View executor logs
    Logs {
        /// Executor ID to view logs for
        executor_id: String,
        /// Follow logs in real-time
        #[arg(short, long)]
        follow: bool,
        /// Number of recent lines to show
        #[arg(short, long)]
        lines: Option<usize>,
    },

    /// Connect directly to an executor
    Connect {
        /// Executor ID to connect to
        executor_id: String,
    },

    /// Run diagnostics on an executor
    Diagnostics {
        /// Executor ID to diagnose
        executor_id: String,
    },

    /// Ping an executor to test connectivity
    Ping {
        /// Executor ID to ping
        executor_id: String,
    },
}

/// Validator management subcommands
#[derive(Subcommand, Debug)]
pub enum ValidatorCommand {
    /// List recent validator interactions
    List {
        /// Number of recent interactions to show
        #[arg(short, long, default_value = "100")]
        limit: i64,
    },

    /// Show SSH access grants for a validator
    ShowAccess {
        /// Validator hotkey
        hotkey: String,
    },
}

/// Service management subcommands
#[derive(Subcommand, Debug)]
pub enum ServiceCommand {
    /// Start the miner service
    Start,

    /// Stop the miner service
    Stop,

    /// Restart the miner service
    Restart,

    /// Show service status
    Status,

    /// Reload service configuration
    Reload,
}

/// Database management subcommands
#[derive(Subcommand, Debug)]
pub enum DatabaseCommand {
    /// Backup the database
    Backup {
        /// Backup file path
        path: String,
    },

    /// Restore database from backup
    Restore {
        /// Backup file path to restore from
        path: String,
    },

    /// Show database statistics
    Stats,

    /// Clean up old database records
    Cleanup {
        /// Number of days to keep records (default: 30)
        #[arg(short, long, default_value = "30")]
        days: u32,
    },

    /// Vacuum database to reclaim space
    Vacuum,

    /// Check database integrity
    Integrity,
}

/// Configuration management subcommands
#[derive(Subcommand, Debug)]
pub enum ConfigCommand {
    /// Validate configuration file
    Validate {
        /// Configuration file path to validate (default: current config)
        #[arg(short, long)]
        path: Option<String>,
    },

    /// Show current configuration
    Show {
        /// Show sensitive fields (default: masked)
        #[arg(long)]
        show_sensitive: bool,
    },

    /// Reload configuration (test only)
    Reload,

    /// Compare configurations
    Diff {
        /// Path to configuration file to compare with
        other_path: String,
    },

    /// Export configuration in different formats
    Export {
        /// Export format (toml, json, yaml)
        #[arg(short, long, default_value = "toml")]
        format: String,
        /// Output file path
        path: String,
    },
}

/// Handle executor management commands
pub async fn handle_executor_command(
    command: ExecutorCommand,
    config: &MinerConfig,
    db: RegistrationDb,
) -> Result<()> {
    match command {
        ExecutorCommand::List => list_executor_health(db).await,
        ExecutorCommand::Show { executor_id } => show_executor_health(db, executor_id).await,
        ExecutorCommand::Health => show_all_executor_health(db).await,
        ExecutorCommand::Restart { executor_id } => {
            handlers::handle_enhanced_executor_command(
                handlers::ExecutorOperation::Restart { executor_id },
                config,
                db,
            )
            .await
        }
        ExecutorCommand::Logs {
            executor_id,
            follow,
            lines,
        } => {
            handlers::handle_enhanced_executor_command(
                handlers::ExecutorOperation::Logs {
                    executor_id,
                    follow,
                    lines,
                },
                config,
                db,
            )
            .await
        }
        ExecutorCommand::Connect { executor_id } => {
            handlers::handle_enhanced_executor_command(
                handlers::ExecutorOperation::Connect { executor_id },
                config,
                db,
            )
            .await
        }
        ExecutorCommand::Diagnostics { executor_id } => {
            handlers::handle_enhanced_executor_command(
                handlers::ExecutorOperation::Diagnostics { executor_id },
                config,
                db,
            )
            .await
        }
        ExecutorCommand::Ping { executor_id } => {
            handlers::handle_enhanced_executor_command(
                handlers::ExecutorOperation::Ping { executor_id },
                config,
                db,
            )
            .await
        }
    }
}

/// Handle validator management commands
pub async fn handle_validator_command(command: ValidatorCommand, db: RegistrationDb) -> Result<()> {
    match command {
        ValidatorCommand::List { limit } => list_validator_interactions(db, limit).await,
        ValidatorCommand::ShowAccess { hotkey } => show_validator_ssh_access(db, hotkey).await,
    }
}

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
        DatabaseCommand::Cleanup { days } => {
            handlers::DatabaseOperation::Cleanup { days: Some(days) }
        }
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
pub async fn show_miner_status(config: &MinerConfig) -> Result<()> {
    println!("=== Basilca Miner Status ===");
    println!("Miner UID: {}", config.bittensor.uid.as_u16());
    println!("Hotkey: {}", config.bittensor.common.hotkey_name);
    println!("Netuid: {}", config.bittensor.common.netuid);
    println!("Axon Port: {}", config.bittensor.axon_port);
    println!("Validator Comms: {:?}", config.validator_comms.auth.method);
    println!();

    // Show configured executors
    println!(
        "Configured Executors: {}",
        config.executor_management.executors.len()
    );
    for executor in &config.executor_management.executors {
        println!("  - {} @ {}", executor.id, executor.grpc_address);
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

    // Show health status summary
    if let Ok(health_records) = db.get_all_executor_health().await {
        let healthy_count = health_records.iter().filter(|h| h.is_healthy).count();
        println!(
            "Healthy Executors: {}/{}",
            healthy_count,
            health_records.len()
        );
    }

    Ok(())
}

/// List executor health status
async fn list_executor_health(db: RegistrationDb) -> Result<()> {
    let health_records = db.get_all_executor_health().await?;

    if health_records.is_empty() {
        println!("No executor health records found");
        return Ok(());
    }

    println!("=== Executor Health Status ===");
    println!(
        "{:<20} {:<10} {:<10} {:<20} Last Check",
        "Executor ID", "Healthy", "Failures", "Last Error"
    );
    println!("{}", "-".repeat(80));

    for record in health_records {
        let last_check = record
            .last_health_check
            .map(|t| t.format("%Y-%m-%d %H:%M:%S").to_string())
            .unwrap_or_else(|| "Never".to_string());

        let last_error = record.last_error.unwrap_or_else(|| "-".to_string());

        println!(
            "{:<20} {:<10} {:<10} {:<20} {}",
            record.executor_id,
            if record.is_healthy { "Yes" } else { "No" },
            record.consecutive_failures,
            if last_error.len() > 20 {
                &last_error[..20]
            } else {
                &last_error
            },
            last_check
        );
    }

    Ok(())
}

/// Show detailed health information for a specific executor
async fn show_executor_health(db: RegistrationDb, executor_id: String) -> Result<()> {
    let health_records = db.get_all_executor_health().await?;

    if let Some(record) = health_records.iter().find(|h| h.executor_id == executor_id) {
        println!("=== Executor Health Details ===");
        println!("Executor ID: {}", record.executor_id);
        println!("Healthy: {}", if record.is_healthy { "Yes" } else { "No" });
        println!("Consecutive Failures: {}", record.consecutive_failures);
        if let Some(last_check) = record.last_health_check {
            println!(
                "Last Health Check: {}",
                last_check.format("%Y-%m-%d %H:%M:%S")
            );
        } else {
            println!("Last Health Check: Never");
        }
        if let Some(error) = &record.last_error {
            println!("Last Error: {error}");
        }
        println!(
            "Updated At: {}",
            record.updated_at.format("%Y-%m-%d %H:%M:%S")
        );
    } else {
        println!("✗ Executor {executor_id} not found in health records");
    }

    Ok(())
}

/// Show all executor health status in a summary view
async fn show_all_executor_health(db: RegistrationDb) -> Result<()> {
    let health_records = db.get_all_executor_health().await?;

    if health_records.is_empty() {
        println!("No executor health records found");
        return Ok(());
    }

    let healthy_count = health_records.iter().filter(|h| h.is_healthy).count();
    let total_count = health_records.len();

    println!("=== Executor Fleet Health Summary ===");
    println!("Total Executors: {total_count}");
    println!(
        "Healthy: {} ({:.1}%)",
        healthy_count,
        (healthy_count as f64 / total_count as f64) * 100.0
    );
    println!(
        "Unhealthy: {} ({:.1}%)",
        total_count - healthy_count,
        ((total_count - healthy_count) as f64 / total_count as f64) * 100.0
    );
    println!();

    // Show unhealthy executors if any
    let unhealthy: Vec<_> = health_records.iter().filter(|h| !h.is_healthy).collect();
    if !unhealthy.is_empty() {
        println!("Unhealthy Executors:");
        for record in unhealthy {
            println!(
                "  - {} (failures: {})",
                record.executor_id, record.consecutive_failures
            );
            if let Some(error) = &record.last_error {
                println!("    Last error: {error}");
            }
        }
    }

    Ok(())
}

/// List recent validator interactions
async fn list_validator_interactions(db: RegistrationDb, limit: i64) -> Result<()> {
    let interactions = db.get_recent_validator_interactions(limit).await?;

    if interactions.is_empty() {
        println!("No validator interactions found");
        return Ok(());
    }

    println!("=== Recent Validator Interactions ===");
    println!(
        "{:<44} {:<20} {:<10} {:<20}",
        "Validator", "Type", "Success", "Time"
    );
    println!("{}", "-".repeat(100));

    for interaction in interactions {
        println!(
            "{:<44} {:<20} {:<10} {:<20}",
            interaction.validator_hotkey,
            interaction.interaction_type,
            if interaction.success { "Yes" } else { "No" },
            interaction.created_at.format("%Y-%m-%d %H:%M:%S")
        );

        if let Some(details) = &interaction.details {
            println!("  Details: {details}");
        }
    }

    Ok(())
}

/// Show SSH access grants for a validator
async fn show_validator_ssh_access(db: RegistrationDb, hotkey: String) -> Result<()> {
    let grants = db.get_active_ssh_grants(&hotkey).await?;

    if grants.is_empty() {
        println!("No active SSH access grants found for validator {hotkey}");
        return Ok(());
    }

    println!("=== SSH Access Grants for {hotkey} ===");
    println!(
        "{:<10} {:<30} {:<20} {:<10}",
        "Grant ID", "Executors", "Granted At", "Active"
    );
    println!("{}", "-".repeat(80));

    for grant in grants {
        let executor_ids: Vec<String> =
            serde_json::from_str(&grant.executor_ids).unwrap_or_default();
        let executors = executor_ids.join(", ");

        println!(
            "{:<10} {:<30} {:<20} {:<10}",
            grant.id,
            if executors.len() > 30 {
                &executors[..30]
            } else {
                &executors
            },
            grant.granted_at.format("%Y-%m-%d %H:%M:%S"),
            if grant.is_active { "Yes" } else { "No" }
        );

        if let Some(expires) = grant.expires_at {
            println!("  Expires: {}", expires.format("%Y-%m-%d %H:%M:%S"));
        }
    }

    Ok(())
}

// TODO: Implement the following for production readiness:
// 1. Interactive CLI mode with prompts and confirmations
// 2. Bulk operations for managing multiple executors
// 3. Export/import functionality for executor configurations
// 4. Advanced filtering and sorting options
// 5. Real-time monitoring dashboard in CLI
// 6. Configuration validation and dry-run modes
// 7. Audit trail for all CLI operations
// 8. Integration with external monitoring tools
