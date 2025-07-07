//! Executor identity-aware CLI handlers
//!
//! This module provides CLI command handlers that support the dual identifier
//! system (UUID + HUID) for executor management operations.

use anyhow::{anyhow, Context, Result};
use clap::{Args, Subcommand};
use common::executor_identity::{ExecutorIdentity, ExecutorIdentityDisplayExt, IdentityDisplay};
use tracing::info;

use crate::config::MinerConfig;
use crate::persistence::RegistrationDb;

/// Executor commands with UUID/HUID support
#[derive(Debug, Clone, Subcommand)]
pub enum ExecutorIdentityCommand {
    /// List executors with optional filtering
    List {
        /// Filter by UUID or HUID prefix (min 3 chars)
        #[clap(short, long)]
        filter: Option<String>,

        /// Show verbose output with UUID + HUID
        #[clap(short, long)]
        verbose: bool,

        /// Output format
        #[clap(short = 'o', long, value_enum, default_value = "table")]
        output: OutputFormat,
    },

    /// Show detailed information about an executor
    Show {
        /// Executor UUID or HUID prefix (min 3 chars)
        executor_id: String,

        /// Output format
        #[clap(short = 'o', long, value_enum, default_value = "text")]
        output: OutputFormat,
    },

    /// Assign an executor to a validator
    Assign {
        /// Executor UUID or HUID prefix (min 3 chars)
        executor_id: String,

        /// Validator address
        validator: String,
    },

    /// Remove an executor assignment
    Unassign {
        /// Executor UUID or HUID prefix (min 3 chars)
        executor_id: String,
    },

    /// Manage executor identity
    Identity(IdentitySubcommand),
}

/// Identity-specific subcommands
#[derive(Debug, Clone, Args)]
#[clap(about = "Manage executor identities")]
pub struct IdentitySubcommand {
    #[clap(subcommand)]
    pub command: IdentityOperation,
}

/// Identity operations
#[derive(Debug, Clone, Subcommand)]
pub enum IdentityOperation {
    /// Show current executor identity
    Show {
        /// Output format
        #[clap(short = 'o', long, value_enum, default_value = "text")]
        output: OutputFormat,
    },

    /// Generate a new identity (for testing)
    Generate {
        /// Number of identities to generate
        #[clap(default_value = "1")]
        count: usize,

        /// Output format
        #[clap(short = 'o', long, value_enum, default_value = "table")]
        output: OutputFormat,
    },

    /// Search for executors by identifier
    Search {
        /// Search query (UUID or HUID prefix)
        query: String,

        /// Show all matches even if ambiguous
        #[clap(short, long)]
        all: bool,
    },
}

/// Output format for commands
#[derive(Debug, Clone, Copy, PartialEq, clap::ValueEnum)]
pub enum OutputFormat {
    /// Human-readable table format
    Table,
    /// Plain text format
    Text,
    /// JSON format
    Json,
    /// Compact format (HUID only)
    Compact,
    /// Verbose format (UUID + HUID)
    Verbose,
}

/// Handle executor identity commands
pub async fn handle_executor_identity_command(
    cmd: ExecutorIdentityCommand,
    config: &MinerConfig,
    db: RegistrationDb,
) -> Result<()> {
    match cmd {
        ExecutorIdentityCommand::List {
            filter,
            verbose,
            output,
        } => list_executors_with_identity(filter, verbose, output, config, db).await,
        ExecutorIdentityCommand::Show {
            executor_id,
            output,
        } => show_executor_with_identity(executor_id, output, config, db).await,
        ExecutorIdentityCommand::Assign {
            executor_id,
            validator,
        } => assign_executor_with_identity(executor_id, validator, config, db).await,
        ExecutorIdentityCommand::Unassign { executor_id } => {
            unassign_executor_with_identity(executor_id, config, db).await
        }
        ExecutorIdentityCommand::Identity(subcmd) => {
            handle_identity_subcommand(subcmd.command, config, db).await
        }
    }
}

/// List executors with identity support
async fn list_executors_with_identity(
    filter: Option<String>,
    verbose: bool,
    output: OutputFormat,
    config: &MinerConfig,
    _db: RegistrationDb,
) -> Result<()> {
    info!("Listing executors with filter: {:?}", filter);

    // Get all configured executors
    let executors = &config.executor_management.executors;

    // Apply filter if provided
    let filtered_executors: Vec<_> = if let Some(query) = &filter {
        // Validate query length
        if query.len() < 3 && !is_valid_uuid(query) {
            return Err(anyhow!(
                "Filter must be a valid UUID or at least 3 characters for HUID prefix"
            ));
        }

        executors
            .iter()
            .filter(|e| {
                // For now, filter by legacy ID until full identity integration
                e.id.starts_with(query)
            })
            .collect()
    } else {
        executors.iter().collect()
    };

    // Format output based on requested format
    match output {
        OutputFormat::Table => {
            print_executor_table(&filtered_executors, verbose);
        }
        OutputFormat::Json => {
            print_executor_json(&filtered_executors)?;
        }
        OutputFormat::Compact => {
            for executor in filtered_executors {
                println!("{}", executor.id);
            }
        }
        OutputFormat::Verbose => {
            for executor in filtered_executors {
                println!("HUID: {}", executor.id);
                println!("Address: {}", executor.grpc_address);
                if let Some(name) = &executor.name {
                    println!("Name: {name}");
                }
                println!();
            }
        }
        _ => {
            print_executor_table(&filtered_executors, verbose);
        }
    }

    Ok(())
}

/// Show detailed executor information
async fn show_executor_with_identity(
    executor_id: String,
    output: OutputFormat,
    config: &MinerConfig,
    db: RegistrationDb,
) -> Result<()> {
    info!("Showing executor: {}", executor_id);

    // Find executor by identifier
    let executor = find_executor_by_identifier(&executor_id, config)
        .await
        .context("Failed to find executor")?;

    // Get health status from database
    let health_records = db.get_all_executor_health().await?;
    let health = health_records.iter().find(|h| h.executor_id == executor.id);

    // Format output
    match output {
        OutputFormat::Json => {
            let json = serde_json::json!({
                "id": executor.id,
                "grpc_address": executor.grpc_address,
                "name": executor.name,
                "metadata": executor.metadata,
                "health": health.map(|h| serde_json::json!({
                    "is_healthy": h.is_healthy,
                    "consecutive_failures": h.consecutive_failures,
                    "last_error": h.last_error,
                    "last_health_check": h.last_health_check,
                })),
            });
            println!("{}", serde_json::to_string_pretty(&json)?);
        }
        _ => {
            println!("Executor Details:");
            println!("  ID: {}", executor.id);
            println!("  Address: {}", executor.grpc_address);
            if let Some(name) = &executor.name {
                println!("  Name: {name}");
            }

            if let Some(h) = health {
                println!("\nHealth Status:");
                println!(
                    "  Status: {}",
                    if h.is_healthy { "Healthy" } else { "Unhealthy" }
                );
                println!("  Consecutive Failures: {}", h.consecutive_failures);
                if let Some(error) = &h.last_error {
                    println!("  Last Error: {error}");
                }
                if let Some(last_check) = &h.last_health_check {
                    println!(
                        "  Last Check: {}",
                        last_check.format("%Y-%m-%d %H:%M:%S UTC")
                    );
                }
            }
        }
    }

    Ok(())
}

/// Assign executor to validator
async fn assign_executor_with_identity(
    executor_id: String,
    validator: String,
    config: &MinerConfig,
    _db: RegistrationDb,
) -> Result<()> {
    info!(
        "Assigning executor {} to validator {}",
        executor_id, validator
    );

    // Find executor by identifier
    let executor = find_executor_by_identifier(&executor_id, config)
        .await
        .context("Failed to find executor")?;

    println!("Assigning {} to validator {}", executor.id, validator);

    // TODO: Implement actual assignment logic
    println!("Assignment operation would be performed here");

    Ok(())
}

/// Remove executor assignment
async fn unassign_executor_with_identity(
    executor_id: String,
    config: &MinerConfig,
    _db: RegistrationDb,
) -> Result<()> {
    info!("Unassigning executor {}", executor_id);

    // Find executor by identifier
    let executor = find_executor_by_identifier(&executor_id, config)
        .await
        .context("Failed to find executor")?;

    println!("Removing assignment for {}", executor.id);

    // TODO: Implement actual unassignment logic
    println!("Unassignment operation would be performed here");

    Ok(())
}

/// Handle identity-specific subcommands
async fn handle_identity_subcommand(
    cmd: IdentityOperation,
    _config: &MinerConfig,
    _db: RegistrationDb,
) -> Result<()> {
    match cmd {
        IdentityOperation::Show { output } => show_current_identity(output).await,
        IdentityOperation::Generate { count, output } => {
            generate_test_identities(count, output).await
        }
        IdentityOperation::Search { query, all } => search_executors_by_identity(query, all).await,
    }
}

/// Show current executor identity
async fn show_current_identity(output: OutputFormat) -> Result<()> {
    // TODO: Get actual identity from identity store
    // For now, show a mock identity

    match output {
        OutputFormat::Json => {
            println!(r#"{{"message": "Identity store integration pending"}}"#);
        }
        _ => {
            println!("Current executor identity:");
            println!("  Identity store integration pending");
        }
    }

    Ok(())
}

/// Generate test identities
async fn generate_test_identities(count: usize, output: OutputFormat) -> Result<()> {
    use common::executor_identity::ExecutorId;

    let mut identities = Vec::new();

    for i in 0..count {
        let id = ExecutorId::new().context(format!("Failed to generate identity {}", i + 1))?;
        identities.push(id);
    }

    match output {
        OutputFormat::Table => {
            println!("{:<20} {:<40} {:<10}", "HUID", "UUID", "Short UUID");
            println!("{}", "-".repeat(72));
            for id in &identities {
                println!(
                    "{:<20} {:<40} {:<10}",
                    id.huid(),
                    id.uuid(),
                    id.short_uuid()
                );
            }
        }
        OutputFormat::Json => {
            let json: Vec<_> = identities
                .iter()
                .map(|id| {
                    serde_json::json!({
                        "uuid": id.uuid().to_string(),
                        "huid": id.huid(),
                        "created_at": id.created_at()
                            .duration_since(std::time::UNIX_EPOCH)
                            .map(|d| d.as_secs())
                            .unwrap_or(0),
                    })
                })
                .collect();
            println!("{}", serde_json::to_string_pretty(&json)?);
        }
        _ => {
            for id in &identities {
                println!("{}", id.display().format_compact());
            }
        }
    }

    Ok(())
}

/// Search for executors by identity
async fn search_executors_by_identity(query: String, show_all: bool) -> Result<()> {
    // Validate query
    if query.len() < 3 && !is_valid_uuid(&query) {
        return Err(anyhow!(
            "Search query must be a valid UUID or at least 3 characters for HUID prefix"
        ));
    }

    // TODO: Implement actual search against identity store
    println!("Searching for executors matching '{query}'");

    if show_all {
        println!("Showing all matches (including ambiguous ones)");
    }

    println!("Search functionality pending identity store integration");

    Ok(())
}

/// Find executor by UUID or HUID identifier
async fn find_executor_by_identifier<'a>(
    identifier: &str,
    config: &'a MinerConfig,
) -> Result<&'a crate::config::ExecutorConfig> {
    // For now, use simple string matching on legacy IDs
    // This will be replaced with proper identity matching

    let executors = &config.executor_management.executors;

    // Try exact match first
    if let Some(executor) = executors.iter().find(|e| e.id == identifier) {
        return Ok(executor);
    }

    // Try prefix matching (min 3 chars)
    if identifier.len() >= 3 {
        let matches: Vec<_> = executors
            .iter()
            .filter(|e| e.id.starts_with(identifier))
            .collect();

        match matches.len() {
            0 => Err(anyhow!("No executor found matching '{}'", identifier)),
            1 => Ok(matches[0]),
            _n => {
                // Multiple matches - show disambiguation
                let ids: Vec<String> = matches.iter().map(|e| e.id.clone()).collect();
                Err(anyhow!(
                    "Multiple executors match '{}': {}",
                    identifier,
                    ids.join(", ")
                ))
            }
        }
    } else {
        Err(anyhow!(
            "Executor identifier must be at least 3 characters for prefix matching"
        ))
    }
}

/// Check if a string is a valid UUID
fn is_valid_uuid(s: &str) -> bool {
    uuid::Uuid::parse_str(s).is_ok()
}

/// Print executor table
fn print_executor_table(executors: &[&crate::config::ExecutorConfig], verbose: bool) {
    if executors.is_empty() {
        println!("No executors found");
        return;
    }

    if verbose {
        // Verbose format with more columns
        println!(
            "{:<40} {:<20} {:<30} {:<10}",
            "ID", "NAME", "ADDRESS", "STATUS"
        );
        println!("{}", "-".repeat(102));

        for executor in executors {
            println!(
                "{:<40} {:<20} {:<30} {:<10}",
                executor.id,
                executor.name.as_ref().unwrap_or(&"".to_string()),
                executor.grpc_address,
                "UNKNOWN" // Status would come from health checks
            );
        }
    } else {
        // Compact format - HUID only
        println!("{:<20} {:<10} {:<30}", "ID", "STATUS", "ADDRESS");
        println!("{}", "-".repeat(62));

        for executor in executors {
            println!(
                "{:<20} {:<10} {:<30}",
                executor.id,
                "UNKNOWN", // Status would come from health checks
                executor.grpc_address
            );
        }
    }
}

/// Print executor JSON
fn print_executor_json(executors: &[&crate::config::ExecutorConfig]) -> Result<()> {
    let json: Vec<_> = executors
        .iter()
        .map(|e| {
            serde_json::json!({
                "id": e.id,
                "grpc_address": e.grpc_address,
                "name": e.name,
                "metadata": e.metadata,
            })
        })
        .collect();

    println!("{}", serde_json::to_string_pretty(&json)?);
    Ok(())
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_is_valid_uuid() {
        assert!(is_valid_uuid("550e8400-e29b-41d4-a716-446655440000"));
        assert!(!is_valid_uuid("not-a-uuid"));
        assert!(!is_valid_uuid("swift-falcon-a3f2"));
    }

    #[test]
    fn test_output_format_parsing() {
        use clap::ValueEnum;

        assert_eq!(
            OutputFormat::from_str("table", false).unwrap(),
            OutputFormat::Table
        );
        assert_eq!(
            OutputFormat::from_str("json", false).unwrap(),
            OutputFormat::Json
        );
    }
}
