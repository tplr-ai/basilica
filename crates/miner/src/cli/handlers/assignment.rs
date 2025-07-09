//! # Assignment CLI Handlers
//!
//! Command handlers for executor assignment management

use crate::cli::AssignmentCommand;
use crate::config::MinerConfig;
use crate::persistence::AssignmentDb;
use crate::services::assignment_manager::{AssignmentManager, AssignmentSuggester};
use anyhow::{anyhow, Result};
use sqlx::SqlitePool;
use tracing::{error, info};

/// Handle assignment commands
pub async fn handle_assignment_command(
    command: &AssignmentCommand,
    config: &MinerConfig,
) -> Result<()> {
    // Create database pool
    let pool = SqlitePool::connect(&config.database.url).await?;
    let assignment_db = AssignmentDb::new(pool.clone());

    // Ensure migrations are run
    assignment_db.run_migrations().await?;

    // Extract executor IDs from config
    let executor_ids: Vec<String> = config
        .executor_management
        .executors
        .iter()
        .map(|e| e.id.clone())
        .collect();

    let assignment_manager = AssignmentManager::new(pool).with_executors(executor_ids);

    match command {
        AssignmentCommand::Assign {
            executor_id,
            validator_hotkey,
            notes,
        } => {
            info!(
                "Assigning executor {} to validator {}",
                executor_id, validator_hotkey
            );

            assignment_db
                .create_assignment(executor_id, validator_hotkey, "cli-user", notes.as_deref())
                .await?;

            println!(
                "Successfully assigned executor {executor_id} to validator {validator_hotkey}"
            );

            if let Some(notes) = notes {
                println!("  Notes: {notes}");
            }
        }

        AssignmentCommand::Unassign { executor_id } => {
            info!("Unassigning executor {}", executor_id);

            // Get current assignment to show what's being removed
            if let Some(assignment) = assignment_db
                .get_assignment_by_executor(executor_id)
                .await?
            {
                assignment_db
                    .delete_assignment(executor_id, "cli-user")
                    .await?;
                println!(
                    "Successfully unassigned executor {} from validator {}",
                    executor_id, assignment.validator_hotkey
                );
            } else {
                return Err(anyhow!(
                    "Executor {} is not currently assigned",
                    executor_id
                ));
            }
        }

        AssignmentCommand::List { validator } => {
            if let Some(validator_hotkey) = validator {
                // List assignments for specific validator
                let assignments = assignment_db
                    .get_assignments_for_validator(validator_hotkey)
                    .await?;

                if assignments.is_empty() {
                    println!("No executors assigned to validator {validator_hotkey}");
                } else {
                    println!("Executors assigned to validator {validator_hotkey}:");
                    println!("{:<20} {:<25} Notes", "Executor ID", "Assigned At");
                    println!("{}", "-".repeat(70));

                    for assignment in assignments {
                        println!(
                            "{:<20} {:<25} {}",
                            assignment.executor_id,
                            assignment.assigned_at.format("%Y-%m-%d %H:%M:%S"),
                            assignment.notes.unwrap_or_else(|| "-".to_string())
                        );
                    }
                }
            } else {
                // List all assignments
                let assignments = assignment_db.get_all_assignments().await?;

                if assignments.is_empty() {
                    println!("No executor assignments found");
                } else {
                    println!("All executor assignments:");
                    println!(
                        "{:<20} {:<50} {:<25} Notes",
                        "Executor ID", "Validator Hotkey", "Assigned At"
                    );
                    println!("{}", "-".repeat(120));

                    for assignment in assignments {
                        println!(
                            "{:<20} {:<50} {:<25} {}",
                            assignment.executor_id,
                            assignment.validator_hotkey,
                            assignment.assigned_at.format("%Y-%m-%d %H:%M:%S"),
                            assignment.notes.unwrap_or_else(|| "-".to_string())
                        );
                    }
                }
            }
        }

        AssignmentCommand::Coverage => {
            let stats = assignment_manager.get_coverage_stats().await?;

            println!("Assignment Coverage Statistics:");
            println!("━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━");
            println!("Total Executors:       {}", stats.total_executors);
            println!("Assigned Executors:    {}", stats.assigned_executors);
            println!(
                "Unassigned Executors:  {}",
                stats.total_executors - stats.assigned_executors
            );
            println!(
                "Covered Stake:         {:.2}%",
                stats.covered_stake_percentage
            );
            println!(
                "Covered Validators:    {}/{}",
                stats.covered_validators, stats.total_validators
            );

            // Coverage quality assessment
            if stats.covered_stake_percentage >= 50.0 {
                println!("Status:                Meeting minimum coverage requirement");
            } else {
                println!("Status:                WARNING: Below minimum 50% coverage requirement");
            }
        }

        AssignmentCommand::Suggest { min_coverage } => {
            let suggester = AssignmentSuggester::new(*min_coverage);

            let executors = assignment_manager.get_all_executors().await?;
            let stakes = assignment_manager.get_validator_stakes().await?;
            let assignments = assignment_manager.get_all_assignments().await?;

            if stakes.is_empty() {
                println!("WARNING: No validator stake data available. Please ensure the stake monitor service is running.");
                return Ok(());
            }

            let suggestions = suggester
                .suggest_assignments(executors, stakes, assignments)
                .await?;

            if suggestions.is_empty() {
                println!("All executors are optimally assigned. No suggestions needed.");
            } else {
                println!("Assignment Suggestions:");
                println!("━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━");
                println!(
                    "{:<10} {:<20} {:<50} Reason",
                    "Priority", "Executor", "Validator"
                );
                println!("{}", "-".repeat(140));

                for suggestion in suggestions {
                    let priority_str = match suggestion.priority {
                        crate::services::assignment_manager::AssignmentPriority::Critical => {
                            "CRITICAL"
                        }
                        crate::services::assignment_manager::AssignmentPriority::High => "HIGH",
                        crate::services::assignment_manager::AssignmentPriority::Medium => {
                            "MEDIUM"
                        }
                        crate::services::assignment_manager::AssignmentPriority::Low => "LOW",
                    };

                    println!(
                        "{:<10} {:<20} {:<50} {}",
                        priority_str,
                        suggestion.executor_id,
                        suggestion.validator_hotkey,
                        suggestion.reason
                    );
                }

                println!("\nTo apply a suggestion, use:");
                println!("  miner assignment assign <executor-id> <validator-hotkey>");
            }
        }

        AssignmentCommand::History { executor_id, limit } => {
            if let Some(executor_id) = executor_id {
                // Show history for specific executor
                let history = assignment_db.get_assignment_history(executor_id).await?;

                if history.is_empty() {
                    println!("No assignment history found for executor {executor_id}");
                } else {
                    println!("Assignment history for executor {executor_id}:");
                    println!(
                        "{:<15} {:<50} {:<25} By",
                        "Action", "Validator", "Performed At"
                    );
                    println!("{}", "-".repeat(110));

                    for record in history {
                        println!(
                            "{:<15} {:<50} {:<25} {}",
                            record.action,
                            record.validator_hotkey.unwrap_or_else(|| "-".to_string()),
                            record.performed_at.format("%Y-%m-%d %H:%M:%S"),
                            record.performed_by
                        );
                    }
                }
            } else {
                // Show recent history for all executors
                let history = assignment_db.get_recent_assignment_history(*limit).await?;

                if history.is_empty() {
                    println!("No assignment history found");
                } else {
                    println!("Recent assignment history (last {limit}):");
                    println!(
                        "{:<20} {:<15} {:<50} {:<25} By",
                        "Executor", "Action", "Validator", "Performed At"
                    );
                    println!("{}", "-".repeat(130));

                    for record in history {
                        println!(
                            "{:<20} {:<15} {:<50} {:<25} {}",
                            record.executor_id,
                            record.action,
                            record.validator_hotkey.unwrap_or_else(|| "-".to_string()),
                            record.performed_at.format("%Y-%m-%d %H:%M:%S"),
                            record.performed_by
                        );
                    }
                }
            }
        }

        AssignmentCommand::Stakes { min_stake } => {
            let stakes = assignment_db.get_all_validator_stakes().await?;

            if stakes.is_empty() {
                println!("WARNING: No validator stake data available. Please ensure the stake monitor service is running.");
                return Ok(());
            }

            let filtered_stakes: Vec<_> = stakes
                .into_iter()
                .filter(|s| min_stake.map_or(true, |min| s.stake_amount >= min))
                .collect();

            if filtered_stakes.is_empty() {
                println!(
                    "No validators found with minimum stake of {:.2} TAO",
                    min_stake.unwrap_or(0.0)
                );
            } else {
                println!(
                    "Validator Stakes{}:",
                    min_stake.map_or(String::new(), |min| format!(" (≥{min:.2} TAO)"))
                );
                println!(
                    "{:<50} {:<15} {:<15} Last Updated",
                    "Validator Hotkey", "Stake (TAO)", "Percentage"
                );
                println!("{}", "-".repeat(100));

                for stake in filtered_stakes {
                    println!(
                        "{:<50} {:<15.2} {:<15.2}% {}",
                        stake.validator_hotkey,
                        stake.stake_amount,
                        stake.percentage_of_total,
                        stake.last_updated.format("%Y-%m-%d %H:%M:%S")
                    );
                }
            }
        }

        AssignmentCommand::Export { path } => {
            let assignments = assignment_db.get_all_assignments().await?;

            let json = serde_json::to_string_pretty(&assignments)?;
            std::fs::write(path, json)?;
            println!("Exported {} assignments to {}", assignments.len(), path);
        }

        AssignmentCommand::Import { path, dry_run } => {
            let content = std::fs::read_to_string(path)?;
            let assignments: Vec<serde_json::Value> = serde_json::from_str(&content)?;

            let mut imported = 0;
            let mut errors = 0;

            for assignment in assignments {
                if let (Some(executor_id), Some(validator_hotkey)) = (
                    assignment.get("executor_id").and_then(|v| v.as_str()),
                    assignment.get("validator_hotkey").and_then(|v| v.as_str()),
                ) {
                    let notes = assignment.get("notes").and_then(|v| v.as_str());

                    if *dry_run {
                        println!("Would import: {executor_id} -> {validator_hotkey}");
                        imported += 1;
                    } else {
                        match assignment_db
                            .create_assignment(executor_id, validator_hotkey, "import", notes)
                            .await
                        {
                            Ok(_) => imported += 1,
                            Err(e) => {
                                error!(
                                    "Failed to import assignment {}->{}: {}",
                                    executor_id, validator_hotkey, e
                                );
                                errors += 1;
                            }
                        }
                    }
                } else {
                    error!("Invalid assignment format: missing executor_id or validator_hotkey");
                    errors += 1;
                }
            }

            if *dry_run {
                println!(
                    "Dry run complete: {imported} assignments would be imported, {errors} errors"
                );
            } else {
                println!("Import complete: {imported} assignments imported, {errors} errors");
            }
        }
    }

    Ok(())
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::config::{ExecutorConfig, ExecutorManagementConfig, MinerConfig};
    use crate::persistence::AssignmentDb;
    use sqlx::SqlitePool;
    use std::time::Duration;

    async fn setup_test_config() -> Result<(MinerConfig, SqlitePool)> {
        let pool = SqlitePool::connect("sqlite::memory:").await?;
        let assignment_db = AssignmentDb::new(pool.clone());
        assignment_db.run_migrations().await?;

        let mut config = MinerConfig::default();
        config.database.url = "sqlite::memory:".to_string();
        config.executor_management = ExecutorManagementConfig {
            executors: vec![
                ExecutorConfig {
                    id: "test-exec-1".to_string(),
                    grpc_address: "127.0.0.1:50051".to_string(),
                    name: Some("Test Executor 1".to_string()),
                    metadata: None,
                },
                ExecutorConfig {
                    id: "test-exec-2".to_string(),
                    grpc_address: "127.0.0.1:50052".to_string(),
                    name: Some("Test Executor 2".to_string()),
                    metadata: None,
                },
            ],
            health_check_interval: Duration::from_secs(60),
            health_check_timeout: Duration::from_secs(10),
            max_retry_attempts: 3,
            auto_recovery: true,
        };

        Ok((config, pool))
    }

    #[tokio::test]
    async fn test_assign_command() -> Result<()> {
        let (_config, pool) = setup_test_config().await?;
        let assignment_db = AssignmentDb::new(pool);

        // Add some validator stakes for testing
        assignment_db
            .update_validator_stake("test-validator", 1000.0, 50.0)
            .await?;

        // Test assignment creation directly through the database
        // (CLI command test would require mocking the database connection)
        assignment_db
            .create_assignment(
                "test-exec-1",
                "test-validator",
                "cli-user",
                Some("Test assignment"),
            )
            .await?;

        // Verify assignment was created
        let assignment = assignment_db
            .get_assignment_by_executor("test-exec-1")
            .await?;
        assert!(assignment.is_some());
        let assignment = assignment.unwrap();
        assert_eq!(assignment.validator_hotkey, "test-validator");
        assert_eq!(assignment.notes, Some("Test assignment".to_string()));
        assert_eq!(assignment.assigned_by, "cli-user");

        Ok(())
    }

    #[tokio::test]
    async fn test_unassign_command() -> Result<()> {
        let (_config, pool) = setup_test_config().await?;
        let assignment_db = AssignmentDb::new(pool);

        // Create initial assignment
        assignment_db
            .create_assignment("test-exec-1", "test-validator", "test", None)
            .await?;

        // Test assignment deletion directly
        assignment_db
            .delete_assignment("test-exec-1", "cli-user")
            .await?;

        // Verify assignment was removed
        let assignment = assignment_db
            .get_assignment_by_executor("test-exec-1")
            .await?;
        assert!(assignment.is_none());

        Ok(())
    }

    #[tokio::test]
    async fn test_list_command_all() -> Result<()> {
        let (_config, pool) = setup_test_config().await?;
        let assignment_db = AssignmentDb::new(pool);

        // Create test assignments
        assignment_db
            .create_assignment("test-exec-1", "validator-1", "test", None)
            .await?;
        assignment_db
            .create_assignment("test-exec-2", "validator-2", "test", Some("Test note"))
            .await?;

        // Test listing all assignments
        let assignments = assignment_db.get_all_assignments().await?;
        assert_eq!(assignments.len(), 2);

        let exec1_assignment = assignments
            .iter()
            .find(|a| a.executor_id == "test-exec-1")
            .unwrap();
        assert_eq!(exec1_assignment.validator_hotkey, "validator-1");
        assert_eq!(exec1_assignment.notes, None);

        let exec2_assignment = assignments
            .iter()
            .find(|a| a.executor_id == "test-exec-2")
            .unwrap();
        assert_eq!(exec2_assignment.validator_hotkey, "validator-2");
        assert_eq!(exec2_assignment.notes, Some("Test note".to_string()));

        Ok(())
    }

    #[tokio::test]
    async fn test_coverage_command() -> Result<()> {
        let (_config, pool) = setup_test_config().await?;
        let assignment_db = AssignmentDb::new(pool);

        // Add validator stakes
        assignment_db
            .update_validator_stake("validator-1", 1000.0, 60.0)
            .await?;
        assignment_db
            .update_validator_stake("validator-2", 500.0, 40.0)
            .await?;

        // Add assignment
        assignment_db
            .create_assignment("test-exec-1", "validator-1", "test", None)
            .await?;

        // Test coverage calculation
        let stats = assignment_db.get_coverage_stats(2).await?; // 2 total executors in config
        assert_eq!(stats.total_executors, 2);
        assert_eq!(stats.assigned_executors, 1);
        assert_eq!(stats.covered_stake_percentage, 60.0);
        assert_eq!(stats.covered_validators, 1);
        assert_eq!(stats.total_validators, 2);

        Ok(())
    }

    #[tokio::test]
    async fn test_suggest_command() -> Result<()> {
        let (_config, pool) = setup_test_config().await?;
        let assignment_db = AssignmentDb::new(pool);

        // Add validator stakes
        assignment_db
            .update_validator_stake("validator-1", 1000.0, 40.0)
            .await?;
        assignment_db
            .update_validator_stake("validator-2", 800.0, 35.0)
            .await?;
        assignment_db
            .update_validator_stake("validator-3", 500.0, 25.0)
            .await?;

        // Test assignment suggestions
        let suggester = AssignmentSuggester::new(0.5);
        let executors = vec!["test-exec-1".to_string(), "test-exec-2".to_string()];
        let stakes = assignment_db.get_all_validator_stakes().await?;
        let assignments = assignment_db.get_all_assignments().await?;

        let suggestions = suggester
            .suggest_assignments(executors, stakes, assignments)
            .await?;

        // Should suggest assignments to reach 50% coverage
        assert!(!suggestions.is_empty());

        Ok(())
    }

    #[tokio::test]
    async fn test_history_command() -> Result<()> {
        let (_config, pool) = setup_test_config().await?;
        let assignment_db = AssignmentDb::new(pool);

        // Create and delete assignment to generate history
        assignment_db
            .create_assignment("test-exec-1", "validator-1", "test", None)
            .await?;
        assignment_db
            .delete_assignment("test-exec-1", "test")
            .await?;

        // Test assignment history
        let history = assignment_db.get_assignment_history("test-exec-1").await?;
        assert_eq!(history.len(), 2); // assign + unassign

        assert_eq!(history[0].action, "unassign"); // Most recent first
        assert_eq!(history[1].action, "assign");
        assert_eq!(history[1].validator_hotkey, Some("validator-1".to_string()));

        Ok(())
    }

    #[tokio::test]
    async fn test_stakes_command() -> Result<()> {
        let (_config, pool) = setup_test_config().await?;
        let assignment_db = AssignmentDb::new(pool);

        // Add validator stakes
        assignment_db
            .update_validator_stake("validator-1", 1000.0, 60.0)
            .await?;
        assignment_db
            .update_validator_stake("validator-2", 500.0, 40.0)
            .await?;

        // Test stakes retrieval
        let all_stakes = assignment_db.get_all_validator_stakes().await?;
        assert_eq!(all_stakes.len(), 2);

        let high_stakes = assignment_db.get_validator_stakes_above(750.0).await?;
        assert_eq!(high_stakes.len(), 1);
        assert_eq!(high_stakes[0].validator_hotkey, "validator-1");
        assert_eq!(high_stakes[0].stake_amount, 1000.0);

        Ok(())
    }
}
