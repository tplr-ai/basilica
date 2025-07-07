//! Integration module for executor identity CLI features
//!
//! This module provides high-level integration functions that combine
//! the identity system with existing executor management functionality.

use anyhow::{anyhow, Result};
use common::executor_identity::{ExecutorId, ExecutorIdentity, SqliteIdentityStore};
use tracing::{debug, info};

use super::disambiguation::{search_by_identifier, DisambiguationOptions, IdentitySearchResult};
use crate::config::{ExecutorConfig, MinerConfig};
use crate::persistence::RegistrationDb;

/// Enhanced executor information with identity support
#[derive(Debug, Clone)]
pub struct ExecutorWithIdentity {
    /// Legacy configuration
    pub config: ExecutorConfig,
    /// Modern identity (if available)
    pub identity: Option<ExecutorId>,
    /// Health status
    pub is_healthy: bool,
}

/// Find an executor by UUID or HUID with full identity support
pub async fn find_executor_with_identity(
    identifier: &str,
    config: &MinerConfig,
    _identity_store: Option<&SqliteIdentityStore>,
) -> Result<ExecutorWithIdentity> {
    info!("Finding executor by identifier: {}", identifier);

    // Validate query length
    super::disambiguation::validate_query_length(identifier, 3)?;

    // For now, use legacy matching until identity store is integrated
    let executors = &config.executor_management.executors;

    // Create matcher function
    let matcher = |exec: &&ExecutorConfig, query: &str| -> bool {
        // Try exact match first
        if exec.id == query {
            return true;
        }

        // Try prefix match if query is long enough
        if query.len() >= 3 {
            exec.id.starts_with(query)
        } else {
            false
        }
    };

    // Search for matches
    let result = search_by_identifier(identifier, executors.iter(), matcher);

    match result {
        IdentitySearchResult::Unique(executor) => {
            Ok(ExecutorWithIdentity {
                config: executor.clone(),
                identity: None, // Will be populated when identity store is integrated
                is_healthy: false, // Will be populated from health checks
            })
        }
        IdentitySearchResult::NotFound => {
            Err(anyhow!("No executor found matching '{}'", identifier))
        }
        IdentitySearchResult::Ambiguous(matches) => {
            let options = DisambiguationOptions::default();
            let display_fn = |e: &&ExecutorConfig| e.id.clone();
            let error = super::disambiguation::format_ambiguous_error(
                identifier, &matches, display_fn, &options,
            );
            Err(anyhow!("{}", error))
        }
    }
}

/// List all executors with identity information
pub async fn list_executors_with_identities(
    config: &MinerConfig,
    db: &RegistrationDb,
    _identity_store: Option<&SqliteIdentityStore>,
) -> Result<Vec<ExecutorWithIdentity>> {
    debug!("Listing all executors with identity information");

    let mut results = Vec::new();

    // Get health records
    let health_records = db.get_all_executor_health().await?;

    for executor in &config.executor_management.executors {
        let health = health_records
            .iter()
            .find(|h| h.executor_id == executor.id)
            .map(|h| h.is_healthy)
            .unwrap_or(false);

        results.push(ExecutorWithIdentity {
            config: executor.clone(),
            identity: None, // Will be populated when identity store is integrated
            is_healthy: health,
        });
    }

    Ok(results)
}

/// Format executor display based on verbosity settings
pub fn format_executor_display(
    executor: &ExecutorWithIdentity,
    verbose: bool,
    show_health: bool,
) -> String {
    let mut output = String::new();

    if let Some(identity) = &executor.identity {
        if verbose {
            // Show both UUID and HUID
            output.push_str(&format!("{:<40} {:<20}", identity.uuid(), identity.huid()));
        } else {
            // Show HUID only
            output.push_str(&format!("{:<20}", identity.huid()));
        }
    } else {
        // Fallback to legacy ID
        output.push_str(&format!("{:<20}", executor.config.id));
    }

    if show_health {
        let status = if executor.is_healthy {
            "HEALTHY"
        } else {
            "UNHEALTHY"
        };
        output.push_str(&format!(" {status:<10}"));
    }

    output.push_str(&format!(" {}", executor.config.grpc_address));

    output
}

/// Create a migration report showing which executors need identity assignment
pub async fn create_identity_migration_report(
    config: &MinerConfig,
    _identity_store: Option<&SqliteIdentityStore>,
) -> Result<String> {
    let mut report = String::from("Executor Identity Migration Report\n");
    report.push_str("=================================\n\n");

    let mut needs_migration = 0;
    let has_identity = 0;

    for executor in &config.executor_management.executors {
        // Check if executor has modern identity
        // For now, all executors are considered to need migration
        needs_migration += 1;

        report.push_str(&format!("- {} : Needs migration\n", executor.id));
    }

    report.push_str(&format!(
        "\nSummary:\n  Total executors: {}\n  With identity: {}\n  Needs migration: {}\n",
        config.executor_management.executors.len(),
        has_identity,
        needs_migration
    ));

    Ok(report)
}

/// Helper to convert legacy executor ID to identity search query
pub fn legacy_id_to_query(legacy_id: &str) -> String {
    // For migration purposes, we can use the legacy ID as-is
    // Later this can be enhanced to handle special cases
    legacy_id.to_string()
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_format_executor_display() {
        let executor = ExecutorWithIdentity {
            config: ExecutorConfig {
                id: "test-executor".to_string(),
                grpc_address: "localhost:50051".to_string(),
                name: Some("Test Executor".to_string()),
                metadata: None,
            },
            identity: None,
            is_healthy: true,
        };

        // Test non-verbose with health
        let display = format_executor_display(&executor, false, true);
        assert!(display.contains("test-executor"));
        assert!(display.contains("HEALTHY"));
        assert!(display.contains("localhost:50051"));

        // Test verbose without health
        let display = format_executor_display(&executor, true, false);
        assert!(display.contains("test-executor"));
        assert!(!display.contains("HEALTHY"));
    }

    #[test]
    fn test_legacy_id_to_query() {
        assert_eq!(legacy_id_to_query("gpu-node-1"), "gpu-node-1");
        assert_eq!(legacy_id_to_query("executor-west-2"), "executor-west-2");
    }
}
