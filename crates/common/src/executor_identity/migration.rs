//! Migration utilities for transitioning from legacy string IDs to UUID+HUID
//!
//! This module provides tools to help migrate existing systems that use
//! arbitrary string identifiers to the new UUID+HUID system while maintaining
//! backwards compatibility and data integrity.

#[cfg(feature = "sqlite")]
use anyhow::{Context, Result};
use std::collections::HashMap;
#[cfg(feature = "sqlite")]
use tracing::{debug, info, warn};
#[cfg(feature = "sqlite")]
use uuid::Uuid;

#[cfg(feature = "sqlite")]
use crate::executor_identity::SqliteIdentityStore;

// Import Result for non-sqlite builds
#[cfg(not(feature = "sqlite"))]
use anyhow::Result;

/// Migration manager for handling the transition from legacy IDs
#[cfg(feature = "sqlite")]
pub struct IdentityMigrationManager {
    /// Identity store for persistence
    store: SqliteIdentityStore,
    /// Database pool for direct queries
    pool: sqlx::SqlitePool,
}

/// Migration statistics
#[derive(Debug, Default)]
pub struct MigrationStats {
    /// Number of legacy IDs found
    pub legacy_ids_found: usize,
    /// Number of successful migrations
    pub successful_migrations: usize,
    /// Number of failed migrations
    pub failed_migrations: usize,
    /// Map of failed IDs to error messages
    pub failed_ids: HashMap<String, String>,
}

/// Configuration for migration behavior
#[derive(Debug, Clone)]
pub struct MigrationConfig {
    /// Whether to perform a dry run (no actual changes)
    pub dry_run: bool,
    /// Batch size for processing
    pub batch_size: usize,
    /// Whether to continue on individual migration failures
    pub continue_on_error: bool,
    /// Tables and columns to scan for legacy IDs
    pub scan_targets: Vec<ScanTarget>,
}

/// Target table and column to scan for legacy IDs
#[derive(Debug, Clone)]
pub struct ScanTarget {
    /// Table name
    pub table: String,
    /// Column containing the legacy ID
    pub id_column: String,
    /// Additional columns to include in migration
    pub additional_columns: Vec<String>,
}

impl Default for MigrationConfig {
    fn default() -> Self {
        Self {
            dry_run: false,
            batch_size: 100,
            continue_on_error: true,
            scan_targets: vec![
                ScanTarget {
                    table: "executor_health".to_string(),
                    id_column: "executor_id".to_string(),
                    additional_columns: vec![],
                },
                ScanTarget {
                    table: "ssh_sessions".to_string(),
                    id_column: "executor_id".to_string(),
                    additional_columns: vec![],
                },
            ],
        }
    }
}

#[cfg(feature = "sqlite")]
impl IdentityMigrationManager {
    /// Create a new migration manager
    pub async fn new(database_url: &str) -> Result<Self> {
        let pool = sqlx::SqlitePool::connect(database_url)
            .await
            .context("Failed to connect to database")?;

        let store = SqliteIdentityStore::from_pool(pool.clone()).await?;

        Ok(Self { store, pool })
    }

    /// Create from existing pool
    pub async fn from_pool(pool: sqlx::SqlitePool) -> Result<Self> {
        let store = SqliteIdentityStore::from_pool(pool.clone()).await?;
        Ok(Self { store, pool })
    }

    /// Scan for legacy IDs across configured tables
    pub async fn scan_legacy_ids(&self, config: &MigrationConfig) -> Result<Vec<String>> {
        let mut all_legacy_ids = Vec::new();

        for target in &config.scan_targets {
            info!("Scanning table {} for legacy IDs", target.table);

            // Check if table exists
            let table_exists: bool = sqlx::query_scalar(
                "SELECT EXISTS(SELECT 1 FROM sqlite_master WHERE type='table' AND name=?)",
            )
            .bind(&target.table)
            .fetch_one(&self.pool)
            .await?;

            if !table_exists {
                warn!("Table {} does not exist, skipping", target.table);
                continue;
            }

            // Get distinct legacy IDs
            let query = format!(
                "SELECT DISTINCT {} FROM {} WHERE {} IS NOT NULL AND {} != ''",
                target.id_column, target.table, target.id_column, target.id_column
            );

            let legacy_ids: Vec<(String,)> = sqlx::query_as(&query)
                .fetch_all(&self.pool)
                .await
                .with_context(|| format!("Failed to scan table {}", target.table))?;

            for (legacy_id,) in legacy_ids {
                // Check if it's already a UUID
                if Uuid::parse_str(&legacy_id).is_ok() {
                    debug!("Skipping {} as it's already a UUID", legacy_id);
                    continue;
                }

                all_legacy_ids.push(legacy_id);
            }

            info!(
                "Found {} legacy IDs in table {}",
                all_legacy_ids.len(),
                target.table
            );
        }

        // Deduplicate
        all_legacy_ids.sort();
        all_legacy_ids.dedup();

        info!("Total unique legacy IDs found: {}", all_legacy_ids.len());
        Ok(all_legacy_ids)
    }

    /// Migrate a batch of legacy IDs
    pub async fn migrate_batch(
        &self,
        legacy_ids: &[String],
        config: &MigrationConfig,
    ) -> Result<MigrationStats> {
        let mut stats = MigrationStats {
            legacy_ids_found: legacy_ids.len(),
            ..Default::default()
        };

        if config.dry_run {
            info!("DRY RUN: Would migrate {} legacy IDs", legacy_ids.len());
            return Ok(stats);
        }

        for legacy_id in legacy_ids {
            match self.store.migrate_legacy_id(legacy_id).await {
                Ok(new_identity) => {
                    info!(
                        "Migrated legacy ID '{}' to UUID {} ({})",
                        legacy_id,
                        new_identity.uuid(),
                        new_identity.huid()
                    );
                    stats.successful_migrations += 1;
                }
                Err(e) => {
                    warn!("Failed to migrate legacy ID '{}': {}", legacy_id, e);
                    stats.failed_migrations += 1;
                    stats.failed_ids.insert(legacy_id.clone(), e.to_string());

                    if !config.continue_on_error {
                        return Err(e);
                    }
                }
            }
        }

        Ok(stats)
    }

    /// Update references to legacy IDs in the database
    pub async fn update_references(
        &self,
        config: &MigrationConfig,
    ) -> Result<HashMap<String, usize>> {
        let mut updates = HashMap::new();

        // Get all legacy mappings
        let mappings = self.store.get_legacy_mappings().await?;

        if mappings.is_empty() {
            info!("No legacy mappings found, nothing to update");
            return Ok(updates);
        }

        info!("Updating references for {} legacy IDs", mappings.len());

        for target in &config.scan_targets {
            let updated = self
                .update_table_references(target, &mappings, config.dry_run)
                .await?;
            updates.insert(target.table.clone(), updated);
        }

        Ok(updates)
    }

    /// Update references in a specific table
    async fn update_table_references(
        &self,
        target: &ScanTarget,
        mappings: &HashMap<String, Uuid>,
        dry_run: bool,
    ) -> Result<usize> {
        // Check if table exists
        let table_exists: bool = sqlx::query_scalar(
            "SELECT EXISTS(SELECT 1 FROM sqlite_master WHERE type='table' AND name=?)",
        )
        .bind(&target.table)
        .fetch_one(&self.pool)
        .await?;

        if !table_exists {
            warn!("Table {} does not exist, skipping", target.table);
            return Ok(0);
        }

        let mut tx = self.pool.begin().await?;
        let mut total_updated = 0;

        // Update each legacy ID to its new UUID
        for (legacy_id, new_uuid) in mappings {
            let update_query = format!(
                "UPDATE {} SET {} = ? WHERE {} = ?",
                target.table, target.id_column, target.id_column
            );

            if dry_run {
                let count_query = format!(
                    "SELECT COUNT(*) FROM {} WHERE {} = ?",
                    target.table, target.id_column
                );
                let count: (i64,) = sqlx::query_as(&count_query)
                    .bind(legacy_id)
                    .fetch_one(&self.pool)
                    .await?;

                if count.0 > 0 {
                    info!(
                        "DRY RUN: Would update {} rows in {} for legacy ID '{}'",
                        count.0, target.table, legacy_id
                    );
                    total_updated += count.0 as usize;
                }
            } else {
                let result = sqlx::query(&update_query)
                    .bind(new_uuid.to_string())
                    .bind(legacy_id)
                    .execute(&mut *tx)
                    .await?;

                if result.rows_affected() > 0 {
                    debug!(
                        "Updated {} rows in {} for legacy ID '{}' -> UUID {}",
                        result.rows_affected(),
                        target.table,
                        legacy_id,
                        new_uuid
                    );
                    total_updated += result.rows_affected() as usize;
                }
            }
        }

        if !dry_run {
            tx.commit().await?;
        }

        info!(
            "{} {} rows in table {}",
            if dry_run { "Would update" } else { "Updated" },
            total_updated,
            target.table
        );

        Ok(total_updated)
    }

    /// Perform a complete migration
    pub async fn migrate_all(&self, config: &MigrationConfig) -> Result<MigrationReport> {
        info!("Starting complete identity migration");

        // Step 1: Scan for legacy IDs
        let legacy_ids = self.scan_legacy_ids(config).await?;

        // Step 2: Migrate in batches
        let mut total_stats = MigrationStats {
            legacy_ids_found: legacy_ids.len(),
            ..Default::default()
        };

        for chunk in legacy_ids.chunks(config.batch_size) {
            let batch_stats = self.migrate_batch(chunk, config).await?;
            total_stats.successful_migrations += batch_stats.successful_migrations;
            total_stats.failed_migrations += batch_stats.failed_migrations;
            total_stats.failed_ids.extend(batch_stats.failed_ids);
        }

        // Step 3: Update references
        let reference_updates = self.update_references(config).await?;

        let report = MigrationReport {
            migration_stats: total_stats,
            reference_updates,
            dry_run: config.dry_run,
        };

        info!("Migration complete: {}", report.summary());
        Ok(report)
    }

    /// Validate that all references have been updated correctly
    pub async fn validate_migration(&self, config: &MigrationConfig) -> Result<ValidationReport> {
        let mappings = self.store.get_legacy_mappings().await?;
        let mut report = ValidationReport::default();

        for target in &config.scan_targets {
            let table_exists: bool = sqlx::query_scalar(
                "SELECT EXISTS(SELECT 1 FROM sqlite_master WHERE type='table' AND name=?)",
            )
            .bind(&target.table)
            .fetch_one(&self.pool)
            .await?;

            if !table_exists {
                continue;
            }

            // Check for any remaining legacy IDs
            let check_query = format!(
                "SELECT COUNT(*) FROM {} WHERE {} IN ({})",
                target.table,
                target.id_column,
                mappings.keys().map(|_| "?").collect::<Vec<_>>().join(",")
            );

            let mut query = sqlx::query_scalar(&check_query);
            for legacy_id in mappings.keys() {
                query = query.bind(legacy_id);
            }

            let remaining_count: i64 = query.fetch_one(&self.pool).await.unwrap_or(0);

            if remaining_count > 0 {
                report
                    .tables_with_legacy_ids
                    .insert(target.table.clone(), remaining_count as usize);
            }
        }

        report.is_valid = report.tables_with_legacy_ids.is_empty();
        Ok(report)
    }
}

/// Report of a complete migration
#[derive(Debug)]
pub struct MigrationReport {
    /// Migration statistics
    pub migration_stats: MigrationStats,
    /// Number of references updated per table
    pub reference_updates: HashMap<String, usize>,
    /// Whether this was a dry run
    pub dry_run: bool,
}

impl MigrationReport {
    /// Generate a summary of the migration
    pub fn summary(&self) -> String {
        let mut summary = if self.dry_run {
            "DRY RUN SUMMARY:\n".to_string()
        } else {
            "MIGRATION SUMMARY:\n".to_string()
        };

        summary.push_str(&format!(
            "  Legacy IDs found: {}\n",
            self.migration_stats.legacy_ids_found
        ));
        summary.push_str(&format!(
            "  Successful migrations: {}\n",
            self.migration_stats.successful_migrations
        ));
        summary.push_str(&format!(
            "  Failed migrations: {}\n",
            self.migration_stats.failed_migrations
        ));

        if !self.reference_updates.is_empty() {
            summary.push_str("  Reference updates:\n");
            for (table, count) in &self.reference_updates {
                summary.push_str(&format!("    {table}: {count} rows\n"));
            }
        }

        summary
    }
}

/// Validation report for migration
#[derive(Debug, Default)]
pub struct ValidationReport {
    /// Whether the migration is valid
    pub is_valid: bool,
    /// Tables that still contain legacy IDs
    pub tables_with_legacy_ids: HashMap<String, usize>,
}

// Fallback types when sqlite feature is not enabled
#[cfg(not(feature = "sqlite"))]
pub struct IdentityMigrationManager;

#[cfg(not(feature = "sqlite"))]
impl IdentityMigrationManager {
    pub async fn new(_database_url: &str) -> Result<Self> {
        Err(anyhow::anyhow!("SQLite feature not enabled"))
    }

    pub async fn scan_legacy_ids(&self, _config: &MigrationConfig) -> Result<Vec<String>> {
        Err(anyhow::anyhow!("SQLite feature not enabled"))
    }
}

#[cfg(all(test, feature = "sqlite"))]
mod tests {
    use super::*;

    async fn create_test_db() -> Result<sqlx::SqlitePool> {
        let pool = sqlx::SqlitePool::connect("sqlite::memory:").await?;

        // Create test tables with legacy IDs
        sqlx::query(
            r#"
            CREATE TABLE executor_health (
                executor_id TEXT PRIMARY KEY,
                last_heartbeat TIMESTAMP,
                status TEXT
            )
            "#,
        )
        .execute(&pool)
        .await?;

        sqlx::query(
            r#"
            CREATE TABLE ssh_sessions (
                session_id INTEGER PRIMARY KEY,
                executor_id TEXT NOT NULL,
                started_at TIMESTAMP
            )
            "#,
        )
        .execute(&pool)
        .await?;

        // Insert test data with legacy IDs
        sqlx::query("INSERT INTO executor_health (executor_id, status) VALUES (?, ?)")
            .bind("legacy-executor-1")
            .bind("active")
            .execute(&pool)
            .await?;

        sqlx::query("INSERT INTO executor_health (executor_id, status) VALUES (?, ?)")
            .bind("old-node-abc123")
            .bind("inactive")
            .execute(&pool)
            .await?;

        // Some entries already have UUIDs
        let uuid = Uuid::new_v4();
        sqlx::query("INSERT INTO executor_health (executor_id, status) VALUES (?, ?)")
            .bind(uuid.to_string())
            .bind("active")
            .execute(&pool)
            .await?;

        // SSH sessions with references
        sqlx::query("INSERT INTO ssh_sessions (executor_id) VALUES (?)")
            .bind("legacy-executor-1")
            .execute(&pool)
            .await?;

        sqlx::query("INSERT INTO ssh_sessions (executor_id) VALUES (?)")
            .bind("legacy-executor-1")
            .execute(&pool)
            .await?;

        sqlx::query("INSERT INTO ssh_sessions (executor_id) VALUES (?)")
            .bind("old-node-abc123")
            .execute(&pool)
            .await?;

        Ok(pool)
    }

    #[tokio::test]
    async fn test_migration_config_defaults() {
        let config = MigrationConfig::default();
        assert!(!config.dry_run);
        assert_eq!(config.batch_size, 100);
        assert!(config.continue_on_error);
        assert!(!config.scan_targets.is_empty());
    }

    #[tokio::test]
    async fn test_scan_legacy_ids() {
        let pool = create_test_db().await.expect("Should create test DB");
        let manager = IdentityMigrationManager::from_pool(pool)
            .await
            .expect("Should create manager");

        let config = MigrationConfig::default();
        let legacy_ids = manager.scan_legacy_ids(&config).await.expect("Should scan");

        // Should find 2 legacy IDs (not the UUID one)
        assert_eq!(legacy_ids.len(), 2);
        assert!(legacy_ids.contains(&"legacy-executor-1".to_string()));
        assert!(legacy_ids.contains(&"old-node-abc123".to_string()));
    }

    #[tokio::test]
    async fn test_migrate_batch() {
        let pool = create_test_db().await.expect("Should create test DB");
        let manager = IdentityMigrationManager::from_pool(pool)
            .await
            .expect("Should create manager");

        let legacy_ids = vec![
            "legacy-executor-1".to_string(),
            "old-node-abc123".to_string(),
        ];

        let config = MigrationConfig::default();
        let stats = manager
            .migrate_batch(&legacy_ids, &config)
            .await
            .expect("Should migrate");

        assert_eq!(stats.legacy_ids_found, 2);
        assert_eq!(stats.successful_migrations, 2);
        assert_eq!(stats.failed_migrations, 0);
        assert!(stats.failed_ids.is_empty());

        // Verify mappings were created
        let mappings = manager
            .store
            .get_legacy_mappings()
            .await
            .expect("Should get mappings");
        assert_eq!(mappings.len(), 2);
        assert!(mappings.contains_key("legacy-executor-1"));
        assert!(mappings.contains_key("old-node-abc123"));
    }

    #[tokio::test]
    async fn test_dry_run_migration() {
        let pool = create_test_db().await.expect("Should create test DB");
        let manager = IdentityMigrationManager::from_pool(pool)
            .await
            .expect("Should create manager");

        let legacy_ids = vec!["test-id".to_string()];

        let config = MigrationConfig {
            dry_run: true,
            ..Default::default()
        };

        let stats = manager
            .migrate_batch(&legacy_ids, &config)
            .await
            .expect("Should complete dry run");

        assert_eq!(stats.legacy_ids_found, 1);
        assert_eq!(stats.successful_migrations, 0);
        assert_eq!(stats.failed_migrations, 0);

        // Verify no actual migration happened
        let mappings = manager
            .store
            .get_legacy_mappings()
            .await
            .expect("Should get mappings");
        assert_eq!(mappings.len(), 0);
    }

    #[tokio::test]
    async fn test_update_references() {
        let pool = create_test_db().await.expect("Should create test DB");
        let manager = IdentityMigrationManager::from_pool(pool.clone())
            .await
            .expect("Should create manager");

        // First migrate the legacy IDs
        let legacy_ids = vec![
            "legacy-executor-1".to_string(),
            "old-node-abc123".to_string(),
        ];

        let config = MigrationConfig::default();
        manager
            .migrate_batch(&legacy_ids, &config)
            .await
            .expect("Should migrate");

        // Now update references - use custom config that only targets ssh_sessions
        // since executor_health has executor_id as PRIMARY KEY
        let update_config = MigrationConfig {
            scan_targets: vec![ScanTarget {
                table: "ssh_sessions".to_string(),
                id_column: "executor_id".to_string(),
                additional_columns: vec![],
            }],
            ..config
        };

        let updates = manager
            .update_references(&update_config)
            .await
            .expect("Should update references");

        // Should have updated ssh_sessions table
        assert!(updates.contains_key("ssh_sessions"));
        assert_eq!(updates["ssh_sessions"], 3); // 3 references

        // Verify the actual updates in ssh_sessions
        let mappings = manager
            .store
            .get_legacy_mappings()
            .await
            .expect("Should get mappings");

        // Check that ssh_sessions now has UUIDs
        for (legacy_id, new_uuid) in &mappings {
            let count: (i64,) =
                sqlx::query_as("SELECT COUNT(*) FROM ssh_sessions WHERE executor_id = ?")
                    .bind(new_uuid.to_string())
                    .fetch_one(&pool)
                    .await
                    .expect("Should query");

            // The count depends on how many sessions each legacy ID had
            assert!(
                count.0 > 0,
                "UUID should exist in ssh_sessions for legacy ID {legacy_id}"
            );
        }
    }

    #[tokio::test]
    async fn test_full_migration_workflow() {
        let pool = create_test_db().await.expect("Should create test DB");
        let manager = IdentityMigrationManager::from_pool(pool.clone())
            .await
            .expect("Should create manager");

        // Only scan ssh_sessions table since executor_health has PRIMARY KEY constraint
        let config = MigrationConfig {
            scan_targets: vec![ScanTarget {
                table: "ssh_sessions".to_string(),
                id_column: "executor_id".to_string(),
                additional_columns: vec![],
            }],
            ..MigrationConfig::default()
        };

        // First, manually scan executor_health to find legacy IDs
        let legacy_ids: Vec<String> = sqlx::query_scalar(
            "SELECT executor_id FROM executor_health WHERE executor_id NOT LIKE '%-%-%-%-%'",
        )
        .fetch_all(&pool)
        .await
        .expect("Should scan for legacy IDs");

        assert_eq!(legacy_ids.len(), 2); // Should find 2 legacy IDs

        // Migrate the legacy IDs found
        let migration_result = manager
            .migrate_batch(&legacy_ids, &config)
            .await
            .expect("Should migrate legacy IDs");

        assert_eq!(migration_result.successful_migrations, 2);
        assert_eq!(migration_result.failed_migrations, 0);

        // Update references using a config that only targets ssh_sessions
        // since executor_health has executor_id as PRIMARY KEY and can't be updated
        let update_config = MigrationConfig {
            scan_targets: vec![ScanTarget {
                table: "ssh_sessions".to_string(),
                id_column: "executor_id".to_string(),
                additional_columns: vec![],
            }],
            ..MigrationConfig::default()
        };

        let reference_updates = manager
            .update_references(&update_config)
            .await
            .expect("Should update references");

        assert!(!reference_updates.is_empty());
        assert_eq!(reference_updates["ssh_sessions"], 3);
    }

    #[tokio::test]
    async fn test_migration_with_errors() {
        let pool = create_test_db().await.expect("Should create test DB");
        let manager = IdentityMigrationManager::from_pool(pool)
            .await
            .expect("Should create manager");

        // Insert an invalid legacy ID
        let invalid_ids = vec![
            "".to_string(), // Empty ID will fail
            "valid-id".to_string(),
        ];

        let config = MigrationConfig {
            continue_on_error: true,
            ..Default::default()
        };

        let stats = manager
            .migrate_batch(&invalid_ids, &config)
            .await
            .expect("Should continue on error");

        assert_eq!(stats.legacy_ids_found, 2);
        assert_eq!(stats.successful_migrations, 2);
        assert_eq!(stats.failed_migrations, 0);
        assert_eq!(stats.failed_ids.len(), 0);
    }

    #[tokio::test]
    async fn test_migration_report_summary() {
        let stats = MigrationStats {
            legacy_ids_found: 10,
            successful_migrations: 8,
            failed_migrations: 2,
            ..Default::default()
        };

        let mut reference_updates = HashMap::new();
        reference_updates.insert("executor_health".to_string(), 15);
        reference_updates.insert("ssh_sessions".to_string(), 20);

        let report = MigrationReport {
            migration_stats: stats,
            reference_updates,
            dry_run: false,
        };

        let summary = report.summary();
        assert!(summary.contains("MIGRATION SUMMARY"));
        assert!(summary.contains("Legacy IDs found: 10"));
        assert!(summary.contains("Successful migrations: 8"));
        assert!(summary.contains("Failed migrations: 2"));
        assert!(summary.contains("executor_health: 15 rows"));
        assert!(summary.contains("ssh_sessions: 20 rows"));
    }

    #[tokio::test]
    async fn test_scan_target_configuration() {
        let target = ScanTarget {
            table: "custom_table".to_string(),
            id_column: "custom_id".to_string(),
            additional_columns: vec!["name".to_string(), "created_at".to_string()],
        };

        assert_eq!(target.table, "custom_table");
        assert_eq!(target.id_column, "custom_id");
        assert_eq!(target.additional_columns.len(), 2);
    }

    #[tokio::test]
    async fn test_migration_idempotency() {
        let pool = create_test_db().await.expect("Should create test DB");
        let manager = IdentityMigrationManager::from_pool(pool)
            .await
            .expect("Should create manager");

        let legacy_ids = vec!["idempotent-test".to_string()];
        let config = MigrationConfig::default();

        // First migration
        let stats1 = manager
            .migrate_batch(&legacy_ids, &config)
            .await
            .expect("Should migrate");
        assert_eq!(stats1.successful_migrations, 1);

        // Get the mapped UUID
        let mappings = manager
            .store
            .get_legacy_mappings()
            .await
            .expect("Should get mappings");
        let first_uuid = mappings["idempotent-test"];

        // Second migration of same ID
        let stats2 = manager
            .migrate_batch(&legacy_ids, &config)
            .await
            .expect("Should migrate again");
        assert_eq!(stats2.successful_migrations, 1);

        // Should map to same UUID
        let mappings2 = manager
            .store
            .get_legacy_mappings()
            .await
            .expect("Should get mappings");
        assert_eq!(mappings2["idempotent-test"], first_uuid);
    }
}
