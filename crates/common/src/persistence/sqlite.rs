//! Complete SQLite persistence layer implementation
//!
//! This module provides a production-ready SQLite persistence layer
//! implementing all traits from the persistence framework.

#[cfg(feature = "sqlite")]
mod sqlite_impl {
    use anyhow::Result;
    use async_trait::async_trait;
    use chrono::{Duration, Utc};
    use sqlx::{migrate::MigrateDatabase, sqlite::SqliteRow, Row, Sqlite, SqlitePool, Transaction};
    use std::collections::HashMap;
    use std::path::Path;
    use tracing::{debug, error, info, warn};

    use super::super::traits::{
        Cleanup, ConnectionStats, DatabaseConnection, DatabaseSizeInfo, DatabaseStats,
        DatabaseStatsProvider, DatabaseTransaction, MigrationManager, MigrationStatus,
        StorageStats,
    };
    use crate::config::DatabaseConfig;

    /// SQLite connection wrapper
    pub struct SqliteConnection {
        pool: SqlitePool,
    }

    impl SqliteConnection {
        /// Create new connection
        pub async fn new(config: &DatabaseConfig) -> Result<Self> {
            info!("Initializing SQLite connection with URL: {}", config.url);

            if config.url.starts_with("sqlite:") {
                let db_path = config.url.strip_prefix("sqlite:").unwrap_or(&config.url);
                if !Path::new(db_path).exists() {
                    info!("Creating new SQLite database: {}", db_path);
                    Sqlite::create_database(&config.url).await?;
                }
            }
            let pool = SqlitePool::connect(&config.url).await?;
            info!("SQLite connection established");

            let connection = Self { pool };
            connection.optimize_settings().await?;
            Ok(connection)
        }

        /// Get connection pool
        pub fn pool(&self) -> &SqlitePool {
            &self.pool
        }

        /// Execute raw SQL
        pub async fn execute_raw(&self, sql: &str) -> Result<u64> {
            let result = sqlx::query(sql).execute(&self.pool).await?;
            Ok(result.rows_affected())
        }

        /// Fetch all rows
        pub async fn fetch_all(&self, sql: &str) -> Result<Vec<SqliteRow>> {
            let rows = sqlx::query(sql).fetch_all(&self.pool).await?;
            Ok(rows)
        }

        /// Fetch one row
        pub async fn fetch_one(&self, sql: &str) -> Result<SqliteRow> {
            let row = sqlx::query(sql).fetch_one(&self.pool).await?;
            Ok(row)
        }

        /// Fetch optional row
        pub async fn fetch_optional(&self, sql: &str) -> Result<Option<SqliteRow>> {
            let row = sqlx::query(sql).fetch_optional(&self.pool).await?;
            Ok(row)
        }

        /// Get table names
        pub async fn get_table_names(&self) -> Result<Vec<String>> {
            let rows =
                sqlx::query("SELECT name FROM sqlite_master WHERE type='table' ORDER BY name")
                    .fetch_all(&self.pool)
                    .await?;

            let mut tables = Vec::new();
            for row in rows {
                tables.push(row.try_get::<String, _>("name")?);
            }
            Ok(tables)
        }

        /// Get database size
        pub async fn get_database_size(&self) -> Result<u64> {
            let page_count: i64 = sqlx::query_scalar("PRAGMA page_count")
                .fetch_one(&self.pool)
                .await?;

            let page_size: i64 = sqlx::query_scalar("PRAGMA page_size")
                .fetch_one(&self.pool)
                .await?;

            Ok((page_count * page_size) as u64)
        }

        /// Vacuum database
        pub async fn vacuum(&self) -> Result<()> {
            info!("Running VACUUM on SQLite database");
            sqlx::query("VACUUM").execute(&self.pool).await?;
            info!("Database VACUUM completed");
            Ok(())
        }

        /// Analyze database
        pub async fn analyze(&self) -> Result<()> {
            info!("Running ANALYZE on SQLite database");
            sqlx::query("ANALYZE").execute(&self.pool).await?;
            info!("Database ANALYZE completed");
            Ok(())
        }

        /// Enable WAL mode
        pub async fn enable_wal_mode(&self) -> Result<()> {
            info!("Enabling WAL mode");
            sqlx::query("PRAGMA journal_mode = WAL")
                .execute(&self.pool)
                .await?;
            info!("WAL mode enabled");
            Ok(())
        }

        /// Set busy timeout
        pub async fn set_busy_timeout(&self, timeout_ms: i32) -> Result<()> {
            let sql = format!("PRAGMA busy_timeout = {timeout_ms}");
            sqlx::query(&sql).execute(&self.pool).await?;
            info!("Busy timeout set to {}ms", timeout_ms);
            Ok(())
        }

        /// Optimize settings for production use
        pub async fn optimize_settings(&self) -> Result<()> {
            info!("Optimizing SQLite settings for production");

            // Enable WAL mode for better concurrency
            self.enable_wal_mode().await?;

            // Set reasonable busy timeout
            self.set_busy_timeout(30000).await?;

            // Optimize cache size (10MB cache)
            sqlx::query("PRAGMA cache_size = 10000")
                .execute(&self.pool)
                .await
                .map_err(|e| anyhow::anyhow!("Failed to set cache_size: {}", e))?;

            // Enable foreign key constraints
            sqlx::query("PRAGMA foreign_keys = ON")
                .execute(&self.pool)
                .await
                .map_err(|e| anyhow::anyhow!("Failed to enable foreign_keys: {}", e))?;

            // Set synchronous mode to NORMAL for balanced performance/safety
            sqlx::query("PRAGMA synchronous = NORMAL")
                .execute(&self.pool)
                .await
                .map_err(|e| anyhow::anyhow!("Failed to set synchronous mode: {}", e))?;

            // Use memory for temporary storage
            sqlx::query("PRAGMA temp_store = MEMORY")
                .execute(&self.pool)
                .await
                .map_err(|e| anyhow::anyhow!("Failed to set temp_store: {}", e))?;

            // Set WAL checkpoint threshold
            sqlx::query("PRAGMA wal_autocheckpoint = 1000")
                .execute(&self.pool)
                .await
                .map_err(|e| anyhow::anyhow!("Failed to set wal_autocheckpoint: {}", e))?;

            // Enable query planner optimization
            sqlx::query("PRAGMA optimize")
                .execute(&self.pool)
                .await
                .map_err(|e| anyhow::anyhow!("Failed to run PRAGMA optimize: {}", e))?;

            info!("SQLite optimization completed successfully");
            Ok(())
        }

        /// Get comprehensive table statistics
        pub async fn get_table_statistics(&self) -> Result<HashMap<String, TableStats>> {
            let tables = self.get_table_names().await?;
            let mut stats = HashMap::new();

            for table in tables {
                let row_count: i64 = sqlx::query_scalar(&format!("SELECT COUNT(*) FROM {table}"))
                    .fetch_one(&self.pool)
                    .await
                    .unwrap_or(0);

                // Rough estimate of table size
                let estimated_size = row_count as u64 * 100; // Rough estimate

                stats.insert(
                    table.clone(),
                    TableStats {
                        row_count: row_count as u64,
                        size_bytes: estimated_size,
                    },
                );
            }

            Ok(stats)
        }

        /// Execute a parameterized query with retry logic
        pub async fn execute_with_retry(&self, sql: &str, max_retries: u32) -> Result<u64> {
            let mut last_error = None;

            for attempt in 0..=max_retries {
                match sqlx::query(sql).execute(&self.pool).await {
                    Ok(result) => return Ok(result.rows_affected()),
                    Err(e) => {
                        last_error = Some(e);
                        if attempt < max_retries {
                            warn!("Query attempt {} failed, retrying: {}", attempt + 1, sql);
                            tokio::time::sleep(std::time::Duration::from_millis(
                                100 * (attempt + 1) as u64,
                            ))
                            .await;
                        }
                    }
                }
            }

            Err(anyhow::anyhow!(
                "Query failed after {} retries: {}",
                max_retries,
                last_error.unwrap()
            ))
        }

        /// Prepare database for shutdown
        pub async fn prepare_shutdown(&self) -> Result<()> {
            info!("Preparing SQLite database for shutdown");

            // Checkpoint WAL file
            sqlx::query("PRAGMA wal_checkpoint(TRUNCATE)")
                .execute(&self.pool)
                .await
                .map_err(|e| anyhow::anyhow!("Failed to checkpoint WAL: {}", e))?;

            // Final optimization
            sqlx::query("PRAGMA optimize")
                .execute(&self.pool)
                .await
                .map_err(|e| anyhow::anyhow!("Failed final optimization: {}", e))?;

            info!("Database prepared for shutdown");
            Ok(())
        }
    }

    /// Table statistics structure
    #[derive(Debug, Clone)]
    pub struct TableStats {
        pub row_count: u64,
        pub size_bytes: u64,
    }

    #[async_trait]
    impl DatabaseConnection for SqliteConnection {
        async fn health_check(&self) -> Result<()> {
            info!("Running SQLite health check");

            // Test basic connectivity
            sqlx::query("SELECT 1")
                .fetch_one(&self.pool)
                .await
                .map_err(|e| anyhow::anyhow!("Basic connectivity test failed: {}", e))?;

            // Check database integrity
            let integrity_result: String = sqlx::query_scalar("PRAGMA integrity_check(10)")
                .fetch_one(&self.pool)
                .await
                .map_err(|e| anyhow::anyhow!("Integrity check failed: {}", e))?;

            if integrity_result != "ok" {
                error!("Database integrity check failed: {}", integrity_result);
                return Err(anyhow::anyhow!(
                    "Database integrity compromised: {}",
                    integrity_result
                ));
            }

            // Check foreign key constraints
            let fk_result: Option<String> = sqlx::query_scalar("PRAGMA foreign_key_check")
                .fetch_optional(&self.pool)
                .await
                .map_err(|e| anyhow::anyhow!("Foreign key check failed: {}", e))?;

            if let Some(fk_result) = fk_result {
                warn!("Foreign key constraint violations detected: {}", fk_result);
            }

            // Verify WAL mode is enabled for better performance
            let journal_mode: String = sqlx::query_scalar("PRAGMA journal_mode")
                .fetch_one(&self.pool)
                .await
                .map_err(|e| anyhow::anyhow!("Journal mode check failed: {}", e))?;

            if journal_mode.to_lowercase() != "wal" {
                info!(
                    "Journal mode is '{}', consider enabling WAL for better performance",
                    journal_mode
                );
            }

            info!(
                "SQLite health check passed - integrity: {}, journal: {}",
                integrity_result, journal_mode
            );
            Ok(())
        }

        async fn close(&self) {
            info!("Closing SQLite connection pool");
            self.pool.close().await;
            info!("SQLite connection pool closed");
        }

        async fn begin_transaction(&self) -> Result<Box<dyn DatabaseTransaction>> {
            debug!("Starting new SQLite transaction");
            let tx = self
                .pool
                .begin()
                .await
                .map_err(|e| anyhow::anyhow!("Failed to begin transaction: {}", e))?;
            Ok(Box::new(SqliteTransaction::new(tx)))
        }

        async fn connection_stats(&self) -> Result<Option<ConnectionStats>> {
            // SQLite connection pool statistics
            let pool_size = self.pool.size();
            let idle_size = self.pool.num_idle() as u32;

            Ok(Some(ConnectionStats {
                active_connections: pool_size.saturating_sub(idle_size),
                idle_connections: idle_size,
                max_connections: self.pool.options().get_max_connections(),
                total_connections: pool_size as u64, // Current total, not lifetime
                failed_connections: 0,               // SQLite doesn't provide this metric easily
            }))
        }
    }

    /// SQLite transaction wrapper with enhanced error handling
    pub struct SqliteTransaction {
        tx: Option<Transaction<'static, Sqlite>>,
        is_completed: bool,
    }

    impl SqliteTransaction {
        fn new(tx: Transaction<'static, Sqlite>) -> Self {
            Self {
                tx: Some(tx),
                is_completed: false,
            }
        }
    }

    #[async_trait]
    impl DatabaseTransaction for SqliteTransaction {
        async fn commit(mut self: Box<Self>) -> Result<()> {
            if self.is_completed {
                return Err(anyhow::anyhow!("Transaction already completed"));
            }

            if let Some(tx) = self.tx.take() {
                debug!("Committing SQLite transaction");
                tx.commit()
                    .await
                    .map_err(|e| anyhow::anyhow!("Transaction commit failed: {}", e))?;
                self.is_completed = true;
                debug!("SQLite transaction committed successfully");
            }
            Ok(())
        }

        async fn rollback(mut self: Box<Self>) -> Result<()> {
            if self.is_completed {
                return Err(anyhow::anyhow!("Transaction already completed"));
            }

            if let Some(tx) = self.tx.take() {
                debug!("Rolling back SQLite transaction");
                tx.rollback()
                    .await
                    .map_err(|e| anyhow::anyhow!("Transaction rollback failed: {}", e))?;
                self.is_completed = true;
                debug!("SQLite transaction rolled back successfully");
            }
            Ok(())
        }
    }

    impl Drop for SqliteTransaction {
        fn drop(&mut self) {
            if !self.is_completed && self.tx.is_some() {
                warn!("Transaction dropped without explicit commit or rollback - will be auto-rolled back");
            }
        }
    }

    /// Implementation of MigrationManager trait for SQLite
    #[async_trait]
    impl MigrationManager for SqliteConnection {
        async fn run_migrations(&self) -> Result<()> {
            info!("Running SQLite migrations");
            sqlx::query(
                r#"
                CREATE TABLE IF NOT EXISTS _migrations (
                    version INTEGER PRIMARY KEY,
                    name TEXT NOT NULL,
                    applied_at DATETIME DEFAULT CURRENT_TIMESTAMP
                )
                "#,
            )
            .execute(&self.pool)
            .await
            .map_err(|e| anyhow::anyhow!("Failed to create migrations table: {}", e))?;
            info!("Migration framework initialized");
            Ok(())
        }

        async fn get_current_version(&self) -> Result<i32> {
            let version: Option<i32> = sqlx::query_scalar("SELECT MAX(version) FROM _migrations")
                .fetch_optional(&self.pool)
                .await
                .map_err(|e| anyhow::anyhow!("Failed to get current migration version: {}", e))?
                .flatten();
            Ok(version.unwrap_or(0))
        }

        async fn validate_schema(&self) -> Result<()> {
            info!("Validating SQLite schema");
            let integrity_result: String = sqlx::query_scalar("PRAGMA integrity_check")
                .fetch_one(&self.pool)
                .await
                .map_err(|e| anyhow::anyhow!("Schema integrity check failed: {}", e))?;
            if integrity_result != "ok" {
                return Err(anyhow::anyhow!(
                    "Schema integrity check failed: {}",
                    integrity_result
                ));
            }
            info!("Schema validation completed successfully");
            Ok(())
        }

        async fn migration_status(&self) -> Result<MigrationStatus> {
            let current_version = self.get_current_version().await? as u64;
            let applied_migrations: Vec<String> =
                sqlx::query_scalar("SELECT name FROM _migrations ORDER BY version")
                    .fetch_all(&self.pool)
                    .await
                    .map_err(|e| anyhow::anyhow!("Failed to get applied migrations: {}", e))?;
            Ok(MigrationStatus {
                current_version,
                latest_version: current_version,
                pending_migrations: Vec::new(),
                applied_migrations,
            })
        }
    }

    /// Implementation of DatabaseStatsProvider trait for SQLite
    #[async_trait]
    impl DatabaseStatsProvider for SqliteConnection {
        async fn get_stats(&self) -> Result<DatabaseStats> {
            let total_size = self.get_database_size().await?;
            let table_count: i64 =
                sqlx::query_scalar("SELECT COUNT(*) FROM sqlite_master WHERE type='table'")
                    .fetch_one(&self.pool)
                    .await
                    .map_err(|e| anyhow::anyhow!("Failed to count tables: {}", e))?;
            let index_count: i64 =
                sqlx::query_scalar("SELECT COUNT(*) FROM sqlite_master WHERE type='index'")
                    .fetch_one(&self.pool)
                    .await
                    .map_err(|e| anyhow::anyhow!("Failed to count indexes: {}", e))?;
            Ok(DatabaseStats {
                total_size_bytes: total_size,
                table_count: table_count as u32,
                index_count: index_count as u32,
            })
        }

        async fn get_table_counts(&self) -> Result<HashMap<String, u64>> {
            let tables = self.get_table_names().await?;
            let mut counts = HashMap::new();
            for table in tables {
                let count: i64 = sqlx::query_scalar(&format!("SELECT COUNT(*) FROM {table}"))
                    .fetch_one(&self.pool)
                    .await
                    .unwrap_or(0);
                counts.insert(table, count as u64);
            }
            Ok(counts)
        }

        async fn get_size_info(&self) -> Result<DatabaseSizeInfo> {
            let total_size = self.get_database_size().await?;
            let table_stats = self.get_table_statistics().await?;
            let mut table_sizes = HashMap::new();
            let mut index_sizes = HashMap::new();
            for (table_name, stats) in table_stats {
                table_sizes.insert(table_name, stats.size_bytes);
            }
            let indexes: Vec<String> = sqlx::query_scalar(
                "SELECT name FROM sqlite_master WHERE type='index' AND name NOT LIKE 'sqlite_%'",
            )
            .fetch_all(&self.pool)
            .await
            .unwrap_or_default();
            for index_name in indexes {
                index_sizes.insert(index_name, 0);
            }
            Ok(DatabaseSizeInfo {
                total_size_bytes: total_size,
                table_sizes,
                index_sizes,
            })
        }
    }

    /// Implementation of Cleanup trait for SQLite
    #[async_trait]
    impl Cleanup for SqliteConnection {
        async fn cleanup_old_records(&self, retention_days: i64) -> Result<u64> {
            info!("Cleaning up records older than {} days", retention_days);
            let cutoff_date = Utc::now() - Duration::days(retention_days);
            let cutoff_str = cutoff_date.format("%Y-%m-%d %H:%M:%S").to_string();
            let tables = self.get_table_names().await?;
            let mut total_deleted = 0u64;
            for table in tables {
                if table.starts_with("sqlite_") || table.starts_with("_") {
                    continue;
                }
                let timestamp_columns =
                    vec!["created_at", "updated_at", "timestamp", "date_created"];
                for col in timestamp_columns {
                    let delete_query = format!("DELETE FROM {table} WHERE {col} < ?");
                    match sqlx::query(&delete_query)
                        .bind(&cutoff_str)
                        .execute(&self.pool)
                        .await
                    {
                        Ok(result) => {
                            let deleted = result.rows_affected();
                            if deleted > 0 {
                                info!("Deleted {} old records from table {}", deleted, table);
                                total_deleted += deleted;
                                break;
                            }
                        }
                        Err(_) => continue,
                    }
                }
            }
            info!("Cleanup completed: {} total records deleted", total_deleted);
            Ok(total_deleted)
        }

        async fn optimize_storage(&self) -> Result<()> {
            info!("Optimizing SQLite storage");
            self.vacuum().await?;
            self.analyze().await?;
            let journal_mode: String = sqlx::query_scalar("PRAGMA journal_mode")
                .fetch_one(&self.pool)
                .await?;
            if journal_mode.to_lowercase() == "wal" {
                sqlx::query("PRAGMA wal_checkpoint(TRUNCATE)")
                    .execute(&self.pool)
                    .await
                    .map_err(|e| anyhow::anyhow!("WAL checkpoint failed: {}", e))?;
                info!("WAL checkpoint completed");
            }
            info!("Storage optimization completed");
            Ok(())
        }

        async fn storage_stats(&self) -> Result<StorageStats> {
            let total_size = self.get_database_size().await?;
            let table_count: i64 =
                sqlx::query_scalar("SELECT COUNT(*) FROM sqlite_master WHERE type='table'")
                    .fetch_one(&self.pool)
                    .await?;
            let index_count: i64 =
                sqlx::query_scalar("SELECT COUNT(*) FROM sqlite_master WHERE type='index'")
                    .fetch_one(&self.pool)
                    .await?;
            Ok(StorageStats {
                total_size_bytes: total_size,
                data_size_bytes: total_size,
                index_size_bytes: 0,
                temp_size_bytes: 0,
                table_count: table_count as u32,
                index_count: index_count as u32,
            })
        }
    }
}

#[cfg(feature = "sqlite")]
pub use sqlite_impl::*;

#[cfg(not(feature = "sqlite"))]
mod fallback {
    /// Placeholder when SQLite feature is disabled
    pub struct SqliteConnection;
    pub struct SqliteTransaction;
    pub struct TableStats;
}

#[cfg(not(feature = "sqlite"))]
pub use fallback::*;
