//! # Registration Database
//!
//! Simplified SQLite database for the miner according to SPEC v1.6:
//! - Track executor health status (no dynamic registration)
//! - Log validator interactions and SSH access grants
//! - Simple audit trail for compliance

use anyhow::{Context, Result};
use chrono::{DateTime, Utc};
use serde::{Deserialize, Serialize};
use sqlx::{FromRow, SqlitePool};
use std::path::Path;
use tokio::fs;
use tracing::{debug, info};

use basilica_common::{
    config::DatabaseConfig,
    executor_identity::{ExecutorId, ExecutorIdentity},
};

/// Registration database client
#[derive(Debug, Clone)]
pub struct RegistrationDb {
    pool: SqlitePool,
}

/// Executor health status
#[derive(Debug, Clone, Serialize, Deserialize, FromRow)]
pub struct ExecutorHealth {
    /// Executor ID (from config)
    pub executor_id: String,
    /// Is the executor healthy?
    pub is_healthy: bool,
    /// Last successful health check
    pub last_health_check: Option<DateTime<Utc>>,
    /// Number of consecutive failures
    pub consecutive_failures: i32,
    /// Last error message
    pub last_error: Option<String>,
    /// When this record was last updated
    pub updated_at: DateTime<Utc>,
}

/// Validator interaction log
#[derive(Debug, Clone, Serialize, Deserialize, FromRow)]
pub struct ValidatorInteraction {
    /// Unique ID for this interaction
    pub id: i64,
    /// Validator hotkey
    pub validator_hotkey: String,
    /// Type of interaction (auth, list_executors, ssh_access)
    pub interaction_type: String,
    /// Was the interaction successful?
    pub success: bool,
    /// Additional details (JSON)
    pub details: Option<String>,
    /// When this occurred
    pub created_at: DateTime<Utc>,
}

/// SSH access grant record
#[derive(Debug, Clone, Serialize, Deserialize, FromRow)]
pub struct SshAccessGrant {
    /// Unique ID for this grant
    pub id: i64,
    /// Validator who was granted access
    pub validator_hotkey: String,
    /// Executor IDs that were granted access to
    pub executor_ids: String, // JSON array
    /// When access was granted
    pub granted_at: DateTime<Utc>,
    /// When access expires (if applicable)
    pub expires_at: Option<DateTime<Utc>>,
    /// Is this grant still active?
    pub is_active: bool,
}

/// SSH session record for tracking temporary SSH access
#[derive(Debug, Clone, Serialize, Deserialize, FromRow)]
pub struct SshSessionRecord {
    /// Session ID
    pub session_id: String,
    /// Validator hotkey
    pub validator_hotkey: String,
    /// Target executor ID
    pub executor_id: String,
    /// SSH username created for this session
    pub ssh_username: String,
    /// When the session was created
    pub created_at: DateTime<Utc>,
    /// When the session expires
    pub expires_at: DateTime<Utc>,
    /// Session status (active, expired, revoked)
    pub status: String,
    /// Optional revocation reason
    pub revocation_reason: Option<String>,
    /// When the session was revoked (if applicable)
    pub revoked_at: Option<DateTime<Utc>>,
}

impl RegistrationDb {
    /// Create a new registration database client
    pub async fn new(config: &DatabaseConfig) -> Result<Self> {
        info!("Creating registration database client");
        debug!("Database URL: {}", config.url);

        // Ensure database directory exists
        Self::ensure_database_directory(&config.url).await?;

        // Add connection mode for read-write-create if not present
        let final_url = if config.url.contains('?') {
            config.url.clone()
        } else {
            format!("{}?mode=rwc", config.url)
        };
        debug!("Final database URL: {}", final_url);

        let pool = SqlitePool::connect(&final_url)
            .await
            .context("Failed to connect to SQLite database")?;

        let db = Self { pool };

        // Run migrations
        if config.run_migrations {
            db.run_migrations().await?;
        }

        Ok(db)
    }

    /// Run database migrations
    async fn run_migrations(&self) -> Result<()> {
        info!("Running database migrations...");

        // Create executor health table
        sqlx::query(
            r#"
            CREATE TABLE IF NOT EXISTS executor_health (
                executor_id TEXT PRIMARY KEY,
                is_healthy BOOLEAN NOT NULL DEFAULT FALSE,
                last_health_check TIMESTAMP,
                consecutive_failures INTEGER NOT NULL DEFAULT 0,
                last_error TEXT,
                updated_at TIMESTAMP NOT NULL DEFAULT CURRENT_TIMESTAMP
            )
            "#,
        )
        .execute(&self.pool)
        .await
        .context("Failed to create executor_health table")?;

        // Create validator interactions table
        sqlx::query(
            r#"
            CREATE TABLE IF NOT EXISTS validator_interactions (
                id INTEGER PRIMARY KEY AUTOINCREMENT,
                validator_hotkey TEXT NOT NULL,
                interaction_type TEXT NOT NULL,
                success BOOLEAN NOT NULL,
                details TEXT,
                created_at TIMESTAMP NOT NULL DEFAULT CURRENT_TIMESTAMP
            )
            "#,
        )
        .execute(&self.pool)
        .await
        .context("Failed to create validator_interactions table")?;

        // Create SSH access grants table
        sqlx::query(
            r#"
            CREATE TABLE IF NOT EXISTS ssh_access_grants (
                id INTEGER PRIMARY KEY AUTOINCREMENT,
                validator_hotkey TEXT NOT NULL,
                executor_ids TEXT NOT NULL,
                granted_at TIMESTAMP NOT NULL DEFAULT CURRENT_TIMESTAMP,
                expires_at TIMESTAMP,
                is_active BOOLEAN NOT NULL DEFAULT TRUE
            )
            "#,
        )
        .execute(&self.pool)
        .await
        .context("Failed to create ssh_access_grants table")?;

        // Create SSH sessions table for temporary access tracking
        sqlx::query(
            r#"
            CREATE TABLE IF NOT EXISTS ssh_sessions (
                session_id TEXT PRIMARY KEY,
                validator_hotkey TEXT NOT NULL,
                executor_id TEXT NOT NULL,
                ssh_username TEXT NOT NULL,
                created_at TIMESTAMP NOT NULL DEFAULT CURRENT_TIMESTAMP,
                expires_at TIMESTAMP NOT NULL,
                status TEXT NOT NULL DEFAULT 'active',
                revocation_reason TEXT,
                revoked_at TIMESTAMP
            )
            "#,
        )
        .execute(&self.pool)
        .await
        .context("Failed to create ssh_sessions table")?;

        // Create indices for performance
        sqlx::query("CREATE INDEX IF NOT EXISTS idx_validator_interactions_hotkey ON validator_interactions(validator_hotkey)")
            .execute(&self.pool)
            .await?;

        sqlx::query("CREATE INDEX IF NOT EXISTS idx_ssh_grants_validator ON ssh_access_grants(validator_hotkey)")
            .execute(&self.pool)
            .await?;

        sqlx::query("CREATE INDEX IF NOT EXISTS idx_ssh_sessions_validator ON ssh_sessions(validator_hotkey)")
            .execute(&self.pool)
            .await?;

        sqlx::query(
            "CREATE INDEX IF NOT EXISTS idx_ssh_sessions_executor ON ssh_sessions(executor_id)",
        )
        .execute(&self.pool)
        .await?;

        sqlx::query("CREATE INDEX IF NOT EXISTS idx_ssh_sessions_status ON ssh_sessions(status)")
            .execute(&self.pool)
            .await?;

        sqlx::query(
            r#"
            CREATE TABLE IF NOT EXISTS executor_uuids (
                executor_address TEXT NOT NULL UNIQUE,
                uuid TEXT NOT NULL UNIQUE,
                huid TEXT NOT NULL UNIQUE,
                created_at TIMESTAMP NOT NULL DEFAULT CURRENT_TIMESTAMP
            )
            "#,
        )
        .execute(&self.pool)
        .await
        .context("Failed to create ssh_sessions table")?;

        info!("Database migrations completed successfully");
        Ok(())
    }

    /// Update executor health status
    pub async fn update_executor_health(&self, executor_id: &str, is_healthy: bool) -> Result<()> {
        let now = Utc::now();

        let consecutive_failures = if is_healthy {
            0
        } else {
            // Get current failures and increment
            let current: Option<(i32,)> = sqlx::query_as(
                "SELECT consecutive_failures FROM executor_health WHERE executor_id = ?",
            )
            .bind(executor_id)
            .fetch_optional(&self.pool)
            .await?;

            current.map(|(f,)| f + 1).unwrap_or(1)
        };

        sqlx::query(
            r#"
            INSERT INTO executor_health (executor_id, is_healthy, last_health_check, consecutive_failures, updated_at)
            VALUES (?, ?, ?, ?, ?)
            ON CONFLICT(executor_id) DO UPDATE SET
                is_healthy = excluded.is_healthy,
                last_health_check = CASE WHEN excluded.is_healthy THEN excluded.last_health_check ELSE executor_health.last_health_check END,
                consecutive_failures = excluded.consecutive_failures,
                updated_at = excluded.updated_at
            "#,
        )
        .bind(executor_id)
        .bind(is_healthy)
        .bind(if is_healthy { Some(now) } else { None })
        .bind(consecutive_failures)
        .bind(now)
        .execute(&self.pool)
        .await?;

        debug!(
            "Updated health status for executor {}: healthy={}",
            executor_id, is_healthy
        );
        Ok(())
    }

    /// Get health status of all executors
    pub async fn get_all_executor_health(&self) -> Result<Vec<ExecutorHealth>> {
        let health_records = sqlx::query_as::<_, ExecutorHealth>(
            "SELECT * FROM executor_health ORDER BY executor_id",
        )
        .fetch_all(&self.pool)
        .await?;

        Ok(health_records)
    }

    /// Check if a specific executor is healthy
    pub async fn is_executor_healthy(&self, executor_id: &str) -> Result<bool> {
        let result = sqlx::query_scalar::<_, bool>(
            "SELECT is_healthy FROM executor_health WHERE executor_id = ?",
        )
        .bind(executor_id)
        .fetch_optional(&self.pool)
        .await?;

        Ok(result.unwrap_or(false))
    }

    /// Record a validator interaction
    pub async fn update_validator_interaction(
        &self,
        validator_hotkey: &str,
        success: bool,
    ) -> Result<()> {
        self.record_validator_interaction(validator_hotkey, "authentication", success, None)
            .await
    }

    /// Record a validator interaction with details
    pub async fn record_validator_interaction(
        &self,
        validator_hotkey: &str,
        interaction_type: &str,
        success: bool,
        details: Option<String>,
    ) -> Result<()> {
        sqlx::query(
            r#"
            INSERT INTO validator_interactions (validator_hotkey, interaction_type, success, details)
            VALUES (?, ?, ?, ?)
            "#,
        )
        .bind(validator_hotkey)
        .bind(interaction_type)
        .bind(success)
        .bind(details)
        .execute(&self.pool)
        .await?;

        debug!(
            "Recorded {} interaction for validator {}",
            interaction_type, validator_hotkey
        );
        Ok(())
    }

    /// Record SSH access grant
    pub async fn record_ssh_access_grant(
        &self,
        validator_hotkey: &str,
        executor_ids: &[String],
    ) -> Result<()> {
        let executor_ids_json = serde_json::to_string(executor_ids)?;

        sqlx::query(
            r#"
            INSERT INTO ssh_access_grants (validator_hotkey, executor_ids)
            VALUES (?, ?)
            "#,
        )
        .bind(validator_hotkey)
        .bind(executor_ids_json)
        .execute(&self.pool)
        .await?;

        info!(
            "Recorded SSH access grant for validator {} to {} executors",
            validator_hotkey,
            executor_ids.len()
        );
        Ok(())
    }

    /// Get recent validator interactions
    pub async fn get_recent_validator_interactions(
        &self,
        limit: i64,
    ) -> Result<Vec<ValidatorInteraction>> {
        let interactions = sqlx::query_as::<_, ValidatorInteraction>(
            "SELECT * FROM validator_interactions ORDER BY created_at DESC LIMIT ?",
        )
        .bind(limit)
        .fetch_all(&self.pool)
        .await?;

        Ok(interactions)
    }

    /// Get active SSH grants for a validator
    pub async fn get_active_ssh_grants(
        &self,
        validator_hotkey: &str,
    ) -> Result<Vec<SshAccessGrant>> {
        let grants = sqlx::query_as::<_, SshAccessGrant>(
            r#"
            SELECT * FROM ssh_access_grants
            WHERE validator_hotkey = ? AND is_active = TRUE
            ORDER BY granted_at DESC
            "#,
        )
        .bind(validator_hotkey)
        .fetch_all(&self.pool)
        .await?;

        Ok(grants)
    }

    /// Record SSH session creation
    pub async fn record_ssh_session_created(
        &self,
        session_id: &str,
        validator_hotkey: &str,
        executor_id: &str,
        expires_at: &DateTime<Utc>,
    ) -> Result<()> {
        sqlx::query(
            r#"
            INSERT INTO ssh_sessions (session_id, validator_hotkey, executor_id, ssh_username, expires_at)
            VALUES (?, ?, ?, ?, ?)
            "#,
        )
        .bind(session_id)
        .bind(validator_hotkey)
        .bind(executor_id)
        .bind(format!("validator_{}", &session_id[..8]))
        .bind(expires_at)
        .execute(&self.pool)
        .await?;

        debug!(
            "Recorded SSH session {} for validator {} -> executor {}",
            session_id, validator_hotkey, executor_id
        );
        Ok(())
    }

    /// Record SSH session revocation
    pub async fn record_ssh_session_revoked(
        &self,
        session_id: &str,
        revocation_reason: &str,
    ) -> Result<()> {
        let now = Utc::now();

        sqlx::query(
            r#"
            UPDATE ssh_sessions
            SET status = 'revoked', revocation_reason = ?, revoked_at = ?
            WHERE session_id = ?
            "#,
        )
        .bind(revocation_reason)
        .bind(now)
        .bind(session_id)
        .execute(&self.pool)
        .await?;

        debug!(
            "Recorded SSH session {} revocation: {}",
            session_id, revocation_reason
        );
        Ok(())
    }

    /// Get active SSH sessions for a validator
    pub async fn get_active_ssh_sessions(
        &self,
        validator_hotkey: &str,
    ) -> Result<Vec<SshSessionRecord>> {
        let sessions = sqlx::query_as::<_, SshSessionRecord>(
            r#"
            SELECT * FROM ssh_sessions
            WHERE validator_hotkey = ? AND status = 'active' AND expires_at > CURRENT_TIMESTAMP
            ORDER BY created_at DESC
            "#,
        )
        .bind(validator_hotkey)
        .fetch_all(&self.pool)
        .await?;

        Ok(sessions)
    }

    /// Clean up expired SSH sessions in database
    pub async fn cleanup_expired_ssh_sessions(&self) -> Result<u64> {
        let now = Utc::now();

        let result = sqlx::query(
            r#"
            UPDATE ssh_sessions
            SET status = 'expired', revocation_reason = 'expired'
            WHERE status = 'active' AND expires_at < ?
            "#,
        )
        .bind(now)
        .execute(&self.pool)
        .await?;

        Ok(result.rows_affected())
    }

    /// Clean up old records (for maintenance)
    pub async fn cleanup_old_records(&self, days_to_keep: i64) -> Result<u64> {
        let cutoff = Utc::now() - chrono::Duration::days(days_to_keep);

        let result = sqlx::query("DELETE FROM validator_interactions WHERE created_at < ?")
            .bind(cutoff)
            .execute(&self.pool)
            .await?;

        Ok(result.rows_affected())
    }

    /// Health check for database connection
    pub async fn health_check(&self) -> Result<()> {
        sqlx::query("SELECT 1")
            .execute(&self.pool)
            .await
            .context("Database health check failed")?;
        Ok(())
    }

    /// Vacuum database to reclaim space
    pub async fn vacuum(&self) -> Result<()> {
        sqlx::query("VACUUM")
            .execute(&self.pool)
            .await
            .context("Database vacuum failed")?;
        Ok(())
    }

    /// Vacuum database into a backup file
    pub async fn vacuum_into(&self, backup_path: &str) -> Result<()> {
        sqlx::query(&format!("VACUUM INTO '{backup_path}'"))
            .execute(&self.pool)
            .await
            .context("Database vacuum into backup failed")?;
        Ok(())
    }

    /// Check database integrity
    pub async fn integrity_check(&self) -> Result<bool> {
        let result: (String,) = sqlx::query_as("PRAGMA integrity_check")
            .fetch_one(&self.pool)
            .await
            .context("Database integrity check failed")?;

        Ok(result.0 == "ok")
    }

    /// Get database statistics
    pub async fn get_database_stats(&self) -> Result<DatabaseStats> {
        // Get page count and page size
        let (page_count,): (i64,) = sqlx::query_as("PRAGMA page_count")
            .fetch_one(&self.pool)
            .await?;

        let (page_size,): (i64,) = sqlx::query_as("PRAGMA page_size")
            .fetch_one(&self.pool)
            .await?;

        // Get table statistics
        let table_stats = self.get_table_statistics().await?;

        Ok(DatabaseStats {
            page_count: page_count as u64,
            page_size: page_size as u64,
            vacuum_count: 0, // SQLite doesn't track this directly
            table_stats,
        })
    }

    /// Get statistics for all tables
    async fn get_table_statistics(&self) -> Result<Vec<TableStatistics>> {
        let table_names: Vec<(String,)> = sqlx::query_as(
            "SELECT name FROM sqlite_master WHERE type='table' AND name NOT LIKE 'sqlite_%'",
        )
        .fetch_all(&self.pool)
        .await?;

        let mut stats = Vec::new();

        for (table_name,) in table_names {
            let (row_count,): (i64,) =
                sqlx::query_as(&format!("SELECT COUNT(*) FROM {table_name}"))
                    .fetch_one(&self.pool)
                    .await
                    .unwrap_or((0,));

            // Estimate size (SQLite doesn't provide exact table sizes easily)
            let size_bytes = (row_count as u64) * 100; // Rough estimate

            stats.push(TableStatistics {
                table_name,
                row_count: row_count as u64,
                size_bytes,
            });
        }

        Ok(stats)
    }

    /// Clean up old validator interactions
    pub async fn cleanup_old_validator_interactions(
        &self,
        cutoff_date: DateTime<Utc>,
    ) -> Result<u64> {
        let result = sqlx::query("DELETE FROM validator_interactions WHERE created_at < ?")
            .bind(cutoff_date)
            .execute(&self.pool)
            .await?;

        Ok(result.rows_affected())
    }

    /// Clean up old SSH grants
    pub async fn cleanup_old_ssh_grants(&self, cutoff_date: DateTime<Utc>) -> Result<u64> {
        let result =
            sqlx::query("DELETE FROM ssh_access_grants WHERE granted_at < ? AND is_active = 0")
                .bind(cutoff_date)
                .execute(&self.pool)
                .await?;

        Ok(result.rows_affected())
    }

    /// Clean up stale executor health records
    pub async fn cleanup_stale_executor_health(&self, cutoff_date: DateTime<Utc>) -> Result<u64> {
        // Only clean up records that haven't been updated recently and are not healthy
        let result = sqlx::query(
            "DELETE FROM executor_health WHERE updated_at < ? AND is_healthy = 0 AND consecutive_failures > 10"
        )
        .bind(cutoff_date)
        .execute(&self.pool)
        .await?;

        Ok(result.rows_affected())
    }

    /// Ensure database directory exists
    async fn ensure_database_directory(database_url: &str) -> Result<()> {
        if let Some(path) = database_url.strip_prefix("sqlite:") {
            let db_path = path.split('?').next().unwrap_or(path);
            if let Some(parent_dir) = Path::new(db_path).parent() {
                if !parent_dir.exists() {
                    debug!("Creating database directory: {:?}", parent_dir);
                    fs::create_dir_all(parent_dir).await.with_context(|| {
                        format!("Failed to create database directory: {parent_dir:?}")
                    })?;
                }
            }
        }
        Ok(())
    }

    pub async fn get_or_create_executor_id(&self, executor_address: &str) -> Result<ExecutorId> {
        // First try to get existing identity
        let existing = sqlx::query_as::<_, (String, String, DateTime<Utc>)>(
            "SELECT uuid, huid, created_at FROM executor_uuids WHERE executor_address = ?",
        )
        .bind(executor_address)
        .fetch_optional(&self.pool)
        .await?;

        if let Some((uid_str, huid, created_at)) = existing {
            let uuid = uuid::Uuid::parse_str(&uid_str)?;
            let executor_id = ExecutorId::from_parts(uuid, huid, created_at.into())?;
            return Ok(executor_id);
        }

        let executor_id = ExecutorId::new(executor_address)?;

        // Try to insert with conflict handling
        match sqlx::query(
            "INSERT INTO executor_uuids (executor_address, uuid, huid, created_at) VALUES (?, ?, ?, ?)",
        )
        .bind(executor_address)
        .bind(executor_id.uuid.to_string())
        .bind(executor_id.huid.clone())
        .bind(DateTime::<Utc>::from(executor_id.created_at()))
        .execute(&self.pool)
        .await
        {
            Ok(_) => Ok(executor_id),
            Err(e) => Err(e.into()),
        }
    }
}

/// Database statistics structure
#[derive(Debug)]
pub struct DatabaseStats {
    pub page_count: u64,
    pub page_size: u64,
    pub vacuum_count: u64,
    pub table_stats: Vec<TableStatistics>,
}

/// Table statistics structure
#[derive(Debug)]
pub struct TableStatistics {
    pub table_name: String,
    pub row_count: u64,
    pub size_bytes: u64,
}

#[cfg(test)]
mod tests {
    use super::*;
    use basilica_common::executor_identity::{constants::is_valid_huid, ExecutorIdentity};

    #[tokio::test]
    async fn test_executor_health_tracking() {
        let config = DatabaseConfig {
            url: "sqlite::memory:".to_string(),
            run_migrations: true,
            ..Default::default()
        };

        let db = RegistrationDb::new(&config).await.unwrap();

        // Update health status
        db.update_executor_health("executor-1", true).await.unwrap();
        db.update_executor_health("executor-2", false)
            .await
            .unwrap();

        // Get all health records
        let health_records = db.get_all_executor_health().await.unwrap();
        assert_eq!(health_records.len(), 2);

        let executor1 = health_records
            .iter()
            .find(|h| h.executor_id == "executor-1")
            .unwrap();
        assert!(executor1.is_healthy);
        assert_eq!(executor1.consecutive_failures, 0);

        let executor2 = health_records
            .iter()
            .find(|h| h.executor_id == "executor-2")
            .unwrap();
        assert!(!executor2.is_healthy);
        assert_eq!(executor2.consecutive_failures, 1);
    }

    #[tokio::test]
    async fn test_validator_interaction_logging() {
        let config = DatabaseConfig {
            url: "sqlite::memory:".to_string(),
            run_migrations: true,
            ..Default::default()
        };

        let db = RegistrationDb::new(&config).await.unwrap();

        // Record interactions
        db.update_validator_interaction("validator-1", true)
            .await
            .unwrap();

        // Small delay to ensure different timestamps
        tokio::time::sleep(tokio::time::Duration::from_millis(10)).await;

        db.record_validator_interaction(
            "validator-1",
            "list_executors",
            true,
            Some(r#"{"count": 5}"#.to_string()),
        )
        .await
        .unwrap();

        // Get recent interactions (should be in reverse chronological order)
        let interactions = db.get_recent_validator_interactions(10).await.unwrap();
        assert_eq!(interactions.len(), 2);

        // Check both interaction types are present (order may vary due to timestamp precision)
        let interaction_types: Vec<&str> = interactions
            .iter()
            .map(|i| i.interaction_type.as_str())
            .collect();
        assert!(interaction_types.contains(&"authentication"));
        assert!(interaction_types.contains(&"list_executors"));
    }

    // ===== AUTOMATIC IDENTITY GENERATION TESTS =====

    #[tokio::test]
    async fn test_get_or_create_executor_id_first_time() {
        let config = DatabaseConfig {
            url: "sqlite::memory:".to_string(),
            run_migrations: true,
            ..Default::default()
        };

        let db = RegistrationDb::new(&config).await.unwrap();

        // First call should create a new identity
        let executor_id = db
            .get_or_create_executor_id("127.0.0.1:50051")
            .await
            .unwrap();

        // Verify the identity was generated correctly
        assert!(is_valid_huid(&executor_id.huid));
        assert_eq!(executor_id.uuid.get_version(), Some(uuid::Version::Random));
        assert!(!executor_id.uuid.to_string().is_empty());
        assert!(!executor_id.huid.is_empty());

        // Verify the identity was stored in the database
        let stored_id = db
            .get_or_create_executor_id("127.0.0.1:50051")
            .await
            .unwrap();
        assert_eq!(stored_id.uuid, executor_id.uuid);
        assert_eq!(stored_id.huid, executor_id.huid);
    }

    #[tokio::test]
    async fn test_get_or_create_executor_id_retrieval_consistency() {
        let config = DatabaseConfig {
            url: "sqlite::memory:".to_string(),
            run_migrations: true,
            ..Default::default()
        };

        let db = RegistrationDb::new(&config).await.unwrap();

        let address = "192.168.1.100:8080";

        // Create identity
        let id1 = db.get_or_create_executor_id(address).await.unwrap();

        // Retrieve multiple times - should always return the same identity
        for _ in 0..5 {
            let id2 = db.get_or_create_executor_id(address).await.unwrap();
            assert_eq!(id2.uuid, id1.uuid);
            assert_eq!(id2.huid, id1.huid);
        }
    }

    #[tokio::test]
    async fn test_get_or_create_executor_id_multiple_executors() {
        let config = DatabaseConfig {
            url: "sqlite::memory:".to_string(),
            run_migrations: true,
            ..Default::default()
        };

        let db = RegistrationDb::new(&config).await.unwrap();

        let addresses = vec![
            "127.0.0.1:50051",
            "127.0.0.1:50052",
            "192.168.1.100:8080",
            "10.0.0.50:9090",
        ];

        let mut identities = Vec::new();

        // Create identities for multiple executors
        for address in &addresses {
            let id = db.get_or_create_executor_id(address).await.unwrap();
            identities.push((address.to_string(), id));
        }

        // Verify all identities are unique
        let mut uuids = std::collections::HashSet::new();
        let mut huids = std::collections::HashSet::new();

        for (_, id) in &identities {
            assert!(uuids.insert(id.uuid));
            assert!(huids.insert(id.huid.clone()));
        }

        // Verify each address maps to the correct identity
        for (address, expected_id) in &identities {
            let retrieved_id = db.get_or_create_executor_id(address).await.unwrap();
            assert_eq!(retrieved_id.uuid, expected_id.uuid);
            assert_eq!(retrieved_id.huid, expected_id.huid);
        }
    }

    #[tokio::test]
    async fn test_get_or_create_executor_id_database_persistence() {
        let config = DatabaseConfig {
            url: "sqlite::memory:".to_string(),
            run_migrations: true,
            ..Default::default()
        };

        let db = RegistrationDb::new(&config).await.unwrap();

        let address = "127.0.0.1:50051";

        // Create identity
        let original_id = db.get_or_create_executor_id(address).await.unwrap();

        // Verify it's stored in the database by querying directly
        let stored = sqlx::query_as::<_, (String, String, String)>(
            "SELECT executor_address, uuid, huid FROM executor_uuids WHERE executor_address = ?",
        )
        .bind(address)
        .fetch_one(&db.pool)
        .await
        .unwrap();

        assert_eq!(stored.0, address);
        assert_eq!(stored.1, original_id.uuid.to_string());
        assert_eq!(stored.2, original_id.huid);
    }

    #[tokio::test]
    async fn test_get_or_create_executor_id_format_validation() {
        let config = DatabaseConfig {
            url: "sqlite::memory:".to_string(),
            run_migrations: true,
            ..Default::default()
        };

        let db = RegistrationDb::new(&config).await.unwrap();

        // Generate multiple identities to test format consistency
        for i in 0..10 {
            let address = format!("127.0.0.1:{}", 50051 + i);
            let id = db.get_or_create_executor_id(&address).await.unwrap();

            // Verify HUID format
            assert!(is_valid_huid(&id.huid), "HUID should be valid: {}", id.huid);

            // Verify UUID format
            assert_eq!(id.uuid.get_version(), Some(uuid::Version::Random));
            assert_eq!(id.uuid.to_string().len(), 36); // Standard UUID length
        }
    }

    #[tokio::test]
    async fn test_get_or_create_executor_id_edge_cases() {
        let config = DatabaseConfig {
            url: "sqlite::memory:".to_string(),
            run_migrations: true,
            ..Default::default()
        };

        let db = RegistrationDb::new(&config).await.unwrap();

        // Test with various address formats
        let test_addresses = vec![
            "localhost:50051",
            "0.0.0.0:8080",
            "::1:9090",
            "example.com:12345",
            "192.168.1.1:1",
            "10.0.0.1:65535",
        ];

        for address in test_addresses {
            let id = db.get_or_create_executor_id(address).await.unwrap();
            assert!(is_valid_huid(&id.huid));
            assert_eq!(id.uuid.get_version(), Some(uuid::Version::Random));
        }
    }

    #[tokio::test]
    async fn test_get_or_create_executor_id_uniqueness_across_generations() {
        let config = DatabaseConfig {
            url: "sqlite::memory:".to_string(),
            run_migrations: true,
            ..Default::default()
        };

        let db = RegistrationDb::new(&config).await.unwrap();

        let mut uuids = std::collections::HashSet::new();
        let mut huids = std::collections::HashSet::new();

        // Generate many identities to test uniqueness
        for i in 0..50 {
            let address = format!("127.0.0.1:{}", 50051 + i);
            let id = db.get_or_create_executor_id(&address).await.unwrap();

            // Verify UUID uniqueness
            assert!(
                uuids.insert(id.uuid),
                "UUID collision detected at iteration {}: {}",
                i,
                id.uuid
            );

            // Verify HUID uniqueness
            assert!(
                huids.insert(id.huid.clone()),
                "HUID collision detected at iteration {}: {}",
                i,
                id.huid
            );
        }

        assert_eq!(uuids.len(), 50);
        assert_eq!(huids.len(), 50);
    }

    #[tokio::test]
    async fn test_get_or_create_executor_id_database_integrity() {
        let config = DatabaseConfig {
            url: "sqlite::memory:".to_string(),
            run_migrations: true,
            ..Default::default()
        };

        let db = RegistrationDb::new(&config).await.unwrap();

        // Create several identities
        let addresses = vec!["127.0.0.1:50051", "127.0.0.1:50052", "192.168.1.100:8080"];

        for address in &addresses {
            db.get_or_create_executor_id(address).await.unwrap();
        }

        // Verify database integrity
        let integrity_check = db.integrity_check().await.unwrap();
        assert!(integrity_check, "Database integrity check should pass");

        // Verify table statistics
        let stats = db.get_database_stats().await.unwrap();
        let executor_uuids_stats = stats
            .table_stats
            .iter()
            .find(|s| s.table_name == "executor_uuids")
            .unwrap();
        assert_eq!(executor_uuids_stats.row_count, 3);
    }

    #[tokio::test]
    async fn test_get_or_create_executor_id_error_handling() {
        // Test with invalid database URL format (should fail gracefully)
        let config = DatabaseConfig {
            url: "invalid://database/url".to_string(),
            run_migrations: true,
            ..Default::default()
        };

        let result = RegistrationDb::new(&config).await;
        assert!(
            result.is_err(),
            "Should fail with invalid database URL format"
        );
    }

    #[tokio::test]
    async fn test_get_or_create_executor_id_empty_address() {
        let config = DatabaseConfig {
            url: "sqlite::memory:".to_string(),
            run_migrations: true,
            ..Default::default()
        };

        let db = RegistrationDb::new(&config).await.unwrap();

        // Test with empty address (edge case)
        let id = db.get_or_create_executor_id("").await.unwrap();
        assert!(is_valid_huid(&id.huid));
        assert_eq!(id.uuid.get_version(), Some(uuid::Version::Random));
    }

    #[tokio::test]
    async fn test_get_or_create_executor_id_special_characters() {
        let config = DatabaseConfig {
            url: "sqlite::memory:".to_string(),
            run_migrations: true,
            ..Default::default()
        };

        let db = RegistrationDb::new(&config).await.unwrap();

        // Test with addresses containing special characters
        let test_addresses = vec![
            "test-host:50051",
            "my-executor.local:8080",
            "executor-01.example.com:9090",
            "192.168.1.100:12345",
        ];

        for address in test_addresses {
            let id = db.get_or_create_executor_id(address).await.unwrap();
            assert!(is_valid_huid(&id.huid));
            assert_eq!(id.uuid.get_version(), Some(uuid::Version::Random));
        }
    }

    #[tokio::test]
    async fn test_executor_id_timestamp_parsing() {
        let config = DatabaseConfig {
            url: "sqlite::memory:".to_string(),
            run_migrations: true,
            ..Default::default()
        };

        let db = RegistrationDb::new(&config).await.unwrap();

        // Create a new executor ID
        let original_executor_id = db
            .get_or_create_executor_id("test-executor:50051")
            .await
            .unwrap();

        // Verify the identity was created correctly
        assert!(is_valid_huid(&original_executor_id.huid));
        assert_eq!(original_executor_id.uuid().to_string().len(), 36);

        // Get the same executor ID back from the database
        let retrieved_executor_id = db
            .get_or_create_executor_id("test-executor:50051")
            .await
            .unwrap();

        // Verify all fields match exactly
        assert_eq!(original_executor_id.uuid(), retrieved_executor_id.uuid());
        assert_eq!(original_executor_id.huid(), retrieved_executor_id.huid());
        assert_eq!(
            original_executor_id.created_at(),
            retrieved_executor_id.created_at()
        );
    }
}
