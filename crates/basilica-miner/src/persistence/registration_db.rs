//! # Registration Database
//!
//! Simplified SQLite database for the miner with node UUID tracking

use anyhow::{Context, Result};
use sqlx::SqlitePool;
use std::path::Path;
use tokio::fs;
use tracing::{debug, info};

use basilica_common::{config::DatabaseConfig, node_identity::NodeId};

/// Registration database client
#[derive(Debug, Clone)]
pub struct RegistrationDb {
    pool: SqlitePool,
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
        info!("Running database migrations");

        sqlx::migrate!("./migrations")
            .run(&self.pool)
            .await
            .context("Failed to run migrations")?;

        info!("Database migrations completed successfully");
        Ok(())
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

    /// Generate a deterministic node ID based on SSH credentials (in-memory only)
    ///
    /// # Arguments
    /// * `username` - SSH username
    /// * `host` - SSH hostname or IP address
    /// * `port` - SSH port
    ///
    /// # Returns
    /// A deterministic NodeId based on the SSH credentials. The same credentials
    /// will always generate the same node ID. This is generated in-memory only
    /// and not persisted to the database.
    pub async fn get_or_create_node_id(
        &self,
        username: &str,
        host: &str,
        port: u16,
    ) -> Result<NodeId> {
        // Create deterministic seed from SSH credentials
        let ssh_credentials = format!("{}@{}:{}", username, host, port);

        // Generate deterministic NodeId from SSH credentials (in-memory only)
        let node_id = NodeId::new(&ssh_credentials)?;

        Ok(node_id)
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
    use basilica_common::node_identity::NodeIdentity;

    // ===== AUTOMATIC IDENTITY GENERATION TESTS =====

    #[tokio::test]
    async fn test_get_or_create_node_id_first_time() {
        let config = DatabaseConfig {
            url: "sqlite::memory:".to_string(),
            run_migrations: true,
            ..Default::default()
        };

        let db = RegistrationDb::new(&config).await.unwrap();

        // First call should create a new identity
        let node_id = db
            .get_or_create_node_id("testuser", "127.0.0.1", 50051)
            .await
            .unwrap();

        // Verify the identity was generated correctly
        assert_eq!(node_id.uuid().get_version(), Some(uuid::Version::Random));
        assert!(!node_id.uuid().to_string().is_empty());

        // Verify the identity is deterministically generated (same credentials = same ID)
        let stored_id = db
            .get_or_create_node_id("testuser", "127.0.0.1", 50051)
            .await
            .unwrap();
        assert_eq!(stored_id.uuid(), node_id.uuid());
        assert_eq!(stored_id.created_at(), node_id.created_at());
    }

    #[tokio::test]
    async fn test_get_or_create_node_id_retrieval_consistency() {
        let config = DatabaseConfig {
            url: "sqlite::memory:".to_string(),
            run_migrations: true,
            ..Default::default()
        };

        let db = RegistrationDb::new(&config).await.unwrap();

        // Create identity
        let id1 = db
            .get_or_create_node_id("testuser", "192.168.1.100", 8080)
            .await
            .unwrap();

        // Generate multiple times - should always return the same identity due to determinism
        for _ in 0..5 {
            let id2 = db
                .get_or_create_node_id("testuser", "192.168.1.100", 8080)
                .await
                .unwrap();
            assert_eq!(id2.uuid(), id1.uuid());
            assert_eq!(id2.created_at(), id1.created_at());
        }
    }

    #[tokio::test]
    async fn test_get_or_create_node_id_multiple_nodes() {
        let config = DatabaseConfig {
            url: "sqlite::memory:".to_string(),
            run_migrations: true,
            ..Default::default()
        };

        let db = RegistrationDb::new(&config).await.unwrap();

        let nodes = vec![
            ("user1", "127.0.0.1", 50051),
            ("user2", "127.0.0.1", 50052),
            ("user3", "192.168.1.100", 8080),
            ("user4", "10.0.0.50", 9090),
        ];

        let mut identities = Vec::new();

        // Create identities for multiple nodes
        for (username, host, port) in &nodes {
            let id = db
                .get_or_create_node_id(username, host, *port)
                .await
                .unwrap();
            identities.push((format!("{}@{}:{}", username, host, port), id));
        }

        // Verify all identities are unique
        let mut uuids = std::collections::HashSet::new();

        for (_, id) in &identities {
            assert!(uuids.insert(*id.uuid()));
        }

        // Verify each node maps to the correct identity
        for (credentials, expected_id) in &identities {
            let parts: Vec<&str> = credentials.split('@').collect();
            let username = parts[0];
            let host_port: Vec<&str> = parts[1].split(':').collect();
            let host = host_port[0];
            let port: u16 = host_port[1].parse().unwrap();

            let retrieved_id = db
                .get_or_create_node_id(username, host, port)
                .await
                .unwrap();
            assert_eq!(retrieved_id.uuid(), expected_id.uuid());
            assert_eq!(retrieved_id.created_at(), expected_id.created_at());
        }
    }

    #[tokio::test]
    async fn test_get_or_create_node_id_determinism() {
        let config = DatabaseConfig {
            url: "sqlite::memory:".to_string(),
            run_migrations: true,
            ..Default::default()
        };

        let db = RegistrationDb::new(&config).await.unwrap();

        // Create identity multiple times with same credentials
        let id1 = db
            .get_or_create_node_id("testuser", "127.0.0.1", 50051)
            .await
            .unwrap();

        let id2 = db
            .get_or_create_node_id("testuser", "127.0.0.1", 50051)
            .await
            .unwrap();

        let id3 = db
            .get_or_create_node_id("testuser", "127.0.0.1", 50051)
            .await
            .unwrap();

        // Verify deterministic generation (same credentials = same IDs)
        assert_eq!(id1.uuid(), id2.uuid());
        assert_eq!(id1.uuid(), id3.uuid());
        assert_eq!(id1.created_at(), id2.created_at());
        assert_eq!(id1.created_at(), id3.created_at());
    }

    #[tokio::test]
    async fn test_get_or_create_node_id_format_validation() {
        let config = DatabaseConfig {
            url: "sqlite::memory:".to_string(),
            run_migrations: true,
            ..Default::default()
        };

        let db = RegistrationDb::new(&config).await.unwrap();

        // Generate multiple identities to test format consistency
        for i in 0..10 {
            let port = 50051 + i;
            let id = db
                .get_or_create_node_id("testuser", "127.0.0.1", port)
                .await
                .unwrap();

            // Verify UUID format
            assert_eq!(id.uuid().get_version(), Some(uuid::Version::Random));
            assert_eq!(id.uuid().to_string().len(), 36); // Standard UUID length
        }
    }

    #[tokio::test]
    async fn test_get_or_create_node_id_edge_cases() {
        let config = DatabaseConfig {
            url: "sqlite::memory:".to_string(),
            run_migrations: true,
            ..Default::default()
        };

        let db = RegistrationDb::new(&config).await.unwrap();

        // Test with various host/port combinations
        let test_nodes = vec![
            ("user", "localhost", 50051),
            ("root", "0.0.0.0", 8080),
            ("admin", "::1", 9090),
            ("ubuntu", "example.com", 12345),
            ("test", "192.168.1.1", 1),
            ("prod", "10.0.0.1", 65535),
        ];

        for (username, host, port) in test_nodes {
            let id = db
                .get_or_create_node_id(username, host, port)
                .await
                .unwrap();
            assert_eq!(id.uuid().get_version(), Some(uuid::Version::Random));
        }
    }

    #[tokio::test]
    async fn test_get_or_create_node_id_uniqueness_across_generations() {
        let config = DatabaseConfig {
            url: "sqlite::memory:".to_string(),
            run_migrations: true,
            ..Default::default()
        };

        let db = RegistrationDb::new(&config).await.unwrap();

        let mut uuids = std::collections::HashSet::new();

        // Generate many identities to test uniqueness
        for i in 0..50 {
            let port = 50051 + i;
            let username = format!("user{}", i);
            let id = db
                .get_or_create_node_id(&username, "127.0.0.1", port)
                .await
                .unwrap();

            // Verify UUID uniqueness
            assert!(
                uuids.insert(*id.uuid()),
                "UUID collision detected at iteration {}: {}",
                i,
                id.uuid()
            );
        }

        assert_eq!(uuids.len(), 50);
    }

    #[tokio::test]
    async fn test_get_or_create_node_id_different_credentials() {
        let config = DatabaseConfig {
            url: "sqlite::memory:".to_string(),
            run_migrations: true,
            ..Default::default()
        };

        let db = RegistrationDb::new(&config).await.unwrap();

        // Create identities with different credentials
        let nodes = vec![
            ("user1", "127.0.0.1", 50051),
            ("user2", "127.0.0.1", 50052),
            ("user3", "192.168.1.100", 8080),
        ];

        let mut ids = Vec::new();
        for (username, host, port) in &nodes {
            let id = db
                .get_or_create_node_id(username, host, *port)
                .await
                .unwrap();
            ids.push(id);
        }

        // Verify all IDs are unique (different credentials = different IDs)
        assert_ne!(ids[0].uuid(), ids[1].uuid());
        assert_ne!(ids[0].uuid(), ids[2].uuid());
        assert_ne!(ids[1].uuid(), ids[2].uuid());

        // Verify database integrity (table exists but empty)
        let integrity_check = db.integrity_check().await.unwrap();
        assert!(integrity_check, "Database integrity check should pass");
    }

    #[tokio::test]
    async fn test_get_or_create_node_id_error_handling() {
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
    async fn test_get_or_create_node_id_empty_username() {
        let config = DatabaseConfig {
            url: "sqlite::memory:".to_string(),
            run_migrations: true,
            ..Default::default()
        };

        let db = RegistrationDb::new(&config).await.unwrap();

        // Test with empty username (edge case)
        let id = db
            .get_or_create_node_id("", "127.0.0.1", 50051)
            .await
            .unwrap();
        assert_eq!(id.uuid().get_version(), Some(uuid::Version::Random));
    }

    #[tokio::test]
    async fn test_get_or_create_node_id_special_characters() {
        let config = DatabaseConfig {
            url: "sqlite::memory:".to_string(),
            run_migrations: true,
            ..Default::default()
        };

        let db = RegistrationDb::new(&config).await.unwrap();

        // Test with hosts containing special characters
        let test_nodes = vec![
            ("user", "test-host", 50051),
            ("admin", "my-node.local", 8080),
            ("root", "node-01.example.com", 9090),
            ("basilica", "192.168.1.100", 12345),
        ];

        for (username, host, port) in test_nodes {
            let id = db
                .get_or_create_node_id(username, host, port)
                .await
                .unwrap();
            assert_eq!(id.uuid().get_version(), Some(uuid::Version::Random));
        }
    }

    #[tokio::test]
    async fn test_node_id_timestamp_parsing() {
        let config = DatabaseConfig {
            url: "sqlite::memory:".to_string(),
            run_migrations: true,
            ..Default::default()
        };

        let db = RegistrationDb::new(&config).await.unwrap();

        // Create a new node ID
        let original_node_id = db
            .get_or_create_node_id("testuser", "test-node", 50051)
            .await
            .unwrap();

        // Verify the identity was created correctly
        assert_eq!(original_node_id.uuid().to_string().len(), 36);

        // Get the same node ID back from the database
        let retrieved_node_id = db
            .get_or_create_node_id("testuser", "test-node", 50051)
            .await
            .unwrap();

        // Verify all fields match exactly
        assert_eq!(original_node_id.uuid(), retrieved_node_id.uuid());
        assert_eq!(
            original_node_id.created_at(),
            retrieved_node_id.created_at()
        );
    }
}
