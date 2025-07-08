//! Integration helpers for using the executor identity system
//!
//! This module provides utilities to help integrate the UUID+HUID system
//! into existing codebases with minimal disruption.

#[cfg(feature = "sqlite")]
use anyhow::{Context, Result};
#[cfg(feature = "sqlite")]
use std::sync::Arc;
#[cfg(feature = "sqlite")]
use tracing::{debug, info};

use crate::executor_identity::ExecutorIdentity;

#[cfg(feature = "sqlite")]
use crate::executor_identity::{
    IdentityMigrationManager, IdentityPersistence, MigrationConfig, SqliteIdentityStore,
};

/// Factory for creating identity-aware database connections
pub struct IdentityDbFactory {
    /// Database URL
    #[allow(dead_code)]
    database_url: String,
    /// Whether to run migrations automatically
    auto_migrate: bool,
}

impl IdentityDbFactory {
    /// Create a new factory
    pub fn new(database_url: String) -> Self {
        Self {
            database_url,
            auto_migrate: true,
        }
    }

    /// Set whether to run migrations automatically
    pub fn with_auto_migrate(mut self, auto_migrate: bool) -> Self {
        self.auto_migrate = auto_migrate;
        self
    }

    /// Create an identity store with automatic setup
    #[cfg(feature = "sqlite")]
    pub async fn create_identity_store(&self) -> Result<SqliteIdentityStore> {
        info!("Creating identity store with URL: {}", self.database_url);

        let store = SqliteIdentityStore::new(&self.database_url)
            .await
            .context("Failed to create identity store")?;

        if self.auto_migrate {
            // Run a basic migration check
            let migration_manager = IdentityMigrationManager::new(&self.database_url).await?;
            let config = MigrationConfig {
                dry_run: true,
                ..Default::default()
            };

            let legacy_ids = migration_manager.scan_legacy_ids(&config).await?;
            if !legacy_ids.is_empty() {
                info!(
                    "Found {} legacy IDs that could be migrated. Run migration manually if needed.",
                    legacy_ids.len()
                );
            }
        }

        Ok(store)
    }

    /// Create a shared identity store wrapped in Arc
    #[cfg(feature = "sqlite")]
    pub async fn create_shared_store(&self) -> Result<Arc<SqliteIdentityStore>> {
        let store = self.create_identity_store().await?;
        Ok(Arc::new(store))
    }
}

/// Extension trait for SQLite pools to add identity support
#[cfg(feature = "sqlite")]
#[async_trait::async_trait]
pub trait IdentityPoolExt {
    /// Create an identity store from this pool
    async fn create_identity_store(&self) -> Result<SqliteIdentityStore>;

    /// Get or create the executor identity for this instance
    async fn get_executor_identity(&self) -> Result<Box<dyn ExecutorIdentity>>;
}

#[cfg(feature = "sqlite")]
#[async_trait::async_trait]
impl IdentityPoolExt for sqlx::SqlitePool {
    async fn create_identity_store(&self) -> Result<SqliteIdentityStore> {
        SqliteIdentityStore::from_pool(self.clone()).await
    }

    async fn get_executor_identity(&self) -> Result<Box<dyn ExecutorIdentity>> {
        let store = self.create_identity_store().await?;
        store.get_or_create().await
    }
}

/// Helper struct for working with executor identities in a transactional context
#[cfg(feature = "sqlite")]
pub struct IdentityTransaction {
    store: Arc<SqliteIdentityStore>,
}

#[cfg(feature = "sqlite")]
impl IdentityTransaction {
    /// Create a new identity transaction helper
    pub fn new(store: Arc<SqliteIdentityStore>) -> Self {
        Self { store }
    }

    /// Get or create identity with retry logic
    pub async fn get_or_create_with_retry(
        &self,
        max_retries: u32,
    ) -> Result<Box<dyn ExecutorIdentity>> {
        let mut last_error = None;

        for attempt in 0..=max_retries {
            match self.store.get_or_create().await {
                Ok(identity) => return Ok(identity),
                Err(e) => {
                    last_error = Some(e);
                    if attempt < max_retries {
                        debug!("Identity creation attempt {} failed, retrying", attempt + 1);
                        tokio::time::sleep(std::time::Duration::from_millis(
                            100 * (attempt + 1) as u64,
                        ))
                        .await;
                    }
                }
            }
        }

        Err(last_error.unwrap_or_else(|| anyhow::anyhow!("Failed to create identity")))
    }

    /// Find identity with caching and fallback
    pub async fn find_with_fallback(
        &self,
        id: &str,
        create_if_missing: bool,
    ) -> Result<Box<dyn ExecutorIdentity>> {
        // Try to find existing
        if let Some(identity) = self.store.find_by_identifier(id).await? {
            return Ok(identity);
        }

        if create_if_missing {
            info!("Identity '{}' not found, creating new one", id);
            self.store.get_or_create().await
        } else {
            anyhow::bail!("Identity '{}' not found", id);
        }
    }
}

/// Configuration for identity-aware database operations
#[derive(Debug, Clone)]
pub struct IdentityConfig {
    /// Whether to cache identities in memory
    pub enable_cache: bool,
    /// Cache size limit
    pub cache_size: usize,
    /// Whether to validate HUID format on all operations
    pub validate_huid: bool,
    /// Whether to log identity operations
    pub log_operations: bool,
}

impl Default for IdentityConfig {
    fn default() -> Self {
        Self {
            enable_cache: true,
            cache_size: 1000,
            validate_huid: true,
            log_operations: true,
        }
    }
}

/// Macro to simplify identity-aware database operations
#[macro_export]
macro_rules! with_identity {
    ($store:expr, $operation:expr) => {{
        let identity = $store.get_or_create().await?;
        if $crate::executor_identity::integration::IdentityConfig::default().log_operations {
            ::tracing::debug!("Executing operation with identity: {}", identity.huid());
        }
        $operation(identity)
    }};
}

/// Helper function to format identity for logging
pub fn format_identity_log(identity: &dyn ExecutorIdentity) -> String {
    format!(
        "ExecutorID[uuid={}, huid={}, created={}]",
        identity.short_uuid(),
        identity.huid(),
        chrono::DateTime::<chrono::Utc>::from(identity.created_at()).format("%Y-%m-%d %H:%M:%S")
    )
}

/// Helper to create identity-aware SQL queries
pub struct IdentityQueryBuilder {
    table: String,
    id_column: String,
}

impl IdentityQueryBuilder {
    /// Create a new query builder
    pub fn new(table: impl Into<String>, id_column: impl Into<String>) -> Self {
        Self {
            table: table.into(),
            id_column: id_column.into(),
        }
    }

    /// Build an INSERT query with identity
    pub fn insert_with_identity(
        &self,
        _identity: &dyn ExecutorIdentity,
        additional_columns: &[(&str, &str)],
    ) -> String {
        let mut columns = vec![self.id_column.as_str()];
        let mut values = vec!["?"];

        for (col, _) in additional_columns {
            columns.push(col);
            values.push("?");
        }

        format!(
            "INSERT INTO {} ({}) VALUES ({})",
            self.table,
            columns.join(", "),
            values.join(", ")
        )
    }

    /// Build an UPDATE query for identity
    pub fn update_by_identity(&self, updates: &[(&str, &str)]) -> String {
        let set_clauses: Vec<String> = updates
            .iter()
            .map(|(col, _)| format!("{col} = ?"))
            .collect();

        format!(
            "UPDATE {} SET {} WHERE {} = ?",
            self.table,
            set_clauses.join(", "),
            self.id_column
        )
    }

    /// Build a SELECT query by identity
    pub fn select_by_identity(&self, columns: &[&str]) -> String {
        let column_list = if columns.is_empty() {
            "*".to_string()
        } else {
            columns.join(", ")
        };

        format!(
            "SELECT {} FROM {} WHERE {} = ?",
            column_list, self.table, self.id_column
        )
    }
}

// Fallback trait when sqlite feature is not enabled
#[cfg(not(feature = "sqlite"))]
pub trait IdentityPoolExt {}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_identity_config_defaults() {
        let config = IdentityConfig::default();
        assert!(config.enable_cache);
        assert_eq!(config.cache_size, 1000);
        assert!(config.validate_huid);
        assert!(config.log_operations);
    }

    #[test]
    fn test_query_builder() {
        let builder = IdentityQueryBuilder::new("executors", "id");

        // Test INSERT query
        let insert_query =
            builder.insert_with_identity(&DummyIdentity, &[("name", "test"), ("status", "active")]);
        assert_eq!(
            insert_query,
            "INSERT INTO executors (id, name, status) VALUES (?, ?, ?)"
        );

        // Test UPDATE query
        let update_query =
            builder.update_by_identity(&[("status", "inactive"), ("updated_at", "now")]);
        assert_eq!(
            update_query,
            "UPDATE executors SET status = ?, updated_at = ? WHERE id = ?"
        );

        // Test SELECT query
        let select_query = builder.select_by_identity(&["id", "name", "status"]);
        assert_eq!(
            select_query,
            "SELECT id, name, status FROM executors WHERE id = ?"
        );
    }

    // Dummy identity for testing
    struct DummyIdentity;

    impl ExecutorIdentity for DummyIdentity {
        fn uuid(&self) -> &uuid::Uuid {
            unimplemented!()
        }
        fn huid(&self) -> &str {
            "test-identity-1234"
        }
        fn created_at(&self) -> std::time::SystemTime {
            std::time::SystemTime::now()
        }
        fn matches(&self, _query: &str) -> bool {
            false
        }
        fn full_display(&self) -> String {
            "test".to_string()
        }
        fn short_uuid(&self) -> String {
            "12345678".to_string()
        }
    }

    #[tokio::test]
    async fn test_identity_db_factory() {
        let factory =
            IdentityDbFactory::new("sqlite::memory:".to_string()).with_auto_migrate(false);

        assert!(!factory.auto_migrate);
    }
}
