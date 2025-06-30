//! # Persistence Traits
//!
//! Core traits for database operations and persistence management.

use async_trait::async_trait;
use serde::{Deserialize, Serialize};
// HashMap import removed as unused in this file
use std::fmt::Debug;

use crate::error::BasilcaError;

/// Generic repository trait for CRUD operations
#[async_trait]
pub trait Repository<T, K>
where
    T: Send + Sync,
    K: Send + Sync,
{
    type Error: BasilcaError;

    /// Create a new entity
    async fn create(&self, entity: &T) -> Result<(), Self::Error>;

    /// Get entity by ID
    async fn get_by_id(&self, id: &K) -> Result<Option<T>, Self::Error>;

    /// Update existing entity
    async fn update(&self, entity: &T) -> Result<(), Self::Error>;

    /// Delete entity by ID
    async fn delete(&self, id: &K) -> Result<bool, Self::Error>;

    /// Check if entity exists
    async fn exists(&self, id: &K) -> Result<bool, Self::Error>;

    /// List entities with pagination
    async fn list(&self, limit: u32, offset: u32) -> Result<Vec<T>, Self::Error>;

    /// Count total entities
    async fn count(&self) -> Result<u64, Self::Error>;
}

/// Database connection abstraction
#[async_trait]
pub trait DatabaseConnection {
    /// Execute a health check
    async fn health_check(&self) -> Result<(), anyhow::Error>;

    /// Close the connection
    async fn close(&self);

    /// Begin a transaction
    async fn begin_transaction(&self) -> Result<Box<dyn DatabaseTransaction>, anyhow::Error>;

    /// Get connection pool statistics if available
    async fn connection_stats(&self) -> Result<Option<ConnectionStats>, anyhow::Error> {
        Ok(None)
    }
}

/// Database transaction abstraction
#[async_trait]
pub trait DatabaseTransaction {
    /// Commit the transaction
    async fn commit(self: Box<Self>) -> Result<(), anyhow::Error>;

    /// Rollback the transaction
    async fn rollback(self: Box<Self>) -> Result<(), anyhow::Error>;
}

/// Migration manager
#[async_trait]
pub trait MigrationManager {
    /// Run all pending migrations
    async fn run_migrations(&self) -> Result<(), anyhow::Error>;

    /// Get current schema version
    async fn get_current_version(&self) -> Result<i32, anyhow::Error>;

    /// Validate schema integrity
    async fn validate_schema(&self) -> Result<(), anyhow::Error>;

    /// Check migration status
    async fn migration_status(&self) -> Result<MigrationStatus, anyhow::Error>;

    /// Rollback to a specific migration version (if supported)
    async fn rollback_to(&self, _version: u64) -> Result<(), anyhow::Error> {
        Err(anyhow::anyhow!("Rollback not supported"))
    }
}

/// Database statistics provider
#[async_trait]
pub trait DatabaseStatsProvider {
    /// Get comprehensive database statistics
    async fn get_stats(&self) -> Result<DatabaseStats, anyhow::Error>;

    /// Get table row counts
    async fn get_table_counts(
        &self,
    ) -> Result<std::collections::HashMap<String, u64>, anyhow::Error>;

    /// Get database size information
    async fn get_size_info(&self) -> Result<DatabaseSizeInfo, anyhow::Error>;
}

/// Cleanup old records and maintain database health
#[async_trait]
pub trait Cleanup {
    /// Remove records older than specified retention period
    async fn cleanup_old_records(&self, retention_days: i64) -> Result<u64, anyhow::Error>;

    /// Vacuum/optimize database storage
    async fn optimize_storage(&self) -> Result<(), anyhow::Error> {
        Ok(()) // Default no-op implementation
    }

    /// Get database storage statistics
    async fn storage_stats(&self) -> Result<StorageStats, anyhow::Error>;
}

/// Query builder trait for complex queries
#[async_trait]
pub trait QueryBuilder<T, F>
where
    T: Send + Sync,
    F: Send + Sync,
{
    type Error: BasilcaError;

    /// Apply filter to query
    fn filter(self, filter: F) -> Self;

    /// Add ordering
    fn order_by(self, field: &str, ascending: bool) -> Self;

    /// Set pagination
    fn paginate(self, limit: u32, offset: u32) -> Self;

    /// Execute query and return results
    async fn execute(self) -> Result<Vec<T>, Self::Error>;

    /// Execute query and return count
    async fn count(self) -> Result<u64, Self::Error>;
}

/// Transaction management for atomic operations
#[async_trait]
pub trait Transactional {
    type Transaction;
    type Error: BasilcaError;

    /// Start a new transaction
    async fn begin_transaction(&self) -> Result<Self::Transaction, Self::Error>;

    /// Commit transaction
    async fn commit_transaction(&self, tx: Self::Transaction) -> Result<(), Self::Error>;

    /// Rollback transaction
    async fn rollback_transaction(&self, tx: Self::Transaction) -> Result<(), Self::Error>;
}

/// Connection pool statistics
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct ConnectionStats {
    pub active_connections: u32,
    pub idle_connections: u32,
    pub max_connections: u32,
    pub total_connections: u64,
    pub failed_connections: u64,
}

/// Migration status information
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct MigrationStatus {
    pub current_version: u64,
    pub latest_version: u64,
    pub pending_migrations: Vec<String>,
    pub applied_migrations: Vec<String>,
}

/// Database statistics
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct DatabaseStats {
    pub total_size_bytes: u64,
    pub table_count: u32,
    pub index_count: u32,
}

/// Database size information
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct DatabaseSizeInfo {
    pub total_size_bytes: u64,
    pub table_sizes: std::collections::HashMap<String, u64>,
    pub index_sizes: std::collections::HashMap<String, u64>,
}

/// Database storage statistics
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct StorageStats {
    pub total_size_bytes: u64,
    pub data_size_bytes: u64,
    pub index_size_bytes: u64,
    pub temp_size_bytes: u64,
    pub table_count: u32,
    pub index_count: u32,
}
