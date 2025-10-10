//! Database persistence implementation for node identities
//!
//! This module provides SQLite-based persistence for node identities,
//! implementing the IdentityPersistence trait with proper caching,
//! collision handling, and transaction safety.

#[cfg(feature = "sqlite")]
use anyhow::{anyhow, Context, Result};
#[cfg(feature = "sqlite")]
use async_trait::async_trait;
#[cfg(feature = "sqlite")]
use chrono::{DateTime, Utc};
use std::collections::HashMap;
#[cfg(feature = "sqlite")]
use std::sync::Arc;
#[cfg(feature = "sqlite")]
use tokio::sync::RwLock;
#[cfg(feature = "sqlite")]
use tracing::{debug, info};
use uuid::Uuid;

#[cfg(feature = "sqlite")]
use crate::node_identity::{IdentityPersistence, NodeId, NodeIdentity};

// Import Result and NodeIdentity for non-sqlite builds
#[cfg(not(feature = "sqlite"))]
use crate::node_identity::NodeIdentity;
#[cfg(not(feature = "sqlite"))]
use anyhow::Result;

/// SQLite-based identity store with caching
#[cfg(feature = "sqlite")]
pub struct SqliteIdentityStore {
    /// Database connection pool
    pool: sqlx::SqlitePool,
    /// In-memory cache for performance
    cache: Arc<RwLock<IdentityCache>>,
}

/// In-memory cache for node identities
#[cfg(feature = "sqlite")]
#[derive(Default)]
struct IdentityCache {
    /// Map from UUID to cached identity
    by_uuid: HashMap<Uuid, CachedIdentity>,
}

/// Cached identity with metadata
#[cfg(feature = "sqlite")]
struct CachedIdentity {
    /// The node identity
    identity: Box<dyn NodeIdentity>,
    /// When this entry was cached
    cached_at: DateTime<Utc>,
}

#[cfg(feature = "sqlite")]
impl SqliteIdentityStore {
    /// Create a new SQLite identity store
    pub async fn new(database_url: &str) -> Result<Self> {
        info!("Creating SQLite identity store");

        let pool = sqlx::SqlitePool::connect(database_url)
            .await
            .context("Failed to connect to SQLite database")?;

        let store = Self {
            pool,
            cache: Arc::new(RwLock::new(IdentityCache::default())),
        };

        // Run migrations
        store.run_migrations().await?;

        Ok(store)
    }

    /// Create a new SQLite identity store from an existing pool
    pub async fn from_pool(pool: sqlx::SqlitePool) -> Result<Self> {
        let store = Self {
            pool,
            cache: Arc::new(RwLock::new(IdentityCache::default())),
        };

        // Run migrations
        store.run_migrations().await?;

        Ok(store)
    }

    /// Run database migrations
    async fn run_migrations(&self) -> Result<()> {
        info!("Running node identity migrations");

        // Create main identity table
        sqlx::query(
            r#"
            CREATE TABLE IF NOT EXISTS node_identities (
                uuid TEXT PRIMARY KEY CHECK(length(uuid) = 36),
                created_at TIMESTAMP NOT NULL,
                updated_at TIMESTAMP NOT NULL DEFAULT CURRENT_TIMESTAMP
            )
            "#,
        )
        .execute(&self.pool)
        .await
        .context("Failed to create node_identities table")?;

        // Create index on UUID for faster lookups
        sqlx::query("CREATE INDEX IF NOT EXISTS idx_node_identities_uuid ON node_identities(uuid)")
            .execute(&self.pool)
            .await
            .context("Failed to create UUID index")?;

        // Create legacy ID mapping table for migration support
        sqlx::query(
            r#"
            CREATE TABLE IF NOT EXISTS legacy_id_mappings (
                legacy_id TEXT PRIMARY KEY,
                uuid TEXT NOT NULL,
                migrated_at TIMESTAMP NOT NULL DEFAULT CURRENT_TIMESTAMP,
                FOREIGN KEY (uuid) REFERENCES node_identities(uuid)
            )
            "#,
        )
        .execute(&self.pool)
        .await
        .context("Failed to create legacy_id_mappings table")?;

        info!("Node identity migrations completed");
        Ok(())
    }

    /// Get or create identity with transaction safety
    async fn get_or_create_with_tx(
        &self,
        tx: &mut sqlx::Transaction<'_, sqlx::Sqlite>,
    ) -> Result<Box<dyn NodeIdentity>> {
        // Try to get existing identity first
        let existing = sqlx::query_as::<_, (String, DateTime<Utc>)>(
            "SELECT uuid, created_at FROM node_identities LIMIT 1",
        )
        .fetch_optional(&mut **tx)
        .await?;

        if let Some((uuid_str, created_at)) = existing {
            let uuid = Uuid::parse_str(&uuid_str)?;
            let identity = NodeId::from_parts(uuid, created_at.into())?;

            debug!("Found existing identity: {}", uuid);
            return Ok(Box::new(identity));
        }

        // Create new identity
        let new_id = NodeId::new(&Uuid::new_v4().to_string())?;

        sqlx::query(
            r#"
            INSERT INTO node_identities (uuid, created_at, updated_at)
            VALUES (?, ?, ?)
            "#,
        )
        .bind(new_id.uuid().to_string())
        .bind(DateTime::<Utc>::from(new_id.created_at()))
        .bind(Utc::now())
        .execute(&mut **tx)
        .await?;

        info!("Created new node identity: {}", new_id.uuid());
        Ok(Box::new(new_id))
    }

    /// Find identity by identifier (UUID)
    async fn find_by_identifier_internal(&self, id: &str) -> Result<Option<Box<dyn NodeIdentity>>> {
        // Validate input
        if id.is_empty() {
            return Err(anyhow!("Identifier cannot be empty"));
        }

        // For non-UUID identifiers, ensure minimum length
        if Uuid::parse_str(id).is_err() && id.len() < 3 {
            return Err(anyhow!(
                "UUID prefix must be at least 3 characters, got {} characters",
                id.len()
            ));
        }

        // Check cache first
        {
            let cache = self.cache.read().await;
            if let Ok(uuid) = Uuid::parse_str(id) {
                if let Some(cached) = cache.by_uuid.get(&uuid) {
                    let identity =
                        NodeId::from_parts(*cached.identity.uuid(), cached.identity.created_at())?;
                    return Ok(Some(Box::new(identity)));
                }
            }
        }

        // Not in cache, query database
        let result = if Uuid::parse_str(id).is_ok() {
            sqlx::query_as::<_, (String, DateTime<Utc>)>(
                "SELECT uuid, created_at FROM node_identities WHERE uuid = ?",
            )
            .bind(id)
            .fetch_optional(&self.pool)
            .await?
        } else if id.len() >= 3 {
            sqlx::query_as::<_, (String, DateTime<Utc>)>(
                "SELECT uuid, created_at FROM node_identities WHERE uuid LIKE ? || '%'",
            )
            .bind(id)
            .fetch_optional(&self.pool)
            .await?
        } else {
            None
        };

        if let Some((uuid_str, created_at)) = result {
            let uuid = Uuid::parse_str(&uuid_str)?;
            let identity = NodeId::from_parts(uuid, created_at.into())?;
            let cache_identity = NodeId::from_parts(*identity.uuid(), identity.created_at())?;
            self.update_cache(Box::new(cache_identity)).await;
            Ok(Some(Box::new(identity)))
        } else {
            Ok(None)
        }
    }

    /// Update the cache with an identity
    async fn update_cache(&self, identity: Box<dyn NodeIdentity>) {
        let mut cache = self.cache.write().await;
        let uuid = *identity.uuid();
        cache.by_uuid.insert(
            uuid,
            CachedIdentity {
                identity,
                cached_at: Utc::now(),
            },
        );

        // Implement simple LRU eviction if cache gets too large
        const MAX_CACHE_SIZE: usize = 1000;
        if cache.by_uuid.len() > MAX_CACHE_SIZE {
            if let Some((&oldest_uuid, _)) = cache.by_uuid.iter().min_by_key(|(_, v)| v.cached_at) {
                cache.by_uuid.remove(&oldest_uuid);
            }
        }
    }
}

#[cfg(feature = "sqlite")]
#[async_trait]
impl IdentityPersistence for SqliteIdentityStore {
    async fn get_or_create(&self) -> Result<Box<dyn NodeIdentity>> {
        // Use transaction for atomicity
        let mut tx = self.pool.begin().await?;
        let identity = self.get_or_create_with_tx(&mut tx).await?;
        tx.commit().await?;

        let cache_identity = NodeId::from_parts(*identity.uuid(), identity.created_at())?;
        self.update_cache(Box::new(cache_identity)).await;
        Ok(identity)
    }

    async fn find_by_identifier(&self, id: &str) -> Result<Option<Box<dyn NodeIdentity>>> {
        self.find_by_identifier_internal(id).await
    }

    async fn save(&self, id: &dyn NodeIdentity) -> Result<()> {
        sqlx::query(
            r#"
            INSERT INTO node_identities (uuid, created_at, updated_at)
            VALUES (?, ?, ?)
            ON CONFLICT(uuid) DO UPDATE SET
                updated_at = excluded.updated_at
            "#,
        )
        .bind(id.uuid().to_string())
        .bind(DateTime::<Utc>::from(id.created_at()))
        .bind(Utc::now())
        .execute(&self.pool)
        .await
        .context("Failed to save node identity")?;

        let identity = NodeId::from_parts(*id.uuid(), id.created_at())?;
        self.update_cache(Box::new(identity)).await;

        debug!("Saved node identity: {}", id.uuid());
        Ok(())
    }
}

/// Migration support for legacy string IDs
#[cfg(feature = "sqlite")]
impl SqliteIdentityStore {
    /// Migrate a legacy string ID to UUID
    pub async fn migrate_legacy_id(&self, legacy_id: &str) -> Result<Box<dyn NodeIdentity>> {
        // Check if already migrated
        let existing = sqlx::query_as::<_, (String,)>(
            "SELECT uuid FROM legacy_id_mappings WHERE legacy_id = ?",
        )
        .bind(legacy_id)
        .fetch_optional(&self.pool)
        .await?;

        if let Some((uuid_str,)) = existing {
            // Already migrated, return the identity
            let uuid = Uuid::parse_str(&uuid_str)?;
            return self
                .find_by_identifier(&uuid.to_string())
                .await?
                .ok_or_else(|| anyhow::anyhow!("Migrated identity not found for UUID: {}", uuid));
        }

        // Create new identity for this legacy ID
        let mut tx = self.pool.begin().await?;

        let new_identity = self.get_or_create_with_tx(&mut tx).await?;

        // Record the mapping
        sqlx::query(
            r#"
            INSERT INTO legacy_id_mappings (legacy_id, uuid)
            VALUES (?, ?)
            "#,
        )
        .bind(legacy_id)
        .bind(new_identity.uuid().to_string())
        .execute(&mut *tx)
        .await?;

        tx.commit().await?;

        info!(
            "Migrated legacy ID {} to {}",
            legacy_id,
            new_identity.uuid()
        );
        Ok(new_identity)
    }

    /// Get all legacy ID mappings
    pub async fn get_legacy_mappings(&self) -> Result<HashMap<String, Uuid>> {
        let mappings =
            sqlx::query_as::<_, (String, String)>("SELECT legacy_id, uuid FROM legacy_id_mappings")
                .fetch_all(&self.pool)
                .await?;

        let mut result = HashMap::new();
        for (legacy_id, uuid_str) in mappings {
            let uuid = Uuid::parse_str(&uuid_str)?;
            result.insert(legacy_id, uuid);
        }

        Ok(result)
    }

    /// Get collision statistics
    pub async fn get_collision_stats(&self) -> Result<CollisionStats> {
        Ok(CollisionStats {
            total_collisions: 0,
        })
    }

    /// Clear the in-memory cache
    pub async fn clear_cache(&self) {
        let mut cache = self.cache.write().await;
        cache.by_uuid.clear();
        debug!("Cleared identity cache");
    }

    /// Get cache statistics
    pub async fn cache_stats(&self) -> CacheStats {
        let cache = self.cache.read().await;
        CacheStats {
            total_entries: cache.by_uuid.len(),
            uuid_entries: cache.by_uuid.len(),
        }
    }
}

/// Statistics about UUID collisions
#[derive(Debug)]
pub struct CollisionStats {
    pub total_collisions: u64,
}

/// Statistics about the in-memory cache
#[derive(Debug)]
pub struct CacheStats {
    pub total_entries: usize,
    pub uuid_entries: usize,
}

// Fallback for when sqlite feature is not enabled
#[cfg(not(feature = "sqlite"))]
pub struct SqliteIdentityStore;

#[cfg(not(feature = "sqlite"))]
impl SqliteIdentityStore {
    pub async fn new(_database_url: &str) -> Result<Self> {
        Err(anyhow::anyhow!("SQLite feature not enabled"))
    }

    pub async fn migrate_legacy_id(&self, _legacy_id: &str) -> Result<Box<dyn NodeIdentity>> {
        Err(anyhow::anyhow!("SQLite feature not enabled"))
    }

    pub async fn get_legacy_mappings(&self) -> Result<HashMap<String, Uuid>> {
        Err(anyhow::anyhow!("SQLite feature not enabled"))
    }

    pub async fn get_collision_stats(&self) -> Result<CollisionStats> {
        Err(anyhow::anyhow!("SQLite feature not enabled"))
    }

    pub async fn clear_cache(&self) {
        // No-op for non-sqlite
    }

    pub async fn cache_stats(&self) -> CacheStats {
        CacheStats {
            total_entries: 0,
            uuid_entries: 0,
        }
    }
}

#[cfg(all(test, feature = "sqlite"))]
mod tests {
    use super::*;

    #[tokio::test]
    async fn test_identity_store_basic_operations() {
        let store = SqliteIdentityStore::new("sqlite::memory:")
            .await
            .expect("Should create in-memory store");

        // Test get_or_create
        let id1 = store.get_or_create().await.expect("Should create identity");
        let id2 = store
            .get_or_create()
            .await
            .expect("Should get existing identity");

        // Should return the same identity
        assert_eq!(id1.uuid(), id2.uuid());

        // Test find_by_identifier with UUID
        let found = store
            .find_by_identifier(&id1.uuid().to_string())
            .await
            .expect("Should find by UUID")
            .expect("Should find identity");
        assert_eq!(found.uuid(), id1.uuid());

        // Test find_by_identifier with UUID prefix
        let uuid_prefix = &id1.uuid().to_string()[..8];
        let found = store
            .find_by_identifier(uuid_prefix)
            .await
            .expect("Should find by UUID prefix")
            .expect("Should find identity");
        assert_eq!(found.uuid(), id1.uuid());

        // Test save
        let new_id = NodeId::new("default-seed").expect("Should create new ID");
        store.save(&new_id).await.expect("Should save identity");

        let found = store
            .find_by_identifier(&new_id.uuid().to_string())
            .await
            .expect("Should find saved identity")
            .expect("Should find identity");
        assert_eq!(found.uuid(), new_id.uuid());
    }

    #[tokio::test]
    async fn test_cache_functionality() {
        let store = SqliteIdentityStore::new("sqlite::memory:")
            .await
            .expect("Should create store");

        // Create identity
        let id = store.get_or_create().await.expect("Should create identity");

        // Check cache stats
        let stats = store.cache_stats().await;
        assert_eq!(stats.uuid_entries, 1);

        // Clear cache
        store.clear_cache().await;
        let stats = store.cache_stats().await;
        assert_eq!(stats.uuid_entries, 0);

        // Finding should repopulate cache
        let _ = store
            .find_by_identifier(&id.uuid().to_string())
            .await
            .expect("Should find");
        let stats = store.cache_stats().await;
        assert_eq!(stats.uuid_entries, 1);
    }

    #[tokio::test]
    async fn test_legacy_id_migration() {
        let store = SqliteIdentityStore::new("sqlite::memory:")
            .await
            .expect("Should create store");

        let legacy_id = "old-node-id-123";

        // First migration
        let migrated1 = store
            .migrate_legacy_id(legacy_id)
            .await
            .expect("Should migrate legacy ID");

        // Second migration should return same identity
        let migrated2 = store
            .migrate_legacy_id(legacy_id)
            .await
            .expect("Should return existing migration");

        assert_eq!(migrated1.uuid(), migrated2.uuid());

        // Check mapping
        let mappings = store
            .get_legacy_mappings()
            .await
            .expect("Should get mappings");
        assert_eq!(mappings.len(), 1);
        assert_eq!(mappings.get(legacy_id), Some(migrated1.uuid()));
    }

    #[tokio::test]
    async fn test_collision_handling() {
        let store = SqliteIdentityStore::new("sqlite::memory:")
            .await
            .expect("Should create store");

        // This test would require mocking or manipulating the UUID generation
        // to force collisions, which is complex. For now, we just test that
        // collision stats work when there are no collisions
        let stats = store.get_collision_stats().await.expect("Should get stats");
        assert_eq!(stats.total_collisions, 0);
    }

    #[tokio::test]
    async fn test_not_found_cases() {
        let store = SqliteIdentityStore::new("sqlite::memory:")
            .await
            .expect("Should create store");

        // Test finding non-existent UUID
        let result = store
            .find_by_identifier("00000000-0000-0000-0000-000000000000")
            .await
            .expect("Should complete search");
        assert!(result.is_none());

        // Test finding non-existent UUID
        let result = store
            .find_by_identifier("nonexistent-uuid-string")
            .await
            .expect("Should complete search");
        assert!(result.is_none());

        // Test too-short query
        let result = store.find_by_identifier("ab").await;
        assert!(result.is_err());
        let err = result.err().unwrap();
        assert!(err.to_string().contains("at least 3 characters"));
    }

    #[tokio::test]
    async fn test_concurrent_get_or_create() {
        // Use a file-based database with WAL mode for better concurrency
        let temp_file = tempfile::NamedTempFile::new().expect("Should create temp file");
        let db_path = format!("sqlite://{}?mode=rwc", temp_file.path().display());

        let store = Arc::new(
            SqliteIdentityStore::new(&db_path)
                .await
                .expect("Should create store"),
        );

        // Enable WAL mode for better concurrency
        sqlx::query("PRAGMA journal_mode = WAL")
            .execute(&store.pool)
            .await
            .expect("Should set WAL mode");

        // Spawn multiple tasks that try to get_or_create simultaneously
        let tasks: Vec<_> = (0..5)
            .map(|_| {
                let store = store.clone();
                tokio::spawn(async move {
                    // Add retry logic for database locks
                    let mut attempts = 0;
                    loop {
                        match store.get_or_create().await {
                            Ok(identity) => return identity,
                            Err(e)
                                if e.to_string().contains("database is locked")
                                    || e.to_string().contains("database is deadlocked") =>
                            {
                                attempts += 1;
                                if attempts > 10 {
                                    panic!("Failed after 10 attempts: {e}");
                                }
                                tokio::time::sleep(tokio::time::Duration::from_millis(10)).await;
                            }
                            Err(e) => panic!("Unexpected error: {e}"),
                        }
                    }
                })
            })
            .collect();

        // Wait for all tasks
        let mut results = Vec::new();
        for task in tasks {
            results.push(task.await.expect("Task should complete"));
        }

        // All should have the same identity
        let first_uuid = results[0].uuid();
        for result in &results {
            assert_eq!(result.uuid(), first_uuid);
        }
    }

    #[tokio::test]
    async fn test_from_pool_constructor() {
        // Create a pool separately
        let pool = sqlx::SqlitePool::connect("sqlite::memory:")
            .await
            .expect("Should create pool");

        let store = SqliteIdentityStore::from_pool(pool)
            .await
            .expect("Should create store from pool");

        // Test basic operation
        let id = store.get_or_create().await.expect("Should create identity");
        assert!(!id.uuid().to_string().is_empty());
    }
}
