//! Examples demonstrating the executor identity system usage
//!
//! This module provides example code showing how to use the identity
//! persistence layer in various scenarios.

#[cfg(feature = "sqlite")]
pub mod sqlite_examples {
    use crate::executor_identity::{
        IdentityDbFactory, IdentityMigrationManager, IdentityPersistence, MigrationConfig,
        SqliteIdentityStore,
    };
    use anyhow::Result;

    /// Example: Basic identity store usage
    pub async fn basic_usage_example() -> Result<()> {
        // Create an in-memory identity store
        let store = SqliteIdentityStore::new("sqlite::memory:").await?;

        // Get or create the executor's identity
        let identity = store.get_or_create().await?;
        println!("Executor UUID: {}", identity.uuid());
        println!("Executor HUID: {}", identity.huid());

        // Find by UUID
        let found = store
            .find_by_identifier(&identity.uuid().to_string())
            .await?;
        assert!(found.is_some());

        // Find by HUID prefix
        let huid_prefix = &identity.huid()[..5];
        let found = store.find_by_identifier(huid_prefix).await?;
        assert!(found.is_some());

        Ok(())
    }

    /// Example: Using the identity factory
    pub async fn factory_usage_example() -> Result<()> {
        let factory = IdentityDbFactory::new("sqlite::memory:".to_string()).with_auto_migrate(true);

        let store = factory.create_identity_store().await?;
        let identity = store.get_or_create().await?;

        println!("Created identity via factory: {}", identity.full_display());

        Ok(())
    }

    /// Example: Migrating legacy IDs
    pub async fn migration_example() -> Result<()> {
        // Set up database with legacy data
        let pool = sqlx::SqlitePool::connect("sqlite::memory:").await?;

        // Create a table with legacy executor IDs
        sqlx::query(
            r#"
            CREATE TABLE executor_health (
                executor_id TEXT PRIMARY KEY,
                is_healthy BOOLEAN DEFAULT FALSE
            )
            "#,
        )
        .execute(&pool)
        .await?;

        // Insert some legacy IDs
        sqlx::query("INSERT INTO executor_health (executor_id) VALUES (?)")
            .bind("legacy-executor-001")
            .execute(&pool)
            .await?;

        sqlx::query("INSERT INTO executor_health (executor_id) VALUES (?)")
            .bind("legacy-executor-002")
            .execute(&pool)
            .await?;

        // Create migration manager
        let migration_manager = IdentityMigrationManager::from_pool(pool.clone()).await?;

        // Configure migration
        let config = MigrationConfig {
            dry_run: false,
            batch_size: 10,
            continue_on_error: true,
            scan_targets: vec![crate::executor_identity::migration::ScanTarget {
                table: "executor_health".to_string(),
                id_column: "executor_id".to_string(),
                additional_columns: vec![],
            }],
        };

        // Run migration
        let report = migration_manager.migrate_all(&config).await?;
        println!("Migration report: {}", report.summary());

        // Verify migration
        let validation = migration_manager.validate_migration(&config).await?;
        assert!(validation.is_valid);

        Ok(())
    }

    /// Example: Working with cached identities
    pub async fn cache_example() -> Result<()> {
        let store = SqliteIdentityStore::new("sqlite::memory:").await?;

        // Create an identity
        let identity = store.get_or_create().await?;
        let uuid = identity.uuid().to_string();

        // First lookup - hits database
        let _ = store.find_by_identifier(&uuid).await?;

        // Check cache stats
        let stats = store.cache_stats().await;
        println!(
            "Cache contains {} UUIDs and {} HUIDs",
            stats.uuid_entries, stats.huid_entries
        );

        // Second lookup - should hit cache
        let cached = store.find_by_identifier(&uuid).await?;
        assert!(cached.is_some());

        // Clear cache
        store.clear_cache().await;
        let stats = store.cache_stats().await;
        assert_eq!(stats.uuid_entries, 0);

        Ok(())
    }

    /// Example: Integration with existing database operations
    pub async fn integration_example() -> Result<()> {
        use crate::executor_identity::integration::IdentityQueryBuilder;

        let pool = sqlx::SqlitePool::connect("sqlite::memory:").await?;
        let store = SqliteIdentityStore::from_pool(pool.clone()).await?;

        // Create executor table
        sqlx::query(
            r#"
            CREATE TABLE executors (
                id TEXT PRIMARY KEY,
                name TEXT NOT NULL,
                status TEXT DEFAULT 'active',
                created_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP
            )
            "#,
        )
        .execute(&pool)
        .await?;

        // Get identity
        let identity = store.get_or_create().await?;

        // Use query builder for type-safe queries
        let builder = IdentityQueryBuilder::new("executors", "id");

        // Insert with identity
        let insert_sql = builder
            .insert_with_identity(&*identity, &[("name", "My Executor"), ("status", "active")]);

        sqlx::query(&insert_sql)
            .bind(identity.uuid().to_string())
            .bind("My Executor")
            .bind("active")
            .execute(&pool)
            .await?;

        // Select by identity
        let select_sql = builder.select_by_identity(&["name", "status"]);
        let row: (String, String) = sqlx::query_as(&select_sql)
            .bind(identity.uuid().to_string())
            .fetch_one(&pool)
            .await?;

        println!("Executor: name={}, status={}", row.0, row.1);

        Ok(())
    }
}

/// Example SQL schema for executor identity tables
pub const EXAMPLE_SCHEMA: &str = r#"
-- Main executor identities table
CREATE TABLE IF NOT EXISTS executor_identities (
    uuid TEXT PRIMARY KEY CHECK(length(uuid) = 36),
    huid TEXT NOT NULL UNIQUE,
    created_at TIMESTAMP NOT NULL,
    updated_at TIMESTAMP NOT NULL DEFAULT CURRENT_TIMESTAMP,
    CHECK(huid GLOB '[a-z]*-[a-z]*-[0-9a-f][0-9a-f][0-9a-f][0-9a-f]')
);

-- Index for HUID prefix searches
CREATE INDEX IF NOT EXISTS idx_executor_identities_huid ON executor_identities(huid);

-- Legacy ID mapping table
CREATE TABLE IF NOT EXISTS legacy_id_mappings (
    legacy_id TEXT PRIMARY KEY,
    uuid TEXT NOT NULL,
    migrated_at TIMESTAMP NOT NULL DEFAULT CURRENT_TIMESTAMP,
    FOREIGN KEY (uuid) REFERENCES executor_identities(uuid)
);

-- HUID collision tracking (for monitoring)
CREATE TABLE IF NOT EXISTS huid_collision_log (
    id INTEGER PRIMARY KEY AUTOINCREMENT,
    attempted_huid TEXT NOT NULL,
    existing_uuid TEXT NOT NULL,
    new_uuid TEXT NOT NULL,
    occurred_at TIMESTAMP NOT NULL DEFAULT CURRENT_TIMESTAMP
);

-- Example: Updated executor_health table using UUIDs
CREATE TABLE IF NOT EXISTS executor_health_v2 (
    executor_id TEXT PRIMARY KEY CHECK(length(executor_id) = 36),
    is_healthy BOOLEAN NOT NULL DEFAULT FALSE,
    last_health_check TIMESTAMP,
    consecutive_failures INTEGER NOT NULL DEFAULT 0,
    last_error TEXT,
    updated_at TIMESTAMP NOT NULL DEFAULT CURRENT_TIMESTAMP,
    FOREIGN KEY (executor_id) REFERENCES executor_identities(uuid)
);

-- Example: SSH sessions with UUID references
CREATE TABLE IF NOT EXISTS ssh_sessions_v2 (
    session_id TEXT PRIMARY KEY,
    validator_hotkey TEXT NOT NULL,
    executor_id TEXT NOT NULL CHECK(length(executor_id) = 36),
    ssh_username TEXT NOT NULL,
    created_at TIMESTAMP NOT NULL DEFAULT CURRENT_TIMESTAMP,
    expires_at TIMESTAMP NOT NULL,
    status TEXT NOT NULL DEFAULT 'active',
    revocation_reason TEXT,
    revoked_at TIMESTAMP,
    FOREIGN KEY (executor_id) REFERENCES executor_identities(uuid)
);
"#;

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_example_schema_is_valid_sql() {
        // Just verify the schema string is defined
        assert!(!EXAMPLE_SCHEMA.is_empty());
        assert!(EXAMPLE_SCHEMA.contains("executor_identities"));
        assert!(EXAMPLE_SCHEMA.contains("legacy_id_mappings"));
        assert!(EXAMPLE_SCHEMA.contains("huid_collision_log"));
    }
}
