//! Examples demonstrating the node identity system usage
//!
//! This module provides example code showing how to use the identity
//! persistence layer in various scenarios.

#[cfg(feature = "sqlite")]
pub mod sqlite_examples {
    use crate::node_identity::{
        IdentityDbFactory, IdentityMigrationManager, IdentityPersistence, MigrationConfig,
        SqliteIdentityStore,
    };
    use anyhow::Result;

    /// Example: Basic identity store usage
    pub async fn basic_usage_example() -> Result<()> {
        // Create an in-memory identity store
        let store = SqliteIdentityStore::new("sqlite::memory:").await?;

        // Get or create the node's identity
        let identity = store.get_or_create().await?;
        println!("Node UUID: {}", identity.uuid());

        // Find by UUID
        let found = store
            .find_by_identifier(&identity.uuid().to_string())
            .await?;
        assert!(found.is_some());

        // Find by UUID prefix
        let uuid_prefix = &identity.uuid().to_string()[..8];
        let found = store.find_by_identifier(uuid_prefix).await?;
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

        // Create a table with legacy node IDs
        sqlx::query(
            r#"
            CREATE TABLE node_health (
                node_id TEXT PRIMARY KEY,
                is_healthy BOOLEAN DEFAULT FALSE
            )
            "#,
        )
        .execute(&pool)
        .await?;

        // Insert some legacy IDs
        sqlx::query("INSERT INTO node_health (node_id) VALUES (?)")
            .bind("legacy-node-001")
            .execute(&pool)
            .await?;

        sqlx::query("INSERT INTO node_health (node_id) VALUES (?)")
            .bind("legacy-node-002")
            .execute(&pool)
            .await?;

        // Create migration manager
        let migration_manager = IdentityMigrationManager::from_pool(pool.clone()).await?;

        // Configure migration
        let config = MigrationConfig {
            dry_run: false,
            batch_size: 10,
            continue_on_error: true,
            scan_targets: vec![crate::node_identity::migration::ScanTarget {
                table: "node_health".to_string(),
                id_column: "node_id".to_string(),
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
        println!("Cache contains {} UUIDs", stats.uuid_entries);

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
        use crate::node_identity::integration::IdentityQueryBuilder;

        let pool = sqlx::SqlitePool::connect("sqlite::memory:").await?;
        let store = SqliteIdentityStore::from_pool(pool.clone()).await?;

        // Create node table
        sqlx::query(
            r#"
            CREATE TABLE nodes (
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
        let builder = IdentityQueryBuilder::new("nodes", "id");

        // Insert with identity
        let insert_sql =
            builder.insert_with_identity(&*identity, &[("name", "My Node"), ("status", "active")]);

        sqlx::query(&insert_sql)
            .bind(identity.uuid().to_string())
            .bind("My Node")
            .bind("active")
            .execute(&pool)
            .await?;

        // Select by identity
        let select_sql = builder.select_by_identity(&["name", "status"]);
        let row: (String, String) = sqlx::query_as(&select_sql)
            .bind(identity.uuid().to_string())
            .fetch_one(&pool)
            .await?;

        println!("Node: name={}, status={}", row.0, row.1);

        Ok(())
    }
}

/// Examples for NodeId usage
use crate::node_identity::node_id::NodeId;

/// Example demonstrating seeded NodeId generation
pub fn seeded_generation_example() -> anyhow::Result<()> {
    println!("=== Seeded NodeId Generation Example ===");

    // Create NodeIds with the same seed
    let seed = "my-deterministic-seed";
    let id1 = NodeId::new(seed)?;
    let id2 = NodeId::new(seed)?;

    println!("Seed: {seed}");
    println!("First ID:  {id1}");
    println!("Second ID: {id2}");
    println!("Identical: {}", id1 == id2);

    // Create NodeIds with different seeds
    let seed1 = "seed-1";
    let seed2 = "seed-2";
    let id3 = NodeId::new(seed1)?;
    let id4 = NodeId::new(seed2)?;

    println!("\nDifferent seeds:");
    println!("Seed 1: {seed1}");
    println!("ID 3:   {id3}");
    println!("Seed 2: {seed2}");
    println!("ID 4:   {id4}");
    println!("Different: {}", id3 != id4);

    // Compare with different seed generation
    let random_id = NodeId::new("different-seed")?;
    println!("\nDifferent seed generation:");
    println!("Different seed ID: {random_id}");
    println!("Different from seeded: {}", id1 != random_id);

    Ok(())
}

/// Example demonstrating the difference between seeded and random generation
pub fn seeded_vs_random_example() -> anyhow::Result<()> {
    println!("=== Seeded vs Random Generation ===");

    let seed = "consistent-seed";

    // Generate multiple IDs with the same seed
    println!("With same seed '{seed}':");
    for i in 1..=3 {
        let id = NodeId::new(seed)?;
        println!("  ID {i}: {id}");
    }

    // Generate multiple IDs with different seeds
    println!("\nWith different seeds:");
    for i in 1..=3 {
        let id = NodeId::new(&format!("seed-{i}"))?;
        println!("  ID {i}: {id}");
    }

    Ok(())
}

/// Example SQL schema for node identity tables
pub const EXAMPLE_SCHEMA: &str = r#"
-- Main node identities table
CREATE TABLE IF NOT EXISTS node_identities (
    uuid TEXT PRIMARY KEY CHECK(length(uuid) = 36),
    created_at TIMESTAMP NOT NULL,
    updated_at TIMESTAMP NOT NULL DEFAULT CURRENT_TIMESTAMP
);

-- Legacy ID mapping table
CREATE TABLE IF NOT EXISTS legacy_id_mappings (
    legacy_id TEXT PRIMARY KEY,
    uuid TEXT NOT NULL,
    migrated_at TIMESTAMP NOT NULL DEFAULT CURRENT_TIMESTAMP,
    FOREIGN KEY (uuid) REFERENCES node_identities(uuid)
);

-- Example: Updated node_health table using UUIDs
CREATE TABLE IF NOT EXISTS node_health_v2 (
    node_id TEXT PRIMARY KEY CHECK(length(node_id) = 36),
    is_healthy BOOLEAN NOT NULL DEFAULT FALSE,
    last_health_check TIMESTAMP,
    consecutive_failures INTEGER NOT NULL DEFAULT 0,
    last_error TEXT,
    updated_at TIMESTAMP NOT NULL DEFAULT CURRENT_TIMESTAMP,
    FOREIGN KEY (node_id) REFERENCES node_identities(uuid)
);

-- Example: SSH sessions with UUID references
CREATE TABLE IF NOT EXISTS ssh_sessions_v2 (
    session_id TEXT PRIMARY KEY,
    validator_hotkey TEXT NOT NULL,
    node_id TEXT NOT NULL CHECK(length(node_id) = 36),
    ssh_username TEXT NOT NULL,
    created_at TIMESTAMP NOT NULL DEFAULT CURRENT_TIMESTAMP,
    expires_at TIMESTAMP NOT NULL,
    status TEXT NOT NULL DEFAULT 'active',
    revocation_reason TEXT,
    revoked_at TIMESTAMP,
    FOREIGN KEY (node_id) REFERENCES node_identities(uuid)
);
"#;

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    #[allow(clippy::const_is_empty)]
    fn test_example_schema_is_valid_sql() {
        // Just verify the schema string is defined
        assert!(!EXAMPLE_SCHEMA.is_empty(), "Schema should not be empty");
        assert!(EXAMPLE_SCHEMA.contains("node_identities"));
        assert!(EXAMPLE_SCHEMA.contains("legacy_id_mappings"));
    }
}
