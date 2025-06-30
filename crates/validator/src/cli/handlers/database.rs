use super::HandlerUtils;
use crate::cli::commands::DatabaseAction;
use crate::config::ValidatorConfig;
use anyhow::Result;
use chrono::Utc;
use sqlx::{Pool, Row, Sqlite};
use std::path::Path;

pub async fn handle_database(action: DatabaseAction) -> Result<()> {
    match action {
        DatabaseAction::Migrate => run_migrations().await,
        DatabaseAction::Reset { confirm } => reset_database(confirm).await,
        DatabaseAction::Status => show_database_status().await,
        DatabaseAction::Cleanup { days } => cleanup_old_records(days).await,
    }
}

async fn run_migrations() -> Result<()> {
    HandlerUtils::print_info("Running database migrations...");

    // Load configuration
    let config = ValidatorConfig::load()?;

    // Extract database path from SQLite URL
    let db_path = if config.database.url.starts_with("sqlite:") {
        &config.database.url[7..] // Remove "sqlite:" prefix
    } else {
        return Err(anyhow::anyhow!("Only SQLite databases are supported"));
    };

    // Ensure data directory exists
    if let Some(parent) = Path::new(db_path).parent() {
        std::fs::create_dir_all(parent)?;
        HandlerUtils::print_info(&format!("Created data directory: {}", parent.display()));
    }

    // Create connection pool
    let pool = sqlx::SqlitePool::connect(&config.database.url).await?;
    HandlerUtils::print_success("Database connection established");

    // Check if migrations table exists
    let migrations_exist = sqlx::query(
        "SELECT name FROM sqlite_master WHERE type='table' AND name='_sqlx_migrations'",
    )
    .fetch_optional(&pool)
    .await?
    .is_some();

    if !migrations_exist {
        HandlerUtils::print_info("Creating migrations tracking table...");
    }

    // Run built-in migrations (same as SimplePersistence)
    create_schema(&pool).await?;

    // Record migration status
    let now = Utc::now().to_rfc3339();
    sqlx::query(
        "CREATE TABLE IF NOT EXISTS _migration_log (
            id INTEGER PRIMARY KEY AUTOINCREMENT,
            migration_name TEXT NOT NULL,
            applied_at TEXT NOT NULL,
            checksum TEXT NOT NULL
        )",
    )
    .execute(&pool)
    .await?;

    // Check if our schema migration was already applied
    let existing_migration = sqlx::query(
        "SELECT applied_at FROM _migration_log WHERE migration_name = 'initial_schema_v1'",
    )
    .fetch_optional(&pool)
    .await?;

    if existing_migration.is_none() {
        sqlx::query(
            "INSERT INTO _migration_log (migration_name, applied_at, checksum) VALUES (?, ?, ?)",
        )
        .bind("initial_schema_v1")
        .bind(&now)
        .bind("sha256:a1b2c3d4e5f6") // Schema checksum placeholder
        .execute(&pool)
        .await?;

        HandlerUtils::print_success("Applied initial schema migration");
    } else {
        HandlerUtils::print_info("Database schema is already up to date");
    }

    pool.close().await;
    HandlerUtils::print_success("Database migrations completed successfully");

    Ok(())
}

async fn reset_database(confirm: bool) -> Result<()> {
    if !confirm {
        HandlerUtils::print_error("Database reset requires --confirm flag");
        HandlerUtils::print_warning("This will permanently delete all validator data");
        HandlerUtils::print_warning("Use: validator database reset --confirm");
        return Ok(());
    }

    HandlerUtils::print_warning("🚨 DESTRUCTIVE OPERATION: Resetting database...");

    // Load configuration
    let config = ValidatorConfig::load()?;

    // Create connection pool
    let pool = sqlx::SqlitePool::connect(&config.database.url).await?;
    HandlerUtils::print_info("Connected to database");

    // Get list of all tables
    let tables: Vec<String> = sqlx::query(
        "SELECT name FROM sqlite_master WHERE type='table' AND name NOT LIKE 'sqlite_%'",
    )
    .fetch_all(&pool)
    .await?
    .into_iter()
    .map(|row| row.get::<String, _>("name"))
    .collect();

    HandlerUtils::print_info(&format!("Found {} tables to drop", tables.len()));

    // Drop all tables
    for table in &tables {
        HandlerUtils::print_info(&format!("Dropping table: {table}"));
        sqlx::query(&format!("DROP TABLE IF EXISTS {table}"))
            .execute(&pool)
            .await?;
    }

    HandlerUtils::print_success("All tables dropped successfully");

    // Close connection before recreating schema
    pool.close().await;

    // Re-run migrations to recreate schema
    HandlerUtils::print_info("Recreating database schema...");
    run_migrations().await?;

    HandlerUtils::print_success("Database reset completed successfully");
    HandlerUtils::print_info("Database has been reset to initial state");

    Ok(())
}

async fn show_database_status() -> Result<()> {
    HandlerUtils::print_info("=== Database Status Report ===");

    // Load configuration
    let config = ValidatorConfig::load()?;

    // Display configuration info
    println!("Database URL: {}", config.database.url);
    println!("Max Connections: {}", config.database.max_connections);
    println!("Connection Timeout: {:?}", config.database.connect_timeout);

    // Test database connectivity
    println!("\n🔗 Connectivity Test:");
    let pool_result = sqlx::SqlitePool::connect(&config.database.url).await;

    let pool = match pool_result {
        Ok(pool) => {
            HandlerUtils::print_success("Database connection successful");
            pool
        }
        Err(e) => {
            HandlerUtils::print_error(&format!("Database connection failed: {e}"));
            return Err(e.into());
        }
    };

    // Check if database file exists and get size
    let db_path = if config.database.url.starts_with("sqlite:") {
        &config.database.url[7..]
    } else {
        "unknown"
    };

    if let Ok(metadata) = std::fs::metadata(db_path) {
        println!(
            "Database file size: {} bytes ({:.2} MB)",
            metadata.len(),
            metadata.len() as f64 / 1024.0 / 1024.0
        );
    } else {
        println!("Database file: Not found (will be created on first use)");
    }

    // Query table statistics
    println!("\n📊 Table Statistics:");

    let tables = vec![
        "miners",
        "miner_executors",
        "verification_requests",
        "verification_logs",
        "rentals",
    ];

    let mut total_records = 0;

    for table in &tables {
        let count_result = sqlx::query(&format!("SELECT COUNT(*) as count FROM {table} WHERE 1=1"))
            .fetch_optional(&pool)
            .await;

        match count_result {
            Ok(Some(row)) => {
                let count: i64 = row.get("count");
                total_records += count;
                println!("  {table}: {count} records");
            }
            Ok(None) => {
                println!("  {table}: Table exists but no data");
            }
            Err(_) => {
                println!("  {table}: Table does not exist");
            }
        }
    }

    println!("\nTotal Records: {total_records}");

    // Check migration status
    println!("\n🔄 Migration Status:");
    let migration_check = sqlx::query(
        "SELECT migration_name, applied_at FROM _migration_log ORDER BY applied_at DESC LIMIT 5",
    )
    .fetch_all(&pool)
    .await;

    match migration_check {
        Ok(migrations) => {
            if migrations.is_empty() {
                println!("  No migrations recorded (database may need initialization)");
            } else {
                println!("  Recent migrations:");
                for migration in migrations {
                    let name: String = migration.get("migration_name");
                    let applied_at: String = migration.get("applied_at");
                    println!("    - {name} (applied: {applied_at})");
                }
            }
        }
        Err(_) => {
            println!("  Migration log table not found (run migrations first)");
        }
    }

    // Database integrity check
    println!("\n🔍 Integrity Check:");
    let integrity_result = sqlx::query("PRAGMA integrity_check").fetch_one(&pool).await;

    match integrity_result {
        Ok(row) => {
            let result: String = row.get(0);
            if result == "ok" {
                HandlerUtils::print_success("Database integrity check passed");
            } else {
                HandlerUtils::print_warning(&format!("Integrity check result: {result}"));
            }
        }
        Err(e) => {
            HandlerUtils::print_error(&format!("Integrity check failed: {e}"));
        }
    }

    pool.close().await;

    println!("\n=== Status Report Complete ===");

    Ok(())
}

async fn cleanup_old_records(days: u32) -> Result<()> {
    if days < 7 {
        HandlerUtils::print_warning("Cleanup period less than 7 days - proceed with caution");
        HandlerUtils::print_warning("This will delete recent data that may still be useful");
    }

    if days == 0 {
        HandlerUtils::print_error("Cleanup period cannot be 0 days");
        return Err(anyhow::anyhow!("Invalid cleanup period"));
    }

    HandlerUtils::print_info(&format!("🧹 Cleaning up records older than {days} days..."));

    // Load configuration and connect
    let config = ValidatorConfig::load()?;
    let pool = sqlx::SqlitePool::connect(&config.database.url).await?;

    // Calculate cutoff date
    let cutoff_date = Utc::now() - chrono::Duration::days(days as i64);
    let cutoff_str = cutoff_date.to_rfc3339();

    HandlerUtils::print_info(&format!(
        "Cutoff date: {} ({})",
        cutoff_str,
        cutoff_date.format("%Y-%m-%d %H:%M:%S UTC")
    ));

    let mut total_deleted = 0;

    // Start transaction for atomic cleanup
    let mut tx = pool.begin().await?;

    // Cleanup verification_logs (main cleanup target)
    let verification_logs_deleted =
        sqlx::query("DELETE FROM verification_logs WHERE created_at < ?")
            .bind(&cutoff_str)
            .execute(&mut *tx)
            .await?
            .rows_affected();

    if verification_logs_deleted > 0 {
        println!("  🗑️  Deleted {verification_logs_deleted} verification log records");
        total_deleted += verification_logs_deleted;
    }

    // Cleanup old verification_requests
    let verification_requests_deleted = sqlx::query(
        "DELETE FROM verification_requests WHERE created_at < ? AND status IN ('completed', 'failed')"
    )
    .bind(&cutoff_str)
    .execute(&mut *tx)
    .await?
    .rows_affected();

    if verification_requests_deleted > 0 {
        println!(
            "  🗑️  Deleted {verification_requests_deleted} completed verification request records"
        );
        total_deleted += verification_requests_deleted;
    }

    // Cleanup old terminated rentals
    let rentals_deleted = sqlx::query(
        "DELETE FROM rentals WHERE created_at < ? AND status IN ('Terminated', 'Failed')",
    )
    .bind(&cutoff_str)
    .execute(&mut *tx)
    .await?
    .rows_affected();

    if rentals_deleted > 0 {
        println!("  🗑️  Deleted {rentals_deleted} terminated rental records");
        total_deleted += rentals_deleted;
    }

    // Clean up miners that haven't been seen for the cutoff period
    let old_miners_deleted = sqlx::query("DELETE FROM miners WHERE last_seen < ?")
        .bind(&cutoff_str)
        .execute(&mut *tx)
        .await?
        .rows_affected();

    if old_miners_deleted > 0 {
        println!("  🗑️  Deleted {old_miners_deleted} inactive miner records");
        total_deleted += old_miners_deleted;
    }

    // Note: miner_executors will be automatically deleted due to CASCADE foreign key

    // Commit transaction
    tx.commit().await?;

    if total_deleted > 0 {
        HandlerUtils::print_success(&format!(
            "Cleanup completed: {total_deleted} total records deleted"
        ));

        // Run VACUUM to reclaim space
        HandlerUtils::print_info("Running VACUUM to reclaim disk space...");
        sqlx::query("VACUUM").execute(&pool).await?;
        HandlerUtils::print_success("Database space reclaimed");

        // Update statistics
        HandlerUtils::print_info("Updating database statistics...");
        sqlx::query("ANALYZE").execute(&pool).await?;
        HandlerUtils::print_success("Database statistics updated");
    } else {
        HandlerUtils::print_info("No records found for cleanup");
    }

    pool.close().await;

    // Log cleanup operation
    let cleanup_summary = format!(
        "Database cleanup completed: {total_deleted} records deleted (older than {days} days)"
    );

    HandlerUtils::print_success(&cleanup_summary);

    Ok(())
}

/// Create the database schema using the same logic as SimplePersistence
async fn create_schema(pool: &Pool<Sqlite>) -> Result<()> {
    // Create main tables
    sqlx::query(
        r#"
        CREATE TABLE IF NOT EXISTS miners (
            id TEXT PRIMARY KEY,
            hotkey TEXT NOT NULL UNIQUE,
            endpoint TEXT NOT NULL,
            verification_score REAL DEFAULT 0.0,
            uptime_percentage REAL DEFAULT 0.0,
            last_seen TEXT NOT NULL,
            registered_at TEXT NOT NULL,
            updated_at TEXT NOT NULL,
            executor_info TEXT NOT NULL DEFAULT '{}'
        );

        CREATE TABLE IF NOT EXISTS miner_executors (
            id TEXT PRIMARY KEY,
            miner_id TEXT NOT NULL,
            executor_id TEXT NOT NULL,
            grpc_address TEXT NOT NULL,
            gpu_count INTEGER NOT NULL,
            gpu_specs TEXT NOT NULL,
            cpu_specs TEXT NOT NULL,
            location TEXT,
            status TEXT DEFAULT 'unknown',
            last_health_check TEXT,
            created_at TEXT NOT NULL,
            updated_at TEXT NOT NULL,
            FOREIGN KEY (miner_id) REFERENCES miners (id) ON DELETE CASCADE
        );

        CREATE TABLE IF NOT EXISTS verification_requests (
            id TEXT PRIMARY KEY,
            miner_id TEXT NOT NULL,
            verification_type TEXT NOT NULL,
            executor_id TEXT,
            status TEXT DEFAULT 'scheduled',
            scheduled_at TEXT NOT NULL,
            completed_at TEXT,
            created_at TEXT NOT NULL,
            FOREIGN KEY (miner_id) REFERENCES miners (id) ON DELETE CASCADE
        );
        "#,
    )
    .execute(pool)
    .await?;

    // Create additional tables
    sqlx::query(
        r#"
        CREATE TABLE IF NOT EXISTS verification_logs (
            id TEXT PRIMARY KEY,
            executor_id TEXT NOT NULL,
            validator_hotkey TEXT NOT NULL,
            verification_type TEXT NOT NULL,
            timestamp TEXT NOT NULL,
            score REAL NOT NULL,
            success INTEGER NOT NULL,
            details TEXT NOT NULL,
            duration_ms INTEGER NOT NULL,
            error_message TEXT,
            created_at TEXT NOT NULL,
            updated_at TEXT NOT NULL
        );

        CREATE TABLE IF NOT EXISTS rentals (
            id TEXT PRIMARY KEY,
            executor_id TEXT NOT NULL,
            customer_public_key TEXT NOT NULL,
            docker_image TEXT NOT NULL,
            env_vars TEXT,
            gpu_requirements TEXT NOT NULL,
            ssh_access_info TEXT NOT NULL,
            max_duration_hours INTEGER NOT NULL,
            cost_per_hour REAL NOT NULL,
            status TEXT NOT NULL,
            created_at TEXT NOT NULL,
            updated_at TEXT NOT NULL,
            started_at TEXT,
            terminated_at TEXT,
            termination_reason TEXT,
            total_cost REAL
        );
        "#,
    )
    .execute(pool)
    .await?;

    // Create indexes for performance
    sqlx::query("CREATE INDEX IF NOT EXISTS idx_verification_logs_created_at ON verification_logs(created_at)")
        .execute(pool)
        .await?;

    sqlx::query("CREATE INDEX IF NOT EXISTS idx_verification_logs_executor_id ON verification_logs(executor_id)")
        .execute(pool)
        .await?;

    sqlx::query("CREATE INDEX IF NOT EXISTS idx_miners_hotkey ON miners(hotkey)")
        .execute(pool)
        .await?;

    sqlx::query("CREATE INDEX IF NOT EXISTS idx_miners_last_seen ON miners(last_seen)")
        .execute(pool)
        .await?;

    sqlx::query("CREATE INDEX IF NOT EXISTS idx_rentals_status ON rentals(status)")
        .execute(pool)
        .await?;

    Ok(())
}
