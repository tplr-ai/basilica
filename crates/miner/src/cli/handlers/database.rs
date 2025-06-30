//! # Database Management Commands
//!
//! Handles database operations including backup, restore, stats,
//! and maintenance operations for the miner SQLite database.

use anyhow::{anyhow, Result};
use chrono::{DateTime, Utc};
use std::fs;
use std::path::{Path, PathBuf};
use tracing::{error, info, warn};

use crate::config::MinerConfig;
use crate::persistence::RegistrationDb;

/// Database operation types
#[derive(Debug, Clone)]
pub enum DatabaseOperation {
    Backup { path: String },
    Restore { path: String },
    Stats,
    Cleanup { days: Option<u32> },
    Vacuum,
    Integrity,
}

/// Database statistics
#[derive(Debug)]
pub struct DatabaseStats {
    pub file_size: u64,
    pub page_count: u64,
    pub page_size: u64,
    pub vacuum_count: u64,
    pub table_stats: Vec<TableStats>,
    pub last_backup: Option<DateTime<Utc>>,
}

/// Table statistics
#[derive(Debug)]
pub struct TableStats {
    pub table_name: String,
    pub row_count: u64,
    pub size_bytes: u64,
}

/// Handle database management commands
pub async fn handle_database_command(
    operation: DatabaseOperation,
    config: &MinerConfig,
) -> Result<()> {
    match operation {
        DatabaseOperation::Backup { path } => backup_database(config, &path).await,
        DatabaseOperation::Restore { path } => restore_database(config, &path).await,
        DatabaseOperation::Stats => show_database_stats(config).await,
        DatabaseOperation::Cleanup { days } => cleanup_database(config, days.unwrap_or(30)).await,
        DatabaseOperation::Vacuum => vacuum_database(config).await,
        DatabaseOperation::Integrity => check_database_integrity(config).await,
    }
}

/// Backup the miner database
async fn backup_database(config: &MinerConfig, backup_path: &str) -> Result<()> {
    info!("Starting database backup to: {}", backup_path);
    println!("💾 Backing up database to: {backup_path}");

    // Parse database URL to get the file path
    let db_path = extract_db_path_from_url(&config.database.url)?;

    // Check if source database exists
    if !Path::new(&db_path).exists() {
        return Err(anyhow!("Database file not found: {}", db_path));
    }

    // Create backup directory if it doesn't exist
    if let Some(parent) = Path::new(backup_path).parent() {
        fs::create_dir_all(parent)
            .map_err(|e| anyhow!("Failed to create backup directory: {}", e))?;
    }

    // Generate timestamped backup filename if path is a directory
    let final_backup_path = if Path::new(backup_path).is_dir() {
        let timestamp = Utc::now().format("%Y%m%d_%H%M%S");
        PathBuf::from(backup_path).join(format!("miner_backup_{timestamp}.db"))
    } else {
        PathBuf::from(backup_path)
    };

    // Create database connection for hot backup
    let db = RegistrationDb::new(&config.database).await?;

    // Perform hot backup using SQLite VACUUM INTO command
    info!("Performing hot backup using VACUUM INTO");
    match perform_hot_backup(&db, &final_backup_path).await {
        Ok(_) => {
            println!("✅ Database backup completed successfully");
            println!("   Backup location: {}", final_backup_path.display());
            info!("Database backup completed: {}", final_backup_path.display());
        }
        Err(e) => {
            warn!("Hot backup failed, falling back to file copy: {}", e);

            // Fallback to file copy
            fs::copy(&db_path, &final_backup_path)
                .map_err(|e| anyhow!("Failed to copy database file: {}", e))?;

            println!("⚠️  Fallback backup completed (file copy)");
            println!("   Backup location: {}", final_backup_path.display());
            warn!("Database backup completed using file copy fallback");
        }
    }

    // Verify backup integrity
    if verify_backup_integrity(&final_backup_path).await? {
        println!("✅ Backup integrity verified");
    } else {
        error!("Backup integrity check failed");
        return Err(anyhow!("Backup integrity check failed"));
    }

    Ok(())
}

/// Restore database from backup
async fn restore_database(config: &MinerConfig, backup_path: &str) -> Result<()> {
    info!("Starting database restore from: {}", backup_path);
    println!("📥 Restoring database from: {backup_path}");

    // Check if backup file exists
    if !Path::new(backup_path).exists() {
        return Err(anyhow!("Backup file not found: {}", backup_path));
    }

    // Verify backup integrity before restore
    if !verify_backup_integrity(Path::new(backup_path)).await? {
        return Err(anyhow!("Backup file is corrupted or invalid"));
    }

    // Parse database URL to get the target path
    let db_path = extract_db_path_from_url(&config.database.url)?;

    // Create backup of current database if it exists
    if Path::new(&db_path).exists() {
        let backup_current = format!(
            "{}.pre_restore_{}",
            db_path,
            Utc::now().format("%Y%m%d_%H%M%S")
        );
        fs::copy(&db_path, &backup_current)
            .map_err(|e| anyhow!("Failed to backup current database: {}", e))?;
        println!("📦 Current database backed up to: {backup_current}");
    }

    // Restore the database
    fs::copy(backup_path, &db_path).map_err(|e| anyhow!("Failed to restore database: {}", e))?;

    println!("✅ Database restored successfully");
    info!("Database restored from: {}", backup_path);

    // Test the restored database
    match RegistrationDb::new(&config.database).await {
        Ok(_) => {
            println!("✅ Restored database connection verified");
        }
        Err(e) => {
            error!("Restored database connection failed: {}", e);
            return Err(anyhow!("Restored database is not accessible: {}", e));
        }
    }

    Ok(())
}

/// Show database statistics
async fn show_database_stats(config: &MinerConfig) -> Result<()> {
    println!("📊 Gathering database statistics...");

    let db = RegistrationDb::new(&config.database).await?;
    let stats = collect_database_stats(&db, config).await?;

    println!("\n=== Database Statistics ===");
    println!(
        "File Size: {:.2} MB",
        stats.file_size as f64 / 1024.0 / 1024.0
    );
    println!("Page Count: {}", stats.page_count);
    println!("Page Size: {} bytes", stats.page_size);
    println!(
        "Database Size: {:.2} MB",
        (stats.page_count * stats.page_size) as f64 / 1024.0 / 1024.0
    );

    if let Some(last_backup) = stats.last_backup {
        println!(
            "Last Backup: {}",
            last_backup.format("%Y-%m-%d %H:%M:%S UTC")
        );
    } else {
        println!("Last Backup: Never");
    }

    println!("\n=== Table Statistics ===");
    println!("{:<25} {:>10} {:>15}", "Table", "Rows", "Size (KB)");
    println!("{}", "-".repeat(52));

    for table in &stats.table_stats {
        println!(
            "{:<25} {:>10} {:>15.1}",
            table.table_name,
            table.row_count,
            table.size_bytes as f64 / 1024.0
        );
    }

    // Show health status
    let health_records = db.get_all_executor_health().await.unwrap_or_default();
    let interaction_count = db
        .get_recent_validator_interactions(100)
        .await
        .unwrap_or_default()
        .len();

    println!("\n=== Content Summary ===");
    println!("Executor Health Records: {}", health_records.len());
    println!("Recent Validator Interactions: {interaction_count}");

    let healthy_executors = health_records.iter().filter(|h| h.is_healthy).count();
    if !health_records.is_empty() {
        println!(
            "Healthy Executors: {}/{} ({:.1}%)",
            healthy_executors,
            health_records.len(),
            (healthy_executors as f64 / health_records.len() as f64) * 100.0
        );
    }

    Ok(())
}

/// Clean up old database records
async fn cleanup_database(config: &MinerConfig, days: u32) -> Result<()> {
    info!("Starting database cleanup (older than {} days)", days);
    println!("🧹 Cleaning up database records older than {days} days...");

    let db = RegistrationDb::new(&config.database).await?;

    // Calculate cutoff date
    let cutoff_date = Utc::now() - chrono::Duration::days(days as i64);
    println!(
        "   Cutoff date: {}",
        cutoff_date.format("%Y-%m-%d %H:%M:%S UTC")
    );

    // Clean up old validator interactions
    let interactions_cleaned = clean_old_validator_interactions(&db, cutoff_date).await?;
    if interactions_cleaned > 0 {
        println!("✅ Cleaned {interactions_cleaned} old validator interactions");
    }

    // Clean up old SSH grants
    let grants_cleaned = clean_old_ssh_grants(&db, cutoff_date).await?;
    if grants_cleaned > 0 {
        println!("✅ Cleaned {grants_cleaned} old SSH grants");
    }

    // Clean up stale executor health records
    let health_cleaned = clean_stale_executor_health(&db, cutoff_date).await?;
    if health_cleaned > 0 {
        println!("✅ Cleaned {health_cleaned} stale executor health records");
    }

    if interactions_cleaned == 0 && grants_cleaned == 0 && health_cleaned == 0 {
        println!("ℹ️  No old records found to clean up");
    }

    println!("✅ Database cleanup completed");
    Ok(())
}

/// Vacuum the database to reclaim space
async fn vacuum_database(config: &MinerConfig) -> Result<()> {
    info!("Starting database vacuum operation");
    println!("🗜️  Vacuuming database to reclaim space...");

    let db = RegistrationDb::new(&config.database).await?;

    // Get size before vacuum
    let stats_before = collect_database_stats(&db, config).await?;
    let size_before = stats_before.file_size;

    // Perform vacuum
    db.vacuum()
        .await
        .map_err(|e| anyhow!("Failed to vacuum database: {}", e))?;

    // Get size after vacuum
    let stats_after = collect_database_stats(&db, config).await?;
    let size_after = stats_after.file_size;

    let space_saved = size_before.saturating_sub(size_after);

    println!("✅ Database vacuum completed");
    println!(
        "   Size before: {:.2} MB",
        size_before as f64 / 1024.0 / 1024.0
    );
    println!(
        "   Size after:  {:.2} MB",
        size_after as f64 / 1024.0 / 1024.0
    );

    if space_saved > 0 {
        println!(
            "   Space saved: {:.2} MB",
            space_saved as f64 / 1024.0 / 1024.0
        );
    } else {
        println!("   No space reclaimed");
    }

    Ok(())
}

/// Check database integrity
async fn check_database_integrity(config: &MinerConfig) -> Result<()> {
    info!("Starting database integrity check");
    println!("🔍 Checking database integrity...");

    let db = RegistrationDb::new(&config.database).await?;

    // Perform integrity check
    let integrity_result = db
        .integrity_check()
        .await
        .map_err(|e| anyhow!("Failed to check database integrity: {}", e))?;

    if integrity_result {
        println!("✅ Database integrity check passed");
        info!("Database integrity check passed");
    } else {
        println!("❌ Database integrity check failed");
        error!("Database integrity check failed");
        return Err(anyhow!("Database integrity check failed"));
    }

    // Additional connection test
    match db.health_check().await {
        Ok(_) => {
            println!("✅ Database connection test passed");
        }
        Err(e) => {
            println!("❌ Database connection test failed: {e}");
            return Err(anyhow!("Database connection test failed: {}", e));
        }
    }

    Ok(())
}

/// Extract database file path from URL
fn extract_db_path_from_url(url: &str) -> Result<String> {
    if url.starts_with("sqlite:") {
        Ok(url.strip_prefix("sqlite:").unwrap().to_string())
    } else {
        Err(anyhow!("Unsupported database URL format: {}", url))
    }
}

/// Perform hot backup using VACUUM INTO
async fn perform_hot_backup(db: &RegistrationDb, backup_path: &Path) -> Result<()> {
    db.vacuum_into(backup_path.to_string_lossy().as_ref()).await
}

/// Verify backup integrity
async fn verify_backup_integrity(backup_path: &Path) -> Result<bool> {
    // Try to open the backup database and perform basic operations
    let backup_url = format!("sqlite:{}", backup_path.display());
    let temp_config = common::config::DatabaseConfig {
        url: backup_url,
        max_connections: 1,
        min_connections: 1,
        run_migrations: false,
        connect_timeout: std::time::Duration::from_secs(5),
        max_lifetime: None,
        idle_timeout: Some(std::time::Duration::from_secs(30)),
        ssl_config: None,
    };

    match RegistrationDb::new(&temp_config).await {
        Ok(backup_db) => {
            // Try to perform a simple query
            backup_db
                .integrity_check()
                .await
                .map_err(|e| anyhow!("Backup integrity check failed: {}", e))
        }
        Err(e) => {
            error!("Failed to open backup database: {}", e);
            Ok(false)
        }
    }
}

/// Collect comprehensive database statistics
async fn collect_database_stats(
    db: &RegistrationDb,
    config: &MinerConfig,
) -> Result<DatabaseStats> {
    let stats = db.get_database_stats().await?;

    // Get file size
    let db_path = extract_db_path_from_url(&config.database.url)?;
    let file_size = fs::metadata(&db_path)
        .map(|metadata| metadata.len())
        .unwrap_or(0);

    Ok(DatabaseStats {
        file_size,
        page_count: stats.page_count,
        page_size: stats.page_size,
        vacuum_count: stats.vacuum_count,
        table_stats: stats
            .table_stats
            .into_iter()
            .map(|ts| TableStats {
                table_name: ts.table_name,
                row_count: ts.row_count,
                size_bytes: ts.size_bytes,
            })
            .collect(),
        last_backup: None, // Would need to track this separately
    })
}

/// Clean old validator interactions
async fn clean_old_validator_interactions(
    db: &RegistrationDb,
    cutoff_date: DateTime<Utc>,
) -> Result<u64> {
    db.cleanup_old_validator_interactions(cutoff_date).await
}

/// Clean old SSH grants
async fn clean_old_ssh_grants(db: &RegistrationDb, cutoff_date: DateTime<Utc>) -> Result<u64> {
    db.cleanup_old_ssh_grants(cutoff_date).await
}

/// Clean stale executor health records
async fn clean_stale_executor_health(
    db: &RegistrationDb,
    cutoff_date: DateTime<Utc>,
) -> Result<u64> {
    db.cleanup_stale_executor_health(cutoff_date).await
}
