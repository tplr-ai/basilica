//! # Registration Database
//!
//! Simplified SQLite database for the miner with node UUID tracking

use anyhow::{Context, Result};
use sqlx::SqlitePool;
use std::path::Path;
use tokio::fs;
use tracing::{debug, info};

use basilica_common::config::DatabaseConfig;

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
}

#[cfg(test)]
mod tests {
    use super::*;

    #[tokio::test]
    async fn test_new_with_invalid_url() {
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
    async fn test_health_check() {
        let config = DatabaseConfig {
            url: "sqlite::memory:".to_string(),
            run_migrations: true,
            ..Default::default()
        };

        let db = RegistrationDb::new(&config).await.unwrap();
        db.health_check().await.unwrap();
    }
}
