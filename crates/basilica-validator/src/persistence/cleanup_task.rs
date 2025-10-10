//! Database cleanup task for periodic maintenance
//!
//! Removes old GPU profiles and emission metrics to prevent database bloat

use anyhow::Result;
use std::sync::Arc;
use tokio::time::{interval, Duration};
use tracing::{error, info};

use crate::persistence::gpu_profile_repository::GpuProfileRepository;

/// Configuration for cleanup tasks
#[derive(Debug, Clone, serde::Serialize, serde::Deserialize)]
pub struct CleanupConfig {
    /// How often to run cleanup (in hours)
    pub run_interval_hours: u64,

    /// Delete GPU profiles older than this many days
    pub profile_retention_days: i64,

    /// Delete emission metrics older than this many days
    pub emission_retention_days: i64,

    /// How often to run stale node cleanup (in minutes)
    pub stale_node_cleanup_cadence_minutes: u64,

    /// Whether cleanup is enabled
    pub enabled: bool,
}

impl Default for CleanupConfig {
    fn default() -> Self {
        Self {
            run_interval_hours: 24,
            profile_retention_days: 30,
            emission_retention_days: 90,
            stale_node_cleanup_cadence_minutes: 30,
            enabled: true,
        }
    }
}

/// Cleanup task runner
pub struct CleanupTask {
    config: CleanupConfig,
    gpu_repo: Arc<GpuProfileRepository>,
}

impl CleanupTask {
    /// Create a new cleanup task
    pub fn new(config: CleanupConfig, gpu_repo: Arc<GpuProfileRepository>) -> Self {
        Self { config, gpu_repo }
    }

    /// Start the cleanup task loop
    pub async fn start(&self) -> Result<()> {
        if !self.config.enabled {
            info!("Database cleanup task is disabled");
            return Ok(());
        }

        info!(
            "Starting database cleanup task - will run every {} hours, stale node cleanup every {} minutes",
            self.config.run_interval_hours, self.config.stale_node_cleanup_cadence_minutes
        );

        // Run both cleanup types concurrently
        tokio::try_join!(
            self.run_database_cleanup_loop(),
            self.run_stale_node_cleanup_loop()
        )?;

        Ok(())
    }

    /// Run the database cleanup loop (daily)
    async fn run_database_cleanup_loop(&self) -> Result<()> {
        let mut interval = interval(Duration::from_secs(self.config.run_interval_hours * 3600));

        loop {
            interval.tick().await;

            if let Err(e) = self.run_cleanup().await {
                error!("Database cleanup failed: {}", e);
            }
        }
    }

    /// Run the stale node cleanup loop
    async fn run_stale_node_cleanup_loop(&self) -> Result<()> {
        let mut interval = interval(Duration::from_secs(
            self.config.stale_node_cleanup_cadence_minutes * 60,
        ));

        loop {
            interval.tick().await;

            if let Err(e) = self.run_stale_node_cleanup().await {
                error!("Stale node cleanup failed: {}", e);
            }
        }
    }

    /// Run a single cleanup cycle
    pub async fn run_cleanup(&self) -> Result<()> {
        info!("Starting database cleanup");

        // Clean up old GPU profiles
        let profile_count = self
            .gpu_repo
            .cleanup_old_profiles(self.config.profile_retention_days)
            .await?;

        if profile_count > 0 {
            info!("Cleaned up {} old GPU profiles", profile_count);
        }

        // Clean up old emission metrics
        let metrics_count = self
            .gpu_repo
            .cleanup_old_emission_metrics(self.config.emission_retention_days)
            .await?;

        if metrics_count > 0 {
            info!("Cleaned up {} old emission metrics", metrics_count);
        }

        info!("Database cleanup completed");
        Ok(())
    }

    /// Run stale node cleanup
    async fn run_stale_node_cleanup(&self) -> Result<()> {
        info!("Starting stale node cleanup");

        let cleaned_count = self.gpu_repo.cleanup_stale_nodes().await?;

        if cleaned_count > 0 {
            info!(
                security = true,
                cleaned_nodes = cleaned_count,
                cleanup_reason = "stale_node_cleanup_task",
                "Cleaned up {} stale nodes and their GPU assignments",
                cleaned_count
            );
        } else {
            info!("No stale nodes found for cleanup");
        }

        Ok(())
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::gpu::MinerGpuProfile;
    use crate::persistence::SimplePersistence;
    use basilica_common::identity::MinerUid;
    use chrono::Utc;
    use std::collections::HashMap;
    use tempfile::NamedTempFile;

    async fn create_test_repo() -> Result<(Arc<GpuProfileRepository>, NamedTempFile)> {
        let temp_file = NamedTempFile::new()?;
        let db_path = temp_file.path().to_str().unwrap();
        let persistence = SimplePersistence::new(db_path, "test".to_string()).await?;
        persistence.run_migrations().await?;
        let repo = Arc::new(GpuProfileRepository::new(persistence.pool().clone()));
        Ok((repo, temp_file))
    }

    #[tokio::test]
    async fn test_cleanup_old_profiles() {
        let (repo, _temp_file) = create_test_repo().await.unwrap();

        // Create old and new profiles
        let mut gpu_counts = HashMap::new();
        gpu_counts.insert("A100".to_string(), 1);

        // Old profile (40 days old)
        let old_profile = MinerGpuProfile {
            miner_uid: MinerUid::new(1),
            gpu_counts: gpu_counts.clone(),
            total_score: 0.5,
            verification_count: 1,
            last_updated: Utc::now() - chrono::Duration::days(40),
            last_successful_validation: None,
        };

        // Manually insert old profile
        let query = r#"
            INSERT INTO miner_gpu_profiles (
                miner_uid, gpu_counts_json,
                total_score, verification_count, last_updated, last_successful_validation, created_at
            ) VALUES (?, ?, ?, ?, ?, ?, CURRENT_TIMESTAMP)
        "#;

        sqlx::query(query)
            .bind(old_profile.miner_uid.as_u16() as i64)
            .bind(serde_json::to_string(&old_profile.gpu_counts).unwrap())
            .bind(old_profile.total_score)
            .bind(old_profile.verification_count as i64)
            .bind(old_profile.last_updated.to_rfc3339())
            .bind(
                old_profile
                    .last_successful_validation
                    .map(|dt| dt.to_rfc3339()),
            )
            .execute(repo.pool())
            .await
            .unwrap();

        // Recent profile
        let recent_profile = MinerGpuProfile {
            miner_uid: MinerUid::new(2),
            gpu_counts,
            total_score: 0.8,
            verification_count: 1,
            last_updated: Utc::now(),
            last_successful_validation: None,
        };

        repo.upsert_gpu_profile(&recent_profile).await.unwrap();

        // Run cleanup
        let config = CleanupConfig {
            run_interval_hours: 24,
            profile_retention_days: 30,
            emission_retention_days: 90,
            stale_node_cleanup_cadence_minutes: 30,
            enabled: true,
        };

        let cleanup_task = CleanupTask::new(config, repo.clone());
        cleanup_task.run_cleanup().await.unwrap();

        // Verify only recent profile remains in the database
        let remaining_count: i64 = sqlx::query_scalar("SELECT COUNT(*) FROM miner_gpu_profiles")
            .fetch_one(repo.pool())
            .await
            .unwrap();
        assert_eq!(remaining_count, 1);

        // Verify it's the recent profile that remains
        let remaining_uid: i64 = sqlx::query_scalar("SELECT miner_uid FROM miner_gpu_profiles")
            .fetch_one(repo.pool())
            .await
            .unwrap();
        assert_eq!(remaining_uid, recent_profile.miner_uid.as_u16() as i64);
    }

    #[tokio::test]
    async fn test_cleanup_config_default() {
        let config = CleanupConfig::default();
        assert_eq!(config.run_interval_hours, 24);
        assert_eq!(config.profile_retention_days, 30);
        assert_eq!(config.emission_retention_days, 90);
        assert!(config.enabled);
    }

    #[tokio::test]
    async fn test_cleanup_disabled() {
        let (repo, _temp_file) = create_test_repo().await.unwrap();

        let config = CleanupConfig {
            enabled: false,
            ..Default::default()
        };

        let cleanup_task = CleanupTask::new(config, repo);
        let result = cleanup_task.start().await;
        assert!(result.is_ok());
    }
}
