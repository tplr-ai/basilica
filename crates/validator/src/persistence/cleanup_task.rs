//! Database cleanup task for periodic maintenance
//!
//! Removes old GPU profiles and emission metrics to prevent database bloat

use anyhow::Result;
use std::sync::Arc;
use tokio::time::{interval, Duration};
use tracing::{error, info, warn};

use crate::persistence::GpuProfileRepository;

/// Configuration for cleanup tasks
#[derive(Debug, Clone)]
pub struct CleanupConfig {
    /// How often to run cleanup (in hours)
    pub run_interval_hours: u64,

    /// Delete GPU profiles older than this many days
    pub profile_retention_days: i64,

    /// Delete emission metrics older than this many days
    pub emission_retention_days: i64,

    /// Whether cleanup is enabled
    pub enabled: bool,
}

impl Default for CleanupConfig {
    fn default() -> Self {
        Self {
            run_interval_hours: 24, // Daily
            profile_retention_days: 30,
            emission_retention_days: 90, // Keep 3 months of emission history
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
            "Starting database cleanup task - will run every {} hours",
            self.config.run_interval_hours
        );

        let mut interval = interval(Duration::from_secs(self.config.run_interval_hours * 3600));

        loop {
            interval.tick().await;

            if let Err(e) = self.run_cleanup().await {
                error!("Database cleanup failed: {}", e);
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

        // Get and log current statistics
        match self.gpu_repo.get_profile_statistics().await {
            Ok(stats) => {
                info!(
                    "Database stats after cleanup: {} profiles, {} H100s, {} H200s",
                    stats.total_profiles,
                    stats.gpu_model_distribution.get("H100").unwrap_or(&0),
                    stats.gpu_model_distribution.get("H200").unwrap_or(&0)
                );
            }
            Err(e) => {
                warn!("Failed to get profile statistics: {}", e);
            }
        }

        info!("Database cleanup completed");
        Ok(())
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::gpu::MinerGpuProfile;
    use crate::persistence::SimplePersistence;
    use chrono::Utc;
    use common::identity::MinerUid;
    use std::collections::HashMap;
    use tempfile::NamedTempFile;

    async fn create_test_repo() -> Result<(Arc<GpuProfileRepository>, NamedTempFile)> {
        let temp_file = NamedTempFile::new()?;
        let db_path = temp_file.path().to_str().unwrap();
        let persistence = SimplePersistence::new(db_path, "test".to_string()).await?;
        let repo = Arc::new(GpuProfileRepository::new(persistence.pool().clone()));
        Ok((repo, temp_file))
    }

    #[tokio::test]
    async fn test_cleanup_old_profiles() {
        let (repo, _temp_file) = create_test_repo().await.unwrap();

        // Create old and new profiles
        let mut gpu_counts = HashMap::new();
        gpu_counts.insert("H100".to_string(), 1);

        // Old profile (40 days old)
        let old_profile = MinerGpuProfile {
            miner_uid: MinerUid::new(1),
            primary_gpu_model: "H100".to_string(),
            gpu_counts: gpu_counts.clone(),
            total_score: 0.5,
            verification_count: 1,
            last_updated: Utc::now() - chrono::Duration::days(40),
        };

        // Manually insert old profile
        let query = r#"
            INSERT INTO miner_gpu_profiles (
                miner_uid, primary_gpu_model, gpu_counts_json, 
                total_score, verification_count, last_updated, created_at
            ) VALUES (?, ?, ?, ?, ?, ?, CURRENT_TIMESTAMP)
        "#;

        sqlx::query(query)
            .bind(old_profile.miner_uid.as_u16() as i64)
            .bind(&old_profile.primary_gpu_model)
            .bind(serde_json::to_string(&old_profile.gpu_counts).unwrap())
            .bind(old_profile.total_score)
            .bind(old_profile.verification_count as i64)
            .bind(old_profile.last_updated.to_rfc3339())
            .execute(repo.pool())
            .await
            .unwrap();

        // Recent profile
        let recent_profile = MinerGpuProfile {
            miner_uid: MinerUid::new(2),
            primary_gpu_model: "H200".to_string(),
            gpu_counts,
            total_score: 0.8,
            verification_count: 1,
            last_updated: Utc::now(),
        };

        repo.upsert_gpu_profile(&recent_profile).await.unwrap();

        // Run cleanup
        let config = CleanupConfig {
            run_interval_hours: 24,
            profile_retention_days: 30,
            emission_retention_days: 90,
            enabled: true,
        };

        let cleanup_task = CleanupTask::new(config, repo.clone());
        cleanup_task.run_cleanup().await.unwrap();

        // Verify only recent profile remains
        let all_profiles = repo.get_all_gpu_profiles().await.unwrap();
        assert_eq!(all_profiles.len(), 1);
        assert_eq!(all_profiles[0].miner_uid, recent_profile.miner_uid);
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
