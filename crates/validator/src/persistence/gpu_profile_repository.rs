//! GPU Profile Repository for emission-based allocation
//!
//! Provides CRUD operations for GPU profiles and emission metrics

use anyhow::Result;
use chrono::{DateTime, Utc};
use sqlx::{Row, SqlitePool};
use std::collections::HashMap;
use tracing::{debug, info};

use super::super::gpu::MinerGpuProfile;
use common::identity::MinerUid;

/// Repository for GPU profile operations
pub struct GpuProfileRepository {
    pool: SqlitePool,
}

impl GpuProfileRepository {
    /// Create a new repository instance
    pub fn new(pool: SqlitePool) -> Self {
        Self { pool }
    }

    /// Get reference to the underlying pool
    pub fn pool(&self) -> &SqlitePool {
        &self.pool
    }

    /// Store or update a miner's GPU profile
    pub async fn upsert_gpu_profile(&self, profile: &MinerGpuProfile) -> Result<()> {
        let gpu_counts_json = serde_json::to_string(&profile.gpu_counts)?;

        let query = r#"
            INSERT INTO miner_gpu_profiles (
                miner_uid, primary_gpu_model, gpu_counts_json, 
                total_score, verification_count, last_updated, created_at
            ) VALUES (?, ?, ?, ?, ?, ?, CURRENT_TIMESTAMP)
            ON CONFLICT(miner_uid) DO UPDATE SET
                primary_gpu_model = excluded.primary_gpu_model,
                gpu_counts_json = excluded.gpu_counts_json,
                total_score = excluded.total_score,
                verification_count = excluded.verification_count,
                last_updated = excluded.last_updated
        "#;

        sqlx::query(query)
            .bind(profile.miner_uid.as_u16() as i64)
            .bind(&profile.primary_gpu_model)
            .bind(&gpu_counts_json)
            .bind(profile.total_score)
            .bind(profile.verification_count as i64)
            .bind(profile.last_updated.to_rfc3339())
            .execute(&self.pool)
            .await?;

        debug!(
            miner_uid = profile.miner_uid.as_u16(),
            primary_gpu = %profile.primary_gpu_model,
            "GPU profile stored"
        );

        Ok(())
    }

    /// Get a specific miner's GPU profile
    pub async fn get_gpu_profile(&self, miner_uid: MinerUid) -> Result<Option<MinerGpuProfile>> {
        let query = r#"
            SELECT miner_uid, primary_gpu_model, gpu_counts_json, 
                   total_score, verification_count, last_updated
            FROM miner_gpu_profiles
            WHERE miner_uid = ?
        "#;

        let row = sqlx::query(query)
            .bind(miner_uid.as_u16() as i64)
            .fetch_optional(&self.pool)
            .await?;

        match row {
            Some(row) => {
                let miner_uid_val: i64 = row.get("miner_uid");
                let primary_gpu_model: String = row.get("primary_gpu_model");
                let gpu_counts_json: String = row.get("gpu_counts_json");
                let total_score: f64 = row.get("total_score");
                let verification_count: i64 = row.get("verification_count");
                let last_updated_str: String = row.get("last_updated");

                let gpu_counts: HashMap<String, u32> = serde_json::from_str(&gpu_counts_json)?;
                let last_updated =
                    DateTime::parse_from_rfc3339(&last_updated_str)?.with_timezone(&Utc);

                Ok(Some(MinerGpuProfile {
                    miner_uid: MinerUid::new(miner_uid_val as u16),
                    primary_gpu_model,
                    gpu_counts,
                    total_score,
                    verification_count: verification_count as u32,
                    last_updated,
                }))
            }
            None => Ok(None),
        }
    }

    /// Get all GPU profiles
    pub async fn get_all_gpu_profiles(&self) -> Result<Vec<MinerGpuProfile>> {
        let query = r#"
            SELECT miner_uid, primary_gpu_model, gpu_counts_json, 
                   total_score, verification_count, last_updated
            FROM miner_gpu_profiles
            ORDER BY total_score DESC
        "#;

        let rows = sqlx::query(query).fetch_all(&self.pool).await?;

        let mut profiles = Vec::new();

        for row in rows {
            let miner_uid_val: i64 = row.get("miner_uid");
            let primary_gpu_model: String = row.get("primary_gpu_model");
            let gpu_counts_json: String = row.get("gpu_counts_json");
            let total_score: f64 = row.get("total_score");
            let verification_count: i64 = row.get("verification_count");
            let last_updated_str: String = row.get("last_updated");

            let gpu_counts: HashMap<String, u32> = serde_json::from_str(&gpu_counts_json)?;
            let last_updated = DateTime::parse_from_rfc3339(&last_updated_str)?.with_timezone(&Utc);

            profiles.push(MinerGpuProfile {
                miner_uid: MinerUid::new(miner_uid_val as u16),
                primary_gpu_model,
                gpu_counts,
                total_score,
                verification_count: verification_count as u32,
                last_updated,
            });
        }

        Ok(profiles)
    }

    /// Get profiles by GPU model
    pub async fn get_profiles_by_gpu_model(&self, gpu_model: &str) -> Result<Vec<MinerGpuProfile>> {
        let query = r#"
            SELECT miner_uid, primary_gpu_model, gpu_counts_json, 
                   total_score, verification_count, last_updated
            FROM miner_gpu_profiles
            WHERE primary_gpu_model = ?
            ORDER BY total_score DESC
        "#;

        let rows = sqlx::query(query)
            .bind(gpu_model)
            .fetch_all(&self.pool)
            .await?;

        let mut profiles = Vec::new();

        for row in rows {
            let miner_uid_val: i64 = row.get("miner_uid");
            let primary_gpu_model: String = row.get("primary_gpu_model");
            let gpu_counts_json: String = row.get("gpu_counts_json");
            let total_score: f64 = row.get("total_score");
            let verification_count: i64 = row.get("verification_count");
            let last_updated_str: String = row.get("last_updated");

            let gpu_counts: HashMap<String, u32> = serde_json::from_str(&gpu_counts_json)?;
            let last_updated = DateTime::parse_from_rfc3339(&last_updated_str)?.with_timezone(&Utc);

            profiles.push(MinerGpuProfile {
                miner_uid: MinerUid::new(miner_uid_val as u16),
                primary_gpu_model,
                gpu_counts,
                total_score,
                verification_count: verification_count as u32,
                last_updated,
            });
        }

        Ok(profiles)
    }

    /// Delete old profiles that haven't been updated in N days
    pub async fn cleanup_old_profiles(&self, days_old: i64) -> Result<u64> {
        let cutoff_date = Utc::now() - chrono::Duration::days(days_old);

        let query = r#"
            DELETE FROM miner_gpu_profiles
            WHERE last_updated < ?
        "#;

        let result = sqlx::query(query)
            .bind(cutoff_date.to_rfc3339())
            .execute(&self.pool)
            .await?;

        let deleted = result.rows_affected();

        if deleted > 0 {
            info!("Cleaned up {} old GPU profiles", deleted);
        }

        Ok(deleted)
    }

    /// Get GPU profile statistics
    pub async fn get_profile_statistics(&self) -> Result<GpuProfileStatistics> {
        // Get total profiles
        let total_query = "SELECT COUNT(*) as count FROM miner_gpu_profiles";
        let total_row = sqlx::query(total_query).fetch_one(&self.pool).await?;
        let total_profiles: i64 = total_row.get("count");

        // Get model distribution and scores
        let stats_query = r#"
            SELECT 
                primary_gpu_model,
                COUNT(*) as miner_count,
                AVG(total_score) as avg_score,
                SUM(json_extract(gpu_counts_json, '$."' || primary_gpu_model || '"')) as total_gpus
            FROM miner_gpu_profiles
            GROUP BY primary_gpu_model
        "#;

        let rows = sqlx::query(stats_query).fetch_all(&self.pool).await?;

        let mut gpu_model_distribution = HashMap::new();
        let mut average_score_by_model = HashMap::new();
        let mut total_gpus_by_model = HashMap::new();

        for row in rows {
            let model: String = row.get("primary_gpu_model");
            let miner_count: i64 = row.get("miner_count");
            let avg_score: f64 = row.get("avg_score");
            let total_gpus: Option<i64> = row.get("total_gpus");

            gpu_model_distribution.insert(model.clone(), miner_count as u64);
            average_score_by_model.insert(model.clone(), avg_score);
            total_gpus_by_model.insert(model, total_gpus.unwrap_or(0) as u32);
        }

        Ok(GpuProfileStatistics {
            total_profiles: total_profiles as u64,
            gpu_model_distribution,
            average_score_by_model,
            total_gpus_by_model,
            last_updated: Utc::now(),
        })
    }
}

/// GPU profile statistics
#[derive(Debug, Clone, serde::Serialize, serde::Deserialize)]
pub struct GpuProfileStatistics {
    pub total_profiles: u64,
    pub gpu_model_distribution: HashMap<String, u64>,
    pub average_score_by_model: HashMap<String, f64>,
    pub total_gpus_by_model: HashMap<String, u32>,
    pub last_updated: DateTime<Utc>,
}

/// Emission metrics tracking
pub struct EmissionMetrics {
    pub id: i64,
    pub timestamp: DateTime<Utc>,
    pub burn_amount: u64,
    pub burn_percentage: f64,
    pub category_distributions: HashMap<String, CategoryDistribution>,
    pub total_miners: u32,
    pub weight_set_block: u64,
}

#[derive(Debug, Clone, serde::Serialize, serde::Deserialize)]
pub struct CategoryDistribution {
    pub category: String,
    pub miner_count: u32,
    pub total_weight: u64,
    pub average_score: f64,
}

impl GpuProfileRepository {
    /// Store emission metrics for a weight setting round
    pub async fn store_emission_metrics(&self, metrics: &EmissionMetrics) -> Result<i64> {
        let distributions_json = serde_json::to_string(&metrics.category_distributions)?;

        let query = r#"
            INSERT INTO emission_metrics (
                timestamp, burn_amount, burn_percentage, 
                category_distributions_json, total_miners, weight_set_block
            ) VALUES (?, ?, ?, ?, ?, ?)
        "#;

        let result = sqlx::query(query)
            .bind(metrics.timestamp.to_rfc3339())
            .bind(metrics.burn_amount as i64)
            .bind(metrics.burn_percentage)
            .bind(&distributions_json)
            .bind(metrics.total_miners as i64)
            .bind(metrics.weight_set_block as i64)
            .execute(&self.pool)
            .await?;

        Ok(result.last_insert_rowid())
    }

    /// Store weight allocation history for auditing
    pub async fn store_weight_allocation(
        &self,
        emission_metrics_id: i64,
        miner_uid: MinerUid,
        gpu_category: &str,
        allocated_weight: u64,
        miner_score: f64,
        category_total_score: f64,
        weight_set_block: u64,
    ) -> Result<()> {
        let query = r#"
            INSERT INTO weight_allocation_history (
                miner_uid, gpu_category, allocated_weight, 
                miner_score, category_total_score, weight_set_block, 
                timestamp, emission_metrics_id
            ) VALUES (?, ?, ?, ?, ?, ?, ?, ?)
        "#;

        sqlx::query(query)
            .bind(miner_uid.as_u16() as i64)
            .bind(gpu_category)
            .bind(allocated_weight as i64)
            .bind(miner_score)
            .bind(category_total_score)
            .bind(weight_set_block as i64)
            .bind(Utc::now().to_rfc3339())
            .bind(emission_metrics_id)
            .execute(&self.pool)
            .await?;

        Ok(())
    }

    /// Get emission metrics history
    pub async fn get_emission_metrics_history(
        &self,
        limit: u32,
        offset: u32,
    ) -> Result<Vec<EmissionMetrics>> {
        let query = r#"
            SELECT id, timestamp, burn_amount, burn_percentage, 
                   category_distributions_json, total_miners, weight_set_block
            FROM emission_metrics
            ORDER BY timestamp DESC
            LIMIT ? OFFSET ?
        "#;

        let rows = sqlx::query(query)
            .bind(limit as i64)
            .bind(offset as i64)
            .fetch_all(&self.pool)
            .await?;

        let mut metrics = Vec::new();

        for row in rows {
            let id: i64 = row.get("id");
            let timestamp_str: String = row.get("timestamp");
            let burn_amount: i64 = row.get("burn_amount");
            let burn_percentage: f64 = row.get("burn_percentage");
            let distributions_json: String = row.get("category_distributions_json");
            let total_miners: i64 = row.get("total_miners");
            let weight_set_block: i64 = row.get("weight_set_block");

            let timestamp = DateTime::parse_from_rfc3339(&timestamp_str)?.with_timezone(&Utc);
            let category_distributions: HashMap<String, CategoryDistribution> =
                serde_json::from_str(&distributions_json)?;

            metrics.push(EmissionMetrics {
                id,
                timestamp,
                burn_amount: burn_amount as u64,
                burn_percentage,
                category_distributions,
                total_miners: total_miners as u32,
                weight_set_block: weight_set_block as u64,
            });
        }

        Ok(metrics)
    }

    /// Get emission metrics by block range
    pub async fn get_emission_metrics_by_block_range(
        &self,
        start_block: u64,
        end_block: u64,
    ) -> Result<Vec<EmissionMetrics>> {
        let query = r#"
            SELECT id, timestamp, burn_amount, burn_percentage, 
                   category_distributions_json, total_miners, weight_set_block
            FROM emission_metrics
            WHERE weight_set_block >= ? AND weight_set_block <= ?
            ORDER BY weight_set_block ASC
        "#;

        let rows = sqlx::query(query)
            .bind(start_block as i64)
            .bind(end_block as i64)
            .fetch_all(&self.pool)
            .await?;

        let mut metrics = Vec::new();

        for row in rows {
            let id: i64 = row.get("id");
            let timestamp_str: String = row.get("timestamp");
            let burn_amount: i64 = row.get("burn_amount");
            let burn_percentage: f64 = row.get("burn_percentage");
            let distributions_json: String = row.get("category_distributions_json");
            let total_miners: i64 = row.get("total_miners");
            let weight_set_block: i64 = row.get("weight_set_block");

            let timestamp = DateTime::parse_from_rfc3339(&timestamp_str)?.with_timezone(&Utc);
            let category_distributions: HashMap<String, CategoryDistribution> =
                serde_json::from_str(&distributions_json)?;

            metrics.push(EmissionMetrics {
                id,
                timestamp,
                burn_amount: burn_amount as u64,
                burn_percentage,
                category_distributions,
                total_miners: total_miners as u32,
                weight_set_block: weight_set_block as u64,
            });
        }

        Ok(metrics)
    }

    /// Get latest emission metrics
    pub async fn get_latest_emission_metrics(&self) -> Result<Option<EmissionMetrics>> {
        let query = r#"
            SELECT id, timestamp, burn_amount, burn_percentage, 
                   category_distributions_json, total_miners, weight_set_block
            FROM emission_metrics
            ORDER BY timestamp DESC
            LIMIT 1
        "#;

        let row = sqlx::query(query).fetch_optional(&self.pool).await?;

        match row {
            Some(row) => {
                let id: i64 = row.get("id");
                let timestamp_str: String = row.get("timestamp");
                let burn_amount: i64 = row.get("burn_amount");
                let burn_percentage: f64 = row.get("burn_percentage");
                let distributions_json: String = row.get("category_distributions_json");
                let total_miners: i64 = row.get("total_miners");
                let weight_set_block: i64 = row.get("weight_set_block");

                let timestamp = DateTime::parse_from_rfc3339(&timestamp_str)?.with_timezone(&Utc);
                let category_distributions: HashMap<String, CategoryDistribution> =
                    serde_json::from_str(&distributions_json)?;

                Ok(Some(EmissionMetrics {
                    id,
                    timestamp,
                    burn_amount: burn_amount as u64,
                    burn_percentage,
                    category_distributions,
                    total_miners: total_miners as u32,
                    weight_set_block: weight_set_block as u64,
                }))
            }
            None => Ok(None),
        }
    }

    /// Clean up old emission metrics
    pub async fn cleanup_old_emission_metrics(&self, days_old: i64) -> Result<u64> {
        let cutoff_date = Utc::now() - chrono::Duration::days(days_old);

        let query = r#"
            DELETE FROM emission_metrics
            WHERE timestamp < ?
        "#;

        let result = sqlx::query(query)
            .bind(cutoff_date.to_rfc3339())
            .execute(&self.pool)
            .await?;

        let deleted = result.rows_affected();

        if deleted > 0 {
            info!("Cleaned up {} old emission metrics", deleted);
        }

        Ok(deleted)
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use tempfile::NamedTempFile;

    async fn create_test_pool() -> Result<(SqlitePool, NamedTempFile)> {
        let temp_file = NamedTempFile::new()?;
        let db_path = temp_file.path().to_str().unwrap();

        // Create a new persistence instance which will handle the database creation and migrations
        let persistence =
            crate::persistence::SimplePersistence::new(db_path, "test".to_string()).await?;

        // Return the pool from the persistence instance along with the temp file to keep it alive
        Ok((persistence.pool().clone(), temp_file))
    }

    #[tokio::test]
    async fn test_gpu_profile_storage() {
        let (pool, _temp_file) = create_test_pool().await.unwrap();
        let repo = GpuProfileRepository::new(pool);

        // Create a test profile
        let mut gpu_counts = HashMap::new();
        gpu_counts.insert("H100".to_string(), 2);

        let profile = MinerGpuProfile {
            miner_uid: MinerUid::new(1),
            primary_gpu_model: "H100".to_string(),
            gpu_counts,
            total_score: 0.85,
            verification_count: 5,
            last_updated: Utc::now(),
        };

        // Store the profile
        repo.upsert_gpu_profile(&profile).await.unwrap();

        // Retrieve the profile
        let retrieved = repo.get_gpu_profile(profile.miner_uid).await.unwrap();
        assert!(retrieved.is_some());

        let retrieved_profile = retrieved.unwrap();
        assert_eq!(
            retrieved_profile.primary_gpu_model,
            profile.primary_gpu_model
        );
        assert_eq!(retrieved_profile.total_score, profile.total_score);
        assert_eq!(
            retrieved_profile.verification_count,
            profile.verification_count
        );
    }

    #[tokio::test]
    async fn test_profile_update() {
        let (pool, _temp_file) = create_test_pool().await.unwrap();
        let repo = GpuProfileRepository::new(pool);

        let miner_uid = MinerUid::new(1);
        let mut gpu_counts = HashMap::new();
        gpu_counts.insert("H100".to_string(), 1);

        // Initial profile
        let profile1 = MinerGpuProfile {
            miner_uid,
            primary_gpu_model: "H100".to_string(),
            gpu_counts: gpu_counts.clone(),
            total_score: 0.5,
            verification_count: 1,
            last_updated: Utc::now(),
        };

        repo.upsert_gpu_profile(&profile1).await.unwrap();

        // Update profile
        gpu_counts.insert("H100".to_string(), 2);
        let profile2 = MinerGpuProfile {
            miner_uid,
            primary_gpu_model: "H100".to_string(),
            gpu_counts,
            total_score: 0.8,
            verification_count: 2,
            last_updated: Utc::now(),
        };

        repo.upsert_gpu_profile(&profile2).await.unwrap();

        // Verify update
        let retrieved = repo.get_gpu_profile(miner_uid).await.unwrap().unwrap();
        assert_eq!(retrieved.total_score, 0.8);
        assert_eq!(retrieved.verification_count, 2);
        assert_eq!(retrieved.gpu_counts.get("H100"), Some(&2));
    }

    #[tokio::test]
    async fn test_get_profiles_by_gpu_model() {
        let (pool, _temp_file) = create_test_pool().await.unwrap();
        let repo = GpuProfileRepository::new(pool);

        // Create profiles with different GPU models
        for i in 0..3 {
            let mut gpu_counts = HashMap::new();
            let model = if i < 2 { "H100" } else { "H200" };
            gpu_counts.insert(model.to_string(), 1);

            let profile = MinerGpuProfile {
                miner_uid: MinerUid::new(i),
                primary_gpu_model: model.to_string(),
                gpu_counts,
                total_score: 0.5 + (i as f64 * 0.1),
                verification_count: 1,
                last_updated: Utc::now(),
            };

            repo.upsert_gpu_profile(&profile).await.unwrap();
        }

        // Query H100 profiles
        let h100_profiles = repo.get_profiles_by_gpu_model("H100").await.unwrap();
        assert_eq!(h100_profiles.len(), 2);

        // Query H200 profiles
        let h200_profiles = repo.get_profiles_by_gpu_model("H200").await.unwrap();
        assert_eq!(h200_profiles.len(), 1);
    }

    #[tokio::test]
    async fn test_emission_metrics_storage() {
        let (pool, _temp_file) = create_test_pool().await.unwrap();
        let repo = GpuProfileRepository::new(pool);

        // Create test metrics
        let mut distributions = HashMap::new();
        distributions.insert(
            "H100".to_string(),
            CategoryDistribution {
                category: "H100".to_string(),
                miner_count: 10,
                total_weight: 4000,
                average_score: 0.7,
            },
        );
        distributions.insert(
            "H200".to_string(),
            CategoryDistribution {
                category: "H200".to_string(),
                miner_count: 5,
                total_weight: 6000,
                average_score: 0.8,
            },
        );

        let metrics = EmissionMetrics {
            id: 0, // Will be set by DB
            timestamp: Utc::now(),
            burn_amount: 1000,
            burn_percentage: 10.0,
            category_distributions: distributions,
            total_miners: 15,
            weight_set_block: 12345,
        };

        // Store metrics
        let metrics_id = repo.store_emission_metrics(&metrics).await.unwrap();
        assert!(metrics_id > 0);

        // Store some allocations
        repo.store_weight_allocation(metrics_id, MinerUid::new(1), "H100", 400, 0.7, 7.0, 12345)
            .await
            .unwrap();

        // Retrieve metrics
        let history = repo.get_emission_metrics_history(10, 0).await.unwrap();
        assert_eq!(history.len(), 1);

        let retrieved = &history[0];
        assert_eq!(retrieved.burn_percentage, 10.0);
        assert_eq!(retrieved.total_miners, 15);
        assert_eq!(retrieved.weight_set_block, 12345);
    }

    #[tokio::test]
    async fn test_cleanup_old_profiles() {
        let (pool, _temp_file) = create_test_pool().await.unwrap();
        let repo = GpuProfileRepository::new(pool);

        // Create an old profile
        let mut gpu_counts = HashMap::new();
        gpu_counts.insert("H100".to_string(), 1);

        let old_profile = MinerGpuProfile {
            miner_uid: MinerUid::new(1),
            primary_gpu_model: "H100".to_string(),
            gpu_counts: gpu_counts.clone(),
            total_score: 0.5,
            verification_count: 1,
            last_updated: Utc::now() - chrono::Duration::days(31),
        };

        // Manually insert with old timestamp
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
            .execute(&repo.pool)
            .await
            .unwrap();

        // Create a recent profile
        let recent_profile = MinerGpuProfile {
            miner_uid: MinerUid::new(2),
            primary_gpu_model: "H200".to_string(),
            gpu_counts,
            total_score: 0.8,
            verification_count: 1,
            last_updated: Utc::now(),
        };

        repo.upsert_gpu_profile(&recent_profile).await.unwrap();

        // Cleanup profiles older than 30 days
        let deleted = repo.cleanup_old_profiles(30).await.unwrap();
        assert_eq!(deleted, 1);

        // Verify only recent profile remains
        let all_profiles = repo.get_all_gpu_profiles().await.unwrap();
        assert_eq!(all_profiles.len(), 1);
        assert_eq!(all_profiles[0].miner_uid, recent_profile.miner_uid);
    }

    #[tokio::test]
    async fn test_profile_statistics() {
        let (pool, _temp_file) = create_test_pool().await.unwrap();
        let repo = GpuProfileRepository::new(pool);

        // Create profiles with different GPU models
        let models = [("H100", 3), ("H200", 2), ("OTHER", 1)];

        for (i, (model, count)) in models.iter().enumerate() {
            let mut gpu_counts = HashMap::new();
            gpu_counts.insert(model.to_string(), *count);

            let profile = MinerGpuProfile {
                miner_uid: MinerUid::new(i as u16),
                primary_gpu_model: model.to_string(),
                gpu_counts,
                total_score: 0.5 + (i as f64 * 0.1),
                verification_count: 1,
                last_updated: Utc::now(),
            };

            repo.upsert_gpu_profile(&profile).await.unwrap();
        }

        // Get statistics
        let stats = repo.get_profile_statistics().await.unwrap();

        assert_eq!(stats.total_profiles, 3);
        assert_eq!(stats.gpu_model_distribution.get("H100"), Some(&1));
        assert_eq!(stats.gpu_model_distribution.get("H200"), Some(&1));
        assert_eq!(stats.gpu_model_distribution.get("OTHER"), Some(&1));

        // Check average scores
        assert!((stats.average_score_by_model.get("H100").unwrap() - 0.5).abs() < 0.01);
        assert!((stats.average_score_by_model.get("H200").unwrap() - 0.6).abs() < 0.01);
        assert!((stats.average_score_by_model.get("OTHER").unwrap() - 0.7).abs() < 0.01);
    }

    #[tokio::test]
    async fn test_emission_metrics_block_range() {
        let (pool, _temp_file) = create_test_pool().await.unwrap();
        let repo = GpuProfileRepository::new(pool);

        // Create metrics at different blocks
        for block in [100, 200, 300, 400, 500] {
            let mut distributions = HashMap::new();
            distributions.insert(
                "H100".to_string(),
                CategoryDistribution {
                    category: "H100".to_string(),
                    miner_count: 10,
                    total_weight: 4000,
                    average_score: 0.7,
                },
            );

            let metrics = EmissionMetrics {
                id: 0,
                timestamp: Utc::now(),
                burn_amount: 1000,
                burn_percentage: 10.0,
                category_distributions: distributions,
                total_miners: 10,
                weight_set_block: block,
            };

            repo.store_emission_metrics(&metrics).await.unwrap();
        }

        // Query specific range
        let range_metrics = repo
            .get_emission_metrics_by_block_range(200, 400)
            .await
            .unwrap();
        assert_eq!(range_metrics.len(), 3);
        assert_eq!(range_metrics[0].weight_set_block, 200);
        assert_eq!(range_metrics[1].weight_set_block, 300);
        assert_eq!(range_metrics[2].weight_set_block, 400);
    }

    #[tokio::test]
    async fn test_latest_emission_metrics() {
        let (pool, _temp_file) = create_test_pool().await.unwrap();
        let repo = GpuProfileRepository::new(pool);

        // Initially no metrics
        let latest = repo.get_latest_emission_metrics().await.unwrap();
        assert!(latest.is_none());

        // Add a metric
        let mut distributions = HashMap::new();
        distributions.insert(
            "H100".to_string(),
            CategoryDistribution {
                category: "H100".to_string(),
                miner_count: 5,
                total_weight: 2000,
                average_score: 0.8,
            },
        );

        let metrics = EmissionMetrics {
            id: 0,
            timestamp: Utc::now(),
            burn_amount: 500,
            burn_percentage: 5.0,
            category_distributions: distributions,
            total_miners: 5,
            weight_set_block: 1000,
        };

        repo.store_emission_metrics(&metrics).await.unwrap();

        // Check latest
        let latest = repo.get_latest_emission_metrics().await.unwrap();
        assert!(latest.is_some());
        let latest = latest.unwrap();
        assert_eq!(latest.weight_set_block, 1000);
        assert_eq!(latest.burn_percentage, 5.0);
    }
}
