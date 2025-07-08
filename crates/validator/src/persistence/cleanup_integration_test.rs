//! Integration test for cleanup functionality

#[cfg(test)]
mod tests {
    use super::*;
    use crate::persistence::{SimplePersistence, GpuProfileRepository, CleanupTask, CleanupConfig};
    use crate::gpu::MinerGpuProfile;
    use crate::persistence::{EmissionMetrics, CategoryDistribution};
    use common::identity::MinerUid;
    use std::collections::HashMap;
    use std::sync::Arc;
    use chrono::{Utc, Duration};
    use tempfile::NamedTempFile;
    
    #[tokio::test]
    async fn test_cleanup_integration() {
        // Create temporary database
        let temp_file = NamedTempFile::new().unwrap();
        let db_path = temp_file.path().to_str().unwrap();
        
        // Initialize persistence
        let persistence = SimplePersistence::new(db_path, "test".to_string()).await.unwrap();
        let repo = Arc::new(GpuProfileRepository::new(persistence.pool().clone()));
        
        // Create old and new data
        let mut gpu_counts = HashMap::new();
        gpu_counts.insert("H100".to_string(), 2);
        
        // Old profile (35 days old)
        let old_profile = MinerGpuProfile {
            miner_uid: MinerUid::new(1),
            primary_gpu_model: "H100".to_string(),
            gpu_counts: gpu_counts.clone(),
            total_score: 0.75,
            verification_count: 10,
            last_updated: Utc::now() - Duration::days(35),
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
            total_score: 0.85,
            verification_count: 5,
            last_updated: Utc::now(),
        };
        
        repo.upsert_gpu_profile(&recent_profile).await.unwrap();
        
        // Old emission metrics (100 days old)
        let mut distributions = HashMap::new();
        distributions.insert("H100".to_string(), CategoryDistribution {
            category: "H100".to_string(),
            miner_count: 10,
            total_weight: 5000,
            average_score: 0.7,
        });
        
        let old_metrics = EmissionMetrics {
            id: 0,
            timestamp: Utc::now() - Duration::days(100),
            burn_amount: 1000,
            burn_percentage: 10.0,
            category_distributions: distributions.clone(),
            total_miners: 10,
            weight_set_block: 1000,
        };
        
        // Manually insert old emission metrics
        let metrics_query = r#"
            INSERT INTO emission_metrics (
                timestamp, burn_amount, burn_percentage, 
                category_distributions_json, total_miners, weight_set_block
            ) VALUES (?, ?, ?, ?, ?, ?)
        "#;
        
        sqlx::query(metrics_query)
            .bind(old_metrics.timestamp.to_rfc3339())
            .bind(old_metrics.burn_amount as i64)
            .bind(old_metrics.burn_percentage)
            .bind(serde_json::to_string(&old_metrics.category_distributions).unwrap())
            .bind(old_metrics.total_miners as i64)
            .bind(old_metrics.weight_set_block as i64)
            .execute(repo.pool())
            .await
            .unwrap();
        
        // Recent emission metrics
        let recent_metrics = EmissionMetrics {
            id: 0,
            timestamp: Utc::now(),
            burn_amount: 500,
            burn_percentage: 5.0,
            category_distributions: distributions,
            total_miners: 15,
            weight_set_block: 2000,
        };
        
        repo.store_emission_metrics(&recent_metrics).await.unwrap();
        
        // Verify initial state
        let all_profiles = repo.get_all_gpu_profiles().await.unwrap();
        assert_eq!(all_profiles.len(), 2);
        
        let all_metrics = repo.get_emission_metrics_history(10, 0).await.unwrap();
        assert_eq!(all_metrics.len(), 2);
        
        // Configure and run cleanup
        let config = CleanupConfig {
            run_interval_hours: 24,
            profile_retention_days: 30,
            emission_retention_days: 90,
            enabled: true,
        };
        
        let cleanup_task = CleanupTask::new(config, repo.clone());
        cleanup_task.run_cleanup().await.unwrap();
        
        // Verify cleanup results
        let remaining_profiles = repo.get_all_gpu_profiles().await.unwrap();
        assert_eq!(remaining_profiles.len(), 1);
        assert_eq!(remaining_profiles[0].miner_uid, recent_profile.miner_uid);
        
        let remaining_metrics = repo.get_emission_metrics_history(10, 0).await.unwrap();
        assert_eq!(remaining_metrics.len(), 1);
        assert_eq!(remaining_metrics[0].weight_set_block, 2000);
        
        // Verify statistics after cleanup
        let stats = repo.get_profile_statistics().await.unwrap();
        assert_eq!(stats.total_profiles, 1);
        assert_eq!(stats.gpu_model_distribution.get("H200"), Some(&1));
        assert_eq!(stats.gpu_model_distribution.get("H100"), Some(&0));
    }
}