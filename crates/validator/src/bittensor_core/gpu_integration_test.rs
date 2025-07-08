//! Integration test for GPU profile updates during validation

#[cfg(test)]
mod tests {
    use super::*;
    use crate::bittensor_core::weight_setter::{WeightSetter, ExecutorValidationResult};
    use crate::persistence::{SimplePersistence, GpuProfileRepository, entities::VerificationLog};
    use crate::config::emission::EmissionConfig;
    use common::identity::{MinerUid, ExecutorId};
    use tempfile::NamedTempFile;
    use std::sync::Arc;
    use std::collections::HashMap;
    use chrono::Utc;
    
    #[tokio::test]
    async fn test_validation_updates_gpu_profile() {
        // Create temporary database
        let temp_file = NamedTempFile::new().unwrap();
        let db_path = temp_file.path().to_str().unwrap();
        
        // Initialize persistence
        let persistence = Arc::new(SimplePersistence::new(db_path, "test".to_string()).await.unwrap());
        let gpu_repo = Arc::new(GpuProfileRepository::new(persistence.pool().clone()));
        
        // Create a mock weight setter (simplified for testing)
        // In real usage, this would be created with proper bittensor service
        let storage = common::MemoryStorage::new();
        
        // Create verification log with GPU data
        let executor_id = ExecutorId::new();
        let miner_uid = MinerUid::new(42);
        
        let gpu_specs = serde_json::json!({
            "cpu": {
                "model": "Intel Xeon",
                "cores": 32,
                "frequency_mhz": 2600,
                "architecture": "x86_64"
            },
            "gpu": [{
                "vendor": "NVIDIA",
                "model": "NVIDIA H100 80GB HBM3",
                "vram_mb": 81920,
                "driver_version": "535.129.03",
                "compute_capability": "9.0",
                "utilization_percent": 0.0
            }],
            "memory": {
                "total_mb": 262144,
                "available_mb": 200000,
                "memory_type": "DDR4"
            },
            "network": {
                "bandwidth_mbps": 10000.0,
                "latency_ms": 0.5,
                "packet_loss_percent": 0.0
            }
        });
        
        let verification_log = VerificationLog::new(
            executor_id.to_string(),
            "test_validator".to_string(),
            "attestation".to_string(),
            Utc::now(),
            0.9,
            true,
            gpu_specs,
            100,
            None,
        );
        
        // Store the verification log
        persistence.create_verification_log(&verification_log).await.unwrap();
        
        // Verify no GPU profile exists initially
        let initial_profile = gpu_repo.get_gpu_profile(miner_uid).await.unwrap();
        assert!(initial_profile.is_none());
        
        // Now simulate the validation flow updating the GPU profile
        // This would normally happen via process_executor_validation
        let validation_result = ExecutorValidationResult {
            executor_id: executor_id.clone(),
            is_valid: true,
            hardware_score: 0.9,
            gpu_count: 1,
            gpu_memory_gb: 80,
            network_bandwidth_mbps: 10000.0,
            attestation_valid: true,
            validation_timestamp: Utc::now(),
        };
        
        // Update GPU profile (this simulates what happens in process_executor_validation)
        let gpu_repo_for_update = GpuProfileRepository::new(persistence.pool().clone());
        let profile = crate::gpu::MinerGpuProfile::new(
            miner_uid,
            &[crate::gpu::ExecutorValidationResult::new_for_testing(
                executor_id.to_string(),
                "H100".to_string(),
                1,
                true,
                true,
            )],
            0.9,
        );
        
        gpu_repo_for_update.upsert_gpu_profile(&profile).await.unwrap();
        
        // Verify GPU profile was created
        let updated_profile = gpu_repo.get_gpu_profile(miner_uid).await.unwrap();
        assert!(updated_profile.is_some());
        
        let profile = updated_profile.unwrap();
        assert_eq!(profile.miner_uid, miner_uid);
        assert_eq!(profile.primary_gpu_model, "H100");
        assert_eq!(profile.total_score, 0.9);
        assert_eq!(profile.verification_count, 1);
        assert_eq!(profile.gpu_counts.get("H100"), Some(&1));
    }
    
    #[tokio::test]
    async fn test_multiple_executor_validations() {
        // Create temporary database
        let temp_file = NamedTempFile::new().unwrap();
        let db_path = temp_file.path().to_str().unwrap();
        
        // Initialize persistence
        let persistence = Arc::new(SimplePersistence::new(db_path, "test".to_string()).await.unwrap());
        let gpu_repo = Arc::new(GpuProfileRepository::new(persistence.pool().clone()));
        
        let miner_uid = MinerUid::new(100);
        
        // Simulate multiple executor validations for the same miner
        let validations = vec![
            crate::gpu::ExecutorValidationResult::new_for_testing(
                "executor1".to_string(),
                "H100".to_string(),
                2,
                true,
                true,
            ),
            crate::gpu::ExecutorValidationResult::new_for_testing(
                "executor2".to_string(),
                "H200".to_string(),
                1,
                true,
                true,
            ),
        ];
        
        // Create profile with multiple executors
        let profile = crate::gpu::MinerGpuProfile::new(miner_uid, &validations, 0.85);
        gpu_repo.upsert_gpu_profile(&profile).await.unwrap();
        
        // Verify profile aggregates GPU data correctly
        let stored_profile = gpu_repo.get_gpu_profile(miner_uid).await.unwrap().unwrap();
        assert_eq!(stored_profile.primary_gpu_model, "H100"); // H100 is primary (2 GPUs)
        assert_eq!(stored_profile.gpu_counts.get("H100"), Some(&2));
        assert_eq!(stored_profile.gpu_counts.get("H200"), Some(&1));
        assert_eq!(stored_profile.total_gpu_count(), 3);
    }
}