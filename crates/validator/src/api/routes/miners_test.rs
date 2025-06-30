#[cfg(test)]
mod tests {
    use super::*;
    use crate::persistence::simple_persistence::SimplePersistence;
    use std::sync::Arc;

    #[tokio::test]
    async fn test_miner_persistence_operations() {
        // Initialize persistence layer with test database
        let persistence = SimplePersistence::new("test_miners_integration.db", "test_validator_hotkey".to_string()).await.unwrap();

        // Test miner registration
        let executors = vec![crate::api::types::ExecutorRegistration {
            executor_id: "test-executor-1".to_string(),
            grpc_address: "127.0.0.1:50051".to_string(),
            gpu_count: 2,
            gpu_specs: vec![
                crate::api::types::GpuSpec {
                    name: "RTX 4090".to_string(),
                    memory_gb: 24,
                    compute_capability: "8.9".to_string(),
                },
            ],
            cpu_specs: crate::api::types::CpuSpec {
                cores: 16,
                model: "Intel i9-13900K".to_string(),
                memory_gb: 64,
            },
        }];

        // Register miner
        persistence.register_miner(
            "test-miner-1",
            "5GrwvaEF5zXb26Fz9rcQpDWS57CtERHpNehXCPcNoHGKutQY",
            "http://localhost:8080",
            &executors
        ).await.unwrap();

        // List miners
        let miners = persistence.get_registered_miners(0, 10).await.unwrap();
        assert!(!miners.is_empty(), "Should have at least one miner");

        // Get specific miner
        let miner = persistence.get_miner_by_id("test-miner-1").await.unwrap();
        assert!(miner.is_some(), "Miner should exist");

        // Update miner
        let update_request = crate::api::types::UpdateMinerRequest {
            endpoint: Some("http://localhost:8081".to_string()),
            signature: "test_signature".to_string(),
            executors: None,
        };
        persistence.update_miner("test-miner-1", &update_request).await.unwrap();

        // Get miner health
        let health = persistence.get_miner_health("test-miner-1").await.unwrap();
        assert!(health.is_some(), "Health data should exist");

        // Schedule verification
        persistence.schedule_verification(
            "test-miner-1",
            "test-verification-1",
            "gpu_attestation",
            Some("test-executor-1")
        ).await.unwrap();

        // Get miner executors
        let executors = persistence.get_miner_executors("test-miner-1").await.unwrap();
        assert!(!executors.is_empty(), "Should have executors");

        // Remove miner
        persistence.remove_miner("test-miner-1").await.unwrap();

        // Verify removal
        let miner = persistence.get_miner_by_id("test-miner-1").await.unwrap();
        assert!(miner.is_none(), "Miner should be removed");

        // Cleanup
        std::fs::remove_file("test_miners_integration.db").ok();
    }

    #[test]
    fn test_crypto_verification_function_exists() {
        // Test that the crypto verification function compiles and is available
        let result = std::panic::catch_unwind(|| {
            // This tests that the function signature is correct
            let _: fn(&str, &str, &str) -> std::pin::Pin<Box<dyn std::future::Future<Output = Result<bool, anyhow::Error>> + Send>> = common::crypto::verify_signature;
        });
        assert!(result.is_ok(), "Crypto verification function should be properly defined");
    }

    #[test]
    fn test_miner_types_serialization() {
        use serde_json;

        // Test that our types can be serialized/deserialized
        let miner_details = MinerDetails {
            miner_id: "test".to_string(),
            hotkey: "test".to_string(),
            endpoint: "http://test".to_string(),
            status: MinerStatus::Active,
            executor_count: 1,
            total_gpu_count: 2,
            verification_score: 0.95,
            uptime_percentage: 99.9,
            last_seen: chrono::Utc::now(),
            registered_at: chrono::Utc::now(),
        };

        let json = serde_json::to_string(&miner_details).unwrap();
        assert!(!json.is_empty());

        let register_request = RegisterMinerRequest {
            miner_id: "test".to_string(),
            hotkey: "test".to_string(),
            endpoint: "http://test".to_string(),
            signature: "test".to_string(),
            executors: vec![],
        };

        let json = serde_json::to_string(&register_request).unwrap();
        assert!(!json.is_empty());
    }
}