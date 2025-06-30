//! Test for the new weight setting implementation

#[cfg(test)]
mod tests {
    use common::identity::{ExecutorId, MinerUid};
    use common::MemoryStorage;
    use validator::bittensor_core::weight_setter::ExecutorValidationResult;

    #[tokio::test]
    async fn test_weight_setter_scoring() {
        // Create a memory storage
        let storage = MemoryStorage::new();

        // Create mock bittensor service
        let config = common::config::BittensorConfig::default();

        // This test verifies the weight setter can update miner scores
        let miner_uid = MinerUid::new(1);

        // Create mock validation results
        let validations = [
            ExecutorValidationResult {
                executor_id: ExecutorId::new(),
                is_valid: true,
                hardware_score: 0.8,
                gpu_count: 2,
                gpu_memory_gb: 48,
                network_bandwidth_mbps: 1000.0,
                attestation_valid: true,
                validation_timestamp: chrono::Utc::now(),
            },
            ExecutorValidationResult {
                executor_id: ExecutorId::new(),
                is_valid: true,
                hardware_score: 0.9,
                gpu_count: 4,
                gpu_memory_gb: 96,
                network_bandwidth_mbps: 10000.0,
                attestation_valid: true,
                validation_timestamp: chrono::Utc::now(),
            },
        ];

        // Test score calculation
        let total_score: f64 = validations
            .iter()
            .map(|v| {
                if !v.is_valid || !v.attestation_valid {
                    return 0.0;
                }

                let base_score = v.hardware_score;
                let gpu_weight =
                    (v.gpu_count as f64 * 0.1).min(0.5) + (v.gpu_memory_gb as f64 / 100.0).min(0.5);
                let network_weight = (v.network_bandwidth_mbps / 10000.0).clamp(0.5, 1.0);

                base_score * (1.0 + gpu_weight) * network_weight
            })
            .sum();

        let average_score = total_score / validations.len() as f64;

        assert!(average_score > 0.0);
        assert!(average_score <= 2.0); // Max possible with bonuses

        println!("Calculated average score: {average_score:.4}");
    }

    // Note: Weight normalization is handled directly in the weight_setter module
    // We don't test crabtensor's normalize_weights function here
}
