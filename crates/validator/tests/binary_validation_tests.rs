//! Binary Validation Unit Tests
//!
//! Unit tests for binary validation functionality

use crate::validation::types::*;
use crate::miner_prover::verification::VerificationEngine;
use crate::config::VerificationConfig;
use std::time::Duration;

#[cfg(test)]
mod tests {
    use super::*;

    #[tokio::test]
    async fn test_binary_validation_score_calculation() {
        let config = VerificationConfig {
            verification_interval: Duration::from_secs(600),
            max_concurrent_verifications: 5,
            challenge_timeout: Duration::from_secs(30),
            min_score_threshold: 0.5,
            max_miners_per_round: 10,
            min_verification_interval: Duration::from_secs(300),
            netuid: 387,
            use_dynamic_discovery: true,
            discovery_timeout: Duration::from_secs(30),
            fallback_to_static: true,
            cache_miner_info_ttl: Duration::from_secs(300),
            grpc_port_offset: None,
            binary_validation: crate::config::BinaryValidationConfig::default(),
        };

        let verification_engine = VerificationEngine::new(config);

        let executor_result = ExecutorResult {
            gpu_name: "RTX 4090".to_string(),
            gpu_uuid: "GPU-12345".to_string(),
            cpu_info: BinaryCpuInfo {
                model: "Intel i9".to_string(),
                cores: 8,
                threads: 16,
                frequency_mhz: 3600,
            },
            memory_info: BinaryMemoryInfo {
                total_gb: 32.0,
                available_gb: 16.0,
            },
            network_info: BinaryNetworkInfo {
                interfaces: vec![],
            },
            matrix_c: CompressedMatrix {
                rows: 2,
                cols: 2,
                data: vec![1.0, 2.0, 3.0, 4.0],
            },
            computation_time_ns: 1_000_000,
            checksum: [0u8; 32],
            sm_utilization: SmUtilizationStats {
                min_utilization: 0.8,
                max_utilization: 1.0,
                avg_utilization: 0.9,
                per_sm_stats: vec![],
            },
            active_sms: 80,
            total_sms: 84,
            memory_bandwidth_gbps: 1000.0,
            anti_debug_passed: true,
            timing_fingerprint: 0x12345678,
        };

        let validation_result = ValidatorBinaryOutput {
            success: true,
            executor_result: Some(executor_result),
            error_message: None,
            execution_time_ms: 150,
            validation_score: 0.0, // Will be calculated
        };

        let score = verification_engine.calculate_binary_validation_score(&validation_result).unwrap();

        // Expected score: 0.3 (base) + 0.2 (anti-debug) + 0.2 (high utilization) + 0.15 (gpu efficiency) + 0.1 (bandwidth) + 0.05 (timing)
        assert!(score >= 0.9 && score <= 1.0, "Score should be between 0.9 and 1.0, got {}", score);
    }

    #[tokio::test]
    async fn test_combined_score_calculation() {
        let config = VerificationConfig {
            verification_interval: Duration::from_secs(600),
            max_concurrent_verifications: 5,
            challenge_timeout: Duration::from_secs(30),
            min_score_threshold: 0.5,
            max_miners_per_round: 10,
            min_verification_interval: Duration::from_secs(300),
            netuid: 387,
            use_dynamic_discovery: true,
            discovery_timeout: Duration::from_secs(30),
            fallback_to_static: true,
            cache_miner_info_ttl: Duration::from_secs(300),
            grpc_port_offset: None,
            binary_validation: crate::config::BinaryValidationConfig::default(),
        };

        let verification_engine = VerificationEngine::new(config);

        let ssh_score = 0.8;
        let binary_score = 0.9;

        let combined_score = verification_engine.calculate_combined_verification_score(
            ssh_score, binary_score, true, true
        );

        // Expected: (0.8 * 0.2) + (0.9 * 0.8) = 0.16 + 0.72 = 0.88
        assert!((combined_score - 0.88).abs() < 0.01, "Combined score should be 0.88, got {}", combined_score);
    }

    #[tokio::test]
    async fn test_ssh_failure_results_in_zero_score() {
        let config = VerificationConfig {
            verification_interval: Duration::from_secs(600),
            max_concurrent_verifications: 5,
            challenge_timeout: Duration::from_secs(30),
            min_score_threshold: 0.5,
            max_miners_per_round: 10,
            min_verification_interval: Duration::from_secs(300),
            netuid: 387,
            use_dynamic_discovery: true,
            discovery_timeout: Duration::from_secs(30),
            fallback_to_static: true,
            cache_miner_info_ttl: Duration::from_secs(300),
            grpc_port_offset: None,
            binary_validation: crate::config::BinaryValidationConfig::default(),
        };

        let verification_engine = VerificationEngine::new(config);

        let combined_score = verification_engine.calculate_combined_verification_score(
            0.0, 0.9, false, true
        );

        assert_eq!(combined_score, 0.0, "SSH failure should result in zero score");
    }

    #[tokio::test]
    async fn test_binary_validation_disabled_uses_ssh_score() {
        let mut config = VerificationConfig {
            verification_interval: Duration::from_secs(600),
            max_concurrent_verifications: 5,
            challenge_timeout: Duration::from_secs(30),
            min_score_threshold: 0.5,
            max_miners_per_round: 10,
            min_verification_interval: Duration::from_secs(300),
            netuid: 387,
            use_dynamic_discovery: true,
            discovery_timeout: Duration::from_secs(30),
            fallback_to_static: true,
            cache_miner_info_ttl: Duration::from_secs(300),
            grpc_port_offset: None,
            binary_validation: crate::config::BinaryValidationConfig::default(),
        };
        
        config.binary_validation.enabled = false;

        let verification_engine = VerificationEngine::new(config);

        let combined_score = verification_engine.calculate_combined_verification_score(
            0.8, 0.9, true, true
        );

        assert_eq!(combined_score, 0.8, "When binary validation is disabled, should use SSH score");
    }
}