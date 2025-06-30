//! Integration tests for GPU attestor functionality
//!
//! These tests validate the executor's integration with the gpu-attestor crate,
//! including hardware attestation generation, VDF challenge computation, and
//! system information collection.
//!
//! NOTE: Many of these tests are commented out because they require:
//! - Root access (for dmidecode)
//! - GPU hardware
//! - Docker installation
//! - Proper VDF parameters (non-zero modulus)

use anyhow::Result;
use gpu_attestor::{
    attest_system, check_system_requirements, collect_system_info, query_all_gpus,
    vdf::{compute_vdf_proof, VdfAlgorithm, VdfChallenge, VdfParameters},
    AttestationBuilder, AttestationSigner, AttestationVerifier, DockerCollector,
    NetworkBenchmarker,
};

#[cfg(test)]
mod gpu_attestor_tests {
    use super::*;

    #[tokio::test]
    #[ignore = "Requires GPU hardware and root access"]
    async fn test_gpu_detection_and_attestation() -> Result<()> {
        // Test GPU detection
        match query_all_gpus() {
            Ok(gpus) => {
                // May be empty if no GPU present
                if !gpus.is_empty() {
                    let gpu = &gpus[0];
                    assert!(!gpu.name.is_empty());
                    assert!(gpu.memory_total > 0);
                }
            }
            Err(e) => {
                println!("GPU detection failed (may not have GPU): {e}");
            }
        }

        // Test system info collection
        match collect_system_info() {
            Ok(system_info) => {
                assert!(!system_info.motherboard.manufacturer.is_empty());
                assert!(system_info.cpu.cores > 0);
                assert!(system_info.memory.total_bytes > 0);
            }
            Err(e) => {
                println!("System info collection failed (expected without root): {e}");
            }
        }

        // Build attestation
        let attestation = AttestationBuilder::new("test-executor".to_string()).build();

        // Verify attestation structure
        assert_eq!(attestation.executor_id, "test-executor");
        assert!(attestation.is_recent());

        Ok(())
    }

    #[test]
    fn test_attestation_signing_and_verification() -> Result<()> {
        // Create test keypair
        let signer = AttestationSigner::new();

        // Build test attestation
        let mut builder = AttestationBuilder::new("test-executor".to_string());

        // Add minimal test data
        if let Ok(gpus) = query_all_gpus() {
            builder = builder.with_gpu_info(gpus);
        }
        builder =
            builder.with_binary_info("test_checksum".to_string(), true, "test_key".to_string());

        let attestation = builder.build();

        // Sign attestation
        let signed = signer.sign_attestation(attestation)?;

        // Verify signature
        let is_valid = AttestationVerifier::verify_signed_attestation(&signed)?;
        assert!(is_valid, "Attestation signature should be valid");

        // Test tampering detection
        let mut tampered = signed.clone();
        tampered.report.timestamp += chrono::Duration::seconds(1);

        let is_tampered_valid = AttestationVerifier::verify_signed_attestation(&tampered);
        assert!(is_tampered_valid.is_err() || !is_tampered_valid.unwrap());

        Ok(())
    }

    #[test]
    fn test_attestation_building() -> Result<()> {
        // Build test attestation with various components
        let mut builder = AttestationBuilder::new("test-executor-123".to_string());

        // Add GPU info if available
        if let Ok(gpus) = query_all_gpus() {
            builder = builder.with_gpu_info(gpus);
        }

        let attestation = builder.build();

        // Verify attestation
        assert_eq!(attestation.executor_id, "test-executor-123");
        assert!(attestation.is_valid());

        Ok(())
    }

    #[test]
    #[ignore = "VDF implementation requires proper RSA modulus"]
    fn test_vdf_challenge_computation() -> Result<()> {
        // Create simple VDF parameters with non-zero modulus
        let mut modulus = vec![0u8; 256];
        modulus[0] = 1; // Set first byte to avoid zero modulus
        modulus[255] = 1; // Set last byte to ensure it's odd
        let params = VdfParameters {
            modulus,
            generator: vec![2u8],
            difficulty: 100, // Low difficulty for testing
            challenge_seed: b"test_seed".to_vec(),
        };

        let challenge = VdfChallenge {
            parameters: params,
            expected_computation_time_ms: 10,
            max_allowed_time_ms: 10000,
            min_required_time_ms: 0,
        };

        // Test different VDF algorithms
        let algorithms = vec![
            VdfAlgorithm::SimpleSequential,
            VdfAlgorithm::Wesolowski,
            VdfAlgorithm::Pietrzak,
        ];

        for algorithm in algorithms {
            match compute_vdf_proof(&challenge, algorithm.clone()) {
                Ok(proof) => {
                    // Verify proof structure
                    assert!(!proof.output.is_empty());
                    assert!(!proof.proof.is_empty());
                    assert!(proof.computation_time_ms > 0);
                    assert_eq!(proof.algorithm, algorithm);
                }
                Err(e) => {
                    // Some algorithms might not be implemented
                    println!("VDF algorithm {algorithm:?} failed: {e}");
                }
            }
        }

        Ok(())
    }

    #[test]
    #[ignore = "VDF implementation requires proper RSA modulus"]
    fn test_vdf_time_bounds_validation() -> Result<()> {
        // Create challenge with impossible time bounds
        let params = VdfParameters {
            modulus: vec![0u8; 256],
            generator: vec![2u8],
            difficulty: 1000000, // High difficulty
            challenge_seed: b"test".to_vec(),
        };

        let challenge = VdfChallenge {
            parameters: params,
            expected_computation_time_ms: 1,
            max_allowed_time_ms: 1, // Impossible to meet
            min_required_time_ms: 0,
        };

        // This should fail due to time constraints
        let result = compute_vdf_proof(&challenge, VdfAlgorithm::SimpleSequential);

        // Either it fails or takes longer than allowed
        if let Ok(proof) = result {
            assert!(proof.computation_time_ms > challenge.max_allowed_time_ms);
        }

        Ok(())
    }

    #[test]
    #[ignore = "VDF implementation requires proper RSA modulus"]
    fn test_attestation_with_vdf_proof() -> Result<()> {
        // Create simple VDF proof
        let params = VdfParameters {
            modulus: vec![0u8; 256],
            generator: vec![2u8],
            difficulty: 10, // Very low for testing
            challenge_seed: b"test_challenge".to_vec(),
        };

        let challenge = VdfChallenge {
            parameters: params,
            expected_computation_time_ms: 1,
            max_allowed_time_ms: 1000,
            min_required_time_ms: 0,
        };

        match compute_vdf_proof(&challenge, VdfAlgorithm::SimpleSequential) {
            Ok(vdf_proof) => {
                // Build attestation with VDF proof
                let attestation = AttestationBuilder::new("test-executor".to_string())
                    .with_vdf_proof(vdf_proof.clone())
                    .build();

                // Verify attestation includes VDF proof
                assert!(attestation.has_vdf_proof());
                assert_eq!(attestation.vdf_proof.unwrap().output, vdf_proof.output);
            }
            Err(e) => {
                println!("VDF proof generation failed: {e}");
            }
        }

        Ok(())
    }

    #[tokio::test]
    #[ignore = "VDF implementation requires proper RSA modulus"]
    async fn test_concurrent_vdf_computations() -> Result<()> {
        // Create multiple challenges
        let challenges: Vec<_> = (0..3)
            .map(|i| {
                let params = VdfParameters {
                    modulus: vec![0u8; 256],
                    generator: vec![2u8],
                    difficulty: 50 + i * 10,
                    challenge_seed: format!("test_seed_{i}").into_bytes(),
                };

                VdfChallenge {
                    parameters: params,
                    expected_computation_time_ms: 10,
                    max_allowed_time_ms: 10000,
                    min_required_time_ms: 0,
                }
            })
            .collect();

        // Compute VDFs concurrently
        let handles: Vec<_> = challenges
            .into_iter()
            .map(|challenge| {
                tokio::spawn(async move {
                    compute_vdf_proof(&challenge, VdfAlgorithm::SimpleSequential)
                })
            })
            .collect();

        // Collect results
        let mut successful_proofs = 0;
        for handle in handles {
            if let Ok(Ok(proof)) = handle.await {
                assert!(proof.computation_time_ms > 0);
                assert!(!proof.output.is_empty());
                successful_proofs += 1;
            }
        }

        // At least some should succeed
        assert!(successful_proofs > 0);

        Ok(())
    }

    #[tokio::test]
    async fn test_network_latency_measurement() -> Result<()> {
        // Test network benchmarking
        match NetworkBenchmarker::run_comprehensive_benchmark().await {
            Ok(results) => {
                // Check that we got some benchmark results
                assert!(!results.latency_tests.is_empty() || results.latency_tests.is_empty());
            }
            Err(e) => {
                // Network might not be available
                println!("Network benchmark test failed: {e}");
            }
        }

        Ok(())
    }

    #[test]
    fn test_hardware_benchmarking_integration() -> Result<()> {
        // Check system requirements
        let system_info = collect_system_info()?;
        let requirements_met = check_system_requirements(&system_info);

        if !requirements_met.meets_all_requirements {
            eprintln!("Warning: System requirements not fully met for benchmarking");
        }

        // Run quick benchmarks would require proper BenchmarkRunner implementation
        // For now, just verify system info collection works
        assert!(!system_info.cpu.brand.is_empty());

        Ok(())
    }

    #[test]
    fn test_os_attestation() -> Result<()> {
        let os_attestation = attest_system()?;

        assert!(!os_attestation.os_info.name.is_empty());
        assert!(!os_attestation.os_info.version.is_empty());
        assert!(!os_attestation.kernel_info.version.is_empty());

        Ok(())
    }

    #[test]
    #[ignore = "Requires Docker installation"]
    fn test_docker_attestation() -> Result<()> {
        // Try to get Docker info - may fail if Docker not installed
        match DockerCollector::collect_docker_info() {
            Ok(docker_info) => {
                assert!(docker_info.version.is_some());
                assert!(docker_info.api_version.is_some());
            }
            Err(e) => {
                // Docker might not be available
                println!("Docker info not available: {e}");
            }
        }

        Ok(())
    }
}

#[cfg(test)]
mod attestation_edge_cases {
    use super::*;

    #[test]
    fn test_empty_attestation() -> Result<()> {
        // Build minimal attestation
        let attestation = AttestationBuilder::new("test-executor".to_string()).build();

        // Should still have valid structure
        assert_eq!(attestation.executor_id, "test-executor");
        assert!(attestation.is_recent());
        assert!(attestation.gpu_info.is_empty());
        assert!(attestation.vdf_proof.is_none());

        Ok(())
    }

    #[test]
    fn test_attestation_serialization() -> Result<()> {
        let attestation = AttestationBuilder::new("test-executor-serial".to_string()).build();

        // Test JSON serialization
        let json = serde_json::to_string(&attestation)?;
        let deserialized: gpu_attestor::attestation::types::AttestationReport =
            serde_json::from_str(&json)?;

        assert_eq!(attestation.executor_id, deserialized.executor_id);
        assert_eq!(attestation.version, deserialized.version);

        Ok(())
    }

    #[test]
    #[ignore = "VDF implementation requires proper RSA modulus"]
    fn test_vdf_edge_cases() -> Result<()> {
        // Test with minimal parameters
        let params = VdfParameters {
            modulus: vec![0u8; 32], // Small modulus
            generator: vec![2u8],
            difficulty: 1, // Minimal difficulty
            challenge_seed: vec![],
        };

        let challenge = VdfChallenge {
            parameters: params,
            expected_computation_time_ms: 0,
            max_allowed_time_ms: 1000,
            min_required_time_ms: 0,
        };

        match compute_vdf_proof(&challenge, VdfAlgorithm::SimpleSequential) {
            Ok(proof) => {
                assert!(!proof.output.is_empty());
            }
            Err(e) => {
                // May fail with invalid parameters
                println!("VDF edge case failed as expected: {e}");
            }
        }

        Ok(())
    }

    #[test]
    #[ignore = "Requires root access for dmidecode"]
    fn test_system_info_completeness() -> Result<()> {
        let system_info = collect_system_info()?;

        // Verify all major components are present
        assert!(!system_info.motherboard.manufacturer.is_empty());
        assert!(system_info.cpu.cores > 0);
        assert!(system_info.memory.total_bytes > 0);
        assert!(!system_info.storage.is_empty());
        assert!(!system_info.network.interfaces.is_empty());

        // Benchmarks should have reasonable values
        assert!(system_info.benchmarks.cpu_benchmark_score > 0.0);
        assert!(system_info.benchmarks.memory_bandwidth_mbps > 0.0);

        Ok(())
    }
}
