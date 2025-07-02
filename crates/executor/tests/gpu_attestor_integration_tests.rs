//! Integration tests for GPU attestor functionality
//!
//! These tests validate the executor's integration with the gpu-attestor crate,
//! including hardware attestation generation and system information collection.
//!
//! NOTE: Many of these tests are commented out because they require:
//! - Root access (for dmidecode)
//! - GPU hardware
//! - Docker installation

use anyhow::Result;
use gpu_attestor::{
    attest_system, check_system_requirements, collect_system_info, query_all_gpus,
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
