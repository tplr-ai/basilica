//! Attestation module for GPU attestation reports
//!
//! This module provides comprehensive attestation functionality including:
//! - Creating and building attestation reports
//! - Signing attestations with cryptographic keys
//! - Verifying attestation signatures and content
//! - Utility functions for saving/loading attestations

pub mod builder;
pub mod signer;
pub mod types;
pub mod utils;
pub mod verifier;

// Re-export main types and functions for convenient access
pub use builder::AttestationBuilder;
pub use signer::AttestationSigner;
pub use types::*;
pub use utils::*;
pub use verifier::{AttestationVerifier, VerificationReport};

use crate::gpu::GpuInfo;
use crate::hardware::SystemInfo;
use crate::network::NetworkBenchmarkResults;
use crate::vdf::VdfProof;
use anyhow::Result;

/// Create a new attestation builder for the given executor
pub fn create_attestation_builder(executor_id: String) -> AttestationBuilder {
    AttestationBuilder::new(executor_id)
}

/// Create a comprehensive attestation with all system information
pub fn create_comprehensive_attestation(executor_id: String) -> AttestationBuilder {
    AttestationBuilder::comprehensive(executor_id)
}

/// Create a minimal attestation with only required information
pub fn create_minimal_attestation(
    executor_id: String,
    gpu_info: Vec<GpuInfo>,
) -> AttestationBuilder {
    AttestationBuilder::minimal(executor_id, gpu_info)
}

/// Sign an attestation report using a new ephemeral key
pub fn sign_attestation(report: AttestationReport) -> Result<SignedAttestation> {
    let signer = AttestationSigner::new();
    signer.sign_attestation(report)
}

/// Sign an attestation report using an existing signer
pub fn sign_with_signer(
    signer: &AttestationSigner,
    report: AttestationReport,
) -> Result<SignedAttestation> {
    signer.sign_attestation(report)
}

/// Verify a signed attestation
pub fn verify_attestation(signed_attestation: &SignedAttestation) -> Result<bool> {
    AttestationVerifier::verify_signed_attestation(signed_attestation)
}

/// Quick verification of an attestation (signature and basic fields only)
pub fn quick_verify_attestation(signed_attestation: &SignedAttestation) -> Result<bool> {
    AttestationVerifier::quick_verify(signed_attestation)
}

/// Detailed verification with comprehensive reporting
pub fn detailed_verify_attestation(
    signed_attestation: &SignedAttestation,
) -> Result<VerificationReport> {
    AttestationVerifier::detailed_verify(signed_attestation)
}

/// Create and sign an attestation in one step
pub fn create_and_sign_attestation(
    executor_id: String,
    gpu_info: Vec<GpuInfo>,
    system_info: SystemInfo,
    network_benchmark: NetworkBenchmarkResults,
    binary_checksum: String,
    signature_verified: bool,
    key_fingerprint: String,
) -> Result<SignedAttestation> {
    let report = AttestationBuilder::new(executor_id)
        .with_gpu_info(gpu_info)
        .with_system_info(system_info)
        .with_network_benchmark(network_benchmark)
        .with_binary_info(binary_checksum, signature_verified, key_fingerprint)
        .build();

    sign_attestation(report)
}

/// Create attestation with VDF proof
pub fn create_attestation_with_vdf(
    executor_id: String,
    gpu_info: Vec<GpuInfo>,
    vdf_proof: VdfProof,
) -> Result<SignedAttestation> {
    let report = AttestationBuilder::minimal(executor_id, gpu_info)
        .with_vdf_proof(vdf_proof)
        .build();

    sign_attestation(report)
}

/// Create attestation with model verification
pub fn create_attestation_with_model_verification(
    executor_id: String,
    gpu_info: Vec<GpuInfo>,
    provider_id: String,
    model_hash: String,
    expected_fingerprint: Vec<u8>,
    observed_fingerprint: Vec<u8>,
) -> Result<SignedAttestation> {
    let report = AttestationBuilder::minimal(executor_id, gpu_info)
        .with_model_verification(
            provider_id,
            model_hash,
            expected_fingerprint,
            observed_fingerprint,
        )
        .build();

    sign_attestation(report)
}

/// Validate attestation report structure and content
pub fn validate_attestation_report(report: &AttestationReport) -> Result<()> {
    if !report.is_valid() {
        anyhow::bail!("Attestation report is invalid");
    }
    Ok(())
}

/// Get attestation summary for quick inspection
pub fn get_attestation_summary(signed_attestation: &SignedAttestation) -> AttestationSummary {
    generate_attestation_summary(signed_attestation)
}

/// Compare two attestations for differences
pub fn compare_two_attestations(
    attestation1: &SignedAttestation,
    attestation2: &SignedAttestation,
) -> AttestationComparison {
    compare_attestations(attestation1, attestation2)
}

/// Batch verify multiple attestations
pub fn verify_multiple_attestations(attestations: &[SignedAttestation]) -> Result<Vec<bool>> {
    AttestationVerifier::verify_multiple(attestations)
}

/// Save attestation with automatic path generation
pub fn save_attestation_with_timestamp(
    signed_attestation: &SignedAttestation,
    base_dir: &std::path::Path,
) -> Result<std::path::PathBuf> {
    let timestamp = chrono::Utc::now().format("%Y%m%d_%H%M%S");
    let executor_id = &signed_attestation.report.executor_id;
    let filename = format!("attestation_{executor_id}_{timestamp}");
    let file_path = base_dir.join(filename);

    save_attestation_to_files(signed_attestation, &file_path)?;
    Ok(file_path)
}

/// Load the most recent attestation for an executor
pub fn load_latest_attestation(
    base_dir: &std::path::Path,
    executor_id: &str,
) -> Result<SignedAttestation> {
    use std::fs;

    let mut latest_file: Option<std::path::PathBuf> = None;
    let mut latest_time: Option<std::time::SystemTime> = None;

    for entry in fs::read_dir(base_dir)? {
        let entry = entry?;
        let path = entry.path();

        if path.extension() == Some(std::ffi::OsStr::new("json")) {
            if let Some(filename) = path.file_stem() {
                if filename.to_string_lossy().contains(executor_id) {
                    if let Ok(metadata) = entry.metadata() {
                        if let Ok(modified) = metadata.modified() {
                            if latest_time.is_none_or(|t| modified > t) {
                                latest_time = Some(modified);
                                latest_file = Some(path.with_extension(""));
                            }
                        }
                    }
                }
            }
        }
    }

    if let Some(file_path) = latest_file {
        load_attestation_from_files(&file_path)
    } else {
        anyhow::bail!("No attestation found for executor: {}", executor_id);
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_create_attestation_builder() {
        let builder = create_attestation_builder("test_executor".to_string());
        let report = builder.build();
        assert_eq!(report.executor_id, "test_executor");
    }

    #[test]
    fn test_comprehensive_attestation() {
        let builder = create_comprehensive_attestation("test_executor".to_string());
        let report = builder.build();
        assert_eq!(report.executor_id, "test_executor");
        assert!(!report.system_info.cpu.brand.is_empty());
    }

    #[test]
    fn test_minimal_attestation() {
        let gpu_info = vec![GpuInfo::new(
            crate::gpu::GpuVendor::Unknown,
            "Test GPU".to_string(),
            "1.0.0".to_string(),
        )];
        let builder = create_minimal_attestation("test_executor".to_string(), gpu_info.clone());
        let report = builder.build();
        assert_eq!(report.executor_id, "test_executor");
        assert_eq!(report.gpu_info.len(), 1);
    }

    #[test]
    fn test_sign_and_verify_attestation() {
        let report = create_attestation_builder("test_executor".to_string()).build();
        let signed_attestation = sign_attestation(report).unwrap();

        // Quick verify should work for basic signature check
        let result = quick_verify_attestation(&signed_attestation);
        assert!(result.is_ok());
    }

    #[test]
    fn test_create_and_sign_attestation() {
        let gpu_info = vec![GpuInfo::new(
            crate::gpu::GpuVendor::Unknown,
            "Test GPU".to_string(),
            "1.0.0".to_string(),
        )];
        let system_info = SystemInfo {
            motherboard: crate::hardware::MotherboardInfo {
                manufacturer: "Unknown".to_string(),
                product_name: "Unknown".to_string(),
                version: "Unknown".to_string(),
                serial_number: None,
                asset_tag: None,
                bios_vendor: "Unknown".to_string(),
                bios_version: "Unknown".to_string(),
                bios_date: "Unknown".to_string(),
            },
            cpu: crate::hardware::CpuInfo {
                brand: "Unknown".to_string(),
                vendor_id: "Unknown".to_string(),
                cores: 0,
                threads: 0,
                frequency_mhz: 0,
                architecture: "Unknown".to_string(),
                features: Vec::new(),
                temperature: None,
            },
            memory: crate::hardware::MemoryInfo {
                total_bytes: 0,
                available_bytes: 0,
                used_bytes: 0,
                swap_total_bytes: 0,
                swap_used_bytes: 0,
                memory_modules: Vec::new(),
            },
            storage: Vec::new(),
            network: crate::hardware::NetworkInfo {
                interfaces: Vec::new(),
                connectivity_test: crate::hardware::ConnectivityTest {
                    can_reach_internet: false,
                    dns_resolution_working: false,
                    latency_ms: None,
                },
            },
            benchmarks: crate::hardware::BenchmarkResults {
                cpu_benchmark_score: 0.0,
                memory_bandwidth_mbps: 0.0,
                disk_sequential_read_mbps: 0.0,
                disk_sequential_write_mbps: 0.0,
                network_throughput_mbps: None,
            },
        };
        let network_benchmark = NetworkBenchmarkResults::default();

        let signed_attestation = create_and_sign_attestation(
            "test_executor".to_string(),
            gpu_info,
            system_info,
            network_benchmark,
            "test_checksum".to_string(),
            true,
            "test_fingerprint".to_string(),
        )
        .unwrap();

        assert_eq!(signed_attestation.report.executor_id, "test_executor");
        assert!(signed_attestation.report.binary_info.signature_verified);
    }

    #[test]
    fn test_detailed_verify_attestation() {
        let report = create_attestation_builder("test_executor".to_string()).build();
        let signed_attestation = sign_attestation(report).unwrap();

        let verification_report = detailed_verify_attestation(&signed_attestation).unwrap();
        assert!(verification_report.signature_valid);
    }

    #[test]
    fn test_validate_attestation_report() {
        let report = create_attestation_builder("test_executor".to_string()).build();

        // Should fail validation due to missing GPU info
        let result = validate_attestation_report(&report);
        assert!(result.is_err());
    }

    #[test]
    fn test_get_attestation_summary() {
        let report = create_attestation_builder("test_executor".to_string()).build();
        let signed_attestation = sign_attestation(report).unwrap();

        let summary = get_attestation_summary(&signed_attestation);
        assert_eq!(summary.executor_id, "test_executor");
        assert_eq!(summary.get_status(), "INVALID"); // No GPU info
    }

    #[test]
    fn test_compare_two_attestations() {
        let report1 = create_attestation_builder("test_executor".to_string()).build();
        let attestation1 = sign_attestation(report1).unwrap();

        let report2 = create_attestation_builder("test_executor".to_string()).build();
        let attestation2 = sign_attestation(report2).unwrap();

        let comparison = compare_two_attestations(&attestation1, &attestation2);
        assert!(comparison.same_executor);
    }

    #[test]
    #[ignore = "Requires GPU hardware"]
    fn test_verify_multiple_attestations() {
        let gpu_info = vec![GpuInfo::new(
            crate::gpu::GpuVendor::Unknown,
            "Test GPU".to_string(),
            "1.0.0".to_string(),
        )];
        let report1 = create_attestation_builder("executor1".to_string())
            .with_gpu_info(gpu_info.clone())
            .with_binary_info("test_checksum".to_string(), true, "test_key".to_string())
            .build();
        let attestation1 = sign_attestation(report1).unwrap();

        let report2 = create_attestation_builder("executor2".to_string())
            .with_gpu_info(gpu_info)
            .with_binary_info("test_checksum2".to_string(), true, "test_key2".to_string())
            .build();
        let attestation2 = sign_attestation(report2).unwrap();

        let attestations = vec![attestation1, attestation2];
        let results = verify_multiple_attestations(&attestations).unwrap();
        assert_eq!(results.len(), 2);
    }

    #[test]
    fn test_create_attestation_with_vdf() {
        use crate::vdf::{VdfAlgorithm, VdfProof};

        let gpu_info = vec![GpuInfo::new(
            crate::gpu::GpuVendor::Unknown,
            "Test GPU".to_string(),
            "1.0.0".to_string(),
        )];
        let vdf_proof = VdfProof::new(
            vec![1, 2, 3],
            vec![4, 5, 6],
            1000,
            VdfAlgorithm::SimpleSequential,
        );

        let signed_attestation =
            create_attestation_with_vdf("test_executor".to_string(), gpu_info, vdf_proof).unwrap();

        assert!(signed_attestation.report.has_vdf_proof());
    }

    #[test]
    fn test_create_attestation_with_model_verification() {
        let gpu_info = vec![GpuInfo::new(
            crate::gpu::GpuVendor::Unknown,
            "Test GPU".to_string(),
            "1.0.0".to_string(),
        )];
        let expected = vec![1, 2, 3, 4];
        let observed = vec![1, 2, 3, 4];

        let signed_attestation = create_attestation_with_model_verification(
            "test_executor".to_string(),
            gpu_info,
            "test_provider".to_string(),
            "test_hash".to_string(),
            expected,
            observed,
        )
        .unwrap();

        assert!(signed_attestation.report.has_model_fingerprint());
    }
}
