//! Attestation builder implementation for creating attestation reports

use chrono::Utc;

use super::types::*;
use crate::docker::DockerAttestation;
use crate::gpu::GpuInfo;
use crate::hardware::SystemInfo;
use crate::network::{IpInfo, NetworkBenchmarkResults};
use crate::os::OsAttestation;
use crate::validation::{PerformanceValidator, VirtualizationDetector};
use crate::vdf::VdfProof;

pub struct AttestationBuilder {
    report: AttestationReport,
}

impl AttestationBuilder {
    pub fn new(executor_id: String) -> Self {
        Self {
            report: AttestationReport::new(executor_id),
        }
    }

    pub fn with_binary_info(
        mut self,
        checksum: String,
        signature_verified: bool,
        key_fingerprint: String,
    ) -> Self {
        self.report.binary_info = BinaryInfo::default()
            .with_checksum(checksum)
            .with_signature_verification(signature_verified)
            .with_key_fingerprint(key_fingerprint);
        self
    }

    pub fn with_binary_info_detailed(mut self, binary_info: BinaryInfo) -> Self {
        self.report.binary_info = binary_info;
        self
    }

    pub fn with_gpu_info(mut self, gpu_info: Vec<GpuInfo>) -> Self {
        self.report.gpu_info = gpu_info;
        self
    }

    pub fn with_system_info(mut self, system_info: SystemInfo) -> Self {
        self.report.system_info = system_info;
        self
    }

    pub fn with_network_benchmark(mut self, network_benchmark: NetworkBenchmarkResults) -> Self {
        self.report.network_benchmark = network_benchmark;
        self
    }

    pub fn with_ipinfo(mut self, ipinfo: IpInfo) -> Self {
        self.report.ipinfo = Some(ipinfo);
        self
    }

    pub fn with_vdf_proof(mut self, vdf_proof: VdfProof) -> Self {
        self.report.vdf_proof = Some(vdf_proof);
        self
    }

    pub fn with_model_fingerprint(mut self, model_fingerprint: ModelFingerprint) -> Self {
        self.report.model_fingerprint = Some(model_fingerprint);
        self
    }

    pub fn with_semantic_similarity(
        mut self,
        semantic_similarity: SemanticSimilarityResult,
    ) -> Self {
        self.report.semantic_similarity = Some(semantic_similarity);
        self
    }

    pub fn with_os_attestation(mut self, os_attestation: OsAttestation) -> Self {
        self.report.os_attestation = Some(os_attestation);
        self
    }

    pub fn with_docker_attestation(mut self, docker_attestation: DockerAttestation) -> Self {
        self.report.docker_attestation = Some(docker_attestation);
        self
    }

    pub fn with_validator_nonce(mut self, nonce: String) -> Self {
        self.report.validator_nonce = Some(nonce);
        self
    }

    pub fn with_gpu_benchmarks(mut self, gpu_benchmarks: GpuBenchmarkResults) -> Self {
        self.report.gpu_benchmarks = Some(gpu_benchmarks);
        self
    }

    /// Validate hardware performance matches claimed specifications
    pub fn validate_hardware_performance(self) -> Result<Self, String> {
        let validator = PerformanceValidator::new();

        for gpu in &self.report.gpu_info {
            match validator.validate_gpu(gpu) {
                Ok(result) => {
                    if !result.is_valid {
                        return Err(format!(
                            "GPU {} failed performance validation: confidence score {}",
                            gpu.name, result.confidence_score
                        ));
                    }
                }
                Err(e) => {
                    return Err(format!("Failed to validate GPU {}: {}", gpu.name, e));
                }
            }
        }

        Ok(self)
    }

    /// Check for virtualization
    pub fn check_virtualization(self) -> Result<Self, String> {
        let detector = VirtualizationDetector::new();

        match detector.detect_virtualization() {
            Ok(status) => {
                if status.is_virtualized && status.confidence > 0.7 {
                    return Err(format!(
                        "Virtualization detected with {:.0}% confidence: {:?} {:?}",
                        status.confidence * 100.0,
                        status.hypervisor,
                        status.container
                    ));
                }
            }
            Err(e) => {
                return Err(format!("Failed to check virtualization: {e}"));
            }
        }

        Ok(self)
    }

    pub fn with_version(mut self, version: String) -> Self {
        self.report.version = version;
        self
    }

    pub fn with_timestamp(mut self, timestamp: chrono::DateTime<Utc>) -> Self {
        self.report.timestamp = timestamp;
        self
    }

    pub fn build(self) -> AttestationReport {
        self.report
    }
}

impl AttestationBuilder {
    /// Create a comprehensive attestation with all available system information
    pub fn comprehensive(executor_id: String) -> Self {
        Self::new(executor_id)
            .with_default_system_info()
            .with_default_network_benchmark()
    }

    /// Create a minimal attestation with only required fields
    pub fn minimal(executor_id: String, gpu_info: Vec<GpuInfo>) -> Self {
        Self::new(executor_id).with_gpu_info(gpu_info)
    }

    /// Add default system info (useful for testing)
    pub fn with_default_system_info(mut self) -> Self {
        self.report.system_info = SystemInfo {
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
        self
    }

    /// Add default network benchmark (useful for testing)
    pub fn with_default_network_benchmark(mut self) -> Self {
        self.report.network_benchmark = NetworkBenchmarkResults::default();
        self
    }

    /// Add model verification components
    pub fn with_model_verification(
        self,
        provider_id: String,
        model_hash: String,
        expected_fingerprint: Vec<u8>,
        observed_fingerprint: Vec<u8>,
    ) -> Self {
        let fingerprint = ModelFingerprint::new(
            provider_id,
            model_hash,
            expected_fingerprint,
            observed_fingerprint,
        );
        self.with_model_fingerprint(fingerprint)
    }

    /// Add semantic similarity test result
    pub fn with_semantic_test(
        self,
        test_case: String,
        expected_embedding: Vec<f32>,
        actual_embedding: Vec<f32>,
        threshold: f32,
    ) -> Self {
        let similarity = SemanticSimilarityResult::new(
            test_case,
            expected_embedding,
            actual_embedding,
            threshold,
        );
        self.with_semantic_similarity(similarity)
    }

    /// Validate the attestation before building
    pub fn validate(self) -> Result<Self, String> {
        if self.report.executor_id.is_empty() {
            return Err("Executor ID cannot be empty".to_string());
        }

        if self.report.gpu_info.is_empty() {
            return Err("At least one GPU must be specified".to_string());
        }

        if !self.report.binary_info.is_valid() {
            return Err("Binary information is invalid".to_string());
        }

        Ok(self)
    }

    /// Get a preview of the report without consuming the builder
    pub fn preview(&self) -> &AttestationReport {
        &self.report
    }

    /// Clone the current state for branching
    pub fn clone_state(&self) -> Self {
        Self {
            report: self.report.clone(),
        }
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_attestation_builder_basic() {
        let report = AttestationBuilder::new("test_executor_123".to_string())
            .with_binary_info(
                "test_checksum".to_string(),
                true,
                "test_fingerprint".to_string(),
            )
            .build();

        assert_eq!(report.executor_id, "test_executor_123");
        assert!(report.binary_info.signature_verified);
        assert_eq!(report.binary_info.checksum, "test_checksum");
    }

    #[test]
    fn test_comprehensive_builder() {
        let report = AttestationBuilder::comprehensive("test_executor".to_string()).build();

        assert_eq!(report.executor_id, "test_executor");
        assert!(!report.system_info.cpu.brand.is_empty());
        assert!(!report
            .network_benchmark
            .dns_resolution_test
            .hostname
            .is_empty());
    }

    #[test]
    fn test_minimal_builder() {
        let gpu_info = vec![GpuInfo::new(
            crate::gpu::GpuVendor::Unknown,
            "Test GPU".to_string(),
            "1.0.0".to_string(),
        )];
        let report =
            AttestationBuilder::minimal("test_executor".to_string(), gpu_info.clone()).build();

        assert_eq!(report.executor_id, "test_executor");
        assert_eq!(report.gpu_info.len(), 1);
    }

    #[test]
    fn test_model_verification_builder() {
        let expected = vec![1, 2, 3, 4];
        let observed = vec![1, 2, 3, 4];

        let report = AttestationBuilder::new("test_executor".to_string())
            .with_model_verification(
                "test_provider".to_string(),
                "test_hash".to_string(),
                expected.clone(),
                observed.clone(),
            )
            .build();

        assert!(report.has_model_fingerprint());
        let fingerprint = report.model_fingerprint.unwrap();
        assert!(fingerprint.fingerprint_verified);
    }

    #[test]
    fn test_semantic_test_builder() {
        let embedding_a = vec![1.0, 0.0, 0.0];
        let embedding_b = vec![1.0, 0.0, 0.0];

        let report = AttestationBuilder::new("test_executor".to_string())
            .with_semantic_test("test_case".to_string(), embedding_a, embedding_b, 0.8)
            .build();

        assert!(report.has_semantic_similarity());
        let similarity = report.semantic_similarity.unwrap();
        assert!(similarity.passed);
    }

    #[test]
    fn test_builder_validation() {
        // Should fail with empty executor ID
        let result = AttestationBuilder::new("".to_string()).validate();
        assert!(result.is_err());

        // Should fail with no GPU info
        let result = AttestationBuilder::new("test".to_string()).validate();
        assert!(result.is_err());

        // Should pass with valid data
        let gpu_info = vec![GpuInfo::new(
            crate::gpu::GpuVendor::Unknown,
            "Test GPU".to_string(),
            "1.0.0".to_string(),
        )];
        let result = AttestationBuilder::new("test".to_string())
            .with_gpu_info(gpu_info)
            .with_binary_info("checksum".to_string(), true, "fingerprint".to_string())
            .validate();
        assert!(result.is_ok());
    }

    #[test]
    fn test_builder_preview() {
        let builder = AttestationBuilder::new("test_executor".to_string());
        let preview = builder.preview();

        assert_eq!(preview.executor_id, "test_executor");
        assert!(preview.gpu_info.is_empty());
    }

    #[test]
    fn test_builder_clone_state() {
        let original = AttestationBuilder::new("test_executor".to_string());
        let cloned = original.clone_state();

        assert_eq!(original.report.executor_id, cloned.report.executor_id);
    }

    #[test]
    fn test_builder_with_version_and_timestamp() {
        let custom_time = Utc::now();
        let report = AttestationBuilder::new("test_executor".to_string())
            .with_version("2.0.0".to_string())
            .with_timestamp(custom_time)
            .build();

        assert_eq!(report.version, "2.0.0");
        assert_eq!(report.timestamp, custom_time);
    }
}
