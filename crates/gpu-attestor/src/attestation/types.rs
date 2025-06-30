//! Attestation data structures and type definitions

use chrono::{DateTime, Utc};
use serde::{Deserialize, Serialize};

use crate::docker::DockerAttestation;
use crate::gpu::GpuInfo;
use crate::hardware::SystemInfo;
use crate::network::{IpInfo, NetworkBenchmarkResults};
use crate::os::OsAttestation;
use crate::vdf::VdfProof;

#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct AttestationReport {
    pub version: String,
    pub timestamp: DateTime<Utc>,
    pub executor_id: String,
    pub binary_info: BinaryInfo,
    pub gpu_info: Vec<GpuInfo>,
    pub system_info: SystemInfo,
    pub network_benchmark: NetworkBenchmarkResults,
    #[serde(skip_serializing_if = "Option::is_none")]
    pub ipinfo: Option<IpInfo>,
    pub vdf_proof: Option<VdfProof>,
    #[serde(skip_serializing_if = "Option::is_none")]
    pub os_attestation: Option<OsAttestation>,
    #[serde(skip_serializing_if = "Option::is_none")]
    pub docker_attestation: Option<DockerAttestation>,
    pub model_fingerprint: Option<ModelFingerprint>,
    pub semantic_similarity: Option<SemanticSimilarityResult>,
    #[serde(skip_serializing_if = "Option::is_none")]
    pub validation_results: Option<ValidationResults>,
    /// GPU benchmark results including CUDA Driver API tests
    #[serde(skip_serializing_if = "Option::is_none")]
    pub gpu_benchmarks: Option<GpuBenchmarkResults>,
    /// Validator-provided nonce to prevent replay attacks
    #[serde(skip_serializing_if = "Option::is_none")]
    pub validator_nonce: Option<String>,
}

#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct GpuBenchmarkResults {
    /// Results from individual GPU benchmarks
    pub gpu_results: Vec<SingleGpuBenchmark>,
    /// Total time taken to run all GPU benchmarks
    pub total_benchmark_time_ms: f64,
}

#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct SingleGpuBenchmark {
    /// GPU index
    pub gpu_index: u32,
    /// GPU name
    pub gpu_name: String,
    /// Memory bandwidth test result (GB/s)
    #[serde(skip_serializing_if = "Option::is_none")]
    pub memory_bandwidth_gbps: Option<f64>,
    /// FP16 compute performance (TFLOPS)
    #[serde(skip_serializing_if = "Option::is_none")]
    pub fp16_tflops: Option<f64>,
    /// CUDA Driver API matrix multiplication results
    #[serde(skip_serializing_if = "Option::is_none")]
    pub cuda_driver_results: Option<CudaDriverBenchmarkResults>,
    /// Backend used for benchmarking
    pub backend: String,
    /// Any errors encountered during benchmarking
    #[serde(skip_serializing_if = "Option::is_none")]
    pub error: Option<String>,
}

#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct CudaDriverBenchmarkResults {
    /// Matrix multiplication results for different sizes
    pub matrix_results: Vec<MatrixMultiplicationResult>,
    /// Whether PTX kernel was loaded successfully
    pub ptx_loaded: bool,
    /// CUDA Driver version detected
    #[serde(skip_serializing_if = "Option::is_none")]
    pub cuda_driver_version: Option<String>,
}

#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct MatrixMultiplicationResult {
    /// Matrix dimensions (N x N)
    pub matrix_size: u32,
    /// Execution time in milliseconds
    pub execution_time_ms: f64,
    /// Performance in GFLOPS
    pub gflops: f64,
    /// Matrix checksum for verification
    #[serde(skip_serializing_if = "Option::is_none")]
    pub checksum: Option<String>,
}

#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct BinaryInfo {
    pub path: String,
    pub checksum: String,
    pub signature_verified: bool,
    pub compilation_timestamp: DateTime<Utc>,
    pub validator_public_key_fingerprint: String,
}

#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct TmpAttestation {
    pub pcr_values: Vec<PcrValue>,
    pub quote: Vec<u8>,
    pub quote_signature: Vec<u8>,
    pub attestation_key_cert: Vec<u8>,
}

#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct PcrValue {
    pub index: u8,
    pub value: Vec<u8>,
    pub algorithm: String,
}

#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct ModelFingerprint {
    pub provider_id: String,
    pub model_hash: String,
    pub fingerprint_verified: bool,
    pub expected_fingerprint: Vec<u8>,
    pub observed_fingerprint: Vec<u8>,
    pub watermark_validation: bool,
}

#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct SemanticSimilarityResult {
    pub test_case: String,
    pub expected_output_embedding: Vec<f32>,
    pub actual_output_embedding: Vec<f32>,
    pub cosine_similarity: f32,
    pub threshold: f32,
    pub passed: bool,
}

#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct SignedAttestation {
    pub report: AttestationReport,
    pub signature: Vec<u8>,
    pub ephemeral_public_key: Vec<u8>,
    pub key_rotation_timestamp: DateTime<Utc>,
}

impl AttestationReport {
    pub fn new(executor_id: String) -> Self {
        Self {
            version: env!("CARGO_PKG_VERSION").to_string(),
            timestamp: Utc::now(),
            executor_id,
            binary_info: BinaryInfo::default(),
            gpu_info: Vec::new(),
            system_info: SystemInfo {
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
            },
            network_benchmark: NetworkBenchmarkResults::default(),
            ipinfo: None,
            vdf_proof: None,
            os_attestation: None,
            docker_attestation: None,
            model_fingerprint: None,
            semantic_similarity: None,
            validation_results: None,
            gpu_benchmarks: None,
            validator_nonce: None,
        }
    }

    pub fn is_valid(&self) -> bool {
        !self.executor_id.is_empty() && !self.gpu_info.is_empty() && self.is_recent()
    }

    pub fn is_recent(&self) -> bool {
        let now = Utc::now();
        let age = now.signed_duration_since(self.timestamp);
        age.num_minutes() <= 60
    }

    pub fn has_vdf_proof(&self) -> bool {
        self.vdf_proof.is_some()
    }

    pub fn has_model_fingerprint(&self) -> bool {
        self.model_fingerprint.is_some()
    }

    pub fn has_semantic_similarity(&self) -> bool {
        self.semantic_similarity.is_some()
    }

    pub fn has_os_attestation(&self) -> bool {
        self.os_attestation.is_some()
    }

    pub fn has_docker_attestation(&self) -> bool {
        self.docker_attestation.is_some()
    }

    pub fn get_age_minutes(&self) -> i64 {
        let now = Utc::now();
        now.signed_duration_since(self.timestamp).num_minutes()
    }
}

impl BinaryInfo {
    pub fn new(path: String) -> Self {
        Self {
            path,
            checksum: String::new(),
            signature_verified: false,
            compilation_timestamp: Utc::now(),
            validator_public_key_fingerprint: String::new(),
        }
    }

    pub fn with_checksum(mut self, checksum: String) -> Self {
        self.checksum = checksum;
        self
    }

    pub fn with_signature_verification(mut self, verified: bool) -> Self {
        self.signature_verified = verified;
        self
    }

    pub fn with_key_fingerprint(mut self, fingerprint: String) -> Self {
        self.validator_public_key_fingerprint = fingerprint;
        self
    }

    pub fn is_valid(&self) -> bool {
        !self.path.is_empty() && !self.checksum.is_empty()
    }
}

impl Default for BinaryInfo {
    fn default() -> Self {
        Self::new(
            std::env::current_exe()
                .map(|p| p.to_string_lossy().to_string())
                .unwrap_or_else(|_| "unknown".to_string()),
        )
    }
}

impl ModelFingerprint {
    pub fn new(
        provider_id: String,
        model_hash: String,
        expected_fingerprint: Vec<u8>,
        observed_fingerprint: Vec<u8>,
    ) -> Self {
        let fingerprint_verified = expected_fingerprint == observed_fingerprint;
        Self {
            provider_id,
            model_hash,
            fingerprint_verified,
            expected_fingerprint,
            observed_fingerprint,
            watermark_validation: false,
        }
    }

    pub fn with_watermark_validation(mut self, validated: bool) -> Self {
        self.watermark_validation = validated;
        self
    }

    pub fn is_valid(&self) -> bool {
        self.fingerprint_verified && !self.provider_id.is_empty() && !self.model_hash.is_empty()
    }
}

impl SemanticSimilarityResult {
    pub fn new(
        test_case: String,
        expected_embedding: Vec<f32>,
        actual_embedding: Vec<f32>,
        threshold: f32,
    ) -> Self {
        let cosine_similarity =
            Self::calculate_cosine_similarity(&expected_embedding, &actual_embedding);
        let passed = cosine_similarity >= threshold;

        Self {
            test_case,
            expected_output_embedding: expected_embedding,
            actual_output_embedding: actual_embedding,
            cosine_similarity,
            threshold,
            passed,
        }
    }

    pub fn is_valid(&self) -> bool {
        self.passed && !self.test_case.is_empty()
    }

    fn calculate_cosine_similarity(a: &[f32], b: &[f32]) -> f32 {
        if a.len() != b.len() || a.is_empty() {
            return 0.0;
        }

        let dot_product: f32 = a.iter().zip(b.iter()).map(|(x, y)| x * y).sum();
        let norm_a: f32 = a.iter().map(|x| x * x).sum::<f32>().sqrt();
        let norm_b: f32 = b.iter().map(|x| x * x).sum::<f32>().sqrt();

        if norm_a == 0.0 || norm_b == 0.0 {
            return 0.0;
        }

        dot_product / (norm_a * norm_b)
    }
}

impl SignedAttestation {
    pub fn new(
        report: AttestationReport,
        signature: Vec<u8>,
        ephemeral_public_key: Vec<u8>,
    ) -> Self {
        Self {
            report,
            signature,
            ephemeral_public_key,
            key_rotation_timestamp: Utc::now(),
        }
    }

    pub fn is_valid(&self) -> bool {
        self.report.is_valid()
            && !self.signature.is_empty()
            && !self.ephemeral_public_key.is_empty()
    }

    pub fn is_recent(&self) -> bool {
        self.report.is_recent()
    }

    pub fn get_age_minutes(&self) -> i64 {
        self.report.get_age_minutes()
    }
}

impl PcrValue {
    pub fn new(index: u8, value: Vec<u8>, algorithm: String) -> Self {
        Self {
            index,
            value,
            algorithm,
        }
    }

    pub fn is_valid(&self) -> bool {
        !self.value.is_empty() && !self.algorithm.is_empty()
    }
}

impl TmpAttestation {
    pub fn new(
        pcr_values: Vec<PcrValue>,
        quote: Vec<u8>,
        quote_signature: Vec<u8>,
        attestation_key_cert: Vec<u8>,
    ) -> Self {
        Self {
            pcr_values,
            quote,
            quote_signature,
            attestation_key_cert,
        }
    }

    pub fn is_valid(&self) -> bool {
        !self.pcr_values.is_empty()
            && !self.quote.is_empty()
            && !self.quote_signature.is_empty()
            && !self.attestation_key_cert.is_empty()
            && self.pcr_values.iter().all(|pcr| pcr.is_valid())
    }
}

#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct ValidationResults {
    pub hardware_performance_valid: bool,
    pub virtualization_detected: bool,
    pub replay_protection_valid: bool,
    pub performance_confidence: f64,
    pub virtualization_confidence: f64,
    pub validation_timestamp: DateTime<Utc>,
    pub validation_details: ValidationDetails,
}

#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct ValidationDetails {
    pub gpu_performance_results: Vec<GpuPerformanceResult>,
    pub virtualization_indicators: Vec<String>,
    pub challenge_response_time_ms: u64,
}

#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct GpuPerformanceResult {
    pub gpu_name: String,
    pub memory_bandwidth_valid: bool,
    pub compute_performance_valid: bool,
    pub measured_tflops: f64,
    pub expected_tflops: f64,
}

impl ValidationResults {
    pub fn new(
        hardware_valid: bool,
        virt_detected: bool,
        replay_valid: bool,
        perf_confidence: f64,
        virt_confidence: f64,
    ) -> Self {
        Self {
            hardware_performance_valid: hardware_valid,
            virtualization_detected: virt_detected,
            replay_protection_valid: replay_valid,
            performance_confidence: perf_confidence,
            virtualization_confidence: virt_confidence,
            validation_timestamp: Utc::now(),
            validation_details: ValidationDetails {
                gpu_performance_results: Vec::new(),
                virtualization_indicators: Vec::new(),
                challenge_response_time_ms: 0,
            },
        }
    }

    pub fn is_valid(&self) -> bool {
        self.hardware_performance_valid
            && !self.virtualization_detected
            && self.replay_protection_valid
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_attestation_report_creation() {
        let report = AttestationReport::new("test_executor_123".to_string());
        assert_eq!(report.executor_id, "test_executor_123");
        assert!(report.is_recent());
        assert!(!report.is_valid()); // Invalid because no GPU info
    }

    #[test]
    fn test_binary_info_builder() {
        let binary_info = BinaryInfo::new("test_path".to_string())
            .with_checksum("test_checksum".to_string())
            .with_signature_verification(true)
            .with_key_fingerprint("test_fingerprint".to_string());

        assert!(binary_info.is_valid());
        assert!(binary_info.signature_verified);
        assert_eq!(binary_info.checksum, "test_checksum");
    }

    #[test]
    fn test_model_fingerprint() {
        let expected = vec![1, 2, 3, 4];
        let observed = vec![1, 2, 3, 4];

        let fingerprint = ModelFingerprint::new(
            "test_provider".to_string(),
            "test_hash".to_string(),
            expected.clone(),
            observed.clone(),
        );

        assert!(fingerprint.is_valid());
        assert!(fingerprint.fingerprint_verified);
    }

    #[test]
    fn test_semantic_similarity() {
        let embedding_a = vec![1.0, 0.0, 0.0];
        let embedding_b = vec![1.0, 0.0, 0.0];

        let similarity =
            SemanticSimilarityResult::new("test_case".to_string(), embedding_a, embedding_b, 0.8);

        assert!(similarity.is_valid());
        assert!(similarity.passed);
        assert!((similarity.cosine_similarity - 1.0).abs() < 0.001);
    }

    #[test]
    fn test_cosine_similarity_calculation() {
        let a = vec![1.0, 0.0, 0.0];
        let b = vec![0.0, 1.0, 0.0];

        let similarity = SemanticSimilarityResult::calculate_cosine_similarity(&a, &b);
        assert!((similarity - 0.0).abs() < 0.001);

        let a = vec![1.0, 1.0];
        let b = vec![1.0, 1.0];

        let similarity = SemanticSimilarityResult::calculate_cosine_similarity(&a, &b);
        assert!((similarity - 1.0).abs() < 0.001);
    }

    #[test]
    fn test_pcr_value() {
        let pcr = PcrValue::new(0, vec![1, 2, 3, 4], "SHA256".to_string());
        assert!(pcr.is_valid());
        assert_eq!(pcr.index, 0);
        assert_eq!(pcr.algorithm, "SHA256");
    }

    #[test]
    fn test_signed_attestation() {
        let report = AttestationReport::new("test_executor".to_string());
        let signed = SignedAttestation::new(report, vec![1, 2, 3, 4], vec![5, 6, 7, 8]);

        assert!(!signed.is_valid()); // Invalid because report has no GPU info
        assert!(signed.is_recent());
        assert!(!signed.signature.is_empty());
    }
}
