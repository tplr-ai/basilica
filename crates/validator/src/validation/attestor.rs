//! Integration with gpu-attestor for hardware verification
//!
//! Provides integration with the gpu-attestor binary for parsing attestation reports
//! and converting them to internal hardware specification formats.
//!
//! NOTE: This module does NOT execute gpu-attestor directly. The actual execution
//! happens remotely on executor machines via SSH (see validator.rs).

use super::types::{
    AttestationReport, CpuInfo, DockerStatus, GpuInfo, HardwareSpecs, MemoryInfo, NetworkInfo,
    StorageInfo, ValidationError, ValidationResult,
};

#[cfg(test)]
use super::types::{
    CpuAttestorInfo, DockerAttestorInfo, GpuAttestorGpu, MemoryAttestorInfo, SystemInfo,
};
use std::path::Path;
use tokio::fs;
use tracing::{debug, info, warn};

/// GPU attestor integration wrapper for parsing and validation
///
/// This handles local operations like file parsing and data conversion.
/// Remote execution is handled by the HardwareValidator via SSH.
#[derive(Debug)]
pub struct GpuAttestorIntegration {
    /// Path to local gpu-attestor binary (for verification and upload)
    binary_path: String,
}

impl GpuAttestorIntegration {
    /// Create new gpu-attestor integration
    pub fn new(binary_path: String) -> Self {
        Self { binary_path }
    }

    /// Check if local gpu-attestor binary exists and is executable
    ///
    /// This verifies the binary is available for upload to remote executors.
    pub async fn verify_local_binary_availability(&self) -> ValidationResult<bool> {
        info!(
            "Verifying local gpu-attestor binary availability at: {}",
            self.binary_path
        );

        let path = Path::new(&self.binary_path);
        if !path.exists() {
            warn!("gpu-attestor binary not found at: {}", self.binary_path);
            return Ok(false);
        }

        // Check if file is executable (Unix-specific)
        #[cfg(unix)]
        {
            use std::os::unix::fs::PermissionsExt;
            let metadata = fs::metadata(path).await.map_err(ValidationError::IoError)?;
            let is_executable = metadata.permissions().mode() & 0o111 != 0;

            if !is_executable {
                warn!(
                    "gpu-attestor binary is not executable: {}",
                    self.binary_path
                );
                return Ok(false);
            }
        }

        debug!("Local gpu-attestor binary verified successfully");
        Ok(true)
    }

    /// Get the path to the local gpu-attestor binary for upload
    pub fn get_binary_path(&self) -> &str {
        &self.binary_path
    }

    /// Generate expected attestation file paths for a given output directory
    ///
    /// This is used by the validator to know which files to download from remote executors.
    pub fn get_expected_attestation_files(&self, output_dir: &str) -> (String, String, String) {
        (
            format!("{output_dir}/attestation.json"),
            format!("{output_dir}/attestation.sig"),
            format!("{output_dir}/attestation.pub"),
        )
    }

    /// Parse attestation JSON file
    pub async fn parse_attestation_file(
        &self,
        file_path: &str,
    ) -> ValidationResult<AttestationReport> {
        debug!("Parsing attestation file: {}", file_path);

        let content = fs::read_to_string(file_path)
            .await
            .map_err(ValidationError::IoError)?;

        let report: AttestationReport =
            serde_json::from_str(&content).map_err(ValidationError::SerializationError)?;

        info!(
            "Successfully parsed attestation report for executor: {}",
            report.executor_id
        );
        Ok(report)
    }

    /// Convert gpu-attestor report to internal HardwareSpecs format
    pub fn convert_to_hardware_specs(
        &self,
        report: &AttestationReport,
    ) -> ValidationResult<HardwareSpecs> {
        debug!("Converting attestation report to HardwareSpecs");

        // Convert CPU info
        let cpu = CpuInfo {
            model: report.system_info.cpu.brand.clone(),
            cores: report.system_info.cpu.cores,
            frequency_mhz: 0, // Not available in current CpuAttestorInfo
            architecture: "unknown".to_string(), // Not available in current CpuAttestorInfo
        };

        // Convert GPU info
        let gpu: Vec<GpuInfo> = report
            .gpu_info
            .iter()
            .map(|gpu_attestor_gpu| {
                GpuInfo {
                    vendor: gpu_attestor_gpu.vendor.clone(),
                    model: gpu_attestor_gpu.name.clone(),
                    vram_mb: gpu_attestor_gpu.memory_total / (1024 * 1024), // Convert bytes to MB
                    driver_version: gpu_attestor_gpu.driver_version.clone(),
                    compute_capability: None, // Could be inferred from model in the future
                    utilization_percent: gpu_attestor_gpu.utilization,
                }
            })
            .collect();

        // Convert memory info
        let memory = MemoryInfo {
            total_mb: report.system_info.memory.total_bytes / (1024 * 1024), // Convert bytes to MB
            available_mb: 0, // Not available in current MemoryAttestorInfo
            memory_type: "unknown".to_string(), // Not available in current MemoryAttestorInfo
        };

        // Convert storage info (basic implementation)
        let storage = StorageInfo {
            total_gb: 0,
            available_gb: 0,
            storage_type: "unknown".to_string(),
            io_performance: None,
        };

        // Convert network info
        let network = if let Some(network_bench) = &report.network_benchmark {
            // Calculate average latency and throughput
            let avg_latency = if !network_bench.latency_tests.is_empty() {
                network_bench
                    .latency_tests
                    .iter()
                    .map(|test| test.avg_latency_ms)
                    .sum::<f64>()
                    / network_bench.latency_tests.len() as f64
            } else {
                0.0
            };

            let avg_throughput = if !network_bench.throughput_tests.is_empty() {
                network_bench
                    .throughput_tests
                    .iter()
                    .map(|test| test.throughput_mbps)
                    .sum::<f64>()
                    / network_bench.throughput_tests.len() as f64
            } else {
                0.0
            };

            let avg_packet_loss = if !network_bench.latency_tests.is_empty() {
                network_bench
                    .latency_tests
                    .iter()
                    .map(|test| test.packet_loss_percent)
                    .sum::<f32>()
                    / network_bench.latency_tests.len() as f32
            } else {
                0.0
            };

            NetworkInfo {
                bandwidth_mbps: avg_throughput,
                latency_ms: avg_latency,
                packet_loss_percent: avg_packet_loss,
            }
        } else {
            NetworkInfo {
                bandwidth_mbps: 0.0,
                latency_ms: 0.0,
                packet_loss_percent: 0.0,
            }
        };

        // Convert Docker status
        let docker_status = DockerStatus {
            is_installed: !report.system_info.docker.version.is_empty(),
            version: report.system_info.docker.version.clone(),
            daemon_running: report.system_info.docker.is_running,
            nvidia_runtime_available: false, // Would need to be checked separately
            available_images: vec![],        // Not provided by current gpu-attestor types
        };

        Ok(HardwareSpecs {
            cpu,
            gpu,
            memory,
            storage,
            network,
            docker_status,
        })
    }

    /// Validate attestation report structure
    pub fn validate_attestation_structure(
        &self,
        report: &AttestationReport,
    ) -> ValidationResult<bool> {
        debug!("Validating attestation report structure");

        // Check required fields
        if report.executor_id.is_empty() {
            return Ok(false);
        }

        if report.timestamp.is_empty() {
            return Ok(false);
        }

        if report.gpu_info.is_empty() {
            warn!("No GPU information found in attestation report");
            return Ok(false);
        }

        // Validate GPU info
        for gpu in &report.gpu_info {
            if gpu.vendor.is_empty() || gpu.name.is_empty() {
                warn!("Invalid GPU information: missing vendor or name");
                return Ok(false);
            }
        }

        // Validate system info
        if report.system_info.cpu.cores == 0 {
            warn!("Invalid CPU information: zero cores");
            return Ok(false);
        }

        if report.system_info.memory.total_bytes == 0 {
            warn!("Invalid memory information: zero total memory");
            return Ok(false);
        }

        info!("Attestation report structure validation passed");
        Ok(true)
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use tempfile::TempDir;

    #[tokio::test]
    async fn test_gpu_attestor_integration_creation() {
        let integration = GpuAttestorIntegration::new("/path/to/gpu-attestor".to_string());
        assert_eq!(integration.get_binary_path(), "/path/to/gpu-attestor");
    }

    #[tokio::test]
    async fn test_local_binary_availability_nonexistent() {
        let integration = GpuAttestorIntegration::new("/nonexistent/path".to_string());
        let available = integration
            .verify_local_binary_availability()
            .await
            .unwrap();
        assert!(!available);
    }

    #[test]
    fn test_get_expected_attestation_files() {
        let integration = GpuAttestorIntegration::new("/fake/path".to_string());
        let (json_file, sig_file, pub_file) =
            integration.get_expected_attestation_files("/tmp/test");

        assert_eq!(json_file, "/tmp/test/attestation.json");
        assert_eq!(sig_file, "/tmp/test/attestation.sig");
        assert_eq!(pub_file, "/tmp/test/attestation.pub");
    }

    #[tokio::test]
    async fn test_parse_attestation_file() {
        let temp_dir = TempDir::new().unwrap();
        let attestation_path = temp_dir.path().join("attestation.json");

        let sample_report = AttestationReport {
            version: "1.0.0".to_string(),
            timestamp: "2024-06-17T14:00:00Z".to_string(),
            executor_id: "test-executor".to_string(),
            binary_info: None,
            gpu_info: vec![GpuAttestorGpu {
                vendor: "NVIDIA".to_string(),
                name: "RTX 4090".to_string(),
                memory_total: 25769803776,
                driver_version: "525.60.11".to_string(),
                temperature: Some(45.0),
                utilization: Some(0.0),
            }],
            system_info: SystemInfo {
                cpu: CpuAttestorInfo {
                    cores: 16,
                    threads: 32,
                    brand: "AMD Ryzen 9 5950X".to_string(),
                },
                memory: MemoryAttestorInfo {
                    total_bytes: 68719476736,
                },
                docker: DockerAttestorInfo {
                    is_running: true,
                    version: "24.0.2".to_string(),
                },
            },
            network_benchmark: None,
            vdf_proof: None,
            validator_nonce: None,
        };

        // Write sample report to file
        let content = serde_json::to_string_pretty(&sample_report).unwrap();
        std::fs::write(&attestation_path, content).unwrap();

        let integration = GpuAttestorIntegration::new("/fake/path".to_string());
        let parsed_report = integration
            .parse_attestation_file(attestation_path.to_str().unwrap())
            .await
            .unwrap();

        assert_eq!(parsed_report.executor_id, "test-executor");
        assert_eq!(parsed_report.gpu_info.len(), 1);
        assert_eq!(parsed_report.gpu_info[0].vendor, "NVIDIA");
    }

    #[test]
    fn test_convert_to_hardware_specs() {
        let integration = GpuAttestorIntegration::new("/fake/path".to_string());

        let report = AttestationReport {
            version: "1.0.0".to_string(),
            timestamp: "2024-06-17T14:00:00Z".to_string(),
            executor_id: "test-executor".to_string(),
            binary_info: None,
            gpu_info: vec![GpuAttestorGpu {
                vendor: "NVIDIA".to_string(),
                name: "RTX 4090".to_string(),
                memory_total: 25769803776, // 24 GB
                driver_version: "525.60.11".to_string(),
                temperature: Some(45.0),
                utilization: Some(0.0),
            }],
            system_info: SystemInfo {
                cpu: CpuAttestorInfo {
                    cores: 16,
                    threads: 32,
                    brand: "AMD Ryzen 9 5950X".to_string(),
                },
                memory: MemoryAttestorInfo {
                    total_bytes: 68719476736, // 64 GB
                },
                docker: DockerAttestorInfo {
                    is_running: true,
                    version: "24.0.2".to_string(),
                },
            },
            network_benchmark: None,
            vdf_proof: None,
            validator_nonce: None,
        };

        let hardware_specs = integration.convert_to_hardware_specs(&report).unwrap();

        assert_eq!(hardware_specs.cpu.cores, 16);
        assert_eq!(hardware_specs.cpu.model, "AMD Ryzen 9 5950X");
        assert_eq!(hardware_specs.gpu.len(), 1);
        assert_eq!(hardware_specs.gpu[0].vendor, "NVIDIA");
        assert_eq!(hardware_specs.gpu[0].vram_mb, 24576); // 24 GB in MB
        assert_eq!(hardware_specs.memory.total_mb, 65536); // 64 GB in MB
        assert!(hardware_specs.docker_status.daemon_running);
    }

    #[test]
    fn test_validate_attestation_structure() {
        let integration = GpuAttestorIntegration::new("/fake/path".to_string());

        let valid_report = AttestationReport {
            version: "1.0.0".to_string(),
            timestamp: "2024-06-17T14:00:00Z".to_string(),
            executor_id: "test-executor".to_string(),
            binary_info: None,
            gpu_info: vec![GpuAttestorGpu {
                vendor: "NVIDIA".to_string(),
                name: "RTX 4090".to_string(),
                memory_total: 25769803776,
                driver_version: "525.60.11".to_string(),
                temperature: Some(45.0),
                utilization: Some(0.0),
            }],
            system_info: SystemInfo {
                cpu: CpuAttestorInfo {
                    cores: 16,
                    threads: 32,
                    brand: "AMD Ryzen 9 5950X".to_string(),
                },
                memory: MemoryAttestorInfo {
                    total_bytes: 68719476736,
                },
                docker: DockerAttestorInfo {
                    is_running: true,
                    version: "24.0.2".to_string(),
                },
            },
            network_benchmark: None,
            vdf_proof: None,
            validator_nonce: None,
        };

        assert!(integration
            .validate_attestation_structure(&valid_report)
            .unwrap());

        // Test invalid report (empty executor ID)
        let mut invalid_report = valid_report.clone();
        invalid_report.executor_id = "".to_string();
        assert!(!integration
            .validate_attestation_structure(&invalid_report)
            .unwrap());
    }
}
