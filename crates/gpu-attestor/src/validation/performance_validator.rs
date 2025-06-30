//! Performance validation to detect hardware spoofing
//!
//! This module validates that claimed hardware specifications match actual performance
//! characteristics, preventing NVML spoofing attacks.

use anyhow::{Context, Result};
use std::collections::HashMap;
use std::time::{Duration, Instant};

use crate::gpu::types::GpuInfo;

#[derive(Debug, Clone)]
pub struct HardwareProfile {
    pub gpu_model: String,
    pub expected_memory_gb: u64,
    pub expected_memory_bandwidth_gbps: f64,
    pub expected_fp16_tflops: f64,
    pub expected_fp32_tflops: f64,
    pub tolerance_percent: f64,
}

pub struct PerformanceValidator {
    profiles: HashMap<String, HardwareProfile>,
}

impl Default for PerformanceValidator {
    fn default() -> Self {
        Self::new()
    }
}

impl PerformanceValidator {
    pub fn new() -> Self {
        let mut profiles = HashMap::new();

        // NVIDIA H200 Profile
        profiles.insert(
            "H200".to_string(),
            HardwareProfile {
                gpu_model: "NVIDIA H200".to_string(),
                expected_memory_gb: 141,
                expected_memory_bandwidth_gbps: 4800.0,
                expected_fp16_tflops: 800.0, // Similar to H100 but with more memory bandwidth
                expected_fp32_tflops: 400.0, // Similar to H100
                tolerance_percent: 15.0,     // Increased tolerance for benchmark variance
            },
        );

        // NVIDIA H100 Profile
        profiles.insert(
            "H100".to_string(),
            HardwareProfile {
                gpu_model: "NVIDIA H100".to_string(),
                expected_memory_gb: 80,
                expected_memory_bandwidth_gbps: 3350.0,
                expected_fp16_tflops: 800.0, // Realistic benchmark value for 8K matrix
                expected_fp32_tflops: 400.0, // Realistic benchmark value
                tolerance_percent: 15.0,     // Increased tolerance for benchmark variance
            },
        );

        // NVIDIA A100 Profile
        profiles.insert(
            "A100".to_string(),
            HardwareProfile {
                gpu_model: "NVIDIA A100".to_string(),
                expected_memory_gb: 80,
                expected_memory_bandwidth_gbps: 2039.0,
                expected_fp16_tflops: 312.0,
                expected_fp32_tflops: 156.0,
                tolerance_percent: 10.0,
            },
        );

        // NVIDIA RTX 4090 Profile
        profiles.insert(
            "RTX 4090".to_string(),
            HardwareProfile {
                gpu_model: "GeForce RTX 4090".to_string(),
                expected_memory_gb: 24,
                expected_memory_bandwidth_gbps: 1008.0,
                expected_fp16_tflops: 82.6,
                expected_fp32_tflops: 82.6,
                tolerance_percent: 15.0,
            },
        );

        Self { profiles }
    }

    /// Validate that GPU performance matches claimed specifications
    pub fn validate_gpu(&self, gpu_info: &GpuInfo) -> Result<ValidationResult> {
        // Find matching profile
        let profile = self.find_matching_profile(&gpu_info.name)?;

        // Run performance benchmarks
        let benchmark_results = self.run_gpu_benchmarks(gpu_info)?;

        // Validate memory size
        let memory_valid = self.validate_memory_size(
            gpu_info.memory_total,
            profile.expected_memory_gb * 1024 * 1024 * 1024,
            profile.tolerance_percent,
        );

        // Validate memory bandwidth
        let bandwidth_valid = self.validate_memory_bandwidth(
            benchmark_results.memory_bandwidth_gbps,
            profile.expected_memory_bandwidth_gbps,
            profile.tolerance_percent,
        );

        // Validate compute performance
        let compute_valid = self.validate_compute_performance(
            benchmark_results.fp16_tflops,
            profile.expected_fp16_tflops,
            profile.tolerance_percent,
        );

        let is_valid = memory_valid && bandwidth_valid && compute_valid;

        Ok(ValidationResult {
            is_valid,
            claimed_model: gpu_info.name.clone(),
            detected_profile: profile.gpu_model.clone(),
            memory_validation: MemoryValidation {
                claimed_bytes: gpu_info.memory_total,
                expected_bytes: profile.expected_memory_gb * 1024 * 1024 * 1024,
                is_valid: memory_valid,
            },
            bandwidth_validation: BandwidthValidation {
                measured_gbps: benchmark_results.memory_bandwidth_gbps,
                expected_gbps: profile.expected_memory_bandwidth_gbps,
                is_valid: bandwidth_valid,
            },
            compute_validation: ComputeValidation {
                measured_tflops: benchmark_results.fp16_tflops,
                expected_tflops: profile.expected_fp16_tflops,
                is_valid: compute_valid,
            },
            confidence_score: self.calculate_confidence_score(
                memory_valid,
                bandwidth_valid,
                compute_valid,
            ),
        })
    }

    fn find_matching_profile(&self, gpu_name: &str) -> Result<&HardwareProfile> {
        // Try exact match first
        for (key, profile) in &self.profiles {
            if gpu_name.contains(key) || gpu_name.contains(&profile.gpu_model) {
                return Ok(profile);
            }
        }

        // Try partial match
        let gpu_upper = gpu_name.to_uppercase();
        for (key, profile) in &self.profiles {
            if gpu_upper.contains(&key.to_uppercase()) {
                return Ok(profile);
            }
        }

        anyhow::bail!("No performance profile found for GPU: {}", gpu_name)
    }

    fn run_gpu_benchmarks(&self, gpu_info: &GpuInfo) -> Result<GpuBenchmarkResults> {
        use crate::gpu::benchmarks::GpuBenchmarkRunner;

        // Find GPU index from gpu_info
        let gpu_index = self.find_gpu_index(gpu_info)?;

        // Create benchmark runner
        let runner =
            GpuBenchmarkRunner::new(gpu_index).context("Failed to create GPU benchmark runner")?;

        // Run memory bandwidth benchmark
        tracing::info!("Running memory bandwidth benchmark on GPU {}", gpu_index);
        let memory_bandwidth_gbps = self.benchmark_memory_bandwidth_with_runner(&runner)?;

        // Run FP16 compute benchmark
        tracing::info!("Running FP16 compute benchmark on GPU {}", gpu_index);
        let fp16_tflops = self.benchmark_fp16_compute_with_runner(&runner)?;

        Ok(GpuBenchmarkResults {
            memory_bandwidth_gbps,
            fp16_tflops,
        })
    }

    fn find_gpu_index(&self, gpu_info: &GpuInfo) -> Result<u32> {
        use crate::gpu::GpuDetector;

        let detector = GpuDetector::new();
        let gpus = detector.detect()?;

        // Find matching GPU by name and memory size
        for (index, gpu) in gpus.iter().enumerate() {
            if gpu.name == gpu_info.name && gpu.memory_total == gpu_info.memory_total {
                return Ok(index as u32);
            }
        }

        // If exact match not found, use first GPU with same name
        for (index, gpu) in gpus.iter().enumerate() {
            if gpu.name == gpu_info.name {
                tracing::warn!("Using GPU {} with different memory size", index);
                return Ok(index as u32);
            }
        }

        Err(anyhow::anyhow!("GPU {} not found", gpu_info.name))
    }

    /// Benchmark memory bandwidth on the default GPU (GPU 0)
    pub fn benchmark_memory_bandwidth(&self) -> Result<f64> {
        // Create a default benchmark runner for GPU 0
        use crate::gpu::benchmarks::GpuBenchmarkRunner;

        let runner =
            GpuBenchmarkRunner::new(0).context("Failed to create benchmark runner for GPU 0")?;
        self.benchmark_memory_bandwidth_with_runner(&runner)
    }

    /// Benchmark memory bandwidth on a specific GPU using provided runner
    pub fn benchmark_memory_bandwidth_with_runner(
        &self,
        runner: &crate::gpu::benchmarks::GpuBenchmarkRunner,
    ) -> Result<f64> {
        // Run actual memory bandwidth benchmark
        let bandwidth = runner
            .benchmark_memory_bandwidth()
            .context("Failed to run memory bandwidth benchmark")?;

        tracing::info!("Measured memory bandwidth: {:.2} GB/s", bandwidth);
        Ok(bandwidth)
    }

    /// Benchmark FP16 compute performance on the default GPU (GPU 0)
    pub fn benchmark_fp16_compute(&self) -> Result<f64> {
        // Create a default benchmark runner for GPU 0
        use crate::gpu::benchmarks::GpuBenchmarkRunner;

        let runner =
            GpuBenchmarkRunner::new(0).context("Failed to create benchmark runner for GPU 0")?;
        self.benchmark_fp16_compute_with_runner(&runner)
    }

    /// Benchmark FP16 compute performance on a specific GPU using provided runner
    pub fn benchmark_fp16_compute_with_runner(
        &self,
        runner: &crate::gpu::benchmarks::GpuBenchmarkRunner,
    ) -> Result<f64> {
        // Run actual FP16 compute benchmark
        let tflops = runner
            .benchmark_fp16_compute()
            .context("Failed to run FP16 compute benchmark")?;

        tracing::info!("Measured FP16 performance: {:.2} TFLOPS", tflops);
        Ok(tflops)
    }

    fn validate_memory_size(&self, actual: u64, expected: u64, tolerance: f64) -> bool {
        let diff_percent = ((actual as f64 - expected as f64) / expected as f64).abs() * 100.0;
        diff_percent <= tolerance
    }

    fn validate_memory_bandwidth(&self, actual: f64, expected: f64, tolerance: f64) -> bool {
        let diff_percent = ((actual - expected) / expected).abs() * 100.0;
        diff_percent <= tolerance
    }

    fn validate_compute_performance(&self, actual: f64, expected: f64, tolerance: f64) -> bool {
        let diff_percent = ((actual - expected) / expected).abs() * 100.0;
        diff_percent <= tolerance
    }

    fn calculate_confidence_score(
        &self,
        memory_valid: bool,
        bandwidth_valid: bool,
        compute_valid: bool,
    ) -> f64 {
        let mut score = 0.0;
        if memory_valid {
            score += 0.33;
        }
        if bandwidth_valid {
            score += 0.33;
        }
        if compute_valid {
            score += 0.34;
        }
        score
    }

    /// Run a comprehensive stress test to detect thermal throttling
    pub fn stress_test_gpu(&self, duration: Duration) -> Result<StressTestResult> {
        let start = Instant::now();
        let mut performance_samples = Vec::new();

        while start.elapsed() < duration {
            let sample_start = Instant::now();
            let tflops = self.benchmark_fp16_compute()?;
            let sample_duration = sample_start.elapsed();

            performance_samples.push(PerformanceSample {
                timestamp: start.elapsed(),
                tflops,
                duration: sample_duration,
            });

            std::thread::sleep(Duration::from_millis(100));
        }

        // Analyze performance degradation
        let initial_performance = performance_samples
            .iter()
            .take(10)
            .map(|s| s.tflops)
            .sum::<f64>()
            / 10.0;

        let final_performance = performance_samples
            .iter()
            .rev()
            .take(10)
            .map(|s| s.tflops)
            .sum::<f64>()
            / 10.0;

        let degradation_percent =
            ((initial_performance - final_performance) / initial_performance) * 100.0;

        Ok(StressTestResult {
            duration: start.elapsed(),
            samples: performance_samples,
            initial_performance,
            final_performance,
            degradation_percent,
            thermal_throttling_detected: degradation_percent > 10.0,
        })
    }
}

#[derive(Debug, Clone)]
pub struct ValidationResult {
    pub is_valid: bool,
    pub claimed_model: String,
    pub detected_profile: String,
    pub memory_validation: MemoryValidation,
    pub bandwidth_validation: BandwidthValidation,
    pub compute_validation: ComputeValidation,
    pub confidence_score: f64,
}

#[derive(Debug, Clone)]
pub struct MemoryValidation {
    pub claimed_bytes: u64,
    pub expected_bytes: u64,
    pub is_valid: bool,
}

#[derive(Debug, Clone)]
pub struct BandwidthValidation {
    pub measured_gbps: f64,
    pub expected_gbps: f64,
    pub is_valid: bool,
}

#[derive(Debug, Clone)]
pub struct ComputeValidation {
    pub measured_tflops: f64,
    pub expected_tflops: f64,
    pub is_valid: bool,
}

#[derive(Debug, Clone)]
struct GpuBenchmarkResults {
    memory_bandwidth_gbps: f64,
    fp16_tflops: f64,
}

#[derive(Debug, Clone)]
pub struct StressTestResult {
    pub duration: Duration,
    pub samples: Vec<PerformanceSample>,
    pub initial_performance: f64,
    pub final_performance: f64,
    pub degradation_percent: f64,
    pub thermal_throttling_detected: bool,
}

#[derive(Debug, Clone)]
pub struct PerformanceSample {
    pub timestamp: Duration,
    pub tflops: f64,
    pub duration: Duration,
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_performance_validator_creation() {
        let validator = PerformanceValidator::new();
        assert!(validator.profiles.contains_key("H200"));
        assert!(validator.profiles.contains_key("H100"));
    }

    #[test]
    fn test_memory_size_validation() {
        let validator = PerformanceValidator::new();

        // Exact match
        assert!(validator.validate_memory_size(1000, 1000, 10.0));

        // Within tolerance
        assert!(validator.validate_memory_size(1050, 1000, 10.0));
        assert!(validator.validate_memory_size(950, 1000, 10.0));

        // Outside tolerance
        assert!(!validator.validate_memory_size(1200, 1000, 10.0));
        assert!(!validator.validate_memory_size(800, 1000, 10.0));
    }

    #[test]
    fn test_profile_matching() {
        let validator = PerformanceValidator::new();

        assert!(validator.find_matching_profile("NVIDIA H200").is_ok());
        assert!(validator.find_matching_profile("GeForce RTX 4090").is_ok());
        assert!(validator.find_matching_profile("Unknown GPU XYZ").is_err());
    }
}
