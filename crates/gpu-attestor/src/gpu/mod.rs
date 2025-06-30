//! GPU hardware detection and information gathering module
//!
//! This module provides functionality to detect and gather information about
//! GPU hardware across different vendors (NVIDIA, AMD, Intel).

pub mod benchmark_collector;
pub mod benchmarks;
pub mod detection;
pub mod types;

#[cfg(feature = "cuda")]
pub mod cuda_ffi;

// CUDA Driver API module for matrix multiplication
pub mod cuda_driver;

// Re-export commonly used items
pub use detection::GpuDetector;
pub use types::{GpuInfo, GpuVendor};

// Re-export CUDA-specific items when CUDA is enabled
#[cfg(feature = "cuda")]
pub use cuda_ffi::{detect_gpu_architecture, GpuArchitecture};

/// Query all available GPU information on the system
///
/// This is a convenience function that combines vendor detection
/// and information gathering in a single call.
pub fn query_all_gpus() -> anyhow::Result<Vec<GpuInfo>> {
    GpuDetector::query_gpu_info()
}

/// Detect the primary GPU vendor on the system
///
/// Returns the vendor of the first detected GPU, or Unknown if no GPUs are found.
pub fn detect_primary_vendor() -> GpuVendor {
    GpuDetector::detect_gpu_vendor()
}

/// Check if the system has any NVIDIA GPUs
pub fn has_nvidia_gpu() -> bool {
    matches!(detect_primary_vendor(), GpuVendor::Nvidia)
}

/// Check if the system has any AMD GPUs
pub fn has_amd_gpu() -> bool {
    matches!(detect_primary_vendor(), GpuVendor::Amd)
}

/// Check if the system has any Intel GPUs
pub fn has_intel_gpu() -> bool {
    matches!(detect_primary_vendor(), GpuVendor::Intel)
}

/// Get a summary of GPU capabilities on the system
pub fn get_gpu_summary() -> anyhow::Result<GpuSummary> {
    let gpus = query_all_gpus()?;

    let total_memory = gpus.iter().map(|gpu| gpu.memory_total).sum();
    let total_used_memory = gpus.iter().map(|gpu| gpu.memory_used).sum();
    let avg_temperature = if gpus.is_empty() {
        None
    } else {
        let temps: Vec<u32> = gpus.iter().filter_map(|gpu| gpu.temperature).collect();
        if temps.is_empty() {
            None
        } else {
            Some(temps.iter().sum::<u32>() / temps.len() as u32)
        }
    };

    let avg_utilization = if gpus.is_empty() {
        None
    } else {
        let utils: Vec<u32> = gpus.iter().filter_map(|gpu| gpu.utilization).collect();
        if utils.is_empty() {
            None
        } else {
            Some(utils.iter().sum::<u32>() / utils.len() as u32)
        }
    };

    Ok(GpuSummary {
        gpu_count: gpus.len(),
        primary_vendor: detect_primary_vendor(),
        total_memory_bytes: total_memory,
        used_memory_bytes: total_used_memory,
        avg_temperature,
        avg_utilization,
        gpus,
    })
}

#[derive(Debug, Clone)]
pub struct GpuSummary {
    pub gpu_count: usize,
    pub primary_vendor: GpuVendor,
    pub total_memory_bytes: u64,
    pub used_memory_bytes: u64,
    pub avg_temperature: Option<u32>,
    pub avg_utilization: Option<u32>,
    pub gpus: Vec<GpuInfo>,
}

impl GpuSummary {
    pub fn memory_utilization_percent(&self) -> f64 {
        if self.total_memory_bytes == 0 {
            0.0
        } else {
            (self.used_memory_bytes as f64 / self.total_memory_bytes as f64) * 100.0
        }
    }

    pub fn total_memory_gb(&self) -> f64 {
        self.total_memory_bytes as f64 / (1024.0 * 1024.0 * 1024.0)
    }

    pub fn used_memory_gb(&self) -> f64 {
        self.used_memory_bytes as f64 / (1024.0 * 1024.0 * 1024.0)
    }

    pub fn has_gpus(&self) -> bool {
        self.gpu_count > 0
    }

    pub fn supports_cuda(&self) -> bool {
        self.gpus.iter().any(|gpu| gpu.vendor.supports_cuda())
    }

    pub fn supports_rocm(&self) -> bool {
        self.gpus.iter().any(|gpu| gpu.vendor.supports_rocm())
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_gpu_summary_calculations() {
        let gpus = vec![
            GpuInfo::new(GpuVendor::Nvidia, "GPU 1".to_string(), "1.0".to_string())
                .with_memory(1024 * 1024 * 1024, 512 * 1024 * 1024) // 1GB total, 512MB used
                .with_temperature(60)
                .with_utilization(50),
            GpuInfo::new(GpuVendor::Nvidia, "GPU 2".to_string(), "1.0".to_string())
                .with_memory(1024 * 1024 * 1024, 256 * 1024 * 1024) // 1GB total, 256MB used
                .with_temperature(70)
                .with_utilization(30),
        ];

        let summary = GpuSummary {
            gpu_count: gpus.len(),
            primary_vendor: GpuVendor::Nvidia,
            total_memory_bytes: gpus.iter().map(|gpu| gpu.memory_total).sum(),
            used_memory_bytes: gpus.iter().map(|gpu| gpu.memory_used).sum(),
            avg_temperature: Some(65), // (60 + 70) / 2
            avg_utilization: Some(40), // (50 + 30) / 2
            gpus,
        };

        assert_eq!(summary.gpu_count, 2);
        assert_eq!(summary.total_memory_gb(), 2.0);
        assert_eq!(summary.memory_utilization_percent(), 37.5); // 768MB / 2048MB
        assert!(summary.supports_cuda());
        assert!(!summary.supports_rocm());
    }

    #[test]
    fn test_vendor_detection_functions() {
        // These tests will work regardless of actual hardware
        let vendor = detect_primary_vendor();

        match vendor {
            GpuVendor::Nvidia => assert!(has_nvidia_gpu()),
            GpuVendor::Amd => assert!(has_amd_gpu()),
            GpuVendor::Intel => assert!(has_intel_gpu()),
            GpuVendor::Unknown => {
                assert!(!has_nvidia_gpu());
                assert!(!has_amd_gpu());
                assert!(!has_intel_gpu());
            }
        }
    }
}
