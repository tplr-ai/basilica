//! GPU monitoring functionality

use super::types::GpuInfo;
use anyhow::{Context, Result};
use tracing::{debug, info, warn};

/// GPU monitoring handler
#[derive(Debug)]
pub struct GpuMonitor;

impl GpuMonitor {
    /// Create new GPU monitor
    pub fn new() -> Self {
        Self
    }

    /// Get GPU information using NVIDIA ML
    pub async fn get_gpu_info(&self) -> Result<Vec<GpuInfo>> {
        debug!("Starting GPU detection with NVML...");
        let mut gpus = Vec::new();

        match self.get_nvidia_device_count() {
            Ok(device_count) => {
                info!("NVML detected {} NVIDIA GPU(s)", device_count);
                for i in 0..device_count {
                    match self.get_nvidia_gpu_info(i).await {
                        Ok(gpu_info) => {
                            debug!("Successfully got NVML info for GPU {}", i);
                            gpus.push(gpu_info);
                        }
                        Err(e) => warn!("Failed to get NVML info for GPU {}: {}", i, e),
                    }
                }
            }
            Err(e) => {
                info!("NVML unavailable: {}", e);
                debug!("This is normal in environments without NVIDIA driver access (like some containers or WSL setups)");
            }
        }

        debug!("GPU detection completed, found {} GPUs", gpus.len());
        Ok(gpus)
    }

    /// Get NVIDIA device count using NVML
    fn get_nvidia_device_count(&self) -> Result<u32> {
        use nvml_wrapper::Nvml;

        debug!("Attempting to initialize NVML...");
        let nvml = Nvml::init().context("Failed to initialize NVML")?;
        debug!("NVML initialized successfully");

        let device_count = nvml.device_count().context("Failed to get device count")?;
        debug!("NVML reported {} devices", device_count);
        Ok(device_count)
    }

    async fn get_nvidia_gpu_info(&self, index: u32) -> Result<GpuInfo> {
        use nvml_wrapper::Nvml;

        let nvml = Nvml::init().context("Failed to initialize NVML")?;
        let device = nvml
            .device_by_index(index)
            .context("Failed to get device by index")?;

        let name = device
            .name()
            .unwrap_or_else(|_| format!("Unknown GPU {index}"));
        let memory_info = device.memory_info().context("Failed to get memory info")?;

        let temperature = device
            .temperature(nvml_wrapper::enum_wrappers::device::TemperatureSensor::Gpu)
            .unwrap_or(0) as f32;

        let utilization = device
            .utilization_rates()
            .map(|u| u.gpu as f32)
            .unwrap_or(0.0);

        let power_usage = device
            .power_usage()
            .map(|p| p as f32 / 1000.0) // Convert from mW to W
            .unwrap_or(0.0);

        let driver_version = nvml
            .sys_driver_version()
            .unwrap_or_else(|_| "Unknown".to_string());

        let cuda_version = nvml
            .sys_cuda_driver_version()
            .ok()
            .map(|v| format!("{}.{}", v / 1000, (v % 1000) / 10));

        let memory_usage_percent = if memory_info.total > 0 {
            (memory_info.used as f32 / memory_info.total as f32) * 100.0
        } else {
            0.0
        };

        Ok(GpuInfo {
            index,
            name,
            memory_total_bytes: memory_info.total,
            memory_used_bytes: memory_info.used,
            memory_usage_percent,
            utilization_percent: utilization,
            temperature_celsius: temperature,
            power_usage_watts: power_usage,
            driver_version,
            cuda_version,
        })
    }
}

impl Default for GpuMonitor {
    fn default() -> Self {
        Self::new()
    }
}
