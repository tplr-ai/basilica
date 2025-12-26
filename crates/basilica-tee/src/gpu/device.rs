//! GPU Device Provider
//!
//! Provides GPU device information using NVIDIA NVML.
//! Feature-gated behind the `nvml` feature flag.

use crate::error::{TeeError, TeeResult};
use crate::types::GpuDeviceInfo;

/// Sanitize GPU ID - remove 'GPU-' prefix and all hyphens
#[allow(dead_code)]
fn sanitize_gpu_id(gpu_id: &str) -> String {
    gpu_id
        .replace("GPU-", "")
        .replace("gpu-", "")
        .replace('-', "")
}

/// GPU Device Provider using NVML
///
/// Provides information about NVIDIA GPUs in the system.
/// Requires the `nvml` feature to be enabled.
pub struct GpuDeviceProvider {
    #[cfg(feature = "nvml")]
    nvml: Option<nvml_wrapper::Nvml>,
    #[cfg(not(feature = "nvml"))]
    _phantom: std::marker::PhantomData<()>,
}

impl GpuDeviceProvider {
    /// Create a new GPU device provider
    #[cfg(feature = "nvml")]
    pub fn new() -> TeeResult<Self> {
        let nvml = nvml_wrapper::Nvml::init().ok();
        if nvml.is_none() {
            tracing::warn!("Failed to initialize NVML - GPU features will be limited");
        }
        Ok(Self { nvml })
    }

    /// Create a new GPU device provider (without NVML feature)
    #[cfg(not(feature = "nvml"))]
    pub fn new() -> TeeResult<Self> {
        Ok(Self {
            _phantom: std::marker::PhantomData,
        })
    }

    /// Check if NVML is available
    #[cfg(feature = "nvml")]
    pub fn is_available(&self) -> bool {
        self.nvml.is_some()
    }

    #[cfg(not(feature = "nvml"))]
    pub fn is_available(&self) -> bool {
        false
    }

    /// Get the number of GPUs
    #[cfg(feature = "nvml")]
    pub fn device_count(&self) -> TeeResult<u32> {
        let nvml = self
            .nvml
            .as_ref()
            .ok_or_else(|| TeeError::Nvml("NVML not initialized".into()))?;

        nvml.device_count()
            .map_err(|e| TeeError::Nvml(e.to_string()))
    }

    #[cfg(not(feature = "nvml"))]
    pub fn device_count(&self) -> TeeResult<u32> {
        Err(TeeError::Nvml("NVML feature not enabled".into()))
    }

    /// Get device info for all GPUs or filtered by IDs
    #[cfg(feature = "nvml")]
    pub fn get_device_info(&self, gpu_ids: Option<&[String]>) -> TeeResult<Vec<GpuDeviceInfo>> {
        use nvml_wrapper::enum_wrappers::device::Clock;

        let nvml = self
            .nvml
            .as_ref()
            .ok_or_else(|| TeeError::Nvml("NVML not initialized".into()))?;

        let device_count = nvml
            .device_count()
            .map_err(|e| TeeError::Nvml(e.to_string()))?;

        let mut all_gpus = Vec::new();

        for i in 0..device_count {
            let device = nvml
                .device_by_index(i)
                .map_err(|e| TeeError::Nvml(e.to_string()))?;

            let name = device.name().map_err(|e| TeeError::Nvml(e.to_string()))?;

            let uuid = sanitize_gpu_id(&device.uuid().map_err(|e| TeeError::Nvml(e.to_string()))?);

            let memory = device
                .memory_info()
                .map_err(|e| TeeError::Nvml(e.to_string()))?
                .total;

            let cc = device
                .cuda_compute_capability()
                .map_err(|e| TeeError::Nvml(e.to_string()))?;

            let clock_rate = device
                .max_clock_info(Clock::Graphics)
                .map(|c| c as f64 * 1000.0)
                .unwrap_or(0.0);

            let ecc = device
                .is_ecc_enabled()
                .ok()
                .map(|state| state.currently_enabled);

            // Extract short model reference (last word of name)
            let model_short_ref = name
                .split_whitespace()
                .last()
                .unwrap_or(&name)
                .to_lowercase();

            // TODO: Check CC mode status
            // This would require nvidia-smi or direct NVML CC query
            let cc_mode_enabled = None;

            all_gpus.push(GpuDeviceInfo {
                uuid,
                name,
                memory,
                major: Some(cc.major as u32),
                minor: Some(cc.minor as u32),
                clock_rate,
                ecc,
                model_short_ref,
                cc_mode_enabled,
            });
        }

        // Filter by GPU IDs if provided
        if let Some(target_ids) = gpu_ids {
            let formatted_ids: Vec<String> =
                target_ids.iter().map(|id| sanitize_gpu_id(id)).collect();

            Ok(all_gpus
                .into_iter()
                .filter(|gpu| formatted_ids.contains(&gpu.uuid))
                .collect())
        } else {
            Ok(all_gpus)
        }
    }

    #[cfg(not(feature = "nvml"))]
    pub fn get_device_info(&self, _gpu_ids: Option<&[String]>) -> TeeResult<Vec<GpuDeviceInfo>> {
        Err(TeeError::Nvml("NVML feature not enabled".into()))
    }

    /// Get device info by index
    #[cfg(feature = "nvml")]
    pub fn get_device_by_index(&self, index: u32) -> TeeResult<GpuDeviceInfo> {
        let all_devices = self.get_device_info(None)?;
        all_devices
            .into_iter()
            .nth(index as usize)
            .ok_or_else(|| TeeError::Nvml(format!("GPU index {} not found", index)))
    }

    #[cfg(not(feature = "nvml"))]
    pub fn get_device_by_index(&self, _index: u32) -> TeeResult<GpuDeviceInfo> {
        Err(TeeError::Nvml("NVML feature not enabled".into()))
    }
}

impl Default for GpuDeviceProvider {
    fn default() -> Self {
        Self::new().unwrap_or({
            #[cfg(feature = "nvml")]
            {
                Self { nvml: None }
            }
            #[cfg(not(feature = "nvml"))]
            {
                Self {
                    _phantom: std::marker::PhantomData,
                }
            }
        })
    }
}

/// Mock GPU device provider for testing
#[cfg(test)]
pub struct MockGpuDeviceProvider {
    devices: Vec<GpuDeviceInfo>,
}

#[cfg(test)]
impl MockGpuDeviceProvider {
    pub fn new(devices: Vec<GpuDeviceInfo>) -> Self {
        Self { devices }
    }

    pub fn with_h100() -> Self {
        Self {
            devices: vec![GpuDeviceInfo {
                uuid: "abc123def456".to_string(),
                name: "NVIDIA H100 PCIe".to_string(),
                memory: 80 * 1024 * 1024 * 1024,
                major: Some(9),
                minor: Some(0),
                clock_rate: 1755000.0,
                ecc: Some(true),
                model_short_ref: "h100".to_string(),
                cc_mode_enabled: Some(true),
            }],
        }
    }

    pub fn get_device_info(&self, _gpu_ids: Option<&[String]>) -> TeeResult<Vec<GpuDeviceInfo>> {
        Ok(self.devices.clone())
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_sanitize_gpu_id() {
        assert_eq!(sanitize_gpu_id("GPU-abc-def-123"), "abcdef123");
        assert_eq!(sanitize_gpu_id("gpu-xyz"), "xyz");
        assert_eq!(sanitize_gpu_id("no-prefix"), "noprefix");
        assert_eq!(sanitize_gpu_id("plain"), "plain");
    }

    #[test]
    fn test_mock_provider() {
        let provider = MockGpuDeviceProvider::with_h100();
        let devices = provider.get_device_info(None).unwrap();

        assert_eq!(devices.len(), 1);
        assert_eq!(devices[0].name, "NVIDIA H100 PCIe");
        assert_eq!(devices[0].cc_mode_enabled, Some(true));
    }

    #[test]
    fn test_provider_default() {
        let _provider = GpuDeviceProvider::default();
        // Without NVML feature, should not be available
        #[cfg(not(feature = "nvml"))]
        assert!(!_provider.is_available());
    }

    #[test]
    fn test_gpu_device_info_serialization() {
        let info = GpuDeviceInfo {
            uuid: "test123".to_string(),
            name: "Test GPU".to_string(),
            memory: 1024,
            major: Some(8),
            minor: Some(0),
            clock_rate: 1500.0,
            ecc: Some(false),
            model_short_ref: "test".to_string(),
            cc_mode_enabled: None,
        };

        let json = serde_json::to_string(&info).unwrap();
        let parsed: GpuDeviceInfo = serde_json::from_str(&json).unwrap();

        assert_eq!(info.uuid, parsed.uuid);
        assert_eq!(info.name, parsed.name);
    }
}
