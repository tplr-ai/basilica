use serde::{Deserialize, Serialize};

#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct GpuInfo {
    pub vendor: GpuVendor,
    pub name: String,
    pub serial: Option<String>,
    pub memory_total: u64,
    pub memory_used: u64,
    pub temperature: Option<u32>,
    pub utilization: Option<u32>,
    pub cuda_version: Option<String>,
    pub driver_version: String,
    pub compute_capability: Option<String>,
    pub power_limit: Option<u32>,
    pub power_usage: Option<u32>,
}

#[derive(Debug, Clone, Serialize, Deserialize, PartialEq)]
pub enum GpuVendor {
    Nvidia,
    Amd,
    Intel,
    Unknown,
}

impl GpuInfo {
    pub fn new(vendor: GpuVendor, name: String, driver_version: String) -> Self {
        Self {
            vendor,
            name,
            serial: None,
            memory_total: 0,
            memory_used: 0,
            temperature: None,
            utilization: None,
            cuda_version: None,
            driver_version,
            compute_capability: None,
            power_limit: None,
            power_usage: None,
        }
    }

    pub fn with_memory(mut self, total: u64, used: u64) -> Self {
        self.memory_total = total;
        self.memory_used = used;
        self
    }

    pub fn with_serial(mut self, serial: String) -> Self {
        self.serial = Some(serial);
        self
    }

    pub fn with_temperature(mut self, temperature: u32) -> Self {
        self.temperature = Some(temperature);
        self
    }

    pub fn with_utilization(mut self, utilization: u32) -> Self {
        self.utilization = Some(utilization);
        self
    }

    pub fn with_cuda_version(mut self, cuda_version: String) -> Self {
        self.cuda_version = Some(cuda_version);
        self
    }

    pub fn with_compute_capability(mut self, compute_capability: String) -> Self {
        self.compute_capability = Some(compute_capability);
        self
    }

    pub fn with_power_info(mut self, limit: Option<u32>, usage: Option<u32>) -> Self {
        self.power_limit = limit;
        self.power_usage = usage;
        self
    }

    pub fn memory_utilization_percent(&self) -> f64 {
        if self.memory_total == 0 {
            0.0
        } else {
            (self.memory_used as f64 / self.memory_total as f64) * 100.0
        }
    }

    pub fn is_nvidia(&self) -> bool {
        matches!(self.vendor, GpuVendor::Nvidia)
    }

    pub fn is_amd(&self) -> bool {
        matches!(self.vendor, GpuVendor::Amd)
    }

    pub fn is_intel(&self) -> bool {
        matches!(self.vendor, GpuVendor::Intel)
    }
}

impl GpuVendor {
    pub fn as_str(&self) -> &'static str {
        match self {
            GpuVendor::Nvidia => "nvidia",
            GpuVendor::Amd => "amd",
            GpuVendor::Intel => "intel",
            GpuVendor::Unknown => "unknown",
        }
    }

    pub fn supports_cuda(&self) -> bool {
        matches!(self, GpuVendor::Nvidia)
    }

    pub fn supports_rocm(&self) -> bool {
        matches!(self, GpuVendor::Amd)
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_gpu_info_builder() {
        let gpu = GpuInfo::new(
            GpuVendor::Nvidia,
            "GeForce RTX 4090".to_string(),
            "535.54.03".to_string(),
        )
        .with_memory(24 * 1024 * 1024 * 1024, 1024 * 1024 * 1024) // 24GB total, 1GB used
        .with_temperature(65)
        .with_utilization(50)
        .with_cuda_version("12.0".to_string())
        .with_compute_capability("8.9".to_string());

        assert_eq!(gpu.vendor, GpuVendor::Nvidia);
        assert_eq!(gpu.name, "GeForce RTX 4090");
        assert_eq!(gpu.memory_total, 24 * 1024 * 1024 * 1024);
        assert_eq!(gpu.temperature, Some(65));
        assert!(gpu.is_nvidia());
        assert!(!gpu.is_amd());
    }

    #[test]
    fn test_memory_utilization_calculation() {
        let gpu = GpuInfo::new(GpuVendor::Nvidia, "Test GPU".to_string(), "1.0".to_string())
            .with_memory(1000, 250); // 25% utilization

        assert_eq!(gpu.memory_utilization_percent(), 25.0);
    }

    #[test]
    fn test_vendor_capabilities() {
        assert!(GpuVendor::Nvidia.supports_cuda());
        assert!(!GpuVendor::Nvidia.supports_rocm());

        assert!(!GpuVendor::Amd.supports_cuda());
        assert!(GpuVendor::Amd.supports_rocm());

        assert!(!GpuVendor::Intel.supports_cuda());
        assert!(!GpuVendor::Intel.supports_rocm());
    }

    #[test]
    fn test_serialization() {
        let gpu = GpuInfo::new(GpuVendor::Nvidia, "Test GPU".to_string(), "1.0".to_string());

        let json = serde_json::to_string(&gpu).unwrap();
        let deserialized: GpuInfo = serde_json::from_str(&json).unwrap();

        assert_eq!(gpu.vendor, deserialized.vendor);
        assert_eq!(gpu.name, deserialized.name);
        assert_eq!(gpu.driver_version, deserialized.driver_version);
    }
}
