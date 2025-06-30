use anyhow::{Context, Result};
use std::path::PathBuf;
use std::process::Command;

use super::types::{GpuInfo, GpuVendor};

pub struct GpuDetector;

impl Default for GpuDetector {
    fn default() -> Self {
        Self::new()
    }
}

impl GpuDetector {
    /// Create a new GPU detector
    pub fn new() -> Self {
        Self
    }

    /// Detect all GPUs on the system
    pub fn detect(&self) -> Result<Vec<GpuInfo>> {
        Self::query_gpu_info()
    }
    /// Find the absolute path of a command by searching common locations
    fn find_command(command: &str) -> Option<PathBuf> {
        // First try the command as-is (might be in PATH)
        if Command::new(command)
            .arg("--help")
            .stdout(std::process::Stdio::null())
            .stderr(std::process::Stdio::null())
            .status()
            .map(|s| s.success())
            .unwrap_or(false)
        {
            return Some(PathBuf::from(command));
        }

        let search_paths = [
            // Standard Unix paths
            "/usr/bin",
            "/usr/local/bin",
            "/bin",
            "/usr/sbin",
            "/usr/local/sbin",
            "/sbin",
            // GPU-specific paths
            "/opt/rocm/bin",
            "/usr/local/cuda/bin",
            "/opt/cuda/bin",
            "/usr/local/cuda-*/bin",
            // WSL paths - Windows executables accessible from WSL
            "/mnt/c/Program Files/NVIDIA Corporation/NVSMI",
            "/mnt/c/Windows/System32",
            "/mnt/c/Windows/SysWOW64",
            // WSL paths - Linux subsystem
            "/usr/lib/wsl/lib",
            "/usr/lib/wsl/drivers",
            // Snap packages
            "/snap/bin",
            "/var/lib/snapd/snap/bin",
            // Flatpak
            "/var/lib/flatpak/exports/bin",
            "/usr/local/share/flatpak/exports/bin",
            // Container paths
            "/usr/local/nvidia/bin",
            "/usr/nvidia/bin",
            // Distribution-specific paths
            "/opt/bin",
            "/usr/games",
            "/usr/local/games",
            // Additional CUDA paths
            "/usr/local/cuda/bin",
            "/opt/cuda/bin",
            "/usr/cuda/bin",
            // ROCm additional paths
            "/opt/rocm/bin",
            "/opt/rocm-*/bin",
            "/usr/rocm/bin",
            // Intel GPU tools
            "/usr/local/intel/bin",
            "/opt/intel/bin",
        ];

        // Search common locations
        for path in &search_paths {
            let full_path = PathBuf::from(path).join(command);
            if full_path.exists() {
                tracing::debug!("Found {} at {}", command, full_path.display());
                return Some(full_path);
            }
        }

        // WSL special case for nvidia-smi.exe
        if command == "nvidia-smi" {
            let wsl_path =
                PathBuf::from("/mnt/c/Program Files/NVIDIA Corporation/NVSMI/nvidia-smi.exe");
            if wsl_path.exists() {
                tracing::debug!("Found {} (WSL) at {}", command, wsl_path.display());
                return Some(wsl_path);
            }
        }

        tracing::debug!("Command {} not found", command);
        None
    }

    /// Detect the primary GPU vendor on the system
    pub fn detect_gpu_vendor() -> GpuVendor {
        // Try NVIDIA first
        if Self::is_nvidia_available() {
            return GpuVendor::Nvidia;
        }

        // Try AMD ROCm
        if Self::is_amd_available() {
            return GpuVendor::Amd;
        }

        // Try Intel
        if Self::is_intel_available() {
            return GpuVendor::Intel;
        }

        GpuVendor::Unknown
    }

    /// Query information for all available GPUs
    pub fn query_gpu_info() -> Result<Vec<GpuInfo>> {
        match Self::detect_gpu_vendor() {
            GpuVendor::Nvidia => Self::query_nvidia_gpus(),
            GpuVendor::Amd => Self::query_amd_gpus(),
            GpuVendor::Intel => Self::query_intel_gpus(),
            GpuVendor::Unknown => Ok(vec![]),
        }
    }

    /// Check if NVIDIA tools are available
    fn is_nvidia_available() -> bool {
        let nvidia_smi = match Self::find_command("nvidia-smi") {
            Some(path) => path,
            None => return false,
        };

        let result = Command::new(&nvidia_smi)
            .arg("--version")
            .stdout(std::process::Stdio::null())
            .stderr(std::process::Stdio::null())
            .status()
            .is_ok_and(|s| s.success());

        tracing::debug!(
            "NVIDIA GPU detection: {} (using {})",
            result,
            nvidia_smi.display()
        );
        result
    }

    /// Check if AMD ROCm tools are available
    fn is_amd_available() -> bool {
        let rocm_smi = match Self::find_command("rocm-smi") {
            Some(path) => path,
            None => return false,
        };

        let result = Command::new(&rocm_smi)
            .arg("--version")
            .stdout(std::process::Stdio::null())
            .stderr(std::process::Stdio::null())
            .status()
            .is_ok_and(|s| s.success());

        tracing::debug!(
            "AMD GPU detection: {} (using {})",
            result,
            rocm_smi.display()
        );
        result
    }

    /// Check if Intel GPU tools are available
    fn is_intel_available() -> bool {
        let intel_gpu_top = match Self::find_command("intel_gpu_top") {
            Some(path) => path,
            None => return false,
        };

        let result = Command::new(&intel_gpu_top)
            .arg("--help")
            .stdout(std::process::Stdio::null())
            .stderr(std::process::Stdio::null())
            .status()
            .is_ok_and(|s| s.success());

        tracing::debug!(
            "Intel GPU detection: {} (using {})",
            result,
            intel_gpu_top.display()
        );
        result
    }

    /// Query NVIDIA GPU information
    fn query_nvidia_gpus() -> Result<Vec<GpuInfo>> {
        // Try NVML first, fall back to nvidia-smi if it fails
        match Self::query_nvidia_nvml() {
            Ok(gpus) => Ok(gpus),
            Err(e) => {
                tracing::debug!("NVML failed ({}), falling back to nvidia-smi", e);
                Self::query_nvidia_smi()
            }
        }
    }

    /// Query AMD GPU information
    fn query_amd_gpus() -> Result<Vec<GpuInfo>> {
        let rocm_smi = Self::find_command("rocm-smi")
            .ok_or_else(|| anyhow::anyhow!("rocm-smi command not found"))?;

        let output = Command::new(&rocm_smi)
            .arg("--allinfo")
            .arg("--json")
            .output()
            .context("Failed to run rocm-smi")?;

        if !output.status.success() {
            anyhow::bail!(
                "rocm-smi command failed: {}",
                String::from_utf8_lossy(&output.stderr)
            );
        }

        Self::parse_rocm_output(&output.stdout)
    }

    /// Query Intel GPU information
    fn query_intel_gpus() -> Result<Vec<GpuInfo>> {
        let lspci = Self::find_command("lspci")
            .ok_or_else(|| anyhow::anyhow!("lspci command not found"))?;

        let output = Command::new(&lspci)
            .args(["-nn", "-d", "8086:"])
            .output()
            .context("Failed to run lspci for Intel GPU detection")?;

        let output_str =
            String::from_utf8(output.stdout).context("Invalid UTF-8 in lspci output")?;

        Self::parse_intel_lspci_output(&output_str)
    }

    /// Parse ROCm SMI JSON output
    fn parse_rocm_output(output: &[u8]) -> Result<Vec<GpuInfo>> {
        let output_str =
            String::from_utf8(output.to_vec()).context("Invalid UTF-8 in rocm-smi output")?;
        let json_data: serde_json::Value =
            serde_json::from_str(&output_str).context("Failed to parse rocm-smi JSON output")?;

        let mut gpus = Vec::new();

        if let Some(devices) = json_data.as_object() {
            for (_device_id, device_info) in devices {
                if let Some(info) = device_info.as_object() {
                    let name = info
                        .get("Card series")
                        .and_then(|v| v.as_str())
                        .unwrap_or("Unknown AMD GPU")
                        .to_string();

                    let memory_total = info
                        .get("VRAM Total Memory (B)")
                        .and_then(|v| v.as_u64())
                        .unwrap_or(0);

                    let memory_used = info
                        .get("VRAM Used Memory (B)")
                        .and_then(|v| v.as_u64())
                        .unwrap_or(0);

                    let temperature = info
                        .get("Temperature (Sensor edge) (C)")
                        .and_then(|v| v.as_u64())
                        .map(|t| t as u32);

                    let utilization = info
                        .get("GPU use (%)")
                        .and_then(|v| v.as_u64())
                        .map(|u| u as u32);

                    let driver_version = info
                        .get("Driver version")
                        .and_then(|v| v.as_str())
                        .unwrap_or("Unknown")
                        .to_string();

                    let mut gpu_info = GpuInfo::new(GpuVendor::Amd, name, driver_version)
                        .with_memory(memory_total, memory_used);

                    if let Some(temp) = temperature {
                        gpu_info = gpu_info.with_temperature(temp);
                    }

                    if let Some(util) = utilization {
                        gpu_info = gpu_info.with_utilization(util);
                    }

                    gpus.push(gpu_info);
                }
            }
        }

        Ok(gpus)
    }

    /// Parse Intel lspci output
    fn parse_intel_lspci_output(output: &str) -> Result<Vec<GpuInfo>> {
        let mut gpus = Vec::new();

        for line in output.lines() {
            if line.to_lowercase().contains("vga") || line.to_lowercase().contains("display") {
                let name = line
                    .split(": ")
                    .nth(1)
                    .unwrap_or("Unknown Intel GPU")
                    .to_string();

                let gpu_info = GpuInfo::new(GpuVendor::Intel, name, "Unknown".to_string());

                gpus.push(gpu_info);
            }
        }

        Ok(gpus)
    }

    fn query_nvidia_nvml() -> Result<Vec<GpuInfo>> {
        use nvml_wrapper::Nvml;

        let nvml = Nvml::init().context("Failed to initialize NVML")?;
        let device_count = nvml.device_count().context("Failed to get device count")?;

        let mut gpus = Vec::new();

        for i in 0..device_count {
            let device = nvml
                .device_by_index(i)
                .context("Failed to get device by index")?;

            let name = device.name().unwrap_or_else(|_| "Unknown".to_string());
            let serial = device.serial().ok();
            let memory_info = device.memory_info().context("Failed to get memory info")?;
            let temperature = device
                .temperature(nvml_wrapper::enum_wrappers::device::TemperatureSensor::Gpu)
                .ok();
            let utilization = device.utilization_rates().ok().map(|u| u.gpu);
            let power_limit = device
                .power_management_limit_default()
                .ok()
                .map(|p| p / 1000); // Convert to watts
            let power_usage = device.power_usage().ok().map(|p| p / 1000); // Convert to watts

            let driver_version = nvml
                .sys_driver_version()
                .unwrap_or_else(|_| "Unknown".to_string());
            let cuda_version = nvml
                .sys_cuda_driver_version()
                .ok()
                .map(|v| format!("{}.{}", v / 1000, (v % 1000) / 10));

            let compute_capability = device
                .cuda_compute_capability()
                .ok()
                .map(|cc| format!("{}.{}", cc.major, cc.minor));

            let mut gpu_info = GpuInfo::new(GpuVendor::Nvidia, name, driver_version)
                .with_memory(memory_info.total, memory_info.used)
                .with_power_info(power_limit, power_usage);

            if let Some(s) = serial {
                gpu_info = gpu_info.with_serial(s);
            }

            if let Some(temp) = temperature {
                gpu_info = gpu_info.with_temperature(temp);
            }

            if let Some(util) = utilization {
                gpu_info = gpu_info.with_utilization(util);
            }

            if let Some(cuda) = cuda_version {
                gpu_info = gpu_info.with_cuda_version(cuda);
            }

            if let Some(cc) = compute_capability {
                gpu_info = gpu_info.with_compute_capability(cc);
            }

            gpus.push(gpu_info);
        }

        Ok(gpus)
    }

    fn query_nvidia_smi() -> Result<Vec<GpuInfo>> {
        let nvidia_smi = Self::find_command("nvidia-smi")
            .ok_or_else(|| anyhow::anyhow!("nvidia-smi command not found"))?;

        let output = Command::new(&nvidia_smi)
            .args([
                "--query-gpu=name,serial,memory.total,memory.used,temperature.gpu,utilization.gpu,driver_version,compute_cap"
            ])
            .arg("--format=csv,noheader,nounits")
            .output()
            .context("Failed to run nvidia-smi")?;

        if !output.status.success() {
            anyhow::bail!(
                "nvidia-smi command failed: {}",
                String::from_utf8_lossy(&output.stderr)
            );
        }

        Self::parse_nvidia_smi_output(&output.stdout)
    }

    fn parse_nvidia_smi_output(output: &[u8]) -> Result<Vec<GpuInfo>> {
        let output_str =
            String::from_utf8(output.to_vec()).context("Invalid UTF-8 in nvidia-smi output")?;
        let mut gpus = Vec::new();

        for line in output_str.lines() {
            let fields: Vec<&str> = line.split(',').map(|s| s.trim()).collect();
            if fields.len() >= 8 {
                let mut gpu_info = GpuInfo::new(
                    GpuVendor::Nvidia,
                    fields[0].to_string(),
                    fields[6].to_string(),
                )
                .with_memory(
                    fields[2].parse().unwrap_or(0) * 1024 * 1024, // Convert MB to bytes
                    fields[3].parse().unwrap_or(0) * 1024 * 1024, // Convert MB to bytes
                );

                if !fields[1].is_empty() {
                    gpu_info = gpu_info.with_serial(fields[1].to_string());
                }

                if let Ok(temp) = fields[4].parse() {
                    gpu_info = gpu_info.with_temperature(temp);
                }

                if let Ok(util) = fields[5].parse() {
                    gpu_info = gpu_info.with_utilization(util);
                }

                if !fields[7].is_empty() {
                    gpu_info = gpu_info.with_compute_capability(fields[7].to_string());
                }

                gpus.push(gpu_info);
            }
        }

        Ok(gpus)
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_find_command() {
        // Test with a command that should exist on most systems
        let ls_path = GpuDetector::find_command("ls");
        assert!(ls_path.is_some());

        // Test with a command that definitely doesn't exist
        let fake_path = GpuDetector::find_command("fake_command_that_doesnt_exist_12345");
        assert!(fake_path.is_none());
    }

    #[test]
    fn test_gpu_vendor_detection() {
        let vendor = GpuDetector::detect_gpu_vendor();
        // This test will pass regardless of the actual GPU vendor
        assert!(matches!(
            vendor,
            GpuVendor::Nvidia | GpuVendor::Amd | GpuVendor::Intel | GpuVendor::Unknown
        ));
    }

    #[test]
    fn test_parse_intel_lspci_output() {
        let sample_output = r#"00:02.0 VGA compatible controller [0300]: Intel Corporation Device [8086:46a6] (rev 0c)
00:02.1 Display controller [0380]: Intel Corporation Device [8086:46a7] (rev 0c)"#;

        let gpus = GpuDetector::parse_intel_lspci_output(sample_output).unwrap();
        assert_eq!(gpus.len(), 2);
        assert!(gpus.iter().all(|gpu| gpu.vendor == GpuVendor::Intel));
    }

    #[test]
    fn test_parse_rocm_empty_output() {
        let empty_json = "{}";
        let gpus = GpuDetector::parse_rocm_output(empty_json.as_bytes()).unwrap();
        assert_eq!(gpus.len(), 0);
    }

    #[test]
    fn test_parse_nvidia_smi_output() {
        let sample_output =
            "GeForce RTX 4090, 1234567890, 24564, 1024, 65, 50, 535.54.03, 12.0, 8.9\n";

        let gpus = GpuDetector::parse_nvidia_smi_output(sample_output.as_bytes()).unwrap();
        assert_eq!(gpus.len(), 1);

        let gpu = &gpus[0];
        assert_eq!(gpu.vendor, GpuVendor::Nvidia);
        assert_eq!(gpu.name, "GeForce RTX 4090");
        assert_eq!(gpu.serial, Some("1234567890".to_string()));
        assert_eq!(gpu.temperature, Some(65));
        assert_eq!(gpu.utilization, Some(50));
    }
}
