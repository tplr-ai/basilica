use anyhow::{Context, Result};
use std::collections::HashMap;
use std::fs;
use std::process::Command;
use sysinfo::{Disks, Networks, System};

use super::types::*;

pub struct SystemInfoCollector;

impl SystemInfoCollector {
    pub fn create_minimal_system_info() -> SystemInfo {
        SystemInfo {
            motherboard: MotherboardInfo {
                manufacturer: "Unknown".to_string(),
                product_name: "Unknown".to_string(),
                version: "Unknown".to_string(),
                serial_number: None,
                asset_tag: None,
                bios_vendor: "Unknown".to_string(),
                bios_version: "Unknown".to_string(),
                bios_date: "Unknown".to_string(),
            },
            cpu: CpuInfo {
                brand: "Unknown".to_string(),
                vendor_id: "Unknown".to_string(),
                cores: 0,
                threads: 0,
                frequency_mhz: 0,
                architecture: "Unknown".to_string(),
                features: Vec::new(),
                temperature: None,
            },
            memory: MemoryInfo {
                total_bytes: 0,
                available_bytes: 0,
                used_bytes: 0,
                swap_total_bytes: 0,
                swap_used_bytes: 0,
                memory_modules: Vec::new(),
            },
            network: NetworkInfo {
                interfaces: Vec::new(),
                connectivity_test: ConnectivityTest {
                    can_reach_internet: false,
                    dns_resolution_working: false,
                    latency_ms: None,
                },
            },
            storage: Vec::new(),
            benchmarks: BenchmarkResults {
                cpu_benchmark_score: 0.0,
                memory_bandwidth_mbps: 0.0,
                disk_sequential_read_mbps: 0.0,
                disk_sequential_write_mbps: 0.0,
                network_throughput_mbps: None,
            },
        }
    }

    pub fn collect_all() -> Result<SystemInfo> {
        Ok(SystemInfo {
            motherboard: Self::collect_motherboard_info()?,
            cpu: Self::collect_cpu_info()?,
            memory: Self::collect_memory_info()?,
            storage: Self::collect_storage_info()?,
            network: Self::collect_network_info()?,
            benchmarks: super::benchmarks::BenchmarkRunner::run_all()?,
        })
    }

    pub fn collect_motherboard_info() -> Result<MotherboardInfo> {
        // Try to get DMI data, but don't fail if it's not available
        let dmi_data = match Self::parse_dmidecode() {
            Ok(data) => data,
            Err(e) => {
                tracing::warn!("Failed to get DMI data: {}. Using fallback methods.", e);
                HashMap::new()
            }
        };

        // Try alternative sources if DMI data is not available
        let mut info = MotherboardInfo {
            manufacturer: dmi_data
                .get("Base Board Information.Manufacturer")
                .unwrap_or(&"Unknown".to_string())
                .clone(),
            product_name: dmi_data
                .get("Base Board Information.Product Name")
                .unwrap_or(&"Unknown".to_string())
                .clone(),
            version: dmi_data
                .get("Base Board Information.Version")
                .unwrap_or(&"Unknown".to_string())
                .clone(),
            serial_number: dmi_data
                .get("Base Board Information.Serial Number")
                .cloned(),
            asset_tag: dmi_data.get("Base Board Information.Asset Tag").cloned(),
            bios_vendor: dmi_data
                .get("BIOS Information.Vendor")
                .unwrap_or(&"Unknown".to_string())
                .clone(),
            bios_version: dmi_data
                .get("BIOS Information.Version")
                .unwrap_or(&"Unknown".to_string())
                .clone(),
            bios_date: dmi_data
                .get("BIOS Information.Release Date")
                .unwrap_or(&"Unknown".to_string())
                .clone(),
        };

        // Try to get some info from /sys if DMI failed
        if info.manufacturer == "Unknown" {
            if let Ok(board_vendor) = fs::read_to_string("/sys/devices/virtual/dmi/id/board_vendor")
            {
                info.manufacturer = board_vendor.trim().to_string();
            }
        }

        if info.product_name == "Unknown" {
            if let Ok(board_name) = fs::read_to_string("/sys/devices/virtual/dmi/id/board_name") {
                info.product_name = board_name.trim().to_string();
            }
        }

        if info.bios_vendor == "Unknown" {
            if let Ok(bios_vendor) = fs::read_to_string("/sys/devices/virtual/dmi/id/bios_vendor") {
                info.bios_vendor = bios_vendor.trim().to_string();
            }
        }

        if info.bios_version == "Unknown" {
            if let Ok(bios_version) = fs::read_to_string("/sys/devices/virtual/dmi/id/bios_version")
            {
                info.bios_version = bios_version.trim().to_string();
            }
        }

        Ok(info)
    }

    pub fn collect_cpu_info() -> Result<CpuInfo> {
        let mut sys = System::new_all();
        sys.refresh_cpu();

        let cpus = sys.cpus();
        let first_cpu = cpus.first().context("No CPU found")?;

        let cpu_info =
            fs::read_to_string("/proc/cpuinfo").context("Failed to read /proc/cpuinfo")?;

        let mut vendor_id = "Unknown".to_string();
        let mut features = Vec::new();

        for line in cpu_info.lines() {
            if line.starts_with("vendor_id") {
                vendor_id = line.split(": ").nth(1).unwrap_or("Unknown").to_string();
            } else if line.starts_with("flags") {
                features = line
                    .split(": ")
                    .nth(1)
                    .unwrap_or("")
                    .split_whitespace()
                    .map(|s| s.to_string())
                    .collect();
            }
        }

        let temperature = Self::get_cpu_temperature();

        Ok(CpuInfo {
            brand: first_cpu.brand().to_string(),
            vendor_id,
            cores: sys.physical_core_count().unwrap_or(cpus.len()),
            threads: cpus.len(),
            frequency_mhz: first_cpu.frequency(),
            architecture: std::env::consts::ARCH.to_string(),
            features,
            temperature,
        })
    }

    pub fn collect_memory_info() -> Result<MemoryInfo> {
        let mut sys = System::new_all();
        sys.refresh_memory();

        let memory_modules = Self::parse_memory_modules()?;

        Ok(MemoryInfo {
            total_bytes: sys.total_memory() * 1024, // sysinfo returns KB
            available_bytes: sys.available_memory() * 1024,
            used_bytes: sys.used_memory() * 1024,
            swap_total_bytes: sys.total_swap() * 1024,
            swap_used_bytes: sys.used_swap() * 1024,
            memory_modules,
        })
    }

    pub fn collect_storage_info() -> Result<Vec<StorageInfo>> {
        let _sys = System::new_all();
        // Remove unused refresh call

        let mut storage_info = Vec::new();
        let disks = Disks::new_with_refreshed_list();

        for disk in &disks {
            storage_info.push(StorageInfo {
                name: disk.name().to_string_lossy().to_string(),
                total_space: disk.total_space(),
                available_space: disk.available_space(),
                mount_point: disk.mount_point().to_string_lossy().to_string(),
                file_system: disk.file_system().to_string_lossy().to_string(),
                disk_type: format!("{:?}", disk.kind()),
            });
        }

        Ok(storage_info)
    }

    pub fn collect_network_info() -> Result<NetworkInfo> {
        let mut networks = Networks::new_with_refreshed_list();
        networks.refresh();

        let mut interfaces = Vec::new();

        for (interface_name, data) in &networks {
            let mut ip_addresses = Vec::new();

            // Try to get IP addresses using ip command
            if let Ok(output) = Command::new("ip")
                .args(["addr", "show", interface_name])
                .output()
            {
                let output_str = String::from_utf8_lossy(&output.stdout);
                for line in output_str.lines() {
                    if line.trim().starts_with("inet ") {
                        if let Some(ip) = line.split_whitespace().nth(1) {
                            if let Some(ip_only) = ip.split('/').next() {
                                ip_addresses.push(ip_only.to_string());
                            }
                        }
                    }
                }
            }

            interfaces.push(NetworkInterface {
                name: interface_name.to_string(),
                mac_address: data
                    .mac_address()
                    .0
                    .iter()
                    .map(|b| format!("{b:02x}"))
                    .collect::<Vec<_>>()
                    .join(":"),
                ip_addresses,
                is_up: data.packets_received() > 0 || data.packets_transmitted() > 0,
                speed_mbps: None, // Would require more complex detection
            });
        }

        let connectivity_test = Self::test_connectivity()?;

        Ok(NetworkInfo {
            interfaces,
            connectivity_test,
        })
    }

    fn parse_dmidecode() -> Result<HashMap<String, String>> {
        // First check if dmidecode exists
        let dmidecode_exists = Command::new("which")
            .arg("dmidecode")
            .stdout(std::process::Stdio::null())
            .status()
            .is_ok_and(|s| s.success());

        if !dmidecode_exists {
            tracing::warn!("dmidecode not found in PATH. Hardware information will be limited.");
            return Ok(HashMap::new());
        }

        // Try with sudo if available
        let output = if Command::new("which")
            .arg("sudo")
            .stdout(std::process::Stdio::null())
            .status()
            .is_ok_and(|s| s.success())
        {
            // Try with sudo first
            let sudo_output = Command::new("sudo").args(["-n", "dmidecode"]).output();

            match sudo_output {
                Ok(output) if output.status.success() => output,
                _ => {
                    // Fall back to direct dmidecode
                    Command::new("dmidecode")
                        .output()
                        .context("Failed to run dmidecode (tried both sudo and direct execution)")?
                }
            }
        } else {
            // No sudo available, try direct
            Command::new("dmidecode")
                .output()
                .context("Failed to run dmidecode")?
        };

        if !output.status.success() {
            let stderr = String::from_utf8_lossy(&output.stderr);
            if stderr.contains("Permission denied") || stderr.contains("Operation not permitted") {
                anyhow::bail!(
                    "dmidecode requires root privileges. Try running with sudo or as root"
                );
            } else {
                anyhow::bail!("dmidecode command failed: {}", stderr);
            }
        }

        let output_str =
            String::from_utf8(output.stdout).context("Invalid UTF-8 in dmidecode output")?;

        let mut dmi_data = HashMap::new();
        let mut current_section = String::new();

        for line in output_str.lines() {
            let line = line.trim();
            if line.ends_with(" Information") || line.ends_with(" Configuration") {
                current_section = line.to_string();
            } else if line.contains(": ") && !current_section.is_empty() {
                let parts: Vec<&str> = line.splitn(2, ": ").collect();
                if parts.len() == 2 {
                    let key = format!("{}.{}", current_section, parts[0]);
                    let value = parts[1].to_string();
                    if !value.is_empty()
                        && value != "Not Specified"
                        && value != "To be filled by O.E.M."
                    {
                        dmi_data.insert(key, value);
                    }
                }
            }
        }

        Ok(dmi_data)
    }

    fn parse_memory_modules() -> Result<Vec<MemoryModule>> {
        // First check if dmidecode exists
        let dmidecode_exists = Command::new("which")
            .arg("dmidecode")
            .stdout(std::process::Stdio::null())
            .status()
            .is_ok_and(|s| s.success());

        if !dmidecode_exists {
            tracing::warn!("dmidecode not found, cannot get detailed memory module information");
            return Ok(Vec::new());
        }

        // Similar approach to parse_dmidecode - try with sudo first if available
        let output = if Command::new("which")
            .arg("sudo")
            .stdout(std::process::Stdio::null())
            .status()
            .is_ok_and(|s| s.success())
        {
            let sudo_output = Command::new("sudo")
                .args(["-n", "dmidecode", "-t", "memory"])
                .output();

            match sudo_output {
                Ok(output) if output.status.success() => output,
                _ => Command::new("dmidecode")
                    .args(["-t", "memory"])
                    .output()
                    .context(
                        "Failed to run dmidecode for memory (tried both sudo and direct execution)",
                    )?,
            }
        } else {
            Command::new("dmidecode")
                .args(["-t", "memory"])
                .output()
                .context("Failed to run dmidecode for memory")?
        };

        if !output.status.success() {
            // Return empty vec instead of failing
            tracing::warn!("dmidecode failed for memory modules, returning empty list");
            return Ok(Vec::new());
        }

        let output_str =
            String::from_utf8(output.stdout).context("Invalid UTF-8 in dmidecode memory output")?;

        let mut modules = Vec::new();
        let mut current_module = HashMap::new();

        for line in output_str.lines() {
            let line = line.trim();
            if line.starts_with("Memory Device") {
                if !current_module.is_empty() {
                    if let Some(module) = Self::parse_memory_module(&current_module) {
                        modules.push(module);
                    }
                    current_module.clear();
                }
            } else if line.contains(": ") {
                let parts: Vec<&str> = line.splitn(2, ": ").collect();
                if parts.len() == 2 {
                    current_module.insert(parts[0].to_string(), parts[1].to_string());
                }
            }
        }

        if !current_module.is_empty() {
            if let Some(module) = Self::parse_memory_module(&current_module) {
                modules.push(module);
            }
        }

        Ok(modules)
    }

    fn parse_memory_module(data: &HashMap<String, String>) -> Option<MemoryModule> {
        let size_str = data.get("Size")?;
        if size_str == "No Module Installed" {
            return None;
        }

        let size_mb = if size_str.ends_with(" MB") {
            size_str.trim_end_matches(" MB").parse().ok()?
        } else if size_str.ends_with(" GB") {
            size_str.trim_end_matches(" GB").parse::<u32>().ok()? * 1024
        } else {
            return None;
        };

        let unknown_string = "Unknown".to_string();
        let speed_str = data.get("Speed").unwrap_or(&unknown_string);
        let speed_mhz = if speed_str.ends_with(" MT/s") {
            speed_str.trim_end_matches(" MT/s").parse().unwrap_or(0)
        } else {
            0
        };

        Some(MemoryModule {
            size_mb,
            speed_mhz,
            memory_type: data.get("Type").unwrap_or(&"Unknown".to_string()).clone(),
            manufacturer: data
                .get("Manufacturer")
                .unwrap_or(&"Unknown".to_string())
                .clone(),
            part_number: data
                .get("Part Number")
                .unwrap_or(&"Unknown".to_string())
                .clone(),
            serial_number: data.get("Serial Number").cloned(),
        })
    }

    fn get_cpu_temperature() -> Option<f32> {
        // Try different methods to get CPU temperature
        if let Ok(temp_str) = fs::read_to_string("/sys/class/thermal/thermal_zone0/temp") {
            if let Ok(temp_millicelsius) = temp_str.trim().parse::<i32>() {
                return Some(temp_millicelsius as f32 / 1000.0);
            }
        }

        // Try sensors command
        if let Ok(output) = Command::new("sensors").arg("-A").output() {
            let output_str = String::from_utf8_lossy(&output.stdout);
            for line in output_str.lines() {
                if line.contains("°C") && (line.contains("Core") || line.contains("CPU")) {
                    if let Some(temp_part) = line.split_whitespace().find(|s| s.contains("°C")) {
                        if let Some(temp_str) = temp_part.split('°').next() {
                            if let Ok(temp) = temp_str.parse::<f32>() {
                                return Some(temp);
                            }
                        }
                    }
                }
            }
        }

        None
    }

    fn test_connectivity() -> Result<ConnectivityTest> {
        let can_reach_internet = Command::new("ping")
            .args(["-c", "1", "-W", "5", "8.8.8.8"])
            .stdout(std::process::Stdio::null())
            .stderr(std::process::Stdio::null())
            .status()
            .is_ok_and(|status| status.success());

        let dns_resolution_working = Command::new("nslookup")
            .arg("google.com")
            .stdout(std::process::Stdio::null())
            .stderr(std::process::Stdio::null())
            .status()
            .is_ok_and(|status| status.success());

        let latency_ms = if can_reach_internet {
            Command::new("ping")
                .args(["-c", "1", "8.8.8.8"])
                .output()
                .ok()
                .and_then(|output| {
                    let output_str = String::from_utf8(output.stdout).ok()?;
                    for line in output_str.lines() {
                        if line.contains("time=") {
                            let time_part = line.split("time=").nth(1)?;
                            let time_str = time_part.split_whitespace().next()?;
                            return time_str.parse::<f32>().ok().map(|t| t as u32);
                        }
                    }
                    None
                })
        } else {
            None
        };

        Ok(ConnectivityTest {
            can_reach_internet,
            dns_resolution_working,
            latency_ms,
        })
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_dmidecode_parser() {
        let _sample_output = r#"BIOS Information
	Vendor: American Megatrends Inc.
	Version: F4
	Release Date: 03/14/2018

Base Board Information
	Manufacturer: Gigabyte Technology Co., Ltd.
	Product Name: Z370 AORUS Gaming 7
	Version: x.x
	Serial Number: 123456789
	Asset Tag: Default string"#;

        // This would require refactoring parse_dmidecode to accept input
        // For now, just test that the collection doesn't panic
        let _ = SystemInfoCollector::collect_cpu_info();
    }

    #[test]
    fn test_memory_module_parsing() {
        let mut data = HashMap::new();
        data.insert("Size".to_string(), "8192 MB".to_string());
        data.insert("Speed".to_string(), "3200 MT/s".to_string());
        data.insert("Type".to_string(), "DDR4".to_string());
        data.insert("Manufacturer".to_string(), "Corsair".to_string());
        data.insert("Part Number".to_string(), "CMK16GX4M2B3200C16".to_string());

        let module = SystemInfoCollector::parse_memory_module(&data).unwrap();
        assert_eq!(module.size_mb, 8192);
        assert_eq!(module.speed_mhz, 3200);
        assert_eq!(module.memory_type, "DDR4");
        assert_eq!(module.manufacturer, "Corsair");
    }

    #[test]
    fn test_empty_memory_module() {
        let mut data = HashMap::new();
        data.insert("Size".to_string(), "No Module Installed".to_string());

        let module = SystemInfoCollector::parse_memory_module(&data);
        assert!(module.is_none());
    }

    #[test]
    fn test_gb_memory_module() {
        let mut data = HashMap::new();
        data.insert("Size".to_string(), "16 GB".to_string());
        data.insert("Type".to_string(), "DDR4".to_string());
        data.insert("Manufacturer".to_string(), "Test".to_string());
        data.insert("Part Number".to_string(), "TEST123".to_string());

        let module = SystemInfoCollector::parse_memory_module(&data).unwrap();
        assert_eq!(module.size_mb, 16384); // 16 * 1024
    }
}
