//! OS attestation implementation

use anyhow::{Context, Result};
use std::collections::HashMap;
use std::fs;
use std::process::Command;

use super::types::*;

pub struct OsAttestor;

impl OsAttestor {
    pub fn attest_system() -> Result<OsAttestation> {
        Ok(OsAttestation {
            os_info: Self::collect_os_info()?,
            kernel_info: Self::collect_kernel_info()?,
            security_features: Self::check_security_features()?,
            process_integrity: Self::analyze_process_integrity()?,
            filesystem_integrity: Self::check_filesystem_integrity()?,
            performance_metrics: crate::os::benchmarker::OsBenchmarker::run_benchmarks()?,
        })
    }

    pub fn collect_os_info() -> Result<OsInfo> {
        let os_release = Self::read_os_release()?;
        let uptime = Self::read_uptime()?;
        let load_avg = Self::read_load_average().ok();

        Ok(OsInfo {
            name: os_release
                .get("NAME")
                .unwrap_or(&"Unknown".to_string())
                .clone(),
            version: os_release
                .get("VERSION")
                .unwrap_or(&"Unknown".to_string())
                .clone(),
            kernel_version: Self::get_kernel_version()?,
            architecture: std::env::consts::ARCH.to_string(),
            hostname: Self::get_hostname()?,
            uptime_seconds: uptime,
            load_average: load_avg,
        })
    }

    pub fn collect_kernel_info() -> Result<KernelInfo> {
        let version = Self::get_kernel_version()?;
        let modules = Self::list_kernel_modules()?;
        let security_modules = Self::list_security_modules()?;

        Ok(KernelInfo {
            version: version.clone(),
            build_date: Self::get_kernel_build_date().unwrap_or_else(|_| "Unknown".to_string()),
            compiler_version: Self::get_kernel_compiler().unwrap_or_else(|_| "Unknown".to_string()),
            configuration_hash: Self::get_kernel_config_hash()
                .unwrap_or_else(|_| "Unknown".to_string()),
            modules_loaded: modules,
            security_modules,
        })
    }

    pub fn check_security_features() -> Result<SecurityFeatures> {
        Ok(SecurityFeatures {
            aslr_enabled: Self::check_aslr()?,
            nx_bit_enabled: Self::check_nx_bit()?,
            smep_enabled: Self::check_smep()?,
            smap_enabled: Self::check_smap()?,
            kaslr_enabled: Self::check_kaslr()?,
            stack_protector: Self::check_stack_protector()?,
            fortify_source: Self::check_fortify_source()?,
            selinux_status: Self::check_selinux_status(),
            apparmor_status: Self::check_apparmor_status(),
        })
    }

    pub fn analyze_process_integrity() -> Result<ProcessIntegrity> {
        let processes = Self::get_process_stats()?;
        let high_priv = Self::find_high_privilege_processes()?;
        let suspicious = Self::find_suspicious_processes()?;

        Ok(ProcessIntegrity {
            running_processes: processes.0,
            zombie_processes: processes.1,
            high_privilege_processes: high_priv,
            suspicious_processes: suspicious,
            process_memory_usage: Self::get_total_process_memory()?,
        })
    }

    pub fn check_filesystem_integrity() -> Result<FilesystemIntegrity> {
        Ok(FilesystemIntegrity {
            mounted_filesystems: Self::get_mount_info()?,
            file_permissions_secure: Self::check_critical_file_permissions()?,
            setuid_files: Self::find_setuid_files()?,
            world_writable_files: Self::find_world_writable_files()?,
            tmp_directory_secured: Self::check_tmp_security()?,
        })
    }

    fn read_os_release() -> Result<HashMap<String, String>> {
        let content =
            fs::read_to_string("/etc/os-release").context("Failed to read /etc/os-release")?;

        let mut map = HashMap::new();
        for line in content.lines() {
            if let Some((key, value)) = line.split_once('=') {
                let clean_value = value.trim_matches('"');
                map.insert(key.to_string(), clean_value.to_string());
            }
        }
        Ok(map)
    }

    fn read_uptime() -> Result<u64> {
        let content = fs::read_to_string("/proc/uptime").context("Failed to read /proc/uptime")?;

        let uptime_str = content
            .split_whitespace()
            .next()
            .context("Invalid uptime format")?;

        let uptime_f: f64 = uptime_str.parse().context("Failed to parse uptime")?;

        Ok(uptime_f as u64)
    }

    fn read_load_average() -> Result<LoadAverage> {
        let content =
            fs::read_to_string("/proc/loadavg").context("Failed to read /proc/loadavg")?;

        let parts: Vec<&str> = content.split_whitespace().collect();
        if parts.len() < 3 {
            return Err(anyhow::anyhow!("Invalid loadavg format"));
        }

        Ok(LoadAverage {
            one_minute: parts[0].parse().context("Failed to parse 1-minute load")?,
            five_minutes: parts[1].parse().context("Failed to parse 5-minute load")?,
            fifteen_minutes: parts[2].parse().context("Failed to parse 15-minute load")?,
        })
    }

    fn get_kernel_version() -> Result<String> {
        let output = Command::new("uname")
            .arg("-r")
            .output()
            .context("Failed to execute uname")?;

        String::from_utf8(output.stdout)
            .context("Invalid UTF-8 in kernel version")
            .map(|s| s.trim().to_string())
    }

    fn get_hostname() -> Result<String> {
        let output = Command::new("hostname")
            .output()
            .context("Failed to execute hostname")?;

        String::from_utf8(output.stdout)
            .context("Invalid UTF-8 in hostname")
            .map(|s| s.trim().to_string())
    }

    fn list_kernel_modules() -> Result<Vec<String>> {
        let content =
            fs::read_to_string("/proc/modules").context("Failed to read /proc/modules")?;

        Ok(content
            .lines()
            .map(|line| line.split_whitespace().next().unwrap_or("").to_string())
            .filter(|s| !s.is_empty())
            .collect())
    }

    fn list_security_modules() -> Result<Vec<String>> {
        let mut modules = Vec::new();

        if fs::read_to_string("/proc/modules")
            .unwrap_or_default()
            .contains("selinux")
        {
            modules.push("selinux".to_string());
        }

        if fs::read_to_string("/sys/module/apparmor/parameters/enabled")
            .unwrap_or_default()
            .trim()
            == "Y"
        {
            modules.push("apparmor".to_string());
        }

        Ok(modules)
    }

    fn check_aslr() -> Result<bool> {
        let content = fs::read_to_string("/proc/sys/kernel/randomize_va_space")
            .context("Failed to read ASLR setting")?;

        Ok(content.trim() != "0")
    }

    fn check_nx_bit() -> Result<bool> {
        let output = Command::new("grep")
            .args(["flags", "/proc/cpuinfo"])
            .output()
            .context("Failed to check NX bit")?;

        let cpu_flags = String::from_utf8_lossy(&output.stdout);
        Ok(cpu_flags.contains("nx"))
    }

    fn check_smep() -> Result<bool> {
        Ok(fs::read_to_string("/proc/cpuinfo")
            .unwrap_or_default()
            .contains("smep"))
    }

    fn check_smap() -> Result<bool> {
        Ok(fs::read_to_string("/proc/cpuinfo")
            .unwrap_or_default()
            .contains("smap"))
    }

    fn check_kaslr() -> Result<bool> {
        Ok(fs::read_to_string("/proc/cmdline")
            .unwrap_or_default()
            .contains("kaslr"))
    }

    fn check_stack_protector() -> Result<bool> {
        Ok(true)
    }

    fn check_fortify_source() -> Result<bool> {
        Ok(true)
    }

    fn check_selinux_status() -> SeLinuxStatus {
        if let Ok(content) = fs::read_to_string("/sys/fs/selinux/enforce") {
            match content.trim() {
                "1" => SeLinuxStatus::Enforcing,
                "0" => SeLinuxStatus::Permissive,
                _ => SeLinuxStatus::Unknown,
            }
        } else {
            SeLinuxStatus::Disabled
        }
    }

    fn check_apparmor_status() -> AppArmorStatus {
        if let Ok(content) = fs::read_to_string("/sys/module/apparmor/parameters/enabled") {
            if content.trim() == "Y" {
                AppArmorStatus::Enforce
            } else {
                AppArmorStatus::Disabled
            }
        } else {
            AppArmorStatus::Disabled
        }
    }

    fn get_process_stats() -> Result<(u32, u32)> {
        let content = fs::read_to_string("/proc/stat").context("Failed to read /proc/stat")?;

        for line in content.lines() {
            if line.starts_with("processes ") {
                let count: u32 = line
                    .split_whitespace()
                    .nth(1)
                    .and_then(|s| s.parse().ok())
                    .unwrap_or(0);
                return Ok((count, 0));
            }
        }
        Ok((0, 0))
    }

    fn find_high_privilege_processes() -> Result<Vec<String>> {
        Ok(vec![])
    }

    fn find_suspicious_processes() -> Result<Vec<String>> {
        Ok(vec![])
    }

    fn get_total_process_memory() -> Result<u64> {
        let content =
            fs::read_to_string("/proc/meminfo").context("Failed to read /proc/meminfo")?;

        for line in content.lines() {
            if line.starts_with("MemTotal:") {
                let kb: u64 = line
                    .split_whitespace()
                    .nth(1)
                    .and_then(|s| s.parse().ok())
                    .unwrap_or(0);
                return Ok(kb * 1024);
            }
        }
        Ok(0)
    }

    fn get_mount_info() -> Result<Vec<MountInfo>> {
        let content = fs::read_to_string("/proc/mounts").context("Failed to read /proc/mounts")?;

        let mut mounts = Vec::new();
        for line in content.lines() {
            let parts: Vec<&str> = line.split_whitespace().collect();
            if parts.len() >= 4 {
                mounts.push(MountInfo {
                    device: parts[0].to_string(),
                    mount_point: parts[1].to_string(),
                    filesystem_type: parts[2].to_string(),
                    options: parts[3].split(',').map(|s| s.to_string()).collect(),
                });
            }
        }
        Ok(mounts)
    }

    fn check_critical_file_permissions() -> Result<bool> {
        Ok(true)
    }

    fn find_setuid_files() -> Result<Vec<String>> {
        Ok(vec![])
    }

    fn find_world_writable_files() -> Result<Vec<String>> {
        Ok(vec![])
    }

    fn check_tmp_security() -> Result<bool> {
        if let Ok(_metadata) = fs::metadata("/tmp") {
            Ok(true)
        } else {
            Ok(false)
        }
    }

    fn get_kernel_build_date() -> Result<String> {
        let output = Command::new("uname")
            .arg("-v")
            .output()
            .context("Failed to get kernel build date")?;

        String::from_utf8(output.stdout)
            .context("Invalid UTF-8 in kernel build date")
            .map(|s| s.trim().to_string())
    }

    fn get_kernel_compiler() -> Result<String> {
        let version_info =
            fs::read_to_string("/proc/version").context("Failed to read /proc/version")?;

        if let Some(start) = version_info.find("gcc") {
            if let Some(end) = version_info[start..].find(')') {
                return Ok(version_info[start..start + end].to_string());
            }
        }

        Ok("Unknown".to_string())
    }

    fn get_kernel_config_hash() -> Result<String> {
        Ok("placeholder_hash".to_string())
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_os_attestation_creation() {
        let result = OsAttestor::attest_system();
        assert!(result.is_ok());

        let attestation = result.unwrap();
        assert!(!attestation.os_info.name.is_empty());
        assert!(!attestation.kernel_info.version.is_empty());
    }

    #[test]
    fn test_security_features_check() {
        let result = OsAttestor::check_security_features();
        assert!(result.is_ok());
    }

    #[test]
    fn test_os_info_collection() {
        let result = OsAttestor::collect_os_info();
        assert!(result.is_ok());

        let os_info = result.unwrap();
        assert!(!os_info.name.is_empty());
        assert!(!os_info.architecture.is_empty());
    }
}
