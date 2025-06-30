//! OS attestation and benchmarking type definitions

use serde::{Deserialize, Serialize};

#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct OsAttestation {
    pub os_info: OsInfo,
    pub kernel_info: KernelInfo,
    pub security_features: SecurityFeatures,
    pub process_integrity: ProcessIntegrity,
    pub filesystem_integrity: FilesystemIntegrity,
    pub performance_metrics: OsPerformanceMetrics,
}

#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct OsInfo {
    pub name: String,
    pub version: String,
    pub kernel_version: String,
    pub architecture: String,
    pub hostname: String,
    pub uptime_seconds: u64,
    pub load_average: Option<LoadAverage>,
}

#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct LoadAverage {
    pub one_minute: f64,
    pub five_minutes: f64,
    pub fifteen_minutes: f64,
}

#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct KernelInfo {
    pub version: String,
    pub build_date: String,
    pub compiler_version: String,
    pub configuration_hash: String,
    pub modules_loaded: Vec<String>,
    pub security_modules: Vec<String>,
}

#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct SecurityFeatures {
    pub aslr_enabled: bool,
    pub nx_bit_enabled: bool,
    pub smep_enabled: bool,
    pub smap_enabled: bool,
    pub kaslr_enabled: bool,
    pub stack_protector: bool,
    pub fortify_source: bool,
    pub selinux_status: SeLinuxStatus,
    pub apparmor_status: AppArmorStatus,
}

#[derive(Debug, Clone, Serialize, Deserialize)]
pub enum SeLinuxStatus {
    Disabled,
    Permissive,
    Enforcing,
    Unknown,
}

#[derive(Debug, Clone, Serialize, Deserialize)]
pub enum AppArmorStatus {
    Disabled,
    Complain,
    Enforce,
    Unknown,
}

#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct ProcessIntegrity {
    pub running_processes: u32,
    pub zombie_processes: u32,
    pub high_privilege_processes: Vec<String>,
    pub suspicious_processes: Vec<String>,
    pub process_memory_usage: u64,
}

#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct FilesystemIntegrity {
    pub mounted_filesystems: Vec<MountInfo>,
    pub file_permissions_secure: bool,
    pub setuid_files: Vec<String>,
    pub world_writable_files: Vec<String>,
    pub tmp_directory_secured: bool,
}

#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct MountInfo {
    pub device: String,
    pub mount_point: String,
    pub filesystem_type: String,
    pub options: Vec<String>,
}

#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct OsPerformanceMetrics {
    pub context_switches_per_second: f64,
    pub interrupts_per_second: f64,
    pub cpu_idle_percentage: f64,
    pub memory_fragmentation: f64,
    pub io_wait_percentage: f64,
    pub system_call_latency_us: f64,
    pub scheduler_latency_us: f64,
    pub filesystem_latency_us: f64,
}

impl OsAttestation {
    pub fn is_secure(&self) -> bool {
        self.security_features.aslr_enabled
            && self.security_features.nx_bit_enabled
            && self.filesystem_integrity.file_permissions_secure
            && self.filesystem_integrity.tmp_directory_secured
    }

    pub fn has_good_performance(&self) -> bool {
        self.performance_metrics.cpu_idle_percentage > 80.0
            && self.performance_metrics.memory_fragmentation < 10.0
            && self.performance_metrics.io_wait_percentage < 5.0
            && self.performance_metrics.system_call_latency_us < 10.0
    }

    pub fn security_score(&self) -> u8 {
        let mut score = 0u8;

        if self.security_features.aslr_enabled {
            score += 15;
        }
        if self.security_features.nx_bit_enabled {
            score += 15;
        }
        if self.security_features.smep_enabled {
            score += 10;
        }
        if self.security_features.smap_enabled {
            score += 10;
        }
        if self.security_features.kaslr_enabled {
            score += 10;
        }
        if self.security_features.stack_protector {
            score += 10;
        }
        if self.security_features.fortify_source {
            score += 10;
        }

        match self.security_features.selinux_status {
            SeLinuxStatus::Enforcing => score += 10,
            SeLinuxStatus::Permissive => score += 5,
            _ => {}
        }

        match self.security_features.apparmor_status {
            AppArmorStatus::Enforce => score += 10,
            AppArmorStatus::Complain => score += 5,
            _ => {}
        }

        score
    }

    pub fn performance_score(&self) -> u8 {
        let mut score = 0u8;

        if self.performance_metrics.cpu_idle_percentage > 90.0 {
            score += 25;
        } else if self.performance_metrics.cpu_idle_percentage > 80.0 {
            score += 20;
        } else if self.performance_metrics.cpu_idle_percentage > 70.0 {
            score += 15;
        }

        if self.performance_metrics.memory_fragmentation < 5.0 {
            score += 25;
        } else if self.performance_metrics.memory_fragmentation < 10.0 {
            score += 20;
        } else if self.performance_metrics.memory_fragmentation < 15.0 {
            score += 15;
        }

        if self.performance_metrics.io_wait_percentage < 2.0 {
            score += 25;
        } else if self.performance_metrics.io_wait_percentage < 5.0 {
            score += 20;
        } else if self.performance_metrics.io_wait_percentage < 10.0 {
            score += 15;
        }

        if self.performance_metrics.system_call_latency_us < 5.0 {
            score += 25;
        } else if self.performance_metrics.system_call_latency_us < 10.0 {
            score += 20;
        } else if self.performance_metrics.system_call_latency_us < 20.0 {
            score += 15;
        }

        score
    }
}

impl OsPerformanceMetrics {
    pub fn new() -> Self {
        Self {
            context_switches_per_second: 0.0,
            interrupts_per_second: 0.0,
            cpu_idle_percentage: 0.0,
            memory_fragmentation: 0.0,
            io_wait_percentage: 0.0,
            system_call_latency_us: 0.0,
            scheduler_latency_us: 0.0,
            filesystem_latency_us: 0.0,
        }
    }
}

impl Default for OsPerformanceMetrics {
    fn default() -> Self {
        Self::new()
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_security_score_calculation() {
        let attestation = OsAttestation {
            os_info: OsInfo {
                name: "Test OS".to_string(),
                version: "1.0".to_string(),
                kernel_version: "5.0.0".to_string(),
                architecture: "x86_64".to_string(),
                hostname: "testhost".to_string(),
                uptime_seconds: 3600,
                load_average: None,
            },
            kernel_info: KernelInfo {
                version: "5.0.0".to_string(),
                build_date: "2024-01-01".to_string(),
                compiler_version: "gcc-9".to_string(),
                configuration_hash: "test_hash".to_string(),
                modules_loaded: vec![],
                security_modules: vec![],
            },
            security_features: SecurityFeatures {
                aslr_enabled: true,
                nx_bit_enabled: true,
                smep_enabled: true,
                smap_enabled: true,
                kaslr_enabled: true,
                stack_protector: true,
                fortify_source: true,
                selinux_status: SeLinuxStatus::Enforcing,
                apparmor_status: AppArmorStatus::Disabled,
            },
            process_integrity: ProcessIntegrity {
                running_processes: 100,
                zombie_processes: 0,
                high_privilege_processes: vec![],
                suspicious_processes: vec![],
                process_memory_usage: 1024 * 1024 * 1024,
            },
            filesystem_integrity: FilesystemIntegrity {
                mounted_filesystems: vec![],
                file_permissions_secure: true,
                setuid_files: vec![],
                world_writable_files: vec![],
                tmp_directory_secured: true,
            },
            performance_metrics: OsPerformanceMetrics {
                context_switches_per_second: 1000.0,
                interrupts_per_second: 500.0,
                cpu_idle_percentage: 95.0,
                memory_fragmentation: 3.0,
                io_wait_percentage: 1.0,
                system_call_latency_us: 2.0,
                scheduler_latency_us: 1.0,
                filesystem_latency_us: 5.0,
            },
        };

        let security_score = attestation.security_score();
        let performance_score = attestation.performance_score();

        assert!(security_score > 80);
        assert!(performance_score > 80);
        assert!(attestation.is_secure());
        assert!(attestation.has_good_performance());
    }
}
