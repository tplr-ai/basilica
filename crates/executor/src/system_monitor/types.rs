//! System monitoring data types and structures

use serde::{Deserialize, Serialize};

/// System information snapshot
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct SystemInfo {
    pub cpu: CpuInfo,
    pub memory: MemoryInfo,
    pub gpu: Vec<GpuInfo>,
    pub disk: Vec<DiskInfo>,
    pub network: NetworkInfo,
    pub system: BasicSystemInfo,
    pub timestamp: i64,
}

/// CPU information
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct CpuInfo {
    pub usage_percent: f32,
    pub cores: usize,
    pub frequency_mhz: u64,
    pub model: String,
    pub vendor: String,
    pub temperature_celsius: Option<f32>,
}

/// Memory information
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct MemoryInfo {
    pub total_bytes: u64,
    pub used_bytes: u64,
    pub available_bytes: u64,
    pub usage_percent: f32,
    pub swap_total_bytes: u64,
    pub swap_used_bytes: u64,
}

/// GPU information
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct GpuInfo {
    pub index: u32,
    pub name: String,
    pub memory_total_bytes: u64,
    pub memory_used_bytes: u64,
    pub memory_usage_percent: f32,
    pub utilization_percent: f32,
    pub temperature_celsius: f32,
    pub power_usage_watts: f32,
    pub driver_version: String,
    pub cuda_version: Option<String>,
}

/// Disk information
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct DiskInfo {
    pub name: String,
    pub mount_point: String,
    pub total_bytes: u64,
    pub used_bytes: u64,
    pub available_bytes: u64,
    pub usage_percent: f32,
    pub filesystem: String,
}

/// Network information
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct NetworkInfo {
    pub interfaces: Vec<NetworkInterface>,
    pub total_bytes_sent: u64,
    pub total_bytes_received: u64,
}

/// Network interface information
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct NetworkInterface {
    pub name: String,
    pub bytes_sent: u64,
    pub bytes_received: u64,
    pub packets_sent: u64,
    pub packets_received: u64,
    pub errors_sent: u64,
    pub errors_received: u64,
    pub is_up: bool,
}

/// Basic system information
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct BasicSystemInfo {
    pub hostname: String,
    pub os_name: String,
    pub os_version: String,
    pub kernel_version: String,
    pub uptime_seconds: u64,
    pub boot_time: u64,
    pub load_average: Vec<f64>,
}

/// System profile for registration
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct SystemProfile {
    pub cpu: CpuProfile,
    pub memory: MemoryProfile,
    pub storage: StorageProfile,
    pub os: OsProfile,
    pub docker: DockerProfile,
}

/// CPU profile
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct CpuProfile {
    pub model: String,
    pub cores: usize,
    pub vendor: String,
}

/// Memory profile
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct MemoryProfile {
    pub total_gb: f32,
}

/// Storage profile
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct StorageProfile {
    pub total_gb: f32,
}

/// OS profile
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct OsProfile {
    pub os_type: String,
    pub version: String,
}

/// Docker profile
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct DockerProfile {
    pub version: String,
}

/// Resource information
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct ResourceInfo {
    pub cpu_cores: usize,
    pub memory_mb: u32,
    pub storage_mb: u32,
    pub gpu_count: u32,
    pub gpu_memory_mb: u32,
}

/// Resource utilization
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct ResourceUtilization {
    pub cpu_percent: f32,
    pub memory_percent: f32,
    pub disk_percent: f32,
    pub gpu_percent: f32,
    pub gpu_memory_percent: f32,
    pub network_bandwidth_mbps: f32,
}
