use serde::{Deserialize, Serialize};

#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct SystemInfo {
    pub motherboard: MotherboardInfo,
    pub cpu: CpuInfo,
    pub memory: MemoryInfo,
    pub storage: Vec<StorageInfo>,
    pub network: NetworkInfo,
    pub benchmarks: BenchmarkResults,
}

#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct MotherboardInfo {
    pub manufacturer: String,
    pub product_name: String,
    pub version: String,
    pub serial_number: Option<String>,
    pub asset_tag: Option<String>,
    pub bios_vendor: String,
    pub bios_version: String,
    pub bios_date: String,
}

#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct CpuInfo {
    pub brand: String,
    pub vendor_id: String,
    pub cores: usize,
    pub threads: usize,
    pub frequency_mhz: u64,
    pub architecture: String,
    pub features: Vec<String>,
    pub temperature: Option<f32>,
}

#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct MemoryInfo {
    pub total_bytes: u64,
    pub available_bytes: u64,
    pub used_bytes: u64,
    pub swap_total_bytes: u64,
    pub swap_used_bytes: u64,
    pub memory_modules: Vec<MemoryModule>,
}

#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct MemoryModule {
    pub size_mb: u32,
    pub speed_mhz: u32,
    pub memory_type: String,
    pub manufacturer: String,
    pub part_number: String,
    pub serial_number: Option<String>,
}

#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct StorageInfo {
    pub name: String,
    pub total_space: u64,
    pub available_space: u64,
    pub mount_point: String,
    pub file_system: String,
    pub disk_type: String,
}

#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct NetworkInfo {
    pub interfaces: Vec<NetworkInterface>,
    pub connectivity_test: ConnectivityTest,
}

#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct NetworkInterface {
    pub name: String,
    pub mac_address: String,
    pub ip_addresses: Vec<String>,
    pub is_up: bool,
    pub speed_mbps: Option<u32>,
}

#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct ConnectivityTest {
    pub can_reach_internet: bool,
    pub dns_resolution_working: bool,
    pub latency_ms: Option<u32>,
}

#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct BenchmarkResults {
    pub cpu_benchmark_score: f64,
    pub memory_bandwidth_mbps: f64,
    pub disk_sequential_read_mbps: f64,
    pub disk_sequential_write_mbps: f64,
    pub network_throughput_mbps: Option<f64>,
}

impl SystemInfo {
    pub fn total_memory_gb(&self) -> f64 {
        self.memory.total_bytes as f64 / (1024.0 * 1024.0 * 1024.0)
    }

    pub fn memory_utilization_percent(&self) -> f64 {
        if self.memory.total_bytes == 0 {
            0.0
        } else {
            (self.memory.used_bytes as f64 / self.memory.total_bytes as f64) * 100.0
        }
    }

    pub fn total_storage_gb(&self) -> f64 {
        self.storage.iter().map(|s| s.total_space).sum::<u64>() as f64 / (1024.0 * 1024.0 * 1024.0)
    }

    pub fn active_network_interfaces(&self) -> Vec<&NetworkInterface> {
        self.network.interfaces.iter().filter(|i| i.is_up).collect()
    }
}

impl MemoryInfo {
    pub fn utilization_percent(&self) -> f64 {
        if self.total_bytes == 0 {
            0.0
        } else {
            (self.used_bytes as f64 / self.total_bytes as f64) * 100.0
        }
    }

    pub fn total_gb(&self) -> f64 {
        self.total_bytes as f64 / (1024.0 * 1024.0 * 1024.0)
    }

    pub fn available_gb(&self) -> f64 {
        self.available_bytes as f64 / (1024.0 * 1024.0 * 1024.0)
    }

    pub fn used_gb(&self) -> f64 {
        self.used_bytes as f64 / (1024.0 * 1024.0 * 1024.0)
    }
}

impl StorageInfo {
    pub fn total_gb(&self) -> f64 {
        self.total_space as f64 / (1024.0 * 1024.0 * 1024.0)
    }

    pub fn available_gb(&self) -> f64 {
        self.available_space as f64 / (1024.0 * 1024.0 * 1024.0)
    }

    pub fn used_gb(&self) -> f64 {
        (self.total_space - self.available_space) as f64 / (1024.0 * 1024.0 * 1024.0)
    }

    pub fn utilization_percent(&self) -> f64 {
        if self.total_space == 0 {
            0.0
        } else {
            ((self.total_space - self.available_space) as f64 / self.total_space as f64) * 100.0
        }
    }
}

impl CpuInfo {
    pub fn has_feature(&self, feature: &str) -> bool {
        self.features.iter().any(|f| f == feature)
    }

    pub fn supports_avx(&self) -> bool {
        self.has_feature("avx") || self.has_feature("avx2")
    }

    pub fn supports_sse(&self) -> bool {
        self.has_feature("sse")
            || self.has_feature("sse2")
            || self.has_feature("sse3")
            || self.has_feature("sse4_1")
            || self.has_feature("sse4_2")
    }

    pub fn thread_per_core_ratio(&self) -> f64 {
        if self.cores == 0 {
            0.0
        } else {
            self.threads as f64 / self.cores as f64
        }
    }
}
