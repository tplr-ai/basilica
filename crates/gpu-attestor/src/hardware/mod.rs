//! Hardware information collection and benchmarking module
//!
//! This module provides comprehensive system hardware detection,
//! information gathering, and performance benchmarking capabilities.

pub mod benchmarks;
pub mod collection;
pub mod types;

// Re-export commonly used items
pub use benchmarks::BenchmarkRunner;
pub use collection::SystemInfoCollector;
pub use types::{
    BenchmarkResults, ConnectivityTest, CpuInfo, MemoryInfo, MemoryModule, MotherboardInfo,
    NetworkInfo, NetworkInterface, StorageInfo, SystemInfo,
};

/// Collect all system information including hardware specs and benchmarks
pub fn collect_system_info() -> anyhow::Result<SystemInfo> {
    SystemInfoCollector::collect_all()
}

/// Collect only hardware information without running benchmarks
pub fn collect_hardware_only() -> anyhow::Result<SystemInfo> {
    Ok(SystemInfo {
        motherboard: SystemInfoCollector::collect_motherboard_info()?,
        cpu: SystemInfoCollector::collect_cpu_info()?,
        memory: SystemInfoCollector::collect_memory_info()?,
        storage: SystemInfoCollector::collect_storage_info()?,
        network: SystemInfoCollector::collect_network_info()?,
        benchmarks: BenchmarkResults {
            cpu_benchmark_score: 0.0,
            memory_bandwidth_mbps: 0.0,
            disk_sequential_read_mbps: 0.0,
            disk_sequential_write_mbps: 0.0,
            network_throughput_mbps: None,
        },
    })
}

/// Run quick benchmarks (faster, less comprehensive)
pub fn quick_benchmark() -> anyhow::Result<BenchmarkResults> {
    BenchmarkRunner::quick_benchmark()
}

/// Run comprehensive benchmarks (slower, more thorough)
pub fn comprehensive_benchmark() -> anyhow::Result<BenchmarkResults> {
    BenchmarkRunner::comprehensive_benchmark()
}

/// Check if system meets minimum requirements for GPU attestation
pub fn check_system_requirements(system_info: &SystemInfo) -> SystemRequirementsCheck {
    SystemRequirementsCheck {
        has_sufficient_memory: system_info.memory.total_gb() >= 4.0, // Minimum 4GB
        has_sufficient_storage: system_info.total_storage_gb() >= 10.0, // Minimum 10GB
        has_docker: false,
        docker_running: false,
        has_network_connectivity: system_info.network.connectivity_test.can_reach_internet,
        dns_working: system_info.network.connectivity_test.dns_resolution_working,
        cpu_cores_sufficient: system_info.cpu.cores >= 2, // Minimum 2 cores
        meets_all_requirements: false,                    // Will be calculated
    }
}

#[derive(Debug, Clone)]
pub struct SystemRequirementsCheck {
    pub has_sufficient_memory: bool,
    pub has_sufficient_storage: bool,
    pub has_docker: bool,
    pub docker_running: bool,
    pub has_network_connectivity: bool,
    pub dns_working: bool,
    pub cpu_cores_sufficient: bool,
    pub meets_all_requirements: bool,
}

impl SystemRequirementsCheck {
    pub fn calculate_requirements(&mut self) {
        self.meets_all_requirements = self.has_sufficient_memory
            && self.has_sufficient_storage
            && self.has_docker
            && self.docker_running
            && self.has_network_connectivity
            && self.dns_working
            && self.cpu_cores_sufficient;
    }

    pub fn get_missing_requirements(&self) -> Vec<&'static str> {
        let mut missing = Vec::new();

        if !self.has_sufficient_memory {
            missing.push("Insufficient memory (minimum 4GB required)");
        }
        if !self.has_sufficient_storage {
            missing.push("Insufficient storage (minimum 10GB required)");
        }
        if !self.has_docker {
            missing.push("Docker not installed");
        }
        if !self.docker_running {
            missing.push("Docker not running");
        }
        if !self.has_network_connectivity {
            missing.push("No internet connectivity");
        }
        if !self.dns_working {
            missing.push("DNS resolution not working");
        }
        if !self.cpu_cores_sufficient {
            missing.push("Insufficient CPU cores (minimum 2 cores required)");
        }

        missing
    }
}

/// Get system summary for quick overview
pub fn get_system_summary(system_info: &SystemInfo) -> SystemSummary {
    SystemSummary {
        cpu_brand: system_info.cpu.brand.clone(),
        cpu_cores: system_info.cpu.cores,
        cpu_threads: system_info.cpu.threads,
        total_memory_gb: system_info.memory.total_gb(),
        total_storage_gb: system_info.total_storage_gb(),
        active_network_interfaces: system_info.active_network_interfaces().len(),
        docker_status: "Not available".to_string(),
        internet_connectivity: system_info.network.connectivity_test.can_reach_internet,
    }
}

#[derive(Debug, Clone)]
pub struct SystemSummary {
    pub cpu_brand: String,
    pub cpu_cores: usize,
    pub cpu_threads: usize,
    pub total_memory_gb: f64,
    pub total_storage_gb: f64,
    pub active_network_interfaces: usize,
    pub docker_status: String,
    pub internet_connectivity: bool,
}
