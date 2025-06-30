//! System profiling service

use super::types::{GrpcResult, SharedExecutorState};
use tracing::info;

/// System profiling handler
pub struct SystemProfileService {
    state: SharedExecutorState,
}

impl SystemProfileService {
    /// Create new system profile service
    pub fn new(state: SharedExecutorState) -> Self {
        Self { state }
    }

    /// Execute system profile
    pub async fn execute_system_profile(&self) -> GrpcResult<String> {
        info!("Executing system profile");

        let system_info = self.state.system_monitor.get_system_info().await?;
        let system_profile = self.state.system_monitor.get_system_profile().await?;

        let full_profile = serde_json::json!({
            "system_profile": {
                "cpu": {
                    "model": system_profile.cpu.model,
                    "cores": system_profile.cpu.cores,
                    "vendor": system_profile.cpu.vendor
                },
                "memory": {
                    "total_gb": system_profile.memory.total_gb
                },
                "storage": {
                    "total_gb": system_profile.storage.total_gb
                },
                "os": {
                    "os_type": system_profile.os.os_type,
                    "version": system_profile.os.version
                },
                "docker": {
                    "version": system_profile.docker.version
                }
            },
            "current_state": {
                "cpu": {
                    "usage_percent": system_info.cpu.usage_percent,
                    "frequency_mhz": system_info.cpu.frequency_mhz,
                    "temperature_celsius": system_info.cpu.temperature_celsius
                },
                "memory": {
                    "usage_percent": system_info.memory.usage_percent,
                    "used_mb": system_info.memory.used_bytes / (1024 * 1024),
                    "available_mb": system_info.memory.available_bytes / (1024 * 1024)
                },
                "gpu": system_info.gpu.iter().map(|gpu| {
                    serde_json::json!({
                        "index": gpu.index,
                        "name": gpu.name,
                        "utilization_percent": gpu.utilization_percent,
                        "memory_usage_percent": gpu.memory_usage_percent,
                        "temperature_celsius": gpu.temperature_celsius,
                        "power_usage_watts": gpu.power_usage_watts
                    })
                }).collect::<Vec<_>>(),
                "disk": system_info.disk.iter().map(|disk| {
                    serde_json::json!({
                        "mount_point": disk.mount_point,
                        "usage_percent": disk.usage_percent,
                        "available_gb": disk.available_bytes / (1024 * 1024 * 1024)
                    })
                }).collect::<Vec<_>>(),
                "uptime_seconds": system_info.system.uptime_seconds,
                "load_average": system_info.system.load_average
            },
            "timestamp": system_info.timestamp
        });

        Ok(full_profile.to_string())
    }

    /// Get basic system information
    pub async fn get_basic_info(&self) -> GrpcResult<String> {
        info!("Getting basic system information");

        let system_info = self.state.system_monitor.get_system_info().await?;
        let basic_info = serde_json::json!({
            "hostname": system_info.system.hostname,
            "os_name": system_info.system.os_name,
            "os_version": system_info.system.os_version,
            "kernel_version": system_info.system.kernel_version,
            "uptime_seconds": system_info.system.uptime_seconds,
            "cpu": {
                "usage_percent": system_info.cpu.usage_percent,
                "cores": system_info.cpu.cores,
                "model": system_info.cpu.model,
                "vendor": system_info.cpu.vendor,
                "frequency_mhz": system_info.cpu.frequency_mhz,
                "temperature_celsius": system_info.cpu.temperature_celsius
            },
            "memory": {
                "usage_percent": system_info.memory.usage_percent,
                "used_bytes": system_info.memory.used_bytes,
                "total_bytes": system_info.memory.total_bytes,
                "available_bytes": system_info.memory.available_bytes,
                "used_mb": system_info.memory.used_bytes / (1024 * 1024),
                "total_mb": system_info.memory.total_bytes / (1024 * 1024)
            },
            "load_average": system_info.system.load_average,
            "timestamp": system_info.timestamp
        });

        Ok(basic_info.to_string())
    }

    /// Get detailed performance metrics
    pub async fn get_performance_metrics(&self) -> GrpcResult<String> {
        info!("Getting performance metrics");

        let system_info = self.state.system_monitor.get_system_info().await?;

        let performance_metrics = serde_json::json!({
            "cpu": {
                "cores": system_info.cpu.cores,
                "usage_percent": system_info.cpu.usage_percent,
                "frequency_mhz": system_info.cpu.frequency_mhz,
                "temperature_celsius": system_info.cpu.temperature_celsius,
                "model": system_info.cpu.model,
                "vendor": system_info.cpu.vendor
            },
            "memory": {
                "total_mb": system_info.memory.total_bytes / (1024 * 1024),
                "used_mb": system_info.memory.used_bytes / (1024 * 1024),
                "available_mb": system_info.memory.available_bytes / (1024 * 1024),
                "usage_percent": system_info.memory.usage_percent,
                "swap_total_mb": system_info.memory.swap_total_bytes / (1024 * 1024),
                "swap_used_mb": system_info.memory.swap_used_bytes / (1024 * 1024)
            },
            "gpu": system_info.gpu.iter().map(|gpu| {
                serde_json::json!({
                    "index": gpu.index,
                    "name": gpu.name,
                    "utilization_percent": gpu.utilization_percent,
                    "memory_used_mb": gpu.memory_used_bytes / (1024 * 1024),
                    "memory_total_mb": gpu.memory_total_bytes / (1024 * 1024),
                    "memory_usage_percent": gpu.memory_usage_percent,
                    "temperature_celsius": gpu.temperature_celsius,
                    "power_usage_watts": gpu.power_usage_watts
                })
            }).collect::<Vec<_>>(),
            "disk": system_info.disk.iter().map(|disk| {
                serde_json::json!({
                    "mount_point": disk.mount_point,
                    "file_system": disk.filesystem,
                    "total_gb": disk.total_bytes / (1024 * 1024 * 1024),
                    "used_gb": disk.used_bytes / (1024 * 1024 * 1024),
                    "available_gb": disk.available_bytes / (1024 * 1024 * 1024),
                    "usage_percent": disk.usage_percent
                })
            }).collect::<Vec<_>>(),
            "network": {
                "interfaces": system_info.network.interfaces.len(),
                "total_bytes_sent": system_info.network.total_bytes_sent,
                "total_bytes_received": system_info.network.total_bytes_received,
                "interfaces_detail": system_info.network.interfaces.iter().map(|iface| {
                    serde_json::json!({
                        "name": iface.name,
                        "bytes_sent": iface.bytes_sent,
                        "bytes_received": iface.bytes_received,
                        "is_up": iface.is_up
                    })
                }).collect::<Vec<_>>()
            },
            "timestamp": system_info.timestamp
        });

        Ok(performance_metrics.to_string())
    }

    /// Get resource availability
    pub async fn get_resource_availability(&self) -> GrpcResult<String> {
        info!("Getting resource availability");

        let system_info = self.state.system_monitor.get_system_info().await?;

        // Calculate available resources
        let cpu_available_percent = 100.0 - system_info.cpu.usage_percent;
        let memory_available_mb = system_info.memory.available_bytes / (1024 * 1024);
        let total_disk_available_gb: u64 = system_info
            .disk
            .iter()
            .map(|d| d.available_bytes / (1024 * 1024 * 1024))
            .sum();

        // Calculate GPU availability
        let gpu_availability: Vec<_> = system_info
            .gpu
            .iter()
            .map(|gpu| {
                let memory_available_mb =
                    (gpu.memory_total_bytes - gpu.memory_used_bytes) / (1024 * 1024);
                let compute_available_percent = 100.0 - gpu.utilization_percent;
                serde_json::json!({
                    "gpu_index": gpu.index,
                    "name": gpu.name,
                    "compute_available_percent": compute_available_percent,
                    "memory_available_mb": memory_available_mb,
                    "memory_total_mb": gpu.memory_total_bytes / (1024 * 1024),
                    "temperature_celsius": gpu.temperature_celsius
                })
            })
            .collect();

        let resource_availability = serde_json::json!({
            "cpu": {
                "available_percent": cpu_available_percent,
                "total_cores": system_info.cpu.cores,
                "frequency_mhz": system_info.cpu.frequency_mhz
            },
            "memory": {
                "available_mb": memory_available_mb,
                "total_mb": system_info.memory.total_bytes / (1024 * 1024),
                "usage_percent": system_info.memory.usage_percent
            },
            "disk": {
                "available_gb": total_disk_available_gb,
                "disks": system_info.disk.iter().map(|d| {
                    serde_json::json!({
                        "mount_point": d.mount_point,
                        "available_gb": d.available_bytes / (1024 * 1024 * 1024),
                        "total_gb": d.total_bytes / (1024 * 1024 * 1024),
                        "usage_percent": d.usage_percent
                    })
                }).collect::<Vec<_>>()
            },
            "gpu": {
                "count": system_info.gpu.len(),
                "devices": gpu_availability
            },
            "network": {
                "interfaces_available": system_info.network.interfaces.iter()
                    .filter(|iface| iface.is_up).count(),
                "total_interfaces": system_info.network.interfaces.len()
            },
            "system_health": {
                "uptime_seconds": system_info.system.uptime_seconds,
                "load_average": system_info.system.load_average
            },
            "timestamp": system_info.timestamp
        });

        Ok(resource_availability.to_string())
    }
}
