//! Unit tests for system monitor

use executor::config::SystemConfig;
use executor::system_monitor::types::BasicSystemInfo;
use executor::system_monitor::{
    CpuInfo, DiskInfo, GpuInfo, MemoryInfo, NetworkInfo, SystemInfo, SystemMonitor,
};
use std::time::Duration;

// Mock system monitor for testing
struct MockSystemMonitor {
    cpu_usage: f32,
    memory_usage: u64,
    disk_usage: u64,
}

impl MockSystemMonitor {
    fn new() -> Self {
        Self {
            cpu_usage: 25.0,
            memory_usage: 4096 * 1024 * 1024,     // 4GB
            disk_usage: 100 * 1024 * 1024 * 1024, // 100GB
        }
    }
}

#[async_trait::async_trait]
impl executor::system_monitor::SystemMonitor for MockSystemMonitor {
    async fn get_system_info(&self) -> anyhow::Result<BasicSystemInfo> {
        Ok(BasicSystemInfo {
            system: executor::system_monitor::types::SystemInfo {
                hostname: "test-host".to_string(),
                os_name: "Linux".to_string(),
                os_version: "5.15.0".to_string(),
                kernel_version: "5.15.0-generic".to_string(),
                uptime_seconds: 3600,
            },
            cpu: executor::system_monitor::types::CpuInfo {
                cores: 8,
                model: "Intel Core i7".to_string(),
                vendor: "Intel".to_string(),
                frequency_mhz: 3600,
                usage_percent: self.cpu_usage,
                temperature_celsius: Some(65.0),
            },
            memory: executor::system_monitor::types::MemoryInfo {
                total_bytes: 16 * 1024 * 1024 * 1024,     // 16GB
                available_bytes: 12 * 1024 * 1024 * 1024, // 12GB
                used_bytes: self.memory_usage,
                cached_bytes: 2 * 1024 * 1024 * 1024,     // 2GB
                swap_total_bytes: 8 * 1024 * 1024 * 1024, // 8GB
                swap_used_bytes: 0,
            },
            disk: vec![executor::system_monitor::types::DiskInfo {
                device: "/dev/sda1".to_string(),
                mount_point: "/".to_string(),
                filesystem: "ext4".to_string(),
                total_bytes: 500 * 1024 * 1024 * 1024, // 500GB
                used_bytes: self.disk_usage,
                available_bytes: 400 * 1024 * 1024 * 1024, // 400GB
            }],
            network: executor::system_monitor::types::NetworkInfo {
                interfaces: vec![executor::system_monitor::types::NetworkInterfaceInfo {
                    name: "eth0".to_string(),
                    ip_addresses: vec!["192.168.1.100".to_string()],
                    mac_address: "00:11:22:33:44:55".to_string(),
                    is_up: true,
                    speed_mbps: Some(1000),
                }],
            },
            gpu: vec![executor::system_monitor::types::GpuInfo {
                index: 0,
                name: "NVIDIA GeForce RTX 3080".to_string(),
                uuid: "GPU-12345678-1234-1234-1234-123456789012".to_string(),
                driver_version: "525.60.13".to_string(),
                cuda_version: Some("12.0".to_string()),
                memory_total_bytes: 10 * 1024 * 1024 * 1024, // 10GB
                memory_used_bytes: 2 * 1024 * 1024 * 1024,   // 2GB
                memory_free_bytes: 8 * 1024 * 1024 * 1024,   // 8GB
                utilization_percent: 30,
                memory_usage_percent: 20.0,
                temperature_celsius: 70,
                power_usage_watts: 250,
                power_limit_watts: 350,
            }],
        })
    }

    async fn get_cpu_usage(&self) -> anyhow::Result<f32> {
        Ok(self.cpu_usage)
    }

    async fn get_memory_usage(&self) -> anyhow::Result<(u64, u64)> {
        Ok((self.memory_usage, 16 * 1024 * 1024 * 1024))
    }

    async fn get_disk_usage(&self) -> anyhow::Result<Vec<(String, u64, u64)>> {
        Ok(vec![(
            "/".to_string(),
            self.disk_usage,
            500 * 1024 * 1024 * 1024,
        )])
    }

    async fn get_gpu_usage(&self) -> anyhow::Result<Vec<(u32, f32, f32)>> {
        Ok(vec![
            (0, 30.0, 20.0), // GPU 0: 30% util, 20% memory
        ])
    }

    async fn start_monitoring(&self, interval: Duration) -> anyhow::Result<()> {
        Ok(())
    }

    async fn stop_monitoring(&self) -> anyhow::Result<()> {
        Ok(())
    }

    async fn get_latest_metrics(&self) -> anyhow::Result<BasicSystemInfo> {
        self.get_system_info().await
    }
}

// Legacy tests
#[tokio::test]
async fn test_system_monitor_creation() {
    let config = SystemConfig::default();
    let monitor = SystemMonitor::new(config);

    assert!(monitor.is_ok());

    let monitor = monitor.unwrap();

    // Health check should pass
    assert!(monitor.health_check().await.is_ok());
}

#[tokio::test]
async fn test_get_system_info() {
    let config = SystemConfig::default();
    let monitor = SystemMonitor::new(config).unwrap();

    let info = monitor.get_system_info().await.unwrap();

    // System info should have all components
    assert!(!info.system.hostname.is_empty());
    assert!(!info.system.os_type.is_empty());
    assert!(!info.system.os_version.is_empty());
    assert!(info.system.uptime_seconds > 0);

    // CPU info
    assert!(info.cpu.total_cores > 0);
    assert!(!info.cpu.model_name.is_empty());
    assert!(info.cpu.usage_percent >= 0.0 && info.cpu.usage_percent <= 100.0);

    // Memory info
    assert!(info.memory.total_mb > 0);
    assert!(info.memory.available_mb <= info.memory.total_mb);
    assert!(info.memory.usage_percent >= 0.0 && info.memory.usage_percent <= 100.0);

    // Network info
    assert!(!info.network.interfaces.is_empty());
}

#[tokio::test]
async fn test_get_cpu_info() {
    let config = SystemConfig::default();
    let monitor = SystemMonitor::new(config).unwrap();

    let cpu_info = monitor.get_cpu_info().await;

    assert!(cpu_info.total_cores > 0);
    assert!(cpu_info.usage_percent >= 0.0);
    assert!(cpu_info.usage_percent <= 100.0);
    assert!(!cpu_info.model_name.is_empty());
    assert!(cpu_info.frequency_mhz > 0);
}

#[tokio::test]
async fn test_get_memory_info() {
    let config = SystemConfig::default();
    let monitor = SystemMonitor::new(config).unwrap();

    let mem_info = monitor.get_memory_info().await;

    assert!(mem_info.total_mb > 0);
    assert!(mem_info.available_mb > 0);
    assert!(mem_info.used_mb > 0);
    assert!(mem_info.available_mb <= mem_info.total_mb);
    assert!(mem_info.usage_percent >= 0.0 && mem_info.usage_percent <= 100.0);
}

// New comprehensive tests using mock
#[tokio::test]
async fn test_mock_system_monitor() {
    let monitor = MockSystemMonitor::new();

    let info = monitor.get_system_info().await.unwrap();
    assert_eq!(info.system.hostname, "test-host");
    assert_eq!(info.cpu.cores, 8);
    assert_eq!(info.cpu.model, "Intel Core i7");
}

#[tokio::test]
async fn test_mock_cpu_info() {
    let monitor = MockSystemMonitor::new();

    let cpu_usage = monitor.get_cpu_usage().await.unwrap();
    assert_eq!(cpu_usage, 25.0);

    let info = monitor.get_system_info().await.unwrap();
    assert_eq!(info.cpu.usage_percent, 25.0);
    assert_eq!(info.cpu.temperature_celsius, Some(65.0));
}

#[tokio::test]
async fn test_mock_memory_info() {
    let monitor = MockSystemMonitor::new();

    let (used, total) = monitor.get_memory_usage().await.unwrap();
    assert_eq!(used, 4 * 1024 * 1024 * 1024); // 4GB
    assert_eq!(total, 16 * 1024 * 1024 * 1024); // 16GB

    let info = monitor.get_system_info().await.unwrap();
    assert_eq!(info.memory.total_bytes, 16 * 1024 * 1024 * 1024);
    assert_eq!(info.memory.available_bytes, 12 * 1024 * 1024 * 1024);
}

#[tokio::test]
async fn test_mock_disk_info() {
    let monitor = MockSystemMonitor::new();

    let disks = monitor.get_disk_usage().await.unwrap();
    assert_eq!(disks.len(), 1);
    assert_eq!(disks[0].0, "/");
    assert_eq!(disks[0].1, 100 * 1024 * 1024 * 1024); // 100GB

    let info = monitor.get_system_info().await.unwrap();
    assert_eq!(info.disk.len(), 1);
    assert_eq!(info.disk[0].mount_point, "/");
    assert_eq!(info.disk[0].filesystem, "ext4");
}

#[tokio::test]
async fn test_mock_network_info() {
    let monitor = MockSystemMonitor::new();

    let info = monitor.get_system_info().await.unwrap();
    assert_eq!(info.network.interfaces.len(), 1);
    assert_eq!(info.network.interfaces[0].name, "eth0");
    assert_eq!(info.network.interfaces[0].ip_addresses[0], "192.168.1.100");
    assert!(info.network.interfaces[0].is_up);
}

#[tokio::test]
async fn test_mock_gpu_info() {
    let monitor = MockSystemMonitor::new();

    let gpu_usage = monitor.get_gpu_usage().await.unwrap();
    assert_eq!(gpu_usage.len(), 1);
    assert_eq!(gpu_usage[0].0, 0); // GPU index
    assert_eq!(gpu_usage[0].1, 30.0); // Utilization
    assert_eq!(gpu_usage[0].2, 20.0); // Memory usage

    let info = monitor.get_system_info().await.unwrap();
    assert_eq!(info.gpu.len(), 1);
    assert_eq!(info.gpu[0].name, "NVIDIA GeForce RTX 3080");
    assert_eq!(info.gpu[0].temperature_celsius, 70);
}

#[tokio::test]
async fn test_monitoring_lifecycle() {
    let monitor = MockSystemMonitor::new();

    // Start monitoring
    let result = monitor.start_monitoring(Duration::from_secs(1)).await;
    assert!(result.is_ok());

    // Get latest metrics
    let metrics = monitor.get_latest_metrics().await.unwrap();
    assert!(!metrics.system.hostname.is_empty());

    // Stop monitoring
    let result = monitor.stop_monitoring().await;
    assert!(result.is_ok());
}

#[test]
fn test_system_config_defaults() {
    let config = SystemConfig::default();
    assert_eq!(config.monitor_interval_secs, 60);
    assert_eq!(config.metric_retention_hours, 24);
    assert!(config.enable_gpu_monitoring);
    assert!(config.enable_disk_monitoring);
    assert!(config.enable_network_monitoring);
}

#[test]
fn test_cpu_info_creation() {
    let cpu = executor::system_monitor::types::CpuInfo {
        cores: 16,
        model: "AMD Ryzen 9".to_string(),
        vendor: "AMD".to_string(),
        frequency_mhz: 4200,
        usage_percent: 15.5,
        temperature_celsius: Some(55.0),
    };

    assert_eq!(cpu.cores, 16);
    assert_eq!(cpu.model, "AMD Ryzen 9");
    assert_eq!(cpu.frequency_mhz, 4200);
}

#[test]
fn test_memory_info_calculations() {
    let memory = executor::system_monitor::types::MemoryInfo {
        total_bytes: 32 * 1024 * 1024 * 1024,      // 32GB
        available_bytes: 24 * 1024 * 1024 * 1024,  // 24GB
        used_bytes: 8 * 1024 * 1024 * 1024,        // 8GB
        cached_bytes: 4 * 1024 * 1024 * 1024,      // 4GB
        swap_total_bytes: 16 * 1024 * 1024 * 1024, // 16GB
        swap_used_bytes: 0,
    };

    // Used + Available should approximately equal Total
    assert!(memory.used_bytes + memory.available_bytes <= memory.total_bytes);
    assert_eq!(memory.swap_used_bytes, 0);
}

#[test]
fn test_disk_info_validation() {
    let disk = executor::system_monitor::types::DiskInfo {
        device: "/dev/nvme0n1p1".to_string(),
        mount_point: "/home".to_string(),
        filesystem: "btrfs".to_string(),
        total_bytes: 1024 * 1024 * 1024 * 1024,    // 1TB
        used_bytes: 200 * 1024 * 1024 * 1024,      // 200GB
        available_bytes: 824 * 1024 * 1024 * 1024, // 824GB
    };

    assert!(disk.used_bytes + disk.available_bytes <= disk.total_bytes);
    assert!(!disk.device.is_empty());
    assert!(!disk.mount_point.is_empty());
}

#[test]
fn test_network_interface_info() {
    let iface = executor::system_monitor::types::NetworkInterfaceInfo {
        name: "wlan0".to_string(),
        ip_addresses: vec!["192.168.0.50".to_string(), "fe80::1".to_string()],
        mac_address: "AA:BB:CC:DD:EE:FF".to_string(),
        is_up: true,
        speed_mbps: Some(300),
    };

    assert_eq!(iface.name, "wlan0");
    assert_eq!(iface.ip_addresses.len(), 2);
    assert!(iface.is_up);
    assert_eq!(iface.speed_mbps, Some(300));
}

#[test]
fn test_gpu_info_validation() {
    let gpu = executor::system_monitor::types::GpuInfo {
        index: 0,
        name: "NVIDIA Tesla V100".to_string(),
        uuid: "GPU-abcdef".to_string(),
        driver_version: "525.60.13".to_string(),
        cuda_version: Some("12.0".to_string()),
        memory_total_bytes: 32 * 1024 * 1024 * 1024, // 32GB
        memory_used_bytes: 16 * 1024 * 1024 * 1024,  // 16GB
        memory_free_bytes: 16 * 1024 * 1024 * 1024,  // 16GB
        utilization_percent: 85,
        memory_usage_percent: 50.0,
        temperature_celsius: 80,
        power_usage_watts: 300,
        power_limit_watts: 300,
    };

    assert_eq!(
        gpu.memory_used_bytes + gpu.memory_free_bytes,
        gpu.memory_total_bytes
    );
    assert!(gpu.utilization_percent <= 100);
    assert!(gpu.memory_usage_percent <= 100.0);
}

#[tokio::test]
async fn test_system_info_completeness() {
    let monitor = MockSystemMonitor::new();
    let info = monitor.get_system_info().await.unwrap();

    // Verify all components are present
    assert!(!info.system.hostname.is_empty());
    assert!(info.cpu.cores > 0);
    assert!(info.memory.total_bytes > 0);
    assert!(!info.disk.is_empty());
    assert!(!info.network.interfaces.is_empty());
    assert!(!info.gpu.is_empty());
}

// Legacy compatibility tests
#[tokio::test]
async fn test_get_disk_info() {
    let config = SystemConfig::default();
    let monitor = SystemMonitor::new(config).unwrap();

    let disk_info = monitor.get_disk_info().await;

    // Should have at least one disk
    assert!(!disk_info.disks.is_empty());

    let first_disk = &disk_info.disks[0];
    assert!(!first_disk.name.is_empty());
    assert!(!first_disk.mount_point.is_empty());
    assert!(first_disk.total_bytes > 0);
    assert!(first_disk.available_bytes <= first_disk.total_bytes);
    assert!(first_disk.usage_percent >= 0.0 && first_disk.usage_percent <= 100.0);
}

#[tokio::test]
async fn test_get_network_info() {
    let config = SystemConfig::default();
    let monitor = SystemMonitor::new(config).unwrap();

    let net_info = monitor.get_network_info().await;

    // Should have at least one network interface
    assert!(!net_info.interfaces.is_empty());

    let first_interface = &net_info.interfaces[0];
    assert!(!first_interface.name.is_empty());
    assert!(first_interface.rx_bytes >= 0);
    assert!(first_interface.tx_bytes >= 0);
}

#[tokio::test]
async fn test_get_gpu_info() {
    let config = SystemConfig::default();
    let monitor = SystemMonitor::new(config).unwrap();

    let gpu_info = monitor.get_gpu_info().await;

    // GPU info may be empty if no GPU present
    // Just verify it doesn't panic
    if !gpu_info.gpus.is_empty() {
        let first_gpu = &gpu_info.gpus[0];
        assert!(!first_gpu.model.is_empty());
        assert!(first_gpu.memory_total_mb > 0);
        assert!(!first_gpu.uuid.is_empty());
    }
}

#[tokio::test]
async fn test_system_monitor_with_disabled_gpu() {
    let mut config = SystemConfig::default();
    config.enable_gpu_monitoring = false;

    let monitor = SystemMonitor::new(config).unwrap();
    let gpu_info = monitor.get_gpu_info().await;

    // Should return empty GPU list when monitoring disabled
    assert!(gpu_info.gpus.is_empty());
}

#[tokio::test]
async fn test_monitor_intervals() {
    let config = SystemConfig {
        monitoring_interval: Duration::from_millis(100),
        resource_update_interval: Duration::from_millis(200),
        ..Default::default()
    };

    let monitor = SystemMonitor::new(config).unwrap();

    // Get info twice with small delay
    let info1 = monitor.get_system_info().await.unwrap();
    tokio::time::sleep(Duration::from_millis(150)).await;
    let info2 = monitor.get_system_info().await.unwrap();

    // CPU usage might have changed
    // Just verify we get valid data both times
    assert!(info1.cpu.usage_percent >= 0.0);
    assert!(info2.cpu.usage_percent >= 0.0);
}
