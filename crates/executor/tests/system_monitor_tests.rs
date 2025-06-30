//! Unit tests for system monitor

use executor::config::SystemConfig;
use executor::system_monitor::{
    BasicSystemInfo, CpuInfo, DiskInfo, GpuInfo, MemoryInfo, NetworkInfo, NetworkInterface,
    SystemInfo, SystemMonitor,
};
use std::time::Duration;

#[test]
fn test_system_monitor_creation() {
    let config = SystemConfig::default();
    let monitor = SystemMonitor::new(config);

    // Should create successfully
    assert!(monitor.is_ok());
}

#[tokio::test]
async fn test_get_system_info() {
    let config = SystemConfig::default();
    let monitor = SystemMonitor::new(config).unwrap();

    // Get system info
    let info = monitor.get_system_info().await.unwrap();

    // Basic checks - values will vary by system
    assert!(info.cpu.cores > 0);
    assert!(info.memory.total_bytes > 0);
    assert!(!info.system.hostname.is_empty());
    assert!(!info.system.os_name.is_empty());
}

#[tokio::test]
async fn test_health_check() {
    let config = SystemConfig::default();
    let monitor = SystemMonitor::new(config).unwrap();

    // Health check should pass
    let result = monitor.health_check().await;
    assert!(result.is_ok());
}

#[test]
fn test_cpu_info_struct() {
    let cpu_info = CpuInfo {
        usage_percent: 25.5,
        cores: 8,
        frequency_mhz: 3600,
        model: "Intel Core i7".to_string(),
        vendor: "Intel".to_string(),
        temperature_celsius: Some(65.0),
    };

    assert_eq!(cpu_info.usage_percent, 25.5);
    assert_eq!(cpu_info.cores, 8);
    assert_eq!(cpu_info.frequency_mhz, 3600);
    assert_eq!(cpu_info.model, "Intel Core i7");
    assert_eq!(cpu_info.vendor, "Intel");
    assert_eq!(cpu_info.temperature_celsius, Some(65.0));
}

#[test]
fn test_memory_info_struct() {
    let memory_info = MemoryInfo {
        total_bytes: 16 * 1024 * 1024 * 1024,    // 16GB
        used_bytes: 8 * 1024 * 1024 * 1024,      // 8GB
        available_bytes: 8 * 1024 * 1024 * 1024, // 8GB
        usage_percent: 50.0,
        swap_total_bytes: 8 * 1024 * 1024 * 1024, // 8GB
        swap_used_bytes: 0,
    };

    assert_eq!(memory_info.total_bytes, 16 * 1024 * 1024 * 1024);
    assert_eq!(memory_info.used_bytes, 8 * 1024 * 1024 * 1024);
    assert_eq!(memory_info.available_bytes, 8 * 1024 * 1024 * 1024);
    assert_eq!(memory_info.usage_percent, 50.0);
    assert_eq!(memory_info.swap_total_bytes, 8 * 1024 * 1024 * 1024);
    assert_eq!(memory_info.swap_used_bytes, 0);
}

#[test]
fn test_gpu_info_struct() {
    let gpu_info = GpuInfo {
        index: 0,
        name: "NVIDIA GeForce RTX 3080".to_string(),
        memory_total_bytes: 10 * 1024 * 1024 * 1024, // 10GB
        memory_used_bytes: 2 * 1024 * 1024 * 1024,   // 2GB
        memory_usage_percent: 20.0,
        utilization_percent: 35.0,
        temperature_celsius: 70.0,
        power_usage_watts: 250.0,
        driver_version: "525.60.13".to_string(),
        cuda_version: Some("12.0".to_string()),
    };

    assert_eq!(gpu_info.index, 0);
    assert_eq!(gpu_info.name, "NVIDIA GeForce RTX 3080");
    assert_eq!(gpu_info.memory_total_bytes, 10 * 1024 * 1024 * 1024);
    assert_eq!(gpu_info.memory_used_bytes, 2 * 1024 * 1024 * 1024);
    assert_eq!(gpu_info.memory_usage_percent, 20.0);
    assert_eq!(gpu_info.utilization_percent, 35.0);
    assert_eq!(gpu_info.temperature_celsius, 70.0);
    assert_eq!(gpu_info.power_usage_watts, 250.0);
    assert_eq!(gpu_info.driver_version, "525.60.13");
    assert_eq!(gpu_info.cuda_version, Some("12.0".to_string()));
}

#[test]
fn test_disk_info_struct() {
    let disk_info = DiskInfo {
        name: "/dev/sda1".to_string(),
        mount_point: "/".to_string(),
        total_bytes: 500 * 1024 * 1024 * 1024,     // 500GB
        used_bytes: 100 * 1024 * 1024 * 1024,      // 100GB
        available_bytes: 400 * 1024 * 1024 * 1024, // 400GB
        usage_percent: 20.0,
        filesystem: "ext4".to_string(),
    };

    assert_eq!(disk_info.name, "/dev/sda1");
    assert_eq!(disk_info.mount_point, "/");
    assert_eq!(disk_info.total_bytes, 500 * 1024 * 1024 * 1024);
    assert_eq!(disk_info.used_bytes, 100 * 1024 * 1024 * 1024);
    assert_eq!(disk_info.available_bytes, 400 * 1024 * 1024 * 1024);
    assert_eq!(disk_info.usage_percent, 20.0);
    assert_eq!(disk_info.filesystem, "ext4");
}

#[test]
fn test_network_interface_struct() {
    let interface = NetworkInterface {
        name: "eth0".to_string(),
        bytes_sent: 1024 * 1024 * 1024,         // 1GB
        bytes_received: 2 * 1024 * 1024 * 1024, // 2GB
        packets_sent: 1000000,
        packets_received: 2000000,
        errors_sent: 0,
        errors_received: 0,
        is_up: true,
    };

    assert_eq!(interface.name, "eth0");
    assert_eq!(interface.bytes_sent, 1024 * 1024 * 1024);
    assert_eq!(interface.bytes_received, 2 * 1024 * 1024 * 1024);
    assert_eq!(interface.packets_sent, 1000000);
    assert_eq!(interface.packets_received, 2000000);
    assert_eq!(interface.errors_sent, 0);
    assert_eq!(interface.errors_received, 0);
    assert!(interface.is_up);
}

#[test]
fn test_network_info_struct() {
    let interface = NetworkInterface {
        name: "eth0".to_string(),
        bytes_sent: 500 * 1024 * 1024,
        bytes_received: 1024 * 1024 * 1024,
        packets_sent: 500000,
        packets_received: 1000000,
        errors_sent: 0,
        errors_received: 0,
        is_up: true,
    };

    let network_info = NetworkInfo {
        interfaces: vec![interface],
        total_bytes_sent: 500 * 1024 * 1024,
        total_bytes_received: 1024 * 1024 * 1024,
    };

    assert_eq!(network_info.interfaces.len(), 1);
    assert_eq!(network_info.interfaces[0].name, "eth0");
    assert_eq!(network_info.total_bytes_sent, 500 * 1024 * 1024);
    assert_eq!(network_info.total_bytes_received, 1024 * 1024 * 1024);
}

#[test]
fn test_basic_system_info_struct() {
    let system_info = BasicSystemInfo {
        hostname: "test-host".to_string(),
        os_name: "Linux".to_string(),
        os_version: "5.15.0".to_string(),
        kernel_version: "5.15.0-generic".to_string(),
        uptime_seconds: 3600,
        boot_time: 1234567890,
        load_average: vec![1.5, 1.2, 0.8],
    };

    assert_eq!(system_info.hostname, "test-host");
    assert_eq!(system_info.os_name, "Linux");
    assert_eq!(system_info.os_version, "5.15.0");
    assert_eq!(system_info.kernel_version, "5.15.0-generic");
    assert_eq!(system_info.uptime_seconds, 3600);
    assert_eq!(system_info.boot_time, 1234567890);
    assert_eq!(system_info.load_average, vec![1.5, 1.2, 0.8]);
}

#[test]
fn test_system_config_for_monitor() {
    let config = SystemConfig::default();

    assert_eq!(config.update_interval, Duration::from_secs(5));
    assert!(config.enable_gpu_monitoring);
    assert!(config.enable_network_monitoring);
    assert!(config.enable_memory_monitoring);
    assert!(config.enable_cpu_monitoring);
    assert_eq!(config.max_cpu_usage, 90.0);
    assert_eq!(config.max_memory_usage, 90.0);
    assert_eq!(config.max_gpu_memory_usage, 90.0);
    assert_eq!(config.min_disk_space_gb, 10);
    assert!(config.enable_metrics_recording);
}

#[test]
fn test_system_info_struct() {
    let cpu_info = CpuInfo {
        usage_percent: 25.0,
        cores: 8,
        frequency_mhz: 3600,
        model: "Intel Core i7".to_string(),
        vendor: "Intel".to_string(),
        temperature_celsius: Some(65.0),
    };

    let memory_info = MemoryInfo {
        total_bytes: 16 * 1024 * 1024 * 1024,
        used_bytes: 8 * 1024 * 1024 * 1024,
        available_bytes: 8 * 1024 * 1024 * 1024,
        usage_percent: 50.0,
        swap_total_bytes: 8 * 1024 * 1024 * 1024,
        swap_used_bytes: 0,
    };

    let basic_info = BasicSystemInfo {
        hostname: "test-host".to_string(),
        os_name: "Linux".to_string(),
        os_version: "5.15.0".to_string(),
        kernel_version: "5.15.0-generic".to_string(),
        uptime_seconds: 3600,
        boot_time: 1234567890,
        load_average: vec![1.5, 1.2, 0.8],
    };

    let system_info = SystemInfo {
        cpu: cpu_info,
        memory: memory_info,
        gpu: vec![],
        disk: vec![],
        network: NetworkInfo {
            interfaces: vec![],
            total_bytes_sent: 0,
            total_bytes_received: 0,
        },
        system: basic_info,
        timestamp: 1234567890,
    };

    assert_eq!(system_info.cpu.cores, 8);
    assert_eq!(system_info.memory.total_bytes, 16 * 1024 * 1024 * 1024);
    assert_eq!(system_info.system.hostname, "test-host");
    assert_eq!(system_info.timestamp, 1234567890);
}

#[tokio::test]
async fn test_monitor_with_disabled_features() {
    let config = SystemConfig {
        enable_gpu_monitoring: false,
        enable_network_monitoring: false,
        ..Default::default()
    };

    let monitor = SystemMonitor::new(config).unwrap();
    let info = monitor.get_system_info().await.unwrap();

    // Should still get basic info even with some features disabled
    assert!(info.cpu.cores > 0);
    assert!(info.memory.total_bytes > 0);
}

#[test]
fn test_memory_calculations() {
    let memory_info = MemoryInfo {
        total_bytes: 16 * 1024 * 1024 * 1024,    // 16GB
        used_bytes: 12 * 1024 * 1024 * 1024,     // 12GB
        available_bytes: 4 * 1024 * 1024 * 1024, // 4GB
        usage_percent: 75.0,
        swap_total_bytes: 8 * 1024 * 1024 * 1024, // 8GB
        swap_used_bytes: 2 * 1024 * 1024 * 1024,  // 2GB
    };

    // Verify calculations
    assert_eq!(
        memory_info.used_bytes + memory_info.available_bytes,
        memory_info.total_bytes
    );
    assert_eq!(memory_info.usage_percent, 75.0);

    // Calculate swap usage percentage
    let swap_usage_percent = if memory_info.swap_total_bytes > 0 {
        (memory_info.swap_used_bytes as f64 / memory_info.swap_total_bytes as f64) * 100.0
    } else {
        0.0
    };
    assert_eq!(swap_usage_percent, 25.0);
}

#[test]
fn test_disk_usage_calculations() {
    let disk_info = DiskInfo {
        name: "/dev/sda1".to_string(),
        mount_point: "/".to_string(),
        total_bytes: 1000 * 1024 * 1024 * 1024,    // 1TB
        used_bytes: 300 * 1024 * 1024 * 1024,      // 300GB
        available_bytes: 700 * 1024 * 1024 * 1024, // 700GB
        usage_percent: 30.0,
        filesystem: "ext4".to_string(),
    };

    // Verify calculations
    assert_eq!(
        disk_info.used_bytes + disk_info.available_bytes,
        disk_info.total_bytes
    );
    assert_eq!(disk_info.usage_percent, 30.0);

    // Check thresholds
    assert!(disk_info.usage_percent < 90.0); // Default threshold
}
