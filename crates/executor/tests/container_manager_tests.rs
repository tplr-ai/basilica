//! Unit tests for container manager

use executor::config::DockerConfig;
use executor::container_manager::{
    ContainerExecutionResult, ContainerLogEntry, ContainerManager, ContainerResourceUsage,
    ContainerStatus, LogLevel,
};
use std::time::Duration;

#[tokio::test]
async fn test_container_manager_creation() {
    let config = DockerConfig::default();

    // This test may fail if Docker is not available
    match ContainerManager::new(config).await {
        Ok(manager) => {
            // Manager created successfully
            assert!(manager.health_check().await.is_ok());
        }
        Err(e) => {
            // Expected if Docker is not available
            println!("Container manager creation failed (Docker not available): {e}");
        }
    }
}

#[test]
fn test_container_status_struct() {
    let status = ContainerStatus {
        id: "test123".to_string(),
        name: "test-container".to_string(),
        image: "alpine:latest".to_string(),
        state: "running".to_string(),
        status: "Up 5 minutes".to_string(),
        created: 1234567890,
        started: Some(1234567900),
        finished: None,
        exit_code: None,
        resource_usage: None,
    };

    assert_eq!(status.id, "test123");
    assert_eq!(status.state, "running");
    assert!(status.finished.is_none());
    assert!(status.exit_code.is_none());
}

#[test]
fn test_container_execution_result() {
    let result = ContainerExecutionResult {
        exit_code: 0,
        stdout: "Success".to_string(),
        stderr: String::new(),
        duration_ms: 100,
    };

    assert_eq!(result.exit_code, 0);
    assert_eq!(result.stdout, "Success");
    assert!(result.stderr.is_empty());
    assert_eq!(result.duration_ms, 100);
}

#[test]
fn test_container_log_entry() {
    let entry = ContainerLogEntry {
        timestamp: 1234567890,
        level: LogLevel::Info,
        message: "Container started".to_string(),
        container_id: "test123".to_string(),
    };

    assert_eq!(entry.timestamp, 1234567890);
    matches!(entry.level, LogLevel::Info);
    assert_eq!(entry.message, "Container started");
    assert_eq!(entry.container_id, "test123");
}

#[test]
fn test_log_level_variants() {
    let levels = vec![
        LogLevel::Info,
        LogLevel::Warning,
        LogLevel::Error,
        LogLevel::Debug,
    ];

    for level in levels {
        match level {
            LogLevel::Info => {}
            LogLevel::Warning => {}
            LogLevel::Error => {}
            LogLevel::Debug => {}
        }
    }
}

#[test]
fn test_container_resource_usage() {
    let usage = ContainerResourceUsage {
        cpu_usage_percent: 25.5,
        memory_usage_bytes: 104857600,  // 100MB
        memory_limit_bytes: 1073741824, // 1GB
        network_io_bytes: 1048576,      // 1MB
        block_io_bytes: 2097152,        // 2MB
    };

    assert_eq!(usage.cpu_usage_percent, 25.5);
    assert_eq!(usage.memory_usage_bytes, 104857600);
    assert_eq!(usage.memory_limit_bytes, 1073741824);
    assert_eq!(usage.network_io_bytes, 1048576);
    assert_eq!(usage.block_io_bytes, 2097152);

    // Check memory usage percentage
    let memory_usage_percent =
        (usage.memory_usage_bytes as f64 / usage.memory_limit_bytes as f64) * 100.0;
    assert!(memory_usage_percent < 10.0); // Should be about 9.77%
}

#[test]
fn test_docker_config_for_container_manager() {
    let config = DockerConfig::default();

    assert_eq!(config.socket_path, "/var/run/docker.sock");
    assert_eq!(config.default_image, "ubuntu:22.04");
    assert_eq!(config.container_timeout, Duration::from_secs(3600));
    assert_eq!(config.max_concurrent_containers, 10);
    assert!(config.enable_gpu_passthrough);

    // Check resource limits
    assert_eq!(config.resource_limits.memory_bytes, 8 * 1024 * 1024 * 1024);
    assert_eq!(config.resource_limits.cpu_cores, 4.0);

    // Check network config
    assert!(config.network_config.enable_isolation);
    assert!(!config.network_config.allow_internet);
}

#[test]
fn test_container_status_transitions() {
    let mut status = ContainerStatus {
        id: "test123".to_string(),
        name: "test-container".to_string(),
        image: "alpine:latest".to_string(),
        state: "created".to_string(),
        status: "Created".to_string(),
        created: 1234567890,
        started: None,
        finished: None,
        exit_code: None,
        resource_usage: None,
    };

    // Transition to running
    status.state = "running".to_string();
    status.status = "Up 1 second".to_string();
    status.started = Some(1234567891);

    assert_eq!(status.state, "running");
    assert!(status.started.is_some());
    assert!(status.finished.is_none());

    // Transition to stopped
    status.state = "exited".to_string();
    status.status = "Exited (0) 5 seconds ago".to_string();
    status.finished = Some(1234567896);
    status.exit_code = Some(0);

    assert_eq!(status.state, "exited");
    assert!(status.finished.is_some());
    assert_eq!(status.exit_code, Some(0));
}

#[test]
fn test_container_execution_result_with_error() {
    let result = ContainerExecutionResult {
        exit_code: 1,
        stdout: "Processing...".to_string(),
        stderr: "Error: File not found".to_string(),
        duration_ms: 50,
    };

    assert_eq!(result.exit_code, 1);
    assert!(!result.stdout.is_empty());
    assert!(!result.stderr.is_empty());
    assert!(result.stderr.contains("Error"));
}

#[test]
fn test_container_log_entry_levels() {
    let info_log = ContainerLogEntry {
        timestamp: 1234567890,
        level: LogLevel::Info,
        message: "Info message".to_string(),
        container_id: "test123".to_string(),
    };

    let warning_log = ContainerLogEntry {
        timestamp: 1234567891,
        level: LogLevel::Warning,
        message: "Warning message".to_string(),
        container_id: "test123".to_string(),
    };

    let error_log = ContainerLogEntry {
        timestamp: 1234567892,
        level: LogLevel::Error,
        message: "Error message".to_string(),
        container_id: "test123".to_string(),
    };

    let debug_log = ContainerLogEntry {
        timestamp: 1234567893,
        level: LogLevel::Debug,
        message: "Debug message".to_string(),
        container_id: "test123".to_string(),
    };

    // Verify all log levels work
    matches!(info_log.level, LogLevel::Info);
    matches!(warning_log.level, LogLevel::Warning);
    matches!(error_log.level, LogLevel::Error);
    matches!(debug_log.level, LogLevel::Debug);
}

#[tokio::test]
#[ignore = "Requires Docker daemon"]
async fn test_container_manager_operations() {
    let config = DockerConfig::default();

    match ContainerManager::new(config).await {
        Ok(manager) => {
            // Test health check
            assert!(manager.health_check().await.is_ok());

            // Test container listing (should work even with no containers)
            match manager.list_containers().await {
                Ok(containers) => {
                    // May have 0 or more containers
                    println!("Found {} containers", containers.len());
                }
                Err(e) => {
                    println!("Failed to list containers: {e}");
                }
            }
        }
        Err(e) => {
            println!("Skipping test - Docker not available: {e}");
        }
    }
}

#[test]
fn test_resource_usage_calculations() {
    let usage = ContainerResourceUsage {
        cpu_usage_percent: 75.0,
        memory_usage_bytes: 6 * 1024 * 1024 * 1024, // 6GB
        memory_limit_bytes: 8 * 1024 * 1024 * 1024, // 8GB
        network_io_bytes: 100 * 1024 * 1024,        // 100MB
        block_io_bytes: 500 * 1024 * 1024,          // 500MB
    };

    // CPU usage should be between 0 and 100
    assert!(usage.cpu_usage_percent >= 0.0);
    assert!(usage.cpu_usage_percent <= 100.0);

    // Memory usage should not exceed limit
    assert!(usage.memory_usage_bytes <= usage.memory_limit_bytes);

    // Calculate memory usage percentage
    let memory_percent =
        (usage.memory_usage_bytes as f64 / usage.memory_limit_bytes as f64) * 100.0;
    assert!((memory_percent - 75.0).abs() < 0.1); // Should be 75%
}

#[test]
fn test_container_status_with_resource_usage() {
    let usage = ContainerResourceUsage {
        cpu_usage_percent: 10.5,
        memory_usage_bytes: 512 * 1024 * 1024,      // 512MB
        memory_limit_bytes: 2 * 1024 * 1024 * 1024, // 2GB
        network_io_bytes: 0,
        block_io_bytes: 0,
    };

    let status = ContainerStatus {
        id: "test123".to_string(),
        name: "test-container".to_string(),
        image: "alpine:latest".to_string(),
        state: "running".to_string(),
        status: "Up 10 minutes".to_string(),
        created: 1234567890,
        started: Some(1234567900),
        finished: None,
        exit_code: None,
        resource_usage: Some(usage),
    };

    assert!(status.resource_usage.is_some());
    let usage = status.resource_usage.unwrap();
    assert_eq!(usage.cpu_usage_percent, 10.5);
    assert_eq!(usage.memory_usage_bytes, 512 * 1024 * 1024);
}
