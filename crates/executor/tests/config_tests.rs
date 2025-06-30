//! Unit tests for executor configuration

use common::config::{LoggingConfig, MetricsConfig, ServerConfig};
use common::identity::Hotkey;
use executor::config::docker::DockerConfigValidation;
use executor::config::system::SystemConfigValidation;
use executor::config::{
    ContainerNetworkConfig, ContainerRegistryConfig, ContainerResourceLimits, DockerConfig,
    ExecutorConfig, PortMapping, SystemConfig,
};
use executor::validation_session::{
    AccessControlConfig, HotkeyVerificationConfig, RateLimitConfig, ValidatorConfig, ValidatorRole,
};
use std::collections::HashMap;
use std::str::FromStr;
use std::time::Duration;

#[test]
fn test_executor_config_default() {
    let config = ExecutorConfig::default();

    assert_eq!(config.server.host, "0.0.0.0");
    assert_eq!(config.server.port, 50051);
    assert_eq!(
        config.managing_miner_hotkey.as_str(),
        "5GrwvaEF5zXb26Fz9rcQpDWS57CtERHpNehXCPcNoHGKutQY"
    );
}

#[test]
fn test_docker_config_default() {
    let config = DockerConfig::default();

    assert_eq!(config.socket_path, "/var/run/docker.sock");
    assert_eq!(config.default_image, "ubuntu:22.04");
    assert_eq!(config.container_timeout, Duration::from_secs(3600));
    assert_eq!(config.max_concurrent_containers, 10);
    assert!(config.enable_gpu_passthrough);
    assert_eq!(config.resource_limits.memory_bytes, 8 * 1024 * 1024 * 1024);
    assert_eq!(config.resource_limits.cpu_cores, 4.0);
    assert!(config.network_config.enable_isolation);
    assert!(!config.network_config.allow_internet);
}

#[test]
fn test_system_config_default() {
    let config = SystemConfig::default();

    assert_eq!(config.update_interval, Duration::from_secs(5));
    assert_eq!(config.max_cpu_usage, 90.0);
    assert_eq!(config.max_memory_usage, 90.0);
    assert_eq!(config.max_gpu_memory_usage, 90.0);
    assert_eq!(config.min_disk_space_gb, 10);
    assert!(config.enable_gpu_monitoring);
    assert!(config.enable_network_monitoring);
    assert!(config.enable_memory_monitoring);
    assert!(config.enable_cpu_monitoring);
    assert!(config.enable_metrics_recording);
}

#[test]
fn test_validator_config_default() {
    let config = ValidatorConfig::default();

    assert!(config.enabled);
    assert!(!config.strict_ssh_restrictions);
    assert_eq!(config.access_config.ip_whitelist.len(), 0);
    assert_eq!(config.access_config.required_permissions.len(), 0);
}

#[test]
fn test_docker_config_validation() {
    let mut config = DockerConfig::default();

    // Valid config should pass
    assert!(config.validate_resource_limits().is_ok());
    assert!(config.validate_network_settings().is_ok());
    assert!(config.validate_registry_settings().is_ok());

    // Zero resource limits should fail
    config.resource_limits.memory_bytes = 0;
    assert!(config.validate_resource_limits().is_err());

    // Reset to valid value
    config.resource_limits.memory_bytes = 1024 * 1024 * 1024;

    // Invalid CPU cores should fail
    config.resource_limits.cpu_cores = 0.0;
    assert!(config.validate_resource_limits().is_err());

    // Reset to valid value
    config.resource_limits.cpu_cores = 1.0;

    // Zero concurrent containers should fail
    config.max_concurrent_containers = 0;
    assert!(config.validate_resource_limits().is_err());
}

#[test]
fn test_system_config_validation() {
    let mut config = SystemConfig::default();

    // Valid config should pass
    assert!(config.validate_usage_limits().is_ok());
    assert!(config.validate_monitoring_settings().is_ok());

    // Invalid CPU usage should fail
    config.max_cpu_usage = 101.0;
    assert!(config.validate_usage_limits().is_err());

    config.max_cpu_usage = 0.0;
    assert!(config.validate_usage_limits().is_err());

    // Reset to valid value
    config.max_cpu_usage = 90.0;

    // Invalid memory usage should fail
    config.max_memory_usage = 150.0;
    assert!(config.validate_usage_limits().is_err());

    // Reset to valid value
    config.max_memory_usage = 90.0;

    // Zero update interval should fail
    config.update_interval = Duration::from_secs(0);
    assert!(config.validate_monitoring_settings().is_err());
}

#[test]
fn test_executor_config_defaults() {
    let config = ExecutorConfig::default();

    // Server config
    assert_eq!(config.server.host, "0.0.0.0");
    assert_eq!(config.server.port, 50051);

    // Docker config
    assert_eq!(config.docker.default_image, "ubuntu:22.04");
    assert_eq!(config.docker.max_concurrent_containers, 10);

    // System config
    assert_eq!(config.system.update_interval, Duration::from_secs(5));
    assert!(config.system.enable_gpu_monitoring);

    // Validator config
    assert!(config.validator.enabled);

    // Managing miner hotkey
    assert_eq!(
        config.managing_miner_hotkey.as_str(),
        "5GrwvaEF5zXb26Fz9rcQpDWS57CtERHpNehXCPcNoHGKutQY"
    );
}

#[test]
fn test_executor_config_custom() {
    let config = ExecutorConfig {
        server: ServerConfig {
            host: "127.0.0.1".to_string(),
            port: 8080,
            max_connections: 100,
            ..Default::default()
        },
        docker: DockerConfig {
            default_image: "alpine:latest".to_string(),
            max_concurrent_containers: 20,
            resource_limits: ContainerResourceLimits {
                cpu_cores: 8.0,
                memory_bytes: 16 * 1024 * 1024 * 1024,
                gpu_memory_bytes: Some(8 * 1024 * 1024 * 1024),
                disk_io_bps: Some(200 * 1024 * 1024),
                network_bps: Some(200 * 1024 * 1024),
            },
            ..Default::default()
        },
        system: SystemConfig {
            update_interval: Duration::from_secs(2),
            max_cpu_usage: 95.0,
            max_memory_usage: 95.0,
            ..Default::default()
        },
        validator: ValidatorConfig {
            enabled: true,
            strict_ssh_restrictions: false,
            access_config: AccessControlConfig {
                ip_whitelist: vec!["192.168.1.0/24".to_string()],
                required_permissions: Default::default(),
                hotkey_verification: HotkeyVerificationConfig {
                    enabled: false,
                    challenge_timeout_seconds: 60,
                    max_signature_attempts: 3,
                    cleanup_interval_seconds: 300,
                },
                rate_limits: RateLimitConfig {
                    ssh_requests_per_minute: 10,
                    api_requests_per_minute: 50,
                    burst_allowance: 5,
                    rate_limit_window_seconds: 60,
                },
                role_assignments: HashMap::new(),
            },
        },
        managing_miner_hotkey: Hotkey::from_str("5GNJqTPyNqANBkUVMN1LPPrxXnFouWXoe2wNSmmEoLctxiZY")
            .unwrap(),
        logging: LoggingConfig::default(),
        metrics: MetricsConfig::default(),
    };

    assert_eq!(config.server.host, "127.0.0.1");
    assert_eq!(config.server.port, 8080);
    assert_eq!(config.docker.default_image, "alpine:latest");
    assert_eq!(config.docker.max_concurrent_containers, 20);
    assert_eq!(config.docker.resource_limits.cpu_cores, 8.0);
    assert_eq!(
        config.docker.resource_limits.memory_bytes,
        16 * 1024 * 1024 * 1024
    );
    assert_eq!(config.system.update_interval, Duration::from_secs(2));
    assert_eq!(config.system.max_cpu_usage, 95.0);
    assert!(config.validator.enabled);
    assert!(!config.validator.strict_ssh_restrictions);
    assert_eq!(config.validator.access_config.ip_whitelist.len(), 1);
}

#[test]
fn test_config_warnings() {
    let mut config = DockerConfig::default();
    config.network_config.allow_internet = true;
    config.registry.verify_signatures = false;
    config.max_concurrent_containers = 100;

    let warnings = config.docker_warnings();
    assert!(!warnings.is_empty());
    assert!(warnings.iter().any(|w| w.contains("internet access")));
    assert!(warnings
        .iter()
        .any(|w| w.contains("signature verification")));
    assert!(warnings.iter().any(|w| w.contains("concurrent container")));
}

#[test]
fn test_container_resource_limits() {
    let limits = ContainerResourceLimits::default();

    assert_eq!(limits.cpu_cores, 4.0);
    assert_eq!(limits.memory_bytes, 8 * 1024 * 1024 * 1024);
    assert_eq!(limits.gpu_memory_bytes, Some(4 * 1024 * 1024 * 1024));
    assert_eq!(limits.disk_io_bps, Some(100 * 1024 * 1024));
    assert_eq!(limits.network_bps, Some(100 * 1024 * 1024));
}

#[test]
fn test_network_config_defaults() {
    let config = ContainerNetworkConfig::default();

    assert!(config.enable_isolation);
    assert!(!config.allow_internet);
    assert_eq!(config.dns_servers.len(), 2);
    assert!(config.dns_servers.contains(&"8.8.8.8".to_string()));
    assert!(config.dns_servers.contains(&"8.8.4.4".to_string()));
    assert!(config.port_mappings.is_empty());
}

#[test]
fn test_registry_config_defaults() {
    let config = ContainerRegistryConfig::default();

    assert_eq!(config.url, "docker.io");
    assert!(config.username.is_none());
    assert!(config.password.is_none());
    assert!(config.verify_signatures);
    assert_eq!(config.allowed_registries.len(), 3);
    assert!(config.allowed_registries.contains(&"docker.io".to_string()));
    assert!(config.allowed_registries.contains(&"ghcr.io".to_string()));
    assert!(config.allowed_registries.contains(&"quay.io".to_string()));
}

#[test]
fn test_port_mapping() {
    let port_mapping = PortMapping {
        host_port: 8080,
        container_port: 80,
        protocol: "tcp".to_string(),
    };

    assert_eq!(port_mapping.host_port, 8080);
    assert_eq!(port_mapping.container_port, 80);
    assert_eq!(port_mapping.protocol, "tcp");
}

#[test]
fn test_network_config_validation() {
    let mut config = DockerConfig::default();

    // Add invalid port mapping
    config.network_config.port_mappings.push(PortMapping {
        host_port: 0,
        container_port: 80,
        protocol: "tcp".to_string(),
    });

    assert!(config.validate_network_settings().is_err());

    // Fix port mapping
    config.network_config.port_mappings.clear();
    config.network_config.port_mappings.push(PortMapping {
        host_port: 8080,
        container_port: 80,
        protocol: "invalid".to_string(),
    });

    assert!(config.validate_network_settings().is_err());

    // Fix protocol
    config.network_config.port_mappings[0].protocol = "udp".to_string();
    assert!(config.validate_network_settings().is_ok());
}

#[test]
fn test_registry_validation() {
    let mut config = DockerConfig::default();

    // Clear allowed registries
    config.registry.allowed_registries.clear();
    assert!(config.validate_registry_settings().is_err());

    // Add at least one registry
    config
        .registry
        .allowed_registries
        .push("docker.io".to_string());
    assert!(config.validate_registry_settings().is_ok());
}

#[test]
fn test_system_config_warnings() {
    let config = SystemConfig {
        max_cpu_usage: 98.0,
        max_memory_usage: 97.0,
        max_gpu_memory_usage: 99.0,
        ..SystemConfig::default()
    };

    let warnings = config.usage_warnings();
    assert!(!warnings.is_empty());
    assert!(warnings.iter().any(|w| w.contains("CPU usage")));
    assert!(warnings.iter().any(|w| w.contains("memory usage")));
    assert!(warnings.iter().any(|w| w.contains("GPU memory usage")));
}

#[test]
fn test_validator_config_custom() {
    let config = ValidatorConfig {
        enabled: true,
        strict_ssh_restrictions: true,
        access_config: AccessControlConfig {
            ip_whitelist: vec!["10.0.0.0/8".to_string(), "192.168.0.0/16".to_string()],
            required_permissions: {
                let mut perms = HashMap::new();
                perms.insert(
                    "execute".to_string(),
                    vec!["admin".to_string(), "validator".to_string()],
                );
                perms
            },
            hotkey_verification: HotkeyVerificationConfig {
                enabled: true,
                challenge_timeout_seconds: 30,
                max_signature_attempts: 5,
                cleanup_interval_seconds: 180,
            },
            rate_limits: RateLimitConfig {
                ssh_requests_per_minute: 20,
                api_requests_per_minute: 100,
                burst_allowance: 10,
                rate_limit_window_seconds: 60,
            },
            role_assignments: {
                let mut roles = HashMap::new();
                roles.insert("admin_validator".to_string(), ValidatorRole::Admin);
                roles
            },
        },
    };

    assert!(config.enabled);
    assert!(config.strict_ssh_restrictions);
    assert_eq!(config.access_config.ip_whitelist.len(), 2);
    assert_eq!(config.access_config.required_permissions.len(), 1);
    assert!(config
        .access_config
        .required_permissions
        .contains_key("execute"));
}
