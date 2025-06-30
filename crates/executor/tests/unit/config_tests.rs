//! Unit tests for executor configuration

use common::identity::Hotkey;
use executor::config::{DockerConfig, ExecutorConfig, SystemConfig};
use executor::validation_session::ValidatorConfig;
use std::fs;
use std::path::PathBuf;
use std::time::Duration;
use tempfile::NamedTempFile;

#[test]
fn test_executor_config_default() {
    let config = ExecutorConfig::default();

    assert_eq!(config.server.host, "0.0.0.0");
    assert_eq!(config.server.port, 50051);
    assert_eq!(
        config.managing_miner_hotkey.to_string(),
        "5GrwvaEF5zXb26Fz9rcQpDWS57CtERHpNehXCPcNoHGKutQY"
    );
}

#[test]
fn test_docker_config_default() {
    let config = DockerConfig::default();

    assert_eq!(config.socket_path, "unix:///var/run/docker.sock");
    assert!(config.enable_gpu);
    assert_eq!(config.allowed_images.len(), 3);
    assert!(config.allowed_images.contains(&"ubuntu:latest".to_string()));
    assert_eq!(config.default_resource_limits.max_cpu_cores, 4);
    assert_eq!(config.default_resource_limits.max_memory_mb, 8192);
}

#[test]
fn test_system_config_default() {
    let config = SystemConfig::default();

    assert_eq!(config.monitoring_interval, Duration::from_secs(10));
    assert_eq!(config.resource_update_interval, Duration::from_secs(30));
    assert_eq!(config.disk_usage_threshold_percent, 90.0);
    assert_eq!(config.memory_usage_threshold_percent, 85.0);
    assert!(config.enable_gpu_monitoring);
}

#[test]
fn test_validator_config_default() {
    let config = ValidatorConfig::default();

    assert!(config.enabled);
    assert_eq!(config.max_concurrent_sessions, 10);
    assert_eq!(config.session_timeout, Duration::from_secs(3600));
    assert_eq!(config.ssh_port, 22);
    assert!(config.require_auth_token);
}

#[test]
fn test_load_config_from_toml() {
    let config_content = r#"
[server]
host = "127.0.0.1"
port = 50052

[logging]
level = "debug"
format = "json"

[metrics]
enabled = true
port = 9091

[system]
monitoring_interval = 5
resource_update_interval = 20
disk_usage_threshold_percent = 95.0
memory_usage_threshold_percent = 90.0
enable_gpu_monitoring = false

[docker]
socket_path = "unix:///custom/docker.sock"
enable_gpu = false
allowed_images = ["custom/image:latest"]
max_containers_per_validator = 5
container_cleanup_interval = 600
log_retention_duration = 7200

[docker.default_resource_limits]
max_cpu_cores = 8
max_memory_mb = 16384
max_storage_mb = 102400
max_containers = 10
max_bandwidth_mbps = 1000.0
max_gpus = 2

[validator]
enabled = false
max_concurrent_sessions = 20
session_timeout = 7200
ssh_port = 2222
require_auth_token = false
allowed_ssh_keys = ["ssh-rsa AAAAB3..."]

managing_miner_hotkey = "5GNJqTPyNqANBkUVMN1LPPrxXnFouWXoe2wNSmmEoLctxiZY"
"#;

    let mut temp_file = NamedTempFile::new().unwrap();
    fs::write(&temp_file, config_content).unwrap();

    let config = ExecutorConfig::load_from_file(temp_file.path()).unwrap();

    // Verify server config
    assert_eq!(config.server.host, "127.0.0.1");
    assert_eq!(config.server.port, 50052);

    // Verify logging config
    assert_eq!(config.logging.level, "debug");
    assert_eq!(config.logging.format, "json");

    // Verify metrics config
    assert!(config.metrics.enabled);
    assert_eq!(config.metrics.port, 9091);

    // Verify system config
    assert_eq!(config.system.monitoring_interval, Duration::from_secs(5));
    assert_eq!(
        config.system.resource_update_interval,
        Duration::from_secs(20)
    );
    assert_eq!(config.system.disk_usage_threshold_percent, 95.0);
    assert_eq!(config.system.memory_usage_threshold_percent, 90.0);
    assert!(!config.system.enable_gpu_monitoring);

    // Verify docker config
    assert_eq!(config.docker.socket_path, "unix:///custom/docker.sock");
    assert!(!config.docker.enable_gpu);
    assert_eq!(config.docker.allowed_images, vec!["custom/image:latest"]);
    assert_eq!(config.docker.max_containers_per_validator, 5);
    assert_eq!(
        config.docker.container_cleanup_interval,
        Duration::from_secs(600)
    );
    assert_eq!(
        config.docker.log_retention_duration,
        Duration::from_secs(7200)
    );

    // Verify resource limits
    assert_eq!(config.docker.default_resource_limits.max_cpu_cores, 8);
    assert_eq!(config.docker.default_resource_limits.max_memory_mb, 16384);
    assert_eq!(config.docker.default_resource_limits.max_storage_mb, 102400);
    assert_eq!(config.docker.default_resource_limits.max_containers, 10);
    assert_eq!(
        config.docker.default_resource_limits.max_bandwidth_mbps,
        1000.0
    );
    assert_eq!(config.docker.default_resource_limits.max_gpus, 2);

    // Verify validator config
    assert!(!config.validator.enabled);
    assert_eq!(config.validator.max_concurrent_sessions, 20);
    assert_eq!(config.validator.session_timeout, Duration::from_secs(7200));
    assert_eq!(config.validator.ssh_port, 2222);
    assert!(!config.validator.require_auth_token);
    assert_eq!(config.validator.allowed_ssh_keys.len(), 1);

    // Verify managing miner hotkey
    assert_eq!(
        config.managing_miner_hotkey.to_string(),
        "5GNJqTPyNqANBkUVMN1LPPrxXnFouWXoe2wNSmmEoLctxiZY"
    );
}

#[test]
fn test_config_serialization() {
    let config = ExecutorConfig {
        managing_miner_hotkey: Hotkey(
            "5GNJqTPyNqANBkUVMN1LPPrxXnFouWXoe2wNSmmEoLctxiZY".to_string(),
        ),
        ..Default::default()
    };

    // Serialize to TOML
    let toml_str = toml::to_string_pretty(&config).unwrap();

    // Deserialize back
    let deserialized: ExecutorConfig = toml::from_str(&toml_str).unwrap();

    assert_eq!(config.server.host, deserialized.server.host);
    assert_eq!(config.server.port, deserialized.server.port);
    assert_eq!(
        config.managing_miner_hotkey.to_string(),
        deserialized.managing_miner_hotkey.to_string()
    );
}
