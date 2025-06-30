//! Unit tests for configuration

use common::identity::Hotkey;
use miner::config::{
    load_config, DiscoveryConfig, ExecutorConfig, ExecutorManagementConfig, MinerConfig,
};
use std::fs;
use std::time::Duration;
use tempfile::NamedTempFile;

#[test]
fn test_miner_config_default() {
    let config = MinerConfig::default();

    assert_eq!(config.netuid, 1);
    assert_eq!(config.network, "finney");
    assert_eq!(config.wallet_name, "default");
    assert_eq!(config.hotkey_name, "default");
    assert_eq!(config.max_concurrent_validators, 10);
    assert_eq!(config.session_timeout, Duration::from_secs(3600));
    assert_eq!(config.rate_limit_per_validator, 100);
}

#[test]
fn test_executor_config_creation() {
    let config = ExecutorConfig {
        id: "test-executor".to_string(),
        grpc_address: "127.0.0.1:50051".to_string(),
        name: Some("Test Executor".to_string()),
        metadata: Some(serde_json::json!({
            "gpu": "RTX 4090",
            "location": "US-East"
        })),
    };

    assert_eq!(config.id, "test-executor");
    assert_eq!(config.grpc_address, "127.0.0.1:50051");
    assert_eq!(config.name, Some("Test Executor".to_string()));
    assert!(config.metadata.is_some());
}

#[test]
fn test_executor_management_config_default() {
    let config = ExecutorManagementConfig::default();

    assert!(config.executors.is_empty());
    assert_eq!(config.health_check_interval, Duration::from_secs(30));
    assert_eq!(config.health_check_timeout, Duration::from_secs(10));
    assert_eq!(config.max_retry_attempts, 3);
    assert!(config.auto_recovery);
}

#[test]
fn test_discovery_config_default() {
    let config = DiscoveryConfig::default();

    assert_eq!(config.max_session_duration, Duration::from_secs(3600));
    assert_eq!(config.min_lease_duration, Duration::from_secs(600));
    assert_eq!(config.max_executors_per_validator, 10);
    assert!(config.require_attestation);
}

#[test]
fn test_load_config_from_file() {
    let config_content = r#"
[miner]
hotkey = "5GrwvaEF5zXb26Fz9rcQpDWS57CtERHpNehXCPcNoHGKutQY"
netuid = 42
network = "test"
wallet_name = "test_wallet"
hotkey_name = "test_hotkey"
max_concurrent_validators = 20
session_timeout = 7200
rate_limit_per_validator = 200

[discovery]
max_session_duration = 7200
min_lease_duration = 300
max_executors_per_validator = 5
require_attestation = false

[executor_management]
health_check_interval = 60
health_check_timeout = 15
max_retry_attempts = 5
auto_recovery = false

[[executor_management.executors]]
id = "exec1"
grpc_address = "192.168.1.100:50051"
name = "Primary Executor"

[[executor_management.executors]]
id = "exec2"
grpc_address = "192.168.1.101:50051"
name = "Secondary Executor"

[server]
host = "0.0.0.0"
port = 8080

[database]
url = "postgresql://user:pass@localhost/miner"
max_connections = 10

[logging]
level = "debug"
format = "json"

[metrics]
enabled = true
endpoint = "/metrics"
port = 9090
"#;

    let mut temp_file = NamedTempFile::new().unwrap();
    fs::write(&temp_file, config_content).unwrap();

    let loaded_config = load_config(temp_file.path()).unwrap();

    // Verify miner config
    assert_eq!(
        loaded_config.miner.hotkey.to_string(),
        "5GrwvaEF5zXb26Fz9rcQpDWS57CtERHpNehXCPcNoHGKutQY"
    );
    assert_eq!(loaded_config.miner.netuid, 42);
    assert_eq!(loaded_config.miner.network, "test");
    assert_eq!(loaded_config.miner.wallet_name, "test_wallet");
    assert_eq!(loaded_config.miner.hotkey_name, "test_hotkey");
    assert_eq!(loaded_config.miner.max_concurrent_validators, 20);
    assert_eq!(
        loaded_config.miner.session_timeout,
        Duration::from_secs(7200)
    );
    assert_eq!(loaded_config.miner.rate_limit_per_validator, 200);

    // Verify discovery config
    assert_eq!(
        loaded_config.miner.discovery.max_session_duration,
        Duration::from_secs(7200)
    );
    assert_eq!(
        loaded_config.miner.discovery.min_lease_duration,
        Duration::from_secs(300)
    );
    assert_eq!(loaded_config.miner.discovery.max_executors_per_validator, 5);
    assert!(!loaded_config.miner.discovery.require_attestation);

    // Verify executor management config
    assert_eq!(
        loaded_config.executor_management.health_check_interval,
        Duration::from_secs(60)
    );
    assert_eq!(
        loaded_config.executor_management.health_check_timeout,
        Duration::from_secs(15)
    );
    assert_eq!(loaded_config.executor_management.max_retry_attempts, 5);
    assert!(!loaded_config.executor_management.auto_recovery);
    assert_eq!(loaded_config.executor_management.executors.len(), 2);

    // Verify executors
    assert_eq!(loaded_config.executor_management.executors[0].id, "exec1");
    assert_eq!(
        loaded_config.executor_management.executors[0].grpc_address,
        "192.168.1.100:50051"
    );
    assert_eq!(
        loaded_config.executor_management.executors[0].name,
        Some("Primary Executor".to_string())
    );

    assert_eq!(loaded_config.executor_management.executors[1].id, "exec2");
    assert_eq!(
        loaded_config.executor_management.executors[1].grpc_address,
        "192.168.1.101:50051"
    );
    assert_eq!(
        loaded_config.executor_management.executors[1].name,
        Some("Secondary Executor".to_string())
    );

    // Verify server config
    assert_eq!(loaded_config.server.host, "0.0.0.0");
    assert_eq!(loaded_config.server.port, 8080);

    // Verify database config
    assert_eq!(
        loaded_config.database.url,
        "postgresql://user:pass@localhost/miner"
    );
    assert_eq!(loaded_config.database.max_connections, 10);

    // Verify logging config
    assert_eq!(loaded_config.logging.level, "debug");
    assert_eq!(loaded_config.logging.format, "json");

    // Verify metrics config
    assert!(loaded_config.metrics.enabled);
    assert_eq!(loaded_config.metrics.endpoint, "/metrics");
    assert_eq!(loaded_config.metrics.port, 9090);
}

#[test]
fn test_config_serialization() {
    let config = MinerConfig {
        hotkey: Hotkey("5GrwvaEF5zXb26Fz9rcQpDWS57CtERHpNehXCPcNoHGKutQY".to_string()),
        netuid: 42,
        network: "test".to_string(),
        wallet_name: "test_wallet".to_string(),
        hotkey_name: "test_hotkey".to_string(),
        max_concurrent_validators: 20,
        session_timeout: Duration::from_secs(7200),
        rate_limit_per_validator: 200,
        discovery: DiscoveryConfig {
            max_session_duration: Duration::from_secs(7200),
            min_lease_duration: Duration::from_secs(300),
            max_executors_per_validator: 5,
            require_attestation: false,
        },
    };

    // Serialize to TOML
    let toml_str = toml::to_string(&config).unwrap();

    // Deserialize back
    let deserialized: MinerConfig = toml::from_str(&toml_str).unwrap();

    assert_eq!(config.hotkey.to_string(), deserialized.hotkey.to_string());
    assert_eq!(config.netuid, deserialized.netuid);
    assert_eq!(config.network, deserialized.network);
    assert_eq!(config.wallet_name, deserialized.wallet_name);
    assert_eq!(config.hotkey_name, deserialized.hotkey_name);
    assert_eq!(
        config.max_concurrent_validators,
        deserialized.max_concurrent_validators
    );
    assert_eq!(config.session_timeout, deserialized.session_timeout);
    assert_eq!(
        config.rate_limit_per_validator,
        deserialized.rate_limit_per_validator
    );
}
