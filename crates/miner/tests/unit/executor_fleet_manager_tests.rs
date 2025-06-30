//! Unit tests for ExecutorFleetManager

use common::config::DatabaseConfig;
use miner::config::{ExecutorConfig, ExecutorManagementConfig};
use miner::executor_fleet_manager::{ExecutorFleetManager, FleetStatistics};
use miner::persistence::RegistrationDb;
use std::time::Duration;
use tokio::time::timeout;

#[tokio::test]
async fn test_fleet_manager_new() {
    let config = create_test_config();
    let db = create_test_db().await;

    let manager = ExecutorFleetManager::new(config.clone(), db).await.unwrap();
    let stats = manager.get_fleet_stats().await;

    assert_eq!(stats.total_executors, 2);
    assert_eq!(stats.healthy_executors, 0);
    assert_eq!(stats.unhealthy_executors, 2);
}

#[tokio::test]
async fn test_list_available_executors_empty_when_unhealthy() {
    let config = create_test_config();
    let db = create_test_db().await;

    let manager = ExecutorFleetManager::new(config, db).await.unwrap();
    let executors = manager.list_available_executors().await.unwrap();

    assert_eq!(executors.len(), 0);
}

#[tokio::test]
async fn test_fleet_stats_calculation() {
    let config = create_test_config();
    let db = create_test_db().await;

    let manager = ExecutorFleetManager::new(config, db).await.unwrap();
    let stats = manager.get_fleet_stats().await;

    assert_eq!(stats.total_executors, 2);
    assert_eq!(stats.healthy_executors, 0);
    assert_eq!(stats.unhealthy_executors, 2);
    assert_eq!(stats.avg_response_time, 0.0);
}

#[tokio::test]
async fn test_parse_resource_stats() {
    let config = create_test_config();
    let db = create_test_db().await;

    let manager = ExecutorFleetManager::new(config, db).await.unwrap();

    let mut resource_status = std::collections::HashMap::new();
    resource_status.insert("cpu_usage".to_string(), "50.5".to_string());
    resource_status.insert("memory_mb".to_string(), "4096".to_string());
    resource_status.insert("network_rx_bytes".to_string(), "1024".to_string());
    resource_status.insert("network_tx_bytes".to_string(), "2048".to_string());
    resource_status.insert("disk_read_bytes".to_string(), "3072".to_string());
    resource_status.insert("disk_write_bytes".to_string(), "4096".to_string());
    resource_status.insert("gpu_utilization".to_string(), "75.0, 80.0".to_string());
    resource_status.insert("gpu_memory_mb".to_string(), "8192, 16384".to_string());

    // Use reflection to test private method
    let stats = manager.parse_resource_stats(&resource_status);

    assert!(stats.is_some());
    let stats = stats.unwrap();
    assert_eq!(stats.cpu_percent, 50.5);
    assert_eq!(stats.memory_mb, 4096);
    assert_eq!(stats.network_rx_bytes, 1024);
    assert_eq!(stats.network_tx_bytes, 2048);
    assert_eq!(stats.disk_read_bytes, 3072);
    assert_eq!(stats.disk_write_bytes, 4096);
    assert_eq!(stats.gpu_utilization, vec![75.0, 80.0]);
    assert_eq!(stats.gpu_memory_mb, vec![8192, 16384]);
}

#[tokio::test]
async fn test_health_monitoring_timeout() {
    let mut config = create_test_config();
    config.health_check_interval = Duration::from_millis(100);
    let db = create_test_db().await;

    let manager = ExecutorFleetManager::new(config, db).await.unwrap();

    // Start monitoring should not block indefinitely
    let result = timeout(Duration::from_secs(1), manager.start_monitoring()).await;

    // Should timeout since we're not sending ctrl+c
    assert!(result.is_err());
}

// Helper functions

fn create_test_config() -> ExecutorManagementConfig {
    ExecutorManagementConfig {
        executors: vec![
            ExecutorConfig {
                id: "test-executor-1".to_string(),
                grpc_address: "127.0.0.1:50051".to_string(),
                name: Some("Test Executor 1".to_string()),
                metadata: None,
            },
            ExecutorConfig {
                id: "test-executor-2".to_string(),
                grpc_address: "127.0.0.1:50052".to_string(),
                name: Some("Test Executor 2".to_string()),
                metadata: None,
            },
        ],
        health_check_interval: Duration::from_secs(30),
        health_check_timeout: Duration::from_secs(5),
        max_retry_attempts: 3,
        auto_recovery: true,
    }
}

async fn create_test_db() -> RegistrationDb {
    let db_config = DatabaseConfig {
        url: "sqlite::memory:".to_string(),
        max_connections: 5,
        min_connections: 1,
        connection_timeout: Duration::from_secs(10),
        idle_timeout: Duration::from_secs(300),
        max_lifetime: Duration::from_secs(3600),
    };

    RegistrationDb::new(&db_config).await.unwrap()
}
