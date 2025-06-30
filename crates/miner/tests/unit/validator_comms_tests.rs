//! Unit tests for ValidatorHandlerService

use common::config::DatabaseConfig;
use common::identity::Hotkey;
use miner::config::{ExecutorConfig, ExecutorManagementConfig, MinerConfig};
use miner::executor_fleet_manager::ExecutorFleetManager;
use miner::persistence::RegistrationDb;
use miner::validator_comms::ValidatorHandlerService;
use protocol::miner::miner_discovery_server::MinerDiscovery;
use protocol::miner::{ListExecutorsRequest, SshAccessRequest, ValidatorAuthRequest};
use std::time::Duration;
use tonic::Request;

#[tokio::test]
async fn test_validator_handler_service_new() {
    let config = create_test_miner_config();
    let fleet_manager = create_test_fleet_manager().await;
    let db = create_test_db().await;

    let service = ValidatorHandlerService::new(config, fleet_manager, db);
    assert!(service.is_ok());
}

#[tokio::test]
async fn test_authenticate_validator_invalid_signature() {
    let config = create_test_miner_config();
    let fleet_manager = create_test_fleet_manager().await;
    let db = create_test_db().await;

    let service = ValidatorHandlerService::new(config, fleet_manager, db).unwrap();

    let request = Request::new(ValidatorAuthRequest {
        validator_hotkey: "5GrwvaEF5zXb26Fz9rcQpDWS57CtERHpNehXCPcNoHGKutQY".to_string(),
        timestamp: chrono::Utc::now().timestamp(),
        signature: "invalid_signature".to_string(),
        requested_executors: 1,
    });

    let response = service.authenticate_validator(request).await;
    assert!(response.is_err());
    assert_eq!(response.unwrap_err().code(), tonic::Code::Unauthenticated);
}

#[tokio::test]
async fn test_authenticate_validator_expired_timestamp() {
    let config = create_test_miner_config();
    let fleet_manager = create_test_fleet_manager().await;
    let db = create_test_db().await;

    let service = ValidatorHandlerService::new(config, fleet_manager, db).unwrap();

    // Timestamp from 10 minutes ago
    let old_timestamp = chrono::Utc::now().timestamp() - 600;

    let request = Request::new(ValidatorAuthRequest {
        validator_hotkey: "5GrwvaEF5zXb26Fz9rcQpDWS57CtERHpNehXCPcNoHGKutQY".to_string(),
        timestamp: old_timestamp,
        signature: "some_signature".to_string(),
        requested_executors: 1,
    });

    let response = service.authenticate_validator(request).await;
    assert!(response.is_err());
    assert_eq!(response.unwrap_err().code(), tonic::Code::Unauthenticated);
}

#[tokio::test]
async fn test_list_executors_no_session() {
    let config = create_test_miner_config();
    let fleet_manager = create_test_fleet_manager().await;
    let db = create_test_db().await;

    let service = ValidatorHandlerService::new(config, fleet_manager, db).unwrap();

    let request = Request::new(ListExecutorsRequest {
        session_token: "invalid_token".to_string(),
        filter_available: true,
    });

    let response = service.list_executors(request).await;
    assert!(response.is_err());
    assert_eq!(response.unwrap_err().code(), tonic::Code::Unauthenticated);
}

#[tokio::test]
async fn test_request_ssh_access_no_session() {
    let config = create_test_miner_config();
    let fleet_manager = create_test_fleet_manager().await;
    let db = create_test_db().await;

    let service = ValidatorHandlerService::new(config, fleet_manager, db).unwrap();

    let request = Request::new(SshAccessRequest {
        session_token: "invalid_token".to_string(),
        executor_ids: vec!["exec1".to_string()],
        ssh_public_key: "ssh-rsa AAAAB3NzaC1yc2E...".to_string(),
        duration_seconds: 3600,
    });

    let response = service.request_ssh_access(request).await;
    assert!(response.is_err());
    assert_eq!(response.unwrap_err().code(), tonic::Code::Unauthenticated);
}

// Helper functions

fn create_test_miner_config() -> MinerConfig {
    MinerConfig {
        hotkey: Hotkey("5GrwvaEF5zXb26Fz9rcQpDWS57CtERHpNehXCPcNoHGKutQY".to_string()),
        netuid: 1,
        network: "finney".to_string(),
        wallet_name: "test".to_string(),
        hotkey_name: "test".to_string(),
        max_concurrent_validators: 10,
        session_timeout: Duration::from_secs(3600),
        rate_limit_per_validator: 100,
        discovery: Default::default(),
    }
}

async fn create_test_fleet_manager() -> ExecutorFleetManager {
    let config = ExecutorManagementConfig {
        executors: vec![ExecutorConfig {
            id: "test-executor-1".to_string(),
            grpc_address: "127.0.0.1:50051".to_string(),
            name: Some("Test Executor 1".to_string()),
            metadata: None,
        }],
        health_check_interval: Duration::from_secs(30),
        health_check_timeout: Duration::from_secs(5),
        max_retry_attempts: 3,
        auto_recovery: true,
    };

    let db = create_test_db().await;
    ExecutorFleetManager::new(config, db).await.unwrap()
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
