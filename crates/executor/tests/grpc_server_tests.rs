//! Unit tests for gRPC server

use common::identity::Hotkey;
use executor::grpc_server::executor_management::ExecutorManagementService;
use executor::grpc_server::{ExecutorControlService, ExecutorServer};
use executor::{ExecutorConfig, ExecutorState};
use protocol::common::Timestamp;
use protocol::executor_control::{
    executor_control_server::ExecutorControl, HealthCheckRequest, SystemProfileRequest,
};
use protocol::executor_management::{
    executor_management_server::ExecutorManagement, HealthCheckRequest as MgmtHealthCheckRequest,
    StatusRequest,
};
use std::collections::HashMap;
use std::net::SocketAddr;
use std::str::FromStr;
use std::sync::Arc;
use tonic::Request;

// Helper function to create test executor state
async fn create_test_executor_state() -> ExecutorState {
    let config = ExecutorConfig {
        managing_miner_hotkey: Hotkey::from_str("5GrwvaEF5zXb26Fz9rcQpDWS57CtERHpNehXCPcNoHGKutQY")
            .unwrap(),
        ..Default::default()
    };
    ExecutorState::new(config).await.unwrap()
}

#[tokio::test]
async fn test_executor_server_creation() {
    let state = create_test_executor_state().await;
    let server = ExecutorServer::new(state);

    // Server should be created successfully
    assert!(!server.state().id.as_uuid().is_nil());
}

#[tokio::test]
async fn test_executor_control_service_creation() {
    let state = Arc::new(create_test_executor_state().await);
    let _service = ExecutorControlService::new(state.clone());

    // Service should have valid state
    assert!(!state.id.as_uuid().is_nil());
}

#[tokio::test]
async fn test_health_check_endpoint() {
    let state = Arc::new(create_test_executor_state().await);
    let service = ExecutorControlService::new(state);

    let request = Request::new(HealthCheckRequest {
        requester: "test_miner".to_string(),
        check_type: "basic".to_string(),
    });

    let response = service.health_check(request).await.unwrap();
    let health_response = response.into_inner();

    assert_eq!(health_response.status, "healthy");
    assert!(!health_response.docker_status.is_empty());
    assert!(health_response.uptime_seconds > 0);
}

#[tokio::test]
async fn test_system_profile_endpoint() {
    let state = Arc::new(create_test_executor_state().await);
    let service = ExecutorControlService::new(state);

    let request = Request::new(SystemProfileRequest {
        validator_hotkey: "test_validator".to_string(),
        session_key: "test_session".to_string(),
        key_mapping: HashMap::new(),
        profile_depth: "basic".to_string(),
        include_benchmarks: false,
    });

    let response = service.execute_system_profile(request).await.unwrap();
    let profile_response = response.into_inner();

    assert!(!profile_response.encrypted_profile.is_empty());
    assert!(!profile_response.profile_hash.is_empty());
    assert!(profile_response.collected_at.is_some());
}

#[tokio::test]
async fn test_executor_management_service() {
    let state = Arc::new(create_test_executor_state().await);
    let service = ExecutorManagementService::new(state);

    // Test health check from authorized miner
    let request = Request::new(MgmtHealthCheckRequest {
        miner_hotkey: "5GrwvaEF5zXb26Fz9rcQpDWS57CtERHpNehXCPcNoHGKutQY".to_string(),
        check_type: "health".to_string(),
        timestamp: Some(Timestamp::default()),
    });

    let response = service.health_check(request).await.unwrap();
    let health_response = response.into_inner();

    assert_eq!(health_response.status, "healthy");
    assert!(!health_response.docker_status.is_empty());
}

#[tokio::test]
async fn test_unauthorized_miner() {
    let state = Arc::new(create_test_executor_state().await);
    let service = ExecutorManagementService::new(state);

    // Test health check from unauthorized miner
    let request = Request::new(MgmtHealthCheckRequest {
        miner_hotkey: "5FHneW46xGXgs5mUiveU4sbTyGBzmstUspZC92UhjJM694ty".to_string(), // Wrong hotkey
        check_type: "health".to_string(),
        timestamp: Some(Timestamp::default()),
    });

    let response = service.health_check(request).await;
    assert!(response.is_err());

    let error = response.unwrap_err();
    assert_eq!(error.code(), tonic::Code::PermissionDenied);
}

#[tokio::test]
async fn test_get_status_endpoint() {
    let state = Arc::new(create_test_executor_state().await);
    let service = ExecutorManagementService::new(state);

    let request = Request::new(StatusRequest { detailed: true });

    let response = service.get_status(request).await.unwrap();
    let status_response = response.into_inner();

    assert_eq!(status_response.status, "operational");
    assert!(status_response.uptime_seconds > 0);
    // Resource usage may or may not be present depending on monitoring state
    if let Some(resource_usage) = status_response.resource_usage {
        assert!(resource_usage.cpu_percent >= 0.0);
    }
}

#[test]
fn test_server_address_parsing() {
    let addr: SocketAddr = "127.0.0.1:50051".parse().unwrap();
    assert_eq!(addr.port(), 50051);
    assert_eq!(addr.ip().to_string(), "127.0.0.1");
}

#[test]
fn test_executor_config_for_grpc() {
    let config = ExecutorConfig::default();

    assert_eq!(config.server.host, "0.0.0.0");
    assert_eq!(config.server.port, 50051);
    assert_eq!(
        config.managing_miner_hotkey.as_str(),
        "5GrwvaEF5zXb26Fz9rcQpDWS57CtERHpNehXCPcNoHGKutQY"
    );
}

#[tokio::test]
async fn test_concurrent_requests() {
    let state = Arc::new(create_test_executor_state().await);
    let service = Arc::new(ExecutorControlService::new(state));

    // Spawn multiple concurrent requests
    let mut handles = vec![];

    for i in 0..5 {
        let service_clone = service.clone();
        let handle = tokio::spawn(async move {
            let request = Request::new(HealthCheckRequest {
                requester: format!("test_miner_{i}"),
                check_type: "basic".to_string(),
            });

            service_clone.health_check(request).await
        });
        handles.push(handle);
    }

    // All requests should succeed
    for handle in handles {
        let result = handle.await.unwrap();
        assert!(result.is_ok());
        let response = result.unwrap();
        assert_eq!(response.into_inner().status, "healthy");
    }
}

#[tokio::test]
async fn test_state_sharing() {
    let state = Arc::new(create_test_executor_state().await);

    // Create multiple services sharing the same state
    let control_service = ExecutorControlService::new(state.clone());
    let management_service = ExecutorManagementService::new(state.clone());

    // Both services should see the same executor ID
    let control_request = Request::new(HealthCheckRequest {
        requester: "test".to_string(),
        check_type: "basic".to_string(),
    });

    let management_request = Request::new(MgmtHealthCheckRequest {
        miner_hotkey: "5GrwvaEF5zXb26Fz9rcQpDWS57CtERHpNehXCPcNoHGKutQY".to_string(),
        check_type: "health".to_string(),
        timestamp: Some(Timestamp::default()),
    });

    let control_response = control_service.health_check(control_request).await.unwrap();
    let management_response = management_service
        .health_check(management_request)
        .await
        .unwrap();

    // Both should report healthy status
    assert_eq!(
        control_response.into_inner().status,
        management_response.into_inner().status
    );
}
