//! Integration tests for executor crate

use common::identity::Hotkey;
use executor::grpc_server::ExecutorServer;
use executor::{ExecutorConfig, ExecutorState};
use protocol::common::Timestamp;
use protocol::executor_control::executor_control_client::ExecutorControlClient;
use protocol::executor_control::{HealthCheckRequest, SystemProfileRequest};
use protocol::executor_management::executor_management_client::ExecutorManagementClient;
use protocol::executor_management::{HealthCheckRequest as MgmtHealthCheckRequest, StatusRequest};
use std::net::SocketAddr;
use std::str::FromStr;
use tokio::time::Duration;
use tonic::transport::Channel;

#[tokio::test]
async fn test_executor_startup_and_shutdown() {
    let config = ExecutorConfig::default();
    let state = ExecutorState::new(config).await.unwrap();

    // Verify state initialized
    assert!(!state.id.as_uuid().is_nil());
}

#[tokio::test]
async fn test_executor_grpc_services() {
    let mut config = ExecutorConfig::default();
    config.server.port = 0; // Use random port

    let state = ExecutorState::new(config).await.unwrap();
    let server = ExecutorServer::new(state);

    // Start server on localhost with random port
    let addr: SocketAddr = "127.0.0.1:0".parse().unwrap();
    let actual_addr = start_test_server(server, addr).await;

    // Give server time to start
    tokio::time::sleep(Duration::from_millis(100)).await;

    // Connect to ExecutorControl service
    let channel = Channel::from_shared(format!("http://{actual_addr}"))
        .unwrap()
        .connect()
        .await
        .unwrap();

    let mut client = ExecutorControlClient::new(channel);

    // Test health check
    let request = tonic::Request::new(HealthCheckRequest {
        requester: "test".to_string(),
        check_type: "basic".to_string(),
    });

    let response = client.health_check(request).await.unwrap();
    let health_response = response.into_inner();
    assert_eq!(health_response.status, "healthy");
}

#[tokio::test]
async fn test_executor_management_service() {
    let mut config = ExecutorConfig::default();
    config.server.port = 0; // Use random port
    config.managing_miner_hotkey =
        Hotkey::from_str("5GrwvaEF5zXb26Fz9rcQpDWS57CtERHpNehXCPcNoHGKutQY").unwrap();

    let state = ExecutorState::new(config).await.unwrap();
    let server = ExecutorServer::new(state);

    // Start server
    let addr: SocketAddr = "127.0.0.1:0".parse().unwrap();
    let actual_addr = start_test_server(server, addr).await;

    // Give server time to start
    tokio::time::sleep(Duration::from_millis(100)).await;

    // Connect to ExecutorManagement service
    let channel = Channel::from_shared(format!("http://{actual_addr}"))
        .unwrap()
        .connect()
        .await
        .unwrap();

    let mut client = ExecutorManagementClient::new(channel);

    // Test health check from miner
    let request = tonic::Request::new(MgmtHealthCheckRequest {
        miner_hotkey: "5GrwvaEF5zXb26Fz9rcQpDWS57CtERHpNehXCPcNoHGKutQY".to_string(),
        check_type: "health".to_string(),
        timestamp: Some(Timestamp::default()),
    });

    let response = client.health_check(request).await.unwrap();
    let health_response = response.into_inner();
    assert_eq!(health_response.status, "healthy");

    // Test get status
    let request = tonic::Request::new(StatusRequest { detailed: true });

    let response = client.get_status(request).await.unwrap();
    let status_response = response.into_inner();
    assert_eq!(status_response.status, "operational");
}

#[tokio::test]
async fn test_system_profile_endpoint() {
    let config = ExecutorConfig::default();
    let state = ExecutorState::new(config).await.unwrap();
    let server = ExecutorServer::new(state);

    // Start server
    let addr: SocketAddr = "127.0.0.1:0".parse().unwrap();
    let actual_addr = start_test_server(server, addr).await;

    // Give server time to start
    tokio::time::sleep(Duration::from_millis(100)).await;

    // Connect and test system profile
    let channel = Channel::from_shared(format!("http://{actual_addr}"))
        .unwrap()
        .connect()
        .await
        .unwrap();

    let mut client = ExecutorControlClient::new(channel);

    let request = tonic::Request::new(SystemProfileRequest {
        validator_hotkey: "test_validator".to_string(),
        session_key: "test_session".to_string(),
        key_mapping: std::collections::HashMap::new(),
        profile_depth: "full".to_string(),
        include_benchmarks: true,
    });

    let response = client.execute_system_profile(request).await.unwrap();
    let profile = response.into_inner();

    // Should have encrypted profile
    assert!(!profile.encrypted_profile.is_empty());
    assert!(!profile.profile_hash.is_empty());
}

#[tokio::test]
async fn test_unauthorized_miner_access() {
    let mut config = ExecutorConfig::default();
    config.server.port = 0;
    config.managing_miner_hotkey =
        Hotkey::from_str("5GrwvaEF5zXb26Fz9rcQpDWS57CtERHpNehXCPcNoHGKutQY").unwrap();

    let state = ExecutorState::new(config).await.unwrap();
    let server = ExecutorServer::new(state);

    // Start server
    let addr: SocketAddr = "127.0.0.1:0".parse().unwrap();
    let actual_addr = start_test_server(server, addr).await;

    tokio::time::sleep(Duration::from_millis(100)).await;

    // Connect with wrong miner hotkey
    let channel = Channel::from_shared(format!("http://{actual_addr}"))
        .unwrap()
        .connect()
        .await
        .unwrap();

    let mut client = ExecutorManagementClient::new(channel);

    let request = tonic::Request::new(MgmtHealthCheckRequest {
        miner_hotkey: "5FHneW46xGXgs5mUiveU4sbTyGBzmstUspZC92UhjJM694ty".to_string(), // Wrong hotkey
        check_type: "health".to_string(),
        timestamp: Some(Timestamp::default()),
    });

    let response = client.health_check(request).await;
    assert!(response.is_err());

    let error = response.unwrap_err();
    assert_eq!(error.code(), tonic::Code::PermissionDenied);
}

// Helper function to start test server and get actual address
async fn start_test_server(server: ExecutorServer, addr: SocketAddr) -> SocketAddr {
    let listener = tokio::net::TcpListener::bind(addr).await.unwrap();
    let actual_addr = listener.local_addr().unwrap();

    let incoming =
        tonic::transport::server::TcpIncoming::from_listener(listener, true, None).unwrap();

    tokio::spawn(async move {
        let control_service =
            executor::grpc_server::ExecutorControlService::new(server.state().clone());
        let management_service =
            executor::grpc_server::executor_management::ExecutorManagementService::new(
                server.state().clone(),
            );
        tonic::transport::Server::builder()
            .add_service(protocol::executor_control::executor_control_server::ExecutorControlServer::new(
                control_service
            ))
            .add_service(protocol::executor_management::executor_management_server::ExecutorManagementServer::new(
                management_service
            ))
            .serve_with_incoming(incoming)
            .await
            .unwrap();
    });

    actual_addr
}
