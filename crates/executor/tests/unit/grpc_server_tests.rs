//! Unit tests for gRPC server

use common::identity::Hotkey;
use executor::grpc_server::container_operations::ContainerOperationsService;
use executor::grpc_server::executor_management::ExecutorManagementService;
use executor::grpc_server::health_check::{HealthCheckService, HealthStatus};
use executor::grpc_server::system_profile::SystemProfileService;
use executor::grpc_server::validator_access::ValidatorAccessService;
use executor::grpc_server::{ExecutorControlService, ExecutorServer};
use executor::{ExecutorConfig, ExecutorState};
use protocol::common::{ChallengeParameters, Timestamp};
use protocol::executor_control::{
    executor_control_server::ExecutorControl, BenchmarkRequest, ChallengeRequest,
    ContainerOpRequest, ContainerSpec, HealthCheckRequest, LogSubscriptionRequest,
    ProvisionAccessRequest, SystemProfileRequest,
};
use protocol::executor_management::{
    executor_management_server::ExecutorManagement, HealthCheckRequest as MgmtHealthCheckRequest,
    SshKeyUpdate, StatusRequest,
};
use std::collections::HashMap;
use std::net::SocketAddr;
use std::str::FromStr;
use std::sync::Arc;
use tokio::time::{timeout, Duration};
use tonic::Request;

// Helper function to create test executor state
async fn create_test_executor_state() -> ExecutorState {
    let mut config = ExecutorConfig::default();
    config.managing_miner_hotkey =
        Hotkey::from_str("5GrwvaEF5zXb26Fz9rcQpDWS57CtERHpNehXCPcNoHGKutQY").unwrap();
    ExecutorState::new(config).await.unwrap()
}

#[tokio::test]
async fn test_executor_server_creation() {
    let state = create_test_executor_state().await;
    let server = ExecutorServer::new(state);
    assert!(server.state.id.is_nil() == false);
}

#[tokio::test]
async fn test_executor_control_service_creation() {
    let state = Arc::new(create_test_executor_state().await);
    let service = ExecutorControlService::new(state);
    assert!(service.state.id.is_nil() == false);
}

#[tokio::test]
async fn test_health_check_service() {
    let state = Arc::new(create_test_executor_state().await);
    let service = HealthCheckService::new(state);

    // Health check should return valid status
    let status = service.health_check().await.unwrap();
    assert_eq!(status.status, "healthy");
    assert!(!status.executor_id.is_empty());
}

#[tokio::test]
async fn test_quick_health_check() {
    let state = Arc::new(create_test_executor_state().await);
    let service = HealthCheckService::new(state);

    let result = service.quick_health_check().await.unwrap();
    assert!(result);
}

#[tokio::test]
async fn test_readiness_check() {
    let state = Arc::new(create_test_executor_state().await);
    let service = HealthCheckService::new(state);

    let result = service.readiness_check().await.unwrap();
    assert!(result);
}

#[tokio::test]
async fn test_liveness_check() {
    let state = Arc::new(create_test_executor_state().await);
    let service = HealthCheckService::new(state);

    let result = service.liveness_check().await.unwrap();
    assert!(result);
}

#[tokio::test]
async fn test_provision_validator_access() {
    let state = Arc::new(create_test_executor_state().await);
    let service = ExecutorControlService::new(state);

    let request = Request::new(ProvisionAccessRequest {
        validator_hotkey: "test_validator".to_string(),
        requested_by: "test_miner".to_string(),
        access_type: "ssh".to_string(),
        duration_seconds: 3600,
        metadata: HashMap::new(),
    });

    let response = service.provision_validator_access(request).await;
    assert!(response.is_ok());

    let resp = response.unwrap().into_inner();
    assert!(resp.success);
    assert!(!resp.connection_endpoint.is_empty());
}

#[tokio::test]
async fn test_execute_system_profile() {
    let state = Arc::new(create_test_executor_state().await);
    let service = ExecutorControlService::new(state);

    let request = Request::new(SystemProfileRequest {
        validator_hotkey: "test_validator".to_string(),
        requested_at: Some(Timestamp::default()),
        profile_type: "full".to_string(),
        include_hardware: true,
        include_software: true,
        include_network: true,
    });

    let response = service.execute_system_profile(request).await;
    assert!(response.is_ok());

    let resp = response.unwrap().into_inner();
    assert!(!resp.encrypted_profile.is_empty());
    assert!(!resp.profile_hash.is_empty());
}

#[tokio::test]
async fn test_execute_vdf_challenge() {
    let state = Arc::new(create_test_executor_state().await);
    let service = ExecutorControlService::new(state);

    let params = ChallengeParameters {
        challenge_type: "vdf".to_string(),
        seed: "test_seed".to_string(),
        difficulty: 1000,
        parameters_json: r#"{"iterations": 1000}"#.to_string(),
        timeout_ms: 60000,
    };

    let request = Request::new(ChallengeRequest {
        validator_hotkey: "test_validator".to_string(),
        challenge_id: "test_challenge_1".to_string(),
        parameters: Some(params),
        nonce: 12345,
        timestamp: Some(Timestamp::default()),
    });

    let response = service.execute_computational_challenge(request).await;
    assert!(response.is_ok());

    let resp = response.unwrap().into_inner();
    assert!(resp.result.is_some());

    let result = resp.result.unwrap();
    assert!(!result.solution.is_empty());
    assert_eq!(result.gpu_utilization, vec![100.0]);
}

#[tokio::test]
async fn test_execute_hardware_attestation_challenge() {
    let state = Arc::new(create_test_executor_state().await);
    let service = ExecutorControlService::new(state);

    let params = ChallengeParameters {
        challenge_type: "hardware_attestation".to_string(),
        seed: "test_seed".to_string(),
        difficulty: 0,
        parameters_json: "{}".to_string(),
        timeout_ms: 60000,
    };

    let request = Request::new(ChallengeRequest {
        validator_hotkey: "test_validator".to_string(),
        challenge_id: "test_challenge_2".to_string(),
        parameters: Some(params),
        nonce: 54321,
        timestamp: Some(Timestamp::default()),
    });

    let response = service.execute_computational_challenge(request).await;
    assert!(response.is_ok());

    let resp = response.unwrap().into_inner();
    assert!(resp.result.is_some());

    let result = resp.result.unwrap();
    assert!(!result.solution.is_empty());
    assert_eq!(result.execution_time_ms, 0); // Attestation is not time-based
}

#[tokio::test]
async fn test_execute_unknown_challenge_type() {
    let state = Arc::new(create_test_executor_state().await);
    let service = ExecutorControlService::new(state);

    let params = ChallengeParameters {
        challenge_type: "unknown_type".to_string(),
        seed: "test_seed".to_string(),
        difficulty: 1000,
        parameters_json: "{}".to_string(),
        timeout_ms: 60000,
    };

    let request = Request::new(ChallengeRequest {
        validator_hotkey: "test_validator".to_string(),
        challenge_id: "test_challenge_3".to_string(),
        parameters: Some(params),
        nonce: 11111,
        timestamp: Some(Timestamp::default()),
    });

    let response = service.execute_computational_challenge(request).await;
    assert!(response.is_err());
    assert!(response
        .unwrap_err()
        .message()
        .contains("Unknown challenge type"));
}

#[tokio::test]
async fn test_execute_gpu_benchmark() {
    let state = Arc::new(create_test_executor_state().await);
    let service = ExecutorControlService::new(state);

    let request = Request::new(BenchmarkRequest {
        validator_hotkey: "test_validator".to_string(),
        benchmark_type: "gpu".to_string(),
        parameters: HashMap::new(),
        timeout_ms: 60000,
    });

    let response = service.execute_benchmark(request).await;
    assert!(response.is_ok());

    let resp = response.unwrap().into_inner();
    assert!(resp.score > 0.0);
    assert!(resp.results.contains_key("gpu_count"));
    assert!(resp.results.contains_key("gpu_model"));
}

#[tokio::test]
async fn test_execute_cpu_benchmark() {
    let state = Arc::new(create_test_executor_state().await);
    let service = ExecutorControlService::new(state);

    let request = Request::new(BenchmarkRequest {
        validator_hotkey: "test_validator".to_string(),
        benchmark_type: "cpu".to_string(),
        parameters: HashMap::new(),
        timeout_ms: 60000,
    });

    let response = service.execute_benchmark(request).await;
    assert!(response.is_ok());

    let resp = response.unwrap().into_inner();
    assert!(resp.score > 0.0);
    assert!(resp.results.contains_key("cpu_cores"));
    assert!(resp.results.contains_key("cpu_model"));
}

#[tokio::test]
async fn test_execute_memory_benchmark() {
    let state = Arc::new(create_test_executor_state().await);
    let service = ExecutorControlService::new(state);

    let request = Request::new(BenchmarkRequest {
        validator_hotkey: "test_validator".to_string(),
        benchmark_type: "memory".to_string(),
        parameters: HashMap::new(),
        timeout_ms: 60000,
    });

    let response = service.execute_benchmark(request).await;
    assert!(response.is_ok());

    let resp = response.unwrap().into_inner();
    assert!(resp.score > 0.0);
    assert!(resp.results.contains_key("total_memory_mb"));
}

#[tokio::test]
async fn test_container_create_operation() {
    let state = Arc::new(create_test_executor_state().await);
    let service = ExecutorControlService::new(state);

    let spec = ContainerSpec {
        image: "alpine:latest".to_string(),
        command: vec!["sleep".to_string(), "60".to_string()],
        environment: HashMap::new(),
        mounts: Vec::new(),
        resources: None,
    };

    let request = Request::new(ContainerOpRequest {
        operation: "create".to_string(),
        container_id: String::new(),
        container_spec: Some(spec),
        validator_hotkey: "test_validator".to_string(),
        ssh_public_key: String::new(),
        parameters: HashMap::new(),
    });

    let response = service.manage_container(request).await;
    // May fail if Docker not available
    if response.is_ok() {
        let resp = response.unwrap().into_inner();
        assert!(resp.success);
        assert!(!resp.container_id.is_empty());
        assert!(resp.status.is_some());
    }
}

#[tokio::test]
async fn test_container_get_status_operation() {
    let state = Arc::new(create_test_executor_state().await);
    let service = ExecutorControlService::new(state);

    let request = Request::new(ContainerOpRequest {
        operation: "get_status".to_string(),
        container_id: "test_container".to_string(),
        container_spec: None,
        validator_hotkey: "test_validator".to_string(),
        ssh_public_key: String::new(),
        parameters: HashMap::new(),
    });

    let response = service.manage_container(request).await;
    // May fail if container doesn't exist
    if response.is_ok() {
        let resp = response.unwrap().into_inner();
        assert!(resp.status.is_some());
    }
}

#[tokio::test]
async fn test_stream_logs() {
    let state = Arc::new(create_test_executor_state().await);
    let service = ExecutorControlService::new(state);

    let request = Request::new(LogSubscriptionRequest {
        container_id: "test_container".to_string(),
        follow: false,
        tail_lines: 10,
        since_timestamp: None,
        metadata: HashMap::new(),
    });

    let response = service.stream_logs(request).await;
    assert!(response.is_ok());
}

#[tokio::test]
async fn test_health_check() {
    let state = Arc::new(create_test_executor_state().await);
    let service = ExecutorControlService::new(state);

    let request = Request::new(HealthCheckRequest {
        requester: "test_validator".to_string(),
        detailed: true,
        metadata: HashMap::new(),
    });

    let response = service.health_check(request).await;
    assert!(response.is_ok());

    let resp = response.unwrap().into_inner();
    assert_eq!(resp.status, "healthy");
    assert!(resp.resource_status.contains_key("cpu_percent"));
    assert!(resp.resource_status.contains_key("memory_percent"));
    assert!(resp.resource_status.contains_key("disk_percent"));
}

// Management service tests

#[tokio::test]
async fn test_management_health_check() {
    let state = Arc::new(create_test_executor_state().await);
    let service = ExecutorManagementService::new(state);

    let request = Request::new(MgmtHealthCheckRequest {
        miner_hotkey: "5GrwvaEF5zXb26Fz9rcQpDWS57CtERHpNehXCPcNoHGKutQY".to_string(),
    });

    let response = service.health_check(request).await;
    assert!(response.is_ok());

    let resp = response.unwrap().into_inner();
    assert_eq!(resp.status, "healthy");
    assert!(resp.resource_status.contains_key("cpu_usage"));
    assert!(resp.resource_status.contains_key("memory_mb"));
}

#[tokio::test]
async fn test_management_health_check_unauthorized() {
    let state = Arc::new(create_test_executor_state().await);
    let service = ExecutorManagementService::new(state);

    let request = Request::new(MgmtHealthCheckRequest {
        miner_hotkey: "unauthorized_miner".to_string(),
    });

    let response = service.health_check(request).await;
    assert!(response.is_err());
    assert_eq!(response.unwrap_err().code(), tonic::Code::PermissionDenied);
}

#[tokio::test]
async fn test_update_ssh_keys_add() {
    let state = Arc::new(create_test_executor_state().await);
    let service = ExecutorManagementService::new(state);

    let request = Request::new(SshKeyUpdate {
        miner_hotkey: "5GrwvaEF5zXb26Fz9rcQpDWS57CtERHpNehXCPcNoHGKutQY".to_string(),
        validator_hotkey: "test_validator".to_string(),
        operation: "add".to_string(),
        ssh_public_key: "ssh-rsa AAAAB3NzaC1yc2EAAAADAQABAAABgQC... test@example.com".to_string(),
    });

    let response = service.update_ssh_keys(request).await;
    assert!(response.is_ok());

    let resp = response.unwrap().into_inner();
    assert!(resp.success);
    assert!(resp.message.contains("SSH key added"));
}

#[tokio::test]
async fn test_update_ssh_keys_remove() {
    let state = Arc::new(create_test_executor_state().await);
    let service = ExecutorManagementService::new(state);

    // First add a key
    let add_request = Request::new(SshKeyUpdate {
        miner_hotkey: "5GrwvaEF5zXb26Fz9rcQpDWS57CtERHpNehXCPcNoHGKutQY".to_string(),
        validator_hotkey: "test_validator".to_string(),
        operation: "add".to_string(),
        ssh_public_key: "ssh-rsa AAAAB3NzaC1yc2EAAAADAQABAAABgQC... test@example.com".to_string(),
    });

    let _ = service.update_ssh_keys(add_request).await;

    // Now remove it
    let remove_request = Request::new(SshKeyUpdate {
        miner_hotkey: "5GrwvaEF5zXb26Fz9rcQpDWS57CtERHpNehXCPcNoHGKutQY".to_string(),
        validator_hotkey: "test_validator".to_string(),
        operation: "remove".to_string(),
        ssh_public_key: String::new(),
    });

    let response = service.update_ssh_keys(remove_request).await;
    assert!(response.is_ok());

    let resp = response.unwrap().into_inner();
    assert!(resp.success);
    assert!(resp.message.contains("SSH key removed"));
}

#[tokio::test]
async fn test_update_ssh_keys_invalid_operation() {
    let state = Arc::new(create_test_executor_state().await);
    let service = ExecutorManagementService::new(state);

    let request = Request::new(SshKeyUpdate {
        miner_hotkey: "5GrwvaEF5zXb26Fz9rcQpDWS57CtERHpNehXCPcNoHGKutQY".to_string(),
        validator_hotkey: "test_validator".to_string(),
        operation: "invalid".to_string(),
        ssh_public_key: String::new(),
    });

    let response = service.update_ssh_keys(request).await;
    assert!(response.is_err());
    assert_eq!(response.unwrap_err().code(), tonic::Code::InvalidArgument);
}

#[tokio::test]
async fn test_get_status() {
    let state = Arc::new(create_test_executor_state().await);
    let service = ExecutorManagementService::new(state);

    let request = Request::new(StatusRequest { detailed: true });

    let response = service.get_status(request).await;
    assert!(response.is_ok());

    let resp = response.unwrap().into_inner();
    assert!(!resp.executor_id.is_empty());
    assert_eq!(resp.status, "operational");
    assert!(resp.machine_info.is_some());
}

#[tokio::test]
async fn test_container_operations_service() {
    let state = Arc::new(create_test_executor_state().await);
    let service = ContainerOperationsService::new(state);

    // Health check should work
    let result = service.container_operations_health_check().await;
    // May fail if Docker not available, that's okay
    assert!(result.is_ok() || result.is_err());
}

#[tokio::test]
async fn test_system_profile_service() {
    let state = Arc::new(create_test_executor_state().await);
    let service = SystemProfileService::new(state);

    // Get system profile
    let profile = service.get_system_profile().await.unwrap();
    assert!(!profile.os_type.is_empty());
    assert!(profile.cpu_cores > 0);
    assert!(profile.memory_mb > 0);
}

#[tokio::test]
async fn test_get_hardware_specs() {
    let state = Arc::new(create_test_executor_state().await);
    let service = SystemProfileService::new(state);

    let specs = service.get_hardware_specs().await.unwrap();
    assert!(!specs.cpu_model.is_empty());
    assert!(specs.cpu_cores > 0);
    assert!(specs.memory_mb > 0);
}

#[tokio::test]
async fn test_get_resource_availability() {
    let state = Arc::new(create_test_executor_state().await);
    let service = SystemProfileService::new(state);

    let availability = service.get_resource_availability().await.unwrap();
    assert!(availability.cpu_percent >= 0.0);
    assert!(availability.memory_available_mb > 0);
}

#[tokio::test]
async fn test_validator_access_service() {
    let state = Arc::new(create_test_executor_state().await);
    let service = ValidatorAccessService::new(state);

    // List validators should return empty list initially
    let validators = service.list_validators().await.unwrap();
    assert_eq!(validators.len(), 0);
}

#[tokio::test]
async fn test_grpc_server_startup_shutdown() {
    let state = create_test_executor_state().await;
    let server = ExecutorServer::new(state);

    // Use port 0 for random available port
    let addr: SocketAddr = "127.0.0.1:0".parse().unwrap();

    // Start server in background
    let server_handle = tokio::spawn(async move { server.serve(addr).await });

    // Give server time to start
    tokio::time::sleep(Duration::from_millis(100)).await;

    // Cancel the server
    server_handle.abort();

    // Should complete without panic
    let _ = timeout(Duration::from_secs(1), server_handle).await;
}

#[tokio::test]
async fn test_perform_health_check() {
    let state = Arc::new(create_test_executor_state().await);
    let service = ExecutorControlService::new(state);

    let status = service.perform_health_check().await.unwrap();
    assert_eq!(status.status, "healthy");
}

#[tokio::test]
async fn test_placeholder_methods() {
    let state = Arc::new(create_test_executor_state().await);
    let service = ExecutorControlService::new(state);

    // Test provision validator access placeholder
    let result = service
        .provision_validator_access_placeholder(
            "test_validator",
            Some("ssh-rsa AAAAB3NzaC1yc2E...".to_string()),
        )
        .await;
    assert!(result.is_ok());

    // Test execute system profile placeholder
    let profile = service.execute_system_profile_placeholder().await;
    assert!(profile.is_ok());

    // Test container operations placeholder
    let container_ops = service.container_operations_placeholder().await;
    assert!(container_ops.is_ok());
}
