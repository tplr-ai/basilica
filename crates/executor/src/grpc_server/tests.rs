//! Tests for gRPC server implementation

use super::*;
use crate::test_utils::create_test_executor_state;
use protocol::common::{ChallengeParameters, Timestamp};
use protocol::executor_control::{
    BenchmarkRequest, ChallengeRequest, ContainerOpRequest, ContainerSpec, HealthCheckRequest,
    LogSubscriptionRequest, ProvisionAccessRequest, SystemProfileRequest,
};
use protocol::executor_management::{
    HealthCheckRequest as MgmtHealthCheckRequest, SshKeyUpdate, StatusRequest,
};
use std::collections::HashMap;
use tonic::Request;

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
    assert!(response.is_ok());

    let resp = response.unwrap().into_inner();
    assert!(resp.success);
    assert!(!resp.container_id.is_empty());
    assert!(resp.status.is_some());
}

#[tokio::test]
async fn test_container_delete_operation() {
    let state = Arc::new(create_test_executor_state().await);
    let service = ExecutorControlService::new(state);

    // First create a container
    let spec = ContainerSpec {
        image: "alpine:latest".to_string(),
        command: vec!["sleep".to_string(), "60".to_string()],
        environment: HashMap::new(),
        mounts: Vec::new(),
        resources: None,
    };

    let create_request = Request::new(ContainerOpRequest {
        operation: "create".to_string(),
        container_id: String::new(),
        container_spec: Some(spec),
        validator_hotkey: "test_validator".to_string(),
        ssh_public_key: String::new(),
        parameters: HashMap::new(),
    });

    let create_response = service.manage_container(create_request).await.unwrap();
    let container_id = create_response.into_inner().container_id;

    // Now delete it
    let mut params = HashMap::new();
    params.insert("force".to_string(), "true".to_string());

    let delete_request = Request::new(ContainerOpRequest {
        operation: "delete".to_string(),
        container_id: container_id.clone(),
        container_spec: None,
        validator_hotkey: "test_validator".to_string(),
        ssh_public_key: String::new(),
        parameters: params,
    });

    let response = service.manage_container(delete_request).await;
    assert!(response.is_ok());

    let resp = response.unwrap().into_inner();
    assert!(resp.success);
    assert_eq!(resp.container_id, container_id);
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
    let service = executor_management::ExecutorManagementService::new(state);

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
    let service = executor_management::ExecutorManagementService::new(state);

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
    let service = executor_management::ExecutorManagementService::new(state);

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
    let service = executor_management::ExecutorManagementService::new(state);

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
async fn test_get_status() {
    let state = Arc::new(create_test_executor_state().await);
    let service = executor_management::ExecutorManagementService::new(state);

    let request = Request::new(StatusRequest { detailed: true });

    let response = service.get_status(request).await;
    assert!(response.is_ok());

    let resp = response.unwrap().into_inner();
    assert!(!resp.executor_id.is_empty());
    assert_eq!(resp.status, "operational");
    assert!(resp.machine_info.is_some());
}

#[cfg(test)]
mod module_tests {
    use super::*;
    use crate::grpc_server::container_operations::ContainerOperationsService;
    use crate::grpc_server::health_check::HealthCheckService;
    use crate::grpc_server::system_profile::SystemProfileService;
    use crate::grpc_server::validator_access::ValidatorAccessService;

    #[tokio::test]
    async fn test_health_check_service() {
        let state = Arc::new(create_test_executor_state().await);
        let service = HealthCheckService::new(state);

        let result = service.health_check().await;
        assert!(result.is_ok());

        let status = result.unwrap();
        assert_eq!(status.status, "healthy");
    }

    #[tokio::test]
    async fn test_system_profile_service() {
        let state = Arc::new(create_test_executor_state().await);
        let service = SystemProfileService::new(state);

        let result = service.execute_system_profile().await;
        assert!(result.is_ok());

        let profile = result.unwrap();
        assert!(!profile.is_empty());
    }

    #[tokio::test]
    async fn test_container_operations_service() {
        let state = Arc::new(create_test_executor_state().await);
        let service = ContainerOperationsService::new(state);

        let result = service.container_operations_health_check().await;
        assert!(result.is_ok());
    }

    #[tokio::test]
    async fn test_validator_access_service() {
        let state = Arc::new(create_test_executor_state().await);
        let service = ValidatorAccessService::new(state);

        let result = service
            .provision_ssh_access(
                "test_validator",
                "ssh-rsa AAAAB3NzaC1yc2EAAAADAQABAAABgQC... test@example.com".to_string(),
            )
            .await;
        assert!(result.is_ok());
    }
}
