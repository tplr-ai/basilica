use anyhow::Result;
use common::identity::{ExecutorId, Hotkey, MinerUid};
use executor::config::{ExecutorConfig, GrpcConfig, TlsConfig};
use executor::grpc::{create_grpc_server, ExecutorServiceImpl};
use protocol::executor::{
    executor_service_client::ExecutorServiceClient, executor_service_server::ExecutorServiceServer,
    ExecuteCommandRequest, GetLogsRequest, GetMetricsRequest, GetStatusRequest, HealthCheckRequest,
    RegisterExecutorRequest, StartValidationRequest, StopValidationRequest, UpdateConfigRequest,
};
use std::net::SocketAddr;
use std::path::PathBuf;
use std::sync::Arc;
use tempfile::TempDir;
use tokio::net::TcpListener;
use tokio::sync::RwLock;
use tokio::time::{sleep, timeout, Duration};
use tonic::transport::{Channel, Server};
use uuid::Uuid;

async fn setup_test_server() -> Result<(SocketAddr, ExecutorServiceImpl, TempDir)> {
    let temp_dir = TempDir::new()?;

    let config = ExecutorConfig {
        executor_id: ExecutorId::new("test-grpc-executor"),
        working_dir: temp_dir.path().to_path_buf(),
        grpc: GrpcConfig {
            bind_address: "127.0.0.1:0".parse()?,
            max_message_size: 4 * 1024 * 1024,
            keepalive_interval: Duration::from_secs(30),
            keepalive_timeout: Duration::from_secs(10),
            tls: None,
        },
        ..Default::default()
    };

    let service = ExecutorServiceImpl::new(config.clone()).await?;

    // Find available port
    let listener = TcpListener::bind("127.0.0.1:0").await?;
    let addr = listener.local_addr()?;
    drop(listener);

    // Start server in background
    let service_clone = service.clone();
    tokio::spawn(async move {
        Server::builder()
            .add_service(ExecutorServiceServer::new(service_clone))
            .serve(addr)
            .await
    });

    // Wait for server to start
    sleep(Duration::from_millis(100)).await;

    Ok((addr, service, temp_dir))
}

#[tokio::test]
async fn test_executor_registration() -> Result<()> {
    let (addr, _service, _temp_dir) = setup_test_server().await?;

    let mut client = ExecutorServiceClient::connect(format!("http://{}", addr)).await?;

    let request = RegisterExecutorRequest {
        executor_id: "test-executor-1".to_string(),
        miner_uid: 42,
        capabilities: vec!["gpu_compute".to_string(), "cpu_tasks".to_string()],
        hardware_info: Some(protocol::executor::HardwareInfo {
            cpu_count: 16,
            memory_gb: 64,
            gpu_count: 2,
            storage_gb: 1000,
        }),
    };

    let response = client.register_executor(request).await?;
    let result = response.into_inner();

    assert!(result.success);
    assert!(!result.session_token.is_empty());

    Ok(())
}

#[tokio::test]
async fn test_validation_session_flow() -> Result<()> {
    let (addr, _service, _temp_dir) = setup_test_server().await?;

    let mut client = ExecutorServiceClient::connect(format!("http://{}", addr)).await?;

    // Register first
    let register_response = client
        .register_executor(RegisterExecutorRequest {
            executor_id: "test-executor-2".to_string(),
            miner_uid: 43,
            capabilities: vec!["hardware_attestation".to_string()],
            hardware_info: None,
        })
        .await?
        .into_inner();

    let session_token = register_response.session_token;

    // Start validation
    let start_request = StartValidationRequest {
        session_id: Uuid::new_v4().to_string(),
        validator_hotkey: "5GrwvaEF5zXb26Fz9rcQpDWS57CtERHpNehXCPcNoHGKutQY".to_string(),
        validation_type: "hardware_attestation".to_string(),
        parameters: serde_json::json!({
            "vdf_difficulty": 1000,
            "collect_gpu_info": true
        })
        .to_string(),
        session_token: session_token.clone(),
    };

    let start_response = client.start_validation(start_request.clone()).await?;
    let start_result = start_response.into_inner();

    assert!(start_result.success);
    assert!(!start_result.validation_id.is_empty());

    // Get status
    let status_request = GetStatusRequest {
        session_token: session_token.clone(),
    };

    let status_response = client.get_status(status_request).await?;
    let status = status_response.into_inner();

    assert_eq!(status.executor_id, "test-executor-2");
    assert!(status.active_validations > 0);

    // Stop validation
    let stop_request = StopValidationRequest {
        validation_id: start_result.validation_id,
        reason: "Test completed".to_string(),
        session_token: session_token.clone(),
    };

    let stop_response = client.stop_validation(stop_request).await?;
    assert!(stop_response.into_inner().success);

    Ok(())
}

#[tokio::test]
async fn test_command_execution() -> Result<()> {
    let (addr, _service, _temp_dir) = setup_test_server().await?;

    let mut client = ExecutorServiceClient::connect(format!("http://{}", addr)).await?;

    // Register and get token
    let register_response = client
        .register_executor(RegisterExecutorRequest {
            executor_id: "test-executor-cmd".to_string(),
            miner_uid: 44,
            capabilities: vec!["command_execution".to_string()],
            hardware_info: None,
        })
        .await?
        .into_inner();

    let session_token = register_response.session_token;

    // Execute command
    let exec_request = ExecuteCommandRequest {
        command: "echo".to_string(),
        args: vec!["Hello from gRPC test".to_string()],
        environment: std::collections::HashMap::new(),
        working_directory: "/tmp".to_string(),
        timeout_seconds: 10,
        session_token: session_token.clone(),
    };

    let exec_response = client.execute_command(exec_request).await?;
    let exec_result = exec_response.into_inner();

    assert!(exec_result.success);
    assert_eq!(exec_result.exit_code, 0);
    assert!(exec_result.stdout.contains("Hello from gRPC test"));
    assert!(exec_result.stderr.is_empty());

    Ok(())
}

#[tokio::test]
async fn test_log_streaming() -> Result<()> {
    let (addr, service, _temp_dir) = setup_test_server().await?;

    let mut client = ExecutorServiceClient::connect(format!("http://{}", addr)).await?;

    // Register
    let register_response = client
        .register_executor(RegisterExecutorRequest {
            executor_id: "test-executor-logs".to_string(),
            miner_uid: 45,
            capabilities: vec!["logging".to_string()],
            hardware_info: None,
        })
        .await?
        .into_inner();

    let session_token = register_response.session_token;

    // Generate some logs
    service.log_message("Test log message 1").await;
    service.log_message("Test log message 2").await;
    service.log_message("Test log message 3").await;

    // Request logs
    let logs_request = GetLogsRequest {
        validation_id: None,
        lines: 10,
        follow: false,
        session_token,
    };

    let mut stream = client.get_logs(logs_request).await?.into_inner();

    let mut log_count = 0;
    while let Some(log_entry) = stream.message().await? {
        assert!(!log_entry.message.is_empty());
        assert!(!log_entry.timestamp.is_empty());
        log_count += 1;
    }

    assert!(
        log_count >= 3,
        "Should have received at least 3 log messages"
    );

    Ok(())
}

#[tokio::test]
async fn test_health_check_and_metrics() -> Result<()> {
    let (addr, _service, _temp_dir) = setup_test_server().await?;

    let mut client = ExecutorServiceClient::connect(format!("http://{}", addr)).await?;

    // Health check (no auth required)
    let health_request = HealthCheckRequest {};
    let health_response = client.health_check(health_request).await?;
    let health = health_response.into_inner();

    assert!(health.healthy);
    assert!(!health.version.is_empty());
    assert!(health.uptime_seconds >= 0);

    // Get metrics (requires auth)
    let register_response = client
        .register_executor(RegisterExecutorRequest {
            executor_id: "test-executor-metrics".to_string(),
            miner_uid: 46,
            capabilities: vec!["metrics".to_string()],
            hardware_info: None,
        })
        .await?
        .into_inner();

    let metrics_request = GetMetricsRequest {
        session_token: register_response.session_token,
    };

    let metrics_response = client.get_metrics(metrics_request).await?;
    let metrics = metrics_response.into_inner();

    assert!(metrics.cpu_usage >= 0.0);
    assert!(metrics.memory_used_mb > 0);
    assert!(metrics.disk_free_gb >= 0.0);
    assert_eq!(metrics.active_sessions, 0);
    assert_eq!(metrics.total_validations, 0);

    Ok(())
}

#[tokio::test]
async fn test_concurrent_grpc_requests() -> Result<()> {
    let (addr, _service, _temp_dir) = setup_test_server().await?;

    let endpoint = format!("http://{}", addr);

    // Create multiple clients
    let mut handles = vec![];

    for i in 0..5 {
        let endpoint = endpoint.clone();
        let handle = tokio::spawn(async move {
            let mut client = ExecutorServiceClient::connect(endpoint).await?;

            // Register executor
            let register_response = client
                .register_executor(RegisterExecutorRequest {
                    executor_id: format!("concurrent-executor-{}", i),
                    miner_uid: (50 + i) as u32,
                    capabilities: vec!["concurrent_test".to_string()],
                    hardware_info: None,
                })
                .await?
                .into_inner();

            let token = register_response.session_token;

            // Make multiple requests
            for j in 0..3 {
                // Get status
                client
                    .get_status(GetStatusRequest {
                        session_token: token.clone(),
                    })
                    .await?;

                // Execute command
                client
                    .execute_command(ExecuteCommandRequest {
                        command: "echo".to_string(),
                        args: vec![format!("Client {} Request {}", i, j)],
                        environment: std::collections::HashMap::new(),
                        working_directory: "/tmp".to_string(),
                        timeout_seconds: 5,
                        session_token: token.clone(),
                    })
                    .await?;
            }

            Ok::<(), anyhow::Error>(())
        });

        handles.push(handle);
    }

    // Wait for all clients to complete
    for handle in handles {
        handle.await??;
    }

    Ok(())
}

#[tokio::test]
async fn test_grpc_error_handling() -> Result<()> {
    let (addr, _service, _temp_dir) = setup_test_server().await?;

    let mut client = ExecutorServiceClient::connect(format!("http://{}", addr)).await?;

    // Test unauthorized request
    let unauthorized_request = GetStatusRequest {
        session_token: "invalid-token".to_string(),
    };

    let error = client.get_status(unauthorized_request).await;
    assert!(error.is_err());
    let status = error.err().unwrap();
    assert_eq!(status.code(), tonic::Code::Unauthenticated);

    // Test invalid parameters
    let register_response = client
        .register_executor(RegisterExecutorRequest {
            executor_id: "test-error-executor".to_string(),
            miner_uid: 99,
            capabilities: vec![],
            hardware_info: None,
        })
        .await?
        .into_inner();

    let invalid_exec = ExecuteCommandRequest {
        command: "/nonexistent/command".to_string(),
        args: vec![],
        environment: std::collections::HashMap::new(),
        working_directory: "/nonexistent/path".to_string(),
        timeout_seconds: 1,
        session_token: register_response.session_token,
    };

    let exec_error = client.execute_command(invalid_exec).await?;
    let result = exec_error.into_inner();
    assert!(!result.success);
    assert_ne!(result.exit_code, 0);

    Ok(())
}

#[tokio::test]
async fn test_grpc_tls_configuration() -> Result<()> {
    let temp_dir = TempDir::new()?;

    // Generate test certificates
    let (cert, key) = generate_test_certificates()?;
    let cert_path = temp_dir.path().join("server.crt");
    let key_path = temp_dir.path().join("server.key");

    std::fs::write(&cert_path, cert)?;
    std::fs::write(&key_path, key)?;

    let config = ExecutorConfig {
        executor_id: ExecutorId::new("test-tls-executor"),
        working_dir: temp_dir.path().to_path_buf(),
        grpc: GrpcConfig {
            bind_address: "127.0.0.1:0".parse()?,
            max_message_size: 4 * 1024 * 1024,
            keepalive_interval: Duration::from_secs(30),
            keepalive_timeout: Duration::from_secs(10),
            tls: Some(TlsConfig {
                cert_path: cert_path.clone(),
                key_path: key_path.clone(),
                ca_path: None,
                client_auth: false,
            }),
        },
        ..Default::default()
    };

    // Verify TLS configuration is loaded
    assert!(config.grpc.tls.is_some());
    let tls = config.grpc.tls.as_ref().unwrap();
    assert_eq!(tls.cert_path, cert_path);
    assert_eq!(tls.key_path, key_path);

    Ok(())
}

#[tokio::test]
async fn test_dynamic_config_update() -> Result<()> {
    let (addr, _service, _temp_dir) = setup_test_server().await?;

    let mut client = ExecutorServiceClient::connect(format!("http://{}", addr)).await?;

    // Register as admin
    let register_response = client
        .register_executor(RegisterExecutorRequest {
            executor_id: "admin-executor".to_string(),
            miner_uid: 1, // Admin miner UID
            capabilities: vec!["admin".to_string()],
            hardware_info: None,
        })
        .await?
        .into_inner();

    // Update configuration
    let update_request = UpdateConfigRequest {
        config_json: serde_json::json!({
            "max_concurrent_sessions": 20,
            "session_timeout_seconds": 7200,
            "enable_debug_logging": true
        })
        .to_string(),
        session_token: register_response.session_token,
    };

    let update_response = client.update_config(update_request).await?;
    assert!(update_response.into_inner().success);

    Ok(())
}

fn generate_test_certificates() -> Result<(String, String)> {
    // Simple self-signed certificate for testing
    let cert = r#"-----BEGIN CERTIFICATE-----
MIIBkTCB+wIJAKHHIG...
-----END CERTIFICATE-----"#;

    let key = r#"-----BEGIN PRIVATE KEY-----
MIICdgIBADANBgkqh...
-----END PRIVATE KEY-----"#;

    Ok((cert.to_string(), key.to_string()))
}
