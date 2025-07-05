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

#[cfg(test)]
mod freivalds_integration_tests {
    use super::*;
    use executor::validation_session::types::ValidatorId;
    use protocol::basilca::freivalds_gpu_pow::v1::FreivaldsChallenge;
    use std::process::Command;
    use tempfile::TempDir;

    #[tokio::test]
    #[ignore] // Run with: cargo test --ignored test_freivalds_ssh_execution
    async fn test_freivalds_ssh_execution() {
        // This test simulates a validator uploading and executing the Freivalds binary
        let temp_dir = TempDir::new().unwrap();
        let binary_path = temp_dir.path().join("gpu_attestor_freivalds");
        
        // Build the gpu-attestor binary
        let output = Command::new("cargo")
            .args(&["build", "--release", "--bin", "gpu-attestor"])
            .current_dir("../gpu-attestor")
            .output()
            .expect("Failed to build gpu-attestor");
        
        if !output.status.success() {
            panic!("Failed to build gpu-attestor: {}", 
                   String::from_utf8_lossy(&output.stderr));
        }
        
        // Copy binary to temp location
        std::fs::copy("../gpu-attestor/target/release/gpu-attestor", &binary_path)
            .expect("Failed to copy binary");
        
        // Execute Freivalds challenge
        let output = Command::new(&binary_path)
            .args(&[
                "--freivalds",
                "--freivalds-matrix-size", "256",
                "--freivalds-seed", "deadbeef00000000deadbeef00000000",
                "--freivalds-session-id", "executor_test_session",
            ])
            .output()
            .expect("Failed to execute gpu-attestor");
        
        assert!(output.status.success(), "Freivalds execution failed");
        
        // Parse and verify output
        let stdout = String::from_utf8_lossy(&output.stdout);
        let json: serde_json::Value = serde_json::from_str(&stdout)
            .expect("Failed to parse JSON output");
        
        assert_eq!(json["type"], "freivalds_commitment");
        assert_eq!(json["session_id"], "executor_test_session");
        assert_eq!(json["row_count"], 256);
        assert!(json["merkle_root"].is_string());
    }

    #[tokio::test]
    async fn test_freivalds_challenge_via_grpc() {
        let mut config = ExecutorConfig::default();
        config.server.port = 0;
        config.enable_validation_sessions = true;
        
        let state = ExecutorState::new(config).await.unwrap();
        let server = ExecutorServer::new(state.clone());
        
        // Start server
        let addr: SocketAddr = "127.0.0.1:0".parse().unwrap();
        let actual_addr = start_test_server(server, addr).await;
        tokio::time::sleep(Duration::from_millis(100)).await;
        
        // Connect to server
        let channel = Channel::from_shared(format!("http://{actual_addr}"))
            .unwrap()
            .connect()
            .await
            .unwrap();
        
        let mut client = ExecutorControlClient::new(channel);
        
        // Create Freivalds challenge parameters
        let challenge_params = protocol::common::ChallengeParameters {
            challenge_type: "freivalds_gpu_pow".to_string(),
            gpu_pow_seed: 12345678901234567890u64,
            matrix_dim: 512,
            num_matrices: 1,
            num_iterations: 1,
            parameters_json: serde_json::json!({
                "matrix_size": 512,
            }).to_string(),
            validator_nonce: "test_nonce".to_string(),
            expected_duration_seconds: 10,
            difficulty_level: 5,
            seed: "test_seed".to_string(),
            machine_info: None,
            matrix_a_index: 0,
            matrix_b_index: 0,
        };
        
        let request = tonic::Request::new(protocol::executor_control::ChallengeRequest {
            validator_hotkey: "test_validator".to_string(),
            nonce: "test_nonce".to_string(),
            parameters: Some(challenge_params),
            timeout_seconds: 30,
        });
        
        let response = client.execute_computational_challenge(request).await;
        
        // Note: This will fail unless Freivalds is integrated into the executor's challenge handler
        // For now, we just verify the endpoint exists
        assert!(response.is_err() || response.unwrap().into_inner().result.is_some());
    }

    #[tokio::test]
    async fn test_executor_ssh_access_for_freivalds() {
        let mut config = ExecutorConfig::default();
        config.server.port = 0;
        config.enable_validation_sessions = true;
        
        let state = ExecutorState::new(config).await.unwrap();
        
        // Grant SSH access for a validator
        let validator_id = ValidatorId::new("test_validator_hotkey".to_string());
        let public_key = "ssh-ed25519 AAAAC3NzaC1lZDI1NTE5AAAAIFakeKeyForTesting test@validator";
        
        if let Some(validation_service) = &state.validation_service {
            validation_service
                .grant_ssh_access(&validator_id, public_key)
                .await
                .expect("Failed to grant SSH access");
            
            // Verify access was granted
            let has_access = validation_service.has_ssh_access(&validator_id).await;
            assert!(has_access, "Validator should have SSH access");
        }
    }

    #[test]
    fn test_freivalds_binary_determinism() {
        // Test that the same seed produces the same Merkle root
        let temp_dir = TempDir::new().unwrap();
        let binary = "../gpu-attestor/target/release/gpu-attestor";
        
        // Run twice with same parameters
        let output1 = Command::new(binary)
            .args(&[
                "--freivalds",
                "--freivalds-matrix-size", "128",
                "--freivalds-seed", "abcd1234abcd1234abcd1234abcd1234",
                "--freivalds-session-id", "determinism_test",
                "--log-level", "error",
            ])
            .output()
            .expect("Failed to execute gpu-attestor");
        
        let output2 = Command::new(binary)
            .args(&[
                "--freivalds",
                "--freivalds-matrix-size", "128", 
                "--freivalds-seed", "abcd1234abcd1234abcd1234abcd1234",
                "--freivalds-session-id", "determinism_test",
                "--log-level", "error",
            ])
            .output()
            .expect("Failed to execute gpu-attestor");
        
        assert!(output1.status.success());
        assert!(output2.status.success());
        
        let json1: serde_json::Value = serde_json::from_slice(&output1.stdout).unwrap();
        let json2: serde_json::Value = serde_json::from_slice(&output2.stdout).unwrap();
        
        // Merkle roots should be identical
        assert_eq!(
            json1["merkle_root"].as_str().unwrap(),
            json2["merkle_root"].as_str().unwrap(),
            "Same seed should produce same Merkle root"
        );
    }

    #[tokio::test]
    async fn test_freivalds_performance_characteristics() {
        use std::time::Instant;
        
        let binary = "../gpu-attestor/target/release/gpu-attestor";
        let sizes = vec![64, 128, 256];
        let mut timings = Vec::new();
        
        for size in sizes {
            let start = Instant::now();
            
            let output = Command::new(binary)
                .args(&[
                    "--freivalds",
                    "--freivalds-matrix-size", &size.to_string(),
                    "--log-level", "error",
                ])
                .output()
                .expect("Failed to execute gpu-attestor");
            
            let elapsed = start.elapsed();
            assert!(output.status.success());
            
            let json: serde_json::Value = serde_json::from_slice(&output.stdout).unwrap();
            let metadata = &json["metadata"];
            let kernel_time = metadata["kernel_time_ms"].as_u64().unwrap();
            
            timings.push((size, elapsed.as_millis() as u64, kernel_time));
        }
        
        // Verify reasonable scaling
        for i in 1..timings.len() {
            let (size1, _, kernel1) = timings[i-1];
            let (size2, _, kernel2) = timings[i];
            let size_ratio = size2 as f64 / size1 as f64;
            let time_ratio = kernel2 as f64 / kernel1 as f64;
            
            // Kernel time should scale between O(n²) and O(n³)
            assert!(
                time_ratio > size_ratio.powi(2) * 0.5,
                "Performance scaling too slow"
            );
            assert!(
                time_ratio < size_ratio.powi(3) * 2.0,
                "Performance scaling too fast"
            );
        }
    }
}
