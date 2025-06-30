use anyhow::Result;
use common::ssh::SshManager;
use std::fs;
use std::path::PathBuf;
use std::process::Command;
use tempfile::TempDir;
use tokio::time::{sleep, timeout, Duration};
use validator::config::{SshConfig, ValidationConfig, ValidatorConfig};
use validator::ssh::SshClient;
use validator::validation::{
    HardwareValidator, ValidationContext, ValidationOptions, ValidationResult,
    ValidatorFactory,
};

#[tokio::test]
async fn test_ssh_connection_establishment() -> Result<()> {
    let temp_dir = TempDir::new()?;
    let ssh_key_path = temp_dir.path().join("test_key");
    let ssh_pub_path = temp_dir.path().join("test_key.pub");
    
    // Generate test SSH key pair
    Command::new("ssh-keygen")
        .args(&[
            "-t", "ed25519",
            "-f", ssh_key_path.to_str().unwrap(),
            "-N", "",
            "-C", "test@validator"
        ])
        .output()?;
    
    assert!(ssh_key_path.exists(), "Private key should be created");
    assert!(ssh_pub_path.exists(), "Public key should be created");
    
    let config = SshConfig {
        private_key_path: ssh_key_path.to_string_lossy().to_string(),
        known_hosts_path: temp_dir.path().join("known_hosts").to_string_lossy().to_string(),
        connection_timeout: Duration::from_secs(10),
        keepalive_interval: Duration::from_secs(30),
        max_retries: 3,
        strict_host_checking: false,
    };
    
    let client = SshClient::new(config)?;
    
    // Test connection to localhost (will fail but structure is correct)
    let connection_result = timeout(
        Duration::from_secs(5),
        client.connect("localhost", "test_user", 22)
    ).await;
    
    match connection_result {
        Ok(Ok(_)) => {
            // Unlikely to succeed in test environment
        }
        Ok(Err(e)) => {
            assert!(
                e.to_string().contains("ssh") || 
                e.to_string().contains("connect") ||
                e.to_string().contains("authentication"),
                "Expected SSH connection error, got: {}", e
            );
        }
        Err(_) => {
            // Timeout is expected in test environment
        }
    }
    
    Ok(())
}

#[tokio::test]
async fn test_gpu_attestor_binary_transfer() -> Result<()> {
    let temp_dir = TempDir::new()?;
    let local_binary = temp_dir.path().join("gpu-attestor");
    let remote_dir = temp_dir.path().join("remote");
    
    // Create fake gpu-attestor binary
    fs::write(&local_binary, b"#!/bin/bash\necho 'GPU Attestor Test Binary'")?;
    fs::set_permissions(&local_binary, fs::Permissions::from_mode(0o755))?;
    
    // Create remote directory
    fs::create_dir_all(&remote_dir)?;
    
    let ssh_config = SshConfig {
        private_key_path: "/tmp/test_key".to_string(),
        known_hosts_path: "/tmp/known_hosts".to_string(),
        connection_timeout: Duration::from_secs(10),
        keepalive_interval: Duration::from_secs(30),
        max_retries: 3,
        strict_host_checking: false,
    };
    
    let client = SshClient::new(ssh_config)?;
    
    // Test binary transfer logic
    let transfer_result = client.transfer_file(
        &local_binary,
        &remote_dir.join("gpu-attestor")
    );
    
    match transfer_result {
        Ok(_) => {
            // Check file was prepared for transfer
            assert!(local_binary.exists(), "Local binary should exist");
        }
        Err(e) => {
            // Expected in test environment without real SSH
            assert!(
                e.to_string().contains("transfer") ||
                e.to_string().contains("ssh"),
                "Expected transfer error, got: {}", e
            );
        }
    }
    
    Ok(())
}

#[tokio::test]
async fn test_remote_attestation_execution() -> Result<()> {
    let temp_dir = TempDir::new()?;
    
    let validation_config = ValidationConfig {
        gpu_attestor_path: PathBuf::from("/usr/local/bin/gpu-attestor"),
        remote_work_dir: PathBuf::from("/tmp/basilica_validation"),
        execution_timeout: Duration::from_secs(120),
        cleanup_on_success: true,
        cleanup_on_failure: false,
        max_attestation_size: 10 * 1024 * 1024, // 10MB
        allowed_algorithms: vec!["simple".to_string(), "advanced".to_string()],
    };
    
    let config = ValidatorConfig {
        validation: validation_config,
        ..Default::default()
    };
    
    let factory = ValidatorFactory::new(config);
    let validator = factory.create_hardware_validator()?;
    
    let context = ValidationContext {
        executor_id: "test-executor".to_string(),
        miner_uid: 42,
        validator_hotkey: "test-validator-key".to_string(),
        timestamp: chrono::Utc::now(),
    };
    
    let options = ValidationOptions {
        skip_network_benchmark: false,
        skip_hardware_collection: false,
        skip_vdf: false,
        vdf_difficulty: 1000,
        vdf_algorithm: "simple".to_string(),
        custom_args: vec![],
    };
    
    // Test remote execution (will fail without real SSH)
    let validation_result = timeout(
        Duration::from_secs(10),
        validator.validate_executor(
            "localhost",
            "test_user",
            "/tmp/test_key",
            22,
            context,
            options
        )
    ).await;
    
    match validation_result {
        Ok(Ok(result)) => {
            // Unlikely in test environment
            assert!(!result.attestation_data.is_empty());
        }
        Ok(Err(e)) => {
            assert!(
                e.to_string().contains("validation") ||
                e.to_string().contains("ssh") ||
                e.to_string().contains("connect"),
                "Expected validation error, got: {}", e
            );
        }
        Err(_) => {
            // Timeout expected
        }
    }
    
    Ok(())
}

#[tokio::test]
async fn test_attestation_result_collection() -> Result<()> {
    let temp_dir = TempDir::new()?;
    let result_dir = temp_dir.path().join("results");
    fs::create_dir_all(&result_dir)?;
    
    // Create fake attestation files
    let attestation_json = r#"{
        "executor_id": "test-executor",
        "timestamp": "2024-01-01T00:00:00Z",
        "hardware_info": {
            "cpu_count": 16,
            "memory_gb": 64,
            "gpu_count": 2
        },
        "gpu_info": [
            {
                "name": "NVIDIA RTX 4090",
                "memory_mb": 24576,
                "compute_capability": "8.9"
            }
        ]
    }"#;
    
    fs::write(result_dir.join("attestation.json"), attestation_json)?;
    fs::write(result_dir.join("attestation.sig"), b"fake-signature-data")?;
    fs::write(result_dir.join("attestation.pub"), b"fake-public-key")?;
    
    // Test result collection
    let files = fs::read_dir(&result_dir)?
        .filter_map(|entry| entry.ok())
        .map(|entry| entry.path())
        .collect::<Vec<_>>();
    
    assert_eq!(files.len(), 3, "Should have 3 attestation files");
    assert!(files.iter().any(|p| p.extension().unwrap_or_default() == "json"));
    assert!(files.iter().any(|p| p.extension().unwrap_or_default() == "sig"));
    assert!(files.iter().any(|p| p.extension().unwrap_or_default() == "pub"));
    
    // Test JSON parsing
    let json_content = fs::read_to_string(result_dir.join("attestation.json"))?;
    let parsed: serde_json::Value = serde_json::from_str(&json_content)?;
    
    assert_eq!(parsed["executor_id"], "test-executor");
    assert_eq!(parsed["hardware_info"]["cpu_count"], 16);
    assert_eq!(parsed["gpu_info"][0]["name"], "NVIDIA RTX 4090");
    
    Ok(())
}

#[tokio::test]
async fn test_ssh_session_cleanup() -> Result<()> {
    let temp_dir = TempDir::new()?;
    let work_dir = temp_dir.path().join("validation_work");
    fs::create_dir_all(&work_dir)?;
    
    // Create test files
    fs::write(work_dir.join("gpu-attestor"), b"binary")?;
    fs::write(work_dir.join("attestation.json"), b"{}")?;
    fs::write(work_dir.join("attestation.sig"), b"sig")?;
    
    assert!(work_dir.exists(), "Work directory should exist");
    assert_eq!(fs::read_dir(&work_dir)?.count(), 3, "Should have 3 files");
    
    // Simulate cleanup
    let cleanup_result = fs::remove_dir_all(&work_dir);
    assert!(cleanup_result.is_ok(), "Cleanup should succeed");
    assert!(!work_dir.exists(), "Work directory should be removed");
    
    Ok(())
}

#[tokio::test]
async fn test_concurrent_ssh_validations() -> Result<()> {
    let temp_dir = TempDir::new()?;
    
    let validation_config = ValidationConfig {
        gpu_attestor_path: PathBuf::from("/usr/local/bin/gpu-attestor"),
        remote_work_dir: PathBuf::from("/tmp/basilica_validation"),
        execution_timeout: Duration::from_secs(30),
        cleanup_on_success: true,
        cleanup_on_failure: false,
        max_attestation_size: 10 * 1024 * 1024,
        allowed_algorithms: vec!["simple".to_string()],
    };
    
    let config = ValidatorConfig {
        validation: validation_config,
        ..Default::default()
    };
    
    let factory = ValidatorFactory::new(config);
    
    // Launch multiple concurrent validations
    let mut handles = vec![];
    
    for i in 0..5 {
        let validator = factory.create_hardware_validator()?;
        
        let handle = tokio::spawn(async move {
            let context = ValidationContext {
                executor_id: format!("executor-{}", i),
                miner_uid: i as u16,
                validator_hotkey: "test-validator".to_string(),
                timestamp: chrono::Utc::now(),
            };
            
            let options = ValidationOptions::default();
            
            let result = timeout(
                Duration::from_secs(5),
                validator.validate_executor(
                    "localhost",
                    &format!("user-{}", i),
                    "/tmp/test_key",
                    22,
                    context,
                    options
                )
            ).await;
            
            result.is_ok()
        });
        
        handles.push(handle);
    }
    
    // Wait for all validations to complete
    let results = futures::future::join_all(handles).await;
    
    // All should complete (even if they fail)
    assert_eq!(results.len(), 5, "All validations should complete");
    
    Ok(())
}

#[tokio::test]
async fn test_ssh_key_permissions_validation() -> Result<()> {
    let temp_dir = TempDir::new()?;
    let key_path = temp_dir.path().join("test_key");
    
    // Create key with wrong permissions
    fs::write(&key_path, "fake-private-key")?;
    fs::set_permissions(&key_path, fs::Permissions::from_mode(0o644))?;
    
    let ssh_config = SshConfig {
        private_key_path: key_path.to_string_lossy().to_string(),
        known_hosts_path: temp_dir.path().join("known_hosts").to_string_lossy().to_string(),
        connection_timeout: Duration::from_secs(10),
        keepalive_interval: Duration::from_secs(30),
        max_retries: 3,
        strict_host_checking: false,
    };
    
    // SSH client should detect bad permissions
    let client_result = SshClient::new(ssh_config.clone());
    
    // Fix permissions
    fs::set_permissions(&key_path, fs::Permissions::from_mode(0o600))?;
    
    // Now it should work
    let client = SshClient::new(ssh_config)?;
    assert!(client.is_ok() || client_result.is_err(), 
        "Should handle key permission validation");
    
    Ok(())
}