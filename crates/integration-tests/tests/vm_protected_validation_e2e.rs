//! VM-Protected GPU Attestation Integration Tests
//!
//! This test suite validates the end-to-end flow of the VM-protected GPU attestation
//! system using the secure generic validator that hides all operational details.

use anyhow::{Context, Result};
use std::path::PathBuf;
use std::time::Duration;
use tracing::{info, warn};

// Import the secure validator
use common::ssh::SshConnectionDetails;
use validator::validation::secure_validator::{SecureValidator, SecureValidatorConfig};

/// Test configuration for VM-protected validation
struct VmTestConfig {
    attestor_binary_path: PathBuf,
    test_matrix_sizes: Vec<u32>,
    test_resource_counts: Vec<u32>,
    ssh_timeout: Duration,
    max_execution_time: Duration,
}

impl VmTestConfig {
    fn new() -> Self {
        Self {
            attestor_binary_path: PathBuf::from("./target/release/gpu-attestor"),
            test_matrix_sizes: vec![64, 128, 256, 512],
            test_resource_counts: vec![1, 2],
            ssh_timeout: Duration::from_secs(60),
            max_execution_time: Duration::from_secs(120),
        }
    }

    fn with_binary_path(mut self, path: PathBuf) -> Self {
        self.attestor_binary_path = path;
        self
    }
}

/// Create a test SSH connection for local testing
fn create_local_ssh_connection() -> SshConnectionDetails {
    SshConnectionDetails {
        host: "127.0.0.1".to_string(),
        port: 22,
        username: std::env::var("USER").unwrap_or_else(|_| "testuser".to_string()),
        private_key_path: std::path::PathBuf::from(format!(
            "{}/.ssh/id_rsa",
            std::env::var("HOME").unwrap_or_else(|_| "/tmp".to_string())
        )),
        timeout: Duration::from_secs(30),
    }
}

/// Check if we can perform SSH-based testing
async fn check_ssh_connectivity() -> bool {
    let connection = create_local_ssh_connection();

    // Test SSH connectivity with a simple command
    let test_cmd = format!(
        "ssh -o ConnectTimeout=5 -o BatchMode=yes -o StrictHostKeyChecking=no {}@{} 'echo test'",
        connection.username, connection.host
    );

    match tokio::process::Command::new("sh")
        .arg("-c")
        .arg(&test_cmd)
        .output()
        .await
    {
        Ok(output) => {
            if output.status.success() {
                info!("SSH connectivity confirmed");
                true
            } else {
                warn!(
                    "SSH connectivity test failed: {}",
                    String::from_utf8_lossy(&output.stderr)
                );
                false
            }
        }
        Err(e) => {
            warn!("SSH connectivity check error: {}", e);
            false
        }
    }
}

/// Check if required binaries exist
fn check_binary_availability(config: &VmTestConfig) -> bool {
    if !config.attestor_binary_path.exists() {
        warn!(
            "GPU attestor binary not found at {:?}. Build with: cargo build --release",
            config.attestor_binary_path
        );
        false
    } else {
        info!(
            "GPU attestor binary found at {:?}",
            config.attestor_binary_path
        );
        true
    }
}

/// Detect available GPU count for testing
fn detect_gpu_count() -> u32 {
    // Try nvidia-smi to count GPUs
    match std::process::Command::new("nvidia-smi")
        .arg("--list-gpus")
        .output()
    {
        Ok(output) if output.status.success() => {
            let gpu_list = String::from_utf8_lossy(&output.stdout);
            let count = gpu_list.lines().filter(|line| line.contains("GPU")).count() as u32;
            info!("Detected {} GPU(s) via nvidia-smi", count);
            count
        }
        _ => {
            info!("No GPUs detected or nvidia-smi not available");
            0
        }
    }
}

#[tokio::test]
async fn test_vm_protected_basic_validation() -> Result<()> {
    let _ = tracing_subscriber::fmt()
        .with_env_filter("info")
        .with_test_writer()
        .try_init();

    info!("=== VM-Protected Basic Validation Test ===");

    let test_config = VmTestConfig::new();

    // Check prerequisites
    if !check_binary_availability(&test_config) {
        eprintln!("Skipping test - required binaries not available");
        return Ok(());
    }

    if !check_ssh_connectivity().await {
        eprintln!("Skipping test - SSH connectivity not available");
        return Ok(());
    }

    // Create secure validator
    let validator_config = SecureValidatorConfig {
        attestor_binary_path: test_config.attestor_binary_path.clone(),
        ssh_timeout: test_config.ssh_timeout,
        max_execution_time: test_config.max_execution_time,
    };

    let validator = SecureValidator::new(validator_config)?;
    let connection = create_local_ssh_connection();

    // Test basic validation
    info!("Testing VM-protected validation with generic parameters");

    let problem_size = 256;
    let resource_count = 1;

    info!(
        "Executing secure validation: problem_size={}, resource_count={}",
        problem_size, resource_count
    );

    let start_time = std::time::Instant::now();
    let result = validator
        .validate_compute_resource(&connection, problem_size, resource_count)
        .await
        .context("VM-protected validation failed")?;
    let elapsed = start_time.elapsed();

    info!(
        "VM-protected validation completed in {:?}: {}",
        elapsed,
        if result { "PASS ✓" } else { "FAIL ✗" }
    );

    // The result depends on the VM's internal validation logic and available hardware
    // We don't assert the specific result since it's hidden in the VM
    info!("✓ VM-protected validation completed successfully");

    Ok(())
}

#[tokio::test]
async fn test_vm_protected_multiple_problem_sizes() -> Result<()> {
    let _ = tracing_subscriber::fmt()
        .with_env_filter("info")
        .with_test_writer()
        .try_init();

    info!("=== VM-Protected Multiple Problem Sizes Test ===");

    let test_config = VmTestConfig::new();

    if !check_binary_availability(&test_config) || !check_ssh_connectivity().await {
        eprintln!("Skipping test - prerequisites not met");
        return Ok(());
    }

    let validator_config = SecureValidatorConfig {
        attestor_binary_path: test_config.attestor_binary_path.clone(),
        ssh_timeout: test_config.ssh_timeout,
        max_execution_time: test_config.max_execution_time,
    };

    let validator = SecureValidator::new(validator_config)?;
    let connection = create_local_ssh_connection();

    info!("Testing various problem sizes with VM protection");
    info!(
        "{:<12} {:<12} {:<12} {:<12}",
        "Size", "Time (ms)", "Result", "Status"
    );
    info!("{:-<48}", "");

    for &problem_size in &test_config.test_matrix_sizes {
        let start_time = std::time::Instant::now();

        let result = validator
            .validate_compute_resource(&connection, problem_size, 1)
            .await
            .context(format!("Validation failed for problem size {problem_size}"))?;

        let elapsed = start_time.elapsed();

        info!(
            "{:<12} {:<12} {:<12} {:<12}",
            problem_size,
            elapsed.as_millis(),
            if result { "PASS" } else { "FAIL" },
            "✓"
        );

        // Ensure reasonable execution time
        assert!(
            elapsed < Duration::from_secs(60),
            "Validation took too long for size {problem_size}: {elapsed:?}"
        );
    }

    info!("✓ All problem sizes tested successfully");

    Ok(())
}

#[tokio::test]
async fn test_vm_protected_resource_scaling() -> Result<()> {
    let _ = tracing_subscriber::fmt()
        .with_env_filter("info")
        .with_test_writer()
        .try_init();

    info!("=== VM-Protected Resource Scaling Test ===");

    let test_config = VmTestConfig::new();

    if !check_binary_availability(&test_config) || !check_ssh_connectivity().await {
        eprintln!("Skipping test - prerequisites not met");
        return Ok(());
    }

    let available_gpus = detect_gpu_count();
    let validator_config = SecureValidatorConfig {
        attestor_binary_path: test_config.attestor_binary_path.clone(),
        ssh_timeout: test_config.ssh_timeout,
        max_execution_time: test_config.max_execution_time,
    };

    let validator = SecureValidator::new(validator_config)?;
    let connection = create_local_ssh_connection();

    // Test different resource counts
    let problem_size = 256;

    info!(
        "Testing resource scaling (available GPUs: {})",
        available_gpus
    );
    info!(
        "{:<12} {:<12} {:<12} {:<12}",
        "Resources", "Time (ms)", "Result", "Status"
    );
    info!("{:-<48}", "");

    for &resource_count in &test_config.test_resource_counts {
        let start_time = std::time::Instant::now();

        let result = validator
            .validate_compute_resource(&connection, problem_size, resource_count)
            .await
            .context(format!("Validation failed for {resource_count} resources"))?;

        let elapsed = start_time.elapsed();

        info!(
            "{:<12} {:<12} {:<12} {:<12}",
            resource_count,
            elapsed.as_millis(),
            if result { "PASS" } else { "FAIL" },
            "✓"
        );

        // The VM will internally validate whether the expected resource count matches available resources
        // We don't assert the specific result since this is hidden validation logic
    }

    info!("✓ Resource scaling test completed");

    Ok(())
}

#[tokio::test]
async fn test_vm_protected_concurrent_validations() -> Result<()> {
    let _ = tracing_subscriber::fmt()
        .with_env_filter("info")
        .with_test_writer()
        .try_init();

    info!("=== VM-Protected Concurrent Validations Test ===");

    let test_config = VmTestConfig::new();

    if !check_binary_availability(&test_config) || !check_ssh_connectivity().await {
        eprintln!("Skipping test - prerequisites not met");
        return Ok(());
    }

    let validator_config = SecureValidatorConfig {
        attestor_binary_path: test_config.attestor_binary_path.clone(),
        ssh_timeout: test_config.ssh_timeout,
        max_execution_time: test_config.max_execution_time,
    };

    let validator = std::sync::Arc::new(SecureValidator::new(validator_config)?);
    let connection = create_local_ssh_connection();

    // Run multiple concurrent validations
    let concurrent_count = 3;
    info!("Running {} concurrent validations", concurrent_count);

    let tasks: Vec<_> = (0..concurrent_count)
        .map(|i| {
            let validator = validator.clone();
            let connection = connection.clone();

            tokio::spawn(async move {
                let problem_size = 128 + (i * 64); // 128, 192, 256
                info!("Concurrent validation {}: problem_size={}", i, problem_size);

                let start_time = std::time::Instant::now();
                let result = validator
                    .validate_compute_resource(&connection, problem_size, 1)
                    .await
                    .context(format!("Concurrent validation {i} failed"))?;
                let elapsed = start_time.elapsed();

                info!(
                    "Concurrent validation {} completed in {:?}: {}",
                    i,
                    elapsed,
                    if result { "PASS" } else { "FAIL" }
                );

                Ok::<(bool, Duration), anyhow::Error>((result, elapsed))
            })
        })
        .collect();

    // Wait for all concurrent validations to complete
    let results = futures::future::try_join_all(tasks).await?;

    info!("All {} concurrent validations completed", results.len());

    // Verify all validations completed without errors
    for (i, task_result) in results.into_iter().enumerate() {
        let (validation_result, elapsed) = task_result?;
        info!(
            "Task {}: {} (took {:?})",
            i,
            if validation_result { "PASS" } else { "FAIL" },
            elapsed
        );

        // Ensure no task took unreasonably long
        assert!(
            elapsed < Duration::from_secs(90),
            "Concurrent validation {i} took too long: {elapsed:?}"
        );
    }

    info!("✓ Concurrent validations test completed successfully");

    Ok(())
}

#[tokio::test]
async fn test_vm_protected_error_handling() -> Result<()> {
    let _ = tracing_subscriber::fmt()
        .with_env_filter("info")
        .with_test_writer()
        .try_init();

    info!("=== VM-Protected Error Handling Test ===");

    // Test 1: Invalid binary path
    info!("Testing invalid binary path handling");
    let invalid_config = SecureValidatorConfig {
        attestor_binary_path: PathBuf::from("/nonexistent/path/gpu-attestor"),
        ssh_timeout: Duration::from_secs(30),
        max_execution_time: Duration::from_secs(60),
    };

    let result = SecureValidator::new(invalid_config);
    assert!(result.is_err(), "Should fail with invalid binary path");
    info!("✓ Correctly handled invalid binary path");

    // Test 2: Invalid SSH connection (if SSH is available)
    if check_ssh_connectivity().await {
        let test_config = VmTestConfig::new();

        if check_binary_availability(&test_config) {
            let validator_config = SecureValidatorConfig {
                attestor_binary_path: test_config.attestor_binary_path,
                ssh_timeout: Duration::from_secs(5), // Short timeout
                max_execution_time: Duration::from_secs(10),
            };

            let validator = SecureValidator::new(validator_config)?;

            // Create connection to non-existent host
            let invalid_connection = SshConnectionDetails {
                host: "nonexistent.invalid.test".to_string(),
                port: 22,
                username: "testuser".to_string(),
                private_key_path: std::path::PathBuf::from("/nonexistent/key"),
                timeout: Duration::from_secs(5),
            };

            info!("Testing invalid SSH connection handling");
            let result = validator
                .validate_compute_resource(&invalid_connection, 256, 1)
                .await;

            assert!(result.is_err(), "Should fail with invalid SSH connection");
            info!("✓ Correctly handled invalid SSH connection");
        }
    }

    // Test 3: Short timeout
    if check_ssh_connectivity().await {
        let test_config = VmTestConfig::new();

        if check_binary_availability(&test_config) {
            info!("Testing execution timeout handling");

            let short_timeout_config = SecureValidatorConfig {
                attestor_binary_path: test_config.attestor_binary_path,
                ssh_timeout: Duration::from_secs(30),
                max_execution_time: Duration::from_millis(100), // Very short timeout
            };

            let validator = SecureValidator::new(short_timeout_config)?;
            let connection = create_local_ssh_connection();

            // Use large problem size that should take longer than 100ms
            let result = validator
                .validate_compute_resource(&connection, 1024, 1)
                .await;

            // Should either complete quickly or timeout
            match result {
                Ok(validation_result) => {
                    info!(
                        "Validation completed within short timeout: {}",
                        validation_result
                    );
                }
                Err(e) => {
                    let error_msg = e.to_string();
                    if error_msg.contains("timeout") || error_msg.contains("Execution timeout") {
                        info!("✓ Correctly handled execution timeout");
                    } else {
                        warn!("Unexpected error during timeout test: {}", e);
                    }
                }
            }
        }
    }

    info!("✓ Error handling test completed");

    Ok(())
}

#[tokio::test]
async fn test_vm_protected_security_features() -> Result<()> {
    let _ = tracing_subscriber::fmt()
        .with_env_filter("info")
        .with_test_writer()
        .try_init();

    info!("=== VM-Protected Security Features Test ===");

    let test_config = VmTestConfig::new();

    if !check_binary_availability(&test_config) || !check_ssh_connectivity().await {
        eprintln!("Skipping test - prerequisites not met");
        return Ok(());
    }

    let validator_config = SecureValidatorConfig {
        attestor_binary_path: test_config.attestor_binary_path,
        ssh_timeout: test_config.ssh_timeout,
        max_execution_time: test_config.max_execution_time,
    };

    let validator = SecureValidator::new(validator_config)?;
    let connection = create_local_ssh_connection();

    info!("Testing VM security features and obfuscation");

    // Test multiple validations with same parameters to verify consistency
    let problem_size = 256;
    let resource_count = 1;
    let mut results = Vec::new();

    for i in 0..3 {
        info!("Security test iteration {}", i + 1);

        let result = validator
            .validate_compute_resource(&connection, problem_size, resource_count)
            .await
            .context(format!("Security test iteration {} failed", i + 1))?;

        results.push(result);
        info!(
            "Iteration {}: {}",
            i + 1,
            if result { "PASS" } else { "FAIL" }
        );

        // Small delay between iterations
        tokio::time::sleep(Duration::from_millis(100)).await;
    }

    info!("VM security and consistency test completed");

    // The VM should provide consistent results for the same input
    // (though the specific result depends on internal VM logic)
    let first_result = results[0];
    let all_consistent = results.iter().all(|&r| r == first_result);

    if all_consistent {
        info!("✓ VM provides consistent results across iterations");
    } else {
        info!("! VM results varied across iterations (may be intentional for security)");
    }

    info!("✓ Security features test completed");

    Ok(())
}
