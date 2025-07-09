//! End-to-end integration tests for secure GPU validation
//! Tests the full VM-protected validation flow using the generic secure validator

use anyhow::{Context, Result};
use std::time::Duration;
use tracing::info;

// Import the secure validator (no protocol dependencies)
use validator::validation::secure_validator::{SecureValidator, SecureValidatorConfig};

// Import SSH connection for testing
use common::ssh::SshConnectionDetails;

/// Create a test SSH connection to localhost (simulating remote validation)
fn create_test_connection() -> SshConnectionDetails {
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

/// Helper function to check if we can run SSH tests
async fn can_run_ssh_tests() -> bool {
    // Check if we can connect to localhost SSH
    let connection = create_test_connection();

    // Try a simple SSH connection test
    match tokio::process::Command::new("ssh")
        .arg("-o")
        .arg("ConnectTimeout=5")
        .arg("-o")
        .arg("BatchMode=yes")
        .arg("-o")
        .arg("StrictHostKeyChecking=no")
        .arg(format!("{}@{}", connection.username, connection.host))
        .arg("echo test")
        .output()
        .await
    {
        Ok(output) => output.status.success(),
        Err(_) => false,
    }
}

#[tokio::test]
async fn test_secure_validator_basic_validation() -> Result<()> {
    // Initialize logging
    let _ = tracing_subscriber::fmt::try_init();

    info!("=== Secure GPU Validation Basic Test ===");

    // Create secure validator with test configuration
    let config = SecureValidatorConfig {
        attestor_binary_path: std::path::PathBuf::from("./debug/release/gpu-attestor"),
        ssh_timeout: Duration::from_secs(30),
        max_execution_time: Duration::from_secs(60),
    };

    // Check if binary exists
    if !config.attestor_binary_path.exists() {
        eprintln!(
            "Skipping test - gpu-attestor binary not found at {:?}",
            config.attestor_binary_path
        );
        return Ok(());
    }

    let validator = SecureValidator::new(config)?;

    // Test connection
    let connection = create_test_connection();

    // Check if we can run SSH tests
    if !can_run_ssh_tests().await {
        eprintln!("Skipping test - SSH to localhost not available");
        return Ok(());
    }

    info!("Testing secure validation with problem_size=256, resource_count=1");

    // Execute secure validation
    let result = validator
        .validate_compute_resource(&connection, 256, 1)
        .await
        .context("Secure validation failed")?;

    info!(
        "Validation result: {}",
        if result { "PASS" } else { "FAIL" }
    );

    // For this test, we expect validation to succeed if GPU hardware is available
    // If no GPUs, the result depends on the VM's validation logic
    assert!(
        result || !has_gpus_available(),
        "Validation should pass if GPUs are available"
    );

    Ok(())
}

#[tokio::test]
async fn test_secure_validator_multiple_problem_sizes() -> Result<()> {
    let _ = tracing_subscriber::fmt::try_init();

    info!("=== Secure GPU Validation Multiple Sizes Test ===");

    let config = SecureValidatorConfig::default();

    if !config.attestor_binary_path.exists() {
        eprintln!("Skipping test - gpu-attestor binary not found");
        return Ok(());
    }

    let validator = SecureValidator::new(config)?;
    let connection = create_test_connection();

    if !can_run_ssh_tests().await {
        eprintln!("Skipping test - SSH not available");
        return Ok(());
    }

    // Test different problem sizes
    let test_sizes = vec![64, 128, 256, 512];

    for size in test_sizes {
        info!("Testing problem size: {}", size);

        let start = std::time::Instant::now();
        let result = validator
            .validate_compute_resource(&connection, size, 1)
            .await
            .context(format!("Validation failed for size {size}"))?;
        let elapsed = start.elapsed();

        info!(
            "Size {}: {} in {:?}",
            size,
            if result { "PASS" } else { "FAIL" },
            elapsed
        );

        // Validation should be consistent regardless of problem size
        assert!(
            result || !has_gpus_available(),
            "Validation consistency check for size {size}"
        );
    }

    Ok(())
}

#[tokio::test]
async fn test_secure_validator_multiple_resources() -> Result<()> {
    let _ = tracing_subscriber::fmt::try_init();

    info!("=== Secure GPU Validation Multiple Resources Test ===");

    let config = SecureValidatorConfig::default();

    if !config.attestor_binary_path.exists() {
        eprintln!("Skipping test - gpu-attestor binary not found");
        return Ok(());
    }

    let validator = SecureValidator::new(config)?;
    let connection = create_test_connection();

    if !can_run_ssh_tests().await {
        eprintln!("Skipping test - SSH not available");
        return Ok(());
    }

    // Test different resource counts
    let resource_counts = vec![1, 2, 4];
    let problem_size = 256;

    for count in resource_counts {
        info!("Testing with {} expected resources", count);

        let result = validator
            .validate_compute_resource(&connection, problem_size, count)
            .await
            .context(format!("Validation failed for {count} resources"))?;

        info!(
            "Resource count {}: {}",
            count,
            if result { "PASS" } else { "FAIL" }
        );

        // The VM should handle resource count validation internally
        // Result depends on actual vs expected resource count
    }

    Ok(())
}

#[tokio::test]
async fn test_secure_validator_timeout_handling() -> Result<()> {
    let _ = tracing_subscriber::fmt::try_init();

    info!("=== Secure GPU Validation Timeout Test ===");

    // Create validator with very short timeout for testing
    let config = SecureValidatorConfig {
        attestor_binary_path: std::path::PathBuf::from("./target/release/gpu-attestor"),
        ssh_timeout: Duration::from_secs(5),
        max_execution_time: Duration::from_secs(1), // Very short timeout
    };

    if !config.attestor_binary_path.exists() {
        eprintln!("Skipping test - gpu-attestor binary not found");
        return Ok(());
    }

    let validator = SecureValidator::new(config)?;
    let connection = create_test_connection();

    if !can_run_ssh_tests().await {
        eprintln!("Skipping test - SSH not available");
        return Ok(());
    }

    // Use a large problem size that might take longer than 1 second
    let result = validator
        .validate_compute_resource(&connection, 1024, 1)
        .await;

    // We expect this to either succeed quickly or timeout
    match result {
        Ok(validation_result) => {
            info!("Validation completed within timeout: {}", validation_result);
        }
        Err(e) => {
            let error_msg = e.to_string();
            if error_msg.contains("timeout") || error_msg.contains("Execution timeout") {
                info!("Validation timed out as expected");
            } else {
                return Err(e).context("Unexpected error during timeout test");
            }
        }
    }

    Ok(())
}

#[tokio::test]
async fn test_secure_validator_error_handling() -> Result<()> {
    let _ = tracing_subscriber::fmt::try_init();

    info!("=== Secure GPU Validation Error Handling Test ===");

    // Test with invalid binary path
    let invalid_config = SecureValidatorConfig {
        attestor_binary_path: std::path::PathBuf::from("/nonexistent/binary"),
        ssh_timeout: Duration::from_secs(30),
        max_execution_time: Duration::from_secs(60),
    };

    let result = SecureValidator::new(invalid_config);
    assert!(result.is_err(), "Should fail with invalid binary path");

    info!("✓ Correctly handled invalid binary path");

    // Test with valid config but invalid connection
    let config = SecureValidatorConfig::default();

    if !config.attestor_binary_path.exists() {
        eprintln!("Skipping connection test - gpu-attestor binary not found");
        return Ok(());
    }

    let validator = SecureValidator::new(config)?;

    // Create invalid connection
    let invalid_connection = SshConnectionDetails {
        host: "invalid.nonexistent.host.test".to_string(),
        port: 22,
        username: "testuser".to_string(),
        private_key_path: std::path::PathBuf::from("/nonexistent/key"),
        timeout: Duration::from_secs(5),
    };

    let result = validator
        .validate_compute_resource(&invalid_connection, 256, 1)
        .await;

    assert!(result.is_err(), "Should fail with invalid connection");

    info!("✓ Correctly handled invalid SSH connection");

    Ok(())
}

#[tokio::test]
async fn test_secure_validator_concurrent_validations() -> Result<()> {
    let _ = tracing_subscriber::fmt::try_init();

    info!("=== Secure GPU Validation Concurrent Test ===");

    let config = SecureValidatorConfig::default();

    if !config.attestor_binary_path.exists() {
        eprintln!("Skipping test - gpu-attestor binary not found");
        return Ok(());
    }

    let validator = std::sync::Arc::new(SecureValidator::new(config)?);
    let connection = create_test_connection();

    if !can_run_ssh_tests().await {
        eprintln!("Skipping test - SSH not available");
        return Ok(());
    }

    // Run multiple validations concurrently
    let tasks: Vec<_> = (0..3)
        .map(|i| {
            let validator = validator.clone();
            let connection = connection.clone();

            tokio::spawn(async move {
                let problem_size = 128 + (i * 64); // Different sizes: 128, 192, 256
                info!("Concurrent validation {} with size {}", i, problem_size);

                let result = validator
                    .validate_compute_resource(&connection, problem_size, 1)
                    .await
                    .context(format!("Concurrent validation {i} failed"))?;

                info!(
                    "Concurrent validation {}: {}",
                    i,
                    if result { "PASS" } else { "FAIL" }
                );

                Ok::<bool, anyhow::Error>(result)
            })
        })
        .collect();

    // Wait for all tasks to complete
    let results = futures::future::try_join_all(tasks).await?;

    info!("All {} concurrent validations completed", results.len());

    // All validations should have completed without errors
    for (i, result) in results.into_iter().enumerate() {
        let validation_result = result?;
        info!(
            "Task {}: {}",
            i,
            if validation_result { "PASS" } else { "FAIL" }
        );
    }

    Ok(())
}

/// Check if GPUs are available on the system
fn has_gpus_available() -> bool {
    // Try to detect GPUs using a simple check
    std::process::Command::new("nvidia-smi")
        .arg("--list-gpus")
        .output()
        .map(|output| output.status.success())
        .unwrap_or(false)
}

#[tokio::test]
async fn test_secure_validator_performance_characteristics() -> Result<()> {
    let _ = tracing_subscriber::fmt::try_init();

    info!("=== Secure GPU Validation Performance Test ===");

    let config = SecureValidatorConfig::default();

    if !config.attestor_binary_path.exists() {
        eprintln!("Skipping test - gpu-attestor binary not found");
        return Ok(());
    }

    let validator = SecureValidator::new(config)?;
    let connection = create_test_connection();

    if !can_run_ssh_tests().await {
        eprintln!("Skipping test - SSH not available");
        return Ok(());
    }

    // Test performance with different problem sizes
    let sizes = vec![64, 128, 256, 512];

    info!("Testing performance characteristics:");
    info!("{:<12} {:<12} {:<12}", "Size", "Time (ms)", "Result");
    info!("{:-<36}", "");

    for size in sizes {
        let start = std::time::Instant::now();

        let result = validator
            .validate_compute_resource(&connection, size, 1)
            .await
            .context(format!("Performance test failed for size {size}"))?;

        let elapsed = start.elapsed();

        info!(
            "{:<12} {:<12} {:<12}",
            size,
            elapsed.as_millis(),
            if result { "PASS" } else { "FAIL" }
        );

        // Ensure validation doesn't take too long
        assert!(
            elapsed < Duration::from_secs(30),
            "Validation for size {size} took too long: {elapsed:?}"
        );
    }

    Ok(())
}
