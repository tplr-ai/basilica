//! Integration test for GPU profile-aware Freivalds validation

use anyhow::Result;
use common::ssh::SshConnectionDetails;
use std::path::PathBuf;
use std::time::Duration;
use validator::validation::freivalds_validator::{FreivaldsGpuValidator, FreivaldsValidatorConfig};

#[tokio::test]
async fn test_freivalds_with_gpu_profiling() -> Result<()> {
    // Skip test if no SSH connection details are provided
    if std::env::var("TEST_SSH_HOST").is_err() {
        eprintln!(
            "Skipping SSH test - set TEST_SSH_HOST, TEST_SSH_USER, and TEST_SSH_KEY_PATH to run"
        );
        return Ok(());
    }

    // Create SSH connection details from environment
    let connection = SshConnectionDetails {
        host: std::env::var("TEST_SSH_HOST").expect("TEST_SSH_HOST not set"),
        username: std::env::var("TEST_SSH_USER").unwrap_or_else(|_| "ubuntu".to_string()),
        port: std::env::var("TEST_SSH_PORT")
            .ok()
            .and_then(|p| p.parse().ok())
            .unwrap_or(22),
        private_key_path: PathBuf::from(
            std::env::var("TEST_SSH_KEY_PATH").expect("TEST_SSH_KEY_PATH not set"),
        ),
        timeout: Duration::from_secs(30),
    };

    // Create validator with configuration
    let config = FreivaldsValidatorConfig {
        gpu_attestor_path: PathBuf::from("./crates/gpu-attestor"),
        temp_dir: PathBuf::from("/tmp/freivalds_test"),
        ssh_timeout: Duration::from_secs(300),
        max_matrix_size: 4096,
        num_spot_checks: 10,
    };

    let validator = FreivaldsGpuValidator::new(config)?;

    // Test 1: Generate challenge with GPU profiling
    println!("Testing challenge generation with GPU profiling...");
    let session_id = format!(
        "test_session_{}",
        std::time::SystemTime::now()
            .duration_since(std::time::UNIX_EPOCH)
            .unwrap()
            .as_millis()
    );

    // Execute challenge with automatic profiling
    match validator
        .execute_challenge_with_profiling(&connection, session_id.clone())
        .await
    {
        Ok((challenge, commitment)) => {
            println!("✅ GPU profiling successful!");
            println!("  - Matrix size: {}×{}", challenge.n, challenge.n);
            println!("  - GPU count: {}", challenge.expected_gpu_count);
            println!(
                "  - Computation timeout: {}ms",
                challenge.computation_timeout_ms
            );
            println!("  - Protocol timeout: {}ms", challenge.protocol_timeout_ms);
            println!("  - Merkle root: {}", hex::encode(&commitment.merkle_root));

            // Verify the commitment
            let result = validator
                .verify_result(&connection, &challenge, &commitment)
                .await?;
            println!(
                "  - Verification: {}",
                if result.verified { "PASSED" } else { "FAILED" }
            );

            assert!(result.verified, "Verification should pass");
            assert_eq!(commitment.session_id, session_id);
        }
        Err(e) => {
            eprintln!("❌ GPU profiling failed: {e}");
            return Err(e);
        }
    }

    // Test 2: Compare with static timeout generation
    println!("\nTesting static challenge generation for comparison...");
    let static_challenge = validator.generate_challenge(format!("static_{session_id}"), 1024, 1);

    println!("Static challenge timeouts:");
    println!(
        "  - Computation timeout: {}ms",
        static_challenge.computation_timeout_ms
    );
    println!(
        "  - Protocol timeout: {}ms",
        static_challenge.protocol_timeout_ms
    );

    Ok(())
}

#[tokio::test]
async fn test_gpu_profile_caching() -> Result<()> {
    use validator::ssh::ValidatorSshClient;
    use validator::validation::GpuProfileQuery;

    // Skip test if no SSH connection details
    if std::env::var("TEST_SSH_HOST").is_err() {
        eprintln!("Skipping SSH test - set TEST_SSH_HOST to run");
        return Ok(());
    }

    let connection = SshConnectionDetails {
        host: std::env::var("TEST_SSH_HOST").expect("TEST_SSH_HOST not set"),
        username: std::env::var("TEST_SSH_USER").unwrap_or_else(|_| "ubuntu".to_string()),
        port: 22,
        private_key_path: PathBuf::from(
            std::env::var("TEST_SSH_KEY_PATH").expect("TEST_SSH_KEY_PATH not set"),
        ),
        timeout: Duration::from_secs(30),
    };

    let ssh_client = ValidatorSshClient::new();
    let gpu_profiler = GpuProfileQuery::new(ssh_client);

    // Assume gpu-attestor binary exists
    let binary_path = PathBuf::from("./target/release/gpu-attestor");

    // First query should hit the executor
    let start1 = std::time::Instant::now();
    let profile1 = gpu_profiler
        .query_profile(&connection, &binary_path)
        .await?;
    let duration1 = start1.elapsed();

    println!("First query took: {duration1:?}");
    println!(
        "Profile: {} GPUs, {:.1} TFLOPS",
        profile1.devices.len(),
        profile1.total_compute_power
    );

    // Second query should be cached
    let start2 = std::time::Instant::now();
    let profile2 = gpu_profiler
        .query_profile(&connection, &binary_path)
        .await?;
    let duration2 = start2.elapsed();

    println!("Second query took: {duration2:?}");

    // Cached query should be much faster
    assert!(
        duration2 < duration1 / 10,
        "Cached query should be much faster"
    );

    // Profiles should be identical
    assert_eq!(profile1.devices.len(), profile2.devices.len());
    assert_eq!(profile1.total_compute_power, profile2.total_compute_power);

    Ok(())
}
