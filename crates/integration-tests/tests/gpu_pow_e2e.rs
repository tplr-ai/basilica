//! GPU Proof-of-Work End-to-End Integration Test
//!
//! This test simulates the complete flow of:
//! 1. A miner with a GPU executor
//! 2. A validator issuing a GPU PoW challenge
//! 3. The executor completing the challenge
//! 4. The validator verifying the result

use anyhow::{Context, Result};
use base64::{engine::general_purpose, Engine as _};
use std::process::{Command, Stdio};
use tracing::{error, info};

use protocol::basilca::common::v1::{ChallengeParameters, ChallengeResult};
use validator::validation::{challenge_generator::ChallengeGenerator, gpu_validator::GpuValidator};

// Custom struct that matches gpu-attestor JSON output
#[derive(serde::Deserialize)]
struct GpuAttestorResult {
    solution: String,
    execution_time_ms: u64,
    gpu_utilization: Vec<f64>,
    memory_usage_mb: u64,
    error_message: String,
    metadata_json: String,
    result_checksum: String, // Hex string in JSON
    success: bool,
    gpu_model: String,
    vram_allocated_mb: u64,
    challenge_id: String,
}

/// Test configuration
struct TestConfig {
    gpu_attestor_path: String,
    validator_gpu_model: String,
    validator_vram_gb: u32,
}

impl TestConfig {
    fn new() -> Result<Self> {
        // Path to gpu-attestor binary - prefer release build for performance
        let gpu_attestor_path = std::env::var("GPU_ATTESTOR_PATH").unwrap_or_else(|_| {
            // Try to find the workspace root
            let manifest_dir = env!("CARGO_MANIFEST_DIR");
            let workspace_root = std::path::Path::new(manifest_dir)
                .parent()
                .and_then(|p| p.parent())
                .unwrap_or_else(|| std::path::Path::new("."));
            
            // Check if release build exists
            let release_path = workspace_root.join("target/release/gpu-attestor");
            let debug_path = workspace_root.join("target/debug/gpu-attestor");
            
            if release_path.exists() {
                release_path.to_string_lossy().to_string()
            } else if debug_path.exists() {
                debug_path.to_string_lossy().to_string()
            } else {
                // Fallback to relative path
                "../../target/debug/gpu-attestor".to_string()
            }
        });

        // Ensure gpu-attestor exists
        if !std::path::Path::new(&gpu_attestor_path).exists() {
            anyhow::bail!(
                "gpu-attestor binary not found at {}. Please build it first with:\n\
                 cargo build --bin gpu-attestor",
                gpu_attestor_path
            );
        }

        // Default to H100 for testing
        Ok(Self {
            gpu_attestor_path,
            validator_gpu_model: "NVIDIA H100 PCIe".to_string(),
            validator_vram_gb: 80,
        })
    }
}

/// Simulate a miner's GPU executor
async fn simulate_miner_executor(
    config: &TestConfig,
    challenge: &ChallengeParameters,
) -> Result<ChallengeResult> {
    info!("Miner: Received challenge, executing on GPU...");

    // Serialize challenge to JSON and base64 encode
    let challenge_json = serde_json::to_string(challenge)?;
    let challenge_b64 = general_purpose::STANDARD.encode(&challenge_json);

    // Execute gpu-attestor with challenge
    let output = Command::new(&config.gpu_attestor_path)
        .arg("--challenge")
        .arg(&challenge_b64)
        .stdout(Stdio::piped())
        .stderr(Stdio::piped())
        .output()
        .context("Failed to execute gpu-attestor")?;

    if !output.status.success() {
        let stderr = String::from_utf8_lossy(&output.stderr);
        anyhow::bail!("gpu-attestor failed: {}", stderr);
    }

    // Parse the result from stdout
    let stdout = String::from_utf8(output.stdout)?;
    let gpu_result: GpuAttestorResult =
        serde_json::from_str(&stdout).context("Failed to parse challenge result")?;

    info!("Miner: Challenge completed successfully");
    info!("  - Checksum: {}", &gpu_result.result_checksum);
    info!("  - Execution time: {} ms", gpu_result.execution_time_ms);
    info!("  - VRAM used: {} MB", gpu_result.vram_allocated_mb);

    // Parse metadata to show bandwidth if available
    if let Ok(metadata) = serde_json::from_str::<serde_json::Value>(&gpu_result.metadata_json) {
        if let Some(bandwidth) = metadata.get("bandwidth_gbps").and_then(|v| v.as_f64()) {
            info!("  - Bandwidth achieved: {:.2} GB/s", bandwidth);
        }
        if let Some(version) = metadata.get("version").and_then(|v| v.as_str()) {
            info!("  - PoW version: {}", version);
        }
    }

    // Convert to protobuf ChallengeResult
    let result = ChallengeResult {
        solution: gpu_result.solution,
        execution_time_ms: gpu_result.execution_time_ms,
        gpu_utilization: gpu_result.gpu_utilization.into_iter().map(|v| v).collect(),
        memory_usage_mb: gpu_result.memory_usage_mb,
        error_message: gpu_result.error_message,
        metadata_json: gpu_result.metadata_json,
        result_checksum: hex::decode(&gpu_result.result_checksum)
            .context("Failed to decode checksum hex")?,
        success: gpu_result.success,
        gpu_model: gpu_result.gpu_model,
        vram_allocated_mb: gpu_result.vram_allocated_mb,
        challenge_id: gpu_result.challenge_id,
    };

    Ok(result)
}

/// Detect GPUs using the unified detector
async fn detect_gpus(config: &TestConfig) -> Result<()> {
    info!("Detecting GPUs on the system using unified detector...");
    
    // Execute gpu-attestor with --detect flag
    let output = Command::new(&config.gpu_attestor_path)
        .arg("--detect")
        .stdout(Stdio::piped())
        .stderr(Stdio::piped())
        .output()
        .context("Failed to execute gpu-attestor --detect")?;

    if !output.status.success() {
        let stderr = String::from_utf8_lossy(&output.stderr);
        anyhow::bail!("gpu-attestor --detect failed: {}", stderr);
    }

    let stdout = String::from_utf8_lossy(&output.stdout);
    
    // Parse the output to check GPU count
    let gpu_count = stdout.lines()
        .find(|line| line.contains("Total GPUs detected:"))
        .and_then(|line| line.split(':').last())
        .and_then(|s| s.trim().parse::<usize>().ok())
        .unwrap_or(0);
    
    info!("Detected {} GPU(s) using unified detector", gpu_count);
    
    // Verify at least one GPU is present
    if gpu_count == 0 {
        anyhow::bail!("No GPUs detected. This test requires at least one CUDA-capable GPU.");
    }
    
    Ok(())
}

/// The main end-to-end test
#[tokio::test]
async fn gpu_pow_end_to_end_flow() -> Result<()> {
    // Initialize logging
    let _ = tracing_subscriber::fmt()
        .with_env_filter("info")
        .with_test_writer()
        .try_init();

    info!("=== GPU PoW End-to-End Test Starting ===");

    // Load test configuration
    let config = TestConfig::new()?;
    
    // Step 0: Detect GPUs using unified detector
    info!("\n--- Step 0: GPU Detection ---");
    detect_gpus(&config).await?;

    // Step 1: Initialize validator
    info!("\n--- Step 1: Validator Initialization ---");
    let mut gpu_validator = GpuValidator::new(config.gpu_attestor_path.clone());

    // Initialize validator's GPU detection
    gpu_validator
        .initialize()
        .await
        .context("Failed to initialize validator GPU")?;

    let validator_gpu = gpu_validator
        .get_gpu_model()
        .ok_or_else(|| anyhow::anyhow!("No GPU detected on validator"))?;

    info!("Validator GPU: {}", validator_gpu);

    // Step 2: Generate challenge
    info!("\n--- Step 2: Challenge Generation ---");
    let challenge_gen = ChallengeGenerator::new();
    let challenge = challenge_gen.generate_challenge(
        &config.validator_gpu_model,
        config.validator_vram_gb,
        Some("e2e_test_nonce_42".to_string()),
    )?;

    info!("Generated challenge:");
    info!("  - Type: {}", challenge.challenge_type);
    info!("  - Seed: {}", challenge.gpu_pow_seed);
    info!(
        "  - Matrix dimension: {}x{}",
        challenge.matrix_dim, challenge.matrix_dim
    );
    info!("  - Number of matrices: {}", challenge.num_matrices);
    info!("  - Number of iterations: {}", challenge.num_iterations);
    info!(
        "  - Verification sample rate: {:.1}%",
        challenge.verification_sample_rate * 100.0
    );
    info!("  - Validator nonce: {}", challenge.validator_nonce);

    // Step 3: Miner executes challenge
    info!("\n--- Step 3: Miner Execution ---");
    let miner_result = simulate_miner_executor(&config, &challenge).await?;

    // Step 4: Validator verifies result
    info!("\n--- Step 4: Validator Verification ---");
    let verification_start = std::time::Instant::now();

    let is_valid = gpu_validator
        .verify_challenge(&challenge, &miner_result)
        .await
        .context("Challenge verification failed")?;

    let verification_time = verification_start.elapsed();

    if is_valid {
        info!("✅ Challenge verification PASSED!");
        info!("  - Checksum matches");
        info!("  - Execution time is reasonable");
        info!("  - GPU model matches");
        info!("  - Verification took: {:?}", verification_time);
    } else {
        error!("❌ Challenge verification FAILED!");
        anyhow::bail!("GPU PoW verification failed");
    }

    // Step 5: Test failure scenarios
    info!("\n--- Step 5: Testing Failure Scenarios ---");

    // Test 5a: Wrong checksum
    info!("Testing wrong checksum...");
    let mut bad_result = miner_result.clone();
    bad_result.result_checksum = vec![0; 32];

    let should_fail = gpu_validator
        .verify_challenge(&challenge, &bad_result)
        .await?;
    assert!(!should_fail, "Should reject wrong checksum");
    info!("✅ Correctly rejected wrong checksum");

    // Test 5b: Wrong GPU model
    info!("Testing wrong GPU model...");
    let mut bad_result = miner_result.clone();
    bad_result.gpu_model = "NVIDIA RTX 4090".to_string();

    let should_fail = gpu_validator
        .verify_challenge(&challenge, &bad_result)
        .await?;
    assert!(!should_fail, "Should reject wrong GPU model");
    info!("✅ Correctly rejected wrong GPU model");

    // Test 5c: Wrong nonce
    info!("Testing wrong nonce...");
    let mut bad_result = miner_result.clone();
    bad_result.challenge_id = "wrong_nonce".to_string();

    let should_fail = gpu_validator
        .verify_challenge(&challenge, &bad_result)
        .await?;
    assert!(!should_fail, "Should reject wrong nonce");
    info!("✅ Correctly rejected wrong nonce");

    // Test 5d: Execution time too fast (suspicious)
    info!("Testing suspiciously fast execution time...");
    let mut bad_result = miner_result.clone();
    bad_result.execution_time_ms = 1; // 1ms is impossibly fast

    let should_fail = gpu_validator
        .verify_challenge(&challenge, &bad_result)
        .await?;
    assert!(!should_fail, "Should reject suspiciously fast execution");
    info!("✅ Correctly rejected suspiciously fast execution");

    info!("\n=== GPU PoW End-to-End Test PASSED ===");

    Ok(())
}

/// Test multiple challenges in sequence
#[tokio::test]
async fn gpu_pow_multiple_challenges() -> Result<()> {
    let _ = tracing_subscriber::fmt()
        .with_env_filter("info")
        .with_test_writer()
        .try_init();

    let config = TestConfig::new()?;
    let mut gpu_validator = GpuValidator::new(config.gpu_attestor_path.clone());
    gpu_validator.initialize().await?;

    let challenge_gen = ChallengeGenerator::new();

    info!("Testing multiple sequential challenges...");

    for i in 0..3 {
        info!("\n--- Challenge {} ---", i + 1);

        let challenge = challenge_gen.generate_challenge(
            &config.validator_gpu_model,
            config.validator_vram_gb,
            Some(format!("multi_test_{i}")),
        )?;

        let result = simulate_miner_executor(&config, &challenge).await?;
        let is_valid = gpu_validator.verify_challenge(&challenge, &result).await?;

        assert!(is_valid, "Challenge {} verification failed", i + 1);
        info!("✅ Challenge {} passed", i + 1);
    }

    info!("\n=== Multiple Challenges Test PASSED ===");
    Ok(())
}

/// Test concurrent challenge execution
#[tokio::test]
async fn gpu_pow_concurrent_challenges() -> Result<()> {
    let _ = tracing_subscriber::fmt()
        .with_env_filter("info")
        .with_test_writer()
        .try_init();

    let config = TestConfig::new()?;

    info!("Testing concurrent challenge generation and verification...");

    // Generate multiple challenges
    let challenge_gen = ChallengeGenerator::new();
    let mut challenges = Vec::new();

    for i in 0..3 {
        let challenge = challenge_gen.generate_challenge(
            &config.validator_gpu_model,
            config.validator_vram_gb,
            Some(format!("concurrent_test_{i}")),
        )?;
        challenges.push(challenge);
    }

    // Execute challenges concurrently (simulating multiple miners)
    let mut handles = Vec::new();

    for (i, challenge) in challenges.iter().enumerate() {
        let config = config.gpu_attestor_path.clone();
        let challenge = challenge.clone();

        let handle = tokio::spawn(async move {
            info!("Executing challenge {} concurrently...", i);

            let challenge_json = serde_json::to_string(&challenge)?;
            let challenge_b64 = general_purpose::STANDARD.encode(&challenge_json);

            let output = tokio::process::Command::new(&config)
                .arg("--challenge")
                .arg(&challenge_b64)
                .output()
                .await?;

            if !output.status.success() {
                anyhow::bail!("Challenge {} execution failed", i);
            }

            let stdout = String::from_utf8(output.stdout)?;
            let gpu_result: GpuAttestorResult = serde_json::from_str(&stdout)?;

            // Convert to protobuf ChallengeResult
            let result = ChallengeResult {
                solution: gpu_result.solution,
                execution_time_ms: gpu_result.execution_time_ms,
                gpu_utilization: gpu_result.gpu_utilization,
                memory_usage_mb: gpu_result.memory_usage_mb,
                error_message: gpu_result.error_message,
                metadata_json: gpu_result.metadata_json,
                result_checksum: hex::decode(&gpu_result.result_checksum)?,
                success: gpu_result.success,
                gpu_model: gpu_result.gpu_model,
                vram_allocated_mb: gpu_result.vram_allocated_mb,
                challenge_id: gpu_result.challenge_id,
            };
            Ok::<(usize, ChallengeResult), anyhow::Error>((i, result))
        });

        handles.push(handle);
    }

    // Wait for all challenges to complete
    let mut results = Vec::new();
    for handle in handles {
        let (idx, result) = handle.await??;
        results.push((idx, result));
    }

    // Verify all results
    let mut gpu_validator = GpuValidator::new(config.gpu_attestor_path);
    gpu_validator.initialize().await?;

    for (idx, result) in results {
        let challenge = &challenges[idx];
        let is_valid = gpu_validator.verify_challenge(challenge, &result).await?;
        assert!(is_valid, "Concurrent challenge {idx} verification failed");
        info!("✅ Concurrent challenge {} verified", idx);
    }

    info!("\n=== Concurrent Challenges Test PASSED ===");
    Ok(())
}

/// Test bandwidth-intensive PoW v2 with statistical sampling
#[tokio::test]
async fn gpu_pow_v2_bandwidth_test() -> Result<()> {
    let _ = tracing_subscriber::fmt()
        .with_env_filter("info")
        .with_test_writer()
        .try_init();

    info!("=== GPU PoW V2 Bandwidth Test Starting ===");

    let config = TestConfig::new()?;

    // Step 1: Initialize validator
    info!("\n--- Step 1: Validator Initialization ---");
    let mut gpu_validator = GpuValidator::new(config.gpu_attestor_path.clone());
    gpu_validator.initialize().await?;

    let challenge_gen = ChallengeGenerator::new();

    // Step 2: Generate bandwidth-intensive challenge for H100
    info!("\n--- Step 2: Generating Bandwidth Challenge ---");
    let challenge = challenge_gen.generate_challenge(
        "NVIDIA H100 PCIe",
        80, // H100 with 80GB
        Some("bandwidth_test_nonce".to_string()),
    )?;

    // Verify it's using v2 parameters
    assert!(
        challenge.num_iterations > 0,
        "Should have iterations for v2"
    );
    assert_eq!(
        challenge.matrix_dim, 1024,
        "H100 should use 1024x1024 matrices"
    );
    assert_eq!(
        challenge.num_iterations, 4200,
        "H100 should use 4200 iterations"
    );

    info!("Generated bandwidth-intensive challenge:");
    info!(
        "  - Matrix dimension: {}x{}",
        challenge.matrix_dim, challenge.matrix_dim
    );
    info!("  - Number of matrices: {}", challenge.num_matrices);
    info!("  - Number of iterations: {}", challenge.num_iterations);
    info!("  - Expected bandwidth transfer: ~100GB");

    // Step 3: Execute challenge (full)
    info!("\n--- Step 3: Full Challenge Execution ---");
    let full_start = std::time::Instant::now();
    let full_result = simulate_miner_executor(&config, &challenge).await?;
    let full_duration = full_start.elapsed();

    info!(
        "Full execution completed in {:.2}s",
        full_duration.as_secs_f64()
    );

    // Check bandwidth from metadata
    let metadata: serde_json::Value = serde_json::from_str(&full_result.metadata_json)?;
    let bandwidth_gbps = metadata
        .get("bandwidth_gbps")
        .and_then(|v| v.as_f64())
        .unwrap_or(0.0);

    info!("Achieved bandwidth: {:.2} GB/s", bandwidth_gbps);
    assert!(
        bandwidth_gbps > 100.0,
        "H100 should achieve >100 GB/s bandwidth"
    );

    // Step 4: Verify with statistical sampling
    info!("\n--- Step 4: Statistical Sampling Verification ---");
    let verify_start = std::time::Instant::now();
    let is_valid = gpu_validator
        .verify_challenge(&challenge, &full_result)
        .await?;
    let verify_duration = verify_start.elapsed();

    assert!(is_valid, "Challenge verification should pass");
    info!("✅ Verification passed using statistical sampling");
    info!(
        "  - Verification time: {:.2}s",
        verify_duration.as_secs_f64()
    );
    info!(
        "  - Speedup: {:.1}x",
        full_duration.as_secs_f64() / verify_duration.as_secs_f64()
    );

    // The verification should be significantly faster due to sampling
    assert!(
        verify_duration.as_secs_f64() < full_duration.as_secs_f64() * 0.3,
        "Sampling verification should be at least 3x faster"
    );

    info!("\n=== GPU PoW V2 Bandwidth Test PASSED ===");
    Ok(())
}

/// Test sampled challenge execution mode
#[tokio::test]
async fn gpu_pow_sampled_execution_test() -> Result<()> {
    let _ = tracing_subscriber::fmt()
        .with_env_filter("info")
        .with_test_writer()
        .try_init();

    info!("=== GPU PoW Sampled Execution Test ===");

    let config = TestConfig::new()?;
    let challenge_gen = ChallengeGenerator::new();

    // Generate challenge
    let challenge = challenge_gen.generate_challenge(
        "NVIDIA H100 PCIe",
        80,
        Some("sampled_test".to_string()),
    )?;

    // Prepare sampled challenge params
    let sample_count = (challenge.num_iterations as f32 * 0.1) as usize; // 10% sampling
    let mut sampled_iterations: Vec<u32> = Vec::new();

    // Generate deterministic sample indices (same as validator would)
    use sha2::{Digest, Sha256};
    let mut hasher = Sha256::new();
    hasher.update(challenge.gpu_pow_seed.to_le_bytes());
    hasher.update(b"sampling_nonce");
    let hash = hasher.finalize();

    // Use hash to seed iteration selection
    let mut rng_state = u64::from_le_bytes(hash[0..8].try_into().unwrap());
    for i in 0..sample_count {
        rng_state = rng_state.wrapping_mul(6364136223846793005).wrapping_add(1);
        let iteration = (rng_state % challenge.num_iterations as u64) as u32;
        if !sampled_iterations.contains(&iteration) {
            sampled_iterations.push(iteration);
        }
    }
    sampled_iterations.sort_unstable();

    info!("Testing sampled execution:");
    info!("  - Total iterations: {}", challenge.num_iterations);
    info!("  - Sampled iterations: {}", sampled_iterations.len());

    // Create sampled params
    let sampled_params = serde_json::json!({
        "original_params": challenge,
        "sampled_iterations": sampled_iterations
    });

    let sampled_json = serde_json::to_string(&sampled_params)?;
    let sampled_b64 = general_purpose::STANDARD.encode(&sampled_json);

    // Execute sampled challenge
    let output = Command::new(&config.gpu_attestor_path)
        .arg("--challenge-sampled")
        .arg(&sampled_b64)
        .stdout(Stdio::piped())
        .stderr(Stdio::piped())
        .output()?;

    assert!(output.status.success(), "Sampled execution should succeed");

    let stdout = String::from_utf8(output.stdout)?;
    let result: serde_json::Value = serde_json::from_str(&stdout)?;

    info!("Sampled execution result:");
    info!(
        "  - Checksum: {}",
        result["sampled_checksum"].as_str().unwrap()
    );
    info!(
        "  - Execution time: {} ms",
        result["execution_time_ms"].as_u64().unwrap()
    );
    info!(
        "  - Samples verified: {}",
        result["num_samples"].as_u64().unwrap()
    );

    info!("\n=== Sampled Execution Test PASSED ===");
    Ok(())
}

/// Test bandwidth PoW with different GPU configurations
#[tokio::test]
async fn gpu_pow_different_gpu_configs() -> Result<()> {
    let _ = tracing_subscriber::fmt()
        .with_env_filter("info")
        .with_test_writer()
        .try_init();

    info!("=== Testing Different GPU Configurations ===");

    let config = TestConfig::new()?;
    let challenge_gen = ChallengeGenerator::new();

    // Test H100 configuration only
    let gpu_model = "NVIDIA H100 PCIe";
    let vram_gb = 80u32;
    let expected_dim = 1024u32;
    let expected_iterations = 4200u32;

    info!("\n--- Testing {} ({}GB) ---", gpu_model, vram_gb);

    let challenge = challenge_gen.generate_challenge(
        gpu_model,
        vram_gb,
        Some(format!("{}_test", gpu_model.replace(" ", "_"))),
    )?;

    // Verify correct parameters for H100
    assert_eq!(
        challenge.matrix_dim, expected_dim,
        "{gpu_model} should use {expected_dim}x{expected_dim} matrices"
    );
    assert_eq!(
        challenge.num_iterations, expected_iterations,
        "{gpu_model} should use {expected_iterations} iterations"
    );

    // Calculate expected memory usage and bandwidth
    let matrix_size_mb = (expected_dim * expected_dim * 8) / (1024 * 1024); // 8 bytes per f64
    let num_matrices = challenge.num_matrices;
    let total_memory_mb = matrix_size_mb * num_matrices; // Total matrices allocated
    let bandwidth_transfer_gb = (matrix_size_mb * 3 * expected_iterations) / 1024; // Each iteration transfers 3 matrices

    info!("  Matrix dimension: {}x{}", expected_dim, expected_dim);
    info!("  Number of matrices: {}", num_matrices);
    info!("  Number of iterations: {}", expected_iterations);
    info!("  Expected memory usage: ~{} MB", total_memory_mb);
    info!(
        "  Expected bandwidth transfer: ~{} GB",
        bandwidth_transfer_gb
    );

    // Verify memory usage is appropriate for H100
    let expected_usage_ratio = total_memory_mb as f32 / (vram_gb as f32 * 1024.0);
    assert!(
        expected_usage_ratio > 0.7 && expected_usage_ratio < 0.95,
        "{} should use 70-95% of VRAM, but uses {:.1}%",
        gpu_model,
        expected_usage_ratio * 100.0
    );

    info!("  ✅ Configuration validated for {}", gpu_model);

    info!("\n=== Different GPU Configurations Test PASSED ===");
    Ok(())
}

/// Test edge cases and error conditions
#[tokio::test]
async fn gpu_pow_edge_cases() -> Result<()> {
    let _ = tracing_subscriber::fmt()
        .with_env_filter("info")
        .with_test_writer()
        .try_init();

    info!("=== Testing Edge Cases ===");

    let config = TestConfig::new()?;
    let mut gpu_validator = GpuValidator::new(config.gpu_attestor_path.clone());
    gpu_validator.initialize().await?;

    let challenge_gen = ChallengeGenerator::new();

    // Test 1: H100 GPU configuration
    info!("\n--- Test 1: H100 GPU Configuration ---");
    let h100_challenge = challenge_gen.generate_challenge(
        "NVIDIA H100 PCIe",
        80, // 80GB VRAM
        Some("h100_gpu_test".to_string()),
    )?;

    assert!(
        h100_challenge.num_iterations > 0,
        "Should always have iterations"
    );
    assert_eq!(
        h100_challenge.matrix_dim, 1024,
        "H100 should use 1024x1024 matrices"
    );
    assert_eq!(
        h100_challenge.num_iterations, 4200,
        "H100 should use 4200 iterations"
    );
    info!("✅ H100 GPU configuration validated");

    // Test 2: Sample rate validation
    info!("\n--- Test 2: Sample Rate Validation ---");
    let challenge = challenge_gen.generate_challenge(
        "NVIDIA H100 PCIe",
        80,
        Some("sample_rate_test".to_string()),
    )?;
    assert!(
        challenge.verification_sample_rate >= 0.0 && challenge.verification_sample_rate <= 1.0,
        "Sample rate should be clamped to valid range"
    );
    assert_eq!(
        challenge.verification_sample_rate, 0.1,
        "Default sample rate should be 10%"
    );
    info!("✅ Sample rate properly validated");

    info!("\n=== Edge Cases Test PASSED ===");
    Ok(())
}
