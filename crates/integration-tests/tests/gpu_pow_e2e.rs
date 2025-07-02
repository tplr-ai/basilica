//! GPU Proof-of-Work End-to-End Integration Test
//! 
//! This test simulates the complete flow of:
//! 1. A miner with a GPU executor
//! 2. A validator issuing a GPU PoW challenge
//! 3. The executor completing the challenge
//! 4. The validator verifying the result

use anyhow::{Context, Result};
use std::process::{Command, Stdio};
use std::time::Duration;
use tokio::time::timeout;
use tracing::{info, warn, error};

use validator::validation::{
    challenge_generator::ChallengeGenerator,
    gpu_validator::GpuValidator,
};
use protocol::basilca::common::v1::{ChallengeParameters, ChallengeResult};

// Custom struct that matches gpu-attestor JSON output
#[derive(serde::Deserialize)]
struct GpuAttestorResult {
    solution: String,
    execution_time_ms: u64,
    gpu_utilization: Vec<f64>,
    memory_usage_mb: u64,
    error_message: String,
    metadata_json: String,
    result_checksum: String,  // Hex string in JSON
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
        // Path to gpu-attestor binary
        let gpu_attestor_path = std::env::var("GPU_ATTESTOR_PATH")
            .unwrap_or_else(|_| "../../../target/debug/gpu-attestor".to_string());
        
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
    let challenge_b64 = base64::encode(&challenge_json);
    
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
    let gpu_result: GpuAttestorResult = serde_json::from_str(&stdout)
        .context("Failed to parse challenge result")?;
    
    info!("Miner: Challenge completed successfully");
    info!("  - Checksum: {}", &gpu_result.result_checksum);
    info!("  - Execution time: {} ms", gpu_result.execution_time_ms);
    info!("  - VRAM used: {} MB", gpu_result.vram_allocated_mb);
    
    // Convert to protobuf ChallengeResult
    let result = ChallengeResult {
        solution: gpu_result.solution,
        execution_time_ms: gpu_result.execution_time_ms,
        gpu_utilization: gpu_result.gpu_utilization.into_iter().map(|v| v as f64).collect(),
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

/// The main end-to-end test
#[tokio::test]
async fn gpu_pow_end_to_end_flow() -> Result<()> {
    // Initialize logging
    tracing_subscriber::fmt()
        .with_env_filter("info")
        .with_test_writer()
        .init();
    
    info!("=== GPU PoW End-to-End Test Starting ===");
    
    // Load test configuration
    let config = TestConfig::new()?;
    
    // Step 1: Initialize validator
    info!("\n--- Step 1: Validator Initialization ---");
    let mut gpu_validator = GpuValidator::new(config.gpu_attestor_path.clone());
    
    // Initialize validator's GPU detection
    gpu_validator.initialize().await
        .context("Failed to initialize validator GPU")?;
    
    let validator_gpu = gpu_validator.get_gpu_model()
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
    info!("  - Matrix dimension: {}x{}", challenge.matrix_dim, challenge.matrix_dim);
    info!("  - Number of matrices: {}", challenge.num_matrices);
    info!("  - Matrix A index: {}", challenge.matrix_a_index);
    info!("  - Matrix B index: {}", challenge.matrix_b_index);
    info!("  - Validator nonce: {}", challenge.validator_nonce);
    
    // Step 3: Miner executes challenge
    info!("\n--- Step 3: Miner Execution ---");
    let miner_result = simulate_miner_executor(&config, &challenge).await?;
    
    // Step 4: Validator verifies result
    info!("\n--- Step 4: Validator Verification ---");
    let verification_start = std::time::Instant::now();
    
    let is_valid = gpu_validator.verify_challenge(&challenge, &miner_result)
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
    
    let should_fail = gpu_validator.verify_challenge(&challenge, &bad_result).await?;
    assert!(!should_fail, "Should reject wrong checksum");
    info!("✅ Correctly rejected wrong checksum");
    
    // Test 5b: Wrong GPU model
    info!("Testing wrong GPU model...");
    let mut bad_result = miner_result.clone();
    bad_result.gpu_model = "NVIDIA RTX 4090".to_string();
    
    let should_fail = gpu_validator.verify_challenge(&challenge, &bad_result).await?;
    assert!(!should_fail, "Should reject wrong GPU model");
    info!("✅ Correctly rejected wrong GPU model");
    
    // Test 5c: Wrong nonce
    info!("Testing wrong nonce...");
    let mut bad_result = miner_result.clone();
    bad_result.challenge_id = "wrong_nonce".to_string();
    
    let should_fail = gpu_validator.verify_challenge(&challenge, &bad_result).await?;
    assert!(!should_fail, "Should reject wrong nonce");
    info!("✅ Correctly rejected wrong nonce");
    
    // Test 5d: Execution time too fast (suspicious)
    info!("Testing suspiciously fast execution time...");
    let mut bad_result = miner_result.clone();
    bad_result.execution_time_ms = 1; // 1ms is impossibly fast
    
    let should_fail = gpu_validator.verify_challenge(&challenge, &bad_result).await?;
    assert!(!should_fail, "Should reject suspiciously fast execution");
    info!("✅ Correctly rejected suspiciously fast execution");
    
    info!("\n=== GPU PoW End-to-End Test PASSED ===");
    
    Ok(())
}

/// Test multiple challenges in sequence
#[tokio::test]
async fn gpu_pow_multiple_challenges() -> Result<()> {
    tracing_subscriber::fmt()
        .with_env_filter("info")
        .with_test_writer()
        .init();
    
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
            Some(format!("multi_test_{}", i)),
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
    tracing_subscriber::fmt()
        .with_env_filter("info")
        .with_test_writer()
        .init();
    
    let config = TestConfig::new()?;
    
    info!("Testing concurrent challenge generation and verification...");
    
    // Generate multiple challenges
    let challenge_gen = ChallengeGenerator::new();
    let mut challenges = Vec::new();
    
    for i in 0..3 {
        let challenge = challenge_gen.generate_challenge(
            &config.validator_gpu_model,
            config.validator_vram_gb,
            Some(format!("concurrent_test_{}", i)),
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
            let challenge_b64 = base64::encode(&challenge_json);
            
            let output = tokio::process::Command::new(&config)
                .arg("--challenge")
                .arg(&challenge_b64)
                .output()
                .await?;
            
            if !output.status.success() {
                anyhow::bail!("Challenge {} execution failed", i);
            }
            
            let result: ChallengeResult = serde_json::from_slice(&output.stdout)?;
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
        assert!(is_valid, "Concurrent challenge {} verification failed", idx);
        info!("✅ Concurrent challenge {} verified", idx);
    }
    
    info!("\n=== Concurrent Challenges Test PASSED ===");
    Ok(())
}