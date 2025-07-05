//! Freivalds GPU Attestation Same Machine End-to-End Test
//!
//! This test simulates the complete Freivalds protocol flow where the miner
//! and validator are on the same machine (similar to gpu_pow_e2e.rs):
//! 1. Validator creates a Freivalds challenge
//! 2. Miner (executor) computes matrix multiplication and generates commitment
//! 3. Validator sends verification request
//! 4. Miner responds with proof
//! 5. Validator verifies the response

use anyhow::{Context, Result};
use std::sync::Arc;
use std::time::Instant;
use tracing::{info, warn};

use gpu_attestor::challenge::freivalds_handler::FreivaldsHandler;
use gpu_attestor::gpu::{
    cuda_driver::{CudaContext, CudaDevice},
    GpuDetector,
};
use gpu_attestor::validation::freivalds_validator::{FreivaldsValidator, FreivaldsValidatorConfig};

/// Test configuration
struct TestConfig {
    matrix_sizes: Vec<u32>,
    spot_check_count: u32,
    expected_gpu_count: usize,
}

impl TestConfig {
    fn new() -> Self {
        Self {
            matrix_sizes: vec![64, 128, 256, 512],
            spot_check_count: 10,
            expected_gpu_count: std::env::var("GPU_COUNT")
                .unwrap_or_else(|_| "0".to_string())
                .parse()
                .unwrap_or(0),
        }
    }
}

/// Create a test handler with all available GPUs
async fn create_multi_gpu_handler() -> Result<(FreivaldsHandler, usize)> {
    info!("Detecting GPUs on the system...");

    let detector = GpuDetector::detect_all().context("Failed to detect GPUs")?;

    if detector.devices.is_empty() {
        anyhow::bail!("No GPUs detected on the system");
    }

    info!("Found {} GPU(s):", detector.devices.len());
    for device in &detector.devices {
        info!(
            "  - Device {}: {} ({:.0} MB)",
            device.device_id,
            device.name,
            device.memory_total / (1024 * 1024)
        );
    }

    // Initialize CUDA contexts for ALL GPUs
    let mut cuda_contexts = Vec::new();

    for device in &detector.devices {
        let cuda_device = CudaDevice::init_device(device.device_id as i32)?;
        let context = Arc::new(CudaContext::new(cuda_device)?);
        cuda_contexts.push(context);
    }

    let gpu_count = detector.devices.len();
    let handler = FreivaldsHandler::new(detector.devices, cuda_contexts)?;

    Ok((handler, gpu_count))
}

/// Main test that runs Freivalds on same machine with all GPUs
#[tokio::test]
async fn freivalds_same_machine_flow() -> Result<()> {
    // Initialize logging
    let _ = tracing_subscriber::fmt()
        .with_env_filter("info")
        .with_test_writer()
        .try_init();

    info!("=== Freivalds Same Machine End-to-End Test ===");

    // Step 0: Detect and initialize GPUs
    info!("\n--- Step 0: GPU Detection ---");
    let (mut handler, gpu_count) = match create_multi_gpu_handler().await {
        Ok((h, c)) => (h, c),
        Err(e) => {
            eprintln!("Skipping test - GPU initialization failed: {e}");
            return Ok(());
        }
    };

    info!("Initialized with {} GPU(s)", gpu_count);

    // Step 1: Initialize validator
    info!("\n--- Step 1: Validator Initialization ---");
    let validator_config = FreivaldsValidatorConfig {
        spot_check_count: 10,
        session_timeout: std::time::Duration::from_secs(60),
        verification_rounds: 1,
        verify_all_spot_checks: true,
        network_latency_ms: 0, // Same machine, no network latency
        timeout_overhead_factor: 2.0,
        enable_concurrent_verification: true,
        max_verification_threads: num_cpus::get().min(8),
    };

    let validator = FreivaldsValidator::new(validator_config);
    info!("Validator configured with concurrent verification");

    // Step 2: Test different matrix sizes
    info!("\n--- Step 2: Running Freivalds Protocol Tests ---");

    let config = TestConfig::new();

    for &matrix_size in &config.matrix_sizes {
        info!("\n{}", "=".repeat(60));
        info!("Testing {}×{} matrices", matrix_size, matrix_size);
        info!("{}", "=".repeat(60));

        let test_start = Instant::now();

        // Create challenge
        info!("\nPhase 1: Creating challenge");
        let challenge = validator.create_challenge(matrix_size, gpu_count as u32)?;

        info!("  Session ID: {}", challenge.session_id);
        info!("  Matrix size: {}×{}", challenge.n, challenge.n);
        info!("  Expected GPUs: {}", challenge.expected_gpu_count);

        // Execute challenge
        info!("\nPhase 2: Executing challenge (computing C = A × B)");
        let exec_start = Instant::now();
        let commitment = handler.execute_challenge(&challenge).await?;
        let exec_time = exec_start.elapsed();

        info!("  Execution time: {:.2}ms", exec_time.as_millis());
        info!(
            "  Merkle root: {}",
            hex::encode(&commitment.merkle_root[..8])
        );

        if let Some(ref metadata) = commitment.metadata {
            info!("  Kernel time: {}ms", metadata.kernel_time_ms);
            info!("  Merkle time: {}ms", metadata.merkle_time_ms);
            info!("  GPUs used: {}", metadata.gpus.len());
        }

        // Process commitment
        info!("\nPhase 3: Processing commitment");
        let verification = validator.process_commitment(commitment)?;

        info!("  Spot checks: {}", verification.spot_check_rows.len());

        // Compute response
        info!("\nPhase 4: Computing response");
        let resp_start = Instant::now();
        let response = handler.compute_response(&verification).await?;
        let resp_time = resp_start.elapsed();

        info!("  Response time: {:.2}ms", resp_time.as_millis());
        info!("  Row proofs: {}", response.row_proofs.len());

        // Verify response
        info!("\nPhase 5: Verifying response");
        let verify_start = Instant::now();
        let result = validator.verify_response(response).await?;
        let verify_time = verify_start.elapsed();

        info!("  Verification time: {:.2}ms", verify_time.as_millis());
        info!(
            "  Result: {}",
            if result.verified {
                "VALID ✓"
            } else {
                "INVALID ✗"
            }
        );

        assert!(result.verified, "Verification should succeed");

        // Performance analysis
        let total_time = test_start.elapsed();
        let gpu_work = exec_time.as_millis() as f64;
        let validator_work = verify_time.as_millis() as f64;

        info!("\n--- Performance Summary ---");
        info!("  Total protocol time: {:.2}ms", total_time.as_millis());
        info!("  GPU computation (O(n³)): {:.2}ms", gpu_work);
        info!("  Validator verification (O(n²)): {:.2}ms", validator_work);

        // Calculate theoretical vs actual advantage
        let n = matrix_size as f64;
        let theoretical_advantage = n; // O(n³) / O(n²) = n
        let actual_advantage = if validator_work > 0.0 {
            gpu_work / validator_work
        } else {
            1.0
        };

        info!("\n--- Asymmetric Verification Analysis ---");
        info!("  Matrix size: {}", matrix_size);
        info!(
            "  Theoretical advantage: {:.0}x (n = {})",
            theoretical_advantage, matrix_size
        );
        info!("  Actual advantage: {:.1}x", actual_advantage);
        info!(
            "  Efficiency: {:.1}%",
            (actual_advantage / theoretical_advantage) * 100.0
        );

        if let Some(metrics) = result.metrics {
            info!(
                "  Computation saved by validator: {:.1}%",
                metrics.computation_saved_percent
            );
            info!("  Spot checks verified: {}", result.spot_checks_performed);
        }
    }

    info!("\n=== Freivalds Same Machine Test PASSED ===");

    Ok(())
}

/// Test that concurrent verification correctly handles all spot checks
#[tokio::test]
async fn freivalds_concurrent_verification_correctness_test() -> Result<()> {
    let _ = tracing_subscriber::fmt()
        .with_env_filter("info")
        .with_test_writer()
        .try_init();

    info!("=== Freivalds Concurrent Verification Correctness Test ===");

    // Initialize components
    let (mut handler, gpu_count) = match create_multi_gpu_handler().await {
        Ok((h, c)) => (h, c),
        Err(e) => {
            eprintln!("Skipping test - GPU initialization failed: {e}");
            return Ok(());
        }
    };

    info!("Testing with {} GPU(s)", gpu_count);

    // Create validator with concurrent verification
    let config = FreivaldsValidatorConfig {
        enable_concurrent_verification: true,
        max_verification_threads: num_cpus::get().min(8),
        ..Default::default()
    };

    let validator = FreivaldsValidator::new(config);

    // Test with different matrix sizes and spot check counts
    let test_cases = vec![
        (256, 10),  // Small matrix, few checks
        (256, 50),  // Small matrix, many checks
        (512, 20),  // Medium matrix, moderate checks
        (512, 100), // Medium matrix, many checks
    ];

    info!("\nVerifying correctness with various configurations:");
    info!(
        "{:<15} {:<20} {:<15} {:<15}",
        "Matrix Size", "Spot Checks (Req/Act)", "All Valid", "Time (ms)"
    );
    info!("{:-<65}", "");

    for (matrix_size, spot_checks) in test_cases {
        // Create challenge
        let challenge = validator.create_challenge(matrix_size, gpu_count as u32)?;

        // Execute challenge
        let commitment = handler.execute_challenge(&challenge).await?;
        validator.store_commitment(&challenge.session_id, &commitment)?;

        // Create verification with specific spot check count
        let verification = validator.create_verification(&challenge.session_id, spot_checks)?;

        // Generate response
        let response = handler.compute_response(&verification).await?;

        // Verify with concurrent verification
        let start = Instant::now();
        let result = validator.verify_response(response).await?;
        let elapsed = start.elapsed();

        // Check that verification succeeded and spot checks were performed
        assert!(result.verified, "Verification should succeed");

        // Note: Due to deduplication, we might get fewer spot checks than requested
        assert!(
            result.spot_checks_performed > 0 && result.spot_checks_performed <= spot_checks,
            "Should perform between 1 and {} spot checks, got {}",
            spot_checks,
            result.spot_checks_performed
        );

        info!(
            "{:<15} {:<20} {:<15} {:<15.2}",
            matrix_size,
            format!("{}/{}", spot_checks, result.spot_checks_performed),
            if result.verified { "✓" } else { "✗" },
            elapsed.as_millis()
        );
    }

    info!("\n✓ All concurrent verifications completed correctly");

    Ok(())
}

/// Test Freivalds with multiple GPUs
#[tokio::test]
async fn freivalds_multi_gpu_test() -> Result<()> {
    let _ = tracing_subscriber::fmt()
        .with_env_filter("info")
        .with_test_writer()
        .try_init();

    info!("=== Freivalds Multi-GPU Test ===");

    let (mut handler, gpu_count) = match create_multi_gpu_handler().await {
        Ok((h, c)) => (h, c),
        Err(e) => {
            eprintln!("Skipping test - GPU initialization failed: {e}");
            return Ok(());
        }
    };

    if gpu_count < 2 {
        warn!(
            "Test requires at least 2 GPUs, found {}. Skipping.",
            gpu_count
        );
        return Ok(());
    }

    info!("Testing with {} GPUs", gpu_count);

    let validator = FreivaldsValidator::new(FreivaldsValidatorConfig::default());

    // Test large matrix that benefits from multi-GPU
    let matrix_size = 1024;
    let challenge = validator.create_challenge(matrix_size, gpu_count as u32)?;

    info!(
        "\nExecuting {}×{} matrix multiplication on {} GPUs...",
        matrix_size, matrix_size, gpu_count
    );

    let start = Instant::now();
    let commitment = handler.execute_challenge(&challenge).await?;
    let elapsed = start.elapsed();

    info!(
        "Multi-GPU execution completed in {:.2}ms",
        elapsed.as_millis()
    );

    // Verify the computation used multiple GPUs
    if let Some(metadata) = &commitment.metadata {
        assert_eq!(
            metadata.gpus.len(),
            gpu_count,
            "Should use all available GPUs"
        );

        info!("\nGPU utilization:");
        for gpu in &metadata.gpus {
            info!(
                "  - GPU {}: {} ({} MB used)",
                gpu.device_id, gpu.model, gpu.vram_mb
            );
        }
    }

    // Complete the protocol
    let verification = validator.process_commitment(commitment)?;
    let response = handler.compute_response(&verification).await?;
    let result = validator.verify_response(response).await?;

    assert!(result.verified, "Multi-GPU verification should succeed");

    info!("\n✓ Multi-GPU test passed!");

    Ok(())
}
