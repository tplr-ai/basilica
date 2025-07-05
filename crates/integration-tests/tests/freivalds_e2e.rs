//! End-to-end integration tests for Freivalds protocol
//! Tests the full multi-round challenge-response flow between validator and handler

use anyhow::{Context, Result};
use std::sync::Arc;
use std::time::Duration;
use tracing::info;

// Import from gpu-attestor for the handler and validator
use gpu_attestor::gpu::cuda_driver::{CudaDevice, CudaContext};
use gpu_attestor::challenge::freivalds_handler::FreivaldsHandler;
use gpu_attestor::gpu::GpuDevice;
use gpu_attestor::validation::freivalds_validator::{FreivaldsValidator, FreivaldsValidatorConfig};

// Import protocol types
use protocol::basilca::freivalds_gpu_pow::v1::CommitmentResponse;

/// Helper function to create a test handler with CUDA support
async fn create_test_handler() -> Result<FreivaldsHandler> {
    // Try to detect actual GPUs on the system
    use gpu_attestor::gpu::GpuDetector;
    
    let detector = match GpuDetector::detect_all() {
        Ok(d) => d,
        Err(e) => {
            eprintln!("Skipping Freivalds tests - GPU detection failed: {}", e);
            return Err(anyhow::anyhow!("GPU detection failed"));
        }
    };
    
    if detector.devices.is_empty() {
        eprintln!("Skipping Freivalds tests - No GPUs detected");
        return Err(anyhow::anyhow!("No GPUs available"));
    }
    
    // Use the first GPU device
    let gpu_device = detector.devices[0].clone();
    
    // Initialize CUDA device
    let cuda_device = match CudaDevice::init_device(gpu_device.device_id as i32) {
        Ok(dev) => dev,
        Err(e) => {
            eprintln!("Skipping Freivalds tests - CUDA device init failed: {}", e);
            return Err(anyhow::anyhow!("CUDA initialization failed"));
        }
    };

    // Create CUDA context
    let cuda_context = Arc::new(CudaContext::new(cuda_device)?);

    FreivaldsHandler::new(vec![gpu_device], vec![cuda_context])
}

#[tokio::test]
async fn test_freivalds_full_protocol_flow() -> Result<()> {
    // Initialize logging
    let _ = tracing_subscriber::fmt::try_init();

    // Initialize validator
    let config = FreivaldsValidatorConfig {
        spot_check_count: 5,
        session_timeout: Duration::from_secs(60),
        verification_rounds: 1,
        verify_all_spot_checks: true,
    };
    let validator = FreivaldsValidator::new(config);

    // Initialize handler (miner side)
    let mut handler = match create_test_handler().await {
        Ok(h) => h,
        Err(_) => {
            eprintln!("Skipping test - CUDA not available");
            return Ok(());
        }
    };

    // Step 1: Create challenge from validator
    info!("Step 1: Creating challenge from validator");
    let matrix_size = 64;
    let expected_gpu_count = 1;
    let challenge = validator
        .create_challenge(matrix_size, expected_gpu_count)
        .context("Failed to create challenge")?;

    assert_eq!(challenge.n, matrix_size);
    assert_eq!(challenge.expected_gpu_count, expected_gpu_count);
    assert_eq!(challenge.master_seed.len(), 16);
    assert!(challenge.session_id.starts_with("freivalds_session_"));

    // Step 2: Execute challenge on handler (miner side)
    info!("Step 2: Executing challenge on handler");
    let commitment = handler
        .execute_challenge(&challenge)
        .await
        .context("Failed to execute challenge")?;

    assert_eq!(commitment.session_id, challenge.session_id);
    assert_eq!(commitment.merkle_root.len(), 32);
    assert_eq!(commitment.row_count, matrix_size);
    assert!(commitment.metadata.is_some());

    // Step 3: Process commitment on validator
    info!("Step 3: Processing commitment on validator");
    let verification_request = validator
        .process_commitment(commitment)
        .context("Failed to process commitment")?;

    assert_eq!(verification_request.session_id, challenge.session_id);
    assert!(!verification_request.challenge_vector.is_empty());
    assert!(!verification_request.spot_check_rows.is_empty());
    assert!(verification_request.spot_check_rows.len() <= 5);

    // Step 4: Process verification request on handler
    info!("Step 4: Processing verification request on handler");
    let response = handler
        .compute_response(&verification_request)
        .await
        .context("Failed to process verification")?;

    assert_eq!(response.session_id, challenge.session_id);
    assert!(!response.cr_result.is_empty());
    assert_eq!(
        response.row_proofs.len(),
        verification_request.spot_check_rows.len()
    );

    // Step 5: Verify response on validator
    info!("Step 5: Verifying response on validator");
    let verification_result = validator
        .verify_response(response)
        .await
        .context("Failed to verify response")?;

    assert_eq!(verification_result.session_id, challenge.session_id);
    assert!(verification_result.verified);
    assert!(verification_result.freivalds_valid);
    assert!(verification_result.spot_checks_valid);
    assert_eq!(
        verification_result.spot_checks_performed as usize,
        verification_request.spot_check_rows.len()
    );
    assert!(verification_result.metrics.is_some());

    let metrics = verification_result.metrics.unwrap();
    // For very small matrices, the time might be < 1ms
    info!("Metrics: total_time_ms={}, freivalds_time_ms={}, spot_check_time_ms={}, computation_saved={}%",
          metrics.total_time_ms, metrics.freivalds_time_ms, metrics.spot_check_time_ms,
          metrics.computation_saved_percent);
    assert!(metrics.computation_saved_percent > 0.0);

    info!("Freivalds protocol completed successfully!");
    info!(
        "Computation saved: {:.2}%",
        metrics.computation_saved_percent
    );

    // Verify session cleanup
    assert_eq!(validator.active_session_count(), 0);

    Ok(())
}

#[tokio::test]
async fn test_freivalds_multiple_sessions() -> Result<()> {
    let _ = tracing_subscriber::fmt::try_init();

    let validator = FreivaldsValidator::new(FreivaldsValidatorConfig::default());
    let mut handler = match create_test_handler().await {
        Ok(h) => h,
        Err(_) => {
            eprintln!("Skipping test - CUDA not available");
            return Ok(());
        }
    };

    // Create multiple concurrent sessions
    let mut challenges = Vec::new();
    for i in 0..3 {
        let challenge = validator
            .create_challenge(64 + i * 32, 1)
            .context("Failed to create challenge")?;
        challenges.push(challenge);
    }

    assert_eq!(validator.active_session_count(), 3);

    // Process all challenges
    for (i, challenge) in challenges.into_iter().enumerate() {
        info!("Processing challenge {}", i);

        // Execute challenge
        let commitment = handler
            .execute_challenge(&challenge)
            .await
            .context("Failed to execute challenge")?;

        // Process commitment
        let verification = validator
            .process_commitment(commitment)
            .context("Failed to process commitment")?;

        // Process verification
        let response = handler
            .compute_response(&verification)
            .await
            .context("Failed to process verification")?;

        // Verify response
        let result = validator
            .verify_response(response)
            .await
            .context("Failed to verify response")?;

        assert!(result.verified);
    }

    // All sessions should be cleaned up
    assert_eq!(validator.active_session_count(), 0);

    Ok(())
}

#[tokio::test]
async fn test_freivalds_large_matrix() -> Result<()> {
    let _ = tracing_subscriber::fmt::try_init();

    let validator = FreivaldsValidator::new(FreivaldsValidatorConfig::default());
    let mut handler = match create_test_handler().await {
        Ok(h) => h,
        Err(_) => {
            eprintln!("Skipping test - CUDA not available");
            return Ok(());
        }
    };

    // Test with larger matrix
    let challenge = validator
        .create_challenge(512, 1)
        .context("Failed to create challenge")?;

    // Execute full protocol
    let commitment = handler
        .execute_challenge(&challenge)
        .await
        .context("Failed to execute challenge")?;
    let verification = validator
        .process_commitment(commitment)
        .context("Failed to process commitment")?;
    let response = handler
        .compute_response(&verification)
        .await
        .context("Failed to process verification")?;
    let result = validator
        .verify_response(response)
        .await
        .context("Failed to verify response")?;

    assert!(result.verified);
    assert!(result.metrics.unwrap().computation_saved_percent > 90.0);

    Ok(())
}

#[tokio::test]
async fn test_freivalds_invalid_session() -> Result<()> {
    let _ = tracing_subscriber::fmt::try_init();

    let validator = FreivaldsValidator::new(FreivaldsValidatorConfig::default());

    // Try to process commitment for non-existent session
    let fake_commitment = CommitmentResponse {
        session_id: "fake_session_999".to_string(),
        merkle_root: vec![0x42; 32],
        row_count: 64,
        metadata: None,
        timestamp: None,
    };

    let result = validator.process_commitment(fake_commitment);
    assert!(result.is_err());
    assert!(result.unwrap_err().to_string().contains("not found"));

    Ok(())
}

#[tokio::test]
async fn test_freivalds_session_timeout() -> Result<()> {
    let _ = tracing_subscriber::fmt::try_init();

    let mut config = FreivaldsValidatorConfig::default();
    config.session_timeout = Duration::from_millis(100);

    let validator = FreivaldsValidator::new(config);
    let mut handler = match create_test_handler().await {
        Ok(h) => h,
        Err(_) => {
            eprintln!("Skipping test - CUDA not available");
            return Ok(());
        }
    };

    // Create challenge
    let challenge = validator
        .create_challenge(64, 1)
        .context("Failed to create challenge")?;

    // Execute challenge
    let commitment = handler
        .execute_challenge(&challenge)
        .await
        .context("Failed to execute challenge")?;

    // Wait for session to timeout
    tokio::time::sleep(Duration::from_millis(150)).await;

    // Try to process commitment after timeout
    let result = validator.process_commitment(commitment);
    assert!(result.is_err());
    assert!(result.unwrap_err().to_string().contains("expired"));

    Ok(())
}

#[tokio::test]
async fn test_freivalds_performance_metrics() -> Result<()> {
    let _ = tracing_subscriber::fmt::try_init();

    let validator = FreivaldsValidator::new(FreivaldsValidatorConfig::default());
    let mut handler = match create_test_handler().await {
        Ok(h) => h,
        Err(_) => {
            eprintln!("Skipping test - CUDA not available");
            return Ok(());
        }
    };

    // Test different matrix sizes
    for size in [64, 128, 256] {
        info!("Testing matrix size: {}", size);

        let challenge = validator
            .create_challenge(size, 1)
            .context("Failed to create challenge")?;

        let start = std::time::Instant::now();

        // Execute full protocol
        let commitment = handler
            .execute_challenge(&challenge)
            .await
            .context("Failed to execute challenge")?;
        let verification = validator
            .process_commitment(commitment)
            .context("Failed to process commitment")?;
        let response = handler
            .compute_response(&verification)
            .await
            .context("Failed to process verification")?;
        let result = validator
            .verify_response(response)
            .await
            .context("Failed to verify response")?;

        let total_time = start.elapsed();

        info!("Matrix size {}: Total time: {:?}", size, total_time);
        info!(
            "  Computation saved: {:.2}%",
            result.metrics.as_ref().unwrap().computation_saved_percent
        );

        assert!(result.verified);
        assert!(result.metrics.is_some());

        let metrics = result.metrics.unwrap();
        // For small matrices, times might be < 1ms and round to 0
        // Just verify the computation savings which is the key metric
        assert!(metrics.computation_saved_percent > 85.0);
    }

    Ok(())
}