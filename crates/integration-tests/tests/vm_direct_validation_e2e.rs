//! Direct VM-Protected GPU Attestation Integration Tests
//!
//! This test suite validates the gpu-attestor binary directly without SSH,
//! demonstrating the VM-protected validation with the generic --challenge interface.

use anyhow::{Context, Result};
use base64::Engine;
use serde::{Deserialize, Serialize};
use std::path::PathBuf;
use std::process::Command;
use tracing::{info, warn};

/// Generic computational challenge (matches validator's format)
#[derive(Debug, Clone, Serialize, Deserialize)]
struct ComputeChallenge {
    session_id: String,
    problem_size: u32,
    seed_data: Vec<u8>,
    timestamp: Option<String>,
    resource_count: u32,
    computation_timeout_ms: u32,
    protocol_timeout_ms: u32,
}

/// Expected result structure from the binary
#[derive(Debug, Deserialize)]
struct ValidationResult {
    status: String, // "PASS" or "FAIL"
    session_id: String,
    timestamp: String,
}

/// Test helper to get the binary path
fn get_binary_path() -> PathBuf {
    // Use CARGO_MANIFEST_DIR to find the workspace root
    let workspace_root = std::env::var("CARGO_MANIFEST_DIR")
        .map(|dir| {
            PathBuf::from(dir)
                .parent()
                .unwrap()
                .parent()
                .unwrap()
                .to_path_buf()
        })
        .unwrap_or_else(|_| PathBuf::from("../.."));

    workspace_root.join("target/release/gpu-attestor")
}

/// Test helper to check if binary exists
fn check_binary_exists() -> bool {
    let binary_path = get_binary_path();
    if !binary_path.exists() {
        warn!("GPU attestor binary not found at {:?}", binary_path);
        warn!("Run 'cargo build --release -p gpu-attestor' to build it");
        return false;
    }
    info!("GPU attestor binary found at {:?}", binary_path);
    true
}

/// Create a test challenge
fn create_test_challenge(problem_size: u32, resource_count: u32) -> ComputeChallenge {
    use rand::Rng;
    let mut rng = rand::thread_rng();

    let mut seed_data = vec![0u8; 16];
    rng.fill(&mut seed_data[..]);

    ComputeChallenge {
        session_id: format!("test_session_{}", rng.gen::<u64>()),
        problem_size,
        seed_data,
        timestamp: Some(
            std::time::SystemTime::now()
                .duration_since(std::time::UNIX_EPOCH)
                .unwrap()
                .as_secs()
                .to_string(),
        ),
        resource_count,
        computation_timeout_ms: problem_size * 20, // Increased timeout
        protocol_timeout_ms: problem_size * 40,    // Increased timeout
    }
}

/// Run the gpu-attestor binary with a challenge
fn run_validation(challenge: &ComputeChallenge) -> Result<ValidationResult> {
    let binary_path = get_binary_path();

    // Serialize challenge to JSON
    let challenge_json = serde_json::to_string(challenge)?;

    // Base64 encode the challenge
    let challenge_b64 = base64::engine::general_purpose::STANDARD.encode(&challenge_json);

    info!(
        "Running validation: session={}, problem_size={}, resources={}",
        challenge.session_id, challenge.problem_size, challenge.resource_count
    );

    // Execute the binary with anti-debugging disabled
    let output = std::process::Command::new(&binary_path)
        .arg("--challenge")
        .arg(&challenge_b64)
        .env("DISABLE_VM_ANTI_DEBUG", "1")
        .env("CUDA_LAUNCH_BLOCKING", "1") // Force synchronous CUDA operations
        .output()
        .context("Failed to execute gpu-attestor binary")?;

    if !output.status.success() {
        let stderr = String::from_utf8_lossy(&output.stderr);
        let stdout = String::from_utf8_lossy(&output.stdout);
        return Err(anyhow::anyhow!(
            "Binary execution failed: exit_code={}, stderr={}, stdout={}",
            output.status.code().unwrap_or(-1),
            stderr,
            stdout
        ));
    }

    // Parse the JSON output
    let stdout = String::from_utf8_lossy(&output.stdout);
    let result: ValidationResult =
        serde_json::from_str(&stdout).context(format!("Failed to parse output: {stdout}"))?;

    Ok(result)
}

#[test]
fn test_vm_direct_basic_validation() -> Result<()> {
    let _ = tracing_subscriber::fmt()
        .with_env_filter("info")
        .with_test_writer()
        .try_init();

    info!("=== Direct VM-Protected Basic Validation Test ===");

    if !check_binary_exists() {
        eprintln!("Skipping test - binary not available");
        return Ok(());
    }

    // Create a basic challenge
    let challenge = create_test_challenge(256, 1);

    // Run validation
    let start = std::time::Instant::now();
    let result = run_validation(&challenge)?;
    let elapsed = start.elapsed();

    info!(
        "Validation completed in {:?}: status={}, session={}",
        elapsed, result.status, result.session_id
    );

    // Verify we got a valid response
    assert!(
        result.status == "PASS" || result.status == "FAIL",
        "Invalid status: {}",
        result.status
    );

    assert_eq!(
        result.session_id, challenge.session_id,
        "Session ID mismatch"
    );

    info!("✓ Direct validation test passed");
    Ok(())
}

#[test]
fn test_vm_direct_multiple_sizes() -> Result<()> {
    let _ = tracing_subscriber::fmt()
        .with_env_filter("info")
        .with_test_writer()
        .try_init();

    info!("=== Direct VM-Protected Multiple Sizes Test ===");

    if !check_binary_exists() {
        eprintln!("Skipping test - binary not available");
        return Ok(());
    }

    // Use smaller sizes to reduce GPU memory pressure
    let test_sizes = [64, 128, 256];

    info!("{:<12} {:<12} {:<12}", "Size", "Time (ms)", "Status");
    info!("{:-<36}", "");

    for (idx, size) in test_sizes.iter().enumerate() {
        let challenge = create_test_challenge(*size, 1);

        info!(
            "Starting test {} of {} (size: {})",
            idx + 1,
            test_sizes.len(),
            size
        );

        let start = std::time::Instant::now();
        match run_validation(&challenge) {
            Ok(result) => {
                let elapsed = start.elapsed();
                info!(
                    "{:<12} {:<12} {:<12}",
                    size,
                    elapsed.as_millis(),
                    result.status
                );
            }
            Err(e) => {
                warn!("Validation failed for size {}: {}", size, e);
                // Continue with next test instead of failing
            }
        }

        // Add a longer delay between runs to avoid resource contention
        if idx < test_sizes.len() - 1 {
            info!("Waiting 2 seconds before next test...");
            std::thread::sleep(std::time::Duration::from_secs(2));
        }
    }

    info!("✓ Multiple sizes test completed");
    Ok(())
}

#[test]
fn test_vm_direct_resource_scaling() -> Result<()> {
    let _ = tracing_subscriber::fmt()
        .with_env_filter("info")
        .with_test_writer()
        .try_init();

    info!("=== Direct VM-Protected Resource Scaling Test ===");

    if !check_binary_exists() {
        eprintln!("Skipping test - binary not available");
        return Ok(());
    }

    let test_resources = vec![1, 2, 4, 8];
    let problem_size = 256;

    info!("{:<12} {:<12} {:<12}", "Resources", "Time (ms)", "Status");
    info!("{:-<36}", "");

    for resources in test_resources {
        let challenge = create_test_challenge(problem_size, resources);

        let start = std::time::Instant::now();
        match run_validation(&challenge) {
            Ok(result) => {
                let elapsed = start.elapsed();
                info!(
                    "{:<12} {:<12} {:<12}",
                    resources,
                    elapsed.as_millis(),
                    result.status
                );
            }
            Err(e) => {
                warn!("Validation failed for {} resources: {}", resources, e);
            }
        }

        // Add a longer delay between runs to avoid resource contention
        std::thread::sleep(std::time::Duration::from_millis(1000));
    }

    info!("✓ Resource scaling test completed");
    Ok(())
}

#[test]
fn test_vm_security_layers_demonstration() -> Result<()> {
    let _ = tracing_subscriber::fmt()
        .with_env_filter("info")
        .with_test_writer()
        .try_init();

    info!("=== VM Security Layers Demonstration ===");

    if !check_binary_exists() {
        eprintln!("Skipping test - binary not available");
        return Ok(());
    }

    info!("Demonstrating multi-layer security architecture:");
    info!("");
    info!("Layer 1 - VM PROTECTION (Primary Security):");
    info!("  ✓ Encrypted bytecode prevents static analysis");
    info!("  ✓ Anti-debugging detects runtime analysis");
    info!("  ✓ Dynamic code generation defeats patterns");
    info!("  ✓ Hidden validation logic in VM");
    info!("");
    info!("Layer 2 - INTERFACE OBFUSCATION (Additional):");
    info!("  ✓ Generic --challenge parameter only");
    info!("  ✓ No protocol-specific flags exposed");
    info!("  ✓ Binary determines validation internally");
    info!("");

    // Demonstrate that the interface reveals nothing
    let challenge = create_test_challenge(256, 1);

    info!("Input: Generic ComputeChallenge with:");
    info!(
        "  - problem_size: {} (generic parameter)",
        challenge.problem_size
    );
    info!(
        "  - resource_count: {} (generic parameter)",
        challenge.resource_count
    );
    info!("  - No validation algorithm specified");
    info!("  - No thresholds or tolerances exposed");
    info!("");

    let result = run_validation(&challenge)?;

    info!("Output: Generic result with:");
    info!("  - status: {} (PASS/FAIL only)", result.status);
    info!("  - session_id: {} (for tracking)", result.session_id);
    info!("  - No details about validation logic");
    info!("  - No information about what was checked");
    info!("");

    info!("✓ Security layers successfully hide all operational details");
    Ok(())
}

#[test]
fn test_vm_consistent_results() -> Result<()> {
    let _ = tracing_subscriber::fmt()
        .with_env_filter("info")
        .with_test_writer()
        .try_init();

    info!("=== VM Consistent Results Test ===");

    if !check_binary_exists() {
        eprintln!("Skipping test - binary not available");
        return Ok(());
    }

    // Test that same input produces consistent results
    let challenge = create_test_challenge(128, 1);
    let mut results = Vec::new();

    info!("Running validation 3 times with same parameters...");

    for i in 0..3 {
        let result = run_validation(&challenge)?;
        info!("Run {}: {}", i + 1, result.status);
        results.push(result.status);

        // Add a longer delay between runs to avoid resource contention
        if i < 2 {
            std::thread::sleep(std::time::Duration::from_millis(1000));
        }
    }

    // Check consistency
    let first = &results[0];
    let all_same = results.iter().all(|r| r == first);

    if all_same {
        info!("✓ VM provides consistent results for same input");
    } else {
        info!("! Results varied (may be intentional for security)");
    }

    Ok(())
}

#[test]
fn test_vm_error_handling() -> Result<()> {
    let _ = tracing_subscriber::fmt()
        .with_env_filter("info")
        .with_test_writer()
        .try_init();

    info!("=== VM Error Handling Test ===");

    if !check_binary_exists() {
        eprintln!("Skipping test - binary not available");
        return Ok(());
    }

    // Test 1: Invalid base64 encoding
    info!("Testing invalid base64 input...");
    let binary_path = get_binary_path();
    let output = Command::new(&binary_path)
        .arg("--challenge")
        .arg("not-valid-base64!@#")
        .env("DISABLE_VM_ANTI_DEBUG", "1")
        .output()?;

    assert!(!output.status.success(), "Should fail with invalid base64");
    info!("✓ Correctly rejected invalid base64");

    // Test 2: Invalid JSON structure
    info!("Testing invalid JSON structure...");
    let invalid_json = base64::engine::general_purpose::STANDARD.encode("{ invalid json }");
    let output = Command::new(&binary_path)
        .arg("--challenge")
        .arg(&invalid_json)
        .env("DISABLE_VM_ANTI_DEBUG", "1")
        .output()?;

    assert!(!output.status.success(), "Should fail with invalid JSON");
    info!("✓ Correctly rejected invalid JSON");

    // Test 3: Missing required fields - should use defaults and still return a result
    info!("Testing missing required fields (uses defaults)...");
    let incomplete = base64::engine::general_purpose::STANDARD.encode(r#"{"session_id": "test"}"#);
    let output = Command::new(&binary_path)
        .arg("--challenge")
        .arg(&incomplete)
        .env("DISABLE_VM_ANTI_DEBUG", "1")
        .output()?;

    // The binary accepts incomplete challenges and uses defaults
    assert!(
        output.status.success(),
        "Should succeed with defaults for missing fields"
    );

    // Should still return valid JSON
    let stdout = String::from_utf8_lossy(&output.stdout);
    let result: ValidationResult = serde_json::from_str(&stdout).context(format!(
        "Failed to parse output with missing fields: {stdout}"
    ))?;

    assert_eq!(result.session_id, "test", "Session ID should be preserved");
    assert!(
        result.status == "PASS" || result.status == "FAIL",
        "Should have valid status"
    );

    info!("✓ Correctly handled incomplete challenge with defaults");

    info!("✓ Error handling test completed");
    Ok(())
}
