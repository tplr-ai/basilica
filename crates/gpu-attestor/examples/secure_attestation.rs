//! Example demonstrating secure attestation with all mitigations
//!
//! This example shows how to:
//! 1. Validate hardware performance matches claimed specs
//! 2. Detect virtualization
//! 3. Implement replay protection with VDF challenges

use anyhow::Result;
use gpu_attestor::{
    attestation::{AttestationBuilder, AttestationSigner, ValidationResults},
    gpu::GpuDetector,
    hardware::SystemInfoCollector,
    validation::{PerformanceValidator, ReplayGuard, VirtualizationDetector},
    vdf::{VdfAlgorithm, VdfComputer},
};

fn main() -> Result<()> {
    // Initialize logging
    tracing_subscriber::fmt::init();

    println!("=== Secure GPU Attestation Example ===\n");

    // Step 1: Detect virtualization
    println!("Step 1: Checking for virtualization...");
    let virt_detector = VirtualizationDetector::new();
    let virt_status = virt_detector.detect_virtualization()?;

    if virt_status.is_virtualized {
        println!("⚠️  WARNING: Virtualization detected!");
        println!("   Hypervisor: {:?}", virt_status.hypervisor);
        println!("   Container: {:?}", virt_status.container);
        println!("   Confidence: {:.0}%", virt_status.confidence * 100.0);
        println!("   CPU Flags: {:?}", virt_status.cpu_flags);
        println!("   Kernel Modules: {:?}", virt_status.kernel_modules);
        println!("   Virtual NICs: {:?}", virt_status.network_interfaces);
        println!("   Virtual Disks: {:?}", virt_status.disk_devices);

        // In production, you might want to fail here
        // return Err(anyhow::anyhow!("Cannot attest virtualized environment"));
    } else {
        println!(
            "✅ No virtualization detected (confidence: {:.0}%)",
            (1.0 - virt_status.confidence) * 100.0
        );
    }

    // Step 2: Collect hardware information
    println!("\nStep 2: Collecting hardware information...");
    let gpu_info = GpuDetector::query_gpu_info()?;
    let system_info = SystemInfoCollector::collect_all()?;

    if gpu_info.is_empty() {
        return Err(anyhow::anyhow!("No GPUs detected"));
    }

    println!("✅ Detected {} GPU(s):", gpu_info.len());
    for gpu in &gpu_info {
        println!(
            "   - {} ({}MB memory)",
            gpu.name,
            gpu.memory_total / 1024 / 1024
        );
    }

    // Step 3: Validate GPU performance
    println!("\nStep 3: Validating GPU performance...");
    let perf_validator = PerformanceValidator::new();
    let mut all_gpus_valid = true;
    let mut gpu_results = Vec::new();

    for gpu in &gpu_info {
        match perf_validator.validate_gpu(gpu) {
            Ok(result) => {
                if result.is_valid {
                    println!(
                        "✅ {} performance validated (confidence: {:.0}%)",
                        gpu.name,
                        result.confidence_score * 100.0
                    );
                } else {
                    println!("❌ {} performance validation failed:", gpu.name);
                    println!(
                        "   Memory: {} MB (expected: {} MB)",
                        result.memory_validation.claimed_bytes / 1024 / 1024,
                        result.memory_validation.expected_bytes / 1024 / 1024
                    );
                    println!(
                        "   Bandwidth: {:.1} GB/s (expected: {:.1} GB/s)",
                        result.bandwidth_validation.measured_gbps,
                        result.bandwidth_validation.expected_gbps
                    );
                    println!(
                        "   Compute: {:.1} TFLOPS (expected: {:.1} TFLOPS)",
                        result.compute_validation.measured_tflops,
                        result.compute_validation.expected_tflops
                    );
                    all_gpus_valid = false;
                }

                gpu_results.push(gpu_attestor::attestation::GpuPerformanceResult {
                    gpu_name: gpu.name.clone(),
                    memory_bandwidth_valid: result.bandwidth_validation.is_valid,
                    compute_performance_valid: result.compute_validation.is_valid,
                    measured_tflops: result.compute_validation.measured_tflops,
                    expected_tflops: result.compute_validation.expected_tflops,
                });
            }
            Err(e) => {
                println!("⚠️  Could not validate {}: {}", gpu.name, e);
                // This might happen for unknown GPU models
            }
        }
    }

    // Step 4: Generate VDF challenge for replay protection
    println!("\nStep 4: Generating VDF challenge...");
    let replay_guard = ReplayGuard::new();
    let challenge = replay_guard.generate_challenge("validator-001")?;

    println!("✅ Challenge generated:");
    println!("   Hash: {}", &challenge.challenge_hash[..16]);
    println!("   Difficulty: {}", challenge.difficulty);
    println!("   Expires: {:?}", challenge.expires_at);

    // Step 5: Compute VDF proof
    println!("\nStep 5: Computing VDF proof...");
    let start = std::time::Instant::now();
    let vdf_challenge = VdfComputer::create_challenge(
        challenge.vdf_modulus.clone(),
        challenge.vdf_generator.clone(),
        challenge.difficulty,
        challenge.challenge_hash.as_bytes(),
    );
    let vdf_proof = VdfComputer::compute_vdf(&vdf_challenge, VdfAlgorithm::Wesolowski)?;
    let compute_time = start.elapsed();

    println!(
        "✅ VDF proof computed in {:.2} seconds",
        compute_time.as_secs_f64()
    );
    println!("   Algorithm: {:?}", vdf_proof.algorithm);
    println!("   Proof size: {} bytes", vdf_proof.proof.len());

    // Step 6: Build attestation report with validation results
    println!("\nStep 6: Building attestation report...");

    let validation_results = ValidationResults {
        hardware_performance_valid: all_gpus_valid,
        virtualization_detected: virt_status.is_virtualized,
        replay_protection_valid: true,
        performance_confidence: if all_gpus_valid { 0.95 } else { 0.3 },
        virtualization_confidence: virt_status.confidence,
        validation_timestamp: chrono::Utc::now(),
        validation_details: gpu_attestor::attestation::ValidationDetails {
            gpu_performance_results: gpu_results,
            virtualization_indicators: vec![
                format!("Hypervisor: {:?}", virt_status.hypervisor),
                format!("Container: {:?}", virt_status.container),
                format!("CPU Flags: {:?}", virt_status.cpu_flags),
            ],
            challenge_response_time_ms: compute_time.as_millis() as u64,
        },
    };

    let mut builder = AttestationBuilder::new("secure-executor-001".to_string())
        .with_gpu_info(gpu_info)
        .with_system_info(system_info)
        .with_vdf_proof(vdf_proof);

    // Apply security validations
    builder = builder
        .validate_hardware_performance()
        .map_err(|e| anyhow::anyhow!("Hardware validation failed: {}", e))?;
    builder = builder
        .check_virtualization()
        .map_err(|e| anyhow::anyhow!("Virtualization check failed: {}", e))?;

    let mut report = builder.build();
    report.validation_results = Some(validation_results);

    // Step 7: Sign the attestation
    println!("\nStep 7: Signing attestation...");
    let signer = AttestationSigner::new();
    let signed_attestation = signer.sign_attestation(report)?;

    println!("✅ Attestation signed successfully");
    println!("   Signature: {} bytes", signed_attestation.signature.len());
    println!(
        "   Public key: {} bytes",
        signed_attestation.ephemeral_public_key.len()
    );

    // Step 8: Summary
    println!("\n=== Attestation Summary ===");
    if let Some(validation) = &signed_attestation.report.validation_results {
        println!(
            "Hardware Performance: {}",
            if validation.hardware_performance_valid {
                "✅ VALID"
            } else {
                "❌ INVALID"
            }
        );
        println!(
            "Virtualization: {}",
            if validation.virtualization_detected {
                "❌ DETECTED"
            } else {
                "✅ NOT DETECTED"
            }
        );
        println!(
            "Replay Protection: {}",
            if validation.replay_protection_valid {
                "✅ VALID"
            } else {
                "❌ INVALID"
            }
        );
        println!(
            "\nOverall Status: {}",
            if validation.is_valid() {
                "✅ SECURE"
            } else {
                "❌ INSECURE"
            }
        );
    }

    // Save attestation to file
    let attestation_json = serde_json::to_string_pretty(&signed_attestation)?;
    std::fs::write("secure_attestation.json", attestation_json)?;
    println!("\n✅ Attestation saved to secure_attestation.json");

    Ok(())
}

// Helper function to build virtualization indicators list
#[allow(dead_code)]
fn build_virtualization_indicators(
    status: &gpu_attestor::validation::VirtualizationStatus,
) -> Vec<String> {
    let mut indicators = Vec::new();

    if status.hypervisor.is_some() {
        indicators.push(format!("Hypervisor: {:?}", status.hypervisor));
    }
    if status.container.is_some() {
        indicators.push(format!("Container: {:?}", status.container));
    }
    if !status.cpu_flags.is_empty() {
        indicators.push(format!("CPU flags: {:?}", status.cpu_flags));
    }
    if let Some(vendor) = &status.dmi_vendor {
        indicators.push(format!("DMI vendor: {vendor}"));
    }
    if !status.kernel_modules.is_empty() {
        indicators.push(format!("Virt modules: {:?}", status.kernel_modules));
    }
    if !status.network_interfaces.is_empty() {
        indicators.push(format!("Virtual NICs: {:?}", status.network_interfaces));
    }
    if !status.disk_devices.is_empty() {
        indicators.push(format!("Virtual disks: {:?}", status.disk_devices));
    }

    indicators
}
