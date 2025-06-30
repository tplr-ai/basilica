//! Hopper/Blackwell GPU Validation Example
//!
//! This example demonstrates how to validate NVIDIA Hopper (H100, H200) and
//! Blackwell GPUs using tensor core optimized benchmarks.

use anyhow::Result;
use gpu_attestor::{
    gpu::{benchmarks::GpuBenchmarkRunner, GpuDetector},
    validation::PerformanceValidator,
};

#[cfg(feature = "cuda")]
use gpu_attestor::gpu::cuda_ffi::{get_cuda_driver_version, get_cuda_runtime_version};

fn main() -> Result<()> {
    // Initialize logging
    tracing_subscriber::fmt::init();

    println!("=== Hopper/Blackwell GPU Validation ===\n");

    // Check CUDA versions
    #[cfg(feature = "cuda")]
    {
        if let Ok((runtime_major, runtime_minor)) = get_cuda_runtime_version() {
            println!("CUDA Runtime Version: {runtime_major}.{runtime_minor}");
        }

        if let Ok((driver_major, driver_minor)) = get_cuda_driver_version() {
            println!("CUDA Driver Version: {driver_major}.{driver_minor}");
        }
        println!();
    }

    // Detect GPUs
    let detector = GpuDetector::new();
    let gpus = detector.detect()?;

    if gpus.is_empty() {
        println!("No GPUs detected.");
        return Ok(());
    }

    println!("Detected {} GPU(s):\n", gpus.len());

    for (idx, gpu) in gpus.iter().enumerate() {
        println!("GPU {}: {}", idx, gpu.name);
        println!("  Vendor: {:?}", gpu.vendor);
        println!(
            "  Memory: {:.1} GB",
            gpu.memory_total as f64 / (1024.0 * 1024.0 * 1024.0)
        );

        // Check if this is a Hopper or Blackwell GPU
        let is_hopper_blackwell = gpu.name.contains("H100")
            || gpu.name.contains("H200")
            || gpu.name.contains("B100")
            || gpu.name.contains("B200");

        if is_hopper_blackwell {
            println!("  ✓ Detected Hopper/Blackwell architecture");

            #[cfg(feature = "cuda")]
            {
                // Run tensor core validation
                println!("\n  Running tensor core validation...");

                unsafe {
                    use gpu_attestor::gpu::cuda_ffi::*;

                    // Set device
                    if cudaSetDevice(idx as i32) != cudaSuccess {
                        println!("  ❌ Failed to set CUDA device");
                        continue;
                    }

                    // Create cuBLAS handle using safe wrapper
                    match CublasHandle::new() {
                        Ok(handle) => {
                            // Validate GPU
                            match gpu_attestor::gpu::cuda_ffi::validate_gpu(
                                handle.as_ptr(),
                                idx as i32,
                            ) {
                                Ok((gpu_name, is_valid)) => {
                                    if is_valid {
                                        println!("  ✅ {gpu_name} validated successfully");
                                    } else {
                                        println!("  ❌ {gpu_name} validation failed - performance outside expected range");
                                    }
                                }
                                Err(e) => {
                                    println!("  ❌ Validation error: {e}");
                                }
                            }
                            // Handle is automatically cleaned up on drop
                        }
                        Err(e) => {
                            println!("  ❌ Failed to create cuBLAS handle: {e}");
                        }
                    }
                }
            }

            #[cfg(not(feature = "cuda"))]
            {
                println!("  ⚠️  CUDA support not enabled - cannot run tensor core validation");
            }
        } else {
            println!("  ℹ️  Not a Hopper/Blackwell GPU");
        }

        println!();
    }

    // Run standard benchmarks for comparison
    println!("\n=== Standard Benchmarks ===\n");

    for (idx, gpu) in gpus.iter().enumerate() {
        println!("GPU {}: {}", idx, gpu.name);

        match GpuBenchmarkRunner::new(idx as u32) {
            Ok(runner) => {
                // Memory bandwidth
                match runner.benchmark_memory_bandwidth() {
                    Ok(bandwidth) => {
                        println!("  Memory Bandwidth: {bandwidth:.2} GB/s");
                    }
                    Err(e) => {
                        println!("  Memory Bandwidth: Failed - {e}");
                    }
                }

                // FP16 compute (will use tensor cores for Hopper/Blackwell)
                match runner.benchmark_fp16_compute() {
                    Ok(tflops) => {
                        println!("  FP16 Performance: {tflops:.2} TFLOPS");

                        // Check if tensor cores were used
                        let is_hopper_blackwell = gpu.name.contains("H100")
                            || gpu.name.contains("H200")
                            || gpu.name.contains("B100")
                            || gpu.name.contains("B200");

                        if is_hopper_blackwell && tflops > 1000.0 {
                            println!("  ✓ Tensor cores detected (>1000 TFLOPS)");
                        }
                    }
                    Err(e) => {
                        println!("  FP16 Performance: Failed - {e}");
                    }
                }
            }
            Err(e) => {
                println!("  Failed to create benchmark runner: {e}");
            }
        }

        println!();
    }

    // Performance validation
    println!("\n=== Performance Validation ===\n");

    let validator = PerformanceValidator::new();

    for gpu in &gpus {
        println!("Validating {}", gpu.name);

        match validator.validate_gpu(gpu) {
            Ok(result) => {
                println!("  Overall Valid: {}", result.is_valid);
                println!("  Confidence Score: {:.2}", result.confidence_score);

                if !result.is_valid {
                    println!("  Issues:");
                    if !result.memory_validation.is_valid {
                        println!("    - Memory size mismatch");
                    }
                    if !result.bandwidth_validation.is_valid {
                        println!("    - Bandwidth outside expected range");
                    }
                    if !result.compute_validation.is_valid {
                        println!("    - Compute performance outside expected range");
                    }
                }

                // Special note for Hopper/Blackwell
                let is_hopper_blackwell = gpu.name.contains("H100")
                    || gpu.name.contains("H200")
                    || gpu.name.contains("B100")
                    || gpu.name.contains("B200");

                if is_hopper_blackwell {
                    println!("\n  Hopper/Blackwell Specific:");
                    println!(
                        "    Expected FP16: {:.1} TFLOPS",
                        result.compute_validation.expected_tflops
                    );
                    println!(
                        "    Measured FP16: {:.1} TFLOPS",
                        result.compute_validation.measured_tflops
                    );

                    if result.compute_validation.measured_tflops > 1000.0 {
                        println!("    ✓ Tensor cores confirmed");
                    }
                }
            }
            Err(e) => {
                println!("  Validation failed: {e}");
            }
        }

        println!();
    }

    println!("\n=== Summary ===");
    println!("\nAdvantages of cuBLAS/CUTLASS for Hopper/Blackwell:");
    println!("1. Optimized for 4th gen Tensor Cores (up to 2x faster)");
    println!("2. Native FP8 support (2x throughput vs FP16)");
    println!("3. Better memory hierarchy utilization");
    println!("4. Automatic selection of optimal algorithms");
    println!("5. Support for new instructions (TMA, async copy)");
    println!("\nValidation ensures:");
    println!("• GPU is genuine Hopper/Blackwell architecture");
    println!("• Performance matches hardware specifications");
    println!("• No virtualization or spoofing detected");

    Ok(())
}
