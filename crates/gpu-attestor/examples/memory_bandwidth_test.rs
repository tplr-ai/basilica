//! Memory Bandwidth Test Example
//!
//! This example demonstrates how to directly benchmark GPU memory bandwidth
//! and use it for performance validation.

use anyhow::Result;
use gpu_attestor::{
    gpu::{benchmarks::GpuBenchmarkRunner, GpuDetector},
    validation::PerformanceValidator,
};

fn main() -> Result<()> {
    // Initialize logging
    tracing_subscriber::fmt::init();

    println!("=== GPU Memory Bandwidth Test ===\n");

    // Method 1: Using GpuBenchmarkRunner directly
    println!("Method 1: Direct Benchmark Runner\n");

    // Detect GPUs
    let detector = GpuDetector::new();
    let gpus = detector.detect()?;

    for (idx, gpu) in gpus.iter().enumerate() {
        println!("GPU {}: {}", idx, gpu.name);
        println!(
            "  Memory: {:.1} GB",
            gpu.memory_total as f64 / (1024.0 * 1024.0 * 1024.0)
        );

        // Create benchmark runner for this specific GPU
        match GpuBenchmarkRunner::new(idx as u32) {
            Ok(runner) => {
                // Run memory bandwidth benchmark
                match runner.benchmark_memory_bandwidth() {
                    Ok(bandwidth) => {
                        println!("  Measured Bandwidth: {bandwidth:.2} GB/s");

                        // Compare with theoretical maximum
                        let theoretical_max = get_theoretical_bandwidth(&gpu.name);
                        let efficiency = (bandwidth / theoretical_max) * 100.0;

                        println!("  Theoretical Max: {theoretical_max:.2} GB/s");
                        println!("  Efficiency: {efficiency:.1}%");

                        if efficiency < 70.0 {
                            println!("  ⚠️  Low efficiency - may indicate issues");
                        } else if efficiency > 95.0 {
                            println!("  ⚠️  Suspiciously high efficiency");
                        } else {
                            println!("  ✅ Normal efficiency range");
                        }
                    }
                    Err(e) => {
                        println!("  ❌ Benchmark failed: {e}");
                    }
                }
            }
            Err(e) => {
                println!("  ❌ Failed to create runner: {e}");
            }
        }
        println!();
    }

    // Method 2: Using PerformanceValidator
    println!("\nMethod 2: Performance Validator\n");

    let validator = PerformanceValidator::new();

    // Benchmark default GPU (GPU 0)
    println!("Default GPU benchmark:");
    match validator.benchmark_memory_bandwidth() {
        Ok(bandwidth) => {
            println!("  Memory Bandwidth: {bandwidth:.2} GB/s");
        }
        Err(e) => {
            println!("  Failed: {e}");
        }
    }

    // Benchmark specific GPUs
    for (idx, gpu) in gpus.iter().enumerate() {
        println!("\nBenchmarking GPU {idx} via validator:");
        match GpuBenchmarkRunner::new(idx as u32) {
            Ok(runner) => match validator.benchmark_memory_bandwidth_with_runner(&runner) {
                Ok(bandwidth) => {
                    println!("  Memory Bandwidth: {bandwidth:.2} GB/s");
                }
                Err(e) => {
                    println!("  Failed: {e}");
                }
            },
            Err(e) => {
                println!("  Failed to create runner: {e}");
            }
        }
    }

    // Method 3: Full validation including bandwidth
    println!("\n\nMethod 3: Full Performance Validation\n");

    for gpu in &gpus {
        println!("Validating {}", gpu.name);
        match validator.validate_gpu(gpu) {
            Ok(result) => {
                println!("  Overall Valid: {}", result.is_valid);
                println!("  Bandwidth Validation:");
                println!(
                    "    Measured: {:.2} GB/s",
                    result.bandwidth_validation.measured_gbps
                );
                println!(
                    "    Expected: {:.2} GB/s",
                    result.bandwidth_validation.expected_gbps
                );
                println!("    Valid: {}", result.bandwidth_validation.is_valid);

                let ratio = result.bandwidth_validation.measured_gbps
                    / result.bandwidth_validation.expected_gbps;
                println!("    Ratio: {ratio:.2}x");
            }
            Err(e) => {
                println!("  Validation failed: {e}");
            }
        }
        println!();
    }

    // Bandwidth testing details
    println!("\n=== Memory Bandwidth Testing Details ===");
    println!("• Uses large buffers (25% of GPU memory, max 4GB)");
    println!("• Performs device-to-device copies for peak bandwidth");
    println!("• Uses vectorized operations (float4) for efficiency");
    println!("• Runs 100 iterations with warm-up");
    println!("• GPU-side timing with CUDA events or OpenCL profiling");
    println!("\nFactors affecting bandwidth:");
    println!("• GPU memory type (HBM2e, HBM3, GDDR6, etc.)");
    println!("• Thermal throttling");
    println!("• ECC memory enabled/disabled");
    println!("• Other processes using GPU memory");
    println!("• PCIe bandwidth (for system memory access)");

    Ok(())
}

fn get_theoretical_bandwidth(gpu_name: &str) -> f64 {
    match gpu_name {
        name if name.contains("H200") => 4800.0,
        name if name.contains("H100") => 3350.0,
        name if name.contains("A100") => 2039.0,
        name if name.contains("RTX 4090") => 1008.0,
        name if name.contains("RTX 4080") => 717.0,
        name if name.contains("RTX 3090") => 936.0,
        name if name.contains("V100") => 900.0,
        _ => 500.0, // Conservative default
    }
}
