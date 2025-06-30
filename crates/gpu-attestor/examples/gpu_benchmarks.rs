//! Example demonstrating production GPU benchmarks
//!
//! This example shows how the performance validator uses actual GPU benchmarks
//! to detect hardware spoofing and validate GPU specifications.

use anyhow::Result;
use gpu_attestor::{
    gpu::{benchmarks::GpuBenchmarkRunner, GpuDetector},
    validation::PerformanceValidator,
};

fn main() -> Result<()> {
    // Initialize logging
    tracing_subscriber::fmt::init();

    println!("=== Production GPU Benchmark Example ===\n");

    // Detect GPUs
    let detector = GpuDetector::new();
    let gpus = detector.detect()?;

    if gpus.is_empty() {
        println!("No GPUs detected on this system");
        return Ok(());
    }

    println!("Found {} GPU(s):", gpus.len());
    for (idx, gpu) in gpus.iter().enumerate() {
        println!(
            "  GPU {}: {} ({:.1} GB)",
            idx,
            gpu.name,
            gpu.memory_total as f64 / (1024.0 * 1024.0 * 1024.0)
        );
    }
    println!();

    // Run benchmarks on each GPU
    for (idx, gpu) in gpus.iter().enumerate() {
        println!("Testing GPU {}: {}", idx, gpu.name);
        println!("{}", "=".repeat(50));

        // Create benchmark runner
        match GpuBenchmarkRunner::new(idx as u32) {
            Ok(runner) => {
                // Run memory bandwidth benchmark
                println!("\nMemory Bandwidth Benchmark:");
                match runner.benchmark_memory_bandwidth() {
                    Ok(bandwidth) => {
                        println!("  Measured: {bandwidth:.2} GB/s");

                        // Compare with expected values
                        let expected = match gpu.name.as_str() {
                            name if name.contains("H200") => 4800.0,
                            name if name.contains("H100") => 3350.0,
                            name if name.contains("A100") => 2039.0,
                            name if name.contains("RTX 4090") => 1008.0,
                            name if name.contains("RTX 4080") => 717.0,
                            name if name.contains("RTX 3090") => 936.0,
                            _ => 500.0, // Conservative estimate
                        };

                        let ratio = bandwidth / expected;
                        println!("  Expected: {expected:.2} GB/s");
                        println!("  Ratio: {ratio:.2}x");

                        if ratio < 0.8 {
                            println!("  ⚠️  WARNING: Performance below expected!");
                        } else if ratio > 1.2 {
                            println!("  ⚠️  WARNING: Performance suspiciously high!");
                        } else {
                            println!("  ✅ Performance within expected range");
                        }
                    }
                    Err(e) => {
                        println!("  ❌ Failed: {e}");
                    }
                }

                // Run FP16 compute benchmark
                println!("\nFP16 Compute Benchmark:");
                match runner.benchmark_fp16_compute() {
                    Ok(tflops) => {
                        println!("  Measured: {tflops:.2} TFLOPS");

                        // Compare with expected values
                        let expected = match gpu.name.as_str() {
                            name if name.contains("H200") => 1979.0,
                            name if name.contains("H100") => 1979.0,
                            name if name.contains("A100") => 312.0,
                            name if name.contains("RTX 4090") => 82.6,
                            name if name.contains("RTX 4080") => 48.7,
                            name if name.contains("RTX 3090") => 35.6,
                            _ => 20.0, // Conservative estimate
                        };

                        let ratio = tflops / expected;
                        println!("  Expected: {expected:.2} TFLOPS");
                        println!("  Ratio: {ratio:.2}x");

                        if ratio < 0.8 {
                            println!("  ⚠️  WARNING: Performance below expected!");
                        } else if ratio > 1.2 {
                            println!("  ⚠️  WARNING: Performance suspiciously high!");
                        } else {
                            println!("  ✅ Performance within expected range");
                        }
                    }
                    Err(e) => {
                        println!("  ❌ Failed: {e}");
                    }
                }
            }
            Err(e) => {
                println!("  Failed to create benchmark runner: {e}");
            }
        }

        println!();
    }

    // Test with performance validator
    println!("\n=== Performance Validator Test ===");
    let validator = PerformanceValidator::new();

    for gpu in &gpus {
        println!("\nValidating {}", gpu.name);
        match validator.validate_gpu(gpu) {
            Ok(result) => {
                println!("  Valid: {}", result.is_valid);
                println!("  Confidence: {:.0}%", result.confidence_score * 100.0);
                println!(
                    "  Memory: {} (expected: {} GB)",
                    if result.memory_validation.is_valid {
                        "✅"
                    } else {
                        "❌"
                    },
                    result.memory_validation.expected_bytes / (1024 * 1024 * 1024)
                );
                println!(
                    "  Bandwidth: {} ({:.0} GB/s measured, {:.0} GB/s expected)",
                    if result.bandwidth_validation.is_valid {
                        "✅"
                    } else {
                        "❌"
                    },
                    result.bandwidth_validation.measured_gbps,
                    result.bandwidth_validation.expected_gbps
                );
                println!(
                    "  Compute: {} ({:.0} TFLOPS measured, {:.0} TFLOPS expected)",
                    if result.compute_validation.is_valid {
                        "✅"
                    } else {
                        "❌"
                    },
                    result.compute_validation.measured_tflops,
                    result.compute_validation.expected_tflops
                );
            }
            Err(e) => {
                println!("  Validation failed: {e}");
            }
        }
    }

    println!("\n=== Summary ===");
    println!("Production GPU benchmarks provide real performance measurements");
    println!("to detect hardware spoofing and validate GPU specifications.");
    println!("\nFeatures demonstrated:");
    println!("- CUDA kernels for NVIDIA GPUs");
    println!("- OpenCL kernels for cross-platform support");
    println!("- Memory bandwidth testing with vectorized operations");
    println!("- FP16 compute testing with matrix multiplication");
    println!("- Comparison against known hardware profiles");
    println!("- Detection of anomalous performance");

    Ok(())
}
