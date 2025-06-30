//! Multi-GPU Benchmark Example
//!
//! This example demonstrates how to benchmark multiple GPUs in a system
//! and validate their performance independently.

use anyhow::Result;
use gpu_attestor::{
    gpu::{
        benchmarks::{BenchmarkBackend, GpuBenchmarkRunner},
        GpuDetector,
    },
    validation::PerformanceValidator,
};

fn main() -> Result<()> {
    // Initialize logging
    tracing_subscriber::fmt::init();

    println!("=== Multi-GPU Benchmark Example ===\n");

    // Detect all GPUs
    let detector = GpuDetector::new();
    let gpus = detector.detect()?;

    println!("Found {} GPU(s) in the system:", gpus.len());
    for (idx, gpu) in gpus.iter().enumerate() {
        println!(
            "  GPU {}: {} ({:.1} GB)",
            idx,
            gpu.name,
            gpu.memory_total as f64 / (1024.0 * 1024.0 * 1024.0)
        );
    }

    if gpus.len() < 2 {
        println!("\n⚠️  This system has less than 2 GPUs. Multi-GPU features are still available.");
    }

    // Benchmark all GPUs using the convenience method
    println!("\n=== Running Benchmarks on All GPUs ===");
    match GpuBenchmarkRunner::benchmark_all_gpus() {
        Ok(results) => {
            for result in &results {
                println!(
                    "\nGPU {}: {} (Backend: {:?})",
                    result.gpu_index, result.gpu_name, result.backend
                );
                println!(
                    "  Memory Bandwidth: {:.2} GB/s",
                    result.memory_bandwidth_gbps
                );
                println!("  FP16 Performance: {:.2} TFLOPS", result.fp16_tflops);
            }

            // Calculate aggregate performance
            if results.len() > 1 {
                let total_bandwidth: f64 = results.iter().map(|r| r.memory_bandwidth_gbps).sum();
                let total_tflops: f64 = results.iter().map(|r| r.fp16_tflops).sum();

                println!("\n=== Aggregate Performance ===");
                println!("Total Memory Bandwidth: {total_bandwidth:.2} GB/s");
                println!("Total FP16 Performance: {total_tflops:.2} TFLOPS");
            }
        }
        Err(e) => {
            println!("Failed to benchmark all GPUs: {e}");
        }
    }

    // Benchmark specific GPUs with different backends
    println!("\n=== Individual GPU Benchmarking ===");

    for (idx, gpu) in gpus.iter().enumerate() {
        println!("\nTesting GPU {idx} with available backends:");

        // Try CUDA
        #[cfg(feature = "cuda")]
        {
            match GpuBenchmarkRunner::with_backend(idx as u32, BenchmarkBackend::Cuda) {
                Ok(runner) => {
                    println!("  CUDA Backend:");
                    benchmark_gpu(&runner);
                }
                Err(e) => {
                    println!("  CUDA Backend: Not available - {e}");
                }
            }
        }

        // Try OpenCL
        #[cfg(feature = "opencl")]
        {
            match GpuBenchmarkRunner::with_backend(idx as u32, BenchmarkBackend::OpenCL) {
                Ok(runner) => {
                    println!("  OpenCL Backend:");
                    benchmark_gpu(&runner);
                }
                Err(e) => {
                    println!("  OpenCL Backend: Not available - {}", e);
                }
            }
        }
    }

    // Performance validation for multi-GPU setups
    println!("\n=== Multi-GPU Performance Validation ===");
    let validator = PerformanceValidator::new();

    for (idx, gpu) in gpus.iter().enumerate() {
        println!("\nValidating GPU {}: {}", idx, gpu.name);
        match validator.validate_gpu(gpu) {
            Ok(result) => {
                if result.is_valid {
                    println!("  ✅ PASSED - Performance matches expected profile");
                } else {
                    println!("  ❌ FAILED - Performance does not match profile");
                    if !result.memory_validation.is_valid {
                        println!("    - Memory size mismatch");
                    }
                    if !result.bandwidth_validation.is_valid {
                        println!("    - Bandwidth below expected");
                    }
                    if !result.compute_validation.is_valid {
                        println!("    - Compute performance below expected");
                    }
                }
                println!("  Confidence: {:.0}%", result.confidence_score * 100.0);
            }
            Err(e) => {
                println!("  ⚠️  Validation error: {e}");
            }
        }
    }

    // Multi-GPU considerations
    println!("\n=== Multi-GPU Considerations ===");
    println!("• Each GPU is benchmarked independently");
    println!("• Performance may vary between GPUs even of the same model");
    println!("• Thermal throttling may affect GPUs differently");
    println!("• PCIe bandwidth limitations may impact multi-GPU scaling");
    println!("• CUDA/OpenCL may enumerate GPUs in different orders");

    if gpus.len() > 1 {
        // Check for GPU variations
        let unique_models: std::collections::HashSet<_> = gpus.iter().map(|g| &g.name).collect();

        if unique_models.len() == 1 {
            println!("\n✅ All GPUs are the same model - good for uniform workloads");
        } else {
            println!("\n⚠️  Mixed GPU models detected - may require load balancing");
        }
    }

    Ok(())
}

fn benchmark_gpu(runner: &GpuBenchmarkRunner) {
    match runner.benchmark_memory_bandwidth() {
        Ok(bandwidth) => {
            println!("    Memory Bandwidth: {bandwidth:.2} GB/s");
        }
        Err(e) => {
            println!("    Memory Bandwidth: Failed - {e}");
        }
    }

    match runner.benchmark_fp16_compute() {
        Ok(tflops) => {
            println!("    FP16 Performance: {tflops:.2} TFLOPS");
        }
        Err(e) => {
            println!("    FP16 Performance: Failed - {e}");
        }
    }
}
