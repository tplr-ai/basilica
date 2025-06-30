//! Example demonstrating CUDA Driver API matrix multiplication
//!
//! This example shows how to use the cuda_driver module for GPU computation
//! using custom PTX kernels as an alternative to cuBLAS.

use anyhow::Result;
use gpu_attestor::gpu::cuda_driver::{CudaMatrixCompute, MatrixCompute, MatrixDimensions};
use std::time::Instant;
use tracing::info;

fn main() -> Result<()> {
    // Initialize logging
    tracing_subscriber::fmt()
        .with_max_level(tracing::Level::INFO)
        .init();

    info!("CUDA Driver API Matrix Multiplication Example");
    info!("============================================");

    // Create CUDA matrix compute engine
    let compute = match CudaMatrixCompute::new() {
        Ok(c) => c,
        Err(e) => {
            eprintln!("Failed to initialize CUDA: {e}");
            eprintln!("Make sure you have an NVIDIA GPU and CUDA drivers installed");
            return Err(e);
        }
    };

    info!("CUDA Driver API initialized successfully");

    // Test different matrix sizes
    let test_sizes = vec![(512, 512), (1024, 1024), (2048, 2048)];

    for (n, k) in test_sizes {
        info!("\nTesting {}x{} matrix multiplication", n, k);

        let dimensions = MatrixDimensions::new(n, k);
        let seed = 42;
        let device_id = 0;

        // Warm up
        info!("Warming up...");
        for _ in 0..3 {
            compute.multiply_matrices(&dimensions, seed, device_id)?;
        }

        // Benchmark
        info!("Running benchmark...");
        let num_iterations = 10;
        let mut total_time = 0.0;

        for i in 0..num_iterations {
            let start = Instant::now();
            let result = compute.multiply_matrices(&dimensions, seed + i as u64, device_id)?;
            let elapsed = start.elapsed();

            total_time += result.execution_time_ms;
            info!(
                "  Iteration {}: {:.2} ms (total elapsed: {:.2} ms)",
                i + 1,
                result.execution_time_ms,
                elapsed.as_secs_f32() * 1000.0
            );
        }

        let avg_time = total_time / num_iterations as f32;
        info!("Average execution time: {:.2} ms", avg_time);

        // Calculate GFLOPS
        let operations = 2.0 * (n as f64) * (n as f64) * (k as f64);
        let gflops = operations / (avg_time as f64 * 1e6);
        info!("Performance: {:.2} GFLOPS", gflops);
    }

    info!("\nComparison with cuBLAS:");
    info!("========================");
    info!("This implementation uses direct CUDA Driver API with custom PTX kernels.");
    info!("For optimized performance with tensor cores, use the cuBLAS-based implementation.");
    info!("The CUDA Driver API approach provides:");
    info!("  - Direct control over GPU execution");
    info!("  - Custom kernel implementation");
    info!("  - Runtime PTX loading capability");
    info!("  - Deterministic execution timing");

    Ok(())
}
