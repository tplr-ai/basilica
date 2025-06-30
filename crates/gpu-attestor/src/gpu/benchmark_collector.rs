//! GPU benchmark collection for attestation
//!
//! This module collects comprehensive GPU benchmarks including:
//! - Memory bandwidth tests
//! - FP16 compute performance
//! - CUDA Driver API matrix multiplication

use anyhow::Result;
use std::time::Instant;
use tracing::{debug, info, warn};

use crate::attestation::types::{
    CudaDriverBenchmarkResults, GpuBenchmarkResults, MatrixMultiplicationResult, SingleGpuBenchmark,
};
use crate::gpu::benchmarks::{BenchmarkBackend, GpuBenchmarkRunner};
use crate::gpu::cuda_driver::{CudaMatrixCompute, MatrixCompute, MatrixDimensions};
use crate::gpu::{GpuDetector, GpuVendor};

/// Collect comprehensive GPU benchmarks for all available GPUs
pub async fn collect_gpu_benchmarks() -> Result<GpuBenchmarkResults> {
    info!("Starting GPU benchmark collection...");
    let start_time = Instant::now();

    let detector = GpuDetector::new();
    let gpus = detector.detect()?;

    if gpus.is_empty() {
        warn!("No GPUs detected, skipping GPU benchmarks");
        return Ok(GpuBenchmarkResults {
            gpu_results: vec![],
            total_benchmark_time_ms: 0.0,
        });
    }

    let mut gpu_results = Vec::new();

    for (idx, gpu_info) in gpus.iter().enumerate() {
        info!("Benchmarking GPU {}: {}", idx, gpu_info.name);

        let mut benchmark_result = SingleGpuBenchmark {
            gpu_index: idx as u32,
            gpu_name: gpu_info.name.clone(),
            memory_bandwidth_gbps: None,
            fp16_tflops: None,
            cuda_driver_results: None,
            backend: "None".to_string(),
            error: None,
        };

        // Try standard GPU benchmarks first
        match run_standard_benchmarks(idx as u32).await {
            Ok((bandwidth, fp16, backend)) => {
                benchmark_result.memory_bandwidth_gbps = Some(bandwidth);
                benchmark_result.fp16_tflops = Some(fp16);
                benchmark_result.backend = format!("{backend:?}");
                info!(
                    "  Standard benchmarks: {:.2} GB/s bandwidth, {:.2} TFLOPS FP16",
                    bandwidth, fp16
                );
            }
            Err(e) => {
                warn!("  Standard benchmarks failed: {}", e);
                benchmark_result.error = Some(format!("Standard benchmarks failed: {e}"));
            }
        }

        // Try CUDA Driver API benchmarks if this is an NVIDIA GPU
        if gpu_info.vendor == GpuVendor::Nvidia {
            match run_cuda_driver_benchmarks(idx as u32).await {
                Ok(cuda_results) => {
                    info!("  CUDA Driver API benchmarks completed successfully");
                    benchmark_result.cuda_driver_results = Some(cuda_results);
                }
                Err(e) => {
                    warn!("  CUDA Driver API benchmarks failed: {}", e);
                    if let Some(ref mut error) = benchmark_result.error {
                        error.push_str(&format!("; CUDA Driver API failed: {e}"));
                    } else {
                        benchmark_result.error = Some(format!("CUDA Driver API failed: {e}"));
                    }
                }
            }
        }

        gpu_results.push(benchmark_result);
    }

    let total_time = start_time.elapsed();
    info!(
        "GPU benchmark collection completed in {:.2}s",
        total_time.as_secs_f64()
    );

    Ok(GpuBenchmarkResults {
        gpu_results,
        total_benchmark_time_ms: total_time.as_secs_f64() * 1000.0,
    })
}

/// Run standard GPU benchmarks using available backends
async fn run_standard_benchmarks(gpu_index: u32) -> Result<(f64, f64, BenchmarkBackend)> {
    let runner = GpuBenchmarkRunner::new(gpu_index)?;
    let backend = runner.backend.clone();

    // Run memory bandwidth test
    debug!("Running memory bandwidth test on GPU {}", gpu_index);
    let bandwidth = runner.benchmark_memory_bandwidth()?;

    // Run FP16 compute test
    debug!("Running FP16 compute test on GPU {}", gpu_index);
    let fp16_tflops = runner.benchmark_fp16_compute()?;

    Ok((bandwidth, fp16_tflops, backend))
}

/// Run CUDA Driver API benchmarks
async fn run_cuda_driver_benchmarks(gpu_index: u32) -> Result<CudaDriverBenchmarkResults> {
    info!("Initializing CUDA Driver API for GPU {}", gpu_index);

    // Create CUDA matrix compute engine
    let compute = CudaMatrixCompute::new()?;

    // Test different matrix sizes
    let test_sizes = vec![512, 1024, 2048, 4096];
    let mut matrix_results = Vec::new();

    for size in test_sizes {
        debug!("Testing {}x{} matrix multiplication", size, size);

        let dimensions = MatrixDimensions::new(size, size);
        let seed = 42;

        // Warm up
        for _ in 0..3 {
            compute.multiply_matrices(&dimensions, seed, gpu_index)?;
        }

        // Benchmark
        let iterations = 10;
        let mut total_time = 0.0;
        let mut checksum = None;

        for i in 0..iterations {
            let result = compute.multiply_matrices(&dimensions, seed + i as u64, gpu_index)?;
            total_time += result.execution_time_ms;

            if i == 0 {
                checksum = result.matrix_checksum.map(|c| format!("{c:016x}"));
            }
        }

        let avg_time = total_time / iterations as f32;

        // Calculate GFLOPS
        let operations = 2.0 * (size as f64).powi(3);
        let gflops = operations / (avg_time as f64 * 1e6);

        matrix_results.push(MatrixMultiplicationResult {
            matrix_size: size as u32,
            execution_time_ms: avg_time as f64,
            gflops,
            checksum,
        });

        info!(
            "  {}x{}: {:.2} ms, {:.2} GFLOPS",
            size, size, avg_time, gflops
        );
    }

    // Get CUDA driver version
    let cuda_driver_version = get_cuda_driver_version();

    Ok(CudaDriverBenchmarkResults {
        matrix_results,
        ptx_loaded: true, // If we got here, PTX was loaded successfully
        cuda_driver_version,
    })
}

/// Get CUDA driver version if available
fn get_cuda_driver_version() -> Option<String> {
    // Try to get CUDA driver version
    match std::process::Command::new("nvidia-smi")
        .args(["--query", "--display=DRIVER"])
        .output()
    {
        Ok(output) => {
            if output.status.success() {
                String::from_utf8(output.stdout).ok().and_then(|s| {
                    s.lines()
                        .find(|line| line.contains("CUDA Version"))
                        .map(|line| line.trim().to_string())
                })
            } else {
                None
            }
        }
        Err(_) => None,
    }
}
