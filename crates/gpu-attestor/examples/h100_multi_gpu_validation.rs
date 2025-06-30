//! H100 Multi-GPU Validation Example
//!
//! This example is specifically designed to detect and validate multi-H100 setups,
//! including 8x H100 systems with NVLink interconnects.

use anyhow::Result;
use gpu_attestor::{gpu::GpuDetector, validation::PerformanceValidator};

#[cfg(feature = "cuda")]
use gpu_attestor::gpu::cuda_ffi::{
    benchmark_tensor_core_fp16, benchmark_tensor_core_fp8, cudaDeviceCanAccessPeer, cudaDeviceProp,
    cudaGetDeviceProperties, cudaSetDevice, cudaSuccess, get_cuda_driver_version,
    get_cuda_runtime_version, validate_gpu, CublasHandle, CUBLAS_TENSOR_OP_MATH,
};

fn main() -> Result<()> {
    // Initialize logging
    tracing_subscriber::fmt::init();

    println!("=== H100 Multi-GPU Detection and Validation ===\n");

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

    // Detect all GPUs
    let detector = GpuDetector::new();
    let gpus = detector.detect()?;

    println!("Detected {} GPU(s)\n", gpus.len());

    // Check if this is a multi-H100 system
    let h100_count = gpus
        .iter()
        .filter(|gpu| gpu.name.contains("H100") || gpu.name.contains("H200"))
        .count();

    if h100_count > 0 {
        println!("✓ Found {h100_count} H100/H200 GPU(s)");
        if h100_count == 8 {
            println!("✓ This appears to be an 8x H100 system!");
        }
    }

    // Detailed GPU information
    println!("\n=== GPU Details ===\n");

    for (idx, gpu) in gpus.iter().enumerate() {
        println!("GPU {}: {}", idx, gpu.name);
        println!(
            "  Memory: {:.1} GB",
            gpu.memory_total as f64 / (1024.0 * 1024.0 * 1024.0)
        );
        println!(
            "  Memory Used: {:.1} GB",
            gpu.memory_used as f64 / (1024.0 * 1024.0 * 1024.0)
        );

        if let Some(temp) = gpu.temperature {
            println!("  Temperature: {temp}°C");
        }

        if let Some(util) = gpu.utilization {
            println!("  Utilization: {util}%");
        }

        // Check if it's an H100
        let is_h100 = gpu.name.contains("H100") || gpu.name.contains("H200");
        if is_h100 {
            println!("  ✓ Hopper architecture GPU detected");
        }

        println!();
    }

    // Check NVLink connectivity
    #[cfg(feature = "cuda")]
    {
        println!("=== NVLink Topology ===\n");
        check_nvlink_topology(&gpus)?;
        println!();
    }

    // Run validation on each H100
    println!("=== Individual GPU Validation ===\n");

    for (idx, gpu) in gpus.iter().enumerate() {
        if gpu.name.contains("H100") || gpu.name.contains("H200") {
            println!("Validating GPU {}: {}", idx, gpu.name);

            #[cfg(feature = "cuda")]
            {
                unsafe {
                    if cudaSetDevice(idx as i32) != cudaSuccess {
                        println!("  ❌ Failed to set CUDA device");
                        continue;
                    }
                }

                // Create cuBLAS handle for this GPU
                match CublasHandle::new() {
                    Ok(handle) => {
                        // Quick validation
                        match unsafe { validate_gpu(handle.as_ptr(), idx as i32) } {
                            Ok((gpu_name, is_valid)) => {
                                if is_valid {
                                    println!("  ✅ {gpu_name} validated successfully");

                                    // Determine if it's PCIe or SXM variant
                                    let props = get_gpu_properties(idx as i32)?;
                                    if props.multiProcessorCount >= 114 {
                                        println!(
                                            "  ✓ H100 SXM variant detected (SM count: {})",
                                            props.multiProcessorCount
                                        );
                                    } else {
                                        println!(
                                            "  ✓ H100 PCIe variant detected (SM count: {})",
                                            props.multiProcessorCount
                                        );
                                    }
                                } else {
                                    println!("  ❌ {gpu_name} validation failed");
                                }
                            }
                            Err(e) => {
                                println!("  ❌ Validation error: {e}");
                            }
                        }
                    }
                    Err(e) => {
                        println!("  ❌ Failed to create cuBLAS handle: {e}");
                    }
                }
            }

            #[cfg(not(feature = "cuda"))]
            {
                println!("  ⚠️  CUDA support not enabled");
            }

            println!();
        }
    }

    // Run performance validation
    println!("=== Performance Validation ===\n");

    let validator = PerformanceValidator::new();
    let mut total_fp16_tflops = 0.0;
    let mut valid_count = 0;

    for (idx, gpu) in gpus.iter().enumerate() {
        if gpu.name.contains("H100") || gpu.name.contains("H200") {
            match validator.validate_gpu(gpu) {
                Ok(result) => {
                    println!("GPU {}: {}", idx, gpu.name);
                    println!("  Valid: {}", result.is_valid);
                    println!(
                        "  FP16 Performance: {:.1} TFLOPS",
                        result.compute_validation.measured_tflops
                    );
                    println!(
                        "  Memory Bandwidth: {:.1} GB/s",
                        result.bandwidth_validation.measured_gbps
                    );

                    if result.is_valid {
                        valid_count += 1;
                        total_fp16_tflops += result.compute_validation.measured_tflops;
                    }
                }
                Err(e) => {
                    println!("GPU {idx} validation failed: {e}");
                }
            }
            println!();
        }
    }

    // Aggregate performance metrics
    if h100_count > 0 {
        println!("\n=== Aggregate Performance ===\n");
        println!("Total H100 GPUs: {h100_count}");
        println!("Valid GPUs: {valid_count}");

        if valid_count > 0 {
            println!(
                "Average FP16 per GPU: {:.1} TFLOPS",
                total_fp16_tflops / valid_count as f64
            );
            println!("Total System FP16: {total_fp16_tflops:.1} TFLOPS");

            // Expected performance for H100 systems
            let expected_per_gpu = if h100_count == 8 {
                // 8x H100 likely means SXM5 variant
                1979.0
            } else {
                // Could be PCIe variant
                1513.0
            };

            let expected_total = expected_per_gpu * h100_count as f64;
            let efficiency = (total_fp16_tflops / expected_total) * 100.0;

            println!("\nExpected Total: {expected_total:.1} TFLOPS");
            println!("Efficiency: {efficiency:.1}%");

            if efficiency > 70.0 {
                println!("✅ System performance validated");
            } else {
                println!("⚠️  System performance below expected");
            }
        }
    }

    // Run a quick benchmark on one GPU
    #[cfg(feature = "cuda")]
    if h100_count > 0 {
        println!("\n=== Quick Benchmark (GPU 0) ===\n");

        unsafe {
            cudaSetDevice(0);
        }

        match CublasHandle::new() {
            Ok(mut handle) => {
                // Enable tensor cores
                if handle.set_math_mode(CUBLAS_TENSOR_OP_MATH).is_ok() {
                    println!("Running FP16 Tensor Core benchmark...");

                    match unsafe {
                        benchmark_tensor_core_fp16(handle.as_ptr(), 4096, 4096, 4096, 10)
                    } {
                        Ok(tflops) => {
                            println!("✓ FP16 Performance: {tflops:.1} TFLOPS");
                        }
                        Err(e) => {
                            println!("❌ FP16 benchmark failed: {e}");
                        }
                    }

                    match unsafe {
                        benchmark_tensor_core_fp8(handle.as_ptr(), 4096, 4096, 4096, 10)
                    } {
                        Ok(tflops) => {
                            println!("✓ FP8 Performance: {tflops:.1} TFLOPS");
                        }
                        Err(e) => {
                            println!("❌ FP8 benchmark failed: {e}");
                        }
                    }
                }
            }
            Err(e) => {
                println!("❌ Failed to create cuBLAS handle: {e}");
            }
        }
    }

    // Summary
    println!("\n=== Summary ===\n");

    if h100_count == 8 && valid_count == 8 {
        println!("✅ All 8 H100 GPUs detected and validated");
        println!("✅ This is a genuine 8x H100 system");

        #[cfg(feature = "cuda")]
        {
            // Check for specific 8x H100 configurations
            if gpus[0].memory_total > 80 * 1024 * 1024 * 1024 {
                println!("✓ H100 80GB variant detected");
            } else {
                println!("✓ H100 40GB variant detected");
            }

            println!("\nSystem appears to be one of:");
            println!("- DGX H100 (8x H100 SXM5)");
            println!("- HGX H100 (8x H100 SXM5)");
            println!("- Custom 8x H100 system");
        }
    } else if h100_count > 0 {
        println!("✓ {h100_count} H100/H200 GPU(s) detected");
        println!("✓ {valid_count} GPU(s) passed validation");
    } else {
        println!("ℹ️  No H100/H200 GPUs detected on this system");
    }

    Ok(())
}

#[cfg(feature = "cuda")]
fn check_nvlink_topology(gpus: &[gpu_attestor::gpu::types::GpuInfo]) -> Result<()> {
    let gpu_count = gpus.len();

    if gpu_count < 2 {
        println!("Single GPU system - no NVLink topology to check");
        return Ok(());
    }

    println!("Checking peer access between GPUs:");
    println!("(✓ = NVLink available, ✗ = No direct connection)\n");

    // Print header
    print!("     ");
    for i in 0..gpu_count {
        print!("GPU{i} ");
    }
    println!();

    // Check peer access
    for i in 0..gpu_count {
        print!("GPU{i} ");

        for j in 0..gpu_count {
            if i == j {
                print!("  -  ");
            } else {
                unsafe {
                    let mut can_access = 0;
                    let result = cudaDeviceCanAccessPeer(&mut can_access, i as i32, j as i32);

                    if result == cudaSuccess && can_access != 0 {
                        print!("  ✓  ");
                    } else {
                        print!("  ✗  ");
                    }
                }
            }
        }
        println!();
    }

    // Count NVLink connections
    let mut nvlink_count = 0;
    for i in 0..gpu_count {
        for j in (i + 1)..gpu_count {
            unsafe {
                let mut can_access = 0;
                let result = cudaDeviceCanAccessPeer(&mut can_access, i as i32, j as i32);

                if result == cudaSuccess && can_access != 0 {
                    nvlink_count += 1;
                }
            }
        }
    }

    println!("\nTotal NVLink connections: {nvlink_count}");

    // Determine topology
    if gpu_count == 8 {
        if nvlink_count >= 28 {
            println!("✓ Full NVLink mesh topology detected (DGX/HGX H100)");
        } else if nvlink_count >= 12 {
            println!("✓ Partial NVLink topology detected");
        } else {
            println!("⚠️  Limited NVLink connectivity");
        }
    }

    Ok(())
}

#[cfg(feature = "cuda")]
fn get_gpu_properties(device: i32) -> Result<cudaDeviceProp> {
    unsafe {
        let mut props = std::mem::zeroed::<cudaDeviceProp>();
        let result = cudaGetDeviceProperties(&mut props, device);
        if result != cudaSuccess {
            return Err(anyhow::anyhow!("Failed to get device properties"));
        }
        Ok(props)
    }
}

#[cfg(not(feature = "cuda"))]
fn check_nvlink_topology(_gpus: &[gpu_attestor::gpu::types::GpuInfo]) -> Result<()> {
    println!("CUDA support not enabled - cannot check NVLink topology");
    Ok(())
}
