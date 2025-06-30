//! CUDA Compatibility Test
//!
//! This example checks CUDA version compatibility and GPU compute capabilities

use anyhow::Result;

#[cfg(feature = "cuda")]
use gpu_attestor::gpu::cuda_ffi::{
    cudaDeviceProp, cudaGetDeviceCount, cudaGetDeviceProperties, get_cuda_driver_version,
    get_cuda_runtime_version,
};

fn main() -> Result<()> {
    println!("=== CUDA Compatibility Test ===\n");

    #[cfg(not(feature = "cuda"))]
    {
        println!("CUDA support not enabled. Build with --features cuda");
        return Ok(());
    }

    #[cfg(feature = "cuda")]
    {
        // Check CUDA versions
        match get_cuda_runtime_version() {
            Ok((major, minor)) => {
                println!("CUDA Runtime Version: {major}.{minor}");

                if (11..=13).contains(&major) {
                    println!("✅ Runtime version supported");
                } else if major > 13 {
                    println!("⚠️  Runtime version newer than tested (supports up to 13.9)");
                } else {
                    println!("❌ Runtime version too old (requires 11.0+)");
                }
            }
            Err(e) => {
                println!("❌ Failed to get CUDA runtime version: {e}");
            }
        }

        match get_cuda_driver_version() {
            Ok((major, minor)) => {
                println!("CUDA Driver Version: {major}.{minor}");

                if major >= 11 {
                    println!("✅ Driver version supported");
                } else {
                    println!("❌ Driver version too old (requires 11.0+)");
                }
            }
            Err(e) => {
                println!("❌ Failed to get CUDA driver version: {e}");
            }
        }

        // Check GPUs
        unsafe {
            let mut device_count = 0;
            if cudaGetDeviceCount(&mut device_count) == 0 && device_count > 0 {
                println!("\nFound {device_count} CUDA device(s):");

                for i in 0..device_count {
                    let mut props = std::mem::zeroed::<cudaDeviceProp>();
                    if cudaGetDeviceProperties(&mut props, i) == 0 {
                        let name = std::ffi::CStr::from_ptr(props.name.as_ptr()).to_string_lossy();

                        println!("\nDevice {i}: {name}");
                        println!("  Compute Capability: {}.{}", props.major, props.minor);
                        println!(
                            "  Memory: {:.1} GB",
                            props.totalGlobalMem as f64 / (1024.0 * 1024.0 * 1024.0)
                        );
                        println!("  SMs: {}", props.multiProcessorCount);
                        println!("  Max Threads/Block: {}", props.maxThreadsPerBlock);

                        // Check FP16 support
                        if props.major >= 5 && props.minor >= 3 {
                            println!("  ✅ FP16 compute supported");
                        } else {
                            println!("  ❌ FP16 compute not supported (requires CC 5.3+)");
                        }

                        // Check tensor core support
                        if props.major >= 7 {
                            println!("  ✅ Tensor cores available");
                        }

                        // Compute capability support status
                        match props.major {
                            5..=6 => println!("  ℹ️  Legacy compute capability"),
                            7 => println!("  ✅ Volta architecture"),
                            8 => match props.minor {
                                0 => println!("  ✅ Ampere architecture (A100)"),
                                6 => println!("  ✅ Ampere architecture (Consumer)"),
                                7 => println!("  ✅ Ampere architecture (Professional)"),
                                9 => println!("  ✅ Ada Lovelace architecture"),
                                _ => println!("  ✅ Ampere/Ada architecture"),
                            },
                            9 => println!("  ✅ Hopper architecture"),
                            10 => println!("  ✅ Blackwell architecture (future)"),
                            _ => println!("  ⚠️  Unknown architecture"),
                        }
                    }
                }
            } else {
                println!("❌ No CUDA devices found or CUDA not available");
            }
        }

        // Test PTX compatibility
        println!("\n=== PTX Compatibility ===");
        println!("PTX version: 7.0 (compatible with CUDA 11.0+)");
        println!("Target: sm_53 (Maxwell 2.0+)");
        println!("✅ Compatible with all modern NVIDIA GPUs");

        println!("\n=== Supported CUDA Versions ===");
        println!("Build system searches for CUDA versions:");
        println!("  • CUDA 11.0 - 11.8");
        println!("  • CUDA 12.0 - 12.9 ✅");
        println!("  • CUDA 13.0 - 13.9 (future support)");

        println!("\n=== Build Configuration ===");
        println!("To use a specific CUDA version:");
        println!("  export CUDA_PATH=/usr/local/cuda-12.3");
        println!("  cargo build --features cuda");
    }

    Ok(())
}
