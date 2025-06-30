//! Production GPU benchmark implementations for performance validation
//!
//! This module provides actual GPU benchmarks using CUDA for NVIDIA GPUs
//! and OpenCL for cross-platform support.

#[cfg(feature = "cuda")]
use crate::cuda_check;
use anyhow::Result;
#[cfg(feature = "cuda")]
use std::ptr;
#[cfg(feature = "opencl")]
use std::time::Instant;

/// GPU benchmark runner for performance measurements
pub struct GpuBenchmarkRunner {
    /// Index of the GPU to benchmark (supports multi-GPU setups)
    pub gpu_index: u32,
    /// Backend used for benchmarking
    pub backend: BenchmarkBackend,
}

#[derive(Debug, Clone)]
pub enum BenchmarkBackend {
    Cuda,
    OpenCL,
    Vulkan,
}

/// Results from GPU benchmarking
#[derive(Debug, Clone)]
pub struct GpuBenchmarkResult {
    pub gpu_index: u32,
    pub gpu_name: String,
    pub memory_bandwidth_gbps: f64,
    pub fp16_tflops: f64,
    pub backend: BenchmarkBackend,
}

impl GpuBenchmarkRunner {
    pub fn new(gpu_index: u32) -> Result<Self> {
        // Validate GPU index
        Self::validate_gpu_index(gpu_index)?;

        // Detect available backend
        let backend = Self::detect_backend()?;
        Ok(Self { gpu_index, backend })
    }

    /// Create a benchmark runner for a specific backend
    pub fn with_backend(gpu_index: u32, backend: BenchmarkBackend) -> Result<Self> {
        Self::validate_gpu_index(gpu_index)?;
        Ok(Self { gpu_index, backend })
    }

    /// Get the GPU index this runner is configured for
    pub fn gpu_index(&self) -> u32 {
        self.gpu_index
    }

    /// Validate that the GPU index exists
    fn validate_gpu_index(gpu_index: u32) -> Result<()> {
        use crate::gpu::GpuDetector;

        let detector = GpuDetector::new();
        let gpus = detector.detect()?;

        if gpu_index as usize >= gpus.len() {
            return Err(anyhow::anyhow!(
                "GPU index {} out of range. Found {} GPU(s)",
                gpu_index,
                gpus.len()
            ));
        }

        Ok(())
    }

    fn detect_backend() -> Result<BenchmarkBackend> {
        // Try CUDA first for NVIDIA GPUs
        #[cfg(feature = "cuda")]
        {
            if Self::cuda_available() {
                return Ok(BenchmarkBackend::Cuda);
            }
        }

        // Try OpenCL as fallback
        #[cfg(feature = "opencl")]
        {
            if Self::opencl_available() {
                return Ok(BenchmarkBackend::OpenCL);
            }
        }

        // Try Vulkan compute
        #[cfg(feature = "vulkan")]
        {
            if Self::vulkan_available() {
                return Ok(BenchmarkBackend::Vulkan);
            }
        }

        Err(anyhow::anyhow!("No GPU compute backend available"))
    }

    #[cfg(feature = "cuda")]
    fn cuda_available() -> bool {
        unsafe {
            use std::os::raw::c_int;

            #[link(name = "cuda")]
            extern "C" {
                fn cudaGetDeviceCount(count: *mut c_int) -> c_int;
            }

            let mut device_count: c_int = 0;
            cudaGetDeviceCount(&mut device_count) == 0 && device_count > 0
        }
    }

    #[cfg(feature = "opencl")]
    fn opencl_available() -> bool {
        ocl::Platform::list().into_iter().any(|_| true)
    }

    #[cfg(feature = "vulkan")]
    fn vulkan_available() -> bool {
        // Check for Vulkan compute support
        false // TODO: Implement Vulkan detection
    }

    #[cfg(feature = "opencl")]
    fn find_opencl_device(&self) -> Result<(ocl::Platform, ocl::Device)> {
        use ocl::{Device, DeviceType, Platform};

        let platforms = Platform::list();
        let mut gpu_count = 0;

        // Search all platforms for GPUs
        for platform in platforms {
            let devices = Device::list(platform, Some(DeviceType::GPU))?;

            for device in devices {
                if gpu_count == self.gpu_index {
                    return Ok((platform, device));
                }
                gpu_count += 1;
            }
        }

        Err(anyhow::anyhow!(
            "OpenCL GPU index {} not found. Total GPUs: {}",
            self.gpu_index,
            gpu_count
        ))
    }

    /// Benchmark memory bandwidth using actual GPU kernels
    /// Returns bandwidth in GB/s
    pub fn benchmark_memory_bandwidth(&self) -> Result<f64> {
        match self.backend {
            #[cfg(feature = "cuda")]
            BenchmarkBackend::Cuda => self.benchmark_memory_bandwidth_cuda(),
            #[cfg(feature = "opencl")]
            BenchmarkBackend::OpenCL => self.benchmark_memory_bandwidth_opencl(),
            #[cfg(feature = "vulkan")]
            BenchmarkBackend::Vulkan => self.benchmark_memory_bandwidth_vulkan(),
            _ => Err(anyhow::anyhow!("Backend not available")),
        }
    }

    /// Benchmark FP16 compute performance using actual matrix multiplication
    /// Returns performance in TFLOPS
    pub fn benchmark_fp16_compute(&self) -> Result<f64> {
        match self.backend {
            #[cfg(feature = "cuda")]
            BenchmarkBackend::Cuda => self.benchmark_fp16_compute_cuda(),
            #[cfg(feature = "opencl")]
            BenchmarkBackend::OpenCL => self.benchmark_fp16_compute_opencl(),
            #[cfg(feature = "vulkan")]
            BenchmarkBackend::Vulkan => self.benchmark_fp16_compute_vulkan(),
            _ => Err(anyhow::anyhow!("Backend not available")),
        }
    }

    /// Benchmark FP16 using cuBLAS tensor cores for Hopper/Blackwell
    #[cfg(feature = "cuda")]
    fn benchmark_fp16_tensor_core_cuda(&self) -> Result<f64> {
        use super::cuda_ffi::*;

        unsafe {
            cuda_check!(cudaSetDevice(self.gpu_index as i32));
        }

        // Create cuBLAS handle using the safe wrapper
        let mut handle = CublasHandle::new()?;

        // Enable tensor cores
        handle.set_math_mode(CUBLAS_TENSOR_OP_MATH)?;

        // Run FP16 tensor core benchmark
        let tflops = unsafe { benchmark_tensor_core_fp16(handle.as_ptr(), 8192, 8192, 8192, 50)? };

        Ok(tflops)
    }

    /// Benchmark FP16 using standard cuBLAS for Ada Lovelace and Ampere
    #[cfg(feature = "cuda")]
    fn benchmark_fp16_standard_cuda(&self) -> Result<f64> {
        use super::cuda_ffi::*;

        unsafe {
            cuda_check!(cudaSetDevice(self.gpu_index as i32));
        }

        // Create cuBLAS handle
        let mut handle = CublasHandle::new()?;

        // Enable tensor cores for architectures that support them
        handle.set_math_mode(CUBLAS_TENSOR_OP_MATH)?;

        // Use appropriate matrix sizes for different architectures
        let arch = super::detect_gpu_architecture(self.gpu_index as i32)?;
        let (m, n, k) = match arch {
            super::GpuArchitecture::AdaLovelace => (4096, 4096, 4096), // Smaller matrices for Ada
            super::GpuArchitecture::Ampere => (4096, 4096, 4096), // Smaller matrices for Ampere
            _ => (8192, 8192, 8192),                              // Default larger size
        };

        // Run FP16 benchmark with appropriate algorithm
        let algo = super::cuda_ffi::get_optimal_gemm_algo(self.gpu_index as i32)?;
        let tflops = unsafe { benchmark_fp16_gemm_with_algo(handle.as_ptr(), m, n, k, 50, algo)? };

        Ok(tflops)
    }

    /// Run benchmarks on all available GPUs
    pub fn benchmark_all_gpus() -> Result<Vec<GpuBenchmarkResult>> {
        use crate::gpu::GpuDetector;

        let detector = GpuDetector::new();
        let gpus = detector.detect()?;
        let mut results = Vec::new();

        for (idx, gpu) in gpus.iter().enumerate() {
            tracing::info!("Benchmarking GPU {}: {}", idx, gpu.name);

            match Self::new(idx as u32) {
                Ok(runner) => {
                    let memory_bandwidth =
                        runner.benchmark_memory_bandwidth().unwrap_or_else(|e| {
                            tracing::warn!("Memory bandwidth benchmark failed: {}", e);
                            0.0
                        });

                    let fp16_tflops = runner.benchmark_fp16_compute().unwrap_or_else(|e| {
                        tracing::warn!("FP16 compute benchmark failed: {}", e);
                        0.0
                    });

                    results.push(GpuBenchmarkResult {
                        gpu_index: idx as u32,
                        gpu_name: gpu.name.clone(),
                        memory_bandwidth_gbps: memory_bandwidth,
                        fp16_tflops,
                        backend: runner.backend.clone(),
                    });
                }
                Err(e) => {
                    tracing::error!("Failed to create benchmark runner for GPU {}: {}", idx, e);
                }
            }
        }

        Ok(results)
    }

    #[cfg(feature = "cuda")]
    fn benchmark_memory_bandwidth_cuda(&self) -> Result<f64> {
        use super::cuda_ffi::*;

        // Check CUDA version first
        if let Ok((major, minor)) = get_cuda_runtime_version() {
            tracing::info!("CUDA runtime version: {}.{}", major, minor);

            // Warn if using an untested version
            if major > 13 || (major == 13 && minor > 9) {
                tracing::warn!(
                    "Using CUDA {}.{} which is newer than tested versions (up to 13.9)",
                    major,
                    minor
                );
            }
        }

        unsafe {
            // Set device
            cuda_check!(cudaSetDevice(self.gpu_index as i32));

            // Get device properties to optimize buffer size
            let mut props = std::mem::zeroed::<cudaDeviceProp>();
            cudaGetDeviceProperties(&mut props, self.gpu_index as i32);

            // Use smaller buffer size for simpler synchronous benchmark (1GB max)
            let buffer_size = (props.totalGlobalMem as usize / 8).min(1024 * 1024 * 1024);

            // Allocate device memory using RAII wrappers
            let d_src = CudaBuffer::new(buffer_size)?;
            let mut d_dst = CudaBuffer::new(buffer_size)?;

            // Initialize source buffer with test pattern
            let pattern = vec![0xAB_u8; 4096];
            for offset in (0..buffer_size).step_by(pattern.len()) {
                let size = pattern.len().min(buffer_size - offset);
                cuda_check!(cudaMemcpy(
                    (d_src.as_ptr() as *mut u8).add(offset) as *mut _,
                    pattern.as_ptr() as *const _,
                    size,
                    cudaMemcpyKind::cudaMemcpyHostToDevice
                ));
            }

            // Warm up with synchronous operations
            for _ in 0..5 {
                cuda_check!(cudaMemcpy(
                    d_dst.as_mut_ptr(),
                    d_src.as_ptr(),
                    buffer_size,
                    cudaMemcpyKind::cudaMemcpyDeviceToDevice
                ));
            }

            // Benchmark with events for precise timing
            const NUM_ITERATIONS: u32 = 50;
            let mut events_start = vec![ptr::null_mut(); NUM_ITERATIONS as usize];
            let mut events_stop = vec![ptr::null_mut(); NUM_ITERATIONS as usize];

            // Create events for precise timing
            for i in 0..NUM_ITERATIONS as usize {
                cuda_check!(cudaEventCreate(&mut events_start[i]));
                cuda_check!(cudaEventCreate(&mut events_stop[i]));
            }

            // Run benchmark with events
            for i in 0..NUM_ITERATIONS as usize {
                cuda_check!(cudaEventRecord(events_start[i], ptr::null_mut()));

                // Synchronous memory copy
                cuda_check!(cudaMemcpy(
                    d_dst.as_mut_ptr(),
                    d_src.as_ptr(),
                    buffer_size,
                    cudaMemcpyKind::cudaMemcpyDeviceToDevice
                ));

                cuda_check!(cudaEventRecord(events_stop[i], ptr::null_mut()));
            }

            // Wait for all events to complete
            for event in events_stop.iter().take(NUM_ITERATIONS as usize) {
                cuda_check!(cudaEventSynchronize(*event));
            }

            // Calculate timings
            let mut total_time_ms = 0.0f32;
            for i in 0..NUM_ITERATIONS as usize {
                let mut time_ms = 0.0f32;
                cuda_check!(cudaEventElapsedTime(
                    &mut time_ms,
                    events_start[i],
                    events_stop[i]
                ));
                total_time_ms += time_ms;

                // Cleanup events
                cudaEventDestroy(events_start[i]);
                cudaEventDestroy(events_stop[i]);
            }

            // Calculate bandwidth
            // Each iteration does one read and one write
            let total_bytes = buffer_size as f64 * NUM_ITERATIONS as f64 * 2.0;
            let total_time_sec = (total_time_ms / 1000.0) as f64;
            let bandwidth_gbps = (total_bytes / 1e9) / total_time_sec;

            Ok(bandwidth_gbps)
        }
    }

    #[cfg(feature = "cuda")]
    fn benchmark_fp16_compute_cuda(&self) -> Result<f64> {
        use super::{detect_gpu_architecture, GpuArchitecture};

        // Detect GPU architecture
        let arch = detect_gpu_architecture(self.gpu_index as i32)?;

        match arch {
            GpuArchitecture::Hopper | GpuArchitecture::Blackwell => {
                // Use optimized tensor core benchmark for Hopper/Blackwell
                self.benchmark_fp16_tensor_core_cuda()
            }
            GpuArchitecture::AdaLovelace | GpuArchitecture::Ampere => {
                // Use standard cuBLAS FP16 benchmark for Ada Lovelace and Ampere
                self.benchmark_fp16_standard_cuda()
            }
            GpuArchitecture::Unknown(major, minor) => {
                // For unknown architectures, check if they support tensor cores
                if major >= 7 {
                    // Volta and newer support tensor cores
                    self.benchmark_fp16_standard_cuda()
                } else {
                    Err(anyhow::anyhow!(
                        "GPU architecture SM {}.{} does not support FP16 tensor cores",
                        major,
                        minor
                    ))
                }
            }
        }
    }

    #[cfg(feature = "opencl")]
    fn benchmark_memory_bandwidth_opencl(&self) -> Result<f64> {
        use ocl::{Buffer, Event, ProQue};

        // Find the correct GPU across all platforms
        let (platform, device) = self.find_opencl_device()?;

        // OpenCL kernel for bandwidth test
        const BANDWIDTH_KERNEL: &str = r#"
            __kernel void bandwidth_test(
                __global const float4* restrict src,
                __global float4* restrict dst,
                const int num_elements
            ) {
                int gid = get_global_id(0);
                if (gid < num_elements) {
                    // Use vector types for better bandwidth utilization
                    float4 data = src[gid];
                    dst[gid] = data;
                }
            }
        "#;

        // Build program
        let pro_que = ProQue::builder()
            .platform(platform)
            .device(device)
            .src(BANDWIDTH_KERNEL)
            .build()?;

        // Get device memory size
        let global_mem_size = device.info(ocl::enums::DeviceInfo::GlobalMemSize)?;
        let global_mem_size_bytes = match global_mem_size {
            ocl::enums::DeviceInfoResult::GlobalMemSize(size) => size,
            _ => return Err(anyhow::anyhow!("Failed to get device memory size")),
        };
        let buffer_size = (global_mem_size_bytes / 4).min(4 * 1024 * 1024 * 1024) as usize; // Max 4GB
        let num_elements = buffer_size / 16; // float4 is 16 bytes

        // Create buffers
        let src_buffer = Buffer::<f32>::builder()
            .queue(pro_que.queue().clone())
            .len(num_elements * 4)
            .build()?;

        let dst_buffer = Buffer::<f32>::builder()
            .queue(pro_que.queue().clone())
            .len(num_elements * 4)
            .build()?;

        // Initialize source buffer
        let init_data = vec![1.0f32; num_elements * 4];
        src_buffer.write(&init_data).enq()?;

        // Build kernel
        let kernel = pro_que
            .kernel_builder("bandwidth_test")
            .arg(&src_buffer)
            .arg(&dst_buffer)
            .arg(num_elements as i32)
            .build()?;

        // Warm up
        for _ in 0..10 {
            unsafe {
                kernel.enq()?;
            }
        }
        pro_que.queue().finish()?;

        // Benchmark
        const NUM_ITERATIONS: u32 = 100;
        let mut events = Vec::new();

        let start = Instant::now();

        for _ in 0..NUM_ITERATIONS {
            let mut event = Event::empty();
            unsafe {
                kernel.cmd().enew(&mut event).enq()?;
            }
            events.push(event);
        }

        // Wait for all events
        for event in &events {
            event.wait_for()?;
        }

        let elapsed = start.elapsed();

        // Use elapsed time instead of precise event timings to avoid enum matching issues
        let total_time_sec = elapsed.as_secs_f64();

        // Calculate bandwidth
        let total_bytes = buffer_size as f64 * NUM_ITERATIONS as f64 * 2.0;
        let bandwidth_gbps = (total_bytes / 1e9) / total_time_sec;

        Ok(bandwidth_gbps)
    }

    #[cfg(feature = "opencl")]
    fn benchmark_fp16_compute_opencl(&self) -> Result<f64> {
        use ocl::{Buffer, ProQue};

        // Find the correct GPU across all platforms
        let (platform, device) = self.find_opencl_device()?;

        // Check for cl_khr_fp16 extension
        let extensions = device.info(ocl::enums::DeviceInfo::Extensions)?;
        if !extensions.to_string().contains("cl_khr_fp16") {
            return Err(anyhow::anyhow!("Device does not support FP16"));
        }

        // FP16 GEMM kernel
        const FP16_GEMM_KERNEL: &str = r#"
            #pragma OPENCL EXTENSION cl_khr_fp16 : enable
            
            __kernel void fp16_gemm(
                __global const half* A,
                __global const half* B,
                __global half* C,
                const int M,
                const int N,
                const int K
            ) {
                const int row = get_global_id(1);
                const int col = get_global_id(0);
                
                if (row < M && col < N) {
                    half sum = 0.0h;
                    
                    for (int k = 0; k < K; k++) {
                        half a = A[row * K + k];
                        half b = B[k * N + col];
                        sum = fma(a, b, sum);
                    }
                    
                    C[row * N + col] = sum;
                }
            }
        "#;

        // Build program
        let pro_que = ProQue::builder()
            .platform(platform)
            .device(device)
            .src(FP16_GEMM_KERNEL)
            .build()?;

        // Matrix dimensions
        const M: usize = 4096;
        const N: usize = 4096;
        const K: usize = 4096;

        // Create buffers (using u16 to represent half)
        let a_buffer = Buffer::<u16>::builder()
            .queue(pro_que.queue().clone())
            .len(M * K)
            .build()?;

        let b_buffer = Buffer::<u16>::builder()
            .queue(pro_que.queue().clone())
            .len(K * N)
            .build()?;

        let c_buffer = Buffer::<u16>::builder()
            .queue(pro_que.queue().clone())
            .len(M * N)
            .build()?;

        // Initialize with random data (simplified)
        let init_a = vec![0x3C00u16; M * K]; // 1.0 in FP16
        let init_b = vec![0x3C00u16; K * N]; // 1.0 in FP16
        a_buffer.write(&init_a).enq()?;
        b_buffer.write(&init_b).enq()?;

        // Build kernel
        let kernel = pro_que
            .kernel_builder("fp16_gemm")
            .arg(&a_buffer)
            .arg(&b_buffer)
            .arg(&c_buffer)
            .arg(M as i32)
            .arg(N as i32)
            .arg(K as i32)
            .global_work_size([N, M])
            .build()?;

        // Warm up
        for _ in 0..10 {
            unsafe {
                kernel.enq()?;
            }
        }
        pro_que.queue().finish()?;

        // Benchmark
        const NUM_ITERATIONS: u32 = 100;
        let start = Instant::now();

        for _ in 0..NUM_ITERATIONS {
            unsafe {
                kernel.enq()?;
            }
        }

        pro_que.queue().finish()?;
        let elapsed = start.elapsed();

        // Calculate TFLOPS
        let total_flops = 2.0 * M as f64 * N as f64 * K as f64 * NUM_ITERATIONS as f64;
        let tflops = total_flops / (elapsed.as_secs_f64() * 1e12);

        Ok(tflops)
    }

    #[cfg(feature = "vulkan")]
    fn benchmark_memory_bandwidth_vulkan(&self) -> Result<f64> {
        // TODO: Implement Vulkan compute benchmarks
        Err(anyhow::anyhow!("Vulkan benchmarks not yet implemented"))
    }

    #[cfg(feature = "vulkan")]
    fn benchmark_fp16_compute_vulkan(&self) -> Result<f64> {
        // TODO: Implement Vulkan FP16 benchmarks
        Err(anyhow::anyhow!(
            "Vulkan FP16 benchmarks not yet implemented"
        ))
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_benchmark_runner_creation() {
        let result = GpuBenchmarkRunner::new(0);
        // May fail if no GPU backend is available
        if let Ok(runner) = result {
            println!(
                "Created benchmark runner with backend: {:?}",
                runner.backend
            );
        }
    }

    #[test]
    #[ignore] // Requires GPU
    fn test_memory_bandwidth_benchmark() {
        let runner = GpuBenchmarkRunner::new(0).unwrap();
        let bandwidth = runner.benchmark_memory_bandwidth().unwrap();

        println!("Measured memory bandwidth: {bandwidth:.2} GB/s");

        // Sanity check - should be between 10 GB/s and 10 TB/s
        assert!(bandwidth > 10.0);
        assert!(bandwidth < 10000.0);
    }

    #[test]
    #[ignore] // Requires GPU with FP16 support
    fn test_fp16_compute_benchmark() {
        let runner = GpuBenchmarkRunner::new(0).unwrap();
        let tflops = runner.benchmark_fp16_compute().unwrap();

        println!("Measured FP16 performance: {tflops:.2} TFLOPS");

        // Sanity check - should be between 1 and 5000 TFLOPS
        assert!(tflops > 1.0);
        assert!(tflops < 5000.0);
    }
}
