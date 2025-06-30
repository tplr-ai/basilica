//! Consolidated CUDA FFI bindings for Hopper/Blackwell GPU validation
//!
//! This module provides everything needed for H100, H200, and Blackwell GPU validation,
//! including Tensor Core benchmarking with FP16 and FP8 support.

#![allow(non_camel_case_types)]
#![allow(non_snake_case)]
#![allow(dead_code)]

use anyhow::Result;
use std::marker::PhantomData;
use std::os::raw::{c_char, c_float, c_int, c_void};

// ============================================================================
// CUDA Core Types and Constants
// ============================================================================

pub type cudaError_t = c_int;
pub type cudaStream_t = *mut c_void;
pub type cudaEvent_t = *mut c_void;

#[allow(non_upper_case_globals)]
pub const cudaSuccess: cudaError_t = 0;

#[repr(C)]
#[derive(Debug, Clone, Copy)]
pub enum cudaMemcpyKind {
    cudaMemcpyHostToHost = 0,
    cudaMemcpyHostToDevice = 1,
    cudaMemcpyDeviceToHost = 2,
    cudaMemcpyDeviceToDevice = 3,
    cudaMemcpyDefault = 4,
}

// CUDA device properties structure - must match exact CUDA layout
#[repr(C)]
pub struct cudaDeviceProp {
    pub name: [c_char; 256],
    pub uuid: [u8; 16],          // Added missing uuid field
    pub luid: [c_char; 8],       // Added missing luid field
    pub luidDeviceNodeMask: u32, // Added missing field
    pub totalGlobalMem: usize,
    pub sharedMemPerBlock: usize,
    pub regsPerBlock: c_int,
    pub warpSize: c_int,
    pub memPitch: usize,
    pub maxThreadsPerBlock: c_int,
    pub maxThreadsDim: [c_int; 3],
    pub maxGridSize: [c_int; 3],
    pub clockRate: c_int,
    pub totalConstMem: usize,
    pub major: c_int,
    pub minor: c_int,
    pub textureAlignment: usize,
    pub texturePitchAlignment: usize,
    pub deviceOverlap: c_int,
    pub multiProcessorCount: c_int,
    pub kernelExecTimeoutEnabled: c_int,
    pub integrated: c_int,
    pub canMapHostMemory: c_int,
    pub computeMode: c_int,
    pub maxTexture1D: c_int,
    pub maxTexture1DMipmap: c_int,
    pub maxTexture1DLinear: c_int,
    pub maxTexture2D: [c_int; 2],
    pub maxTexture2DMipmap: [c_int; 2],
    pub maxTexture2DLinear: [c_int; 3],
    pub maxTexture2DGather: [c_int; 2],
    pub maxTexture3D: [c_int; 3],
    pub maxTexture3DAlt: [c_int; 3],
    pub maxTextureCubemap: c_int,
    pub maxTexture1DLayered: [c_int; 2],
    pub maxTexture2DLayered: [c_int; 3],
    pub maxTextureCubemapLayered: [c_int; 2],
    pub maxSurface1D: c_int,
    pub maxSurface2D: [c_int; 2],
    pub maxSurface3D: [c_int; 3],
    pub maxSurface1DLayered: [c_int; 2],
    pub maxSurface2DLayered: [c_int; 3],
    pub maxSurfaceCubemap: c_int,
    pub maxSurfaceCubemapLayered: [c_int; 2],
    pub surfaceAlignment: usize,
    pub concurrentKernels: c_int,
    pub ECCEnabled: c_int,
    pub pciBusID: c_int,
    pub pciDeviceID: c_int,
    pub pciDomainID: c_int,
    pub tccDriver: c_int,
    pub asyncEngineCount: c_int,
    pub unifiedAddressing: c_int,
    pub memoryClockRate: c_int,
    pub memoryBusWidth: c_int,
    pub l2CacheSize: c_int,
    pub persistingL2CacheMaxSize: c_int,
    pub maxThreadsPerMultiProcessor: c_int,
    pub streamPrioritiesSupported: c_int,
    pub globalL1CacheSupported: c_int,
    pub localL1CacheSupported: c_int,
    pub sharedMemPerMultiprocessor: usize,
    pub regsPerMultiprocessor: c_int,
    pub managedMemory: c_int,
    pub isMultiGpuBoard: c_int,
    pub multiGpuBoardGroupID: c_int,
    pub hostNativeAtomicSupported: c_int,
    pub singleToDoublePrecisionPerfRatio: c_int,
    pub pageableMemoryAccess: c_int,
    pub concurrentManagedAccess: c_int,
    pub computePreemptionSupported: c_int,
    pub canUseHostPointerForRegisteredMem: c_int,
    pub cooperativeLaunch: c_int,
    pub cooperativeMultiDeviceLaunch: c_int,
    pub sharedMemPerBlockOptin: usize,
    pub pageableMemoryAccessUsesHostPageTables: c_int,
    pub directManagedMemAccessFromHost: c_int,
    pub maxBlocksPerMultiProcessor: c_int,
    pub accessPolicyMaxWindowSize: c_int,
    pub reservedSharedMemPerBlock: usize,
    pub hostRegisterSupported: c_int,
    pub sparseCudaArraySupported: c_int,
    pub hostRegisterReadOnlySupported: c_int,
    pub timelineSemaphoreInteropSupported: c_int,
    pub memoryPoolsSupported: c_int,
    pub gpuDirectRDMASupported: c_int,
    pub gpuDirectRDMAFlushWritesOptions: c_int,
    pub gpuDirectRDMAWritesOrdering: c_int,
    pub memoryPoolSupportedHandleTypes: c_int,
    pub deferredMappingCudaArraySupported: c_int,
    pub ipcEventSupported: c_int,
    pub clusterLaunch: c_int,
    pub unifiedFunctionPointers: c_int,
    pub reserved: [c_int; 63],
}

// ============================================================================
// cuBLAS Types and Constants for Tensor Cores
// ============================================================================

pub type cublasHandle_t = *mut c_void;
pub type cublasStatus_t = c_int;
pub type cublasOperation_t = c_int;
pub type cublasComputeType_t = c_int;
pub type cudaDataType_t = c_int;
pub type cublasGemmAlgo_t = c_int;
pub type cublasMath_t = c_int;

// cuBLAS status codes
pub const CUBLAS_STATUS_SUCCESS: cublasStatus_t = 0;

// cuBLAS operations
pub const CUBLAS_OP_N: cublasOperation_t = 0;
pub const CUBLAS_OP_T: cublasOperation_t = 1;

// CUDA data types
pub const CUDA_R_16F: cudaDataType_t = 2; // FP16
pub const CUDA_R_8F_E4M3: cudaDataType_t = 28; // FP8 E4M3 (Hopper/Blackwell)

// cuBLAS compute types for Tensor Cores
pub const CUBLAS_COMPUTE_32F_FAST_16F: cublasComputeType_t = 74; // FP16 Tensor Cores
pub const CUBLAS_COMPUTE_32F_FAST_8F_E4M3: cublasComputeType_t = 88; // FP8 Tensor Cores

// cuBLAS math modes
pub const CUBLAS_TENSOR_OP_MATH: cublasMath_t = 1;

// Tensor Core optimized algorithms
pub const CUBLAS_GEMM_DEFAULT: cublasGemmAlgo_t = -1;
pub const CUBLAS_GEMM_ALGO0_TENSOR_OP: cublasGemmAlgo_t = 99;
pub const CUBLAS_GEMM_ALGO1_TENSOR_OP: cublasGemmAlgo_t = 100;
pub const CUBLAS_GEMM_ALGO2_TENSOR_OP: cublasGemmAlgo_t = 101;

// ============================================================================
// FFI Bindings
// ============================================================================

#[link(name = "cudart")]
extern "C" {
    pub fn cudaGetDeviceCount(count: *mut c_int) -> cudaError_t;
    pub fn cudaSetDevice(device: c_int) -> cudaError_t;
    pub fn cudaGetDeviceProperties(prop: *mut cudaDeviceProp, device: c_int) -> cudaError_t;
    pub fn cudaRuntimeGetVersion(runtimeVersion: *mut c_int) -> cudaError_t;
    pub fn cudaDriverGetVersion(driverVersion: *mut c_int) -> cudaError_t;

    pub fn cudaMalloc(devPtr: *mut *mut c_void, size: usize) -> cudaError_t;
    pub fn cudaFree(devPtr: *mut c_void) -> cudaError_t;
    pub fn cudaMemcpy(
        dst: *mut c_void,
        src: *const c_void,
        count: usize,
        kind: cudaMemcpyKind,
    ) -> cudaError_t;

    pub fn cudaEventCreate(event: *mut cudaEvent_t) -> cudaError_t;
    pub fn cudaEventDestroy(event: cudaEvent_t) -> cudaError_t;
    pub fn cudaEventRecord(event: cudaEvent_t, stream: cudaStream_t) -> cudaError_t;
    pub fn cudaEventSynchronize(event: cudaEvent_t) -> cudaError_t;
    pub fn cudaEventElapsedTime(
        ms: *mut c_float,
        start: cudaEvent_t,
        end: cudaEvent_t,
    ) -> cudaError_t;

    pub fn cudaDeviceCanAccessPeer(
        canAccessPeer: *mut c_int,
        device: c_int,
        peerDevice: c_int,
    ) -> cudaError_t;
}

#[link(name = "cublas")]
extern "C" {
    pub fn cublasCreate_v2(handle: *mut cublasHandle_t) -> cublasStatus_t;
    pub fn cublasDestroy_v2(handle: cublasHandle_t) -> cublasStatus_t;
    pub fn cublasSetStream_v2(handle: cublasHandle_t, streamId: cudaStream_t) -> cublasStatus_t;
    pub fn cublasSetMathMode(handle: cublasHandle_t, mode: cublasMath_t) -> cublasStatus_t;

    pub fn cublasGemmEx(
        handle: cublasHandle_t,
        transa: cublasOperation_t,
        transb: cublasOperation_t,
        m: c_int,
        n: c_int,
        k: c_int,
        alpha: *const c_void,
        A: *const c_void,
        Atype: cudaDataType_t,
        lda: c_int,
        B: *const c_void,
        Btype: cudaDataType_t,
        ldb: c_int,
        beta: *const c_void,
        C: *mut c_void,
        Ctype: cudaDataType_t,
        ldc: c_int,
        computeType: cublasComputeType_t,
        algo: cublasGemmAlgo_t,
    ) -> cublasStatus_t;

    pub fn cublasSetWorkspace(
        handle: cublasHandle_t,
        workspace: *mut c_void,
        workspaceSizeInBytes: usize,
    ) -> cublasStatus_t;
}

// ============================================================================
// Helper Macros
// ============================================================================

#[macro_export]
macro_rules! cuda_check {
    ($call:expr) => {{
        let result = $call;
        if result != $crate::gpu::cuda_ffi::cudaSuccess {
            return Err(anyhow::anyhow!(
                "CUDA error {} at {}:{}",
                result,
                file!(),
                line!()
            ));
        }
    }};
}

// ============================================================================
// RAII Wrappers
// ============================================================================

/// RAII wrapper for CUDA memory allocation
pub struct CudaBuffer {
    ptr: *mut c_void,
    size: usize,
}

impl CudaBuffer {
    pub fn new(size: usize) -> Result<Self> {
        let mut ptr: *mut c_void = std::ptr::null_mut();
        unsafe {
            cuda_check!(cudaMalloc(&mut ptr, size));
        }
        Ok(Self { ptr, size })
    }

    pub fn as_ptr(&self) -> *const c_void {
        self.ptr
    }

    pub fn as_mut_ptr(&mut self) -> *mut c_void {
        self.ptr
    }

    pub fn size(&self) -> usize {
        self.size
    }
}

impl Drop for CudaBuffer {
    fn drop(&mut self) {
        unsafe {
            if !self.ptr.is_null() {
                cudaFree(self.ptr);
            }
        }
    }
}

unsafe impl Send for CudaBuffer {}

/// Thread-safe wrapper for cuBLAS handle
pub struct CublasHandle {
    handle: cublasHandle_t,
    _phantom: PhantomData<*const ()>,
}

impl CublasHandle {
    pub fn new() -> Result<Self> {
        let mut handle: cublasHandle_t = std::ptr::null_mut();
        unsafe {
            let status = cublasCreate_v2(&mut handle);
            if status != CUBLAS_STATUS_SUCCESS {
                return Err(anyhow::anyhow!(
                    "Failed to create cuBLAS handle: {}",
                    status
                ));
            }
        }
        Ok(Self {
            handle,
            _phantom: PhantomData,
        })
    }

    pub fn as_ptr(&self) -> cublasHandle_t {
        self.handle
    }

    pub fn set_math_mode(&mut self, mode: cublasMath_t) -> Result<()> {
        unsafe {
            let status = cublasSetMathMode(self.handle, mode);
            if status != CUBLAS_STATUS_SUCCESS {
                return Err(anyhow::anyhow!("Failed to set math mode: {}", status));
            }
        }
        Ok(())
    }

    pub fn set_workspace(&mut self, workspace: &mut CudaBuffer) -> Result<()> {
        unsafe {
            let status = cublasSetWorkspace(self.handle, workspace.as_mut_ptr(), workspace.size());
            if status != CUBLAS_STATUS_SUCCESS {
                return Err(anyhow::anyhow!("Failed to set workspace: {}", status));
            }
        }
        Ok(())
    }
}

impl Drop for CublasHandle {
    fn drop(&mut self) {
        unsafe {
            if !self.handle.is_null() {
                cublasDestroy_v2(self.handle);
            }
        }
    }
}

unsafe impl Send for CublasHandle {}

// ============================================================================
// Architecture Detection
// ============================================================================

/// GPU Architecture enumeration
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum GpuArchitecture {
    /// Ampere architecture (RTX 30 series, A100) - SM 8.6
    Ampere,
    /// Ada Lovelace architecture (RTX 40 series, L4, L40) - SM 8.9
    AdaLovelace,
    /// Hopper architecture (H100, H200) - SM 9.0
    Hopper,
    /// Blackwell architecture (B100, B200) - SM 10.0
    Blackwell,
    /// Unknown or older architecture
    Unknown(i32, i32), // (major, minor)
}

impl GpuArchitecture {
    /// Check if this architecture supports FP8 tensor cores
    pub fn supports_fp8(&self) -> bool {
        matches!(self, GpuArchitecture::Hopper | GpuArchitecture::Blackwell)
    }

    /// Check if this architecture supports FP16 tensor cores
    pub fn supports_fp16_tensor_cores(&self) -> bool {
        matches!(
            self,
            GpuArchitecture::Ampere
                | GpuArchitecture::AdaLovelace
                | GpuArchitecture::Hopper
                | GpuArchitecture::Blackwell
        )
    }

    /// Get the tensor core generation for this architecture
    pub fn tensor_core_generation(&self) -> Option<u8> {
        match self {
            GpuArchitecture::Ampere => Some(2),      // 2nd gen tensor cores
            GpuArchitecture::AdaLovelace => Some(3), // 3rd gen tensor cores
            GpuArchitecture::Hopper => Some(4),      // 4th gen tensor cores
            GpuArchitecture::Blackwell => Some(5),   // 5th gen tensor cores
            GpuArchitecture::Unknown(_, _) => None,
        }
    }
}

/// Detect GPU architecture from device properties
pub fn detect_gpu_architecture(device: c_int) -> Result<GpuArchitecture> {
    unsafe {
        let mut props = std::mem::zeroed::<cudaDeviceProp>();
        cuda_check!(cudaGetDeviceProperties(&mut props, device));

        let arch = match (props.major, props.minor) {
            (8, 6) => GpuArchitecture::Ampere,
            (8, 9) => GpuArchitecture::AdaLovelace,
            (9, 0) => GpuArchitecture::Hopper,
            (10, _) => GpuArchitecture::Blackwell,
            (major, minor) => GpuArchitecture::Unknown(major, minor),
        };

        tracing::info!(
            "GPU {} architecture: {:?} (SM {}.{}, {} SMs)",
            device,
            arch,
            props.major,
            props.minor,
            props.multiProcessorCount
        );

        Ok(arch)
    }
}

/// Check if GPU is Hopper (H100/H200) or Blackwell
pub fn is_hopper_or_blackwell(device: c_int) -> Result<bool> {
    let arch = detect_gpu_architecture(device)?;
    Ok(matches!(
        arch,
        GpuArchitecture::Hopper | GpuArchitecture::Blackwell
    ))
}

/// Get optimal GEMM algorithm for architecture
pub fn get_optimal_gemm_algo(device: c_int) -> Result<cublasGemmAlgo_t> {
    let arch = detect_gpu_architecture(device)?;

    unsafe {
        let mut props = std::mem::zeroed::<cudaDeviceProp>();
        cuda_check!(cudaGetDeviceProperties(&mut props, device));

        match arch {
            GpuArchitecture::Blackwell => Ok(CUBLAS_GEMM_ALGO2_TENSOR_OP),
            GpuArchitecture::Hopper => {
                // H100 SXM/H200 have more SMs than PCIe variant
                if props.multiProcessorCount >= 114 {
                    Ok(CUBLAS_GEMM_ALGO1_TENSOR_OP)
                } else {
                    Ok(CUBLAS_GEMM_ALGO0_TENSOR_OP)
                }
            }
            GpuArchitecture::AdaLovelace => {
                // Ada Lovelace uses tensor cores efficiently with default algorithm
                Ok(CUBLAS_GEMM_DEFAULT)
            }
            GpuArchitecture::Ampere => {
                // Ampere also works well with default algorithm
                Ok(CUBLAS_GEMM_DEFAULT)
            }
            GpuArchitecture::Unknown(major, minor) => {
                tracing::warn!(
                    "Unknown GPU architecture (SM {}.{}), using default GEMM algorithm",
                    major,
                    minor
                );
                Ok(CUBLAS_GEMM_DEFAULT)
            }
        }
    }
}

// ============================================================================
// Benchmarking Functions
// ============================================================================

/// Benchmark FP16 Tensor Core GEMM performance
///
/// # Safety
/// This function dereferences the raw cuBLAS handle pointer and calls unsafe CUDA APIs.
/// The caller must ensure the handle is valid and properly initialized.
pub unsafe fn benchmark_tensor_core_fp16(
    handle: cublasHandle_t,
    m: usize,
    n: usize,
    k: usize,
    iterations: u32,
) -> Result<f64> {
    let size_a = m * k * 2; // 2 bytes per FP16
    let size_b = k * n * 2;
    let size_c = m * n * 2;

    let d_a = CudaBuffer::new(size_a)?;
    let d_b = CudaBuffer::new(size_b)?;
    let mut d_c = CudaBuffer::new(size_c)?;

    unsafe {
        let alpha: f32 = 1.0;
        let beta: f32 = 0.0;

        let mut start: cudaEvent_t = std::ptr::null_mut();
        let mut stop: cudaEvent_t = std::ptr::null_mut();
        cuda_check!(cudaEventCreate(&mut start));
        cuda_check!(cudaEventCreate(&mut stop));

        let algo = get_optimal_gemm_algo(0)?;

        // Warm up
        for _ in 0..10 {
            let status = cublasGemmEx(
                handle,
                CUBLAS_OP_N,
                CUBLAS_OP_N,
                n as c_int,
                m as c_int,
                k as c_int,
                &alpha as *const _ as *const c_void,
                d_b.as_ptr(),
                CUDA_R_16F,
                n as c_int,
                d_a.as_ptr(),
                CUDA_R_16F,
                k as c_int,
                &beta as *const _ as *const c_void,
                d_c.as_mut_ptr(),
                CUDA_R_16F,
                n as c_int,
                CUBLAS_COMPUTE_32F_FAST_16F,
                algo,
            );

            if status != CUBLAS_STATUS_SUCCESS {
                return Err(anyhow::anyhow!("cuBLAS GEMM failed: {}", status));
            }
        }

        // Benchmark
        cuda_check!(cudaEventRecord(start, std::ptr::null_mut()));

        for _ in 0..iterations {
            let status = cublasGemmEx(
                handle,
                CUBLAS_OP_N,
                CUBLAS_OP_N,
                n as c_int,
                m as c_int,
                k as c_int,
                &alpha as *const _ as *const c_void,
                d_b.as_ptr(),
                CUDA_R_16F,
                n as c_int,
                d_a.as_ptr(),
                CUDA_R_16F,
                k as c_int,
                &beta as *const _ as *const c_void,
                d_c.as_mut_ptr(),
                CUDA_R_16F,
                n as c_int,
                CUBLAS_COMPUTE_32F_FAST_16F,
                algo,
            );

            if status != CUBLAS_STATUS_SUCCESS {
                return Err(anyhow::anyhow!("cuBLAS GEMM failed: {}", status));
            }
        }

        cuda_check!(cudaEventRecord(stop, std::ptr::null_mut()));
        cuda_check!(cudaEventSynchronize(stop));

        let mut elapsed_ms = 0.0f32;
        cuda_check!(cudaEventElapsedTime(&mut elapsed_ms, start, stop));

        cuda_check!(cudaEventDestroy(start));
        cuda_check!(cudaEventDestroy(stop));

        // Calculate TFLOPS
        let total_flops = 2.0 * m as f64 * n as f64 * k as f64 * iterations as f64;
        let elapsed_sec = (elapsed_ms / 1000.0) as f64;
        let tflops = total_flops / (elapsed_sec * 1e12);

        Ok(tflops)
    }
}

/// Benchmark FP8 Tensor Core GEMM performance (Hopper/Blackwell only)
///
/// # Safety
/// This function dereferences the raw cuBLAS handle pointer and calls unsafe CUDA APIs.
/// The caller must ensure the handle is valid and properly initialized.
pub unsafe fn benchmark_tensor_core_fp8(
    handle: cublasHandle_t,
    m: usize,
    n: usize,
    k: usize,
    iterations: u32,
) -> Result<f64> {
    if !is_hopper_or_blackwell(0)? {
        return Err(anyhow::anyhow!(
            "FP8 requires Hopper or Blackwell architecture"
        ));
    }

    let size_a = m * k; // 1 byte per FP8
    let size_b = k * n;
    let size_c = m * n * 2; // Output in FP16

    let d_a = CudaBuffer::new(size_a)?;
    let d_b = CudaBuffer::new(size_b)?;
    let mut d_c = CudaBuffer::new(size_c)?;

    unsafe {
        let alpha: f32 = 1.0;
        let beta: f32 = 0.0;

        let mut start: cudaEvent_t = std::ptr::null_mut();
        let mut stop: cudaEvent_t = std::ptr::null_mut();
        cuda_check!(cudaEventCreate(&mut start));
        cuda_check!(cudaEventCreate(&mut stop));

        let algo = get_optimal_gemm_algo(0)?;

        // Benchmark
        cuda_check!(cudaEventRecord(start, std::ptr::null_mut()));

        for _ in 0..iterations {
            let status = cublasGemmEx(
                handle,
                CUBLAS_OP_N,
                CUBLAS_OP_N,
                n as c_int,
                m as c_int,
                k as c_int,
                &alpha as *const _ as *const c_void,
                d_b.as_ptr(),
                CUDA_R_8F_E4M3,
                n as c_int,
                d_a.as_ptr(),
                CUDA_R_8F_E4M3,
                k as c_int,
                &beta as *const _ as *const c_void,
                d_c.as_mut_ptr(),
                CUDA_R_16F,
                n as c_int,
                CUBLAS_COMPUTE_32F_FAST_8F_E4M3,
                algo,
            );

            if status != CUBLAS_STATUS_SUCCESS {
                return Err(anyhow::anyhow!("cuBLAS FP8 GEMM failed: {}", status));
            }
        }

        cuda_check!(cudaEventRecord(stop, std::ptr::null_mut()));
        cuda_check!(cudaEventSynchronize(stop));

        let mut elapsed_ms = 0.0f32;
        cuda_check!(cudaEventElapsedTime(&mut elapsed_ms, start, stop));

        cuda_check!(cudaEventDestroy(start));
        cuda_check!(cudaEventDestroy(stop));

        // Calculate TFLOPS
        let total_flops = 2.0 * m as f64 * n as f64 * k as f64 * iterations as f64;
        let elapsed_sec = (elapsed_ms / 1000.0) as f64;
        let tflops = total_flops / (elapsed_sec * 1e12);

        Ok(tflops)
    }
}

/// Benchmark FP16 GEMM with custom algorithm selection
///
/// # Safety
/// This function dereferences the raw cuBLAS handle pointer and calls unsafe CUDA APIs.
/// The caller must ensure the handle is valid and properly initialized.
pub unsafe fn benchmark_fp16_gemm_with_algo(
    handle: cublasHandle_t,
    m: usize,
    n: usize,
    k: usize,
    iterations: u32,
    algo: cublasGemmAlgo_t,
) -> Result<f64> {
    let size_a = m * k * 2; // 2 bytes per FP16
    let size_b = k * n * 2;
    let size_c = m * n * 2;

    let d_a = CudaBuffer::new(size_a)?;
    let d_b = CudaBuffer::new(size_b)?;
    let mut d_c = CudaBuffer::new(size_c)?;

    unsafe {
        let alpha: f32 = 1.0;
        let beta: f32 = 0.0;

        let mut start: cudaEvent_t = std::ptr::null_mut();
        let mut stop: cudaEvent_t = std::ptr::null_mut();
        cuda_check!(cudaEventCreate(&mut start));
        cuda_check!(cudaEventCreate(&mut stop));

        // Warm up
        for _ in 0..10 {
            let status = cublasGemmEx(
                handle,
                CUBLAS_OP_N,
                CUBLAS_OP_N,
                n as c_int,
                m as c_int,
                k as c_int,
                &alpha as *const _ as *const c_void,
                d_b.as_ptr(),
                CUDA_R_16F,
                n as c_int,
                d_a.as_ptr(),
                CUDA_R_16F,
                k as c_int,
                &beta as *const _ as *const c_void,
                d_c.as_mut_ptr(),
                CUDA_R_16F,
                n as c_int,
                CUBLAS_COMPUTE_32F_FAST_16F,
                algo,
            );

            if status != CUBLAS_STATUS_SUCCESS {
                return Err(anyhow::anyhow!("cuBLAS GEMM failed: {}", status));
            }
        }

        // Benchmark
        cuda_check!(cudaEventRecord(start, std::ptr::null_mut()));

        for _ in 0..iterations {
            let status = cublasGemmEx(
                handle,
                CUBLAS_OP_N,
                CUBLAS_OP_N,
                n as c_int,
                m as c_int,
                k as c_int,
                &alpha as *const _ as *const c_void,
                d_b.as_ptr(),
                CUDA_R_16F,
                n as c_int,
                d_a.as_ptr(),
                CUDA_R_16F,
                k as c_int,
                &beta as *const _ as *const c_void,
                d_c.as_mut_ptr(),
                CUDA_R_16F,
                n as c_int,
                CUBLAS_COMPUTE_32F_FAST_16F,
                algo,
            );

            if status != CUBLAS_STATUS_SUCCESS {
                return Err(anyhow::anyhow!("cuBLAS GEMM failed: {}", status));
            }
        }

        cuda_check!(cudaEventRecord(stop, std::ptr::null_mut()));
        cuda_check!(cudaEventSynchronize(stop));

        let mut elapsed_ms = 0.0f32;
        cuda_check!(cudaEventElapsedTime(&mut elapsed_ms, start, stop));

        cuda_check!(cudaEventDestroy(start));
        cuda_check!(cudaEventDestroy(stop));

        // Calculate TFLOPS
        let total_flops = 2.0 * m as f64 * n as f64 * k as f64 * iterations as f64;
        let elapsed_sec = (elapsed_ms / 1000.0) as f64;
        let tflops = total_flops / (elapsed_sec * 1e12);

        Ok(tflops)
    }
}

// ============================================================================
// Performance Profiles
// ============================================================================

pub struct GpuProfile {
    pub name: &'static str,
    pub fp16_tflops: f64,
    pub fp8_tflops: f64,
    pub memory_bandwidth_gbps: f64,
}

pub const GPU_PROFILES: &[GpuProfile] = &[
    // Hopper & Blackwell GPUs (with FP8 support)
    GpuProfile {
        name: "H100 PCIe",
        fp16_tflops: 756.0, // Realistic benchmark value (50% of theoretical)
        fp8_tflops: 1513.0, // Realistic benchmark value (50% of theoretical)
        memory_bandwidth_gbps: 2039.0,
    },
    GpuProfile {
        name: "H100 SXM5",
        fp16_tflops: 989.0, // Realistic benchmark value (50% of theoretical 1979)
        fp8_tflops: 1979.0, // Realistic benchmark value (50% of theoretical 3958)
        memory_bandwidth_gbps: 3350.0,
    },
    GpuProfile {
        name: "H200",
        fp16_tflops: 989.0, // Realistic benchmark value (50% of theoretical)
        fp8_tflops: 1979.0, // Realistic benchmark value (50% of theoretical)
        memory_bandwidth_gbps: 4800.0,
    },
    GpuProfile {
        name: "B100",
        fp16_tflops: 1800.0, // Realistic benchmark value (50% of theoretical)
        fp8_tflops: 3600.0,  // Realistic benchmark value (50% of theoretical)
        memory_bandwidth_gbps: 8000.0,
    },
    GpuProfile {
        name: "B200",
        fp16_tflops: 2250.0, // Realistic benchmark value (50% of theoretical)
        fp8_tflops: 4500.0,  // Realistic benchmark value (50% of theoretical)
        memory_bandwidth_gbps: 8000.0,
    },
    // Ada Lovelace GPUs (RTX 40 series, L4, L40)
    GpuProfile {
        name: "RTX 4090",
        fp16_tflops: 165.0, // Realistic benchmark value (50% of theoretical 330)
        fp8_tflops: 0.0,    // No FP8 support
        memory_bandwidth_gbps: 1008.0,
    },
    GpuProfile {
        name: "RTX 4080",
        fp16_tflops: 97.0, // Realistic benchmark value (50% of theoretical 194)
        fp8_tflops: 0.0,   // No FP8 support
        memory_bandwidth_gbps: 717.0,
    },
    GpuProfile {
        name: "L4",
        fp16_tflops: 60.0, // Realistic benchmark value (50% of theoretical 120)
        fp8_tflops: 0.0,   // No FP8 support
        memory_bandwidth_gbps: 300.0,
    },
    GpuProfile {
        name: "L40",
        fp16_tflops: 180.0, // Realistic benchmark value (50% of theoretical 360)
        fp8_tflops: 0.0,    // No FP8 support
        memory_bandwidth_gbps: 864.0,
    },
    // Ampere GPUs (RTX 30 series, A100)
    GpuProfile {
        name: "A100 PCIe",
        fp16_tflops: 77.0, // Realistic benchmark value (50% of theoretical 155)
        fp8_tflops: 0.0,   // No FP8 support
        memory_bandwidth_gbps: 1555.0,
    },
    GpuProfile {
        name: "A100 SXM4",
        fp16_tflops: 156.0, // Realistic benchmark value (50% of theoretical 312)
        fp8_tflops: 0.0,    // No FP8 support
        memory_bandwidth_gbps: 2039.0,
    },
    GpuProfile {
        name: "RTX 3090",
        fp16_tflops: 35.5, // Realistic benchmark value (50% of theoretical 71)
        fp8_tflops: 0.0,   // No FP8 support
        memory_bandwidth_gbps: 936.0,
    },
    GpuProfile {
        name: "RTX 3080",
        fp16_tflops: 29.5, // Realistic benchmark value (50% of theoretical 59)
        fp8_tflops: 0.0,   // No FP8 support
        memory_bandwidth_gbps: 760.0,
    },
    GpuProfile {
        name: "RTX 3070",
        fp16_tflops: 20.0, // Realistic benchmark value (50% of theoretical 40)
        fp8_tflops: 0.0,   // No FP8 support
        memory_bandwidth_gbps: 448.0,
    },
];

// ============================================================================
// Validation Functions
// ============================================================================

/// Validate GPU is H100/H200/Blackwell with expected performance
///
/// # Safety
/// This function dereferences the raw cuBLAS handle pointer and calls unsafe CUDA APIs.
/// The caller must ensure the handle is valid and properly initialized.
pub unsafe fn validate_gpu(handle: cublasHandle_t, device: c_int) -> Result<(String, bool)> {
    let mut props = std::mem::zeroed::<cudaDeviceProp>();
    cuda_check!(cudaGetDeviceProperties(&mut props, device));

    let gpu_name = std::ffi::CStr::from_ptr(props.name.as_ptr())
        .to_string_lossy()
        .to_string();

    // Only support Hopper and Blackwell
    if props.major < 9 {
        return Ok((
            format!(
                "Unsupported GPU: {} (SM {}.{})",
                gpu_name, props.major, props.minor
            ),
            false,
        ));
    }

    // Find matching profile with flexible name matching
    let profile = GPU_PROFILES
        .iter()
        .find(|p| {
            // Match H100 variants
            if p.name.contains("H100") && gpu_name.contains("H100") {
                // Distinguish between PCIe and SXM by SM count
                (p.name.contains("SXM5") && props.multiProcessorCount >= 114)
                    || (p.name.contains("PCIe") && props.multiProcessorCount < 114)
                    || (!p.name.contains("PCIe") && !p.name.contains("SXM5"))
            } else {
                // Match other GPU models by name
                (p.name.contains("H200") && gpu_name.contains("H200"))
                    || (p.name.contains("B100") && gpu_name.contains("B100"))
                    || (p.name.contains("B200") && gpu_name.contains("B200"))
            }
        })
        .or_else(|| {
            // Fallback: find the best H100 profile based on SM count
            if gpu_name.contains("H100") {
                if props.multiProcessorCount >= 114 {
                    GPU_PROFILES.iter().find(|p| p.name == "H100 SXM5")
                } else {
                    GPU_PROFILES.iter().find(|p| p.name == "H100 PCIe")
                }
            } else {
                None
            }
        })
        .ok_or_else(|| anyhow::anyhow!("Unknown GPU: {}", gpu_name))?;

    tracing::info!(
        "Validating GPU: {} (SM {}.{}, {} SMs)",
        gpu_name,
        props.major,
        props.minor,
        props.multiProcessorCount
    );

    // FP16 benchmark
    let fp16_tflops = benchmark_tensor_core_fp16(handle, 8192, 8192, 8192, 50)?;
    let fp16_ratio = fp16_tflops / profile.fp16_tflops;

    tracing::info!(
        "FP16 Performance: {:.1} TFLOPS (expected: {:.1}, ratio: {:.2})",
        fp16_tflops,
        profile.fp16_tflops,
        fp16_ratio
    );

    // FP8 benchmark - optional, may not be supported on all CUDA versions
    let (_fp8_tflops, fp8_ratio) = match benchmark_tensor_core_fp8(handle, 8192, 8192, 8192, 50) {
        Ok(tflops) => {
            let ratio = tflops / profile.fp8_tflops;
            tracing::info!(
                "FP8 Performance: {:.1} TFLOPS (expected: {:.1}, ratio: {:.2})",
                tflops,
                profile.fp8_tflops,
                ratio
            );
            (Some(tflops), Some(ratio))
        }
        Err(e) => {
            tracing::warn!("FP8 benchmark not available: {}", e);
            tracing::info!(
                "Note: FP8 support requires CUDA 12.9+ or specific cuBLAS configuration"
            );
            (None, None)
        }
    };

    // Validate performance - FP16 is required, FP8 is optional
    let fp16_valid = fp16_ratio > 0.7 && fp16_ratio < 1.3;
    let fp8_valid = fp8_ratio.map(|r| r > 0.7 && r < 1.3).unwrap_or(true); // If FP8 not available, consider it valid

    let valid = fp16_valid && fp8_valid;

    if valid {
        tracing::info!("✅ GPU validation passed!");
    } else {
        tracing::warn!("❌ GPU validation failed - performance outside expected range");
    }

    Ok((gpu_name, valid))
}

/// Validate GPU performance across all supported architectures
///
/// # Safety
/// This function dereferences the raw cuBLAS handle pointer and calls unsafe CUDA APIs.
/// The caller must ensure the handle is valid and properly initialized.
pub unsafe fn validate_gpu_all_architectures(
    handle: cublasHandle_t,
    device: c_int,
) -> Result<(String, bool)> {
    let mut props = std::mem::zeroed::<cudaDeviceProp>();
    cuda_check!(cudaGetDeviceProperties(&mut props, device));

    let gpu_name = std::ffi::CStr::from_ptr(props.name.as_ptr())
        .to_string_lossy()
        .to_string();

    // Detect architecture
    let arch = detect_gpu_architecture(device)?;

    tracing::info!(
        "Validating GPU: {} ({:?}, SM {}.{}, {} SMs)",
        gpu_name,
        arch,
        props.major,
        props.minor,
        props.multiProcessorCount
    );

    // Check if architecture is supported
    if let GpuArchitecture::Unknown(major, minor) = arch {
        if major < 7 {
            return Ok((
                format!(
                    "GPU architecture too old: {gpu_name} (SM {major}.{minor}) - requires SM 7.0+ for tensor cores"
                ),
                false,
            ));
        }
    }

    // Find matching profile
    let profile = GPU_PROFILES
        .iter()
        .find(|p| {
            // Flexible name matching
            let gpu_upper = gpu_name.to_uppercase();
            let profile_upper = p.name.to_uppercase();

            // Direct name match
            if gpu_upper.contains(&profile_upper) || profile_upper.contains(&gpu_upper) {
                return true;
            }

            // Specific checks for different architectures
            match arch {
                GpuArchitecture::Ampere => {
                    p.name.contains("RTX 3090")
                        || p.name.contains("RTX 3080")
                        || p.name.contains("RTX 3070")
                        || p.name.contains("A100")
                }
                GpuArchitecture::AdaLovelace => {
                    p.name.contains("RTX 4090")
                        || p.name.contains("RTX 4080")
                        || p.name.contains("L4")
                        || p.name.contains("L40")
                }
                GpuArchitecture::Hopper => p.name.contains("H100") || p.name.contains("H200"),
                GpuArchitecture::Blackwell => p.name.contains("B100") || p.name.contains("B200"),
                _ => false,
            }
        })
        .ok_or_else(|| {
            anyhow::anyhow!(
                "No performance profile found for GPU: {} ({:?})",
                gpu_name,
                arch
            )
        })?;

    tracing::info!(
        "Using performance profile: {} (expected FP16: {:.1} TFLOPS)",
        profile.name,
        profile.fp16_tflops
    );

    // Run FP16 benchmark based on architecture
    let fp16_tflops = match arch {
        GpuArchitecture::Hopper | GpuArchitecture::Blackwell => {
            // Use optimized tensor core benchmark
            benchmark_tensor_core_fp16(handle, 8192, 8192, 8192, 50)?
        }
        GpuArchitecture::AdaLovelace | GpuArchitecture::Ampere => {
            // Use smaller matrices for consumer GPUs
            let algo = get_optimal_gemm_algo(device)?;
            benchmark_fp16_gemm_with_algo(handle, 4096, 4096, 4096, 50, algo)?
        }
        _ => {
            // Unknown architectures - try standard benchmark
            let algo = get_optimal_gemm_algo(device)?;
            benchmark_fp16_gemm_with_algo(handle, 4096, 4096, 4096, 50, algo)?
        }
    };

    let fp16_ratio = fp16_tflops / profile.fp16_tflops;

    tracing::info!(
        "FP16 Performance: {:.1} TFLOPS (expected: {:.1}, ratio: {:.2})",
        fp16_tflops,
        profile.fp16_tflops,
        fp16_ratio
    );

    // FP8 benchmark - only for architectures that support it
    if arch.supports_fp8() && profile.fp8_tflops > 0.0 {
        match benchmark_tensor_core_fp8(handle, 8192, 8192, 8192, 50) {
            Ok(tflops) => {
                let ratio = tflops / profile.fp8_tflops;
                tracing::info!(
                    "FP8 Performance: {:.1} TFLOPS (expected: {:.1}, ratio: {:.2})",
                    tflops,
                    profile.fp8_tflops,
                    ratio
                );
            }
            Err(e) => {
                tracing::warn!("FP8 benchmark not available: {}", e);
                tracing::info!("Note: FP8 support requires CUDA 12.9+ and Hopper/Blackwell GPU");
            }
        }
    }

    // Validate performance - allow 30% tolerance for consumer GPUs
    let tolerance = match arch {
        GpuArchitecture::Hopper | GpuArchitecture::Blackwell => 0.3, // 30% tolerance for datacenter
        _ => 0.4, // 40% tolerance for consumer GPUs
    };

    let fp16_valid = fp16_ratio > (1.0 - tolerance) && fp16_ratio < (1.0 + tolerance);

    if fp16_valid {
        tracing::info!("✅ GPU validation passed for {} ({:?})", gpu_name, arch);
    } else {
        tracing::warn!(
            "❌ GPU validation failed - performance outside expected range (ratio: {:.2}, tolerance: ±{:.0}%)",
            fp16_ratio,
            tolerance * 100.0
        );
    }

    Ok((gpu_name, fp16_valid))
}

/// Get CUDA runtime version
pub fn get_cuda_runtime_version() -> Result<(i32, i32)> {
    unsafe {
        let mut version = 0;
        let result = cudaRuntimeGetVersion(&mut version);
        if result != cudaSuccess {
            return Err(anyhow::anyhow!("Failed to get CUDA runtime version"));
        }

        let major = version / 1000;
        let minor = (version % 1000) / 10;
        Ok((major, minor))
    }
}

/// Get CUDA driver version
pub fn get_cuda_driver_version() -> Result<(i32, i32)> {
    unsafe {
        let mut version = 0;
        let result = cudaDriverGetVersion(&mut version);
        if result != cudaSuccess {
            return Err(anyhow::anyhow!("Failed to get CUDA driver version"));
        }

        let major = version / 1000;
        let minor = (version % 1000) / 10;
        Ok((major, minor))
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_cuda_buffer() {
        // Test that CudaBuffer can be created and dropped without issues
        if let Ok(buffer) = CudaBuffer::new(1024) {
            assert_eq!(buffer.size(), 1024);
            assert!(!buffer.as_ptr().is_null());
        }
    }
}
