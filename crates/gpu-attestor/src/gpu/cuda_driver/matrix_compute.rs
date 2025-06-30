//! CUDA Driver API matrix multiplication implementation

use super::{
    cuda_driver_ffi::*,
    ptx_source::PTX_SOURCE,
    timing::CudaTimer,
    types::{ComputeResult, MatrixCompute, MatrixDimensions},
};
use anyhow::{anyhow, Result};
use std::os::raw::{c_int, c_uint, c_void};

const BLOCK_SIZE: usize = 16;

pub struct CudaMatrixCompute {
    context: CUcontext,
    module: CUmodule,
    kernel: CUfunction,
}

// SAFETY: CudaMatrixCompute only accesses CUDA through the CUDA driver API
// which handles thread safety internally
unsafe impl Send for CudaMatrixCompute {}
unsafe impl Sync for CudaMatrixCompute {}

impl CudaMatrixCompute {
    pub fn new() -> Result<Self> {
        let mut context: CUcontext = std::ptr::null_mut();
        let mut module: CUmodule = std::ptr::null_mut();
        let mut kernel: CUfunction = std::ptr::null_mut();

        unsafe {
            // Initialize CUDA driver
            let init_result = cuInit(0);
            if init_result != CUresult::CUDA_SUCCESS {
                return Err(anyhow!("Failed to initialize CUDA: {:?}", init_result));
            }

            let mut device: CUdevice = 0;
            let result = cuDeviceGet(&mut device, 0);
            if result != CUresult::CUDA_SUCCESS {
                return Err(anyhow!("Failed to get device: {:?}", result));
            }

            let result = cuCtxCreate_v2(&mut context, 0, device);
            if result != CUresult::CUDA_SUCCESS {
                return Err(anyhow!("Failed to create context: {:?}", result));
            }

            // Use embedded PTX source
            let ptx_cstring = std::ffi::CString::new(PTX_SOURCE)
                .map_err(|e| anyhow!("Invalid PTX string: {}", e))?;

            let result = cuModuleLoadData(&mut module, ptx_cstring.as_ptr() as *const c_void);
            if result != CUresult::CUDA_SUCCESS {
                cuCtxDestroy_v2(context);
                return Err(anyhow!("Failed to load PTX module: {:?}", result));
            }

            let kernel_name = std::ffi::CString::new("matrix_multiply").unwrap();
            let result = cuModuleGetFunction(&mut kernel, module, kernel_name.as_ptr());
            if result != CUresult::CUDA_SUCCESS {
                cuModuleUnload(module);
                cuCtxDestroy_v2(context);
                return Err(anyhow!("Failed to get kernel function: {:?}", result));
            }
        }

        Ok(Self {
            context,
            module,
            kernel,
        })
    }

    fn generate_random_matrix(&self, size: usize, seed: u64) -> Result<CUdeviceptr> {
        let mut d_matrix: CUdeviceptr = 0;
        let bytes = size * std::mem::size_of::<f64>();

        unsafe {
            let result = cuMemAlloc_v2(&mut d_matrix, bytes);
            if result != CUresult::CUDA_SUCCESS {
                return Err(anyhow!("Failed to allocate memory: {:?}", result));
            }

            let mut rng_state: CUdeviceptr = 0;
            let result = cuMemAlloc_v2(&mut rng_state, 8); // unsigned long long
            if result != CUresult::CUDA_SUCCESS {
                cuMemFree_v2(d_matrix);
                return Err(anyhow!("Failed to allocate RNG state: {:?}", result));
            }

            let mut init_kernel: CUfunction = std::ptr::null_mut();
            let kernel_name = std::ffi::CString::new("init_rng").unwrap();
            let result = cuModuleGetFunction(&mut init_kernel, self.module, kernel_name.as_ptr());
            if result != CUresult::CUDA_SUCCESS {
                cuMemFree_v2(d_matrix);
                cuMemFree_v2(rng_state);
                return Err(anyhow!("Failed to get RNG init kernel: {:?}", result));
            }

            let mut params = [
                &rng_state as *const _ as *mut c_void,
                &seed as *const _ as *mut c_void,
            ];

            let result = cuLaunchKernel(
                init_kernel,
                1,
                1,
                1,
                1,
                1,
                1,
                0,
                std::ptr::null_mut(),
                params.as_mut_ptr(),
                std::ptr::null_mut(),
            );

            if result != CUresult::CUDA_SUCCESS {
                cuMemFree_v2(d_matrix);
                cuMemFree_v2(rng_state);
                return Err(anyhow!("Failed to launch RNG init: {:?}", result));
            }

            let mut gen_kernel: CUfunction = std::ptr::null_mut();
            let kernel_name = std::ffi::CString::new("generate_random").unwrap();
            let result = cuModuleGetFunction(&mut gen_kernel, self.module, kernel_name.as_ptr());
            if result != CUresult::CUDA_SUCCESS {
                cuMemFree_v2(d_matrix);
                cuMemFree_v2(rng_state);
                return Err(anyhow!("Failed to get random gen kernel: {:?}", result));
            }

            let count = size as c_uint;
            let mut params = [
                &d_matrix as *const _ as *mut c_void,
                &rng_state as *const _ as *mut c_void,
                &count as *const _ as *mut c_void,
            ];

            let threads_per_block = 256;
            let blocks = size.div_ceil(threads_per_block);

            let result = cuLaunchKernel(
                gen_kernel,
                blocks as c_uint,
                1,
                1,
                threads_per_block as c_uint,
                1,
                1,
                0,
                std::ptr::null_mut(),
                params.as_mut_ptr(),
                std::ptr::null_mut(),
            );

            cuMemFree_v2(rng_state);

            if result != CUresult::CUDA_SUCCESS {
                cuMemFree_v2(d_matrix);
                return Err(anyhow!("Failed to launch random gen: {:?}", result));
            }

            cuCtxSynchronize();
        }

        Ok(d_matrix)
    }
}

impl MatrixCompute for CudaMatrixCompute {
    fn set_device(&self, _device_id: u32) -> Result<()> {
        unsafe {
            cuCtxSetCurrent(self.context);
        }
        Ok(())
    }

    fn multiply_matrices(
        &self,
        dimensions: &MatrixDimensions,
        seed: u64,
        device_id: u32,
    ) -> Result<ComputeResult> {
        self.set_device(device_id)?;

        let d_a = self.generate_random_matrix(dimensions.n * dimensions.n, seed)?;
        let d_b = self.generate_random_matrix(dimensions.n * dimensions.k, seed.wrapping_add(1))?;

        let mut d_c: CUdeviceptr = 0;
        let c_bytes = dimensions.n * dimensions.k * std::mem::size_of::<f64>();

        unsafe {
            let result = cuMemAlloc_v2(&mut d_c, c_bytes);
            if result != CUresult::CUDA_SUCCESS {
                cuMemFree_v2(d_a);
                cuMemFree_v2(d_b);
                return Err(anyhow!("Failed to allocate result matrix: {:?}", result));
            }

            let timer = CudaTimer::new()?;
            timer.start()?;

            let grid_x = dimensions.k.div_ceil(BLOCK_SIZE) as c_uint;
            let grid_y = dimensions.n.div_ceil(BLOCK_SIZE) as c_uint;

            let n = dimensions.n as c_int;
            let k = dimensions.k as c_int;

            let mut params = [
                &d_a as *const _ as *mut c_void,
                &d_b as *const _ as *mut c_void,
                &d_c as *const _ as *mut c_void,
                &n as *const _ as *mut c_void,
                &n as *const _ as *mut c_void,
                &k as *const _ as *mut c_void,
            ];

            let result = cuLaunchKernel(
                self.kernel,
                grid_x,
                grid_y,
                1,
                BLOCK_SIZE as c_uint,
                BLOCK_SIZE as c_uint,
                1,
                0,
                std::ptr::null_mut(),
                params.as_mut_ptr(),
                std::ptr::null_mut(),
            );

            if result != CUresult::CUDA_SUCCESS {
                cuMemFree_v2(d_a);
                cuMemFree_v2(d_b);
                cuMemFree_v2(d_c);
                return Err(anyhow!("Failed to launch kernel: {:?}", result));
            }

            timer.stop()?;
            let elapsed_ms = timer.elapsed_ms()?;

            cuMemFree_v2(d_a);
            cuMemFree_v2(d_b);
            cuMemFree_v2(d_c);

            Ok(ComputeResult {
                execution_time_ms: elapsed_ms,
                matrix_checksum: None,
            })
        }
    }
}

impl Drop for CudaMatrixCompute {
    fn drop(&mut self) {
        unsafe {
            cuModuleUnload(self.module);
            cuCtxDestroy_v2(self.context);
        }
    }
}
