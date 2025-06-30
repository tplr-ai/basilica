//! CUDA timing utilities for Driver API

use super::cuda_driver_ffi::*;
use anyhow::{anyhow, Result};

pub struct CudaTimer {
    start: CUevent,
    stop: CUevent,
}

impl CudaTimer {
    pub fn new() -> Result<Self> {
        let mut start: CUevent = std::ptr::null_mut();
        let mut stop: CUevent = std::ptr::null_mut();

        unsafe {
            let result = cuEventCreate(&mut start, 0);
            if result != CUresult::CUDA_SUCCESS {
                return Err(anyhow!("Failed to create start event: {:?}", result));
            }

            let result = cuEventCreate(&mut stop, 0);
            if result != CUresult::CUDA_SUCCESS {
                cuEventDestroy_v2(start);
                return Err(anyhow!("Failed to create stop event: {:?}", result));
            }
        }

        Ok(Self { start, stop })
    }

    pub fn start(&self) -> Result<()> {
        unsafe {
            let result = cuEventRecord(self.start, std::ptr::null_mut());
            if result != CUresult::CUDA_SUCCESS {
                return Err(anyhow!("Failed to record start event: {:?}", result));
            }
        }
        Ok(())
    }

    pub fn stop(&self) -> Result<()> {
        unsafe {
            let result = cuEventRecord(self.stop, std::ptr::null_mut());
            if result != CUresult::CUDA_SUCCESS {
                return Err(anyhow!("Failed to record stop event: {:?}", result));
            }
        }
        Ok(())
    }

    pub fn elapsed_ms(&self) -> Result<f32> {
        unsafe {
            let result = cuEventSynchronize(self.stop);
            if result != CUresult::CUDA_SUCCESS {
                return Err(anyhow!("Failed to synchronize stop event: {:?}", result));
            }

            let mut elapsed: f32 = 0.0;
            let result = cuEventElapsedTime(&mut elapsed, self.start, self.stop);
            if result != CUresult::CUDA_SUCCESS {
                return Err(anyhow!("Failed to get elapsed time: {:?}", result));
            }

            Ok(elapsed)
        }
    }
}

impl Drop for CudaTimer {
    fn drop(&mut self) {
        unsafe {
            cuEventDestroy_v2(self.start);
            cuEventDestroy_v2(self.stop);
        }
    }
}
