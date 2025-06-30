//! CUDA Driver API implementation for matrix multiplication
//!
//! This module provides an alternative GPU computation backend using the
//! CUDA Driver API with custom PTX kernels, complementing the existing
//! cuBLAS-based implementation.

pub mod cuda_driver_ffi;
pub mod matrix_compute;
pub mod ptx_source;
pub mod timing;
pub mod types;

pub use matrix_compute::CudaMatrixCompute;
pub use types::{ComputeResult, MatrixCompute, MatrixDimensions};

#[cfg(test)]
mod test;
