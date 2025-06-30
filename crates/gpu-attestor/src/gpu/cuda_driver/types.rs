//! Types for CUDA Driver API matrix computation

use serde::{Deserialize, Serialize};

#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct ComputeResult {
    pub execution_time_ms: f32,
    pub matrix_checksum: Option<u64>,
}

#[derive(Debug, Clone)]
pub struct MatrixDimensions {
    pub n: usize,
    pub k: usize,
}

impl MatrixDimensions {
    pub fn new(n: usize, k: usize) -> Self {
        Self { n, k }
    }

    #[allow(dead_code)]
    pub fn memory_required(&self) -> usize {
        let size_a = self.n * self.n * std::mem::size_of::<f64>();
        let size_b = self.n * self.k * std::mem::size_of::<f64>();
        let size_c = self.n * self.k * std::mem::size_of::<f64>();
        size_a + size_b + size_c
    }
}

/// Trait for matrix computation engines
pub trait MatrixCompute: Send + Sync {
    fn set_device(&self, device_id: u32) -> anyhow::Result<()>;

    fn multiply_matrices(
        &self,
        dimensions: &MatrixDimensions,
        seed: u64,
        device_id: u32,
    ) -> anyhow::Result<ComputeResult>;
}
