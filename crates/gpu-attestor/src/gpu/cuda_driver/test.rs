//! Tests for CUDA Driver API implementation

#[cfg(test)]
mod tests {
    use super::super::*;

    #[test]
    fn test_matrix_dimensions() {
        let dims = MatrixDimensions::new(1024, 2048);
        assert_eq!(dims.n, 1024);
        assert_eq!(dims.k, 2048);

        let memory = dims.memory_required();
        // 3 matrices (A: n×n, B: n×k, C: n×k) × 8 bytes per double
        let expected = (1024 * 1024 + 1024 * 2048 + 1024 * 2048) * 8;
        assert_eq!(memory, expected);
    }

    #[test]
    fn test_compute_result_serialization() {
        let result = ComputeResult {
            execution_time_ms: 42.5,
            matrix_checksum: Some(0x123456789ABCDEF0),
        };

        let json = serde_json::to_string(&result).unwrap();
        let deserialized: ComputeResult = serde_json::from_str(&json).unwrap();

        assert_eq!(result.execution_time_ms, deserialized.execution_time_ms);
        assert_eq!(result.matrix_checksum, deserialized.matrix_checksum);
    }

    #[test]
    #[ignore] // Requires CUDA hardware
    fn test_cuda_matrix_compute_creation() {
        match CudaMatrixCompute::new() {
            Ok(_) => println!("CUDA Driver API initialized successfully"),
            Err(e) => println!("Expected error without CUDA hardware: {e}"),
        }
    }
}
