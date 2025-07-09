//! Tests for the challenge generator with H100 optimization

#[cfg(test)]
mod tests {
    use crate::validation::challenge_generator::ChallengeGenerator;

    #[test]
    fn test_h100_configuration() {
        let generator = ChallengeGenerator::new();

        // Test H100 80GB
        let challenge = generator
            .generate_challenge("NVIDIA H100", 80, Some("test_nonce".to_string()))
            .unwrap();

        // Should use 1024x1024 matrices for H100
        assert_eq!(challenge.matrix_dim, 1024);

        // Calculate expected memory usage
        let matrix_size_bytes = (1024u64 * 1024 * 8) as f64; // 8MB per matrix
        let total_memory = challenge.num_matrices as f64 * matrix_size_bytes;
        let memory_gb = total_memory / (1024.0 * 1024.0 * 1024.0);

        // Should use ~72GB (90% of 80GB)
        assert!(memory_gb > 65.0 && memory_gb < 75.0);

        // Should not exceed the cap
        assert!(challenge.num_matrices <= 20000);

        println!(
            "H100 config: {}x{} matrices, {} matrices total, {:.2}GB memory",
            challenge.matrix_dim, challenge.matrix_dim, challenge.num_matrices, memory_gb
        );
    }

    #[test]
    fn test_a100_configuration() {
        let generator = ChallengeGenerator::new();

        // Test A100 40GB
        let challenge = generator
            .generate_challenge("NVIDIA A100", 40, None)
            .unwrap();

        // Should use 512x512 matrices for A100
        assert_eq!(challenge.matrix_dim, 512);

        // Calculate memory usage
        let matrix_size_bytes = (512u64 * 512 * 8) as f64; // 2MB per matrix
        let total_memory = challenge.num_matrices as f64 * matrix_size_bytes;
        let memory_gb = total_memory / (1024.0 * 1024.0 * 1024.0);

        // Should use close to 30GB (limited by matrix cap of 15000)
        println!("A100 actual memory usage: {memory_gb:.2}GB");
        assert!(memory_gb > 28.0 && memory_gb < 32.0);

        // Should not exceed A100 cap
        assert!(challenge.num_matrices <= 15000);

        println!(
            "A100 config: {}x{} matrices, {} matrices total, {:.2}GB memory",
            challenge.matrix_dim, challenge.matrix_dim, challenge.num_matrices, memory_gb
        );
    }

    #[test]
    fn test_small_gpu_configuration() {
        let generator = ChallengeGenerator::new();

        // Test RTX 4080 16GB
        let challenge = generator
            .generate_challenge("NVIDIA RTX 4080", 16, None)
            .unwrap();

        // Should use default 256x256 matrices for smaller GPUs
        assert_eq!(challenge.matrix_dim, 256);

        // Calculate memory usage
        let matrix_size_bytes = (256u64 * 256 * 8) as f64; // 512KB per matrix
        let total_memory = challenge.num_matrices as f64 * matrix_size_bytes;
        let memory_gb = total_memory / (1024.0 * 1024.0 * 1024.0);

        // Should use close to 5GB (limited by matrix cap of 10000)
        println!("RTX 4080 actual memory usage: {memory_gb:.2}GB");
        assert!(memory_gb > 4.5 && memory_gb < 6.0);

        // Should not exceed small GPU cap
        assert!(challenge.num_matrices <= 10000);

        println!(
            "RTX 4080 config: {}x{} matrices, {} matrices total, {:.2}GB memory",
            challenge.matrix_dim, challenge.matrix_dim, challenge.num_matrices, memory_gb
        );
    }

    #[test]
    fn test_memory_saturation_percentage() {
        let generator = ChallengeGenerator::new().with_vram_utilization(0.85); // Test with 85% utilization

        let challenge = generator
            .generate_challenge("NVIDIA H100", 80, None)
            .unwrap();

        let matrix_size_bytes =
            (challenge.matrix_dim as u64 * challenge.matrix_dim as u64 * 8) as f64;
        let total_memory = challenge.num_matrices as f64 * matrix_size_bytes;
        let memory_gb = total_memory / (1024.0 * 1024.0 * 1024.0);

        // Should use ~68GB (85% of 80GB)
        assert!(memory_gb > 65.0 && memory_gb < 70.0);
    }
}
