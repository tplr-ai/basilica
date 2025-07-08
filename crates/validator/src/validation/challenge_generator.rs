//! GPU PoW Challenge Generator
//!
//! Generates challenge parameters for GPU Proof-of-Work validation.

use anyhow::Result;
use rand::{thread_rng, Rng};
use tracing::info;

use protocol::basilca::common::v1::ChallengeParameters;

/// Challenge generator for GPU PoW
pub struct ChallengeGenerator {
    default_matrix_dim: u32,
    vram_utilization_target: f64,
}

impl ChallengeGenerator {
    /// Create a new challenge generator
    pub fn new() -> Self {
        Self {
            default_matrix_dim: 256,
            vram_utilization_target: 0.9, // Target 90% VRAM usage
        }
    }

    /// Get optimal matrix dimension based on GPU memory
    fn get_optimal_matrix_dim(&self, claimed_vram_gb: u32) -> u32 {
        if claimed_vram_gb >= 80 {
            // H100 (80GB) or H200 (96GB): Use 1024x1024 for better performance
            1024
        } else if claimed_vram_gb >= 40 {
            // A100 (40GB) or similar: Use 512x512
            512
        } else {
            // Smaller GPUs: Use default 256x256
            self.default_matrix_dim
        }
    }

    /// Calculate number of iterations for bandwidth testing
    fn calculate_iterations(&self, claimed_vram_gb: u32) -> u32 {
        match claimed_vram_gb {
            80.. => 4200, // H100: ~100GB transfer
            40.. => 2500, // A100: ~60GB transfer
            24.. => 1000, // RTX 4090: ~24GB transfer
            16.. => 500,  // RTX 4080: ~12GB transfer
            _ => 250,     // Default: ~6GB transfer
        }
    }

    /// Generate a challenge for the given GPU
    pub fn generate_challenge(
        &self,
        claimed_gpu: &str,
        claimed_vram_gb: u32,
        validator_nonce: Option<String>,
    ) -> Result<ChallengeParameters> {
        info!(
            "Generating GPU PoW challenge for {} with {} GB VRAM",
            claimed_gpu, claimed_vram_gb
        );

        // Use optimal matrix dimension based on GPU memory
        let matrix_dim = self.get_optimal_matrix_dim(claimed_vram_gb);

        // Calculate number of matrices to fill target VRAM usage
        let matrix_size_bytes = (matrix_dim * matrix_dim * 8) as u64; // 8 bytes per f64
        let vram_bytes = (claimed_vram_gb as u64) * 1024 * 1024 * 1024;
        let target_usage = (vram_bytes as f64 * self.vram_utilization_target) as u64;
        // Dynamic cap based on GPU memory to prevent u32 overflow
        // For H100 with 1024x1024 matrices: ~8960 matrices for 70GB
        let max_matrices = if claimed_vram_gb >= 80 {
            // H100 or larger: use larger matrices but fewer of them
            20000
        } else if claimed_vram_gb >= 40 {
            // A100 40GB: moderate cap
            15000
        } else {
            // Smaller GPUs: keep original cap
            10000
        };
        let num_matrices = ((target_usage / matrix_size_bytes).max(2) as u32).min(max_matrices);

        // Generate random parameters
        let mut rng = thread_rng();
        let seed: u64 = rng.gen();

        // Generate validator nonce if not provided
        let validator_nonce =
            validator_nonce.unwrap_or_else(|| format!("val_{}", rng.gen::<u64>()));

        // Calculate iterations for bandwidth testing
        let num_iterations = self.calculate_iterations(claimed_vram_gb);

        let memory_usage_gb =
            (num_matrices as u64 * matrix_size_bytes) as f64 / (1024.0 * 1024.0 * 1024.0);
        let bandwidth_gb =
            (num_iterations as f64 * 24.0 * matrix_size_bytes as f64) / (1024.0 * 1024.0 * 1024.0);

        info!(
            "Challenge params: seed={}, dim={}, matrices={}, memory={:.2}GB, iterations={}, bandwidth={:.2}GB, nonce={}",
            seed, matrix_dim, num_matrices, memory_usage_gb, num_iterations, bandwidth_gb, validator_nonce
        );

        Ok(ChallengeParameters {
            challenge_type: "matrix_multiplication_pow".to_string(),
            parameters_json: String::new(), // Not used for GPU PoW
            expected_duration_seconds: 10,  // Max 10 seconds
            difficulty_level: 1,
            seed: seed.to_string(), // Legacy field
            machine_info: None,
            gpu_pow_seed: seed,
            matrix_dim,
            num_matrices,
            matrix_a_index: 0, // Deprecated - not used in v2
            matrix_b_index: 0, // Deprecated - not used in v2
            validator_nonce,
            num_iterations,
            verification_sample_rate: 0.1, // 10% sampling by default
        })
    }

    /// Set custom matrix dimension
    pub fn with_matrix_dim(mut self, dim: u32) -> Self {
        self.default_matrix_dim = dim;
        self
    }

    /// Set custom VRAM utilization target (0.0 - 1.0)
    pub fn with_vram_utilization(mut self, target: f64) -> Self {
        self.vram_utilization_target = target.clamp(0.1, 0.95);
        self
    }
}

impl Default for ChallengeGenerator {
    fn default() -> Self {
        Self::new()
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_challenge_generation() {
        let generator = ChallengeGenerator::new();
        let challenge = generator
            .generate_challenge(
                "NVIDIA H100",
                80, // 80GB VRAM
                Some("test_nonce".to_string()),
            )
            .unwrap();

        assert_eq!(challenge.challenge_type, "matrix_multiplication_pow");
        assert_eq!(challenge.matrix_dim, 1024); // H100 should use 1024x1024
        assert!(challenge.num_matrices > 0);
        assert!(challenge.num_iterations > 0); // Should always have iterations
        assert_eq!(challenge.num_iterations, 4200); // H100 should use 4200 iterations
        assert_eq!(challenge.validator_nonce, "test_nonce");
    }

    #[test]
    fn test_vram_calculation() {
        let generator = ChallengeGenerator::new();

        // Test with 8GB VRAM
        let challenge = generator
            .generate_challenge("NVIDIA RTX 4080", 8, None)
            .unwrap();

        let matrix_size_bytes = (challenge.matrix_dim * challenge.matrix_dim * 8) as u64;
        let total_memory = challenge.num_matrices as u64 * matrix_size_bytes;
        let vram_bytes = 8u64 * 1024 * 1024 * 1024;

        // Should use ~60% of VRAM (limited by matrix cap)
        assert!((total_memory as f64) > (vram_bytes as f64 * 0.5));
        assert!((total_memory as f64) < (vram_bytes as f64 * 0.75));
    }
}
