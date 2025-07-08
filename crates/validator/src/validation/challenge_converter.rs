//! Converter from protocol-specific ChallengeParameters to generic ComputeChallenge
//!
//! This module provides conversion utilities to transform the protocol-specific
//! challenge format into the generic format expected by the VM-protected attestor.

use anyhow::Result;
use protocol::basilca::common::v1::ChallengeParameters;
use serde::{Deserialize, Serialize};

/// Generic computational challenge for VM-protected attestor
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct ComputeChallenge {
    /// Unique session identifier
    pub session_id: String,
    /// Problem size parameter (matrix dimension)
    pub problem_size: u32,
    /// Random seed data
    pub seed_data: Vec<u8>,
    /// Challenge timestamp
    pub timestamp: Option<String>,
    /// Expected resource count (number of GPUs)
    pub resource_count: u32,
    /// Computation timeout in milliseconds
    pub computation_timeout_ms: u32,
    /// Protocol timeout in milliseconds
    pub protocol_timeout_ms: u32,
}

impl From<&ChallengeParameters> for ComputeChallenge {
    fn from(params: &ChallengeParameters) -> Self {
        // Convert seed to bytes
        let seed_bytes = params.gpu_pow_seed.to_le_bytes().to_vec();
        
        // Calculate appropriate timeouts based on matrix dimension and iterations
        let computation_timeout_ms = calculate_computation_timeout(
            params.matrix_dim,
            params.num_iterations
        );
        
        let protocol_timeout_ms = computation_timeout_ms + 2000; // Add 2s overhead
        
        ComputeChallenge {
            session_id: params.validator_nonce.clone(),
            problem_size: params.matrix_dim,
            seed_data: seed_bytes,
            timestamp: Some(
                std::time::SystemTime::now()
                    .duration_since(std::time::UNIX_EPOCH)
                    .unwrap()
                    .as_secs()
                    .to_string(),
            ),
            resource_count: 1, // Default to 1 GPU, could be derived from params
            computation_timeout_ms,
            protocol_timeout_ms,
        }
    }
}

/// Calculate computation timeout based on problem parameters
fn calculate_computation_timeout(matrix_dim: u32, iterations: u32) -> u32 {
    // Base time per iteration for different matrix sizes (in milliseconds)
    let time_per_iteration_ms = match matrix_dim {
        0..=128 => 0.1,
        129..=256 => 0.5,
        257..=512 => 2.0,
        513..=1024 => 8.0,
        1025..=2048 => 32.0,
        _ => 64.0,
    };
    
    // Calculate total time with safety margin
    let base_time = (iterations as f64 * time_per_iteration_ms) as u32;
    let safety_margin = 2.0; // 2x safety margin
    
    (base_time as f64 * safety_margin) as u32
}

/// Convert ChallengeParameters to JSON string for gpu-attestor
pub fn challenge_params_to_json(params: &ChallengeParameters) -> Result<String> {
    let compute_challenge: ComputeChallenge = params.into();
    let json = serde_json::to_string(&compute_challenge)?;
    Ok(json)
}

/// Convert ChallengeParameters to base64-encoded challenge for gpu-attestor
pub fn challenge_params_to_base64(params: &ChallengeParameters) -> Result<String> {
    use base64::{engine::general_purpose, Engine as _};
    
    let json = challenge_params_to_json(params)?;
    let base64 = general_purpose::STANDARD.encode(&json);
    Ok(base64)
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_challenge_conversion() {
        let params = ChallengeParameters {
            challenge_type: "matrix_multiplication_pow".to_string(),
            parameters_json: String::new(),
            expected_duration_seconds: 10,
            difficulty_level: 1,
            seed: "12345".to_string(),
            machine_info: None,
            gpu_pow_seed: 12345,
            matrix_dim: 1024,
            num_matrices: 100,
            matrix_a_index: 0,
            matrix_b_index: 0,
            validator_nonce: "test_session_123".to_string(),
            num_iterations: 1000,
            verification_sample_rate: 0.1,
        };

        let compute_challenge: ComputeChallenge = (&params).into();
        
        assert_eq!(compute_challenge.session_id, "test_session_123");
        assert_eq!(compute_challenge.problem_size, 1024);
        assert_eq!(compute_challenge.resource_count, 1);
        assert!(compute_challenge.computation_timeout_ms > 0);
        assert!(compute_challenge.protocol_timeout_ms > compute_challenge.computation_timeout_ms);
    }

    #[test]
    fn test_base64_encoding() {
        let params = ChallengeParameters {
            challenge_type: "matrix_multiplication_pow".to_string(),
            parameters_json: String::new(),
            expected_duration_seconds: 10,
            difficulty_level: 1,
            seed: "12345".to_string(),
            machine_info: None,
            gpu_pow_seed: 12345,
            matrix_dim: 256,
            num_matrices: 50,
            matrix_a_index: 0,
            matrix_b_index: 0,
            validator_nonce: "validator_test".to_string(),
            num_iterations: 500,
            verification_sample_rate: 0.1,
        };

        let base64 = challenge_params_to_base64(&params).unwrap();
        assert!(!base64.is_empty());
        
        // Verify it's valid base64
        use base64::{engine::general_purpose, Engine as _};
        let decoded = general_purpose::STANDARD.decode(&base64).unwrap();
        
        // Verify it's valid JSON
        let json = String::from_utf8(decoded).unwrap();
        let _: ComputeChallenge = serde_json::from_str(&json).unwrap();
    }
}