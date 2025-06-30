//! Verifiable Delay Function (VDF) module
//!
//! This module provides VDF computation and verification capabilities for GPU attestation.
//! VDFs are cryptographic functions that require a specific amount of sequential computation
//! to evaluate, providing proof of work that cannot be parallelized.

pub mod computer;
pub mod types;

// Re-export main types and functions for convenient access
pub use computer::VdfComputer;
pub use types::*;

use anyhow::Result;

/// Generate VDF public parameters with the specified bit size
pub fn generate_vdf_parameters(bits: usize) -> Result<(Vec<u8>, Vec<u8>)> {
    VdfComputer::generate_public_params(bits)
}

/// Create a VDF challenge from given parameters and previous attestation
pub fn create_vdf_challenge(
    modulus: Vec<u8>,
    generator: Vec<u8>,
    difficulty: u64,
    prev_attestation_sig: &[u8],
) -> VdfChallenge {
    VdfComputer::create_challenge(modulus, generator, difficulty, prev_attestation_sig)
}

/// Compute a VDF proof for the given challenge using the specified algorithm
pub fn compute_vdf_proof(challenge: &VdfChallenge, algorithm: VdfAlgorithm) -> Result<VdfProof> {
    VdfComputer::compute_vdf(challenge, algorithm)
}

/// Verify a VDF proof against the given challenge
pub fn verify_vdf_proof(challenge: &VdfChallenge, proof: &VdfProof) -> Result<bool> {
    VdfComputer::verify_vdf_proof(challenge, proof)
}

/// Quick VDF computation using SimpleSequential algorithm with low difficulty
pub fn compute_quick_vdf_proof(prev_attestation_sig: &[u8]) -> Result<(VdfChallenge, VdfProof)> {
    let (modulus, generator) = generate_vdf_parameters(64)?;
    let mut challenge = create_vdf_challenge(modulus, generator, 10, prev_attestation_sig);

    // Adjust for quick computation
    challenge.min_required_time_ms = 0;
    challenge.max_allowed_time_ms = 5000;

    let proof = compute_vdf_proof(&challenge, VdfAlgorithm::SimpleSequential)?;
    Ok((challenge, proof))
}

/// Comprehensive VDF computation with higher difficulty for production use
pub fn compute_production_vdf_proof(
    modulus: Vec<u8>,
    generator: Vec<u8>,
    difficulty: u64,
    prev_attestation_sig: &[u8],
    algorithm: VdfAlgorithm,
) -> Result<(VdfChallenge, VdfProof)> {
    let challenge = create_vdf_challenge(modulus, generator, difficulty, prev_attestation_sig);
    let proof = compute_vdf_proof(&challenge, algorithm)?;
    Ok((challenge, proof))
}

/// Validate VDF parameters for correctness
pub fn validate_vdf_parameters(params: &VdfParameters) -> bool {
    params.is_valid()
}

/// Get VDF algorithm from string representation
pub fn parse_vdf_algorithm(algorithm: &str) -> Option<VdfAlgorithm> {
    match algorithm.to_lowercase().as_str() {
        "wesolowski" => Some(VdfAlgorithm::Wesolowski),
        "pietrzak" => Some(VdfAlgorithm::Pietrzak),
        "simple" | "sequential" | "simple_sequential" => Some(VdfAlgorithm::SimpleSequential),
        _ => None,
    }
}

/// Get string representation of VDF algorithm
pub fn vdf_algorithm_to_string(algorithm: &VdfAlgorithm) -> &'static str {
    match algorithm {
        VdfAlgorithm::Wesolowski => "wesolowski",
        VdfAlgorithm::Pietrzak => "pietrzak",
        VdfAlgorithm::SimpleSequential => "simple_sequential",
    }
}

/// Estimate VDF computation time based on difficulty
pub fn estimate_vdf_time(difficulty: u64) -> u64 {
    let base_time_per_squaring_us = 10;
    (difficulty * base_time_per_squaring_us) / 1000
}

/// Check if VDF proof meets time requirements
pub fn validate_vdf_timing(challenge: &VdfChallenge, proof: &VdfProof) -> bool {
    challenge.is_within_time_bounds(proof.computation_time_ms)
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_vdf_parameter_generation() {
        let (modulus, generator) = generate_vdf_parameters(128).unwrap();
        assert!(!modulus.is_empty());
        assert!(!generator.is_empty());
    }

    #[test]
    fn test_vdf_challenge_creation() {
        let (modulus, generator) = generate_vdf_parameters(64).unwrap();
        let prev_sig = b"test_signature";

        let challenge = create_vdf_challenge(modulus, generator, 100, prev_sig);
        assert_eq!(challenge.parameters.difficulty, 100);
        assert!(validate_vdf_parameters(&challenge.parameters));
    }

    #[test]
    fn test_quick_vdf_computation() {
        let prev_sig = b"test_signature";
        let (challenge, proof) = compute_quick_vdf_proof(prev_sig).unwrap();

        assert!(proof.is_valid());
        let is_valid = verify_vdf_proof(&challenge, &proof).unwrap();
        assert!(is_valid);
    }

    #[test]
    fn test_vdf_algorithm_parsing() {
        assert_eq!(
            parse_vdf_algorithm("wesolowski"),
            Some(VdfAlgorithm::Wesolowski)
        );
        assert_eq!(
            parse_vdf_algorithm("pietrzak"),
            Some(VdfAlgorithm::Pietrzak)
        );
        assert_eq!(
            parse_vdf_algorithm("simple"),
            Some(VdfAlgorithm::SimpleSequential)
        );
        assert_eq!(parse_vdf_algorithm("invalid"), None);
    }

    #[test]
    fn test_vdf_algorithm_string_conversion() {
        assert_eq!(
            vdf_algorithm_to_string(&VdfAlgorithm::Wesolowski),
            "wesolowski"
        );
        assert_eq!(vdf_algorithm_to_string(&VdfAlgorithm::Pietrzak), "pietrzak");
        assert_eq!(
            vdf_algorithm_to_string(&VdfAlgorithm::SimpleSequential),
            "simple_sequential"
        );
    }

    #[test]
    fn test_vdf_time_estimation() {
        let difficulty = 1000;
        let estimated_time = estimate_vdf_time(difficulty);
        assert_eq!(estimated_time, 10); // 1000 * 10 / 1000 = 10ms
    }

    #[test]
    fn test_vdf_timing_validation() {
        let (modulus, generator) = generate_vdf_parameters(64).unwrap();
        let prev_sig = b"test_signature";
        let challenge = create_vdf_challenge(modulus, generator, 10, prev_sig);

        let valid_proof = VdfProof::new(
            vec![1, 2, 3],
            vec![4, 5, 6],
            challenge.expected_computation_time_ms,
            VdfAlgorithm::SimpleSequential,
        );

        assert!(validate_vdf_timing(&challenge, &valid_proof));

        if challenge.min_required_time_ms > 0 {
            let too_fast_proof = VdfProof::new(
                vec![1, 2, 3],
                vec![4, 5, 6],
                challenge.min_required_time_ms - 1,
                VdfAlgorithm::SimpleSequential,
            );

            assert!(!validate_vdf_timing(&challenge, &too_fast_proof));
        }
    }

    #[test]
    fn test_production_vdf_computation() {
        let (modulus, generator) = generate_vdf_parameters(64).unwrap();
        let prev_sig = b"test_signature";

        let (challenge, proof) = compute_production_vdf_proof(
            modulus,
            generator,
            5, // Very low difficulty for testing
            prev_sig,
            VdfAlgorithm::SimpleSequential,
        )
        .unwrap();

        assert!(proof.is_valid());
        let is_valid = verify_vdf_proof(&challenge, &proof).unwrap();
        assert!(is_valid);
    }
}
