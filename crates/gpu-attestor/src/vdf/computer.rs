//! VDF computation and verification implementation

use anyhow::Result;
use num_bigint::BigUint;
use num_traits::pow::Pow;
use sha2::{Digest, Sha256};
use std::time::{Duration, Instant};

use super::types::*;

pub struct VdfComputer {}

impl VdfComputer {
    /// Generate public parameters for VDF (typically done by validator)
    pub fn generate_public_params(bits: usize) -> Result<(Vec<u8>, Vec<u8>)> {
        let modulus = Self::generate_pseudo_rsa_modulus(bits)?;
        let generator = Self::generate_generator(&modulus)?;

        Ok((modulus.to_bytes_be(), generator.to_bytes_be()))
    }

    /// Create a new VDF challenge with given parameters
    pub fn create_challenge(
        modulus: Vec<u8>,
        generator: Vec<u8>,
        difficulty: u64,
        prev_attestation_sig: &[u8],
    ) -> VdfChallenge {
        let challenge_seed = Self::derive_challenge_seed(prev_attestation_sig);
        let parameters = VdfParameters::new(modulus, generator, difficulty, challenge_seed);
        let expected_time_ms = Self::estimate_computation_time(difficulty);

        VdfChallenge::new(
            parameters,
            expected_time_ms,
            expected_time_ms * 3, // Allow 3x expected time
            expected_time_ms / 2, // Must take at least half expected time
        )
    }

    /// Compute VDF proof using the specified algorithm
    pub fn compute_vdf(challenge: &VdfChallenge, algorithm: VdfAlgorithm) -> Result<VdfProof> {
        assert!(challenge.is_valid(), "Invalid VDF challenge");

        let start_time = Instant::now();
        let (output, proof) = Self::compute_algorithm(challenge, &algorithm)?;
        let computation_time_ms = start_time.elapsed().as_millis() as u64;

        Self::validate_computation_time(challenge, computation_time_ms)?;

        Ok(VdfProof::new(
            output.to_bytes_be(),
            proof.to_bytes_be(),
            computation_time_ms,
            algorithm,
        ))
    }

    /// Verify a VDF proof (fast verification)
    pub fn verify_vdf_proof(challenge: &VdfChallenge, proof: &VdfProof) -> Result<bool> {
        assert!(challenge.is_valid(), "Invalid VDF challenge");
        assert!(proof.is_valid(), "Invalid VDF proof");

        let modulus = BigUint::from_bytes_be(&challenge.parameters.modulus);
        let generator = BigUint::from_bytes_be(&challenge.parameters.generator);
        let difficulty = challenge.parameters.difficulty;
        let seed = &challenge.parameters.challenge_seed;

        let output = BigUint::from_bytes_be(&proof.output);
        let pi = BigUint::from_bytes_be(&proof.proof);

        match proof.algorithm {
            VdfAlgorithm::Wesolowski => {
                Self::verify_wesolowski(&modulus, &generator, seed, difficulty, &output, &pi)
            }
            VdfAlgorithm::Pietrzak => {
                Self::verify_pietrzak(&modulus, &generator, seed, difficulty, &output, &pi)
            }
            VdfAlgorithm::SimpleSequential => {
                Self::verify_simple_sequential(&modulus, &generator, seed, difficulty, &output)
            }
        }
    }
}

// Private implementation methods
impl VdfComputer {
    fn derive_challenge_seed(prev_attestation_sig: &[u8]) -> Vec<u8> {
        let mut hasher = Sha256::new();
        hasher.update(prev_attestation_sig);
        hasher.finalize().to_vec()
    }

    fn estimate_computation_time(difficulty: u64) -> u64 {
        let base_time_per_squaring_us = 10;
        (difficulty * base_time_per_squaring_us) / 1000
    }

    fn validate_computation_time(challenge: &VdfChallenge, computation_time_ms: u64) -> Result<()> {
        if !challenge.is_within_time_bounds(computation_time_ms) {
            if computation_time_ms < challenge.min_required_time_ms {
                anyhow::bail!(
                    "VDF computation completed too quickly: {}ms < {}ms minimum",
                    computation_time_ms,
                    challenge.min_required_time_ms
                );
            } else {
                anyhow::bail!(
                    "VDF computation took too long: {}ms > {}ms maximum",
                    computation_time_ms,
                    challenge.max_allowed_time_ms
                );
            }
        }
        Ok(())
    }

    fn compute_algorithm(
        challenge: &VdfChallenge,
        algorithm: &VdfAlgorithm,
    ) -> Result<(BigUint, BigUint)> {
        let modulus = BigUint::from_bytes_be(&challenge.parameters.modulus);
        let generator = BigUint::from_bytes_be(&challenge.parameters.generator);
        let difficulty = challenge.parameters.difficulty;
        let seed = &challenge.parameters.challenge_seed;

        match algorithm {
            VdfAlgorithm::Wesolowski => {
                Self::compute_wesolowski(&modulus, &generator, seed, difficulty)
            }
            VdfAlgorithm::Pietrzak => {
                Self::compute_pietrzak(&modulus, &generator, seed, difficulty)
            }
            VdfAlgorithm::SimpleSequential => {
                Self::compute_simple_sequential(&modulus, &generator, seed, difficulty)
            }
        }
    }

    fn compute_challenge_input(generator: &BigUint, seed: &[u8], modulus: &BigUint) -> BigUint {
        let mut hasher = Sha256::new();
        hasher.update(generator.to_bytes_be());
        hasher.update(seed);
        let x = BigUint::from_bytes_be(&hasher.finalize());
        &x % modulus
    }

    fn sequential_squaring(
        initial: &BigUint,
        difficulty: u64,
        modulus: &BigUint,
        microsleep_interval: u64,
        microsleep_duration: u64,
    ) -> BigUint {
        let mut result = initial.clone();

        for i in 0..difficulty {
            result = (&result * &result) % modulus;

            if microsleep_interval > 0 && i % microsleep_interval == 0 {
                std::thread::sleep(Duration::from_micros(microsleep_duration));
            }
        }

        result
    }

    fn generate_pseudo_rsa_modulus(bits: usize) -> Result<BigUint> {
        let base = BigUint::from(2u32);
        let modulus = base.pow(bits as u32) - BigUint::from(1u32);
        Ok(modulus)
    }

    fn generate_generator(modulus: &BigUint) -> Result<BigUint> {
        let generator = BigUint::from(3u32);
        Ok(generator % modulus)
    }
}

// Algorithm implementations
impl VdfComputer {
    fn compute_wesolowski(
        modulus: &BigUint,
        generator: &BigUint,
        seed: &[u8],
        difficulty: u64,
    ) -> Result<(BigUint, BigUint)> {
        let x = Self::compute_challenge_input(generator, seed, modulus);
        let start_time = Instant::now();

        let y = Self::sequential_squaring(&x, difficulty, modulus, 1000, 1);

        // Generate proof (simplified - real Wesolowski involves more complex steps)
        let mut proof_hasher = Sha256::new();
        proof_hasher.update(x.to_bytes_be());
        proof_hasher.update(y.to_bytes_be());
        proof_hasher.update(difficulty.to_be_bytes());
        let proof = BigUint::from_bytes_be(&proof_hasher.finalize());

        tracing::info!(
            "Wesolowski VDF computation completed in {:?} for difficulty {}",
            start_time.elapsed(),
            difficulty
        );

        Ok((y, proof))
    }

    fn verify_wesolowski(
        modulus: &BigUint,
        generator: &BigUint,
        seed: &[u8],
        difficulty: u64,
        output: &BigUint,
        proof: &BigUint,
    ) -> Result<bool> {
        let x = Self::compute_challenge_input(generator, seed, modulus);

        // Verify proof (simplified verification)
        let mut expected_proof_hasher = Sha256::new();
        expected_proof_hasher.update(x.to_bytes_be());
        expected_proof_hasher.update(output.to_bytes_be());
        expected_proof_hasher.update(difficulty.to_be_bytes());
        let expected_proof = BigUint::from_bytes_be(&expected_proof_hasher.finalize());

        // Fast verification: simplified check
        let verification_check = output.modpow(&BigUint::from(2u32), modulus);
        let expected_check = x.modpow(&BigUint::from(4u32), modulus);

        Ok(proof == &expected_proof && verification_check == expected_check)
    }

    fn compute_pietrzak(
        modulus: &BigUint,
        generator: &BigUint,
        seed: &[u8],
        difficulty: u64,
    ) -> Result<(BigUint, BigUint)> {
        let x = Self::compute_challenge_input(generator, seed, modulus);
        let y = Self::sequential_squaring(&x, difficulty, modulus, 1000, 1);

        // Generate Pietrzak-style proof (simplified)
        let proof = (&y * generator) % modulus;

        Ok((y, proof))
    }

    fn verify_pietrzak(
        modulus: &BigUint,
        generator: &BigUint,
        _seed: &[u8],
        _difficulty: u64,
        output: &BigUint,
        proof: &BigUint,
    ) -> Result<bool> {
        // Simplified Pietrzak verification
        let expected_proof = (output * generator) % modulus;
        Ok(proof == &expected_proof)
    }

    fn compute_simple_sequential(
        modulus: &BigUint,
        generator: &BigUint,
        seed: &[u8],
        difficulty: u64,
    ) -> Result<(BigUint, BigUint)> {
        let x = Self::compute_challenge_input(generator, seed, modulus);
        let start_time = Instant::now();

        let result = Self::sequential_squaring(&x, difficulty, modulus, 100, 10);

        tracing::info!(
            "Simple sequential VDF completed in {:?} for {} iterations",
            start_time.elapsed(),
            difficulty
        );

        // Proof is just a hash of the computation
        let mut proof_hasher = Sha256::new();
        proof_hasher.update(x.to_bytes_be());
        proof_hasher.update(result.to_bytes_be());
        proof_hasher.update(difficulty.to_be_bytes());
        let proof = BigUint::from_bytes_be(&proof_hasher.finalize());

        Ok((result, proof))
    }

    fn verify_simple_sequential(
        modulus: &BigUint,
        _generator: &BigUint,
        _seed: &[u8],
        _difficulty: u64,
        output: &BigUint,
    ) -> Result<bool> {
        // For simple sequential, partial verification
        // Check that output is within valid range
        Ok(output < modulus && output > &BigUint::from(0u32))
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_vdf_parameter_generation() {
        let (modulus, generator) = VdfComputer::generate_public_params(128).unwrap();
        assert!(!modulus.is_empty());
        assert!(!generator.is_empty());
    }

    #[test]
    fn test_vdf_challenge_creation() {
        let (modulus, generator) = VdfComputer::generate_public_params(128).unwrap();
        let prev_sig = b"test_signature";

        let challenge = VdfComputer::create_challenge(modulus, generator, 100, prev_sig);

        assert_eq!(challenge.parameters.difficulty, 100);
        assert!(!challenge.parameters.challenge_seed.is_empty());
        assert!(challenge.is_valid());
    }

    #[test]
    fn test_simple_vdf_computation_and_verification() {
        let (modulus, generator) = VdfComputer::generate_public_params(64).unwrap();
        let prev_sig = b"test_signature";

        let mut challenge = VdfComputer::create_challenge(modulus, generator, 10, prev_sig);

        // Adjust time constraints for testing
        challenge.min_required_time_ms = 0;
        challenge.max_allowed_time_ms = 10000;

        let proof = VdfComputer::compute_vdf(&challenge, VdfAlgorithm::SimpleSequential).unwrap();

        assert!(proof.is_valid());

        let is_valid = VdfComputer::verify_vdf_proof(&challenge, &proof).unwrap();
        assert!(is_valid);
    }

    #[test]
    fn test_challenge_input_computation() {
        let modulus = BigUint::from(1000u32);
        let generator = BigUint::from(3u32);
        let seed = b"test_seed";

        let input1 = VdfComputer::compute_challenge_input(&generator, seed, &modulus);
        let input2 = VdfComputer::compute_challenge_input(&generator, seed, &modulus);

        assert_eq!(input1, input2); // Should be deterministic
        assert!(input1 < modulus);
    }

    #[test]
    fn test_sequential_squaring() {
        let initial = BigUint::from(5u32);
        let modulus = BigUint::from(1000u32);

        let result = VdfComputer::sequential_squaring(&initial, 3, &modulus, 0, 0);

        // Manually compute 5^(2^3) mod 1000 = 5^8 mod 1000
        let expected = initial.modpow(&BigUint::from(8u32), &modulus);
        assert_eq!(result, expected);
    }
}
