//! Replay attack prevention using VDF challenges
//!
//! This module ensures attestations are fresh and cannot be replayed
//! by requiring computation of time-bound VDF challenges.

use anyhow::{Context, Result};
use chrono::{DateTime, Duration, Utc};
use serde::{Deserialize, Serialize};
use sha2::{Digest, Sha256};
use std::collections::HashMap;

use crate::vdf::types::VdfProof;

#[derive(Debug, Clone)]
pub struct ReplayGuard {
    /// Maximum age of a valid attestation
    max_attestation_age: Duration,
    /// Minimum VDF difficulty
    min_vdf_difficulty: u64,
    /// Used challenges cache to prevent reuse
    used_challenges: HashMap<String, DateTime<Utc>>,
}

impl Default for ReplayGuard {
    fn default() -> Self {
        Self::new()
    }
}

impl ReplayGuard {
    pub fn new() -> Self {
        Self {
            max_attestation_age: Duration::minutes(5),
            min_vdf_difficulty: 100_000,
            used_challenges: HashMap::new(),
        }
    }

    pub fn with_max_age(mut self, max_age: Duration) -> Self {
        self.max_attestation_age = max_age;
        self
    }

    pub fn with_min_difficulty(mut self, difficulty: u64) -> Self {
        self.min_vdf_difficulty = difficulty;
        self
    }

    /// Generate a fresh challenge for attestation
    pub fn generate_challenge(&self, validator_id: &str) -> Result<AttestationChallenge> {
        use crate::vdf::VdfComputer;

        let timestamp = Utc::now();
        let nonce = self.generate_nonce();

        // Generate VDF parameters for this challenge
        let (modulus, generator) = VdfComputer::generate_public_params(2048)
            .context("Failed to generate VDF parameters")?;

        // Create challenge incorporating validator ID and timestamp
        let challenge_data = format!(
            "{}:{}:{}:{}",
            validator_id,
            timestamp.timestamp_nanos_opt().unwrap_or(0),
            nonce,
            self.min_vdf_difficulty
        );

        let challenge_hash = Self::hash_challenge(&challenge_data);

        Ok(AttestationChallenge {
            validator_id: validator_id.to_string(),
            timestamp,
            nonce,
            challenge_hash,
            difficulty: self.min_vdf_difficulty,
            expires_at: timestamp + self.max_attestation_age,
            vdf_modulus: modulus,
            vdf_generator: generator,
        })
    }

    /// Verify an attestation response
    pub fn verify_response(
        &mut self,
        challenge: &AttestationChallenge,
        response: &AttestationResponse,
    ) -> Result<VerificationResult> {
        // Check if challenge has expired
        let now = Utc::now();
        if now > challenge.expires_at {
            return Ok(VerificationResult {
                is_valid: false,
                reason: "Challenge has expired".to_string(),
                timestamp_valid: false,
                challenge_valid: false,
                vdf_valid: false,
                not_replayed: false,
            });
        }

        // Check if response timestamp is reasonable
        let time_diff = response
            .timestamp
            .signed_duration_since(challenge.timestamp);
        if time_diff < chrono::Duration::zero() {
            return Ok(VerificationResult {
                is_valid: false,
                reason: "Response timestamp is before challenge".to_string(),
                timestamp_valid: false,
                challenge_valid: true,
                vdf_valid: false,
                not_replayed: true,
            });
        }

        if time_diff > self.max_attestation_age {
            return Ok(VerificationResult {
                is_valid: false,
                reason: "Response took too long".to_string(),
                timestamp_valid: false,
                challenge_valid: true,
                vdf_valid: false,
                not_replayed: true,
            });
        }

        // Verify challenge hasn't been used before
        if self.used_challenges.contains_key(&challenge.challenge_hash) {
            return Ok(VerificationResult {
                is_valid: false,
                reason: "Challenge has already been used".to_string(),
                timestamp_valid: true,
                challenge_valid: true,
                vdf_valid: false,
                not_replayed: false,
            });
        }

        // Verify VDF proof
        let vdf_valid = self.verify_vdf_proof(challenge, &response.vdf_proof)?;

        if !vdf_valid {
            return Ok(VerificationResult {
                is_valid: false,
                reason: "Invalid VDF proof".to_string(),
                timestamp_valid: true,
                challenge_valid: true,
                vdf_valid: false,
                not_replayed: true,
            });
        }

        // Mark challenge as used
        self.used_challenges
            .insert(challenge.challenge_hash.clone(), now);

        // Clean up old challenges
        self.cleanup_old_challenges(now);

        Ok(VerificationResult {
            is_valid: true,
            reason: "Valid attestation".to_string(),
            timestamp_valid: true,
            challenge_valid: true,
            vdf_valid: true,
            not_replayed: true,
        })
    }

    fn generate_nonce(&self) -> String {
        use rand::Rng;
        let mut rng = rand::thread_rng();
        let nonce: u128 = rng.gen();
        format!("{nonce:032x}")
    }

    fn hash_challenge(data: &str) -> String {
        let mut hasher = Sha256::new();
        hasher.update(data.as_bytes());
        format!("{:x}", hasher.finalize())
    }

    fn verify_vdf_proof(&self, challenge: &AttestationChallenge, proof: &VdfProof) -> Result<bool> {
        use crate::vdf::VdfComputer;

        // First check computation time is reasonable
        let expected_time_ms = self.estimate_computation_time(challenge.difficulty);
        let time_ratio = proof.computation_time_ms as f64 / expected_time_ms as f64;

        // Allow 20% faster (powerful hardware) to 5x slower (weak hardware)
        if !(0.8..=5.0).contains(&time_ratio) {
            tracing::warn!(
                "VDF computation time suspicious: {} ms (expected ~{} ms, ratio: {:.2})",
                proof.computation_time_ms,
                expected_time_ms,
                time_ratio
            );
            return Ok(false);
        }

        // Create VDF challenge from attestation challenge
        let prev_attestation_sig = challenge.challenge_hash.as_bytes();
        let vdf_challenge = VdfComputer::create_challenge(
            challenge.vdf_modulus.clone(),
            challenge.vdf_generator.clone(),
            challenge.difficulty,
            prev_attestation_sig,
        );

        // Verify the proof using the VDF computer
        match VdfComputer::verify_vdf_proof(&vdf_challenge, proof) {
            Ok(is_valid) => {
                if !is_valid {
                    tracing::warn!("VDF proof verification failed");
                    return Ok(false);
                }
            }
            Err(e) => {
                tracing::warn!("VDF verification error: {}", e);
                return Ok(false);
            }
        }

        tracing::info!(
            "VDF proof verified successfully (algorithm: {:?}, time: {} ms)",
            proof.algorithm,
            proof.computation_time_ms
        );

        Ok(true)
    }

    fn estimate_computation_time(&self, difficulty: u64) -> u64 {
        // Estimate based on typical single-threaded performance
        // Roughly 1ms per 100 iterations on modern hardware
        // For small difficulties (like in tests), return a minimum of 10ms
        ((difficulty / 100).max(1)).max(10)
    }

    fn cleanup_old_challenges(&mut self, now: DateTime<Utc>) {
        let cutoff = now - self.max_attestation_age * 2;
        self.used_challenges
            .retain(|_, timestamp| *timestamp > cutoff);
    }

    /// Create a time-bound attestation package
    pub fn create_attestation_package(
        &self,
        challenge: &AttestationChallenge,
        hardware_data: &[u8],
        vdf_proof: VdfProof,
    ) -> Result<AttestationPackage> {
        let timestamp = Utc::now();

        // Create attestation data hash
        let mut hasher = Sha256::new();
        hasher.update(&challenge.challenge_hash);
        hasher.update(hardware_data);
        hasher.update(timestamp.timestamp_nanos_opt().unwrap_or(0).to_le_bytes());
        let data_hash = format!("{:x}", hasher.finalize());

        Ok(AttestationPackage {
            challenge_hash: challenge.challenge_hash.clone(),
            timestamp,
            data_hash,
            vdf_proof,
            hardware_data_size: hardware_data.len(),
        })
    }
}

#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct AttestationChallenge {
    pub validator_id: String,
    pub timestamp: DateTime<Utc>,
    pub nonce: String,
    pub challenge_hash: String,
    pub difficulty: u64,
    pub expires_at: DateTime<Utc>,
    /// VDF parameters needed for verification
    pub vdf_modulus: Vec<u8>,
    pub vdf_generator: Vec<u8>,
}

#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct AttestationResponse {
    pub challenge_hash: String,
    pub timestamp: DateTime<Utc>,
    pub vdf_proof: VdfProof,
    pub attestation_data_hash: String,
}

#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct AttestationPackage {
    pub challenge_hash: String,
    pub timestamp: DateTime<Utc>,
    pub data_hash: String,
    pub vdf_proof: VdfProof,
    pub hardware_data_size: usize,
}

#[derive(Debug, Clone)]
pub struct VerificationResult {
    pub is_valid: bool,
    pub reason: String,
    pub timestamp_valid: bool,
    pub challenge_valid: bool,
    pub vdf_valid: bool,
    pub not_replayed: bool,
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_replay_guard_creation() {
        let guard = ReplayGuard::new();
        assert_eq!(guard.min_vdf_difficulty, 100_000);
    }

    #[test]
    fn test_challenge_generation() {
        let guard = ReplayGuard::new();
        let challenge = guard.generate_challenge("validator-123").unwrap();

        assert_eq!(challenge.validator_id, "validator-123");
        assert_eq!(challenge.difficulty, 100_000);
        assert!(!challenge.nonce.is_empty());
        assert!(!challenge.challenge_hash.is_empty());
        assert!(!challenge.vdf_modulus.is_empty());
        assert!(!challenge.vdf_generator.is_empty());
    }

    #[test]
    fn test_challenge_uniqueness() {
        let guard = ReplayGuard::new();
        let challenge1 = guard.generate_challenge("validator-123").unwrap();
        let challenge2 = guard.generate_challenge("validator-123").unwrap();

        // Challenges should be unique even for same validator
        assert_ne!(challenge1.challenge_hash, challenge2.challenge_hash);
        assert_ne!(challenge1.nonce, challenge2.nonce);
    }

    #[test]
    fn test_expired_challenge() {
        let mut guard = ReplayGuard::new();
        let challenge = guard.generate_challenge("validator-123").unwrap();

        // Create a response with expired timestamp
        let mut expired_challenge = challenge.clone();
        expired_challenge.expires_at = Utc::now() - Duration::hours(1);

        let response = AttestationResponse {
            challenge_hash: challenge.challenge_hash.clone(),
            timestamp: Utc::now(),
            vdf_proof: VdfProof {
                output: vec![1, 2, 3, 4],
                proof: vec![5, 6, 7, 8],
                computation_time_ms: 100,
                algorithm: crate::vdf::types::VdfAlgorithm::SimpleSequential,
            },
            attestation_data_hash: "dummy_hash".to_string(),
        };

        let result = guard
            .verify_response(&expired_challenge, &response)
            .unwrap();
        assert!(!result.is_valid);
        assert!(result.reason.contains("expired"));
    }
}
