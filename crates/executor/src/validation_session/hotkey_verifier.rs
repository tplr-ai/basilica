//! Bittensor hotkey signature verification for validator authentication
//!
//! This module implements cryptographic signature verification for Bittensor validator hotkeys,
//! providing a secure challenge-response authentication mechanism that replaces simple SSH-based
//! access control.

use anyhow::{Context, Result};
use common::crypto::P256PublicKey;
use serde::{Deserialize, Serialize};
use std::collections::HashMap;
use std::time::{Duration, SystemTime};
use tokio::sync::RwLock;
use tracing::{debug, info, warn};

/// Challenge data for hotkey signature verification
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct SignatureChallenge {
    pub challenge_id: String,
    pub challenge_data: Vec<u8>,
    pub created_at: SystemTime,
    pub expires_at: SystemTime,
    pub validator_hotkey: String,
}

impl SignatureChallenge {
    /// Create a new signature challenge for a validator
    pub fn new(validator_hotkey: String, timeout: Duration) -> Self {
        let now = SystemTime::now();
        let challenge_id = format!(
            "challenge_{}_{}",
            validator_hotkey,
            now.duration_since(SystemTime::UNIX_EPOCH)
                .unwrap_or_default()
                .as_millis()
        );

        // Generate random challenge data (32 bytes)
        let challenge_data = (0..32).map(|_| rand::random::<u8>()).collect();

        Self {
            challenge_id,
            challenge_data,
            created_at: now,
            expires_at: now + timeout,
            validator_hotkey,
        }
    }

    /// Check if the challenge has expired
    pub fn is_expired(&self) -> bool {
        SystemTime::now() >= self.expires_at
    }

    /// Get time remaining until expiry
    pub fn time_remaining(&self) -> Option<Duration> {
        self.expires_at.duration_since(SystemTime::now()).ok()
    }
}

/// Configuration for hotkey signature verification
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct HotkeyVerificationConfig {
    pub enabled: bool,
    pub challenge_timeout_seconds: u64,
    pub max_signature_attempts: u32,
    pub cleanup_interval_seconds: u64,
}

impl Default for HotkeyVerificationConfig {
    fn default() -> Self {
        Self {
            enabled: true,
            challenge_timeout_seconds: 300, // 5 minutes
            max_signature_attempts: 3,
            cleanup_interval_seconds: 600, // 10 minutes
        }
    }
}

/// Signature verification attempt tracking
#[derive(Debug, Clone)]
struct VerificationAttempt {
    #[allow(dead_code)]
    validator_hotkey: String,
    attempts: u32,
    last_attempt: SystemTime,
}

/// Bittensor hotkey signature verifier
pub struct HotkeySignatureVerifier {
    config: HotkeyVerificationConfig,
    active_challenges: RwLock<HashMap<String, SignatureChallenge>>,
    verification_attempts: RwLock<HashMap<String, VerificationAttempt>>,
}

impl HotkeySignatureVerifier {
    /// Create a new hotkey signature verifier
    pub fn new(config: HotkeyVerificationConfig) -> Self {
        Self {
            config,
            active_challenges: RwLock::new(HashMap::new()),
            verification_attempts: RwLock::new(HashMap::new()),
        }
    }

    /// Generate a new signature challenge for a validator hotkey
    pub async fn generate_challenge(&self, validator_hotkey: &str) -> Result<SignatureChallenge> {
        if !self.config.enabled {
            return Err(anyhow::anyhow!("Hotkey verification is disabled"));
        }

        // Check if validator has exceeded attempt limits
        if self.has_exceeded_attempts(validator_hotkey).await {
            return Err(anyhow::anyhow!(
                "Maximum signature attempts exceeded for validator: {}",
                validator_hotkey
            ));
        }

        let timeout = Duration::from_secs(self.config.challenge_timeout_seconds);
        let challenge = SignatureChallenge::new(validator_hotkey.to_string(), timeout);

        // Store the challenge
        let mut challenges = self.active_challenges.write().await;
        challenges.insert(challenge.challenge_id.clone(), challenge.clone());

        info!(
            "Generated signature challenge {} for validator: {}",
            challenge.challenge_id, validator_hotkey
        );

        Ok(challenge)
    }

    /// Verify a signature response to a challenge
    pub async fn verify_signature(
        &self,
        challenge_id: &str,
        signature_bytes: &[u8],
        public_key_bytes: &[u8],
    ) -> Result<bool> {
        if !self.config.enabled {
            return Ok(true); // Allow if verification is disabled
        }

        // Retrieve and validate challenge
        let challenge = {
            let challenges = self.active_challenges.read().await;
            challenges.get(challenge_id).cloned()
        };

        let challenge = match challenge {
            Some(c) => c,
            None => {
                warn!("Invalid or expired challenge ID: {}", challenge_id);
                return Ok(false);
            }
        };

        if challenge.is_expired() {
            warn!("Challenge {} has expired", challenge_id);
            self.remove_challenge(challenge_id).await;
            return Ok(false);
        }

        // Track verification attempt
        self.track_verification_attempt(&challenge.validator_hotkey)
            .await?;

        // Verify the signature
        let verification_result = self
            .verify_p256_signature(&challenge.challenge_data, signature_bytes, public_key_bytes)
            .await;

        match verification_result {
            Ok(true) => {
                info!(
                    "Signature verification successful for challenge: {}",
                    challenge_id
                );
                // Remove the challenge on successful verification
                self.remove_challenge(challenge_id).await;
                // Reset attempt counter on success
                self.reset_verification_attempts(&challenge.validator_hotkey)
                    .await;
                Ok(true)
            }
            Ok(false) => {
                warn!(
                    "Signature verification failed for challenge: {}",
                    challenge_id
                );
                Ok(false)
            }
            Err(e) => {
                warn!(
                    "Error during signature verification for challenge {}: {}",
                    challenge_id, e
                );
                Ok(false)
            }
        }
    }

    /// Verify P256 ECDSA signature
    async fn verify_p256_signature(
        &self,
        message: &[u8],
        signature_bytes: &[u8],
        public_key_bytes: &[u8],
    ) -> Result<bool> {
        // Parse the public key from compressed bytes
        let public_key = P256PublicKey::from_compressed_bytes(public_key_bytes)
            .context("Failed to parse P256 public key")?;

        // Verify the signature using our crypto module
        match common::crypto::verify_p256_signature(&public_key, message, signature_bytes) {
            Ok(()) => {
                debug!("P256 signature verification successful");
                Ok(true)
            }
            Err(e) => {
                debug!("P256 signature verification failed: {}", e);
                Ok(false)
            }
        }
    }

    /// Check if validator has exceeded signature attempt limits
    async fn has_exceeded_attempts(&self, validator_hotkey: &str) -> bool {
        let attempts = self.verification_attempts.read().await;
        if let Some(attempt) = attempts.get(validator_hotkey) {
            attempt.attempts >= self.config.max_signature_attempts
        } else {
            false
        }
    }

    /// Track a verification attempt for rate limiting
    async fn track_verification_attempt(&self, validator_hotkey: &str) -> Result<()> {
        let mut attempts = self.verification_attempts.write().await;
        let now = SystemTime::now();

        match attempts.get_mut(validator_hotkey) {
            Some(attempt) => {
                attempt.attempts += 1;
                attempt.last_attempt = now;
            }
            None => {
                attempts.insert(
                    validator_hotkey.to_string(),
                    VerificationAttempt {
                        validator_hotkey: validator_hotkey.to_string(),
                        attempts: 1,
                        last_attempt: now,
                    },
                );
            }
        }

        Ok(())
    }

    /// Reset verification attempts for a validator (on successful auth)
    async fn reset_verification_attempts(&self, validator_hotkey: &str) {
        let mut attempts = self.verification_attempts.write().await;
        attempts.remove(validator_hotkey);
    }

    /// Remove a challenge from active challenges
    async fn remove_challenge(&self, challenge_id: &str) {
        let mut challenges = self.active_challenges.write().await;
        challenges.remove(challenge_id);
    }

    /// Clean up expired challenges and attempt records
    pub async fn cleanup_expired(&self) -> Result<u32> {
        let mut cleaned = 0;
        let now = SystemTime::now();

        // Clean up expired challenges
        {
            let mut challenges = self.active_challenges.write().await;
            let expired_ids: Vec<String> = challenges
                .iter()
                .filter(|(_, challenge)| challenge.is_expired())
                .map(|(id, _)| id.clone())
                .collect();

            for id in expired_ids {
                challenges.remove(&id);
                cleaned += 1;
            }
        }

        // Clean up old verification attempts (older than 1 hour)
        {
            let mut attempts = self.verification_attempts.write().await;
            let cutoff = now - Duration::from_secs(3600);
            attempts.retain(|_, attempt| attempt.last_attempt > cutoff);
        }

        if cleaned > 0 {
            debug!("Cleaned up {} expired signature challenges", cleaned);
        }

        Ok(cleaned)
    }

    /// Get active challenge statistics
    pub async fn get_stats(&self) -> (usize, usize) {
        let challenges = self.active_challenges.read().await;
        let attempts = self.verification_attempts.read().await;
        (challenges.len(), attempts.len())
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use common::crypto::P256KeyPair;

    #[tokio::test]
    async fn test_challenge_generation() {
        let config = HotkeyVerificationConfig::default();
        let verifier = HotkeySignatureVerifier::new(config);

        let challenge = verifier.generate_challenge("test_validator").await.unwrap();

        assert_eq!(challenge.validator_hotkey, "test_validator");
        assert_eq!(challenge.challenge_data.len(), 32);
        assert!(!challenge.is_expired());
    }

    #[tokio::test]
    async fn test_signature_verification() {
        let config = HotkeyVerificationConfig::default();
        let verifier = HotkeySignatureVerifier::new(config);

        // Generate a test key pair
        let key_pair = P256KeyPair::generate();
        let public_key = key_pair.public_key();

        // Generate challenge
        let challenge = verifier.generate_challenge("test_validator").await.unwrap();

        // Sign the challenge
        let signature = key_pair.private_key().sign(&challenge.challenge_data);

        // Verify the signature
        let result = verifier
            .verify_signature(
                &challenge.challenge_id,
                &signature.to_bytes(),
                &public_key.to_compressed_bytes(),
            )
            .await
            .unwrap();

        assert!(result);
    }

    #[tokio::test]
    async fn test_expired_challenge() {
        let config = HotkeyVerificationConfig {
            challenge_timeout_seconds: 1, // 1 second timeout
            ..Default::default()
        };
        let verifier = HotkeySignatureVerifier::new(config);

        let challenge = verifier.generate_challenge("test_validator").await.unwrap();

        // Wait for challenge to expire
        tokio::time::sleep(Duration::from_secs(2)).await;

        // Try to verify expired challenge
        let result = verifier
            .verify_signature(&challenge.challenge_id, &[0u8; 64], &[0u8; 33])
            .await
            .unwrap();

        assert!(!result);
    }

    #[tokio::test]
    async fn test_attempt_limiting() {
        let config = HotkeyVerificationConfig {
            max_signature_attempts: 2,
            ..Default::default()
        };
        let verifier = HotkeySignatureVerifier::new(config);

        // Generate challenge and make attempts
        let challenge1 = verifier.generate_challenge("test_validator").await.unwrap();

        // First attempt (fail)
        verifier
            .verify_signature(&challenge1.challenge_id, &[0u8; 64], &[0u8; 33])
            .await
            .unwrap();

        // Second attempt (fail)
        let challenge2 = verifier.generate_challenge("test_validator").await.unwrap();
        verifier
            .verify_signature(&challenge2.challenge_id, &[0u8; 64], &[0u8; 33])
            .await
            .unwrap();

        // Third attempt should be denied
        let result = verifier.generate_challenge("test_validator").await;
        assert!(result.is_err());
    }
}
