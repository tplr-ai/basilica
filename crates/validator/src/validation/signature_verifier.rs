//! P256 ECDSA signature verification for attestation validation
//!
//! Provides cryptographic signature verification capabilities for gpu-attestor
//! attestation reports using P256 ECDSA algorithm.

use super::types::{P256PublicKey, SignatureVerificationResult, ValidationError, ValidationResult};
use common::crypto::P256PublicKey as CryptoP256PublicKey;
use sha2::{Digest, Sha256};
use std::collections::HashMap;
use std::time::{Duration, SystemTime};
use tracing::{debug, info, warn};

/// P256 ECDSA signature verifier
#[derive(Debug)]
pub struct P256SignatureVerifier {
    /// Trusted public keys for verification
    trusted_keys: HashMap<String, P256PublicKey>,
    /// Verification cache to avoid repeated verifications
    verification_cache: HashMap<String, (SignatureVerificationResult, SystemTime)>,
    /// Cache TTL
    cache_ttl: Duration,
}

impl P256SignatureVerifier {
    /// Create a new P256 signature verifier
    pub fn new() -> Self {
        Self {
            trusted_keys: HashMap::new(),
            verification_cache: HashMap::new(),
            cache_ttl: Duration::from_secs(300), // 5 minutes
        }
    }

    /// Add a trusted public key for verification
    pub fn add_trusted_key(&mut self, key: P256PublicKey) {
        info!("Adding trusted P256 key: {}", key.key_id);
        self.trusted_keys.insert(key.key_id.clone(), key);
    }

    /// Remove a trusted public key
    pub fn remove_trusted_key(&mut self, key_id: &str) -> bool {
        if self.trusted_keys.remove(key_id).is_some() {
            info!("Removed trusted P256 key: {}", key_id);
            // Clear any cached verifications for this key
            self.verification_cache
                .retain(|cache_key, _| !cache_key.contains(key_id));
            true
        } else {
            false
        }
    }

    /// Get a trusted key by ID
    pub fn get_trusted_key(&self, key_id: &str) -> Option<&P256PublicKey> {
        self.trusted_keys.get(key_id)
    }

    /// List all trusted key IDs
    pub fn list_trusted_key_ids(&self) -> Vec<String> {
        self.trusted_keys.keys().cloned().collect()
    }

    /// Verify a signature against data using a specific key
    pub fn verify_signature(
        &mut self,
        data: &[u8],
        signature_hex: &str,
        key_id: &str,
    ) -> ValidationResult<SignatureVerificationResult> {
        debug!("Verifying P256 ECDSA signature with key: {}", key_id);

        // Check cache first
        let cache_key = self.generate_cache_key(data, signature_hex, key_id);
        if let Some((cached_result, cached_at)) = self.verification_cache.get(&cache_key) {
            if cached_at.elapsed().unwrap_or(Duration::MAX) < self.cache_ttl {
                debug!("Using cached signature verification result");
                return Ok(cached_result.clone());
            }
        }

        let result = self.perform_signature_verification(data, signature_hex, key_id)?;

        // Cache the result
        self.verification_cache
            .insert(cache_key, (result.clone(), SystemTime::now()));

        Ok(result)
    }

    /// Verify a signature against data using any trusted key
    pub fn verify_signature_any_key(
        &mut self,
        data: &[u8],
        signature_hex: &str,
    ) -> ValidationResult<SignatureVerificationResult> {
        debug!("Attempting signature verification with any trusted key");

        let key_ids: Vec<String> = self.trusted_keys.keys().cloned().collect();
        for key_id in key_ids {
            match self.verify_signature(data, signature_hex, &key_id) {
                Ok(result) if result.is_valid => {
                    info!("Signature verified successfully with key: {}", key_id);
                    return Ok(result);
                }
                Ok(_) => {
                    debug!("Signature verification failed with key: {}", key_id);
                    continue;
                }
                Err(e) => {
                    warn!("Error verifying signature with key {}: {}", key_id, e);
                    continue;
                }
            }
        }

        Ok(SignatureVerificationResult {
            is_valid: false,
            key_id: "none".to_string(),
            verified_at: SystemTime::now(),
            error_message: Some("No trusted key could verify the signature".to_string()),
        })
    }

    /// Verify attestation file signature
    pub fn verify_attestation_signature(
        &mut self,
        attestation_json: &str,
        signature_hex: &str,
        public_key_pem: &str,
    ) -> ValidationResult<SignatureVerificationResult> {
        info!("Verifying attestation signature");

        // Extract key ID from PEM (simplified approach)
        let key_id = self.extract_key_id_from_pem(public_key_pem)?;

        // Parse and add the public key as trusted (temporarily for this verification)
        let temp_key = self.parse_pem_to_p256_key(&key_id, public_key_pem)?;
        let was_already_trusted = self.trusted_keys.contains_key(&key_id);

        if !was_already_trusted {
            self.add_trusted_key(temp_key);
        }

        // Verify the signature
        let result = self.verify_signature(attestation_json.as_bytes(), signature_hex, &key_id);

        // Remove temporary key if it wasn't already trusted
        if !was_already_trusted {
            self.remove_trusted_key(&key_id);
        }

        result
    }

    /// Perform the actual signature verification
    fn perform_signature_verification(
        &self,
        data: &[u8],
        signature_hex: &str,
        key_id: &str,
    ) -> ValidationResult<SignatureVerificationResult> {
        let now = SystemTime::now();

        // Get the trusted key
        let trusted_key = self.trusted_keys.get(key_id).ok_or_else(|| {
            ValidationError::AttestationValidationFailed(format!("Trusted key not found: {key_id}"))
        })?;

        if !trusted_key.is_trusted {
            return Ok(SignatureVerificationResult {
                is_valid: false,
                key_id: key_id.to_string(),
                verified_at: now,
                error_message: Some("Key is not marked as trusted".to_string()),
            });
        }

        // Validate signature format
        if signature_hex.len() != 128 {
            // 64 bytes = 128 hex chars for P256 signature
            return Ok(SignatureVerificationResult {
                is_valid: false,
                key_id: key_id.to_string(),
                verified_at: now,
                error_message: Some("Invalid signature length".to_string()),
            });
        }

        if !signature_hex.chars().all(|c| c.is_ascii_hexdigit()) {
            return Ok(SignatureVerificationResult {
                is_valid: false,
                key_id: key_id.to_string(),
                verified_at: now,
                error_message: Some("Invalid signature format".to_string()),
            });
        }

        // Validate public key format
        if trusted_key.compressed_key_hex.len() != 66 {
            // 33 bytes = 66 hex chars for compressed P256 key
            return Ok(SignatureVerificationResult {
                is_valid: false,
                key_id: key_id.to_string(),
                verified_at: now,
                error_message: Some("Invalid public key length".to_string()),
            });
        }

        // Perform real P256 ECDSA verification
        let is_valid = self.perform_real_p256_verification(
            data,
            signature_hex,
            &trusted_key.compressed_key_hex,
        )?;

        Ok(SignatureVerificationResult {
            is_valid,
            key_id: key_id.to_string(),
            verified_at: now,
            error_message: if is_valid {
                None
            } else {
                Some("Signature verification failed".to_string())
            },
        })
    }

    /// Perform real P256 ECDSA signature verification
    ///
    /// Verifies a P256 ECDSA signature against the provided data using the given public key.
    /// This implementation uses the industry-standard p256 crate for cryptographic operations.
    fn perform_real_p256_verification(
        &self,
        data: &[u8],
        signature_hex: &str,
        public_key_hex: &str,
    ) -> ValidationResult<bool> {
        debug!(
            "Performing real P256 ECDSA verification for {} bytes of data",
            data.len()
        );

        // Decode hex signature (64 bytes for P256 signature)
        let signature_bytes = hex::decode(signature_hex).map_err(|e| {
            ValidationError::SignatureVerificationFailed(format!(
                "Invalid signature hex encoding: {e}"
            ))
        })?;

        if signature_bytes.len() != 64 {
            return Err(ValidationError::SignatureVerificationFailed(format!(
                "Invalid signature length: expected 64 bytes, got {}",
                signature_bytes.len()
            )));
        }

        // Decode hex public key (33 bytes for compressed P256 key)
        let pubkey_bytes = hex::decode(public_key_hex).map_err(|e| {
            ValidationError::SignatureVerificationFailed(format!(
                "Invalid public key hex encoding: {e}"
            ))
        })?;

        if pubkey_bytes.len() != 33 {
            return Err(ValidationError::SignatureVerificationFailed(format!(
                "Invalid public key length: expected 33 bytes, got {}",
                pubkey_bytes.len()
            )));
        }

        // Parse P256 public key from compressed bytes
        let public_key =
            CryptoP256PublicKey::from_compressed_bytes(&pubkey_bytes).map_err(|e| {
                ValidationError::SignatureVerificationFailed(format!(
                    "Invalid P256 public key format: {e}"
                ))
            })?;

        // Perform signature verification using our crypto module
        match common::crypto::verify_p256_signature(&public_key, data, &signature_bytes) {
            Ok(_) => {
                debug!("P256 ECDSA signature verification successful");
                Ok(true)
            }
            Err(e) => {
                debug!("P256 ECDSA signature verification failed: {}", e);
                Ok(false)
            }
        }
    }

    /// Generate cache key for verification results
    fn generate_cache_key(&self, data: &[u8], signature_hex: &str, key_id: &str) -> String {
        let mut hasher = Sha256::new();
        hasher.update(data);
        hasher.update(signature_hex.as_bytes());
        hasher.update(key_id.as_bytes());
        hex::encode(hasher.finalize())
    }

    /// Extract key ID from PEM format
    fn extract_key_id_from_pem(&self, pem_content: &str) -> ValidationResult<String> {
        // Simple key ID extraction based on content hash
        let mut hasher = Sha256::new();
        hasher.update(pem_content.as_bytes());
        let hash = hasher.finalize();
        Ok(format!("pem_{}", hex::encode(&hash[..8])))
    }

    /// Parse PEM format to P256 key structure
    fn parse_pem_to_p256_key(
        &self,
        key_id: &str,
        pem_content: &str,
    ) -> ValidationResult<P256PublicKey> {
        // Use our crypto module to parse the PEM properly
        let crypto_public_key = CryptoP256PublicKey::from_pem(pem_content).map_err(|e| {
            ValidationError::AttestationValidationFailed(format!(
                "Failed to parse P256 public key from PEM: {e}"
            ))
        })?;

        // Convert to our local P256PublicKey type
        Ok(P256PublicKey {
            key_id: key_id.to_string(),
            compressed_key_hex: crypto_public_key.to_hex(),
            created_at: SystemTime::now(),
            is_trusted: true,
        })
    }

    /// Clear verification cache
    pub fn clear_cache(&mut self) {
        self.verification_cache.clear();
        debug!("Cleared signature verification cache");
    }

    /// Get cache statistics
    pub fn get_cache_stats(&self) -> (usize, usize) {
        let total_entries = self.verification_cache.len();
        let expired_entries = self
            .verification_cache
            .values()
            .filter(|(_, cached_at)| {
                cached_at.elapsed().unwrap_or(Duration::ZERO) >= self.cache_ttl
            })
            .count();

        (total_entries, expired_entries)
    }
}

impl Default for P256SignatureVerifier {
    fn default() -> Self {
        Self::new()
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_add_remove_trusted_key() {
        let mut verifier = P256SignatureVerifier::new();

        let key = P256PublicKey {
            key_id: "test_key".to_string(),
            compressed_key_hex: "02".to_string() + &"a".repeat(64),
            created_at: SystemTime::now(),
            is_trusted: true,
        };

        verifier.add_trusted_key(key);
        assert!(verifier.get_trusted_key("test_key").is_some());

        assert!(verifier.remove_trusted_key("test_key"));
        assert!(verifier.get_trusted_key("test_key").is_none());
    }

    #[test]
    fn test_cache_key_generation() {
        let verifier = P256SignatureVerifier::new();
        let data = b"test data";
        let signature = "abcd1234";
        let key_id = "test_key";

        let cache_key1 = verifier.generate_cache_key(data, signature, key_id);
        let cache_key2 = verifier.generate_cache_key(data, signature, key_id);

        assert_eq!(cache_key1, cache_key2);
        assert_eq!(cache_key1.len(), 64); // SHA-256 hex string
    }

    #[test]
    fn test_pem_key_id_extraction() {
        let verifier = P256SignatureVerifier::new();
        let pem_content = "-----BEGIN PUBLIC KEY-----\nMFkwEwYHKoZIzj0CAQYIKoZIzj0DAQcDQgAE...\n-----END PUBLIC KEY-----";

        let key_id = verifier.extract_key_id_from_pem(pem_content).unwrap();
        assert!(key_id.starts_with("pem_"));
        assert_eq!(key_id.len(), 20); // "pem_" + 16 hex chars
    }

    #[tokio::test]
    async fn test_signature_verification_flow() {
        use common::crypto::P256KeyPair;

        let mut verifier = P256SignatureVerifier::new();

        // Generate a real key pair
        let key_pair = P256KeyPair::generate();
        let public_key = key_pair.public_key();

        // Create a valid P256 public key
        let key = P256PublicKey {
            key_id: "test_key".to_string(),
            compressed_key_hex: public_key.to_hex(),
            created_at: SystemTime::now(),
            is_trusted: true,
        };

        verifier.add_trusted_key(key);

        let data = b"test attestation data";
        // Create a real signature but for different data
        let wrong_data = b"different data";
        let signature_for_wrong_data = key_pair.private_key().sign(wrong_data);
        let signature_hex = hex::encode(signature_for_wrong_data.to_bytes());

        let result = verifier
            .verify_signature(data, &signature_hex, "test_key")
            .unwrap();
        // This should fail because the signature is for different data
        assert!(!result.is_valid);
        assert_eq!(result.key_id, "test_key");
    }

    // Real cryptographic tests using p256 crate
    #[test]
    fn test_real_p256_signature_verification_valid() {
        use common::crypto::P256KeyPair;

        let mut verifier = P256SignatureVerifier::new();

        // Generate a real P256 key pair
        let key_pair = P256KeyPair::generate();
        let public_key = key_pair.public_key();

        // Create test data and sign it
        let test_data = b"test attestation data for real crypto verification";
        let signature = key_pair.private_key().sign(test_data);

        // Convert to our trusted key format
        let key_id = "real_crypto_test_key";
        let compressed_key_hex = public_key.to_hex();
        let trusted_key = P256PublicKey {
            key_id: key_id.to_string(),
            compressed_key_hex: compressed_key_hex.clone(),
            created_at: SystemTime::now(),
            is_trusted: true,
        };

        verifier.add_trusted_key(trusted_key);

        // Verify the real signature
        let signature_hex = hex::encode(signature.to_bytes());
        let result = verifier
            .verify_signature(test_data, &signature_hex, key_id)
            .unwrap();

        if !result.is_valid {
            println!("Verification failed. Error: {:?}", result.error_message);
            println!("Key ID: {}", result.key_id);
            println!("Signature hex: {signature_hex}");
            println!("Public key hex: {compressed_key_hex}");
        }
        assert!(
            result.is_valid,
            "Real P256 signature should verify successfully. Error: {:?}",
            result.error_message
        );
        assert_eq!(result.key_id, key_id);
        assert!(result.error_message.is_none());
    }

    #[test]
    fn test_real_p256_signature_verification_invalid() {
        use common::crypto::P256KeyPair;

        let mut verifier = P256SignatureVerifier::new();

        // Generate a real P256 key pair
        let key_pair = P256KeyPair::generate();
        let public_key = key_pair.public_key();

        // Convert to our trusted key format
        let key_id = "real_crypto_invalid_test_key";
        let compressed_key_hex = public_key.to_hex();
        let trusted_key = P256PublicKey {
            key_id: key_id.to_string(),
            compressed_key_hex,
            created_at: SystemTime::now(),
            is_trusted: true,
        };

        verifier.add_trusted_key(trusted_key);

        // Create a different key pair to generate a signature with the wrong key
        let wrong_key_pair = P256KeyPair::generate();
        let test_data = b"test attestation data for invalid signature test";
        let signature_with_wrong_key = wrong_key_pair.private_key().sign(test_data);
        let fake_signature = hex::encode(signature_with_wrong_key.to_bytes());

        let result = verifier
            .verify_signature(test_data, &fake_signature, key_id)
            .unwrap();

        assert!(
            !result.is_valid,
            "Signature with wrong key should fail verification"
        );
        assert_eq!(result.key_id, key_id);
        // Error message might be present when verification fails
        if let Some(error) = &result.error_message {
            assert_eq!(error, "Signature verification failed");
        }
    }

    #[test]
    fn test_signature_verification_malformed_signature() {
        let mut verifier = P256SignatureVerifier::new();

        // Add a dummy trusted key
        let trusted_key = P256PublicKey {
            key_id: "malformed_test_key".to_string(),
            compressed_key_hex: "02".to_string() + &hex::encode([0u8; 32]),
            created_at: SystemTime::now(),
            is_trusted: true,
        };
        verifier.add_trusted_key(trusted_key);

        let test_data = b"test data";

        // Test with invalid hex characters (but correct length)
        let invalid_signature = "g".repeat(128); // 128 chars but invalid hex
        let result = verifier.verify_signature(test_data, &invalid_signature, "malformed_test_key");
        assert!(result.is_ok());
        let verification_result = result.unwrap();
        assert!(!verification_result.is_valid);
        assert!(verification_result
            .error_message
            .as_ref()
            .unwrap()
            .contains("Invalid signature format"));

        // Test with wrong length signature
        let short_signature = "ab".repeat(30); // Only 30 bytes instead of 64
        let result = verifier.verify_signature(test_data, &short_signature, "malformed_test_key");
        assert!(result.is_ok());
        let verification_result = result.unwrap();
        assert!(!verification_result.is_valid);
        assert!(verification_result
            .error_message
            .as_ref()
            .unwrap()
            .contains("Invalid signature length"));
    }

    #[test]
    fn test_signature_verification_malformed_public_key() {
        let verifier = P256SignatureVerifier::new();

        // Test the internal verification method directly with malformed public key
        let test_data = b"test data";
        let valid_signature = "a".repeat(128);

        // Test with invalid hex in public key
        let result =
            verifier.perform_real_p256_verification(test_data, &valid_signature, "invalid_hex_gg");
        assert!(result.is_err());
        assert!(result
            .unwrap_err()
            .to_string()
            .contains("Invalid public key hex encoding"));

        // Test with wrong length public key
        let short_pubkey = "02".to_string() + &"ab".repeat(15); // Only 16 bytes instead of 33
        let result =
            verifier.perform_real_p256_verification(test_data, &valid_signature, &short_pubkey);
        assert!(result.is_err());
        assert!(result
            .unwrap_err()
            .to_string()
            .contains("Invalid public key length"));
    }

    #[test]
    fn test_attestation_signature_verification_with_real_crypto() {
        use common::crypto::P256KeyPair;

        let mut verifier = P256SignatureVerifier::new();

        // Generate a real P256 key pair
        let key_pair = P256KeyPair::generate();
        let public_key = key_pair.public_key();

        // Create test attestation data
        let attestation_json =
            r#"{"executor_id":"test-executor","timestamp":"2024-06-17T14:00:00Z","gpu_info":[]}"#;

        // Sign the attestation
        let signature = key_pair.private_key().sign(attestation_json.as_bytes());

        // Get the actual public key PEM from our generated key
        let public_key_pem = public_key.to_pem().unwrap();

        // Verify the attestation signature
        let signature_hex = hex::encode(signature.to_bytes());
        let result = verifier
            .verify_attestation_signature(attestation_json, &signature_hex, &public_key_pem)
            .unwrap();

        // Now that we have proper PEM parsing, this should succeed
        assert!(
            result.is_valid,
            "Signature should verify successfully with correct key"
        );
        assert!(result.key_id.starts_with("pem_"));
    }

    #[test]
    fn test_verify_signature_any_key_with_real_crypto() {
        use common::crypto::P256KeyPair;

        let mut verifier = P256SignatureVerifier::new();

        // Generate multiple real P256 key pairs
        let key_pair1 = P256KeyPair::generate();
        let public_key1 = key_pair1.public_key();
        let key_pair2 = P256KeyPair::generate();
        let public_key2 = key_pair2.public_key();

        // Add both as trusted keys
        let trusted_key1 = P256PublicKey {
            key_id: "key1".to_string(),
            compressed_key_hex: public_key1.to_hex(),
            created_at: SystemTime::now(),
            is_trusted: true,
        };
        let trusted_key2 = P256PublicKey {
            key_id: "key2".to_string(),
            compressed_key_hex: public_key2.to_hex(),
            created_at: SystemTime::now(),
            is_trusted: true,
        };

        verifier.add_trusted_key(trusted_key1);
        verifier.add_trusted_key(trusted_key2);

        // Sign data with key2
        let test_data = b"test data for any key verification";
        let signature = key_pair2.private_key().sign(test_data);
        let signature_hex = hex::encode(signature.to_bytes());

        // Verify using any key (should find key2)
        let result = verifier
            .verify_signature_any_key(test_data, &signature_hex)
            .unwrap();

        assert!(result.is_valid, "Should verify with the correct key");
        assert_eq!(result.key_id, "key2");
    }
}
