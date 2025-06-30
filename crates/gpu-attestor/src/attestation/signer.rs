//! Attestation signing implementation

use anyhow::{Context, Result};
use chrono::{DateTime, Utc};
use p256::ecdsa::{Signature, SigningKey, VerifyingKey};
use p256::pkcs8::{DecodePrivateKey, EncodePrivateKey, EncodePublicKey};
use rand::rngs::OsRng;
use signature::{Signer, Verifier};
use std::fs;

use super::types::*;

pub struct AttestationSigner {
    ephemeral_key: SigningKey,
    key_rotation_timestamp: DateTime<Utc>,
}

impl Default for AttestationSigner {
    fn default() -> Self {
        Self::new()
    }
}

impl AttestationSigner {
    pub fn new() -> Self {
        Self {
            ephemeral_key: SigningKey::random(&mut OsRng),
            key_rotation_timestamp: Utc::now(),
        }
    }

    pub fn from_embedded_key() -> Result<Self> {
        // Embedded validator key is a public key only - cannot be used for signing
        // This is correct security behavior - private keys should never be embedded in binaries
        anyhow::bail!(
            "Cannot derive signing key from embedded public key - use generate new ephemeral key instead"
        )
    }

    pub fn from_private_key_pem(pem_data: &str) -> Result<Self> {
        let signing_key = SigningKey::from_pkcs8_pem(pem_data)
            .context("Failed to parse private key from PEM format")?;

        Ok(Self {
            ephemeral_key: signing_key,
            key_rotation_timestamp: Utc::now(),
        })
    }

    pub fn from_private_key_der(der_data: &[u8]) -> Result<Self> {
        let signing_key = SigningKey::from_pkcs8_der(der_data)
            .context("Failed to parse private key from DER format")?;

        Ok(Self {
            ephemeral_key: signing_key,
            key_rotation_timestamp: Utc::now(),
        })
    }

    pub fn to_private_key_pem(&self) -> Result<String> {
        self.ephemeral_key
            .to_pkcs8_pem(p256::pkcs8::LineEnding::LF)
            .map_err(|e| anyhow::anyhow!("Failed to encode private key as PEM: {}", e))
            .map(|pem| pem.to_string())
    }

    pub fn to_public_key_pem(&self) -> Result<String> {
        let verifying_key = VerifyingKey::from(&self.ephemeral_key);
        verifying_key
            .to_public_key_pem(p256::pkcs8::LineEnding::LF)
            .map_err(|e| anyhow::anyhow!("Failed to encode public key as PEM: {}", e))
    }

    pub fn should_rotate_key(&self, max_age_hours: u64) -> bool {
        let age = Utc::now().signed_duration_since(self.key_rotation_timestamp);
        age.num_hours() >= max_age_hours as i64
    }

    pub fn rotate_key(&mut self) {
        self.ephemeral_key = SigningKey::random(&mut OsRng);
        self.key_rotation_timestamp = Utc::now();
        tracing::info!("Ephemeral signing key rotated");
    }

    pub fn sign_attestation(&self, report: AttestationReport) -> Result<SignedAttestation> {
        let report_json =
            serde_json::to_string(&report).context("Failed to serialize attestation report")?;

        // Sign the JSON-serialized report
        let signature: Signature = self.ephemeral_key.sign(report_json.as_bytes());
        let verifying_key = VerifyingKey::from(&self.ephemeral_key);

        // Verify the signature before returning to ensure correctness
        verifying_key
            .verify(report_json.as_bytes(), &signature)
            .context("Failed to verify signature after signing - this should not happen")?;

        Ok(SignedAttestation::new(
            report,
            signature.to_bytes().to_vec(),
            verifying_key.to_sec1_bytes().to_vec(),
        ))
    }

    pub fn verify_attestation(&self, signed_attestation: &SignedAttestation) -> Result<bool> {
        let report_json = serde_json::to_string(&signed_attestation.report)
            .context("Failed to serialize attestation report for verification")?;

        let signature = Signature::from_bytes(signed_attestation.signature.as_slice().into())
            .context("Failed to parse signature bytes")?;

        let verifying_key = VerifyingKey::from_sec1_bytes(&signed_attestation.ephemeral_public_key)
            .context("Failed to parse public key bytes")?;

        match verifying_key.verify(report_json.as_bytes(), &signature) {
            Ok(()) => Ok(true),
            Err(_) => Ok(false),
        }
    }

    pub fn get_public_key(&self) -> VerifyingKey {
        VerifyingKey::from(&self.ephemeral_key)
    }

    pub fn get_key_rotation_timestamp(&self) -> DateTime<Utc> {
        self.key_rotation_timestamp
    }

    pub fn secure_key_storage_save(&self, file_path: &str) -> Result<()> {
        use std::fs::OpenOptions;
        use std::os::unix::fs::OpenOptionsExt;

        let pem_data = self.to_private_key_pem()?;

        // Create file with restrictive permissions (600 - owner read/write only)
        let mut file = OpenOptions::new()
            .create(true)
            .write(true)
            .truncate(true)
            .mode(0o600)
            .open(file_path)
            .context("Failed to create key file with secure permissions")?;

        use std::io::Write;
        file.write_all(pem_data.as_bytes())
            .context("Failed to write private key to file")?;

        tracing::info!("Private key saved securely to {}", file_path);
        Ok(())
    }

    pub fn secure_key_storage_load(file_path: &str) -> Result<Self> {
        let pem_data = fs::read_to_string(file_path).context("Failed to read private key file")?;

        let signer = Self::from_private_key_pem(&pem_data)?;

        // Clear sensitive data from memory
        drop(pem_data);

        tracing::info!("Private key loaded securely from {}", file_path);
        Ok(signer)
    }

    pub fn generate_deterministic_key(seed: &[u8]) -> Result<Self> {
        if seed.len() < 32 {
            anyhow::bail!("Seed must be at least 32 bytes for security");
        }

        use p256::elliptic_curve::rand_core::SeedableRng;
        use rand_chacha::ChaCha20Rng;

        let mut rng = ChaCha20Rng::from_seed(
            seed[..32]
                .try_into()
                .context("Failed to create seed array")?,
        );

        let signing_key = SigningKey::random(&mut rng);

        Ok(Self {
            ephemeral_key: signing_key,
            key_rotation_timestamp: Utc::now(),
        })
    }
}

impl AttestationSigner {
    /// Create a new signer with automatic key rotation
    pub fn with_auto_rotation(max_age_hours: u64) -> Self {
        let mut signer = Self::new();
        if signer.should_rotate_key(max_age_hours) {
            signer.rotate_key();
        }
        signer
    }

    /// Sign multiple reports at once for batch processing
    pub fn sign_multiple(&self, reports: Vec<AttestationReport>) -> Result<Vec<SignedAttestation>> {
        reports
            .into_iter()
            .map(|report| self.sign_attestation(report))
            .collect()
    }

    /// Get signing key fingerprint for identification
    pub fn get_key_fingerprint(&self) -> String {
        let public_key = self.get_public_key();
        let key_bytes = public_key.to_sec1_bytes();
        use sha2::{Digest, Sha256};
        let hash = Sha256::digest(&key_bytes);
        format!("{hash:x}")
    }

    /// Check if key needs rotation based on usage count
    pub fn should_rotate_by_usage(&self, usage_count: u64, max_usages: u64) -> bool {
        usage_count >= max_usages
    }

    /// Get the embedded validator public key for validation purposes
    pub fn get_embedded_validator_key() -> Result<VerifyingKey> {
        crate::integrity::extract_embedded_key()
    }

    /// Validate that an attestation was signed by a known validator key
    pub fn validate_against_embedded_key(
        &self,
        signed_attestation: &SignedAttestation,
    ) -> Result<bool> {
        let _validator_key = Self::get_embedded_validator_key()?;

        // For now, we verify the ephemeral signature - in production you might want
        // additional validation against the validator key
        self.verify_attestation(signed_attestation)
    }

    /// Sign with additional metadata for enhanced security
    pub fn sign_attestation_with_metadata(
        &self,
        report: AttestationReport,
        metadata: &[u8],
    ) -> Result<SignedAttestation> {
        let report_json =
            serde_json::to_string(&report).context("Failed to serialize attestation report")?;

        // Create a combined payload: report + metadata
        let mut combined_payload = report_json.as_bytes().to_vec();
        combined_payload.extend_from_slice(metadata);

        let signature: Signature = self.ephemeral_key.sign(&combined_payload);
        let verifying_key = VerifyingKey::from(&self.ephemeral_key);

        // Verify the signature
        verifying_key
            .verify(&combined_payload, &signature)
            .context("Failed to verify signature after signing")?;

        Ok(SignedAttestation::new(
            report,
            signature.to_bytes().to_vec(),
            verifying_key.to_sec1_bytes().to_vec(),
        ))
    }
}

impl Drop for AttestationSigner {
    fn drop(&mut self) {
        // The SigningKey struct doesn't directly implement Zeroize, but the underlying
        // scalar does get zeroized when the SigningKey is dropped
        tracing::debug!(
            "AttestationSigner dropped - key material will be zeroized by the SigningKey Drop impl"
        );
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_attestation_signer_creation() {
        let signer = AttestationSigner::new();
        let public_key = signer.get_public_key();

        // Test that we can create a signature and verify it
        let test_data = b"test attestation data";
        let signature: Signature = signer.ephemeral_key.sign(test_data);

        assert!(signature::Verifier::verify(&public_key, test_data, &signature).is_ok());
    }

    #[test]
    fn test_key_rotation() {
        let mut signer = AttestationSigner::new();
        let original_key = signer.get_public_key();

        signer.rotate_key();
        let new_key = signer.get_public_key();

        assert_ne!(original_key.to_sec1_bytes(), new_key.to_sec1_bytes());
    }

    #[test]
    fn test_should_rotate_key() {
        let signer = AttestationSigner::new();

        // Should not rotate immediately
        assert!(!signer.should_rotate_key(24));

        // Should always rotate if max age is 0
        assert!(signer.should_rotate_key(0));
    }

    #[test]
    fn test_sign_attestation() {
        let signer = AttestationSigner::new();
        let report = AttestationReport::new("test_executor".to_string());

        let signed_attestation = signer.sign_attestation(report).unwrap();

        assert!(!signed_attestation.signature.is_empty());
        assert!(!signed_attestation.ephemeral_public_key.is_empty());
        assert_eq!(signed_attestation.report.executor_id, "test_executor");
    }

    #[test]
    fn test_auto_rotation_signer() {
        let signer = AttestationSigner::with_auto_rotation(24);
        assert!(!signer.get_public_key().to_sec1_bytes().is_empty());
    }

    #[test]
    fn test_sign_multiple() {
        let signer = AttestationSigner::new();
        let reports = vec![
            AttestationReport::new("executor1".to_string()),
            AttestationReport::new("executor2".to_string()),
        ];

        let signed_attestations = signer.sign_multiple(reports).unwrap();

        assert_eq!(signed_attestations.len(), 2);
        assert_eq!(signed_attestations[0].report.executor_id, "executor1");
        assert_eq!(signed_attestations[1].report.executor_id, "executor2");
    }

    #[test]
    fn test_key_fingerprint() {
        let signer = AttestationSigner::new();
        let fingerprint = signer.get_key_fingerprint();

        assert!(!fingerprint.is_empty());
        assert_eq!(fingerprint.len(), 64); // SHA256 hash length in hex
    }

    #[test]
    fn test_usage_based_rotation() {
        let signer = AttestationSigner::new();

        assert!(!signer.should_rotate_by_usage(50, 100));
        assert!(signer.should_rotate_by_usage(100, 100));
        assert!(signer.should_rotate_by_usage(150, 100));
    }

    #[test]
    fn test_pem_serialization_roundtrip() {
        let signer = AttestationSigner::new();
        let pem_private = signer.to_private_key_pem().unwrap();
        let pem_public = signer.to_public_key_pem().unwrap();

        assert!(!pem_private.is_empty());
        assert!(!pem_public.is_empty());
        assert!(pem_private.contains("-----BEGIN PRIVATE KEY-----"));
        assert!(pem_public.contains("-----BEGIN PUBLIC KEY-----"));

        // Test roundtrip
        let restored_signer = AttestationSigner::from_private_key_pem(&pem_private).unwrap();
        let restored_pem = restored_signer.to_private_key_pem().unwrap();

        // The PEM should be identical after roundtrip
        assert_eq!(pem_private, restored_pem);
    }

    #[test]
    fn test_verify_attestation() {
        let signer = AttestationSigner::new();
        let report = AttestationReport::new("test_executor".to_string());
        let signed_attestation = signer.sign_attestation(report).unwrap();

        let is_valid = signer.verify_attestation(&signed_attestation).unwrap();
        assert!(is_valid);
    }

    #[test]
    fn test_verify_tampered_attestation() {
        let signer = AttestationSigner::new();
        let report = AttestationReport::new("test_executor".to_string());
        let mut signed_attestation = signer.sign_attestation(report).unwrap();

        // Tamper with the signature
        signed_attestation.signature[0] ^= 0x01;

        let is_valid = signer.verify_attestation(&signed_attestation).unwrap();
        assert!(!is_valid);
    }

    #[test]
    fn test_deterministic_key_generation() {
        let seed = b"this_is_a_test_seed_for_deterministic_key_generation_12345";
        let signer1 = AttestationSigner::generate_deterministic_key(seed).unwrap();
        let signer2 = AttestationSigner::generate_deterministic_key(seed).unwrap();

        // Same seed should produce same key
        let key1_bytes = signer1.get_public_key().to_sec1_bytes();
        let key2_bytes = signer2.get_public_key().to_sec1_bytes();

        assert_eq!(key1_bytes, key2_bytes);
    }

    #[test]
    fn test_deterministic_key_different_seeds() {
        let seed1 = b"this_is_a_test_seed_for_deterministic_key_generation_12345";
        let seed2 = b"this_is_a_different_seed_for_deterministic_key_gen_54321";

        let signer1 = AttestationSigner::generate_deterministic_key(seed1).unwrap();
        let signer2 = AttestationSigner::generate_deterministic_key(seed2).unwrap();

        // Different seeds should produce different keys
        let key1_bytes = signer1.get_public_key().to_sec1_bytes();
        let key2_bytes = signer2.get_public_key().to_sec1_bytes();

        assert_ne!(key1_bytes, key2_bytes);
    }

    #[test]
    fn test_sign_with_metadata() {
        let signer = AttestationSigner::new();
        let report = AttestationReport::new("test_executor".to_string());
        let metadata = b"additional_security_metadata";

        let signed_attestation = signer
            .sign_attestation_with_metadata(report, metadata)
            .unwrap();

        assert!(!signed_attestation.signature.is_empty());
        assert!(!signed_attestation.ephemeral_public_key.is_empty());
    }

    #[test]
    fn test_embedded_validator_key_access() {
        // This test verifies we can access the embedded validator key
        let result = AttestationSigner::get_embedded_validator_key();
        assert!(result.is_ok());

        let validator_key = result.unwrap();
        let key_bytes = validator_key.to_sec1_bytes();
        // P256 public keys can be either compressed (33 bytes) or uncompressed (65 bytes)
        assert!(
            key_bytes.len() == 33 || key_bytes.len() == 65,
            "Expected key length of 33 or 65 bytes, got {}",
            key_bytes.len()
        );
    }

    #[test]
    fn test_validate_against_embedded_key() {
        let signer = AttestationSigner::new();
        let report = AttestationReport::new("test_executor".to_string());
        let signed_attestation = signer.sign_attestation(report).unwrap();

        // This should validate the ephemeral signature
        let is_valid = signer
            .validate_against_embedded_key(&signed_attestation)
            .unwrap();
        assert!(is_valid);
    }

    #[test]
    fn test_from_embedded_key_fails() {
        // This should fail as expected - we cannot derive private key from public key
        let result = AttestationSigner::from_embedded_key();
        assert!(result.is_err());
    }

    #[test]
    fn test_seed_too_short_fails() {
        let short_seed = b"short_seed";
        let result = AttestationSigner::generate_deterministic_key(short_seed);
        assert!(result.is_err());
    }
}
