//! Ed25519 cryptographic operations
//!
//! This module provides Ed25519 key generation, signing, and verification functionality
//! using the ed25519-dalek library with proper key formatting for SSH and PEM formats.

use crate::error::CryptoError;
use base64::{engine::general_purpose::STANDARD as BASE64, Engine};
use ed25519_dalek::{
    Signature, Signer, SigningKey, Verifier, VerifyingKey, PUBLIC_KEY_LENGTH, SECRET_KEY_LENGTH,
};
use rand::rngs::OsRng;
use std::fmt;
use zeroize::ZeroizeOnDrop;

/// Ed25519 private key wrapper with secure memory handling
#[derive(ZeroizeOnDrop)]
pub struct Ed25519PrivateKey {
    #[zeroize(skip)]
    signing_key: SigningKey,
}

/// Ed25519 public key wrapper
#[derive(Clone, Debug, PartialEq, Eq)]
pub struct Ed25519PublicKey {
    verifying_key: VerifyingKey,
}

/// Ed25519 key pair containing both private and public keys
#[derive(ZeroizeOnDrop)]
pub struct Ed25519KeyPair {
    private_key: Ed25519PrivateKey,
    #[zeroize(skip)]
    public_key: Ed25519PublicKey,
}

impl Ed25519PrivateKey {
    /// Generate a new random Ed25519 private key
    pub fn generate() -> Self {
        let signing_key = SigningKey::generate(&mut OsRng);
        Self { signing_key }
    }

    /// Create from raw bytes (32 bytes)
    pub fn from_bytes(bytes: &[u8; SECRET_KEY_LENGTH]) -> Result<Self, CryptoError> {
        let signing_key = SigningKey::from_bytes(bytes);
        Ok(Self { signing_key })
    }

    /// Create from PKCS#8 PEM string
    pub fn from_pem(pem_str: &str) -> Result<Self, CryptoError> {
        // Parse PEM and extract the key bytes
        let pem_lines: Vec<&str> = pem_str.lines().collect();
        if pem_lines.len() < 3 {
            return Err(CryptoError::KeyGenerationFailed {
                details: "Invalid PEM format".to_string(),
            });
        }

        // Extract base64 content between header and footer
        let base64_content = pem_lines[1..pem_lines.len() - 1].join("");
        let der_bytes =
            BASE64
                .decode(base64_content)
                .map_err(|e| CryptoError::KeyGenerationFailed {
                    details: format!("Failed to decode PEM base64: {e}"),
                })?;

        // Parse PKCS#8 structure to extract the raw key
        // For Ed25519, the key is typically at a specific offset in the PKCS#8 structure
        if der_bytes.len() < 48 {
            return Err(CryptoError::KeyGenerationFailed {
                details: "Invalid PKCS#8 structure".to_string(),
            });
        }

        // Extract the 32-byte Ed25519 private key from PKCS#8
        // The actual key typically starts at offset 16 in the PKCS#8 structure
        let key_start = der_bytes.len() - 32;
        let key_bytes: [u8; 32] =
            der_bytes[key_start..]
                .try_into()
                .map_err(|_| CryptoError::KeyGenerationFailed {
                    details: "Failed to extract Ed25519 key from PKCS#8".to_string(),
                })?;

        Self::from_bytes(&key_bytes)
    }

    /// Export as PKCS#8 PEM string
    pub fn to_pem(&self) -> String {
        // Create PKCS#8 structure for Ed25519
        // This is a simplified version - in production, use proper ASN.1 encoding
        let mut pkcs8_bytes = Vec::new();

        // PKCS#8 header for Ed25519
        pkcs8_bytes.extend_from_slice(&[
            0x30, 0x2e, // SEQUENCE (46 bytes)
            0x02, 0x01, 0x00, // INTEGER version (0)
            0x30, 0x05, // SEQUENCE (5 bytes) - AlgorithmIdentifier
            0x06, 0x03, 0x2b, 0x65, 0x70, // OID for Ed25519
            0x04, 0x22, // OCTET STRING (34 bytes)
            0x04, 0x20, // OCTET STRING (32 bytes) - the actual key
        ]);

        // Add the actual private key bytes
        pkcs8_bytes.extend_from_slice(self.signing_key.as_bytes());

        // Encode to base64 and format as PEM
        let base64_key = BASE64.encode(&pkcs8_bytes);
        format!("-----BEGIN PRIVATE KEY-----\n{base64_key}\n-----END PRIVATE KEY-----")
    }

    /// Get the corresponding public key
    pub fn public_key(&self) -> Ed25519PublicKey {
        Ed25519PublicKey {
            verifying_key: self.signing_key.verifying_key(),
        }
    }

    /// Sign a message
    pub fn sign(&self, message: &[u8]) -> Vec<u8> {
        let signature = self.signing_key.sign(message);
        signature.to_bytes().to_vec()
    }

    /// Get raw bytes (32 bytes)
    pub fn to_bytes(&self) -> [u8; SECRET_KEY_LENGTH] {
        self.signing_key.to_bytes()
    }
}

impl Ed25519PublicKey {
    /// Create from raw bytes (32 bytes)
    pub fn from_bytes(bytes: &[u8; PUBLIC_KEY_LENGTH]) -> Result<Self, CryptoError> {
        let verifying_key =
            VerifyingKey::from_bytes(bytes).map_err(|e| CryptoError::InvalidPublicKey {
                details: format!("Invalid Ed25519 public key: {e}"),
            })?;
        Ok(Self { verifying_key })
    }

    /// Export as OpenSSH format
    pub fn to_openssh(&self) -> Result<String, CryptoError> {
        // OpenSSH format for Ed25519:
        // "ssh-ed25519 <base64-encoded-key> <comment>"

        // The key data includes a header and the actual public key
        let mut key_data = Vec::new();

        // SSH wire format: string "ssh-ed25519"
        let key_type = b"ssh-ed25519";
        key_data.extend_from_slice(&(key_type.len() as u32).to_be_bytes());
        key_data.extend_from_slice(key_type);

        // Followed by the public key bytes
        let public_key_bytes = self.verifying_key.to_bytes();
        key_data.extend_from_slice(&(public_key_bytes.len() as u32).to_be_bytes());
        key_data.extend_from_slice(&public_key_bytes);

        // Base64 encode the key data
        let base64_key = BASE64.encode(&key_data);

        // Format as OpenSSH with comment
        Ok(format!("ssh-ed25519 {base64_key} basilica-ephemeral-key"))
    }

    /// Export as PEM format
    pub fn to_pem(&self) -> String {
        // Create SubjectPublicKeyInfo structure for Ed25519
        let mut spki_bytes = Vec::new();

        // SubjectPublicKeyInfo header
        spki_bytes.extend_from_slice(&[
            0x30, 0x2a, // SEQUENCE (42 bytes)
            0x30, 0x05, // SEQUENCE (5 bytes) - AlgorithmIdentifier
            0x06, 0x03, 0x2b, 0x65, 0x70, // OID for Ed25519
            0x03, 0x21, // BIT STRING (33 bytes)
            0x00, // unused bits
        ]);

        // Add the public key bytes
        spki_bytes.extend_from_slice(self.verifying_key.as_bytes());

        // Encode to base64 and format as PEM
        let base64_key = BASE64.encode(&spki_bytes);
        format!("-----BEGIN PUBLIC KEY-----\n{base64_key}\n-----END PUBLIC KEY-----")
    }

    /// Verify a signature
    pub fn verify(&self, message: &[u8], signature: &[u8]) -> Result<(), CryptoError> {
        let sig = Signature::from_slice(signature).map_err(|e| CryptoError::InvalidSignature {
            details: format!("Invalid Ed25519 signature format: {e}"),
        })?;

        self.verifying_key
            .verify(message, &sig)
            .map_err(|e| CryptoError::InvalidSignature {
                details: format!("Ed25519 signature verification failed: {e}"),
            })
    }

    /// Get raw bytes (32 bytes)
    pub fn to_bytes(&self) -> [u8; PUBLIC_KEY_LENGTH] {
        self.verifying_key.to_bytes()
    }
}

impl Ed25519KeyPair {
    /// Generate a new random Ed25519 key pair
    pub fn generate() -> Self {
        let private_key = Ed25519PrivateKey::generate();
        let public_key = private_key.public_key();
        Self {
            private_key,
            public_key,
        }
    }

    /// Create from private key
    pub fn from_private_key(private_key: Ed25519PrivateKey) -> Self {
        let public_key = private_key.public_key();
        Self {
            private_key,
            public_key,
        }
    }

    /// Get the private key
    pub fn private_key(&self) -> &Ed25519PrivateKey {
        &self.private_key
    }

    /// Get the public key
    pub fn public_key(&self) -> &Ed25519PublicKey {
        &self.public_key
    }

    /// Export as SSH key pair format (private PEM, public OpenSSH)
    pub fn to_ssh_format(&self) -> Result<(String, String), CryptoError> {
        let private_pem = self.private_key.to_pem();
        let public_openssh = self.public_key.to_openssh()?;
        Ok((private_pem, public_openssh))
    }
}

impl fmt::Display for Ed25519PublicKey {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        write!(f, "{}", hex::encode(self.verifying_key.as_bytes()))
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_ed25519_key_generation() {
        let keypair = Ed25519KeyPair::generate();

        // Test private key export
        let private_pem = keypair.private_key().to_pem();
        assert!(private_pem.contains("-----BEGIN PRIVATE KEY-----"));
        assert!(private_pem.contains("-----END PRIVATE KEY-----"));

        // Test public key export
        let public_openssh = keypair.public_key().to_openssh().unwrap();
        assert!(public_openssh.starts_with("ssh-ed25519"));
        assert!(public_openssh.contains("basilica-ephemeral-key"));

        let public_pem = keypair.public_key().to_pem();
        assert!(public_pem.contains("-----BEGIN PUBLIC KEY-----"));
        assert!(public_pem.contains("-----END PUBLIC KEY-----"));
    }

    #[test]
    fn test_ed25519_signing_and_verification() {
        let keypair = Ed25519KeyPair::generate();
        let message = b"Test message for signing";

        // Sign message
        let signature = keypair.private_key().sign(message);
        assert_eq!(signature.len(), 64); // Ed25519 signatures are 64 bytes

        // Verify signature
        let result = keypair.public_key().verify(message, &signature);
        assert!(result.is_ok());

        // Verify with wrong message should fail
        let wrong_message = b"Different message";
        let result = keypair.public_key().verify(wrong_message, &signature);
        assert!(result.is_err());
    }

    #[test]
    fn test_ed25519_key_serialization() {
        let keypair = Ed25519KeyPair::generate();

        // Get raw bytes
        let private_bytes = keypair.private_key().to_bytes();
        let public_bytes = keypair.public_key().to_bytes();

        // Recreate from bytes
        let private_key2 = Ed25519PrivateKey::from_bytes(&private_bytes).unwrap();
        let public_key2 = Ed25519PublicKey::from_bytes(&public_bytes).unwrap();

        // Verify they produce the same signatures
        let message = b"Test serialization";
        let sig1 = keypair.private_key().sign(message);
        let sig2 = private_key2.sign(message);
        assert_eq!(sig1, sig2);

        // Verify public keys are equal
        assert_eq!(keypair.public_key().to_bytes(), public_key2.to_bytes());
    }

    #[test]
    fn test_ed25519_different_keypairs() {
        let keypair1 = Ed25519KeyPair::generate();
        let keypair2 = Ed25519KeyPair::generate();

        // Keys should be different
        assert_ne!(
            keypair1.private_key().to_bytes(),
            keypair2.private_key().to_bytes()
        );
        assert_ne!(
            keypair1.public_key().to_bytes(),
            keypair2.public_key().to_bytes()
        );

        // Cross-signature verification should fail
        let message = b"Test message";
        let signature = keypair1.private_key().sign(message);
        let result = keypair2.public_key().verify(message, &signature);
        assert!(result.is_err());
    }
}
