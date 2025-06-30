//! P256 (secp256r1) ECDSA cryptographic operations
//!
//! This module provides P256 key generation, signing, and verification functionality
//! using the p256 crate with proper key formatting for PEM and DER formats.

use crate::error::CryptoError;
use p256::{
    ecdsa::{signature::Signer, signature::Verifier, Signature, SigningKey, VerifyingKey},
    elliptic_curve::sec1::ToEncodedPoint,
    pkcs8::{DecodePrivateKey, DecodePublicKey, EncodePrivateKey, EncodePublicKey, LineEnding},
    PublicKey,
};
use rand::rngs::OsRng;
use std::fmt;
use zeroize::ZeroizeOnDrop;

/// P256 signature wrapper
#[derive(Clone, Debug, PartialEq, Eq)]
pub struct P256Signature {
    signature: Signature,
}

/// P256 private key wrapper with secure memory handling
#[derive(ZeroizeOnDrop)]
pub struct P256PrivateKey {
    #[zeroize(skip)]
    signing_key: SigningKey,
}

/// P256 public key wrapper
#[derive(Clone, Debug, PartialEq, Eq)]
pub struct P256PublicKey {
    verifying_key: VerifyingKey,
}

/// P256 key pair containing both private and public keys
#[derive(ZeroizeOnDrop)]
pub struct P256KeyPair {
    private_key: P256PrivateKey,
    #[zeroize(skip)]
    public_key: P256PublicKey,
}

impl P256Signature {
    /// Create from raw bytes
    pub fn from_bytes(bytes: &[u8]) -> Result<Self, CryptoError> {
        let signature =
            Signature::from_slice(bytes).map_err(|e| CryptoError::InvalidSignature {
                details: format!("Invalid P256 signature format: {e}"),
            })?;
        Ok(Self { signature })
    }

    /// Get raw bytes
    pub fn to_bytes(&self) -> Vec<u8> {
        self.signature.to_bytes().to_vec()
    }

    /// Get DER encoded signature
    pub fn to_der(&self) -> Vec<u8> {
        self.signature.to_der().to_bytes().to_vec()
    }
}

impl P256PrivateKey {
    /// Generate a new random P256 private key
    pub fn generate() -> Self {
        let signing_key = SigningKey::random(&mut OsRng);
        Self { signing_key }
    }

    /// Create from PKCS#8 PEM string
    pub fn from_pem(pem_str: &str) -> Result<Self, CryptoError> {
        let signing_key =
            SigningKey::from_pkcs8_pem(pem_str).map_err(|e| CryptoError::KeyGenerationFailed {
                details: format!("Failed to parse P256 private key from PEM: {e}"),
            })?;
        Ok(Self { signing_key })
    }

    /// Create from PKCS#8 DER bytes
    pub fn from_der(der_bytes: &[u8]) -> Result<Self, CryptoError> {
        let signing_key = SigningKey::from_pkcs8_der(der_bytes).map_err(|e| {
            CryptoError::KeyGenerationFailed {
                details: format!("Failed to parse P256 private key from DER: {e}"),
            }
        })?;
        Ok(Self { signing_key })
    }

    /// Export as PKCS#8 PEM string
    pub fn to_pem(&self) -> Result<String, CryptoError> {
        self.signing_key
            .to_pkcs8_pem(LineEnding::LF)
            .map_err(|e| CryptoError::KeyGenerationFailed {
                details: format!("Failed to encode P256 private key as PEM: {e}"),
            })
            .map(|pem| pem.to_string())
    }

    /// Export as PKCS#8 DER bytes
    pub fn to_der(&self) -> Result<Vec<u8>, CryptoError> {
        self.signing_key
            .to_pkcs8_der()
            .map_err(|e| CryptoError::KeyGenerationFailed {
                details: format!("Failed to encode P256 private key as DER: {e}"),
            })
            .map(|der| der.to_bytes().to_vec())
    }

    /// Get the corresponding public key
    pub fn public_key(&self) -> P256PublicKey {
        P256PublicKey {
            verifying_key: VerifyingKey::from(&self.signing_key),
        }
    }

    /// Sign a message
    pub fn sign(&self, message: &[u8]) -> P256Signature {
        let signature: Signature = self.signing_key.sign(message);
        P256Signature { signature }
    }
}

impl P256PublicKey {
    /// Create from PEM string
    pub fn from_pem(pem_str: &str) -> Result<Self, CryptoError> {
        let public_key =
            PublicKey::from_public_key_pem(pem_str).map_err(|e| CryptoError::InvalidPublicKey {
                details: format!("Failed to parse P256 public key from PEM: {e}"),
            })?;
        let verifying_key = VerifyingKey::from(public_key);
        Ok(Self { verifying_key })
    }

    /// Create from DER bytes
    pub fn from_der(der_bytes: &[u8]) -> Result<Self, CryptoError> {
        let public_key = PublicKey::from_public_key_der(der_bytes).map_err(|e| {
            CryptoError::InvalidPublicKey {
                details: format!("Failed to parse P256 public key from DER: {e}"),
            }
        })?;
        let verifying_key = VerifyingKey::from(public_key);
        Ok(Self { verifying_key })
    }

    /// Create from compressed point bytes (33 bytes)
    pub fn from_compressed_bytes(bytes: &[u8]) -> Result<Self, CryptoError> {
        if bytes.len() != 33 {
            return Err(CryptoError::InvalidPublicKey {
                details: format!(
                    "Expected 33 bytes for compressed P256 public key, got {}",
                    bytes.len()
                ),
            });
        }

        // Parse the compressed point
        let public_key =
            PublicKey::from_sec1_bytes(bytes).map_err(|e| CryptoError::InvalidPublicKey {
                details: format!("Failed to parse compressed P256 public key: {e}"),
            })?;

        let verifying_key = VerifyingKey::from(public_key);
        Ok(Self { verifying_key })
    }

    /// Export as PEM string
    pub fn to_pem(&self) -> Result<String, CryptoError> {
        self.verifying_key
            .to_public_key_pem(LineEnding::LF)
            .map_err(|e| CryptoError::KeyGenerationFailed {
                details: format!("Failed to encode P256 public key as PEM: {e}"),
            })
    }

    /// Export as DER bytes
    pub fn to_der(&self) -> Result<Vec<u8>, CryptoError> {
        PublicKey::from(&self.verifying_key)
            .to_public_key_der()
            .map_err(|e| CryptoError::KeyGenerationFailed {
                details: format!("Failed to encode P256 public key as DER: {e}"),
            })
            .map(|der| der.to_vec())
    }

    /// Export as compressed SEC1 bytes (33 bytes)
    pub fn to_compressed_bytes(&self) -> Vec<u8> {
        let point = PublicKey::from(&self.verifying_key).to_encoded_point(true);
        point.as_bytes().to_vec()
    }

    /// Export as uncompressed SEC1 bytes (65 bytes)
    pub fn to_uncompressed_bytes(&self) -> Vec<u8> {
        let point = PublicKey::from(&self.verifying_key).to_encoded_point(false);
        point.as_bytes().to_vec()
    }

    /// Verify a signature
    pub fn verify(&self, message: &[u8], signature: &P256Signature) -> Result<(), CryptoError> {
        self.verifying_key
            .verify(message, &signature.signature)
            .map_err(|e| CryptoError::InvalidSignature {
                details: format!("P256 signature verification failed: {e}"),
            })
    }

    /// Get hex representation of compressed public key
    pub fn to_hex(&self) -> String {
        hex::encode(self.to_compressed_bytes())
    }
}

impl P256KeyPair {
    /// Generate a new random P256 key pair
    pub fn generate() -> Self {
        let private_key = P256PrivateKey::generate();
        let public_key = private_key.public_key();
        Self {
            private_key,
            public_key,
        }
    }

    /// Create from private key
    pub fn from_private_key(private_key: P256PrivateKey) -> Self {
        let public_key = private_key.public_key();
        Self {
            private_key,
            public_key,
        }
    }

    /// Get the private key
    pub fn private_key(&self) -> &P256PrivateKey {
        &self.private_key
    }

    /// Get the public key
    pub fn public_key(&self) -> &P256PublicKey {
        &self.public_key
    }
}

impl fmt::Display for P256PublicKey {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        write!(f, "{}", self.to_hex())
    }
}

/// Verify a P256 signature given a public key, message, and signature
pub fn verify_p256_signature(
    public_key: &P256PublicKey,
    message: &[u8],
    signature: &[u8],
) -> Result<(), CryptoError> {
    let sig = P256Signature::from_bytes(signature)?;
    public_key.verify(message, &sig)
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_p256_key_generation() {
        let keypair = P256KeyPair::generate();

        // Test private key export
        let private_pem = keypair.private_key().to_pem().unwrap();
        assert!(private_pem.contains("-----BEGIN PRIVATE KEY-----"));
        assert!(private_pem.contains("-----END PRIVATE KEY-----"));

        let private_der = keypair.private_key().to_der().unwrap();
        assert!(!private_der.is_empty());

        // Test public key export
        let public_pem = keypair.public_key().to_pem().unwrap();
        assert!(public_pem.contains("-----BEGIN PUBLIC KEY-----"));
        assert!(public_pem.contains("-----END PUBLIC KEY-----"));

        let public_der = keypair.public_key().to_der().unwrap();
        assert!(!public_der.is_empty());

        // Test compressed format
        let compressed = keypair.public_key().to_compressed_bytes();
        assert_eq!(compressed.len(), 33);
        assert!(compressed[0] == 0x02 || compressed[0] == 0x03); // Compressed point prefix

        // Test hex format
        let hex = keypair.public_key().to_hex();
        assert_eq!(hex.len(), 66); // 33 bytes * 2 hex chars
    }

    #[test]
    fn test_p256_signing_and_verification() {
        let keypair = P256KeyPair::generate();
        let message = b"Test message for P256 signing";

        // Sign message
        let signature = keypair.private_key().sign(message);
        let sig_bytes = signature.to_bytes();
        assert_eq!(sig_bytes.len(), 64); // P256 signatures are 64 bytes

        // Verify signature
        let result = keypair.public_key().verify(message, &signature);
        assert!(result.is_ok());

        // Verify with wrong message should fail
        let wrong_message = b"Different message";
        let result = keypair.public_key().verify(wrong_message, &signature);
        assert!(result.is_err());
    }

    #[test]
    fn test_p256_key_serialization() {
        let keypair = P256KeyPair::generate();

        // Test PEM round-trip
        let private_pem = keypair.private_key().to_pem().unwrap();
        let private_key2 = P256PrivateKey::from_pem(&private_pem).unwrap();

        let public_pem = keypair.public_key().to_pem().unwrap();
        let public_key2 = P256PublicKey::from_pem(&public_pem).unwrap();

        // Test DER round-trip
        let private_der = keypair.private_key().to_der().unwrap();
        let private_key3 = P256PrivateKey::from_der(&private_der).unwrap();

        let public_der = keypair.public_key().to_der().unwrap();
        let public_key3 = P256PublicKey::from_der(&public_der).unwrap();

        // Test compressed bytes round-trip
        let compressed = keypair.public_key().to_compressed_bytes();
        let public_key4 = P256PublicKey::from_compressed_bytes(&compressed).unwrap();

        // Verify all produce the same signatures
        let message = b"Test serialization";
        let sig1 = keypair.private_key().sign(message);
        let sig2 = private_key2.sign(message);
        let sig3 = private_key3.sign(message);

        assert_eq!(sig1.to_bytes(), sig2.to_bytes());
        assert_eq!(sig1.to_bytes(), sig3.to_bytes());

        // Verify all public keys can verify the signature
        assert!(public_key2.verify(message, &sig1).is_ok());
        assert!(public_key3.verify(message, &sig1).is_ok());
        assert!(public_key4.verify(message, &sig1).is_ok());
    }

    #[test]
    fn test_p256_different_keypairs() {
        let keypair1 = P256KeyPair::generate();
        let keypair2 = P256KeyPair::generate();

        // Keys should be different
        assert_ne!(
            keypair1.public_key().to_compressed_bytes(),
            keypair2.public_key().to_compressed_bytes()
        );

        // Cross-signature verification should fail
        let message = b"Test message";
        let signature = keypair1.private_key().sign(message);
        let result = keypair2.public_key().verify(message, &signature);
        assert!(result.is_err());
    }

    #[test]
    fn test_verify_p256_signature_helper() {
        let keypair = P256KeyPair::generate();
        let message = b"Test helper function";

        let signature = keypair.private_key().sign(message);
        let sig_bytes = signature.to_bytes();

        // Should verify successfully
        let result = verify_p256_signature(keypair.public_key(), message, &sig_bytes);
        assert!(result.is_ok());

        // Should fail with tampered signature
        let mut bad_sig = sig_bytes.clone();
        bad_sig[0] ^= 1;
        let result = verify_p256_signature(keypair.public_key(), message, &bad_sig);
        assert!(result.is_err());
    }
}
