//! High-level key generation and management functions
//!
//! This module provides convenient functions for generating various types of
//! cryptographic keys used throughout the Basilica system.

use super::{ed25519::Ed25519KeyPair, p256::P256KeyPair};
use crate::error::CryptoError;

/// Generate a new Ed25519 key pair for ephemeral use
///
/// # Returns
/// * `(private_key_pem, public_key_openssh)` - Private key in PEM format, public key in OpenSSH format
///
/// # Example
/// ```rust
/// use common::crypto::keys::generate_ed25519_keypair;
///
/// let (private_pem, public_openssh) = generate_ed25519_keypair();
/// assert!(private_pem.contains("-----BEGIN PRIVATE KEY-----"));
/// assert!(public_openssh.starts_with("ssh-ed25519"));
/// ```
///
/// # Security Notes
/// - Private keys are automatically zeroized when dropped
/// - Use for ephemeral authentication only
/// - Do not store private keys in logs or insecure storage
pub fn generate_ed25519_keypair() -> (String, String) {
    let keypair = Ed25519KeyPair::generate();

    match keypair.to_ssh_format() {
        Ok((private_pem, public_openssh)) => {
            tracing::debug!("Generated new Ed25519 keypair");
            (private_pem, public_openssh)
        }
        Err(e) => {
            tracing::error!("Failed to format Ed25519 keypair: {}", e);
            // Return a properly formatted error message as fallback
            // This should not happen with a properly generated keypair
            (
                "-----BEGIN PRIVATE KEY-----\nERROR: Key generation failed\n-----END PRIVATE KEY-----".to_string(),
                "ssh-ed25519 ERROR basilica-ephemeral-key".to_string()
            )
        }
    }
}

/// Generate a new P256 (secp256r1) key pair
///
/// # Returns
/// * `Ok(P256KeyPair)` - The generated key pair
/// * `Err(CryptoError)` - If key generation fails
///
/// # Example
/// ```rust
/// use common::crypto::keys::generate_p256_keypair;
///
/// let keypair = generate_p256_keypair().unwrap();
/// let private_pem = keypair.private_key().to_pem().unwrap();
/// let public_hex = keypair.public_key().to_hex();
/// ```
///
/// # Security Notes
/// - Private keys are automatically zeroized when dropped
/// - P256 is suitable for ECDSA signatures and ECDH key agreement
/// - Used in validator/attestor cryptographic operations
pub fn generate_p256_keypair() -> Result<P256KeyPair, CryptoError> {
    Ok(P256KeyPair::generate())
}

/// Generate a P256 key pair and return in various formats
///
/// # Returns
/// * `Ok((private_pem, public_pem, public_hex))` - Private key PEM, public key PEM, and compressed public key hex
/// * `Err(CryptoError)` - If key generation or formatting fails
///
/// # Example
/// ```rust
/// use common::crypto::keys::generate_p256_keypair_formatted;
///
/// let (private_pem, public_pem, public_hex) = generate_p256_keypair_formatted().unwrap();
/// assert!(private_pem.contains("-----BEGIN PRIVATE KEY-----"));
/// assert!(public_pem.contains("-----BEGIN PUBLIC KEY-----"));
/// assert_eq!(public_hex.len(), 66); // 33 bytes * 2 hex chars
/// ```
pub fn generate_p256_keypair_formatted() -> Result<(String, String, String), CryptoError> {
    let keypair = P256KeyPair::generate();

    let private_pem = keypair.private_key().to_pem()?;
    let public_pem = keypair.public_key().to_pem()?;
    let public_hex = keypair.public_key().to_hex();

    tracing::debug!("Generated new P256 keypair with formatted output");

    Ok((private_pem, public_pem, public_hex))
}

/// Generate an Ed25519 key pair and return in PEM format
///
/// # Returns
/// * `Ok((private_pem, public_pem))` - Both keys in PEM format
/// * `Err(CryptoError)` - If key generation fails
///
/// # Example
/// ```rust
/// use common::crypto::keys::generate_ed25519_keypair_pem;
///
/// let (private_pem, public_pem) = generate_ed25519_keypair_pem().unwrap();
/// assert!(private_pem.contains("-----BEGIN PRIVATE KEY-----"));
/// assert!(public_pem.contains("-----BEGIN PUBLIC KEY-----"));
/// ```
pub fn generate_ed25519_keypair_pem() -> Result<(String, String), CryptoError> {
    let keypair = Ed25519KeyPair::generate();

    let private_pem = keypair.private_key().to_pem();
    let public_pem = keypair.public_key().to_pem();

    Ok((private_pem, public_pem))
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_generate_ed25519_keypair() {
        let (private_pem, public_openssh) = generate_ed25519_keypair();

        // Check format
        assert!(private_pem.contains("-----BEGIN PRIVATE KEY-----"));
        assert!(private_pem.contains("-----END PRIVATE KEY-----"));
        assert!(public_openssh.starts_with("ssh-ed25519"));
        assert!(public_openssh.contains("basilica-ephemeral-key"));

        // Generate another pair - should be different
        let (private_pem2, public_openssh2) = generate_ed25519_keypair();
        assert_ne!(private_pem, private_pem2);
        assert_ne!(public_openssh, public_openssh2);
    }

    #[test]
    fn test_generate_p256_keypair() {
        let keypair = generate_p256_keypair().unwrap();

        // Test that we can use the keypair
        let message = b"test message";
        let signature = keypair.private_key().sign(message);
        let verify_result = keypair.public_key().verify(message, &signature);
        assert!(verify_result.is_ok());
    }

    #[test]
    fn test_generate_p256_keypair_formatted() {
        let (private_pem, public_pem, public_hex) = generate_p256_keypair_formatted().unwrap();

        // Check formats
        assert!(private_pem.contains("-----BEGIN PRIVATE KEY-----"));
        assert!(private_pem.contains("-----END PRIVATE KEY-----"));
        assert!(public_pem.contains("-----BEGIN PUBLIC KEY-----"));
        assert!(public_pem.contains("-----END PUBLIC KEY-----"));
        assert_eq!(public_hex.len(), 66); // 33 bytes * 2 hex chars

        // Verify hex format
        assert!(public_hex.chars().all(|c| c.is_ascii_hexdigit()));

        // Generate another pair - should be different
        let (private_pem2, public_pem2, public_hex2) = generate_p256_keypair_formatted().unwrap();
        assert_ne!(private_pem, private_pem2);
        assert_ne!(public_pem, public_pem2);
        assert_ne!(public_hex, public_hex2);
    }

    #[test]
    fn test_generate_ed25519_keypair_pem() {
        let (private_pem, public_pem) = generate_ed25519_keypair_pem().unwrap();

        // Check format
        assert!(private_pem.contains("-----BEGIN PRIVATE KEY-----"));
        assert!(private_pem.contains("-----END PRIVATE KEY-----"));
        assert!(public_pem.contains("-----BEGIN PUBLIC KEY-----"));
        assert!(public_pem.contains("-----END PUBLIC KEY-----"));

        // Should be able to parse back
        use super::super::ed25519::Ed25519PrivateKey;
        let parsed_private = Ed25519PrivateKey::from_pem(&private_pem);
        assert!(parsed_private.is_ok());
    }
}
