//! AEAD (Authenticated Encryption with Associated Data) utilities
//!
//! This module provides AES-256-GCM encryption with a simple base64 format:
//! "<base64_nonce>:<base64_ciphertext>"

use crate::crypto::{decrypt_aes_gcm, encrypt_aes_gcm, AES_KEY_SIZE};
use anyhow::{anyhow, Result};
use data_encoding::BASE64;
use zeroize::Zeroizing;

/// AEAD wrapper for AES-256-GCM encryption
///
/// This struct provides authenticated encryption with associated data (AEAD) using AES-256-GCM
/// with a simple base64 format: "<base64_nonce>:<base64_ciphertext>".
///
/// # Examples
/// ```rust
/// # fn main() -> Result<(), Box<dyn std::error::Error>> {
/// use basilica_common::crypto::Aead;
///
/// let key_hex = "0123456789abcdef0123456789abcdef0123456789abcdef0123456789abcdef";
/// let aead = Aead::new(key_hex)?;
/// let encrypted = aead.encrypt("secret data")?;
/// let decrypted = aead.decrypt(&encrypted)?;
/// # assert_eq!(decrypted, "secret data");
/// # Ok(())
/// # }
/// ```
pub struct Aead {
    key: Zeroizing<Vec<u8>>,
}

impl Aead {
    /// Create a new Aead instance from a hex-encoded key
    ///
    /// # Arguments
    /// * `key_hex` - 32-byte key encoded as hexadecimal (64 hex characters)
    ///
    /// # Returns
    /// * `Ok(Aead)` - Initialized encryption instance
    /// * `Err` - If key is invalid
    pub fn new(key_hex: &str) -> Result<Self> {
        let key_bytes = hex::decode(key_hex)?;
        if key_bytes.len() != AES_KEY_SIZE {
            return Err(anyhow!(
                "AEAD key must be {} bytes ({} hex characters)",
                AES_KEY_SIZE,
                AES_KEY_SIZE * 2
            ));
        }
        Ok(Self {
            key: Zeroizing::new(key_bytes),
        })
    }

    /// Create Aead instance from raw key bytes
    ///
    /// # Arguments
    /// * `key_bytes` - 32-byte key as raw bytes
    ///
    /// # Returns
    /// * `Ok(Aead)` - Initialized encryption instance
    /// * `Err` - If key size is invalid
    pub fn from_bytes(key_bytes: Vec<u8>) -> Result<Self> {
        if key_bytes.len() != AES_KEY_SIZE {
            return Err(anyhow!("AEAD key must be {} bytes", AES_KEY_SIZE));
        }
        Ok(Self {
            key: Zeroizing::new(key_bytes),
        })
    }

    /// Encrypt plaintext and return a base64 formatted string
    ///
    /// # Arguments
    /// * `plaintext` - Data to encrypt
    ///
    /// # Returns
    /// * Encrypted data in format: "<base64_nonce>:<base64_ciphertext>"
    pub fn encrypt(&self, plaintext: &str) -> Result<String> {
        let (ciphertext, nonce) = encrypt_aes_gcm(plaintext.as_bytes(), &self.key)
            .map_err(|e| anyhow!("Encryption failed: {}", e))?;

        Ok(format!(
            "{}:{}",
            BASE64.encode(&nonce),
            BASE64.encode(&ciphertext)
        ))
    }

    /// Decrypt a base64 formatted ciphertext string
    ///
    /// # Arguments
    /// * `data` - Encrypted string in format: "<base64_nonce>:<base64_ciphertext>"
    ///
    /// # Returns
    /// * Decrypted plaintext as string
    pub fn decrypt(&self, data: &str) -> Result<String> {
        let parts: Vec<&str> = data.split(':').collect();
        if parts.len() != 2 {
            return Err(anyhow!(
                "Invalid ciphertext format, expected 'nonce:ciphertext'"
            ));
        }

        let nonce = BASE64
            .decode(parts[0].as_bytes())
            .map_err(|e| anyhow!("Invalid nonce base64: {}", e))?;
        let ciphertext = BASE64
            .decode(parts[1].as_bytes())
            .map_err(|e| anyhow!("Invalid ciphertext base64: {}", e))?;

        let plaintext = decrypt_aes_gcm(&ciphertext, &self.key, &nonce)
            .map_err(|e| anyhow!("Decryption failed: {}", e))?;

        Ok(String::from_utf8(plaintext)?)
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    const TEST_KEY: &str = "0123456789abcdef0123456789abcdef0123456789abcdef0123456789abcdef";

    #[test]
    fn test_aead_roundtrip() {
        let aead = Aead::new(TEST_KEY).unwrap();
        let plaintext = "test data";

        let encrypted = aead.encrypt(plaintext).unwrap();
        // Should be "nonce:ciphertext" format
        assert_eq!(encrypted.matches(':').count(), 1);

        let decrypted = aead.decrypt(&encrypted).unwrap();
        assert_eq!(decrypted, plaintext);
    }

    #[test]
    fn test_aead_format_structure() {
        let aead = Aead::new(TEST_KEY).unwrap();
        let encrypted = aead.encrypt("test").unwrap();

        // Check format structure - should have exactly 2 parts separated by ':'
        let parts: Vec<&str> = encrypted.split(':').collect();
        assert_eq!(parts.len(), 2);

        // Both parts should be valid base64
        assert!(BASE64.decode(parts[0].as_bytes()).is_ok());
        assert!(BASE64.decode(parts[1].as_bytes()).is_ok());
    }

    #[test]
    fn test_invalid_key_length() {
        // Too short key
        let result = Aead::new("0123456789abcdef");
        assert!(result.is_err());

        // Invalid hex
        let result = Aead::new("not_hex_at_all");
        assert!(result.is_err());
    }

    #[test]
    fn test_invalid_format_deserialization() {
        let aead = Aead::new(TEST_KEY).unwrap();

        // Invalid format - too many parts
        let result = aead.decrypt("part1:part2:part3");
        assert!(result.is_err());

        // Invalid format - only one part
        let result = aead.decrypt("singlepart");
        assert!(result.is_err());

        // Invalid base64
        let result = aead.decrypt("invalid_base64:also_invalid");
        assert!(result.is_err());
    }

    #[test]
    fn test_from_bytes_constructor() {
        let key_bytes = hex::decode(TEST_KEY).unwrap();
        let aead = Aead::from_bytes(key_bytes).unwrap();

        let plaintext = "test with byte constructor";
        let encrypted = aead.encrypt(plaintext).unwrap();
        let decrypted = aead.decrypt(&encrypted).unwrap();

        assert_eq!(decrypted, plaintext);
    }

    #[test]
    fn test_encryption_uniqueness() {
        let aead = Aead::new(TEST_KEY).unwrap();
        let plaintext = "same plaintext";

        // Encrypt same plaintext twice
        let encrypted1 = aead.encrypt(plaintext).unwrap();
        let encrypted2 = aead.encrypt(plaintext).unwrap();

        // Should be different due to random nonce
        assert_ne!(encrypted1, encrypted2);

        // But should decrypt to same plaintext
        assert_eq!(aead.decrypt(&encrypted1).unwrap(), plaintext);
        assert_eq!(aead.decrypt(&encrypted2).unwrap(), plaintext);
    }
}
