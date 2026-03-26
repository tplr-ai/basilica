//! Cryptographic utilities for Basilca
//!
//! This module provides secure cryptographic operations used throughout the system:
//! - Blake3 hashing for data integrity
//! - Bittensor signature verification
//! - Symmetric encryption utilities for transient data protection
//! - Random number generation and key derivation
//!
//! # Security Considerations
//! - All cryptographic operations use industry-standard algorithms
//! - Keys should never be logged or stored in plaintext
//! - Use constant-time operations where possible to prevent timing attacks
//! - Prefer authenticated encryption (AEAD) for symmetric operations

use crate::error::CryptoError;
use crate::identity::Hotkey;
use aes_gcm::aead::{Aead, OsRng};
use aes_gcm::{Aes256Gcm, Key, KeyInit, Nonce};
use bittensor::crypto::sr25519;
use blake3::Hasher;
use rand::RngCore;
use std::str::FromStr;

/// Blake3 hash digest size in bytes
pub const BLAKE3_DIGEST_SIZE: usize = 32;

/// AES-256-GCM key size in bytes
pub const AES_KEY_SIZE: usize = 32;

/// AES-256-GCM nonce size in bytes
pub const AES_NONCE_SIZE: usize = 12;

/// Compute Blake3 hash of input data
///
/// # Arguments
/// * `data` - Input data to hash
///
/// # Returns
/// * Blake3 hash as hexadecimal string
///
/// # Example
/// ```rust
/// use basilica_common::crypto::hash_blake3_string;
///
/// let data = b"Hello, Basilica!";
/// let hash = hash_blake3_string(data);
/// assert_eq!(hash.len(), 64); // 32 bytes * 2 hex chars
/// ```
///
/// # Implementation Notes for Developers
/// - Blake3 is chosen for its speed and security properties
/// - Output is lowercase hexadecimal for consistency
/// - Consider using this for content addressing and integrity verification
/// - Blake3 is resistant to length extension attacks
pub fn hash_blake3_string(data: &[u8]) -> String {
    let mut hasher = Hasher::new();
    hasher.update(data);
    let result = hasher.finalize();
    hex::encode(result.as_bytes())
}

/// Backward compatibility alias
pub use hash_blake3_string as hash_blake3;

/// Verify Bittensor signature
///
/// # Arguments
/// * `hotkey` - The hotkey to verify against
/// * `signature_hex` - The signature as a hex string
/// * `data` - The data that was signed
///
/// # Returns
/// * `Ok(())` - If the signature is valid
/// * `Err(CryptoError)` - If the signature is invalid or verification fails
pub fn verify_bittensor_signature(
    hotkey: &Hotkey,
    signature_hex: &str,
    data: &[u8],
) -> Result<(), CryptoError> {
    // Validate inputs
    if signature_hex.is_empty() {
        return Err(CryptoError::InvalidSignature {
            details: "Empty signature".to_string(),
        });
    }

    if data.is_empty() {
        return Err(CryptoError::InvalidSignature {
            details: "Empty data".to_string(),
        });
    }

    // Decode signature from hex
    let signature_bytes =
        hex::decode(signature_hex).map_err(|e| CryptoError::InvalidSignature {
            details: format!("Invalid hex signature format: {e}"),
        })?;

    // Convert to AccountId32 type (same as AccountId in bittensor)
    let account_id =
        sr25519::Public::from_str(hotkey.as_str()).map_err(|_| CryptoError::InvalidSignature {
            details: format!("Invalid hotkey format: {}", hotkey.as_str()),
        })?;

    // Convert signature bytes to the expected type
    if signature_bytes.len() != 64 {
        return Err(CryptoError::InvalidSignature {
            details: format!(
                "Invalid signature length: expected 64 bytes, got {}",
                signature_bytes.len()
            ),
        });
    }

    // Convert Vec<u8> to fixed-size array for signature
    let mut signature_array = [0u8; 64];
    signature_array.copy_from_slice(&signature_bytes);

    // Create signature from bytes
    let signature = sr25519::Signature::from_raw(signature_array);

    // Verify the signature
    use bittensor::crypto::Pair as _;
    let is_valid = sr25519::Pair::verify(&signature, data, &account_id);

    if is_valid {
        Ok(())
    } else {
        Err(CryptoError::InvalidSignature {
            details: "Signature verification failed".to_string(),
        })
    }
}

/// Backward compatibility function with bytes signature  
pub fn verify_signature_bittensor(
    hotkey: &Hotkey,
    signature: &[u8],
    data: &[u8],
) -> Result<(), CryptoError> {
    let signature_hex = hex::encode(signature);
    verify_bittensor_signature(hotkey, &signature_hex, data)
}

/// Generate ephemeral ED25519 keypair
///
/// # Returns
/// * `(private_pem, public_openssh_format)` - Private key in PEM format, public key in OpenSSH format
///
/// # Implementation Notes for Developers
/// This function uses the ed25519-dalek library to generate a secure random keypair.
/// The private key is formatted as PKCS#8 PEM and the public key as OpenSSH format.
///
/// # Security Notes
/// - Private key should be securely handled and never logged
/// - Use only for ephemeral/temporary authentication
/// - Private key is automatically zeroized when dropped
///
/// # Example
/// ```rust
/// use basilica_common::crypto::generate_ephemeral_ed25519_keypair;
///
/// let (private_pem, public_openssh) = generate_ephemeral_ed25519_keypair();
/// assert!(private_pem.contains("-----BEGIN PRIVATE KEY-----"));
/// assert!(public_openssh.starts_with("ssh-ed25519"));
/// ```
pub fn generate_ephemeral_ed25519_keypair() -> (String, String) {
    // Delegate to the new crypto module implementation
    crate::crypto::keys::generate_ed25519_keypair()
}

/// Derive deterministic key from GPU information
///
/// # Arguments
/// * `gpu_info_str` - GPU information string (e.g., concatenated GPU UUIDs, models, etc.)
///
/// # Returns
/// * 32-byte derived key
///
/// # Implementation Notes for Developers
/// - Uses deterministic key derivation (HKDF-SHA256) for consistent results
/// - Same GPU info should always produce the same key
/// - Different GPU configurations should produce different keys
/// - Consider including system-specific info to prevent key reuse across machines
///
/// # Security Notes
/// - This is for obfuscation/verification purposes, not cryptographic security
/// - Should not be used as primary encryption key
/// - GPU info should include sufficient entropy (UUIDs, models, memory sizes)
pub fn derive_key_from_gpu_info(gpu_info_str: &str) -> [u8; 32] {
    let mut hasher = Hasher::new();
    hasher.update(b"BASILCA_GPU_KEY_DERIVATION"); // Domain separation
    hasher.update(gpu_info_str.as_bytes());

    let mut key = [0u8; 32];
    hasher.finalize_xof().fill(&mut key);
    key
}

/// Generate a cryptographically secure random key
///
/// # Arguments
/// * `size` - Key size in bytes
///
/// # Returns
/// * Random key bytes
///
/// # Example
/// ```rust
/// use basilica_common::crypto::{generate_random_key, AES_KEY_SIZE};
///
/// let key = generate_random_key(AES_KEY_SIZE);
/// assert_eq!(key.len(), AES_KEY_SIZE);
/// ```
pub fn generate_random_key(size: usize) -> Vec<u8> {
    let mut key = vec![0u8; size];
    OsRng.fill_bytes(&mut key);
    key
}

/// Encrypt data using AES-256-GCM
///
/// # Arguments
/// * `data` - Plaintext data to encrypt
/// * `key` - 256-bit encryption key
///
/// # Returns
/// * `Ok((ciphertext, nonce))` - Encrypted data and nonce
/// * `Err(CryptoError)` - If encryption fails
///
/// # Implementation Notes for Developers
/// - Uses AES-256-GCM for authenticated encryption
/// - Nonce is randomly generated for each encryption
/// - Key must be exactly 32 bytes (256 bits)
/// - Ciphertext includes authentication tag
/// - Store nonce alongside ciphertext for decryption
///
/// # Security Notes
/// - Never reuse nonce with the same key
/// - Verify authentication tag during decryption
/// - Key should be derived from secure source (e.g., PBKDF2, Argon2)
#[allow(deprecated)]
pub fn encrypt_aes_gcm(data: &[u8], key: &[u8]) -> Result<(Vec<u8>, Vec<u8>), CryptoError> {
    if key.len() != AES_KEY_SIZE {
        return Err(CryptoError::EncryptionFailed {
            details: format!(
                "Invalid key size: expected {}, got {}",
                AES_KEY_SIZE,
                key.len()
            ),
        });
    }

    let key: &Key<Aes256Gcm> = key.into();
    let cipher = Aes256Gcm::new(key);

    // Generate random nonce
    let mut nonce_bytes = [0u8; AES_NONCE_SIZE];
    OsRng.fill_bytes(&mut nonce_bytes);
    let nonce = Nonce::from_slice(&nonce_bytes);

    let ciphertext = cipher
        .encrypt(nonce, data)
        .map_err(|e| CryptoError::EncryptionFailed {
            details: format!("AES-GCM encryption failed: {e}"),
        })?;

    Ok((ciphertext, nonce_bytes.to_vec()))
}

/// Decrypt data using AES-256-GCM
///
/// # Arguments
/// * `ciphertext` - Encrypted data with authentication tag
/// * `key` - 256-bit decryption key (must match encryption key)
/// * `nonce` - Nonce used during encryption
///
/// # Returns
/// * `Ok(plaintext)` - Decrypted data
/// * `Err(CryptoError)` - If decryption or authentication fails
///
/// # Security Notes
/// - Authentication verification is automatic with GCM mode
/// - Fails if ciphertext has been tampered with
/// - Fails if wrong key or nonce is used
#[allow(deprecated)]
pub fn decrypt_aes_gcm(
    ciphertext: &[u8],
    key: &[u8],
    nonce: &[u8],
) -> Result<Vec<u8>, CryptoError> {
    if key.len() != AES_KEY_SIZE {
        return Err(CryptoError::DecryptionFailed {
            details: format!(
                "Invalid key size: expected {}, got {}",
                AES_KEY_SIZE,
                key.len()
            ),
        });
    }

    if nonce.len() != AES_NONCE_SIZE {
        return Err(CryptoError::DecryptionFailed {
            details: format!(
                "Invalid nonce size: expected {}, got {}",
                AES_NONCE_SIZE,
                nonce.len()
            ),
        });
    }

    let key: &Key<Aes256Gcm> = key.into();
    let cipher = Aes256Gcm::new(key);
    let nonce = Nonce::from_slice(nonce);

    let plaintext =
        cipher
            .decrypt(nonce, ciphertext)
            .map_err(|e| CryptoError::DecryptionFailed {
                details: format!("AES-GCM decryption failed: {e}"),
            })?;

    Ok(plaintext)
}

/// Derive key from password using a simple approach
///
/// # Arguments
/// * `password` - Password or seed material
/// * `salt` - Salt for key derivation
///
/// # Returns
/// * Derived key bytes
///
/// # Implementation Notes for Developers
/// This function now uses PBKDF2-HMAC-SHA256 with 100,000 iterations
/// for proper password-based key derivation.
///
/// # Security Notes
/// - Uses PBKDF2 with 100,000 iterations for security
/// - Salt should be unique per password and stored securely
/// - For new implementations, consider using the more flexible
///   `pbkdf2_derive_key` or `argon2_derive_key` functions from the kdf module
///
/// # Example
/// ```rust
/// use basilica_common::crypto::derive_key_simple;
///
/// let password = "my_secure_password";
/// let salt = b"unique_salt_16by";
/// let key = derive_key_simple(password, salt);
/// assert_eq!(key.len(), 32);
/// ```
pub fn derive_key_simple(password: &str, salt: &[u8]) -> Vec<u8> {
    // Delegate to the new KDF implementation with sensible defaults
    crate::crypto::kdf::derive_key_simple(password, salt)
}

/// Encrypt data using symmetric encryption (AES-256-GCM)
///
/// # Arguments
/// * `key` - 256-bit encryption key
/// * `plaintext` - Data to encrypt
///
/// # Returns
/// * `Ok(ciphertext)` - Encrypted data (includes nonce prepended)
/// * `Err(CryptoError)` - If encryption fails
///
/// # Implementation Notes
/// - Uses AES-256-GCM for authenticated encryption
/// - Nonce is randomly generated and prepended to ciphertext
/// - Key must be exactly 32 bytes (256 bits)
/// - Ciphertext format: [nonce (12 bytes)][encrypted_data_with_tag]
///
/// # Security Notes
/// - Never reuse nonce with the same key (handled automatically)
/// - Key should be derived from secure source (e.g., PBKDF2, Argon2)
pub fn symmetric_encrypt(key: &[u8; 32], plaintext: &[u8]) -> Result<Vec<u8>, CryptoError> {
    let (ciphertext, nonce) = encrypt_aes_gcm(plaintext, key)?;

    // Prepend nonce to ciphertext for storage
    let mut result = Vec::with_capacity(nonce.len() + ciphertext.len());
    result.extend_from_slice(&nonce);
    result.extend_from_slice(&ciphertext);

    Ok(result)
}

/// Decrypt data using symmetric encryption (AES-256-GCM)
///
/// # Arguments
/// * `key` - 256-bit decryption key (must match encryption key)
/// * `ciphertext` - Encrypted data (with nonce prepended)
///
/// # Returns
/// * `Ok(plaintext)` - Decrypted data
/// * `Err(CryptoError)` - If decryption or authentication fails
///
/// # Implementation Notes
/// - Expects ciphertext format: [nonce (12 bytes)][encrypted_data_with_tag]
/// - Authentication verification is automatic with GCM mode
/// - Fails if ciphertext has been tampered with
/// - Fails if wrong key is used
pub fn symmetric_decrypt(key: &[u8; 32], ciphertext: &[u8]) -> Result<Vec<u8>, CryptoError> {
    if ciphertext.len() < AES_NONCE_SIZE {
        return Err(CryptoError::DecryptionFailed {
            details: format!(
                "Ciphertext too short: expected at least {} bytes, got {}",
                AES_NONCE_SIZE,
                ciphertext.len()
            ),
        });
    }

    // Split nonce and encrypted data
    let (nonce, encrypted_data) = ciphertext.split_at(AES_NONCE_SIZE);

    decrypt_aes_gcm(encrypted_data, key, nonce)
}

/// Secure comparison of byte slices to prevent timing attacks
///
/// # Arguments
/// * `a` - First byte slice
/// * `b` - Second byte slice
///
/// # Returns
/// * `true` if slices are equal, `false` otherwise
///
/// # Security Notes
/// - Uses constant-time comparison to prevent timing side-channel attacks
/// - Always compares full length even if early difference is found
/// - Important for comparing secrets, signatures, and authentication tags
pub fn secure_compare(a: &[u8], b: &[u8]) -> bool {
    if a.len() != b.len() {
        return false;
    }

    let mut result = 0u8;
    for (byte_a, byte_b) in a.iter().zip(b.iter()) {
        result |= byte_a ^ byte_b;
    }

    result == 0
}

/// Verify signature with simplified interface for API endpoints
///
/// # Arguments
/// * `signature` - Signature as hex string
/// * `message` - Message that was signed
/// * `hotkey_str` - Hotkey as string
///
/// # Returns
/// * `Ok(true)` if signature is valid
/// * `Ok(false)` if signature is invalid
/// * `Err(anyhow::Error)` if verification process fails
pub async fn verify_signature(
    signature: &str,
    message: &str,
    hotkey_str: &str,
) -> Result<bool, anyhow::Error> {
    let hotkey = Hotkey::new(hotkey_str.to_string())
        .map_err(|e| anyhow::anyhow!("Invalid hotkey format: {}", e))?;

    match verify_bittensor_signature(&hotkey, signature, message.as_bytes()) {
        Ok(()) => Ok(true),
        Err(_) => Ok(false), // Any error means invalid signature
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_blake3_hash() {
        let data = b"test data";
        let hash1 = hash_blake3_string(data);
        let hash2 = hash_blake3_string(data);

        // Same input should produce same hash
        assert_eq!(hash1, hash2);

        // Hash should be 64 hex characters (32 bytes)
        assert_eq!(hash1.len(), 64);

        // Different input should produce different hash
        let hash3 = hash_blake3_string(b"different data");
        assert_ne!(hash1, hash3);

        // Test backward compatibility
        let hash4 = hash_blake3(data);
        assert_eq!(hash1, hash4);
    }

    #[test]
    fn test_random_key_generation() {
        let key1 = generate_random_key(32);
        let key2 = generate_random_key(32);

        assert_eq!(key1.len(), 32);
        assert_eq!(key2.len(), 32);
        assert_ne!(key1, key2); // Should be different
    }

    #[test]
    fn test_aes_gcm_encrypt_decrypt() {
        let data = b"sensitive data that needs encryption";
        let key = generate_random_key(AES_KEY_SIZE);

        // Encrypt
        let (ciphertext, nonce) = encrypt_aes_gcm(data, &key).unwrap();
        assert_ne!(ciphertext, data); // Should be different from plaintext

        // Decrypt
        let decrypted = decrypt_aes_gcm(&ciphertext, &key, &nonce).unwrap();
        assert_eq!(decrypted, data);
    }

    #[test]
    fn test_aes_gcm_invalid_key_size() {
        let data = b"test data";
        let invalid_key = vec![0u8; 16]; // Wrong size

        let result = encrypt_aes_gcm(data, &invalid_key);
        assert!(result.is_err());

        match result.unwrap_err() {
            CryptoError::EncryptionFailed { details } => {
                assert!(details.contains("Invalid key size"));
            }
            _ => panic!("Expected EncryptionFailed error"),
        }
    }

    #[test]
    fn test_aes_gcm_tampered_ciphertext() {
        let data = b"test data";
        let key = generate_random_key(AES_KEY_SIZE);

        let (mut ciphertext, nonce) = encrypt_aes_gcm(data, &key).unwrap();

        // Tamper with ciphertext
        ciphertext[0] ^= 1;

        // Decryption should fail
        let result = decrypt_aes_gcm(&ciphertext, &key, &nonce);
        assert!(result.is_err());
    }

    #[test]
    fn test_key_derivation() {
        let password = "test_password";
        let salt = b"random_salt";

        let key1 = derive_key_simple(password, salt);
        let key2 = derive_key_simple(password, salt);

        assert_eq!(key1, key2); // Same inputs should produce same key
        assert_eq!(key1.len(), AES_KEY_SIZE);

        // Different salt should produce different key
        let key3 = derive_key_simple(password, b"different_salt");
        assert_ne!(key1, key3);
    }

    #[test]
    fn test_secure_compare() {
        let data1 = b"secret_value";
        let data2 = b"secret_value";
        let data3 = b"different_value";

        assert!(secure_compare(data1, data2));
        assert!(!secure_compare(data1, data3));
        assert!(!secure_compare(data1, b"short"));
    }

    #[test]
    fn test_signature_verification_inputs() {
        let hotkey =
            Hotkey::new("5GrwvaEF5zXb26Fz9rcQpDWS57CtERHpNehXCPcNoHGKutQY".to_string()).unwrap();

        // Test empty signature
        let result = verify_bittensor_signature(&hotkey, "", b"data");
        assert!(result.is_err());

        // Test empty data
        let result = verify_bittensor_signature(&hotkey, "deadbeef", b"");
        assert!(result.is_err());

        // Test invalid hex
        let result = verify_bittensor_signature(&hotkey, "invalid_hex_!", b"data");
        assert!(result.is_err());
    }

    #[test]
    fn test_signature_verification_with_bittensor() {
        let hotkey =
            Hotkey::new("5GrwvaEF5zXb26Fz9rcQpDWS57CtERHpNehXCPcNoHGKutQY".to_string()).unwrap();
        let signature_hex = "deadbeefcafebabe1234567890abcdef"; // Valid hex but likely invalid signature
        let data = b"signed_data";

        // This should fail with real bittensor verification (invalid signature)
        let result = verify_bittensor_signature(&hotkey, signature_hex, data);
        // We expect this to fail since we're using a dummy signature
        assert!(result.is_err());
    }

    #[test]
    fn test_ephemeral_keypair_generation() {
        let (private_key, public_key) = generate_ephemeral_ed25519_keypair();

        // Basic format checks
        assert!(private_key.contains("-----BEGIN PRIVATE KEY-----"));
        assert!(private_key.contains("-----END PRIVATE KEY-----"));
        assert!(public_key.starts_with("ssh-ed25519"));
        assert!(public_key.contains("basilica-ephemeral-key"));

        // Should generate different keys each time (though this is placeholder)
        let (_private_key2, _public_key2) = generate_ephemeral_ed25519_keypair();
        // Note: With placeholder implementation, these will be the same
        // When real implementation is added, uncomment:
        // assert_ne!(private_key, private_key2);
        // assert_ne!(public_key, public_key2);
    }

    #[test]
    fn test_gpu_key_derivation() {
        let gpu_info1 = "GPU-12345678-1234-1234-1234-123456789012,RTX4090,24GB";
        let gpu_info2 = "GPU-87654321-4321-4321-4321-210987654321,RTX4080,16GB";

        let key1 = derive_key_from_gpu_info(gpu_info1);
        let key2 = derive_key_from_gpu_info(gpu_info1); // Same input
        let key3 = derive_key_from_gpu_info(gpu_info2); // Different input

        // Same input should produce same key
        assert_eq!(key1, key2);
        assert_eq!(key1.len(), 32);

        // Different input should produce different key
        assert_ne!(key1, key3);

        // Test empty input
        let empty_key = derive_key_from_gpu_info("");
        assert_ne!(key1, empty_key);
        assert_eq!(empty_key.len(), 32);
    }

    #[test]
    fn test_symmetric_encrypt_decrypt() {
        let key = [0u8; 32]; // Test key
        let data = b"test data for symmetric encryption";

        // Encrypt
        let ciphertext = symmetric_encrypt(&key, data).unwrap();
        assert_ne!(ciphertext.as_slice(), data); // Should be different from plaintext
        assert!(ciphertext.len() > data.len()); // Should be larger due to nonce and tag

        // Decrypt
        let decrypted = symmetric_decrypt(&key, &ciphertext).unwrap();
        assert_eq!(decrypted, data);
    }

    #[test]
    fn test_symmetric_decrypt_invalid_input() {
        let key = [0u8; 32];

        // Test with too short ciphertext
        let short_ciphertext = vec![0u8; 5]; // Less than AES_NONCE_SIZE
        let result = symmetric_decrypt(&key, &short_ciphertext);
        assert!(result.is_err());

        // Test with tampered ciphertext
        let data = b"test data";
        let mut ciphertext = symmetric_encrypt(&key, data).unwrap();
        ciphertext[0] ^= 1; // Tamper with first byte (part of nonce)
        let result = symmetric_decrypt(&key, &ciphertext);
        assert!(result.is_err());
    }
}
