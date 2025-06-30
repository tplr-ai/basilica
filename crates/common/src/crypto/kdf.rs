//! Key Derivation Functions (KDF)
//!
//! This module provides secure key derivation functions including PBKDF2 and Argon2
//! for deriving cryptographic keys from passwords or other key material.

use crate::error::CryptoError;
use argon2::{
    password_hash::{PasswordHasher, SaltString},
    Argon2, Params, Version,
};
use pbkdf2::pbkdf2_hmac_array;
use rand::{rngs::OsRng, RngCore};
use sha2::Sha256;

/// Default number of iterations for PBKDF2
pub const PBKDF2_DEFAULT_ITERATIONS: u32 = 100_000;

/// Default memory cost for Argon2 (64 MiB)
pub const ARGON2_DEFAULT_MEMORY: u32 = 65536;

/// Default number of iterations for Argon2
pub const ARGON2_DEFAULT_ITERATIONS: u32 = 3;

/// Default parallelism for Argon2
pub const ARGON2_DEFAULT_PARALLELISM: u32 = 4;

/// Parameters for key derivation functions
#[derive(Clone, Debug)]
pub struct KdfParams {
    /// Salt for key derivation (should be unique per password)
    pub salt: Vec<u8>,
    /// Number of iterations (for PBKDF2) or time cost (for Argon2)
    pub iterations: u32,
    /// Memory cost in KiB (for Argon2 only)
    pub memory_cost: Option<u32>,
    /// Parallelism degree (for Argon2 only)
    pub parallelism: Option<u32>,
    /// Output key length in bytes
    pub key_length: usize,
}

impl KdfParams {
    /// Create default parameters for PBKDF2
    pub fn pbkdf2_default() -> Self {
        let mut salt = vec![0u8; 16];
        OsRng.fill_bytes(&mut salt);

        Self {
            salt,
            iterations: PBKDF2_DEFAULT_ITERATIONS,
            memory_cost: None,
            parallelism: None,
            key_length: 32,
        }
    }

    /// Create default parameters for Argon2
    pub fn argon2_default() -> Self {
        let mut salt = vec![0u8; 16];
        OsRng.fill_bytes(&mut salt);

        Self {
            salt,
            iterations: ARGON2_DEFAULT_ITERATIONS,
            memory_cost: Some(ARGON2_DEFAULT_MEMORY),
            parallelism: Some(ARGON2_DEFAULT_PARALLELISM),
            key_length: 32,
        }
    }

    /// Create parameters with a specific salt
    pub fn with_salt(mut self, salt: Vec<u8>) -> Self {
        self.salt = salt;
        self
    }

    /// Set the number of iterations
    pub fn with_iterations(mut self, iterations: u32) -> Self {
        self.iterations = iterations;
        self
    }

    /// Set the output key length
    pub fn with_key_length(mut self, key_length: usize) -> Self {
        self.key_length = key_length;
        self
    }

    /// Set memory cost for Argon2
    pub fn with_memory_cost(mut self, memory_cost: u32) -> Self {
        self.memory_cost = Some(memory_cost);
        self
    }

    /// Set parallelism for Argon2
    pub fn with_parallelism(mut self, parallelism: u32) -> Self {
        self.parallelism = Some(parallelism);
        self
    }
}

/// Derive a key using PBKDF2-HMAC-SHA256
///
/// # Arguments
/// * `password` - Password or passphrase to derive from
/// * `params` - KDF parameters including salt and iterations
///
/// # Returns
/// * Derived key bytes
///
/// # Security Notes
/// - Use a unique salt for each password (at least 16 bytes)
/// - Use at least 100,000 iterations for passwords
/// - Consider using Argon2 for better resistance against GPU/ASIC attacks
pub fn pbkdf2_derive_key(password: &str, params: &KdfParams) -> Result<Vec<u8>, CryptoError> {
    if params.salt.is_empty() {
        return Err(CryptoError::KeyDerivationFailed {
            details: "Salt cannot be empty".to_string(),
        });
    }

    if params.iterations == 0 {
        return Err(CryptoError::KeyDerivationFailed {
            details: "Iterations must be greater than 0".to_string(),
        });
    }

    if params.key_length == 0 || params.key_length > 1024 {
        return Err(CryptoError::KeyDerivationFailed {
            details: "Invalid key length".to_string(),
        });
    }

    // For keys up to 32 bytes, use the optimized array version
    if params.key_length == 32 {
        let key: [u8; 32] =
            pbkdf2_hmac_array::<Sha256, 32>(password.as_bytes(), &params.salt, params.iterations);
        Ok(key.to_vec())
    } else {
        // For other key lengths, use the dynamic version
        let mut key = vec![0u8; params.key_length];
        pbkdf2::pbkdf2_hmac::<Sha256>(
            password.as_bytes(),
            &params.salt,
            params.iterations,
            &mut key,
        );
        Ok(key)
    }
}

/// Derive a key using Argon2id
///
/// # Arguments
/// * `password` - Password or passphrase to derive from
/// * `params` - KDF parameters including salt, iterations, memory cost, and parallelism
///
/// # Returns
/// * Derived key bytes
///
/// # Security Notes
/// - Argon2id is recommended for password hashing as it's memory-hard
/// - Use appropriate memory cost based on available resources (64 MiB default)
/// - Adjust parallelism based on available CPU cores
pub fn argon2_derive_key(password: &str, params: &KdfParams) -> Result<Vec<u8>, CryptoError> {
    if params.salt.is_empty() {
        return Err(CryptoError::KeyDerivationFailed {
            details: "Salt cannot be empty".to_string(),
        });
    }

    if params.key_length == 0 || params.key_length > 1024 {
        return Err(CryptoError::KeyDerivationFailed {
            details: "Invalid key length".to_string(),
        });
    }

    // Convert parameters
    let memory_cost = params.memory_cost.unwrap_or(ARGON2_DEFAULT_MEMORY);
    let parallelism = params.parallelism.unwrap_or(ARGON2_DEFAULT_PARALLELISM);

    // Create Argon2 parameters
    let argon2_params = Params::new(
        memory_cost,
        params.iterations,
        parallelism,
        Some(params.key_length),
    )
    .map_err(|e| CryptoError::KeyDerivationFailed {
        details: format!("Invalid Argon2 parameters: {e}"),
    })?;

    // Create Argon2 instance
    let argon2 = Argon2::new(argon2::Algorithm::Argon2id, Version::V0x13, argon2_params);

    // Generate output
    let salt_string =
        SaltString::encode_b64(&params.salt).map_err(|e| CryptoError::KeyDerivationFailed {
            details: format!("Failed to encode salt: {e}"),
        })?;

    let hash = argon2
        .hash_password(password.as_bytes(), &salt_string)
        .map_err(|e| CryptoError::KeyDerivationFailed {
            details: format!("Argon2 hashing failed: {e}"),
        })?;

    // Extract the raw hash bytes
    let hash_bytes = hash.hash.ok_or_else(|| CryptoError::KeyDerivationFailed {
        details: "No hash output from Argon2".to_string(),
    })?;

    Ok(hash_bytes.as_bytes().to_vec())
}

/// Simple wrapper for backward compatibility - uses PBKDF2 with sensible defaults
///
/// # Arguments
/// * `password` - Password to derive from
/// * `salt` - Salt for derivation
///
/// # Returns
/// * 32-byte derived key
///
/// # Note
/// This function exists for backward compatibility. New code should use
/// `pbkdf2_derive_key` or `argon2_derive_key` directly with proper parameters.
pub fn derive_key_simple(password: &str, salt: &[u8]) -> Vec<u8> {
    let params = KdfParams::pbkdf2_default()
        .with_salt(salt.to_vec())
        .with_iterations(100_000)
        .with_key_length(32);

    pbkdf2_derive_key(password, &params).unwrap_or_else(|_| {
        // Fallback to a simple hash if KDF fails (should not happen with valid inputs)
        let mut result = vec![0u8; 32];
        result.copy_from_slice(&blake3::hash(password.as_bytes()).as_bytes()[..32]);
        result
    })
}

/// Securely compare password-derived keys
///
/// # Arguments
/// * `password` - Password to verify
/// * `salt` - Salt used in original derivation
/// * `expected_key` - Expected derived key
/// * `params` - KDF parameters used in original derivation
///
/// # Returns
/// * `true` if the password derives to the expected key, `false` otherwise
pub fn verify_password_pbkdf2(
    password: &str,
    salt: &[u8],
    expected_key: &[u8],
    iterations: u32,
) -> bool {
    let params = KdfParams::pbkdf2_default()
        .with_salt(salt.to_vec())
        .with_iterations(iterations)
        .with_key_length(expected_key.len());

    match pbkdf2_derive_key(password, &params) {
        Ok(derived_key) => {
            // Use constant-time comparison
            if derived_key.len() != expected_key.len() {
                return false;
            }

            let mut result = 0u8;
            for (a, b) in derived_key.iter().zip(expected_key.iter()) {
                result |= a ^ b;
            }
            result == 0
        }
        Err(_) => false,
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_pbkdf2_derivation() {
        let password = "test_password123";
        let params = KdfParams::pbkdf2_default()
            .with_salt(b"test_salt_16byte".to_vec())
            .with_iterations(1000)
            .with_key_length(32);

        let key1 = pbkdf2_derive_key(password, &params).unwrap();
        let key2 = pbkdf2_derive_key(password, &params).unwrap();

        // Same inputs should produce same key
        assert_eq!(key1, key2);
        assert_eq!(key1.len(), 32);

        // Different salt should produce different key
        let params2 = params.clone().with_salt(b"different_salt16".to_vec());
        let key3 = pbkdf2_derive_key(password, &params2).unwrap();
        assert_ne!(key1, key3);

        // Different password should produce different key
        let key4 = pbkdf2_derive_key("different_password", &params).unwrap();
        assert_ne!(key1, key4);
    }

    #[test]
    fn test_argon2_derivation() {
        let password = "test_password123";
        let params = KdfParams::argon2_default()
            .with_salt(b"test_salt_16byte".to_vec())
            .with_iterations(1)
            .with_memory_cost(1024) // 1 MiB for testing
            .with_parallelism(1)
            .with_key_length(32);

        let key1 = argon2_derive_key(password, &params).unwrap();
        let key2 = argon2_derive_key(password, &params).unwrap();

        // Same inputs should produce same key
        assert_eq!(key1, key2);
        assert_eq!(key1.len(), 32);

        // Different salt should produce different key
        let params2 = params.clone().with_salt(b"different_salt16".to_vec());
        let key3 = argon2_derive_key(password, &params2).unwrap();
        assert_ne!(key1, key3);
    }

    #[test]
    fn test_different_key_lengths() {
        let password = "test_password";
        let salt = b"test_salt_16byte".to_vec();

        // Test PBKDF2 with different key lengths
        for key_length in [16, 24, 32, 48, 64] {
            let params = KdfParams::pbkdf2_default()
                .with_salt(salt.clone())
                .with_iterations(1000)
                .with_key_length(key_length);

            let key = pbkdf2_derive_key(password, &params).unwrap();
            assert_eq!(key.len(), key_length);
        }

        // Test Argon2 with different key lengths
        for key_length in [16, 24, 32, 48, 64] {
            let params = KdfParams::argon2_default()
                .with_salt(salt.clone())
                .with_iterations(1)
                .with_memory_cost(1024)
                .with_key_length(key_length);

            let key = argon2_derive_key(password, &params).unwrap();
            assert_eq!(key.len(), key_length);
        }
    }

    #[test]
    fn test_derive_key_simple_compatibility() {
        let password = "test_password";
        let salt = b"test_salt";

        let key1 = derive_key_simple(password, salt);
        let key2 = derive_key_simple(password, salt);

        assert_eq!(key1, key2);
        assert_eq!(key1.len(), 32);

        // Should be equivalent to PBKDF2 with 100k iterations
        let params = KdfParams::pbkdf2_default()
            .with_salt(salt.to_vec())
            .with_iterations(100_000)
            .with_key_length(32);

        let key3 = pbkdf2_derive_key(password, &params).unwrap();
        assert_eq!(key1, key3);
    }

    #[test]
    fn test_password_verification() {
        let password = "correct_password";
        let salt = b"verification_salt";
        let iterations = 10_000;

        let params = KdfParams::pbkdf2_default()
            .with_salt(salt.to_vec())
            .with_iterations(iterations)
            .with_key_length(32);

        let expected_key = pbkdf2_derive_key(password, &params).unwrap();

        // Correct password should verify
        assert!(verify_password_pbkdf2(
            password,
            salt,
            &expected_key,
            iterations
        ));

        // Wrong password should not verify
        assert!(!verify_password_pbkdf2(
            "wrong_password",
            salt,
            &expected_key,
            iterations
        ));

        // Wrong salt should not verify
        assert!(!verify_password_pbkdf2(
            password,
            b"wrong_salt",
            &expected_key,
            iterations
        ));

        // Wrong iterations should not verify
        assert!(!verify_password_pbkdf2(
            password,
            salt,
            &expected_key,
            iterations + 1
        ));
    }

    #[test]
    fn test_invalid_parameters() {
        let password = "test";

        // Empty salt should fail
        let params = KdfParams::pbkdf2_default().with_salt(vec![]);
        assert!(pbkdf2_derive_key(password, &params).is_err());
        assert!(argon2_derive_key(password, &params).is_err());

        // Zero iterations should fail for PBKDF2
        let params = KdfParams::pbkdf2_default().with_iterations(0);
        assert!(pbkdf2_derive_key(password, &params).is_err());

        // Invalid key length should fail
        let params = KdfParams::pbkdf2_default().with_key_length(0);
        assert!(pbkdf2_derive_key(password, &params).is_err());
        assert!(argon2_derive_key(password, &params).is_err());
    }
}
