//! Ephemeral key rotation system for gpu-attestor binary
//!
//! Manages cryptographic keys for binary signing and validation with automatic rotation.

use super::types::{EphemeralKey, KeyRotationConfig, ValidationError, ValidationResult};
use p256::elliptic_curve::sec1::ToEncodedPoint;
use p256::{
    ecdsa::{Signature, SigningKey, VerifyingKey},
    PublicKey, SecretKey,
};
use rand::rngs::OsRng;
use signature::Verifier;
use std::collections::HashMap;
use std::time::{Duration, SystemTime};
use tokio::fs;
use tracing::{debug, info, warn};

/// Manages ephemeral key rotation for binary validation
#[derive(Debug)]
pub struct EphemeralKeyManager {
    /// Configuration for key rotation
    config: KeyRotationConfig,
    /// Currently managed keys
    keys: HashMap<String, EphemeralKey>,
    /// Last rotation timestamp
    last_rotation: Option<SystemTime>,
}

impl EphemeralKeyManager {
    /// Create a new ephemeral key manager
    pub fn new(config: KeyRotationConfig) -> Self {
        Self {
            config,
            keys: HashMap::new(),
            last_rotation: None,
        }
    }

    /// Initialize the key manager and load existing keys
    pub async fn initialize(&mut self) -> ValidationResult<()> {
        info!("Initializing ephemeral key manager");

        // Create key storage directory if it doesn't exist
        if !self.config.key_storage_dir.exists() {
            fs::create_dir_all(&self.config.key_storage_dir)
                .await
                .map_err(ValidationError::IoError)?;
        }

        // Load existing keys from disk
        self.load_keys_from_disk().await?;

        // Perform initial cleanup
        self.cleanup_expired_keys().await?;

        // Check if we need to rotate immediately
        if self.should_rotate_keys() {
            self.rotate_keys().await?;
        }

        Ok(())
    }

    /// Check if keys should be rotated
    pub fn should_rotate_keys(&self) -> bool {
        match self.last_rotation {
            Some(last) => {
                let elapsed = last.elapsed().unwrap_or(Duration::MAX);
                elapsed >= self.config.rotation_interval
            }
            None => true, // First rotation
        }
    }

    /// Get the currently active key
    pub fn get_active_key(&self) -> Option<&EphemeralKey> {
        self.keys
            .values()
            .find(|key| key.is_active && !self.is_key_expired(key))
    }

    /// Get a key by ID for verification
    pub fn get_key_by_id(&self, key_id: &str) -> Option<&EphemeralKey> {
        self.keys.get(key_id)
    }

    /// Check if a key is expired
    pub fn is_key_expired(&self, key: &EphemeralKey) -> bool {
        SystemTime::now() > key.expires_at
    }

    /// Rotate keys - generate new active key and deactivate old ones
    pub async fn rotate_keys(&mut self) -> ValidationResult<String> {
        info!("Rotating ephemeral keys");

        // Deactivate current active key
        for key in self.keys.values_mut() {
            key.is_active = false;
        }

        // Generate new key
        let new_key = self.generate_new_key().await?;
        let key_id = new_key.key_id.clone();

        // Store new key
        self.keys.insert(key_id.clone(), new_key);

        // Update rotation timestamp
        self.last_rotation = Some(SystemTime::now());

        // Persist keys to disk
        self.save_keys_to_disk().await?;

        // Cleanup old keys
        self.cleanup_expired_keys().await?;

        info!("Key rotation completed. New active key: {}", key_id);
        Ok(key_id)
    }

    /// Generate a new ephemeral key
    async fn generate_new_key(&self) -> ValidationResult<EphemeralKey> {
        let now = SystemTime::now();
        let key_id = format!(
            "key_{}",
            now.duration_since(SystemTime::UNIX_EPOCH)
                .unwrap_or(Duration::ZERO)
                .as_secs()
        );

        // In a real implementation, this would generate actual cryptographic keys
        // For now, we simulate with a placeholder
        let public_key_hex = self.generate_p256_key(&key_id).await?;

        Ok(EphemeralKey {
            key_id,
            public_key_hex,
            created_at: now,
            expires_at: now + self.config.key_retention_period,
            is_active: true,
        })
    }

    /// Generate P256 ECDSA keypair for cryptographic signing
    async fn generate_p256_key(&self, key_id: &str) -> ValidationResult<String> {
        debug!("Generating P256 ECDSA keypair for key_id: {}", key_id);

        // Generate a cryptographically secure P256 private key
        let secret_key = SecretKey::random(&mut OsRng);
        let signing_key = SigningKey::from(secret_key.clone());

        // Extract the public key
        let verifying_key = VerifyingKey::from(&signing_key);
        let public_key = PublicKey::from(&verifying_key);

        // Encode the public key as compressed point for storage
        let public_key_bytes = public_key.to_encoded_point(true); // true = compressed format
        let public_key_hex = hex::encode(public_key_bytes.as_bytes());

        // Store the private key securely for later use in signing
        // In production, this should be stored in a secure key store
        let private_key_hex = hex::encode(secret_key.to_bytes());

        // For now, store both keys in a secure location
        // TODO: In production, implement proper key storage with HSM or secure enclave
        let key_storage_path = format!("/tmp/basilica_keys/{}.key", key_id);
        if let Some(parent) = std::path::Path::new(&key_storage_path).parent() {
            tokio::fs::create_dir_all(parent).await.map_err(|e| {
                ValidationError::CryptoError(format!("Failed to create key directory: {}", e))
            })?;
        }

        // Store private key with restricted permissions
        tokio::fs::write(&key_storage_path, private_key_hex.as_bytes())
            .await
            .map_err(|e| {
                ValidationError::CryptoError(format!("Failed to store private key: {}", e))
            })?;

        // Set restrictive permissions (Unix only)
        #[cfg(unix)]
        {
            use std::os::unix::fs::PermissionsExt;
            let permissions = std::fs::Permissions::from_mode(0o600); // Owner read/write only
            std::fs::set_permissions(&key_storage_path, permissions).map_err(|e| {
                ValidationError::CryptoError(format!("Failed to set key permissions: {}", e))
            })?;
        }

        info!("Generated and stored P256 keypair for key_id: {}", key_id);
        Ok(public_key_hex)
    }

    /// Load keys from disk storage
    async fn load_keys_from_disk(&mut self) -> ValidationResult<()> {
        let keys_file = self.config.key_storage_dir.join("keys.json");

        if !keys_file.exists() {
            debug!("No existing keys file found");
            return Ok(());
        }

        let content = fs::read_to_string(&keys_file)
            .await
            .map_err(ValidationError::IoError)?;

        let loaded_keys: HashMap<String, EphemeralKey> =
            serde_json::from_str(&content).map_err(ValidationError::SerializationError)?;

        self.keys = loaded_keys;
        info!("Loaded {} keys from disk", self.keys.len());

        Ok(())
    }

    /// Save keys to disk storage
    async fn save_keys_to_disk(&self) -> ValidationResult<()> {
        let keys_file = self.config.key_storage_dir.join("keys.json");

        let content = serde_json::to_string_pretty(&self.keys)
            .map_err(ValidationError::SerializationError)?;

        fs::write(&keys_file, content)
            .await
            .map_err(ValidationError::IoError)?;

        debug!("Saved {} keys to disk", self.keys.len());
        Ok(())
    }

    /// Remove expired keys from memory and disk
    async fn cleanup_expired_keys(&mut self) -> ValidationResult<()> {
        let now = SystemTime::now();
        let initial_count = self.keys.len();

        // Remove expired keys beyond retention period
        self.keys.retain(|_, key| {
            now <= key.expires_at
                || (now.duration_since(key.expires_at).unwrap_or(Duration::ZERO)
                    < self.config.key_retention_period)
        });

        // Limit total number of keys
        if self.keys.len() > self.config.max_keys {
            let mut keys_vec: Vec<_> = self
                .keys
                .iter()
                .map(|(id, key)| (id.clone(), key.created_at))
                .collect();
            keys_vec.sort_by_key(|(_, created_at)| *created_at);

            let to_remove = self.keys.len() - self.config.max_keys;
            let keys_to_remove: Vec<String> = keys_vec
                .iter()
                .take(to_remove)
                .map(|(id, _)| id.clone())
                .collect();

            for key_id in keys_to_remove {
                self.keys.remove(&key_id);
            }
        }

        let removed_count = initial_count - self.keys.len();
        if removed_count > 0 {
            info!("Cleaned up {} expired keys", removed_count);
            self.save_keys_to_disk().await?;
        }

        Ok(())
    }

    /// Verify P256 ECDSA signature against stored key
    pub async fn verify_signature(
        &self,
        data: &[u8],
        signature_hex: &str,
        key_id: &str,
    ) -> ValidationResult<bool> {
        if let Some(key) = self.get_key_by_id(key_id) {
            if self.is_key_expired(key) {
                warn!("Attempting to verify with expired key: {}", key_id);
                return Ok(false);
            }

            debug!("Verifying P256 ECDSA signature with key: {}", key_id);

            // Load the corresponding private key to reconstruct the public key
            let key_storage_path = format!("/tmp/basilica_keys/{}.key", key_id);

            let private_key_hex = match tokio::fs::read_to_string(&key_storage_path).await {
                Ok(content) => content,
                Err(e) => {
                    warn!("Failed to read private key for verification: {}", e);
                    return Ok(false);
                }
            };

            // Parse the private key from hex
            let private_key_bytes = match hex::decode(private_key_hex.trim()) {
                Ok(bytes) => bytes,
                Err(e) => {
                    warn!("Failed to decode private key hex: {}", e);
                    return Ok(false);
                }
            };

            // Reconstruct the secret key
            let secret_key = match SecretKey::from_slice(&private_key_bytes) {
                Ok(key) => key,
                Err(e) => {
                    warn!("Failed to reconstruct secret key: {}", e);
                    return Ok(false);
                }
            };

            // Get the verifying key (public key) from the signing key
            let signing_key = SigningKey::from(secret_key);
            let verifying_key = VerifyingKey::from(&signing_key);

            // Parse the signature from hex
            let signature_bytes = match hex::decode(signature_hex) {
                Ok(bytes) => bytes,
                Err(e) => {
                    warn!("Failed to decode signature hex: {}", e);
                    return Ok(false);
                }
            };

            // Parse the signature
            let signature = match Signature::from_slice(&signature_bytes) {
                Ok(sig) => sig,
                Err(e) => {
                    warn!("Failed to parse signature: {}", e);
                    return Ok(false);
                }
            };

            // Verify the signature
            match verifying_key.verify(data, &signature) {
                Ok(()) => {
                    debug!("Signature verification successful for key: {}", key_id);
                    Ok(true)
                }
                Err(e) => {
                    warn!("Signature verification failed for key {}: {}", key_id, e);
                    Ok(false)
                }
            }
        } else {
            warn!("Key not found for verification: {}", key_id);
            Ok(false)
        }
    }

    /// Get active key for binary building
    pub fn get_active_key_for_build(&self) -> ValidationResult<String> {
        if let Some(key) = self.get_active_key() {
            Ok(key.public_key_hex.clone())
        } else {
            Err(ValidationError::ConfigError(
                "No active key available for binary building".to_string(),
            ))
        }
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use tempfile::TempDir;

    #[tokio::test]
    async fn test_key_manager_initialization() {
        let temp_dir = TempDir::new().unwrap();
        let config = KeyRotationConfig {
            key_storage_dir: temp_dir.path().to_path_buf(),
            ..Default::default()
        };

        let mut manager = EphemeralKeyManager::new(config);
        assert!(manager.initialize().await.is_ok());
        assert!(manager.get_active_key().is_some());
    }

    #[tokio::test]
    async fn test_key_rotation() {
        let temp_dir = TempDir::new().unwrap();
        let config = KeyRotationConfig {
            key_storage_dir: temp_dir.path().to_path_buf(),
            rotation_interval: Duration::from_secs(1),
            ..Default::default()
        };

        let mut manager = EphemeralKeyManager::new(config);
        manager.initialize().await.unwrap();

        let initial_key_id = manager.get_active_key().unwrap().key_id.clone();

        // Wait for rotation interval
        tokio::time::sleep(Duration::from_secs(2)).await;

        let new_key_id = manager.rotate_keys().await.unwrap();
        assert_ne!(initial_key_id, new_key_id);
    }

    #[tokio::test]
    async fn test_key_persistence() {
        let temp_dir = TempDir::new().unwrap();
        let config = KeyRotationConfig {
            key_storage_dir: temp_dir.path().to_path_buf(),
            ..Default::default()
        };

        // Create manager and rotate keys
        let mut manager1 = EphemeralKeyManager::new(config.clone());
        manager1.initialize().await.unwrap();
        let key_id = manager1.rotate_keys().await.unwrap();

        // Create new manager and verify key persistence
        let mut manager2 = EphemeralKeyManager::new(config);
        manager2.initialize().await.unwrap();

        assert!(manager2.get_key_by_id(&key_id).is_some());
    }
}
