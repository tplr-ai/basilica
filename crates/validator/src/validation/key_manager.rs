//! Ephemeral key rotation system for gpu-attestor binary
//!
//! Manages cryptographic keys for binary signing and validation with automatic rotation.

use super::types::{EphemeralKey, KeyRotationConfig, ValidationError, ValidationResult};
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
        let public_key_hex = self.generate_placeholder_key(&key_id).await?;

        Ok(EphemeralKey {
            key_id,
            public_key_hex,
            created_at: now,
            expires_at: now + self.config.key_retention_period,
            is_active: true,
        })
    }

    /// Generate placeholder key (replace with actual key generation)
    async fn generate_placeholder_key(&self, key_id: &str) -> ValidationResult<String> {
        // TODO: Replace with actual P256 key generation
        // This is a placeholder implementation
        use sha2::{Digest, Sha256};

        let mut hasher = Sha256::new();
        hasher.update(key_id.as_bytes());
        hasher.update(
            SystemTime::now()
                .duration_since(SystemTime::UNIX_EPOCH)
                .unwrap_or(Duration::ZERO)
                .as_secs()
                .to_le_bytes(),
        );

        let hash = hasher.finalize();
        Ok(hex::encode(hash))
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

    /// Verify a signature using any available key
    pub fn verify_signature(
        &self,
        _data: &[u8],
        _signature: &str,
        key_id: &str,
    ) -> ValidationResult<bool> {
        if let Some(key) = self.get_key_by_id(key_id) {
            if self.is_key_expired(key) {
                warn!("Attempting to verify with expired key: {}", key_id);
                return Ok(false);
            }

            // TODO: Implement actual signature verification
            // This is a placeholder implementation
            debug!("Verifying signature with key: {}", key_id);
            Ok(true) // Placeholder
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
