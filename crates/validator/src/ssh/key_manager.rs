//! SSH Key Management for Validator Sessions
//!
//! Handles generation, storage, and cleanup of ephemeral SSH keys for validator-executor sessions.

use anyhow::{Context, Result};
use ssh_key::{Algorithm, LineEnding, PrivateKey, PublicKey};
use std::path::PathBuf;
use std::time::SystemTime;
use tokio::fs;
use tracing::{debug, info, warn};

/// Manages SSH keys for validator sessions
#[derive(Debug, Clone)]
pub struct ValidatorSshKeyManager {
    /// Directory for storing ephemeral keys
    key_dir: PathBuf,
    /// SSH key algorithm to use
    key_algorithm: Algorithm,
    /// Persistent SSH key for validator (loaded once at startup)
    persistent_key: Option<(String, PathBuf)>, // (public_key, private_key_path)
}

impl ValidatorSshKeyManager {
    /// Create a new SSH key manager
    pub async fn new(key_dir: PathBuf) -> Result<Self> {
        // Ensure key directory exists with proper permissions
        fs::create_dir_all(&key_dir)
            .await
            .context("Failed to create SSH key directory")?;

        #[cfg(unix)]
        {
            use std::os::unix::fs::PermissionsExt;
            let metadata = fs::metadata(&key_dir).await?;
            let mut perms = metadata.permissions();
            perms.set_mode(0o700); // rwx------
            fs::set_permissions(&key_dir, perms)
                .await
                .context("Failed to set key directory permissions")?;
        }

        Ok(Self {
            key_dir,
            key_algorithm: Algorithm::Ed25519,
            persistent_key: None,
        })
    }

    /// Load or generate persistent SSH key for validator
    pub async fn load_or_generate_persistent_key(
        &mut self,
        key_path: Option<PathBuf>,
    ) -> Result<(String, PathBuf)> {
        let persistent_key_path =
            key_path.unwrap_or_else(|| self.key_dir.join("validator_persistent.pem"));

        // Check if persistent key already exists
        if persistent_key_path.exists() {
            info!(
                "Loading existing persistent SSH key from {}",
                persistent_key_path.display()
            );
            let (public_key, private_key_path) = self
                .load_existing_persistent_key(&persistent_key_path)
                .await?;
            self.persistent_key = Some((public_key.clone(), private_key_path.clone()));
            Ok((public_key, private_key_path))
        } else {
            info!(
                "Generating new persistent SSH key at {}",
                persistent_key_path.display()
            );
            let (public_key, private_key_path) =
                self.generate_persistent_key(&persistent_key_path).await?;
            self.persistent_key = Some((public_key.clone(), private_key_path.clone()));
            Ok((public_key, private_key_path))
        }
    }

    /// Load existing persistent SSH key
    async fn load_existing_persistent_key(&self, key_path: &PathBuf) -> Result<(String, PathBuf)> {
        // Read private key file
        let key_content = fs::read_to_string(key_path)
            .await
            .context("Failed to read persistent SSH key file")?;

        // Parse private key
        let private_key = PrivateKey::from_openssh(&key_content)
            .context("Failed to parse persistent SSH private key")?;

        // Get public key
        let public_key = private_key.public_key();
        let public_key_str = Self::get_public_key_openssh(public_key)?;

        debug!("Loaded persistent SSH key from {}", key_path.display());
        Ok((public_key_str, key_path.clone()))
    }

    /// Generate new persistent SSH key
    async fn generate_persistent_key(&self, key_path: &PathBuf) -> Result<(String, PathBuf)> {
        // Generate keypair
        let mut rng = rand::thread_rng();
        let private_key = PrivateKey::random(&mut rng, self.key_algorithm.clone())
            .context("Failed to generate persistent private key")?;
        let public_key = private_key.public_key();

        // Convert to OpenSSH format
        let key_content = private_key
            .to_openssh(LineEnding::default())
            .context("Failed to convert persistent key to OpenSSH format")?;
        let public_key_str = Self::get_public_key_openssh(public_key)?;

        // Ensure parent directory exists
        if let Some(parent) = key_path.parent() {
            fs::create_dir_all(parent)
                .await
                .context("Failed to create persistent key directory")?;
        }

        // Write private key to file
        fs::write(key_path, key_content.as_bytes())
            .await
            .context("Failed to write persistent private key")?;

        // Set permissions to 0600
        #[cfg(unix)]
        {
            use std::os::unix::fs::PermissionsExt;
            let metadata = fs::metadata(key_path).await?;
            let mut perms = metadata.permissions();
            perms.set_mode(0o600); // rw-------
            fs::set_permissions(key_path, perms)
                .await
                .context("Failed to set persistent key permissions")?;
        }

        info!("Generated persistent SSH key at {}", key_path.display());
        Ok((public_key_str, key_path.clone()))
    }

    /// Get the persistent SSH key (public key and private key path)
    pub fn get_persistent_key(&self) -> Option<&(String, PathBuf)> {
        self.persistent_key.as_ref()
    }

    /// Generate ephemeral SSH keypair for a session
    pub async fn generate_session_keypair(
        &self,
        session_id: &str,
    ) -> Result<(PrivateKey, PublicKey, PathBuf)> {
        info!("Generating SSH keypair for session {}", session_id);

        // Generate keypair (do this synchronously to avoid Send issues)
        let (private_key, public_key, key_content) = {
            let mut rng = rand::thread_rng();
            let private_key = PrivateKey::random(&mut rng, self.key_algorithm.clone())
                .context("Failed to generate private key")?;
            let public_key = private_key.public_key().clone();
            let key_content = private_key
                .to_openssh(LineEnding::default())
                .context("Failed to convert key to OpenSSH format")?;
            (private_key, public_key, key_content)
        };

        // Save private key with restricted permissions
        let key_path = self.key_dir.join(format!("session_{session_id}.pem"));

        // Write key to file
        fs::write(&key_path, key_content.as_bytes())
            .await
            .context("Failed to write private key")?;

        // Set permissions to 0600
        #[cfg(unix)]
        {
            use std::os::unix::fs::PermissionsExt;
            let metadata = fs::metadata(&key_path).await?;
            let mut perms = metadata.permissions();
            perms.set_mode(0o600); // rw-------
            fs::set_permissions(&key_path, perms)
                .await
                .context("Failed to set key permissions")?;
        }

        debug!(
            "Generated SSH keypair for session {} at {}",
            session_id,
            key_path.display()
        );

        Ok((private_key, public_key, key_path))
    }

    /// Get the path to a session's private key
    pub fn get_session_key_path(&self, session_id: &str) -> PathBuf {
        self.key_dir.join(format!("session_{session_id}.pem"))
    }

    /// Clean up session keys
    pub async fn cleanup_session_keys(&self, session_id: &str) -> Result<()> {
        let key_path = self.get_session_key_path(session_id);
        if key_path.exists() {
            info!("Cleaning up SSH key for session {}", session_id);
            fs::remove_file(&key_path)
                .await
                .context("Failed to remove session key")?;
        }
        Ok(())
    }

    /// Clean up all expired session keys
    pub async fn cleanup_expired_keys(&self, max_age_secs: u64) -> Result<usize> {
        let mut cleaned = 0;
        let now = SystemTime::now();

        let mut entries = fs::read_dir(&self.key_dir)
            .await
            .context("Failed to read key directory")?;

        while let Some(entry) = entries.next_entry().await? {
            let path = entry.path();

            // Only process session key files
            if let Some(filename) = path.file_name().and_then(|n| n.to_str()) {
                if !filename.starts_with("session_") || !filename.ends_with(".pem") {
                    continue;
                }
            } else {
                continue;
            }

            // Check file age
            if let Ok(metadata) = entry.metadata().await {
                if let Ok(modified) = metadata.modified() {
                    if let Ok(age) = now.duration_since(modified) {
                        if age.as_secs() >= max_age_secs {
                            if let Err(e) = fs::remove_file(&path).await {
                                warn!("Failed to remove expired key {}: {}", path.display(), e);
                            } else {
                                cleaned += 1;
                                debug!("Removed expired key: {}", path.display());
                            }
                        }
                    }
                }
            }
        }

        if cleaned > 0 {
            info!("Cleaned up {} expired SSH keys", cleaned);
        }

        Ok(cleaned)
    }

    /// Get public key in OpenSSH format
    pub fn get_public_key_openssh(public_key: &PublicKey) -> Result<String> {
        public_key
            .to_openssh()
            .context("Failed to convert public key to OpenSSH format")
    }

    /// Run periodic cleanup task
    pub async fn run_cleanup_task(&self, cleanup_interval: std::time::Duration, max_age_secs: u64) {
        let mut interval = tokio::time::interval(cleanup_interval);

        loop {
            interval.tick().await;

            if let Err(e) = self.cleanup_expired_keys(max_age_secs).await {
                warn!("SSH key cleanup failed: {}", e);
            }
        }
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use std::time::Duration;
    use tempfile::TempDir;

    #[tokio::test]
    async fn test_ssh_key_generation() {
        let temp_dir = TempDir::new().unwrap();
        let key_manager = ValidatorSshKeyManager::new(temp_dir.path().to_path_buf())
            .await
            .unwrap();

        let session_id = "test-session-123";
        let (private_key, public_key, key_path) = key_manager
            .generate_session_keypair(session_id)
            .await
            .unwrap();

        // Verify key files exist
        assert!(key_path.exists());

        // Verify key formats
        let private_openssh = private_key.to_openssh(LineEnding::default()).unwrap();
        assert!(private_openssh
            .as_str()
            .contains("BEGIN OPENSSH PRIVATE KEY"));

        let public_openssh = ValidatorSshKeyManager::get_public_key_openssh(&public_key).unwrap();
        assert!(public_openssh.starts_with("ssh-"));

        // Verify permissions (Unix only)
        #[cfg(unix)]
        {
            use std::os::unix::fs::PermissionsExt;
            let metadata = fs::metadata(&key_path).await.unwrap();
            assert_eq!(metadata.permissions().mode() & 0o777, 0o600);
        }
    }

    #[tokio::test]
    async fn test_key_cleanup() {
        let temp_dir = TempDir::new().unwrap();
        let key_manager = ValidatorSshKeyManager::new(temp_dir.path().to_path_buf())
            .await
            .unwrap();

        // Generate a key
        let session_id = "test-cleanup-456";
        let (_, _, key_path) = key_manager
            .generate_session_keypair(session_id)
            .await
            .unwrap();

        assert!(key_path.exists());

        // Clean up the key
        key_manager.cleanup_session_keys(session_id).await.unwrap();
        assert!(!key_path.exists());
    }

    #[tokio::test]
    async fn test_expired_key_cleanup() {
        let temp_dir = TempDir::new().unwrap();
        let key_manager = ValidatorSshKeyManager::new(temp_dir.path().to_path_buf())
            .await
            .unwrap();

        // Generate multiple keys
        for i in 0..3 {
            let session_id = format!("test-expire-{i}");
            key_manager
                .generate_session_keypair(&session_id)
                .await
                .unwrap();
        }

        // Sleep briefly to ensure files have some age
        tokio::time::sleep(Duration::from_millis(100)).await;

        // Clean up with very short max age (0 seconds)
        let cleaned = key_manager.cleanup_expired_keys(0).await.unwrap();
        assert_eq!(cleaned, 3);

        // Verify all keys are gone
        let mut entries = fs::read_dir(&key_manager.key_dir).await.unwrap();
        let mut count = 0;
        while let Some(entry) = entries.next_entry().await.unwrap() {
            if entry.file_name().to_str().unwrap().starts_with("session_") {
                count += 1;
            }
        }
        assert_eq!(count, 0);
    }
}
