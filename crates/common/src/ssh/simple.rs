//! Simple SSH utilities for basic operations
//!
//! This module provides simplified SSH operations without the complexity
//! of the full trait-based system. Designed for straightforward use cases.

use anyhow::Result;
use std::fs::{self, OpenOptions};
use std::io::{Read, Seek, Write};
use std::os::unix::fs::PermissionsExt;
use std::path::Path;
use std::process::Command;
use tracing::{debug, error, info, warn};

/// Simple SSH key utilities
pub struct SimpleSshKeys;

impl SimpleSshKeys {
    /// Add SSH public key to user's authorized_keys (idempotent - only adds if not exists)
    pub async fn add_key_if_missing(
        username: &str,
        public_key: &str,
        restrictions: &[&str],
    ) -> Result<bool> {
        info!("Adding SSH key for user: {} (checking if exists)", username);

        let home_dir = format!("/home/{username}");
        let ssh_dir = format!("{home_dir}/.ssh");
        let auth_keys_path = format!("{ssh_dir}/authorized_keys");

        // Ensure the SSH directory exists with proper ownership from the start
        if !Path::new(&ssh_dir).exists() {
            info!("Creating SSH directory: {}", ssh_dir);
            fs::create_dir_all(&ssh_dir)?;

            // Set ownership immediately after creation to avoid timing issues
            Self::set_ownership(&ssh_dir, username)?;
            Self::set_permissions(&ssh_dir, 0o700)?;
            info!("SSH directory created with proper ownership: {}", ssh_dir);
        }

        // Format key entry with restrictions
        let key_entry = if restrictions.is_empty() {
            format!("{}\n", public_key.trim())
        } else {
            format!("{} {}\n", restrictions.join(","), public_key.trim())
        };

        debug!(
            "Formatted key entry for authorized_keys: {} (length: {} chars)",
            key_entry.trim(),
            key_entry.len()
        );

        // Atomically check and append key with proper file locking
        info!(
            "Atomically checking and adding SSH key to authorized_keys file: {}",
            auth_keys_path
        );
        let key_added =
            Self::append_key_safely_atomic(&auth_keys_path, &key_entry, public_key.trim())?;

        if key_added {
            // Set permissions and ownership for the authorized_keys file
            Self::set_permissions(&auth_keys_path, 0o600)?;
            Self::set_ownership(&auth_keys_path, username)?;

            // Verify the file was created correctly
            if Path::new(&auth_keys_path).exists() {
                info!(
                    "SSH key added successfully for user: {} (file size: {} bytes)",
                    username,
                    fs::metadata(&auth_keys_path).map(|m| m.len()).unwrap_or(0)
                );
            } else {
                return Err(anyhow::anyhow!("authorized_keys file was not created"));
            }
        } else {
            info!("SSH key already exists for user {}, skipping", username);
        }

        Ok(key_added)
    }

    /// Add SSH public key to user's authorized_keys
    pub async fn add_key(username: &str, public_key: &str, restrictions: &[&str]) -> Result<()> {
        info!("Adding SSH key for user: {}", username);

        let home_dir = format!("/home/{username}");
        let ssh_dir = format!("{home_dir}/.ssh");
        let auth_keys_path = format!("{ssh_dir}/authorized_keys");

        // Ensure the SSH directory exists with proper ownership from the start
        if !Path::new(&ssh_dir).exists() {
            info!("Creating SSH directory: {}", ssh_dir);
            fs::create_dir_all(&ssh_dir)?;

            // Set ownership immediately after creation to avoid timing issues
            Self::set_ownership(&ssh_dir, username)?;
            Self::set_permissions(&ssh_dir, 0o700)?;
            info!("SSH directory created with proper ownership: {}", ssh_dir);
        }

        // Format key entry with restrictions
        let key_entry = if restrictions.is_empty() {
            format!("{}\n", public_key.trim())
        } else {
            format!("{} {}\n", restrictions.join(","), public_key.trim())
        };

        debug!(
            "Formatted key entry for authorized_keys: {} (length: {} chars)",
            key_entry.trim(),
            key_entry.len()
        );

        // Safely append key to authorized_keys with proper file locking
        info!(
            "Appending SSH key to authorized_keys file: {}",
            auth_keys_path
        );
        Self::append_key_safely(&auth_keys_path, &key_entry, public_key.trim())?;

        // Set permissions and ownership for the authorized_keys file
        Self::set_permissions(&auth_keys_path, 0o600)?;
        Self::set_ownership(&auth_keys_path, username)?;

        // Verify the file was created correctly
        if Path::new(&auth_keys_path).exists() {
            info!(
                "SSH key added successfully for user: {} (file size: {} bytes)",
                username,
                fs::metadata(&auth_keys_path).map(|m| m.len()).unwrap_or(0)
            );
        } else {
            return Err(anyhow::anyhow!("authorized_keys file was not created"));
        }

        Ok(())
    }

    /// Atomically check for duplicates and append SSH key with file locking
    fn append_key_safely_atomic(
        auth_keys_path: &str,
        key_entry: &str,
        public_key: &str,
    ) -> Result<bool> {
        use std::io::SeekFrom;
        use std::os::unix::io::AsRawFd;

        info!("Processing atomic SSH key addition to: {}", auth_keys_path);

        // Extract the actual public key part (without restrictions) for comparison
        let key_parts: Vec<&str> = public_key.split_whitespace().collect();
        let key_type_and_data = if key_parts.len() >= 2 {
            format!("{} {}", key_parts[0], key_parts[1])
        } else {
            public_key.trim().to_string()
        };

        debug!(
            "Extracted key data for comparison: {} (length: {})",
            key_type_and_data.chars().take(50).collect::<String>() + "...",
            key_type_and_data.len()
        );

        // Open file for read+write+create - this allows us to read existing content and append
        let mut file = match OpenOptions::new()
            .create(true)
            .read(true)
            .write(true)
            .truncate(false)
            .open(auth_keys_path)
        {
            Ok(f) => f,
            Err(e) => {
                error!("Failed to open authorized_keys file: {}", e);
                return Err(e.into());
            }
        };

        // Apply exclusive file lock to prevent any concurrent access
        let fd = file.as_raw_fd();
        let lock_result = unsafe { libc::flock(fd, libc::LOCK_EX) };

        if lock_result != 0 {
            warn!("Failed to acquire file lock for authorized_keys (errno: {}), proceeding without lock", 
                  std::io::Error::last_os_error());
        } else {
            debug!("Successfully acquired exclusive file lock for authorized_keys");
        }

        // Read existing keys to check for duplicates while holding the lock
        let mut existing_keys = String::new();
        if let Err(e) = file.read_to_string(&mut existing_keys) {
            warn!("Failed to read existing authorized_keys file: {}", e);
            existing_keys = String::new();
        } else if !existing_keys.is_empty() {
            info!(
                "Read existing authorized_keys file: {} bytes, {} lines",
                existing_keys.len(),
                existing_keys.lines().count()
            );
        } else {
            info!("authorized_keys file is empty, will add first key");
        }

        // Check if key already exists (compare the actual key data, not restrictions)
        let mut found_duplicate = false;
        for (line_num, line) in existing_keys.lines().enumerate() {
            let line = line.trim();
            if !line.is_empty() {
                // Extract key data from existing line
                // SSH key format: [restrictions] ssh-type key-data [comment]
                // We need to find the ssh-type and key-data parts
                let existing_parts: Vec<&str> = line.split_whitespace().collect();

                // Look for SSH key type (ssh-rsa, ssh-ed25519, etc.)
                for (i, part) in existing_parts.iter().enumerate() {
                    if part.starts_with("ssh-") || part.starts_with("ecdsa-") {
                        // Found the key type, next part should be the key data
                        if i + 1 < existing_parts.len() {
                            let existing_key_data = format!("{} {}", part, existing_parts[i + 1]);
                            if existing_key_data == key_type_and_data {
                                info!("Identical SSH key already exists at line {} in authorized_keys, skipping duplicate", line_num + 1);
                                found_duplicate = true;
                                break;
                            }
                        }
                        break;
                    }
                }
                if found_duplicate {
                    break;
                }
            }
        }

        if found_duplicate {
            // Release lock and return false (no key added)
            if lock_result == 0 {
                unsafe {
                    libc::flock(fd, libc::LOCK_UN);
                }
                debug!("Released file lock for authorized_keys (duplicate found)");
            }
            return Ok(false);
        }

        info!("Adding new SSH key to authorized_keys file");

        // Seek to end of file for appending
        if let Err(e) = file.seek(SeekFrom::End(0)) {
            error!("Failed to seek to end of authorized_keys file: {}", e);
            if lock_result == 0 {
                unsafe {
                    libc::flock(fd, libc::LOCK_UN);
                }
            }
            return Err(e.into());
        }

        // Write the key entry while holding the lock
        match file.write_all(key_entry.as_bytes()) {
            Ok(_) => {
                info!(
                    "Successfully wrote SSH key entry ({} bytes)",
                    key_entry.len()
                );
            }
            Err(e) => {
                error!("Failed to write SSH key to authorized_keys: {}", e);
                if lock_result == 0 {
                    unsafe {
                        libc::flock(fd, libc::LOCK_UN);
                    }
                }
                return Err(e.into());
            }
        }

        if let Err(e) = file.flush() {
            error!("Failed to flush authorized_keys file: {}", e);
            if lock_result == 0 {
                unsafe {
                    libc::flock(fd, libc::LOCK_UN);
                }
            }
            return Err(e.into());
        }

        // Release lock
        if lock_result == 0 {
            unsafe {
                libc::flock(fd, libc::LOCK_UN);
            }
            debug!("Released file lock for authorized_keys (key added)");
        }

        info!("SSH key successfully appended to authorized_keys file");
        Ok(true) // Key was added
    }

    /// Safely append SSH key to authorized_keys file with deduplication
    fn append_key_safely(auth_keys_path: &str, key_entry: &str, public_key: &str) -> Result<()> {
        // Use the atomic version and ignore the return value for backward compatibility
        Self::append_key_safely_atomic(auth_keys_path, key_entry, public_key)?;
        Ok(())
    }

    /// Remove SSH key from user's authorized_keys
    pub async fn remove_key(username: &str) -> Result<()> {
        info!("Removing SSH key for user: {}", username);

        let auth_keys_path = format!("/home/{username}/.ssh/authorized_keys");

        if Path::new(&auth_keys_path).exists() {
            fs::remove_file(&auth_keys_path)?;
            debug!("Removed authorized_keys file for user: {}", username);
        } else {
            debug!("No authorized_keys file found for user: {}", username);
        }

        Ok(())
    }

    /// Check if user has SSH key
    pub async fn has_key(username: &str) -> bool {
        let auth_keys_path = format!("/home/{username}/.ssh/authorized_keys");
        Path::new(&auth_keys_path).exists()
    }

    /// Get default SSH key restrictions
    pub fn get_default_restrictions() -> Vec<&'static str> {
        vec![
            "no-agent-forwarding",
            "no-port-forwarding",
            "no-X11-forwarding",
        ]
    }

    /// Get strict SSH key restrictions
    pub fn get_strict_restrictions() -> Vec<&'static str> {
        vec![
            "no-agent-forwarding",
            "no-port-forwarding",
            "no-X11-forwarding",
            "no-pty",
        ]
    }

    fn set_permissions(path: &str, mode: u32) -> Result<()> {
        info!("Setting permissions of {} to {:o}", path, mode);

        let path_obj = Path::new(path);
        if path_obj.exists() {
            let mut perms = fs::metadata(path_obj)?.permissions();
            perms.set_mode(mode);
            match fs::set_permissions(path_obj, perms) {
                Ok(_) => {
                    info!("Successfully set permissions of {} to {:o}", path, mode);
                }
                Err(e) => {
                    error!("Failed to set permissions for {}: {}", path, e);
                    return Err(e.into());
                }
            }
        } else {
            warn!("Cannot set permissions: path {} does not exist", path);
        }
        Ok(())
    }

    fn set_ownership(path: &str, username: &str) -> Result<()> {
        info!("Setting ownership of {} to {}", path, username);

        let output = Command::new("chown")
            .arg("-R")
            .arg(format!("{username}:{username}"))
            .arg(path)
            .output()?;

        if !output.status.success() {
            let error = String::from_utf8_lossy(&output.stderr);
            let stdout = String::from_utf8_lossy(&output.stdout);
            error!(
                "Failed to set ownership for {}: stderr={}, stdout={}",
                path, error, stdout
            );
            return Err(anyhow::anyhow!(
                "Failed to set ownership for {}: {}",
                path,
                error
            ));
        } else {
            info!("Successfully set ownership of {} to {}", path, username);
        }

        Ok(())
    }
}

/// Simple SSH user utilities
pub struct SimpleSshUsers;

impl SimpleSshUsers {
    /// Create system user for SSH access
    pub async fn create_user(username: &str) -> Result<()> {
        info!("Creating system user: {}", username);

        // Check if user exists first
        if Self::user_exists(username)? {
            info!("User {} already exists, skipping creation", username);
            return Ok(());
        }

        // Create user with home directory
        info!("Executing useradd command for user: {}", username);
        let output = Command::new("useradd")
            .arg("-m") // Create home directory
            .arg("-s")
            .arg("/bin/bash") // Set shell
            .arg("-G")
            .arg("docker") // Add to docker group
            .arg("-c")
            .arg("Validator User") // Comment
            .arg(username)
            .output()?;

        if !output.status.success() {
            let error = String::from_utf8_lossy(&output.stderr);
            let stdout = String::from_utf8_lossy(&output.stdout);

            // Check if the error is because the user already exists (race condition)
            if error.contains("already exists") {
                info!(
                    "User {} was created by another process, continuing",
                    username
                );
                return Ok(());
            }

            error!(
                "useradd command failed for user {}: stderr={}, stdout={}",
                username, error, stdout
            );
            return Err(anyhow::anyhow!(
                "Failed to create user {}: {}",
                username,
                error
            ));
        }

        info!("Successfully created system user: {}", username);

        // Verify the user was created and has a home directory
        let home_dir = format!("/home/{username}");
        if !Path::new(&home_dir).exists() {
            warn!(
                "Home directory {} was not created for user {}",
                home_dir, username
            );
        } else {
            info!("Verified home directory exists: {}", home_dir);
        }

        Ok(())
    }

    /// Remove system user
    pub async fn remove_user(username: &str) -> Result<()> {
        info!("Removing system user: {}", username);

        if !Self::user_exists(username)? {
            debug!("User {} does not exist", username);
            return Ok(());
        }

        let output = Command::new("userdel")
            .arg("-r") // Remove home directory
            .arg(username)
            .output()?;

        if !output.status.success() {
            let error = String::from_utf8_lossy(&output.stderr);
            warn!("Failed to remove user {}: {}", username, error);
        } else {
            info!("Successfully removed system user: {}", username);
        }

        Ok(())
    }

    /// Check if system user exists
    pub fn user_exists(username: &str) -> Result<bool> {
        let output = Command::new("id").arg(username).output()?;
        Ok(output.status.success())
    }

    /// Generate username for validator
    /// Creates a Linux-compliant username by hashing the hotkey to ensure:
    /// - Maximum 32 characters (Linux limit)
    /// - Only alphanumeric characters and underscores
    /// - Starts with a letter
    pub fn validator_username(hotkey: &str) -> String {
        use std::collections::hash_map::DefaultHasher;
        use std::hash::{Hash, Hasher};

        // Create a hash of the hotkey
        let mut hasher = DefaultHasher::new();
        hotkey.hash(&mut hasher);
        let hash = hasher.finish();

        // Convert to hex and take first 24 characters to ensure total length < 32
        let hash_hex = format!("{hash:x}");
        let truncated_hash = &hash_hex[..hash_hex.len().min(24)];

        // Ensure username starts with letter, contains only valid chars, and is under 32 chars
        format!("val_{truncated_hash}")
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_validator_username_generation() {
        let hotkey = "5FZbLC8tt9BsYXVyK3aq2aFHeQhfQexNRyQy83oatY7vkvn8";
        let username = SimpleSshUsers::validator_username(hotkey);

        // Check username constraints
        assert!(
            username.len() <= 32,
            "Username too long: {}",
            username.len()
        );
        assert!(
            username.starts_with("val_"),
            "Username should start with 'val_'"
        );
        assert!(
            username
                .chars()
                .all(|c| c.is_ascii_alphanumeric() || c == '_'),
            "Username contains invalid characters: {username}"
        );

        println!("Generated username: {username}");
    }

    #[test]
    fn test_validator_username_consistency() {
        let hotkey = "5FZbLC8tt9BsYXVyK3aq2aFHeQhfQexNRyQy83oatY7vkvn8";
        let username1 = SimpleSshUsers::validator_username(hotkey);
        let username2 = SimpleSshUsers::validator_username(hotkey);

        assert_eq!(
            username1, username2,
            "Username generation should be deterministic"
        );
    }

    #[test]
    fn test_validator_username_different_hotkeys() {
        let hotkey1 = "5FZbLC8tt9BsYXVyK3aq2aFHeQhfQexNRyQy83oatY7vkvn8";
        let hotkey2 = "5FHneW46xGXgs5mUiveU4sbTyGBzmstUspZC92UhjJM694ty";

        let username1 = SimpleSshUsers::validator_username(hotkey1);
        let username2 = SimpleSshUsers::validator_username(hotkey2);

        assert_ne!(
            username1, username2,
            "Different hotkeys should generate different usernames"
        );
    }
}
