//! Simple SSH utilities for basic operations
//!
//! This module provides simplified SSH operations without the complexity
//! of the full trait-based system. Designed for straightforward use cases.

use anyhow::Result;
use std::fs;
use std::os::unix::fs::PermissionsExt;
use std::path::Path;
use std::process::Command;
use tracing::{debug, info, warn};

/// Simple SSH key utilities
pub struct SimpleSshKeys;

impl SimpleSshKeys {
    /// Add SSH public key to user's authorized_keys
    pub async fn add_key(username: &str, public_key: &str, restrictions: &[&str]) -> Result<()> {
        info!("Adding SSH key for user: {}", username);

        let home_dir = format!("/home/{username}");
        let ssh_dir = format!("{home_dir}/.ssh");
        let auth_keys_path = format!("{ssh_dir}/authorized_keys");

        // Create .ssh directory if it doesn't exist
        fs::create_dir_all(&ssh_dir)?;
        Self::set_permissions(&ssh_dir, 0o700)?;

        // Format key entry with restrictions
        let key_entry = if restrictions.is_empty() {
            format!("{}\n", public_key.trim())
        } else {
            format!("{} {}\n", restrictions.join(","), public_key.trim())
        };

        // Write key to authorized_keys
        fs::write(&auth_keys_path, key_entry)?;
        Self::set_permissions(&auth_keys_path, 0o600)?;

        // Set ownership
        Self::set_ownership(&ssh_dir, username)?;
        Self::set_ownership(&auth_keys_path, username)?;

        info!("SSH key added successfully for user: {}", username);
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
        let path_obj = Path::new(path);
        if path_obj.exists() {
            let mut perms = fs::metadata(path_obj)?.permissions();
            perms.set_mode(mode);
            fs::set_permissions(path_obj, perms)?;
        }
        Ok(())
    }

    fn set_ownership(path: &str, username: &str) -> Result<()> {
        let output = Command::new("chown")
            .arg("-R")
            .arg(format!("{username}:{username}"))
            .arg(path)
            .output()?;

        if !output.status.success() {
            let error = String::from_utf8_lossy(&output.stderr);
            warn!("Failed to set ownership for {}: {}", path, error);
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

        if Self::user_exists(username)? {
            debug!("User {} already exists", username);
            return Ok(());
        }

        // Create user with home directory
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
            return Err(anyhow::anyhow!(
                "Failed to create user {}: {}",
                username,
                error
            ));
        }

        info!("Successfully created system user: {}", username);
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
    pub fn validator_username(hotkey: &str) -> String {
        format!("validator_{hotkey}")
    }
}
