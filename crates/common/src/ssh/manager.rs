//! Concrete SSH manager implementations

use super::config::SshConfig;
use super::traits::{SshAuditLogger, SshKeyManager, SshUserManager};
use super::types::{
    SshError, SshKeyAlgorithm, SshKeyId, SshKeyInfo, SshKeyParams, SshResult, SshUserInfo,
    SshUsername,
};
use async_trait::async_trait;
use std::collections::HashMap;
use std::path::{Path, PathBuf};
use std::process::Command;
use std::sync::Arc;
use std::time::SystemTime;
use tokio::fs;
use tokio::sync::RwLock;
use tracing::{debug, error, info, warn};

/// Standard SSH key manager implementation
pub struct StandardSshKeyManager {
    config: SshConfig,
    active_keys: Arc<RwLock<HashMap<SshKeyId, SshKeyInfo>>>,
    audit_logger: Option<Arc<dyn SshAuditLogger>>,
}

impl StandardSshKeyManager {
    /// Create new SSH key manager
    pub fn new(
        config: SshConfig,
        audit_logger: Option<Arc<dyn SshAuditLogger>>,
    ) -> SshResult<Self> {
        // Validate configuration
        config.validate().map_err(SshError::InvalidConfiguration)?;

        // Ensure key directory exists
        std::fs::create_dir_all(&config.key_directory).map_err(|e| {
            SshError::KeyGenerationFailed(format!("Failed to create key directory: {e}"))
        })?;

        Ok(Self {
            config,
            active_keys: Arc::new(RwLock::new(HashMap::new())),
            audit_logger,
        })
    }

    /// Generate unique key filename
    fn generate_key_filename(&self, id: &SshKeyId, algorithm: &SshKeyAlgorithm) -> PathBuf {
        PathBuf::from(&self.config.key_directory).join(format!("{id}_{algorithm}"))
    }

    /// Generate SSH key using ssh-keygen
    async fn generate_key_pair(
        &self,
        id: &SshKeyId,
        params: &SshKeyParams,
    ) -> SshResult<(String, String)> {
        let private_key_path = self.generate_key_filename(id, &params.algorithm);
        let public_key_path = private_key_path.with_extension("pub");

        let mut cmd = Command::new("ssh-keygen");
        cmd.args([
            "-t",
            &params.algorithm.to_string(),
            "-b",
            &params.key_size.to_string(),
            "-f",
            private_key_path.to_str().unwrap(),
            "-N",
            params.passphrase.as_deref().unwrap_or(""), // Empty string for no passphrase
            "-C",
            params
                .comment
                .as_deref()
                .unwrap_or(&format!("basilica-{id}")),
        ]);

        let output = cmd.output().map_err(|e| {
            SshError::KeyGenerationFailed(format!("Failed to execute ssh-keygen: {e}"))
        })?;

        if !output.status.success() {
            return Err(SshError::KeyGenerationFailed(format!(
                "ssh-keygen failed: {}",
                String::from_utf8_lossy(&output.stderr)
            )));
        }

        // Read the generated keys
        let public_key = fs::read_to_string(&public_key_path)
            .await
            .map_err(|e| SshError::KeyGenerationFailed(format!("Failed to read public key: {e}")))?
            .trim()
            .to_string();

        let private_key_path_str = private_key_path.to_string_lossy().to_string();

        Ok((public_key, private_key_path_str))
    }

    /// Generate SSH key fingerprint
    async fn generate_fingerprint(&self, public_key_path: &Path) -> SshResult<String> {
        let output = Command::new("ssh-keygen")
            .args(["-lf", public_key_path.to_str().unwrap()])
            .output()
            .map_err(|e| {
                SshError::KeyGenerationFailed(format!("Failed to generate fingerprint: {e}"))
            })?;

        if !output.status.success() {
            return Err(SshError::KeyGenerationFailed(format!(
                "ssh-keygen fingerprint failed: {}",
                String::from_utf8_lossy(&output.stderr)
            )));
        }

        let fingerprint_line = String::from_utf8_lossy(&output.stdout);
        // Extract fingerprint from output like "2048 SHA256:... comment"
        if let Some(start) = fingerprint_line.find("SHA256:") {
            if let Some(end) = fingerprint_line[start..].find(' ') {
                return Ok(fingerprint_line[start..start + end].to_string());
            }
        }

        Err(SshError::KeyGenerationFailed(
            "Failed to parse key fingerprint".to_string(),
        ))
    }
}

#[async_trait]
impl SshKeyManager for StandardSshKeyManager {
    async fn generate_key(&self, id: &SshKeyId, params: SshKeyParams) -> SshResult<SshKeyInfo> {
        info!("Generating SSH key for ID: {}", id);

        // Check if key already exists
        {
            let active_keys = self.active_keys.read().await;
            if let Some(existing_key) = active_keys.get(id) {
                if !existing_key.is_expired(SystemTime::now()) {
                    warn!("SSH key already exists for ID: {}", id);
                    return Ok(existing_key.clone());
                }
            }
        }

        let now = SystemTime::now();
        let expires_at = now + params.validity_duration;

        // Generate key pair
        let (public_key, private_key_path) = self.generate_key_pair(id, &params).await?;

        // Generate fingerprint
        let fingerprint = self
            .generate_fingerprint(&PathBuf::from(&private_key_path).with_extension("pub"))
            .await?;

        let key_info = SshKeyInfo {
            id: id.clone(),
            public_key,
            private_key_path,
            username: format!("{}_{}", self.config.username_prefix, id),
            fingerprint,
            created_at: now,
            expires_at,
            algorithm: params.algorithm,
            key_size: params.key_size,
        };

        // Store in active keys
        {
            let mut active_keys = self.active_keys.write().await;
            active_keys.insert(id.clone(), key_info.clone());
        }

        // Log key generation
        if let Some(audit_logger) = &self.audit_logger {
            if let Err(e) = audit_logger.log_key_generated(id, &key_info.username).await {
                warn!("Failed to log key generation: {}", e);
            }
        }

        info!("SSH key generated successfully for ID: {}", id);
        Ok(key_info)
    }

    async fn install_key(&self, key_info: &SshKeyInfo, username: &SshUsername) -> SshResult<()> {
        debug!("Installing SSH key for user: {}", username);

        let ssh_dir = format!("/home/{username}/.ssh");
        let authorized_keys_path = format!("{ssh_dir}/authorized_keys");

        // Create .ssh directory
        fs::create_dir_all(&ssh_dir).await.map_err(|e| {
            SshError::KeyInstallationFailed(format!("Failed to create .ssh directory: {e}"))
        })?;

        // Append public key to authorized_keys
        let key_entry = format!(
            "# Basilica SSH Key - {} - {}\n{}\n",
            key_info.id,
            key_info
                .created_at
                .duration_since(SystemTime::UNIX_EPOCH)
                .unwrap()
                .as_secs(),
            key_info.public_key
        );

        let existing_content = fs::read_to_string(&authorized_keys_path)
            .await
            .unwrap_or_default();
        let new_content = format!("{existing_content}{key_entry}");

        fs::write(&authorized_keys_path, new_content)
            .await
            .map_err(|e| {
                SshError::KeyInstallationFailed(format!("Failed to write authorized_keys: {e}"))
            })?;

        // Set correct permissions
        self.set_ssh_permissions(&ssh_dir, &authorized_keys_path)
            .await?;

        debug!("SSH key installed successfully for user: {}", username);
        Ok(())
    }

    async fn revoke_key(&self, id: &SshKeyId) -> SshResult<()> {
        info!("Revoking SSH key for ID: {}", id);

        let key_info = {
            let mut active_keys = self.active_keys.write().await;
            active_keys.remove(id)
        };

        if let Some(key_info) = key_info {
            // Remove public key from authorized_keys
            self.remove_key_from_authorized_keys(&key_info).await?;

            // Remove private key file
            if let Err(e) = fs::remove_file(&key_info.private_key_path).await {
                warn!("Failed to remove private key file: {}", e);
            }

            // Remove public key file
            let public_key_path = PathBuf::from(&key_info.private_key_path).with_extension("pub");
            if let Err(e) = fs::remove_file(&public_key_path).await {
                warn!("Failed to remove public key file: {}", e);
            }

            // Log key revocation
            if let Some(audit_logger) = &self.audit_logger {
                if let Err(e) = audit_logger.log_key_revoked(id, &key_info.username).await {
                    warn!("Failed to log key revocation: {}", e);
                }
            }

            info!("SSH key revoked successfully for ID: {}", id);
        } else {
            warn!("No SSH key found for ID: {}", id);
        }

        Ok(())
    }

    async fn get_key(&self, id: &SshKeyId) -> SshResult<Option<SshKeyInfo>> {
        let active_keys = self.active_keys.read().await;
        Ok(active_keys.get(id).cloned())
    }

    async fn list_keys(&self) -> SshResult<Vec<SshKeyInfo>> {
        let active_keys = self.active_keys.read().await;
        Ok(active_keys.values().cloned().collect())
    }

    async fn key_exists(&self, id: &SshKeyId) -> SshResult<bool> {
        let active_keys = self.active_keys.read().await;
        Ok(active_keys.contains_key(id))
    }

    async fn cleanup_expired_keys(&self) -> SshResult<u32> {
        let mut cleaned = 0;
        let now = SystemTime::now();

        let expired_keys: Vec<SshKeyId> = {
            let active_keys = self.active_keys.read().await;
            active_keys
                .iter()
                .filter(|(_, key_info)| key_info.is_expired(now))
                .map(|(key_id, _)| key_id.clone())
                .collect()
        };

        for key_id in expired_keys {
            if let Err(e) = self.revoke_key(&key_id).await {
                error!("Failed to revoke expired key {}: {}", key_id, e);
            } else {
                cleaned += 1;
            }
        }

        if cleaned > 0 {
            info!("Cleaned up {} expired SSH keys", cleaned);
        }

        Ok(cleaned)
    }

    async fn authorize_key(
        &self,
        id: &SshKeyId,
        public_key: &str,
        duration: std::time::Duration,
    ) -> SshResult<SshKeyInfo> {
        info!("Authorizing provided SSH public key for ID: {}", id);

        // Check if key already exists
        {
            let active_keys = self.active_keys.read().await;
            if let Some(existing_key) = active_keys.get(id) {
                if !existing_key.is_expired(SystemTime::now()) {
                    warn!("SSH key already exists for ID: {}", id);
                    return Ok(existing_key.clone());
                }
            }
        }

        let now = SystemTime::now();
        let expires_at = now + duration;

        // Parse the public key to extract algorithm and key size
        let (algorithm, key_size) = self.parse_public_key_info(public_key)?;

        // Generate fingerprint from the public key
        let fingerprint = self.generate_fingerprint_from_public_key(public_key)?;

        let key_info = SshKeyInfo {
            id: id.clone(),
            public_key: public_key.to_string(),
            private_key_path: "".to_string(), // No private key since we're only authorizing the public key
            username: format!("{}_{}", self.config.username_prefix, id),
            fingerprint,
            created_at: now,
            expires_at,
            algorithm,
            key_size,
        };

        // Store in active keys
        {
            let mut active_keys = self.active_keys.write().await;
            active_keys.insert(id.clone(), key_info.clone());
        }

        // Log key authorization
        if let Some(audit_logger) = &self.audit_logger {
            if let Err(e) = audit_logger.log_key_generated(id, &key_info.username).await {
                warn!("Failed to log key authorization: {}", e);
            }
        }

        info!("SSH public key authorized successfully for ID: {}", id);
        Ok(key_info)
    }
}

impl StandardSshKeyManager {
    /// Set SSH directory and file permissions
    async fn set_ssh_permissions(
        &self,
        ssh_dir: &str,
        authorized_keys_path: &str,
    ) -> SshResult<()> {
        // Set .ssh directory permissions (700)
        Command::new("chmod")
            .args(["700", ssh_dir])
            .output()
            .map_err(|e| {
                SshError::PermissionFailed(format!("Failed to set directory permissions: {e}"))
            })?;

        // Set authorized_keys permissions (600)
        Command::new("chmod")
            .args(["600", authorized_keys_path])
            .output()
            .map_err(|e| {
                SshError::PermissionFailed(format!("Failed to set file permissions: {e}"))
            })?;

        Ok(())
    }

    /// Remove key from authorized_keys file
    async fn remove_key_from_authorized_keys(&self, key_info: &SshKeyInfo) -> SshResult<()> {
        let authorized_keys_path = format!("/home/{}/.ssh/authorized_keys", key_info.username);

        if !Path::new(&authorized_keys_path).exists() {
            return Ok(());
        }

        // Read current authorized_keys
        let content = fs::read_to_string(&authorized_keys_path)
            .await
            .map_err(|e| {
                SshError::KeyRevocationFailed(format!("Failed to read authorized_keys: {e}"))
            })?;

        // Filter out the key
        let lines: Vec<&str> = content
            .lines()
            .filter(|line| !line.contains(&key_info.public_key) && !line.contains(&key_info.id))
            .collect();

        // Write back filtered content
        fs::write(&authorized_keys_path, lines.join("\n"))
            .await
            .map_err(|e| {
                SshError::KeyRevocationFailed(format!("Failed to update authorized_keys: {e}"))
            })?;

        Ok(())
    }

    /// Parse SSH public key to extract algorithm and key size information
    fn parse_public_key_info(&self, public_key: &str) -> SshResult<(SshKeyAlgorithm, u32)> {
        let parts: Vec<&str> = public_key.split_whitespace().collect();
        if parts.is_empty() {
            return Err(SshError::InvalidConfiguration(
                "Invalid public key format".to_string(),
            ));
        }

        let algorithm = match parts[0] {
            "ssh-rsa" => SshKeyAlgorithm::Rsa,
            "ssh-ed25519" => SshKeyAlgorithm::Ed25519,
            "ecdsa-sha2-nistp256" | "ecdsa-sha2-nistp384" | "ecdsa-sha2-nistp521" => {
                SshKeyAlgorithm::Ecdsa
            }
            _ => {
                return Err(SshError::InvalidConfiguration(format!(
                    "Unsupported key algorithm: {}",
                    parts[0]
                )))
            }
        };

        // Estimate key size based on algorithm
        let key_size = match algorithm {
            SshKeyAlgorithm::Rsa => 2048, // Default RSA size, could be 1024, 2048, 4096
            SshKeyAlgorithm::Ed25519 => 256, // Ed25519 is always 256 bits
            SshKeyAlgorithm::Ecdsa => 256, // Default ECDSA size
        };

        Ok((algorithm, key_size))
    }

    /// Generate fingerprint from SSH public key
    fn generate_fingerprint_from_public_key(&self, public_key: &str) -> SshResult<String> {
        // For simplicity, we'll create a basic fingerprint
        // In production, you'd want to use proper SSH key fingerprinting
        use std::collections::hash_map::DefaultHasher;
        use std::hash::{Hash, Hasher};

        let mut hasher = DefaultHasher::new();
        public_key.hash(&mut hasher);
        let hash = hasher.finish();

        // Format as a traditional SSH fingerprint (simplified)
        Ok(format!("SHA256:{hash:016x}"))
    }
}

/// Standard SSH user manager implementation
pub struct StandardSshUserManager {
    #[allow(dead_code)]
    config: SshConfig,
    active_users: Arc<RwLock<HashMap<SshUsername, SshUserInfo>>>,
    audit_logger: Option<Arc<dyn SshAuditLogger>>,
}

impl StandardSshUserManager {
    /// Create new SSH user manager
    pub fn new(
        config: SshConfig,
        audit_logger: Option<Arc<dyn SshAuditLogger>>,
    ) -> SshResult<Self> {
        Ok(Self {
            config,
            active_users: Arc::new(RwLock::new(HashMap::new())),
            audit_logger,
        })
    }

    /// Get current shell for a user
    async fn get_user_shell(&self, username: &SshUsername) -> SshResult<String> {
        let output = Command::new("getent")
            .args(["passwd", username])
            .output()
            .map_err(|e| SshError::UserCreationFailed(format!("Failed to get user info: {e}")))?;

        if !output.status.success() {
            return Err(SshError::UserCreationFailed(format!(
                "User {username} not found"
            )));
        }

        let passwd_line = String::from_utf8_lossy(&output.stdout);
        let fields: Vec<&str> = passwd_line.trim().split(':').collect();

        if fields.len() >= 7 {
            Ok(fields[6].to_string())
        } else {
            Err(SshError::UserCreationFailed(
                "Invalid passwd entry format".to_string(),
            ))
        }
    }
}

#[async_trait]
impl SshUserManager for StandardSshUserManager {
    async fn create_user(&self, user_info: &SshUserInfo) -> SshResult<()> {
        info!("Creating SSH user: {}", user_info.username);

        // Check if user already exists
        let output = Command::new("id")
            .arg(&user_info.username)
            .output()
            .map_err(|e| {
                SshError::UserCreationFailed(format!("Failed to check user existence: {e}"))
            })?;

        if output.status.success() {
            debug!("User {} already exists", user_info.username);
            return Ok(());
        }

        // Create user with specified shell and home directory
        let mut cmd = Command::new("useradd");
        cmd.args([
            "-m", // Create home directory
            "-s",
            &user_info.shell,
            "-d",
            &user_info.home_directory,
        ]);

        // Add groups if specified
        if !user_info.groups.is_empty() {
            cmd.args(["-G", &user_info.groups.join(",")]);
        }

        cmd.arg(&user_info.username);

        let output = cmd
            .output()
            .map_err(|e| SshError::UserCreationFailed(format!("Failed to execute useradd: {e}")))?;

        if !output.status.success() {
            return Err(SshError::UserCreationFailed(format!(
                "useradd failed: {}",
                String::from_utf8_lossy(&output.stderr)
            )));
        }

        // Store user info
        {
            let mut active_users = self.active_users.write().await;
            active_users.insert(user_info.username.clone(), user_info.clone());
        }

        // Log user creation
        if let Some(audit_logger) = &self.audit_logger {
            if let Err(e) = audit_logger.log_user_created(&user_info.username).await {
                warn!("Failed to log user creation: {}", e);
            }
        }

        info!("SSH user created successfully: {}", user_info.username);
        Ok(())
    }

    async fn remove_user(&self, username: &SshUsername) -> SshResult<()> {
        info!("Removing SSH user: {}", username);

        let output = Command::new("userdel")
            .args(["-r", username]) // Remove home directory
            .output()
            .map_err(|e| SshError::UserRemovalFailed(format!("Failed to execute userdel: {e}")))?;

        if !output.status.success() {
            warn!(
                "Failed to remove user {}: {}",
                username,
                String::from_utf8_lossy(&output.stderr)
            );
        }

        // Remove from active users
        {
            let mut active_users = self.active_users.write().await;
            active_users.remove(username);
        }

        // Log user removal
        if let Some(audit_logger) = &self.audit_logger {
            if let Err(e) = audit_logger.log_user_removed(username).await {
                warn!("Failed to log user removal: {}", e);
            }
        }

        info!("SSH user removed successfully: {}", username);
        Ok(())
    }

    async fn get_user(&self, username: &SshUsername) -> SshResult<Option<SshUserInfo>> {
        let active_users = self.active_users.read().await;
        Ok(active_users.get(username).cloned())
    }

    async fn user_exists(&self, username: &SshUsername) -> SshResult<bool> {
        let output = Command::new("id").arg(username).output().map_err(|e| {
            SshError::UserCreationFailed(format!("Failed to check user existence: {e}"))
        })?;

        Ok(output.status.success())
    }

    async fn list_users(&self) -> SshResult<Vec<SshUserInfo>> {
        let active_users = self.active_users.read().await;
        Ok(active_users.values().cloned().collect())
    }

    async fn update_user(&self, username: &SshUsername, user_info: &SshUserInfo) -> SshResult<()> {
        info!("Updating SSH user: {}", username);

        // Check if user exists
        if !self.user_exists(username).await? {
            return Err(SshError::UserCreationFailed(format!(
                "User {username} does not exist"
            )));
        }

        // Update user groups if specified
        if !user_info.groups.is_empty() {
            let output = Command::new("usermod")
                .args(["-G", &user_info.groups.join(","), username])
                .output()
                .map_err(|e| {
                    SshError::UserCreationFailed(format!("Failed to execute usermod: {e}"))
                })?;

            if !output.status.success() {
                warn!(
                    "Failed to update groups for user {}: {}",
                    username,
                    String::from_utf8_lossy(&output.stderr)
                );
            }
        }

        // Update shell if different
        let current_shell = self.get_user_shell(username).await?;
        if current_shell != user_info.shell {
            let output = Command::new("usermod")
                .args(["-s", &user_info.shell, username])
                .output()
                .map_err(|e| {
                    SshError::UserCreationFailed(format!("Failed to execute usermod: {e}"))
                })?;

            if !output.status.success() {
                warn!(
                    "Failed to update shell for user {}: {}",
                    username,
                    String::from_utf8_lossy(&output.stderr)
                );
            }
        }

        // Update stored user info
        {
            let mut active_users = self.active_users.write().await;
            active_users.insert(username.clone(), user_info.clone());
        }

        // Log user update
        if let Some(audit_logger) = &self.audit_logger {
            if let Err(e) = audit_logger.log_user_created(username).await {
                warn!("Failed to log user update: {}", e);
            }
        }

        info!("SSH user updated successfully: {}", username);
        Ok(())
    }

    async fn cleanup_expired_users(&self) -> SshResult<u32> {
        let mut cleaned = 0;
        let now = SystemTime::now();

        let expired_users: Vec<SshUsername> = {
            let active_users = self.active_users.read().await;
            active_users
                .iter()
                .filter(|(_, user_info)| user_info.is_expired(now))
                .map(|(username, _)| username.clone())
                .collect()
        };

        for username in expired_users {
            if let Err(e) = self.remove_user(&username).await {
                error!("Failed to remove expired user {}: {}", username, e);
            } else {
                cleaned += 1;
            }
        }

        if cleaned > 0 {
            info!("Cleaned up {} expired SSH users", cleaned);
        }

        Ok(cleaned)
    }
}

/// Default SSH service implementation combining all managers
pub struct DefaultSshService {
    key_manager: StandardSshKeyManager,
    user_manager: StandardSshUserManager,
}

impl DefaultSshService {
    pub fn new(config: impl Into<SshConfig>) -> Result<Self, SshError> {
        let ssh_config = config.into();

        Ok(Self {
            key_manager: StandardSshKeyManager::new(ssh_config.clone(), None)?,
            user_manager: StandardSshUserManager::new(ssh_config, None)?,
        })
    }

    /// Check if SSH daemon is running
    async fn check_ssh_daemon_status(&self) -> bool {
        // Check if sshd service is active
        if let Ok(output) = Command::new("systemctl")
            .args(["is-active", "ssh"])
            .output()
        {
            if output.status.success() {
                return true;
            }
        }

        // Fallback: check if sshd service is active (alternative name)
        match Command::new("systemctl")
            .args(["is-active", "sshd"])
            .output()
        {
            Ok(output) => output.status.success(),
            Err(_) => {
                // Fallback: check if SSH port is listening
                self.check_ssh_port_listening().await
            }
        }
    }

    /// Check if SSH port is listening
    async fn check_ssh_port_listening(&self) -> bool {
        if let Ok(output) = Command::new("ss")
            .args(["-tln", "sport", "=", ":22"])
            .output()
        {
            if output.status.success() {
                let output_str = String::from_utf8_lossy(&output.stdout);
                return output_str.contains(":22");
            }
        }

        // Alternative check using netstat
        if let Ok(output) = Command::new("netstat").args(["-tln"]).output() {
            if output.status.success() {
                let output_str = String::from_utf8_lossy(&output.stdout);
                return output_str.contains(":22");
            }
        }

        false
    }

    /// Check if key directory is accessible
    async fn check_key_directory_access(&self) -> bool {
        use std::fs;

        // Try to access the key directory
        match fs::metadata(&self.key_manager.config.key_directory) {
            Ok(metadata) => {
                // Check if it's a directory and readable
                if metadata.is_dir() {
                    // Try to read the directory contents
                    match fs::read_dir(&self.key_manager.config.key_directory) {
                        Ok(_) => true,
                        Err(e) => {
                            debug!("Key directory not readable: {}", e);
                            false
                        }
                    }
                } else {
                    debug!("Key directory path is not a directory");
                    false
                }
            }
            Err(e) => {
                debug!("Key directory not accessible: {}", e);
                false
            }
        }
    }

    /// Check if user management commands are available
    async fn check_user_management_availability(&self) -> bool {
        // Check if useradd command is available
        let useradd_available = match Command::new("which").arg("useradd").output() {
            Ok(output) => output.status.success(),
            Err(_) => false,
        };

        // Check if userdel command is available
        let userdel_available = match Command::new("which").arg("userdel").output() {
            Ok(output) => output.status.success(),
            Err(_) => false,
        };

        // Check if usermod command is available
        let usermod_available = match Command::new("which").arg("usermod").output() {
            Ok(output) => output.status.success(),
            Err(_) => false,
        };

        useradd_available && userdel_available && usermod_available
    }
}

#[async_trait]
impl super::traits::SshService for DefaultSshService {
    async fn provision_access(
        &self,
        id: &SshKeyId,
        username: &SshUsername,
        key_params: super::types::SshKeyParams,
        user_info: &SshUserInfo,
        _access_config: &super::types::SshAccessConfig,
        _duration: std::time::Duration,
    ) -> SshResult<SshKeyInfo> {
        // Create the user first
        self.user_manager.create_user(user_info).await?;

        // Generate SSH key
        let key_info = self.key_manager.generate_key(id, key_params).await?;

        // Install the key for the user
        self.key_manager.install_key(&key_info, username).await?;

        Ok(key_info)
    }

    async fn revoke_access(&self, id: &SshKeyId, username: &SshUsername) -> SshResult<()> {
        // Revoke the SSH key
        self.key_manager.revoke_key(id).await?;

        // Remove the user
        self.user_manager.remove_user(username).await?;

        Ok(())
    }

    async fn get_access_info(
        &self,
        id: &SshKeyId,
    ) -> SshResult<Option<(SshKeyInfo, SshUserInfo, super::types::SshAccessConfig)>> {
        if let Some(key_info) = self.key_manager.get_key(id).await? {
            if let Some(user_info) = self.user_manager.get_user(&key_info.username).await? {
                let access_config = super::types::SshAccessConfig::default();
                return Ok(Some((key_info, user_info, access_config)));
            }
        }
        Ok(None)
    }

    async fn list_access(&self) -> SshResult<Vec<(SshKeyInfo, SshUserInfo)>> {
        let keys = self.key_manager.list_keys().await?;
        let mut result = Vec::new();

        for key in keys {
            if let Some(user) = self.user_manager.get_user(&key.username).await? {
                result.push((key, user));
            }
        }

        Ok(result)
    }

    async fn health_check(&self) -> SshResult<super::traits::SshHealthStatus> {
        let keys = self.key_manager.list_keys().await?;
        let users = self.user_manager.list_users().await?;

        let now = std::time::SystemTime::now();
        let expired_keys = keys.iter().filter(|k| k.is_expired(now)).count() as u32;
        let expired_users = users.iter().filter(|u| u.is_expired(now)).count() as u32;

        // Check SSH daemon status
        let ssh_daemon_running = self.check_ssh_daemon_status().await;

        // Check key directory accessibility
        let key_directory_accessible = self.check_key_directory_access().await;

        // Check user management availability
        let user_management_available = self.check_user_management_availability().await;

        Ok(super::traits::SshHealthStatus {
            ssh_daemon_running,
            key_directory_accessible,
            user_management_available,
            total_active_keys: keys.len() as u32,
            total_active_users: users.len() as u32,
            expired_keys,
            expired_users,
        })
    }

    async fn cleanup_expired(&self) -> SshResult<super::traits::SshCleanupStats> {
        let mut errors_encountered = 0;

        // Clean up expired keys and track errors
        let cleaned_keys = match self.key_manager.cleanup_expired_keys().await {
            Ok(count) => count,
            Err(e) => {
                error!("Failed to cleanup expired keys: {}", e);
                errors_encountered += 1;
                0
            }
        };

        // Clean up expired users and track errors
        let cleaned_users = match self.user_manager.cleanup_expired_users().await {
            Ok(count) => count,
            Err(e) => {
                error!("Failed to cleanup expired users: {}", e);
                errors_encountered += 1;
                0
            }
        };

        info!(
            "Cleanup completed: {} keys, {} users, {} errors",
            cleaned_keys, cleaned_users, errors_encountered
        );

        Ok(super::traits::SshCleanupStats {
            cleaned_keys,
            cleaned_users,
            errors_encountered,
        })
    }
}
