//! SSH Key Manager Builder for Guaranteed Initialization
//!
//! Provides builder pattern for ValidatorSshKeyManager with comprehensive validation
//! and guaranteed initialization for production environments.

use super::key_manager::ValidatorSshKeyManager;
use crate::config::SshSessionConfig;
use anyhow::{Context, Result};
use common::identity::Hotkey;
use std::sync::Arc;
use tokio::fs;
use tracing::info;

/// Builder for ValidatorSshKeyManager with guaranteed initialization
pub struct ValidatorSshKeyManagerBuilder {
    config: SshSessionConfig,
    validator_hotkey: Hotkey,
}

impl ValidatorSshKeyManagerBuilder {
    /// Create a new builder instance
    pub fn new(config: SshSessionConfig, validator_hotkey: Hotkey) -> Self {
        Self {
            config,
            validator_hotkey,
        }
    }

    /// Build and validate SSH key manager with guaranteed initialization
    pub async fn build(&self) -> Result<Arc<ValidatorSshKeyManager>> {
        info!(
            "Building SSH key manager for validator {}",
            self.validator_hotkey
                .to_string()
                .chars()
                .take(8)
                .collect::<String>()
                + "..."
        );

        // Phase 1: Pre-build validation
        self.validate_build_prerequisites()
            .await
            .context("Pre-build validation failed")?;

        // Phase 2: Ensure SSH directory exists with proper permissions
        self.ensure_ssh_directory_exists()
            .await
            .context("Failed to ensure SSH directory exists")?;

        // Phase 3: Initialize key manager
        let key_manager = self
            .initialize_key_manager()
            .await
            .context("Failed to initialize SSH key manager")?;

        // Phase 4: Validate key manager functionality
        self.validate_key_manager_functionality(&key_manager)
            .await
            .context("Key manager functionality validation failed")?;

        // Phase 5: Test key generation and cleanup
        self.test_key_lifecycle(&key_manager)
            .await
            .context("Key lifecycle test failed")?;

        info!(
            "SSH key manager successfully built and validated for validator {}",
            self.validator_hotkey
                .to_string()
                .chars()
                .take(8)
                .collect::<String>()
                + "..."
        );

        Ok(Arc::new(key_manager))
    }

    /// Validate prerequisites for building SSH key manager
    async fn validate_build_prerequisites(&self) -> Result<()> {
        info!("Validating SSH key manager build prerequisites");

        // Validate SSH key directory path
        if self.config.ssh_key_directory.as_os_str().is_empty() {
            return Err(anyhow::anyhow!("SSH key directory path cannot be empty"));
        }

        // Validate SSH key algorithm
        if self.config.key_algorithm.is_empty() {
            return Err(anyhow::anyhow!("SSH key algorithm cannot be empty"));
        }

        let supported_algorithms = ["ed25519", "rsa", "ecdsa"];
        if !supported_algorithms.contains(&self.config.key_algorithm.as_str()) {
            return Err(anyhow::anyhow!(
                "Unsupported SSH key algorithm: {}. Supported: {:?}",
                self.config.key_algorithm,
                supported_algorithms
            ));
        }

        // Validate session duration constraints
        if self.config.default_session_duration == 0 {
            return Err(anyhow::anyhow!("Default session duration cannot be zero"));
        }

        if self.config.max_session_duration < self.config.default_session_duration {
            return Err(anyhow::anyhow!(
                "Maximum session duration ({}) cannot be less than default session duration ({})",
                self.config.max_session_duration,
                self.config.default_session_duration
            ));
        }

        // Validate validator hotkey format
        if self.validator_hotkey.to_string().is_empty() {
            return Err(anyhow::anyhow!("Validator hotkey cannot be empty"));
        }

        info!("SSH key manager build prerequisites validated successfully");
        Ok(())
    }

    /// Ensure SSH directory exists with proper permissions
    async fn ensure_ssh_directory_exists(&self) -> Result<()> {
        let ssh_dir = &self.config.ssh_key_directory;
        info!("Ensuring SSH directory exists: {}", ssh_dir.display());

        // Create parent directories if they don't exist
        if let Some(parent) = ssh_dir.parent() {
            if !parent.exists() {
                info!("Creating parent directories: {}", parent.display());
                fs::create_dir_all(parent)
                    .await
                    .context("Failed to create parent directories")?;
            }
        }

        // Create SSH key directory
        fs::create_dir_all(ssh_dir)
            .await
            .context("Failed to create SSH key directory")?;

        // Set proper permissions on Unix systems
        #[cfg(unix)]
        {
            use std::os::unix::fs::PermissionsExt;
            let metadata = fs::metadata(ssh_dir)
                .await
                .context("Failed to get SSH directory metadata")?;
            let mut perms = metadata.permissions();
            perms.set_mode(0o700); // rwx------
            fs::set_permissions(ssh_dir, perms)
                .await
                .context("Failed to set SSH directory permissions")?;

            info!("Set SSH directory permissions to 0700");
        }

        // Verify directory is writable
        let test_file = ssh_dir.join(".write_test");
        fs::write(&test_file, b"test")
            .await
            .context("SSH directory is not writable")?;
        fs::remove_file(&test_file)
            .await
            .context("Failed to clean up write test file")?;

        info!("SSH directory validated and ready: {}", ssh_dir.display());
        Ok(())
    }

    /// Initialize the SSH key manager
    async fn initialize_key_manager(&self) -> Result<ValidatorSshKeyManager> {
        info!("Initializing SSH key manager");

        let key_manager = ValidatorSshKeyManager::new(self.config.ssh_key_directory.clone())
            .await
            .context("Failed to create ValidatorSshKeyManager")?;

        info!("SSH key manager initialized successfully");
        Ok(key_manager)
    }

    /// Validate key manager functionality
    async fn validate_key_manager_functionality(
        &self,
        key_manager: &ValidatorSshKeyManager,
    ) -> Result<()> {
        info!("Validating SSH key manager functionality");

        // Test 1: Verify key directory accessibility
        let key_dir = &self.config.ssh_key_directory;
        if !key_dir.exists() {
            return Err(anyhow::anyhow!(
                "Key directory does not exist after initialization"
            ));
        }

        if !key_dir.is_dir() {
            return Err(anyhow::anyhow!("Key path is not a directory"));
        }

        // Test 2: Verify directory permissions on Unix
        #[cfg(unix)]
        {
            use std::os::unix::fs::PermissionsExt;
            let metadata = fs::metadata(key_dir)
                .await
                .context("Failed to read key directory metadata")?;
            let mode = metadata.permissions().mode() & 0o777;
            if mode != 0o700 {
                return Err(anyhow::anyhow!(
                    "Key directory has incorrect permissions: {:o}, expected 0700",
                    mode
                ));
            }
        }

        // Test 3: Test session key path generation
        let test_session_id = "validation-test-session";
        let key_path = key_manager.get_session_key_path(test_session_id);

        if !key_path.starts_with(key_dir) {
            return Err(anyhow::anyhow!(
                "Generated key path is outside key directory"
            ));
        }

        if !key_path
            .file_name()
            .unwrap()
            .to_str()
            .unwrap()
            .contains(test_session_id)
        {
            return Err(anyhow::anyhow!(
                "Generated key path does not contain session ID"
            ));
        }

        info!("SSH key manager functionality validation completed successfully");
        Ok(())
    }

    /// Test complete key lifecycle (generation, usage, cleanup)
    async fn test_key_lifecycle(&self, key_manager: &ValidatorSshKeyManager) -> Result<()> {
        info!("Testing SSH key lifecycle");

        let test_session_id = format!("lifecycle-test-{}", chrono::Utc::now().timestamp());

        // Test 1: Key generation
        let (private_key, public_key, key_path) = key_manager
            .generate_session_keypair(&test_session_id)
            .await
            .context("Failed to generate test session keypair")?;

        // Verify key file exists
        if !key_path.exists() {
            return Err(anyhow::anyhow!("Generated key file does not exist"));
        }

        // Verify key file permissions on Unix
        #[cfg(unix)]
        {
            use std::os::unix::fs::PermissionsExt;
            let metadata = fs::metadata(&key_path)
                .await
                .context("Failed to read key file metadata")?;
            let mode = metadata.permissions().mode() & 0o777;
            if mode != 0o600 {
                return Err(anyhow::anyhow!(
                    "Key file has incorrect permissions: {:o}, expected 0600",
                    mode
                ));
            }
        }

        // Test 2: Key format validation
        let private_openssh = private_key
            .to_openssh(ssh_key::LineEnding::default())
            .context("Failed to convert private key to OpenSSH format")?;

        if !private_openssh
            .as_str()
            .contains("BEGIN OPENSSH PRIVATE KEY")
        {
            return Err(anyhow::anyhow!(
                "Private key is not in valid OpenSSH format"
            ));
        }

        let public_openssh = ValidatorSshKeyManager::get_public_key_openssh(&public_key)
            .context("Failed to convert public key to OpenSSH format")?;

        if !public_openssh.starts_with("ssh-") {
            return Err(anyhow::anyhow!("Public key is not in valid OpenSSH format"));
        }

        // Test 3: Key cleanup
        key_manager
            .cleanup_session_keys(&test_session_id)
            .await
            .context("Failed to cleanup test session keys")?;

        // Verify key file is removed
        if key_path.exists() {
            return Err(anyhow::anyhow!("Key file still exists after cleanup"));
        }

        info!("SSH key lifecycle test completed successfully");
        Ok(())
    }

    /// Get configuration summary for logging
    pub fn get_config_summary(&self) -> String {
        format!(
            "ssh_dir={}, algorithm={}, default_duration={}s, max_duration={}s, automated={}",
            self.config.ssh_key_directory.display(),
            self.config.key_algorithm,
            self.config.default_session_duration,
            self.config.max_session_duration,
            self.config.enable_automated_sessions
        )
    }
}

/// Result of SSH key manager build validation
#[derive(Debug, Clone)]
pub struct KeyManagerValidationResult {
    pub is_valid: bool,
    pub errors: Vec<String>,
    pub warnings: Vec<String>,
    pub config_summary: String,
}

impl Default for KeyManagerValidationResult {
    fn default() -> Self {
        Self::new()
    }
}

impl KeyManagerValidationResult {
    pub fn new() -> Self {
        Self {
            is_valid: true,
            errors: Vec::new(),
            warnings: Vec::new(),
            config_summary: String::new(),
        }
    }

    pub fn add_error(&mut self, error: String) {
        self.is_valid = false;
        self.errors.push(error);
    }

    pub fn add_warning(&mut self, warning: String) {
        self.warnings.push(warning);
    }

    pub fn has_issues(&self) -> bool {
        !self.errors.is_empty() || !self.warnings.is_empty()
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::config::SshSessionConfig;
    use std::time::Duration;
    use tempfile::TempDir;

    fn create_test_config(temp_dir: &TempDir) -> SshSessionConfig {
        SshSessionConfig {
            ssh_key_directory: temp_dir.path().to_path_buf(),
            key_algorithm: "ed25519".to_string(),
            default_session_duration: 300,
            max_session_duration: 3600,
            key_cleanup_interval: Duration::from_secs(60),
            enable_automated_sessions: true,
            max_concurrent_sessions: 5,
            session_rate_limit: 20,
            enable_audit_logging: true,
            audit_log_path: temp_dir.path().join("audit.log"),
            ssh_connection_timeout: Duration::from_secs(30),
            ssh_command_timeout: Duration::from_secs(60),
            ssh_retry_attempts: 3,
            ssh_retry_delay: Duration::from_secs(2),
        }
    }

    fn create_test_hotkey() -> Hotkey {
        Hotkey::new("5DAAnrj7VHTznn2AWBemMuyBwZWs6FNFjdyVXUeYum3PTXFy".to_string()).unwrap()
    }

    #[tokio::test]
    async fn test_successful_build() {
        let temp_dir = TempDir::new().unwrap();
        let config = create_test_config(&temp_dir);
        let hotkey = create_test_hotkey();

        let builder = ValidatorSshKeyManagerBuilder::new(config, hotkey);
        let result = builder.build().await;

        assert!(result.is_ok(), "Build should succeed: {:?}", result.err());

        let key_manager = result.unwrap();

        // Verify the key manager is functional
        let test_session = "test-build-session";
        let (_, _, key_path) = key_manager
            .generate_session_keypair(test_session)
            .await
            .unwrap();

        assert!(key_path.exists());

        key_manager
            .cleanup_session_keys(test_session)
            .await
            .unwrap();
        assert!(!key_path.exists());
    }

    #[tokio::test]
    async fn test_invalid_config_validation() {
        let temp_dir = TempDir::new().unwrap();
        let mut config = create_test_config(&temp_dir);
        let hotkey = create_test_hotkey();

        // Test empty key algorithm
        config.key_algorithm = String::new();
        let builder = ValidatorSshKeyManagerBuilder::new(config.clone(), hotkey.clone());
        let result = builder.build().await;
        assert!(result.is_err());
        let error = result.unwrap_err();
        println!("Actual error: {}", error);
        // Check if the error chain contains our expected message
        let mut source = error.source();
        let mut found = false;
        while let Some(err) = source {
            if err
                .to_string()
                .contains("SSH key algorithm cannot be empty")
            {
                found = true;
                break;
            }
            source = err.source();
        }
        assert!(
            found
                || error
                    .to_string()
                    .contains("SSH key algorithm cannot be empty")
        );

        // Test invalid session duration
        config.key_algorithm = "ed25519".to_string();
        config.default_session_duration = 0;
        let builder = ValidatorSshKeyManagerBuilder::new(config.clone(), hotkey.clone());
        let result = builder.build().await;
        assert!(result.is_err());
        assert!(result
            .unwrap_err()
            .to_string()
            .contains("duration cannot be zero"));

        // Test max < default duration
        config.default_session_duration = 300;
        config.max_session_duration = 100;
        let builder = ValidatorSshKeyManagerBuilder::new(config, hotkey);
        let result = builder.build().await;
        assert!(result.is_err());
        assert!(result
            .unwrap_err()
            .to_string()
            .contains("Maximum session duration"));
    }

    #[tokio::test]
    async fn test_unsupported_algorithm() {
        let temp_dir = TempDir::new().unwrap();
        let mut config = create_test_config(&temp_dir);
        config.key_algorithm = "unsupported".to_string();
        let hotkey = create_test_hotkey();

        let builder = ValidatorSshKeyManagerBuilder::new(config, hotkey);
        let result = builder.build().await;

        assert!(result.is_err());
        let error = result.unwrap_err();
        // Check if the error chain contains our expected message
        let mut source = error.source();
        let mut found = false;
        while let Some(err) = source {
            if err.to_string().contains("Unsupported SSH key algorithm") {
                found = true;
                break;
            }
            source = err.source();
        }
        assert!(found || error.to_string().contains("Unsupported SSH key algorithm"));
    }

    #[tokio::test]
    async fn test_config_summary() {
        let temp_dir = TempDir::new().unwrap();
        let config = create_test_config(&temp_dir);
        let hotkey = create_test_hotkey();

        let builder = ValidatorSshKeyManagerBuilder::new(config, hotkey);
        let summary = builder.get_config_summary();

        assert!(summary.contains("algorithm=ed25519"));
        assert!(summary.contains("default_duration=300s"));
        assert!(summary.contains("max_duration=3600s"));
        assert!(summary.contains("automated=true"));
    }
}
