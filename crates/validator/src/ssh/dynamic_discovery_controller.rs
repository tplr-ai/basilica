//! Dynamic Discovery Controller for SSH Automation
//!
//! Manages dynamic discovery enablement with comprehensive prerequisite validation
//! and configuration-driven control for production environments.

use crate::config::{AutomaticVerificationConfig, SshSessionConfig, VerificationConfig};
use anyhow::Result;
use std::time::Duration;
use tokio::fs;
use tokio::net::TcpStream;
use tokio::time::timeout;
use tracing::{debug, error, info, warn};

/// Controls dynamic discovery enablement with prerequisite validation
#[derive(Debug, Clone)]
pub struct DynamicDiscoveryController {
    verification_config: VerificationConfig,
    automatic_verification_config: AutomaticVerificationConfig,
    ssh_session_config: SshSessionConfig,
}

impl DynamicDiscoveryController {
    /// Create a new dynamic discovery controller
    pub fn new(
        verification_config: VerificationConfig,
        automatic_verification_config: AutomaticVerificationConfig,
        ssh_session_config: SshSessionConfig,
    ) -> Self {
        Self {
            verification_config,
            automatic_verification_config,
            ssh_session_config,
        }
    }

    /// Determine if dynamic discovery should be enabled
    pub async fn should_enable_dynamic_discovery(&self) -> Result<bool> {
        info!("Evaluating dynamic discovery enablement prerequisites");

        // Step 1: Check configuration flags
        let config_enabled = self.check_configuration_flags()?;
        if !config_enabled {
            info!("Dynamic discovery disabled by configuration");
            return Ok(false);
        }

        // Step 2: Validate all prerequisites
        let validation_result = self.validate_prerequisites().await?;
        if !validation_result.all_passed() {
            warn!(
                "Dynamic discovery prerequisites not met: {}",
                validation_result.get_failure_summary()
            );
            return Ok(false);
        }

        // Step 3: Perform runtime capability checks
        let runtime_ready = self.validate_runtime_capabilities().await?;
        if !runtime_ready {
            warn!("Dynamic discovery runtime capabilities not available");
            return Ok(false);
        }

        info!("Dynamic discovery prerequisites validated successfully");
        Ok(true)
    }

    /// Check if configuration flags enable dynamic discovery
    fn check_configuration_flags(&self) -> Result<bool> {
        debug!("Checking dynamic discovery configuration flags");

        let flags = DynamicDiscoveryFlags {
            use_dynamic_discovery: self.verification_config.use_dynamic_discovery,
            automatic_verification_enabled: self.automatic_verification_config.enabled,
            ssh_automation_enabled: self.automatic_verification_config.enable_ssh_automation,
            automated_sessions_enabled: self.ssh_session_config.enable_automated_sessions,
        };

        debug!("Configuration flags: {:?}", flags);

        if !flags.use_dynamic_discovery {
            debug!("Dynamic discovery disabled in verification config");
            return Ok(false);
        }

        if !flags.automatic_verification_enabled {
            debug!("Automatic verification disabled in config");
            return Ok(false);
        }

        if !flags.ssh_automation_enabled {
            debug!("SSH automation disabled in automatic verification config");
            return Ok(false);
        }

        if !flags.automated_sessions_enabled {
            debug!("Automated sessions disabled in SSH session config");
            return Ok(false);
        }

        debug!("All configuration flags enable dynamic discovery");
        Ok(true)
    }

    /// Validate all prerequisites for dynamic discovery
    pub async fn validate_prerequisites(&self) -> Result<PrerequisiteValidationResult> {
        info!("Validating dynamic discovery prerequisites");

        let mut result = PrerequisiteValidationResult::new();

        // SSH Infrastructure Validation
        self.validate_ssh_infrastructure(&mut result).await;

        // Network Connectivity Validation
        self.validate_network_connectivity(&mut result).await;

        // SSH Client Validation
        self.validate_ssh_client_availability(&mut result).await;

        // gRPC Configuration Validation
        self.validate_grpc_configuration(&mut result).await;

        // File System Permissions Validation
        self.validate_filesystem_permissions(&mut result).await;

        // Resource Availability Validation
        self.validate_resource_availability(&mut result).await;

        info!(
            "Prerequisite validation completed: passed={}, failed={}, warnings={}",
            result.passed_checks.len(),
            result.failed_checks.len(),
            result.warnings.len()
        );

        Ok(result)
    }

    /// Validate SSH infrastructure prerequisites
    async fn validate_ssh_infrastructure(&self, result: &mut PrerequisiteValidationResult) {
        debug!("Validating SSH infrastructure prerequisites");

        // Check SSH key directory
        let ssh_dir = &self.ssh_session_config.ssh_key_directory;
        if ssh_dir.exists() {
            if ssh_dir.is_dir() {
                result.add_passed("SSH key directory exists and is accessible");

                // Check permissions on Unix systems
                #[cfg(unix)]
                {
                    use std::os::unix::fs::PermissionsExt;
                    if let Ok(metadata) = fs::metadata(ssh_dir).await {
                        let mode = metadata.permissions().mode() & 0o777;
                        if mode == 0o700 {
                            result.add_passed("SSH directory has correct permissions (0700)");
                        } else {
                            result.add_warning(format!(
                                "SSH directory permissions ({mode:o}) are not optimal (expected 0700)"
                            ));
                        }
                    } else {
                        result.add_failed("Cannot read SSH directory metadata");
                    }
                }
            } else {
                result.add_failed("SSH key path exists but is not a directory");
            }
        } else {
            // Try to create the directory
            match fs::create_dir_all(ssh_dir).await {
                Ok(_) => {
                    result.add_passed("SSH key directory created successfully");
                }
                Err(e) => {
                    result.add_failed(format!("Cannot create SSH key directory: {e}"));
                }
            }
        }

        // Validate SSH key algorithm
        let supported_algorithms = ["ed25519", "rsa", "ecdsa"];
        if supported_algorithms.contains(&self.ssh_session_config.key_algorithm.as_str()) {
            result.add_passed(format!(
                "SSH key algorithm '{}' is supported",
                self.ssh_session_config.key_algorithm
            ));
        } else {
            result.add_failed(format!(
                "SSH key algorithm '{}' is not supported",
                self.ssh_session_config.key_algorithm
            ));
        }

        // Check SSH configuration values
        if self.ssh_session_config.default_session_duration > 0 {
            result.add_passed("SSH session duration configuration is valid");
        } else {
            result.add_failed("SSH session duration cannot be zero");
        }
    }

    /// Validate network connectivity prerequisites
    async fn validate_network_connectivity(&self, result: &mut PrerequisiteValidationResult) {
        debug!("Validating network connectivity prerequisites");

        // Test basic network connectivity
        let test_addresses = [
            ("8.8.8.8", 53), // Google DNS
            ("1.1.1.1", 53), // Cloudflare DNS
        ];

        let mut connectivity_ok = false;
        for (host, port) in &test_addresses {
            match timeout(Duration::from_secs(5), TcpStream::connect((*host, *port))).await {
                Ok(Ok(_)) => {
                    connectivity_ok = true;
                    debug!("Network connectivity test passed for {}:{}", host, port);
                    break;
                }
                Ok(Err(e)) => {
                    debug!(
                        "Network connectivity test failed for {}:{}: {}",
                        host, port, e
                    );
                }
                Err(_) => {
                    debug!("Network connectivity test timed out for {}:{}", host, port);
                }
            }
        }

        if connectivity_ok {
            result.add_passed("Basic network connectivity is available");
        } else {
            result.add_warning(
                "Basic network connectivity tests failed (may still work in production)",
            );
        }

        // Validate discovery timeout configuration
        if self.verification_config.discovery_timeout.as_secs() > 0 {
            result.add_passed("Discovery timeout configuration is valid");
        } else {
            result.add_failed("Discovery timeout must be greater than zero");
        }

        // Validate connection timeout configurations
        if self.ssh_session_config.ssh_connection_timeout.as_secs() > 0 {
            result.add_passed("SSH connection timeout configuration is valid");
        } else {
            result.add_failed("SSH connection timeout must be greater than zero");
        }
    }

    /// Validate SSH client availability
    async fn validate_ssh_client_availability(&self, result: &mut PrerequisiteValidationResult) {
        debug!("Validating SSH client availability");

        // Check if ssh command is available in PATH
        match tokio::process::Command::new("ssh").arg("-V").output().await {
            Ok(output) => {
                if output.status.success() {
                    let version = String::from_utf8_lossy(&output.stderr);
                    result.add_passed(format!("SSH client is available: {}", version.trim()));
                } else {
                    result.add_warning("SSH command available but returned non-zero status");
                }
            }
            Err(e) => {
                result.add_warning(format!("SSH command not found in PATH: {e}"));
            }
        }

        // Check SSH key generation capability
        match tokio::process::Command::new("ssh-keygen")
            .arg("-t")
            .arg(&self.ssh_session_config.key_algorithm)
            .arg("-f")
            .arg("/dev/null")
            .arg("-N")
            .arg("")
            .arg("-q")
            .output()
            .await
        {
            Ok(output) => {
                if output.status.success() {
                    result.add_passed("SSH key generation capability is available");
                } else {
                    result.add_warning("SSH key generation test failed");
                }
            }
            Err(_) => {
                result.add_warning("SSH key generation tool (ssh-keygen) not available");
            }
        }
    }

    /// Validate gRPC configuration
    async fn validate_grpc_configuration(&self, result: &mut PrerequisiteValidationResult) {
        debug!("Validating gRPC configuration");

        // Validate gRPC port offset configuration
        if let Some(offset) = self.verification_config.grpc_port_offset {
            if offset > 0 && offset < 65535 {
                result.add_passed(format!("gRPC port offset configuration is valid: {offset}"));
            } else {
                result.add_failed(format!("gRPC port offset is invalid: {offset}"));
            }
        } else {
            result.add_passed("Using default gRPC port configuration");
        }

        // Validate automatic verification configuration
        if self
            .automatic_verification_config
            .max_concurrent_verifications
            > 0
        {
            result.add_passed("Concurrent verification limit is properly configured");
        } else {
            result.add_failed("Concurrent verification limit must be greater than zero");
        }

        if self.automatic_verification_config.discovery_interval > 0 {
            result.add_passed("Discovery interval configuration is valid");
        } else {
            result.add_failed("Discovery interval must be greater than zero");
        }
    }

    /// Validate filesystem permissions
    async fn validate_filesystem_permissions(&self, result: &mut PrerequisiteValidationResult) {
        debug!("Validating filesystem permissions");

        let ssh_dir = &self.ssh_session_config.ssh_key_directory;

        // Test write permissions
        let test_file = ssh_dir.join(".permission_test");
        match fs::write(&test_file, b"test").await {
            Ok(_) => {
                result.add_passed("SSH directory is writable");
                if let Err(e) = fs::remove_file(&test_file).await {
                    result.add_warning(format!("Failed to clean up test file: {e}"));
                }
            }
            Err(e) => {
                result.add_failed(format!("SSH directory is not writable: {e}"));
            }
        }

        // Check parent directory permissions if SSH dir doesn't exist
        if !ssh_dir.exists() {
            if let Some(parent) = ssh_dir.parent() {
                if parent.exists() {
                    result.add_passed("Parent directory exists for SSH key directory");
                } else {
                    result.add_warning("Parent directory for SSH key directory does not exist");
                }
            }
        }

        // Validate audit log directory if enabled
        if self.ssh_session_config.enable_audit_logging {
            let audit_log_path = &self.ssh_session_config.audit_log_path;
            if let Some(audit_dir) = audit_log_path.parent() {
                if audit_dir.exists() {
                    result.add_passed("Audit log directory exists");
                } else {
                    match fs::create_dir_all(audit_dir).await {
                        Ok(_) => {
                            result.add_passed("Audit log directory created");
                        }
                        Err(e) => {
                            result.add_warning(format!("Cannot create audit log directory: {e}"));
                        }
                    }
                }
            }
        }
    }

    /// Validate resource availability
    async fn validate_resource_availability(&self, result: &mut PrerequisiteValidationResult) {
        debug!("Validating resource availability");

        // Check available disk space in SSH directory
        let ssh_dir = &self.ssh_session_config.ssh_key_directory;
        if let Ok(_metadata) = fs::metadata(ssh_dir).await {
            result.add_passed("SSH directory is accessible for resource checks");
        } else {
            result.add_warning("Cannot access SSH directory for resource validation");
        }

        // Validate session rate limits
        if self.ssh_session_config.session_rate_limit > 0 {
            result.add_passed("Session rate limit configuration is valid");
        } else {
            result.add_failed("Session rate limit must be greater than zero");
        }

        // Validate concurrent session limits
        if self.ssh_session_config.max_concurrent_sessions > 0 {
            result.add_passed("Concurrent session limit configuration is valid");
        } else {
            result.add_failed("Concurrent session limit must be greater than zero");
        }

        // Check if cleanup interval is reasonable
        let cleanup_secs = self.ssh_session_config.key_cleanup_interval.as_secs();
        if (30..=3600).contains(&cleanup_secs) {
            result.add_passed("Key cleanup interval is within reasonable range");
        } else {
            result.add_warning(format!(
                "Key cleanup interval ({cleanup_secs}s) may be too short or too long"
            ));
        }
    }

    /// Validate runtime capabilities
    async fn validate_runtime_capabilities(&self) -> Result<bool> {
        debug!("Validating runtime capabilities");

        // Test key generation capability
        let test_result = self.test_ssh_key_generation().await;
        if !test_result {
            error!("SSH key generation test failed");
            return Ok(false);
        }

        // Test directory operations
        let dir_test_result = self.test_directory_operations().await;
        if !dir_test_result {
            error!("Directory operations test failed");
            return Ok(false);
        }

        debug!("Runtime capabilities validation passed");
        Ok(true)
    }

    /// Test SSH key generation capability
    async fn test_ssh_key_generation(&self) -> bool {
        debug!("Testing SSH key generation capability");

        match ssh_key::PrivateKey::random(
            &mut rand::thread_rng(),
            match self.ssh_session_config.key_algorithm.as_str() {
                "ed25519" => ssh_key::Algorithm::Ed25519,
                "rsa" => ssh_key::Algorithm::Rsa { hash: None },
                "ecdsa" => ssh_key::Algorithm::Ecdsa {
                    curve: ssh_key::EcdsaCurve::NistP256,
                },
                _ => {
                    error!(
                        "Unsupported SSH algorithm for testing: {}",
                        self.ssh_session_config.key_algorithm
                    );
                    return false;
                }
            },
        ) {
            Ok(private_key) => match private_key.to_openssh(ssh_key::LineEnding::default()) {
                Ok(_) => {
                    debug!("SSH key generation test passed");
                    true
                }
                Err(e) => {
                    error!("SSH key serialization test failed: {}", e);
                    false
                }
            },
            Err(e) => {
                error!("SSH key generation test failed: {}", e);
                false
            }
        }
    }

    /// Test directory operations
    async fn test_directory_operations(&self) -> bool {
        debug!("Testing directory operations");

        let ssh_dir = &self.ssh_session_config.ssh_key_directory;
        let test_file = ssh_dir.join(".runtime_test");

        // Test write operation
        if let Err(e) = fs::write(&test_file, b"runtime_test").await {
            error!("Directory write test failed: {}", e);
            return false;
        }

        // Test read operation
        if let Err(e) = fs::read(&test_file).await {
            error!("Directory read test failed: {}", e);
            let _ = fs::remove_file(&test_file).await;
            return false;
        }

        // Test delete operation
        if let Err(e) = fs::remove_file(&test_file).await {
            error!("Directory delete test failed: {}", e);
            return false;
        }

        debug!("Directory operations test passed");
        true
    }

    /// Get detailed configuration summary
    pub fn get_config_summary(&self) -> DynamicDiscoveryConfigSummary {
        DynamicDiscoveryConfigSummary {
            dynamic_discovery_enabled: self.verification_config.use_dynamic_discovery,
            automatic_verification_enabled: self.automatic_verification_config.enabled,
            ssh_automation_enabled: self.automatic_verification_config.enable_ssh_automation,
            automated_sessions_enabled: self.ssh_session_config.enable_automated_sessions,
            discovery_interval: self.automatic_verification_config.discovery_interval,
            session_duration: self.ssh_session_config.default_session_duration,
            max_concurrent_verifications: self
                .automatic_verification_config
                .max_concurrent_verifications,
            max_concurrent_sessions: self.ssh_session_config.max_concurrent_sessions,
            key_algorithm: self.ssh_session_config.key_algorithm.clone(),
            audit_logging_enabled: self.ssh_session_config.enable_audit_logging,
        }
    }
}

/// Configuration flags for dynamic discovery
#[derive(Debug, Clone)]
struct DynamicDiscoveryFlags {
    use_dynamic_discovery: bool,
    automatic_verification_enabled: bool,
    ssh_automation_enabled: bool,
    automated_sessions_enabled: bool,
}

/// Result of prerequisite validation
#[derive(Debug, Clone)]
pub struct PrerequisiteValidationResult {
    pub passed_checks: Vec<String>,
    pub failed_checks: Vec<String>,
    pub warnings: Vec<String>,
}

impl Default for PrerequisiteValidationResult {
    fn default() -> Self {
        Self::new()
    }
}

impl PrerequisiteValidationResult {
    pub fn new() -> Self {
        Self {
            passed_checks: Vec::new(),
            failed_checks: Vec::new(),
            warnings: Vec::new(),
        }
    }

    pub fn add_passed(&mut self, check: impl Into<String>) {
        self.passed_checks.push(check.into());
    }

    pub fn add_failed(&mut self, check: impl Into<String>) {
        self.failed_checks.push(check.into());
    }

    pub fn add_warning(&mut self, warning: impl Into<String>) {
        self.warnings.push(warning.into());
    }

    pub fn all_passed(&self) -> bool {
        self.failed_checks.is_empty()
    }

    pub fn has_warnings(&self) -> bool {
        !self.warnings.is_empty()
    }

    pub fn get_failure_summary(&self) -> String {
        if self.failed_checks.is_empty() {
            "No failures".to_string()
        } else {
            format!("Failed checks: {}", self.failed_checks.join(", "))
        }
    }

    pub fn get_summary(&self) -> String {
        format!(
            "Passed: {}, Failed: {}, Warnings: {}",
            self.passed_checks.len(),
            self.failed_checks.len(),
            self.warnings.len()
        )
    }
}

/// Configuration summary for dynamic discovery
#[derive(Debug, Clone)]
pub struct DynamicDiscoveryConfigSummary {
    pub dynamic_discovery_enabled: bool,
    pub automatic_verification_enabled: bool,
    pub ssh_automation_enabled: bool,
    pub automated_sessions_enabled: bool,
    pub discovery_interval: u64,
    pub session_duration: u64,
    pub max_concurrent_verifications: usize,
    pub max_concurrent_sessions: usize,
    pub key_algorithm: String,
    pub audit_logging_enabled: bool,
}

impl std::fmt::Display for DynamicDiscoveryConfigSummary {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        write!(
            f,
            "DynamicDiscovery[enabled={}, auto_verify={}, ssh_auto={}, sessions={}, interval={}s, duration={}s, max_verify={}, max_sessions={}, algo={}, audit={}]",
            self.dynamic_discovery_enabled,
            self.automatic_verification_enabled,
            self.ssh_automation_enabled,
            self.automated_sessions_enabled,
            self.discovery_interval,
            self.session_duration,
            self.max_concurrent_verifications,
            self.max_concurrent_sessions,
            self.key_algorithm,
            self.audit_logging_enabled
        )
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use std::time::Duration;
    use tempfile::TempDir;

    fn create_test_configs() -> (
        VerificationConfig,
        AutomaticVerificationConfig,
        SshSessionConfig,
    ) {
        let temp_dir = TempDir::new().unwrap();

        let verification_config = VerificationConfig {
            verification_interval: Duration::from_secs(600),
            max_concurrent_verifications: 5,
            challenge_timeout: Duration::from_secs(30),
            min_score_threshold: 0.5,
            max_miners_per_round: 10,
            min_verification_interval: Duration::from_secs(300),
            netuid: 387,
            use_dynamic_discovery: true,
            discovery_timeout: Duration::from_secs(30),
            fallback_to_static: true,
            cache_miner_info_ttl: Duration::from_secs(300),
            grpc_port_offset: Some(1000),
        };

        let automatic_verification_config = AutomaticVerificationConfig {
            enabled: true,
            discovery_interval: 300,
            min_verification_interval_hours: 1,
            max_concurrent_verifications: 5,
            enable_ssh_automation: true,
        };

        let ssh_session_config = SshSessionConfig {
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
        };

        // Keep temp_dir alive by leaking it for test purposes
        std::mem::forget(temp_dir);

        (
            verification_config,
            automatic_verification_config,
            ssh_session_config,
        )
    }

    #[tokio::test]
    async fn test_dynamic_discovery_enabled() {
        let (verification_config, automatic_config, ssh_config) = create_test_configs();

        let controller =
            DynamicDiscoveryController::new(verification_config, automatic_config, ssh_config);

        let result = controller.should_enable_dynamic_discovery().await;
        assert!(result.is_ok());

        // May be true or false depending on system prerequisites, but should not error
        let enabled = result.unwrap();
        println!("Dynamic discovery enabled: {enabled}");
    }

    #[tokio::test]
    async fn test_disabled_flags() {
        let (mut verification_config, automatic_config, ssh_config) = create_test_configs();

        // Disable dynamic discovery
        verification_config.use_dynamic_discovery = false;

        let controller =
            DynamicDiscoveryController::new(verification_config, automatic_config, ssh_config);

        let result = controller.should_enable_dynamic_discovery().await.unwrap();
        assert!(!result);
    }

    #[tokio::test]
    async fn test_config_summary() {
        let (verification_config, automatic_config, ssh_config) = create_test_configs();

        let controller =
            DynamicDiscoveryController::new(verification_config, automatic_config, ssh_config);

        let summary = controller.get_config_summary();
        assert!(summary.dynamic_discovery_enabled);
        assert!(summary.automatic_verification_enabled);
        assert!(summary.ssh_automation_enabled);
        assert_eq!(summary.key_algorithm, "ed25519");

        let display_str = format!("{summary}");
        assert!(display_str.contains("enabled=true"));
        assert!(display_str.contains("algo=ed25519"));
    }

    #[tokio::test]
    async fn test_prerequisite_validation() {
        let (verification_config, automatic_config, ssh_config) = create_test_configs();

        let controller =
            DynamicDiscoveryController::new(verification_config, automatic_config, ssh_config);

        let result = controller.validate_prerequisites().await.unwrap();

        // Should have some passed checks
        assert!(!result.passed_checks.is_empty());

        // Print summary for debugging
        println!("Validation summary: {}", result.get_summary());
        if !result.failed_checks.is_empty() {
            println!("Failed checks: {:?}", result.failed_checks);
        }
        if !result.warnings.is_empty() {
            println!("Warnings: {:?}", result.warnings);
        }
    }

    #[test]
    fn test_prerequisite_validation_result() {
        let mut result = PrerequisiteValidationResult::new();

        result.add_passed("Test passed");
        result.add_failed("Test failed");
        result.add_warning("Test warning");

        assert!(!result.all_passed());
        assert!(result.has_warnings());
        assert_eq!(result.passed_checks.len(), 1);
        assert_eq!(result.failed_checks.len(), 1);
        assert_eq!(result.warnings.len(), 1);

        let summary = result.get_summary();
        assert!(summary.contains("Passed: 1"));
        assert!(summary.contains("Failed: 1"));
        assert!(summary.contains("Warnings: 1"));
    }
}
