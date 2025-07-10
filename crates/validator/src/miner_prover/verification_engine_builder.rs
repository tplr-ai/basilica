//! Verification Engine Builder with Enhanced SSH Automation
//!
//! Provides builder pattern for VerificationEngine with guaranteed SSH key manager
//! initialization and comprehensive validation for production environments.

use super::miner_client::MinerClientConfig;
use super::verification::VerificationEngine;
use crate::config::{AutomaticVerificationConfig, SshSessionConfig, VerificationConfig};
use crate::ssh::{SshAutomationComponents, ValidatorSshClient};
use anyhow::{Context, Result};
use common::identity::Hotkey;
use std::sync::Arc;
use tracing::{info, warn};

/// Builder for VerificationEngine with guaranteed SSH automation components
pub struct VerificationEngineBuilder {
    config: VerificationConfig,
    automatic_verification_config: AutomaticVerificationConfig,
    ssh_session_config: SshSessionConfig,
    validator_hotkey: Hotkey,
    bittensor_service: Option<Arc<bittensor::Service>>,
    ssh_client: Option<Arc<ValidatorSshClient>>,
}

impl VerificationEngineBuilder {
    /// Create a new builder instance
    pub fn new(
        config: VerificationConfig,
        automatic_verification_config: AutomaticVerificationConfig,
        ssh_session_config: SshSessionConfig,
        validator_hotkey: Hotkey,
    ) -> Self {
        Self {
            config,
            automatic_verification_config,
            ssh_session_config,
            validator_hotkey,
            bittensor_service: None,
            ssh_client: None,
        }
    }

    /// Set Bittensor service for authentication
    pub fn with_bittensor_service(mut self, bittensor_service: Arc<bittensor::Service>) -> Self {
        self.bittensor_service = Some(bittensor_service);
        self
    }

    /// Set custom SSH client
    pub fn with_ssh_client(mut self, ssh_client: Arc<ValidatorSshClient>) -> Self {
        self.ssh_client = Some(ssh_client);
        self
    }

    /// Build VerificationEngine with guaranteed SSH automation components
    pub async fn build(self) -> Result<VerificationEngine> {
        info!(
            "Building VerificationEngine with SSH automation for validator {}",
            self.validator_hotkey
                .to_string()
                .chars()
                .take(8)
                .collect::<String>()
                + "..."
        );

        let build_start = std::time::Instant::now();

        // Phase 1: Build SSH automation components
        info!("Phase 1: Building SSH automation components");
        let ssh_automation = self
            .build_ssh_automation_components()
            .await
            .context("Failed to build SSH automation components")?;

        // Phase 2: Validate SSH automation readiness
        info!("Phase 2: Validating SSH automation readiness");
        self.validate_ssh_automation_readiness(&ssh_automation)?;

        // Phase 3: Initialize other components
        info!("Phase 3: Initializing other components");
        let miner_client_config = self.build_miner_client_config();
        let ssh_client = self
            .ssh_client
            .unwrap_or_else(|| Arc::new(ValidatorSshClient::new()));

        // Phase 4: Construct VerificationEngine
        info!("Phase 4: Constructing VerificationEngine");
        let verification_engine = VerificationEngine::with_ssh_automation(
            self.config.clone(),
            miner_client_config,
            self.validator_hotkey.clone(),
            ssh_client,
            ssh_automation.enable_dynamic_discovery,
            Some(ssh_automation.ssh_key_manager.clone()),
            self.bittensor_service,
        )?;

        info!(
            "VerificationEngine built successfully in {:?} - dynamic_discovery={}, ssh_automation={}",
            build_start.elapsed(),
            ssh_automation.enable_dynamic_discovery,
            ssh_automation.is_ready_for_automation()
        );

        Ok(verification_engine)
    }

    /// Build SSH automation components with validation
    async fn build_ssh_automation_components(&self) -> Result<SshAutomationComponents> {
        SshAutomationComponents::build(
            self.config.clone(),
            self.automatic_verification_config.clone(),
            self.ssh_session_config.clone(),
            self.validator_hotkey.clone(),
        )
        .await
        .context("Failed to build SSH automation components")
    }

    /// Validate SSH automation readiness
    fn validate_ssh_automation_readiness(
        &self,
        ssh_automation: &SshAutomationComponents,
    ) -> Result<()> {
        info!("Validating SSH automation readiness");

        // Check if SSH automation is ready
        if !ssh_automation.is_ready_for_automation() {
            warn!(
                "SSH automation is not fully ready: dynamic_discovery={}, health={}",
                ssh_automation.enable_dynamic_discovery,
                ssh_automation.health_status.overall_status
            );

            // Log detailed status for debugging
            let status_report = ssh_automation.get_status_report();
            info!("SSH Automation Status: {}", status_report);

            // Don't fail if automation is not ready - just warn
            // This allows the service to start even with degraded SSH capabilities
        } else {
            info!("SSH automation is ready for production use");
        }

        Ok(())
    }

    /// Build miner client configuration
    fn build_miner_client_config(&self) -> MinerClientConfig {
        MinerClientConfig {
            timeout: self.config.discovery_timeout,
            grpc_port_offset: self.config.grpc_port_offset,
            ..Default::default()
        }
    }

    /// Get detailed configuration summary for logging
    pub fn get_config_summary(&self) -> String {
        format!(
            "VerificationEngine[netuid={}, use_dynamic={}, discovery_timeout={:?}, max_concurrent={}, ssh_automation={}, ssh_algorithm={}]",
            self.config.netuid,
            self.config.use_dynamic_discovery,
            self.config.discovery_timeout,
            self.config.max_concurrent_verifications,
            self.automatic_verification_config.enable_ssh_automation,
            self.ssh_session_config.key_algorithm
        )
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::config::{AutomaticVerificationConfig, SshSessionConfig, VerificationConfig};
    use crate::miner_prover::verification::SshAutomationStatus;
    use std::path::PathBuf;
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
            binary_validation: crate::config::BinaryValidationConfig::default(),
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
            persistent_ssh_key_path: None,
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

        // Keep temp_dir alive
        std::mem::forget(temp_dir);

        (
            verification_config,
            automatic_verification_config,
            ssh_session_config,
        )
    }

    fn create_test_hotkey() -> Hotkey {
        Hotkey::new("5DAAnrj7VHTznn2AWBemMuyBwZWs6FNFjdyVXUeYum3PTXFy".to_string()).unwrap()
    }

    #[tokio::test]
    async fn test_verification_engine_builder() {
        let (verification_config, automatic_config, ssh_config) = create_test_configs();
        let hotkey = create_test_hotkey();

        let builder = VerificationEngineBuilder::new(
            verification_config,
            automatic_config,
            ssh_config,
            hotkey,
        );

        let result = builder.build().await;
        assert!(result.is_ok(), "Builder should succeed: {:?}", result.err());

        let engine = result.unwrap();

        // Verify SSH automation status
        let status = engine.get_ssh_automation_status();
        assert!(status.ssh_key_manager_available);

        // Test SSH automation readiness
        let is_ready = engine.is_ssh_automation_ready();
        assert!(is_ready);

        // Test configuration summary
        let summary = engine.get_config_summary();
        assert!(summary.contains("VerificationEngine"));
        assert!(summary.contains("ssh_key_manager=true"));
    }

    #[tokio::test]
    async fn test_builder_with_components() {
        let (verification_config, automatic_config, ssh_config) = create_test_configs();
        let hotkey = create_test_hotkey();

        let ssh_client = Arc::new(ValidatorSshClient::new());

        let builder = VerificationEngineBuilder::new(
            verification_config,
            automatic_config,
            ssh_config,
            hotkey,
        )
        .with_ssh_client(ssh_client);

        let engine = builder.build().await.unwrap();

        // Verify components are properly initialized
        let status = engine.get_ssh_automation_status();
        assert!(status.ssh_key_manager_available);
        assert!(!status.bittensor_service_available); // Not set in this test
    }

    #[test]
    fn test_config_summary() {
        let (verification_config, automatic_config, ssh_config) = create_test_configs();
        let hotkey = create_test_hotkey();

        let builder = VerificationEngineBuilder::new(
            verification_config,
            automatic_config,
            ssh_config,
            hotkey,
        );

        let summary = builder.get_config_summary();
        assert!(summary.contains("VerificationEngine"));
        assert!(summary.contains("netuid=387"));
        assert!(summary.contains("use_dynamic=true"));
        assert!(summary.contains("ssh_automation=true"));
        assert!(summary.contains("ssh_algorithm=ed25519"));
    }

    #[test]
    fn test_ssh_automation_status_display() {
        let status = SshAutomationStatus {
            dynamic_discovery_enabled: true,
            ssh_key_manager_available: true,
            bittensor_service_available: true,
            fallback_key_path: Some(PathBuf::from("/test/key")),
        };

        let display_str = format!("{status}");
        assert!(display_str.contains("dynamic=true"));
        assert!(display_str.contains("key_manager=true"));
        assert!(display_str.contains("bittensor=true"));
        assert!(display_str.contains("fallback_key=/test/key"));
    }

    #[test]
    fn test_verification_engine_with_ssh_automation() {
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
            binary_validation: crate::config::BinaryValidationConfig::default(),
        };

        let miner_client_config = MinerClientConfig::default();
        let hotkey = create_test_hotkey();
        let ssh_client = Arc::new(ValidatorSshClient::new());

        // Test creation with dynamic discovery but no SSH key manager (should fail)
        let result = VerificationEngine::with_ssh_automation(
            verification_config.clone(),
            miner_client_config.clone(),
            hotkey.clone(),
            ssh_client.clone(),
            true, // dynamic discovery enabled
            None, // no SSH key manager
            None, // no bittensor service
        );
        assert!(result.is_err());

        // Test creation with dynamic discovery disabled (should succeed)
        let result = VerificationEngine::with_ssh_automation(
            verification_config,
            miner_client_config,
            hotkey,
            ssh_client,
            false, // dynamic discovery disabled
            None,  // no SSH key manager
            None,  // no bittensor service
        );
        assert!(result.is_ok());

        let engine = result.unwrap();
        assert!(!engine.use_dynamic_discovery());
        assert!(!engine.is_ssh_automation_ready()); // Not ready without key manager or fallback path
    }
}
