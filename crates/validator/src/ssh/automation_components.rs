//! SSH Automation Components
//!
//! Encapsulates all SSH automation dependencies and provides a unified interface
//! for managing SSH automation components in the validator service.

use super::dynamic_discovery_controller::{
    DynamicDiscoveryConfigSummary, DynamicDiscoveryController,
};
use super::key_manager::ValidatorSshKeyManager;
use super::key_manager_builder::ValidatorSshKeyManagerBuilder;
use crate::config::{AutomaticVerificationConfig, SshSessionConfig, VerificationConfig};
use anyhow::{Context, Result};
use common::identity::Hotkey;
use std::sync::Arc;
use std::time::Duration;
use tracing::{error, info, warn};

/// Complete SSH automation components for validator service
#[derive(Clone)]
pub struct SshAutomationComponents {
    /// SSH key manager for session key generation and management
    pub ssh_key_manager: Arc<ValidatorSshKeyManager>,
    /// Dynamic discovery controller for prerequisite validation
    pub discovery_controller: DynamicDiscoveryController,
    /// Whether dynamic discovery is enabled based on configuration and prerequisites
    pub enable_dynamic_discovery: bool,
    /// Configuration summary for logging and debugging
    pub config_summary: DynamicDiscoveryConfigSummary,
    /// Initialization timestamp for monitoring
    pub initialized_at: chrono::DateTime<chrono::Utc>,
    /// Health status of components
    pub health_status: SshAutomationHealthStatus,
}

impl SshAutomationComponents {
    /// Build SSH automation components with comprehensive validation
    pub async fn build(
        verification_config: VerificationConfig,
        automatic_verification_config: AutomaticVerificationConfig,
        ssh_session_config: SshSessionConfig,
        validator_hotkey: Hotkey,
    ) -> Result<Self> {
        info!(
            "Building SSH automation components for validator {}",
            validator_hotkey
                .to_string()
                .chars()
                .take(8)
                .collect::<String>()
                + "..."
        );

        let build_start = std::time::Instant::now();

        // Phase 1: Build SSH key manager with validation
        info!("Phase 1: Building SSH key manager");
        let ssh_key_manager = Self::build_ssh_key_manager(&ssh_session_config, &validator_hotkey)
            .await
            .context("Failed to build SSH key manager")?;

        // Phase 2: Initialize dynamic discovery controller
        info!("Phase 2: Initializing dynamic discovery controller");
        let discovery_controller = DynamicDiscoveryController::new(
            verification_config.clone(),
            automatic_verification_config.clone(),
            ssh_session_config.clone(),
        );

        // Phase 3: Determine if dynamic discovery should be enabled
        info!("Phase 3: Evaluating dynamic discovery enablement");
        let enable_dynamic_discovery = discovery_controller
            .should_enable_dynamic_discovery()
            .await
            .context("Failed to evaluate dynamic discovery prerequisites")?;

        if enable_dynamic_discovery {
            info!("Dynamic discovery enabled - all prerequisites met");
        } else {
            warn!("Dynamic discovery disabled - prerequisites not met or configuration disabled");
        }

        // Phase 4: Generate configuration summary
        let config_summary = discovery_controller.get_config_summary();
        info!("Configuration summary: {}", config_summary);

        // Phase 5: Perform initial health check
        info!("Phase 5: Performing initial health check");
        let health_status = Self::perform_initial_health_check(
            &ssh_key_manager,
            &discovery_controller,
            enable_dynamic_discovery,
        )
        .await?;

        let components = Self {
            ssh_key_manager,
            discovery_controller,
            enable_dynamic_discovery,
            config_summary,
            initialized_at: chrono::Utc::now(),
            health_status,
        };

        info!(
            "SSH automation components built successfully in {:?} - dynamic_discovery={}, health={}",
            build_start.elapsed(),
            enable_dynamic_discovery,
            components.health_status.overall_status
        );

        Ok(components)
    }

    /// Build SSH key manager with guaranteed initialization
    async fn build_ssh_key_manager(
        ssh_session_config: &SshSessionConfig,
        validator_hotkey: &Hotkey,
    ) -> Result<Arc<ValidatorSshKeyManager>> {
        let builder = ValidatorSshKeyManagerBuilder::new(
            ssh_session_config.clone(),
            validator_hotkey.clone(),
        );

        builder
            .build()
            .await
            .context("SSH key manager builder failed")
    }

    /// Perform initial health check of all components
    async fn perform_initial_health_check(
        ssh_key_manager: &Arc<ValidatorSshKeyManager>,
        discovery_controller: &DynamicDiscoveryController,
        enable_dynamic_discovery: bool,
    ) -> Result<SshAutomationHealthStatus> {
        info!("Performing initial SSH automation health check");

        let mut health_status = SshAutomationHealthStatus::new();

        // Test SSH key manager functionality
        let key_manager_healthy = Self::test_ssh_key_manager_health(ssh_key_manager).await;
        health_status.ssh_key_manager_healthy = key_manager_healthy;

        if key_manager_healthy {
            info!("SSH key manager health check: PASSED");
        } else {
            error!("SSH key manager health check: FAILED");
        }

        // Test discovery controller prerequisites if enabled
        if enable_dynamic_discovery {
            let discovery_healthy =
                Self::test_discovery_controller_health(discovery_controller).await;
            health_status.discovery_controller_healthy = discovery_healthy;

            if discovery_healthy {
                info!("Discovery controller health check: PASSED");
            } else {
                error!("Discovery controller health check: FAILED");
            }
        } else {
            health_status.discovery_controller_healthy = true; // Not required if disabled
            info!("Discovery controller health check: SKIPPED (disabled)");
        }

        // Determine overall health status
        health_status.update_overall_status();

        info!(
            "Initial health check completed: {}",
            health_status.overall_status
        );
        Ok(health_status)
    }

    /// Test SSH key manager health
    async fn test_ssh_key_manager_health(ssh_key_manager: &Arc<ValidatorSshKeyManager>) -> bool {
        // Test key generation and cleanup
        let test_session_id = format!("health-check-{}", chrono::Utc::now().timestamp());

        match ssh_key_manager
            .generate_session_keypair(&test_session_id)
            .await
        {
            Ok((_, _, key_path)) => {
                // Verify key file exists
                if !key_path.exists() {
                    error!("Generated key file does not exist");
                    return false;
                }

                // Test cleanup
                match ssh_key_manager.cleanup_session_keys(&test_session_id).await {
                    Ok(_) => {
                        // Verify key file is removed
                        if key_path.exists() {
                            error!("Key file still exists after cleanup");
                            return false;
                        }
                        true
                    }
                    Err(e) => {
                        error!("Key cleanup test failed: {}", e);
                        false
                    }
                }
            }
            Err(e) => {
                error!("SSH key generation test failed: {}", e);
                false
            }
        }
    }

    /// Test discovery controller health
    async fn test_discovery_controller_health(
        discovery_controller: &DynamicDiscoveryController,
    ) -> bool {
        // Re-validate prerequisites to ensure they're still met
        match discovery_controller.validate_prerequisites().await {
            Ok(validation_result) => {
                if validation_result.all_passed() {
                    info!("Discovery controller prerequisites validation: PASSED");
                    true
                } else {
                    warn!(
                        "Discovery controller prerequisites validation: FAILED - {}",
                        validation_result.get_failure_summary()
                    );
                    false
                }
            }
            Err(e) => {
                error!("Discovery controller prerequisites validation error: {}", e);
                false
            }
        }
    }

    /// Perform runtime health check
    pub async fn perform_health_check(&mut self) -> Result<()> {
        info!("Performing runtime SSH automation health check");

        let previous_status = self.health_status.overall_status.clone();

        // Update health status
        self.health_status = Self::perform_initial_health_check(
            &self.ssh_key_manager,
            &self.discovery_controller,
            self.enable_dynamic_discovery,
        )
        .await?;

        self.health_status.last_check_at = Some(chrono::Utc::now());

        // Log status changes
        if self.health_status.overall_status != previous_status {
            warn!(
                "SSH automation health status changed: {} -> {}",
                previous_status, self.health_status.overall_status
            );
        } else {
            info!(
                "SSH automation health status: {}",
                self.health_status.overall_status
            );
        }

        Ok(())
    }

    /// Get runtime metrics for monitoring
    pub fn get_runtime_metrics(&self) -> SshAutomationMetrics {
        let uptime = chrono::Utc::now().signed_duration_since(self.initialized_at);

        SshAutomationMetrics {
            uptime_seconds: uptime.num_seconds() as u64,
            dynamic_discovery_enabled: self.enable_dynamic_discovery,
            ssh_key_manager_healthy: self.health_status.ssh_key_manager_healthy,
            discovery_controller_healthy: self.health_status.discovery_controller_healthy,
            overall_healthy: matches!(
                self.health_status.overall_status,
                OverallHealthStatus::Healthy
            ),
            last_health_check: self.health_status.last_check_at,
            config_summary: self.config_summary.clone(),
        }
    }

    /// Check if components are ready for SSH automation
    pub fn is_ready_for_automation(&self) -> bool {
        matches!(
            self.health_status.overall_status,
            OverallHealthStatus::Healthy
        ) && self.enable_dynamic_discovery
    }

    /// Get detailed status report for debugging
    pub fn get_status_report(&self) -> SshAutomationStatusReport {
        SshAutomationStatusReport {
            initialized_at: self.initialized_at,
            enable_dynamic_discovery: self.enable_dynamic_discovery,
            health_status: self.health_status.clone(),
            config_summary: self.config_summary.clone(),
            runtime_metrics: self.get_runtime_metrics(),
        }
    }

    /// Start background health monitoring task
    pub fn start_health_monitoring(
        components: Arc<tokio::sync::RwLock<Self>>,
        interval: Duration,
    ) -> tokio::task::JoinHandle<()> {
        info!(
            "Starting SSH automation health monitoring with interval: {:?}",
            interval
        );

        tokio::spawn(async move {
            let mut interval_timer = tokio::time::interval(interval);

            loop {
                interval_timer.tick().await;

                let mut components_guard = components.write().await;
                if let Err(e) = components_guard.perform_health_check().await {
                    error!("SSH automation health check failed: {}", e);
                }
            }
        })
    }
}

/// Health status of SSH automation components
#[derive(Debug, Clone)]
pub struct SshAutomationHealthStatus {
    pub ssh_key_manager_healthy: bool,
    pub discovery_controller_healthy: bool,
    pub overall_status: OverallHealthStatus,
    pub last_check_at: Option<chrono::DateTime<chrono::Utc>>,
    pub error_messages: Vec<String>,
}

impl Default for SshAutomationHealthStatus {
    fn default() -> Self {
        Self::new()
    }
}

impl SshAutomationHealthStatus {
    pub fn new() -> Self {
        Self {
            ssh_key_manager_healthy: false,
            discovery_controller_healthy: false,
            overall_status: OverallHealthStatus::Unknown,
            last_check_at: None,
            error_messages: Vec::new(),
        }
    }

    pub fn update_overall_status(&mut self) {
        self.overall_status = if self.ssh_key_manager_healthy && self.discovery_controller_healthy {
            OverallHealthStatus::Healthy
        } else if self.ssh_key_manager_healthy || self.discovery_controller_healthy {
            OverallHealthStatus::Degraded
        } else {
            OverallHealthStatus::Unhealthy
        };
    }

    pub fn add_error(&mut self, error: String) {
        self.error_messages.push(error);
        self.update_overall_status();
    }
}

/// Overall health status enumeration
#[derive(Debug, Clone, PartialEq)]
pub enum OverallHealthStatus {
    Unknown,
    Healthy,
    Degraded,
    Unhealthy,
}

impl std::fmt::Display for OverallHealthStatus {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        match self {
            OverallHealthStatus::Unknown => write!(f, "UNKNOWN"),
            OverallHealthStatus::Healthy => write!(f, "HEALTHY"),
            OverallHealthStatus::Degraded => write!(f, "DEGRADED"),
            OverallHealthStatus::Unhealthy => write!(f, "UNHEALTHY"),
        }
    }
}

/// Runtime metrics for SSH automation
#[derive(Debug, Clone)]
pub struct SshAutomationMetrics {
    pub uptime_seconds: u64,
    pub dynamic_discovery_enabled: bool,
    pub ssh_key_manager_healthy: bool,
    pub discovery_controller_healthy: bool,
    pub overall_healthy: bool,
    pub last_health_check: Option<chrono::DateTime<chrono::Utc>>,
    pub config_summary: DynamicDiscoveryConfigSummary,
}

/// Comprehensive status report for SSH automation
#[derive(Debug, Clone)]
pub struct SshAutomationStatusReport {
    pub initialized_at: chrono::DateTime<chrono::Utc>,
    pub enable_dynamic_discovery: bool,
    pub health_status: SshAutomationHealthStatus,
    pub config_summary: DynamicDiscoveryConfigSummary,
    pub runtime_metrics: SshAutomationMetrics,
}

impl std::fmt::Display for SshAutomationStatusReport {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        write!(
            f,
            "SSH Automation Status Report:\n\
             - Initialized: {}\n\
             - Dynamic Discovery: {}\n\
             - Overall Health: {}\n\
             - SSH Key Manager: {}\n\
             - Discovery Controller: {}\n\
             - Uptime: {}s\n\
             - Configuration: {}",
            self.initialized_at.to_rfc3339(),
            self.enable_dynamic_discovery,
            self.health_status.overall_status,
            if self.health_status.ssh_key_manager_healthy {
                "HEALTHY"
            } else {
                "UNHEALTHY"
            },
            if self.health_status.discovery_controller_healthy {
                "HEALTHY"
            } else {
                "UNHEALTHY"
            },
            self.runtime_metrics.uptime_seconds,
            self.config_summary
        )
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::config::{AutomaticVerificationConfig, SshSessionConfig, VerificationConfig};
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
    async fn test_ssh_automation_components_build() {
        let (verification_config, automatic_config, ssh_config) = create_test_configs();
        let hotkey = create_test_hotkey();

        let result = SshAutomationComponents::build(
            verification_config,
            automatic_config,
            ssh_config,
            hotkey,
        )
        .await;

        assert!(
            result.is_ok(),
            "Components build should succeed: {:?}",
            result.err()
        );

        let components = result.unwrap();

        // Verify components are initialized
        assert!(!components
            .ssh_key_manager
            .get_session_key_path("test")
            .as_os_str()
            .is_empty());
        assert!(components.initialized_at <= chrono::Utc::now());

        // Test health check
        assert!(matches!(
            components.health_status.overall_status,
            OverallHealthStatus::Healthy | OverallHealthStatus::Degraded
        ));
    }

    #[tokio::test]
    async fn test_health_status_update() {
        let mut health_status = SshAutomationHealthStatus::new();

        health_status.ssh_key_manager_healthy = true;
        health_status.discovery_controller_healthy = true;
        health_status.update_overall_status();

        assert!(matches!(
            health_status.overall_status,
            OverallHealthStatus::Healthy
        ));

        health_status.discovery_controller_healthy = false;
        health_status.update_overall_status();

        assert!(matches!(
            health_status.overall_status,
            OverallHealthStatus::Degraded
        ));

        health_status.ssh_key_manager_healthy = false;
        health_status.update_overall_status();

        assert!(matches!(
            health_status.overall_status,
            OverallHealthStatus::Unhealthy
        ));
    }

    #[tokio::test]
    async fn test_runtime_metrics() {
        let (verification_config, automatic_config, ssh_config) = create_test_configs();
        let hotkey = create_test_hotkey();

        let components = SshAutomationComponents::build(
            verification_config,
            automatic_config,
            ssh_config,
            hotkey,
        )
        .await
        .unwrap();

        let metrics = components.get_runtime_metrics();

        assert!(metrics.uptime_seconds >= 0);
        assert_eq!(metrics.config_summary.key_algorithm, "ed25519");

        let report = components.get_status_report();
        let report_str = format!("{}", report);
        assert!(report_str.contains("SSH Automation Status Report"));
        assert!(report_str.contains("Dynamic Discovery"));
    }

    #[tokio::test]
    async fn test_automation_readiness() {
        let (verification_config, automatic_config, ssh_config) = create_test_configs();
        let hotkey = create_test_hotkey();

        let components = SshAutomationComponents::build(
            verification_config,
            automatic_config,
            ssh_config,
            hotkey,
        )
        .await
        .unwrap();

        // Test automation readiness
        let is_ready = components.is_ready_for_automation();

        // Should depend on health status and dynamic discovery enablement
        if components.enable_dynamic_discovery
            && matches!(
                components.health_status.overall_status,
                OverallHealthStatus::Healthy
            )
        {
            assert!(is_ready);
        } else {
            assert!(!is_ready);
        }
    }

    #[test]
    fn test_overall_health_status_display() {
        assert_eq!(format!("{}", OverallHealthStatus::Unknown), "UNKNOWN");
        assert_eq!(format!("{}", OverallHealthStatus::Healthy), "HEALTHY");
        assert_eq!(format!("{}", OverallHealthStatus::Degraded), "DEGRADED");
        assert_eq!(format!("{}", OverallHealthStatus::Unhealthy), "UNHEALTHY");
    }
}
