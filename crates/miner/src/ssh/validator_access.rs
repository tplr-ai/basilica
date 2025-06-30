//! SSH access management specifically for validators accessing executors

use anyhow::Result;
use common::ssh::{SshCleanupStats, SshHealthStatus, SshService};
use std::sync::Arc;
use tracing::{debug, info, warn};

use super::{MinerSshConfig, SshSessionManager, ValidatorSession};
use crate::executor_manager::ExecutorManager;

/// High-level validator access service combining session management with executor communication
#[derive(Clone)]
pub struct ValidatorAccessService {
    session_manager: SshSessionManager,
    executor_manager: Arc<ExecutorManager>,
    ssh_service: Arc<dyn SshService>,
}

impl ValidatorAccessService {
    /// Create a new validator access service
    pub async fn new(
        config: MinerSshConfig,
        ssh_service: Arc<dyn SshService>,
        executor_manager: Arc<ExecutorManager>,
        db: crate::persistence::RegistrationDb,
    ) -> Result<Self> {
        let session_manager = SshSessionManager::new(config, ssh_service.clone(), db).await?;

        Ok(Self {
            session_manager,
            executor_manager,
            ssh_service,
        })
    }

    /// Provision SSH access for a validator to an executor
    pub async fn provision_validator_access(
        &self,
        validator_hotkey: &str,
        executor_id: &str,
        session_timeout_hours: Option<u32>,
    ) -> Result<String> {
        // Validate executor exists and is available
        self.validate_executor_access(executor_id).await?;

        // Create SSH session
        let timeout =
            session_timeout_hours.map(|hours| std::time::Duration::from_secs(hours as u64 * 3600));

        let session = self
            .session_manager
            .create_session(validator_hotkey, executor_id, timeout)
            .await?;

        // Deploy SSH key to the actual executor machine
        self.deploy_to_executor(&session).await?;

        info!(
            "SSH access provisioned for validator {} to executor {} (session: {})",
            validator_hotkey, executor_id, session.session_id
        );

        Ok(session.connection_string)
    }

    /// Revoke validator access
    pub async fn revoke_validator_access(&self, session_id: &str) -> Result<()> {
        // Get session info before revoking
        let session = self
            .session_manager
            .get_session(session_id)
            .await?
            .ok_or_else(|| anyhow::anyhow!("Session not found: {}", session_id))?;

        // Remove from executor machine
        self.cleanup_from_executor(&session).await?;

        // Revoke the session
        self.session_manager.revoke_session(session_id).await?;

        info!("SSH access revoked for session {}", session_id);
        Ok(())
    }

    /// List active validator SSH sessions
    pub async fn list_validator_sessions(
        &self,
        validator_hotkey: &str,
    ) -> Result<Vec<ValidatorSession>> {
        self.session_manager
            .list_validator_sessions(validator_hotkey)
            .await
    }

    /// Get health status of SSH system
    pub async fn get_health_status(&self) -> Result<SshHealthStatus> {
        self.ssh_service
            .health_check()
            .await
            .map_err(|e| anyhow::anyhow!("SSH health check failed: {}", e))
    }

    /// Perform cleanup of expired resources
    pub async fn cleanup_expired(&self) -> Result<SshCleanupStats> {
        // Cleanup sessions
        let cleaned_sessions = self.session_manager.cleanup_expired_sessions().await?;

        // Cleanup SSH service resources
        let ssh_stats = self
            .ssh_service
            .cleanup_expired()
            .await
            .map_err(|e| anyhow::anyhow!("SSH service cleanup failed: {}", e))?;

        Ok(SshCleanupStats {
            cleaned_keys: ssh_stats.cleaned_keys,
            cleaned_users: ssh_stats.cleaned_users + cleaned_sessions,
            errors_encountered: ssh_stats.errors_encountered,
        })
    }

    /// Validate that executor is available for SSH access
    async fn validate_executor_access(&self, executor_id: &str) -> Result<()> {
        let executors = self.executor_manager.list_available().await?;

        let _executor = executors
            .iter()
            .find(|e| e.id == executor_id)
            .ok_or_else(|| {
                anyhow::anyhow!("Executor {} not found or not available", executor_id)
            })?;

        debug!(
            "Validated executor {} is available for SSH access",
            _executor.id
        );
        Ok(())
    }

    /// Deploy SSH key to the actual executor machine
    async fn deploy_to_executor(&self, session: &ValidatorSession) -> Result<()> {
        debug!("Deploying SSH key to executor {}", session.executor_id);

        // Get executor machine info
        let executors = self.executor_manager.list_available().await?;
        let executor = executors
            .iter()
            .find(|e| e.id == session.executor_id)
            .ok_or_else(|| anyhow::anyhow!("Executor {} not found", session.executor_id))?;

        // Use executor manager to actually deploy the SSH key to the remote machine
        // This performs real SSH operations to deploy the public key
        self.executor_manager
            .deploy_ssh_key_to_executor(
                &session.executor_id,
                &session.user_info.username,
                &session.key_info.public_key,
            )
            .await
            .map_err(|e| {
                anyhow::anyhow!(
                    "Failed to deploy SSH key to executor {}: {}",
                    executor.id,
                    e
                )
            })?;

        info!(
            "SSH key successfully deployed to executor {} for session {}",
            session.executor_id, session.session_id
        );

        Ok(())
    }

    /// Cleanup SSH configuration from executor machine
    async fn cleanup_from_executor(&self, session: &ValidatorSession) -> Result<()> {
        debug!(
            "Cleaning up SSH configuration from executor {}",
            session.executor_id
        );

        // Get executor machine info
        let executors = self.executor_manager.list_available().await?;
        let _executor = executors.iter().find(|e| e.id == session.executor_id);

        if _executor.is_none() {
            warn!("Executor {} not found during cleanup", session.executor_id);
            return Ok(());
        }

        // Use executor manager to actually remove the SSH user and keys from the remote machine
        // This performs real SSH operations to clean up
        if let Some(executor) = _executor {
            self.executor_manager
                .cleanup_ssh_key_from_executor(&session.executor_id, &session.user_info.username)
                .await
                .map_err(|e| {
                    anyhow::anyhow!("Failed to cleanup SSH from executor {}: {}", executor.id, e)
                })?;
        }

        info!(
            "SSH configuration cleaned up from executor {} for session {}",
            session.executor_id, session.session_id
        );

        Ok(())
    }
}

impl std::fmt::Debug for ValidatorAccessService {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        f.debug_struct("ValidatorAccessService")
            .field("session_manager", &"<session_manager>")
            .field("executor_manager", &"<executor_manager>")
            .finish()
    }
}
