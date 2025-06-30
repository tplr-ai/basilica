//! SSH session management for validator access

use anyhow::{Context, Result};
use chrono::{DateTime, Utc};
use common::ssh::{SshAccessConfig, SshKeyInfo, SshKeyParams, SshService, SshUserInfo};
use serde::{Deserialize, Serialize};
use std::collections::HashMap;
use std::sync::Arc;
use std::time::{Duration, SystemTime};
use tokio::sync::RwLock;
use tracing::{debug, info, warn};
use uuid::Uuid;

use super::config::MinerSshConfig;
use crate::persistence::RegistrationDb;

/// Information about an active validator SSH session
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct ValidatorSession {
    /// Session identifier
    pub session_id: String,
    /// Validator hotkey that owns this session
    pub validator_hotkey: String,
    /// Target executor ID
    pub executor_id: String,
    /// SSH key information
    pub key_info: SshKeyInfo,
    /// SSH user information
    pub user_info: SshUserInfo,
    /// Access configuration
    pub access_config: SshAccessConfig,
    /// Session creation time
    pub created_at: DateTime<Utc>,
    /// Session expiration time
    pub expires_at: DateTime<Utc>,
    /// SSH connection string
    pub connection_string: String,
}

impl ValidatorSession {
    /// Check if session is expired
    pub fn is_expired(&self) -> bool {
        Utc::now() >= self.expires_at
    }

    /// Get connection details for the validator
    pub fn get_connection_info(&self) -> (String, String) {
        (
            self.connection_string.clone(),
            self.key_info.private_key_path.clone(),
        )
    }
}

/// Manages SSH sessions for validator access to executors
pub struct SshSessionManager {
    config: MinerSshConfig,
    ssh_service: Arc<dyn SshService>,
    sessions: Arc<RwLock<HashMap<String, ValidatorSession>>>,
    db: RegistrationDb,
}

impl SshSessionManager {
    /// Create a new SSH session manager
    pub async fn new(
        config: MinerSshConfig,
        ssh_service: Arc<dyn SshService>,
        db: RegistrationDb,
    ) -> Result<Self> {
        let manager = Self {
            config,
            ssh_service,
            sessions: Arc::new(RwLock::new(HashMap::new())),
            db,
        };

        info!("SSH session manager initialized");
        Ok(manager)
    }

    /// Create a new validator SSH session
    pub async fn create_session(
        &self,
        validator_hotkey: &str,
        executor_id: &str,
        session_timeout: Option<Duration>,
    ) -> Result<ValidatorSession> {
        info!(
            "Creating SSH session for validator {} to executor {}",
            validator_hotkey, executor_id
        );

        let timeout =
            session_timeout.unwrap_or(Duration::from_secs(self.config.default_session_timeout));

        if timeout.as_secs() > self.config.max_session_timeout {
            return Err(anyhow::anyhow!(
                "Session timeout {} exceeds maximum allowed {}",
                timeout.as_secs(),
                self.config.max_session_timeout
            ));
        }

        // Generate session ID and username
        let session_id = format!("ssh_{}", Uuid::new_v4().simple());
        let ssh_username = format!("{}_{}", self.config.username_prefix, &session_id[..8]);

        // Create SSH key parameters
        let key_params = SshKeyParams {
            algorithm: self.config.default_algorithm.clone(),
            key_size: self.config.default_key_size,
            validity_duration: timeout,
            comment: Some(format!("basilica_session_{session_id}")),
            passphrase: None,
        };

        // Create user info
        let user_info = SshUserInfo {
            username: ssh_username.clone(),
            home_directory: format!("/home/{ssh_username}"),
            shell: self.config.default_shell.clone(),
            groups: self.config.default_groups.clone(),
            is_temporary: true,
            created_at: SystemTime::now(),
            expires_at: Some(SystemTime::now() + timeout),
        };

        // Create access config
        let access_config = SshAccessConfig::default();

        let now = Utc::now();
        let expires_at = now + chrono::Duration::from_std(timeout)?;

        // Provision complete SSH access using the service
        let key_info = self
            .ssh_service
            .provision_access(
                &session_id,
                &ssh_username,
                key_params,
                &user_info,
                &access_config,
                timeout,
            )
            .await
            .map_err(|e| anyhow::anyhow!("Failed to provision SSH access: {}", e))?;

        // Build connection string
        let connection_string = format!(
            "ssh -i {} -o StrictHostKeyChecking=no -o UserKnownHostsFile=/dev/null {}@{}",
            key_info.private_key_path,
            ssh_username,
            executor_id // This should be resolved to actual host:port
        );

        // Create session object
        let session = ValidatorSession {
            session_id: session_id.clone(),
            validator_hotkey: validator_hotkey.to_string(),
            executor_id: executor_id.to_string(),
            key_info,
            user_info,
            access_config,
            created_at: now,
            expires_at,
            connection_string,
        };

        // Store session
        {
            let mut sessions = self.sessions.write().await;
            sessions.insert(session_id.clone(), session.clone());
        }

        // Record in database
        self.db
            .record_ssh_session_created(&session_id, validator_hotkey, executor_id, &expires_at)
            .await
            .context("Failed to record SSH session in database")?;

        info!(
            "SSH session {} created for validator {} -> executor {} (expires: {})",
            session_id, validator_hotkey, executor_id, expires_at
        );

        Ok(session)
    }

    /// Get an existing SSH session
    pub async fn get_session(&self, session_id: &str) -> Result<Option<ValidatorSession>> {
        let sessions = self.sessions.read().await;
        Ok(sessions.get(session_id).cloned())
    }

    /// Revoke a session immediately
    pub async fn revoke_session(&self, session_id: &str) -> Result<()> {
        info!("Revoking SSH session {}", session_id);

        let session = {
            let mut sessions = self.sessions.write().await;
            sessions.remove(session_id)
        };

        if let Some(session) = session {
            // Revoke access using the SSH service
            if let Err(e) = self
                .ssh_service
                .revoke_access(&session.key_info.id, &session.user_info.username)
                .await
            {
                warn!("Failed to revoke SSH access: {}", e);
            }

            // Record revocation in database
            self.db
                .record_ssh_session_revoked(session_id, "manual_revocation")
                .await
                .context("Failed to record SSH session revocation")?;

            info!("SSH session {} revoked successfully", session_id);
        } else {
            warn!("Attempted to revoke non-existent session {}", session_id);
        }

        Ok(())
    }

    /// List active sessions for a validator
    pub async fn list_validator_sessions(
        &self,
        validator_hotkey: &str,
    ) -> Result<Vec<ValidatorSession>> {
        let sessions = self.sessions.read().await;
        Ok(sessions
            .values()
            .filter(|s| s.validator_hotkey == validator_hotkey)
            .cloned()
            .collect())
    }

    /// Get session statistics
    pub async fn get_session_stats(&self) -> Result<SessionStats> {
        let sessions = self.sessions.read().await;
        let now = Utc::now();

        let total_sessions = sessions.len();
        let expired_sessions = sessions.values().filter(|s| s.expires_at < now).count();
        let active_sessions = total_sessions - expired_sessions;

        Ok(SessionStats {
            total_sessions,
            active_sessions,
            expired_sessions,
        })
    }

    /// Clean up expired sessions
    pub async fn cleanup_expired_sessions(&self) -> Result<u32> {
        let now = Utc::now();
        let expired_sessions: Vec<ValidatorSession> = {
            let mut sessions = self.sessions.write().await;
            let mut expired = Vec::new();

            sessions.retain(|_, session| {
                if session.expires_at < now {
                    expired.push(session.clone());
                    false
                } else {
                    true
                }
            });

            expired
        };

        let cleaned_count = expired_sessions.len() as u32;

        if !expired_sessions.is_empty() {
            info!("Cleaning up {} expired SSH sessions", cleaned_count);

            for session in expired_sessions {
                if let Err(e) = self
                    .ssh_service
                    .revoke_access(&session.key_info.id, &session.user_info.username)
                    .await
                {
                    warn!("Failed to cleanup session {}: {}", session.session_id, e);
                } else {
                    // Record cleanup in database
                    if let Err(e) = self
                        .db
                        .record_ssh_session_revoked(&session.session_id, "expired")
                        .await
                    {
                        warn!("Failed to record session cleanup in database: {}", e);
                    }
                }
            }

            debug!("Cleaned up {} expired sessions", cleaned_count);
        }

        Ok(cleaned_count)
    }
}

impl Clone for SshSessionManager {
    fn clone(&self) -> Self {
        Self {
            config: self.config.clone(),
            ssh_service: self.ssh_service.clone(),
            sessions: self.sessions.clone(),
            db: self.db.clone(),
        }
    }
}

impl std::fmt::Debug for SshSessionManager {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        f.debug_struct("SshSessionManager")
            .field("config", &self.config)
            .field("sessions", &"<sessions>")
            .field("db", &"<db>")
            .finish()
    }
}

/// Session statistics
#[derive(Debug, Clone, Serialize)]
pub struct SessionStats {
    pub total_sessions: usize,
    pub active_sessions: usize,
    pub expired_sessions: usize,
}
