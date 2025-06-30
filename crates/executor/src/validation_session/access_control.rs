use super::hotkey_verifier::{HotkeySignatureVerifier, SignatureChallenge};
use super::rate_limiter::{RateLimitStatus, RequestType, ValidatorRateLimiter};
use super::types::{
    AccessControlConfig, SessionStats, ValidatorAccess, ValidatorId, ValidatorRole,
};
use anyhow::Result;
use common::journal::{log_security_violation, SecuritySeverity};
use std::collections::HashMap;
use std::net::IpAddr;
use std::path::PathBuf;
use std::sync::Arc;
use std::time::SystemTime;
use tokio::sync::RwLock;
use tracing::{debug, info, warn};

pub struct ValidatorAccessControl {
    pub config: AccessControlConfig,
    active_sessions: Arc<RwLock<HashMap<String, ValidatorAccess>>>,
    storage_path: PathBuf,
    hotkey_verifier: HotkeySignatureVerifier,
    rate_limiter: ValidatorRateLimiter,
}

impl ValidatorAccessControl {
    pub fn new(config: AccessControlConfig) -> Self {
        let storage_path = PathBuf::from("/var/lib/basilica/validator_access.json");

        // Initialize hotkey verifier and rate limiter
        let hotkey_verifier = HotkeySignatureVerifier::new(config.hotkey_verification.clone());
        let rate_limiter = ValidatorRateLimiter::new(config.rate_limits.clone());

        let instance = Self {
            config,
            active_sessions: Arc::new(RwLock::new(HashMap::new())),
            storage_path,
            hotkey_verifier,
            rate_limiter,
        };

        // Load existing data on startup (synchronously)
        if let Err(e) = instance.load_from_storage_sync() {
            warn!("Failed to load validator access from storage: {}", e);
        }

        instance
    }

    fn load_from_storage_sync(&self) -> Result<()> {
        if !self.storage_path.exists() {
            return Ok(());
        }

        let content = std::fs::read_to_string(&self.storage_path)?;
        let sessions: HashMap<String, ValidatorAccess> = serde_json::from_str(&content)?;

        // Filter out expired sessions during load
        let now = SystemTime::now();
        let active_sessions: HashMap<String, ValidatorAccess> = sessions
            .into_iter()
            .filter(|(_, access)| !access.is_expired(now))
            .collect();

        if !active_sessions.is_empty() {
            info!(
                "Loaded {} active validator sessions from storage",
                active_sessions.len()
            );
            // Use try_write in sync context during initialization
            match self.active_sessions.try_write() {
                Ok(mut sessions_guard) => {
                    *sessions_guard = active_sessions;
                }
                Err(_) => {
                    warn!("Failed to acquire write lock during storage load");
                }
            }
        }

        Ok(())
    }

    async fn save_to_storage(&self) -> Result<()> {
        let sessions = self.active_sessions.read().await;
        let content = serde_json::to_string_pretty(&*sessions)?;

        // Create parent directory if it doesn't exist
        if let Some(parent) = self.storage_path.parent() {
            std::fs::create_dir_all(parent)?;
        }

        std::fs::write(&self.storage_path, content)?;
        Ok(())
    }

    pub async fn grant_access(&self, access: &ValidatorAccess) -> Result<()> {
        info!("Granting access to validator: {}", access.validator_id);

        {
            let mut sessions = self.active_sessions.write().await;
            sessions.insert(access.validator_id.hotkey.clone(), access.clone());
        }

        // Save to persistent storage
        if let Err(e) = self.save_to_storage().await {
            warn!("Failed to save validator access to storage: {}", e);
        }

        info!(
            "Access granted successfully to validator: {}",
            access.validator_id
        );
        Ok(())
    }

    pub async fn revoke_access(&self, validator_id: &ValidatorId) -> Result<()> {
        info!("Revoking access for validator: {}", validator_id);

        let removed = {
            let mut sessions = self.active_sessions.write().await;
            sessions.remove(&validator_id.hotkey)
        };

        if removed.is_some() {
            // Save to persistent storage
            if let Err(e) = self.save_to_storage().await {
                warn!("Failed to save validator access to storage: {}", e);
            }

            info!(
                "Access revoked successfully for validator: {}",
                validator_id
            );
        } else {
            warn!("No active access found for validator: {}", validator_id);
        }

        Ok(())
    }

    pub async fn has_access(&self, validator_id: &ValidatorId) -> bool {
        let sessions = self.active_sessions.read().await;
        if let Some(access) = sessions.get(&validator_id.hotkey) {
            !access.is_expired(SystemTime::now())
        } else {
            false
        }
    }

    pub async fn get_access(&self, validator_id: &ValidatorId) -> Option<ValidatorAccess> {
        let sessions = self.active_sessions.read().await;
        sessions.get(&validator_id.hotkey).cloned()
    }

    pub async fn list_active_access(&self) -> Result<Vec<ValidatorAccess>> {
        let sessions = self.active_sessions.read().await;
        let now = SystemTime::now();

        Ok(sessions
            .values()
            .filter(|access| !access.is_expired(now))
            .cloned()
            .collect())
    }

    pub async fn authenticate_request(
        &self,
        validator_id: &ValidatorId,
        client_ip: Option<IpAddr>,
        request_type: RequestType,
    ) -> Result<bool> {
        debug!("Authenticating request from validator: {}", validator_id);

        // Check rate limits first
        let rate_check_result = match request_type {
            RequestType::Ssh => self.rate_limiter.check_ssh_request(validator_id).await?,
            RequestType::Api => self.rate_limiter.check_api_request(validator_id).await?,
        };

        if !rate_check_result {
            log_security_violation(
                Some(&validator_id.hotkey),
                "rate_limit_exceeded",
                &format!("Rate limit exceeded for request type: {request_type:?}"),
                client_ip.map(|ip| ip.to_string()).as_deref(),
                SecuritySeverity::Medium,
                HashMap::new(),
            );
            return Ok(false);
        }

        // Check IP whitelist
        if !self.is_ip_allowed(client_ip).await {
            let ip_str = client_ip.map(|ip| ip.to_string());
            log_security_violation(
                Some(&validator_id.hotkey),
                "ip_not_whitelisted",
                &format!("Access denied for IP: {client_ip:?}"),
                ip_str.as_deref(),
                SecuritySeverity::High,
                HashMap::new(),
            );
            return Ok(false);
        }

        // Check active access
        if !self.has_access(validator_id).await {
            let ip_str = client_ip.map(|ip| ip.to_string());
            log_security_violation(
                Some(&validator_id.hotkey),
                "no_active_access",
                "No active access found for validator",
                ip_str.as_deref(),
                SecuritySeverity::High,
                HashMap::new(),
            );
            return Ok(false);
        }

        debug!("Authentication successful for validator: {}", validator_id);
        Ok(true)
    }

    pub async fn authorize_operation(
        &self,
        validator_id: &ValidatorId,
        operation: &str,
    ) -> Result<bool> {
        debug!(
            "Authorizing operation '{}' for validator: {}",
            operation, validator_id
        );

        // Check if validator has active access
        let access = match self.get_access(validator_id).await {
            Some(access) => access,
            None => {
                warn!("No active access found for validator: {}", validator_id);
                return Ok(false);
            }
        };

        // Enhanced role-based authorization
        let has_permission = match operation {
            "ssh_access" => {
                // SSH access requires basic role or higher
                access.role.has_permission("ssh_access")
            }
            "system_info" => {
                // System info requires basic role or higher
                access.role.has_permission("system_info")
            }
            "container_list" => {
                // Container listing requires standard role or higher
                access.role.has_permission("container_list")
            }
            "container_exec" => {
                // Container execution requires admin role or higher
                access.role.has_permission("container_exec")
            }
            "config_change" => {
                // Config changes require owner role
                matches!(access.role, ValidatorRole::Owner)
            }
            "service_control" => {
                // Service control requires owner role
                matches!(access.role, ValidatorRole::Owner)
            }
            _ => {
                // For custom operations, check role assignments from config
                if let Some(role) = self.config.role_assignments.get(&validator_id.hotkey) {
                    role.has_permission(operation)
                } else {
                    // Default to checking if current role has permission
                    access.role.has_permission(operation)
                }
            }
        };

        // Additional check for hotkey verification requirement
        if has_permission && self.config.hotkey_verification.enabled {
            // For sensitive operations, require hotkey verification
            let requires_hotkey_verification = matches!(
                operation,
                "container_exec" | "config_change" | "service_control"
            );

            if requires_hotkey_verification && !access.hotkey_verified {
                warn!(
                    "Operation '{}' requires hotkey verification for validator: {}",
                    operation, validator_id
                );
                return Ok(false);
            }
        }

        if !has_permission {
            log_security_violation(
                Some(&validator_id.hotkey),
                "insufficient_permissions",
                &format!(
                    "Operation '{}' requires role: {:?}, validator has: {:?}",
                    operation,
                    self.get_required_role_for_operation(operation),
                    access.role
                ),
                None,
                SecuritySeverity::Medium,
                HashMap::new(),
            );
        }

        Ok(has_permission)
    }

    /// Get the minimum required role for an operation
    fn get_required_role_for_operation(&self, operation: &str) -> ValidatorRole {
        match operation {
            "ssh_access" | "system_info" => ValidatorRole::Basic,
            "container_list" => ValidatorRole::Standard,
            "container_exec" => ValidatorRole::Admin,
            "config_change" | "service_control" => ValidatorRole::Owner,
            _ => ValidatorRole::Basic,
        }
    }

    pub async fn cleanup_expired_sessions(&self) -> Result<u32> {
        let mut cleaned = 0;
        let now = SystemTime::now();

        let expired_validators: Vec<ValidatorId> = {
            let sessions = self.active_sessions.read().await;
            sessions
                .iter()
                .filter(|(_, access)| access.is_expired(now))
                .map(|(_, access)| access.validator_id.clone())
                .collect()
        };

        for validator_id in expired_validators {
            self.revoke_access(&validator_id).await?;
            cleaned += 1;
        }

        if cleaned > 0 {
            info!("Cleaned up {} expired validator sessions", cleaned);
        }

        Ok(cleaned)
    }

    async fn is_ip_allowed(&self, client_ip: Option<IpAddr>) -> bool {
        if self.config.ip_whitelist.is_empty() {
            // No whitelist configured, allow all
            return true;
        }

        if let Some(ip) = client_ip {
            self.config.ip_whitelist.iter().any(|allowed_ip| {
                if let Ok(allowed) = allowed_ip.parse::<IpAddr>() {
                    ip == allowed
                } else {
                    // Could be a CIDR range, for now just do string comparison
                    ip.to_string() == *allowed_ip
                }
            })
        } else {
            false
        }
    }

    /// Generate hotkey signature challenge for validator authentication
    pub async fn generate_hotkey_challenge(
        &self,
        validator_id: &ValidatorId,
    ) -> Result<SignatureChallenge> {
        info!(
            "Generating hotkey challenge for validator: {}",
            validator_id
        );

        let challenge = self
            .hotkey_verifier
            .generate_challenge(&validator_id.hotkey)
            .await?;

        debug!(
            "Generated challenge {} for validator: {}",
            challenge.challenge_id, validator_id
        );
        Ok(challenge)
    }

    /// Verify hotkey signature challenge response
    pub async fn verify_hotkey_signature(
        &self,
        challenge_id: &str,
        signature_bytes: &[u8],
        public_key_bytes: &[u8],
    ) -> Result<bool> {
        debug!("Verifying hotkey signature for challenge: {}", challenge_id);

        let result = self
            .hotkey_verifier
            .verify_signature(challenge_id, signature_bytes, public_key_bytes)
            .await?;

        if result {
            info!(
                "Hotkey signature verification successful for challenge: {}",
                challenge_id
            );
        } else {
            warn!(
                "Hotkey signature verification failed for challenge: {}",
                challenge_id
            );
        }

        Ok(result)
    }

    /// Grant access with hotkey verification status and role
    pub async fn grant_access_with_verification(
        &self,
        validator_id: &ValidatorId,
        public_key: &str,
        role: ValidatorRole,
        hotkey_verified: bool,
    ) -> Result<()> {
        info!(
            "Granting access to validator: {} with role: {:?}, verified: {}",
            validator_id, role, hotkey_verified
        );

        let access =
            ValidatorAccess::new_with_role(validator_id.clone(), public_key, role, hotkey_verified);

        {
            let mut sessions = self.active_sessions.write().await;
            sessions.insert(validator_id.hotkey.clone(), access);
        }

        // Save to persistent storage
        if let Err(e) = self.save_to_storage().await {
            warn!("Failed to save validator access to storage: {}", e);
        }

        info!("Access granted successfully to validator: {}", validator_id);
        Ok(())
    }

    /// Check if validator has specific role permission
    pub async fn check_role_permission(
        &self,
        validator_id: &ValidatorId,
        permission: &str,
    ) -> bool {
        if let Some(access) = self.get_access(validator_id).await {
            access.role.has_permission(permission)
        } else {
            false
        }
    }

    /// Get rate limit status for validator
    pub async fn get_rate_limit_status(&self, validator_id: &ValidatorId) -> RateLimitStatus {
        self.rate_limiter.get_status(validator_id).await
    }

    /// Reset rate limits for validator (after successful auth)
    pub async fn reset_rate_limits(&self, validator_id: &ValidatorId) {
        self.rate_limiter.reset_limits(validator_id).await;
    }

    /// Clean up expired challenges and rate limit entries
    pub async fn cleanup_expired_data(&self) -> Result<(u32, u32)> {
        let cleaned_sessions = self.cleanup_expired_sessions().await?;
        let cleaned_challenges = self.hotkey_verifier.cleanup_expired().await?;
        let _cleaned_rate_limits = self.rate_limiter.cleanup_old_entries().await?;

        Ok((cleaned_sessions, cleaned_challenges))
    }

    pub async fn get_session_stats(&self) -> SessionStats {
        let sessions = self.active_sessions.read().await;
        let now = SystemTime::now();

        let total_sessions = sessions.len();
        let active_sessions = sessions
            .values()
            .filter(|access| !access.is_expired(now))
            .count();
        let expired_sessions = total_sessions - active_sessions;

        let ssh_sessions = sessions
            .values()
            .filter(|access| !access.is_expired(now) && access.has_ssh_access())
            .count();

        SessionStats {
            total_sessions,
            active_sessions,
            expired_sessions,
            ssh_sessions,
        }
    }
}
