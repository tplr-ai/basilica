//! Validation session types

use crate::validation_session::hotkey_verifier::HotkeyVerificationConfig;
use common::ssh::SshKeyInfo;
use serde::{Deserialize, Serialize};
use std::collections::HashMap;
use std::time::{Duration, SystemTime};

/// Role-based permissions for validator operations
#[derive(Debug, Clone, PartialEq, Eq, Serialize, Deserialize, Default)]
pub enum ValidatorRole {
    /// Basic access - SSH and system info only
    #[default]
    Basic,
    /// Standard access - includes container listing
    Standard,
    /// Admin access - includes container execution
    Admin,
    /// Owner access - full privileges
    Owner,
}

impl ValidatorRole {
    /// Get permissions for this role
    pub fn permissions(&self) -> Vec<&'static str> {
        match self {
            ValidatorRole::Basic => vec!["ssh_access", "system_info"],
            ValidatorRole::Standard => vec!["ssh_access", "system_info", "container_list"],
            ValidatorRole::Admin => vec![
                "ssh_access",
                "system_info",
                "container_list",
                "container_exec",
            ],
            ValidatorRole::Owner => vec!["all"],
        }
    }

    /// Check if role has specific permission
    pub fn has_permission(&self, permission: &str) -> bool {
        match self {
            ValidatorRole::Owner => true, // Owner has all permissions
            _ => self.permissions().contains(&permission),
        }
    }
}

/// Rate limiting configuration per validator
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct RateLimitConfig {
    pub ssh_requests_per_minute: u32,
    pub api_requests_per_minute: u32,
    pub burst_allowance: u32,
    pub rate_limit_window_seconds: u64,
}

impl Default for RateLimitConfig {
    fn default() -> Self {
        Self {
            ssh_requests_per_minute: 10,
            api_requests_per_minute: 100,
            burst_allowance: 20,
            rate_limit_window_seconds: 60,
        }
    }
}

#[derive(Debug, Clone, PartialEq, Eq, Hash, Serialize, Deserialize)]
pub struct ValidatorId {
    pub hotkey: String,
    pub label: Option<String>,
}

impl ValidatorId {
    pub fn new(hotkey: String) -> Self {
        Self {
            hotkey,
            label: None,
        }
    }

    pub fn with_label(hotkey: String, label: String) -> Self {
        Self {
            hotkey,
            label: Some(label),
        }
    }
}

impl std::fmt::Display for ValidatorId {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        if let Some(label) = &self.label {
            write!(f, "{}({})", self.hotkey, label)
        } else {
            write!(f, "{}", self.hotkey)
        }
    }
}

#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct ValidatorAccess {
    pub validator_id: ValidatorId,
    pub ssh_key: SshKeyInfo,
    pub granted_at: SystemTime,
    pub expires_at: SystemTime,
    pub metadata: HashMap<String, String>,
    pub role: ValidatorRole,
    pub hotkey_verified: bool,
}

impl ValidatorAccess {
    pub fn new(validator_id: ValidatorId, public_key: &str) -> Self {
        let now = SystemTime::now();
        let far_future = now + Duration::from_secs(365 * 24 * 3600); // 1 year

        let ssh_key = SshKeyInfo {
            id: format!("validator_{}", validator_id.hotkey),
            public_key: public_key.to_string(),
            private_key_path: String::new(),
            username: format!("validator_{}", validator_id.hotkey),
            fingerprint: String::new(),
            created_at: now,
            expires_at: far_future,
            algorithm: common::ssh::SshKeyAlgorithm::Ed25519,
            key_size: 256,
        };

        Self {
            validator_id,
            ssh_key,
            granted_at: now,
            expires_at: far_future,
            metadata: HashMap::new(),
            role: ValidatorRole::default(),
            hotkey_verified: false,
        }
    }

    /// Create new access with specific role and hotkey verification status
    pub fn new_with_role(
        validator_id: ValidatorId,
        public_key: &str,
        role: ValidatorRole,
        hotkey_verified: bool,
    ) -> Self {
        let mut access = Self::new(validator_id, public_key);
        access.role = role;
        access.hotkey_verified = hotkey_verified;
        access
    }

    pub fn is_expired(&self, now: SystemTime) -> bool {
        now >= self.expires_at
    }

    pub fn time_until_expiry(&self, now: SystemTime) -> Option<Duration> {
        self.expires_at.duration_since(now).ok()
    }

    pub fn has_ssh_access(&self) -> bool {
        true
    }
}

#[derive(Debug, Clone, Serialize, Deserialize, Default)]
pub struct AccessControlConfig {
    pub ip_whitelist: Vec<String>,
    pub required_permissions: HashMap<String, Vec<String>>,
    pub hotkey_verification: HotkeyVerificationConfig,
    pub rate_limits: RateLimitConfig,
    pub role_assignments: HashMap<String, ValidatorRole>,
}

#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct ValidatorConfig {
    pub access_config: AccessControlConfig,
    pub enabled: bool,
    pub strict_ssh_restrictions: bool,
}

impl Default for ValidatorConfig {
    fn default() -> Self {
        Self {
            access_config: AccessControlConfig::default(),
            enabled: true,
            strict_ssh_restrictions: false,
        }
    }
}

#[derive(Debug, Clone)]
pub struct SessionStats {
    pub total_sessions: usize,
    pub active_sessions: usize,
    pub expired_sessions: usize,
    pub ssh_sessions: usize,
}
