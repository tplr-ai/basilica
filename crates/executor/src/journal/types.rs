//! Executor-specific journal types
//!
//! This module provides executor-specific types for structured logging.

use serde::{Deserialize, Serialize};
use std::collections::HashMap;
use std::time::SystemTime;

#[derive(Debug, Clone, PartialEq, Eq, Hash, Serialize, Deserialize)]
pub struct ValidatorId {
    pub hotkey: String,
}

impl ValidatorId {
    pub fn new(hotkey: String) -> Self {
        Self { hotkey }
    }

    pub fn as_str(&self) -> &str {
        &self.hotkey
    }
}

impl std::fmt::Display for ValidatorId {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        write!(f, "{}", self.hotkey)
    }
}

#[derive(Debug, Clone, PartialEq, Eq, Serialize, Deserialize)]
pub enum AccessType {
    Ssh,
    Token,
    Container,
    Monitoring,
    Full,
}

impl std::fmt::Display for AccessType {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        match self {
            Self::Ssh => write!(f, "ssh"),
            Self::Token => write!(f, "token"),
            Self::Container => write!(f, "container"),
            Self::Monitoring => write!(f, "monitoring"),
            Self::Full => write!(f, "full"),
        }
    }
}

#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct ValidatorAccess {
    pub validator_id: ValidatorId,
    pub access_type: AccessType,
    pub granted_at: SystemTime,
    pub expires_at: SystemTime,
    pub permissions: Vec<String>,
    pub metadata: HashMap<String, String>,
}

impl ValidatorAccess {
    pub fn new(
        validator_id: ValidatorId,
        access_type: AccessType,
        duration: std::time::Duration,
        ssh_key: Option<common::ssh::SshKeyInfo>,
    ) -> Self {
        let now = SystemTime::now();
        let mut permissions = Vec::new();
        let mut metadata = HashMap::new();

        // Set permissions based on access type
        match access_type {
            AccessType::Ssh => {
                permissions.push("ssh".to_string());
            }
            AccessType::Token => {
                permissions.push("api_access".to_string());
            }
            AccessType::Container => {
                permissions.extend_from_slice(&[
                    "container_read".to_string(),
                    "container_write".to_string(),
                ]);
            }
            AccessType::Monitoring => {
                permissions.push("monitoring".to_string());
            }
            AccessType::Full => {
                permissions.extend_from_slice(&[
                    "ssh".to_string(),
                    "api_access".to_string(),
                    "container_read".to_string(),
                    "container_write".to_string(),
                    "monitoring".to_string(),
                ]);
            }
        }

        // Add SSH key metadata if provided
        if let Some(key) = ssh_key {
            metadata.insert("ssh_username".to_string(), key.username);
            metadata.insert("ssh_public_key".to_string(), key.public_key);
        }

        Self {
            validator_id,
            access_type,
            granted_at: now,
            expires_at: now + duration,
            permissions,
            metadata,
        }
    }

    pub fn is_expired(&self, now: SystemTime) -> bool {
        now >= self.expires_at
    }

    pub fn has_ssh_access(&self) -> bool {
        self.permissions.contains(&"ssh".to_string())
    }

    pub fn has_token_access(&self) -> bool {
        self.permissions.contains(&"api_access".to_string())
    }
}
