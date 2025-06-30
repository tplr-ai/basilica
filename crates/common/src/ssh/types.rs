//! SSH types and data structures

use serde::{Deserialize, Serialize};
use std::time::{Duration, SystemTime};

/// SSH key identifier type
pub type SshKeyId = String;

/// SSH username type
pub type SshUsername = String;

/// SSH key information
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct SshKeyInfo {
    /// Unique identifier for the SSH key
    pub id: SshKeyId,
    /// Public key content
    pub public_key: String,
    /// Private key path (for internal use)
    pub private_key_path: String,
    /// SSH username for the key owner
    pub username: SshUsername,
    /// Key fingerprint
    pub fingerprint: String,
    /// Key creation time
    pub created_at: SystemTime,
    /// Key expiration time
    pub expires_at: SystemTime,
    /// Key algorithm (rsa, ed25519, etc.)
    pub algorithm: SshKeyAlgorithm,
    /// Key size in bits
    pub key_size: u32,
}

impl SshKeyInfo {
    /// Check if the SSH key is expired
    pub fn is_expired(&self, now: SystemTime) -> bool {
        now >= self.expires_at
    }

    /// Get time until expiry
    pub fn time_until_expiry(&self, now: SystemTime) -> Option<Duration> {
        self.expires_at.duration_since(now).ok()
    }

    /// Check if the key is still valid
    pub fn is_valid(&self, now: SystemTime) -> bool {
        now >= self.created_at && now < self.expires_at
    }
}

/// SSH key algorithms
#[derive(Debug, Clone, PartialEq, Eq, Serialize, Deserialize)]
pub enum SshKeyAlgorithm {
    /// RSA algorithm
    Rsa,
    /// Ed25519 algorithm
    Ed25519,
    /// ECDSA algorithm
    Ecdsa,
}

impl std::fmt::Display for SshKeyAlgorithm {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        match self {
            SshKeyAlgorithm::Rsa => write!(f, "rsa"),
            SshKeyAlgorithm::Ed25519 => write!(f, "ed25519"),
            SshKeyAlgorithm::Ecdsa => write!(f, "ecdsa"),
        }
    }
}

impl std::str::FromStr for SshKeyAlgorithm {
    type Err = String;

    fn from_str(s: &str) -> Result<Self, Self::Err> {
        match s.to_lowercase().as_str() {
            "rsa" => Ok(SshKeyAlgorithm::Rsa),
            "ed25519" => Ok(SshKeyAlgorithm::Ed25519),
            "ecdsa" => Ok(SshKeyAlgorithm::Ecdsa),
            _ => Err(format!("Unknown SSH key algorithm: {s}")),
        }
    }
}

/// SSH key generation parameters
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct SshKeyParams {
    /// Key algorithm
    pub algorithm: SshKeyAlgorithm,
    /// Key size in bits
    pub key_size: u32,
    /// Key validity duration
    pub validity_duration: Duration,
    /// Comment for the key
    pub comment: Option<String>,
    /// Passphrase for the private key (None for no passphrase)
    pub passphrase: Option<String>,
}

impl Default for SshKeyParams {
    fn default() -> Self {
        Self {
            algorithm: SshKeyAlgorithm::Rsa,
            key_size: 2048,
            validity_duration: Duration::from_secs(3600), // 1 hour
            comment: None,
            passphrase: None,
        }
    }
}

/// SSH user account information
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct SshUserInfo {
    /// Username
    pub username: SshUsername,
    /// User home directory
    pub home_directory: String,
    /// Shell to use
    pub shell: String,
    /// Additional groups
    pub groups: Vec<String>,
    /// Whether the user is temporary
    pub is_temporary: bool,
    /// User creation time
    pub created_at: SystemTime,
    /// User expiration time
    pub expires_at: Option<SystemTime>,
}

impl SshUserInfo {
    /// Check if user is expired
    pub fn is_expired(&self, now: SystemTime) -> bool {
        self.expires_at.is_some_and(|expires| now >= expires)
    }
}

/// SSH access configuration
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct SshAccessConfig {
    /// Allowed source IP addresses/ranges
    pub allowed_ips: Vec<String>,
    /// Allowed commands (empty = all commands allowed)
    pub allowed_commands: Vec<String>,
    /// SSH port restrictions
    pub port_forwarding_allowed: bool,
    /// X11 forwarding allowed
    pub x11_forwarding_allowed: bool,
    /// Agent forwarding allowed
    pub agent_forwarding_allowed: bool,
    /// TTY allocation allowed
    pub pty_allowed: bool,
    /// Environment variables to set
    pub environment: std::collections::HashMap<String, String>,
}

impl Default for SshAccessConfig {
    fn default() -> Self {
        Self {
            allowed_ips: vec![],
            allowed_commands: vec![],
            port_forwarding_allowed: false,
            x11_forwarding_allowed: false,
            agent_forwarding_allowed: false,
            pty_allowed: true,
            environment: std::collections::HashMap::new(),
        }
    }
}

/// SSH operation result
pub type SshResult<T> = Result<T, SshError>;

/// SSH management errors
#[derive(Debug, thiserror::Error)]
pub enum SshError {
    /// Key generation failed
    #[error("SSH key generation failed: {0}")]
    KeyGenerationFailed(String),

    /// Key installation failed
    #[error("SSH key installation failed: {0}")]
    KeyInstallationFailed(String),

    /// Key revocation failed
    #[error("SSH key revocation failed: {0}")]
    KeyRevocationFailed(String),

    /// User creation failed
    #[error("SSH user creation failed: {0}")]
    UserCreationFailed(String),

    /// User removal failed
    #[error("SSH user removal failed: {0}")]
    UserRemovalFailed(String),

    /// Permission setting failed
    #[error("SSH permission setting failed: {0}")]
    PermissionFailed(String),

    /// Invalid configuration
    #[error("Invalid SSH configuration: {0}")]
    InvalidConfiguration(String),

    /// Key not found
    #[error("SSH key not found: {0}")]
    KeyNotFound(SshKeyId),

    /// User not found
    #[error("SSH user not found: {0}")]
    UserNotFound(SshUsername),

    /// IO error
    #[error("SSH IO error: {0}")]
    IoError(#[from] std::io::Error),

    /// Command execution failed
    #[error("SSH command execution failed: {0}")]
    CommandFailed(String),
}
