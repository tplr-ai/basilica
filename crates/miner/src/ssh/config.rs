//! SSH configuration for miner operations

use common::ssh::{SshConfigProvider, SshKeyAlgorithm};
use serde::{Deserialize, Serialize};
use std::path::PathBuf;

/// Miner-specific SSH configuration
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct MinerSshConfig {
    /// Directory to store SSH keys
    pub key_directory: PathBuf,
    /// Default session timeout in seconds
    pub default_session_timeout: u64,
    /// Maximum session timeout in seconds
    pub max_session_timeout: u64,
    /// Cleanup interval in seconds
    pub cleanup_interval: u64,
    /// Username prefix for validator sessions
    pub username_prefix: String,
    /// Default SSH key algorithm
    pub default_algorithm: SshKeyAlgorithm,
    /// Default SSH key size
    pub default_key_size: u32,
    /// Default shell for SSH users
    pub default_shell: String,
    /// Default groups for SSH users
    pub default_groups: Vec<String>,
}

impl Default for MinerSshConfig {
    fn default() -> Self {
        Self {
            key_directory: PathBuf::from("/opt/basilica/ssh_keys"),
            default_session_timeout: 3600, // 1 hour
            max_session_timeout: 86400,    // 24 hours
            cleanup_interval: 300,         // 5 minutes
            username_prefix: "validator".to_string(),
            default_algorithm: SshKeyAlgorithm::Ed25519,
            default_key_size: 256,
            default_shell: "/bin/bash".to_string(),
            default_groups: vec![],
        }
    }
}

impl SshConfigProvider for MinerSshConfig {
    fn get_key_directory(&self) -> &str {
        self.key_directory.to_str().unwrap_or("/tmp/ssh_keys")
    }

    fn get_default_algorithm(&self) -> SshKeyAlgorithm {
        self.default_algorithm.clone()
    }

    fn get_default_key_size(&self) -> u32 {
        self.default_key_size
    }

    fn get_username_prefix(&self) -> &str {
        &self.username_prefix
    }

    fn get_default_shell(&self) -> &str {
        &self.default_shell
    }

    fn get_default_groups(&self) -> &[String] {
        &self.default_groups
    }

    fn get_port_range(&self) -> (u16, u16) {
        (2200, 2299) // Port range for dynamic SSH allocation
    }

    fn get_max_connections(&self) -> u32 {
        100
    }
}

impl From<MinerSshConfig> for common::ssh::SshConfig {
    fn from(val: MinerSshConfig) -> Self {
        common::ssh::SshConfig {
            key_directory: val.key_directory.to_string_lossy().to_string(),
            default_algorithm: val.default_algorithm,
            default_key_size: val.default_key_size,
            username_prefix: val.username_prefix,
            default_shell: val.default_shell,
            default_groups: val.default_groups,
            port_range: (2200, 2299),
            max_connections: 100,
            sshd_config_path: "/etc/ssh/sshd_config".to_string(),
            strict_host_key_checking: false,
            connection_timeout: 30,
            enable_key_rotation: false,
            key_rotation_interval: 24,
        }
    }
}
