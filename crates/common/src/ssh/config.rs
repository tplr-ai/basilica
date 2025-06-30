//! SSH configuration structures and implementations

use super::traits::SshConfigProvider;
use super::types::SshKeyAlgorithm;
use serde::{Deserialize, Serialize};

/// SSH configuration
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct SshConfig {
    /// SSH key directory
    pub key_directory: String,
    /// Default SSH key algorithm
    pub default_algorithm: SshKeyAlgorithm,
    /// Default SSH key size in bits
    pub default_key_size: u32,
    /// SSH username prefix
    pub username_prefix: String,
    /// Default shell for SSH users
    pub default_shell: String,
    /// Default groups for SSH users
    pub default_groups: Vec<String>,
    /// SSH port range for allocation
    pub port_range: (u16, u16),
    /// Maximum concurrent SSH connections
    pub max_connections: u32,
    /// SSH daemon configuration file path
    pub sshd_config_path: String,
    /// Enable strict host key checking
    pub strict_host_key_checking: bool,
    /// SSH connection timeout in seconds
    pub connection_timeout: u32,
    /// Enable SSH key rotation
    pub enable_key_rotation: bool,
    /// Key rotation interval in hours
    pub key_rotation_interval: u32,
}

impl Default for SshConfig {
    fn default() -> Self {
        Self {
            key_directory: "./ssh_keys".to_string(),
            default_algorithm: SshKeyAlgorithm::Rsa,
            default_key_size: 2048,
            username_prefix: "basilica_user".to_string(),
            default_shell: "/bin/rbash".to_string(),
            default_groups: vec!["docker".to_string()],
            port_range: (2200, 2299),
            max_connections: 10,
            sshd_config_path: "/etc/ssh/sshd_config".to_string(),
            strict_host_key_checking: true,
            connection_timeout: 30,
            enable_key_rotation: false,
            key_rotation_interval: 24,
        }
    }
}

impl SshConfig {
    /// Validate SSH configuration
    pub fn validate(&self) -> Result<(), String> {
        // Validate key directory
        if self.key_directory.is_empty() {
            return Err("SSH key directory cannot be empty".to_string());
        }

        // Validate key size
        match self.default_algorithm {
            SshKeyAlgorithm::Rsa => {
                if self.default_key_size < 1024 || self.default_key_size > 8192 {
                    return Err("RSA key size must be between 1024 and 8192 bits".to_string());
                }
            }
            SshKeyAlgorithm::Ed25519 => {
                if self.default_key_size != 256 {
                    return Err("Ed25519 key size must be 256 bits".to_string());
                }
            }
            SshKeyAlgorithm::Ecdsa => {
                if ![256, 384, 521].contains(&self.default_key_size) {
                    return Err("ECDSA key size must be 256, 384, or 521 bits".to_string());
                }
            }
        }

        // Validate username prefix
        if self.username_prefix.is_empty() {
            return Err("Username prefix cannot be empty".to_string());
        }

        // Validate username prefix format
        if !self
            .username_prefix
            .chars()
            .all(|c| c.is_alphanumeric() || c == '_')
        {
            return Err(
                "Username prefix can only contain alphanumeric characters and underscores"
                    .to_string(),
            );
        }

        // Validate port range
        if self.port_range.0 >= self.port_range.1 {
            return Err("Invalid port range: start port must be less than end port".to_string());
        }

        if self.port_range.0 < 1024 {
            return Err("Port range start must be >= 1024".to_string());
        }

        // Validate max connections
        if self.max_connections == 0 {
            return Err("Maximum connections must be greater than 0".to_string());
        }

        // Validate connection timeout
        if self.connection_timeout == 0 {
            return Err("Connection timeout must be greater than 0".to_string());
        }

        // Validate key rotation interval
        if self.enable_key_rotation && self.key_rotation_interval == 0 {
            return Err(
                "Key rotation interval must be greater than 0 when rotation is enabled".to_string(),
            );
        }

        Ok(())
    }

    /// Get configuration warnings
    pub fn warnings(&self) -> Vec<String> {
        let mut warnings = Vec::new();

        // Warn about insecure key sizes
        if self.default_algorithm == SshKeyAlgorithm::Rsa && self.default_key_size < 2048 {
            warnings.push(
                "RSA key size less than 2048 bits is not recommended for security".to_string(),
            );
        }

        // Warn about default key directory
        if self.key_directory == "/tmp/basilica_ssh_keys" {
            warnings.push(
                "Using temporary directory for SSH keys - keys will be lost on reboot".to_string(),
            );
        }

        // Warn about high connection limits
        if self.max_connections > 50 {
            warnings
                .push("High maximum connection limit may impact system performance".to_string());
        }

        // Warn about disabled strict host key checking
        if !self.strict_host_key_checking {
            warnings.push("Disabled strict host key checking reduces security".to_string());
        }

        // Warn about long connection timeout
        if self.connection_timeout > 300 {
            warnings.push("Long connection timeout may allow resource exhaustion".to_string());
        }

        // Warn about disabled key rotation
        if !self.enable_key_rotation {
            warnings.push(
                "Key rotation is disabled - consider enabling for better security".to_string(),
            );
        }

        warnings
    }
}

impl SshConfigProvider for SshConfig {
    fn get_key_directory(&self) -> &str {
        &self.key_directory
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
        self.port_range
    }

    fn get_max_connections(&self) -> u32 {
        self.max_connections
    }
}

/// SSH security configuration
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct SshSecurityConfig {
    /// Enable security hardening
    pub enable_hardening: bool,
    /// Use restricted shell by default
    pub use_restricted_shell: bool,
    /// Enable no-new-privileges
    pub enable_no_new_privileges: bool,
    /// Enable seccomp filtering
    pub enable_seccomp: bool,
    /// Allowed SSH ciphers
    pub allowed_ciphers: Vec<String>,
    /// Allowed SSH MACs
    pub allowed_macs: Vec<String>,
    /// Allowed SSH key exchange algorithms
    pub allowed_kex_algorithms: Vec<String>,
    /// Disable password authentication
    pub disable_password_auth: bool,
    /// Disable root login
    pub disable_root_login: bool,
    /// Enable SSH banner
    pub enable_banner: bool,
    /// SSH banner message
    pub banner_message: Option<String>,
}

impl Default for SshSecurityConfig {
    fn default() -> Self {
        Self {
            enable_hardening: true,
            use_restricted_shell: true,
            enable_no_new_privileges: true,
            enable_seccomp: true,
            allowed_ciphers: vec![
                "chacha20-poly1305@openssh.com".to_string(),
                "aes256-gcm@openssh.com".to_string(),
                "aes128-gcm@openssh.com".to_string(),
                "aes256-ctr".to_string(),
                "aes192-ctr".to_string(),
                "aes128-ctr".to_string(),
            ],
            allowed_macs: vec![
                "umac-128-etm@openssh.com".to_string(),
                "hmac-sha2-256-etm@openssh.com".to_string(),
                "hmac-sha2-512-etm@openssh.com".to_string(),
            ],
            allowed_kex_algorithms: vec![
                "curve25519-sha256".to_string(),
                "curve25519-sha256@libssh.org".to_string(),
                "ecdh-sha2-nistp256".to_string(),
                "ecdh-sha2-nistp384".to_string(),
                "ecdh-sha2-nistp521".to_string(),
                "diffie-hellman-group14-sha256".to_string(),
                "diffie-hellman-group16-sha512".to_string(),
            ],
            disable_password_auth: true,
            disable_root_login: true,
            enable_banner: true,
            banner_message: Some(
                "Authorized access only. All activities are monitored and logged.".to_string(),
            ),
        }
    }
}
