//! SSH management traits following SOLID principles

use super::types::{
    SshAccessConfig, SshKeyId, SshKeyInfo, SshKeyParams, SshResult, SshUserInfo, SshUsername,
};
use async_trait::async_trait;
use std::time::Duration;

/// Core SSH key management operations (Single Responsibility)
#[async_trait]
pub trait SshKeyManager: Send + Sync {
    /// Generate a new SSH key pair
    async fn generate_key(&self, id: &SshKeyId, params: SshKeyParams) -> SshResult<SshKeyInfo>;

    /// Install SSH public key for user access
    async fn install_key(&self, key_info: &SshKeyInfo, username: &SshUsername) -> SshResult<()>;

    /// Revoke SSH key access
    async fn revoke_key(&self, id: &SshKeyId) -> SshResult<()>;

    /// Get SSH key information
    async fn get_key(&self, id: &SshKeyId) -> SshResult<Option<SshKeyInfo>>;

    /// List all active SSH keys
    async fn list_keys(&self) -> SshResult<Vec<SshKeyInfo>>;

    /// Check if SSH key exists
    async fn key_exists(&self, id: &SshKeyId) -> SshResult<bool>;

    /// Clean up expired keys
    async fn cleanup_expired_keys(&self) -> SshResult<u32>;

    /// Authorize a provided SSH public key for access
    async fn authorize_key(
        &self,
        id: &SshKeyId,
        public_key: &str,
        duration: Duration,
    ) -> SshResult<SshKeyInfo>;
}

/// SSH user account management operations (Single Responsibility)
#[async_trait]
pub trait SshUserManager: Send + Sync {
    /// Create SSH user account
    async fn create_user(&self, user_info: &SshUserInfo) -> SshResult<()>;

    /// Remove SSH user account
    async fn remove_user(&self, username: &SshUsername) -> SshResult<()>;

    /// Get user information
    async fn get_user(&self, username: &SshUsername) -> SshResult<Option<SshUserInfo>>;

    /// Check if user exists
    async fn user_exists(&self, username: &SshUsername) -> SshResult<bool>;

    /// List all SSH users
    async fn list_users(&self) -> SshResult<Vec<SshUserInfo>>;

    /// Update user configuration
    async fn update_user(&self, username: &SshUsername, user_info: &SshUserInfo) -> SshResult<()>;

    /// Clean up expired users
    async fn cleanup_expired_users(&self) -> SshResult<u32>;
}

/// SSH access control operations (Single Responsibility)
#[async_trait]
pub trait SshAccessController: Send + Sync {
    /// Configure SSH access for a user
    async fn configure_access(
        &self,
        username: &SshUsername,
        config: &SshAccessConfig,
    ) -> SshResult<()>;

    /// Remove SSH access configuration
    async fn remove_access(&self, username: &SshUsername) -> SshResult<()>;

    /// Get access configuration for user
    async fn get_access_config(&self, username: &SshUsername)
        -> SshResult<Option<SshAccessConfig>>;

    /// Validate access permissions
    async fn validate_access(
        &self,
        username: &SshUsername,
        source_ip: Option<&str>,
        command: Option<&str>,
    ) -> SshResult<bool>;
}

/// SSH system operations (Single Responsibility)
#[async_trait]
pub trait SshSystemManager: Send + Sync {
    /// Set file/directory permissions
    async fn set_permissions(&self, path: &str, mode: u32) -> SshResult<()>;

    /// Set file/directory ownership
    async fn set_ownership(&self, path: &str, username: &SshUsername) -> SshResult<()>;

    /// Execute system command
    async fn execute_command(&self, command: &str, args: &[&str]) -> SshResult<String>;

    /// Check if command exists
    async fn command_exists(&self, command: &str) -> SshResult<bool>;

    /// Get system SSH configuration
    async fn get_ssh_config(&self) -> SshResult<std::collections::HashMap<String, String>>;
}

/// Audit logging for SSH operations (Single Responsibility)
#[async_trait]
pub trait SshAuditLogger: Send + Sync {
    /// Log SSH key generation
    async fn log_key_generated(&self, key_id: &SshKeyId, username: &SshUsername) -> SshResult<()>;

    /// Log SSH key revocation
    async fn log_key_revoked(&self, key_id: &SshKeyId, username: &SshUsername) -> SshResult<()>;

    /// Log user creation
    async fn log_user_created(&self, username: &SshUsername) -> SshResult<()>;

    /// Log user removal
    async fn log_user_removed(&self, username: &SshUsername) -> SshResult<()>;

    /// Log access configuration
    async fn log_access_configured(&self, username: &SshUsername) -> SshResult<()>;

    /// Log security violation
    async fn log_security_violation(
        &self,
        username: &SshUsername,
        violation: &str,
        details: &str,
    ) -> SshResult<()>;
}

/// Comprehensive SSH service combining all management aspects (Facade Pattern)
#[async_trait]
pub trait SshService: Send + Sync {
    /// Provision complete SSH access (user + key + access config)
    async fn provision_access(
        &self,
        id: &SshKeyId,
        username: &SshUsername,
        key_params: SshKeyParams,
        user_info: &SshUserInfo,
        access_config: &SshAccessConfig,
        duration: Duration,
    ) -> SshResult<SshKeyInfo>;

    /// Revoke complete SSH access
    async fn revoke_access(&self, id: &SshKeyId, username: &SshUsername) -> SshResult<()>;

    /// Get complete access information
    async fn get_access_info(
        &self,
        id: &SshKeyId,
    ) -> SshResult<Option<(SshKeyInfo, SshUserInfo, SshAccessConfig)>>;

    /// List all active SSH access
    async fn list_access(&self) -> SshResult<Vec<(SshKeyInfo, SshUserInfo)>>;

    /// Perform health check on SSH system
    async fn health_check(&self) -> SshResult<SshHealthStatus>;

    /// Clean up all expired SSH resources
    async fn cleanup_expired(&self) -> SshResult<SshCleanupStats>;
}

/// SSH health status
#[derive(Debug, Clone)]
pub struct SshHealthStatus {
    pub ssh_daemon_running: bool,
    pub key_directory_accessible: bool,
    pub user_management_available: bool,
    pub total_active_keys: u32,
    pub total_active_users: u32,
    pub expired_keys: u32,
    pub expired_users: u32,
}

/// SSH cleanup statistics
#[derive(Debug, Clone)]
pub struct SshCleanupStats {
    pub cleaned_keys: u32,
    pub cleaned_users: u32,
    pub errors_encountered: u32,
}

/// SSH configuration provider trait (Dependency Inversion)
pub trait SshConfigProvider: Send + Sync {
    /// Get SSH key directory
    fn get_key_directory(&self) -> &str;

    /// Get default key algorithm
    fn get_default_algorithm(&self) -> super::types::SshKeyAlgorithm;

    /// Get default key size
    fn get_default_key_size(&self) -> u32;

    /// Get username prefix
    fn get_username_prefix(&self) -> &str;

    /// Get default shell
    fn get_default_shell(&self) -> &str;

    /// Get default user groups
    fn get_default_groups(&self) -> &[String];

    /// Get SSH port range
    fn get_port_range(&self) -> (u16, u16);

    /// Get maximum concurrent connections
    fn get_max_connections(&self) -> u32;
}

/// Factory trait for creating SSH managers (Abstract Factory Pattern)
pub trait SshManagerFactory: Send + Sync {
    type KeyManager: SshKeyManager;
    type UserManager: SshUserManager;
    type AccessController: SshAccessController;
    type SystemManager: SshSystemManager;
    type AuditLogger: SshAuditLogger;
    type Service: SshService;

    /// Create SSH key manager
    fn create_key_manager(&self) -> SshResult<Self::KeyManager>;

    /// Create SSH user manager
    fn create_user_manager(&self) -> SshResult<Self::UserManager>;

    /// Create SSH access controller
    fn create_access_controller(&self) -> SshResult<Self::AccessController>;

    /// Create SSH system manager
    fn create_system_manager(&self) -> SshResult<Self::SystemManager>;

    /// Create SSH audit logger
    fn create_audit_logger(&self) -> SshResult<Self::AuditLogger>;

    /// Create complete SSH service
    fn create_service(&self) -> SshResult<Self::Service>;
}
