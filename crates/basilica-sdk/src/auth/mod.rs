//! Authentication module for Basilica SDK
//!
//! This module provides OAuth 2.0 authentication capabilities including:
//! - PKCE (Proof Key for Code Exchange) browser-based flow
//! - Device authorization flow for headless environments
//! - Secure token storage and management
//! - Automatic token refresh
//! - Support for both direct tokens and file-based authentication

pub mod callback_server;
pub mod device_flow;
pub mod oauth_flow;
pub mod refresh;
pub mod simple_manager;
pub mod token_store;
pub mod types;

// Re-export commonly used types and functions
pub use callback_server::{CallbackData, CallbackServer};
pub use device_flow::{DeviceAuthInstructions, DeviceAuthResponse, DeviceFlow, DeviceFlowPending};
pub use oauth_flow::OAuthFlow;
pub use refresh::refresh_access_token;
pub use simple_manager::TokenManager;
pub use token_store::TokenStore;
pub use types::{get_sdk_data_dir, AuthConfig, AuthError, AuthMethod, AuthResult, TokenSet};

/// Environment detection utilities for determining authentication flow

/// Detect if running in Windows Subsystem for Linux (WSL)
pub fn is_wsl_environment() -> bool {
    std::fs::read_to_string("/proc/version")
        .map(|content| content.contains("Microsoft") || content.contains("WSL"))
        .unwrap_or(false)
}

/// Detect if running in an SSH session
pub fn is_ssh_session() -> bool {
    std::env::var("SSH_CLIENT").is_ok() || std::env::var("SSH_TTY").is_ok()
}

/// Detect if running inside a container runtime
pub fn is_container_runtime() -> bool {
    std::path::Path::new("/.dockerenv").exists()
        || std::path::Path::new("/run/.containerenv").exists()
}

/// Determine if device flow should be used for authentication
///
/// Device flow is preferred when:
/// - Running in WSL environment
/// - Running in SSH session
/// - Running in container
/// - Browser cannot be opened (fallback)
pub fn should_use_device_flow() -> bool {
    is_wsl_environment() || is_ssh_session() || is_container_runtime()
}

/// Create default auth configuration for Basilica
pub fn create_default_auth_config() -> AuthConfig {
    create_auth_config_with_port(0)
}

/// Create auth configuration with a specific callback port
pub fn create_auth_config_with_port(port: u16) -> AuthConfig {
    AuthConfig {
        client_id: basilica_common::auth0_client_id().to_string(),
        auth_endpoint: format!("{}/authorize", basilica_common::auth0_issuer()),
        token_endpoint: format!("{}/oauth/token", basilica_common::auth0_issuer()),
        device_auth_endpoint: Some(format!(
            "{}/oauth/device/code",
            basilica_common::auth0_issuer()
        )),
        revoke_endpoint: Some(format!("{}/oauth/revoke", basilica_common::auth0_issuer())),
        redirect_uri: format!("http://localhost:{}/callback", port),
        scopes: vec![
            "openid".to_string(),
            "profile".to_string(),
            "email".to_string(),
            "offline_access".to_string(),
        ],
        additional_params: std::collections::HashMap::new(),
    }
}
