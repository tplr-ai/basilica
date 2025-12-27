//! Authentication module for Basilica CLI
//!
//! This module re-exports auth types from the SDK and provides CLI-specific
//! wrappers with terminal UI feedback.

pub mod token_store;
pub mod types;

// Re-export auth types from SDK
pub use basilica_sdk::auth::{
    is_container_runtime, is_ssh_session, is_wsl_environment, should_use_device_flow,
    CallbackServer, DeviceAuthInstructions, DeviceFlow, OAuthFlow, TokenSet,
};

// Re-export CLI-specific types
pub use token_store::TokenStore;
pub use types::{AuthConfig, AuthError, AuthResult};
