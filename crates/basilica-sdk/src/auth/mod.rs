//! Authentication module for Basilica SDK
//!
//! This module provides OAuth 2.0 authentication capabilities including:
//! - Secure token storage and management
//! - Automatic token refresh
//! - Support for both direct tokens and file-based authentication

pub mod refresh;
pub mod simple_manager;
pub mod token_store;
pub mod types;
pub mod wallet;

// Re-export commonly used types and functions
pub use refresh::refresh_access_token;
pub use simple_manager::TokenManager;
pub use token_store::TokenStore;
pub use types::{AuthConfig, AuthError, AuthMethod, AuthResult, TokenSet};
