//! Simplified token management with automatic refresh
//!
//! This module provides a clean and simple token manager that handles
//! both direct token provision and file-based token storage.

use super::refresh::refresh_access_token;
use super::token_store::TokenStore;
use super::types::{get_sdk_data_dir, AuthError, AuthMethod, AuthResult, TokenSet};
use std::sync::Arc;
use std::time::Duration;
use tokio::sync::Mutex;
use tracing::{debug, info};

/// Manages tokens with automatic refresh
#[derive(Debug)]
pub struct TokenManager {
    auth_method: Arc<Mutex<AuthMethod>>,
    api_key: Option<String>,
}

impl TokenManager {
    /// Pre-emptive refresh threshold (60 minutes before expiry)
    const REFRESH_THRESHOLD: Duration = Duration::from_secs(3600);

    /// Create a new token manager with direct tokens
    pub fn new_direct(access_token: String, refresh_token: String) -> Self {
        let tokens = TokenSet::new(access_token, refresh_token);
        let auth_method = AuthMethod::Direct { tokens };

        Self {
            auth_method: Arc::new(Mutex::new(auth_method)),
            api_key: None,
        }
    }

    /// Create a new token manager with file-based authentication
    pub fn new_file_based() -> AuthResult<Self> {
        // Check for API key in environment variable first
        let api_key = std::env::var("BASILICA_API_TOKEN").ok();

        let data_dir = get_sdk_data_dir()?;
        let store = TokenStore::new(data_dir)?;
        let auth_method = AuthMethod::FileBased { store };

        Ok(Self {
            auth_method: Arc::new(Mutex::new(auth_method)),
            api_key,
        })
    }

    /// Create a new token manager with API key authentication
    pub fn new_api_key(api_key: String) -> Self {
        Self {
            auth_method: Arc::new(Mutex::new(AuthMethod::Direct {
                tokens: TokenSet::new(String::new(), String::new()),
            })),
            api_key: Some(api_key),
        }
    }

    /// Get valid access token (handles refresh automatically)
    pub async fn get_access_token(&self) -> AuthResult<String> {
        debug!("Getting access token from TokenManager");

        // If API key is set, return it directly
        if let Some(api_key) = &self.api_key {
            debug!("Using API key authentication");
            return Ok(api_key.clone());
        }

        let mut auth_method = self.auth_method.lock().await;

        match &mut *auth_method {
            AuthMethod::Direct { tokens } => {
                // Check if token needs refresh
                if self.should_refresh(tokens) {
                    debug!("Direct token needs refresh");
                    let new_tokens =
                        refresh_access_token(&tokens.refresh_token, None, None).await?;
                    info!("Token refreshed successfully");
                    *tokens = new_tokens.clone();
                    Ok(new_tokens.access_token)
                } else {
                    debug!("Using current direct token");
                    Ok(tokens.access_token.clone())
                }
            }
            AuthMethod::FileBased { store } => {
                // Read tokens from file
                let stored_tokens = store.retrieve().await?.ok_or(AuthError::UserNotLoggedIn)?;

                // Check if token needs refresh
                if self.should_refresh(&stored_tokens) {
                    debug!("File-based token needs refresh");
                    let new_tokens =
                        refresh_access_token(&stored_tokens.refresh_token, None, None).await?;
                    info!("Token refreshed successfully");

                    // Store the new tokens
                    store.store(&new_tokens).await?;
                    Ok(new_tokens.access_token)
                } else {
                    debug!("Using stored token from file");
                    Ok(stored_tokens.access_token)
                }
            }
        }
    }

    /// Check if token should be refreshed
    fn should_refresh(&self, token_set: &TokenSet) -> bool {
        if token_set.is_expired() {
            return true;
        }

        // Pre-emptive refresh if expiring within threshold
        token_set.expires_within(Self::REFRESH_THRESHOLD)
    }
}
