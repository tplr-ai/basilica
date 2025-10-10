//! Secure token storage and management
//!
//! This module provides secure storage for OAuth tokens using file-based storage.

use super::types::{AuthError, AuthResult};
use basilica_sdk::auth::TokenSet;
use std::path::PathBuf;
use std::time::Duration;
use tokio::fs;

const REFRESH_BUFFER_MINUTES: u64 = 5;

/// Secure token storage implementation
pub struct TokenStore {
    auth_file_path: PathBuf,
}

impl TokenStore {
    /// Create a new token store with the provided data directory
    ///
    /// The token file name is determined by the current Auth0 domain:
    /// - Development (matches basilica_common::AUTH0_DOMAIN): auth.dev.json
    /// - Production (different domain via env var): auth.json
    pub async fn new(data_dir: PathBuf) -> AuthResult<Self> {
        fs::create_dir_all(&data_dir).await.map_err(|e| {
            AuthError::StorageError(format!("Failed to create data directory: {}", e))
        })?;

        // Detect environment based on Auth0 domain
        let auth_file_name = if basilica_common::is_development_environment() {
            "auth.dev.json"
        } else {
            "auth.json"
        };

        let auth_file_path = data_dir.join(auth_file_name);

        Ok(Self { auth_file_path })
    }

    /// Check if token needs refresh (with 5 minute buffer)
    pub fn needs_refresh(&self, tokens: &TokenSet) -> bool {
        tokens.expires_within(Duration::from_secs(REFRESH_BUFFER_MINUTES * 60))
    }

    /// Store tokens securely
    pub async fn store_tokens(&self, tokens: &TokenSet) -> AuthResult<()> {
        // Write tokens directly to file
        let json = serde_json::to_string_pretty(tokens)
            .map_err(|e| AuthError::StorageError(format!("Failed to serialize tokens: {}", e)))?;

        fs::write(&self.auth_file_path, json)
            .await
            .map_err(|e| AuthError::StorageError(format!("Failed to write auth file: {}", e)))?;

        Ok(())
    }

    /// Retrieve stored tokens
    pub async fn retrieve_tokens(&self) -> AuthResult<Option<TokenSet>> {
        // Check if file exists
        match fs::try_exists(&self.auth_file_path).await {
            Ok(false) => return Ok(None),
            Err(e) => {
                return Err(AuthError::StorageError(format!(
                    "Failed to check if auth file exists: {}",
                    e
                )))
            }
            Ok(true) => {}
        }

        let content = fs::read_to_string(&self.auth_file_path)
            .await
            .map_err(|e| AuthError::StorageError(format!("Failed to read auth file: {}", e)))?;

        // Parse as TokenSet
        let tokens = serde_json::from_str::<TokenSet>(&content)
            .map_err(|e| AuthError::StorageError(format!("Failed to parse auth file: {}", e)))?;

        Ok(Some(tokens))
    }

    /// Delete stored tokens
    pub async fn delete_tokens(&self) -> AuthResult<()> {
        // Check if file exists before attempting deletion
        if let Ok(true) = fs::try_exists(&self.auth_file_path).await {
            fs::remove_file(&self.auth_file_path).await.map_err(|e| {
                AuthError::StorageError(format!("Failed to delete auth file: {}", e))
            })?;
        }
        Ok(())
    }
}
