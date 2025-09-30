//! CLI-specific client creation and authentication management
//!
//! This module handles creating authenticated BasilicaClient instances specifically
//! for CLI usage, including JWT token retrieval, refresh, and fallback authentication.
//!
//! This is distinct from the general HTTP client library in basilica-api/src/client.rs
//! which provides the underlying HTTP client functionality.

use std::time::Duration;

use crate::auth::{AuthError, OAuthFlow, TokenStore};
use crate::config::CliConfig;
use crate::error::{CliError, Result};
use basilica_sdk::{BasilicaClient, ClientBuilder};
use color_eyre::eyre::{eyre, Context};
use tracing::{debug, warn};

/// Creates an authenticated BasilicaClient with JWT
///
/// This function:
/// 1. Attempts to use JWT tokens from TokenStore
/// 2. Refreshes expired tokens if possible
///
/// # Arguments
/// * `config` - CLI configuration
pub async fn create_authenticated_client(config: &CliConfig) -> Result<BasilicaClient> {
    let api_url = config.api.base_url.clone();

    let mut builder = ClientBuilder::default()
        .base_url(api_url)
        .timeout(Duration::from_secs(config.api.request_timeout));

    // Use JWT authentication with token manager support
    if let Ok(tokens) = get_valid_jwt_tokens(config).await {
        debug!("Using JWT authentication with automatic token refresh");
        builder = builder.with_tokens(tokens.access_token, tokens.refresh_token);
    } else {
        // Provide a more helpful error message using our custom error type
        return Err(CliError::from(AuthError::UserNotLoggedIn));
    }

    builder
        .build()
        .map_err(|e| eyre!("Failed to build client: {}", e).into())
}

/// Alias for create_authenticated_client for backward compatibility
pub async fn create_client(config: &CliConfig) -> Result<BasilicaClient> {
    create_authenticated_client(config).await
}

/// Gets valid JWT tokens with pre-emptive refresh
///
/// This function checks if the stored token needs refresh and refreshes it
/// before returning, ensuring the API client always gets valid tokens.
async fn get_valid_jwt_tokens(_config: &CliConfig) -> Result<basilica_sdk::auth::TokenSet> {
    let data_dir = CliConfig::data_dir().wrap_err("Failed to get data directory")?;
    let token_store = TokenStore::new(data_dir).wrap_err("Failed to initialize token store")?;

    // Try to get stored tokens
    let mut tokens = token_store
        .retrieve()
        .await
        .wrap_err("Failed to retrieve authentication tokens")?
        .ok_or_else(|| CliError::from(AuthError::UserNotLoggedIn))?;

    if tokens.needs_refresh() {
        debug!("Token needs refresh, attempting to refresh pre-emptively");

        // refresh_token is now always present (not optional)
        {
            let refresh_token = &tokens.refresh_token;
            let auth_config = crate::config::create_auth_config_with_port(0);
            let oauth_flow = OAuthFlow::new(auth_config);

            match oauth_flow.refresh_access_token(refresh_token).await {
                Ok(new_tokens) => {
                    debug!("Successfully refreshed tokens pre-emptively");
                    // Store new tokens
                    if let Err(e) = token_store.store(&new_tokens).await {
                        warn!("Failed to store refreshed tokens: {}", e);
                    }
                    tokens = new_tokens;
                }
                Err(e) => {
                    warn!("Failed to refresh token pre-emptively: {}", e);
                    // Continue with existing token - it might still work
                }
            }
        }
    }

    Ok(tokens)
}

/// Checks if the user is authenticated (has valid tokens)
pub async fn is_authenticated() -> bool {
    let data_dir = match CliConfig::data_dir() {
        Ok(dir) => dir,
        Err(_) => return false,
    };
    let token_store = match TokenStore::new(data_dir) {
        Ok(store) => store,
        Err(_) => return false,
    };

    match token_store.retrieve().await {
        Ok(Some(tokens)) => !tokens.is_expired(),
        Ok(None) => false,
        Err(_) => false,
    }
}

/// Clears stored authentication tokens
pub async fn clear_authentication() -> Result<()> {
    let data_dir = CliConfig::data_dir().wrap_err("Failed to get data directory")?;
    let token_store = TokenStore::new(data_dir).wrap_err("Failed to initialize token store")?;
    token_store
        .delete_tokens()
        .await
        .wrap_err("Failed to delete authentication tokens")?;
    Ok(())
}
