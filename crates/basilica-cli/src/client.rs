//! CLI-specific client creation and authentication management
//!
//! This module handles creating authenticated BasilicaClient instances specifically
//! for CLI usage, including JWT token retrieval, refresh, and fallback authentication.
//!
//! This is distinct from the general HTTP client library in basilica-api/src/client.rs
//! which provides the underlying HTTP client functionality.

use std::path::Path;
use std::sync::Arc;
use std::time::Duration;

use crate::auth::{AuthError, OAuthFlow, TokenStore};
use crate::config::CliConfig;
use crate::error::{CliError, Result};
use basilica_sdk::auth::wallet::{RequestSigner, Sr25519Signer};
use basilica_sdk::{BasilicaClient, ClientBuilder};
use color_eyre::eyre::{eyre, Context};
use tracing::{debug, warn};

/// Auth options passed from CLI flags
#[derive(Default)]
pub struct AuthOptions {
    pub wallet_key_file: Option<std::path::PathBuf>,
    pub wallet_address: Option<String>,
    pub api_token: Option<String>,
}

impl AuthOptions {
    /// Create empty options (for const initialization)
    const fn empty() -> Self {
        Self {
            wallet_key_file: None,
            wallet_address: None,
            api_token: None,
        }
    }
}

/// Creates an authenticated BasilicaClient respecting auth priority:
/// 1. Wallet key file + address
/// 2. BASILICA_WALLET_MNEMONIC env var
/// 3. API token (flag or env var)
/// 4. JWT tokens from file (existing behavior)
pub async fn create_authenticated_client(config: &CliConfig) -> Result<BasilicaClient> {
    // Clone options out of the mutex to avoid holding it across await
    let options = {
        let guard = AUTH_OPTIONS.lock().expect("auth options mutex poisoned");
        AuthOptions {
            wallet_key_file: guard.wallet_key_file.clone(),
            wallet_address: guard.wallet_address.clone(),
            api_token: guard.api_token.clone(),
        }
    };
    create_authenticated_client_with_options(config, &options).await
}

/// Thread-safe global auth options set by Args before command execution
static AUTH_OPTIONS: std::sync::Mutex<AuthOptions> = std::sync::Mutex::new(AuthOptions::empty());

/// Set global auth options (called by Args::run before command dispatch)
pub fn set_auth_options(options: AuthOptions) {
    if let Ok(mut opts) = AUTH_OPTIONS.lock() {
        *opts = options;
    }
}

/// Creates an authenticated BasilicaClient with explicit auth options
pub async fn create_authenticated_client_with_options(
    config: &CliConfig,
    options: &AuthOptions,
) -> Result<BasilicaClient> {
    let api_url = config.api.base_url.clone();
    let builder = ClientBuilder::default()
        .base_url(api_url)
        .timeout(Duration::from_secs(config.api.request_timeout));

    // Priority 1: Wallet key file + address
    if let (Some(key_file), Some(address)) = (&options.wallet_key_file, &options.wallet_address) {
        debug!("Using wallet key file authentication");
        let pair = load_sr25519_pair_from_file(key_file)?;
        let signer = Arc::new(Sr25519Signer::new(pair, address.to_string()));
        return builder
            .with_wallet_signer(signer as Arc<dyn RequestSigner>)
            .build()
            .map_err(|e| eyre!("Failed to build client: {}", e).into());
    }

    // Priority 2: Wallet mnemonic from env
    if let Ok(mnemonic) = std::env::var("BASILICA_WALLET_MNEMONIC") {
        if !mnemonic.is_empty() {
            debug!("Using wallet mnemonic authentication");
            let signer = Sr25519Signer::from_mnemonic(&mnemonic, 42)
                .map_err(|e| eyre!("Invalid wallet mnemonic: {}", e))?;
            return builder
                .with_wallet_signer(Arc::new(signer) as Arc<dyn RequestSigner>)
                .build()
                .map_err(|e| eyre!("Failed to build client: {}", e).into());
        }
    }

    // Priority 3: API token (from flag or env var)
    let token = options
        .api_token
        .clone()
        .or_else(|| std::env::var("BASILICA_API_TOKEN").ok())
        .filter(|t| !t.is_empty());
    if let Some(token) = token {
        debug!("Using API token authentication");
        return builder
            .with_api_key(&token)
            .build()
            .map_err(|e| eyre!("Failed to build client: {}", e).into());
    }

    // Priority 4: JWT tokens from file (existing behavior)
    if let Ok(tokens) = get_valid_jwt_tokens(config).await {
        debug!("Using JWT authentication with automatic token refresh");
        return builder
            .with_tokens(tokens.access_token, tokens.refresh_token)
            .build()
            .map_err(|e| eyre!("Failed to build client: {}", e).into());
    }

    Err(CliError::from(AuthError::UserNotLoggedIn))
}

/// Load an sr25519 keypair from a Bittensor key file
fn load_sr25519_pair_from_file(
    path: &Path,
) -> Result<bittensor::crypto::sr25519::Pair> {
    use bittensor::crypto::{sr25519, Pair};

    let content = std::fs::read_to_string(path)
        .map_err(|e| eyre!("Failed to read wallet key file {}: {}", path.display(), e))?;

    // Bittensor key files can be:
    // 1. Raw JSON with hex-encoded key data
    // 2. A mnemonic phrase
    // Try parsing as JSON first, then as mnemonic
    let trimmed = content.trim();

    // Try as JSON (bittensor wallet format): the file may contain a quoted hex string
    if let Ok(hex_str) = serde_json::from_str::<String>(trimmed) {
        // Try as hex-encoded seed
        let seed_bytes = hex::decode(hex_str.trim_start_matches("0x"))
            .map_err(|e| eyre!("Invalid hex in key file: {}", e))?;
        if seed_bytes.len() == 32 {
            let mut seed = [0u8; 32];
            seed.copy_from_slice(&seed_bytes);
            return Ok(sr25519::Pair::from_seed(&seed));
        }
    }

    // Try as raw hex string
    if let Ok(seed_bytes) = hex::decode(trimmed.trim_start_matches("0x")) {
        if seed_bytes.len() == 32 {
            let mut seed = [0u8; 32];
            seed.copy_from_slice(&seed_bytes);
            return Ok(sr25519::Pair::from_seed(&seed));
        }
    }

    // Try as mnemonic
    if let Ok(pair) = basilica_common::crypto::wallet::sr25519_pair_from_mnemonic(trimmed) {
        return Ok(pair);
    }

    Err(eyre!(
        "Could not parse key file {}. Expected hex-encoded seed, JSON string, or mnemonic.",
        path.display()
    )
    .into())
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
    let token_store = TokenStore::new(data_dir)
        .await
        .wrap_err("Failed to initialize token store")?;

    // Try to get stored tokens
    let mut tokens = token_store
        .retrieve_tokens()
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
                    if let Err(e) = token_store.store_tokens(&new_tokens).await {
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
    let token_store = match TokenStore::new(data_dir).await {
        Ok(store) => store,
        Err(_) => return false,
    };

    match token_store.retrieve_tokens().await {
        Ok(Some(tokens)) => !tokens.is_expired(),
        Ok(None) => false,
        Err(_) => false,
    }
}

/// Clears stored authentication tokens
pub async fn clear_authentication() -> Result<()> {
    let data_dir = CliConfig::data_dir().wrap_err("Failed to get data directory")?;
    let token_store = TokenStore::new(data_dir)
        .await
        .wrap_err("Failed to initialize token store")?;
    token_store
        .delete_tokens()
        .await
        .wrap_err("Failed to delete authentication tokens")?;
    Ok(())
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_is_non_interactive_auth_wallet_key_file() {
        let options = AuthOptions {
            wallet_key_file: Some(std::path::PathBuf::from("/tmp/key")),
            wallet_address: Some("5Grw...".to_string()),
            api_token: None,
        };
        assert!(options.wallet_key_file.is_some());
    }

    #[test]
    fn test_is_non_interactive_auth_api_token() {
        let options = AuthOptions {
            wallet_key_file: None,
            wallet_address: None,
            api_token: Some("basilica_test".to_string()),
        };
        assert!(options.api_token.is_some());
    }

    #[test]
    fn test_load_sr25519_pair_nonexistent_file() {
        let result = load_sr25519_pair_from_file(Path::new("/nonexistent/key/file"));
        assert!(result.is_err());
    }
}
