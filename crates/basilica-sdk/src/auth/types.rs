//! Authentication-related types and data structures
//!
//! This module defines all the types used throughout the auth module
//! including configuration, token data, and error types.

use base64::{engine::general_purpose::URL_SAFE_NO_PAD, Engine};
use etcetera::{choose_base_strategy, BaseStrategy};
use serde::{Deserialize, Serialize};
use std::path::PathBuf;
use std::time::{Duration, SystemTime, UNIX_EPOCH};

/// Ceiling on how early a token is renewed, however long its lifetime.
const MAX_REFRESH_LEAD: Duration = Duration::from_secs(60 * 60);

/// Lead used when the token carries no `iat`, so its lifetime is unknown.
const FALLBACK_REFRESH_LEAD: Duration = Duration::from_secs(5 * 60);

/// Result type for authentication operations
pub type AuthResult<T> = Result<T, AuthError>;

/// Authentication configuration
#[derive(Debug, Clone)]
pub struct AuthConfig {
    /// OAuth client ID
    pub client_id: String,
    /// OAuth authorization endpoint URL
    pub auth_endpoint: String,
    /// OAuth token endpoint URL
    pub token_endpoint: String,
    /// OAuth device authorization endpoint URL (for device flow)
    pub device_auth_endpoint: Option<String>,
    /// OAuth token revocation endpoint URL
    pub revoke_endpoint: Option<String>,
    /// Redirect URI for OAuth callback
    pub redirect_uri: String,
    /// OAuth scopes to request
    pub scopes: Vec<String>,
    /// Additional OAuth parameters
    pub additional_params: std::collections::HashMap<String, String>,
}

/// OAuth token set containing access token and refresh token
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct TokenSet {
    /// Access token for API requests
    pub access_token: String,
    /// Refresh token for token renewal (always required)
    pub refresh_token: String,
}

impl TokenSet {
    /// Create a new token set
    pub fn new(access_token: String, refresh_token: String) -> Self {
        Self {
            access_token,
            refresh_token,
        }
    }

    /// Read a numeric claim from the access token's payload.
    fn decode_jwt_claim(token: &str, claim: &str) -> Option<u64> {
        // JWT has three parts: header.payload.signature
        let parts: Vec<&str> = token.split('.').collect();
        if parts.len() != 3 {
            return None;
        }

        // Decode the payload (second part)
        let payload = parts[1];

        // Decode base64url without padding (JWT uses base64url encoding)
        let decoded = URL_SAFE_NO_PAD.decode(payload).ok()?;

        // Parse JSON and extract the requested claim
        let json: serde_json::Value = serde_json::from_slice(&decoded).ok()?;
        json.get(claim)?.as_u64()
    }

    /// Get the expiration time by decoding JWT
    fn get_expiration(&self) -> Option<u64> {
        // Always decode from JWT token
        Self::decode_jwt_claim(&self.access_token, "exp")
    }

    /// Total lifetime the issuer gave this token, from its own claims.
    fn lifetime(&self) -> Option<Duration> {
        let exp = Self::decode_jwt_claim(&self.access_token, "exp")?;
        let iat = Self::decode_jwt_claim(&self.access_token, "iat")?;
        exp.checked_sub(iat).map(Duration::from_secs)
    }

    /// How long before expiry a refresh should happen.
    ///
    /// Half the token's own lifetime, capped at an hour. A fixed lead is wrong
    /// for short-lived tokens: with a one-hour token, a fixed one-hour lead is
    /// satisfied the instant the token is minted, so every single call would
    /// refresh. Scaling with the lifetime keeps roughly half the token usable
    /// before the first renewal, whatever the issuer configures.
    ///
    /// A 24-hour token still gets the one-hour lead it had before this was
    /// derived, so long-lived tokens behave exactly as they always did.
    fn refresh_lead(&self) -> Duration {
        match self.lifetime() {
            Some(lifetime) => MAX_REFRESH_LEAD.min(lifetime / 2),
            // No `iat` to measure against; fall back to the same small buffer
            // the token stores use rather than assuming a long lifetime.
            None => FALLBACK_REFRESH_LEAD,
        }
    }

    /// Check if the access token is expired
    pub fn is_expired(&self) -> bool {
        match self.get_expiration() {
            Some(expires_at) => {
                let now = SystemTime::now()
                    .duration_since(UNIX_EPOCH)
                    .unwrap()
                    .as_secs();
                now >= expires_at
            }
            None => false, // No expiration time means token doesn't expire
        }
    }

    /// Check whether the token is close enough to expiry to renew.
    ///
    /// The window scales with the token's own lifetime; see [`Self::refresh_lead`].
    pub fn needs_refresh(&self) -> bool {
        self.expires_within(self.refresh_lead())
    }

    /// Check if the token expires within the specified duration
    pub fn expires_within(&self, duration: Duration) -> bool {
        match self.get_expiration() {
            Some(expires_at) => {
                let now = SystemTime::now()
                    .duration_since(UNIX_EPOCH)
                    .unwrap()
                    .as_secs();
                let threshold = now + duration.as_secs();
                expires_at <= threshold
            }
            None => false, // No expiration time means token doesn't expire soon
        }
    }

    /// Get time until token expiration
    pub fn time_until_expiry(&self) -> Option<Duration> {
        match self.get_expiration() {
            Some(expires_at) => {
                let now = SystemTime::now()
                    .duration_since(UNIX_EPOCH)
                    .unwrap()
                    .as_secs();
                if expires_at > now {
                    Some(Duration::from_secs(expires_at - now))
                } else {
                    Some(Duration::from_secs(0)) // Already expired
                }
            }
            None => None, // No expiration time
        }
    }
}

/// Authentication method for the SDK
#[derive(Debug, Clone)]
pub enum AuthMethod {
    /// Direct tokens provided by the user
    Direct { tokens: TokenSet },
    /// Tokens loaded from file storage  
    FileBased {
        store: crate::auth::token_store::TokenStore,
    },
}

/// Authentication errors
#[derive(Debug, thiserror::Error)]
pub enum AuthError {
    /// OAuth authorization was denied by user
    #[error("Authorization denied: {0}")]
    AuthorizationDenied(String),

    /// Network error during OAuth flow
    #[error("Network error: {0}")]
    NetworkError(String),

    /// Invalid OAuth response
    #[error("Invalid OAuth response: {0}")]
    InvalidResponse(String),

    /// Token storage error
    #[error("Token storage error: {0}")]
    StorageError(String),

    /// Configuration error
    #[error("Configuration error: {0}")]
    ConfigError(String),

    /// PKCE generation or validation error
    #[error("PKCE error: {0}")]
    PkceError(String),

    /// State parameter mismatch (CSRF protection)
    #[error("State mismatch: expected {expected}, got {actual}")]
    StateMismatch { expected: String, actual: String },

    /// Token expired
    #[error("Token expired")]
    TokenExpired,

    /// Invalid token format
    #[error("Invalid token: {0}")]
    InvalidToken(String),

    /// Authentication required
    #[error("Authentication required")]
    AuthenticationRequired,

    /// Callback server error
    #[error("Callback server error: {0}")]
    CallbackServerError(String),

    /// Device flow specific errors
    #[error("Device flow error: {0}")]
    DeviceFlowError(String),

    /// Timeout during authorization flow
    #[error("Authorization timeout")]
    Timeout,

    /// User is not logged in / no tokens found
    #[error("Authentication required. Please use one of the following methods:\n  • Run 'basilica login' to authenticate via CLI\n  • Provide access_token and refresh_token to the client\n  • Set BASILICA_API_TOKEN and BASILICA_REFRESH_TOKEN environment variables")]
    UserNotLoggedIn,

    /// Generic IO error
    #[error("IO error: {0}")]
    IoError(#[from] std::io::Error),

    /// Serialization/deserialization error
    #[error("Serialization error: {0}")]
    SerdeError(#[from] serde_json::Error),
}

/// Get the default data directory for SDK token storage
/// Returns platform-specific data directory (e.g., ~/.local/share/basilica on Linux)
pub fn get_sdk_data_dir() -> AuthResult<PathBuf> {
    let strategy = choose_base_strategy().map_err(|e| {
        AuthError::ConfigError(format!("Failed to determine base directories: {}", e))
    })?;

    // Use the same path as the CLI for consistency
    Ok(strategy.data_dir().join("basilica"))
}

#[cfg(test)]
mod tests {
    use super::*;

    /// Build an access token whose payload carries the given claims. Only the
    /// payload is read, so the header and signature can be anything.
    fn token_with(iat: u64, exp: u64) -> TokenSet {
        let payload = URL_SAFE_NO_PAD.encode(format!(r#"{{"iat":{iat},"exp":{exp}}}"#));
        TokenSet::new(format!("header.{payload}.signature"), "refresh".to_string())
    }

    fn now() -> u64 {
        SystemTime::now()
            .duration_since(UNIX_EPOCH)
            .unwrap()
            .as_secs()
    }

    #[test]
    fn lead_is_half_the_lifetime_for_a_short_token() {
        // One-hour token: renew in the final half hour, not immediately.
        let t = token_with(now(), now() + 3600);
        assert_eq!(t.refresh_lead(), Duration::from_secs(1800));
    }

    #[test]
    fn a_freshly_minted_short_token_does_not_need_refresh() {
        // The bug this replaces: a fixed one-hour lead made every call to a
        // one-hour token refresh, because it was always "expiring within" it.
        let t = token_with(now(), now() + 3600);
        assert!(!t.needs_refresh());
    }

    #[test]
    fn a_short_token_past_halfway_needs_refresh() {
        let t = token_with(now() - 1900, now() + 1700);
        assert!(t.needs_refresh());
    }

    #[test]
    fn long_lived_tokens_keep_the_one_hour_lead() {
        // A 24-hour token behaves exactly as it did before the lead was derived.
        let t = token_with(now(), now() + 86_400);
        assert_eq!(t.refresh_lead(), Duration::from_secs(3600));
        assert!(!t.needs_refresh());
    }

    #[test]
    fn a_long_token_inside_the_last_hour_needs_refresh() {
        let t = token_with(now() - 84_000, now() + 2_400);
        assert!(t.needs_refresh());
    }

    #[test]
    fn without_iat_the_lead_falls_back_to_the_small_buffer() {
        let payload = URL_SAFE_NO_PAD.encode(format!(r#"{{"exp":{}}}"#, now() + 3600));
        let t = TokenSet::new(format!("header.{payload}.signature"), "refresh".to_string());
        assert_eq!(t.refresh_lead(), FALLBACK_REFRESH_LEAD);
        assert!(!t.needs_refresh());
    }

    #[test]
    fn an_expired_token_needs_refresh() {
        let t = token_with(now() - 7200, now() - 3600);
        assert!(t.is_expired());
        assert!(t.needs_refresh());
    }
}
