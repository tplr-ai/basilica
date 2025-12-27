//! OAuth 2.0 with PKCE (Proof Key for Code Exchange) implementation
//!
//! This module implements the OAuth 2.0 authorization code flow with PKCE
//! for secure authentication without requiring client secrets.

use super::callback_server::CallbackServer;
use super::types::{AuthConfig, AuthError, AuthResult, TokenSet};
use base64::engine::general_purpose::URL_SAFE_NO_PAD;
use base64::Engine;
use oauth2::{
    basic::BasicClient, reqwest::async_http_client, AuthUrl, AuthorizationCode, ClientId,
    CsrfToken, PkceCodeChallenge, PkceCodeVerifier, RedirectUrl, Scope, TokenResponse, TokenUrl,
};
use rand::Rng;
use sha2::{Digest, Sha256};
use tracing::{debug, info};

/// Extract port number from a redirect URI
fn extract_port_from_redirect_uri(redirect_uri: &str) -> AuthResult<u16> {
    let url = url::Url::parse(redirect_uri)
        .map_err(|e| AuthError::ConfigError(format!("Invalid redirect URI: {}", e)))?;

    url.port()
        .ok_or_else(|| AuthError::ConfigError("Redirect URI must contain a port".to_string()))
}

/// OAuth flow implementation with PKCE support
pub struct OAuthFlow {
    config: AuthConfig,
    code_verifier: Option<String>,
    code_challenge: Option<String>,
    state: Option<String>,
}

/// Generate cryptographically secure random 32-byte verifier
pub fn generate_pkce_verifier() -> String {
    debug!("Generating PKCE code verifier");
    let mut rng = rand::thread_rng();
    let verifier_bytes: [u8; 32] = rng.gen();
    let verifier = URL_SAFE_NO_PAD.encode(verifier_bytes);
    debug!("PKCE verifier generated: {} bytes", verifier.len());
    verifier
}

/// Create SHA256 hash of verifier in base64url encoding
pub fn generate_pkce_challenge(verifier: &str) -> String {
    debug!("Generating PKCE code challenge from verifier");
    let mut hasher = Sha256::new();
    hasher.update(verifier.as_bytes());
    let challenge_bytes = hasher.finalize();
    let challenge = URL_SAFE_NO_PAD.encode(challenge_bytes);
    debug!("PKCE challenge generated: {} bytes", challenge.len());
    challenge
}

/// Generate random state parameter
pub fn generate_state() -> String {
    debug!("Generating state parameter for CSRF protection");
    let mut rng = rand::thread_rng();
    let state_bytes: [u8; 32] = rng.gen();
    let state = URL_SAFE_NO_PAD.encode(state_bytes);
    debug!("State parameter generated: {} bytes", state.len());
    state
}

impl OAuthFlow {
    /// Create a new OAuth flow instance
    pub fn new(config: AuthConfig) -> Self {
        debug!(
            "Initializing OAuth flow with client_id: {}",
            config.client_id
        );
        Self {
            config,
            code_verifier: None,
            code_challenge: None,
            state: None,
        }
    }

    /// Generate PKCE code verifier and challenge
    fn generate_pkce_pair(&mut self) -> AuthResult<()> {
        debug!("Generating PKCE verifier and challenge");
        let verifier = generate_pkce_verifier();
        let challenge = generate_pkce_challenge(&verifier);
        self.code_verifier = Some(verifier);
        self.code_challenge = Some(challenge);
        debug!("PKCE pair generated successfully");
        Ok(())
    }

    /// Generate a secure random state parameter
    fn generate_state_internal(&mut self) -> AuthResult<String> {
        let state = generate_state();
        self.state = Some(state.clone());
        Ok(state)
    }

    /// Build the authorization URL for the OAuth provider
    pub fn build_auth_url(&mut self) -> AuthResult<String> {
        debug!("Building authorization URL");

        if self.code_verifier.is_none() || self.code_challenge.is_none() {
            self.generate_pkce_pair()?;
        }

        if self.state.is_none() {
            self.generate_state_internal()?;
        }

        let client = BasicClient::new(
            ClientId::new(self.config.client_id.clone()),
            None, // No client secret for PKCE flow
            AuthUrl::new(self.config.auth_endpoint.clone())
                .map_err(|e| AuthError::ConfigError(format!("Invalid auth endpoint: {}", e)))?,
            Some(
                TokenUrl::new(self.config.token_endpoint.clone())
                    .map_err(|e| AuthError::ConfigError(format!("Invalid token endpoint: {}", e)))?,
            ),
        )
        .set_redirect_uri(
            RedirectUrl::new(self.config.redirect_uri.clone())
                .map_err(|e| AuthError::ConfigError(format!("Invalid redirect URI: {}", e)))?,
        );

        let pkce_verifier = PkceCodeVerifier::new(self.code_verifier.as_ref().unwrap().clone());
        let pkce_challenge = PkceCodeChallenge::from_code_verifier_sha256(&pkce_verifier);

        let mut auth_request = client
            .authorize_url(|| CsrfToken::new(self.state.as_ref().unwrap().clone()))
            .set_pkce_challenge(pkce_challenge);

        debug!(
            "Adding scopes to authorization request: {:?}",
            self.config.scopes
        );
        for scope in &self.config.scopes {
            auth_request = auth_request.add_scope(Scope::new(scope.clone()));
        }

        for (key, value) in &self.config.additional_params {
            auth_request = auth_request.add_extra_param(key, value);
        }

        let audience = basilica_common::auth0_audience();
        debug!("Adding audience parameter: {}", audience);
        auth_request = auth_request.add_extra_param("audience", audience);

        let (auth_url, _csrf_token) = auth_request.url();
        let url_string = auth_url.to_string();

        debug!("Authorization URL built: {}", url_string);
        Ok(url_string)
    }

    /// Get the authorization URL (for display purposes)
    pub fn get_auth_url(&mut self) -> AuthResult<String> {
        self.build_auth_url()
    }

    /// Get the expected state for verification
    pub fn get_expected_state(&self) -> Option<&str> {
        self.state.as_deref()
    }

    /// Start the OAuth flow by opening the browser and starting callback server
    ///
    /// This is the main entry point for browser-based OAuth.
    pub async fn start_flow(&mut self) -> AuthResult<TokenSet> {
        info!("Starting OAuth flow");

        let port = extract_port_from_redirect_uri(&self.config.redirect_uri)?;
        let callback_server = CallbackServer::new(port, std::time::Duration::from_secs(300));

        let auth_url = self.build_auth_url()?;

        let expected_state = self
            .state
            .as_ref()
            .ok_or_else(|| AuthError::ConfigError("State not set for OAuth flow".to_string()))?;

        // Open browser
        info!("Opening browser for authentication");
        webbrowser::open(&auth_url)
            .map_err(|e| AuthError::ConfigError(format!("Failed to open browser: {}", e)))?;

        // Wait for callback
        let callback_data = callback_server.start_and_wait(expected_state).await?;

        let code = callback_data.code.ok_or_else(|| {
            AuthError::CallbackServerError("No authorization code received".to_string())
        })?;

        let token_set = self.exchange_code_for_token(&code).await?;

        info!("OAuth flow completed successfully");
        Ok(token_set)
    }

    /// Exchange authorization code for access token
    pub async fn exchange_code_for_token(&self, code: &str) -> AuthResult<TokenSet> {
        debug!("Exchanging authorization code for tokens");

        let code_verifier = self
            .code_verifier
            .as_ref()
            .ok_or_else(|| AuthError::PkceError("Code verifier not generated".to_string()))?;

        let client = BasicClient::new(
            ClientId::new(self.config.client_id.clone()),
            None,
            AuthUrl::new(self.config.auth_endpoint.clone())
                .map_err(|e| AuthError::ConfigError(format!("Invalid auth endpoint: {}", e)))?,
            Some(
                TokenUrl::new(self.config.token_endpoint.clone())
                    .map_err(|e| AuthError::ConfigError(format!("Invalid token endpoint: {}", e)))?,
            ),
        )
        .set_redirect_uri(
            RedirectUrl::new(self.config.redirect_uri.clone())
                .map_err(|e| AuthError::ConfigError(format!("Invalid redirect URI: {}", e)))?,
        );

        let token_response = client
            .exchange_code(AuthorizationCode::new(code.to_string()))
            .set_pkce_verifier(PkceCodeVerifier::new(code_verifier.clone()))
            .request_async(async_http_client)
            .await
            .map_err(|e| AuthError::NetworkError(format!("Token exchange failed: {}", e)))?;

        let access_token = token_response.access_token().secret().to_string();
        let refresh_token = token_response
            .refresh_token()
            .map(|rt| rt.secret().to_string());

        let token_set = TokenSet::new(
            access_token,
            refresh_token.ok_or(AuthError::InvalidResponse(
                "No refresh token provided".to_string(),
            ))?,
        );

        info!("Token exchange completed successfully");
        Ok(token_set)
    }

    /// Refresh an expired access token using the refresh token
    pub async fn refresh_access_token(&self, refresh_token: &str) -> AuthResult<TokenSet> {
        super::refresh_access_token(
            refresh_token,
            Some(&self.config.client_id),
            Some(&self.config.token_endpoint),
        )
        .await
    }

    /// Revoke a token with the OAuth provider
    pub async fn revoke_token(&self, token_set: &TokenSet) -> AuthResult<()> {
        debug!("Starting token revocation");

        let revoke_endpoint = self.config.revoke_endpoint.as_ref().ok_or_else(|| {
            AuthError::ConfigError("Revoke endpoint not configured".to_string())
        })?;

        let token_to_revoke = &token_set.refresh_token;

        debug!("Revoking token at endpoint: {}", revoke_endpoint);

        let client = reqwest::Client::new();
        let revoke_request = serde_json::json!({
            "client_id": self.config.client_id,
            "token": token_to_revoke
        });

        let response = client
            .post(revoke_endpoint)
            .header("Content-Type", "application/json")
            .json(&revoke_request)
            .send()
            .await
            .map_err(|e| {
                AuthError::NetworkError(format!("Token revocation request failed: {}", e))
            })?;

        if response.status().is_success() {
            info!("Token revocation completed successfully");
            Ok(())
        } else {
            let status = response.status();
            let error_text = response
                .text()
                .await
                .unwrap_or_else(|_| "Unknown error".to_string());

            Err(AuthError::NetworkError(format!(
                "Token revocation failed with status {}: {}",
                status, error_text
            )))
        }
    }
}

