//! Device Authorization Grant implementation (RFC 8628)
//!
//! This module implements OAuth 2.0 Device Authorization Grant for
//! devices that lack a web browser or have limited input capabilities.

use super::types::{AuthConfig, AuthError, AuthResult, TokenSet};
use serde::{Deserialize, Serialize};
use std::time::{Duration, Instant};
use tracing::{debug, info};

/// Device authorization response from the OAuth provider
#[derive(Debug, Clone, Deserialize)]
pub struct DeviceAuthResponse {
    pub device_code: String,
    pub user_code: String,
    pub verification_uri: String,
    pub verification_uri_complete: Option<String>,
    pub expires_in: u64,
    pub interval: Option<u64>,
}

/// Device authorization request
#[derive(Debug, Serialize)]
struct DeviceAuthRequest {
    client_id: String,
    scope: String,
    #[serde(skip_serializing_if = "Option::is_none")]
    audience: Option<String>,
}

/// Token request for device flow
#[derive(Debug, Serialize)]
struct DeviceTokenRequest {
    grant_type: String,
    device_code: String,
    client_id: String,
}

/// Device flow polling response
#[derive(Debug, Deserialize)]
struct PollResponse {
    error: Option<String>,
    error_description: Option<String>,
    access_token: Option<String>,
    refresh_token: Option<String>,
    #[allow(dead_code)]
    scope: Option<String>,
}

/// Device authorization flow implementation
pub struct DeviceFlow {
    config: AuthConfig,
}

impl DeviceFlow {
    /// Create a new device flow instance
    pub fn new(config: AuthConfig) -> Self {
        Self { config }
    }

    /// Initiate device authorization flow
    pub async fn initiate_device_auth(&self) -> AuthResult<DeviceAuthResponse> {
        let device_endpoint = self.config.device_auth_endpoint.as_ref().ok_or_else(|| {
            AuthError::ConfigError("Device authorization endpoint not configured".to_string())
        })?;

        let scope = self.config.scopes.join(" ");
        let request_body = DeviceAuthRequest {
            client_id: self.config.client_id.clone(),
            scope,
            audience: Some(basilica_common::auth0_audience().to_string()),
        };

        let client = reqwest::Client::new();
        let response = client
            .post(device_endpoint)
            .header("Content-Type", "application/x-www-form-urlencoded")
            .form(&request_body)
            .send()
            .await
            .map_err(|e| AuthError::NetworkError(e.to_string()))?;

        if !response.status().is_success() {
            let error_text = response
                .text()
                .await
                .unwrap_or_else(|_| "Unknown error".to_string());
            return Err(AuthError::InvalidResponse(format!(
                "Device auth request failed: {}",
                error_text
            )));
        }

        let auth_response: DeviceAuthResponse = response.json().await.map_err(|e| {
            AuthError::InvalidResponse(format!("Failed to parse device auth response: {}", e))
        })?;

        Ok(auth_response)
    }

    /// Get formatted user instructions for device authorization
    pub fn get_user_instructions(&self, response: &DeviceAuthResponse) -> DeviceAuthInstructions {
        DeviceAuthInstructions {
            verification_uri: response.verification_uri.clone(),
            user_code: response.user_code.clone(),
            verification_uri_complete: response.verification_uri_complete.clone(),
        }
    }

    /// Poll for device authorization completion
    pub async fn poll_for_token(
        &self,
        device_code: &str,
        interval: Duration,
    ) -> AuthResult<TokenSet> {
        let client = reqwest::Client::new();
        let mut current_interval = interval;
        let start_time = Instant::now();
        let timeout_duration = Duration::from_secs(600); // 10 minute timeout

        let request_body = DeviceTokenRequest {
            grant_type: "urn:ietf:params:oauth:grant-type:device_code".to_string(),
            device_code: device_code.to_string(),
            client_id: self.config.client_id.clone(),
        };

        loop {
            if start_time.elapsed() > timeout_duration {
                return Err(AuthError::Timeout);
            }

            tokio::time::sleep(current_interval).await;

            let response = client
                .post(&self.config.token_endpoint)
                .header("Content-Type", "application/x-www-form-urlencoded")
                .form(&request_body)
                .send()
                .await
                .map_err(|e| AuthError::NetworkError(e.to_string()))?;

            let response_text = response
                .text()
                .await
                .map_err(|e| AuthError::NetworkError(e.to_string()))?;

            match self.handle_poll_response(&response_text)? {
                Some(token_set) => return Ok(token_set),
                None => {
                    if let Ok(poll_response) = serde_json::from_str::<PollResponse>(&response_text)
                    {
                        if poll_response.error.as_deref() == Some("slow_down") {
                            current_interval = Duration::from_secs(current_interval.as_secs() + 5);
                            debug!(
                                "Rate limited, slowing down polling interval to {} seconds",
                                current_interval.as_secs()
                            );
                        }
                    }
                    continue;
                }
            }
        }
    }

    /// Start complete device flow (returns instructions for caller to display)
    pub async fn start_flow(&self) -> AuthResult<(DeviceAuthInstructions, DeviceFlowPending)> {
        let device_response = self.initiate_device_auth().await?;
        let instructions = self.get_user_instructions(&device_response);
        let poll_interval = Duration::from_secs(device_response.interval.unwrap_or(5));

        Ok((
            instructions,
            DeviceFlowPending {
                device_code: device_response.device_code,
                poll_interval,
                config: self.config.clone(),
            },
        ))
    }

    /// Handle different polling responses
    fn handle_poll_response(&self, response_body: &str) -> AuthResult<Option<TokenSet>> {
        let poll_response: PollResponse = serde_json::from_str(response_body).map_err(|e| {
            AuthError::InvalidResponse(format!("Failed to parse poll response: {}", e))
        })?;

        if let Some(error) = &poll_response.error {
            match error.as_str() {
                "authorization_pending" => return Ok(None),
                "slow_down" => return Ok(None),
                "access_denied" => {
                    return Err(AuthError::AuthorizationDenied(
                        poll_response
                            .error_description
                            .unwrap_or_else(|| "User denied authorization".to_string()),
                    ));
                }
                "expired_token" => {
                    return Err(AuthError::DeviceFlowError("Device code expired".to_string()));
                }
                _ => {
                    return Err(AuthError::DeviceFlowError(format!(
                        "Unknown error: {} - {}",
                        error,
                        poll_response.error_description.unwrap_or_default()
                    )));
                }
            }
        }

        if let Some(access_token) = poll_response.access_token {
            let refresh_token = poll_response
                .refresh_token
                .ok_or(AuthError::InvalidResponse(
                    "No refresh token provided".to_string(),
                ))?;
            let token_set = TokenSet::new(access_token, refresh_token);
            info!("Device flow completed successfully");
            Ok(Some(token_set))
        } else {
            Err(AuthError::InvalidResponse(
                "Response contains neither error nor access token".to_string(),
            ))
        }
    }
}

/// Instructions for user to complete device authorization
#[derive(Debug, Clone)]
pub struct DeviceAuthInstructions {
    pub verification_uri: String,
    pub user_code: String,
    pub verification_uri_complete: Option<String>,
}

/// Pending device flow that can be polled for completion
pub struct DeviceFlowPending {
    device_code: String,
    poll_interval: Duration,
    config: AuthConfig,
}

impl DeviceFlowPending {
    /// Wait for user to complete authorization and return tokens
    pub async fn wait_for_completion(&self) -> AuthResult<TokenSet> {
        let flow = DeviceFlow::new(self.config.clone());
        flow.poll_for_token(&self.device_code, self.poll_interval)
            .await
    }
}

