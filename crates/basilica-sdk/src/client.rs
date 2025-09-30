//! HTTP client for the Basilica API
//!
//! This module provides a type-safe client for interacting with the Basilica API.
//! It supports both authenticated and unauthenticated requests.
//!
//! # Authentication
//!
//! The client uses Auth0 JWT Bearer token authentication:
//!
//! ## Auth0 JWT Authentication
//! - Uses `Authorization: Bearer {token}` header with Auth0-issued JWT tokens
//! - Thread-safe token management with async/await support
//! - Secure authentication via Auth0 identity provider
//!
//! # Usage Examples
//!
//! ```rust,no_run
//! use basilica_sdk::{BasilicaClient, ClientBuilder};
//! use std::sync::Arc;
//!
//! # async fn example() -> basilica_sdk::Result<()> {
//! // Direct token authentication with refresh support
//! let client = ClientBuilder::default()
//!     .base_url("https://api.basilica.ai")
//!     .with_tokens("access_token", "refresh_token")
//!     .build()?;
//!
//! // Or use file-based authentication (reads from ~/.local/share/basilica/)
//! let client = ClientBuilder::default()
//!     .base_url("https://api.basilica.ai")
//!     .with_file_auth()
//!     .build()?;
//!
//! # Ok(())
//! # }
//! ```

use crate::{
    auth::TokenManager,
    error::{ApiError, ErrorResponse, Result},
    types::{
        ApiKeyInfo, ApiKeyResponse, ApiListRentalsResponse, CreateApiKeyRequest,
        HealthCheckResponse, ListAvailableExecutorsQuery, ListRentalsQuery,
        RentalStatusWithSshResponse,
    },
    StartRentalApiRequest,
};

/// Default API URL when not specified
pub const DEFAULT_API_URL: &str = "https://api.basilica.ai";

/// Default timeout in seconds for API requests
pub const DEFAULT_TIMEOUT_SECS: u64 = 1200;
use basilica_common::ApiKeyName;
use basilica_validator::api::types::ListAvailableExecutorsResponse;
use basilica_validator::rental::RentalResponse;
use reqwest::{RequestBuilder, Response, StatusCode};
use serde::{de::DeserializeOwned, Serialize};
use std::sync::Arc;
use std::time::Duration;

/// HTTP client for interacting with the Basilica API
#[derive(Debug)]
pub struct BasilicaClient {
    http_client: reqwest::Client,
    base_url: String,
    token_manager: Arc<TokenManager>,
}

impl BasilicaClient {
    /// Create a new client (private - use ClientBuilder instead)
    fn new(
        base_url: impl Into<String>,
        timeout: Duration,
        token_manager: Arc<TokenManager>,
    ) -> Result<Self> {
        let http_client = reqwest::Client::builder()
            .timeout(timeout)
            .build()
            .map_err(ApiError::HttpClient)?;

        Ok(Self {
            http_client,
            base_url: base_url.into(),
            token_manager,
        })
    }

    // ===== Rentals =====

    /// Get rental status
    pub async fn get_rental_status(&self, rental_id: &str) -> Result<RentalStatusWithSshResponse> {
        let path = format!("/rentals/{rental_id}");
        self.get(&path).await
    }

    /// Start a new rental
    pub async fn start_rental(&self, request: StartRentalApiRequest) -> Result<RentalResponse> {
        self.post("/rentals", &request).await
    }

    /// Stop a rental
    pub async fn stop_rental(&self, rental_id: &str) -> Result<()> {
        let path = format!("/rentals/{rental_id}");
        let response: Response = self.delete_empty(&path).await?;
        if response.status().is_success() {
            Ok(())
        } else {
            let err = self
                .handle_error_response::<serde_json::Value>(response)
                .await
                .err()
                .unwrap_or(ApiError::Internal {
                    message: "Unknown error".into(),
                });
            Err(err)
        }
    }

    /// Get rental logs
    pub async fn get_rental_logs(
        &self,
        rental_id: &str,
        follow: bool,
        tail: Option<u32>,
    ) -> Result<reqwest::Response> {
        let url = format!("{}/rentals/{}/logs", self.base_url, rental_id);
        let mut request = self.http_client.get(&url);

        let mut params: Vec<(&str, String)> = vec![];
        if follow {
            params.push(("follow", "true".to_string()));
        }
        if let Some(tail_lines) = tail {
            params.push(("tail", tail_lines.to_string()));
        }

        if !params.is_empty() {
            request = request.query(&params);
        }

        let request = self.apply_auth(request).await?;
        request.send().await.map_err(ApiError::HttpClient)
    }

    /// List rentals
    pub async fn list_rentals(
        &self,
        query: Option<ListRentalsQuery>,
    ) -> Result<ApiListRentalsResponse> {
        let url = format!("{}/rentals", self.base_url);
        let mut request = self.http_client.get(&url);

        if let Some(q) = &query {
            request = request.query(&q);
        }

        let request = self.apply_auth(request).await?;
        let response = request.send().await.map_err(ApiError::HttpClient)?;
        self.handle_response(response).await
    }

    /// List available executors for rental
    pub async fn list_available_executors(
        &self,
        query: Option<ListAvailableExecutorsQuery>,
    ) -> Result<ListAvailableExecutorsResponse> {
        let url = format!("{}/executors", self.base_url);
        let mut request = self.http_client.get(&url);

        if let Some(q) = &query {
            request = request.query(&q);
        }

        let request = self.apply_auth(request).await?;
        let response = request.send().await.map_err(ApiError::HttpClient)?;
        self.handle_response(response).await
    }

    // ===== Health & Discovery =====

    /// Health check
    pub async fn health_check(&self) -> Result<HealthCheckResponse> {
        self.get("/health").await
    }

    // ===== API Key Management =====

    /// Create a new API key (requires JWT authentication)
    /// The API key will inherit scopes from the current JWT token
    pub async fn create_api_key(&self, name: &str) -> Result<ApiKeyResponse> {
        // Validate the name early to provide better error messages
        ApiKeyName::new(name).map_err(|e| ApiError::InvalidRequest {
            message: format!("Invalid API key name: {}", e),
        })?;

        let request = CreateApiKeyRequest {
            name: name.to_string(),
            scopes: None, // Will inherit from JWT
        };
        self.post("/api-keys", &request).await
    }

    /// Get current API key info (requires JWT authentication)
    /// Returns the first (and only) key if it exists
    pub async fn get_api_key(&self) -> Result<Option<ApiKeyInfo>> {
        let keys: Vec<ApiKeyInfo> = self.get("/api-keys").await?;
        Ok(keys.into_iter().next())
    }

    /// List all API keys for the authenticated user (requires JWT authentication)
    pub async fn list_api_keys(&self) -> Result<Vec<ApiKeyInfo>> {
        self.get("/api-keys").await
    }

    /// Delete a specific API key by name (requires JWT authentication)
    pub async fn revoke_api_key(&self, name: &str) -> Result<()> {
        let encoded_name = urlencoding::encode(name);
        let response = self
            .delete_empty(&format!("/api-keys/{}", encoded_name))
            .await?;
        if response.status().is_success() {
            Ok(())
        } else {
            self.handle_error_response(response).await
        }
    }

    // ===== Private Helper Methods =====

    /// Apply authentication to request
    /// Uses TokenManager for automatic token refresh
    async fn apply_auth(&self, request: RequestBuilder) -> Result<RequestBuilder> {
        let token =
            self.token_manager
                .get_access_token()
                .await
                .map_err(|e| ApiError::Internal {
                    message: format!("Failed to get access token: {}", e),
                })?;
        Ok(request.header("Authorization", format!("Bearer {}", token)))
    }

    /// Generic GET request
    async fn get<T: DeserializeOwned>(&self, path: &str) -> Result<T> {
        let url = format!("{}{}", self.base_url, path);
        let request = self.http_client.get(&url);
        let request = self.apply_auth(request).await?;

        let response = request.send().await.map_err(ApiError::HttpClient)?;
        self.handle_response(response).await
    }

    /// Generic POST request
    async fn post<B: Serialize, T: DeserializeOwned>(&self, path: &str, body: &B) -> Result<T> {
        let url = format!("{}{}", self.base_url, path);
        let request = self.http_client.post(&url).json(body);
        let request = self.apply_auth(request).await?;

        let response = request.send().await.map_err(ApiError::HttpClient)?;
        self.handle_response(response).await
    }

    /// Generic DELETE request without body
    async fn delete_empty(&self, path: &str) -> Result<Response> {
        let url = format!("{}{}", self.base_url, path);
        let request = self.http_client.delete(&url);
        let request = self.apply_auth(request).await?;

        let response = request.send().await.map_err(ApiError::HttpClient)?;
        Ok(response)
    }

    /// Handle successful response
    async fn handle_response<T: DeserializeOwned>(&self, response: Response) -> Result<T> {
        if response.status().is_success() {
            response.json().await.map_err(ApiError::HttpClient)
        } else {
            self.handle_error_response(response).await
        }
    }

    /// Handle error response
    async fn handle_error_response<T>(&self, response: Response) -> Result<T> {
        let status = response.status();
        let error_text = response.text().await.unwrap_or_default();

        // Try to parse error response
        if let Ok(error_response) = serde_json::from_str::<ErrorResponse>(&error_text) {
            match status {
                StatusCode::UNAUTHORIZED => {
                    // Distinguish between missing auth and expired/invalid auth based on error code
                    match error_response.error.code.as_str() {
                        "BASILICA_API_AUTH_MISSING" => Err(ApiError::MissingAuthentication {
                            message: error_response.error.message,
                        }),
                        _ => Err(ApiError::Authentication {
                            message: error_response.error.message,
                        }),
                    }
                }
                StatusCode::FORBIDDEN => Err(ApiError::Authorization {
                    message: error_response.error.message,
                }),
                StatusCode::TOO_MANY_REQUESTS => Err(ApiError::RateLimitExceeded),
                StatusCode::NOT_FOUND => Err(ApiError::NotFound {
                    resource: error_response.error.message,
                }),
                StatusCode::BAD_REQUEST => Err(ApiError::BadRequest {
                    message: error_response.error.message,
                }),
                StatusCode::CONFLICT => Err(ApiError::Conflict {
                    message: error_response.error.message,
                }),
                _ => Err(ApiError::Internal {
                    message: error_response.error.message,
                }),
            }
        } else {
            // Fallback if we can't parse the error
            match status {
                StatusCode::UNAUTHORIZED => Err(ApiError::Authentication {
                    message: "Authentication failed".into(),
                }),
                StatusCode::FORBIDDEN => Err(ApiError::Authorization {
                    message: "Access forbidden".into(),
                }),
                StatusCode::TOO_MANY_REQUESTS => Err(ApiError::RateLimitExceeded),
                StatusCode::NOT_FOUND => Err(ApiError::NotFound {
                    resource: "Resource not found".into(),
                }),
                StatusCode::BAD_REQUEST => Err(ApiError::BadRequest {
                    message: error_text,
                }),
                StatusCode::CONFLICT => Err(ApiError::Conflict {
                    message: error_text,
                }),
                _ => Err(ApiError::Internal {
                    message: format!("Request failed with status {status}: {error_text}"),
                }),
            }
        }
    }
}

/// Builder for constructing a BasilicaClient with custom configuration
#[derive(Default)]
pub struct ClientBuilder {
    base_url: Option<String>,
    access_token: Option<String>,
    refresh_token: Option<String>,
    timeout: Option<Duration>,
    connect_timeout: Option<Duration>,
    pool_max_idle_per_host: Option<usize>,
    use_file_auth: bool,
    api_key: Option<String>,
}

impl ClientBuilder {
    /// Create a new builder with default values
    pub fn new() -> Self {
        Self::default()
    }

    /// Set the base URL for the API
    pub fn base_url(mut self, url: impl Into<String>) -> Self {
        self.base_url = Some(url.into());
        self
    }

    /// Set tokens for direct authentication (both tokens required)
    pub fn with_tokens(
        mut self,
        access_token: impl Into<String>,
        refresh_token: impl Into<String>,
    ) -> Self {
        self.access_token = Some(access_token.into());
        self.refresh_token = Some(refresh_token.into());
        self.use_file_auth = false;
        self
    }

    /// Use file-based authentication (reads tokens from ~/.local/share/basilica/)
    pub fn with_file_auth(mut self) -> Self {
        self.use_file_auth = true;
        self.access_token = None;
        self.refresh_token = None;
        self
    }

    /// Set the request timeout
    pub fn timeout(mut self, timeout: Duration) -> Self {
        self.timeout = Some(timeout);
        self
    }

    /// Set the connection timeout
    pub fn connect_timeout(mut self, timeout: Duration) -> Self {
        self.connect_timeout = Some(timeout);
        self
    }

    /// Set the maximum idle connections per host
    pub fn pool_max_idle_per_host(mut self, max: usize) -> Self {
        self.pool_max_idle_per_host = Some(max);
        self
    }

    /// Use API key for authentication (from provided string)
    pub fn with_api_key(mut self, api_key: &str) -> Self {
        self.api_key = Some(api_key.to_string());
        self
    }

    /// Build the client with automatic authentication detection
    /// This will automatically find and use CLI tokens if available
    pub async fn build_auto(self) -> Result<BasilicaClient> {
        let base_url = self.base_url.unwrap_or_else(|| DEFAULT_API_URL.to_string());

        // Always try file-based auth for auto mode
        let token_manager = TokenManager::new_file_based().map_err(|e| ApiError::Internal {
            message: format!("Failed to create file-based token manager: {}", e),
        })?;

        let timeout = self
            .timeout
            .unwrap_or(Duration::from_secs(DEFAULT_TIMEOUT_SECS));

        BasilicaClient::new(base_url, timeout, Arc::new(token_manager))
    }

    /// Build the client
    pub fn build(self) -> Result<BasilicaClient> {
        let base_url = self.base_url.unwrap_or_else(|| DEFAULT_API_URL.to_string());

        // Create token manager based on auth configuration
        let token_manager = if let Some(api_key) = self.api_key {
            // API key takes precedence
            TokenManager::new_api_key(api_key)
        } else if self.use_file_auth {
            // File-based auth (also checks for BASILICA_API_KEY env var)
            TokenManager::new_file_based().map_err(|e| ApiError::Internal {
                message: format!("Failed to create file-based token manager: {}", e),
            })?
        } else if let (Some(access_token), Some(refresh_token)) =
            (self.access_token, self.refresh_token)
        {
            TokenManager::new_direct(access_token, refresh_token)
        } else {
            return Err(ApiError::InvalidRequest {
                message: "Either use with_tokens() with both access and refresh tokens, with_file_auth(), or with_api_key()"
                    .into(),
            });
        };

        let timeout = self
            .timeout
            .unwrap_or(Duration::from_secs(DEFAULT_TIMEOUT_SECS));

        BasilicaClient::new(base_url, timeout, Arc::new(token_manager))
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use serde_json::json;
    use wiremock::matchers::{header, method, path};
    use wiremock::{Mock, MockServer, ResponseTemplate};

    #[tokio::test]
    async fn test_health_check() {
        let mock_server = MockServer::start().await;

        Mock::given(method("GET"))
            .and(path("/health"))
            .and(header("Authorization", "Bearer test-token"))
            .respond_with(ResponseTemplate::new(200).set_body_json(json!({
                "status": "healthy",
                "version": "1.0.0",
                "timestamp": "2024-01-01T00:00:00Z",
                "healthy_validators": 10,
                "total_validators": 10,
            })))
            .mount(&mock_server)
            .await;

        let client = ClientBuilder::default()
            .base_url(mock_server.uri())
            .with_tokens("test-token", "refresh-token")
            .build()
            .unwrap();
        let health = client.health_check().await.unwrap();

        assert_eq!(health.status, "healthy");
        assert_eq!(health.version, "1.0.0");
    }

    #[tokio::test]
    async fn test_token_auth_with_refresh() {
        let mock_server = MockServer::start().await;

        Mock::given(method("GET"))
            .and(path("/health"))
            .and(header("Authorization", "Bearer test-token"))
            .respond_with(ResponseTemplate::new(200).set_body_json(json!({
                "status": "healthy",
                "version": "1.0.0",
                "timestamp": "2024-01-01T00:00:00Z",
                "healthy_validators": 10,
                "total_validators": 10,
            })))
            .mount(&mock_server)
            .await;

        let client = ClientBuilder::default()
            .base_url(mock_server.uri())
            .with_tokens("test-token", "refresh-token")
            .build()
            .unwrap();

        let result = client.health_check().await;
        assert!(result.is_ok());
    }

    #[tokio::test]
    async fn test_error_handling() {
        let mock_server = MockServer::start().await;

        Mock::given(method("GET"))
            .and(path("/health"))
            .and(header("Authorization", "Bearer test-token"))
            .respond_with(ResponseTemplate::new(401).set_body_json(json!({
                "error": {
                    "code": "BASILICA_API_AUTH_MISSING",
                    "message": "Authentication required",
                    "timestamp": "2024-01-01T00:00:00Z",
                    "retryable": false,
                }
            })))
            .mount(&mock_server)
            .await;

        let client = ClientBuilder::default()
            .base_url(mock_server.uri())
            .with_tokens("test-token", "refresh-token")
            .build()
            .unwrap();
        let result = client.health_check().await;

        assert!(result.is_err());
        assert!(matches!(
            result.unwrap_err(),
            ApiError::MissingAuthentication { .. }
        ));
    }

    #[test]
    fn test_builder_requires_auth() {
        let result = ClientBuilder::default().build();
        assert!(result.is_err());
        assert!(matches!(
            result.unwrap_err(),
            ApiError::InvalidRequest { .. }
        ));
    }

    #[test]
    fn test_builder_with_all_options() {
        let client = ClientBuilder::default()
            .base_url("https://api.basilica.ai")
            .with_tokens("test-token", "refresh-token")
            .timeout(Duration::from_secs(60))
            .connect_timeout(Duration::from_secs(10))
            .pool_max_idle_per_host(100)
            .build();

        assert!(client.is_ok());
    }
}
