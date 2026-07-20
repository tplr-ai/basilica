//! Managed Inference client for the Basilica SDK
//!
//! This module is the single source of truth for the managed-inference client
//! surface (see `MANAGED-INFERENCE-ENDPOINT-ARCHITECTURE.md` §4). Both the CLI
//! and the Python SDK (via PyO3) consume it.
//!
//! Two hosts are involved:
//!
//! - The **inference gateway** (`https://inference.basilica.ai`, override via
//!   `BASILICA_INFERENCE_ENDPOINT`) serves the OpenAI-compatible catalog:
//!   `GET /v1/models` and `GET /v1/models/{id}`.
//! - The **management plane** (basilica-api, the same base URL the main
//!   [`crate::BasilicaClient`] uses) serves usage rollups:
//!   `GET /v1/inference/usage/summary`.
//!
//! Both authenticate with `Authorization: Bearer <token>`, where the token is
//! a `basilica_...` API key or an OAuth JWT — resolved through the crate's
//! existing [`TokenManager`], so API-key pass-through and JWT refresh behave
//! exactly as they do for the main client.
//!
//! # Example
//!
//! ```rust,no_run
//! use basilica_sdk::inference::{InferenceClient, UsageQuery};
//!
//! # async fn example() -> Result<(), basilica_sdk::inference::InferenceError> {
//! let client = InferenceClient::new("basilica_...")?;
//!
//! let models = client.list_models().await?;
//! for model in &models {
//!     println!("{}", model.id);
//! }
//!
//! let rows = client.usage(UsageQuery::default()).await?;
//! for row in &rows {
//!     println!("{} {}: {} credits", row.date, row.model, row.charge_credits);
//! }
//!
//! // Hand this to the stock OpenAI client for chat completions:
//! let base_url = client.openai_base_url();
//! assert!(base_url.ends_with("/v1"));
//! # Ok(())
//! # }
//! ```

use crate::auth::TokenManager;
use crate::client::{DEFAULT_API_URL, DEFAULT_TIMEOUT_SECS};
use chrono::NaiveDate;
use reqwest::{RequestBuilder, Response, StatusCode};
use rust_decimal::Decimal;
use serde::{Deserialize, Serialize};
use std::sync::Arc;
use std::time::Duration;
use thiserror::Error;

/// Built-in default inference gateway endpoint
pub const DEFAULT_INFERENCE_ENDPOINT: &str = "https://inference.basilica.ai";

/// Environment variable overriding the inference gateway endpoint
pub const INFERENCE_ENDPOINT_ENV: &str = "BASILICA_INFERENCE_ENDPOINT";

/// Management-plane path serving usage rollups (basilica-api)
const USAGE_SUMMARY_PATH: &str = "/v1/inference/usage/summary";

/// Maximum characters of a raw error body retained in error messages
const ERROR_EXCERPT_LEN: usize = 500;

// ============================================================================
// Public types (wire contract)
// ============================================================================

/// One entry of the OpenAI-compatible `GET /v1/models` catalog.
///
/// The gateway renders `{"id", "object", "created", "owned_by"}` per model;
/// everything but `id` is tolerated as missing so registry extensions do not
/// break older clients.
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct InferenceModel {
    /// Model identifier (e.g. `llama-3.1-70b-instruct`)
    pub id: String,

    /// OpenAI object tag (`"model"`)
    #[serde(default)]
    pub object: Option<String>,

    /// Unix timestamp advertised in the OpenAI model object
    #[serde(default)]
    pub created: Option<i64>,

    /// Owning organization (`"basilica"`)
    #[serde(default)]
    pub owned_by: Option<String>,
}

/// Filter for the usage summary rollup.
///
/// Unset fields are omitted from the query string.
#[derive(Debug, Clone, Default, Serialize, Deserialize)]
pub struct UsageQuery {
    /// First day to include (UTC, inclusive)
    #[serde(skip_serializing_if = "Option::is_none")]
    pub from: Option<NaiveDate>,

    /// Last day to include (UTC, inclusive)
    #[serde(skip_serializing_if = "Option::is_none")]
    pub to: Option<NaiveDate>,

    /// Restrict to a single model
    #[serde(skip_serializing_if = "Option::is_none")]
    pub model: Option<String>,

    /// Restrict to a single API key ID
    #[serde(skip_serializing_if = "Option::is_none")]
    pub kid: Option<String>,
}

/// One row of the usage summary rollup.
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct InferenceUsageRow {
    /// UTC day the usage belongs to
    pub date: NaiveDate,

    /// Model the usage belongs to
    pub model: String,

    /// Tenant the usage belongs to (Auth0 subject)
    #[serde(default)]
    pub tenant_id: Option<String>,

    /// API key ID the usage was made with
    #[serde(default)]
    pub kid: Option<String>,

    /// Prompt (input) tokens
    pub prompt_tokens: u64,

    /// Completion (output) tokens
    pub completion_tokens: u64,

    /// Cached prompt tokens
    #[serde(default)]
    pub cached_tokens: u64,

    /// Credits charged (wire format is a decimal string, e.g. `"1.23456"`)
    pub charge_credits: Decimal,
}

/// Which quota cap tripped on a `429` (spec §4.3).
///
/// Parses the exact cap names the gateway emits in `error.cap`.
#[derive(Debug, Clone, Copy, PartialEq, Eq, Serialize, Deserialize)]
#[serde(rename_all = "lowercase")]
pub enum QuotaCap {
    /// Requests-per-minute window
    Rpm,
    /// Tokens-per-minute window (estimated at admission)
    Tpm,
    /// Concurrent open streams
    Concurrency,
    /// Monthly estimated-spend cap
    Budget,
}

impl QuotaCap {
    /// The exact string the gateway surfaces in the §4.3 `429` body
    pub fn as_str(self) -> &'static str {
        match self {
            QuotaCap::Rpm => "rpm",
            QuotaCap::Tpm => "tpm",
            QuotaCap::Concurrency => "concurrency",
            QuotaCap::Budget => "budget",
        }
    }

    /// Parse the gateway's cap name; `None` for anything outside the contract
    fn parse(raw: &str) -> Option<Self> {
        match raw {
            "rpm" => Some(QuotaCap::Rpm),
            "tpm" => Some(QuotaCap::Tpm),
            "concurrency" => Some(QuotaCap::Concurrency),
            "budget" => Some(QuotaCap::Budget),
            _ => None,
        }
    }
}

impl std::fmt::Display for QuotaCap {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        f.write_str(self.as_str())
    }
}

// ============================================================================
// Errors
// ============================================================================

/// Errors from the managed inference surface.
///
/// Mapped totally from the OpenAI-style error contract (spec §4.3): any
/// status/body combination yields a typed error — this module never panics on
/// malformed responses.
#[derive(Debug, Error)]
pub enum InferenceError {
    /// `401` — bad, missing, or revoked credential
    #[error("authentication failed (401): bad or missing credential")]
    Auth,

    /// `403` — the credential lacks the required inference scope
    #[error("permission denied (403): credential lacks the required inference scope")]
    Forbidden,

    /// `400` — the request was rejected as invalid, or the server spoke
    /// outside the contract (details carried in the message)
    #[error("invalid request: {0}")]
    Invalid(String),

    /// `404` — the model does not exist or is not available to this tenant
    #[error("model not found: {model}")]
    ModelNotFound {
        /// The model name that was requested
        model: String,
    },

    /// `402` — balance too low to serve the request
    #[error("insufficient credits (402): top up your balance")]
    InsufficientCredits,

    /// `429` — a quota cap tripped; `Retry-After` (seconds) when provided
    #[error(
        "quota exceeded (429): {cap} cap tripped{}",
        retry_after
            .map(|secs| format!("; retry after {secs}s"))
            .unwrap_or_default()
    )]
    QuotaExceeded {
        /// Which cap tripped
        cap: QuotaCap,
        /// Server-provided retry hint, seconds
        retry_after: Option<u64>,
    },

    /// `5xx` — pool saturated/unavailable or billing-degraded
    #[error(
        "inference service unavailable (5xx){}",
        retry_after
            .map(|secs| format!("; retry after {secs}s"))
            .unwrap_or_default()
    )]
    Unavailable {
        /// Server-provided retry hint, seconds
        retry_after: Option<u64>,
    },

    /// Transport-level failure (connect, timeout, TLS, ...)
    #[error("transport error: {0}")]
    Transport(#[from] reqwest::Error),

    /// A successful (2xx) response that did not match the wire contract
    #[error("failed to decode response: {0}")]
    Decode(String),
}

impl InferenceError {
    /// Whether the call is worth retrying (transport failures and 5xx)
    pub fn is_retryable(&self) -> bool {
        matches!(
            self,
            InferenceError::Transport(_) | InferenceError::Unavailable { .. }
        )
    }
}

// ============================================================================
// Client
// ============================================================================

/// Client for Basilica's managed inference platform.
///
/// Cheap to clone (one `Arc`); safe to share across tasks.
#[derive(Debug, Clone)]
pub struct InferenceClient {
    inner: Arc<Inner>,
}

#[derive(Debug)]
struct Inner {
    http: reqwest::Client,
    /// Management-plane base URL (basilica-api) — serves the usage summary
    api_base: String,
    /// Resolved inference gateway endpoint — serves the OpenAI catalog
    endpoint: String,
    token_manager: Arc<TokenManager>,
}

impl InferenceClient {
    /// Create a client authenticated with a `basilica_...` API key.
    ///
    /// Uses the default management-plane URL and resolves the gateway
    /// endpoint from `BASILICA_INFERENCE_ENDPOINT`, falling back to
    /// [`DEFAULT_INFERENCE_ENDPOINT`]. Use [`InferenceClient::builder`] for
    /// explicit control.
    pub fn new(api_key: impl Into<String>) -> Result<Self, InferenceError> {
        let api_key = api_key.into();
        InferenceClientBuilder::new().with_api_key(&api_key).build()
    }

    /// Start building a client with custom configuration
    pub fn builder() -> InferenceClientBuilder {
        InferenceClientBuilder::new()
    }

    /// The resolved inference gateway endpoint (no trailing slash)
    pub fn endpoint(&self) -> &str {
        &self.inner.endpoint
    }

    /// The management-plane base URL the usage summary is fetched from
    pub fn api_base(&self) -> &str {
        &self.inner.api_base
    }

    /// Base URL for OpenAI-compatible clients: `{endpoint}/v1`.
    ///
    /// Wire this into the stock OpenAI client together with the API key.
    pub fn openai_base_url(&self) -> String {
        format!("{}/v1", self.inner.endpoint)
    }

    /// List the tenant-visible model catalog (`GET {endpoint}/v1/models`)
    pub async fn list_models(&self) -> Result<Vec<InferenceModel>, InferenceError> {
        let url = format!("{}/v1/models", self.inner.endpoint);
        let body = self.get_body(&url, &[], None).await?;
        let payload: ModelListPayload =
            serde_json::from_str(&body).map_err(|e| decode_error(&e, &body))?;
        Ok(payload.into_models())
    }

    /// Fetch a single model's metadata (`GET {endpoint}/v1/models/{name}`)
    ///
    /// # Errors
    ///
    /// * [`InferenceError::ModelNotFound`] — unknown or unavailable model
    pub async fn get_model(&self, name: &str) -> Result<InferenceModel, InferenceError> {
        let url = format!(
            "{}/v1/models/{}",
            self.inner.endpoint,
            urlencoding::encode(name)
        );
        let body = self.get_body(&url, &[], Some(name)).await?;
        serde_json::from_str(&body).map_err(|e| decode_error(&e, &body))
    }

    /// Fetch usage rollups from the management plane
    /// (`GET {api_base}/v1/inference/usage/summary`)
    ///
    /// # Errors
    ///
    /// * [`InferenceError::Invalid`] — `from` is after `to`
    pub async fn usage(&self, query: UsageQuery) -> Result<Vec<InferenceUsageRow>, InferenceError> {
        if let (Some(from), Some(to)) = (query.from, query.to) {
            if from > to {
                return Err(InferenceError::Invalid(
                    "`from` must be on or before `to`".to_string(),
                ));
            }
        }

        let mut params: Vec<(&str, String)> = Vec::new();
        if let Some(from) = &query.from {
            params.push(("from", from.format("%Y-%m-%d").to_string()));
        }
        if let Some(to) = &query.to {
            params.push(("to", to.format("%Y-%m-%d").to_string()));
        }
        if let Some(model) = &query.model {
            params.push(("model", model.clone()));
        }
        if let Some(kid) = &query.kid {
            params.push(("kid", kid.clone()));
        }

        let url = format!("{}{}", self.inner.api_base, USAGE_SUMMARY_PATH);
        let body = self.get_body(&url, &params, None).await?;
        let payload: UsageSummaryPayload =
            serde_json::from_str(&body).map_err(|e| decode_error(&e, &body))?;
        Ok(payload.into_rows())
    }

    // ===== Private helpers =====

    /// Apply Bearer authentication to a request, same as the main client
    async fn apply_auth(&self, request: RequestBuilder) -> Result<RequestBuilder, InferenceError> {
        let token = self
            .inner
            .token_manager
            .get_access_token()
            .await
            .map_err(|_| InferenceError::Auth)?;
        Ok(request.header("Authorization", format!("Bearer {token}")))
    }

    /// GET a URL, returning the raw body on success or the mapped contract
    /// error otherwise. `model` names the requested model so a `404` can be
    /// reported as [`InferenceError::ModelNotFound`].
    async fn get_body(
        &self,
        url: &str,
        query: &[(&str, String)],
        model: Option<&str>,
    ) -> Result<String, InferenceError> {
        let mut request = self.inner.http.get(url);
        if !query.is_empty() {
            request = request.query(query);
        }
        let request = self.apply_auth(request).await?;

        let response = request.send().await.map_err(InferenceError::Transport)?;
        let status = response.status();
        let retry_after = retry_after_hint(&response);
        let body = response.text().await.map_err(InferenceError::Transport)?;

        if !status.is_success() {
            return Err(map_error_response(status, retry_after, &body, model));
        }
        Ok(body)
    }
}

/// Builder for constructing an [`InferenceClient`] with custom configuration
#[derive(Default)]
pub struct InferenceClientBuilder {
    api_base: Option<String>,
    endpoint: Option<String>,
    timeout: Option<Duration>,
    api_key: Option<String>,
    access_token: Option<String>,
    refresh_token: Option<String>,
    token_manager: Option<Arc<TokenManager>>,
}

impl InferenceClientBuilder {
    /// Create a new builder with default values
    pub fn new() -> Self {
        Self::default()
    }

    /// Set the management-plane base URL (basilica-api) the usage summary is
    /// served from. Defaults to [`DEFAULT_API_URL`]. Pass the same base URL
    /// the main [`crate::BasilicaClient`] was built with.
    pub fn api_base(mut self, url: impl Into<String>) -> Self {
        self.api_base = Some(url.into());
        self
    }

    /// Explicit inference gateway endpoint override.
    ///
    /// Precedence: this override > `BASILICA_INFERENCE_ENDPOINT` env var >
    /// [`DEFAULT_INFERENCE_ENDPOINT`].
    pub fn endpoint(mut self, url: impl Into<String>) -> Self {
        self.endpoint = Some(url.into());
        self
    }

    /// Set the request timeout
    pub fn timeout(mut self, timeout: Duration) -> Self {
        self.timeout = Some(timeout);
        self
    }

    /// Use a `basilica_...` API key for authentication
    pub fn with_api_key(mut self, api_key: &str) -> Self {
        self.api_key = Some(api_key.to_string());
        self
    }

    /// Use OAuth tokens for authentication (both tokens required)
    pub fn with_tokens(
        mut self,
        access_token: impl Into<String>,
        refresh_token: impl Into<String>,
    ) -> Self {
        self.access_token = Some(access_token.into());
        self.refresh_token = Some(refresh_token.into());
        self
    }

    /// Share an existing token manager (e.g. the one backing the main client)
    pub fn with_token_manager(mut self, token_manager: Arc<TokenManager>) -> Self {
        self.token_manager = Some(token_manager);
        self
    }

    /// Build the client
    pub fn build(self) -> Result<InferenceClient, InferenceError> {
        // API key takes precedence, same as ClientBuilder in client.rs
        let token_manager = if let Some(token_manager) = self.token_manager {
            token_manager
        } else if let Some(api_key) = self.api_key {
            Arc::new(TokenManager::new_api_key(api_key))
        } else if let (Some(access_token), Some(refresh_token)) =
            (self.access_token, self.refresh_token)
        {
            Arc::new(TokenManager::new_direct(access_token, refresh_token))
        } else {
            return Err(InferenceError::Invalid(
                "no credentials configured: use with_api_key(), with_tokens(), or \
                 with_token_manager()"
                    .to_string(),
            ));
        };

        let timeout = self
            .timeout
            .unwrap_or(Duration::from_secs(DEFAULT_TIMEOUT_SECS));
        let http = reqwest::Client::builder()
            .timeout(timeout)
            .build()
            .map_err(InferenceError::Transport)?;

        let api_base = self.api_base.unwrap_or_else(|| DEFAULT_API_URL.to_string());
        let api_base = api_base.trim_end_matches('/').to_string();

        let endpoint = resolve_endpoint(
            self.endpoint.as_deref(),
            std::env::var(INFERENCE_ENDPOINT_ENV).ok().as_deref(),
        );

        Ok(InferenceClient {
            inner: Arc::new(Inner {
                http,
                api_base,
                endpoint,
                token_manager,
            }),
        })
    }
}

// ============================================================================
// Wire envelopes and error mapping (private)
// ============================================================================

/// OpenAI-compatible model list payload (tolerates a bare array)
#[derive(Debug, Deserialize)]
#[serde(untagged)]
enum ModelListPayload {
    Object { data: Vec<InferenceModel> },
    List(Vec<InferenceModel>),
}

impl ModelListPayload {
    fn into_models(self) -> Vec<InferenceModel> {
        match self {
            ModelListPayload::Object { data } => data,
            ModelListPayload::List(models) => models,
        }
    }
}

/// Usage summary payload: `{"rows": [...]}` (tolerates a bare array)
#[derive(Debug, Deserialize)]
#[serde(untagged)]
enum UsageSummaryPayload {
    Object { rows: Vec<InferenceUsageRow> },
    Rows(Vec<InferenceUsageRow>),
}

impl UsageSummaryPayload {
    fn into_rows(self) -> Vec<InferenceUsageRow> {
        match self {
            UsageSummaryPayload::Object { rows } => rows,
            UsageSummaryPayload::Rows(rows) => rows,
        }
    }
}

/// The subset of the OpenAI-style error body this client acts on
/// (`{"error": {"message", "cap", ...}}`); unknown fields are ignored and a
/// body that does not match at all falls back to a raw excerpt.
#[derive(Debug, Deserialize)]
struct OpenAiErrorEnvelope {
    error: OpenAiErrorDetail,
}

#[derive(Debug, Deserialize)]
struct OpenAiErrorDetail {
    #[serde(default)]
    message: Option<String>,
    #[serde(default)]
    cap: Option<String>,
}

/// Resolve the inference gateway endpoint.
///
/// Precedence: explicit override > env var > built-in default. Empty values
/// are treated as unset; trailing slashes are stripped.
fn resolve_endpoint(explicit: Option<&str>, env: Option<&str>) -> String {
    explicit
        .filter(|v| !v.trim().is_empty())
        .or_else(|| env.filter(|v| !v.trim().is_empty()))
        .unwrap_or(DEFAULT_INFERENCE_ENDPOINT)
        .trim_end_matches('/')
        .to_string()
}

/// Extract the `Retry-After` hint (integer seconds) from a response
fn retry_after_hint(response: &Response) -> Option<u64> {
    response
        .headers()
        .get(reqwest::header::RETRY_AFTER)?
        .to_str()
        .ok()?
        .trim()
        .parse()
        .ok()
}

/// Parse the OpenAI-style error envelope; `None` on any deviation
fn parse_openai_error(body: &str) -> Option<OpenAiErrorDetail> {
    serde_json::from_str::<OpenAiErrorEnvelope>(body)
        .ok()
        .map(|envelope| envelope.error)
}

/// Truncate a raw body for inclusion in an error message (UTF-8 safe)
fn excerpt(body: &str) -> String {
    if body.trim().is_empty() {
        return "<empty body>".to_string();
    }
    let truncated: String = body.chars().take(ERROR_EXCERPT_LEN).collect();
    if body.chars().count() > ERROR_EXCERPT_LEN {
        format!("{truncated}...")
    } else {
        truncated
    }
}

/// A 2xx body that did not match the expected shape
fn decode_error(error: &serde_json::Error, body: &str) -> InferenceError {
    InferenceError::Decode(format!("{error}; body: {}", excerpt(body)))
}

/// Map a non-2xx status plus OpenAI-style error body to a typed error.
///
/// Total by construction: every status and every body — JSON or not —
/// produces a value. An unrecognized `429` cap name surfaces as
/// [`InferenceError::Invalid`] carrying the raw body excerpt rather than
/// being silently dropped.
fn map_error_response(
    status: StatusCode,
    retry_after: Option<u64>,
    body: &str,
    model: Option<&str>,
) -> InferenceError {
    let parsed = parse_openai_error(body);
    let message = parsed.as_ref().and_then(|detail| detail.message.clone());

    match status {
        StatusCode::UNAUTHORIZED => InferenceError::Auth,
        StatusCode::FORBIDDEN => InferenceError::Forbidden,
        StatusCode::BAD_REQUEST => {
            InferenceError::Invalid(message.unwrap_or_else(|| excerpt(body)))
        }
        StatusCode::PAYMENT_REQUIRED => InferenceError::InsufficientCredits,
        StatusCode::NOT_FOUND => match model {
            Some(name) => InferenceError::ModelNotFound {
                model: name.to_string(),
            },
            None => InferenceError::Invalid(format!(
                "resource not found (404): {}",
                message.unwrap_or_else(|| excerpt(body))
            )),
        },
        StatusCode::TOO_MANY_REQUESTS => {
            let cap_raw = parsed.as_ref().and_then(|detail| detail.cap.as_deref());
            match cap_raw {
                Some(raw) => match QuotaCap::parse(raw) {
                    Some(cap) => InferenceError::QuotaExceeded { cap, retry_after },
                    None => InferenceError::Invalid(format!(
                        "unrecognized quota cap `{raw}` in 429 response: {}",
                        excerpt(body)
                    )),
                },
                None => InferenceError::Invalid(format!(
                    "429 quota response is missing the cap name: {}",
                    excerpt(body)
                )),
            }
        }
        StatusCode::SERVICE_UNAVAILABLE => InferenceError::Unavailable { retry_after },
        s if s.is_server_error() => InferenceError::Unavailable { retry_after },
        s => InferenceError::Invalid(format!(
            "unexpected status {s}: {}",
            message.unwrap_or_else(|| excerpt(body))
        )),
    }
}

// ============================================================================
// Tests
// ============================================================================

#[cfg(test)]
mod tests {
    use super::*;
    use serde_json::json;
    use wiremock::matchers::{header, method, path, query_param};
    use wiremock::{Mock, MockServer, ResponseTemplate};

    const TEST_KEY: &str = "basilica_test-key";

    /// Client pointed at mock servers; `api_base` serves usage, `endpoint`
    /// serves the model catalog
    fn test_client(api_base: &str, endpoint: &str) -> InferenceClient {
        InferenceClient::builder()
            .api_base(api_base)
            .endpoint(endpoint)
            .with_api_key(TEST_KEY)
            .timeout(Duration::from_secs(5))
            .build()
            .unwrap()
    }

    fn gateway_client(mock_server: &MockServer) -> InferenceClient {
        test_client(&mock_server.uri(), &mock_server.uri())
    }

    fn openai_error_body(message: &str) -> serde_json::Value {
        json!({
            "error": {
                "message": message,
                "type": "invalid_request_error",
                "param": null,
                "code": "some_error_code",
            }
        })
    }

    fn quota_error_body(cap: &str) -> serde_json::Value {
        json!({
            "error": {
                "message": format!("quota exceeded: {cap} cap tripped (61 > 60)"),
                "type": "rate_limit_error",
                "param": null,
                "code": "quota_exceeded",
                "cap": cap,
            }
        })
    }

    /// Mount a 429 with `Retry-After: 30` naming `cap`, then list models
    async fn quota_error_for_cap(cap: &str) -> InferenceError {
        let mock_server = MockServer::start().await;
        Mock::given(method("GET"))
            .and(path("/v1/models"))
            .respond_with(
                ResponseTemplate::new(429)
                    .insert_header("Retry-After", "30")
                    .set_body_json(quota_error_body(cap)),
            )
            .mount(&mock_server)
            .await;
        gateway_client(&mock_server)
            .list_models()
            .await
            .unwrap_err()
    }

    // ------------------------------------------------------------------
    // Model catalog
    // ------------------------------------------------------------------

    #[tokio::test]
    async fn list_models_parses_openai_envelope_and_sends_auth() {
        let mock_server = MockServer::start().await;
        Mock::given(method("GET"))
            .and(path("/v1/models"))
            .and(header("Authorization", format!("Bearer {TEST_KEY}").as_str()))
            .respond_with(ResponseTemplate::new(200).set_body_json(json!({
                "object": "list",
                "data": [
                    {"id": "llama-3.1-70b-instruct", "object": "model", "created": 1700000000, "owned_by": "basilica"},
                    {"id": "qwen2.5-7b", "object": "model", "created": 1700000100, "owned_by": "basilica"}
                ]
            })))
            .mount(&mock_server)
            .await;

        let models = gateway_client(&mock_server).list_models().await.unwrap();

        assert_eq!(models.len(), 2);
        assert_eq!(models[0].id, "llama-3.1-70b-instruct");
        assert_eq!(models[0].object.as_deref(), Some("model"));
        assert_eq!(models[0].created, Some(1700000000));
        assert_eq!(models[0].owned_by.as_deref(), Some("basilica"));
        assert_eq!(models[1].id, "qwen2.5-7b");
    }

    #[tokio::test]
    async fn list_models_tolerates_bare_array_and_missing_fields() {
        let mock_server = MockServer::start().await;
        Mock::given(method("GET"))
            .and(path("/v1/models"))
            .respond_with(ResponseTemplate::new(200).set_body_json(json!([
                {"id": "llama-3.1-70b-instruct"}
            ])))
            .mount(&mock_server)
            .await;

        let models = gateway_client(&mock_server).list_models().await.unwrap();

        assert_eq!(models.len(), 1);
        assert_eq!(models[0].id, "llama-3.1-70b-instruct");
        assert!(models[0].owned_by.is_none());
        assert!(models[0].created.is_none());
        assert!(models[0].object.is_none());
    }

    #[tokio::test]
    async fn get_model_parses_model_object() {
        let mock_server = MockServer::start().await;
        Mock::given(method("GET"))
            .and(path("/v1/models/llama-3.1-70b-instruct"))
            .and(header(
                "Authorization",
                format!("Bearer {TEST_KEY}").as_str(),
            ))
            .respond_with(ResponseTemplate::new(200).set_body_json(json!({
                "id": "llama-3.1-70b-instruct",
                "object": "model",
                "created": 1700000000,
                "owned_by": "basilica"
            })))
            .mount(&mock_server)
            .await;

        let model = gateway_client(&mock_server)
            .get_model("llama-3.1-70b-instruct")
            .await
            .unwrap();

        assert_eq!(model.id, "llama-3.1-70b-instruct");
        assert_eq!(model.created, Some(1700000000));
    }

    #[tokio::test]
    async fn get_model_404_is_model_not_found() {
        let mock_server = MockServer::start().await;
        // The gateway's exact 404 shape for unknown/retired models
        Mock::given(method("GET"))
            .and(path("/v1/models/no-such-model"))
            .respond_with(ResponseTemplate::new(404).set_body_json(json!({
                "error": {
                    "message": "The model does not exist or is not available.",
                    "type": "invalid_request_error",
                    "param": "model",
                    "code": "model_not_found"
                }
            })))
            .mount(&mock_server)
            .await;

        let err = gateway_client(&mock_server)
            .get_model("no-such-model")
            .await
            .unwrap_err();

        match &err {
            InferenceError::ModelNotFound { model } => assert_eq!(model, "no-such-model"),
            other => panic!("expected ModelNotFound, got {other:?}"),
        }
        assert!(!err.is_retryable());
    }

    // ------------------------------------------------------------------
    // Usage summary
    // ------------------------------------------------------------------

    #[tokio::test]
    async fn usage_parses_rows_with_decimal_dates_and_query_params() {
        let mock_server = MockServer::start().await;
        Mock::given(method("GET"))
            .and(path(USAGE_SUMMARY_PATH))
            .and(query_param("from", "2026-07-01"))
            .and(query_param("to", "2026-07-20"))
            .and(query_param("model", "llama-3.1-70b-instruct"))
            .and(query_param("kid", "0123456789abcdef0123456789abcdef"))
            .and(header(
                "Authorization",
                format!("Bearer {TEST_KEY}").as_str(),
            ))
            .respond_with(ResponseTemplate::new(200).set_body_json(json!({
                "rows": [
                    {
                        "date": "2026-07-19",
                        "model": "llama-3.1-70b-instruct",
                        "tenant_id": "github|434149",
                        "kid": "0123456789abcdef0123456789abcdef",
                        "prompt_tokens": 1200,
                        "completion_tokens": 340,
                        "cached_tokens": 100,
                        "charge_credits": "1.23456"
                    },
                    {
                        "date": "2026-07-20",
                        "model": "llama-3.1-70b-instruct",
                        "prompt_tokens": 800,
                        "completion_tokens": 160,
                        "charge_credits": "0.00544"
                    }
                ]
            })))
            .mount(&mock_server)
            .await;

        let query = UsageQuery {
            from: NaiveDate::from_ymd_opt(2026, 7, 1),
            to: NaiveDate::from_ymd_opt(2026, 7, 20),
            model: Some("llama-3.1-70b-instruct".to_string()),
            kid: Some("0123456789abcdef0123456789abcdef".to_string()),
        };
        let rows = gateway_client(&mock_server).usage(query).await.unwrap();

        assert_eq!(rows.len(), 2);

        let first = &rows[0];
        assert_eq!(first.date, NaiveDate::from_ymd_opt(2026, 7, 19).unwrap());
        assert_eq!(first.model, "llama-3.1-70b-instruct");
        assert_eq!(first.tenant_id.as_deref(), Some("github|434149"));
        assert_eq!(
            first.kid.as_deref(),
            Some("0123456789abcdef0123456789abcdef")
        );
        assert_eq!(first.prompt_tokens, 1200);
        assert_eq!(first.completion_tokens, 340);
        assert_eq!(first.cached_tokens, 100);
        assert_eq!(first.charge_credits, "1.23456".parse::<Decimal>().unwrap());

        // Optional fields tolerated as missing
        let second = &rows[1];
        assert!(second.tenant_id.is_none());
        assert!(second.kid.is_none());
        assert_eq!(second.cached_tokens, 0);
        assert_eq!(second.charge_credits, "0.00544".parse::<Decimal>().unwrap());
    }

    #[tokio::test]
    async fn usage_tolerates_bare_array_and_sends_no_params_when_unset() {
        let mock_server = MockServer::start().await;
        Mock::given(method("GET"))
            .and(path(USAGE_SUMMARY_PATH))
            .respond_with(ResponseTemplate::new(200).set_body_json(json!([
                {
                    "date": "2026-07-19",
                    "model": "qwen2.5-7b",
                    "tenant_id": "github|434149",
                    "kid": null,
                    "prompt_tokens": 1,
                    "completion_tokens": 2,
                    "cached_tokens": 3,
                    "charge_credits": "0.5"
                }
            ])))
            .mount(&mock_server)
            .await;

        let rows = gateway_client(&mock_server)
            .usage(UsageQuery::default())
            .await
            .unwrap();

        assert_eq!(rows.len(), 1);
        assert_eq!(rows[0].charge_credits, "0.5".parse::<Decimal>().unwrap());

        let requests = mock_server.received_requests().await.unwrap();
        assert_eq!(requests.len(), 1);
        assert_eq!(requests[0].url.query(), None);
    }

    #[tokio::test]
    async fn usage_rejects_inverted_date_range_without_http() {
        let mock_server = MockServer::start().await;
        let query = UsageQuery {
            from: NaiveDate::from_ymd_opt(2026, 7, 20),
            to: NaiveDate::from_ymd_opt(2026, 7, 1),
            ..Default::default()
        };
        let err = gateway_client(&mock_server).usage(query).await.unwrap_err();
        assert!(matches!(err, InferenceError::Invalid(_)));
        // No request may have been issued
        let requests = mock_server.received_requests().await.unwrap();
        assert!(requests.is_empty());
    }

    #[tokio::test]
    async fn usage_hits_the_management_plane_not_the_gateway() {
        let api_server = MockServer::start().await;
        let gateway_server = MockServer::start().await;
        Mock::given(method("GET"))
            .and(path(USAGE_SUMMARY_PATH))
            .respond_with(ResponseTemplate::new(200).set_body_json(json!({"rows": []})))
            .mount(&api_server)
            .await;

        let client = test_client(&api_server.uri(), &gateway_server.uri());
        let rows = client.usage(UsageQuery::default()).await.unwrap();
        assert!(rows.is_empty());
        let gateway_requests = gateway_server.received_requests().await.unwrap();
        assert!(gateway_requests.is_empty());
    }

    // ------------------------------------------------------------------
    // Error mapping (spec §4.3)
    // ------------------------------------------------------------------

    #[tokio::test]
    async fn error_401_maps_to_auth() {
        let mock_server = MockServer::start().await;
        Mock::given(method("GET"))
            .and(path("/v1/models"))
            .respond_with(ResponseTemplate::new(401).set_body_json(openai_error_body("bad key")))
            .mount(&mock_server)
            .await;

        let err = gateway_client(&mock_server)
            .list_models()
            .await
            .unwrap_err();
        assert!(matches!(err, InferenceError::Auth));
        assert!(!err.is_retryable());
    }

    #[tokio::test]
    async fn error_403_maps_to_forbidden() {
        let mock_server = MockServer::start().await;
        Mock::given(method("GET"))
            .and(path("/v1/models"))
            .respond_with(
                ResponseTemplate::new(403).set_body_json(openai_error_body("missing scope")),
            )
            .mount(&mock_server)
            .await;

        let err = gateway_client(&mock_server)
            .list_models()
            .await
            .unwrap_err();
        assert!(matches!(err, InferenceError::Forbidden));
    }

    #[tokio::test]
    async fn error_400_maps_to_invalid_with_server_message() {
        let mock_server = MockServer::start().await;
        Mock::given(method("GET"))
            .and(path("/v1/models"))
            .respond_with(
                ResponseTemplate::new(400)
                    .set_body_json(openai_error_body("reserved field `cache_salt` rejected")),
            )
            .mount(&mock_server)
            .await;

        let err = gateway_client(&mock_server)
            .list_models()
            .await
            .unwrap_err();
        match err {
            InferenceError::Invalid(message) => {
                assert!(message.contains("cache_salt"), "{message}")
            }
            other => panic!("expected Invalid, got {other:?}"),
        }
    }

    #[tokio::test]
    async fn error_402_maps_to_insufficient_credits() {
        let mock_server = MockServer::start().await;
        Mock::given(method("GET"))
            .and(path("/v1/models"))
            .respond_with(
                ResponseTemplate::new(402).set_body_json(openai_error_body("balance too low")),
            )
            .mount(&mock_server)
            .await;

        let err = gateway_client(&mock_server)
            .list_models()
            .await
            .unwrap_err();
        assert!(matches!(err, InferenceError::InsufficientCredits));
    }

    #[tokio::test]
    async fn error_404_on_usage_maps_to_invalid_not_model_not_found() {
        let mock_server = MockServer::start().await;
        Mock::given(method("GET"))
            .and(path(USAGE_SUMMARY_PATH))
            .respond_with(
                ResponseTemplate::new(404).set_body_json(openai_error_body("no such route")),
            )
            .mount(&mock_server)
            .await;

        let err = gateway_client(&mock_server)
            .usage(UsageQuery::default())
            .await
            .unwrap_err();
        assert!(matches!(err, InferenceError::Invalid(_)));
    }

    #[tokio::test]
    async fn error_429_maps_each_cap_with_retry_after() {
        for (raw, expected) in [
            ("rpm", QuotaCap::Rpm),
            ("tpm", QuotaCap::Tpm),
            ("concurrency", QuotaCap::Concurrency),
            ("budget", QuotaCap::Budget),
        ] {
            let err = quota_error_for_cap(raw).await;
            match err {
                InferenceError::QuotaExceeded { cap, retry_after } => {
                    assert_eq!(cap, expected, "cap raw={raw}");
                    assert_eq!(retry_after, Some(30), "cap raw={raw}");
                }
                other => panic!("expected QuotaExceeded for cap {raw}, got {other:?}"),
            }
        }
    }

    #[tokio::test]
    async fn error_429_without_retry_after_header() {
        let mock_server = MockServer::start().await;
        Mock::given(method("GET"))
            .and(path("/v1/models"))
            .respond_with(ResponseTemplate::new(429).set_body_json(quota_error_body("tpm")))
            .mount(&mock_server)
            .await;

        let err = gateway_client(&mock_server)
            .list_models()
            .await
            .unwrap_err();
        match err {
            InferenceError::QuotaExceeded { cap, retry_after } => {
                assert_eq!(cap, QuotaCap::Tpm);
                assert_eq!(retry_after, None);
            }
            other => panic!("expected QuotaExceeded, got {other:?}"),
        }
    }

    #[tokio::test]
    async fn error_429_unknown_cap_is_invalid_with_raw_excerpt() {
        let err = quota_error_for_cap("qps").await;
        match err {
            InferenceError::Invalid(message) => {
                assert!(message.contains("qps"), "{message}");
                // The raw body excerpt is preserved, not silently lost
                assert!(message.contains("quota_exceeded"), "{message}");
            }
            other => panic!("expected Invalid, got {other:?}"),
        }
    }

    #[tokio::test]
    async fn error_429_missing_cap_is_invalid_with_raw_excerpt() {
        let mock_server = MockServer::start().await;
        Mock::given(method("GET"))
            .and(path("/v1/models"))
            .respond_with(ResponseTemplate::new(429).set_body_json(json!({
                "error": {"message": "slow down", "type": "rate_limit_error"}
            })))
            .mount(&mock_server)
            .await;

        let err = gateway_client(&mock_server)
            .list_models()
            .await
            .unwrap_err();
        match err {
            InferenceError::Invalid(message) => {
                assert!(message.contains("slow down"), "{message}")
            }
            other => panic!("expected Invalid, got {other:?}"),
        }
    }

    #[tokio::test]
    async fn error_503_maps_to_unavailable_with_retry_after() {
        let mock_server = MockServer::start().await;
        Mock::given(method("GET"))
            .and(path("/v1/models"))
            .respond_with(
                ResponseTemplate::new(503)
                    .insert_header("Retry-After", "5")
                    .set_body_json(openai_error_body("pool saturated")),
            )
            .mount(&mock_server)
            .await;

        let err = gateway_client(&mock_server)
            .list_models()
            .await
            .unwrap_err();
        match err {
            InferenceError::Unavailable { retry_after } => {
                assert_eq!(retry_after, Some(5))
            }
            other => panic!("expected Unavailable, got {other:?}"),
        }
        assert!(err.is_retryable());
    }

    #[tokio::test]
    async fn error_503_without_retry_after_header() {
        let mock_server = MockServer::start().await;
        Mock::given(method("GET"))
            .and(path("/v1/models"))
            .respond_with(ResponseTemplate::new(503).set_body_string("pool empty"))
            .mount(&mock_server)
            .await;

        let err = gateway_client(&mock_server)
            .list_models()
            .await
            .unwrap_err();
        assert!(matches!(
            err,
            InferenceError::Unavailable { retry_after: None }
        ));
    }

    #[tokio::test]
    async fn error_other_5xx_maps_to_unavailable() {
        let mock_server = MockServer::start().await;
        Mock::given(method("GET"))
            .and(path("/v1/models"))
            .respond_with(ResponseTemplate::new(500).set_body_string("boom"))
            .mount(&mock_server)
            .await;

        let err = gateway_client(&mock_server)
            .list_models()
            .await
            .unwrap_err();
        assert!(matches!(err, InferenceError::Unavailable { .. }));
        assert!(err.is_retryable());
    }

    #[tokio::test]
    async fn error_unexpected_status_maps_to_invalid_with_excerpt() {
        let mock_server = MockServer::start().await;
        Mock::given(method("GET"))
            .and(path("/v1/models"))
            .respond_with(ResponseTemplate::new(418).set_body_string("teapot"))
            .mount(&mock_server)
            .await;

        let err = gateway_client(&mock_server)
            .list_models()
            .await
            .unwrap_err();
        match err {
            InferenceError::Invalid(message) => {
                assert!(message.contains("418"), "{message}");
                assert!(message.contains("teapot"), "{message}");
            }
            other => panic!("expected Invalid, got {other:?}"),
        }
    }

    #[tokio::test]
    async fn error_mapping_is_total_on_weird_bodies() {
        for (status, body) in [
            (401, "not json at all"),
            (400, ""),
            (429, r#"{"error": "a bare string"}"#),
            (429, r#"{"error": {"cap": 42}}"#),
            (503, "<html>proxy error</html>"),
            (404, r#"{"unexpected": true}"#),
        ] {
            let mock_server = MockServer::start().await;
            Mock::given(method("GET"))
                .and(path("/v1/models"))
                .respond_with(ResponseTemplate::new(status).set_body_string(body))
                .mount(&mock_server)
                .await;

            // Must produce a typed error, never panic
            let err = gateway_client(&mock_server)
                .list_models()
                .await
                .unwrap_err();
            match status {
                401 => assert!(matches!(err, InferenceError::Auth), "{err:?}"),
                400 => assert!(matches!(err, InferenceError::Invalid(_)), "{err:?}"),
                429 => assert!(matches!(err, InferenceError::Invalid(_)), "{err:?}"),
                503 => assert!(matches!(err, InferenceError::Unavailable { .. }), "{err:?}"),
                404 => assert!(matches!(err, InferenceError::Invalid(_)), "{err:?}"),
                _ => unreachable!(),
            }
        }
    }

    // ------------------------------------------------------------------
    // Decode and transport failures
    // ------------------------------------------------------------------

    #[tokio::test]
    async fn decode_error_on_undecodable_success_body() {
        let mock_server = MockServer::start().await;
        Mock::given(method("GET"))
            .and(path("/v1/models"))
            .respond_with(ResponseTemplate::new(200).set_body_string(r#"{"surprise": true}"#))
            .mount(&mock_server)
            .await;

        let err = gateway_client(&mock_server)
            .list_models()
            .await
            .unwrap_err();
        match &err {
            InferenceError::Decode(message) => {
                assert!(message.contains("surprise"), "{message}")
            }
            other => panic!("expected Decode, got {other:?}"),
        }
        assert!(!err.is_retryable());
    }

    #[tokio::test]
    async fn transport_error_on_connection_refused() {
        let client = InferenceClient::builder()
            .api_base("http://127.0.0.1:1")
            .endpoint("http://127.0.0.1:1")
            .with_api_key(TEST_KEY)
            .timeout(Duration::from_millis(500))
            .build()
            .unwrap();

        let err = client.list_models().await.unwrap_err();
        assert!(matches!(err, InferenceError::Transport(_)), "{err:?}");
        assert!(err.is_retryable());
    }

    // ------------------------------------------------------------------
    // Endpoint resolution and accessors
    // ------------------------------------------------------------------

    #[test]
    fn endpoint_resolution_precedence() {
        // Default
        assert_eq!(resolve_endpoint(None, None), DEFAULT_INFERENCE_ENDPOINT);
        // Env beats default
        assert_eq!(resolve_endpoint(None, Some("http://env:1")), "http://env:1");
        // Explicit beats env
        assert_eq!(
            resolve_endpoint(Some("http://flag:2"), Some("http://env:1")),
            "http://flag:2"
        );
        // Blank values are treated as unset
        assert_eq!(
            resolve_endpoint(Some(""), Some("  ")),
            DEFAULT_INFERENCE_ENDPOINT
        );
        // Trailing slashes are stripped
        assert_eq!(
            resolve_endpoint(Some("http://flag:2/"), None),
            "http://flag:2"
        );
    }

    #[test]
    fn builder_uses_explicit_endpoint_and_trims_api_base() {
        let client = InferenceClient::builder()
            .api_base("http://api:8080/")
            .endpoint("http://gateway:9090/")
            .with_api_key(TEST_KEY)
            .build()
            .unwrap();

        assert_eq!(client.endpoint(), "http://gateway:9090");
        assert_eq!(client.api_base(), "http://api:8080");
        assert_eq!(client.openai_base_url(), "http://gateway:9090/v1");
    }

    #[test]
    fn builder_requires_credentials() {
        let err = InferenceClient::builder().build().unwrap_err();
        assert!(matches!(err, InferenceError::Invalid(_)));
    }

    #[test]
    fn new_uses_default_api_base() {
        let client = InferenceClient::new(TEST_KEY).unwrap();
        assert_eq!(client.api_base(), DEFAULT_API_URL);
        // Endpoint came from explicit-env-default resolution; with the env var
        // unset in CI this is the built-in default.
        if std::env::var(INFERENCE_ENDPOINT_ENV).is_err() {
            assert_eq!(client.endpoint(), DEFAULT_INFERENCE_ENDPOINT);
            assert_eq!(
                client.openai_base_url(),
                format!("{DEFAULT_INFERENCE_ENDPOINT}/v1")
            );
        }
    }

    #[test]
    fn client_is_cheap_to_clone_and_shares_state() {
        let client = InferenceClient::new(TEST_KEY).unwrap();
        let clone = client.clone();
        assert!(Arc::ptr_eq(&client.inner, &clone.inner));
    }

    // ------------------------------------------------------------------
    // Types
    // ------------------------------------------------------------------

    #[test]
    fn quota_cap_parses_exact_names() {
        assert_eq!(QuotaCap::parse("rpm"), Some(QuotaCap::Rpm));
        assert_eq!(QuotaCap::parse("tpm"), Some(QuotaCap::Tpm));
        assert_eq!(QuotaCap::parse("concurrency"), Some(QuotaCap::Concurrency));
        assert_eq!(QuotaCap::parse("budget"), Some(QuotaCap::Budget));
        for bad in ["RPM", "qps", "", "rate", "rppm"] {
            assert_eq!(QuotaCap::parse(bad), None, "{bad}");
        }
        assert_eq!(QuotaCap::Concurrency.to_string(), "concurrency");
    }

    #[test]
    fn quota_cap_serde_uses_contract_names() {
        assert_eq!(serde_json::to_string(&QuotaCap::Tpm).unwrap(), "\"tpm\"");
        assert_eq!(
            serde_json::from_str::<QuotaCap>("\"budget\"").unwrap(),
            QuotaCap::Budget
        );
    }

    #[test]
    fn usage_query_serializes_only_set_fields() {
        let query = UsageQuery {
            from: NaiveDate::from_ymd_opt(2026, 7, 1),
            model: Some("m".to_string()),
            ..Default::default()
        };
        let value = serde_json::to_value(&query).unwrap();
        assert_eq!(value, json!({"from": "2026-07-01", "model": "m"}));
    }

    #[test]
    fn usage_row_serializes_decimal_as_string() {
        let row = InferenceUsageRow {
            date: NaiveDate::from_ymd_opt(2026, 7, 19).unwrap(),
            model: "m".to_string(),
            tenant_id: None,
            kid: None,
            prompt_tokens: 1,
            completion_tokens: 2,
            cached_tokens: 3,
            charge_credits: "1.5".parse::<Decimal>().unwrap(),
        };
        let value = serde_json::to_value(&row).unwrap();
        assert_eq!(value["charge_credits"], json!("1.5"));
        assert_eq!(value["date"], json!("2026-07-19"));
    }

    #[test]
    fn quota_exceeded_display_includes_retry_hint() {
        let with_hint = InferenceError::QuotaExceeded {
            cap: QuotaCap::Rpm,
            retry_after: Some(30),
        };
        assert!(with_hint.to_string().contains("30"), "{with_hint}");
        let without_hint = InferenceError::QuotaExceeded {
            cap: QuotaCap::Budget,
            retry_after: None,
        };
        assert!(
            without_hint.to_string().contains("budget"),
            "{without_hint}"
        );
    }
}
