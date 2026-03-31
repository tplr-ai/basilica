use anyhow::{Context, Result};
use async_trait::async_trait;
use chrono::{DateTime, Utc};
use reqwest::Client;
use rust_decimal::Decimal;
use serde::de::DeserializeOwned;
use serde::{Deserialize, Serialize};
use serde_json::Value;
use std::collections::HashMap;
use std::sync::atomic::{AtomicU32, Ordering};
use std::sync::Arc;
use std::time::{Duration, Instant};
use tokio::sync::{Mutex, RwLock};
use tracing::{debug, warn};

#[derive(Debug, thiserror::Error)]
pub enum BasilicaApiError {
    #[error("api transport error: {0}")]
    Transport(String),
    #[error("api parse error: {0}")]
    Parse(String),
    #[error("api signing error: {0}")]
    Signing(String),
    #[error("api returned status {status}: {body}")]
    HttpStatus {
        status: reqwest::StatusCode,
        body: String,
    },
}

pub trait ValidatorSigner: Send + Sync {
    fn hotkey(&self) -> String;
    fn sign(&self, message: &[u8]) -> Result<String>;
}

fn classify_incentive_status(
    status: reqwest::StatusCode,
    body: Option<String>,
) -> std::result::Result<(), BasilicaApiError> {
    if status.is_success() {
        return Ok(());
    }

    Err(BasilicaApiError::HttpStatus {
        status,
        body: body.unwrap_or_default(),
    })
}

impl ValidatorSigner for bittensor::Service {
    fn hotkey(&self) -> String {
        self.get_account_id().to_string()
    }

    fn sign(&self, message: &[u8]) -> Result<String> {
        self.sign_data(message)
            .map_err(|e| anyhow::anyhow!("Failed to sign request: {e}"))
    }
}

#[derive(Debug, Default)]
struct PriceCache {
    prices: Option<HashMap<String, f64>>,
    fetched_at: Option<Instant>,
}

impl PriceCache {
    fn get_if_valid(&self, ttl: Duration) -> Option<HashMap<String, f64>> {
        match (self.prices.as_ref(), self.fetched_at) {
            (Some(prices), Some(fetched_at)) if fetched_at.elapsed() <= ttl => Some(prices.clone()),
            _ => None,
        }
    }

    fn get_any(&self) -> Option<HashMap<String, f64>> {
        self.prices.clone()
    }

    fn update(&mut self, prices: HashMap<String, f64>) {
        self.prices = Some(prices);
        self.fetched_at = Some(Instant::now());
    }

    #[cfg(test)]
    fn update_with_timestamp(&mut self, prices: HashMap<String, f64>, fetched_at: Instant) {
        self.prices = Some(prices);
        self.fetched_at = Some(fetched_at);
    }
}

#[derive(Debug)]
struct CircuitBreaker {
    failures: AtomicU32,
    threshold: u32,
    reset_timeout: Duration,
    last_failure: RwLock<Option<Instant>>,
}

impl CircuitBreaker {
    fn new(threshold: u32, reset_timeout: Duration) -> Self {
        Self {
            failures: AtomicU32::new(0),
            threshold,
            reset_timeout,
            last_failure: RwLock::new(None),
        }
    }

    async fn is_open(&self) -> bool {
        let failures = self.failures.load(Ordering::SeqCst);
        if failures < self.threshold {
            return false;
        }

        let mut last_failure = self.last_failure.write().await;
        if let Some(ts) = *last_failure {
            if ts.elapsed() > self.reset_timeout {
                self.failures.store(0, Ordering::SeqCst);
                *last_failure = None;
                return false;
            }
        }
        true
    }

    async fn record_success(&self) {
        self.failures.store(0, Ordering::SeqCst);
        *self.last_failure.write().await = None;
    }

    async fn record_failure(&self) {
        let failures = self.failures.fetch_add(1, Ordering::SeqCst) + 1;
        if failures >= self.threshold {
            *self.last_failure.write().await = Some(Instant::now());
        }
    }
}

#[async_trait]
pub trait BaselinePriceFetcher: Send + Sync {
    async fn fetch(&self, client: &BasilicaApiClient) -> Result<HashMap<String, f64>>;
}

#[async_trait]
pub trait TokenPriceFetcher: Send + Sync {
    async fn fetch(&self, client: &BasilicaApiClient, netuid: u16) -> Result<TokenPriceSnapshot>;
}

pub struct HttpBaselinePriceFetcher;

#[async_trait]
impl BaselinePriceFetcher for HttpBaselinePriceFetcher {
    async fn fetch(&self, client: &BasilicaApiClient) -> Result<HashMap<String, f64>> {
        let url = format!(
            "{}/v1/prices/baseline",
            client.api_endpoint.trim_end_matches('/')
        );
        let response = client.signed_get(&url, &()).await?;
        let body: BaselinePricesResponse = client.read_json_response(response).await?;
        let mut prices = HashMap::new();
        for price in body.prices {
            prices.insert(price.gpu_category, price.price_per_hour);
        }
        Ok(prices)
    }
}

pub struct HttpTokenPriceFetcher;

#[async_trait]
impl TokenPriceFetcher for HttpTokenPriceFetcher {
    async fn fetch(&self, client: &BasilicaApiClient, netuid: u16) -> Result<TokenPriceSnapshot> {
        let query = TokenPricesQuery {
            netuid: netuid as u32,
        };
        let url = format!(
            "{}/v1/prices/tokens",
            client.api_endpoint.trim_end_matches('/')
        );
        let response = client.signed_get(&url, &query).await?;
        let payload: TokenPricesResponse = client.read_json_response(response).await?;

        Ok(TokenPriceSnapshot {
            tao_price_usd: Decimal::from_str_exact(&payload.tao_price_usd)
                .context("Invalid tao_price_usd")?,
            alpha_price_usd: Decimal::from_str_exact(&payload.alpha_price_usd)
                .context("Invalid alpha_price_usd")?,
            alpha_price_tao: Decimal::from_str_exact(&payload.alpha_price_tao)
                .context("Invalid alpha_price_tao")?,
            tao_reserve: Decimal::from_str_exact(&payload.tao_reserve)
                .context("Invalid tao_reserve")?,
            alpha_reserve: Decimal::from_str_exact(&payload.alpha_reserve)
                .context("Invalid alpha_reserve")?,
            fetched_at: payload.fetched_at,
        })
    }
}

pub struct BasilicaApiClient {
    api_endpoint: String,
    signer: Arc<dyn ValidatorSigner>,
    http_client: Client,
    baseline_cache: Arc<RwLock<PriceCache>>,
    baseline_cache_ttl: Duration,
    baseline_fetcher: Arc<dyn BaselinePriceFetcher>,
    baseline_circuit_breaker: CircuitBreaker,
    token_cache: Arc<RwLock<HashMap<u16, CachedTokenPrices>>>,
    token_cache_ttl: Duration,
    token_fetcher: Arc<dyn TokenPriceFetcher>,
    token_fetch_lock: Arc<Mutex<()>>,
}

impl BasilicaApiClient {
    pub fn new(
        api_endpoint: String,
        signer: Arc<dyn ValidatorSigner>,
        timeout_secs: u64,
        baseline_cache_ttl: Duration,
        token_cache_ttl: Duration,
    ) -> Result<Self> {
        let http_client = Client::builder()
            .timeout(Duration::from_secs(timeout_secs))
            .build()
            .context("Failed to build HTTP client")?;
        Ok(Self::new_with_fetchers(
            api_endpoint,
            signer,
            http_client,
            baseline_cache_ttl,
            token_cache_ttl,
            Arc::new(HttpBaselinePriceFetcher),
            Arc::new(HttpTokenPriceFetcher),
        ))
    }

    pub fn new_with_fetchers(
        api_endpoint: String,
        signer: Arc<dyn ValidatorSigner>,
        http_client: Client,
        baseline_cache_ttl: Duration,
        token_cache_ttl: Duration,
        baseline_fetcher: Arc<dyn BaselinePriceFetcher>,
        token_fetcher: Arc<dyn TokenPriceFetcher>,
    ) -> Self {
        Self {
            api_endpoint,
            signer,
            http_client,
            baseline_cache: Arc::new(RwLock::new(PriceCache::default())),
            baseline_cache_ttl,
            baseline_fetcher,
            baseline_circuit_breaker: CircuitBreaker::new(3, Duration::from_secs(30)),
            token_cache: Arc::new(RwLock::new(HashMap::new())),
            token_cache_ttl,
            token_fetcher,
            token_fetch_lock: Arc::new(Mutex::new(())),
        }
    }

    pub async fn get_baseline_prices(&self) -> Result<HashMap<String, f64>> {
        if let Some(cached) = self
            .baseline_cache
            .read()
            .await
            .get_if_valid(self.baseline_cache_ttl)
        {
            debug!("Using cached baseline prices");
            return Ok(cached);
        }

        let stale = self.baseline_cache.read().await.get_any();
        if self.baseline_circuit_breaker.is_open().await {
            if let Some(stale_prices) = stale {
                warn!("Price circuit open; using stale cache");
                return Ok(stale_prices);
            }
            return Err(anyhow::anyhow!(
                "Price circuit open and no cached baseline prices available"
            ));
        }

        match self.baseline_fetcher.fetch(self).await {
            Ok(prices) => {
                self.baseline_cache.write().await.update(prices.clone());
                self.baseline_circuit_breaker.record_success().await;
                Ok(prices)
            }
            Err(err) => {
                self.baseline_circuit_breaker.record_failure().await;
                if let Some(stale_prices) = stale {
                    warn!("Price fetch failed; using stale cache: {}", err);
                    Ok(stale_prices)
                } else {
                    Err(err)
                }
            }
        }
    }

    pub async fn get_token_prices(&self, netuid: u16) -> Result<TokenPriceSnapshot> {
        if let Some(snapshot) = self.get_cached_token_prices_if_valid(netuid).await {
            debug!(netuid = netuid, "Using cached token prices");
            return Ok(snapshot);
        }

        let _guard = self.token_fetch_lock.lock().await;
        if let Some(snapshot) = self.get_cached_token_prices_if_valid(netuid).await {
            debug!(netuid = netuid, "Using cached token prices (post-lock)");
            return Ok(snapshot);
        }

        let cached = self.token_cache.read().await.get(&netuid).cloned();
        match self.token_fetcher.fetch(self, netuid).await {
            Ok(snapshot) => {
                self.token_cache
                    .write()
                    .await
                    .insert(netuid, CachedTokenPrices::new(snapshot.clone()));
                Ok(snapshot)
            }
            Err(err) => {
                if let Some(cached) = cached {
                    warn!(
                        netuid = netuid,
                        error = %err,
                        "Token price fetch failed; using cached value"
                    );
                    Ok(cached.snapshot)
                } else {
                    Err(err)
                }
            }
        }
    }

    pub async fn get_alpha_price_usd(&self, netuid: u16) -> Result<Decimal> {
        Ok(self.get_token_prices(netuid).await?.alpha_price_usd)
    }

    pub async fn get_miner_delivery(
        &self,
        since: DateTime<Utc>,
        until: DateTime<Utc>,
    ) -> Result<Vec<basilica_protocol::billing::MinerDelivery>> {
        let query = MinerDeliveryQuery {
            since_epoch_seconds: since.timestamp(),
            until_epoch_seconds: until.timestamp(),
        };

        let url = format!(
            "{}/v1/weights/miner-delivery",
            self.api_endpoint.trim_end_matches('/')
        );

        let response = self.signed_get(&url, &query).await?;
        let body: MinerDeliveryResponse = self.read_json_response(response).await?;

        Ok(body
            .deliveries
            .into_iter()
            .map(|delivery| basilica_protocol::billing::MinerDelivery {
                miner_hotkey: delivery.miner_hotkey,
                miner_uid: delivery.miner_uid,
                total_hours: delivery.total_hours,
                revenue_usd: delivery.revenue_usd,
                gpu_category: delivery.gpu_category,
                node_id: delivery.node_id,
            })
            .collect())
    }

    pub async fn get_incentive_config(
        &self,
    ) -> std::result::Result<IncentiveConfigResponse, BasilicaApiError> {
        let url = format!(
            "{}/v1/incentive/config",
            self.api_endpoint.trim_end_matches('/')
        );

        let response = self.signed_get_typed(&url, &()).await?;
        self.read_json_response_typed(response).await
    }

    pub async fn get_cus(
        &self,
        epoch_start: DateTime<Utc>,
        epoch_end: DateTime<Utc>,
    ) -> std::result::Result<Vec<CuLedgerRowResponse>, BasilicaApiError> {
        let query = EpochWindowQuery {
            epoch_start: epoch_start.to_rfc3339(),
            epoch_end: epoch_end.to_rfc3339(),
        };
        let url = format!(
            "{}/v1/incentive/cus",
            self.api_endpoint.trim_end_matches('/')
        );

        let response = self.signed_get_typed(&url, &query).await?;
        let body: GetCusResponse = self.read_json_response_typed(response).await?;
        Ok(body.rows)
    }

    pub async fn get_rus(
        &self,
        epoch_start: DateTime<Utc>,
        epoch_end: DateTime<Utc>,
    ) -> std::result::Result<Vec<RuLedgerRowResponse>, BasilicaApiError> {
        let query = EpochWindowQuery {
            epoch_start: epoch_start.to_rfc3339(),
            epoch_end: epoch_end.to_rfc3339(),
        };
        let url = format!(
            "{}/v1/incentive/rus",
            self.api_endpoint.trim_end_matches('/')
        );

        let response = self.signed_get_typed(&url, &query).await?;
        let body: GetRusResponse = self.read_json_response_typed(response).await?;
        Ok(body.rows)
    }

    pub async fn submit_cus(
        &self,
        rows: Vec<NewCuLedgerRowRequest>,
    ) -> std::result::Result<usize, BasilicaApiError> {
        let payload = PostCusRequest { cus: rows };
        let url = format!(
            "{}/v1/incentive/cus",
            self.api_endpoint.trim_end_matches('/')
        );

        let response = self.signed_post_typed(&url, &payload).await?;
        let body: PostCusResponse = self.read_json_response_typed(response).await?;
        Ok(body.inserted)
    }

    pub async fn slash_node(
        &self,
        node_id: &str,
        slash_pct: u32,
        idempotency_key: &str,
    ) -> std::result::Result<PostSlashResponse, BasilicaApiError> {
        let payload = PostSlashRequest {
            node_id: node_id.to_string(),
            slash_pct,
            idempotency_key: idempotency_key.to_string(),
        };
        let url = format!(
            "{}/v1/incentive/slash",
            self.api_endpoint.trim_end_matches('/')
        );

        let response = self.signed_post_typed(&url, &payload).await?;
        self.read_json_response_typed(response).await
    }

    async fn signed_get<Q: Serialize>(&self, url: &str, query: &Q) -> Result<reqwest::Response> {
        self.signed_get_typed(url, query)
            .await
            .map_err(|e| anyhow::anyhow!(e.to_string()))
    }

    #[allow(dead_code)]
    fn signed_headers<T: Serialize>(&self, payload: &T) -> Result<(String, String)> {
        self.signed_headers_typed(payload)
            .map_err(|e| anyhow::anyhow!(e.to_string()))
    }

    fn signed_headers_typed<T: Serialize>(
        &self,
        payload: &T,
    ) -> std::result::Result<(String, String), BasilicaApiError> {
        let timestamp = Utc::now().timestamp().to_string();
        let payload_json =
            serde_json::to_string(payload).map_err(|e| BasilicaApiError::Parse(e.to_string()))?;
        let message = format!("{timestamp}:{payload_json}");
        let signature = self
            .signer
            .sign(message.as_bytes())
            .map_err(|e| BasilicaApiError::Signing(e.to_string()))?;
        Ok((signature, timestamp))
    }

    async fn signed_get_typed<Q: Serialize>(
        &self,
        url: &str,
        query: &Q,
    ) -> std::result::Result<reqwest::Response, BasilicaApiError> {
        let (signature, timestamp) = self.signed_headers_typed(query)?;
        debug!(url = url, "Sending signed GET request");
        let response = self
            .http_client
            .get(url)
            .header("X-Validator-Hotkey", self.signer.hotkey())
            .header("X-Validator-Signature", signature)
            .header("X-Timestamp", timestamp)
            .query(query)
            .send()
            .await
            .map_err(|e| {
                BasilicaApiError::Transport(format!("API request to {url} failed: {e}"))
            })?;
        debug!(url = url, status = %response.status(), "Received API response");
        Ok(response)
    }

    async fn signed_post_typed<B: Serialize>(
        &self,
        url: &str,
        body: &B,
    ) -> std::result::Result<reqwest::Response, BasilicaApiError> {
        let (signature, timestamp) = self.signed_headers_typed(body)?;
        debug!(url = url, "Sending signed POST request");
        let response = self
            .http_client
            .post(url)
            .header("X-Validator-Hotkey", self.signer.hotkey())
            .header("X-Validator-Signature", signature)
            .header("X-Timestamp", timestamp)
            .json(body)
            .send()
            .await
            .map_err(|e| {
                BasilicaApiError::Transport(format!("API request to {url} failed: {e}"))
            })?;
        debug!(url = url, status = %response.status(), "Received API response");
        Ok(response)
    }

    async fn read_json_response<T: DeserializeOwned>(
        &self,
        response: reqwest::Response,
    ) -> Result<T> {
        self.read_json_response_typed(response)
            .await
            .map_err(|e| anyhow::anyhow!(e.to_string()))
    }

    async fn read_json_response_typed<T: DeserializeOwned>(
        &self,
        response: reqwest::Response,
    ) -> std::result::Result<T, BasilicaApiError> {
        if !response.status().is_success() {
            let status = response.status();
            let body: Option<Value> = response.json().await.ok();
            return match classify_incentive_status(status, body.map(|v| v.to_string())) {
                Ok(()) => unreachable!("non-success status should not classify as success"),
                Err(err) => Err(err),
            };
        }
        response
            .json::<T>()
            .await
            .map_err(|e| BasilicaApiError::Parse(e.to_string()))
    }

    async fn get_cached_token_prices_if_valid(&self, netuid: u16) -> Option<TokenPriceSnapshot> {
        let cache = self.token_cache.read().await;
        let cached = cache.get(&netuid)?;
        if cached.is_valid(self.token_cache_ttl) {
            return Some(cached.snapshot.clone());
        }
        None
    }
}

#[derive(Debug, Serialize)]
struct EpochWindowQuery {
    epoch_start: String,
    epoch_end: String,
}

#[derive(Debug, Clone, Serialize, Deserialize, PartialEq)]
pub struct IncentiveConfigResponse {
    pub gpu_categories: HashMap<String, IncentiveGpuCategoryConfig>,
    pub window_hours: u32,
    pub max_cu_value_usd: Decimal,
    pub revenue_share_pct: Option<u32>,
    pub slash_pct: u32,
}

#[derive(Debug, Clone, Serialize, Deserialize, PartialEq)]
pub struct IncentiveGpuCategoryConfig {
    /// Number of nodes to target. Each node is assumed to have 8 GPUs,
    /// so target_count=1 means 8 GPUs.
    pub target_count: u32,
    /// Price per individual GPU per hour in USD.
    pub price_per_gpu_usd: Decimal,
}

#[derive(Debug, Clone, Serialize, Deserialize, PartialEq)]
pub struct CuLedgerRowResponse {
    pub id: uuid::Uuid,
    pub hotkey: String,
    pub miner_uid: u32,
    pub node_id: String,
    pub cu_amount: Decimal,
    pub earned_at: DateTime<Utc>,
    pub is_rented: bool,
    pub gpu_category: String,
    pub window_hours: u32,
    pub price_usd: Decimal,
    pub idempotency_key: String,
    pub is_slashed: bool,
    pub slash_audit_id: Option<uuid::Uuid>,
    pub created_at: DateTime<Utc>,
}

#[derive(Debug, Clone, Serialize, Deserialize, PartialEq)]
pub struct RuLedgerRowResponse {
    pub id: uuid::Uuid,
    pub hotkey: String,
    pub miner_uid: u32,
    pub node_id: String,
    pub ru_amount: Decimal,
    pub earned_at: DateTime<Utc>,
    pub gpu_category: String,
    pub window_hours: u32,
    pub revenue_share_pct: u32,
    pub period_start: DateTime<Utc>,
    pub period_end: DateTime<Utc>,
    pub is_slashed: bool,
    pub slash_audit_id: Option<uuid::Uuid>,
    pub created_at: DateTime<Utc>,
}

#[derive(Debug, Clone, Serialize, Deserialize, PartialEq)]
pub struct GetCusResponse {
    pub rows: Vec<CuLedgerRowResponse>,
}

#[derive(Debug, Clone, Serialize, Deserialize, PartialEq)]
pub struct GetRusResponse {
    pub rows: Vec<RuLedgerRowResponse>,
}

#[derive(Debug, Clone, Serialize, Deserialize, PartialEq)]
pub struct NewCuLedgerRowRequest {
    pub hotkey: String,
    pub miner_uid: u32,
    pub node_id: String,
    pub cu_amount: Decimal,
    pub earned_at: DateTime<Utc>,
    pub is_rented: bool,
    pub gpu_category: String,
    pub window_hours: u32,
    pub price_usd: Decimal,
    pub idempotency_key: String,
}

#[derive(Debug, Clone, Serialize, Deserialize, PartialEq)]
pub struct PostCusRequest {
    pub cus: Vec<NewCuLedgerRowRequest>,
}

#[derive(Debug, Clone, Serialize, Deserialize, PartialEq)]
pub struct PostCusResponse {
    pub inserted: usize,
}

#[derive(Debug, Clone, Serialize, Deserialize, PartialEq)]
pub struct PostSlashRequest {
    pub node_id: String,
    pub slash_pct: u32,
    pub idempotency_key: String,
}

#[derive(Debug, Clone, Serialize, Deserialize, PartialEq)]
pub struct PostSlashResponse {
    pub slashed_cu_count: usize,
    pub slashed_ru_count: usize,
}

#[derive(Debug, Serialize)]
struct MinerDeliveryQuery {
    since_epoch_seconds: i64,
    until_epoch_seconds: i64,
}

#[derive(Debug, Deserialize)]
struct MinerDeliveryResponse {
    deliveries: Vec<MinerDeliveryItem>,
}

#[derive(Debug, Deserialize)]
struct MinerDeliveryItem {
    miner_hotkey: String,
    miner_uid: u32,
    total_hours: f64,
    revenue_usd: f64,
    gpu_category: String,
    #[serde(default)]
    node_id: String,
}

#[derive(Debug, Deserialize)]
struct BaselinePricesResponse {
    prices: Vec<BaselinePrice>,
}

#[derive(Debug, Deserialize)]
struct BaselinePrice {
    gpu_category: String,
    price_per_hour: f64,
}

#[derive(Debug, Serialize)]
struct TokenPricesQuery {
    netuid: u32,
}

#[derive(Debug, Deserialize)]
struct TokenPricesResponse {
    tao_price_usd: String,
    alpha_price_usd: String,
    alpha_price_tao: String,
    tao_reserve: String,
    alpha_reserve: String,
    fetched_at: String,
}

#[derive(Debug, Clone)]
pub struct TokenPriceSnapshot {
    pub tao_price_usd: Decimal,
    pub alpha_price_usd: Decimal,
    pub alpha_price_tao: Decimal,
    pub tao_reserve: Decimal,
    pub alpha_reserve: Decimal,
    pub fetched_at: String,
}

#[derive(Debug, Clone)]
struct CachedTokenPrices {
    snapshot: TokenPriceSnapshot,
    cached_at: Instant,
}

impl CachedTokenPrices {
    fn new(snapshot: TokenPriceSnapshot) -> Self {
        Self {
            snapshot,
            cached_at: Instant::now(),
        }
    }

    #[cfg(test)]
    fn with_timestamp(snapshot: TokenPriceSnapshot, cached_at: Instant) -> Self {
        Self {
            snapshot,
            cached_at,
        }
    }

    fn is_valid(&self, ttl: Duration) -> bool {
        self.cached_at.elapsed() <= ttl
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use serde_json::json;
    use std::sync::atomic::{AtomicUsize, Ordering};
    use wiremock::matchers::{body_json, header_exists, method, path, query_param};
    use wiremock::{Mock, MockServer, ResponseTemplate};

    struct TestBaselineFetcher {
        calls: Arc<AtomicUsize>,
        response: Arc<dyn Fn() -> Result<HashMap<String, f64>> + Send + Sync>,
    }

    #[async_trait]
    impl BaselinePriceFetcher for TestBaselineFetcher {
        async fn fetch(&self, _client: &BasilicaApiClient) -> Result<HashMap<String, f64>> {
            self.calls.fetch_add(1, Ordering::SeqCst);
            (self.response)()
        }
    }

    struct TestTokenFetcher {
        calls: Arc<AtomicUsize>,
        response: Arc<dyn Fn() -> Result<TokenPriceSnapshot> + Send + Sync>,
    }

    #[async_trait]
    impl TokenPriceFetcher for TestTokenFetcher {
        async fn fetch(
            &self,
            _client: &BasilicaApiClient,
            _netuid: u16,
        ) -> Result<TokenPriceSnapshot> {
            self.calls.fetch_add(1, Ordering::SeqCst);
            (self.response)()
        }
    }

    #[derive(Clone)]
    struct RecordingSigner {
        last_message: Arc<std::sync::Mutex<Vec<u8>>>,
    }

    impl RecordingSigner {
        fn new() -> Self {
            Self {
                last_message: Arc::new(std::sync::Mutex::new(Vec::new())),
            }
        }

        fn last_message(&self) -> Vec<u8> {
            self.last_message.lock().expect("lock").clone()
        }
    }

    impl ValidatorSigner for RecordingSigner {
        fn hotkey(&self) -> String {
            "test_hotkey".to_string()
        }

        fn sign(&self, message: &[u8]) -> Result<String> {
            let mut guard = self.last_message.lock().expect("lock");
            guard.clear();
            guard.extend_from_slice(message);
            Ok("deadbeef".to_string())
        }
    }

    #[derive(Clone)]
    struct FailingSigner;

    impl ValidatorSigner for FailingSigner {
        fn hotkey(&self) -> String {
            "test_hotkey".to_string()
        }

        fn sign(&self, _message: &[u8]) -> Result<String> {
            Err(anyhow::anyhow!("sign failed"))
        }
    }

    fn make_prices() -> HashMap<String, f64> {
        let mut prices = HashMap::new();
        prices.insert("H100".to_string(), 2.0);
        prices
    }

    fn snapshot(tao: &str, alpha: &str) -> TokenPriceSnapshot {
        TokenPriceSnapshot {
            tao_price_usd: Decimal::from_str_exact(tao).unwrap(),
            alpha_price_usd: Decimal::from_str_exact(alpha).unwrap(),
            alpha_price_tao: Decimal::from_str_exact("0.1").unwrap(),
            tao_reserve: Decimal::from_str_exact("10").unwrap(),
            alpha_reserve: Decimal::from_str_exact("20").unwrap(),
            fetched_at: "2024-01-01T00:00:00Z".to_string(),
        }
    }

    fn build_client(
        baseline_fetcher: Arc<dyn BaselinePriceFetcher>,
        token_fetcher: Arc<dyn TokenPriceFetcher>,
        signer: Arc<dyn ValidatorSigner>,
    ) -> BasilicaApiClient {
        BasilicaApiClient::new_with_fetchers(
            "http://localhost".to_string(),
            signer,
            Client::new(),
            Duration::from_secs(60),
            Duration::from_secs(60),
            baseline_fetcher,
            token_fetcher,
        )
    }

    fn build_http_client(
        api_endpoint: String,
        signer: Arc<dyn ValidatorSigner>,
    ) -> BasilicaApiClient {
        BasilicaApiClient::new_with_fetchers(
            api_endpoint,
            signer,
            Client::new(),
            Duration::from_secs(60),
            Duration::from_secs(60),
            Arc::new(HttpBaselinePriceFetcher),
            Arc::new(HttpTokenPriceFetcher),
        )
    }

    #[test]
    fn test_deserialize_incentive_config_response() {
        let json = r#"
        {
            "gpu_categories": {
                "H100": { "target_count": 2, "price_per_gpu_usd": "3.00" }
            },
            "window_hours": 72,
            "max_cu_value_usd": "0.05",
            "revenue_share_pct": 30,
            "slash_pct": 100
        }
        "#;

        let parsed: IncentiveConfigResponse = serde_json::from_str(json).unwrap();
        assert_eq!(parsed.gpu_categories["H100"].target_count, 2);
        assert_eq!(
            parsed.gpu_categories["H100"].price_per_gpu_usd,
            Decimal::from_str_exact("3.00").unwrap()
        );
        assert_eq!(parsed.window_hours, 72);
        assert_eq!(
            parsed.max_cu_value_usd,
            Decimal::from_str_exact("0.05").unwrap()
        );
    }

    #[test]
    fn test_deserialize_cu_row_response() {
        let json = r#"
        {
            "id": "11111111-1111-1111-1111-111111111111",
            "hotkey": "5miner",
            "miner_uid": 42,
            "node_id": "node-abc",
            "cu_amount": "8.0",
            "earned_at": "2025-03-15T10:00:00Z",
            "is_rented": false,
            "gpu_category": "H100",
            "window_hours": 72,
            "price_usd": "3.00",
            "idempotency_key": "node-abc:1710496800",
            "is_slashed": false,
            "slash_audit_id": "33333333-3333-3333-3333-333333333333",
            "created_at": "2025-03-15T10:05:00Z"
        }
        "#;

        let parsed: CuLedgerRowResponse = serde_json::from_str(json).unwrap();
        assert_eq!(parsed.hotkey, "5miner");
        assert_eq!(parsed.miner_uid, 42);
        assert_eq!(parsed.cu_amount, Decimal::from_str_exact("8.0").unwrap());
        assert_eq!(parsed.idempotency_key, "node-abc:1710496800");
        assert!(!parsed.is_slashed);
        assert_eq!(
            parsed.slash_audit_id,
            Some(uuid::Uuid::parse_str("33333333-3333-3333-3333-333333333333").unwrap())
        );
        assert_eq!(
            parsed.created_at,
            DateTime::parse_from_rfc3339("2025-03-15T10:05:00Z")
                .unwrap()
                .with_timezone(&Utc)
        );
    }

    #[test]
    fn test_deserialize_ru_row_response() {
        let json = r#"
        {
            "id": "22222222-2222-2222-2222-222222222222",
            "hotkey": "5miner",
            "miner_uid": 42,
            "node_id": "node-abc",
            "ru_amount": "9.5",
            "earned_at": "2025-03-15T10:00:00Z",
            "gpu_category": "H100",
            "window_hours": 72,
            "revenue_share_pct": 30,
            "period_start": "2025-03-15T09:00:00Z",
            "period_end": "2025-03-15T10:00:00Z",
            "is_slashed": true,
            "slash_audit_id": null,
            "created_at": "2025-03-15T10:05:00Z"
        }
        "#;

        let parsed: RuLedgerRowResponse = serde_json::from_str(json).unwrap();
        assert_eq!(parsed.node_id, "node-abc");
        assert_eq!(parsed.ru_amount, Decimal::from_str_exact("9.5").unwrap());
        assert!(parsed.is_slashed);
        assert_eq!(parsed.slash_audit_id, None);
        assert_eq!(
            parsed.created_at,
            DateTime::parse_from_rfc3339("2025-03-15T10:05:00Z")
                .unwrap()
                .with_timezone(&Utc)
        );
    }

    #[test]
    fn test_deserialize_get_cus_response() {
        let json = r#"
        {
            "rows": [
                {
                    "id": "11111111-1111-1111-1111-111111111111",
                    "hotkey": "5miner",
                    "miner_uid": 42,
                    "node_id": "node-abc",
                    "cu_amount": "8.0",
                    "earned_at": "2025-03-15T10:00:00Z",
                    "is_rented": false,
                    "gpu_category": "H100",
                    "window_hours": 72,
                    "price_usd": "3.00",
                    "idempotency_key": "node-abc:1710496800",
                    "is_slashed": false,
                    "slash_audit_id": "33333333-3333-3333-3333-333333333333",
                    "created_at": "2025-03-15T10:05:00Z"
                }
            ]
        }
        "#;

        let parsed: GetCusResponse = serde_json::from_str(json).unwrap();
        assert_eq!(parsed.rows.len(), 1);
        assert_eq!(parsed.rows[0].node_id, "node-abc");
    }

    #[test]
    fn test_deserialize_get_rus_response() {
        let json = r#"
        {
            "rows": [
                {
                    "id": "22222222-2222-2222-2222-222222222222",
                    "hotkey": "5miner",
                    "miner_uid": 42,
                    "node_id": "node-abc",
                    "ru_amount": "9.5",
                    "earned_at": "2025-03-15T10:00:00Z",
                    "gpu_category": "H100",
                    "window_hours": 72,
                    "revenue_share_pct": 30,
                    "period_start": "2025-03-15T09:00:00Z",
                    "period_end": "2025-03-15T10:00:00Z",
                    "is_slashed": true,
                    "slash_audit_id": null,
                    "created_at": "2025-03-15T10:05:00Z"
                }
            ]
        }
        "#;

        let parsed: GetRusResponse = serde_json::from_str(json).unwrap();
        assert_eq!(parsed.rows.len(), 1);
        assert_eq!(parsed.rows[0].node_id, "node-abc");
    }

    #[test]
    fn test_deserialize_post_cus_request() {
        let json = r#"
        {
            "cus": [
                {
                    "hotkey": "5miner",
                    "miner_uid": 42,
                    "node_id": "node-abc",
                    "cu_amount": "8.0",
                    "earned_at": "2025-03-15T10:00:00Z",
                    "is_rented": false,
                    "gpu_category": "H100",
                    "window_hours": 72,
                    "price_usd": "3.00",
                    "idempotency_key": "node-abc:1710496800"
                }
            ]
        }
        "#;

        let parsed: PostCusRequest = serde_json::from_str(json).unwrap();
        assert_eq!(parsed.cus.len(), 1);
        assert_eq!(parsed.cus[0].node_id, "node-abc");
    }

    #[test]
    fn test_deserialize_post_slash_request() {
        let json = r#"
        {
            "node_id": "node-abc",
            "slash_pct": 100,
            "idempotency_key": "rental:rental-1"
        }
        "#;

        let parsed: PostSlashRequest = serde_json::from_str(json).unwrap();
        assert_eq!(parsed.node_id, "node-abc");
        assert_eq!(parsed.slash_pct, 100);
        assert_eq!(parsed.idempotency_key, "rental:rental-1");
    }

    #[test]
    fn test_incentive_status_500_remains_http_status() {
        let err = classify_incentive_status(
            reqwest::StatusCode::INTERNAL_SERVER_ERROR,
            Some("{\"error\":\"boom\"}".to_string()),
        )
        .unwrap_err();

        assert!(matches!(
            err,
            BasilicaApiError::HttpStatus {
                status: reqwest::StatusCode::INTERNAL_SERVER_ERROR,
                ..
            }
        ));
    }

    #[test]
    fn test_signed_headers_typed_maps_signing_errors() {
        #[derive(Serialize)]
        struct TestPayload {
            foo: String,
        }

        let client = BasilicaApiClient::new_with_fetchers(
            "http://localhost".to_string(),
            Arc::new(FailingSigner),
            Client::new(),
            Duration::from_secs(60),
            Duration::from_secs(60),
            Arc::new(HttpBaselinePriceFetcher),
            Arc::new(HttpTokenPriceFetcher),
        );

        let err = client
            .signed_headers_typed(&TestPayload {
                foo: "bar".to_string(),
            })
            .unwrap_err();

        assert!(matches!(err, BasilicaApiError::Signing(_)));
    }

    #[tokio::test]
    async fn test_signed_get_typed_maps_transport_errors() {
        #[derive(Serialize)]
        struct TestQuery {
            foo: String,
        }

        let signer: Arc<dyn ValidatorSigner> = Arc::new(RecordingSigner::new());
        let client = build_client(
            Arc::new(HttpBaselinePriceFetcher),
            Arc::new(HttpTokenPriceFetcher),
            signer,
        );

        let err = client
            .signed_get_typed(
                "http://127.0.0.1:9/unreachable",
                &TestQuery {
                    foo: "bar".to_string(),
                },
            )
            .await
            .unwrap_err();

        assert!(matches!(err, BasilicaApiError::Transport(_)));
    }

    #[tokio::test]
    async fn test_read_json_response_typed_invalid_json_maps_parse() {
        let server = MockServer::start().await;
        Mock::given(method("GET"))
            .and(path("/typed-invalid-json"))
            .respond_with(ResponseTemplate::new(200).set_body_raw("not-json", "text/plain"))
            .mount(&server)
            .await;

        let signer: Arc<dyn ValidatorSigner> = Arc::new(RecordingSigner::new());
        let client = build_client(
            Arc::new(HttpBaselinePriceFetcher),
            Arc::new(HttpTokenPriceFetcher),
            signer,
        );

        let response = client
            .signed_get_typed(&format!("{}/typed-invalid-json", server.uri()), &())
            .await
            .unwrap();
        let err = client
            .read_json_response_typed::<IncentiveConfigResponse>(response)
            .await
            .unwrap_err();

        assert!(matches!(err, BasilicaApiError::Parse(_)));
    }

    #[tokio::test]
    async fn test_get_incentive_config_parses_success() {
        let server = MockServer::start().await;
        Mock::given(method("GET"))
            .and(path("/v1/incentive/config"))
            .respond_with(ResponseTemplate::new(200).set_body_json(json!({
                "gpu_categories": {
                    "H100": { "target_count": 2, "price_per_gpu_usd": "3.00" }
                },
                "window_hours": 72,
                "max_cu_value_usd": "0.05",
                "revenue_share_pct": 30,
                "slash_pct": 100
            })))
            .mount(&server)
            .await;

        let signer: Arc<dyn ValidatorSigner> = Arc::new(RecordingSigner::new());
        let client = build_http_client(server.uri(), signer);

        let config = client.get_incentive_config().await.unwrap();
        assert_eq!(config.gpu_categories["H100"].target_count, 2);
    }

    #[tokio::test]
    async fn test_get_incentive_config_404_returns_http_status() {
        let server = MockServer::start().await;
        Mock::given(method("GET"))
            .and(path("/v1/incentive/config"))
            .respond_with(
                ResponseTemplate::new(404).set_body_json(json!({ "error": "not configured" })),
            )
            .mount(&server)
            .await;

        let signer: Arc<dyn ValidatorSigner> = Arc::new(RecordingSigner::new());
        let client = build_http_client(server.uri(), signer);

        let err = client.get_incentive_config().await.unwrap_err();
        assert!(matches!(
            err,
            BasilicaApiError::HttpStatus {
                status: reqwest::StatusCode::NOT_FOUND,
                ..
            }
        ));
    }

    #[tokio::test]
    async fn test_get_incentive_config_500_returns_http_status() {
        let server = MockServer::start().await;
        Mock::given(method("GET"))
            .and(path("/v1/incentive/config"))
            .respond_with(ResponseTemplate::new(500).set_body_string("boom"))
            .mount(&server)
            .await;

        let signer: Arc<dyn ValidatorSigner> = Arc::new(RecordingSigner::new());
        let client = build_http_client(server.uri(), signer);

        let err = client.get_incentive_config().await.unwrap_err();
        assert!(matches!(
            err,
            BasilicaApiError::HttpStatus {
                status: reqwest::StatusCode::INTERNAL_SERVER_ERROR,
                ..
            }
        ));
    }

    #[tokio::test]
    async fn test_get_incentive_config_invalid_json_returns_parse_error() {
        let server = MockServer::start().await;
        Mock::given(method("GET"))
            .and(path("/v1/incentive/config"))
            .respond_with(ResponseTemplate::new(200).set_body_raw("not-json", "text/plain"))
            .mount(&server)
            .await;

        let signer: Arc<dyn ValidatorSigner> = Arc::new(RecordingSigner::new());
        let client = build_http_client(server.uri(), signer);

        let err = client.get_incentive_config().await.unwrap_err();
        assert!(matches!(err, BasilicaApiError::Parse(_)));
    }

    #[tokio::test]
    async fn test_get_cus_parses_rows() {
        let epoch_start = DateTime::parse_from_rfc3339("2025-03-15T09:00:00Z")
            .unwrap()
            .with_timezone(&Utc);
        let epoch_end = DateTime::parse_from_rfc3339("2025-03-15T11:00:00Z")
            .unwrap()
            .with_timezone(&Utc);

        let server = MockServer::start().await;
        Mock::given(method("GET"))
            .and(path("/v1/incentive/cus"))
            .and(query_param("epoch_start", epoch_start.to_rfc3339()))
            .and(query_param("epoch_end", epoch_end.to_rfc3339()))
            .respond_with(ResponseTemplate::new(200).set_body_json(json!({
                "rows": [{
                    "id": "11111111-1111-1111-1111-111111111111",
                    "hotkey": "5miner",
                    "miner_uid": 42,
                    "node_id": "node-abc",
                    "cu_amount": "8.0",
                    "earned_at": "2025-03-15T10:00:00Z",
                    "is_rented": false,
                    "gpu_category": "H100",
                    "window_hours": 72,
                    "price_usd": "3.00",
                    "idempotency_key": "node-abc:1710496800",
                    "is_slashed": false,
                    "slash_audit_id": null,
                    "created_at": "2025-03-15T10:05:00Z"
                }]
            })))
            .mount(&server)
            .await;

        let signer: Arc<dyn ValidatorSigner> = Arc::new(RecordingSigner::new());
        let client = build_http_client(server.uri(), signer);

        let rows = client.get_cus(epoch_start, epoch_end).await.unwrap();
        assert_eq!(rows.len(), 1);
        assert_eq!(rows[0].node_id, "node-abc");
    }

    #[tokio::test]
    async fn test_get_rus_parses_rows() {
        let epoch_start = DateTime::parse_from_rfc3339("2025-03-15T09:00:00Z")
            .unwrap()
            .with_timezone(&Utc);
        let epoch_end = DateTime::parse_from_rfc3339("2025-03-15T11:00:00Z")
            .unwrap()
            .with_timezone(&Utc);

        let server = MockServer::start().await;
        Mock::given(method("GET"))
            .and(path("/v1/incentive/rus"))
            .and(query_param("epoch_start", epoch_start.to_rfc3339()))
            .and(query_param("epoch_end", epoch_end.to_rfc3339()))
            .respond_with(ResponseTemplate::new(200).set_body_json(json!({
                "rows": [{
                    "id": "22222222-2222-2222-2222-222222222222",
                    "hotkey": "5miner",
                    "miner_uid": 42,
                    "node_id": "node-abc",
                    "ru_amount": "9.5",
                    "earned_at": "2025-03-15T10:00:00Z",
                    "gpu_category": "H100",
                    "window_hours": 72,
                    "revenue_share_pct": 30,
                    "period_start": "2025-03-15T09:00:00Z",
                    "period_end": "2025-03-15T10:00:00Z",
                    "is_slashed": false,
                    "slash_audit_id": null,
                    "created_at": "2025-03-15T10:05:00Z"
                }]
            })))
            .mount(&server)
            .await;

        let signer: Arc<dyn ValidatorSigner> = Arc::new(RecordingSigner::new());
        let client = build_http_client(server.uri(), signer);

        let rows = client.get_rus(epoch_start, epoch_end).await.unwrap();
        assert_eq!(rows.len(), 1);
        assert_eq!(rows[0].node_id, "node-abc");
    }

    #[tokio::test]
    async fn test_submit_cus_posts_signed_json_body() {
        let server = MockServer::start().await;
        let payload = PostCusRequest {
            cus: vec![NewCuLedgerRowRequest {
                hotkey: "5miner".to_string(),
                miner_uid: 42,
                node_id: "node-abc".to_string(),
                cu_amount: Decimal::from_str_exact("8.0").unwrap(),
                earned_at: DateTime::parse_from_rfc3339("2025-03-15T10:00:00Z")
                    .unwrap()
                    .with_timezone(&Utc),
                is_rented: false,
                gpu_category: "H100".to_string(),
                window_hours: 72,
                price_usd: Decimal::from_str_exact("3.00").unwrap(),
                idempotency_key: "node-abc:1710496800".to_string(),
            }],
        };

        Mock::given(method("POST"))
            .and(path("/v1/incentive/cus"))
            .and(header_exists("X-Validator-Hotkey"))
            .and(header_exists("X-Validator-Signature"))
            .and(header_exists("X-Timestamp"))
            .and(body_json(json!({
                "cus": [{
                    "hotkey": "5miner",
                    "miner_uid": 42,
                    "node_id": "node-abc",
                    "cu_amount": "8.0",
                    "earned_at": "2025-03-15T10:00:00Z",
                    "is_rented": false,
                    "gpu_category": "H100",
                    "window_hours": 72,
                    "price_usd": "3.00",
                    "idempotency_key": "node-abc:1710496800"
                }]
            })))
            .respond_with(ResponseTemplate::new(200).set_body_json(json!({ "inserted": 1 })))
            .mount(&server)
            .await;

        let signer: Arc<dyn ValidatorSigner> = Arc::new(RecordingSigner::new());
        let client = build_http_client(server.uri(), signer);

        let inserted = client.submit_cus(payload.cus).await.unwrap();
        assert_eq!(inserted, 1);
    }

    #[tokio::test]
    async fn test_slash_node_posts_signed_json_body() {
        let server = MockServer::start().await;
        Mock::given(method("POST"))
            .and(path("/v1/incentive/slash"))
            .and(header_exists("X-Validator-Hotkey"))
            .and(header_exists("X-Validator-Signature"))
            .and(header_exists("X-Timestamp"))
            .and(body_json(json!({
                "node_id": "node-abc",
                "slash_pct": 100,
                "idempotency_key": "test-key-1"
            })))
            .respond_with(ResponseTemplate::new(200).set_body_json(json!({
                "slashed_cu_count": 5,
                "slashed_ru_count": 2
            })))
            .mount(&server)
            .await;

        let signer: Arc<dyn ValidatorSigner> = Arc::new(RecordingSigner::new());
        let client = build_http_client(server.uri(), signer);

        let response = client
            .slash_node("node-abc", 100, "test-key-1")
            .await
            .unwrap();
        assert_eq!(response.slashed_cu_count, 5);
        assert_eq!(response.slashed_ru_count, 2);
    }

    #[tokio::test]
    async fn test_slash_node_403_returns_http_status() {
        let server = MockServer::start().await;
        Mock::given(method("POST"))
            .and(path("/v1/incentive/slash"))
            .respond_with(ResponseTemplate::new(403).set_body_string("forbidden"))
            .mount(&server)
            .await;

        let signer: Arc<dyn ValidatorSigner> = Arc::new(RecordingSigner::new());
        let client = build_http_client(server.uri(), signer);

        let err = client
            .slash_node("node-abc", 100, "test-key-2")
            .await
            .unwrap_err();
        assert!(matches!(
            err,
            BasilicaApiError::HttpStatus {
                status: reqwest::StatusCode::FORBIDDEN,
                ..
            }
        ));
    }

    #[tokio::test]
    async fn test_baseline_cache_hit_returns_cached() {
        let calls = Arc::new(AtomicUsize::new(0));
        let fetcher = Arc::new(TestBaselineFetcher {
            calls: calls.clone(),
            response: Arc::new(|| Ok(make_prices())),
        });
        let token_fetcher = Arc::new(TestTokenFetcher {
            calls: Arc::new(AtomicUsize::new(0)),
            response: Arc::new(|| Ok(snapshot("1.0", "2.0"))),
        });
        let signer: Arc<dyn ValidatorSigner> = Arc::new(RecordingSigner::new());
        let client = build_client(fetcher, token_fetcher, signer);

        client
            .baseline_cache
            .write()
            .await
            .update_with_timestamp(make_prices(), Instant::now());

        let prices = client.get_baseline_prices().await.unwrap();
        assert_eq!(prices.get("H100"), Some(&2.0));
        assert_eq!(calls.load(Ordering::SeqCst), 0);
    }

    #[tokio::test]
    async fn test_baseline_cache_miss_fetch_success() {
        let calls = Arc::new(AtomicUsize::new(0));
        let fetcher = Arc::new(TestBaselineFetcher {
            calls: calls.clone(),
            response: Arc::new(|| Ok(make_prices())),
        });
        let token_fetcher = Arc::new(TestTokenFetcher {
            calls: Arc::new(AtomicUsize::new(0)),
            response: Arc::new(|| Ok(snapshot("1.0", "2.0"))),
        });
        let signer: Arc<dyn ValidatorSigner> = Arc::new(RecordingSigner::new());
        let client = build_client(fetcher, token_fetcher, signer);

        let prices = client.get_baseline_prices().await.unwrap();
        assert_eq!(prices.get("H100"), Some(&2.0));
        assert_eq!(calls.load(Ordering::SeqCst), 1);
    }

    #[tokio::test]
    async fn test_baseline_expired_cache_fetch_failure_returns_stale() {
        let calls = Arc::new(AtomicUsize::new(0));
        let fetcher = Arc::new(TestBaselineFetcher {
            calls: calls.clone(),
            response: Arc::new(|| Err(anyhow::anyhow!("fetch failed"))),
        });
        let token_fetcher = Arc::new(TestTokenFetcher {
            calls: Arc::new(AtomicUsize::new(0)),
            response: Arc::new(|| Ok(snapshot("1.0", "2.0"))),
        });
        let signer: Arc<dyn ValidatorSigner> = Arc::new(RecordingSigner::new());
        let client = build_client(fetcher, token_fetcher, signer);

        client
            .baseline_cache
            .write()
            .await
            .update_with_timestamp(make_prices(), Instant::now() - Duration::from_secs(61));

        let prices = client.get_baseline_prices().await.unwrap();
        assert_eq!(prices.get("H100"), Some(&2.0));
        assert_eq!(calls.load(Ordering::SeqCst), 1);
    }

    #[tokio::test]
    async fn test_token_cache_hit_within_ttl() {
        let baseline_fetcher = Arc::new(TestBaselineFetcher {
            calls: Arc::new(AtomicUsize::new(0)),
            response: Arc::new(|| Ok(make_prices())),
        });
        let calls = Arc::new(AtomicUsize::new(0));
        let token_fetcher = Arc::new(TestTokenFetcher {
            calls: calls.clone(),
            response: Arc::new(|| Err(anyhow::anyhow!("fetch failed"))),
        });
        let signer: Arc<dyn ValidatorSigner> = Arc::new(RecordingSigner::new());
        let client = build_client(baseline_fetcher, token_fetcher, signer);

        let snap = snapshot("1.0", "2.0");
        client.token_cache.write().await.insert(
            1,
            CachedTokenPrices::with_timestamp(snap.clone(), Instant::now()),
        );

        let result = client.get_token_prices(1).await.unwrap();
        assert_eq!(result.tao_price_usd, snap.tao_price_usd);
        assert_eq!(calls.load(Ordering::SeqCst), 0);
    }

    #[tokio::test]
    async fn test_token_ttl_expired_refresh_success() {
        let baseline_fetcher = Arc::new(TestBaselineFetcher {
            calls: Arc::new(AtomicUsize::new(0)),
            response: Arc::new(|| Ok(make_prices())),
        });
        let calls = Arc::new(AtomicUsize::new(0));
        let token_fetcher = Arc::new(TestTokenFetcher {
            calls: calls.clone(),
            response: Arc::new(|| Ok(snapshot("3.0", "4.0"))),
        });
        let signer: Arc<dyn ValidatorSigner> = Arc::new(RecordingSigner::new());
        let client = build_client(baseline_fetcher, token_fetcher, signer);

        let stale = snapshot("1.0", "2.0");
        client.token_cache.write().await.insert(
            1,
            CachedTokenPrices::with_timestamp(stale, Instant::now() - Duration::from_secs(61)),
        );

        let result = client.get_token_prices(1).await.unwrap();
        assert_eq!(
            result.tao_price_usd,
            Decimal::from_str_exact("3.0").unwrap()
        );
        assert_eq!(calls.load(Ordering::SeqCst), 1);
    }

    #[tokio::test]
    async fn test_token_ttl_expired_refresh_failure_uses_cache() {
        let baseline_fetcher = Arc::new(TestBaselineFetcher {
            calls: Arc::new(AtomicUsize::new(0)),
            response: Arc::new(|| Ok(make_prices())),
        });
        let calls = Arc::new(AtomicUsize::new(0));
        let token_fetcher = Arc::new(TestTokenFetcher {
            calls: calls.clone(),
            response: Arc::new(|| Err(anyhow::anyhow!("fetch failed"))),
        });
        let signer: Arc<dyn ValidatorSigner> = Arc::new(RecordingSigner::new());
        let client = build_client(baseline_fetcher, token_fetcher, signer);

        let stale = snapshot("1.0", "2.0");
        client.token_cache.write().await.insert(
            1,
            CachedTokenPrices::with_timestamp(
                stale.clone(),
                Instant::now() - Duration::from_secs(61),
            ),
        );

        let result = client.get_token_prices(1).await.unwrap();
        assert_eq!(result.tao_price_usd, stale.tao_price_usd);
        assert_eq!(calls.load(Ordering::SeqCst), 1);
    }

    #[test]
    fn test_signed_headers_message_format() {
        #[derive(Serialize)]
        struct TestPayload {
            foo: String,
            bar: u32,
        }

        let signer = RecordingSigner::new();
        let client = BasilicaApiClient::new_with_fetchers(
            "http://localhost".to_string(),
            Arc::new(signer.clone()),
            Client::new(),
            Duration::from_secs(60),
            Duration::from_secs(60),
            Arc::new(HttpBaselinePriceFetcher),
            Arc::new(HttpTokenPriceFetcher),
        );

        let payload = TestPayload {
            foo: "hello".to_string(),
            bar: 42,
        };

        let (_sig, timestamp) = client.signed_headers(&payload).unwrap();
        let payload_json = serde_json::to_string(&payload).unwrap();
        let expected = format!("{timestamp}:{payload_json}");
        assert_eq!(String::from_utf8(signer.last_message()).unwrap(), expected);
    }
}
