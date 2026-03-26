use crate::basilica_api::ValidatorSigner;
use crate::collateral::evidence::{EvidenceStore, SlashEvidence};
use crate::collateral::grace_tracker::GracePeriodTracker;
use crate::collateral::manager::{hotkey_ss58_to_hex, node_id_to_hex};
use crate::config::collateral::{CollateralConfig, TrusteeKeySource};
use crate::metrics::ValidatorPrometheusMetrics;
use alloy_primitives::U256;
use anyhow::{Context, Result};
use async_trait::async_trait;
use aws_config::meta::region::RegionProviderChain;
use aws_config::Region;
use aws_sdk_secretsmanager::Client as SecretsClient;
use chrono::Utc;
use collateral_contract::config::{CollateralNetworkConfig, Network};
use collateral_contract::{alpha_collaterals, slash_collateral};
use rust_decimal::prelude::ToPrimitive;
use rust_decimal::Decimal;
use sha2::{Digest, Sha256};
use std::collections::{HashMap, VecDeque};
use std::env;
use std::path::PathBuf;
use std::sync::Arc;
use std::time::{Duration as StdDuration, Instant};
use tokio::fs;
use tokio::sync::Mutex;
use tracing::{info, warn};

#[async_trait]
pub trait CollateralChainClient: Send + Sync {
    async fn alpha_collaterals(
        &self,
        hotkey_bytes: [u8; 32],
        node_bytes: [u8; 16],
        network_config: &CollateralNetworkConfig,
    ) -> Result<U256>;

    #[allow(clippy::too_many_arguments)]
    async fn submit_slash(
        &self,
        private_key: &str,
        hotkey_bytes: [u8; 32],
        node_bytes: [u8; 16],
        alpha_amount: U256,
        url: &str,
        checksum: u128,
        network_config: &CollateralNetworkConfig,
    ) -> Result<()>;
}

struct OnchainCollateralClient;

#[async_trait]
impl CollateralChainClient for OnchainCollateralClient {
    async fn alpha_collaterals(
        &self,
        hotkey_bytes: [u8; 32],
        node_bytes: [u8; 16],
        network_config: &CollateralNetworkConfig,
    ) -> Result<U256> {
        Ok(alpha_collaterals(hotkey_bytes, node_bytes, network_config).await?)
    }

    async fn submit_slash(
        &self,
        private_key: &str,
        hotkey_bytes: [u8; 32],
        node_bytes: [u8; 16],
        alpha_amount: U256,
        url: &str,
        checksum: u128,
        network_config: &CollateralNetworkConfig,
    ) -> Result<()> {
        slash_collateral(
            private_key,
            hotkey_bytes,
            node_bytes,
            alpha_amount,
            url,
            checksum,
            network_config,
        )
        .await?;
        Ok(())
    }
}

#[derive(Clone)]
pub struct SlashExecutor {
    config: CollateralConfig,
    evidence_store: EvidenceStore,
    grace_tracker: Arc<GracePeriodTracker>,
    metrics: Option<Arc<ValidatorPrometheusMetrics>>,
    rate_limiter: Arc<SlashRateLimiter>,
    signer: Option<Arc<dyn ValidatorSigner>>,
    chain_client: Arc<dyn CollateralChainClient>,
}

#[async_trait]
trait KeyProvider: Send + Sync {
    async fn load_private_key(&self) -> Result<String>;
}

struct FileKeyProvider {
    path: PathBuf,
}

struct EnvKeyProvider {
    env_var: String,
}

struct AwsSecretsKeyProvider {
    secret_name: String,
    region: Option<String>,
}

#[async_trait]
impl KeyProvider for FileKeyProvider {
    async fn load_private_key(&self) -> Result<String> {
        let contents = fs::read_to_string(&self.path).await?;
        normalize_private_key(&contents, "trustee_private_key_file")
    }
}

#[async_trait]
impl KeyProvider for EnvKeyProvider {
    async fn load_private_key(&self) -> Result<String> {
        let value = env::var(&self.env_var).with_context(|| {
            format!(
                "{} is required when trustee_key_source=env_var",
                self.env_var
            )
        })?;
        normalize_private_key(&value, "trustee_key_env_var")
    }
}

#[async_trait]
impl KeyProvider for AwsSecretsKeyProvider {
    async fn load_private_key(&self) -> Result<String> {
        let region_provider = match &self.region {
            Some(region) => {
                RegionProviderChain::first_try(Region::new(region.clone())).or_default_provider()
            }
            None => RegionProviderChain::default_provider(),
        };
        let config = aws_config::from_env().region(region_provider).load().await;
        let client = SecretsClient::new(&config);
        let response = client
            .get_secret_value()
            .secret_id(&self.secret_name)
            .send()
            .await?;
        // TODO: Support structured secret payloads (e.g., JSON) and rotation metadata.
        let secret = if let Some(value) = response.secret_string() {
            value.to_string()
        } else if let Some(binary) = response.secret_binary() {
            String::from_utf8(binary.as_ref().to_vec())
                .context("aws_secret_name contains non-utf8 secret_binary")?
        } else {
            anyhow::bail!("aws_secret_name returned empty secret");
        };
        normalize_private_key(&secret, "aws_secret_name")
    }
}

fn normalize_private_key(raw: &str, source: &str) -> Result<String> {
    let trimmed = raw.trim();
    if trimmed.is_empty() {
        anyhow::bail!("{} resolved to empty private key", source);
    }
    Ok(trimmed.to_string())
}

impl SlashExecutor {
    pub fn new(
        config: CollateralConfig,
        evidence_store: EvidenceStore,
        grace_tracker: Arc<GracePeriodTracker>,
        metrics: Option<Arc<ValidatorPrometheusMetrics>>,
        signer: Option<Arc<dyn ValidatorSigner>>,
    ) -> Self {
        Self::new_with_chain_client(
            config,
            evidence_store,
            grace_tracker,
            metrics,
            signer,
            Arc::new(OnchainCollateralClient),
        )
    }

    pub fn is_shadow_mode(&self) -> bool {
        self.config.shadow_mode
    }

    pub fn new_with_chain_client(
        config: CollateralConfig,
        evidence_store: EvidenceStore,
        grace_tracker: Arc<GracePeriodTracker>,
        metrics: Option<Arc<ValidatorPrometheusMetrics>>,
        signer: Option<Arc<dyn ValidatorSigner>>,
        chain_client: Arc<dyn CollateralChainClient>,
    ) -> Self {
        let rate_limiter = Arc::new(SlashRateLimiter::new(&config));
        Self {
            config,
            evidence_store,
            grace_tracker,
            metrics,
            rate_limiter,
            signer,
            chain_client,
        }
    }

    pub async fn execute_slash(
        &self,
        miner_hotkey: &str,
        node_id: &str,
        misbehaviour_type: &str,
        details: &str,
        validator_hotkey: &str,
        rental_id: &str,
    ) -> Result<()> {
        let (url, checksum) = self
            .prepare_evidence(
                misbehaviour_type,
                details,
                validator_hotkey,
                rental_id,
                miner_hotkey,
                node_id,
            )
            .await?;
        self.record_slash_triggered(misbehaviour_type);
        self.exclude_node(miner_hotkey, node_id).await;

        let hotkey_bytes = hotkey_ss58_to_bytes(miner_hotkey)?;
        let node_bytes = node_id_to_bytes(node_id)?;
        let network_config = to_network_config(&self.config)?;

        if self.config.shadow_mode {
            self.log_shadow_slash(
                &network_config,
                &hotkey_bytes,
                &node_bytes,
                miner_hotkey,
                node_id,
                &url,
            )
            .await?;
            return Ok(());
        }

        self.enforce_rate_limit(miner_hotkey, node_id).await?;
        let private_key = self.load_private_key().await?;
        let alpha_amount = self
            .resolve_slash_amount(
                &network_config,
                &hotkey_bytes,
                &node_bytes,
                miner_hotkey,
                node_id,
            )
            .await?;

        let submission = SlashSubmission {
            private_key: &private_key,
            hotkey_bytes,
            node_bytes,
            alpha_amount,
            url: &url,
            checksum,
            network_config: &network_config,
        };
        self.submit_slash(submission).await?;
        self.record_slash_executed();
        Ok(())
    }

    fn compute_slash_amount(&self, collateral: U256) -> U256 {
        if collateral.is_zero() || self.config.slash_fraction >= Decimal::ONE {
            return collateral;
        }
        let numerator = (self.config.slash_fraction * Decimal::from(10_000u64))
            .round()
            .to_u64()
            .unwrap_or(0);
        if numerator == 0 || numerator >= 10_000 {
            return collateral;
        }
        let amount = collateral * U256::from(numerator) / U256::from(10_000u64);
        if amount.is_zero() {
            U256::from(1u64)
        } else {
            amount
        }
    }

    async fn prepare_evidence(
        &self,
        misbehaviour_type: &str,
        details: &str,
        validator_hotkey: &str,
        rental_id: &str,
        miner_hotkey: &str,
        node_id: &str,
    ) -> Result<(String, u128)> {
        let evidence = self.build_evidence(
            misbehaviour_type,
            details,
            validator_hotkey,
            rental_id,
            miner_hotkey,
            node_id,
        )?;
        let (url, json) = self.evidence_store.store(&evidence).await?;
        let checksum = compute_sha256_checksum_truncated(&json);
        Ok((url, checksum))
    }

    fn build_evidence(
        &self,
        misbehaviour_type: &str,
        details: &str,
        validator_hotkey: &str,
        rental_id: &str,
        miner_hotkey: &str,
        node_id: &str,
    ) -> Result<SlashEvidence> {
        let mut evidence = SlashEvidence {
            rental_id: rental_id.to_string(),
            misbehaviour_type: misbehaviour_type.to_string(),
            timestamp: Utc::now(),
            details: details.to_string(),
            miner_hotkey: miner_hotkey.to_string(),
            node_id: node_id.to_string(),
            validator_hotkey: validator_hotkey.to_string(),
            shadow_mode: self.config.shadow_mode,
            signature: None,
        };

        if let Some(signer) = self.signer.as_ref() {
            let payload = evidence.signing_payload()?;
            let signature = signer
                .sign(&payload)
                .with_context(|| "failed to sign slash evidence payload")?;
            evidence.signature = Some(signature);
        }

        Ok(evidence)
    }

    fn record_slash_triggered(&self, misbehaviour_type: &str) {
        if let Some(metrics) = &self.metrics {
            metrics.record_collateral_slash_triggered(misbehaviour_type);
        }
    }

    async fn exclude_node(&self, miner_hotkey: &str, node_id: &str) {
        if let Err(err) = self
            .grace_tracker
            .force_exclude(miner_hotkey, node_id)
            .await
        {
            warn!("Failed to mark node excluded after slash trigger: {}", err);
        }
    }

    async fn log_shadow_slash(
        &self,
        network_config: &CollateralNetworkConfig,
        hotkey_bytes: &[u8; 32],
        node_bytes: &[u8; 16],
        miner_hotkey: &str,
        node_id: &str,
        url: &str,
    ) -> Result<()> {
        match self
            .resolve_onchain_alpha_amount(
                network_config,
                hotkey_bytes,
                node_bytes,
                miner_hotkey,
                node_id,
            )
            .await
        {
            Ok(collateral) => {
                let amount = self.compute_slash_amount(collateral);
                info!(
                    "[SHADOW] Would slash {} alpha (wei) for node {} (hotkey: {}). Evidence: {}",
                    amount, node_id, miner_hotkey, url
                );
            }
            Err(err) => {
                warn!(
                    "Failed to fetch on-chain alpha collateral for shadow slash: {}",
                    err
                );
            }
        }
        if let Some(metrics) = &self.metrics {
            metrics.record_collateral_slash_shadow();
        }
        Ok(())
    }

    async fn enforce_rate_limit(&self, miner_hotkey: &str, node_id: &str) -> Result<()> {
        if let Err(err) = self.rate_limiter.check_and_record(miner_hotkey).await {
            warn!(
                "Slash rate limited for miner {} (node {}): {}",
                miner_hotkey, node_id, err
            );
            return Err(err);
        }
        Ok(())
    }

    async fn load_private_key(&self) -> Result<String> {
        let provider: Box<dyn KeyProvider> = match self.config.trustee_key_source {
            TrusteeKeySource::File => {
                let path =
                    self.config.trustee_private_key_file.clone().context(
                        "trustee_private_key_file is required when trustee_key_source=file",
                    )?;
                Box::new(FileKeyProvider { path })
            }
            TrusteeKeySource::AwsSecrets => {
                let secret_name = self.config.aws_secret_name.clone().unwrap_or_default();
                if secret_name.trim().is_empty() {
                    anyhow::bail!(
                        "aws_secret_name is required when trustee_key_source=aws_secrets"
                    );
                }
                Box::new(AwsSecretsKeyProvider {
                    secret_name,
                    region: self.config.aws_region.clone(),
                })
            }
            TrusteeKeySource::EnvVar => Box::new(EnvKeyProvider {
                env_var: "TRUSTEE_PRIVATE_KEY".to_string(),
            }),
        };
        provider.load_private_key().await
    }

    async fn resolve_slash_amount(
        &self,
        network_config: &CollateralNetworkConfig,
        hotkey_bytes: &[u8; 32],
        node_bytes: &[u8; 16],
        miner_hotkey: &str,
        node_id: &str,
    ) -> Result<U256> {
        let onchain_collateral = self
            .resolve_onchain_alpha_amount(
                network_config,
                hotkey_bytes,
                node_bytes,
                miner_hotkey,
                node_id,
            )
            .await?;
        Ok(self.compute_slash_amount(onchain_collateral))
    }

    async fn submit_slash(&self, submission: SlashSubmission<'_>) -> Result<()> {
        self.chain_client
            .submit_slash(
                submission.private_key,
                submission.hotkey_bytes,
                submission.node_bytes,
                submission.alpha_amount,
                submission.url,
                submission.checksum,
                submission.network_config,
            )
            .await?;
        Ok(())
    }

    fn record_slash_executed(&self) {
        if let Some(metrics) = &self.metrics {
            metrics.record_collateral_slash_executed();
        }
    }
}

struct SlashSubmission<'a> {
    private_key: &'a str,
    hotkey_bytes: [u8; 32],
    node_bytes: [u8; 16],
    alpha_amount: U256,
    url: &'a str,
    checksum: u128,
    network_config: &'a CollateralNetworkConfig,
}

struct SlashRateLimiter {
    cooldown: StdDuration,
    max_per_window: usize,
    window: StdDuration,
    breaker_threshold: usize,
    breaker_window: StdDuration,
    breaker_cooldown: StdDuration,
    state: Mutex<SlashRateLimiterState>,
}

struct SlashRateLimiterState {
    per_miner_last_slash: HashMap<String, Instant>,
    global_slashes: VecDeque<Instant>,
    breaker_events: VecDeque<Instant>,
    circuit_open_until: Option<Instant>,
}

impl SlashRateLimiter {
    fn new(config: &CollateralConfig) -> Self {
        Self {
            cooldown: StdDuration::from_secs(config.slash_cooldown_secs),
            max_per_window: config.slash_max_per_window as usize,
            window: StdDuration::from_secs(config.slash_window_secs),
            breaker_threshold: config.slash_circuit_breaker_threshold as usize,
            breaker_window: StdDuration::from_secs(config.slash_circuit_breaker_window_secs),
            breaker_cooldown: StdDuration::from_secs(config.slash_circuit_breaker_cooldown_secs),
            state: Mutex::new(SlashRateLimiterState {
                per_miner_last_slash: HashMap::new(),
                global_slashes: VecDeque::new(),
                breaker_events: VecDeque::new(),
                circuit_open_until: None,
            }),
        }
    }

    async fn check_and_record(&self, miner_hotkey: &str) -> Result<()> {
        let mut state = self.state.lock().await;
        let now = Instant::now();

        if let Some(open_until) = state.circuit_open_until {
            if now < open_until {
                anyhow::bail!("circuit breaker open; retry after cooldown");
            }
            state.circuit_open_until = None;
        }

        state.prune(now, self.window, self.breaker_window);

        if let Some(last_slash) = state.per_miner_last_slash.get(miner_hotkey) {
            if now.duration_since(*last_slash) < self.cooldown {
                anyhow::bail!("per-miner slash cooldown active");
            }
        }

        if state.global_slashes.len() >= self.max_per_window {
            anyhow::bail!("global slash rate limit exceeded");
        }

        state
            .per_miner_last_slash
            .insert(miner_hotkey.to_string(), now);
        state.global_slashes.push_back(now);
        state.breaker_events.push_back(now);

        if state.breaker_events.len() >= self.breaker_threshold {
            state.circuit_open_until = now.checked_add(self.breaker_cooldown);
            state.breaker_events.clear();
            anyhow::bail!("slash circuit breaker opened");
        }

        Ok(())
    }
}

impl SlashRateLimiterState {
    fn prune(&mut self, now: Instant, window: StdDuration, breaker_window: StdDuration) {
        Self::prune_queue(&mut self.global_slashes, now, window);
        Self::prune_queue(&mut self.breaker_events, now, breaker_window);
    }

    fn prune_queue(queue: &mut VecDeque<Instant>, now: Instant, window: StdDuration) {
        let Some(cutoff) = now.checked_sub(window) else {
            queue.clear();
            return;
        };
        while queue.front().is_some_and(|ts| *ts < cutoff) {
            queue.pop_front();
        }
    }
}

fn compute_sha256_checksum_truncated(contents: &[u8]) -> u128 {
    let digest = Sha256::digest(contents);
    let mut truncated = [0u8; 16];
    truncated.copy_from_slice(&digest[..16]);
    u128::from_be_bytes(truncated)
}

fn to_network_config(config: &CollateralConfig) -> Result<CollateralNetworkConfig> {
    let network = match config.network.as_str() {
        "mainnet" => Network::Mainnet,
        "testnet" => Network::Testnet,
        "local" => Network::Local,
        other => {
            anyhow::bail!("Unsupported collateral network: {}", other);
        }
    };
    CollateralNetworkConfig::from_network(&network, Some(config.contract_address.clone()))
}

fn hotkey_ss58_to_bytes(hotkey: &str) -> Result<[u8; 32]> {
    let hex = hotkey_ss58_to_hex(hotkey)?;
    let decoded = hex::decode(hex)?;
    let bytes: [u8; 32] = decoded
        .as_slice()
        .try_into()
        .map_err(|_| anyhow::anyhow!("hotkey bytes length mismatch"))?;
    Ok(bytes)
}

fn node_id_to_bytes(node_id: &str) -> Result<[u8; 16]> {
    let hex = node_id_to_hex(node_id)?;
    let decoded = hex::decode(hex)?;
    let bytes: [u8; 16] = decoded
        .as_slice()
        .try_into()
        .map_err(|_| anyhow::anyhow!("node_id bytes length mismatch"))?;
    Ok(bytes)
}

impl SlashExecutor {
    async fn resolve_onchain_alpha_amount(
        &self,
        network_config: &CollateralNetworkConfig,
        hotkey_bytes: &[u8; 32],
        node_bytes: &[u8; 16],
        miner_hotkey: &str,
        node_id: &str,
    ) -> Result<U256> {
        let amount = self
            .chain_client
            .alpha_collaterals(*hotkey_bytes, *node_bytes, network_config)
            .await?;
        if amount.is_zero() {
            anyhow::bail!(
                "alpha collateral is zero for node {} (hotkey: {})",
                node_id,
                miner_hotkey
            );
        }
        Ok(amount)
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use rust_decimal::Decimal;
    use tempfile::tempdir;
    use uuid::Uuid;

    #[tokio::test]
    async fn test_sha256_checksum_truncated_nonzero() {
        let checksum = compute_sha256_checksum_truncated(b"test");
        assert!(checksum > 0);
    }

    #[tokio::test]
    async fn test_shadow_mode_does_not_error() {
        let temp = tempdir().unwrap();
        let config = CollateralConfig {
            shadow_mode: true,
            evidence_storage_path: temp.path().to_path_buf(),
            evidence_base_url: "https://validator.example.com/evidence".to_string(),
            network: "local".to_string(),
            contract_address: "0x0000000000000000000000000000000000000001".to_string(),
            ..CollateralConfig::default()
        };
        let store = EvidenceStore::new_local(
            config.evidence_base_url.clone(),
            config.evidence_storage_path.clone(),
        );
        let grace_tracker = Arc::new(GracePeriodTracker::new(
            Arc::new(
                crate::persistence::SimplePersistence::for_testing()
                    .await
                    .unwrap(),
            ),
            config.grace_period(),
        ));
        let executor = SlashExecutor::new(config, store, grace_tracker, None, None);
        executor
            .execute_slash(
                "5GrwvaEF5zXb26Fz9rcQpDWS57CtERHpNehXCPcNoHGKutQY",
                &Uuid::new_v4().to_string(),
                "deployment_failed",
                "{}",
                "validator-hotkey",
                "rental-1",
            )
            .await
            .unwrap();
    }

    #[tokio::test]
    async fn test_compute_slash_amount() {
        let temp = tempdir().unwrap();
        let config = CollateralConfig {
            slash_fraction: Decimal::new(5, 1),
            evidence_storage_path: temp.path().to_path_buf(),
            contract_address: "0x0000000000000000000000000000000000000001".to_string(),
            ..CollateralConfig::default()
        };
        let store = EvidenceStore::new_local(
            config.evidence_base_url.clone(),
            config.evidence_storage_path.clone(),
        );
        let grace_tracker = Arc::new(GracePeriodTracker::new(
            Arc::new(
                crate::persistence::SimplePersistence::for_testing()
                    .await
                    .unwrap(),
            ),
            config.grace_period(),
        ));
        let executor = SlashExecutor::new(config, store, grace_tracker, None, None);
        assert_eq!(
            executor.compute_slash_amount(U256::from(1000u64)),
            U256::from(500u64)
        );
        assert_eq!(executor.compute_slash_amount(U256::ZERO), U256::ZERO);
    }
}
