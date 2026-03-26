use crate::basilica_api::BasilicaApiClient;
use crate::collateral::evaluator::{CollateralEvaluator, CollateralState, CollateralStatus};
use crate::collateral::grace_tracker::GracePeriodTracker;
use crate::metrics::ValidatorPrometheusMetrics;
use crate::persistence::SimplePersistence;
use anyhow::Result;
use basilica_common::identity::Hotkey;
use hex::encode;
use rust_decimal::prelude::ToPrimitive;
use rust_decimal::Decimal;
use std::str::FromStr;
use std::sync::Arc;
use tracing::warn;
use uuid::Uuid;

#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum CollateralPreference {
    Preferred,
    Fallback,
    Excluded,
}

#[derive(Clone)]
pub struct CollateralManager {
    persistence: Arc<SimplePersistence>,
    api_client: Arc<BasilicaApiClient>,
    evaluator: Arc<CollateralEvaluator>,
    grace_tracker: Arc<GracePeriodTracker>,
    netuid: u16,
    metrics: Option<Arc<ValidatorPrometheusMetrics>>,
}

impl CollateralManager {
    pub fn new(
        persistence: Arc<SimplePersistence>,
        api_client: Arc<BasilicaApiClient>,
        evaluator: Arc<CollateralEvaluator>,
        grace_tracker: Arc<GracePeriodTracker>,
        netuid: u16,
        metrics: Option<Arc<ValidatorPrometheusMetrics>>,
    ) -> Self {
        Self {
            persistence,
            api_client,
            evaluator,
            grace_tracker,
            netuid,
            metrics,
        }
    }

    pub async fn get_collateral_status(
        &self,
        hotkey: &str,
        node_id: &str,
        gpu_category: &str,
        gpu_count: u32,
    ) -> Result<(CollateralState, CollateralStatus)> {
        let alpha_price_usd = match self.api_client.get_alpha_price_usd(self.netuid).await {
            Ok(price) => Some(price),
            Err(err) => {
                warn!("Alpha price unavailable: {}", err);
                None
            }
        };

        if let Some(metrics) = &self.metrics {
            if let Some(alpha_usd) = &alpha_price_usd {
                let alpha_usd = alpha_usd.to_f64().unwrap_or_default();
                metrics.record_collateral_price(alpha_usd);
            }
        }

        let collateral_alpha = self
            .get_collateral_alpha(hotkey, node_id)
            .await
            .unwrap_or(Decimal::ZERO);

        let (state, status) = self
            .evaluator
            .evaluate(
                hotkey,
                node_id,
                gpu_category,
                gpu_count,
                collateral_alpha,
                alpha_price_usd,
            )
            .await?;
        if let Some(metrics) = &self.metrics {
            metrics.record_collateral_node_status(hotkey, node_id, gpu_category, &status.status);
        }
        Ok((state, status))
    }

    pub async fn get_preference(
        &self,
        hotkey: &str,
        node_id: &str,
        gpu_category: &str,
        gpu_count: u32,
    ) -> CollateralPreference {
        match self
            .get_collateral_status(hotkey, node_id, gpu_category, gpu_count)
            .await
        {
            Ok((state, _)) => match state {
                CollateralState::Sufficient { .. } | CollateralState::Warning { .. } => {
                    CollateralPreference::Preferred
                }
                CollateralState::Undercollateralized { .. } | CollateralState::Unknown { .. } => {
                    CollateralPreference::Fallback
                }
                CollateralState::Excluded { .. } => CollateralPreference::Excluded,
            },
            Err(_) => CollateralPreference::Fallback,
        }
    }

    pub async fn refresh_price_cache(&self) {
        // TTL-only pricing: no background refresh loop
    }

    pub async fn is_eligible_for_bids(
        &self,
        hotkey: &str,
        node_id: &str,
        gpu_category: &str,
        gpu_count: u32,
    ) -> bool {
        match self
            .get_collateral_status(hotkey, node_id, gpu_category, gpu_count)
            .await
        {
            Ok((state, _)) => !matches!(state, CollateralState::Excluded { .. }),
            Err(_) => true,
        }
    }

    pub async fn force_exclude(&self, hotkey: &str, node_id: &str) -> Result<()> {
        self.grace_tracker.force_exclude(hotkey, node_id).await
    }

    pub async fn get_collateral_alpha(&self, hotkey: &str, node_id: &str) -> Result<Decimal> {
        let hotkey_hex = match hotkey_ss58_to_hex(hotkey) {
            Ok(val) => val,
            Err(err) => {
                warn!("Failed to convert hotkey to hex: {}", err);
                return Ok(Decimal::ZERO);
            }
        };
        let node_hex = match node_id_to_hex(node_id) {
            Ok(val) => val,
            Err(err) => {
                warn!("Failed to convert node_id to hex: {}", err);
                return Ok(Decimal::ZERO);
            }
        };

        let amount = self
            .persistence
            .get_alpha_collateral_amount(&hotkey_hex, &node_hex)
            .await?;
        let amount = amount.unwrap_or_default();
        Ok(u256_to_alpha(amount))
    }
}

pub fn hotkey_ss58_to_hex(hotkey: &str) -> Result<String> {
    let hotkey =
        Hotkey::new(hotkey.to_string()).map_err(|e| anyhow::anyhow!("invalid hotkey: {e}"))?;
    let account_id = hotkey
        .to_account_id()
        .map_err(|e| anyhow::anyhow!("hotkey conversion failed: {e}"))?;
    let account_bytes: &[u8] = account_id.as_ref();
    Ok(encode(account_bytes))
}

pub fn node_id_to_hex(node_id: &str) -> Result<String> {
    let uuid = Uuid::parse_str(node_id)?;
    Ok(encode(uuid.as_bytes()))
}

fn u256_to_alpha(amount: alloy_primitives::U256) -> Decimal {
    let amount_str = amount.to_string();
    match Decimal::from_str(&amount_str) {
        Ok(value) => value * Decimal::from_i128_with_scale(1, 18),
        Err(_) => {
            warn!(
                "Collateral amount {} exceeds Decimal precision; capping at Decimal::MAX",
                amount_str
            );
            // TODO: Switch to BigDecimal or fixed-point U256 conversion to avoid loss.
            Decimal::MAX * Decimal::from_i128_with_scale(1, 18)
        }
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::basilica_api::{
        BaselinePriceFetcher, BasilicaApiClient, TokenPriceFetcher, TokenPriceSnapshot,
        ValidatorSigner,
    };
    use crate::config::collateral::CollateralConfig;
    use crate::persistence::SimplePersistence;
    use chrono::Duration;
    use rust_decimal::Decimal;
    use std::collections::HashMap;

    struct TestSigner;

    impl ValidatorSigner for TestSigner {
        fn hotkey(&self) -> String {
            "test_hotkey".to_string()
        }

        fn sign(&self, _message: &[u8]) -> Result<String> {
            Ok("deadbeef".to_string())
        }
    }

    struct TestFetcher;

    #[async_trait::async_trait]
    impl TokenPriceFetcher for TestFetcher {
        async fn fetch(
            &self,
            _client: &BasilicaApiClient,
            _netuid: u16,
        ) -> Result<TokenPriceSnapshot> {
            anyhow::bail!("unused")
        }
    }

    struct TestBaselineFetcher;

    #[async_trait::async_trait]
    impl BaselinePriceFetcher for TestBaselineFetcher {
        async fn fetch(&self, _client: &BasilicaApiClient) -> Result<HashMap<String, f64>> {
            Ok(HashMap::new())
        }
    }

    #[tokio::test]
    async fn test_node_id_to_hex() {
        let uuid = Uuid::new_v4();
        let hex = node_id_to_hex(&uuid.to_string()).unwrap();
        assert_eq!(hex.len(), 32);
    }

    #[tokio::test]
    async fn test_get_collateral_alpha_missing_returns_zero() {
        let persistence = Arc::new(SimplePersistence::for_testing().await.unwrap());
        let config = CollateralConfig::default();
        let grace_tracker = Arc::new(GracePeriodTracker::new(
            persistence.clone(),
            Duration::hours(24),
        ));
        let evaluator = Arc::new(CollateralEvaluator::new(
            config.clone(),
            grace_tracker.clone(),
        ));
        let signer: Arc<dyn ValidatorSigner> = Arc::new(TestSigner);
        let api_client = Arc::new(BasilicaApiClient::new_with_fetchers(
            "http://localhost".to_string(),
            signer,
            reqwest::Client::new(),
            std::time::Duration::from_secs(60),
            std::time::Duration::from_secs(60),
            Arc::new(TestBaselineFetcher),
            Arc::new(TestFetcher),
        ));
        let manager = CollateralManager::new(
            persistence.clone(),
            api_client,
            evaluator,
            grace_tracker,
            1,
            None,
        );
        let alpha = manager
            .get_collateral_alpha(
                "5GrwvaEF5zXb26Fz9rcQpDWS57CtERHpNehXCPcNoHGKutQY",
                &Uuid::new_v4().to_string(),
            )
            .await
            .unwrap();
        assert_eq!(alpha, Decimal::ZERO);
    }
}
