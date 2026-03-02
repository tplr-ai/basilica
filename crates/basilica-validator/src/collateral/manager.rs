use crate::basilica_api::BasilicaApiClient;
use crate::collateral::evaluator::{CollateralEvaluator, CollateralState, CollateralStatus};
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
}

#[derive(Clone)]
pub struct CollateralManager {
    persistence: Arc<SimplePersistence>,
    api_client: Arc<BasilicaApiClient>,
    evaluator: Arc<CollateralEvaluator>,
    netuid: u16,
    metrics: Option<Arc<ValidatorPrometheusMetrics>>,
}

impl CollateralManager {
    pub fn new(
        persistence: Arc<SimplePersistence>,
        api_client: Arc<BasilicaApiClient>,
        evaluator: Arc<CollateralEvaluator>,
        netuid: u16,
        metrics: Option<Arc<ValidatorPrometheusMetrics>>,
    ) -> Self {
        Self {
            persistence,
            api_client,
            evaluator,
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
            },
            Err(_) => CollateralPreference::Fallback,
        }
    }

    pub async fn refresh_price_cache(&self) {
        // TTL-only pricing: no background refresh loop
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
    Ok(format!("0x{}", encode(account_bytes)))
}

pub fn node_id_to_hex(node_id: &str) -> Result<String> {
    let uuid = Uuid::parse_str(node_id)?;
    Ok(format!("0x{}", encode(uuid.as_bytes())))
}

fn u256_to_alpha(amount: alloy_primitives::U256) -> Decimal {
    let amount_str = amount.to_string();
    // alphaCollaterals stores RAO (1e9 = 1 alpha), not wei (1e18).
    match Decimal::from_str(&amount_str) {
        Ok(value) => value * Decimal::from_i128_with_scale(1, 9),
        Err(_) => {
            warn!(
                "Collateral amount {} exceeds Decimal precision; capping at Decimal::MAX",
                amount_str
            );
            // TODO: Switch to BigDecimal or fixed-point U256 conversion to avoid loss.
            Decimal::MAX * Decimal::from_i128_with_scale(1, 9)
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

    struct FixedPriceFetcher;

    #[async_trait::async_trait]
    impl TokenPriceFetcher for FixedPriceFetcher {
        async fn fetch(
            &self,
            _client: &BasilicaApiClient,
            _netuid: u16,
        ) -> Result<TokenPriceSnapshot> {
            Ok(TokenPriceSnapshot {
                tao_price_usd: Decimal::ONE,
                alpha_price_usd: Decimal::ONE,
                alpha_price_tao: Decimal::ONE,
                tao_reserve: Decimal::ONE,
                alpha_reserve: Decimal::ONE,
                fetched_at: "2026-01-01T00:00:00Z".to_string(),
            })
        }
    }

    struct TestBaselineFetcher;

    #[async_trait::async_trait]
    impl BaselinePriceFetcher for TestBaselineFetcher {
        async fn fetch(&self, _client: &BasilicaApiClient) -> Result<HashMap<String, f64>> {
            Ok(HashMap::new())
        }
    }

    #[test]
    fn test_u256_to_alpha_zero() {
        let alpha = u256_to_alpha(alloy_primitives::U256::ZERO);
        assert_eq!(alpha, Decimal::ZERO);
    }

    #[test]
    fn test_u256_to_alpha_one_rao() {
        // 1 RAO = 1e-9 alpha
        let alpha = u256_to_alpha(alloy_primitives::U256::from(1u64));
        assert_eq!(alpha, Decimal::from_i128_with_scale(1, 9));
    }

    #[test]
    fn test_u256_to_alpha_one_alpha() {
        // 1e9 RAO = 1 alpha
        let alpha = u256_to_alpha(alloy_primitives::U256::from(1_000_000_000u64));
        assert_eq!(alpha, Decimal::ONE);
    }

    #[test]
    fn test_u256_to_alpha_fractional() {
        // 500_000_000 RAO = 0.5 alpha
        let alpha = u256_to_alpha(alloy_primitives::U256::from(500_000_000u64));
        assert_eq!(alpha, Decimal::new(5, 1));
    }

    #[test]
    fn test_u256_to_alpha_large_amount() {
        // 5e9 RAO = 5 alpha
        let alpha = u256_to_alpha(alloy_primitives::U256::from(5_000_000_000u64));
        assert_eq!(alpha, Decimal::from(5));
    }

    #[test]
    fn test_u256_to_alpha_is_not_wei() {
        // Regression: old code divided by 1e18 (wei). Verify 1e9 RAO = 1 alpha, NOT 1e-9 alpha.
        let alpha = u256_to_alpha(alloy_primitives::U256::from(1_000_000_000u64));
        assert_ne!(
            alpha,
            Decimal::from_i128_with_scale(1, 9),
            "should NOT treat input as wei"
        );
        assert_eq!(alpha, Decimal::ONE, "1e9 RAO must equal 1 alpha");
    }

    #[tokio::test]
    async fn test_node_id_to_hex() {
        let uuid = Uuid::new_v4();
        let hex = node_id_to_hex(&uuid.to_string()).unwrap();
        assert_eq!(hex.len(), 34); // "0x" + 32 hex chars
        assert!(hex.starts_with("0x"));
    }

    #[tokio::test]
    async fn test_get_collateral_alpha_missing_returns_zero() {
        let persistence = Arc::new(SimplePersistence::for_testing().await.unwrap());
        let config = CollateralConfig::default();
        let evaluator = Arc::new(CollateralEvaluator::new(config.clone()));
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
        let manager = CollateralManager::new(persistence.clone(), api_client, evaluator, 1, None);
        let alpha = manager
            .get_collateral_alpha(
                "5GrwvaEF5zXb26Fz9rcQpDWS57CtERHpNehXCPcNoHGKutQY",
                &Uuid::new_v4().to_string(),
            )
            .await
            .unwrap();
        assert_eq!(alpha, Decimal::ZERO);
    }

    #[tokio::test]
    async fn test_get_collateral_alpha_converts_rao_to_alpha() {
        let persistence = Arc::new(SimplePersistence::for_testing().await.unwrap());
        let hotkey = "5GrwvaEF5zXb26Fz9rcQpDWS57CtERHpNehXCPcNoHGKutQY";
        let node_id = Uuid::new_v4().to_string();
        let hotkey_hex = hotkey_ss58_to_hex(hotkey).unwrap();
        let node_hex = node_id_to_hex(&node_id).unwrap();

        // Insert 5e9 RAO (= 5 alpha) into the DB, mimicking on-chain event data.
        sqlx::query(
            "INSERT INTO collateral_status (hotkey, node_id, miner, tao_collateral, alpha_collateral, pending_tao_reclaim, pending_alpha_reclaim, updated_at) VALUES (?, ?, ?, ?, ?, ?, ?, CURRENT_TIMESTAMP)",
        )
        .bind(&hotkey_hex)
        .bind(&node_hex)
        .bind("0x0000000000000000000000000000000000000001")
        .bind("0")
        .bind("5000000000") // 5e9 RAO = 5 alpha
        .bind("0")
        .bind("0")
        .execute(persistence.pool())
        .await
        .unwrap();

        let config = CollateralConfig::default();
        let evaluator = Arc::new(CollateralEvaluator::new(config.clone()));
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
        let manager = CollateralManager::new(persistence, api_client, evaluator, 1, None);

        let alpha = manager
            .get_collateral_alpha(hotkey, &node_id)
            .await
            .unwrap();
        assert_eq!(alpha, Decimal::from(5), "5e9 RAO should convert to 5 alpha");
    }

    #[tokio::test]
    async fn test_tao_is_non_authoritative_for_collateral_status() {
        let persistence = Arc::new(SimplePersistence::for_testing().await.unwrap());
        let hotkey = "5GrwvaEF5zXb26Fz9rcQpDWS57CtERHpNehXCPcNoHGKutQY";
        let node_id = Uuid::new_v4().to_string();
        let hotkey_hex = hotkey_ss58_to_hex(hotkey).unwrap();
        let node_hex = node_id_to_hex(&node_id).unwrap();

        // Persist very high TAO with zero alpha to prove policy uses alpha only.
        sqlx::query(
            "INSERT INTO collateral_status (hotkey, node_id, miner, tao_collateral, alpha_collateral, pending_tao_reclaim, pending_alpha_reclaim, updated_at) VALUES (?, ?, ?, ?, ?, ?, ?, CURRENT_TIMESTAMP)",
        )
        .bind(hotkey_hex)
        .bind(node_hex)
        .bind("0x0000000000000000000000000000000000000001")
        .bind("1000000000000000000000000")
        .bind("0")
        .bind("0")
        .bind("0")
        .execute(persistence.pool())
        .await
        .unwrap();

        let config = CollateralConfig::default();
        let evaluator = Arc::new(CollateralEvaluator::new(config.clone()));
        let signer: Arc<dyn ValidatorSigner> = Arc::new(TestSigner);
        let api_client = Arc::new(BasilicaApiClient::new_with_fetchers(
            "http://localhost".to_string(),
            signer,
            reqwest::Client::new(),
            std::time::Duration::from_secs(60),
            std::time::Duration::from_secs(60),
            Arc::new(TestBaselineFetcher),
            Arc::new(FixedPriceFetcher),
        ));
        let manager = CollateralManager::new(persistence, api_client, evaluator, 1, None);

        let (state, status) = manager
            .get_collateral_status(hotkey, &node_id, "H100", 1)
            .await
            .unwrap();
        assert!(matches!(state, CollateralState::Undercollateralized { .. }));
        assert_eq!(status.current_alpha, Decimal::ZERO);
        assert_eq!(status.status, "undercollateralized");
    }
}
