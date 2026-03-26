use alloy_primitives::U256;
use anyhow::Result;
use async_trait::async_trait;
use basilica_validator::basilica_api::ValidatorSigner;
use basilica_validator::collateral::evidence::EvidenceStore;
use basilica_validator::collateral::grace_tracker::GracePeriodTracker;
use basilica_validator::collateral::{CollateralChainClient, SlashExecutor};
use basilica_validator::config::collateral::{CollateralConfig, TrusteeKeySource};
use basilica_validator::metrics::ValidatorPrometheusMetrics;
use basilica_validator::persistence::SimplePersistence;
use metrics_exporter_prometheus::{PrometheusBuilder, PrometheusHandle};
use once_cell::sync::OnceCell;
use sqlx::SqlitePool;
use std::sync::Arc;
use tempfile::tempdir;
use tokio::sync::Mutex;
use uuid::Uuid;

struct TestSigner;

impl ValidatorSigner for TestSigner {
    fn hotkey(&self) -> String {
        "validator_hotkey".to_string()
    }

    fn sign(&self, _message: &[u8]) -> Result<String> {
        Ok("test-signature".to_string())
    }
}

#[derive(Clone, Debug)]
struct MockSlashCall {
    private_key: String,
    hotkey_bytes: [u8; 32],
    node_bytes: [u8; 16],
    alpha_amount: U256,
    url: String,
    checksum: u128,
}

#[derive(Clone)]
struct MockChainClient {
    alpha_collateral: U256,
    calls: Arc<Mutex<Vec<MockSlashCall>>>,
}

impl MockChainClient {
    fn new(alpha_collateral: U256) -> Self {
        Self {
            alpha_collateral,
            calls: Arc::new(Mutex::new(Vec::new())),
        }
    }

    async fn calls(&self) -> Vec<MockSlashCall> {
        self.calls.lock().await.clone()
    }
}

#[async_trait]
impl CollateralChainClient for MockChainClient {
    async fn alpha_collaterals(
        &self,
        _hotkey_bytes: [u8; 32],
        _node_bytes: [u8; 16],
        _network_config: &collateral_contract::config::CollateralNetworkConfig,
    ) -> Result<U256> {
        Ok(self.alpha_collateral)
    }

    async fn submit_slash(
        &self,
        private_key: &str,
        hotkey_bytes: [u8; 32],
        node_bytes: [u8; 16],
        alpha_amount: U256,
        url: &str,
        checksum: u128,
        _network_config: &collateral_contract::config::CollateralNetworkConfig,
    ) -> Result<()> {
        let mut calls = self.calls.lock().await;
        calls.push(MockSlashCall {
            private_key: private_key.to_string(),
            hotkey_bytes,
            node_bytes,
            alpha_amount,
            url: url.to_string(),
            checksum,
        });
        Ok(())
    }
}

fn init_metrics_handle() -> PrometheusHandle {
    static HANDLE: OnceCell<PrometheusHandle> = OnceCell::new();
    HANDLE
        .get_or_init(|| {
            PrometheusBuilder::new()
                .install_recorder()
                .expect("failed to install Prometheus recorder")
        })
        .clone()
}

async fn build_persistence() -> Result<Arc<SimplePersistence>> {
    let pool = SqlitePool::connect(":memory:").await?;
    let persistence = SimplePersistence::with_pool(pool);
    persistence.run_migrations().await?;
    Ok(Arc::new(persistence))
}

async fn build_executor(
    mut config: CollateralConfig,
    temp_path: &std::path::Path,
    persistence: Arc<SimplePersistence>,
    chain_client: Arc<dyn CollateralChainClient>,
    metrics: Option<Arc<ValidatorPrometheusMetrics>>,
    signer: Option<Arc<dyn ValidatorSigner>>,
) -> Result<SlashExecutor> {
    config.shadow_mode = false;
    config.trustee_key_source = TrusteeKeySource::EnvVar;
    config.evidence_storage_path = temp_path.to_path_buf();
    config.evidence_base_url = "https://validator.example.com/evidence".to_string();
    config.network = "local".to_string();
    config.contract_address = "0x0000000000000000000000000000000000000001".to_string();

    let store = EvidenceStore::new_local(
        config.evidence_base_url.clone(),
        config.evidence_storage_path.clone(),
    );
    let grace_tracker = Arc::new(GracePeriodTracker::new(
        persistence.clone(),
        config.grace_period(),
    ));
    let executor = SlashExecutor::new_with_chain_client(
        config,
        store,
        grace_tracker,
        metrics,
        signer,
        chain_client,
    );
    Ok(executor)
}

#[tokio::test]
async fn test_slash_flow_executes_and_emits_metrics() -> Result<()> {
    std::env::set_var("TRUSTEE_PRIVATE_KEY", "test-private-key");
    let handle = init_metrics_handle();
    let temp = tempdir()?;
    let chain_client = Arc::new(MockChainClient::new(U256::from(1000u64)));
    let persistence = build_persistence().await?;
    let metrics = Arc::new(ValidatorPrometheusMetrics::new(persistence.clone())?);
    let signer = Arc::new(TestSigner);
    let executor = build_executor(
        CollateralConfig::default(),
        temp.path(),
        persistence,
        chain_client.clone(),
        Some(metrics),
        Some(signer),
    )
    .await?;

    let node_id = Uuid::new_v4().to_string();
    executor
        .execute_slash(
            "5GrwvaEF5zXb26Fz9rcQpDWS57CtERHpNehXCPcNoHGKutQY",
            &node_id,
            "deployment_failed",
            "{}",
            "validator_hotkey",
            "rental-1",
        )
        .await?;

    let calls = chain_client.calls().await;
    assert_eq!(calls.len(), 1);
    assert_eq!(calls[0].alpha_amount, U256::from(1000u64));
    assert_eq!(calls[0].private_key, "test-private-key");
    assert!(calls[0].checksum > 0);
    assert!(calls[0].url.contains("validator.example.com"));
    assert!(calls[0].hotkey_bytes.iter().any(|byte| *byte != 0));
    assert!(calls[0].node_bytes.iter().any(|byte| *byte != 0));

    let mut dir = tokio::fs::read_dir(temp.path()).await?;
    let mut evidence_path = None;
    while let Some(entry) = dir.next_entry().await? {
        let name = entry.file_name();
        if name.to_string_lossy().starts_with("evidence-") {
            evidence_path = Some(entry.path());
            break;
        }
    }
    let evidence_path = evidence_path.expect("evidence file not written");
    let json = tokio::fs::read(evidence_path).await?;
    let value: serde_json::Value = serde_json::from_slice(&json)?;
    let signature = value.get("signature").and_then(|v| v.as_str());
    assert_eq!(signature, Some("test-signature"));

    let rendered = handle.render();
    assert!(rendered.contains("basilica_validator_collateral_slash_triggered_total"));
    assert!(rendered.contains("basilica_validator_collateral_slash_executed_total"));
    Ok(())
}

#[tokio::test]
async fn test_rate_limiter_blocks_repeat_slash() -> Result<()> {
    std::env::set_var("TRUSTEE_PRIVATE_KEY", "test-private-key");
    let temp = tempdir()?;
    let chain_client = Arc::new(MockChainClient::new(U256::from(1000u64)));
    let config = CollateralConfig {
        slash_cooldown_secs: 3600,
        ..Default::default()
    };
    let signer = Arc::new(TestSigner);
    let persistence = build_persistence().await?;
    let executor = build_executor(
        config,
        temp.path(),
        persistence,
        chain_client,
        None,
        Some(signer),
    )
    .await?;

    let node_id = Uuid::new_v4().to_string();
    executor
        .execute_slash(
            "5GrwvaEF5zXb26Fz9rcQpDWS57CtERHpNehXCPcNoHGKutQY",
            &node_id,
            "deployment_failed",
            "{}",
            "validator_hotkey",
            "rental-1",
        )
        .await?;

    let err = executor
        .execute_slash(
            "5GrwvaEF5zXb26Fz9rcQpDWS57CtERHpNehXCPcNoHGKutQY",
            &node_id,
            "deployment_failed",
            "{}",
            "validator_hotkey",
            "rental-2",
        )
        .await
        .expect_err("expected per-miner cooldown to reject second slash");
    assert!(err.to_string().contains("per-miner slash cooldown active"));
    Ok(())
}

#[tokio::test]
async fn test_circuit_breaker_trips_on_burst() -> Result<()> {
    std::env::set_var("TRUSTEE_PRIVATE_KEY", "test-private-key");
    let temp = tempdir()?;
    let chain_client = Arc::new(MockChainClient::new(U256::from(1000u64)));
    let config = CollateralConfig {
        slash_circuit_breaker_threshold: 2,
        slash_circuit_breaker_window_secs: 3600,
        slash_circuit_breaker_cooldown_secs: 3600,
        slash_max_per_window: 100,
        ..Default::default()
    };
    let signer = Arc::new(TestSigner);
    let persistence = build_persistence().await?;
    let executor = build_executor(
        config,
        temp.path(),
        persistence,
        chain_client,
        None,
        Some(signer),
    )
    .await?;

    executor
        .execute_slash(
            "5GrwvaEF5zXb26Fz9rcQpDWS57CtERHpNehXCPcNoHGKutQY",
            &Uuid::new_v4().to_string(),
            "deployment_failed",
            "{}",
            "validator_hotkey",
            "rental-1",
        )
        .await?;

    let err = executor
        .execute_slash(
            "5DAAnrj7VHTznn2AWBemMuyBwZWs6FNFjdyVXUeYum3PTXFy",
            &Uuid::new_v4().to_string(),
            "deployment_failed",
            "{}",
            "validator_hotkey",
            "rental-2",
        )
        .await
        .expect_err("expected circuit breaker to trip on burst");
    let err_message = err.to_string();
    assert!(
        err_message.contains("slash circuit breaker opened"),
        "unexpected error: {}",
        err_message
    );
    Ok(())
}
