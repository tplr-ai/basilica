use anyhow::Result;
use basilica_validator::basilica_api::ValidatorSigner;
use basilica_validator::collateral::evidence::EvidenceStore;
use basilica_validator::collateral::grace_tracker::GracePeriodTracker;
use basilica_validator::collateral::SlashExecutor;
use basilica_validator::config::collateral::CollateralConfig;
use basilica_validator::persistence::SimplePersistence;
use sqlx::SqlitePool;
use std::sync::Arc;
use tempfile::tempdir;
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

#[tokio::test]
async fn test_collateral_slash_flow_writes_signed_evidence() {
    let temp = tempdir().unwrap();
    let config = CollateralConfig {
        shadow_mode: true,
        network: "local".to_string(),
        contract_address: "0x0000000000000000000000000000000000000001".to_string(),
        evidence_storage_path: temp.path().to_path_buf(),
        evidence_base_url: "https://validator.example.com/evidence".to_string(),
        ..Default::default()
    };

    let store = EvidenceStore::new_local(
        config.evidence_base_url.clone(),
        config.evidence_storage_path.clone(),
    );
    let pool = SqlitePool::connect(":memory:").await.unwrap();
    let persistence = SimplePersistence::with_pool(pool);
    persistence.run_migrations().await.unwrap();
    let persistence = Arc::new(persistence);
    let grace_tracker = Arc::new(GracePeriodTracker::new(persistence, config.grace_period()));
    let signer = Arc::new(TestSigner);
    let executor = SlashExecutor::new(config, store, grace_tracker, None, Some(signer));

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
        .await
        .unwrap();

    let mut dir = tokio::fs::read_dir(temp.path()).await.unwrap();
    let mut evidence_path = None;
    while let Some(entry) = dir.next_entry().await.unwrap() {
        let name = entry.file_name();
        if name.to_string_lossy().starts_with("evidence-") {
            evidence_path = Some(entry.path());
            break;
        }
    }

    let evidence_path = evidence_path.expect("evidence file not written");
    let json = tokio::fs::read(evidence_path).await.unwrap();
    let value: serde_json::Value = serde_json::from_slice(&json).unwrap();
    let signature = value.get("signature").and_then(|v| v.as_str());
    assert_eq!(signature, Some("test-signature"));
    assert_eq!(
        value.get("misbehaviour_type").and_then(|v| v.as_str()),
        Some("deployment_failed")
    );
}
