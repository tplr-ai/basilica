use anyhow::{Context, Result};
use aws_config::BehaviorVersion;
use aws_sdk_s3::primitives::ByteStream;
use aws_sdk_s3::Client as S3Client;
use chrono::{DateTime, Utc};
use serde::Serialize;
use std::path::PathBuf;
use tokio::fs;
use uuid::Uuid;

#[derive(Debug, Serialize)]
pub struct SlashEvidence {
    pub rental_id: String,
    pub misbehaviour_type: String,
    pub timestamp: DateTime<Utc>,
    pub details: String,
    pub miner_hotkey: String,
    pub node_id: String,
    pub validator_hotkey: String,
    pub shadow_mode: bool,
    #[serde(skip_serializing_if = "Option::is_none")]
    pub signature: Option<String>,
}

#[derive(Debug, Serialize)]
struct SlashEvidencePayload<'a> {
    rental_id: &'a str,
    misbehaviour_type: &'a str,
    timestamp: DateTime<Utc>,
    details: &'a str,
    miner_hotkey: &'a str,
    node_id: &'a str,
    validator_hotkey: &'a str,
    shadow_mode: bool,
}

impl SlashEvidence {
    pub fn signing_payload(&self) -> Result<Vec<u8>> {
        let payload = SlashEvidencePayload {
            rental_id: &self.rental_id,
            misbehaviour_type: &self.misbehaviour_type,
            timestamp: self.timestamp,
            details: &self.details,
            miner_hotkey: &self.miner_hotkey,
            node_id: &self.node_id,
            validator_hotkey: &self.validator_hotkey,
            shadow_mode: self.shadow_mode,
        };
        Ok(serde_json::to_vec(&payload)?)
    }
}

#[derive(Clone)]
pub struct EvidenceStore {
    public_url_base: String,
    storage_path: PathBuf,
    r2_client: Option<S3Client>,
    r2_bucket: Option<String>,
    require_upload: bool,
}

impl EvidenceStore {
    pub fn new_local(base_url: String, storage_path: PathBuf) -> Self {
        Self {
            public_url_base: base_url,
            storage_path,
            r2_client: None,
            r2_bucket: None,
            require_upload: false,
        }
    }

    pub async fn from_config(config: &crate::config::collateral::CollateralConfig) -> Result<Self> {
        let public_url_base = config
            .evidence_public_url_base
            .clone()
            .unwrap_or_else(|| config.evidence_base_url.clone());
        let storage_path = config.evidence_storage_path.clone();
        let require_upload = !config.shadow_mode;

        let r2_client = if config.evidence_r2_bucket.is_some() {
            Some(build_r2_client(config).await?)
        } else {
            None
        };

        Ok(Self {
            public_url_base,
            storage_path,
            r2_client,
            r2_bucket: config.evidence_r2_bucket.clone(),
            require_upload,
        })
    }

    pub async fn store(&self, evidence: &SlashEvidence) -> Result<(String, Vec<u8>)> {
        let file_id = format!("evidence-{}.json", Uuid::new_v4());
        let json = serde_json::to_vec_pretty(evidence)?;

        fs::create_dir_all(&self.storage_path).await?;
        let path = self.storage_path.join(&file_id);
        fs::write(&path, &json).await?;

        if self.require_upload && self.r2_client.is_none() {
            anyhow::bail!("R2 evidence upload is required but not configured");
        }

        if let Some(client) = &self.r2_client {
            let bucket = self
                .r2_bucket
                .as_ref()
                .context("R2 bucket missing for evidence upload")?;
            client
                .put_object()
                .bucket(bucket)
                .key(&file_id)
                .body(ByteStream::from(json.clone()))
                .content_type("application/json")
                .send()
                .await?;
        }

        let url = self.build_url(&file_id);
        Ok((url, json))
    }

    fn build_url(&self, file_name: &str) -> String {
        let base = self.public_url_base.trim_end_matches('/');
        format!("{}/{}", base, file_name)
    }
}

async fn build_r2_client(config: &crate::config::collateral::CollateralConfig) -> Result<S3Client> {
    let account_id = config
        .evidence_r2_account_id
        .as_ref()
        .context("evidence_r2_account_id is required for R2")?;
    let access_key_id = config
        .evidence_r2_access_key_id
        .as_ref()
        .context("evidence_r2_access_key_id is required for R2")?;
    let secret_access_key = config
        .evidence_r2_secret_access_key
        .as_ref()
        .context("evidence_r2_secret_access_key is required for R2")?;

    let endpoint = format!("https://{}.r2.cloudflarestorage.com", account_id);
    let credentials = aws_sdk_s3::config::Credentials::new(
        access_key_id,
        secret_access_key,
        None,
        None,
        "basilica-validator",
    );
    let shared_config = aws_config::defaults(BehaviorVersion::latest())
        .credentials_provider(credentials)
        .region(aws_sdk_s3::config::Region::new("auto"))
        .endpoint_url(endpoint)
        .load()
        .await;
    // TODO: Add support for bucket-level retention policies once R2 lifecycle rules are defined.
    Ok(S3Client::new(&shared_config))
}

#[cfg(test)]
mod tests {
    use super::*;
    use tempfile::tempdir;

    #[tokio::test]
    async fn test_store_evidence() {
        let temp = tempdir().unwrap();
        let store = EvidenceStore::new_local(
            "https://validator.example.com/evidence".to_string(),
            temp.path().to_path_buf(),
        );
        let evidence = SlashEvidence {
            rental_id: "rental-1".to_string(),
            misbehaviour_type: "deployment_failed".to_string(),
            timestamp: Utc::now(),
            details: "{}".to_string(),
            miner_hotkey: "hk".to_string(),
            node_id: "node".to_string(),
            validator_hotkey: "vhk".to_string(),
            shadow_mode: true,
            signature: None,
        };
        let (url, json) = store.store(&evidence).await.unwrap();
        assert!(url.contains("validator.example.com"));
        assert!(!json.is_empty());
    }
}
