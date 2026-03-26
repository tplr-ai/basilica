use anyhow::Result;
use chrono::Duration;
use rust_decimal::Decimal;
use serde::{Deserialize, Serialize};
use std::collections::HashMap;
use std::path::PathBuf;

#[derive(Debug, Clone, Serialize, Deserialize)]
#[serde(rename_all = "snake_case")]
pub enum TrusteeKeySource {
    File,
    AwsSecrets,
    EnvVar,
}

#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct CollateralConfig {
    #[serde(default = "default_shadow_mode")]
    pub shadow_mode: bool,
    #[serde(default = "default_warning_threshold_multiplier")]
    pub warning_threshold_multiplier: Decimal,
    #[serde(default = "default_grace_period_hours")]
    pub grace_period_hours: u64,
    #[serde(default = "default_exclude_on_prolonged_price_failure")]
    pub exclude_on_prolonged_price_failure: bool,
    #[serde(default = "default_minimum_usd_per_gpu")]
    pub minimum_usd_per_gpu: HashMap<String, Decimal>,
    pub contract_address: String,
    #[serde(default = "default_collateral_network")]
    pub network: String,
    #[serde(default = "default_slash_fraction")]
    pub slash_fraction: Decimal,
    #[serde(default = "default_slash_cooldown_secs")]
    pub slash_cooldown_secs: u64,
    #[serde(default = "default_slash_max_per_window")]
    pub slash_max_per_window: u64,
    #[serde(default = "default_slash_window_secs")]
    pub slash_window_secs: u64,
    #[serde(default = "default_slash_circuit_breaker_threshold")]
    pub slash_circuit_breaker_threshold: u64,
    #[serde(default = "default_slash_circuit_breaker_window_secs")]
    pub slash_circuit_breaker_window_secs: u64,
    #[serde(default = "default_slash_circuit_breaker_cooldown_secs")]
    pub slash_circuit_breaker_cooldown_secs: u64,
    #[serde(default)]
    pub trustee_private_key_file: Option<PathBuf>,
    #[serde(default = "default_trustee_key_source")]
    pub trustee_key_source: TrusteeKeySource,
    #[serde(default)]
    pub aws_secret_name: Option<String>,
    #[serde(default)]
    pub aws_region: Option<String>,
    #[serde(default = "default_evidence_base_url")]
    pub evidence_base_url: String,
    #[serde(default = "default_evidence_storage_path")]
    pub evidence_storage_path: PathBuf,
    #[serde(default)]
    pub evidence_r2_account_id: Option<String>,
    #[serde(default)]
    pub evidence_r2_access_key_id: Option<String>,
    #[serde(default)]
    pub evidence_r2_secret_access_key: Option<String>,
    #[serde(default)]
    pub evidence_r2_bucket: Option<String>,
    #[serde(default)]
    pub evidence_public_url_base: Option<String>,
}

impl Default for CollateralConfig {
    fn default() -> Self {
        Self {
            shadow_mode: default_shadow_mode(),
            warning_threshold_multiplier: default_warning_threshold_multiplier(),
            grace_period_hours: default_grace_period_hours(),
            exclude_on_prolonged_price_failure: default_exclude_on_prolonged_price_failure(),
            minimum_usd_per_gpu: default_minimum_usd_per_gpu(),
            contract_address: String::new(),
            network: default_collateral_network(),
            slash_fraction: default_slash_fraction(),
            slash_cooldown_secs: default_slash_cooldown_secs(),
            slash_max_per_window: default_slash_max_per_window(),
            slash_window_secs: default_slash_window_secs(),
            slash_circuit_breaker_threshold: default_slash_circuit_breaker_threshold(),
            slash_circuit_breaker_window_secs: default_slash_circuit_breaker_window_secs(),
            slash_circuit_breaker_cooldown_secs: default_slash_circuit_breaker_cooldown_secs(),
            trustee_private_key_file: None,
            trustee_key_source: default_trustee_key_source(),
            aws_secret_name: None,
            aws_region: None,
            evidence_base_url: default_evidence_base_url(),
            evidence_storage_path: default_evidence_storage_path(),
            evidence_r2_account_id: None,
            evidence_r2_access_key_id: None,
            evidence_r2_secret_access_key: None,
            evidence_r2_bucket: None,
            evidence_public_url_base: None,
        }
    }
}

impl CollateralConfig {
    pub fn grace_period(&self) -> Duration {
        Duration::hours(self.grace_period_hours as i64)
    }

    pub fn validate(&self) -> Result<()> {
        if self.contract_address.trim().is_empty() {
            anyhow::bail!("collateral.contract_address is required when collateral is configured");
        }
        if self.warning_threshold_multiplier < Decimal::ONE {
            anyhow::bail!("collateral.warning_threshold_multiplier must be >= 1.0");
        }
        if self.grace_period_hours == 0 {
            anyhow::bail!("collateral.grace_period_hours must be > 0");
        }
        if self.minimum_usd_per_gpu.is_empty() {
            anyhow::bail!("collateral.minimum_usd_per_gpu cannot be empty");
        }
        if !(Decimal::ZERO < self.slash_fraction && self.slash_fraction <= Decimal::ONE) {
            anyhow::bail!("collateral.slash_fraction must be within (0.0, 1.0]");
        }
        if self.slash_cooldown_secs == 0 {
            anyhow::bail!("collateral.slash_cooldown_secs must be > 0");
        }
        if self.slash_max_per_window == 0 {
            anyhow::bail!("collateral.slash_max_per_window must be > 0");
        }
        if self.slash_window_secs == 0 {
            anyhow::bail!("collateral.slash_window_secs must be > 0");
        }
        if self.slash_circuit_breaker_threshold == 0 {
            anyhow::bail!("collateral.slash_circuit_breaker_threshold must be > 0");
        }
        if self.slash_circuit_breaker_window_secs == 0 {
            anyhow::bail!("collateral.slash_circuit_breaker_window_secs must be > 0");
        }
        if self.slash_circuit_breaker_cooldown_secs == 0 {
            anyhow::bail!("collateral.slash_circuit_breaker_cooldown_secs must be > 0");
        }
        if !self.shadow_mode {
            let missing = [
                (
                    "collateral.evidence_r2_account_id",
                    self.evidence_r2_account_id.as_ref(),
                ),
                (
                    "collateral.evidence_r2_access_key_id",
                    self.evidence_r2_access_key_id.as_ref(),
                ),
                (
                    "collateral.evidence_r2_secret_access_key",
                    self.evidence_r2_secret_access_key.as_ref(),
                ),
                (
                    "collateral.evidence_r2_bucket",
                    self.evidence_r2_bucket.as_ref(),
                ),
                (
                    "collateral.evidence_public_url_base",
                    self.evidence_public_url_base.as_ref(),
                ),
            ]
            .into_iter()
            .filter(|(_, value)| value.map(|v| v.trim().is_empty()).unwrap_or(true))
            .map(|(name, _)| name)
            .collect::<Vec<_>>();

            if !missing.is_empty() {
                anyhow::bail!(
                    "Missing R2 evidence config when shadow_mode=false: {}",
                    missing.join(", ")
                );
            }
        }
        if !self.shadow_mode {
            match self.trustee_key_source {
                TrusteeKeySource::File => {
                    if self.trustee_private_key_file.is_none() {
                        anyhow::bail!(
                            "collateral.trustee_private_key_file is required when shadow_mode=false"
                        );
                    }
                }
                TrusteeKeySource::AwsSecrets => {
                    let name = self
                        .aws_secret_name
                        .as_ref()
                        .map(|value| value.trim())
                        .unwrap_or("");
                    if name.is_empty() {
                        anyhow::bail!(
                            "collateral.aws_secret_name is required when trustee_key_source=aws_secrets"
                        );
                    }
                }
                TrusteeKeySource::EnvVar => {}
            }
        }
        Ok(())
    }
}

fn default_shadow_mode() -> bool {
    false
}

fn default_trustee_key_source() -> TrusteeKeySource {
    TrusteeKeySource::File
}

fn default_warning_threshold_multiplier() -> Decimal {
    Decimal::new(15, 1)
}

fn default_grace_period_hours() -> u64 {
    24
}

fn default_exclude_on_prolonged_price_failure() -> bool {
    true
}

fn default_minimum_usd_per_gpu() -> HashMap<String, Decimal> {
    let mut map = HashMap::new();
    map.insert("H100".to_string(), Decimal::from(50));
    map.insert("A100".to_string(), Decimal::from(25));
    map.insert("H200".to_string(), Decimal::from(75));
    map.insert("B200".to_string(), Decimal::from(75));
    map.insert("DEFAULT".to_string(), Decimal::from(10));
    map
}

fn default_collateral_network() -> String {
    "mainnet".to_string()
}

fn default_slash_fraction() -> Decimal {
    Decimal::ONE
}

fn default_slash_cooldown_secs() -> u64 {
    3600
}

fn default_slash_max_per_window() -> u64 {
    20
}

fn default_slash_window_secs() -> u64 {
    3600
}

fn default_slash_circuit_breaker_threshold() -> u64 {
    10
}

fn default_slash_circuit_breaker_window_secs() -> u64 {
    300
}

fn default_slash_circuit_breaker_cooldown_secs() -> u64 {
    1800
}

fn default_evidence_base_url() -> String {
    "https://validator.example.com/evidence".to_string()
}

fn default_evidence_storage_path() -> PathBuf {
    PathBuf::from("./evidence")
}
