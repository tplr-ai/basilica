//! Network Speed Test Validation Module
//!
//! This module handles network speed testing for nodes using a bash script
//! to measure download and upload speeds. It collects network performance metrics
//! and stores them for validation scoring purposes.

use crate::persistence::SimplePersistence;
use crate::ssh::ValidatorSshClient;
use anyhow::Result;
use basilica_common::ssh::SshConnectionDetails;
use serde::{Deserialize, Serialize};
use std::sync::Arc;
use tracing::{info, warn};

#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct NetworkSpeedProfile {
    pub download_mbps: Option<f64>,
    pub upload_mbps: Option<f64>,
    pub test_timestamp: chrono::DateTime<chrono::Utc>,
    pub test_server: Option<String>,
    pub full_json: String,
}

impl NetworkSpeedProfile {
    /// Parse speedtest script output
    pub fn from_speedtest_output(output: &str) -> Result<Self> {
        let now = chrono::Utc::now();
        let mut download_mbps: Option<f64> = None;
        let mut upload_mbps: Option<f64> = None;

        for line in output.lines() {
            let line = line.trim();

            if line.starts_with("Download:") {
                let speed_part = line
                    .strip_prefix("Download:")
                    .map(|s| s.trim())
                    .unwrap_or("");

                if let Some(mbps_pos) = speed_part.to_lowercase().find("mbps") {
                    let numeric_part = &speed_part[..mbps_pos].trim();
                    download_mbps = numeric_part.parse::<f64>().ok();
                } else if speed_part.ends_with("Mbps") || speed_part.ends_with("mbps") {
                    let numeric_part = speed_part
                        .trim_end_matches("Mbps")
                        .trim_end_matches("mbps")
                        .trim();
                    download_mbps = numeric_part.parse::<f64>().ok();
                }
            } else if line.starts_with("Upload:") {
                let speed_part = line.strip_prefix("Upload:").map(|s| s.trim()).unwrap_or("");

                if let Some(mbps_pos) = speed_part.to_lowercase().find("mbps") {
                    let numeric_part = &speed_part[..mbps_pos].trim();
                    upload_mbps = numeric_part.parse::<f64>().ok();
                } else if speed_part.ends_with("Mbps") || speed_part.ends_with("mbps") {
                    let numeric_part = speed_part
                        .trim_end_matches("Mbps")
                        .trim_end_matches("mbps")
                        .trim();
                    upload_mbps = numeric_part.parse::<f64>().ok();
                }
            }
        }

        if let Some(dl) = download_mbps {
            if !(0.0..=100000.0).contains(&dl) {
                download_mbps = None;
            }
        }
        if let Some(ul) = upload_mbps {
            if !(0.0..=100000.0).contains(&ul) {
                upload_mbps = None;
            }
        }

        let result_json = serde_json::json!({
            "download_mbps": download_mbps,
            "upload_mbps": upload_mbps,
            "test_timestamp": now.to_rfc3339(),
            "test_server": "cloudflare",
            "raw_output": output,
        });

        Ok(NetworkSpeedProfile {
            download_mbps,
            upload_mbps,
            test_timestamp: now,
            test_server: Some("cloudflare".to_string()),
            full_json: result_json.to_string(),
        })
    }
}

/// Network speed test collector for gathering network performance from nodes
#[derive(Clone)]
pub struct NetworkSpeedCollector {
    ssh_client: Arc<ValidatorSshClient>,
    persistence: Arc<SimplePersistence>,
}

impl NetworkSpeedCollector {
    /// Create a new network speed collector
    pub fn new(ssh_client: Arc<ValidatorSshClient>, persistence: Arc<SimplePersistence>) -> Self {
        Self {
            ssh_client,
            persistence,
        }
    }

    /// Collect network speed profile from node
    pub async fn collect(
        &self,
        node_id: &str,
        ssh_details: &SshConnectionDetails,
    ) -> Result<NetworkSpeedProfile> {
        info!(
            node_id = node_id,
            "[SPEEDTEST] Starting network speed test collection"
        );

        let speedtest_script = include_str!("scripts/speedtest.sh");

        let script_path = "/tmp/speedtest_basilica.sh";
        let create_script_cmd =
            format!("cat > {} << 'EOF'\n{}\nEOF", script_path, speedtest_script);

        self.ssh_client
            .execute_command(ssh_details, &create_script_cmd, false)
            .await?;

        self.ssh_client
            .execute_command(ssh_details, &format!("chmod +x {}", script_path), false)
            .await?;

        self.ssh_client
            .ensure_installed(ssh_details, "curl", "curl")
            .await?;

        let speedtest_output = self
            .ssh_client
            .execute_command(
                ssh_details,
                &format!("timeout 60 bash {}", script_path),
                true,
            )
            .await?;

        let _ = self
            .ssh_client
            .execute_command(ssh_details, &format!("rm -f {}", script_path), false)
            .await;

        let speed_profile = NetworkSpeedProfile::from_speedtest_output(&speedtest_output)?;

        info!(
            node_id = node_id,
            download_mbps = speed_profile.download_mbps.unwrap_or(0.0),
            upload_mbps = speed_profile.upload_mbps.unwrap_or(0.0),
            "[SPEEDTEST] Successfully collected network speed profile"
        );

        Ok(speed_profile)
    }

    /// Store network speed profile in database
    pub async fn store(
        &self,
        miner_uid: u16,
        node_id: &str,
        speed_profile: &NetworkSpeedProfile,
    ) -> Result<()> {
        self.persistence
            .store_node_speedtest_profile(
                miner_uid,
                node_id,
                speed_profile.download_mbps,
                speed_profile.upload_mbps,
                &speed_profile.test_timestamp.to_rfc3339(),
                speed_profile.test_server.clone(),
                &speed_profile.full_json,
            )
            .await?;

        info!(
            miner_uid = miner_uid,
            node_id = node_id,
            "[SPEEDTEST] Stored network speed profile in database"
        );

        Ok(())
    }

    /// Collect network speed profile from node and store in database
    pub async fn collect_and_store(
        &self,
        node_id: &str,
        miner_uid: u16,
        ssh_details: &SshConnectionDetails,
    ) -> Result<NetworkSpeedProfile> {
        let speed_profile = self.collect(node_id, ssh_details).await?;
        self.store(miner_uid, node_id, &speed_profile).await?;
        Ok(speed_profile)
    }

    /// Collect network speed profile with error handling (non-critical operation)
    pub async fn collect_with_fallback(
        &self,
        node_id: &str,
        miner_uid: u16,
        ssh_details: &SshConnectionDetails,
    ) -> Option<NetworkSpeedProfile> {
        match self.collect(node_id, ssh_details).await {
            Ok(profile) => {
                if let Err(e) = self.store(miner_uid, node_id, &profile).await {
                    warn!(
                        miner_uid = miner_uid,
                        node_id = node_id,
                        error = %e,
                        "[SPEEDTEST] Failed to store network speed profile (non-critical)"
                    );
                }
                Some(profile)
            }
            Err(e) => {
                warn!(
                    miner_uid = miner_uid,
                    node_id = node_id,
                    error = %e,
                    "[SPEEDTEST] Failed to collect network speed profile (non-critical)"
                );
                None
            }
        }
    }

    /// Retrieve network speed profile from database
    pub async fn retrieve(
        &self,
        miner_uid: u16,
        node_id: &str,
    ) -> Result<Option<NetworkSpeedProfile>> {
        let result = self
            .persistence
            .get_node_speedtest_profile(miner_uid, node_id)
            .await?;

        match result {
            Some((full_json, download_mbps, upload_mbps, test_timestamp, test_server)) => {
                let test_timestamp = chrono::DateTime::parse_from_rfc3339(&test_timestamp)
                    .map_err(|e| anyhow::anyhow!("Failed to parse timestamp: {}", e))?
                    .with_timezone(&chrono::Utc);

                let profile = NetworkSpeedProfile {
                    download_mbps,
                    upload_mbps,
                    test_timestamp,
                    test_server,
                    full_json,
                };

                info!(
                    miner_uid = miner_uid,
                    node_id = node_id,
                    "[SPEEDTEST] Retrieved network speed profile from database"
                );

                Ok(Some(profile))
            }
            None => {
                info!(
                    miner_uid = miner_uid,
                    node_id = node_id,
                    "[SPEEDTEST] No network speed profile found in database"
                );
                Ok(None)
            }
        }
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_parse_speedtest_output() {
        let test_output = r#"Running download test (50MB)...
Running upload test (15MB)...
Download: 125.45 Mbps
Upload: 67.89 Mbps"#;

        let result = NetworkSpeedProfile::from_speedtest_output(test_output);
        assert!(result.is_ok());

        let profile = result.unwrap();
        assert_eq!(profile.download_mbps, Some(125.45));
        assert_eq!(profile.upload_mbps, Some(67.89));
        assert_eq!(profile.test_server, Some("cloudflare".to_string()));
        assert!(profile.full_json.contains("125.45"));
        assert!(profile.full_json.contains("67.89"));
    }

    #[test]
    fn test_parse_speedtest_output_partial() {
        let test_output = r#"Running download test (50MB)...
Download: 200.00 Mbps
Upload test failed or timed out
Upload: 0.0 Mbps"#;

        let result = NetworkSpeedProfile::from_speedtest_output(test_output);
        assert!(result.is_ok());

        let profile = result.unwrap();
        assert_eq!(profile.download_mbps, Some(200.00));
        assert_eq!(profile.upload_mbps, Some(0.0));
    }

    #[test]
    fn test_parse_speedtest_output_malformed() {
        let test_output = "Some random output without speed information";

        let result = NetworkSpeedProfile::from_speedtest_output(test_output);
        assert!(result.is_ok());

        let profile = result.unwrap();
        assert_eq!(profile.download_mbps, None);
        assert_eq!(profile.upload_mbps, None);
        assert!(profile.full_json.contains("raw_output"));
    }
}
