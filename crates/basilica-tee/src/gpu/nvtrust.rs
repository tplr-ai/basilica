//! NvEvidence Provider
//!
//! Generates GPU attestation evidence using the chutes-nvevidence CLI tool.

use crate::config::GpuCcConfig;
use crate::error::{TeeError, TeeResult};
use std::path::Path;
use tokio::process::Command;
use tracing::{debug, error, info};

/// NVIDIA Evidence Provider
///
/// Wraps the chutes-nvevidence CLI tool to generate GPU attestation evidence.
pub struct NvEvidenceProvider {
    /// Path to the nvevidence binary
    binary_path: String,
    /// Output directory for evidence files
    output_dir: String,
}

impl NvEvidenceProvider {
    /// Create a new NvEvidenceProvider with default paths
    pub fn new() -> Self {
        Self {
            binary_path: "chutes-nvevidence".to_string(),
            output_dir: "/var/log/attestation-service".to_string(),
        }
    }

    /// Create a new NvEvidenceProvider from config
    pub fn from_config(config: &GpuCcConfig) -> Self {
        Self {
            binary_path: config.nvevidence_path.to_string_lossy().to_string(),
            output_dir: config.evidence_output_dir.to_string_lossy().to_string(),
        }
    }

    /// Create a new NvEvidenceProvider with custom paths
    pub fn with_config(binary_path: &str, output_dir: &str) -> Self {
        Self {
            binary_path: binary_path.to_string(),
            output_dir: output_dir.to_string(),
        }
    }

    /// Check if the nvevidence binary exists
    pub fn is_available(&self) -> bool {
        // Check if it's in PATH or at the specified path
        Path::new(&self.binary_path).exists()
            || std::process::Command::new("which")
                .arg(&self.binary_path)
                .output()
                .map(|o| o.status.success())
                .unwrap_or(false)
    }

    /// Get GPU attestation evidence
    ///
    /// # Arguments
    /// * `name` - Name of the node to include in evidence
    /// * `nonce` - Nonce to include in evidence (hex string)
    /// * `gpu_ids` - Optional list of GPU IDs to filter
    ///
    /// # Returns
    /// JSON string containing attestation evidence
    pub async fn get_evidence(
        &self,
        name: &str,
        nonce: &str,
        gpu_ids: Option<&[String]>,
    ) -> TeeResult<String> {
        info!(
            "[NvEvidence] Generating evidence for {} with nonce {}",
            name,
            &nonce[..8.min(nonce.len())]
        );

        // Ensure output directory exists
        if let Err(e) = tokio::fs::create_dir_all(&self.output_dir).await {
            debug!("Failed to create output dir (may already exist): {}", e);
        }

        let result = Command::new(&self.binary_path)
            .args(["--name", name, "--nonce", nonce])
            .current_dir(&self.output_dir)
            .output()
            .await
            .map_err(|e| TeeError::GpuAttestation(format!("Failed to execute nvevidence: {}", e)))?;

        if result.status.success() {
            let output_str = String::from_utf8_lossy(&result.stdout);

            // Get the last non-empty line (the JSON evidence)
            let evidence_json = output_str
                .lines()
                .filter(|line| !line.trim().is_empty())
                .last()
                .ok_or_else(|| TeeError::GpuAttestation("No output from evidence command".into()))?;

            info!("[NvEvidence] Successfully generated evidence");

            // Filter evidence if GPU IDs specified
            let filtered = self.filter_evidence(evidence_json, gpu_ids)?;
            Ok(filtered)
        } else {
            let stderr = String::from_utf8_lossy(&result.stderr);
            error!("[NvEvidence] Failed to gather GPU evidence: {}", stderr);
            Err(TeeError::GpuAttestation(format!(
                "Failed to gather GPU evidence: {}",
                stderr
            )))
        }
    }

    /// Filter evidence to only include specified GPU IDs
    fn filter_evidence(
        &self,
        evidence: &str,
        target_gpu_ids: Option<&[String]>,
    ) -> TeeResult<String> {
        if target_gpu_ids.is_none() {
            return Ok(evidence.to_string());
        }

        let target_ids = target_gpu_ids.unwrap();
        if target_ids.is_empty() {
            return Ok(evidence.to_string());
        }

        // Format target IDs
        let formatted_targets: Vec<String> = target_ids
            .iter()
            .map(|id| {
                if id.starts_with("GPU") {
                    id.clone()
                } else {
                    format!("GPU-{}", id)
                }
            })
            .collect();

        // Parse evidence as JSON array
        let evidence_list: serde_json::Value = serde_json::from_str(evidence)?;

        if let Some(arr) = evidence_list.as_array() {
            let filtered: Vec<&serde_json::Value> = arr
                .iter()
                .filter(|item| {
                    if let Some(gpu_id) = item.get("gpu_uuid").and_then(|v| v.as_str()) {
                        formatted_targets.iter().any(|t| gpu_id.contains(t))
                    } else {
                        false
                    }
                })
                .collect();

            Ok(serde_json::to_string(&filtered)?)
        } else {
            // Not an array, return as-is
            Ok(evidence.to_string())
        }
    }

    /// Get evidence with a random nonce
    pub async fn get_evidence_with_random_nonce(
        &self,
        name: &str,
        gpu_ids: Option<&[String]>,
    ) -> TeeResult<(String, [u8; 32])> {
        let nonce: [u8; 32] = rand::random();
        let nonce_hex = hex::encode(nonce);
        let evidence = self.get_evidence(name, &nonce_hex, gpu_ids).await?;
        Ok((evidence, nonce))
    }
}

impl Default for NvEvidenceProvider {
    fn default() -> Self {
        Self::new()
    }
}

/// Mock NvEvidence provider for testing
#[cfg(test)]
pub struct MockNvEvidenceProvider {
    evidence: String,
    should_fail: bool,
}

#[cfg(test)]
impl MockNvEvidenceProvider {
    pub fn new(evidence: String) -> Self {
        Self {
            evidence,
            should_fail: false,
        }
    }

    pub fn failing() -> Self {
        Self {
            evidence: String::new(),
            should_fail: true,
        }
    }

    pub fn with_sample_evidence() -> Self {
        let evidence = serde_json::json!([{
            "gpu_uuid": "GPU-abc123",
            "attestation_report": "deadbeef",
            "signature": "cafebabe",
            "driver_version": "555.0",
            "nonce": "test_nonce"
        }]);
        Self::new(evidence.to_string())
    }

    pub async fn get_evidence(
        &self,
        _name: &str,
        _nonce: &str,
        _gpu_ids: Option<&[String]>,
    ) -> TeeResult<String> {
        if self.should_fail {
            Err(TeeError::GpuAttestation("Mock failure".into()))
        } else {
            Ok(self.evidence.clone())
        }
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_provider_new() {
        let provider = NvEvidenceProvider::new();
        assert_eq!(provider.binary_path, "chutes-nvevidence");
    }

    #[test]
    fn test_provider_with_config() {
        let provider = NvEvidenceProvider::with_config("/custom/path", "/custom/output");
        assert_eq!(provider.binary_path, "/custom/path");
        assert_eq!(provider.output_dir, "/custom/output");
    }

    #[test]
    fn test_filter_evidence_no_filter() {
        let provider = NvEvidenceProvider::new();
        let evidence = r#"[{"gpu_uuid": "GPU-123"}]"#;

        let result = provider.filter_evidence(evidence, None).unwrap();
        assert_eq!(result, evidence);
    }

    #[test]
    fn test_filter_evidence_with_filter() {
        let provider = NvEvidenceProvider::new();
        let evidence = r#"[
            {"gpu_uuid": "GPU-123"},
            {"gpu_uuid": "GPU-456"}
        ]"#;

        let result = provider
            .filter_evidence(evidence, Some(&["GPU-123".to_string()]))
            .unwrap();

        let parsed: Vec<serde_json::Value> = serde_json::from_str(&result).unwrap();
        assert_eq!(parsed.len(), 1);
        assert_eq!(
            parsed[0]["gpu_uuid"].as_str().unwrap(),
            "GPU-123"
        );
    }

    #[test]
    fn test_filter_evidence_adds_gpu_prefix() {
        let provider = NvEvidenceProvider::new();
        let evidence = r#"[{"gpu_uuid": "GPU-abc"}]"#;

        // Should work even without GPU- prefix in filter
        let result = provider
            .filter_evidence(evidence, Some(&["abc".to_string()]))
            .unwrap();

        let parsed: Vec<serde_json::Value> = serde_json::from_str(&result).unwrap();
        assert_eq!(parsed.len(), 1);
    }

    #[tokio::test]
    async fn test_mock_provider_success() {
        let provider = MockNvEvidenceProvider::with_sample_evidence();
        let result = provider.get_evidence("test", "nonce", None).await.unwrap();

        let parsed: Vec<serde_json::Value> = serde_json::from_str(&result).unwrap();
        assert_eq!(parsed[0]["gpu_uuid"].as_str().unwrap(), "GPU-abc123");
    }

    #[tokio::test]
    async fn test_mock_provider_failure() {
        let provider = MockNvEvidenceProvider::failing();
        let result = provider.get_evidence("test", "nonce", None).await;

        assert!(result.is_err());
    }
}

