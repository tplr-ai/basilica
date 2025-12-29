//! NvEvidence Provider
//!
//! Generates GPU attestation evidence using NVIDIA's official attestation tools
//! or nvidia-smi for basic CC mode verification.

use async_trait::async_trait;
use std::path::Path;
use tokio::process::Command;
use tracing::{debug, error, info};

use crate::config::GpuCcConfig;
use crate::error::{TeeError, TeeResult};
use crate::gpu::utils::{gpu_id_contains_any, normalize_gpu_id};
use crate::traits::EvidenceProvider;

/// NVIDIA Evidence Provider
///
/// Wraps NVIDIA's official attestation tools to generate GPU attestation evidence.
/// Supports: nv-attestation-tool, nvidia-attestation, or falls back to nvidia-smi
/// for basic CC mode verification.
pub struct NvEvidenceProvider {
    /// Path to the attestation binary
    binary_path: String,
    /// Output directory for evidence files
    output_dir: String,
}

impl NvEvidenceProvider {
    /// Create a new NvEvidenceProvider with default paths
    /// Tries to find NVIDIA's official attestation tools
    pub fn new() -> Self {
        // Try to find an available attestation tool
        let binary_path = Self::find_attestation_tool().unwrap_or_else(|| "nvidia-smi".to_string());

        Self {
            binary_path,
            output_dir: "/var/log/attestation-service".to_string(),
        }
    }

    /// Find available NVIDIA attestation tool
    fn find_attestation_tool() -> Option<String> {
        let tools = [
            "nv-attestation-tool",
            "nvidia-attestation",
            "/usr/bin/nvidia-attestation",
            "/usr/local/bin/nv-attestation-tool",
        ];

        for tool in tools {
            if Path::new(tool).exists() {
                return Some(tool.to_string());
            }
            // Check if it's in PATH
            if std::process::Command::new("which")
                .arg(tool)
                .output()
                .map(|o| o.status.success())
                .unwrap_or(false)
            {
                return Some(tool.to_string());
            }
        }
        None
    }

    /// Check if Python SDK is available
    fn check_python_sdk_available() -> bool {
        std::process::Command::new("python3")
            .args(["-c", "import nv_attestation_sdk"])
            .output()
            .map(|o| o.status.success())
            .unwrap_or(false)
    }

    /// Get evidence using Python SDK
    async fn get_evidence_python_sdk(nonce: &str) -> TeeResult<String> {
        let script = format!(
            r#"
import json
from nv_attestation_sdk import attestation
nonce = bytes.fromhex('{}')
evidence = attestation.get_evidence(nonce)
print(json.dumps(evidence))
"#,
            nonce
        );

        let output = Command::new("python3")
            .args(["-c", &script])
            .output()
            .await
            .map_err(|e| TeeError::GpuAttestation(format!("Failed to run Python SDK: {}", e)))?;

        if output.status.success() {
            let stdout = String::from_utf8_lossy(&output.stdout);
            Ok(stdout.trim().to_string())
        } else {
            let stderr = String::from_utf8_lossy(&output.stderr);
            Err(TeeError::GpuAttestation(format!(
                "Python SDK failed: {}",
                stderr
            )))
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

        // Try native attestation tool first
        let native_result = self.try_native_attestation(name, nonce).await;

        let evidence_json = match native_result {
            Ok(json) => json,
            Err(e) => {
                debug!("[NvEvidence] Native tool failed ({}), trying Python SDK", e);

                // Try Python SDK as fallback
                if Self::check_python_sdk_available() {
                    info!("[NvEvidence] Trying Python SDK fallback");
                    Self::get_evidence_python_sdk(nonce).await?
                } else {
                    return Err(e);
                }
            }
        };

        info!("[NvEvidence] Successfully generated evidence");

        // Filter evidence if GPU IDs specified
        let filtered = self.filter_evidence(&evidence_json, gpu_ids)?;
        Ok(filtered)
    }

    /// Try to get evidence using native attestation tool
    async fn try_native_attestation(&self, name: &str, nonce: &str) -> TeeResult<String> {
        let result = Command::new(&self.binary_path)
            .args(["--name", name, "--nonce", nonce])
            .current_dir(&self.output_dir)
            .output()
            .await
            .map_err(|e| {
                TeeError::GpuAttestation(format!("Failed to execute {}: {}", self.binary_path, e))
            })?;

        if result.status.success() {
            let output_str = String::from_utf8_lossy(&result.stdout);

            // Get the last non-empty line (the JSON evidence)
            let evidence_json = output_str
                .lines()
                .filter(|line| !line.trim().is_empty())
                .next_back()
                .ok_or_else(|| {
                    TeeError::GpuAttestation("No output from evidence command".into())
                })?;

            Ok(evidence_json.to_string())
        } else {
            let stderr = String::from_utf8_lossy(&result.stderr);
            error!("[NvEvidence] Native tool failed: {}", stderr);
            Err(TeeError::GpuAttestation(format!(
                "Native attestation tool failed: {}",
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

        // Format target IDs with GPU- prefix
        let formatted_targets: Vec<String> =
            target_ids.iter().map(|id| normalize_gpu_id(id)).collect();

        // Parse evidence as JSON array
        let evidence_list: serde_json::Value = serde_json::from_str(evidence)?;

        if let Some(arr) = evidence_list.as_array() {
            let filtered: Vec<&serde_json::Value> = arr
                .iter()
                .filter(|item| {
                    if let Some(gpu_id) = item.get("gpu_uuid").and_then(|v| v.as_str()) {
                        gpu_id_contains_any(gpu_id, &formatted_targets)
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

#[async_trait]
impl EvidenceProvider for NvEvidenceProvider {
    async fn generate_evidence(
        &self,
        name: &str,
        nonce: &str,
        gpu_ids: Option<&[String]>,
    ) -> TeeResult<String> {
        self.get_evidence(name, nonce, gpu_ids).await
    }

    fn is_available(&self) -> bool {
        NvEvidenceProvider::is_available(self)
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
        // Auto-detects available tool, falls back to nvidia-smi
        assert!(!provider.binary_path.is_empty());
        // Should be one of the known tools or nvidia-smi
        let valid_tools = ["nvidia-smi", "nv-attestation-tool", "nvidia-attestation"];
        assert!(
            valid_tools.iter().any(|t| provider.binary_path.contains(t)),
            "Expected valid attestation tool, got: {}",
            provider.binary_path
        );
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
        assert_eq!(parsed[0]["gpu_uuid"].as_str().unwrap(), "GPU-123");
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
