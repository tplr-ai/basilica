//! NVIDIA GPU Remote Attestation Verification
//!
//! Provides integration with NVIDIA's Remote Attestation Service (NRAS)
//! for verifying GPU confidential computing attestation.

use crate::error::{TeeError, TeeResult};
use crate::types::{GpuAttestationEvidence, GpuCcVerificationResult};
use serde::Deserialize;
#[cfg(feature = "remote-attestation")]
use serde::Serialize;
use std::time::Duration;
#[cfg(feature = "remote-attestation")]
use tracing::{info, warn};

/// NVIDIA Remote Attestation Service configuration
#[derive(Debug, Clone)]
pub struct NrasConfig {
    /// URL of NRAS service
    pub service_url: String,
    /// Request timeout
    pub timeout: Duration,
    /// API key for NRAS (if required)
    pub api_key: Option<String>,
    /// Whether to verify TLS certificates
    pub verify_tls: bool,
}

impl Default for NrasConfig {
    fn default() -> Self {
        Self {
            // NVIDIA's attestation service
            service_url: "https://nras.attestation.nvidia.com/v1/attest".into(),
            timeout: Duration::from_secs(30),
            api_key: None,
            verify_tls: true,
        }
    }
}

/// Request body for NRAS verification
#[cfg(feature = "remote-attestation")]
#[derive(Debug, Serialize)]
pub struct NrasVerificationRequest {
    /// Base64-encoded attestation evidence
    pub evidence: String,
    /// Nonce for freshness verification
    pub nonce: Option<String>,
    /// GPU UUID
    pub gpu_uuid: String,
}

/// Response from NRAS verification
#[derive(Debug, Deserialize)]
#[allow(dead_code)]
pub struct NrasVerificationResponse {
    /// Overall verification status
    pub status: String,
    /// Detailed verification result
    pub result: NrasResult,
    /// Timestamp of verification
    pub timestamp: Option<String>,
}

/// Detailed NRAS verification result
#[derive(Debug, Deserialize)]
#[allow(dead_code)]
pub struct NrasResult {
    /// Whether attestation is valid
    pub attestation_valid: bool,
    /// GPU is in CC mode
    pub cc_mode_verified: bool,
    /// Firmware version verified
    pub firmware_valid: bool,
    /// Security advisory status
    pub security_status: String,
    /// Any error message
    pub error: Option<String>,
}

/// NVIDIA GPU Remote Attestation Verifier
pub struct RemoteGpuVerifier {
    config: NrasConfig,
}

impl RemoteGpuVerifier {
    /// Create a new remote verifier with the given configuration
    pub fn new(config: NrasConfig) -> Self {
        Self { config }
    }

    /// Create a verifier with NVIDIA's default NRAS endpoint
    pub fn nvidia_nras() -> Self {
        Self::new(NrasConfig::default())
    }

    /// Verify GPU attestation evidence using NRAS
    #[cfg(feature = "remote-attestation")]
    pub async fn verify(
        &self,
        evidence: &GpuAttestationEvidence,
        nonce: Option<&str>,
    ) -> TeeResult<GpuCcVerificationResult> {
        use base64::Engine;

        info!(
            "[GPU Remote] Sending attestation for GPU {} to NRAS",
            evidence.gpu_uuid
        );

        let evidence_base64 =
            base64::engine::general_purpose::STANDARD.encode(&evidence.attestation_report);

        let request = NrasVerificationRequest {
            evidence: evidence_base64,
            nonce: nonce.map(String::from),
            gpu_uuid: evidence.gpu_uuid.clone(),
        };

        // Build HTTP client
        let client = reqwest::Client::builder()
            .timeout(self.config.timeout)
            .danger_accept_invalid_certs(!self.config.verify_tls)
            .build()
            .map_err(|e| TeeError::GpuAttestation(format!("Failed to build HTTP client: {}", e)))?;

        // Make request
        let mut req = client.post(&self.config.service_url).json(&request);

        if let Some(ref api_key) = self.config.api_key {
            req = req.header("X-API-Key", api_key);
        }

        let response = req.send().await.map_err(|e| {
            TeeError::GpuAttestation(format!("Remote verification request failed: {}", e))
        })?;

        if !response.status().is_success() {
            let status = response.status();
            let body = response.text().await.unwrap_or_default();
            return Err(TeeError::GpuAttestation(format!(
                "Remote verification failed with status {}: {}",
                status, body
            )));
        }

        let verification: NrasVerificationResponse = response.json().await.map_err(|e| {
            TeeError::GpuAttestation(format!("Failed to parse verification response: {}", e))
        })?;

        if verification.status != "success" {
            warn!(
                "[GPU Remote] Verification failed: status={}, error={:?}",
                verification.status, verification.result.error
            );
        } else {
            info!(
                "[GPU Remote] Verification succeeded for GPU {}",
                evidence.gpu_uuid
            );
        }

        // Check nonce locally
        let nonce_verified = nonce.is_none_or(|n| evidence.nonce == n);

        Ok(GpuCcVerificationResult {
            cc_mode_enabled: verification.result.cc_mode_verified,
            attestation_valid: verification.result.attestation_valid,
            gpu_uuid: evidence.gpu_uuid.clone(),
            nonce_verified,
            gpu_model: evidence.gpu_model.clone(),
            driver_version: evidence.driver_version.clone(),
            verified_at: chrono::Utc::now(),
        })
    }

    /// Stub verification for when remote-attestation feature is not enabled
    #[cfg(not(feature = "remote-attestation"))]
    pub async fn verify(
        &self,
        _evidence: &GpuAttestationEvidence,
        _nonce: Option<&str>,
    ) -> TeeResult<GpuCcVerificationResult> {
        Err(TeeError::GpuAttestation(
            "Remote attestation feature not enabled. Compile with --features remote-attestation"
                .into(),
        ))
    }

    /// Verify multiple GPUs
    #[cfg(feature = "remote-attestation")]
    pub async fn verify_all(
        &self,
        evidence_list: &[GpuAttestationEvidence],
        nonce: Option<&str>,
    ) -> TeeResult<Vec<GpuCcVerificationResult>> {
        let mut results = Vec::new();
        for evidence in evidence_list {
            results.push(self.verify(evidence, nonce).await?);
        }
        Ok(results)
    }

    /// Get the configured service URL
    pub fn service_url(&self) -> &str {
        &self.config.service_url
    }
}

/// GPU attestation verification status
#[derive(Debug, Clone, PartialEq, Eq)]
pub enum GpuVerificationStatus {
    /// Attestation is valid and GPU is trusted
    Valid,
    /// Attestation valid but firmware needs update
    FirmwareOutOfDate,
    /// Attestation failed
    Invalid,
    /// Security advisory applies
    SecurityAdvisory,
    /// GPU is not in CC mode
    NotInCcMode,
    /// Unknown status
    Unknown(String),
}

impl From<&str> for GpuVerificationStatus {
    fn from(s: &str) -> Self {
        match s {
            "valid" | "success" => Self::Valid,
            "firmware_out_of_date" => Self::FirmwareOutOfDate,
            "invalid" | "failed" => Self::Invalid,
            "security_advisory" => Self::SecurityAdvisory,
            "not_cc_mode" => Self::NotInCcMode,
            other => Self::Unknown(other.to_string()),
        }
    }
}

impl GpuVerificationStatus {
    /// Check if the GPU is acceptable for confidential workloads
    pub fn is_acceptable(&self) -> bool {
        matches!(self, Self::Valid | Self::FirmwareOutOfDate)
    }

    /// Check if the GPU is fully trusted
    pub fn is_fully_trusted(&self) -> bool {
        matches!(self, Self::Valid)
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_gpu_verification_status_from_str() {
        assert_eq!(
            GpuVerificationStatus::from("valid"),
            GpuVerificationStatus::Valid
        );
        assert_eq!(
            GpuVerificationStatus::from("success"),
            GpuVerificationStatus::Valid
        );
        assert_eq!(
            GpuVerificationStatus::from("invalid"),
            GpuVerificationStatus::Invalid
        );
        assert!(matches!(
            GpuVerificationStatus::from("unknown_status"),
            GpuVerificationStatus::Unknown(_)
        ));
    }

    #[test]
    fn test_gpu_verification_status_acceptability() {
        assert!(GpuVerificationStatus::Valid.is_acceptable());
        assert!(GpuVerificationStatus::FirmwareOutOfDate.is_acceptable());
        assert!(!GpuVerificationStatus::Invalid.is_acceptable());
        assert!(!GpuVerificationStatus::NotInCcMode.is_acceptable());
    }

    #[test]
    fn test_default_config() {
        let config = NrasConfig::default();
        assert!(config.service_url.contains("nvidia.com"));
        assert_eq!(config.timeout, Duration::from_secs(30));
        assert!(config.verify_tls);
    }
}
