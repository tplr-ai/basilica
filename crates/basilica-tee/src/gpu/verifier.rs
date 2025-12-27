//! GPU Evidence Verification
//!
//! Verifies GPU attestation evidence for Confidential Computing.
//!
//! Provides both local verification (stub) and remote verification via
//! NVIDIA Remote Attestation Service (NRAS).

use async_trait::async_trait;
use serde::{Deserialize, Serialize};
use tracing::{debug, error, info, warn};

use crate::error::{TeeError, TeeResult};
use crate::traits::GpuVerifier;
use crate::types::{GpuAttestationEvidence, GpuCcVerificationResult};

/// Local GPU evidence verifier.
///
/// Performs local verification of GPU attestation evidence.
/// Note: This is currently a stub implementation that validates
/// basic evidence structure but does not perform cryptographic
/// verification of signatures.
///
/// For production use, consider:
/// - Using NVIDIA Remote Attestation Service (NRAS) via `RemoteGpuVerifier`
/// - Implementing local verification using RIM files and signature chains
#[derive(Debug, Clone, Default)]
pub struct LocalGpuVerifier;

impl LocalGpuVerifier {
    /// Create a new local GPU verifier.
    pub fn new() -> Self {
        Self
    }

    /// Verify attestation signature.
    ///
    /// Note: This is a stub implementation that always returns true.
    /// In production, this should verify:
    /// 1. The attestation report signature
    /// 2. The certificate chain back to NVIDIA's root
    /// 3. The report contents match expected format
    fn verify_signature(&self, _evidence: &GpuAttestationEvidence) -> TeeResult<bool> {
        debug!("[GpuVerifier] Signature verification stub - returning true");
        debug!("[GpuVerifier] NOTE: Implement actual verification for production use");

        Ok(true)
    }

    /// Check if CC mode is enabled based on evidence.
    ///
    /// Currently checks if the attestation report is non-empty.
    fn check_cc_mode(&self, evidence: &GpuAttestationEvidence) -> bool {
        !evidence.attestation_report.is_empty()
    }

    /// Verify the nonce matches expected value.
    fn verify_nonce(
        &self,
        evidence: &GpuAttestationEvidence,
        expected_nonce: Option<&str>,
    ) -> bool {
        if let Some(expected) = expected_nonce {
            let matches = evidence.nonce == expected;
            if !matches {
                warn!(
                    "[GpuVerifier] Nonce mismatch: expected {}, got {}",
                    expected, evidence.nonce
                );
            }
            matches
        } else {
            true
        }
    }
}

#[async_trait]
impl GpuVerifier for LocalGpuVerifier {
    async fn verify(
        &self,
        evidence: &GpuAttestationEvidence,
        expected_nonce: Option<&str>,
    ) -> TeeResult<GpuCcVerificationResult> {
        debug!(
            "[GpuVerifier] Verifying evidence for GPU {}",
            evidence.gpu_uuid
        );

        let nonce_verified = self.verify_nonce(evidence, expected_nonce);
        let attestation_valid = self.verify_signature(evidence)?;
        let cc_mode_enabled = self.check_cc_mode(evidence);

        Ok(GpuCcVerificationResult {
            cc_mode_enabled,
            attestation_valid,
            gpu_uuid: evidence.gpu_uuid.clone(),
            nonce_verified,
            gpu_model: evidence.gpu_model.clone(),
            driver_version: evidence.driver_version.clone(),
            verified_at: chrono::Utc::now(),
        })
    }
}

/// Convenience function to verify evidence using the default verifier.
pub async fn verify_evidence(
    evidence: &GpuAttestationEvidence,
    expected_nonce: Option<&str>,
) -> TeeResult<GpuCcVerificationResult> {
    LocalGpuVerifier::new()
        .verify(evidence, expected_nonce)
        .await
}

/// Verify multiple evidence entries using the default verifier.
pub async fn verify_all_evidence(
    evidence_list: &[GpuAttestationEvidence],
    expected_nonce: Option<&str>,
) -> TeeResult<Vec<GpuCcVerificationResult>> {
    LocalGpuVerifier::new()
        .verify_all(evidence_list, expected_nonce)
        .await
}

// ============================================================================
// NVIDIA Remote Attestation Service (NRAS) Verifier
// ============================================================================

/// Configuration for NVIDIA Remote Attestation Service
#[derive(Debug, Clone)]
pub struct NrasConfig {
    /// NRAS API endpoint URL
    pub api_url: String,
    /// API key for authentication (if required)
    pub api_key: Option<String>,
    /// Request timeout in seconds
    pub timeout_secs: u64,
}

impl Default for NrasConfig {
    fn default() -> Self {
        Self {
            // NVIDIA's attestation verification service
            // Note: This URL may change - check NVIDIA's documentation
            api_url: "https://nras.attestation.nvidia.com/v1/attest".to_string(),
            api_key: None,
            timeout_secs: 30,
        }
    }
}

/// Request body for NRAS verification
#[derive(Debug, Serialize)]
struct NrasVerifyRequest {
    /// Base64-encoded attestation evidence (EAT token)
    evidence: String,
    /// Optional nonce for freshness verification
    #[serde(skip_serializing_if = "Option::is_none")]
    nonce: Option<String>,
}

/// Response from NRAS verification
#[derive(Debug, Deserialize)]
struct NrasVerifyResponse {
    /// Verification result
    #[serde(default)]
    result: String,
    /// Whether attestation is valid
    #[serde(default)]
    attested: bool,
    /// Error message if verification failed
    #[serde(default)]
    error: Option<String>,
    /// Detailed claims from the attestation (optional)
    #[serde(default)]
    claims: Option<NrasClaims>,
}

/// Claims extracted from the attestation
#[derive(Debug, Deserialize, Default)]
#[allow(dead_code)]
struct NrasClaims {
    /// GPU device ID
    #[serde(default)]
    device_id: Option<String>,
    /// Driver version
    #[serde(default)]
    driver_version: Option<String>,
    /// VBIOS version
    #[serde(default)]
    vbios_version: Option<String>,
    /// Whether CC mode is enabled
    #[serde(default)]
    cc_mode: Option<bool>,
}

/// NVIDIA Remote Attestation Service verifier
///
/// Verifies GPU attestation evidence using NVIDIA's cloud-based
/// attestation verification service.
#[cfg(feature = "remote-attestation")]
pub struct NrasVerifier {
    config: NrasConfig,
    client: reqwest::Client,
}

#[cfg(feature = "remote-attestation")]
impl NrasVerifier {
    /// Create a new NRAS verifier with the given configuration
    pub fn new(config: NrasConfig) -> TeeResult<Self> {
        let client = reqwest::Client::builder()
            .timeout(std::time::Duration::from_secs(config.timeout_secs))
            .build()
            .map_err(|e| TeeError::Configuration(format!("Failed to create HTTP client: {}", e)))?;

        Ok(Self { config, client })
    }

    /// Create with default configuration
    pub fn with_defaults() -> TeeResult<Self> {
        Self::new(NrasConfig::default())
    }

    /// Verify attestation evidence with NRAS
    async fn verify_with_nras(
        &self,
        evidence: &GpuAttestationEvidence,
        nonce: Option<&str>,
    ) -> TeeResult<(bool, Option<NrasClaims>)> {
        info!("[NRAS] Verifying GPU attestation for {}", evidence.gpu_uuid);

        // Build the EAT token from evidence
        let eat_token = self.build_eat_token(evidence)?;

        let request = NrasVerifyRequest {
            evidence: eat_token,
            nonce: nonce.map(String::from),
        };

        let mut req = self.client.post(&self.config.api_url).json(&request);

        // Add API key if configured
        if let Some(ref api_key) = self.config.api_key {
            req = req.header("Authorization", format!("Bearer {}", api_key));
        }

        let response = req
            .send()
            .await
            .map_err(|e| TeeError::GpuCcVerification(format!("NRAS request failed: {}", e)))?;

        let status = response.status();
        if !status.is_success() {
            let error_text = response.text().await.unwrap_or_default();
            error!(
                "[NRAS] Verification request failed: status={}, body={}",
                status, error_text
            );
            return Err(TeeError::GpuCcVerification(format!(
                "NRAS returned status {}: {}",
                status, error_text
            )));
        }

        let result: NrasVerifyResponse = response.json().await.map_err(|e| {
            TeeError::GpuCcVerification(format!("Failed to parse NRAS response: {}", e))
        })?;

        if let Some(ref err) = result.error {
            warn!("[NRAS] Verification error: {}", err);
            return Ok((false, None));
        }

        let verified = result.attested || result.result.to_lowercase() == "success";

        if verified {
            info!("[NRAS] GPU attestation verified successfully");
        } else {
            warn!(
                "[NRAS] GPU attestation verification failed: result={}",
                result.result
            );
        }

        Ok((verified, result.claims))
    }

    /// Build an EAT (Entity Attestation Token) from evidence
    ///
    /// Note: This is a simplified implementation. Real EAT tokens have a specific
    /// CBOR/COSE structure defined in the IETF draft specification.
    fn build_eat_token(&self, evidence: &GpuAttestationEvidence) -> TeeResult<String> {
        // For NVIDIA attestation, the evidence typically comes as a JSON structure
        // that can be base64-encoded for transport
        let evidence_json = serde_json::json!({
            "attestation_report": evidence.attestation_report,
            "signature": evidence.signature,
            "cert_chain": evidence.cert_chain,
            "nonce": evidence.nonce,
        });

        let json_string = serde_json::to_string(&evidence_json).map_err(|e| {
            TeeError::GpuCcVerification(format!("Failed to serialize evidence: {}", e))
        })?;

        Ok(base64::Engine::encode(
            &base64::engine::general_purpose::STANDARD,
            json_string.as_bytes(),
        ))
    }
}

#[cfg(feature = "remote-attestation")]
#[async_trait]
impl GpuVerifier for NrasVerifier {
    async fn verify(
        &self,
        evidence: &GpuAttestationEvidence,
        expected_nonce: Option<&str>,
    ) -> TeeResult<GpuCcVerificationResult> {
        // First verify with NRAS
        let (attestation_valid, claims) = self.verify_with_nras(evidence, expected_nonce).await?;

        // Check nonce
        let nonce_verified = expected_nonce.map(|n| evidence.nonce == n).unwrap_or(true);

        // Determine CC mode from claims or evidence
        let cc_mode_enabled = claims
            .as_ref()
            .and_then(|c| c.cc_mode)
            .unwrap_or(!evidence.attestation_report.is_empty());

        Ok(GpuCcVerificationResult {
            cc_mode_enabled,
            attestation_valid,
            gpu_uuid: evidence.gpu_uuid.clone(),
            nonce_verified,
            gpu_model: evidence.gpu_model.clone(),
            driver_version: claims
                .and_then(|c| c.driver_version)
                .unwrap_or_else(|| evidence.driver_version.clone()),
            verified_at: chrono::Utc::now(),
        })
    }
}

/// Verify evidence using NVIDIA Remote Attestation Service
///
/// Returns an error if the remote-attestation feature is not enabled.
#[cfg(feature = "remote-attestation")]
pub async fn verify_evidence_remote(
    evidence: &GpuAttestationEvidence,
    expected_nonce: Option<&str>,
    config: Option<NrasConfig>,
) -> TeeResult<GpuCcVerificationResult> {
    let verifier = NrasVerifier::new(config.unwrap_or_default())?;
    verifier.verify(evidence, expected_nonce).await
}

#[cfg(not(feature = "remote-attestation"))]
pub async fn verify_evidence_remote(
    _evidence: &GpuAttestationEvidence,
    _expected_nonce: Option<&str>,
    _config: Option<NrasConfig>,
) -> TeeResult<GpuCcVerificationResult> {
    Err(TeeError::GpuCcVerification(
        "Remote attestation feature not enabled. Compile with --features remote-attestation".into(),
    ))
}

#[cfg(test)]
mod tests {
    use super::*;

    fn sample_evidence() -> GpuAttestationEvidence {
        GpuAttestationEvidence {
            gpu_uuid: "GPU-abc123".to_string(),
            attestation_report: "deadbeef".to_string(),
            signature: "cafebabe".to_string(),
            cert_chain: vec!["cert1".to_string(), "cert2".to_string()],
            nonce: "test_nonce_123".to_string(),
            gpu_model: "NVIDIA H100".to_string(),
            driver_version: "555.0".to_string(),
            timestamp: chrono::Utc::now(),
        }
    }

    #[tokio::test]
    async fn test_verify_evidence_with_matching_nonce() {
        let evidence = sample_evidence();
        let result = verify_evidence(&evidence, Some("test_nonce_123"))
            .await
            .unwrap();

        assert!(result.nonce_verified);
        assert!(result.attestation_valid);
        assert!(result.cc_mode_enabled);
    }

    #[tokio::test]
    async fn test_verify_evidence_with_wrong_nonce() {
        let evidence = sample_evidence();
        let result = verify_evidence(&evidence, Some("wrong_nonce"))
            .await
            .unwrap();

        assert!(!result.nonce_verified);
    }

    #[tokio::test]
    async fn test_verify_evidence_no_nonce_check() {
        let evidence = sample_evidence();
        let result = verify_evidence(&evidence, None).await.unwrap();

        assert!(result.nonce_verified);
    }

    #[tokio::test]
    async fn test_verify_all() {
        let evidence_list = vec![
            GpuAttestationEvidence {
                gpu_uuid: "GPU-1".to_string(),
                attestation_report: "report1".to_string(),
                nonce: "nonce".to_string(),
                ..sample_evidence()
            },
            GpuAttestationEvidence {
                gpu_uuid: "GPU-2".to_string(),
                attestation_report: "report2".to_string(),
                nonce: "nonce".to_string(),
                ..sample_evidence()
            },
        ];

        let results = verify_all_evidence(&evidence_list, Some("nonce"))
            .await
            .unwrap();

        assert_eq!(results.len(), 2);
        assert!(results.iter().all(|r| r.nonce_verified));
    }

    #[tokio::test]
    async fn test_cc_mode_detection() {
        // With attestation report = CC mode enabled
        let with_report = GpuAttestationEvidence {
            attestation_report: "report".to_string(),
            ..sample_evidence()
        };
        let result = verify_evidence(&with_report, None).await.unwrap();
        assert!(result.cc_mode_enabled);

        // Without attestation report = CC mode not detected
        let without_report = GpuAttestationEvidence {
            attestation_report: "".to_string(),
            ..sample_evidence()
        };
        let result = verify_evidence(&without_report, None).await.unwrap();
        assert!(!result.cc_mode_enabled);
    }
}
