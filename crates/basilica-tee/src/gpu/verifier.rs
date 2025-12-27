//! GPU Evidence Verification
//!
//! Verifies GPU attestation evidence for Confidential Computing.

use async_trait::async_trait;
use tracing::{debug, warn};

use crate::error::TeeResult;
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
