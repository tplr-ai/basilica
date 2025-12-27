//! GPU Evidence Parsing and Verification
//!
//! This module provides backward compatibility by re-exporting from
//! the refactored `evidence_parser` and `verifier` modules.
//!
//! For new code, prefer using:
//! - `evidence_parser::JsonEvidenceParser` for parsing
//! - `verifier::LocalGpuVerifier` for verification

use crate::error::TeeResult;
use crate::traits::{EvidenceParser, GpuVerifier};
use crate::types::{GpuAttestationEvidence, GpuCcVerificationResult};

pub use super::evidence_parser::{parse_evidence, JsonEvidenceParser};
pub use super::verifier::{verify_all_evidence, verify_evidence, LocalGpuVerifier};

/// GPU Evidence Parser (deprecated, use `JsonEvidenceParser` instead)
///
/// Maintained for backward compatibility.
/// Parses attestation evidence from JSON and performs verification.
#[deprecated(
    since = "0.2.0",
    note = "Use JsonEvidenceParser for parsing and LocalGpuVerifier for verification"
)]
pub struct GpuEvidenceParser;

#[allow(deprecated)]
impl GpuEvidenceParser {
    /// Parse evidence from JSON string
    pub fn parse(evidence_json: &str) -> TeeResult<Vec<GpuAttestationEvidence>> {
        JsonEvidenceParser::new().parse(evidence_json)
    }

    /// Verify attestation evidence
    pub fn verify(
        evidence: &GpuAttestationEvidence,
        expected_nonce: Option<&str>,
    ) -> TeeResult<GpuCcVerificationResult> {
        // Use a blocking runtime since this is a sync method
        let rt = tokio::runtime::Handle::try_current();
        match rt {
            Ok(handle) => {
                // We're in an async context, use block_in_place
                tokio::task::block_in_place(|| {
                    handle.block_on(LocalGpuVerifier::new().verify(evidence, expected_nonce))
                })
            }
            Err(_) => {
                // We're not in an async context, create a new runtime
                let rt = tokio::runtime::Runtime::new()
                    .map_err(|e| crate::error::TeeError::GpuAttestation(e.to_string()))?;
                rt.block_on(LocalGpuVerifier::new().verify(evidence, expected_nonce))
            }
        }
    }

    /// Verify multiple evidence entries
    pub fn verify_all(
        evidence_list: &[GpuAttestationEvidence],
        expected_nonce: Option<&str>,
    ) -> TeeResult<Vec<GpuCcVerificationResult>> {
        evidence_list
            .iter()
            .map(|e| Self::verify(e, expected_nonce))
            .collect()
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    fn sample_evidence_json() -> String {
        serde_json::json!([{
            "gpu_uuid": "GPU-abc123",
            "attestation_report": "deadbeef",
            "signature": "cafebabe",
            "cert_chain": ["cert1", "cert2"],
            "nonce": "test_nonce_123",
            "gpu_model": "NVIDIA H100",
            "driver_version": "555.0"
        }])
        .to_string()
    }

    #[test]
    #[allow(deprecated)]
    fn test_backward_compat_parse() {
        let json = sample_evidence_json();
        let evidence = GpuEvidenceParser::parse(&json).unwrap();

        assert_eq!(evidence.len(), 1);
        assert_eq!(evidence[0].gpu_uuid, "GPU-abc123");
    }

    #[tokio::test]
    async fn test_new_api_parse() {
        let json = sample_evidence_json();
        let evidence = parse_evidence(&json).unwrap();

        assert_eq!(evidence.len(), 1);
        assert_eq!(evidence[0].gpu_uuid, "GPU-abc123");
    }

    #[tokio::test]
    async fn test_new_api_verify() {
        let json = sample_evidence_json();
        let evidence = parse_evidence(&json).unwrap();
        let result = verify_evidence(&evidence[0], Some("test_nonce_123"))
            .await
            .unwrap();

        assert!(result.nonce_verified);
        assert!(result.attestation_valid);
    }
}
