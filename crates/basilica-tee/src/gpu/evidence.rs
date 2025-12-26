//! GPU Evidence Parser
//!
//! Parses and validates GPU attestation evidence.

use crate::error::{TeeError, TeeResult};
use crate::types::{GpuAttestationEvidence, GpuCcVerificationResult};
use serde_json::Value;
use tracing::{debug, warn};

/// GPU Evidence Parser
///
/// Parses attestation evidence from JSON and performs verification.
pub struct GpuEvidenceParser;

impl GpuEvidenceParser {
    /// Parse evidence from JSON string
    pub fn parse(evidence_json: &str) -> TeeResult<Vec<GpuAttestationEvidence>> {
        let value: Value = serde_json::from_str(evidence_json)?;

        let evidence_list = if value.is_array() {
            value
                .as_array()
                .unwrap()
                .iter()
                .map(Self::parse_single)
                .collect::<TeeResult<Vec<_>>>()?
        } else {
            vec![Self::parse_single(&value)?]
        };

        Ok(evidence_list)
    }

    /// Parse a single evidence object
    fn parse_single(value: &Value) -> TeeResult<GpuAttestationEvidence> {
        let gpu_uuid = value
            .get("gpu_uuid")
            .and_then(|v| v.as_str())
            .ok_or_else(|| TeeError::GpuAttestation("Missing gpu_uuid".into()))?
            .to_string();

        let attestation_report = value
            .get("attestation_report")
            .and_then(|v| v.as_str())
            .unwrap_or("")
            .to_string();

        let signature = value
            .get("signature")
            .and_then(|v| v.as_str())
            .unwrap_or("")
            .to_string();

        let cert_chain = value
            .get("cert_chain")
            .and_then(|v| v.as_array())
            .map(|arr| {
                arr.iter()
                    .filter_map(|v| v.as_str().map(|s| s.to_string()))
                    .collect()
            })
            .unwrap_or_default();

        let nonce = value
            .get("nonce")
            .and_then(|v| v.as_str())
            .unwrap_or("")
            .to_string();

        let gpu_model = value
            .get("gpu_model")
            .or_else(|| value.get("model"))
            .and_then(|v| v.as_str())
            .unwrap_or("Unknown")
            .to_string();

        let driver_version = value
            .get("driver_version")
            .and_then(|v| v.as_str())
            .unwrap_or("Unknown")
            .to_string();

        Ok(GpuAttestationEvidence {
            gpu_uuid,
            attestation_report,
            signature,
            cert_chain,
            nonce,
            gpu_model,
            driver_version,
            timestamp: chrono::Utc::now(),
        })
    }

    /// Verify attestation evidence
    ///
    /// Note: This is a stub implementation. In production, this should
    /// verify the attestation signature chain using NVIDIA's attestation
    /// service or local RIM files.
    pub fn verify(
        evidence: &GpuAttestationEvidence,
        expected_nonce: Option<&str>,
    ) -> TeeResult<GpuCcVerificationResult> {
        debug!(
            "[GpuEvidence] Verifying evidence for GPU {}",
            evidence.gpu_uuid
        );

        // Verify nonce if provided
        let nonce_verified = if let Some(expected) = expected_nonce {
            let matches = evidence.nonce == expected;
            if !matches {
                warn!(
                    "[GpuEvidence] Nonce mismatch: expected {}, got {}",
                    expected, evidence.nonce
                );
            }
            matches
        } else {
            true
        };

        // TODO: Implement actual attestation verification
        // Options:
        // 1. Use NVIDIA Remote Attestation Service (NRAS)
        // 2. Verify locally using RIM files and signature chain
        let attestation_valid = Self::verify_signature(evidence)?;

        // Check if evidence indicates CC mode
        let cc_mode_enabled = !evidence.attestation_report.is_empty();

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

    /// Verify attestation signature
    ///
    /// Note: This is a stub implementation.
    fn verify_signature(_evidence: &GpuAttestationEvidence) -> TeeResult<bool> {
        // TODO: Implement actual signature verification
        // This would verify:
        // 1. The attestation report signature
        // 2. The certificate chain back to NVIDIA's root
        // 3. The report contents match expected format

        debug!("[GpuEvidence] Signature verification stub - returning true");
        debug!("[GpuEvidence] NOTE: Implement actual verification for production use");

        Ok(true)
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
    fn test_parse_evidence_array() {
        let json = sample_evidence_json();
        let evidence = GpuEvidenceParser::parse(&json).unwrap();

        assert_eq!(evidence.len(), 1);
        assert_eq!(evidence[0].gpu_uuid, "GPU-abc123");
        assert_eq!(evidence[0].attestation_report, "deadbeef");
        assert_eq!(evidence[0].nonce, "test_nonce_123");
        assert_eq!(evidence[0].cert_chain.len(), 2);
    }

    #[test]
    fn test_parse_evidence_single() {
        let json = serde_json::json!({
            "gpu_uuid": "GPU-single",
            "attestation_report": "report"
        })
        .to_string();

        let evidence = GpuEvidenceParser::parse(&json).unwrap();
        assert_eq!(evidence.len(), 1);
        assert_eq!(evidence[0].gpu_uuid, "GPU-single");
    }

    #[test]
    fn test_parse_evidence_missing_uuid() {
        let json = r#"{"attestation_report": "report"}"#;
        let result = GpuEvidenceParser::parse(json);
        assert!(result.is_err());
    }

    #[test]
    fn test_parse_evidence_optional_fields() {
        let json = r#"{"gpu_uuid": "GPU-123"}"#;
        let evidence = GpuEvidenceParser::parse(json).unwrap();

        assert_eq!(evidence[0].gpu_uuid, "GPU-123");
        assert_eq!(evidence[0].attestation_report, "");
        assert_eq!(evidence[0].gpu_model, "Unknown");
    }

    #[test]
    fn test_verify_evidence_with_matching_nonce() {
        let json = sample_evidence_json();
        let evidence = GpuEvidenceParser::parse(&json).unwrap();

        let result = GpuEvidenceParser::verify(&evidence[0], Some("test_nonce_123")).unwrap();

        assert!(result.nonce_verified);
        assert!(result.attestation_valid); // Stub returns true
        assert!(result.cc_mode_enabled);
    }

    #[test]
    fn test_verify_evidence_with_wrong_nonce() {
        let json = sample_evidence_json();
        let evidence = GpuEvidenceParser::parse(&json).unwrap();

        let result = GpuEvidenceParser::verify(&evidence[0], Some("wrong_nonce")).unwrap();

        assert!(!result.nonce_verified);
    }

    #[test]
    fn test_verify_evidence_no_nonce_check() {
        let json = sample_evidence_json();
        let evidence = GpuEvidenceParser::parse(&json).unwrap();

        let result = GpuEvidenceParser::verify(&evidence[0], None).unwrap();

        assert!(result.nonce_verified);
    }

    #[test]
    fn test_verify_all() {
        let json = serde_json::json!([
            {"gpu_uuid": "GPU-1", "attestation_report": "report1", "nonce": "nonce"},
            {"gpu_uuid": "GPU-2", "attestation_report": "report2", "nonce": "nonce"}
        ])
        .to_string();

        let evidence = GpuEvidenceParser::parse(&json).unwrap();
        let results = GpuEvidenceParser::verify_all(&evidence, Some("nonce")).unwrap();

        assert_eq!(results.len(), 2);
        assert!(results.iter().all(|r| r.nonce_verified));
    }

    #[test]
    fn test_cc_mode_detection() {
        // With attestation report = CC mode enabled
        let with_report =
            GpuEvidenceParser::parse(r#"{"gpu_uuid": "GPU-1", "attestation_report": "report"}"#)
                .unwrap();
        let result = GpuEvidenceParser::verify(&with_report[0], None).unwrap();
        assert!(result.cc_mode_enabled);

        // Without attestation report = CC mode not detected
        let without_report =
            GpuEvidenceParser::parse(r#"{"gpu_uuid": "GPU-2", "attestation_report": ""}"#).unwrap();
        let result = GpuEvidenceParser::verify(&without_report[0], None).unwrap();
        assert!(!result.cc_mode_enabled);
    }
}
