//! GPU Evidence Parser
//!
//! Parses GPU attestation evidence from JSON format.

use serde_json::Value;

use crate::error::{TeeError, TeeResult};
use crate::traits::EvidenceParser;
use crate::types::GpuAttestationEvidence;

/// GPU Evidence Parser
///
/// Parses attestation evidence from JSON format into structured types.
/// Handles both single evidence objects and arrays.
#[derive(Debug, Clone, Default)]
pub struct JsonEvidenceParser;

impl JsonEvidenceParser {
    /// Create a new JSON evidence parser.
    pub fn new() -> Self {
        Self
    }

    /// Parse a single evidence object from JSON value.
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
}

impl EvidenceParser for JsonEvidenceParser {
    fn parse(&self, json: &str) -> TeeResult<Vec<GpuAttestationEvidence>> {
        let value: Value = serde_json::from_str(json)?;

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
}

/// Convenience function to parse evidence using the default parser.
pub fn parse_evidence(json: &str) -> TeeResult<Vec<GpuAttestationEvidence>> {
    JsonEvidenceParser::new().parse(json)
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
        let parser = JsonEvidenceParser::new();
        let evidence = parser.parse(&json).unwrap();

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

        let evidence = parse_evidence(&json).unwrap();
        assert_eq!(evidence.len(), 1);
        assert_eq!(evidence[0].gpu_uuid, "GPU-single");
    }

    #[test]
    fn test_parse_evidence_missing_uuid() {
        let json = r#"{"attestation_report": "report"}"#;
        let result = parse_evidence(json);
        assert!(result.is_err());
    }

    #[test]
    fn test_parse_evidence_optional_fields() {
        let json = r#"{"gpu_uuid": "GPU-123"}"#;
        let evidence = parse_evidence(json).unwrap();

        assert_eq!(evidence[0].gpu_uuid, "GPU-123");
        assert_eq!(evidence[0].attestation_report, "");
        assert_eq!(evidence[0].gpu_model, "Unknown");
    }

    #[test]
    fn test_parse_with_model_alias() {
        let json = r#"{"gpu_uuid": "GPU-123", "model": "H100"}"#;
        let evidence = parse_evidence(json).unwrap();

        assert_eq!(evidence[0].gpu_model, "H100");
    }
}
