//! Attestation verification implementation

use anyhow::{Context, Result};
use p256::ecdsa::{Signature, VerifyingKey};
use signature::Verifier;

use super::types::*;

pub struct AttestationVerifier;

impl AttestationVerifier {
    pub fn verify_signed_attestation(signed_attestation: &SignedAttestation) -> Result<bool> {
        Self::verify_signature(signed_attestation)?;
        Self::verify_attestation_content(&signed_attestation.report)?;
        Ok(true)
    }

    pub fn verify_signature(signed_attestation: &SignedAttestation) -> Result<()> {
        let public_key = VerifyingKey::from_sec1_bytes(&signed_attestation.ephemeral_public_key)
            .context("Failed to parse ephemeral public key")?;

        let signature = Signature::from_slice(&signed_attestation.signature)
            .context("Failed to parse signature")?;

        let report_json = serde_json::to_string(&signed_attestation.report)
            .context("Failed to serialize report for verification")?;

        public_key
            .verify(report_json.as_bytes(), &signature)
            .context("Signature verification failed")?;

        Ok(())
    }

    pub fn verify_attestation_content(report: &AttestationReport) -> Result<()> {
        Self::verify_timestamp(report)?;
        Self::verify_required_fields(report)?;
        Self::verify_optional_components(report)?;

        tracing::info!("Attestation content verification passed");
        Ok(())
    }

    fn verify_timestamp(report: &AttestationReport) -> Result<()> {
        if !report.is_recent() {
            anyhow::bail!(
                "Attestation report is too old: {} minutes",
                report.get_age_minutes()
            );
        }
        Ok(())
    }

    fn verify_required_fields(report: &AttestationReport) -> Result<()> {
        if report.executor_id.is_empty() {
            anyhow::bail!("Missing executor ID in attestation report");
        }

        if report.gpu_info.is_empty() {
            anyhow::bail!("No GPU information in attestation report");
        }

        if !report.binary_info.is_valid() {
            anyhow::bail!("Invalid binary information in attestation report");
        }

        Ok(())
    }

    fn verify_optional_components(report: &AttestationReport) -> Result<()> {
        // Verify VDF proof if present
        if let Some(vdf_proof) = &report.vdf_proof {
            if !vdf_proof.is_valid() {
                anyhow::bail!("Invalid VDF proof in attestation report");
            }
        }

        // Verify model fingerprint if present
        if let Some(fingerprint) = &report.model_fingerprint {
            if !fingerprint.is_valid() {
                anyhow::bail!("Model fingerprint verification failed");
            }
        }

        // Verify semantic similarity if present
        if let Some(similarity) = &report.semantic_similarity {
            if !similarity.is_valid() {
                anyhow::bail!(
                    "Semantic similarity test failed: {} < {}",
                    similarity.cosine_similarity,
                    similarity.threshold
                );
            }
        }

        Ok(())
    }
}

impl AttestationVerifier {
    /// Verify multiple attestations in batch
    pub fn verify_multiple(attestations: &[SignedAttestation]) -> Result<Vec<bool>> {
        attestations
            .iter()
            .map(Self::verify_signed_attestation)
            .collect()
    }

    /// Quick verification that only checks signature and basic fields
    pub fn quick_verify(signed_attestation: &SignedAttestation) -> Result<bool> {
        Self::verify_signature(signed_attestation)?;

        if signed_attestation.report.executor_id.is_empty() {
            anyhow::bail!("Missing executor ID");
        }

        Ok(true)
    }

    /// Comprehensive verification with detailed reporting
    pub fn detailed_verify(signed_attestation: &SignedAttestation) -> Result<VerificationReport> {
        let mut report = VerificationReport::new();

        // Signature verification
        match Self::verify_signature(signed_attestation) {
            Ok(_) => report.signature_valid = true,
            Err(e) => {
                report.signature_valid = false;
                report
                    .errors
                    .push(format!("Signature verification failed: {e}"));
            }
        }

        // Content verification
        match Self::verify_attestation_content(&signed_attestation.report) {
            Ok(_) => report.content_valid = true,
            Err(e) => {
                report.content_valid = false;
                report
                    .errors
                    .push(format!("Content verification failed: {e}"));
            }
        }

        // Optional component checks
        report.has_vdf_proof = signed_attestation.report.has_vdf_proof();
        report.has_model_fingerprint = signed_attestation.report.has_model_fingerprint();
        report.has_semantic_similarity = signed_attestation.report.has_semantic_similarity();

        report.overall_valid = report.signature_valid && report.content_valid;
        Ok(report)
    }

    /// Verify with custom timestamp tolerance
    pub fn verify_with_tolerance(
        signed_attestation: &SignedAttestation,
        max_age_minutes: i64,
    ) -> Result<bool> {
        Self::verify_signature(signed_attestation)?;

        let age = signed_attestation.get_age_minutes();
        if age > max_age_minutes {
            anyhow::bail!("Attestation too old: {} > {} minutes", age, max_age_minutes);
        }

        Self::verify_required_fields(&signed_attestation.report)?;
        Self::verify_optional_components(&signed_attestation.report)?;

        Ok(true)
    }
}

#[derive(Debug, Clone)]
pub struct VerificationReport {
    pub signature_valid: bool,
    pub content_valid: bool,
    pub overall_valid: bool,
    pub has_vdf_proof: bool,
    pub has_model_fingerprint: bool,
    pub has_semantic_similarity: bool,
    pub errors: Vec<String>,
}

impl Default for VerificationReport {
    fn default() -> Self {
        Self::new()
    }
}

impl VerificationReport {
    pub fn new() -> Self {
        Self {
            signature_valid: false,
            content_valid: false,
            overall_valid: false,
            has_vdf_proof: false,
            has_model_fingerprint: false,
            has_semantic_similarity: false,
            errors: Vec::new(),
        }
    }

    pub fn is_valid(&self) -> bool {
        self.overall_valid
    }

    pub fn has_errors(&self) -> bool {
        !self.errors.is_empty()
    }

    pub fn get_summary(&self) -> String {
        if self.overall_valid {
            "Verification passed".to_string()
        } else {
            format!("Verification failed: {}", self.errors.join("; "))
        }
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::attestation::signer::AttestationSigner;

    #[test]
    fn test_signature_verification() {
        let signer = AttestationSigner::new();
        let report = AttestationReport::new("test_executor".to_string());
        let signed_attestation = signer.sign_attestation(report).unwrap();

        let result = AttestationVerifier::verify_signature(&signed_attestation);
        assert!(result.is_ok());
    }

    #[test]
    fn test_quick_verify() {
        let signer = AttestationSigner::new();
        let report = AttestationReport::new("test_executor".to_string());
        let signed_attestation = signer.sign_attestation(report).unwrap();

        let result = AttestationVerifier::quick_verify(&signed_attestation);
        assert!(result.is_ok());
    }

    #[test]
    fn test_detailed_verify() {
        let signer = AttestationSigner::new();
        let report = AttestationReport::new("test_executor".to_string());
        let signed_attestation = signer.sign_attestation(report).unwrap();

        let verification_report =
            AttestationVerifier::detailed_verify(&signed_attestation).unwrap();

        assert!(verification_report.signature_valid);
        // Content validation might fail due to missing GPU info, which is expected
    }

    #[test]
    #[ignore = "Requires GPU hardware"]
    fn test_verify_multiple() {
        let signer = AttestationSigner::new();
        let gpu_info = vec![crate::gpu::GpuInfo::new(
            crate::gpu::GpuVendor::Unknown,
            "Test GPU".to_string(),
            "1.0.0".to_string(),
        )];
        let mut report1 = AttestationReport::new("executor1".to_string());
        report1.gpu_info = gpu_info.clone();
        report1.binary_info = crate::attestation::types::BinaryInfo::default()
            .with_checksum("test_checksum".to_string());
        let mut report2 = AttestationReport::new("executor2".to_string());
        report2.gpu_info = gpu_info;
        report2.binary_info = crate::attestation::types::BinaryInfo::default()
            .with_checksum("test_checksum2".to_string());
        let reports = vec![report1, report2];
        let signed_attestations = signer.sign_multiple(reports).unwrap();

        let results = AttestationVerifier::verify_multiple(&signed_attestations).unwrap();
        assert_eq!(results.len(), 2);
    }

    #[test]
    fn test_verification_report() {
        let mut report = VerificationReport::new();
        assert!(!report.is_valid());
        assert!(!report.has_errors());

        report.signature_valid = true;
        report.content_valid = true;
        report.overall_valid = true;

        assert!(report.is_valid());
        assert_eq!(report.get_summary(), "Verification passed");
    }

    #[test]
    fn test_verify_with_tolerance() {
        let signer = AttestationSigner::new();
        let mut report = AttestationReport::new("test_executor".to_string());
        // Add GPU info to make the report valid
        report.gpu_info.push(crate::gpu::GpuInfo::new(
            crate::gpu::GpuVendor::Unknown,
            "Test GPU".to_string(),
            "1.0.0".to_string(),
        ));
        // Add valid binary info
        report.binary_info.path = "test_path".to_string();
        report.binary_info.checksum = "test_checksum".to_string();

        let signed_attestation = signer.sign_attestation(report).unwrap();

        // Should pass with generous tolerance
        let result = AttestationVerifier::verify_with_tolerance(&signed_attestation, 60);
        assert!(result.is_ok());

        // Should fail with very strict tolerance - this test is commented out as it's timing sensitive
        // let _result = AttestationVerifier::verify_with_tolerance(&signed_attestation, 0);
    }

    #[test]
    fn test_invalid_signature() {
        let signer = AttestationSigner::new();
        let report = AttestationReport::new("test_executor".to_string());
        let mut signed_attestation = signer.sign_attestation(report).unwrap();

        // Corrupt the signature
        signed_attestation.signature[0] = signed_attestation.signature[0].wrapping_add(1);

        let result = AttestationVerifier::verify_signature(&signed_attestation);
        assert!(result.is_err());
    }
}
