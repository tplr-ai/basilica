//! Attestation utility functions

use anyhow::{Context, Result};
use p256::pkcs8::{DecodePublicKey, EncodePublicKey};
use std::fs;
use std::path::Path;

use super::types::SignedAttestation;

/// Save attestation to separate files (JSON, signature, public key)
pub fn save_attestation_to_files(
    signed_attestation: &SignedAttestation,
    base_path: &Path,
) -> Result<()> {
    let report_path = base_path.with_extension("json");
    let signature_path = base_path.with_extension("sig");
    let pubkey_path = base_path.with_extension("pub");

    // Save the report as pretty-printed JSON
    let report_json = serde_json::to_string_pretty(&signed_attestation.report)
        .context("Failed to serialize attestation report")?;
    fs::write(&report_path, report_json).context("Failed to write attestation report file")?;

    // Save the signature
    fs::write(&signature_path, &signed_attestation.signature)
        .context("Failed to write signature file")?;

    // Save the public key in PEM format for compatibility with validator
    // Convert SEC1 bytes to PEM format
    let verifying_key =
        p256::ecdsa::VerifyingKey::from_sec1_bytes(&signed_attestation.ephemeral_public_key)
            .context("Failed to parse public key bytes")?;
    let pubkey_pem = verifying_key
        .to_public_key_pem(p256::pkcs8::LineEnding::LF)
        .map_err(|e| anyhow::anyhow!("Failed to encode public key as PEM: {}", e))?;
    fs::write(&pubkey_path, pubkey_pem).context("Failed to write public key file")?;

    tracing::info!(
        "Attestation saved to: {}, {}, {}",
        report_path.display(),
        signature_path.display(),
        pubkey_path.display()
    );

    Ok(())
}

/// Load attestation from separate files
pub fn load_attestation_from_files(base_path: &Path) -> Result<SignedAttestation> {
    let report_path = base_path.with_extension("json");
    let signature_path = base_path.with_extension("sig");
    let pubkey_path = base_path.with_extension("pub");

    // Load the report
    let report_json =
        fs::read_to_string(&report_path).context("Failed to read attestation report file")?;
    let report =
        serde_json::from_str(&report_json).context("Failed to parse attestation report")?;

    // Load the signature
    let signature = fs::read(&signature_path).context("Failed to read signature file")?;

    // Load the public key from PEM format
    let pubkey_pem = fs::read_to_string(&pubkey_path).context("Failed to read public key file")?;

    // Try to parse as PEM first, fall back to raw bytes for backwards compatibility
    let ephemeral_public_key = if pubkey_pem.starts_with("-----BEGIN PUBLIC KEY-----") {
        // Parse PEM format
        let verifying_key = p256::ecdsa::VerifyingKey::from_public_key_pem(&pubkey_pem)
            .context("Failed to parse public key from PEM")?;
        verifying_key.to_sec1_bytes().to_vec()
    } else {
        // Assume raw bytes for backwards compatibility
        fs::read(&pubkey_path).context("Failed to read public key file")?
    };

    Ok(SignedAttestation::new(
        report,
        signature,
        ephemeral_public_key,
    ))
}

/// Save attestation as a single JSON file with all components
pub fn save_attestation_as_json(
    signed_attestation: &SignedAttestation,
    file_path: &Path,
) -> Result<()> {
    let json = serde_json::to_string_pretty(signed_attestation)
        .context("Failed to serialize signed attestation")?;

    fs::write(file_path, json).context("Failed to write attestation JSON file")?;

    tracing::info!("Attestation saved to: {}", file_path.display());
    Ok(())
}

/// Load attestation from a single JSON file
pub fn load_attestation_from_json(file_path: &Path) -> Result<SignedAttestation> {
    let json = fs::read_to_string(file_path).context("Failed to read attestation JSON file")?;

    let signed_attestation =
        serde_json::from_str(&json).context("Failed to parse attestation JSON")?;

    Ok(signed_attestation)
}

/// Generate a summary report of the attestation
pub fn generate_attestation_summary(signed_attestation: &SignedAttestation) -> AttestationSummary {
    let report = &signed_attestation.report;

    AttestationSummary {
        executor_id: report.executor_id.clone(),
        version: report.version.clone(),
        timestamp: report.timestamp,
        age_minutes: report.get_age_minutes(),
        gpu_count: report.gpu_info.len(),
        has_vdf_proof: report.has_vdf_proof(),
        has_model_fingerprint: report.has_model_fingerprint(),
        has_semantic_similarity: report.has_semantic_similarity(),
        binary_verified: report.binary_info.signature_verified,
        is_recent: report.is_recent(),
        is_valid: signed_attestation.is_valid(),
    }
}

/// Compare two attestations for differences
pub fn compare_attestations(
    attestation1: &SignedAttestation,
    attestation2: &SignedAttestation,
) -> AttestationComparison {
    let report1 = &attestation1.report;
    let report2 = &attestation2.report;

    AttestationComparison {
        same_executor: report1.executor_id == report2.executor_id,
        same_version: report1.version == report2.version,
        timestamp_diff_minutes: (report2.timestamp - report1.timestamp).num_minutes(),
        gpu_count_diff: report2.gpu_info.len() as i32 - report1.gpu_info.len() as i32,
        both_have_vdf: report1.has_vdf_proof() && report2.has_vdf_proof(),
        both_have_model_fingerprint: report1.has_model_fingerprint()
            && report2.has_model_fingerprint(),
        both_verified: report1.binary_info.signature_verified
            && report2.binary_info.signature_verified,
    }
}

/// Validate attestation file paths exist
pub fn validate_attestation_files(base_path: &Path) -> Result<()> {
    let report_path = base_path.with_extension("json");
    let signature_path = base_path.with_extension("sig");
    let pubkey_path = base_path.with_extension("pub");

    if !report_path.exists() {
        anyhow::bail!(
            "Attestation report file not found: {}",
            report_path.display()
        );
    }
    if !signature_path.exists() {
        anyhow::bail!("Signature file not found: {}", signature_path.display());
    }
    if !pubkey_path.exists() {
        anyhow::bail!("Public key file not found: {}", pubkey_path.display());
    }

    Ok(())
}

/// Extract executor ID from attestation file without full parsing
pub fn extract_executor_id(file_path: &Path) -> Result<String> {
    let json = fs::read_to_string(file_path).context("Failed to read attestation file")?;

    // Parse just enough to get the executor ID
    let value: serde_json::Value = serde_json::from_str(&json).context("Failed to parse JSON")?;

    let executor_id = value
        .get("report")
        .and_then(|r| r.get("executor_id"))
        .and_then(|id| id.as_str())
        .unwrap_or("unknown");

    Ok(executor_id.to_string())
}

#[derive(Debug, Clone)]
pub struct AttestationSummary {
    pub executor_id: String,
    pub version: String,
    pub timestamp: chrono::DateTime<chrono::Utc>,
    pub age_minutes: i64,
    pub gpu_count: usize,
    pub has_vdf_proof: bool,
    pub has_model_fingerprint: bool,
    pub has_semantic_similarity: bool,
    pub binary_verified: bool,
    pub is_recent: bool,
    pub is_valid: bool,
}

#[derive(Debug, Clone)]
pub struct AttestationComparison {
    pub same_executor: bool,
    pub same_version: bool,
    pub timestamp_diff_minutes: i64,
    pub gpu_count_diff: i32,
    pub both_have_vdf: bool,
    pub both_have_model_fingerprint: bool,
    pub both_verified: bool,
}

impl AttestationSummary {
    pub fn get_status(&self) -> &'static str {
        if !self.is_valid {
            "INVALID"
        } else if !self.is_recent {
            "EXPIRED"
        } else if !self.binary_verified {
            "UNVERIFIED"
        } else {
            "VALID"
        }
    }

    pub fn get_completeness_score(&self) -> f32 {
        let mut score = 0.0;
        let total_components = 5.0;

        if self.gpu_count > 0 {
            score += 1.0;
        }
        if self.has_vdf_proof {
            score += 1.0;
        }
        if self.has_model_fingerprint {
            score += 1.0;
        }
        if self.has_semantic_similarity {
            score += 1.0;
        }
        if self.binary_verified {
            score += 1.0;
        }

        score / total_components
    }
}

impl AttestationComparison {
    pub fn is_upgrade(&self) -> bool {
        self.same_executor && self.timestamp_diff_minutes > 0
    }

    pub fn has_improvements(&self) -> bool {
        self.gpu_count_diff > 0 || (self.both_have_vdf && self.both_have_model_fingerprint)
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::attestation::{builder::AttestationBuilder, signer::AttestationSigner};
    use tempfile::tempdir;

    #[test]
    fn test_save_and_load_attestation_files() {
        let temp_dir = tempdir().unwrap();
        let base_path = temp_dir.path().join("test_attestation");

        let signer = AttestationSigner::new();
        let report = AttestationBuilder::new("test_executor".to_string()).build();
        let signed_attestation = signer.sign_attestation(report).unwrap();

        // Save attestation
        save_attestation_to_files(&signed_attestation, &base_path).unwrap();

        // Load attestation
        let loaded_attestation = load_attestation_from_files(&base_path).unwrap();

        assert_eq!(
            signed_attestation.report.executor_id,
            loaded_attestation.report.executor_id
        );
        assert_eq!(signed_attestation.signature, loaded_attestation.signature);
    }

    #[test]
    fn test_save_and_load_attestation_json() {
        let temp_dir = tempdir().unwrap();
        let file_path = temp_dir.path().join("test_attestation.json");

        let signer = AttestationSigner::new();
        let report = AttestationBuilder::new("test_executor".to_string()).build();
        let signed_attestation = signer.sign_attestation(report).unwrap();

        // Save as JSON
        save_attestation_as_json(&signed_attestation, &file_path).unwrap();

        // Load from JSON
        let loaded_attestation = load_attestation_from_json(&file_path).unwrap();

        assert_eq!(
            signed_attestation.report.executor_id,
            loaded_attestation.report.executor_id
        );
    }

    #[test]
    fn test_generate_attestation_summary() {
        let signer = AttestationSigner::new();
        let report = AttestationBuilder::new("test_executor".to_string()).build();
        let signed_attestation = signer.sign_attestation(report).unwrap();

        let summary = generate_attestation_summary(&signed_attestation);

        assert_eq!(summary.executor_id, "test_executor");
        assert!(summary.is_recent);
        assert_eq!(summary.get_status(), "INVALID"); // No GPU info
    }

    #[test]
    fn test_compare_attestations() {
        let signer = AttestationSigner::new();

        let report1 = AttestationBuilder::new("test_executor".to_string()).build();
        let attestation1 = signer.sign_attestation(report1).unwrap();

        let report2 = AttestationBuilder::new("test_executor".to_string()).build();
        let attestation2 = signer.sign_attestation(report2).unwrap();

        let comparison = compare_attestations(&attestation1, &attestation2);

        assert!(comparison.same_executor);
        assert!(comparison.same_version);
        assert!(comparison.timestamp_diff_minutes >= 0);
    }

    #[test]
    fn test_validate_attestation_files() {
        let temp_dir = tempdir().unwrap();
        let base_path = temp_dir.path().join("nonexistent");

        // Should fail for non-existent files
        let result = validate_attestation_files(&base_path);
        assert!(result.is_err());
    }

    #[test]
    fn test_extract_executor_id() {
        let temp_dir = tempdir().unwrap();
        let file_path = temp_dir.path().join("test_attestation.json");

        let signer = AttestationSigner::new();
        let report = AttestationBuilder::new("test_executor_123".to_string()).build();
        let signed_attestation = signer.sign_attestation(report).unwrap();

        save_attestation_as_json(&signed_attestation, &file_path).unwrap();

        let executor_id = extract_executor_id(&file_path).unwrap();
        assert_eq!(executor_id, "test_executor_123");
    }

    #[test]
    fn test_attestation_summary_methods() {
        let summary = AttestationSummary {
            executor_id: "test".to_string(),
            version: "1.0.0".to_string(),
            timestamp: chrono::Utc::now(),
            age_minutes: 5,
            gpu_count: 2,
            has_vdf_proof: true,
            has_model_fingerprint: true,
            has_semantic_similarity: false,
            binary_verified: true,
            is_recent: true,
            is_valid: true,
        };

        assert_eq!(summary.get_status(), "VALID");
        assert_eq!(summary.get_completeness_score(), 0.8); // 4/5 components
    }

    #[test]
    fn test_attestation_comparison_methods() {
        let comparison = AttestationComparison {
            same_executor: true,
            same_version: true,
            timestamp_diff_minutes: 10,
            gpu_count_diff: 1,
            both_have_vdf: true,
            both_have_model_fingerprint: true,
            both_verified: true,
        };

        assert!(comparison.is_upgrade());
        assert!(comparison.has_improvements());
    }
}
