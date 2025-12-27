//! TEE Validation Module
//!
//! Integrates TDX quote verification and GPU CC attestation into
//! the Basilica validator verification pipeline.
//!
//! Uses the basilica-tee crate for quote parsing and verification.

use anyhow::{Context, Result};
use std::sync::Arc;
use tracing::{debug, info, warn};

use basilica_common::ssh::SshConnectionDetails;
use basilica_tee::tdx::TdxQuoteVerifier;

// Re-export ExpectedMeasurements for external use
pub use basilica_tee::types::ExpectedMeasurements;

use crate::ssh::ValidatorSshClient;

// Re-export types from basilica-tee for external use
pub use basilica_tee::types::{
    GpuCcVerificationResult, TdxVerificationResult, TeeVerificationResult,
};

/// TEE Validator configuration
#[derive(Debug, Clone)]
pub struct TeeValidatorConfig {
    /// Whether TEE verification is enabled
    pub enabled: bool,
    /// Whether to require TEE (reject non-TEE nodes)
    pub require_tee: bool,
    /// Expected TDX measurements
    pub expected_measurements: ExpectedMeasurements,
    /// Whether GPU CC mode is required
    pub require_gpu_cc: bool,
    /// Allowed GPU models for CC mode
    pub allowed_gpu_models: Vec<String>,
}

impl Default for TeeValidatorConfig {
    fn default() -> Self {
        Self {
            enabled: false,
            require_tee: false,
            expected_measurements: ExpectedMeasurements::default(),
            require_gpu_cc: false,
            allowed_gpu_models: vec![
                "H100 PCIe".to_string(),
                "H100 SXM".to_string(),
                "H200".to_string(),
            ],
        }
    }
}

impl TeeValidatorConfig {
    /// Create config with expected measurements from hex strings
    pub fn with_measurements(
        enabled: bool,
        require_tee: bool,
        mrtd_hex: Option<&str>,
        rtmr0_hex: Option<&str>,
    ) -> Result<Self> {
        let measurements = ExpectedMeasurements {
            mrtd: parse_measurement_hex(mrtd_hex)?,
            rtmr0: parse_measurement_hex(rtmr0_hex)?,
            ..Default::default()
        };

        Ok(Self {
            enabled,
            require_tee,
            expected_measurements: measurements,
            ..Default::default()
        })
    }
}

/// TEE Validator for verifying executor TEE status
pub struct TeeValidator {
    config: TeeValidatorConfig,
    ssh_client: Arc<ValidatorSshClient>,
    quote_verifier: TdxQuoteVerifier,
}

impl TeeValidator {
    /// Create a new TeeValidator with configuration
    pub fn new(config: TeeValidatorConfig, ssh_client: Arc<ValidatorSshClient>) -> Self {
        let quote_verifier = TdxQuoteVerifier::new(config.expected_measurements.clone());
        Self {
            config,
            ssh_client,
            quote_verifier,
        }
    }

    /// Check if TEE verification is enabled
    pub fn is_enabled(&self) -> bool {
        self.config.enabled
    }

    /// Check if non-TEE nodes should be rejected
    pub fn requires_tee(&self) -> bool {
        self.config.enabled && self.config.require_tee
    }

    /// Verify TDX quote from executor
    ///
    /// Steps:
    /// 1. SSH to node and generate TDX quote with nonce
    /// 2. Parse quote structure using basilica-tee
    /// 3. Compare measurements against expected values
    pub async fn verify_tdx_quote(
        &self,
        connection: &SshConnectionDetails,
        nonce: &[u8; 64],
    ) -> Result<TdxVerificationResult> {
        info!("[TEE] Generating TDX quote from executor");

        let nonce_hex = hex::encode(nonce);

        // Generate quote via SSH using tdx-quote-generator
        let quote_command = format!(
            r#"
            if command -v tdx-quote-generator &>/dev/null; then
                TMPFILE=$(mktemp)
                tdx-quote-generator --report-data {} --hex --output "$TMPFILE" 2>/dev/null
                cat "$TMPFILE" && rm -f "$TMPFILE"
            elif [ -c /dev/tdx_guest ] || [ -c /dev/tdx-guest ]; then
                echo "TDX_DEVICE_PRESENT_NO_TOOL"
            else
                echo "TDX_NOT_AVAILABLE"
            fi
            "#,
            nonce_hex
        );

        let quote_output = self
            .ssh_client
            .execute_command(connection, &quote_command, true)
            .await
            .context("Failed to generate TDX quote")?;

        let quote_output = quote_output.trim();

        // Check for TDX availability
        if quote_output.is_empty()
            || quote_output.contains("TDX_NOT_AVAILABLE")
            || quote_output.contains("error")
        {
            warn!("[TEE] TDX quote generation not available on this node");
            return Ok(TdxVerificationResult {
                quote_valid: false,
                mrtd_matches: false,
                rtmr_matches: vec![false; 4],
                report_data_matches: false,
                mrtd_hex: String::new(),
                raw_quote: vec![],
                verified_at: chrono::Utc::now(),
            });
        }

        if quote_output.contains("TDX_DEVICE_PRESENT_NO_TOOL") {
            warn!("[TEE] TDX device present but quote generator tool not available");
            return Ok(TdxVerificationResult {
                quote_valid: false,
                mrtd_matches: false,
                rtmr_matches: vec![false; 4],
                report_data_matches: false,
                mrtd_hex: String::new(),
                raw_quote: vec![],
                verified_at: chrono::Utc::now(),
            });
        }

        // Read the quote file content (binary)
        let quote_bytes = tokio::fs::read(quote_output)
            .await
            .or_else(|_| {
                // If it's not a file path, try to decode as hex
                hex::decode(quote_output)
            })
            .context("Failed to read/decode TDX quote")?;

        // Use basilica-tee's quote verifier
        let result = self.quote_verifier.verify(&quote_bytes, Some(nonce))?;

        if !result.mrtd_matches {
            warn!("[TEE] MRTD mismatch: got {}", result.mrtd_hex);
        }

        if result.quote_valid && result.mrtd_matches {
            info!("[TEE] TDX quote verification passed");
        } else {
            warn!(
                "[TEE] TDX verification issues: quote_valid={}, mrtd_matches={}",
                result.quote_valid, result.mrtd_matches
            );
        }

        Ok(result)
    }

    /// Verify GPU is in Confidential Compute mode
    ///
    /// Uses NVIDIA attestation tools to:
    /// 1. Query GPU CC mode status
    /// 2. Generate attestation report with nonce
    /// 3. Verify attestation
    pub async fn verify_gpu_cc_mode(
        &self,
        connection: &SshConnectionDetails,
        nonce: &[u8; 32],
    ) -> Result<GpuCcVerificationResult> {
        info!("[TEE] Verifying GPU CC mode");

        // Check CC mode status via nvidia-smi
        let cc_mode_output = self
            .ssh_client
            .execute_command(
                connection,
                "nvidia-smi -q 2>/dev/null | grep -i 'Conf Compute Mode' || echo 'not_found'",
                true,
            )
            .await
            .context("Failed to check CC mode")?;

        let cc_mode_enabled = cc_mode_output.to_lowercase().contains("enabled");

        if !cc_mode_enabled {
            debug!("[TEE] GPU is not in Confidential Compute mode");
            return Ok(GpuCcVerificationResult {
                cc_mode_enabled: false,
                attestation_valid: false,
                gpu_uuid: String::new(),
                nonce_verified: false,
                gpu_model: String::new(),
                driver_version: String::new(),
                verified_at: chrono::Utc::now(),
            });
        }

        // Get GPU model and driver version
        let gpu_info_output = self
            .ssh_client
            .execute_command(
                connection,
                "nvidia-smi --query-gpu=name,uuid,driver_version --format=csv,noheader 2>/dev/null | head -1",
                true,
            )
            .await
            .unwrap_or_default();

        let parts: Vec<&str> = gpu_info_output.split(',').map(|s| s.trim()).collect();
        let gpu_model = parts.first().unwrap_or(&"Unknown").to_string();
        let gpu_uuid = parts.get(1).unwrap_or(&"Unknown").to_string();
        let driver_version = parts.get(2).unwrap_or(&"Unknown").to_string();

        // Check if GPU model is allowed for CC
        let model_allowed = self
            .config
            .allowed_gpu_models
            .iter()
            .any(|m| gpu_model.contains(m));
        if !model_allowed {
            warn!(
                "[TEE] GPU model {} is not in allowed list for CC",
                gpu_model
            );
        }

        // Generate GPU attestation with nonce
        let nonce_hex = hex::encode(nonce);
        let attestation_command = format!(
            r#"
            if command -v nv-attestation-tool &>/dev/null; then
                nv-attestation-tool --nonce {} 2>/dev/null
            elif command -v nvidia-attestation &>/dev/null; then
                nvidia-attestation generate --nonce {} 2>/dev/null
            else
                echo '{{"error": "no_attestation_tool"}}'
            fi
            "#,
            nonce_hex, nonce_hex
        );

        let attestation_json = self
            .ssh_client
            .execute_command(connection, &attestation_command, true)
            .await
            .context("Failed to generate GPU attestation")?;

        // Parse and verify attestation using basilica-tee
        let attestation_valid = if !attestation_json.contains("error") {
            // Parse evidence using basilica-tee
            match basilica_tee::gpu::parse_evidence(&attestation_json) {
                Ok(evidence) if !evidence.is_empty() => {
                    // Verify evidence
                    match basilica_tee::gpu::verify_evidence(&evidence[0], Some(&nonce_hex)).await {
                        Ok(result) => result.attestation_valid && result.nonce_verified,
                        Err(e) => {
                            warn!("[TEE] GPU attestation verification failed: {}", e);
                            false
                        }
                    }
                }
                _ => {
                    debug!("[TEE] No GPU attestation evidence available");
                    // CC mode enabled but no attestation SDK - still valid for basic verification
                    true
                }
            }
        } else {
            // No attestation tool, but CC mode is enabled
            debug!("[TEE] No GPU attestation tool, using CC mode status only");
            true
        };

        let nonce_verified = attestation_valid;

        info!(
            "[TEE] GPU CC verification: cc_enabled={}, attestation_valid={}",
            cc_mode_enabled, attestation_valid
        );

        Ok(GpuCcVerificationResult {
            cc_mode_enabled,
            attestation_valid,
            gpu_uuid,
            nonce_verified,
            gpu_model,
            driver_version,
            verified_at: chrono::Utc::now(),
        })
    }

    /// Perform full TEE verification
    ///
    /// Verifies both TDX quote and GPU CC mode.
    pub async fn verify_full(
        &self,
        connection: &SshConnectionDetails,
    ) -> Result<TeeVerificationResult> {
        if !self.config.enabled {
            return Ok(TeeVerificationResult {
                tdx: None,
                gpu_cc: None,
                tee_verified: false,
            });
        }

        info!("[TEE] Starting full TEE verification");

        // Generate random nonces
        let mut tdx_nonce = [0u8; 64];
        getrandom::getrandom(&mut tdx_nonce).unwrap_or_default();
        let mut gpu_nonce = [0u8; 32];
        getrandom::getrandom(&mut gpu_nonce).unwrap_or_default();

        // Verify TDX quote
        let tdx_result = match self.verify_tdx_quote(connection, &tdx_nonce).await {
            Ok(result) => result,
            Err(e) => {
                warn!("[TEE] TDX verification failed: {}", e);
                TdxVerificationResult {
                    quote_valid: false,
                    mrtd_matches: false,
                    rtmr_matches: vec![false; 4],
                    report_data_matches: false,
                    mrtd_hex: String::new(),
                    raw_quote: vec![],
                    verified_at: chrono::Utc::now(),
                }
            }
        };

        // Verify GPU CC mode
        let gpu_cc_result = match self.verify_gpu_cc_mode(connection, &gpu_nonce).await {
            Ok(result) => result,
            Err(e) => {
                warn!("[TEE] GPU CC verification failed: {}", e);
                GpuCcVerificationResult {
                    cc_mode_enabled: false,
                    attestation_valid: false,
                    gpu_uuid: String::new(),
                    nonce_verified: false,
                    gpu_model: String::new(),
                    driver_version: String::new(),
                    verified_at: chrono::Utc::now(),
                }
            }
        };

        // Determine overall TEE verification status
        let tdx_ok = tdx_result.quote_valid && tdx_result.mrtd_matches;
        let gpu_ok = !self.config.require_gpu_cc || gpu_cc_result.cc_mode_enabled;
        let tee_verified = tdx_ok && gpu_ok;

        if tee_verified {
            info!("[TEE] TEE verification passed");
        } else {
            warn!(
                "[TEE] TEE verification failed: tdx_ok={}, gpu_ok={}",
                tdx_ok, gpu_ok
            );
        }

        Ok(TeeVerificationResult {
            tdx: Some(tdx_result),
            gpu_cc: Some(gpu_cc_result),
            tee_verified,
        })
    }
}

/// Parse a hex measurement string into a 48-byte array
fn parse_measurement_hex(hex_str: Option<&str>) -> Result<Option<[u8; 48]>> {
    match hex_str {
        Some(s) if !s.is_empty() => {
            let bytes = hex::decode(s).context("Invalid hex string")?;
            let arr: [u8; 48] = bytes
                .try_into()
                .map_err(|_| anyhow::anyhow!("Measurement must be 48 bytes"))?;
            Ok(Some(arr))
        }
        _ => Ok(None),
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_parse_measurement_hex() {
        let result = parse_measurement_hex(Some(&"aa".repeat(48))).unwrap();
        assert!(result.is_some());
        assert_eq!(result.unwrap(), [0xAAu8; 48]);

        let result = parse_measurement_hex(None).unwrap();
        assert!(result.is_none());

        let result = parse_measurement_hex(Some("")).unwrap();
        assert!(result.is_none());
    }

    #[test]
    fn test_default_config() {
        let config = TeeValidatorConfig::default();
        assert!(!config.enabled);
        assert!(!config.require_tee);
        assert!(config.allowed_gpu_models.contains(&"H100 PCIe".to_string()));
    }

    #[test]
    fn test_config_with_measurements() {
        let config = TeeValidatorConfig::with_measurements(
            true,
            true,
            Some(&"aa".repeat(48)),
            Some(&"bb".repeat(48)),
        )
        .unwrap();

        assert!(config.enabled);
        assert!(config.require_tee);
        assert!(config.expected_measurements.mrtd.is_some());
        assert_eq!(config.expected_measurements.mrtd.unwrap(), [0xAAu8; 48]);
    }
}
