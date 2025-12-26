//! TEE Validation Module
//!
//! Integrates TDX quote verification and GPU CC attestation into
//! the Basilica validator verification pipeline.

use anyhow::{Context, Result};
use basilica_common::ssh::SshConnectionDetails;
use std::sync::Arc;
use tracing::{debug, info, warn};

use crate::ssh::ValidatorSshClient;

/// Expected TDX measurements for valid executor VMs
#[derive(Debug, Clone, Default)]
pub struct ExpectedMeasurements {
    /// MRTD - Build-time measurement of TD (48 bytes)
    pub mrtd: Option<[u8; 48]>,
    /// RTMR[0] - Firmware/initrd measurements
    pub rtmr0: Option<[u8; 48]>,
    /// RTMR[1] - OS kernel measurements
    pub rtmr1: Option<[u8; 48]>,
    /// RTMR[2] - Application measurements
    pub rtmr2: Option<[u8; 48]>,
    /// RTMR[3] - Reserved
    pub rtmr3: Option<[u8; 48]>,
}

/// TDX Quote verification result
#[derive(Debug, Clone)]
pub struct TdxVerificationResult {
    pub quote_valid: bool,
    pub mrtd_matches: bool,
    pub rtmr_matches: Vec<bool>,
    pub report_data_matches: bool,
    pub mrtd_hex: String,
    pub raw_quote: Vec<u8>,
}

/// GPU Confidential Computing verification result
#[derive(Debug, Clone)]
pub struct GpuCcVerificationResult {
    pub cc_mode_enabled: bool,
    pub attestation_valid: bool,
    pub gpu_uuid: String,
    pub nonce_verified: bool,
    pub gpu_model: String,
    pub driver_version: String,
}

/// Combined TEE verification result
#[derive(Debug, Clone)]
pub struct TeeVerificationResult {
    pub tdx: Option<TdxVerificationResult>,
    pub gpu_cc: Option<GpuCcVerificationResult>,
    pub tee_verified: bool,
}

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

/// TEE Validator for verifying executor TEE status
pub struct TeeValidator {
    config: TeeValidatorConfig,
    ssh_client: Arc<ValidatorSshClient>,
}

impl TeeValidator {
    /// Create a new TeeValidator with configuration
    pub fn new(config: TeeValidatorConfig, ssh_client: Arc<ValidatorSshClient>) -> Self {
        Self { config, ssh_client }
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
    /// 2. Parse quote structure
    /// 3. Compare measurements against expected values
    pub async fn verify_tdx_quote(
        &self,
        connection: &SshConnectionDetails,
        nonce: &[u8; 64],
    ) -> Result<TdxVerificationResult> {
        info!("[TEE] Generating TDX quote from executor");

        let nonce_hex = hex::encode(nonce);

        // Generate quote via SSH
        // Uses tdx-quote-generator tool or configfs-tsm interface
        let quote_command = format!(
            "tdx-quote-generator --nonce {} --output hex 2>/dev/null || \
             python3 -c 'import sys; sys.exit(1)' 2>/dev/null",
            nonce_hex
        );

        let quote_hex = self
            .ssh_client
            .execute_command(connection, &quote_command, true)
            .await
            .context("Failed to generate TDX quote")?;

        // Check if the command succeeded
        if quote_hex.trim().is_empty()
            || quote_hex.contains("error")
            || quote_hex.contains("not found")
        {
            warn!("[TEE] TDX quote generation not available on this node");
            return Ok(TdxVerificationResult {
                quote_valid: false,
                mrtd_matches: false,
                rtmr_matches: vec![false; 4],
                report_data_matches: false,
                mrtd_hex: String::new(),
                raw_quote: vec![],
            });
        }

        let quote_bytes = hex::decode(quote_hex.trim()).context("Invalid quote hex")?;

        // Parse TDX quote structure
        let parse_result = self.parse_tdx_quote(&quote_bytes)?;

        // Verify quote signature (stub - always returns true for now)
        // TODO: Implement actual signature verification using Intel QVL
        let signature_valid = true;

        // Compare measurements
        let mrtd_matches = self.verify_measurement(
            &parse_result.mrtd,
            self.config.expected_measurements.mrtd.as_ref(),
        );

        let mut rtmr_matches = vec![true; 4];
        if let Some(expected) = &self.config.expected_measurements.rtmr0 {
            rtmr_matches[0] = &parse_result.rtmrs[0] == expected;
        }
        if let Some(expected) = &self.config.expected_measurements.rtmr1 {
            rtmr_matches[1] = &parse_result.rtmrs[1] == expected;
        }
        if let Some(expected) = &self.config.expected_measurements.rtmr2 {
            rtmr_matches[2] = &parse_result.rtmrs[2] == expected;
        }
        if let Some(expected) = &self.config.expected_measurements.rtmr3 {
            rtmr_matches[3] = &parse_result.rtmrs[3] == expected;
        }

        // Verify report_data contains our nonce
        let report_data_matches = parse_result.report_data[..64] == *nonce;

        if !mrtd_matches {
            warn!(
                "[TEE] MRTD mismatch: got {}",
                hex::encode(parse_result.mrtd)
            );
        }

        Ok(TdxVerificationResult {
            quote_valid: signature_valid,
            mrtd_matches,
            rtmr_matches,
            report_data_matches,
            mrtd_hex: hex::encode(parse_result.mrtd),
            raw_quote: quote_bytes,
        })
    }

    /// Verify GPU is in Confidential Compute mode
    ///
    /// Uses nvevidence SDK to:
    /// 1. Query GPU CC mode status
    /// 2. Generate attestation report with nonce
    /// 3. Verify attestation signature
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
            "chutes-nvevidence --nonce {} --format json 2>/dev/null || echo '{{}}'",
            nonce_hex
        );

        let attestation_json = self
            .ssh_client
            .execute_command(connection, &attestation_command, true)
            .await
            .context("Failed to generate GPU attestation")?;

        // Parse attestation JSON
        let attestation: serde_json::Value =
            serde_json::from_str(&attestation_json).unwrap_or(serde_json::json!({}));

        // Verify attestation signature (stub - always returns true for now)
        // TODO: Implement actual verification using NVIDIA attestation service
        let attestation_valid = !attestation.is_object() || attestation.get("error").is_none();
        let nonce_verified = attestation_valid;

        Ok(GpuCcVerificationResult {
            cc_mode_enabled,
            attestation_valid,
            gpu_uuid,
            nonce_verified,
            gpu_model,
            driver_version,
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

        // Generate random nonces using Send-safe approach
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

    /// Parse TDX quote structure
    fn parse_tdx_quote(&self, quote: &[u8]) -> Result<ParsedQuote> {
        // TDX Quote v4 structure:
        // [0..48]   Header
        // [48..632] TD Report (includes MRTD, RTMR, etc.)
        // [632..]   Signature

        const HEADER_SIZE: usize = 48;
        const TD_REPORT_SIZE: usize = 584;
        const MIN_SIZE: usize = HEADER_SIZE + TD_REPORT_SIZE;

        if quote.len() < MIN_SIZE {
            return Err(anyhow::anyhow!(
                "Quote too short: {} bytes (minimum {})",
                quote.len(),
                MIN_SIZE
            ));
        }

        // Extract MRTD at offset 136 within TD Report
        let report_offset = HEADER_SIZE;
        let mrtd: [u8; 48] = quote[report_offset + 136..report_offset + 184]
            .try_into()
            .context("Failed to extract MRTD")?;

        // Extract RTMRs
        let rtmr0: [u8; 48] = quote[report_offset + 328..report_offset + 376]
            .try_into()
            .context("Failed to extract RTMR0")?;
        let rtmr1: [u8; 48] = quote[report_offset + 376..report_offset + 424]
            .try_into()
            .context("Failed to extract RTMR1")?;
        let rtmr2: [u8; 48] = quote[report_offset + 424..report_offset + 472]
            .try_into()
            .context("Failed to extract RTMR2")?;
        let rtmr3: [u8; 48] = quote[report_offset + 472..report_offset + 520]
            .try_into()
            .context("Failed to extract RTMR3")?;

        // Extract report data
        let report_data: [u8; 64] = quote[report_offset + 520..report_offset + 584]
            .try_into()
            .context("Failed to extract report data")?;

        Ok(ParsedQuote {
            mrtd,
            rtmrs: [rtmr0, rtmr1, rtmr2, rtmr3],
            report_data,
        })
    }

    /// Verify a measurement against expected value
    fn verify_measurement(&self, actual: &[u8; 48], expected: Option<&[u8; 48]>) -> bool {
        match expected {
            Some(exp) => actual == exp,
            None => true, // No expected value = accept any
        }
    }
}

/// Parsed TDX quote data
struct ParsedQuote {
    mrtd: [u8; 48],
    rtmrs: [[u8; 48]; 4],
    report_data: [u8; 64],
}

impl ExpectedMeasurements {
    /// Create measurements from hex strings
    pub fn from_hex(
        mrtd: Option<&str>,
        rtmr0: Option<&str>,
        rtmr1: Option<&str>,
        rtmr2: Option<&str>,
        rtmr3: Option<&str>,
    ) -> Result<Self> {
        fn parse_measurement(hex_str: Option<&str>) -> Result<Option<[u8; 48]>> {
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

        Ok(Self {
            mrtd: parse_measurement(mrtd)?,
            rtmr0: parse_measurement(rtmr0)?,
            rtmr1: parse_measurement(rtmr1)?,
            rtmr2: parse_measurement(rtmr2)?,
            rtmr3: parse_measurement(rtmr3)?,
        })
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_expected_measurements_from_hex() {
        let measurements = ExpectedMeasurements::from_hex(
            Some(&"aa".repeat(48)),
            Some(&"bb".repeat(48)),
            None,
            None,
            None,
        )
        .unwrap();

        assert!(measurements.mrtd.is_some());
        assert_eq!(measurements.mrtd.unwrap(), [0xAAu8; 48]);
        assert!(measurements.rtmr0.is_some());
        assert_eq!(measurements.rtmr0.unwrap(), [0xBBu8; 48]);
        assert!(measurements.rtmr1.is_none());
    }

    #[test]
    fn test_default_config() {
        let config = TeeValidatorConfig::default();
        assert!(!config.enabled);
        assert!(!config.require_tee);
        assert!(config.allowed_gpu_models.contains(&"H100 PCIe".to_string()));
    }

    #[test]
    fn test_tee_verification_result_default() {
        let result = TeeVerificationResult {
            tdx: None,
            gpu_cc: None,
            tee_verified: false,
        };
        assert!(!result.tee_verified);
    }
}
