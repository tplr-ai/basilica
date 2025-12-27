//! Verification Result Types
//!
//! Types representing the results of TEE verification operations.

use serde::{Deserialize, Serialize};

use super::serde_utils::hex_bytes;

/// TDX Quote verification result.
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct TdxVerificationResult {
    /// Whether the quote signature is valid
    pub quote_valid: bool,
    /// Whether MRTD matches expected value
    pub mrtd_matches: bool,
    /// Whether each RTMR matches expected value (4 elements)
    pub rtmr_matches: Vec<bool>,
    /// Whether report data contains expected nonce
    pub report_data_matches: bool,
    /// Raw quote bytes
    #[serde(with = "hex_bytes")]
    pub raw_quote: Vec<u8>,
    /// MRTD value from quote (hex encoded)
    pub mrtd_hex: String,
    /// Timestamp of verification
    pub verified_at: chrono::DateTime<chrono::Utc>,
}

impl TdxVerificationResult {
    /// Check if all verification checks passed.
    pub fn is_valid(&self) -> bool {
        self.quote_valid && self.mrtd_matches && self.report_data_matches
    }

    /// Check if all RTMRs match.
    pub fn all_rtmrs_match(&self) -> bool {
        self.rtmr_matches.iter().all(|&m| m)
    }
}

/// GPU Confidential Computing verification result.
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct GpuCcVerificationResult {
    /// Whether CC mode is enabled on the GPU
    pub cc_mode_enabled: bool,
    /// Whether attestation signature is valid
    pub attestation_valid: bool,
    /// GPU UUID
    pub gpu_uuid: String,
    /// Whether nonce in attestation matches
    pub nonce_verified: bool,
    /// GPU model name
    pub gpu_model: String,
    /// Driver version
    pub driver_version: String,
    /// Timestamp of verification
    pub verified_at: chrono::DateTime<chrono::Utc>,
}

impl GpuCcVerificationResult {
    /// Check if all verification checks passed.
    pub fn is_valid(&self) -> bool {
        self.cc_mode_enabled && self.attestation_valid && self.nonce_verified
    }
}

/// Combined TEE verification result.
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct TeeVerificationResult {
    /// TDX verification result
    pub tdx: Option<TdxVerificationResult>,
    /// GPU CC verification result
    pub gpu_cc: Option<GpuCcVerificationResult>,
    /// Overall TEE verification passed
    pub tee_verified: bool,
}

impl TeeVerificationResult {
    /// Create a new result indicating TEE verification passed.
    pub fn verified(tdx: TdxVerificationResult, gpu_cc: GpuCcVerificationResult) -> Self {
        let tee_verified = tdx.quote_valid && tdx.mrtd_matches && gpu_cc.cc_mode_enabled;
        Self {
            tdx: Some(tdx),
            gpu_cc: Some(gpu_cc),
            tee_verified,
        }
    }

    /// Create a new result indicating no TEE verification was done.
    pub fn not_verified() -> Self {
        Self {
            tdx: None,
            gpu_cc: None,
            tee_verified: false,
        }
    }

    /// Create a result with only TDX verification.
    pub fn tdx_only(tdx: TdxVerificationResult) -> Self {
        let tee_verified = tdx.quote_valid && tdx.mrtd_matches;
        Self {
            tdx: Some(tdx),
            gpu_cc: None,
            tee_verified,
        }
    }

    /// Create a result with only GPU verification.
    pub fn gpu_only(gpu_cc: GpuCcVerificationResult) -> Self {
        let tee_verified = gpu_cc.cc_mode_enabled && gpu_cc.attestation_valid;
        Self {
            tdx: None,
            gpu_cc: Some(gpu_cc),
            tee_verified,
        }
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    fn sample_tdx_result(valid: bool) -> TdxVerificationResult {
        TdxVerificationResult {
            quote_valid: valid,
            mrtd_matches: valid,
            rtmr_matches: vec![valid, valid, valid, valid],
            report_data_matches: valid,
            raw_quote: vec![],
            mrtd_hex: "00".repeat(48),
            verified_at: chrono::Utc::now(),
        }
    }

    fn sample_gpu_result(valid: bool) -> GpuCcVerificationResult {
        GpuCcVerificationResult {
            cc_mode_enabled: valid,
            attestation_valid: valid,
            gpu_uuid: "GPU-123".to_string(),
            nonce_verified: valid,
            gpu_model: "H100".to_string(),
            driver_version: "555.0".to_string(),
            verified_at: chrono::Utc::now(),
        }
    }

    #[test]
    fn test_tee_verified() {
        let result =
            TeeVerificationResult::verified(sample_tdx_result(true), sample_gpu_result(true));
        assert!(result.tee_verified);
    }

    #[test]
    fn test_not_verified() {
        let result = TeeVerificationResult::not_verified();
        assert!(!result.tee_verified);
        assert!(result.tdx.is_none());
        assert!(result.gpu_cc.is_none());
    }

    #[test]
    fn test_tdx_only() {
        let result = TeeVerificationResult::tdx_only(sample_tdx_result(true));
        assert!(result.tee_verified);
        assert!(result.tdx.is_some());
        assert!(result.gpu_cc.is_none());
    }

    #[test]
    fn test_gpu_only() {
        let result = TeeVerificationResult::gpu_only(sample_gpu_result(true));
        assert!(result.tee_verified);
        assert!(result.tdx.is_none());
        assert!(result.gpu_cc.is_some());
    }

    #[test]
    fn test_tdx_result_is_valid() {
        assert!(sample_tdx_result(true).is_valid());
        assert!(!sample_tdx_result(false).is_valid());
    }

    #[test]
    fn test_gpu_result_is_valid() {
        assert!(sample_gpu_result(true).is_valid());
        assert!(!sample_gpu_result(false).is_valid());
    }
}
