//! Shared data types for TEE operations

use serde::{Deserialize, Serialize};

/// TDX Quote verification result
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

/// GPU Confidential Computing verification result
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

/// Combined TEE verification result
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct TeeVerificationResult {
    /// TDX verification result
    pub tdx: Option<TdxVerificationResult>,
    /// GPU CC verification result
    pub gpu_cc: Option<GpuCcVerificationResult>,
    /// Overall TEE verification passed
    pub tee_verified: bool,
}

/// GPU device information
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct GpuDeviceInfo {
    /// Unique GPU identifier (UUID without "GPU-" prefix and hyphens)
    pub uuid: String,
    /// Full GPU product name, e.g., "NVIDIA H100 PCIe"
    pub name: String,
    /// Total memory in bytes
    pub memory: u64,
    /// CUDA compute capability major version
    pub major: Option<u32>,
    /// CUDA compute capability minor version
    pub minor: Option<u32>,
    /// Clock rate in MHz
    pub clock_rate: f64,
    /// ECC enabled
    pub ecc: Option<bool>,
    /// Short name for the GPU model
    pub model_short_ref: String,
    /// Whether GPU is in CC mode
    pub cc_mode_enabled: Option<bool>,
}

/// Attestation evidence from GPU
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct GpuAttestationEvidence {
    /// GPU UUID
    pub gpu_uuid: String,
    /// Attestation report (raw bytes, hex encoded)
    pub attestation_report: String,
    /// Signature over the report
    pub signature: String,
    /// Certificate chain for verification
    pub cert_chain: Vec<String>,
    /// Nonce used in attestation
    pub nonce: String,
    /// GPU model
    pub gpu_model: String,
    /// Driver version
    pub driver_version: String,
    /// Timestamp
    pub timestamp: chrono::DateTime<chrono::Utc>,
}

/// Expected measurements for TDX verification
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

impl ExpectedMeasurements {
    /// Create new expected measurements from config
    pub fn from_config(config: &crate::config::TdxConfig) -> Self {
        Self {
            mrtd: config.expected_mrtd_bytes(),
            rtmr0: config.expected_rtmr_bytes(0),
            rtmr1: config.expected_rtmr_bytes(1),
            rtmr2: config.expected_rtmr_bytes(2),
            rtmr3: config.expected_rtmr_bytes(3),
        }
    }

    /// Check if MRTD matches
    pub fn matches_mrtd(&self, mrtd: &[u8; 48]) -> bool {
        self.mrtd.as_ref().map_or(true, |expected| expected == mrtd)
    }

    /// Check if RTMR at index matches
    pub fn matches_rtmr(&self, index: usize, rtmr: &[u8; 48]) -> bool {
        let expected = match index {
            0 => &self.rtmr0,
            1 => &self.rtmr1,
            2 => &self.rtmr2,
            3 => &self.rtmr3,
            _ => return true,
        };
        expected.as_ref().map_or(true, |e| e == rtmr)
    }
}

impl TeeVerificationResult {
    /// Create a new result indicating TEE verification passed
    pub fn verified(tdx: TdxVerificationResult, gpu_cc: GpuCcVerificationResult) -> Self {
        let tee_verified = tdx.quote_valid && tdx.mrtd_matches && gpu_cc.cc_mode_enabled;
        Self {
            tdx: Some(tdx),
            gpu_cc: Some(gpu_cc),
            tee_verified,
        }
    }

    /// Create a new result indicating no TEE verification was done
    pub fn not_verified() -> Self {
        Self {
            tdx: None,
            gpu_cc: None,
            tee_verified: false,
        }
    }
}

/// Helper module for hex serialization of byte vectors
mod hex_bytes {
    use serde::{Deserialize, Deserializer, Serializer};

    pub fn serialize<S>(bytes: &Vec<u8>, serializer: S) -> Result<S::Ok, S::Error>
    where
        S: Serializer,
    {
        serializer.serialize_str(&hex::encode(bytes))
    }

    pub fn deserialize<'de, D>(deserializer: D) -> Result<Vec<u8>, D::Error>
    where
        D: Deserializer<'de>,
    {
        let s = String::deserialize(deserializer)?;
        hex::decode(&s).map_err(serde::de::Error::custom)
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_expected_measurements_matches_mrtd() {
        let expected = ExpectedMeasurements {
            mrtd: Some([0x01u8; 48]),
            ..Default::default()
        };

        assert!(expected.matches_mrtd(&[0x01u8; 48]));
        assert!(!expected.matches_mrtd(&[0x02u8; 48]));
    }

    #[test]
    fn test_expected_measurements_empty_matches_any() {
        let expected = ExpectedMeasurements::default();

        // Empty expected should match anything
        assert!(expected.matches_mrtd(&[0x01u8; 48]));
        assert!(expected.matches_mrtd(&[0x00u8; 48]));
        assert!(expected.matches_rtmr(0, &[0xFFu8; 48]));
    }

    #[test]
    fn test_expected_measurements_rtmr_matching() {
        let expected = ExpectedMeasurements {
            rtmr0: Some([0xAAu8; 48]),
            rtmr1: Some([0xBBu8; 48]),
            ..Default::default()
        };

        assert!(expected.matches_rtmr(0, &[0xAAu8; 48]));
        assert!(!expected.matches_rtmr(0, &[0x00u8; 48]));
        assert!(expected.matches_rtmr(1, &[0xBBu8; 48]));
        // Index 2 not set, should match any
        assert!(expected.matches_rtmr(2, &[0x00u8; 48]));
        // Invalid index should always match
        assert!(expected.matches_rtmr(5, &[0x00u8; 48]));
    }

    #[test]
    fn test_tee_verification_result_verified() {
        let tdx = TdxVerificationResult {
            quote_valid: true,
            mrtd_matches: true,
            rtmr_matches: vec![true, true, true, true],
            report_data_matches: true,
            raw_quote: vec![],
            mrtd_hex: "00".repeat(48),
            verified_at: chrono::Utc::now(),
        };

        let gpu = GpuCcVerificationResult {
            cc_mode_enabled: true,
            attestation_valid: true,
            gpu_uuid: "GPU-123".to_string(),
            nonce_verified: true,
            gpu_model: "H100".to_string(),
            driver_version: "555.0".to_string(),
            verified_at: chrono::Utc::now(),
        };

        let result = TeeVerificationResult::verified(tdx, gpu);
        assert!(result.tee_verified);
    }

    #[test]
    fn test_tee_verification_result_not_verified() {
        let result = TeeVerificationResult::not_verified();
        assert!(!result.tee_verified);
        assert!(result.tdx.is_none());
        assert!(result.gpu_cc.is_none());
    }

    #[test]
    fn test_gpu_device_info_serialization() {
        let info = GpuDeviceInfo {
            uuid: "abc123".to_string(),
            name: "NVIDIA H100 PCIe".to_string(),
            memory: 80 * 1024 * 1024 * 1024,
            major: Some(9),
            minor: Some(0),
            clock_rate: 1755.0,
            ecc: Some(true),
            model_short_ref: "h100".to_string(),
            cc_mode_enabled: Some(true),
        };

        let json = serde_json::to_string(&info).unwrap();
        let parsed: GpuDeviceInfo = serde_json::from_str(&json).unwrap();
        assert_eq!(info.uuid, parsed.uuid);
        assert_eq!(info.name, parsed.name);
        assert_eq!(info.cc_mode_enabled, parsed.cc_mode_enabled);
    }

    #[test]
    fn test_tdx_verification_result_hex_serialization() {
        let result = TdxVerificationResult {
            quote_valid: true,
            mrtd_matches: true,
            rtmr_matches: vec![true],
            report_data_matches: true,
            raw_quote: vec![0xDE, 0xAD, 0xBE, 0xEF],
            mrtd_hex: "00".repeat(48),
            verified_at: chrono::Utc::now(),
        };

        let json = serde_json::to_string(&result).unwrap();
        assert!(json.contains("deadbeef")); // hex encoded

        let parsed: TdxVerificationResult = serde_json::from_str(&json).unwrap();
        assert_eq!(parsed.raw_quote, vec![0xDE, 0xAD, 0xBE, 0xEF]);
    }
}

