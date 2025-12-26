//! TEE Configuration types

use serde::{Deserialize, Serialize};
use std::path::PathBuf;

/// TEE Configuration
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct TeeConfig {
    /// Enable TEE verification
    #[serde(default)]
    pub enabled: bool,

    /// Require TEE for all nodes (reject non-TEE)
    #[serde(default)]
    pub require_tee: bool,

    /// TDX configuration
    #[serde(default)]
    pub tdx: TdxConfig,

    /// GPU CC configuration
    #[serde(default)]
    pub gpu: GpuCcConfig,

    /// Attestation server configuration
    #[serde(default)]
    pub attestation_server: Option<AttestationServerConfig>,
}

/// TDX-specific configuration
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct TdxConfig {
    /// Expected MRTD measurement (hex encoded, 48 bytes / 96 hex chars)
    pub expected_mrtd: Option<String>,

    /// Expected RTMR[0] (firmware/initrd)
    pub expected_rtmr0: Option<String>,

    /// Expected RTMR[1] (OS kernel)
    pub expected_rtmr1: Option<String>,

    /// Expected RTMR[2] (application)
    pub expected_rtmr2: Option<String>,

    /// Expected RTMR[3] (reserved)
    pub expected_rtmr3: Option<String>,

    /// Path to TDX quote generator binary
    #[serde(default = "default_quote_generator_path")]
    pub quote_generator_path: PathBuf,

    /// Path to server certificate for cert hash binding
    pub server_cert_path: Option<PathBuf>,
}

fn default_quote_generator_path() -> PathBuf {
    PathBuf::from("/usr/bin/tdx-quote-generator")
}

/// GPU Confidential Computing configuration
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct GpuCcConfig {
    /// Require GPU CC mode
    #[serde(default = "default_true")]
    pub require_cc_mode: bool,

    /// Allowed GPU models for CC
    #[serde(default = "default_allowed_models")]
    pub allowed_models: Vec<String>,

    /// Path to nvevidence binary
    #[serde(default = "default_nvevidence_path")]
    pub nvevidence_path: PathBuf,

    /// Output directory for evidence files
    #[serde(default = "default_evidence_output_dir")]
    pub evidence_output_dir: PathBuf,
}

fn default_true() -> bool {
    true
}

fn default_allowed_models() -> Vec<String> {
    vec![
        "H100 PCIe".to_string(),
        "H100 SXM".to_string(),
        "H200".to_string(),
    ]
}

fn default_nvevidence_path() -> PathBuf {
    PathBuf::from("chutes-nvevidence")
}

fn default_evidence_output_dir() -> PathBuf {
    PathBuf::from("/var/log/attestation-service")
}

/// Attestation server configuration
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct AttestationServerConfig {
    /// Enable remote attestation server
    #[serde(default)]
    pub enabled: bool,

    /// Server URL
    pub url: String,

    /// API key for authentication
    pub api_key: Option<String>,

    /// TLS configuration
    pub tls: Option<TlsConfig>,

    /// Server bind host
    #[serde(default = "default_bind_host")]
    pub bind_host: String,

    /// Server bind port
    #[serde(default = "default_bind_port")]
    pub bind_port: u16,
}

fn default_bind_host() -> String {
    "0.0.0.0".to_string()
}

fn default_bind_port() -> u16 {
    8443
}

/// TLS configuration for attestation server
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct TlsConfig {
    /// Path to TLS certificate
    pub cert_path: PathBuf,

    /// Path to TLS key
    pub key_path: PathBuf,

    /// Require client certificates (mTLS)
    #[serde(default)]
    pub mtls_required: bool,

    /// Path to client CA certificate
    pub client_ca_path: Option<PathBuf>,
}

impl Default for TeeConfig {
    fn default() -> Self {
        Self {
            enabled: false,
            require_tee: false,
            tdx: TdxConfig::default(),
            gpu: GpuCcConfig::default(),
            attestation_server: None,
        }
    }
}

impl Default for TdxConfig {
    fn default() -> Self {
        Self {
            expected_mrtd: None,
            expected_rtmr0: None,
            expected_rtmr1: None,
            expected_rtmr2: None,
            expected_rtmr3: None,
            quote_generator_path: default_quote_generator_path(),
            server_cert_path: None,
        }
    }
}

impl Default for GpuCcConfig {
    fn default() -> Self {
        Self {
            require_cc_mode: true,
            allowed_models: default_allowed_models(),
            nvevidence_path: default_nvevidence_path(),
            evidence_output_dir: default_evidence_output_dir(),
        }
    }
}

impl TeeConfig {
    /// Create a new TeeConfig with TEE enabled
    pub fn enabled() -> Self {
        Self {
            enabled: true,
            ..Default::default()
        }
    }

    /// Check if TEE verification should be performed
    pub fn should_verify(&self) -> bool {
        self.enabled
    }

    /// Check if non-TEE nodes should be rejected
    pub fn should_reject_non_tee(&self) -> bool {
        self.enabled && self.require_tee
    }
}

impl TdxConfig {
    /// Parse expected MRTD from hex string to bytes
    pub fn expected_mrtd_bytes(&self) -> Option<[u8; 48]> {
        self.expected_mrtd.as_ref().and_then(|s| {
            hex::decode(s)
                .ok()
                .and_then(|v| v.try_into().ok())
        })
    }

    /// Parse expected RTMR from hex string to bytes
    pub fn expected_rtmr_bytes(&self, index: usize) -> Option<[u8; 48]> {
        let rtmr_hex = match index {
            0 => &self.expected_rtmr0,
            1 => &self.expected_rtmr1,
            2 => &self.expected_rtmr2,
            3 => &self.expected_rtmr3,
            _ => return None,
        };

        rtmr_hex.as_ref().and_then(|s| {
            hex::decode(s)
                .ok()
                .and_then(|v| v.try_into().ok())
        })
    }
}

impl GpuCcConfig {
    /// Check if a GPU model is allowed for CC
    pub fn is_model_allowed(&self, model: &str) -> bool {
        self.allowed_models.iter().any(|m| model.contains(m))
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_default_tee_config() {
        let config = TeeConfig::default();
        assert!(!config.enabled);
        assert!(!config.require_tee);
        assert!(!config.should_verify());
        assert!(!config.should_reject_non_tee());
    }

    #[test]
    fn test_enabled_tee_config() {
        let config = TeeConfig::enabled();
        assert!(config.enabled);
        assert!(!config.require_tee);
        assert!(config.should_verify());
        assert!(!config.should_reject_non_tee());
    }

    #[test]
    fn test_require_tee() {
        let config = TeeConfig {
            enabled: true,
            require_tee: true,
            ..Default::default()
        };
        assert!(config.should_reject_non_tee());
    }

    #[test]
    fn test_tdx_mrtd_bytes_parsing() {
        let config = TdxConfig {
            expected_mrtd: Some("00".repeat(48)),
            ..Default::default()
        };
        let bytes = config.expected_mrtd_bytes().unwrap();
        assert_eq!(bytes, [0u8; 48]);
    }

    #[test]
    fn test_tdx_mrtd_bytes_invalid() {
        let config = TdxConfig {
            expected_mrtd: Some("invalid".to_string()),
            ..Default::default()
        };
        assert!(config.expected_mrtd_bytes().is_none());
    }

    #[test]
    fn test_tdx_rtmr_bytes_parsing() {
        let mut config = TdxConfig::default();
        config.expected_rtmr0 = Some("01".repeat(48));
        config.expected_rtmr1 = Some("02".repeat(48));

        let rtmr0 = config.expected_rtmr_bytes(0).unwrap();
        assert_eq!(rtmr0, [0x01u8; 48]);

        let rtmr1 = config.expected_rtmr_bytes(1).unwrap();
        assert_eq!(rtmr1, [0x02u8; 48]);

        // Index 2 and 3 should be None
        assert!(config.expected_rtmr_bytes(2).is_none());
        assert!(config.expected_rtmr_bytes(4).is_none());
    }

    #[test]
    fn test_gpu_allowed_models() {
        let config = GpuCcConfig::default();
        assert!(config.is_model_allowed("NVIDIA H100 PCIe"));
        assert!(config.is_model_allowed("H100 SXM5"));
        assert!(config.is_model_allowed("H200"));
        assert!(!config.is_model_allowed("RTX 4090"));
        assert!(!config.is_model_allowed("A100"));
    }

    #[test]
    fn test_config_serialization() {
        let config = TeeConfig::enabled();
        let json = serde_json::to_string(&config).unwrap();
        let parsed: TeeConfig = serde_json::from_str(&json).unwrap();
        assert_eq!(config.enabled, parsed.enabled);
    }

    #[test]
    fn test_default_paths() {
        let tdx = TdxConfig::default();
        assert_eq!(
            tdx.quote_generator_path,
            PathBuf::from("/usr/bin/tdx-quote-generator")
        );

        let gpu = GpuCcConfig::default();
        assert_eq!(gpu.nvevidence_path, PathBuf::from("chutes-nvevidence"));
    }
}

