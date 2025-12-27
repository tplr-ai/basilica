//! Evidence Types
//!
//! Types representing attestation evidence and device information.

use serde::{Deserialize, Serialize};

/// GPU device information.
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

impl GpuDeviceInfo {
    /// Check if this GPU supports Confidential Computing.
    ///
    /// Currently, only H100 and H200 GPUs support CC.
    pub fn supports_cc(&self) -> bool {
        self.name.contains("H100") || self.name.contains("H200")
    }

    /// Get the compute capability as a string (e.g., "9.0").
    pub fn compute_capability(&self) -> Option<String> {
        match (self.major, self.minor) {
            (Some(major), Some(minor)) => Some(format!("{}.{}", major, minor)),
            _ => None,
        }
    }
}

/// Attestation evidence from GPU.
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

impl GpuAttestationEvidence {
    /// Check if the evidence has an attestation report.
    pub fn has_report(&self) -> bool {
        !self.attestation_report.is_empty()
    }

    /// Check if the evidence has a certificate chain.
    pub fn has_cert_chain(&self) -> bool {
        !self.cert_chain.is_empty()
    }
}

#[cfg(test)]
mod tests {
    use super::*;

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
    fn test_supports_cc() {
        let h100 = GpuDeviceInfo {
            name: "NVIDIA H100 PCIe".to_string(),
            uuid: String::new(),
            memory: 0,
            major: None,
            minor: None,
            clock_rate: 0.0,
            ecc: None,
            model_short_ref: String::new(),
            cc_mode_enabled: None,
        };
        assert!(h100.supports_cc());

        let a100 = GpuDeviceInfo {
            name: "NVIDIA A100".to_string(),
            ..h100.clone()
        };
        assert!(!a100.supports_cc());
    }

    #[test]
    fn test_compute_capability() {
        let info = GpuDeviceInfo {
            major: Some(9),
            minor: Some(0),
            uuid: String::new(),
            name: String::new(),
            memory: 0,
            clock_rate: 0.0,
            ecc: None,
            model_short_ref: String::new(),
            cc_mode_enabled: None,
        };
        assert_eq!(info.compute_capability(), Some("9.0".to_string()));

        let no_cc = GpuDeviceInfo {
            major: None,
            minor: None,
            ..info
        };
        assert_eq!(no_cc.compute_capability(), None);
    }

    #[test]
    fn test_evidence_has_report() {
        let evidence = GpuAttestationEvidence {
            gpu_uuid: "GPU-123".to_string(),
            attestation_report: "deadbeef".to_string(),
            signature: String::new(),
            cert_chain: vec![],
            nonce: String::new(),
            gpu_model: String::new(),
            driver_version: String::new(),
            timestamp: chrono::Utc::now(),
        };
        assert!(evidence.has_report());

        let no_report = GpuAttestationEvidence {
            attestation_report: String::new(),
            ..evidence
        };
        assert!(!no_report.has_report());
    }
}
