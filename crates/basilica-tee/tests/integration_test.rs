//! Integration tests for basilica-tee crate
//!
//! These tests verify the complete TEE attestation flow.

use basilica_tee::{
    config::{GpuCcConfig, TdxConfig, TeeConfig},
    gpu::GpuEvidenceParser,
    tdx::TdxQuoteVerifier,
    types::ExpectedMeasurements,
};
use std::path::PathBuf;

mod quote_fixtures {
    /// Create a valid TDX Quote V4 structure for testing
    pub fn create_test_quote(mrtd: [u8; 48], rtmr0: [u8; 48], report_data: [u8; 64]) -> Vec<u8> {
        let mut quote = vec![0u8; 700];

        // Header (48 bytes)
        // Version = 4
        quote[0..2].copy_from_slice(&4u16.to_le_bytes());
        // TEE Type = TDX (0x81)
        quote[4..6].copy_from_slice(&0x0081u16.to_le_bytes());

        // TD Report starts at offset 48
        let report_offset = 48;

        // MRTD at offset 136 within report
        quote[report_offset + 136..report_offset + 184].copy_from_slice(&mrtd);

        // RTMR0 at offset 328 within report
        quote[report_offset + 328..report_offset + 376].copy_from_slice(&rtmr0);

        // Report data at offset 520 within report
        quote[report_offset + 520..report_offset + 584].copy_from_slice(&report_data);

        quote
    }
}

/// TDX quote parsing and verification tests
mod tdx_integration {
    use super::*;
    use quote_fixtures::create_test_quote;

    #[test]
    fn test_quote_parse_and_verify_success() {
        let mrtd = [0xAAu8; 48];
        let rtmr0 = [0xBBu8; 48];
        let mut report_data = [0u8; 64];
        let nonce = b"test_nonce_12345678901234567890";
        report_data[..nonce.len()].copy_from_slice(nonce);

        let quote = create_test_quote(mrtd, rtmr0, report_data);

        // Parse
        let parsed = basilica_tee::tdx::TdxQuoteV4::parse(&quote).unwrap();

        // Verify measurements
        assert_eq!(parsed.mrtd(), &mrtd);
        assert_eq!(parsed.rtmrs()[0], rtmr0);
        assert!(parsed.verify_nonce(nonce));
    }

    #[test]
    fn test_quote_verifier_with_expected_measurements() {
        let mrtd = [0xAAu8; 48];
        let rtmr0 = [0xBBu8; 48];
        let report_data = [0u8; 64];

        let quote = create_test_quote(mrtd, rtmr0, report_data);

        let expected = ExpectedMeasurements {
            mrtd: Some(mrtd),
            rtmr0: Some(rtmr0),
            ..Default::default()
        };

        let verifier = TdxQuoteVerifier::new(expected);
        let result = verifier.verify(&quote, None).unwrap();

        assert!(result.quote_valid);
        assert!(result.mrtd_matches);
        assert!(result.rtmr_matches[0]);
    }

    #[test]
    fn test_quote_verifier_mrtd_mismatch() {
        let mrtd = [0xAAu8; 48];
        let quote = create_test_quote(mrtd, [0u8; 48], [0u8; 64]);

        let expected = ExpectedMeasurements {
            mrtd: Some([0xFFu8; 48]), // Different!
            ..Default::default()
        };

        let verifier = TdxQuoteVerifier::new(expected);
        let result = verifier.verify(&quote, None).unwrap();

        assert!(!result.mrtd_matches, "MRTD should not match");
    }
}

/// GPU attestation tests
mod gpu_integration {
    use super::*;

    #[test]
    fn test_gpu_evidence_parse_and_verify() {
        let json = serde_json::json!([{
            "gpu_uuid": "GPU-test-123",
            "attestation_report": "report_data",
            "signature": "sig",
            "cert_chain": ["cert"],
            "nonce": "my_nonce",
            "gpu_model": "NVIDIA H100"
        }])
        .to_string();

        let evidence = GpuEvidenceParser::parse(&json).unwrap();

        assert_eq!(evidence.len(), 1);
        assert_eq!(evidence[0].gpu_uuid, "GPU-test-123");
        assert_eq!(evidence[0].nonce, "my_nonce");
    }

    #[test]
    fn test_gpu_evidence_verify_nonce() {
        let json = serde_json::json!({
            "gpu_uuid": "GPU-1",
            "nonce": "expected_nonce",
            "attestation_report": "report"
        })
        .to_string();

        let evidence = GpuEvidenceParser::parse(&json).unwrap();

        // Correct nonce
        let result = GpuEvidenceParser::verify(&evidence[0], Some("expected_nonce")).unwrap();
        assert!(result.nonce_verified);

        // Wrong nonce
        let result = GpuEvidenceParser::verify(&evidence[0], Some("wrong_nonce")).unwrap();
        assert!(!result.nonce_verified);
    }

    #[test]
    fn test_gpu_cc_mode_detection() {
        // With attestation report = CC mode
        let with_report = serde_json::json!({
            "gpu_uuid": "GPU-1",
            "attestation_report": "some_report"
        })
        .to_string();

        let evidence = GpuEvidenceParser::parse(&with_report).unwrap();
        let result = GpuEvidenceParser::verify(&evidence[0], None).unwrap();
        assert!(result.cc_mode_enabled);

        // Without attestation report = no CC mode
        let without_report = serde_json::json!({
            "gpu_uuid": "GPU-1",
            "attestation_report": ""
        })
        .to_string();

        let evidence = GpuEvidenceParser::parse(&without_report).unwrap();
        let result = GpuEvidenceParser::verify(&evidence[0], None).unwrap();
        assert!(!result.cc_mode_enabled);
    }
}

/// Configuration tests
mod config_integration {
    use super::*;

    #[test]
    fn test_tee_config_roundtrip() {
        let config = TeeConfig {
            enabled: true,
            require_tee: false,
            tdx: TdxConfig {
                quote_generator_path: PathBuf::from("/usr/bin/tdx-quote"),
                expected_mrtd: Some("aa".repeat(48)),
                ..Default::default()
            },
            gpu: GpuCcConfig {
                require_cc_mode: false,
                allowed_models: vec!["H100".to_string(), "H200".to_string()],
                ..Default::default()
            },
            attestation_server: None,
        };

        // Serialize
        let json = serde_json::to_string(&config).unwrap();

        // Deserialize
        let parsed: TeeConfig = serde_json::from_str(&json).unwrap();

        assert!(parsed.enabled);
        assert!(!parsed.require_tee);
        assert!(!parsed.gpu.require_cc_mode);
        assert_eq!(parsed.gpu.allowed_models.len(), 2);
    }

    #[test]
    fn test_expected_measurements_from_config() {
        let config = TdxConfig {
            expected_mrtd: Some("aa".repeat(48)),
            expected_rtmr0: Some("bb".repeat(48)),
            ..Default::default()
        };

        let measurements = ExpectedMeasurements::from_config(&config);

        assert_eq!(measurements.mrtd, Some([0xAAu8; 48]));
        assert_eq!(measurements.rtmr0, Some([0xBBu8; 48]));
        assert!(measurements.rtmr1.is_none());
    }

    #[test]
    fn test_default_config() {
        let config = TeeConfig::default();

        assert!(!config.enabled);
        assert!(!config.require_tee);
    }
}

/// Server endpoint tests (when server feature is enabled)
#[cfg(feature = "server")]
mod server_integration {
    use super::*;

    #[tokio::test]
    async fn test_server_config() {
        let config = TeeConfig::default();

        // Verify default config is valid
        assert!(!config.enabled);
    }
}
