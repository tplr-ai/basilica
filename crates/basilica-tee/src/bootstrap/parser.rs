//! Bootstrap Output Parsers
//!
//! Parsers for interpreting command output from bootstrap scripts.

use super::types::GpuInfo;

/// Parse TDX hardware check output
pub fn parse_tdx_hardware_check(output: &str) -> bool {
    output.contains("TDX_SUPPORTED")
}

/// Parse quote generator check output
pub fn parse_quote_generator_check(output: &str) -> Option<String> {
    if output.starts_with("FOUND:") {
        Some(output.trim_start_matches("FOUND:").trim().to_string())
    } else {
        None
    }
}

/// Parse GPU check output
pub fn parse_gpu_check(output: &str) -> Option<GpuInfo> {
    if output.starts_with("GPU_DETECTED:") {
        let data = output.trim_start_matches("GPU_DETECTED:");
        let parts: Vec<&str> = data.split('|').collect();
        if parts.len() >= 4 {
            return Some(GpuInfo {
                model: parts[0].to_string(),
                uuid: parts[1].to_string(),
                driver: parts[2].to_string(),
                cc_capable: parts[3] == "true",
            });
        }
    }
    None
}

/// Parse CC mode check output
pub fn parse_cc_mode_check(output: &str) -> bool {
    output.contains("CC_ENABLED")
}

/// Parse test quote output
pub fn parse_test_quote(output: &str) -> bool {
    output.starts_with("QUOTE_OK:") || output.starts_with("DEVICE_OK:")
}

/// Parse test attestation output
/// Returns true if CC mode is verified (with or without full attestation SDK)
pub fn parse_test_attestation(output: &str) -> bool {
    output.starts_with("ATTESTATION_OK:") || output.starts_with("CC_MODE_OK:")
}

/// Parse SDK install output
pub fn parse_sdk_install(output: &str) -> bool {
    output.contains("INSTALLED") || output.contains("ALREADY_INSTALLED")
}

/// Parse attestation tool check output
pub fn parse_attestation_tool_check(output: &str) -> bool {
    output.starts_with("FOUND:")
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_parse_tdx_hardware_check() {
        assert!(parse_tdx_hardware_check("TDX_SUPPORTED:dev"));
        assert!(parse_tdx_hardware_check("TDX_SUPPORTED:firmware"));
        assert!(!parse_tdx_hardware_check("TDX_NOT_SUPPORTED"));
    }

    #[test]
    fn test_parse_quote_generator_check() {
        let result = parse_quote_generator_check("FOUND:tdx_attest:/usr/bin/tdx_attest");
        assert_eq!(result, Some("tdx_attest:/usr/bin/tdx_attest".to_string()));

        let result = parse_quote_generator_check("NOT_FOUND");
        assert!(result.is_none());
    }

    #[test]
    fn test_parse_gpu_check() {
        let output = "GPU_DETECTED:NVIDIA H100 PCIe|GPU-12345|535.104.05|true";
        let result = parse_gpu_check(output).unwrap();
        assert_eq!(result.model, "NVIDIA H100 PCIe");
        assert!(result.cc_capable);

        let output = "NO_GPU";
        assert!(parse_gpu_check(output).is_none());
    }

    #[test]
    fn test_parse_cc_mode() {
        assert!(parse_cc_mode_check("CC_ENABLED"));
        assert!(!parse_cc_mode_check("CC_DISABLED:Off"));
    }

    #[test]
    fn test_parse_test_quote() {
        assert!(parse_test_quote("QUOTE_OK:tdx_attest:4096"));
        assert!(parse_test_quote("DEVICE_OK:tdx_guest"));
        assert!(!parse_test_quote("QUOTE_FAILED"));
    }

    #[test]
    fn test_parse_test_attestation() {
        assert!(parse_test_attestation("ATTESTATION_OK:nv-attestation-tool"));
        assert!(parse_test_attestation("CC_MODE_OK:no-attestation-sdk"));
        assert!(!parse_test_attestation("CC_MODE_DISABLED"));
    }
}
