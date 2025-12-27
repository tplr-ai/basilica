//! TEE Bootstrap Module
//!
//! Remote TEE setup capability for validators to configure executor nodes.
//! Minimizes operator involvement by automatically detecting and configuring
//! TEE capabilities (TDX, GPU CC) over SSH.
//!
//! ## Module Structure
//!
//! - [`commands`]: Shell command strings for remote execution
//! - [`parser`]: Output parsers for command results
//! - [`types`]: Configuration and result types

pub mod commands;
pub mod parser;
pub mod types;

// Re-export types for convenience
pub use types::{
    GpuCcBootstrapResult, GpuCommandOutputs, GpuInfo, TdxBootstrapResult, TdxCommandOutputs,
    TeeBootstrapConfig, TeeBootstrapResult,
};

// Re-export command modules for backward compatibility
pub use commands::gpu as gpu_commands;
pub use commands::tdx as tdx_commands;

/// TEE Bootstrap executor
///
/// Runs setup commands on remote executor nodes via SSH.
/// Provides methods for detecting and configuring TEE capabilities.
pub struct TeeBootstrap {
    #[allow(dead_code)]
    config: TeeBootstrapConfig,
}

impl TeeBootstrap {
    /// Create new bootstrap executor with config
    pub fn new(config: TeeBootstrapConfig) -> Self {
        Self { config }
    }

    /// Create with default config (all features enabled)
    pub fn default_config() -> Self {
        Self::new(TeeBootstrapConfig {
            setup_tdx: true,
            setup_gpu_cc: true,
            command_timeout_secs: 120,
            allow_package_install: false,
        })
    }

    /// Get TDX detection commands
    pub fn tdx_detect_commands(&self) -> Vec<&'static str> {
        vec![
            commands::tdx::CHECK_TDX_HARDWARE,
            commands::tdx::CHECK_QUOTE_GENERATOR,
        ]
    }

    /// Get TDX setup commands
    pub fn tdx_setup_commands(&self) -> Vec<&'static str> {
        vec![
            commands::tdx::INSTALL_INTEL_TDX_SDK,
            commands::tdx::SETUP_TDX_QGS,
            commands::tdx::TEST_QUOTE_GENERATION,
        ]
    }

    /// Get GPU CC detection commands
    pub fn gpu_detect_commands(&self) -> Vec<&'static str> {
        vec![
            commands::gpu::CHECK_GPU,
            commands::gpu::CHECK_CC_MODE,
            commands::gpu::CHECK_ATTESTATION_TOOLS,
        ]
    }

    /// Get GPU CC setup commands
    pub fn gpu_setup_commands(&self) -> Vec<&'static str> {
        vec![
            commands::gpu::INSTALL_ATTESTATION_SDK,
            commands::gpu::TEST_ATTESTATION,
        ]
    }

    /// Parse TDX hardware check output (deprecated, use parser module)
    #[deprecated(note = "Use parser::parse_tdx_hardware_check instead")]
    pub fn parse_tdx_hardware_check(output: &str) -> bool {
        parser::parse_tdx_hardware_check(output)
    }

    /// Parse quote generator check output (deprecated, use parser module)
    #[deprecated(note = "Use parser::parse_quote_generator_check instead")]
    pub fn parse_quote_generator_check(output: &str) -> Option<String> {
        parser::parse_quote_generator_check(output)
    }

    /// Parse GPU check output (deprecated, use parser module)
    #[deprecated(note = "Use parser::parse_gpu_check instead")]
    pub fn parse_gpu_check(output: &str) -> Option<GpuInfo> {
        parser::parse_gpu_check(output)
    }

    /// Parse CC mode check output (deprecated, use parser module)
    #[deprecated(note = "Use parser::parse_cc_mode_check instead")]
    pub fn parse_cc_mode_check(output: &str) -> bool {
        parser::parse_cc_mode_check(output)
    }

    /// Parse test quote output (deprecated, use parser module)
    #[deprecated(note = "Use parser::parse_test_quote instead")]
    pub fn parse_test_quote(output: &str) -> bool {
        parser::parse_test_quote(output)
    }

    /// Parse test attestation output (deprecated, use parser module)
    #[deprecated(note = "Use parser::parse_test_attestation instead")]
    pub fn parse_test_attestation(output: &str) -> bool {
        parser::parse_test_attestation(output)
    }

    /// Create a bootstrap result from command outputs
    pub fn build_result(
        &self,
        tdx_outputs: Option<TdxCommandOutputs>,
        gpu_outputs: Option<GpuCommandOutputs>,
    ) -> TeeBootstrapResult {
        let tdx = tdx_outputs.map(|o| TdxBootstrapResult {
            hardware_supported: parser::parse_tdx_hardware_check(&o.hardware_check),
            quote_generator_available: parser::parse_quote_generator_check(&o.generator_check)
                .is_some(),
            sdk_installed: o
                .install_sdk
                .as_ref()
                .map(|s| parser::parse_sdk_install(s))
                .unwrap_or(false),
            test_quote_ok: o
                .test_quote
                .as_ref()
                .map(|s| parser::parse_test_quote(s))
                .unwrap_or(false),
            error: o.error,
        });

        let gpu_cc = gpu_outputs.map(|o| {
            let gpu_info = parser::parse_gpu_check(&o.gpu_check);
            GpuCcBootstrapResult {
                gpu_detected: gpu_info.is_some(),
                gpu_model: gpu_info.as_ref().map(|g| g.model.clone()),
                cc_capable: gpu_info.as_ref().map(|g| g.cc_capable).unwrap_or(false),
                cc_mode_enabled: parser::parse_cc_mode_check(&o.cc_mode_check),
                attestation_tool_available: parser::parse_attestation_tool_check(
                    &o.attestation_check,
                ),
                test_attestation_ok: o
                    .test_attestation
                    .as_ref()
                    .map(|s| parser::parse_test_attestation(s))
                    .unwrap_or(false),
                error: o.error,
            }
        });

        let tdx_ok = tdx
            .as_ref()
            .map(|t| t.hardware_supported && t.test_quote_ok)
            .unwrap_or(true);
        let gpu_ok = gpu_cc
            .as_ref()
            .map(|g| !g.cc_capable || g.test_attestation_ok)
            .unwrap_or(true);

        let success = tdx_ok && gpu_ok;

        let summary = Self::build_summary(&tdx, &gpu_cc);

        TeeBootstrapResult {
            tdx,
            gpu_cc,
            success,
            summary,
        }
    }

    fn build_summary(
        tdx: &Option<TdxBootstrapResult>,
        gpu: &Option<GpuCcBootstrapResult>,
    ) -> String {
        let mut parts = Vec::new();

        if let Some(t) = tdx {
            if t.hardware_supported {
                if t.test_quote_ok {
                    parts.push("TDX: ready".to_string());
                } else {
                    parts.push("TDX: hardware ok, quote generation failed".to_string());
                }
            } else {
                parts.push("TDX: not supported".to_string());
            }
        }

        if let Some(g) = gpu {
            if g.gpu_detected {
                let model = g.gpu_model.as_deref().unwrap_or("unknown");
                if g.cc_capable {
                    if g.cc_mode_enabled && g.test_attestation_ok {
                        parts.push(format!("GPU CC: ready ({})", model));
                    } else if g.cc_mode_enabled {
                        parts.push(format!(
                            "GPU CC: enabled but attestation failed ({})",
                            model
                        ));
                    } else {
                        parts.push(format!("GPU CC: capable but not enabled ({})", model));
                    }
                } else {
                    parts.push(format!("GPU CC: not capable ({})", model));
                }
            } else {
                parts.push("GPU: not detected".to_string());
            }
        }

        if parts.is_empty() {
            "No TEE capabilities detected".to_string()
        } else {
            parts.join("; ")
        }
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_build_summary() {
        let tdx = TdxBootstrapResult {
            hardware_supported: true,
            quote_generator_available: true,
            sdk_installed: true,
            test_quote_ok: true,
            error: None,
        };

        let gpu = GpuCcBootstrapResult {
            gpu_detected: true,
            gpu_model: Some("NVIDIA H100".to_string()),
            cc_capable: true,
            cc_mode_enabled: true,
            attestation_tool_available: true,
            test_attestation_ok: true,
            error: None,
        };

        let summary = TeeBootstrap::build_summary(&Some(tdx), &Some(gpu));
        assert!(summary.contains("TDX: ready"));
        assert!(summary.contains("GPU CC: ready"));
    }

    #[test]
    fn test_build_result() {
        let bootstrap = TeeBootstrap::default_config();

        let tdx_outputs = TdxCommandOutputs {
            hardware_check: "TDX_SUPPORTED:dev".to_string(),
            generator_check: "FOUND:tdx_attest:/usr/bin/tdx_attest".to_string(),
            install_sdk: Some("INSTALLED:tdx_attest".to_string()),
            test_quote: Some("QUOTE_OK:tdx_attest:4096".to_string()),
            ..Default::default()
        };

        let gpu_outputs = GpuCommandOutputs {
            gpu_check: "GPU_DETECTED:NVIDIA H100|GPU-123|535.0|true".to_string(),
            cc_mode_check: "CC_ENABLED".to_string(),
            attestation_check: "FOUND:nv-attestation-tool:/usr/bin/nv-attestation-tool".to_string(),
            test_attestation: Some("ATTESTATION_OK:nv-attestation-tool".to_string()),
            ..Default::default()
        };

        let result = bootstrap.build_result(Some(tdx_outputs), Some(gpu_outputs));
        assert!(result.success);
        assert!(result.tdx.as_ref().unwrap().test_quote_ok);
        assert!(result.tdx.as_ref().unwrap().sdk_installed);
        assert!(result.gpu_cc.as_ref().unwrap().test_attestation_ok);
    }

    #[test]
    fn test_detect_commands() {
        let bootstrap = TeeBootstrap::default_config();

        let tdx_cmds = bootstrap.tdx_detect_commands();
        assert_eq!(tdx_cmds.len(), 2);

        let gpu_cmds = bootstrap.gpu_detect_commands();
        assert_eq!(gpu_cmds.len(), 3);
    }

    #[test]
    fn test_setup_commands() {
        let bootstrap = TeeBootstrap::default_config();

        let tdx_cmds = bootstrap.tdx_setup_commands();
        assert_eq!(tdx_cmds.len(), 3);

        let gpu_cmds = bootstrap.gpu_setup_commands();
        assert_eq!(gpu_cmds.len(), 2);
    }
}
