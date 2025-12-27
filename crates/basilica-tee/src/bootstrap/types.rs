//! Bootstrap Types
//!
//! Types used for TEE bootstrap results and configuration.

use serde::{Deserialize, Serialize};

/// TEE bootstrap configuration
#[derive(Debug, Clone, Default)]
pub struct TeeBootstrapConfig {
    /// Whether to attempt TDX setup
    pub setup_tdx: bool,
    /// Whether to attempt GPU CC setup
    pub setup_gpu_cc: bool,
    /// Timeout for setup commands (seconds)
    pub command_timeout_secs: u64,
    /// Whether to install packages (requires sudo)
    pub allow_package_install: bool,
}

/// Result of TEE bootstrap attempt
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct TeeBootstrapResult {
    /// TDX setup result
    pub tdx: Option<TdxBootstrapResult>,
    /// GPU CC setup result
    pub gpu_cc: Option<GpuCcBootstrapResult>,
    /// Overall success
    pub success: bool,
    /// Human-readable summary
    pub summary: String,
}

/// TDX bootstrap result
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct TdxBootstrapResult {
    /// Hardware supports TDX
    pub hardware_supported: bool,
    /// Intel TDX SDK/attestation tools available
    pub quote_generator_available: bool,
    /// Intel TDX SDK installed successfully
    pub sdk_installed: bool,
    /// Test quote generation succeeded
    pub test_quote_ok: bool,
    /// Error message if any
    pub error: Option<String>,
}

/// GPU CC bootstrap result
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct GpuCcBootstrapResult {
    /// GPU detected
    pub gpu_detected: bool,
    /// GPU model (e.g., "NVIDIA H100")
    pub gpu_model: Option<String>,
    /// GPU supports CC mode
    pub cc_capable: bool,
    /// CC mode currently enabled
    pub cc_mode_enabled: bool,
    /// Attestation tool available
    pub attestation_tool_available: bool,
    /// Test attestation succeeded
    pub test_attestation_ok: bool,
    /// Error message if any
    pub error: Option<String>,
}

/// GPU info parsed from check command
#[derive(Debug, Clone)]
pub struct GpuInfo {
    pub model: String,
    pub uuid: String,
    pub driver: String,
    pub cc_capable: bool,
}

/// Outputs from TDX commands
#[derive(Debug, Default)]
pub struct TdxCommandOutputs {
    pub hardware_check: String,
    pub generator_check: String,
    pub install_sdk: Option<String>,
    pub setup_qgs: Option<String>,
    pub test_quote: Option<String>,
    pub error: Option<String>,
}

/// Outputs from GPU commands
#[derive(Debug, Default)]
pub struct GpuCommandOutputs {
    pub gpu_check: String,
    pub cc_mode_check: String,
    pub attestation_check: String,
    pub install_attestation_sdk: Option<String>,
    pub test_attestation: Option<String>,
    pub error: Option<String>,
}
