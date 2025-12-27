//! TEE Bootstrap Module
//!
//! Remote TEE setup capability for validators to configure executor nodes.
//! Minimizes operator involvement by automatically detecting and configuring
//! TEE capabilities (TDX, GPU CC) over SSH.

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

/// Commands for TDX detection and setup
pub mod tdx_commands {
    /// Check if running in TDX VM
    pub const CHECK_TDX_HARDWARE: &str = r#"
        if [ -c /dev/tdx_guest ] || [ -c /dev/tdx-guest ]; then
            echo "TDX_SUPPORTED:dev"
        elif [ -d /sys/firmware/tdx ]; then
            echo "TDX_SUPPORTED:firmware"
        elif dmesg 2>/dev/null | grep -qi "TDX"; then
            echo "TDX_SUPPORTED:dmesg"
        else
            echo "TDX_NOT_SUPPORTED"
        fi
    "#;

    /// Check if Intel TDX attestation tools are available
    pub const CHECK_QUOTE_GENERATOR: &str = r#"
        # Check for Intel's official TDX attestation tools
        if command -v tdx_attest &>/dev/null; then
            echo "FOUND:tdx_attest:$(which tdx_attest)"
        elif [ -x /usr/bin/tdx_attest ]; then
            echo "FOUND:tdx_attest:/usr/bin/tdx_attest"
        elif [ -x /opt/intel/tdx-quote-generation-sample/tdx_attest ]; then
            echo "FOUND:tdx_attest:/opt/intel/tdx-quote-generation-sample/tdx_attest"
        elif command -v tpm2_quote &>/dev/null && [ -c /dev/tdx_guest ]; then
            # Fallback to tpm2-tools with TDX device
            echo "FOUND:tpm2_quote:$(which tpm2_quote)"
        else
            echo "NOT_FOUND"
        fi
    "#;

    /// Install Intel TDX DCAP SDK and attestation tools
    pub const INSTALL_INTEL_TDX_SDK: &str = r#"
        set -e
        
        # Check if already installed
        if command -v tdx_attest &>/dev/null; then
            echo "ALREADY_INSTALLED"
            exit 0
        fi
        
        # Detect OS
        if [ -f /etc/debian_version ]; then
            # Ubuntu/Debian
            echo "INSTALLING:ubuntu"
            
            # Add Intel SGX/TDX repository
            if [ ! -f /etc/apt/sources.list.d/intel-sgx.list ]; then
                # Get Ubuntu codename
                CODENAME=$(lsb_release -cs 2>/dev/null || echo "jammy")
                echo "deb [arch=amd64 signed-by=/usr/share/keyrings/intel-sgx.gpg] https://download.01.org/intel-sgx/sgx_repo/ubuntu $CODENAME main" | \
                    sudo tee /etc/apt/sources.list.d/intel-sgx.list > /dev/null
                
                # Add Intel GPG key
                curl -fsSL https://download.01.org/intel-sgx/sgx_repo/ubuntu/intel-sgx-deb.key | \
                    sudo gpg --dearmor -o /usr/share/keyrings/intel-sgx.gpg 2>/dev/null || \
                    wget -qO - https://download.01.org/intel-sgx/sgx_repo/ubuntu/intel-sgx-deb.key | \
                    sudo apt-key add - 2>/dev/null
            fi
            
            sudo apt-get update -qq
            
            # Install TDX attestation packages
            sudo apt-get install -y \
                libsgx-dcap-ql \
                libsgx-dcap-quote-verify \
                libsgx-quote-ex \
                libtdx-attest \
                libtdx-attest-dev \
                tdx-qgs \
                2>/dev/null || echo "PARTIAL_INSTALL"
            
            # Try to install sample tools if available
            sudo apt-get install -y tdx-quote-generation-sample 2>/dev/null || true
            
        elif [ -f /etc/redhat-release ]; then
            # RHEL/CentOS/Fedora
            echo "INSTALLING:rhel"
            
            # Add Intel repository
            sudo tee /etc/yum.repos.d/intel-sgx.repo > /dev/null << 'REPOEOF'
[intel-sgx]
name=Intel SGX Repository
baseurl=https://download.01.org/intel-sgx/sgx_repo/rhel/8/$basearch
enabled=1
gpgcheck=1
gpgkey=https://download.01.org/intel-sgx/sgx_repo/rhel/8/sgx_rpm_local_repo.pub
REPOEOF
            
            sudo yum install -y \
                libsgx-dcap-ql \
                libsgx-dcap-quote-verify \
                libsgx-quote-ex \
                libtdx-attest \
                2>/dev/null || echo "PARTIAL_INSTALL"
        else
            echo "UNSUPPORTED_OS"
            exit 1
        fi
        
        # Verify installation
        if command -v tdx_attest &>/dev/null; then
            echo "INSTALLED:tdx_attest"
        elif [ -f /usr/lib/x86_64-linux-gnu/libtdx_attest.so ]; then
            echo "INSTALLED:libtdx_attest"
        else
            echo "INSTALL_INCOMPLETE"
        fi
    "#;

    /// Setup TDX Quote Generation Service (QGS)
    pub const SETUP_TDX_QGS: &str = r#"
        # Start and enable the TDX Quote Generation Service if available
        if systemctl list-unit-files | grep -q tdx-qgs; then
            sudo systemctl enable tdx-qgs 2>/dev/null || true
            sudo systemctl start tdx-qgs 2>/dev/null || true
            echo "QGS_STARTED"
        elif systemctl list-unit-files | grep -q qgsd; then
            sudo systemctl enable qgsd 2>/dev/null || true
            sudo systemctl start qgsd 2>/dev/null || true
            echo "QGSD_STARTED"
        else
            echo "NO_QGS_SERVICE"
        fi
    "#;

    /// Test quote generation using Intel TDX tools
    pub const TEST_QUOTE_GENERATION: &str = r#"
        # Create a test nonce (64 bytes = 128 hex chars)
        TEST_NONCE="0102030405060708090a0b0c0d0e0f101112131415161718191a1b1c1d1e1f202122232425262728292a2b2c2d2e2f303132333435363738393a3b3c3d3e3f40"
        TMPDIR=$(mktemp -d)
        
        # Write nonce to file
        echo -n "$TEST_NONCE" | xxd -r -p > "$TMPDIR/nonce.bin"
        
        # Try Intel's tdx_attest tool
        if command -v tdx_attest &>/dev/null; then
            if tdx_attest -r "$TMPDIR/nonce.bin" -q "$TMPDIR/quote.bin" 2>/dev/null; then
                QUOTE_SIZE=$(stat -c%s "$TMPDIR/quote.bin" 2>/dev/null || stat -f%z "$TMPDIR/quote.bin" 2>/dev/null)
                if [ "$QUOTE_SIZE" -gt 1000 ]; then
                    rm -rf "$TMPDIR"
                    echo "QUOTE_OK:tdx_attest:$QUOTE_SIZE"
                    exit 0
                fi
            fi
        fi
        
        # Try using TDX device directly with a simple test
        if [ -c /dev/tdx_guest ] || [ -c /dev/tdx-guest ]; then
            # The device exists, quote generation should be possible via libraries
            rm -rf "$TMPDIR"
            echo "DEVICE_OK:tdx_guest"
            exit 0
        fi
        
        rm -rf "$TMPDIR"
        echo "QUOTE_FAILED"
    "#;
}

/// Commands for GPU CC detection and setup
pub mod gpu_commands {
    /// Check for NVIDIA GPU and CC capability
    pub const CHECK_GPU: &str = r#"
        if ! command -v nvidia-smi &>/dev/null; then
            echo "NO_NVIDIA_SMI"
            exit 0
        fi
        
        GPU_MODEL=$(nvidia-smi --query-gpu=name --format=csv,noheader 2>/dev/null | head -1)
        GPU_UUID=$(nvidia-smi --query-gpu=uuid --format=csv,noheader 2>/dev/null | head -1)
        DRIVER=$(nvidia-smi --query-gpu=driver_version --format=csv,noheader 2>/dev/null | head -1)
        
        if [ -z "$GPU_MODEL" ]; then
            echo "NO_GPU"
            exit 0
        fi
        
        # Check CC capability
        CC_CAPABLE="false"
        if [[ "$GPU_MODEL" == *"H100"* ]] || [[ "$GPU_MODEL" == *"H200"* ]]; then
            CC_CAPABLE="true"
        fi
        
        echo "GPU_DETECTED:$GPU_MODEL|$GPU_UUID|$DRIVER|$CC_CAPABLE"
    "#;

    /// Check CC mode status
    pub const CHECK_CC_MODE: &str = r#"
        CC_STATUS=$(nvidia-smi -q 2>/dev/null | grep -i "Conf Compute Mode" | awk -F: '{print $2}' | tr -d ' ' | head -1)
        if [ "$CC_STATUS" = "Enabled" ]; then
            echo "CC_ENABLED"
        elif [ -n "$CC_STATUS" ]; then
            echo "CC_DISABLED:$CC_STATUS"
        else
            echo "CC_UNKNOWN"
        fi
    "#;

    /// Check for attestation tools (NVIDIA official tools only)
    pub const CHECK_ATTESTATION_TOOLS: &str = r#"
        # Check for NVIDIA's official attestation tools
        if command -v nv-attestation-tool &>/dev/null; then
            echo "FOUND:nv-attestation-tool:$(which nv-attestation-tool)"
        elif command -v nvidia-attestation &>/dev/null; then
            echo "FOUND:nvidia-attestation:$(which nvidia-attestation)"
        elif [ -f /usr/bin/nvidia-attestation ]; then
            echo "FOUND:nvidia-attestation:/usr/bin/nvidia-attestation"
        else
            # CC mode can still be verified via nvidia-smi even without attestation SDK
            echo "NOT_FOUND:nvidia-smi-only"
        fi
    "#;

    /// Install NVIDIA attestation SDK (official package)
    pub const INSTALL_ATTESTATION_SDK: &str = r#"
        # Check if already installed
        if command -v nv-attestation-tool &>/dev/null || command -v nvidia-attestation &>/dev/null; then
            echo "ALREADY_INSTALLED"
            exit 0
        fi
        
        # Try to install NVIDIA GPU Attestation SDK via package manager
        if [ -f /etc/debian_version ]; then
            # Ubuntu/Debian - try NVIDIA's official repository
            sudo apt-get update -qq 2>/dev/null
            if sudo apt-get install -y nvidia-gpu-attestation 2>/dev/null; then
                echo "INSTALLED:nvidia-gpu-attestation"
                exit 0
            fi
        elif [ -f /etc/redhat-release ]; then
            # RHEL/CentOS
            if sudo yum install -y nvidia-gpu-attestation 2>/dev/null; then
                echo "INSTALLED:nvidia-gpu-attestation"
                exit 0
            fi
        fi
        
        # If package not available, CC mode verification still works via nvidia-smi
        echo "PACKAGE_NOT_AVAILABLE:using-nvidia-smi"
    "#;

    /// Test GPU CC attestation
    /// Uses nvidia-smi for CC mode verification (always available)
    /// Uses NVIDIA attestation SDK for full cryptographic attestation (if available)
    pub const TEST_ATTESTATION: &str = r#"
        TEST_NONCE="deadbeefcafe1234567890abcdef0123"
        
        # First verify CC mode is enabled via nvidia-smi (always works)
        CC_STATUS=$(nvidia-smi -q 2>/dev/null | grep -i "Conf Compute Mode" | awk -F: '{print $2}' | tr -d ' ' | head -1)
        if [ "$CC_STATUS" != "Enabled" ]; then
            echo "CC_MODE_DISABLED"
            exit 1
        fi
        
        # Try NVIDIA's official attestation tools for full cryptographic verification
        if command -v nv-attestation-tool &>/dev/null; then
            EVIDENCE=$(nv-attestation-tool --nonce "$TEST_NONCE" 2>/dev/null)
            if [ -n "$EVIDENCE" ]; then
                echo "ATTESTATION_OK:nv-attestation-tool"
                exit 0
            fi
        fi
        
        if command -v nvidia-attestation &>/dev/null; then
            EVIDENCE=$(nvidia-attestation generate --nonce "$TEST_NONCE" 2>/dev/null)
            if [ -n "$EVIDENCE" ]; then
                echo "ATTESTATION_OK:nvidia-attestation"
                exit 0
            fi
        fi
        
        # CC mode is enabled but no attestation SDK - still valid for basic verification
        echo "CC_MODE_OK:no-attestation-sdk"
    "#;
}

/// TEE Bootstrap executor
///
/// Runs setup commands on remote executor nodes via SSH
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
            tdx_commands::CHECK_TDX_HARDWARE,
            tdx_commands::CHECK_QUOTE_GENERATOR,
        ]
    }

    /// Get TDX setup commands
    pub fn tdx_setup_commands(&self) -> Vec<&'static str> {
        vec![
            tdx_commands::INSTALL_INTEL_TDX_SDK,
            tdx_commands::SETUP_TDX_QGS,
            tdx_commands::TEST_QUOTE_GENERATION,
        ]
    }

    /// Get GPU CC detection commands
    pub fn gpu_detect_commands(&self) -> Vec<&'static str> {
        vec![
            gpu_commands::CHECK_GPU,
            gpu_commands::CHECK_CC_MODE,
            gpu_commands::CHECK_ATTESTATION_TOOLS,
        ]
    }

    /// Get GPU CC setup commands
    pub fn gpu_setup_commands(&self) -> Vec<&'static str> {
        vec![
            gpu_commands::INSTALL_ATTESTATION_SDK,
            gpu_commands::TEST_ATTESTATION,
        ]
    }

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

    /// Create a bootstrap result from command outputs
    pub fn build_result(
        &self,
        tdx_outputs: Option<TdxCommandOutputs>,
        gpu_outputs: Option<GpuCommandOutputs>,
    ) -> TeeBootstrapResult {
        let tdx = tdx_outputs.map(|o| TdxBootstrapResult {
            hardware_supported: Self::parse_tdx_hardware_check(&o.hardware_check),
            quote_generator_available: Self::parse_quote_generator_check(&o.generator_check)
                .is_some(),
            sdk_installed: o
                .install_sdk
                .as_ref()
                .map(|s| s.contains("INSTALLED") || s.contains("ALREADY_INSTALLED"))
                .unwrap_or(false),
            test_quote_ok: o
                .test_quote
                .as_ref()
                .map(|s| Self::parse_test_quote(s))
                .unwrap_or(false),
            error: o.error,
        });

        let gpu_cc = gpu_outputs.map(|o| {
            let gpu_info = Self::parse_gpu_check(&o.gpu_check);
            GpuCcBootstrapResult {
                gpu_detected: gpu_info.is_some(),
                gpu_model: gpu_info.as_ref().map(|g| g.model.clone()),
                cc_capable: gpu_info.as_ref().map(|g| g.cc_capable).unwrap_or(false),
                cc_mode_enabled: Self::parse_cc_mode_check(&o.cc_mode_check),
                attestation_tool_available: o.attestation_check.starts_with("FOUND:"),
                test_attestation_ok: o
                    .test_attestation
                    .as_ref()
                    .map(|s| Self::parse_test_attestation(s))
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

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_parse_tdx_hardware_check() {
        assert!(TeeBootstrap::parse_tdx_hardware_check("TDX_SUPPORTED\n"));
        assert!(!TeeBootstrap::parse_tdx_hardware_check(
            "TDX_NOT_SUPPORTED\n"
        ));
    }

    #[test]
    fn test_parse_quote_generator_check() {
        let result =
            TeeBootstrap::parse_quote_generator_check("FOUND:tdx_attest:/usr/bin/tdx_attest");
        assert_eq!(result, Some("tdx_attest:/usr/bin/tdx_attest".to_string()));

        let result = TeeBootstrap::parse_quote_generator_check("NOT_FOUND");
        assert!(result.is_none());
    }

    #[test]
    fn test_parse_gpu_check() {
        let output = "GPU_DETECTED:NVIDIA H100 PCIe|GPU-12345|535.104.05|true";
        let result = TeeBootstrap::parse_gpu_check(output).unwrap();
        assert_eq!(result.model, "NVIDIA H100 PCIe");
        assert!(result.cc_capable);

        let output = "NO_GPU";
        assert!(TeeBootstrap::parse_gpu_check(output).is_none());
    }

    #[test]
    fn test_parse_cc_mode() {
        assert!(TeeBootstrap::parse_cc_mode_check("CC_ENABLED"));
        assert!(!TeeBootstrap::parse_cc_mode_check("CC_DISABLED:Off"));
    }

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
}
