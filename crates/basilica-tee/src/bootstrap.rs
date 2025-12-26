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
    /// Quote generator available
    pub quote_generator_available: bool,
    /// configfs-tsm configured
    pub configfs_configured: bool,
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
        if [ -d /sys/firmware/tdx ] || [ -c /dev/tdx_guest ]; then
            echo "TDX_SUPPORTED"
        elif dmesg 2>/dev/null | grep -qi "TDX"; then
            echo "TDX_SUPPORTED"
        else
            echo "TDX_NOT_SUPPORTED"
        fi
    "#;

    /// Check if quote generator is available
    pub const CHECK_QUOTE_GENERATOR: &str = r#"
        if command -v tdx-quote-generator &>/dev/null; then
            echo "FOUND:$(which tdx-quote-generator)"
        elif [ -x /usr/local/bin/tdx-quote-generator ]; then
            echo "FOUND:/usr/local/bin/tdx-quote-generator"
        elif [ -d /sys/kernel/config/tsm/report ]; then
            echo "FOUND:configfs-tsm"
        else
            echo "NOT_FOUND"
        fi
    "#;

    /// Setup configfs-tsm for quote generation
    pub const SETUP_CONFIGFS_TSM: &str = r#"
        set -e
        # Mount configfs if needed
        if ! mountpoint -q /sys/kernel/config 2>/dev/null; then
            sudo mount -t configfs none /sys/kernel/config 2>/dev/null || true
        fi
        # Create TSM report entry
        sudo mkdir -p /sys/kernel/config/tsm/report/tdx0 2>/dev/null || true
        # Set permissions
        sudo chmod 755 /sys/kernel/config/tsm/report/tdx0 2>/dev/null || true
        sudo chmod 666 /sys/kernel/config/tsm/report/tdx0/inblob 2>/dev/null || true
        sudo chmod 444 /sys/kernel/config/tsm/report/tdx0/outblob 2>/dev/null || true
        echo "CONFIGFS_TSM_OK"
    "#;

    /// Install TDX quote generation tool (minimal C program)
    pub const INSTALL_QUOTE_GENERATOR: &str = r#"
        set -e
        if command -v tdx-quote-generator &>/dev/null; then
            echo "ALREADY_INSTALLED"
            exit 0
        fi
        
        # Check for compiler
        if ! command -v gcc &>/dev/null; then
            echo "NEED_GCC"
            exit 1
        fi
        
        # Create minimal quote generator
        cat > /tmp/tdx_quote_gen.c << 'CEOF'
#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include <fcntl.h>
#include <unistd.h>

#define REPORT_DATA_SIZE 64
#define QUOTE_MAX_SIZE 8192

int main(int argc, char *argv[]) {
    char *report_data_hex = NULL;
    char *output_file = NULL;
    int output_hex = 0;
    
    for (int i = 1; i < argc; i++) {
        if (strcmp(argv[i], "--report-data") == 0 && i + 1 < argc) {
            report_data_hex = argv[++i];
        } else if (strcmp(argv[i], "--output") == 0 && i + 1 < argc) {
            output_file = argv[++i];
        } else if (strcmp(argv[i], "--hex") == 0) {
            output_hex = 1;
        }
    }
    
    const char *tsm_base = "/sys/kernel/config/tsm/report/tdx0";
    char inblob_path[256], outblob_path[256];
    snprintf(inblob_path, sizeof(inblob_path), "%s/inblob", tsm_base);
    snprintf(outblob_path, sizeof(outblob_path), "%s/outblob", tsm_base);
    
    unsigned char report_data[REPORT_DATA_SIZE] = {0};
    if (report_data_hex) {
        size_t len = strlen(report_data_hex);
        for (size_t i = 0; i < len && i < REPORT_DATA_SIZE * 2; i += 2) {
            sscanf(&report_data_hex[i], "%2hhx", &report_data[i/2]);
        }
    }
    
    int fd = open(inblob_path, O_WRONLY);
    if (fd < 0) { perror("inblob"); return 1; }
    write(fd, report_data, REPORT_DATA_SIZE);
    close(fd);
    
    fd = open(outblob_path, O_RDONLY);
    if (fd < 0) { perror("outblob"); return 1; }
    
    unsigned char quote[QUOTE_MAX_SIZE];
    ssize_t quote_size = read(fd, quote, QUOTE_MAX_SIZE);
    close(fd);
    
    if (quote_size <= 0) { fprintf(stderr, "read failed\n"); return 1; }
    
    FILE *out = output_file ? fopen(output_file, "wb") : stdout;
    if (!out) { perror("output"); return 1; }
    
    if (output_hex) {
        for (ssize_t i = 0; i < quote_size; i++) fprintf(out, "%02x", quote[i]);
        fprintf(out, "\n");
    } else {
        fwrite(quote, 1, quote_size, out);
    }
    
    if (output_file) fclose(out);
    return 0;
}
CEOF
        
        gcc -O2 -o /tmp/tdx-quote-generator /tmp/tdx_quote_gen.c
        sudo mv /tmp/tdx-quote-generator /usr/local/bin/
        sudo chmod +x /usr/local/bin/tdx-quote-generator
        rm -f /tmp/tdx_quote_gen.c
        echo "INSTALLED"
    "#;

    /// Test quote generation
    pub const TEST_QUOTE_GENERATION: &str = r#"
        TEST_NONCE="0102030405060708090a0b0c0d0e0f101112131415161718191a1b1c1d1e1f20"
        if command -v tdx-quote-generator &>/dev/null; then
            QUOTE=$(tdx-quote-generator --report-data "$TEST_NONCE" --hex 2>/dev/null)
            if [ -n "$QUOTE" ] && [ ${#QUOTE} -gt 1000 ]; then
                echo "QUOTE_OK:${#QUOTE}"
            else
                echo "QUOTE_FAILED"
            fi
        else
            echo "NO_TOOL"
        fi
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

    /// Check for attestation tools
    pub const CHECK_ATTESTATION_TOOLS: &str = r#"
        if command -v chutes-nvevidence &>/dev/null; then
            echo "FOUND:chutes-nvevidence:$(which chutes-nvevidence)"
        elif command -v nvidia-attestation &>/dev/null; then
            echo "FOUND:nvidia-attestation:$(which nvidia-attestation)"
        elif command -v nv-attestation-tool &>/dev/null; then
            echo "FOUND:nv-attestation-tool:$(which nv-attestation-tool)"
        else
            echo "NOT_FOUND"
        fi
    "#;

    /// Install chutes-nvevidence (if Rust/cargo available)
    pub const INSTALL_NVEVIDENCE: &str = r#"
        if command -v chutes-nvevidence &>/dev/null; then
            echo "ALREADY_INSTALLED"
            exit 0
        fi
        
        if ! command -v cargo &>/dev/null; then
            echo "NO_CARGO"
            exit 1
        fi
        
        # Try to install from crates.io or build from source
        cargo install chutes-nvevidence 2>/dev/null && echo "INSTALLED" && exit 0
        
        # Fallback: clone and build
        TMPDIR=$(mktemp -d)
        cd "$TMPDIR"
        git clone --depth 1 https://github.com/chutes-ai/chutes-nvevidence.git 2>/dev/null
        if [ -d chutes-nvevidence ]; then
            cd chutes-nvevidence
            cargo build --release 2>/dev/null
            if [ -f target/release/chutes-nvevidence ]; then
                sudo cp target/release/chutes-nvevidence /usr/local/bin/
                echo "INSTALLED"
            else
                echo "BUILD_FAILED"
            fi
        else
            echo "CLONE_FAILED"
        fi
        rm -rf "$TMPDIR"
    "#;

    /// Test GPU attestation
    pub const TEST_ATTESTATION: &str = r#"
        TEST_NONCE="deadbeefcafe1234567890abcdef0123"
        
        if command -v chutes-nvevidence &>/dev/null; then
            EVIDENCE=$(chutes-nvevidence --nonce "$TEST_NONCE" --format json 2>/dev/null)
            if [ -n "$EVIDENCE" ] && [ "$EVIDENCE" != "{}" ] && [ "$EVIDENCE" != "[]" ]; then
                echo "ATTESTATION_OK:chutes-nvevidence"
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
        
        echo "ATTESTATION_FAILED"
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
            tdx_commands::SETUP_CONFIGFS_TSM,
            tdx_commands::INSTALL_QUOTE_GENERATOR,
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
            gpu_commands::INSTALL_NVEVIDENCE,
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
        output.starts_with("QUOTE_OK:")
    }

    /// Parse test attestation output
    pub fn parse_test_attestation(output: &str) -> bool {
        output.starts_with("ATTESTATION_OK:")
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
            configfs_configured: o
                .configfs_setup
                .as_ref()
                .map(|s| s.contains("CONFIGFS_TSM_OK"))
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
    pub configfs_setup: Option<String>,
    pub install_generator: Option<String>,
    pub test_quote: Option<String>,
    pub error: Option<String>,
}

/// Outputs from GPU commands
#[derive(Debug, Default)]
pub struct GpuCommandOutputs {
    pub gpu_check: String,
    pub cc_mode_check: String,
    pub attestation_check: String,
    pub install_nvevidence: Option<String>,
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
            TeeBootstrap::parse_quote_generator_check("FOUND:/usr/local/bin/tdx-quote-generator");
        assert_eq!(
            result,
            Some("/usr/local/bin/tdx-quote-generator".to_string())
        );

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
            configfs_configured: true,
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
            hardware_check: "TDX_SUPPORTED".to_string(),
            generator_check: "FOUND:/usr/local/bin/tdx-quote-generator".to_string(),
            configfs_setup: Some("CONFIGFS_TSM_OK".to_string()),
            test_quote: Some("QUOTE_OK:2048".to_string()),
            ..Default::default()
        };

        let gpu_outputs = GpuCommandOutputs {
            gpu_check: "GPU_DETECTED:NVIDIA H100|GPU-123|535.0|true".to_string(),
            cc_mode_check: "CC_ENABLED".to_string(),
            attestation_check: "FOUND:chutes-nvevidence:/usr/local/bin/chutes-nvevidence"
                .to_string(),
            test_attestation: Some("ATTESTATION_OK:chutes-nvevidence".to_string()),
            ..Default::default()
        };

        let result = bootstrap.build_result(Some(tdx_outputs), Some(gpu_outputs));
        assert!(result.success);
        assert!(result.tdx.as_ref().unwrap().test_quote_ok);
        assert!(result.gpu_cc.as_ref().unwrap().test_attestation_ok);
    }
}
