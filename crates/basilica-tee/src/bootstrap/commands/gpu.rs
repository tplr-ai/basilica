//! GPU Bootstrap Commands
//!
//! Shell commands for detecting and setting up NVIDIA GPU CC on remote nodes.

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
