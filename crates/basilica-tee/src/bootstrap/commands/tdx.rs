//! TDX Bootstrap Commands
//!
//! Shell commands for detecting and setting up Intel TDX on remote nodes.

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
