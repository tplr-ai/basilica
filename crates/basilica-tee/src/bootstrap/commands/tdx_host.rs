//! TDX Host Setup Commands
//!
//! Self-contained shell commands for Intel TDX host configuration.
//! Runs via SSH on bare-metal nodes to prepare them for TDX VMs.
//!
//! These commands are completely self-sufficient - no external repos,
//! submodules, or reference files are required.

/// Check current TDX host status
pub const CHECK_TDX_HOST_STATUS: &str = r#"
    # Check TDX module initialization (in dmesg)
    TDX_INIT="no"
    if dmesg 2>/dev/null | grep -qi "tdx.*module initialized\|tdx: module initialized\|virt/tdx"; then
        TDX_INIT="yes"
    fi
    
    # Check CPU TDX capability
    # TDX requires: Intel CPU with TME (Total Memory Encryption) support
    # Known TDX-capable: Sapphire Rapids (8xxx), Emerald Rapids (8xxx), Sierra Forest, Granite Rapids
    CPU_TDX="no"
    
    # Method 1: Check for TDX sysfs (kernel exposes this if CPU supports TDX)
    if [ -d /sys/firmware/tdx ] || [ -d /sys/devices/system/cpu/tdx ]; then
        CPU_TDX="yes"
    fi
    
    # Method 2: Check CPU model name for known TDX-capable CPUs
    if [ "$CPU_TDX" = "no" ]; then
        CPU_MODEL=$(cat /proc/cpuinfo 2>/dev/null | grep "model name" | head -1 | cut -d: -f2)
        # Xeon Scalable 4th gen+ (Sapphire Rapids, Emerald Rapids) - 8xxx series
        if echo "$CPU_MODEL" | grep -qiE "8[0-9]{3}|platinum 8|gold 6[4-9]|silver 4[4-9]|sapphire|emerald|granite|sierra"; then
            CPU_TDX="yes"
        fi
    fi
    
    # Method 3: Check for TME flag (required for TDX)
    if [ "$CPU_TDX" = "no" ]; then
        if grep -qw "tme" /proc/cpuinfo 2>/dev/null; then
            CPU_TDX="yes"
        fi
    fi
    
    # Kernel info
    KERNEL=$(uname -r)
    INTEL_KERNEL=$(dpkg -l 2>/dev/null | grep -q "linux-image-.*intel" && echo "yes" || echo "no")
    
    # Services
    PCCS=$(systemctl is-active pccs 2>/dev/null || echo "inactive")
    QGS=$(systemctl is-active qgsd 2>/dev/null || echo "inactive")
    
    # BIOS TDX (check dmesg for BIOS enabled message, or MSR access)
    BIOS_TDX="no"
    if dmesg 2>/dev/null | grep -qi "tdx.*BIOS\|TDX enabled by BIOS\|tdx: BIOS"; then
        BIOS_TDX="yes"
    # Also check if TDX is reported as inactive by kernel (means BIOS has it off)
    elif dmesg 2>/dev/null | grep -qi "tdx: BIOS.*not.*enabled\|tdx: disabled"; then
        BIOS_TDX="no"
    # If TDX module initialized, BIOS must have it enabled
    elif [ "$TDX_INIT" = "yes" ]; then
        BIOS_TDX="yes"
    fi
    
    echo "TDX_STATUS:init=$TDX_INIT,cpu=$CPU_TDX,kernel=$KERNEL,intel=$INTEL_KERNEL,pccs=$PCCS,qgs=$QGS,bios=$BIOS_TDX"
"#;

/// Install TDX host packages (kernel, QEMU, libvirt)
pub const INSTALL_TDX_HOST_PACKAGES: &str = r#"
    set -e
    export DEBIAN_FRONTEND=noninteractive
    
    # Detect Ubuntu version
    UBUNTU_VER=$(lsb_release -rs 2>/dev/null || echo "unknown")
    
    if [[ "$UBUNTU_VER" != "24.04" && "$UBUNTU_VER" != "25.04" ]]; then
        echo "UNSUPPORTED_UBUNTU:$UBUNTU_VER"
        exit 1
    fi
    
    # Add kobuk-team TDX PPA
    apt-get install -y software-properties-common >/dev/null 2>&1
    add-apt-repository -y ppa:kobuk-team/tdx-release >/dev/null 2>&1
    apt-get update -qq
    
    # Install TDX host stack
    apt-get install -y --allow-downgrades \
        linux-image-intel \
        qemu-system-x86 \
        libvirt-daemon-system \
        libvirt-clients \
        ovmf \
        gawk \
        >/dev/null 2>&1
    
    # Get kernel version and install modules-extra
    KERNEL_VER=$(apt show linux-image-intel 2>/dev/null | grep -oP "Depends:.*linux-image-\K[^, ]+" | head -1)
    apt-get install -y --allow-downgrades linux-modules-extra-${KERNEL_VER} >/dev/null 2>&1 || true
    
    echo "PACKAGES_INSTALLED:kernel=$KERNEL_VER"
"#;

/// Configure GRUB for TDX (nohibernate required)
pub const CONFIGURE_GRUB_TDX: &str = r#"
    set -e
    
    CHANGED="no"
    
    # Add nohibernate (TDX cannot survive S3/S4 sleep states)
    if ! grep -q "nohibernate" /etc/default/grub; then
        sed -i 's/GRUB_CMDLINE_LINUX="\([^"]*\)"/GRUB_CMDLINE_LINUX="\1 nohibernate"/' /etc/default/grub
        CHANGED="yes"
    fi
    
    if [ "$CHANGED" = "yes" ]; then
        update-grub >/dev/null 2>&1
        grub-install --no-nvram >/dev/null 2>&1 || true
        echo "GRUB_UPDATED:reboot_required"
    else
        echo "GRUB_OK:no_changes"
    fi
"#;

/// Install attestation packages (PCCS, QGS, Intel tools)
pub const INSTALL_ATTESTATION_PACKAGES: &str = r#"
    set -e
    export DEBIAN_FRONTEND=noninteractive
    
    # Add attestation PPA
    add-apt-repository -y ppa:kobuk-team/tdx-attestation-release >/dev/null 2>&1
    apt-get update -qq
    
    # Install DCAP attestation stack
    apt-get install -y --allow-downgrades \
        sgx-dcap-pccs \
        tdx-qgs \
        libsgx-dcap-default-qpl \
        sgx-ra-service \
        sgx-pck-id-retrieval-tool \
        >/dev/null 2>&1
    
    echo "ATTESTATION_PACKAGES_INSTALLED"
"#;

/// Configure PCCS service (non-interactive with provided API key)
/// Usage: Pass INTEL_API_KEY and PCCS_PASSWORD as environment variables or script arguments
pub const CONFIGURE_PCCS: &str = r#"
    INTEL_API_KEY="${INTEL_API_KEY:-$1}"
    PCCS_PASSWORD="${PCCS_PASSWORD:-$2}"
    
    if [ -z "$INTEL_API_KEY" ]; then
        echo "PCCS_CONFIG_ERROR:missing_api_key"
        exit 1
    fi
    
    # Write PCCS config
    mkdir -p /opt/intel/sgx-dcap-pccs/config
    cat > /opt/intel/sgx-dcap-pccs/config/default.json << EOF
{
    "HTTPS_PORT": 8081,
    "hosts": "127.0.0.1",
    "uri": "https://api.trustedservices.intel.com/sgx/certification/v4/",
    "ApiKey": "$INTEL_API_KEY",
    "proxy": "",
    "RefreshSchedule": "0 0 1 * * *",
    "UserTokenHash": "",
    "AdminTokenHash": "",
    "CachingFillMode": "LAZY",
    "LogLevel": "info"
}
EOF
    
    # Restart services
    systemctl restart pccs >/dev/null 2>&1 || true
    systemctl enable pccs >/dev/null 2>&1 || true
    
    sleep 2
    
    if systemctl is-active --quiet pccs; then
        echo "PCCS_CONFIGURED:running"
    else
        echo "PCCS_CONFIGURED:failed_to_start"
    fi
"#;

/// Register platform with Intel PCCS
pub const REGISTER_PLATFORM: &str = r#"
    # Run PCK ID retrieval tool
    if command -v PCKIDRetrievalTool &>/dev/null; then
        OUTPUT=$(PCKIDRetrievalTool -url https://localhost:8081 -use_secure_cert false 2>&1) || true
        if echo "$OUTPUT" | grep -qi "success"; then
            echo "PLATFORM_REGISTRATION:success"
        else
            echo "PLATFORM_REGISTRATION:attempted"
        fi
    else
        echo "PLATFORM_REGISTRATION:tool_not_found"
    fi
"#;

/// Full TDX host verification after setup
pub const VERIFY_TDX_HOST_FULL: &str = r#"
    ERRORS=""
    WARNINGS=""
    
    # Check TDX module initialized
    if ! dmesg | grep -qi "tdx.*module initialized"; then
        ERRORS="${ERRORS}tdx_not_initialized;"
    fi
    
    # Check PCCS running
    if ! systemctl is-active --quiet pccs; then
        WARNINGS="${WARNINGS}pccs_not_running;"
    fi
    
    # Check QGS running
    if ! systemctl is-active --quiet qgsd; then
        WARNINGS="${WARNINGS}qgs_not_running;"
    fi
    
    # Check SGX devices exist
    if [ ! -c /dev/sgx_enclave ] && [ ! -c /dev/sgx_provision ]; then
        WARNINGS="${WARNINGS}sgx_devices_missing;"
    fi
    
    # Check QEMU available
    if ! command -v qemu-system-x86_64 &>/dev/null; then
        ERRORS="${ERRORS}qemu_not_installed;"
    fi
    
    # Check libvirt available
    if ! command -v virsh &>/dev/null; then
        ERRORS="${ERRORS}libvirt_not_installed;"
    fi
    
    if [ -z "$ERRORS" ]; then
        echo "TDX_HOST_VERIFIED:ok|warnings=$WARNINGS"
    else
        echo "TDX_HOST_VERIFIED:failed|errors=$ERRORS|warnings=$WARNINGS"
    fi
"#;

/// Check if reboot is required
pub const CHECK_REBOOT_REQUIRED: &str = r#"
    # Check if running kernel matches installed Intel kernel
    RUNNING=$(uname -r)
    INSTALLED=$(dpkg -l | grep "linux-image-.*intel" | grep -oP '\d+\.\d+\.\d+-\d+-intel' | head -1)
    
    if [ -n "$INSTALLED" ] && [ "$RUNNING" != "$INSTALLED" ]; then
        echo "REBOOT_REQUIRED:current=$RUNNING,target=$INSTALLED"
    elif [ -f /var/run/reboot-required ]; then
        echo "REBOOT_REQUIRED:system_flag"
    else
        echo "REBOOT_NOT_REQUIRED"
    fi
"#;

/// Check if host is ready to launch TDX VMs
/// Verifies: /dev/kvm, QEMU TDX support, libvirt TDX capabilities, OVMF firmware
pub const CHECK_TDX_VM_READINESS: &str = r#"
    ERRORS=""
    WARNINGS=""
    
    # Check /dev/kvm exists
    if [ -c /dev/kvm ]; then
        KVM="yes"
    else
        KVM="no"
        ERRORS="${ERRORS}kvm_device_missing;"
    fi
    
    # Check QEMU with TDX support
    QEMU_PATH=$(which qemu-system-x86_64 2>/dev/null || echo "")
    if [ -n "$QEMU_PATH" ]; then
        QEMU="yes"
        # Check if QEMU supports TDX (look for tdx-guest object)
        if $QEMU_PATH -object help 2>/dev/null | grep -q "tdx-guest"; then
            QEMU_TDX="yes"
        else
            QEMU_TDX="no"
            WARNINGS="${WARNINGS}qemu_no_tdx_support;"
        fi
    else
        QEMU="no"
        QEMU_TDX="no"
        ERRORS="${ERRORS}qemu_not_installed;"
    fi
    
    # Check libvirt and virsh
    if command -v virsh &>/dev/null; then
        VIRSH="yes"
        # Check if libvirt sees TDX capability
        if virsh domcapabilities 2>/dev/null | grep -q "tdx"; then
            LIBVIRT_TDX="yes"
        else
            LIBVIRT_TDX="no"
            WARNINGS="${WARNINGS}libvirt_no_tdx_capability;"
        fi
    else
        VIRSH="no"
        LIBVIRT_TDX="no"
        ERRORS="${ERRORS}virsh_not_installed;"
    fi
    
    # Check OVMF firmware for TDX
    OVMF_TDX=""
    if [ -f /usr/share/OVMF/OVMF_CODE_4M.ms.fd ]; then
        OVMF_TDX="/usr/share/OVMF/OVMF_CODE_4M.ms.fd"
    elif [ -f /usr/share/qemu/OVMF_CODE_4M.ms.fd ]; then
        OVMF_TDX="/usr/share/qemu/OVMF_CODE_4M.ms.fd"
    elif [ -f /usr/share/OVMF/OVMF_CODE.fd ]; then
        OVMF_TDX="/usr/share/OVMF/OVMF_CODE.fd"
    fi
    
    if [ -n "$OVMF_TDX" ]; then
        OVMF="yes"
    else
        OVMF="no"
        ERRORS="${ERRORS}ovmf_firmware_missing;"
    fi
    
    # Check libvirt is running
    if systemctl is-active --quiet libvirtd; then
        LIBVIRTD="active"
    else
        LIBVIRTD="inactive"
        WARNINGS="${WARNINGS}libvirtd_not_running;"
    fi
    
    # Determine overall readiness
    if [ -z "$ERRORS" ]; then
        READY="yes"
    else
        READY="no"
    fi
    
    echo "VM_READINESS:ready=$READY,kvm=$KVM,qemu=$QEMU,qemu_tdx=$QEMU_TDX,virsh=$VIRSH,libvirt_tdx=$LIBVIRT_TDX,ovmf=$OVMF,ovmf_path=$OVMF_TDX,libvirtd=$LIBVIRTD"
    if [ -n "$ERRORS" ]; then
        echo "VM_READINESS_ERRORS:$ERRORS"
    fi
    if [ -n "$WARNINGS" ]; then
        echo "VM_READINESS_WARNINGS:$WARNINGS"
    fi
"#;

/// Check libvirt TDX domain capabilities in detail
pub const CHECK_LIBVIRT_TDX: &str = r#"
    if ! command -v virsh &>/dev/null; then
        echo "LIBVIRT_TDX:not_installed"
        exit 0
    fi
    
    # Get domain capabilities
    CAPS=$(virsh domcapabilities 2>/dev/null || echo "")
    
    if [ -z "$CAPS" ]; then
        echo "LIBVIRT_TDX:no_capabilities"
        exit 0
    fi
    
    # Check for TDX in capabilities
    if echo "$CAPS" | grep -q "<launchSecurity.*tdx"; then
        TDX_SUPPORTED="yes"
        # Extract TDX-specific info if available
        TDX_SECTION=$(echo "$CAPS" | grep -A20 "launchSecurity.*tdx" | head -25)
        echo "LIBVIRT_TDX:supported"
        echo "LIBVIRT_TDX_CAPS:$TDX_SECTION"
    else
        echo "LIBVIRT_TDX:not_supported"
    fi
    
    # Also check machine types
    if echo "$CAPS" | grep -q "q35"; then
        echo "LIBVIRT_Q35:supported"
    else
        echo "LIBVIRT_Q35:not_supported"
    fi
"#;

/// Full TDX host setup - combines all steps into one command
/// This is the main entry point for automated TDX host setup
pub const FULL_TDX_HOST_SETUP: &str = r#"
    set -e
    export DEBIAN_FRONTEND=noninteractive
    
    echo "=== Starting TDX Host Setup ==="
    
    # Step 1: Check Ubuntu version
    UBUNTU_VER=$(lsb_release -rs 2>/dev/null || echo "unknown")
    if [[ "$UBUNTU_VER" != "24.04" && "$UBUNTU_VER" != "25.04" ]]; then
        echo "SETUP_FAILED:unsupported_ubuntu=$UBUNTU_VER"
        exit 1
    fi
    echo "Ubuntu version: $UBUNTU_VER"
    
    # Step 2: Add TDX PPAs
    apt-get update -qq
    apt-get install -y software-properties-common >/dev/null 2>&1
    add-apt-repository -y ppa:kobuk-team/tdx-release >/dev/null 2>&1
    add-apt-repository -y ppa:kobuk-team/tdx-attestation-release >/dev/null 2>&1
    apt-get update -qq
    echo "PPAs added"
    
    # Step 3: Install TDX host packages
    apt-get install -y --allow-downgrades \
        linux-image-intel \
        qemu-system-x86 \
        libvirt-daemon-system \
        libvirt-clients \
        ovmf \
        gawk \
        >/dev/null 2>&1
    echo "TDX host packages installed"
    
    # Step 4: Install kernel modules-extra
    KERNEL_VER=$(apt show linux-image-intel 2>/dev/null | grep -oP "Depends:.*linux-image-\K[^, ]+" | head -1)
    apt-get install -y --allow-downgrades linux-modules-extra-${KERNEL_VER} >/dev/null 2>&1 || true
    echo "Kernel modules installed: $KERNEL_VER"
    
    # Step 5: Install attestation packages
    apt-get install -y --allow-downgrades \
        sgx-dcap-pccs \
        tdx-qgs \
        libsgx-dcap-default-qpl \
        sgx-ra-service \
        sgx-pck-id-retrieval-tool \
        >/dev/null 2>&1 || true
    echo "Attestation packages installed"
    
    # Step 6: Configure GRUB
    if ! grep -q "nohibernate" /etc/default/grub; then
        sed -i 's/GRUB_CMDLINE_LINUX="\([^"]*\)"/GRUB_CMDLINE_LINUX="\1 nohibernate"/' /etc/default/grub
        update-grub >/dev/null 2>&1
        grub-install --no-nvram >/dev/null 2>&1 || true
        echo "GRUB configured"
    else
        echo "GRUB already configured"
    fi
    
    # Step 7: Add user to kvm group
    LOG_USER=$(logname 2>/dev/null || echo "")
    if [ -n "$LOG_USER" ] && [ "$LOG_USER" != "root" ]; then
        usermod -aG kvm $LOG_USER 2>/dev/null || true
    fi
    
    # Check if reboot needed
    RUNNING=$(uname -r)
    if [ "$RUNNING" != "$KERNEL_VER" ]; then
        echo "SETUP_COMPLETE:reboot_required,current_kernel=$RUNNING,target_kernel=$KERNEL_VER"
    else
        echo "SETUP_COMPLETE:no_reboot_needed"
    fi
"#;

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    #[allow(clippy::const_is_empty)]
    fn test_commands_not_empty() {
        assert!(!CHECK_TDX_HOST_STATUS.is_empty());
        assert!(!INSTALL_TDX_HOST_PACKAGES.is_empty());
        assert!(!CONFIGURE_GRUB_TDX.is_empty());
        assert!(!INSTALL_ATTESTATION_PACKAGES.is_empty());
        assert!(!CONFIGURE_PCCS.is_empty());
        assert!(!REGISTER_PLATFORM.is_empty());
        assert!(!VERIFY_TDX_HOST_FULL.is_empty());
        assert!(!CHECK_REBOOT_REQUIRED.is_empty());
        assert!(!FULL_TDX_HOST_SETUP.is_empty());
        assert!(!CHECK_TDX_VM_READINESS.is_empty());
        assert!(!CHECK_LIBVIRT_TDX.is_empty());
    }

    #[test]
    fn test_vm_readiness_output_format() {
        assert!(CHECK_TDX_VM_READINESS.contains("VM_READINESS:"));
        assert!(CHECK_TDX_VM_READINESS.contains("kvm="));
        assert!(CHECK_TDX_VM_READINESS.contains("qemu_tdx="));
    }

    #[test]
    fn test_check_status_has_expected_output_format() {
        assert!(CHECK_TDX_HOST_STATUS.contains("TDX_STATUS:"));
    }

    #[test]
    fn test_install_packages_checks_ubuntu_version() {
        assert!(INSTALL_TDX_HOST_PACKAGES.contains("24.04"));
        assert!(INSTALL_TDX_HOST_PACKAGES.contains("25.04"));
    }

    #[test]
    fn test_grub_config_adds_nohibernate() {
        assert!(CONFIGURE_GRUB_TDX.contains("nohibernate"));
    }

    #[test]
    fn test_attestation_installs_required_packages() {
        assert!(INSTALL_ATTESTATION_PACKAGES.contains("sgx-dcap-pccs"));
        assert!(INSTALL_ATTESTATION_PACKAGES.contains("tdx-qgs"));
    }
}
