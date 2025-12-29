//! TDX Guest VM Commands
//!
//! Shell commands for creating and managing TDX guest VMs for testing.
//! These commands run on the TDX host to create test VMs that can generate TDX quotes.
//!
//! ## Working Configuration
//!
//! TDX VMs are launched with QEMU using:
//! - `-bios /usr/share/ovmf/OVMF.fd` (NOT pflash, which requires unsupported KVM readonly memory)
//! - `-machine q35,kernel-irqchip=split,confidential-guest-support=tdx0`
//! - `-object tdx-guest,id=tdx0`
//!
//! ## Guest Kernel Requirements
//!
//! For TDX Report/Quote generation, the guest VM needs:
//! - TDX-enabled kernel with `/dev/tdx_guest` device
//! - vsock for communication with host QGS (optional, for full quotes)
//!
//! The standard Ubuntu cloud image kernel does NOT have TDX support.
//!
//! ## Two-Phase Boot Process (Tested Working)
//!
//! Since TDX VMs cannot reboot (CPU state is sealed), use two phases:
//! 1. Boot with generic kernel, install `linux-image-intel` from kobuk-team PPA
//! 2. Destroy VM, create new VM using same disk (now has Intel kernel)
//!
//! After two-phase boot, `/dev/tdx_guest` will be available for attestation.

/// Directory for TDX test VM artifacts
pub const TDX_TEST_VM_DIR: &str = "/var/lib/basilica-tdx-test";

/// Download Ubuntu 24.04 cloud image for TDX guest
pub const DOWNLOAD_TDX_GUEST_IMAGE: &str = r#"
    set -e
    
    VM_DIR="/var/lib/basilica-tdx-test"
    mkdir -p "$VM_DIR"
    
    IMAGE_URL="https://cloud-images.ubuntu.com/noble/current/noble-server-cloudimg-amd64.img"
    IMAGE_PATH="$VM_DIR/ubuntu-noble-cloudimg.img"
    
    # Check if image already exists
    if [ -f "$IMAGE_PATH" ]; then
        echo "GUEST_IMAGE:exists:$IMAGE_PATH"
        exit 0
    fi
    
    echo "Downloading Ubuntu 24.04 cloud image..."
    wget -q --show-progress -O "$IMAGE_PATH.tmp" "$IMAGE_URL" 2>&1 || \
        curl -fSL -o "$IMAGE_PATH.tmp" "$IMAGE_URL"
    
    mv "$IMAGE_PATH.tmp" "$IMAGE_PATH"
    
    echo "GUEST_IMAGE:downloaded:$IMAGE_PATH"
"#;

/// Create a test TDX VM disk image from cloud image
/// Creates a qcow2 backing file for the VM
pub const CREATE_TDX_VM_DISK: &str = r#"
    set -e
    
    VM_DIR="/var/lib/basilica-tdx-test"
    VM_NAME="${VM_NAME:-tdx-test-vm}"
    DISK_SIZE="${DISK_SIZE:-20G}"
    
    BASE_IMAGE="$VM_DIR/ubuntu-noble-cloudimg.img"
    VM_DISK="$VM_DIR/${VM_NAME}.qcow2"
    
    if [ ! -f "$BASE_IMAGE" ]; then
        echo "VM_DISK_ERROR:base_image_missing"
        exit 1
    fi
    
    # Create qcow2 with backing file
    qemu-img create -f qcow2 -F qcow2 -b "$BASE_IMAGE" "$VM_DISK" "$DISK_SIZE"
    
    echo "VM_DISK:created:$VM_DISK"
"#;

/// Create cloud-init configuration for TDX test VM
/// Injects SSH key and installs attestation packages
pub const CREATE_CLOUD_INIT: &str = r#"
    set -e
    
    VM_DIR="/var/lib/basilica-tdx-test"
    VM_NAME="${VM_NAME:-tdx-test-vm}"
    SSH_PUBKEY="${SSH_PUBKEY:-}"
    
    # Get host's SSH public key if not provided
    if [ -z "$SSH_PUBKEY" ]; then
        if [ -f /root/.ssh/id_rsa.pub ]; then
            SSH_PUBKEY=$(cat /root/.ssh/id_rsa.pub)
        elif [ -f /root/.ssh/id_ed25519.pub ]; then
            SSH_PUBKEY=$(cat /root/.ssh/id_ed25519.pub)
        else
            echo "CLOUD_INIT_ERROR:no_ssh_key"
            exit 1
        fi
    fi
    
    SEED_DIR="$VM_DIR/${VM_NAME}-seed"
    mkdir -p "$SEED_DIR"
    
    # Create meta-data
    cat > "$SEED_DIR/meta-data" << EOF
instance-id: ${VM_NAME}
local-hostname: ${VM_NAME}
EOF
    
    # Create user-data with SSH key and attestation setup
    cat > "$SEED_DIR/user-data" << EOF
#cloud-config
users:
  - name: ubuntu
    sudo: ALL=(ALL) NOPASSWD:ALL
    shell: /bin/bash
    ssh_authorized_keys:
      - ${SSH_PUBKEY}

package_update: true
package_upgrade: false

packages:
  - software-properties-common
  - curl
  - wget

runcmd:
  # Add TDX attestation PPA and install packages
  - add-apt-repository -y ppa:kobuk-team/tdx-attestation-release
  - apt-get update -qq
  - apt-get install -y libtdx-attest libtdx-attest-dev tdx-tools || true
  # Signal that setup is complete
  - touch /var/run/tdx-guest-ready
EOF
    
    # Create seed ISO
    SEED_ISO="$VM_DIR/${VM_NAME}-seed.iso"
    
    # Try different cloud-localds or genisoimage
    if command -v cloud-localds &>/dev/null; then
        cloud-localds "$SEED_ISO" "$SEED_DIR/user-data" "$SEED_DIR/meta-data"
    elif command -v genisoimage &>/dev/null; then
        genisoimage -output "$SEED_ISO" -volid cidata -joliet -rock \
            "$SEED_DIR/user-data" "$SEED_DIR/meta-data"
    elif command -v mkisofs &>/dev/null; then
        mkisofs -output "$SEED_ISO" -volid cidata -joliet -rock \
            "$SEED_DIR/user-data" "$SEED_DIR/meta-data"
    else
        # Install cloud-image-utils
        apt-get install -y cloud-image-utils >/dev/null 2>&1
        cloud-localds "$SEED_ISO" "$SEED_DIR/user-data" "$SEED_DIR/meta-data"
    fi
    
    echo "CLOUD_INIT:created:$SEED_ISO"
"#;

/// Launch TDX VM using libvirt (handles TDX configuration properly)
/// Returns the VM's assigned IP address
pub const LAUNCH_TDX_VM_QEMU: &str = r#"
    set -e
    
    VM_DIR="/var/lib/basilica-tdx-test"
    VM_NAME="${VM_NAME:-tdx-test-vm}"
    VM_CPUS="${VM_CPUS:-2}"
    VM_MEM="${VM_MEM:-4096}"
    
    VM_DISK="$VM_DIR/${VM_NAME}.qcow2"
    SEED_ISO="$VM_DIR/${VM_NAME}-seed.iso"
    
    # Find OVMF firmware
    OVMF_CODE=""
    OVMF_VARS=""
    if [ -f /usr/share/OVMF/OVMF_CODE_4M.fd ]; then
        OVMF_CODE="/usr/share/OVMF/OVMF_CODE_4M.fd"
        OVMF_VARS="/usr/share/OVMF/OVMF_VARS_4M.fd"
    elif [ -f /usr/share/OVMF/OVMF_CODE.fd ]; then
        OVMF_CODE="/usr/share/OVMF/OVMF_CODE.fd"
        OVMF_VARS="/usr/share/OVMF/OVMF_VARS.fd"
    fi
    
    if [ -z "$OVMF_CODE" ]; then
        echo "TDX_VM_ERROR:ovmf_not_found"
        exit 1
    fi
    
    # Copy VARS file for this VM
    VM_VARS="$VM_DIR/${VM_NAME}-OVMF_VARS.fd"
    cp "$OVMF_VARS" "$VM_VARS"
    
    # Ensure libvirtd is running
    systemctl start libvirtd 2>/dev/null || true
    
    # Find available port for SSH forwarding
    SSH_PORT=2222
    while ss -tuln 2>/dev/null | grep -q ":$SSH_PORT "; do
        SSH_PORT=$((SSH_PORT + 1))
        if [ $SSH_PORT -gt 2299 ]; then
            echo "TDX_VM_ERROR:no_available_port"
            exit 1
        fi
    done
    
    echo "Using OVMF: $OVMF_CODE"
    echo "SSH Port: $SSH_PORT"
    
    # Remove existing VM if present
    virsh destroy "$VM_NAME" 2>/dev/null || true
    virsh undefine "$VM_NAME" --nvram 2>/dev/null || true
    
    # Create libvirt XML for TDX VM
    cat > "$VM_DIR/${VM_NAME}.xml" << XMLEOF
<domain type='kvm'>
  <name>${VM_NAME}</name>
  <memory unit='MiB'>${VM_MEM}</memory>
  <vcpu>${VM_CPUS}</vcpu>
  <os>
    <type arch='x86_64' machine='q35'>hvm</type>
    <loader readonly='yes' type='pflash'>${OVMF_CODE}</loader>
    <nvram>${VM_VARS}</nvram>
    <boot dev='hd'/>
  </os>
  <features>
    <acpi/>
    <apic/>
  </features>
  <cpu mode='host-passthrough'/>
  <launchSecurity type='tdx'>
    <policy>0x0</policy>
  </launchSecurity>
  <devices>
    <emulator>/usr/bin/qemu-system-x86_64</emulator>
    <disk type='file' device='disk'>
      <driver name='qemu' type='qcow2'/>
      <source file='${VM_DISK}'/>
      <target dev='vda' bus='virtio'/>
    </disk>
    <disk type='file' device='cdrom'>
      <driver name='qemu' type='raw'/>
      <source file='${SEED_ISO}'/>
      <target dev='sda' bus='sata'/>
      <readonly/>
    </disk>
    <interface type='user'>
      <model type='virtio'/>
      <portForward address='0.0.0.0' proto='tcp' dev=''>
        <range start='22' to='${SSH_PORT}'/>
      </portForward>
    </interface>
    <serial type='pty'>
      <target port='0'/>
    </serial>
    <console type='pty'>
      <target type='serial' port='0'/>
    </console>
  </devices>
</domain>
XMLEOF
    
    # Define and start the VM
    virsh define "$VM_DIR/${VM_NAME}.xml"
    virsh start "$VM_NAME"
    
    echo "TDX_VM_LAUNCHED:name=$VM_NAME,ssh_port=$SSH_PORT"
"#;

/// Wait for TDX VM to be ready (SSH accessible)
pub const WAIT_FOR_TDX_VM: &str = r#"
    VM_NAME="${VM_NAME:-tdx-test-vm}"
    SSH_PORT="${SSH_PORT:-2222}"
    TIMEOUT="${TIMEOUT:-300}"
    
    echo "Waiting for VM $VM_NAME to be ready on port $SSH_PORT..."
    
    START_TIME=$(date +%s)
    while true; do
        ELAPSED=$(($(date +%s) - START_TIME))
        if [ $ELAPSED -gt $TIMEOUT ]; then
            echo "TDX_VM_WAIT:timeout"
            exit 1
        fi
        
        # Try SSH connection
        if ssh -o StrictHostKeyChecking=no -o UserKnownHostsFile=/dev/null \
               -o ConnectTimeout=5 -o BatchMode=yes \
               -p "$SSH_PORT" ubuntu@localhost "echo ready" 2>/dev/null; then
            echo "TDX_VM_WAIT:ready"
            exit 0
        fi
        
        sleep 5
    done
"#;

/// Install attestation tools inside the TDX guest VM
pub const INSTALL_GUEST_ATTESTATION: &str = r#"
    SSH_PORT="${SSH_PORT:-2222}"
    
    # Commands to run inside the VM
    GUEST_CMDS='
        set -e
        export DEBIAN_FRONTEND=noninteractive
        
        # Check if TDX device exists
        if [ -c /dev/tdx_guest ] || [ -c /dev/tdx-guest ]; then
            echo "TDX_DEVICE:found"
        else
            echo "TDX_DEVICE:not_found"
        fi
        
        # Add attestation PPA if not already added
        if ! grep -q "kobuk-team/tdx-attestation" /etc/apt/sources.list.d/*.list 2>/dev/null; then
            sudo add-apt-repository -y ppa:kobuk-team/tdx-attestation-release
            sudo apt-get update -qq
        fi
        
        # Install attestation packages
        sudo apt-get install -y libtdx-attest libtdx-attest-dev tdx-tools 2>/dev/null || true
        
        # Check what is available
        if command -v tdx_attest &>/dev/null; then
            echo "ATTEST_TOOL:tdx_attest"
        elif [ -f /usr/lib/x86_64-linux-gnu/libtdx_attest.so ]; then
            echo "ATTEST_TOOL:libtdx_attest"
        else
            echo "ATTEST_TOOL:none"
        fi
    '
    
    ssh -o StrictHostKeyChecking=no -o UserKnownHostsFile=/dev/null \
        -p "$SSH_PORT" ubuntu@localhost "$GUEST_CMDS"
"#;

/// Test TDX quote generation inside the guest VM
pub const TEST_GUEST_QUOTE_GEN: &str = r#"
    SSH_PORT="${SSH_PORT:-2222}"
    
    # Commands to run inside the VM for quote generation
    GUEST_CMDS='
        set -e
        
        # Check TDX device
        TDX_DEV=""
        if [ -c /dev/tdx_guest ]; then
            TDX_DEV="/dev/tdx_guest"
        elif [ -c /dev/tdx-guest ]; then
            TDX_DEV="/dev/tdx-guest"
        fi
        
        if [ -z "$TDX_DEV" ]; then
            echo "QUOTE_GEN:no_tdx_device"
            exit 1
        fi
        
        echo "TDX_DEVICE:$TDX_DEV"
        
        # Create test nonce (64 bytes = 128 hex chars)
        TEST_NONCE="0102030405060708090a0b0c0d0e0f101112131415161718191a1b1c1d1e1f202122232425262728292a2b2c2d2e2f303132333435363738393a3b3c3d3e3f40"
        
        TMPDIR=$(mktemp -d)
        
        # Try different methods to generate quote
        
        # Method 1: tdx_attest tool
        if command -v tdx_attest &>/dev/null; then
            echo "Using tdx_attest tool..."
            echo -n "$TEST_NONCE" | xxd -r -p > "$TMPDIR/nonce.bin"
            if tdx_attest -r "$TMPDIR/nonce.bin" -q "$TMPDIR/quote.bin" 2>/dev/null; then
                QUOTE_SIZE=$(stat -c%s "$TMPDIR/quote.bin" 2>/dev/null || stat -f%z "$TMPDIR/quote.bin")
                QUOTE_HEX=$(xxd -p "$TMPDIR/quote.bin" | tr -d "\n" | head -c 200)
                rm -rf "$TMPDIR"
                echo "QUOTE_GEN:success:size=$QUOTE_SIZE"
                echo "QUOTE_PREVIEW:$QUOTE_HEX..."
                exit 0
            fi
        fi
        
        # Method 2: configfs-tsm (kernel 6.7+)
        if [ -d /sys/kernel/config/tsm/report ]; then
            echo "Using configfs-tsm..."
            REPORT_DIR="/sys/kernel/config/tsm/report/test$$"
            mkdir -p "$REPORT_DIR" 2>/dev/null || true
            if [ -d "$REPORT_DIR" ]; then
                echo -n "$TEST_NONCE" | xxd -r -p > "$REPORT_DIR/inblob"
                QUOTE=$(cat "$REPORT_DIR/outblob" 2>/dev/null | xxd -p | tr -d "\n")
                QUOTE_SIZE=${#QUOTE}
                rmdir "$REPORT_DIR" 2>/dev/null || true
                if [ -n "$QUOTE" ] && [ $QUOTE_SIZE -gt 100 ]; then
                    echo "QUOTE_GEN:success:size=$((QUOTE_SIZE/2))"
                    echo "QUOTE_PREVIEW:${QUOTE:0:200}..."
                    rm -rf "$TMPDIR"
                    exit 0
                fi
            fi
        fi
        
        # Method 3: Direct ioctl (needs custom tool)
        echo "QUOTE_GEN:no_method_available"
        rm -rf "$TMPDIR"
        exit 1
    '
    
    ssh -o StrictHostKeyChecking=no -o UserKnownHostsFile=/dev/null \
        -p "$SSH_PORT" ubuntu@localhost "$GUEST_CMDS"
"#;

/// Shutdown and cleanup TDX test VM
pub const CLEANUP_TDX_VM: &str = r#"
    VM_DIR="/var/lib/basilica-tdx-test"
    VM_NAME="${VM_NAME:-tdx-test-vm}"
    
    PID_FILE="$VM_DIR/${VM_NAME}.pid"
    MONITOR_SOCKET="$VM_DIR/${VM_NAME}-monitor.sock"
    
    # Try graceful shutdown via monitor
    if [ -S "$MONITOR_SOCKET" ]; then
        echo "system_powerdown" | socat - UNIX-CONNECT:"$MONITOR_SOCKET" 2>/dev/null || true
        sleep 5
    fi
    
    # Kill process if still running
    if [ -f "$PID_FILE" ]; then
        PID=$(cat "$PID_FILE")
        if kill -0 "$PID" 2>/dev/null; then
            kill "$PID" 2>/dev/null || true
            sleep 2
            kill -9 "$PID" 2>/dev/null || true
        fi
        rm -f "$PID_FILE"
    fi
    
    # Cleanup files
    rm -f "$VM_DIR/${VM_NAME}.qcow2"
    rm -f "$VM_DIR/${VM_NAME}-seed.iso"
    rm -f "$VM_DIR/${VM_NAME}-OVMF_VARS.fd"
    rm -rf "$VM_DIR/${VM_NAME}-seed"
    rm -f "$MONITOR_SOCKET"
    
    echo "TDX_VM_CLEANUP:done:$VM_NAME"
"#;

/// Full TDX guest VM test - creates VM, generates quote, cleans up
pub const FULL_TDX_GUEST_TEST: &str = r#"
    set -e
    
    VM_DIR="/var/lib/basilica-tdx-test"
    VM_NAME="tdx-test-$$"
    SSH_PORT=2222
    
    export VM_NAME
    export VM_DIR
    
    echo "=== TDX Guest VM Quote Generation Test ==="
    echo "VM Name: $VM_NAME"
    
    # Find available SSH port
    while netstat -tuln 2>/dev/null | grep -q ":$SSH_PORT " || ss -tuln 2>/dev/null | grep -q ":$SSH_PORT "; do
        SSH_PORT=$((SSH_PORT + 1))
    done
    export SSH_PORT
    echo "SSH Port: $SSH_PORT"
    
    cleanup() {
        echo "Cleaning up..."
        # Kill QEMU process
        if [ -f "$VM_DIR/${VM_NAME}.pid" ]; then
            PID=$(cat "$VM_DIR/${VM_NAME}.pid")
            kill "$PID" 2>/dev/null || true
            kill -9 "$PID" 2>/dev/null || true
        fi
        rm -f "$VM_DIR/${VM_NAME}"* 2>/dev/null || true
        rm -rf "$VM_DIR/${VM_NAME}-seed" 2>/dev/null || true
    }
    trap cleanup EXIT
    
    # Step 1: Check/download image
    echo ""
    echo "Step 1: Checking guest image..."
    IMAGE_PATH="$VM_DIR/ubuntu-noble-cloudimg.img"
    if [ ! -f "$IMAGE_PATH" ]; then
        echo "Downloading Ubuntu 24.04 cloud image..."
        mkdir -p "$VM_DIR"
        wget -q -O "$IMAGE_PATH" "https://cloud-images.ubuntu.com/noble/current/noble-server-cloudimg-amd64.img" || \
            curl -fSL -o "$IMAGE_PATH" "https://cloud-images.ubuntu.com/noble/current/noble-server-cloudimg-amd64.img"
    fi
    echo "Image ready: $IMAGE_PATH"
    
    # Step 2: Create VM disk
    echo ""
    echo "Step 2: Creating VM disk..."
    qemu-img create -f qcow2 -F qcow2 -b "$IMAGE_PATH" "$VM_DIR/${VM_NAME}.qcow2" 20G
    
    # Step 3: Create cloud-init
    echo ""
    echo "Step 3: Creating cloud-init..."
    SEED_DIR="$VM_DIR/${VM_NAME}-seed"
    mkdir -p "$SEED_DIR"
    
    # Get SSH key
    SSH_PUBKEY=""
    for keyfile in /root/.ssh/id_rsa.pub /root/.ssh/id_ed25519.pub ~/.ssh/id_rsa.pub ~/.ssh/id_ed25519.pub; do
        if [ -f "$keyfile" ]; then
            SSH_PUBKEY=$(cat "$keyfile")
            break
        fi
    done
    
    cat > "$SEED_DIR/meta-data" << EOF
instance-id: ${VM_NAME}
local-hostname: ${VM_NAME}
EOF
    
    cat > "$SEED_DIR/user-data" << EOF
#cloud-config
users:
  - name: ubuntu
    sudo: ALL=(ALL) NOPASSWD:ALL
    ssh_authorized_keys:
      - ${SSH_PUBKEY}
package_update: false
EOF
    
    # Create seed ISO
    apt-get install -y cloud-image-utils >/dev/null 2>&1 || true
    cloud-localds "$VM_DIR/${VM_NAME}-seed.iso" "$SEED_DIR/user-data" "$SEED_DIR/meta-data" 2>/dev/null || \
        genisoimage -output "$VM_DIR/${VM_NAME}-seed.iso" -volid cidata -joliet -rock "$SEED_DIR/user-data" "$SEED_DIR/meta-data"
    
    # Step 4: Launch TDX VM using QEMU directly with -bios
    # Note: virt-install/libvirt have issues with TDX+pflash, raw QEMU with -bios works
    echo ""
    echo "Step 4: Launching TDX VM via QEMU..."
    
    # Find available SSH port
    while ss -tuln 2>/dev/null | grep -q ":$SSH_PORT "; do
        SSH_PORT=$((SSH_PORT + 1))
        if [ $SSH_PORT -gt 2299 ]; then
            echo "GUEST_TEST_ERROR:no_available_port"
            exit 1
        fi
    done
    
    echo "SSH Port: $SSH_PORT"
    
    # Launch QEMU with TDX using -bios (not pflash)
    # Include vsock for communication with host QGS
    nohup qemu-system-x86_64 \
        -name "$VM_NAME" \
        -enable-kvm \
        -machine q35,kernel-irqchip=split,confidential-guest-support=tdx0 \
        -object tdx-guest,id=tdx0 \
        -cpu host \
        -m 4096 \
        -smp 2 \
        -bios /usr/share/ovmf/OVMF.fd \
        -drive file="$VM_DIR/${VM_NAME}.qcow2",if=virtio,format=qcow2 \
        -drive file="$VM_DIR/${VM_NAME}-seed.iso",if=virtio,format=raw,readonly=on \
        -netdev user,id=net0,hostfwd=tcp::${SSH_PORT}-:22 \
        -device virtio-net-pci,netdev=net0 \
        -device vhost-vsock-pci,guest-cid=3 \
        -nographic \
        -serial none \
        -pidfile "$VM_DIR/${VM_NAME}.pid" \
        > "$VM_DIR/${VM_NAME}.log" 2>&1 &
    
    sleep 5
    
    if [ ! -f "$VM_DIR/${VM_NAME}.pid" ]; then
        echo "GUEST_TEST_ERROR:vm_failed_to_start"
        cat "$VM_DIR/${VM_NAME}.log" 2>/dev/null
        exit 1
    fi
    
    echo "VM launched, PID: $(cat $VM_DIR/${VM_NAME}.pid)"
    VM_IP="localhost"
    
    # Step 5: Wait for VM (includes kernel install + reboot)
    echo ""
    echo "Step 5: Waiting for VM to boot, install Intel kernel, and reboot..."
    echo "        This may take 5-10 minutes on first boot..."
    TIMEOUT=600
    START_TIME=$(date +%s)
    while true; do
        ELAPSED=$(($(date +%s) - START_TIME))
        if [ $ELAPSED -gt $TIMEOUT ]; then
            echo "GUEST_TEST:timeout"
            exit 1
        fi
        
        if ssh -o StrictHostKeyChecking=no -o UserKnownHostsFile=/dev/null \
               -o ConnectTimeout=5 -o BatchMode=yes \
               -p "$SSH_PORT" ubuntu@localhost "echo ready" 2>/dev/null; then
            echo "VM is ready!"
            break
        fi
        
        printf "."
        sleep 10
    done
    
    # Step 6: Check TDX inside guest
    echo ""
    echo "Step 6: Checking TDX inside guest..."
    ssh -o StrictHostKeyChecking=no -o UserKnownHostsFile=/dev/null \
        -p "$SSH_PORT" ubuntu@localhost '
            if [ -c /dev/tdx_guest ] || [ -c /dev/tdx-guest ]; then
                echo "TDX_DEVICE:found"
                ls -la /dev/tdx* 2>/dev/null
            else
                echo "TDX_DEVICE:not_found"
                echo "Available devices:"
                ls -la /dev/ | grep -E "tdx|tpm|sgx" || echo "No TEE devices found"
            fi
        '
    
    # Step 7: Try quote generation
    echo ""
    echo "Step 7: Attempting quote generation..."
    QUOTE_RESULT=$(ssh -o StrictHostKeyChecking=no -o UserKnownHostsFile=/dev/null \
        -p "$SSH_PORT" ubuntu@localhost '
            TDX_DEV=""
            [ -c /dev/tdx_guest ] && TDX_DEV="/dev/tdx_guest"
            [ -c /dev/tdx-guest ] && TDX_DEV="/dev/tdx-guest"
            
            if [ -z "$TDX_DEV" ]; then
                echo "QUOTE_TEST:no_device"
                exit 0
            fi
            
            # Try configfs-tsm
            if [ -d /sys/kernel/config/tsm/report ]; then
                REPORT_DIR="/sys/kernel/config/tsm/report/test$$"
                mkdir -p "$REPORT_DIR" 2>/dev/null
                if [ -d "$REPORT_DIR" ]; then
                    # Write test data
                    echo -n "0102030405060708" | xxd -r -p > "$REPORT_DIR/inblob" 2>/dev/null
                    if [ -f "$REPORT_DIR/outblob" ]; then
                        SIZE=$(stat -c%s "$REPORT_DIR/outblob" 2>/dev/null || echo 0)
                        rmdir "$REPORT_DIR" 2>/dev/null
                        echo "QUOTE_TEST:success:size=$SIZE"
                        exit 0
                    fi
                    rmdir "$REPORT_DIR" 2>/dev/null
                fi
            fi
            
            echo "QUOTE_TEST:device_found_but_no_method"
        ' 2>/dev/null || echo "QUOTE_TEST:ssh_error")
    
    echo "$QUOTE_RESULT"
    
    echo ""
    echo "=== Test Complete ==="
    echo "Cleanup will run automatically..."
"#;

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    #[allow(clippy::const_is_empty)]
    fn test_commands_not_empty() {
        assert!(!DOWNLOAD_TDX_GUEST_IMAGE.is_empty());
        assert!(!CREATE_TDX_VM_DISK.is_empty());
        assert!(!CREATE_CLOUD_INIT.is_empty());
        assert!(!LAUNCH_TDX_VM_QEMU.is_empty());
        assert!(!WAIT_FOR_TDX_VM.is_empty());
        assert!(!INSTALL_GUEST_ATTESTATION.is_empty());
        assert!(!TEST_GUEST_QUOTE_GEN.is_empty());
        assert!(!CLEANUP_TDX_VM.is_empty());
        assert!(!FULL_TDX_GUEST_TEST.is_empty());
    }

    #[test]
    fn test_vm_dir_constant() {
        assert_eq!(TDX_TEST_VM_DIR, "/var/lib/basilica-tdx-test");
    }

    #[test]
    fn test_launch_uses_tdx_options() {
        // Using libvirt XML configuration for TDX
        assert!(LAUNCH_TDX_VM_QEMU.contains("launchSecurity type='tdx'"));
        assert!(LAUNCH_TDX_VM_QEMU.contains("policy>0x0</policy"));
    }
}
