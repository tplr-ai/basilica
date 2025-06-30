#!/bin/bash
set -e

# Simple, idempotent remote server setup for Basilica
# Usage: ./scripts/setup-ops/remote.sh <role> <user@host> [-p port]
# Example: ./scripts/setup-ops/remote.sh validator root@51.159.183.42 -p 43738

ROLE="$1"
HOST="$2"
PORT_FLAG="$3"
PORT="$4"

if [[ $# -lt 2 ]]; then
    echo "Usage: $0 <role> <user@host> [-p port]"
    echo "Roles: validator, miner, executor"
    echo "Example: $0 validator root@51.159.183.42 -p 43738"
    exit 1
fi

# Construct SSH command
if [[ "$PORT_FLAG" == "-p" && -n "$PORT" ]]; then
    SSH_CMD="ssh $HOST $PORT_FLAG $PORT"
else
    SSH_CMD="ssh $HOST"
fi

echo "=== Setting up $ROLE on $HOST ==="

# Install basic dependencies
echo "Installing basic dependencies..."
$SSH_CMD "
apt update && apt install -y \
    curl wget git build-essential pkg-config libssl-dev \
    ca-certificates gnupg lsb-release rsync
"

# Install Rust (idempotent)
echo "Installing Rust..."
$SSH_CMD "
if ! command -v rustc &> /dev/null; then
    curl --proto '=https' --tlsv1.2 -sSf https://sh.rustup.rs | sh -s -- -y
fi
source ~/.cargo/env || true
rustc --version
"

# Install Docker (idempotent)
echo "Installing Docker..."
$SSH_CMD "
if ! command -v docker &> /dev/null; then
    curl -fsSL https://download.docker.com/linux/ubuntu/gpg | gpg --dearmor -o /usr/share/keyrings/docker-archive-keyring.gpg
    DISTRO=\$(lsb_release -cs)
    echo \"deb [arch=amd64 signed-by=/usr/share/keyrings/docker-archive-keyring.gpg] https://download.docker.com/linux/ubuntu \$DISTRO stable\" > /etc/apt/sources.list.d/docker.list
    apt update
    apt install -y docker-ce docker-ce-cli containerd.io docker-compose-plugin
fi
service docker start 2>/dev/null || true
docker --version
"

# Setup NVIDIA Docker for executor
if [[ "$ROLE" == "executor" ]]; then
    echo "Setting up NVIDIA Docker runtime for executor..."
    $SSH_CMD "
    if ! grep -q nvidia /etc/docker/daemon.json 2>/dev/null; then
        distribution=\$(. /etc/os-release;echo \$ID\$VERSION_ID)
        curl -fsSL https://nvidia.github.io/libnvidia-container/gpgkey | gpg --dearmor -o /usr/share/keyrings/nvidia-container-toolkit-keyring.gpg
        curl -s -L https://nvidia.github.io/libnvidia-container/\$distribution/libnvidia-container.list | sed 's#deb https://#deb [signed-by=/usr/share/keyrings/nvidia-container-toolkit-keyring.gpg] https://#g' > /etc/apt/sources.list.d/nvidia-container-toolkit.list
        apt update
        apt install -y nvidia-container-toolkit
        nvidia-ctk runtime configure --runtime=docker
        pkill dockerd 2>/dev/null || true
        dockerd --host=unix:///var/run/docker.sock &
        sleep 5
    fi
    nvidia-smi --query-gpu=name --format=csv,noheader,nounits || echo 'GPU check failed'
    "
fi

# Setup SSH keys for cross-machine access
echo "Setting up SSH keys..."
$SSH_CMD "
mkdir -p ~/.ssh
chmod 700 ~/.ssh

# Generate key if doesn't exist
if [[ ! -f ~/.ssh/basilica ]]; then
    ssh-keygen -t ed25519 -f ~/.ssh/basilica -N ''
fi

# Show public key
echo 'Public key for this machine:'
cat ~/.ssh/basilica.pub
"

# Create workspace directory
echo "Creating workspace..."
$SSH_CMD "
mkdir -p /opt/basilica
echo 'Workspace created at /opt/basilica'
"

# Final verification
echo "Final verification..."
$SSH_CMD "
echo '=== $ROLE SETUP COMPLETE ==='
echo 'System:' \$(uname -a)
echo 'Docker:' \$(docker --version)
source ~/.cargo/env
echo 'Rust:' \$(rustc --version)
echo 'OpenSSL:' \$(openssl version)
echo 'Rsync:' \$(rsync --version | head -1)
if [[ '$ROLE' == 'executor' ]]; then
    echo 'NVIDIA:' \$(nvidia-smi --query-gpu=name --format=csv,noheader,nounits 2>/dev/null || echo 'Not available')
fi
echo 'SSH key available at ~/.ssh/basilica.pub'
echo 'Workspace: /opt/basilica'
echo '=== SETUP COMPLETE ==='
"

echo ""
echo "=== NEXT STEPS ==="
echo "Run './scripts/setup-ops/setup-ssh.sh' to automatically:"
echo "1. Collect public keys from all servers"
echo "2. Distribute keys to all machines"
echo "3. Test cross-machine connectivity"
echo ""
echo "Or run './scripts/setup-ops/setup-all.sh' for complete automation"