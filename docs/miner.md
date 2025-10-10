# Basilica Miner Guide

Comprehensive guide for running a Basilica miner node that provides GPU compute resources to the Bittensor network.

---

## Quick Start (TL;DR)

**What it does**: Miner orchestrates validator access to your GPU nodes via SSH. No executor binaries needed.

**Minimum Requirements**:

- Miner server: Linux with 8+ CPU cores, 16GB RAM, public IP
- GPU node(s): NVIDIA GPU (A100/H100/B200), CUDA ≥12.8, Docker with nvidia runtime
- Bittensor wallet registered on subnet 39 (mainnet) or 387 (testnet)

**Quick Setup** (5 steps):

```bash
# 1. Generate SSH key for node access
ssh-keygen -t ed25519 -f ~/.ssh/miner_node_key -N ""

# 2. Deploy key to GPU nodes
ssh-copy-id -i ~/.ssh/miner_node_key.pub basilica@<gpu_node_ip>

# 3. Create minimal config (also see config/miner.prod.toml)
cat > miner.toml <<EOF
[bittensor]
wallet_name = "your_wallet"
hotkey_name = "your_hotkey"
external_ip = "your_public_ip"
axon_port = 8080
network = "finney"
netuid = 39
weight_interval_secs = 300

[node_management]
nodes = [
  { host = "<node 1 IP>", port = 22, username = "root" },
  { host = "<node 2 IP>", port = 22, username = "root" },
]

[validator_comms]
host = "0.0.0.0"
port = 8080

[ssh_session]
miner_node_key_path = "~/.ssh/miner_node_key"

[validator_assignment]
enabled = true
strategy = "highest_stake"
min_stake_threshold = 12000.0
validator_hotkey = "5G3qVaXzKMPDm5AJ3dpzbpUC27kpccBvDwzSWXrq8M6qMmbC"
EOF

# 4. Build and run (with docker compose)
cp ./scripts/miner/compose.prod.yml compose.yml
docker compose up -d

# Check status
docker compose ps
# View logs
docker compose logs -f miner

# 4. Build and run (with self compiled binary)
./scripts/miner/build.sh
./basilica-miner --config miner.toml

# 5. Verify validators can discover nodes
# Check logs for "Node registered" and "Validator authenticated" messages
```

**Need details?** See sections below for architecture explanation, security hardening, troubleshooting, and advanced configuration.

---

## Table of Contents

1. [Overview](#overview)
2. [Architecture](#architecture)
3. [Prerequisites](#prerequisites)
4. [Understanding the System](#understanding-the-system)
5. [SSH Key Setup](#ssh-key-setup)
6. [GPU Node Preparation](#gpu-node-preparation)
7. [Miner Configuration](#miner-configuration)
8. [Deployment Methods](#deployment-methods)
9. [Validator Access Flow](#validator-access-flow)
10. [Security & Best Practices](#security--best-practices)
11. [Monitoring](#monitoring)
12. [Troubleshooting](#troubleshooting)
13. [Advanced Topics](#advanced-topics)

---

## Overview

The Basilica miner manages a fleet of GPU nodes and provides validators with **direct SSH access** to these nodes for verification and rental operations. Unlike traditional architectures that require intermediary agents, Basilica miners act as **access control orchestrators**, deploying validator SSH keys to nodes on-demand.

### What You Need

- **Miner Server**: Linux system with network connectivity (no GPU required)
  - 8+ CPU cores, 16GB+ RAM recommended
  - Public IP address or port forwarding
  - SSH access to your GPU nodes

- **GPU Nodes**: One or more servers with:
  - NVIDIA GPU (A100, H100, or B200 supported)
  - NVIDIA CUDA drivers version ≥12.8
  - Linux OS with SSH server
  - Docker installed (for container workloads with nvidia runtime)
  - All ports need to be open, the NAT or firewall should allow inbound SSH connections from the miner and validator server
  - the validator shall be in control of which ports need to have open internet access
  - at least 1TB of free disk space recommended (for container images and data)

- **Bittensor Wallet**: Registered on subnet 39 (mainnet) or 387 (testnet)

---

## Architecture

### Direct SSH Access Model

```text
┌──────────────────┐
│   Validator      │
│                  │
└────────┬─────────┘
         │
         │ 1. Discovery Request
         │    (with SSH public key and signature)
         ↓
┌──────────────────┐
│   Miner (gRPC)   │
│  - Authenticates │
│  - Deploys keys  │
│  - Returns nodes │
└────────┬─────────┘
         │
         │ 2. SSH Key Deployment
         │    (miner → nodes)
         |────────────────────────────|────────────────────────────|
┌──────────────────┐         ┌──────────────────┐         ┌──────────────────┐
│   GPU Node 1     │         │   GPU Node 2     │         │   GPU Node N     │
│  (SSH endpoint)  │         │  (SSH endpoint)  │         │  (SSH endpoint)  │
└────────▲─────────┘         └────────▲─────────┘         └────────▲─────────┘
         │                            │                            │
         └────────────────────────────┴────────────────────────────┘
                            3. Validator connects directly via SSH
```

---

## Prerequisites

### On Miner Server

```bash
# Update system
sudo apt update && sudo apt upgrade -y

# Install build dependencies
sudo apt install -y \
    build-essential \
    libssl-dev \
    pkg-config \
    protobuf-compiler \
    git \
    curl

# Install Rust (if building from source)
curl --proto '=https' --tlsv1.2 -sSf https://sh.rustup.rs | sh
source ~/.cargo/env

# Install Docker (optional, for Docker deployment)
curl -fsSL https://get.docker.com | sudo sh
sudo usermod -aG docker $USER
```

### On GPU Nodes

```bash
# Install NVIDIA drivers and CUDA (if not already installed)
# Follow NVIDIA's official installation guide for your GPU model

# Install Docker
curl -fsSL https://get.docker.com | sudo sh

# Install NVIDIA Container Toolkit
distribution=$(. /etc/os-release;echo $ID$VERSION_ID)
curl -fsSL https://nvidia.github.io/libnvidia-container/gpgkey | sudo gpg --dearmor -o /usr/share/keyrings/nvidia-container-toolkit-keyring.gpg
curl -s -L https://nvidia.github.io/libnvidia-container/$distribution/libnvidia-container.list | \
    sed 's#deb https://#deb [signed-by=/usr/share/keyrings/nvidia-container-toolkit-keyring.gpg] https://#g' | \
    sudo tee /etc/apt/sources.list.d/nvidia-container-toolkit.list
sudo apt update
sudo apt install -y nvidia-container-toolkit
sudo nvidia-ctk runtime configure --runtime=docker
sudo systemctl restart docker

# Verify GPU access
nvidia-smi
docker run --rm --gpus all nvidia/cuda:12.8.0-base-ubuntu24.04 nvidia-smi
```

### Bittensor Wallet Setup

```bash
# Install btcli if not already installed
pip install bittensor

# Create wallet (if you don't have one)
btcli wallet new_coldkey --wallet.name miner_wallet
btcli wallet new_hotkey --wallet.name miner_wallet --wallet.hotkey default

# Register on subnet (requires TAO for registration fee)
# Mainnet (subnet 39)
btcli subnet register --netuid 39 --wallet.name miner_wallet --wallet.hotkey default

# Testnet (subnet 387)
btcli subnet register --netuid 387 --wallet.name miner_wallet --wallet.hotkey default --subtensor.network test
```

**Verify wallet location:**

```bash
ls ~/.bittensor/wallets/miner_wallet/hotkeys/default
# Should show the hotkey file
```

---

## Understanding the System

### Node Identity

Nodes are identified by **deterministic UUIDs** generated from their SSH credentials:

```text
node_id = UUID_v5_namespace(username@host:port)
```

**Key characteristics:**

- Same credentials always generate the same node ID
- No database persistence required for node identity
- Node IDs stored in miner's SQLite database
- Path: `~/.bittensor/wallets/{wallet_name}/hotkeys/{hotkey_name}`

**Example:**

```toml
[node_management]
nodes = [
  { host = "192.168.1.100", port = 22, username = "basilica" }
]
```

Generates: `node_id = "a3f2b1c4-5d6e-7f8a-9b0c-1d2e3f4a5b6c"` (deterministic)

### Authentication Flow

Basilica uses **mutual authentication** between validators and miners:

#### Phase 1: Validator → Miner Authentication

1. **Validator sends authentication request:**

   ```rust
   ValidatorAuthRequest {
     validator_hotkey: "5G3qVaXz..." (SS58-encoded Bittensor hotkey)
     signature: Ed25519 signature
     nonce: "uuid-1234-5678"
     timestamp: Unix timestamp
     target_miner_hotkey: "5FHd7..."
   }
   ```

2. **Signature payload:**

   ```text
   BASILICA_AUTH_V1:{nonce}:{target_miner_hotkey}:{timestamp_seconds}
   ```

3. **Miner verifies:**
   - Signature is valid for validator's hotkey
   - Timestamp is within 5 minutes (prevents replay attacks)
   - `target_miner_hotkey` matches miner's actual hotkey (prevents MITM)

#### Phase 2: Miner → Validator Authentication (Mutual)

1. **Miner signs response:**

   ```rust
   MinerAuthResponse {
     authenticated: true
     session_token: "hex-token-32-bytes"
     expires_at: timestamp + 3600 seconds
     miner_hotkey: "5FHd7..."
     miner_signature: Ed25519 signature
     response_nonce: "uuid-8765-4321"
   }
   ```

2. **Miner signature payload:**

   ```text
   MINER_AUTH_RESPONSE:{validator_hotkey}:{response_nonce}:{session_token}
   ```

3. **Validator verifies miner's signature** to ensure communication with legitimate miner

### SSH Key Deployment Mechanism

**When validator discovers nodes:**

1. **Validator includes SSH public key in discovery request:**

   ```rust
   DiscoverNodesRequest {
     validator_hotkey: "5G3qVaXz..."
     validator_public_key: "ssh-ed25519 AAAAC3NzaC1lZDI1NTE5..."
     signature: ...
     nonce: ...
   }
   ```

2. **Miner validates SSH key format:**
   - Supports: `ssh-rsa`, `ssh-ed25519`, `ecdsa-sha2-*`, `ssh-dss`
   - Rejects malformed keys

3. **Miner deploys key to ALL nodes:**

   ```bash
   # For each node, miner executes via SSH:
   mkdir -p ~/.ssh && \
   echo 'ssh-ed25519 AAAAC3... validator-5G3qVaXz...' >> ~/.ssh/authorized_keys && \
   chmod 600 ~/.ssh/authorized_keys
   ```

4. **Key tagging**: Keys are tagged with `validator-{hotkey}` for easy revocation:

   ```text
   ssh-ed25519 AAAAC3NzaC1lZDI1NTE5... validator-5G3qVaXzKMPDm5AJ3dpzbpUC27kpccBvDwzSWXrq8M6qMmbC
   ```

5. **Miner returns node connection details:**

   ```rust
   NodeConnectionDetails {
     node_id: "a3f2b1c4-..."
     host: "192.168.1.100"
     port: "22"
     username: "basilica"
     additional_opts: "-o StrictHostKeyChecking=no"
     status: "available"
   }
   ```

6. **Validator connects directly:**

   ```bash
   ssh -i ~/.basilica/ssh/validator_persistent.pem basilica@192.168.1.100
   ```

### Validator Assignment Strategies

Miners can control which validators receive access to their nodes:

#### 1. **Highest Stake** (Recommended)

Assigns ALL nodes to the validator with highest stake above threshold.

```toml
[validator_assignment]
strategy = "highest_stake"
min_stake_threshold = 12000.0  # TAO
validator_hotkey = "5G3qVaXzKMPDm5AJ3dpzbpUC27kpccBvDwzSWXrq8M6qMmbC"
```

**Use cases:**

- Production deployment with single trusted validator
- Maximize security by working with most invested validator
- Simplify operations (single validator relationship)

**Behavior:**

- Fetches validators from Bittensor metagraph
- Filters by `validator_permit = true` and stake ≥ threshold
- If `validator_hotkey` specified: validates it meets criteria, uses it
- Otherwise: selects highest-staked validator
- Only considers online validators (with axon endpoints)

#### 2. **Disabled** (Open Access)

All validators can discover and access all nodes.

```toml
[validator_assignment]
enabled = false
```

**⚠️ Warning**: Not recommended for production. Increases security surface.

### Discovery Process

**Validator's perspective:**

1. **Query Bittensor metagraph** for miners on subnet
2. **Extract miner endpoints** from axon data
3. **Connect to miner's gRPC service** (port 8080 by default)
4. **Authenticate** using Bittensor hotkey signature
5. **Send discovery request** with SSH public key
6. **Receive node connection details**
7. **SSH directly to nodes** for validation/rental

**Miner's perspective:**

1. **Register nodes** from config at startup
2. **Start gRPC server** listening on configured port
3. **Wait for validator connections**
4. **Verify validator signatures** and authorization
5. **Deploy validator SSH keys** to nodes automatically
6. **Return node details** to authorized validators
7. **Monitor node health** and update status

---

## SSH Key Setup

Proper SSH key management is **critical** for security and functionality.

### Generate Miner's SSH Key

The miner needs an SSH key to access your GPU nodes for key deployment.

```bash
# Generate Ed25519 key (recommended for security and performance)
ssh-keygen -t ed25519 -f ~/.ssh/miner_node_key -C "basilica-miner" -N ""

# Set proper permissions (critical for security)
chmod 600 ~/.ssh/miner_node_key
chmod 644 ~/.ssh/miner_node_key.pub

# Verify key was created
ls -la ~/.ssh/miner_node_key*
```

**Alternative: RSA key** (if Ed25519 not supported):

```bash
ssh-keygen -t rsa -b 4096 -f ~/.ssh/miner_node_key -C "basilica-miner" -N ""
```

### Deploy Miner's Public Key to GPU Nodes

The miner needs SSH access to deploy validator keys.

**For each GPU node:**

```bash
# Copy public key to node
ssh-copy-id -i ~/.ssh/miner_node_key.pub basilica@192.168.1.100

# Or manually:
cat ~/.ssh/miner_node_key.pub | ssh basilica@192.168.1.100 \
  "mkdir -p ~/.ssh && cat >> ~/.ssh/authorized_keys && chmod 600 ~/.ssh/authorized_keys"
```

**Verify access:**

```bash
ssh -i ~/.ssh/miner_node_key basilica@192.168.1.100 "hostname && nvidia-smi --query-gpu=name --format=csv,noheader"
```

Expected output:

```text
gpu-node-1
NVIDIA H100 PCIe
```

### SSH Configuration Best Practices

**On Miner Server** (`~/.ssh/config`):

```text
# Miner's SSH configuration for GPU nodes
Host gpu-node-*
    User basilica
    IdentityFile ~/.ssh/miner_node_key
    StrictHostKeyChecking no
    UserKnownHostsFile /dev/null
    ServerAliveInterval 60
    ServerAliveCountMax 3
    ConnectTimeout 30
```

**On GPU Nodes** (`/etc/ssh/sshd_config`):

```text
# Security hardening
PermitRootLogin no
PasswordAuthentication no
PubkeyAuthentication yes
ChallengeResponseAuthentication no
UsePAM yes

# Performance
MaxStartups 30:30:100
MaxSessions 100

# Keep connections alive
ClientAliveInterval 60
ClientAliveCountMax 3
```

After editing `sshd_config`:

```bash
sudo systemctl restart sshd
```

---

## GPU Node Preparation

### Create Dedicated User Account

**On each GPU node:**

```bash
# Create user for validator access
sudo useradd -m -s /bin/bash basilica

# Add to docker group (for container workloads)
sudo usermod -aG docker basilica

# Optional: Add to sudo group (if validators need elevated privileges)
# sudo usermod -aG sudo basilica

# Set up SSH directory
sudo -u basilica mkdir -p /home/basilica/.ssh
sudo chmod 700 /home/basilica/.ssh
sudo -u basilica touch /home/basilica/.ssh/authorized_keys
sudo chmod 600 /home/basilica/.ssh/authorized_keys
```

### Verify GPU Access

```bash
# Test as basilica user
sudo -u basilica nvidia-smi

# Test Docker GPU access
sudo -u basilica docker run --rm --gpus all nvidia/cuda:12.8.0-base-ubuntu24.04 nvidia-smi
```

### Network Configuration

**Ensure SSH port is accessible:**

```bash
# Check SSH is running
sudo systemctl status sshd

# Allow SSH through firewall (if using UFW)
sudo ufw allow 22/tcp
sudo ufw enable

# For custom SSH port:
sudo ufw allow 2222/tcp
```

**Verify connectivity from miner server:**

```bash
# From miner server
ssh -i ~/.ssh/miner_node_key basilica@<node-ip> "echo 'Connection successful'"
```

### GPU Node Checklist

Before adding nodes to miner config, verify:

- [ ] NVIDIA drivers installed (`nvidia-smi` works)
- [ ] Docker installed and configured with NVIDIA runtime
- [ ] Dedicated user account created (`basilica` or similar)
- [ ] User added to `docker` group
- [ ] SSH server running and accessible
- [ ] Miner's SSH public key deployed to node
- [ ] Firewall allows SSH connections from miner
- [ ] GPU accessible to Docker containers
- [ ] Sufficient disk space for containers/data

---

## Miner Configuration

```text
### Using the Simplified Configuration

Use the simplified configuration template provided:

```bash
# Copy the simplified config
cp config/miner.simplified.toml miner.toml

# Edit with your settings
vim miner.toml
```

### Essential Configuration Sections

#### 1. Bittensor Network Configuration

```toml
[bittensor]
# Wallet configuration (path: ~/.bittensor/wallets/{wallet_name}/hotkeys/{hotkey_name})
wallet_name = "miner_wallet"        # Your coldkey/wallet name
hotkey_name = "default"              # Your hotkey name

# Network settings
network = "finney"                   # Options: finney (mainnet), test, local
netuid = 39                          # Basilica subnet ID (39=mainnet, 387=testnet)
weight_interval_secs = 300           # Weight setting interval (5 minutes)

# Axon configuration (for Bittensor network registration)
external_ip = "<your public ip>"     # YOUR SERVER'S PUBLIC IP
axon_port = 8080

# Advanced settings (usually don't need to change)
max_weight_uids = 256
```

```bash
# Find your public IP
curl -4 ifconfig.me
```

#### 2. Database Configuration

```toml
[database]
url = "sqlite:///opt/basilica/data/miner.db"
run_migrations = true
```

**Ensure database directory exists:**

```bash
sudo mkdir -p /opt/basilica/data
sudo chown $USER:$USER /opt/basilica/data
```

#### 3. Validator Communications (gRPC Server)

```toml
[validator_comms]
host = "0.0.0.0"      # Internal binding (0.0.0.0 = all interfaces)
port = 8080            # gRPC server port (must match external_ip routing)
```

**⚠️ Firewall**: Ensure port 8080 is accessible:

```bash
# UFW
sudo ufw allow 8080/tcp

# Or iptables
sudo iptables -A INPUT -p tcp --dport 8080 -j ACCEPT
```

#### 4. GPU Node Management

```toml
[node_management]
# List your GPU compute nodes with SSH access details
nodes = [
  { host = "192.168.1.100", port = 22, username = "basilica" },
  { host = "192.168.1.101", port = 22, username = "basilica" },
  { host = "10.0.0.50", port = 2222, username = "gpu_user" },
]
```

**Node configuration fields:**

- `host`: IP address or hostname of GPU node
- `port`: SSH port (typically 22)
- `username`: SSH username on the node
- `additional_opts` (optional): Extra SSH options like `"-o StrictHostKeyChecking=no"`

#### 5. SSH Access Configuration

```toml
[ssh_session]
# Path to your SSH private key for accessing nodes
miner_node_key_path = "~/.ssh/miner_node_key"

# Default username for SSH access to nodes (used as fallback)
default_node_username = "node"
```

**Verify key path is correct:**

```bash
ls -la ~/.ssh/miner_node_key
# Should show: -rw------- (permissions 600)
```

#### 6. Security Configuration

```toml
[security]
verify_signatures = true   # ALWAYS true for production

# Optional: Ethereum private key for collateral contract (advanced)
# private_key_file = "/opt/basilica/keys/private_key.pem"
```

#### 7. Metrics Configuration

```toml
[metrics]
enabled = true

[metrics.prometheus]
host = "127.0.0.1"    # Bind to localhost for security
port = 9090
```

**Access metrics:**

```bash
curl http://localhost:9090/metrics
```

#### 8. Validator Assignment Strategy

```toml
[validator_assignment]
strategy = "highest_stake"           # Options: highest_stake, fixed_assignment
min_stake_threshold = 12000.0        # Minimum TAO stake required

# Optional: Assign to specific validator
# validator_hotkey = "5G3qVaXzKMPDm5AJ3dpzbpUC27kpccBvDwzSWXrq8M6qMmbC"
```

**Choosing a strategy:**

- **Production**: Use `highest_stake` with high threshold (≥12000 TAO)
- **Testing**: Use `highest_stake` to assign all nodes to the top validator
- **Development**: Disable assignment by omitting this section

#### 9. Advertised Addresses (Optional)

Override auto-detected addresses for NAT/proxy scenarios:

```toml
[advertised_addresses]
# Only needed if miner is behind NAT/proxy
# grpc_endpoint = "http://203.0.113.45:8080"
# axon_endpoint = "http://203.0.113.45:8080"
# metrics_endpoint = "http://203.0.113.45:9090"
```

### Configuration Validation

Before starting the miner, validate your configuration:

```bash
# Validate configuration
cargo run -p basilica-miner -- --config miner.toml config validate

# Expected output:
# Configuration validation passed
```

**If validation fails**, check:

- Wallet names match your actual wallet
- External IP is correct
- Database directory exists
- SSH key paths are valid
- Node SSH credentials are accessible

---

## Deployment Methods

Choose the deployment method that best fits your infrastructure.

### Method 1: Binary Deployment (Simplest)

Best for: Development, testing, simple setups.

#### Build the Binary

```bash
# Clone repository
git clone https://github.com/your-org/basilica.git
cd basilica/basilica

# Build miner binary using build script
./scripts/miner/build.sh

# Binary will be at: ./basilica-miner
```

**Build options:**

```bash
# Release build (optimized, recommended for production)
./scripts/miner/build.sh --release
```

#### Deploy and Run

```bash
# Create data directory
sudo mkdir -p /opt/basilica/data
sudo mkdir -p /opt/basilica/config
sudo chown -R $USER:$USER /opt/basilica

# Copy binary and config
sudo cp basilica-miner /opt/basilica/
sudo cp miner.toml /opt/basilica/config/

# Copy Bittensor wallet
sudo mkdir -p /root/.bittensor
sudo cp -r ~/.bittensor/wallets /root/.bittensor/

# Run miner
cd /opt/basilica
sudo ./basilica-miner --config config/miner.toml
```

**⚠️ Note:** Miner requires root/sudo for:

- SSH key deployment to nodes
- Database access (if in protected directory)
- Network port binding (if port < 1024)

### Method 2: Systemd Service (Production)

Best for: Production deployments requiring auto-restart and logging.

#### Create Systemd Service

```bash
# Create service file
sudo tee /etc/systemd/system/basilica-miner.service > /dev/null <<EOF
[Unit]
Description=Basilica Miner
After=network-online.target
Wants=network-online.target

[Service]
Type=simple
User=root
WorkingDirectory=/opt/basilica
ExecStart=/opt/basilica/basilica-miner --config /opt/basilica/config/miner.toml
Restart=always
RestartSec=10
StandardOutput=journal
StandardError=journal
SyslogIdentifier=basilica-miner

# Security hardening
NoNewPrivileges=true
PrivateTmp=true
ProtectSystem=strict
ReadWritePaths=/opt/basilica/data /var/log/basilica

[Install]
WantedBy=multi-user.target
EOF
```

#### Enable and Start

```bash
# Reload systemd
sudo systemctl daemon-reload

# Enable auto-start on boot
sudo systemctl enable basilica-miner

# Start service
sudo systemctl start basilica-miner

# Check status
sudo systemctl status basilica-miner

# View logs
sudo journalctl -u basilica-miner -f
```

#### Service Management Commands

```bash
# Stop miner
sudo systemctl stop basilica-miner

# Restart miner
sudo systemctl restart basilica-miner

# Disable auto-start
sudo systemctl disable basilica-miner

# View logs (last 100 lines)
sudo journalctl -u basilica-miner -n 100

# View logs (follow in real-time)
sudo journalctl -u basilica-miner -f
```

### Method 3: Docker (Containerized)

Best for: Isolated environments, easy updates, multi-host deployments.

#### Build Docker Image

```bash
# Build using provided Dockerfile
docker build -f scripts/miner/Dockerfile -t basilica-miner:latest .

# Or pull from registry (if available)
docker pull ghcr.io/your-org/basilica/miner:latest
```

#### Run Container

```bash
# Create required directories
sudo mkdir -p /opt/basilica/config
sudo mkdir -p /opt/basilica/data

# Copy configuration
sudo cp miner.toml /opt/basilica/config/

# Run miner container
docker run -d \
  --name basilica-miner \
  --restart unless-stopped \
  -v ~/.bittensor:/root/.bittensor:ro \
  -v /opt/basilica/config:/opt/basilica/config:ro \
  -v /opt/basilica/data:/opt/basilica/data \
  -v ~/.ssh:/root/.ssh:ro \
  -p 8080:8080 \
  -p 9090:9090 \
  basilica-miner:latest --config /opt/basilica/config/miner.toml
```

**Volume mappings explained:**

- `~/.bittensor` - Bittensor wallet (read-only)
- `/opt/basilica/config` - Miner configuration (read-only)
- `/opt/basilica/data` - Database and logs (read-write)
- `~/.ssh` - SSH keys for node access (read-only)

#### Container Management

```bash
# View logs
docker logs -f basilica-miner

# Stop container
docker stop basilica-miner

# Start container
docker start basilica-miner

# Remove container
docker rm -f basilica-miner

# Exec into container (debugging)
docker exec -it basilica-miner /bin/bash
```

### Method 4: Docker Compose (Production with Auto-Updates)

Best for: Production deployments with automatic updates and monitoring.

#### Create docker-compose.yml

```bash
# Navigate to miner scripts directory
cd scripts/miner

# Use provided production compose file
cp compose.prod.yml docker-compose.yml
```

**Or create custom compose file:**

```yaml
version: '3.8'

services:
  miner:
    image: ghcr.io/your-org/basilica/miner:latest
    container_name: basilica-miner
    restart: unless-stopped
    volumes:
      - ~/.bittensor:/root/.bittensor:ro
      - ./config/miner.toml:/opt/basilica/config/miner.toml:ro
      - ./data:/opt/basilica/data
      - ~/.ssh:/root/.ssh:ro
      - /var/log/basilica:/var/log/basilica
    ports:
      - "8080:8080"
      - "9090:9090"
    command: ["--config", "/opt/basilica/config/miner.toml"]
    healthcheck:
      test: ["CMD", "curl", "-f", "http://localhost:8080/health"]
      interval: 30s
      timeout: 10s
      retries: 3
      start_period: 40s

  watchtower:
    image: containrrr/watchtower:latest
    container_name: basilica-watchtower
    restart: unless-stopped
    volumes:
      - /var/run/docker.sock:/var/run/docker.sock
    command: ["--cleanup", "--interval", "300", "basilica-miner"]
```

#### Deploy

```bash
# Ensure config exists
sudo cp miner.toml /opt/basilica/config/

# Start services
docker compose up -d

# View logs
docker compose logs -f miner

# Check status
docker compose ps
```

**Watchtower** automatically:

- Checks for image updates every 5 minutes
- Pulls new images when available
- Restarts containers with new images
- Cleans up old images

---

## Validator Access Flow

Understanding how validators access your nodes helps with troubleshooting and security.

### Step-by-Step Flow

#### 1. Validator Discovery

**Validator queries Bittensor metagraph:**

```bash
# Validator perspective (simplified)
# Query metagraph for miners on subnet 39
validators_list = bittensor_service.get_neurons(netuid=39, filter="miners")

# Extract miner endpoints
for miner in validators_list:
    miner_endpoint = miner.axon.ip + ":" + miner.axon.port
    # miner_endpoint = "203.0.113.45:8080"
```

#### 2. gRPC Connection

**Validator connects to miner's gRPC service:**

```bash
# Validator connects to miner
grpc_endpoint = "http://203.0.113.45:8080"
miner_client = MinerDiscoveryClient(grpc_endpoint)
```

#### 3. Authentication

**Validator authenticates with Bittensor signature:**

```python
# Validator generates signature
nonce = uuid4()
timestamp = int(time.time())
payload = f"BASILICA_AUTH_V1:{nonce}:{miner_hotkey}:{timestamp}"
signature = validator_keypair.sign(payload.encode())

# Send authentication request
auth_response = miner_client.authenticate_validator(
    validator_hotkey=validator_hotkey,
    signature=signature,
    nonce=nonce,
    timestamp=timestamp,
    target_miner_hotkey=miner_hotkey
)

# Verify miner's signature in response
session_token = auth_response.session_token
```

#### 4. Node Discovery with SSH Key

**Validator sends SSH public key:**

```python
# Validator's persistent SSH key
ssh_public_key = load_public_key("~/.basilica/ssh/validator_persistent.pem.pub")

# Discover nodes with SSH key
discovery_response = miner_client.discover_nodes(
    validator_hotkey=validator_hotkey,
    validator_public_key=ssh_public_key,
    signature=signature,
    nonce=uuid4()
)

# Response contains node connection details
nodes = discovery_response.nodes
# [
#   NodeConnectionDetails(node_id="a3f2b1c4-...", host="192.168.1.100", port="22", username="basilica"),
#   NodeConnectionDetails(node_id="b4e3c2d5-...", host="192.168.1.101", port="22", username="basilica")
# ]
```

#### 5. Miner Deploys SSH Key

**Miner automatically deploys validator's SSH key to all nodes:**

```bash
# Miner executes on each node (using miner's SSH key)
ssh -i ~/.ssh/miner_node_key basilica@192.168.1.100 << 'EOF'
mkdir -p ~/.ssh
echo 'ssh-ed25519 AAAAC3NzaC1lZDI1NTE5... validator-5G3qVaXz...' >> ~/.ssh/authorized_keys
chmod 600 ~/.ssh/authorized_keys
EOF
```

#### 6. Validator Connects Directly

**Validator establishes direct SSH connection:**

```bash
# Validator connects to node (using validator's SSH key)
ssh -i ~/.basilica/ssh/validator_persistent.pem basilica@192.168.1.100

# Validator can now:
# - Run GPU validation workloads
# - Execute container rentals
# - Monitor GPU status
```

### What Happens Behind the Scenes

**On miner:**

```rust
// Node manager handles discovery request
async fn handle_discover_nodes(&self, request: DiscoverNodesRequest) -> Result<Response> {
    // 1. Validate SSH public key format
    self.validate_ssh_key(&request.validator_public_key)?;

    // 2. Deploy key to all nodes
    self.authorize_validator(
        &request.validator_hotkey,
        &request.validator_public_key
    ).await?;

    // 3. Get node list
    let nodes = self.list_nodes().await?;

    // 4. Return connection details
    Ok(ListNodeConnectionDetailsResponse { nodes })
}
```

**SSH key deployment:**

```rust
async fn authorize_validator(&self, validator_hotkey: &str, ssh_key: &str) -> Result<()> {
    for node in self.list_nodes().await? {
        // Tag key with validator identifier
        let key_entry = format!("{} validator-{}", ssh_key, validator_hotkey);

        // Deploy via SSH
        let ssh_command = format!(
            "mkdir -p ~/.ssh && echo '{}' >> ~/.ssh/authorized_keys && chmod 600 ~/.ssh/authorized_keys",
            key_entry
        );

        self.ssh_client.execute_command(&node.connection_details, &ssh_command).await?;
    }
    Ok(())
}
```

### Security Considerations

**Validator authentication ensures:**

- Only validators with valid Bittensor hotkeys can discover nodes
- Replay attacks prevented by nonce + timestamp
- Man-in-the-middle attacks prevented by target hotkey verification
- Mutual authentication (both parties verify each other)

**SSH key management ensures:**

- Only authorized validators' keys deployed to nodes
- Keys tagged with validator identity for audit/revocation
- Miner controls key deployment (nodes remain passive)
- Standard SSH security model applies

---

## Security & Best Practices

### SSH Security

#### Key Management

**DO:**

- Generate separate keys for miner and each service
- Use Ed25519 keys (faster, more secure than RSA)
- Set proper permissions (600 for private keys, 644 for public keys)
- Store keys outside of git repositories
- Rotate keys periodically (every 6-12 months)
- Use strong passphrases for keys (if not automated)

**DON'T:**

- Reuse SSH keys across services
- Store private keys in containers
- Share private keys between systems
- Use weak key types (DSA, RSA <2048 bits)
- Leave default permissions (world-readable)

#### SSH Hardening on GPU Nodes

```bash
# Edit /etc/ssh/sshd_config
sudo vim /etc/ssh/sshd_config
```

Add/modify:

```text
# Disable root login
PermitRootLogin no

# Disable password authentication
PasswordAuthentication no
ChallengeResponseAuthentication no

# Only allow public key authentication
PubkeyAuthentication yes

# Limit authentication attempts
MaxAuthTries 3

# Limit concurrent connections
MaxStartups 30:30:100
MaxSessions 100

# Disable forwarding (if not needed)
AllowTcpForwarding no
X11Forwarding no

# Log authentication attempts
LogLevel VERBOSE
```

Restart SSH:

```bash
sudo systemctl restart sshd
```

### Network Security

#### Firewall Configuration

**On miner server:**

```bash
# Allow only necessary ports
sudo ufw default deny incoming
sudo ufw default allow outgoing
sudo ufw allow 22/tcp         # SSH (restrict to known IPs in production)
sudo ufw allow 8080/tcp       # gRPC (for validators)
sudo ufw allow 9090/tcp       # Metrics (optional, can be localhost-only)
sudo ufw enable
```

**On GPU nodes:**

```bash
# Allow SSH from miner only
sudo ufw default deny incoming
sudo ufw default allow outgoing
sudo ufw allow from <MINER_IP> to any port 22
sudo ufw enable
```

#### TLS/Encryption

**For production, consider enabling TLS:**

```toml
[validator_comms]
tls_enabled = true
tls_cert_path = "/opt/basilica/certs/server.crt"
tls_key_path = "/opt/basilica/certs/server.key"
```

Generate self-signed certificate:

```bash
openssl req -x509 -newkey rsa:4096 -nodes \
  -keyout /opt/basilica/certs/server.key \
  -out /opt/basilica/certs/server.crt \
  -days 365 -subj "/CN=miner.basilica.local"
```

### Access Control

#### Validator Assignment Strategy

**Production recommendation:**

```toml
[validator_assignment]
enabled = true
strategy = "highest_stake"
min_stake_threshold = 12000.0    # High threshold for security
validator_hotkey = "5G3qVaXzKMPDm5AJ3dpzbpUC27kpccBvDwzSWXrq8M6qMmbC"
```

**Benefits:**

- Limits exposure to most invested validators
- Reduces attack surface
- Simplifies operations and monitoring
- Easier to establish trust relationships

#### SSH Key Auditing

**Track deployed validator keys:**

```bash
# On GPU node, view authorized_keys
cat ~/.ssh/authorized_keys | grep validator-

# Sample output:
# ssh-ed25519 AAAAC3... validator-5G3qVaXzKMP...
# ssh-ed25519 AAAAC3... validator-5FHd7oPqk...
```

**Audit SSH connections:**

```bash
# View SSH access logs
sudo grep 'Accepted publickey' /var/log/auth.log | tail -20

# View by user
sudo grep 'Accepted publickey for basilica' /var/log/auth.log
```

### Operational Security

#### Monitoring and Alerts

**Set up alerts for:**

- Failed authentication attempts
- Unauthorized SSH access
- Unexpected configuration changes
- Database anomalies
- Node connectivity issues

**Example: Alert on authentication failures**

```bash
# Create alert script
sudo tee /opt/basilica/scripts/auth-alert.sh > /dev/null << 'EOF'
#!/bin/bash
FAILED_LOGINS=$(grep "Failed publickey" /var/log/auth.log | wc -l)
if [ $FAILED_LOGINS -gt 10 ]; then
    echo "WARNING: $FAILED_LOGINS failed SSH attempts detected" | \
        mail -s "Security Alert: SSH Failures" admin@yourdomain.com
fi
EOF

chmod +x /opt/basilica/scripts/auth-alert.sh

# Run hourly via cron
echo "0 * * * * /opt/basilica/scripts/auth-alert.sh" | crontab -
```

#### Backup and Recovery

**Critical data to backup:**

- Bittensor wallet (`~/.bittensor/wallets/`)
- Miner configuration (`/opt/basilica/config/`)
- SSH keys (`~/.ssh/miner_node_key*`)
- Database (`/opt/basilica/data/miner.db`)

**Backup script:**

```bash
#!/bin/bash
BACKUP_DIR="/backup/basilica/$(date +%Y%m%d)"
mkdir -p $BACKUP_DIR

# Backup wallet
cp -r ~/.bittensor/wallets $BACKUP_DIR/

# Backup config
cp /opt/basilica/config/miner.toml $BACKUP_DIR/

# Backup SSH keys
cp ~/.ssh/miner_node_key* $BACKUP_DIR/

# Backup database
sqlite3 /opt/basilica/data/miner.db ".backup $BACKUP_DIR/miner.db"

# Encrypt backup
tar -czf - $BACKUP_DIR | gpg --encrypt --recipient admin@yourdomain.com > $BACKUP_DIR.tar.gz.gpg

echo "Backup completed: $BACKUP_DIR.tar.gz.gpg"
```

#### Update Management

**Keep system updated:**

```bash
# System updates
sudo apt update && sudo apt upgrade -y

# Miner updates
cd /opt/basilica
./scripts/miner/build.sh --release
sudo systemctl restart basilica-miner
```

**For Docker deployments**, Watchtower handles automatic updates.

### Rate Limiting

**Protect against DoS:**

```bash
# Limit SSH connection rate with iptables
sudo iptables -A INPUT -p tcp --dport 22 -m state --state NEW -m recent --set
sudo iptables -A INPUT -p tcp --dport 22 -m state --state NEW -m recent --update --seconds 60 --hitcount 10 -j DROP
```

**Application-level rate limiting** (future enhancement):

```toml
[validator_comms.rate_limit]
enabled = true
requests_per_second = 10
burst_capacity = 20
```

---

## Monitoring

### Health Checks

**Check miner health:**

```bash
# HTTP health endpoint
curl http://localhost:8080/health

# Expected response:
# {"status":"healthy","timestamp":1234567890}
```

**Check database connectivity:**

```bash
# Run miner health check
./basilica-miner --config miner.toml health-check

# Or via database CLI command
./basilica-miner --config miner.toml database health
```

### Metrics Collection

**Prometheus metrics endpoint:**

```bash
# Access metrics
curl http://localhost:9090/metrics

# Sample metrics:
# basilica_miner_node_count 3
# basilica_miner_validator_connections_total 12
# basilica_miner_ssh_deployments_total 45
# basilica_miner_authentication_requests_total 120
```

**Grafana dashboard** (if available):

```text
https://basilica-grafana.tplr.ai/
```

### Log Management

**View miner logs:**

```bash
# Systemd service
sudo journalctl -u basilica-miner -f

# Docker container
docker logs -f basilica-miner

# Binary (if logging to file)
tail -f /opt/basilica/miner.log
```

**Important log patterns to monitor:**

```bash
# Authentication events
grep "Successfully authenticated validator" /opt/basilica/miner.log

# Node registration events
grep "Registered node" /opt/basilica/miner.log

# SSH key deployment events
grep "Deploying SSH key for validator" /opt/basilica/miner.log

# Errors
grep "ERROR" /opt/basilica/miner.log
```

### Node Monitoring

**Check node connectivity:**

```bash
# Test SSH access to all nodes
for node in 192.168.1.100 192.168.1.101; do
    echo "Testing $node..."
    ssh -i ~/.ssh/miner_node_key basilica@$node "nvidia-smi --query-gpu=name,utilization.gpu --format=csv,noheader"
done
```

**Monitor GPU utilization:**

```bash
# Create monitoring script
cat > /opt/basilica/scripts/monitor-gpus.sh << 'EOF'
#!/bin/bash
while true; do
    echo "=== GPU Status $(date) ==="
    ssh -i ~/.ssh/miner_node_key basilica@192.168.1.100 nvidia-smi --query-gpu=index,name,utilization.gpu,utilization.memory,temperature.gpu --format=csv
    sleep 60
done
EOF

chmod +x /opt/basilica/scripts/monitor-gpus.sh
./opt/basilica/scripts/monitor-gpus.sh
```

### Performance Metrics

**Key metrics to track:**

- **Node availability**: Percentage of time nodes are accessible
- **Validator requests**: Number of authentication/discovery requests
- **SSH deployments**: Number of successful key deployments
- **Response times**: gRPC endpoint latency
- **Error rates**: Failed authentications, SSH failures
- **Database performance**: Query times, connection pool usage

---

## Troubleshooting

### Common Issues

#### 1. Miner Won't Start

##### **Error: Database connection failed**

```text
Error: unable to open database file
```

**Solution:**

```bash
# Ensure database directory exists
sudo mkdir -p /opt/basilica/data
sudo chown $USER:$USER /opt/basilica/data

# Check database URL in config
# Should be: url = "sqlite:///opt/basilica/data/miner.db"
```

##### **Error: Wallet loading failed**

```text
Error: Failed to load hotkey: Invalid format
```

**Solution:**

```bash
# Verify wallet exists
ls ~/.bittensor/wallets/miner_wallet/hotkeys/default

# Check wallet_name and hotkey_name in config match filesystem
# wallet_name = "miner_wallet"
# hotkey_name = "default"

# Verify wallet format (should be JSON with secretPhrase)
cat ~/.bittensor/wallets/miner_wallet/hotkeys/default
```

##### **Error: Port already in use**

```text
Error: Address already in use (os error 98)
```

**Solution:**

```bash
# Check what's using port 8080
sudo lsof -i :8080

# Kill process or change port in config
# [validator_comms]
# port = 8081
```

#### 2. SSH Connection Issues

##### **Error: Permission denied (publickey)**

```text
Error: Failed to connect to node 192.168.1.100: Permission denied (publickey)
```

**Solution:**

```bash
# Verify miner's public key is on node
ssh basilica@192.168.1.100 'cat ~/.ssh/authorized_keys | grep miner_node_key'

# If not present, deploy it
ssh-copy-id -i ~/.ssh/miner_node_key.pub basilica@192.168.1.100

# Test connection
ssh -i ~/.ssh/miner_node_key basilica@192.168.1.100 'echo "Connection successful"'
```

##### **Error: Connection timed out**

```text
Error: Connection to 192.168.1.100:22 timed out
```

**Solution:**

```bash
# Check network connectivity
ping 192.168.1.100

# Check SSH port is open
nc -zv 192.168.1.100 22

# Check firewall on node
ssh basilica@192.168.1.100 'sudo ufw status'

# Allow SSH from miner IP
ssh basilica@192.168.1.100 'sudo ufw allow from <MINER_IP> to any port 22'
```

##### **Error: Key permissions too open**

```text
Error: Permissions 0644 for '~/.ssh/miner_node_key' are too open
```

**Solution:**

```bash
# Fix key permissions
chmod 600 ~/.ssh/miner_node_key
chmod 644 ~/.ssh/miner_node_key.pub
```

#### 3. Validator Discovery Issues

##### **Error: No validators discovered**

```text
WARN: No validators found matching criteria
```

**Solution:**

```bash
# Check validator assignment config
# [validator_assignment]
# enabled = true
# min_stake_threshold = 12000.0  # May be too high

# Lower threshold or disable assignment for testing
# min_stake_threshold = 1000.0
# OR
# enabled = false

# Restart miner
sudo systemctl restart basilica-miner
```

##### **Error: Validator authentication failed**

```text
ERROR: Validator authentication failed: Invalid signature
```

**Solution:**

```bash
# Verify security config
# [security]
# verify_signatures = true

# For local testing only:
# verify_signatures = false  # NEVER use in production

# Check timestamp freshness (should be within 5 minutes)
# Ensure system clocks are synchronized
sudo timedatectl set-ntp true
```

#### 4. Node Registration Issues

##### **Error: No nodes registered**

```text
WARN: No nodes registered - miner will not be able to serve validators
```

**Solution:**

```bash
# Check node_management config
# [node_management]
# nodes = [
#   { host = "192.168.1.100", port = 22, username = "basilica" }
# ]

# Verify SSH access to each node
for node in $(grep 'host = ' miner.toml | cut -d'"' -f2); do
    echo "Testing $node..."
    ssh -i ~/.ssh/miner_node_key basilica@$node 'hostname'
done
```

##### **Error: Node ID generation failed**

```text
ERROR: Failed to generate node ID for basilica@192.168.1.100:22
```

**Solution:**

```bash
# This usually indicates SSH credentials are invalid
# Verify each field in node config:
# - host: Must be accessible IP/hostname
# - port: Must be correct SSH port
# - username: Must exist on node

# Test connection
ssh -p 22 -i ~/.ssh/miner_node_key basilica@192.168.1.100
```

#### 5. Bittensor Network Issues

##### **Error: Failed to serve axon on network**

```text
Error: Failed to register on network: Insufficient funds
```

**Solution:**

```bash
# Check wallet balance
btcli wallet balance --wallet.name miner_wallet

# Ensure sufficient TAO for registration fee
# Transfer TAO to wallet if needed

# Check if already registered
btcli subnet list --netuid 39 | grep <YOUR_HOTKEY>
```

##### **Error: Metadata incompatibility**

```text
Error: Metadata error: the generated code is not compatible with the node
```

**Solution:**

```bash
# Regenerate metadata
./scripts/generate-metadata.sh --network finney

# Rebuild miner
./scripts/miner/build.sh --release

# Restart miner
sudo systemctl restart basilica-miner
```

### Debugging Strategies

#### Enable Debug Logging

```bash
# Run miner with verbose logging
./basilica-miner --config miner.toml -vvv

# Or set in config (not recommended for production)
# [logging]
# level = "debug"
```

#### Validate Configuration

```bash
# Validate config before starting
./basilica-miner --config miner.toml config validate

# Show effective configuration
./basilica-miner --config miner.toml config show
```

#### Test Individual Components

```bash
# Test SSH connectivity
./basilica-miner --config miner.toml service test-ssh

# Test database connection
./basilica-miner --config miner.toml database health

# Test Bittensor connection
./basilica-miner --config miner.toml service test-bittensor
```

#### Network Diagnostics

```bash
# Check miner's gRPC endpoint
curl -v http://localhost:8080

# Test from external (validator perspective)
curl -v http://<MINER_PUBLIC_IP>:8080

# Check metrics endpoint
curl http://localhost:9090/metrics | grep basilica_miner
```

### Getting Help

If you're still experiencing issues:

1. **Check logs** for detailed error messages
2. **Search GitHub issues** for similar problems
3. **Join Discord** for community support
4. **Create GitHub issue** with:
   - Miner version
   - Configuration (redacted sensitive info)
   - Full error logs
   - Steps to reproduce

---

## Advanced Topics

### Custom Validator Assignment Logic

You can implement custom assignment strategies by creating a new strategy type:

```rust
// Example: Geographic assignment strategy
pub struct GeographicAssignment {
    preferred_region: String,
}

#[async_trait]
impl AssignmentStrategy for GeographicAssignment {
    async fn select_validators(
        &self,
        validators: Vec<ValidatorInfo>,
        nodes: Vec<RegisteredNode>,
    ) -> Result<Vec<(ValidatorInfo, Vec<RegisteredNode>)>> {
        // Filter validators by region
        let regional_validators: Vec<_> = validators
            .into_iter()
            .filter(|v| self.is_in_region(v))
            .collect();

        // Assign nodes to regional validators
        // ... implementation
    }
}
```

### Multi-Region Deployment

For geo-distributed GPU nodes:

```toml
[node_management]
nodes = [
    # US East
    { host = "us-east-1.example.com", port = 22, username = "basilica" },
    { host = "us-east-2.example.com", port = 22, username = "basilica" },

    # EU West
    { host = "eu-west-1.example.com", port = 22, username = "basilica" },
    { host = "eu-west-2.example.com", port = 22, username = "basilica" },

    # Asia Pacific
    { host = "ap-south-1.example.com", port = 22, username = "basilica" },
]
```

**Considerations:**

- Network latency between miner and nodes
- Regional compliance requirements
- Validator geographic distribution
- Cost optimization

### Automated SSH Key Rotation

For enhanced security:

```bash
#!/bin/bash
# Rotate miner SSH key monthly

OLD_KEY=~/.ssh/miner_node_key
NEW_KEY=~/.ssh/miner_node_key.new

# Generate new key
ssh-keygen -t ed25519 -f $NEW_KEY -N "" -C "basilica-miner-$(date +%Y%m)"

# Deploy new key to all nodes
for node in $(grep 'host = ' /opt/basilica/config/miner.toml | cut -d'"' -f2); do
    ssh-copy-id -i $NEW_KEY.pub basilica@$node
done

# Update miner config
sed -i "s|miner_node_key|miner_node_key.new|g" /opt/basilica/config/miner.toml

# Restart miner
sudo systemctl restart basilica-miner

# Remove old key after verification
sleep 60
rm -f $OLD_KEY $OLD_KEY.pub
mv $NEW_KEY $OLD_KEY
mv $NEW_KEY.pub $OLD_KEY.pub
```

### Performance Tuning

**Optimize SSH connections:**

```text
# ~/.ssh/config
Host gpu-*
    ControlMaster auto
    ControlPath ~/.ssh/control-%r@%h:%p
    ControlPersist 10m
    Compression yes
    ServerAliveInterval 60
```

**Optimize database:**

```toml
[database]
max_connections = 50        # Increase for high validator traffic
connection_timeout = 10     # Faster timeout for quicker failover
```

**System tuning:**

```bash
# Increase file descriptor limits
sudo tee -a /etc/security/limits.conf << EOF
root soft nofile 65536
root hard nofile 65536
EOF

# Apply without reboot
ulimit -n 65536
```

---

### Next Steps

1. **Deploy your first miner** following this guide
2. **Test validator connectivity** and monitor initial interactions
3. **Set up monitoring** with Prometheus/Grafana
4. **Join the community** for support and updates
5. **Review the [Validator Guide](validator.md)** to understand how your miner is evaluated

### Additional Resources

- **GitHub Repository**: <https://github.com/one-covenant/basilica>
- **Discord**: <https://discord.gg/jYcGzwed>
- **Website**: <https://www.basilica.ai/>
- **Validator Guide**: [docs/validator.md](validator.md)
- **Architecture Overview**: [docs/architecture.md](architecture.md)
- **API Documentation**: [docs/api.md](api.md)

---

**Happy Mining! ⛏️🫡**
