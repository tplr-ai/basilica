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

# 3. Copy and edit config from template
cp config/miner.toml.example miner.toml
# Edit miner.toml with your settings:
# - [bittensor] wallet_name, hotkey_name, external_ip
# - [node_management] nodes list with your GPU nodes
# - [bidding.strategy.static.static_prices] prices per GPU category
# - [ssh_session] miner_node_key_path

# Minimal example configuration:
cat > miner.toml <<EOF
[bittensor]
wallet_name = "your_wallet"
hotkey_name = "your_hotkey"
external_ip = "your_public_ip"
axon_port = 50051
network = "finney"
netuid = 39
chain_endpoint = "wss://entrypoint-finney.opentensor.ai:443"

[database]
url = "sqlite:///opt/basilica/data/miner.db"

[node_management]
nodes = [
  { host = "<node_ip>", port = 22, username = "basilica", gpu_category = "H100", gpu_count = 8 },
]

[ssh_session]
miner_node_key_path = "~/.ssh/miner_node_key"
default_node_username = "basilica"

[validator_assignment]
strategy = "highest_stake"

[bidding.strategy.static.static_prices]
H100 = 2.50
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
    - [Node Identity](#node-identity)
    - [Authentication Flow](#authentication-flow)
    - [SSH Key Deployment Mechanism](#ssh-key-deployment-mechanism)
    - [Validator Assignment Strategies](#validator-assignment-strategies)
    - [Discovery Process](#discovery-process)
    - [Bidding & Registration Protocol](#bidding--registration-protocol)
5. [SSH Key Setup](#ssh-key-setup)
6. [GPU Node Preparation](#gpu-node-preparation)
7. [Miner Configuration](#miner-configuration)
8. [Deployment Methods](#deployment-methods)
9. [Validator Access Flow](#validator-access-flow)
10. [Security & Best Practices](#security--best-practices)
11. [Monitoring](#monitoring)
12. [Troubleshooting](#troubleshooting)
13. [Advanced Topics](#advanced-topics)
    - [Delivery-Based Emissions](#delivery-based-emissions)
    - [Miners & the Ban System](#miners--the-ban-system)

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

### Miner-Initiated Bidding Model

```text
┌──────────────────┐                          ┌──────────────────┐
│      Miner       │                          │    Validator     │
│                  │                          │   (gRPC server)  │
└────────┬─────────┘                          └────────▲─────────┘
         │                                             │
         │ 1. Discover validator via metagraph         │
         │    + /discovery HTTP endpoint               │
         │                                             │
         │ 2. RegisterBid gRPC ──────────────────────→ │
         │    (nodes + SSH details + prices)           │
         │                                             │
         │ 3. Response ←─────────────────────────────  │
         │    (validator SSH public key,               │
         │     health check interval,                  │
         │     collateral status)                      │
         │                                             │
         │ 4. Deploy validator SSH key                 │
         │    (miner → nodes)                          │
         │                                             │
         │ 5. Periodic HealthCheck gRPC ─────────────→ │
         │                                             │
         |─────────────|─────────────|                 │
┌──────────────────┐  ┌──────────────────┐  ┌──────────────────┐
│   GPU Node 1     │  │   GPU Node 2     │  │   GPU Node N     │
│  (SSH endpoint)  │  │  (SSH endpoint)  │  │  (SSH endpoint)  │
└────────▲─────────┘  └────────▲─────────┘  └────────▲─────────┘
         │                     │                     │
         └─────────────────────┴─────────────────────┘
                  6. Validator connects directly via SSH
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
  { host = "192.168.1.100", port = 22, username = "basilica", gpu_category = "H100", gpu_count = 8 }
]
```

Generates: `node_id = "a3f2b1c4-5d6e-7f8a-9b0c-1d2e3f4a5b6c"` (deterministic)

### Authentication Flow

All gRPC requests from the miner are signed with the miner's **Bittensor hotkey**. The validator verifies each signature using the miner's public key from the metagraph.

#### Signature Format

Each RPC has a specific message format that is signed:

| RPC | Signature payload |
|---|---|
| **RegisterBid** | `{hotkey}\|{timestamp}` |
| **UpdateBid** | `{hotkey}\|{node_id}\|{hourly_rate_cents}\|{timestamp}` |
| **RemoveBid** | `{hotkey}\|{node_ids_csv}\|{timestamp}` |
| **HealthCheck** | `{hotkey}\|{node_ids_csv}\|{timestamp}` |

**Example** (RegisterBid):

```text
5GrwvaEF5zXb26Fz9rcQpDWS57CtERHpNehXCPcNoHGKutQY|1718000000
```

#### Verification

The validator verifies:
- **Signature validity**: The signature matches the miner's Bittensor hotkey
- **Timestamp freshness**: Timestamp is within the allowed window (default 300 seconds) to prevent replay attacks
- **Miner identity**: The hotkey corresponds to a known miner on the subnet

### SSH Key Deployment Mechanism

After a successful `RegisterBid`, the miner receives the validator's SSH public key and deploys it to all managed nodes.

1. **Miner receives validator's SSH public key** from `RegisterBidResponse.validator_ssh_public_key`

2. **Miner validates SSH key format** using OpenSSH key parsing (supports `ssh-rsa`, `ssh-ed25519`, `ecdsa-sha2-*`)

3. **Miner deploys key with exclusive access** — old validator keys are removed before adding the new one. For each node, the miner SSHes in (using its own key) and atomically rewrites `~/.ssh/authorized_keys`:
   - Filters out any existing lines containing `validator-`
   - Appends the new validator key
   - Preserves the miner's own key and any non-validator keys

4. **Key tagging**: Keys are normalized with a `validator-{hotkey}` identifier:

   ```text
   ssh-ed25519 AAAAC3NzaC1lZDI1NTE5... validator-5G3qVaXzKMPDm5AJ3dpzbpUC27kpccBvDwzSWXrq8M6qMmbC
   ```

5. **Validator connects directly** to nodes using its private key:

   ```bash
   ssh -i ~/.basilica/ssh/validator_persistent.pem basilica@192.168.1.100
   ```

### Validator Assignment Strategies

Miners can control which validators receive access to their nodes:

#### 1. **Highest Stake** (Recommended)

Automatically assigns ALL nodes to the validator with highest stake. No configuration required beyond setting the strategy.

```toml
[validator_assignment]
strategy = "highest_stake"
```

**Use cases:**

- Production deployment (recommended default)
- Maximize security by working with most invested validator
- Simplify operations (automatic validator selection)

**Behavior:**

- Fetches validators from Bittensor metagraph
- Selects highest-staked validator with validator_permit
- `validator_hotkey` config is ignored with this strategy (use `fixed_assignment` if you need a specific validator)

#### 2. **Fixed Assignment**

Assign nodes to a specific validator by hotkey. Mainly useful for debugging or testing.

```toml
[validator_assignment]
strategy = "fixed_assignment"
validator_hotkey = "5G3qVaXzKMPDm5AJ3dpzbpUC27kpccBvDwzSWXrq8M6qMmbC"
```

**Use cases:**

- Debugging with a specific known validator
- Testing environments where you need deterministic validator assignment

### Discovery Process

**Miner's perspective** (miner-initiated):

1. **Query Bittensor metagraph** for validators with permits on the subnet
2. **Select validator** using the configured assignment strategy (e.g., `highest_stake`)
3. **Call validator's `/discovery` HTTP endpoint** (via the axon address from metagraph) to learn the gRPC port for bid registration
4. **Call `RegisterBid` gRPC** with all nodes, SSH details, and pricing
5. **Receive validator's SSH public key** + health check interval + collateral status
6. **Deploy validator SSH key** to all managed nodes
7. **Enter health check loop** — send periodic `HealthCheck` RPCs to keep registrations active

**Validator's perspective** (passive):

1. **Runs gRPC registration server** listening for `RegisterBid` from miners
2. **Receives bids** — verifies signature, checks bid floor, checks collateral, upserts nodes into DB
3. **Returns SSH public key** so the miner can deploy it to nodes
4. **Tracks miner heartbeat** — expects periodic `HealthCheck` RPCs from the miner process (default every 60s)
5. **Tracks node liveness separately** — uses validator SSH verification (`last_node_check`) for node online/offline decisions and stale cleanup
6. **Connects directly to nodes via SSH** for validation and rental operations

### Bidding & Registration Protocol

The miner communicates with the validator via the `MinerRegistration` gRPC service. There are four RPCs:

#### 1. RegisterBid

Called **once at startup** after validator discovery. Sends all nodes with SSH connection details and pricing. The validator returns its SSH public key, the health check interval, and collateral status.

```protobuf
rpc RegisterBid(RegisterBidRequest) returns (RegisterBidResponse);

message RegisterBidRequest {
  string miner_hotkey = 1;              // Miner's Bittensor hotkey
  repeated NodeRegistration nodes = 2;  // Nodes with SSH details + prices
  int64 timestamp = 3;                  // Unix timestamp
  bytes signature = 4;                  // Signature over "{hotkey}|{timestamp}"
}

message NodeRegistration {
  string node_id = 1;               // Deterministic UUID from username@host:port
  string host = 2;                  // SSH host
  uint32 port = 3;                  // SSH port
  string username = 4;              // SSH username
  string gpu_category = 5;          // "H100", "A100", "B200", etc.
  uint32 gpu_count = 6;             // Number of GPUs
  uint32 hourly_rate_cents = 7;     // Price in cents per GPU per hour
}

message RegisterBidResponse {
  bool accepted = 1;
  string registration_id = 2;
  string validator_ssh_public_key = 3;     // Deploy this to your nodes
  uint32 health_check_interval_secs = 4;   // How often to send health checks
  string error_message = 5;
  CollateralStatus collateral_status = 6;  // Worst status across all nodes
}
```

**Validator processing**: Verifies signature → checks timestamp freshness → validates node fields → enforces bid floor → checks collateral → upserts nodes in DB → deactivates any previously-registered nodes not in this request.

**Availability note**: A successful `RegisterBid` does not make a node immediately visible in `GET /nodes` (and therefore `basilica ls`). The node is shown only after at least one successful full validation has populated `gpu_uuid_assignments` for that node. If full validation later fails and assignments are cleaned up, the node is hidden again until a subsequent successful full validation.

#### 2. UpdateBid

Update the hourly rate for a specific node. Can be called at any time after registration.

```protobuf
rpc UpdateBid(UpdateBidRequest) returns (UpdateBidResponse);

message UpdateBidRequest {
  string miner_hotkey = 1;
  string node_id = 2;               // Node to update
  uint32 hourly_rate_cents = 3;     // New price in cents per GPU per hour
  int64 timestamp = 4;
  bytes signature = 5;              // Signature over "{hotkey}|{node_id}|{hourly_rate_cents}|{timestamp}"
}
```

#### 3. RemoveBid

Explicitly remove nodes from availability. If `node_ids` is empty, all nodes are removed.

```protobuf
rpc RemoveBid(RemoveBidRequest) returns (RemoveBidResponse);

message RemoveBidRequest {
  string miner_hotkey = 1;
  repeated string node_ids = 2;     // Empty = remove all
  int64 timestamp = 3;
  bytes signature = 4;              // Signature over "{hotkey}|{node_ids_csv}|{timestamp}"
}

message RemoveBidResponse {
  bool accepted = 1;
  uint32 nodes_removed = 2;
  string error_message = 3;
}
```

#### 4. HealthCheck

Periodic miner-process heartbeat. The interval is returned by `RegisterBidResponse.health_check_interval_secs` (default 60s). Node bid eligibility/staleness is based on validator SSH verification (`last_node_check`), not heartbeat freshness.

```protobuf
rpc HealthCheck(HealthCheckRequest) returns (HealthCheckResponse);

message HealthCheckRequest {
  string miner_hotkey = 1;
  repeated string node_ids = 2;     // Empty = all nodes
  int64 timestamp = 3;
  bytes signature = 4;              // Signature over "{hotkey}|{node_ids_csv}|{timestamp}"
}

message HealthCheckResponse {
  bool accepted = 1;
  uint32 nodes_active = 2;
  string error_message = 3;
}
```

#### Collateral Status

The `RegisterBidResponse` includes a `CollateralStatus` message reflecting the **worst** collateral status across all registered nodes:

```protobuf
message CollateralStatus {
  double current_alpha = 1;           // Raw Alpha amount deposited
  double current_usd_value = 2;       // USD value at current price
  double minimum_usd_required = 3;    // Minimum for this GPU category
  string status = 4;                  // "sufficient", "warning", "undercollateralized", "excluded"
  string grace_period_remaining = 5;  // e.g. "23h 45m" or empty
  string action_required = 6;         // e.g. "Deposit 25 Alpha (~$50) to maintain eligibility"
  double alpha_usd_price = 7;         // Current Alpha/USD price used
  bool price_stale = 8;              // True if price data is > 1hr old
}
```

| Status | Meaning |
|---|---|
| `sufficient` | Collateral meets requirements |
| `warning` | Collateral is low — action recommended |
| `undercollateralized` | Below minimum — may be excluded soon |
| `excluded` | Insufficient collateral — excluded from rentals |

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

### Using the Configuration Template

Use the configuration template provided in the config directory:

```bash
# Copy the example config
cp config/miner.toml.example miner.toml

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
axon_port = 50051

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

#### 3. Validator Discovery (Automatic)

The miner automatically discovers the validator to register with:

1. At startup, the miner queries the Bittensor metagraph for validators
2. Based on the `[validator_assignment]` strategy (e.g., `highest_stake`), it selects a validator
3. It calls the validator's `/discovery` HTTP endpoint (via the axon address) to learn the gRPC port for bid registration
4. The BidManager then registers nodes and runs health checks against the discovered gRPC endpoint

No manual `validator_registration_endpoint` configuration is needed.

**⚠️ Firewall**: Ensure the axon port (default 50051) is accessible for Bittensor network registration:

```bash
# UFW
sudo ufw allow 50051/tcp

# Or iptables
sudo iptables -A INPUT -p tcp --dport 50051 -j ACCEPT
```

#### 4. GPU Node Management

```toml
[node_management]
# List your GPU compute nodes with SSH access details
# Pricing is configured separately in [bidding.strategy.static.static_prices]
nodes = [
  { host = "192.168.1.100", port = 22, username = "basilica", gpu_category = "H100", gpu_count = 8 },
  { host = "192.168.1.101", port = 22, username = "basilica", gpu_category = "A100", gpu_count = 4 },
]
health_check_interval = 60   # Health check interval in seconds
health_check_timeout = 10    # Health check timeout in seconds
max_retry_attempts = 3
auto_recovery = true
```

**Node configuration fields:**

- `host`: IP address or hostname of GPU node (required)
- `port`: SSH port, typically 22 (required)
- `username`: SSH username on the node (required)
- `gpu_category`: GPU model category, e.g., "H100", "A100", "B200" (required)
- `gpu_count`: Number of GPUs on this node (required)
- `additional_opts` (optional): Extra SSH options like `"-o StrictHostKeyChecking=no"`

#### 4b. Bidding Configuration (GPU Pricing)

Pricing is configured separately from nodes in the `[bidding]` section. Every GPU category
listed in your nodes **must** have a corresponding price entry. Prices are defined in **dollars per GPU-hour** and are converted to **cents** internally before being sent to the validator.

```toml
[bidding]
# Static prices by GPU category (in dollars per GPU-hour)
[bidding.strategy.static.static_prices]
H100 = 2.50    # $2.50/hour per H100 GPU → sent as 250 cents
A100 = 1.20    # $1.20/hour per A100 GPU → sent as 120 cents
```

The BidManager runs automatically after validator discovery and registers your nodes with these prices.

**Startup validation**: The miner will refuse to start if any GPU category in your `[node_management]` nodes is missing a price entry. For example, if you have a node with `gpu_category = "B200"` but no `B200 = ...` in `static_prices`, the miner will exit with an error.

**Bid floor enforcement**: Validators enforce a minimum bid price (default 10% of the baseline market price for each GPU category). If your bid is below this floor, the `RegisterBid` RPC will be rejected with an error explaining the minimum acceptable price.

**Collateral requirements**: The validator checks your collateral (Alpha token deposit) during registration. The response includes a `CollateralStatus` with one of these states:

| Status | Meaning |
|---|---|
| `sufficient` | Collateral meets requirements — no action needed |
| `warning` | Collateral is low — deposit more Alpha to maintain eligibility |
| `undercollateralized` | Below minimum — may be excluded from rentals soon |
| `excluded` | Insufficient collateral — nodes are excluded from rental selection |

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

# Optional: Assign to specific validator (required for fixed_assignment)
# validator_hotkey = "5G3qVaXzKMPDm5AJ3dpzbpUC27kpccBvDwzSWXrq8M6qMmbC"
```

**Choosing a strategy:**

- **Production**: Use `highest_stake` to assign all nodes to the top validator
- **Fixed**: Use `fixed_assignment` with a specific `validator_hotkey` for known validators
- **Development**: Use default `highest_stake` without specific hotkey

#### 9. Advertised Addresses (Optional)

Override auto-detected addresses for NAT/proxy scenarios:

```toml
[advertised_addresses]
# Only needed if miner is behind NAT/proxy
# grpc_endpoint = "http://203.0.113.45:50051"
# axon_endpoint = "http://203.0.113.45:50051"
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
  -p 50051:50051 \
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
      - "50051:50051"
      - "9090:9090"
    command: ["--config", "/opt/basilica/config/miner.toml"]
    healthcheck:
      test: ["CMD", "curl", "-f", "http://localhost:9090/metrics"]
      interval: 30s
      timeout: 10s
      retries: 3
      start_period: 40s

  watchtower:
    image: nickfedor/watchtower:1.14.0
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

#### 1. Miner Startup & Validator Discovery

**Miner queries metagraph and selects a validator:**

```text
Miner starts up
  → Queries Bittensor metagraph for validators with permits
  → Applies assignment strategy (e.g., highest_stake)
  → Calls validator's /discovery HTTP endpoint
  → Learns the validator's gRPC port for bid registration
```

The `/discovery` endpoint returns:

```json
{
  "bid_grpc_port": 50052,
  "version": "0.1.0"
}
```

The miner constructs the full gRPC endpoint from the validator's axon IP and the returned port (e.g., `http://203.0.113.45:50052`).

#### 2. RegisterBid gRPC Call

**Miner registers all nodes with the validator:**

```text
Miner builds RegisterBidRequest:
  - miner_hotkey: SS58-encoded Bittensor hotkey
  - nodes: [{node_id, host, port, username, gpu_category, gpu_count, hourly_rate_cents}, ...]
  - timestamp: current Unix timestamp
  - signature: sign("{hotkey}|{timestamp}")

Sends RegisterBid RPC to validator's gRPC endpoint
```

#### 3. Validator Processes Registration

**Validator validates and stores the registration:**

1. Verifies the miner's signature against its Bittensor hotkey
2. Checks timestamp freshness (within 300s tolerance)
3. Validates all node fields (host, port, username, gpu_category, gpu_count, hourly_rate_cents)
4. Enforces **bid floor** — rejects bids below minimum fraction (default 10%) of the baseline price
5. Checks **collateral status** — reports whether the miner has sufficient Alpha deposited
6. Upserts nodes into database; deactivates any previously-registered nodes not in this request
7. Returns `RegisterBidResponse` with validator's SSH public key, health check interval, and collateral status

#### 4. Miner Deploys SSH Key to Nodes

**Miner deploys the validator's SSH key using exclusive access:**

```text
For each managed node:
  1. SSH into node using miner's private key
  2. Read authorized_keys, filter out existing "validator-" entries
  3. Append new key: "ssh-ed25519 AAAAC3... validator-{hotkey}"
  4. Atomically write updated authorized_keys
```

This ensures only the **current** validator has SSH access — previous validator keys are removed.

#### 5. Miner Enters Health Check Loop

**Miner sends periodic HealthCheck RPCs:**

```text
Every {health_check_interval_secs} seconds (default 60):
  → Build HealthCheckRequest with miner_hotkey, timestamp, signature
  → Send HealthCheck RPC to validator
  → Validator updates miner heartbeat timestamps for the miner's nodes
```

If health checks stop, the miner heartbeat becomes stale, but node online/offline and stale cleanup still depend on validator verification timestamps (`last_node_check`).

#### 6. Validator Connects Directly via SSH

**Validator establishes direct SSH connection to nodes:**

```bash
# Validator connects to node (using validator's SSH key)
ssh -i ~/.basilica/ssh/validator_persistent.pem basilica@192.168.1.100

# Validator can now:
# - Run GPU validation workloads
# - Execute container rentals
# - Monitor GPU status
```

### Security Considerations

**Signature-based authentication ensures:**

- Only miners with valid Bittensor hotkeys can register nodes
- Replay attacks prevented by timestamp freshness checks (default 300s window)
- Each RPC type has a distinct signature payload to prevent cross-RPC replay

**SSH key management ensures:**

- Exclusive access: only the current validator's key is deployed (old keys removed)
- Keys tagged with `validator-{hotkey}` identifier for auditing
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
sudo ufw allow 50051/tcp      # gRPC/Axon (for validators and Bittensor network)
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

**For production, consider enabling TLS** on your infrastructure level (e.g., reverse proxy or load balancer in front of the miner's axon port).

### Access Control

#### Validator Assignment Strategy

**Production recommendation:**

```toml
[validator_assignment]
strategy = "highest_stake"
```

**Benefits:**

- Automatically selects the most invested validator
- Reduces attack surface
- Simplifies operations and monitoring
- No manual validator tracking required

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

**Application-level rate limiting** may be added in a future release.

---

## Monitoring

### Health Checks

**Check miner health:**

```bash
# Prometheus metrics endpoint (health check)
curl http://localhost:9090/metrics | grep basilica_miner

# Check gRPC server is responding
grpcurl -plaintext localhost:50051 list
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

### Delivery-Based Emissions

Miners earn emissions based on **rental revenue**, not uptime or validation scores. When your GPU nodes are actively rented and generating revenue, the billing system records delivery records that the validator uses to set weights. Your share of emissions within each GPU category is proportional to the `revenue_usd` your nodes generate relative to other miners in the same category. For the full weight calculation details, see [Scoring and Weight Setting](scoring-and-weights.md).

### Miners & the Ban System

Validators actively track executor misbehaviour to protect rentals. Each miner/executor pair has an independent ban state backed by persistent storage.

#### Failure thresholds & ban durations (per executor instance)

| Repeated issue | Window | Threshold | Ban duration (baseline) |
| --- | --- | --- | --- |
| Light or full validation failures | 1 hour | 2 failures | 30 minutes |
| Any misbehaviour | 6 hours | 3 failures | 12 hours |
| Any misbehaviour | 12 hours | 3 failures | 24 hours |
| Any misbehaviour | 48 hours | 3 failures | 3 days |
| Any misbehaviour | 7 days | 3 failures | 7 days |

#### What counts as a misbehaviour event?

- Validation failures (lightweight or full)
- Deployment or startup failures during rentals
- Executor health checks reporting unhealthy state
- Connection errors caused by the miner misrouting a validator

#### How validators enforce bans

- Banned executors are excluded from discovery/rental routing
- Active bans surface in validator logs and Prometheus metrics
- Validation requests return a specific `executor_banned` error to the miner
- When a ban expires, validators automatically clear it; miners can attempt deployments again

---

#### Recovering from a ban

1. **Fix the root cause** (ensure executor is reachable, healthy, and has compatible drivers/container images)
2. **Confirm the ban timer** via validator metrics (`validator_executor_ban_active_status`)
3. **Wait for the ban duration to elapse** (no manual action needed for standard bans)
4. **After expiry**, monitor deployment logs to verify validators reconnect successfully

#### Best practices to avoid bans

- Keep executors patched (CUDA, drivers, containers) and aligned with validator expectations
- Automate health checks and restart loops so degraded nodes self-heal quickly
- Validate images locally before exposing them to validators
- Enforce network/firewall rules to avoid intermittent reachability
- Maintain enough GPU capacity to handle assigned validators without overcommit
- Document procedures: who on your team handles ban remediation, monitoring, and redeploys

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
# Check what's using the port
sudo lsof -i :50051

# Kill process or change axon_port in config
# [bittensor]
# axon_port = 50052
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
# strategy = "highest_stake"
# validator_hotkey = "..." (optional)

# For testing with a specific validator, use fixed_assignment:
# [validator_assignment]
# strategy = "fixed_assignment"
# validator_hotkey = "5G3qVaXz..."

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
#   { host = "192.168.1.100", port = 22, username = "basilica", gpu_category = "H100", gpu_count = 8 }
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
grpcurl -plaintext localhost:50051 list

# Test from external (validator perspective)
grpcurl -plaintext <MINER_PUBLIC_IP>:50051 list

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
    { host = "us-east-1.example.com", port = 22, username = "basilica", gpu_category = "H100", gpu_count = 8 },
    { host = "us-east-2.example.com", port = 22, username = "basilica", gpu_category = "H100", gpu_count = 8 },

    # EU West
    { host = "eu-west-1.example.com", port = 22, username = "basilica", gpu_category = "A100", gpu_count = 4 },
    { host = "eu-west-2.example.com", port = 22, username = "basilica", gpu_category = "A100", gpu_count = 4 },

    # Asia Pacific
    { host = "ap-south-1.example.com", port = 22, username = "basilica", gpu_category = "H100", gpu_count = 8 },
]

# Prices apply uniformly across all regions
[bidding.strategy.static.static_prices]
H100 = 2.50
A100 = 1.20
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
- **Discord**: <https://discord.gg/Cy7c9vPsNK>
- **Website**: <https://www.basilica.ai/>
- **Validator Guide**: [docs/validator.md](validator.md)
- **Architecture Overview**: [docs/architecture.md](architecture.md)
- **API Documentation**: [docs/api.md](api.md)

---

**Happy Mining!**
