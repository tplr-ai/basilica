# Basilica Validator Guide

Comprehensive guide for running a Basilica validator node that verifies GPU compute resources and maintains network quality on the Bittensor network.

---

## Quick Start (TL;DR)

**What it does**: Validator discovers miners via Bittensor metagraph, SSH directly to their GPU nodes for verification, scores performance, and sets network weights.

**Minimum Requirements**:

- Linux server: 48+ CPU cores, 140GB RAM, stable internet
- GPU: Nvidia 1xB200 is minimum for validator permit
- Bittensor wallet: Registered on subnet 39 (mainnet) or 387 (testnet) with sufficient stake
- SSH access: For remote node verification (ephemeral keys auto-generated)

**Quick Setup** (5 steps):

```bash
# 1. Ensure Bittensor wallet exists
btcli wallet list
# Should show your validator wallet and hotkey

# 2. Create minimal config
cat > validator.toml <<EOF
[bittensor]
wallet_name = "your_validator_wallet"
hotkey_name = "your_hotkey"
network = "finney"
netuid = 39
axon_port = 9090
external_ip = "your_public_ip"

[database]
url = "sqlite:./data/validator.db"
run_migrations = true

[verification]
verification_interval = { secs = 600, nanos = 0 }
max_concurrent_verifications = 50
netuid = 39

[api]
bind_address = "0.0.0.0:8080"

[ssh_session]
ssh_key_directory = "/tmp/validator_ssh_keys"

[emission]
burn_percentage = 0.0
burn_uid = 204
EOF

# 3. Build and run
./scripts/validator/build.sh
./basilica-validator --config validator.toml start

# 4. Verify operation
# Check logs for "Discovered X miners from metagraph"
# Check logs for "Verification completed" messages

# 5. Monitor via API
curl http://localhost:8080/health
curl http://localhost:8080/miners
```

**Need details?** See sections below for architecture explanation, verification workflow, weight setting, and advanced configuration.

---

## Table of Contents

1. [Overview](#overview)
2. [Architecture](#architecture)
3. [Prerequisites](#prerequisites)
4. [Understanding the System](#understanding-the-system)
5. [SSH Key Management](#ssh-key-management)
6. [Validator Configuration](#validator-configuration)
7. [Deployment Methods](#deployment-methods)
8. [Verification Flow](#verification-flow)
9. [Weight Setting and Emissions](#weight-setting-and-emissions)
10. [Security & Best Practices](#security--best-practices)
11. [Monitoring](#monitoring)
12. [Troubleshooting](#troubleshooting)
13. [Advanced Topics](#advanced-topics)

---

## Overview

The Basilica validator performs critical network functions that ensure GPU provider quality and distribute rewards fairly across the Bittensor network.

### What You Need

- **Validator Server**: Linux system (no GPU required)
  - 8+ CPU cores, 16GB+ RAM recommended
  - Stable internet connection with low latency
  - Public IP address or proper port forwarding
  - SQLite database (PostgreSQL supported)

- **Bittensor Wallet**: Registered on subnet
  - Mainnet (finney): netuid 39
  - Testnet: netuid 387
  - Sufficient TAO stake for validator permit
  - Hotkey registered on the subnet

- **Network Access**:
  - Outbound SSH access to miner nodes (port 22)
  - Inbound access on axon port (default: 9090)
  - Inbound access on API port (default: 8080)

### Core Responsibilities

1. **Miner Discovery**: Query Bittensor metagraph to discover all miners on the subnet
2. **Node Verification**: SSH directly to GPU nodes for cryptographic verification
3. **Delivery-Based Scoring**: Set miner weights based on rental revenue from the billing API
4. **Weight Setting**: Distribute emissions proportionally to revenue within each GPU category
5. **API Service**: Provide external access for rentals and network queries

---

## Architecture

### SSH-Based Verification Model

Unlike traditional verification systems that rely on intermediary agents, Basilica validators use **direct SSH access** to GPU nodes for verification. This eliminates intermediaries and ensures cryptographic integrity.

```text
┌─────────────────────────────────────────────────────────────┐
│                   BITTENSOR NETWORK                         │
│                     (Metagraph Query)                       │
└────────────────────────┬───────────────────────────────────┘
                         │
                         │ 1. Query metagraph for miners
                         ↓
            ┌────────────────────────┐
            │   VALIDATOR            │
            │                        │
            │  ┌──────────────────┐  │
            │  │ Miner Discovery  │  │
            │  │   (metagraph)    │  │
            │  └──────────────────┘  │
            │  ┌──────────────────┐  │
            │  │ Verification     │  │
            │  │   Scheduler      │  │
            │  └──────────────────┘  │
            │  ┌──────────────────┐  │
            │  │  Weight Setter   │  │
            │  └──────────────────┘  │
            │  ┌──────────────────┐  │
            │  │   REST API       │  │
            │  └──────────────────┘  │
            └────────┬───────────────┘
                     │
                     │ 2. Authenticate via gRPC
                     ↓
        ┌────────────────────────┐
        │   MINER (gRPC Server)  │
        │  - Validates signature │
        │  - Returns SSH details │
        └────────┬───────────────┘
                 │
                 │ 3. SSH Key Authorization
                 ↓
┌────────────────────────────────────────────────┐
│              GPU NODES (SSH endpoints)         │
│  ┌──────────┐  ┌──────────┐  ┌──────────┐    │
│  │  Node 1  │  │  Node 2  │  │  Node N  │    │
│  │ GPU: H100│  │ GPU: A100│  │ GPU: ...  │   │
│  └─────▲────┘  └─────▲────┘  └─────▲────┘    │
│        │             │             │          │
│        └─────────────┴─────────────┘          │
│         4. Validator SSHs directly            │
│            to execute verification            │
└────────────────────────────────────────────────┘
```

### Verification Architecture

The validator employs a **two-tier verification strategy** that optimizes for both security and efficiency:

**Full Validation** (Binary + Hardware Profiling):

- Triggered: New nodes, >6 hours since last validation, or failed lightweight checks
- SSH to node → Upload binaries → Execute verification → Download results
- Validates: GPU attestation, Docker capability, storage, network, hardware specs
- Frequency: Every 6 hours per node
- Score weight: 100% (50% SSH success + 50% binary validation)

**Lightweight Validation** (SSH Accessibility):

- Triggered: Recently validated nodes (<6 hours)
- Quick SSH connection test
- Updates: Last seen timestamp
- Frequency: Every 10 minutes
- Score weight: Reuses previous validation score

### Component Breakdown

#### MinerProver

**Location**: `crates/basilica-validator/src/miner_prover/`

**Purpose**: Main orchestrator for miner verification

**Sub-components**:

- `MinerDiscovery`: Fetches miners from Bittensor metagraph
- `VerificationScheduler`: Dual pipeline (full + lightweight) task scheduling
- `VerificationEngine`: Executes validation against nodes
- `MinerClient`: gRPC communication with miners

**Flow**:

1. Discovery queries metagraph every verification_interval (default: 10 min)
2. Scheduler determines which miners need verification
3. Engine spawns concurrent verification tasks
4. Results stored in database and aggregated for scoring

#### WeightSetter

**Location**: `crates/basilica-validator/src/bittensor_core/weight_setter.rs`

**Purpose**: Distributes emissions based on GPU scoring

**Flow**:

1. Checks current blockchain block every 12 seconds
2. Every N blocks (default: 360), triggers weight setting
3. Syncs delivery records from the billing API for the current epoch
4. Groups deliveries by GPU category, using `revenue_usd` as the scoring metric
5. Allocates weights proportionally to revenue within each category
6. Burns allocation for empty categories (no active rentals) plus configured burn percentage
7. Submits weights to Bittensor chain

#### ApiHandler

**Location**: `crates/basilica-validator/src/api/mod.rs`

**Purpose**: REST API for external services

**Endpoints**:

- Rental management (start, stop, status, logs)
- Node discovery (list available nodes)
- Miner queries (health, nodes, profiles)
- GPU profiles (list by category)
- Verification results

**Node discovery visibility contract**: `GET /nodes` only returns nodes that are bid-active, not rented, not offline, and have at least one row in `gpu_uuid_assignments`. Newly registered nodes are hidden until successful full validation stores GPU UUID assignments. Nodes are hidden again if full-validation failure cleanup removes those assignments.

---

## Prerequisites

### System Requirements

**Hardware**:

- **CPU**: 8+ cores recommended (4 minimum)
- **RAM**: 16GB+ recommended (8GB minimum)
- **Storage**: 100GB+ SSD (for database and logs)
- **Network**: Stable connection with <100ms latency to Bittensor chain

**Operating System**:

- Ubuntu 22.04 LTS (recommended)
- Debian 11+
- Any modern Linux distribution with systemd

### Software Dependencies

```bash
# Update system
sudo apt update && sudo apt upgrade -y

# Install build dependencies (if building from source)
sudo apt install -y \
    build-essential \
    libssl-dev \
    pkg-config \
    protobuf-compiler \
    git \
    curl \
    sqlite3

# Install Rust (if building from source)
curl --proto '=https' --tlsv1.2 -sSf https://sh.rustup.rs | sh
source ~/.cargo/env

# Install Docker (optional, for Docker deployment)
curl -fsSL https://get.docker.com | sh
sudo usermod -aG docker $USER
```

### Bittensor Wallet Setup

```bash
# Install Bittensor CLI
pip install bittensor

# Create validator wallet (if you don't have one)
btcli wallet new_coldkey --wallet.name validator
btcli wallet new_hotkey --wallet.name validator --wallet.hotkey default

# Fund your coldkey with TAO for registration and staking

# Register on subnet
btcli subnet register --netuid 39 --wallet.name validator --wallet.hotkey default

# Check registration
btcli wallet overview --wallet.name validator

# Add stake for validator permit (amount depends on subnet requirements)
btcli stake add --wallet.name validator --wallet.hotkey default --amount 10000
```

**Validator Permit Requirements**:

- Minimum stake varies by subnet configuration
- Check current validators: `btcli metagraph --netuid 39`
- Your stake must be competitive with existing validators

### Network Configuration

**Firewall Rules**:

```bash
# Allow Bittensor axon port (for other validators/miners)
sudo ufw allow 9090/tcp

# Allow API port (for external services)
sudo ufw allow 8080/tcp

# Allow SSH for administration
sudo ufw allow 22/tcp

# Enable firewall
sudo ufw enable
```

**Port Forwarding** (if behind NAT):

- Forward external port 9090 → validator server:9090
- Forward external port 8080 → validator server:8080
- Ensure external_ip is set correctly in config

---

## Understanding the System

This section explains the deep technical theory of how validation works in Basilica.

### Miner Discovery via Metagraph

**How it Works** (code: `miner_prover/discovery.rs:40-122`):

1. **Metagraph Query**:
   - Validator queries Bittensor subtensor for subnet state
   - Retrieves all neurons (validators + miners) on the configured netuid
   - Metagraph contains: UID, hotkey, stake, endpoint (AxonInfo)

2. **Miner Filtering**:

   ```rust
   // Filters out validators, keeps only miners
   if neuron.validator_permit {
       continue; // Skip validators
   }
   ```

3. **Endpoint Extraction**:
   - Parses IP address from u128 format → IPv4/IPv6 string
   - Validates IP is not 0.0.0.0 or ::
   - Validates port is not 0
   - Formats as: `http://{ip}:{port}`

4. **Result**:

   ```rust
   Vec<MinerInfo> {
       uid: u16,
       hotkey: String,  // SS58 AccountId
       endpoint: String, // http://ip:port
       stake: u64,      // in RAO
   }
   ```

**Key Insight**: Validators discover ALL miners from metagraph. There's no centralized registry or manual configuration required.

### gRPC Authentication with Miners

**Authentication Flow** (code: `miner_prover/miner_client.rs:49-150`):

**Step 1: Validator → Miner Authentication**:

```rust
ValidatorAuthRequest {
    validator_hotkey: "5G3qVa...",  // Validator's SS58 hotkey
    timestamp: 1704067200,          // Current UTC timestamp
    signature: "0xabcd...",          // Sr25519 signature
    ssh_public_key: "ssh-ed25519 AAAA...", // Validator's ephemeral SSH key
}
```

**Signature Payload**:

```rust
let payload = format!(
    "BASILICA_AUTH_V1:{}:{}:{}",
    nonce,
    target_miner_hotkey,
    timestamp_secs
);
// Signed with validator's Bittensor keypair
```

**Step 2: Miner Validates**:

- Verifies signature using validator_hotkey
- Checks timestamp is fresh (within 5 minutes)
- Validates nonce uniqueness (replay attack prevention)
- Deploys SSH public key to all nodes (if provided)

**Step 3: Miner → Validator Response**:

```rust
MinerAuthResponse {
    success: true,
    message: "Authenticated successfully",
    session_token: "uuid-session-token", // 1-hour expiry
}
```

**Why Cryptographic Auth?**

- Prevents impersonation (must control Bittensor hotkey)
- No passwords or API keys to manage
- Timestamp + nonce prevent replay attacks
- Aligns with Bittensor's identity system

### Dynamic SSH Endpoint Discovery

**Discovery Protocol** (code: `miner_prover/miner_client.rs:160-210`):

After authentication, validator calls `DiscoverNodes` RPC:

**Request**:

```rust
DiscoverNodesRequest {
    validator_hotkey: "5G3qVa...",
}
```

**Response** (streaming):

```rust
NodeConnectionDetails {
    node_id: "550e8400-e29b-41d4-a716-446655440000", // UUID
    host: "192.168.1.100",
    port: 22,
    username: "basilica",
    ssh_endpoint: "ssh://192.168.1.100:22",
}
```

**Key Details**:

- Miner has already deployed validator's SSH public key to these nodes
- node_id is deterministic UUID from `username@host:port`
- SSH access is immediate (no manual key exchange needed)

**Fallback Mechanism**:

- If `use_dynamic_discovery = false` or gRPC fails
- Falls back to static SSH configuration from database
- Requires manual node configuration (not recommended)

### Two-Tier Validation Strategy

**Strategy Selection Logic** (code: `miner_prover/validation_strategy.rs`):

```rust
fn determine_strategy(node: &Node, last_validation: Option<Timestamp>) -> Strategy {
    match last_validation {
        None => Strategy::Full,  // Never validated
        Some(ts) if now - ts > 6_hours => Strategy::Full,  // Too old
        Some(ts) if previous_failures > 0 => Strategy::Full,  // Had issues
        Some(_) => Strategy::Lightweight,  // Recently validated successfully
    }
}
```

**Full Validation Workflow** (code: `miner_prover/verification.rs:1583-1596`):

1. **SSH Connection**:

   ```bash
   ssh -i /tmp/validator_ssh_keys/ephemeral_key.pem basilica@node_ip
   ```

2. **Binary Upload**:

   ```bash
   # Upload validator-binary (verification executor)
   scp validator-binary basilica@node:/tmp/

   # Upload executor-binary (for GPU attestation)
   scp executor-binary basilica@node:/tmp/
   ```

3. **Remote Execution**:

   ```bash
   # Execute validation binary
   /tmp/validator-binary \
       --executor-binary /tmp/executor-binary \
       --output-format json \
       > /tmp/validation_results.json
   ```

4. **Result Download**:

   ```bash
   scp basilica@node:/tmp/validation_results.json ./results/
   ```

5. **Validation Parsing**:

   ```json
   {
     "gpu_attestation": {
       "gpus": [
         {
           "uuid": "GPU-550e8400-e29b-41d4-a716-446655440000",
           "model": "NVIDIA H100 PCIe",
           "vram_gb": 80,
           "signature": "0xabcd..."
         }
       ]
     },
     "hardware_profile": { "cpu": "...", "ram_gb": 512, "disk_gb": 2000 },
     "docker_validation": { "service_active": true, "version": "24.0.7" },
     "network_profile": { "download_mbps": 10000, "upload_mbps": 5000 },
     "storage_validation": { "available_bytes": 1099511627776 }
   }
   ```

6. **Score Calculation**:

   ```rust
   // Full validation score
   let ssh_score = if ssh_connected { 0.5 } else { 0.0 };
   let binary_score = if validation_passed { 0.5 } else { 0.0 };
   let total_score = ssh_score + binary_score;  // 0.0 - 1.0
   ```

7. **Database Storage**:
   - Store GPU UUIDs in `gpu_uuid_assignments` table
   - Store hardware profile in `node_hardware_profile` table
   - Store validation result in `verification_logs` table
   - Update `miner_gpu_profiles` for scoring

**Lightweight Validation Workflow** (code: `miner_prover/verification.rs:1566-1581`):

1. **SSH Connection Test**:

   ```bash
   ssh -i ephemeral_key.pem -o ConnectTimeout=10 basilica@node echo "ok"
   ```

2. **Update Node Verification Timestamp**:

   ```sql
   UPDATE miner_nodes
   SET status = 'online',
       last_node_check = datetime('now')
   WHERE node_id = ? AND miner_id = ?;
   ```

3. **Score Reuse**:
   - If SSH succeeds: Reuse previous validation score
   - If SSH fails: Set score to 0.0 and trigger full validation next round

**Why Two Tiers?**

- **Security**: Full validation every 6 hours ensures integrity
- **Efficiency**: Lightweight checks every 10 minutes provide fast feedback
- **Resource Optimization**: Avoid uploading binaries unnecessarily
- **Network Health**: Quick detection of offline nodes

### Parallel Verification Execution

**Concurrency Model** (code: `miner_prover/scheduler.rs:268-338`):

```rust
// Spawn concurrent verification tasks
let tasks = miners.iter()
    .map(|miner| verify_miner(miner))
    .collect::<Vec<_>>();

// Execute with semaphore-like limit
let results = futures::stream::iter(tasks)
    .buffer_unordered(config.max_concurrent_verifications)  // Default: 50
    .collect()
    .await;
```

**Dual Pipeline Architecture**:

- **Full Validation Pipeline**: Runs independently with its own scheduler
- **Lightweight Validation Pipeline**: Runs in parallel with full pipeline
- **Cleanup Pipeline**: Runs every 15 minutes to remove stale tasks

**Resource Limits**:

- `max_concurrent_verifications`: 50 (lightweight SSH checks)
- `max_concurrent_full_validations`: 1024 (binary validation requests)
- `max_miners_per_round`: 20 (miners verified per cycle)

### Delivery-Based Scoring

**Weight Setter** (code: `bittensor_core/weight_setter.rs`):

Miners are scored based on rental revenue, not validation ratios or GPU counts. The weight setter fetches delivery records from the billing API and distributes weights proportionally to `revenue_usd` within each GPU category.

1. **Delivery Sync**: At each epoch, the weight setter calls the billing API to fetch delivery records for the period window.

2. **Category Grouping**: Deliveries are grouped by GPU category. Only deliveries with positive `revenue_usd` and a recognized GPU category contribute.

3. **Revenue-Based Scoring**: Within each category, a miner's weight share equals their share of total revenue:

   ```rust
   // For each miner in a category
   let miner_share = miner_revenue / total_category_revenue;
   let miner_weight = category_pool * miner_share;
   ```

**Multi-GPU Miners**:

- Miners with multiple GPU categories earn weight in each category based on rental revenue
- Each category has independent weight allocation
- Example: Miner with H100 and A100 rentals earns weight in both categories proportional to revenue

### Weight Allocation Algorithm

**Emission Distribution** (code: `bittensor_core/weight_allocation.rs`):

**Configuration**:

```toml
[emission]
burn_percentage = 95.0  # Burn 95% of emissions
burn_uid = 204          # Send burn weight to UID 204

[emission.gpu_allocations]
A100 = { weight = 45.0, min_gpu_count = 8, min_gpu_vram = 1 }
H100 = { weight = 55.0, min_gpu_count = 8, min_gpu_vram = 1 }
```

**Weight Calculation**:

1. **Burn Weight**: Configured percentage of total emissions goes to the burn UID.

2. **Category Allocation**: Remaining emissions are split by configured category weights.

3. **Empty Category Burn**: If a category has no active rentals, its allocation is added to burn.

4. **Miner Weights**: Within each active category, weight is proportional to `revenue_usd`.

5. **Example**:

   ```
   Total emissions: 65535 (u16::MAX)
   Burn (95%): 62258 → UID 204
   Remaining: 3277

   A100 allocation (45%): 1475
     - Miner 1 ($500 revenue, 62.5%): 922
     - Miner 2 ($300 revenue, 37.5%): 553

   H100 allocation (55%): 1802
     - Miner 3 ($1000 revenue, 100%): 1802
   ```

**Weight Setting Frequency**:

- Checks blockchain block every 12 seconds (Bittensor block time)
- Sets weights every N blocks (default: 360 blocks ≈ 72 minutes)
- Configurable via `weight_set_interval_blocks`

---

## SSH Key Management

Validators use **ephemeral SSH keys** for node verification, enhancing security through short-lived credentials.

### Automatic Key Generation

**Key Manager** (code: `ssh/key_manager.rs`):

```rust
// Validator generates ephemeral key pair
let keypair = Ed25519KeyPair::generate();

// Stores in configured directory
ssh_key_directory = "/tmp/validator_ssh_keys"

// Key files
/tmp/validator_ssh_keys/validator_{hotkey}_{timestamp}.pem        // Private key
/tmp/validator_ssh_keys/validator_{hotkey}_{timestamp}.pem.pub    // Public key
```

**Default Settings**:

- **Algorithm**: ed25519 (recommended for performance and security)
- **Key Lifetime**: Determined by miner's authorization TTL (typically 1 hour)
- **Storage**: Temporary directory (cleaned up after use)
- **Cleanup Interval**: 60 seconds (removes expired keys)

### Key Distribution Flow

**Step-by-Step** (code: `miner_prover/verification.rs:440-585`):

1. **Validator Generates Key**:

   ```bash
   ssh-keygen -t ed25519 -f /tmp/validator_ssh_keys/ephemeral_key
   ```

2. **Validator Sends Public Key to Miner**:

   ```rust
   ValidatorAuthRequest {
       validator_hotkey: "5G3qVa...",
       ssh_public_key: "ssh-ed25519 AAAAC3Nza... validator@basilica",
       ...
   }
   ```

3. **Miner Deploys Key to All Nodes**:

   ```bash
   # On each node, miner adds key to authorized_keys
   ssh root@node1 'echo "ssh-ed25519 AAAAC3Nza... validator-5G3qVa..." >> ~/.ssh/authorized_keys'
   ssh root@node2 'echo "ssh-ed25519 AAAAC3Nza... validator-5G3qVa..." >> ~/.ssh/authorized_keys'
   ```

4. **Validator SSHs to Nodes**:

   ```bash
   ssh -i /tmp/validator_ssh_keys/ephemeral_key basilica@node1
   ```

5. **Key Cleanup (after verification)**:

   ```bash
   # Validator removes local private key
   rm /tmp/validator_ssh_keys/ephemeral_key

   # Miner removes authorized key from nodes (after session expiry)
   ssh root@node1 'sed -i "/validator-5G3qVa/d" ~/.ssh/authorized_keys'
   ```

### Persistent SSH Keys (Optional)

For long-term access (e.g., rentals), validators can use persistent keys:

```toml
[ssh_session]
persistent_ssh_key_path = "/opt/basilica/keys/validator_persistent.pem"
ssh_key_directory = "/tmp/validator_ssh_keys"
```

**Persistent Key Setup**:

```bash
# Generate persistent key
ssh-keygen -t ed25519 -f /opt/basilica/keys/validator_persistent -C "validator-persistent"

# Set secure permissions
chmod 600 /opt/basilica/keys/validator_persistent
```

**Use Cases**:

- GPU rentals with extended duration
- Manual node administration
- Debugging and troubleshooting

### Security Considerations

**Ephemeral Keys Advantages**:

- ✅ Short-lived credentials (reduced exposure window)
- ✅ Automatic rotation per verification session
- ✅ No long-term key storage on validator
- ✅ Miner controls access duration

**SSH Security Settings** (code: `ssh/session.rs`):

```rust
SshSessionConfig {
    ssh_connection_timeout: Duration::from_secs(30),
    ssh_command_timeout: Duration::from_secs(60),
    ssh_retry_attempts: 3,
    ssh_retry_delay: Duration::from_secs(2),
    strict_host_key_checking: false,  // Nodes have dynamic IPs
    known_hosts_file: None,            // Trust miner-provided endpoints
}
```

**Audit Logging**:

```toml
[ssh_session]
enable_audit_logging = true
audit_log_path = "/var/log/basilica/ssh_audit.log"
```

**Audit Log Format**:

```
2024-01-01T12:00:00Z validator-5G3qVa connected to node-550e8400 (192.168.1.100:22)
2024-01-01T12:00:15Z validator-5G3qVa executed command on node-550e8400: /tmp/validator-binary
2024-01-01T12:01:30Z validator-5G3qVa disconnected from node-550e8400 (duration: 90s)
```

---

## Validator Configuration

Comprehensive breakdown of all configuration options with examples and explanations.

### Configuration File Structure

**Location**: `validator.toml`

**Layered Loading** (priority order):

1. Environment variables (highest priority)
2. TOML configuration file
3. Compiled defaults (lowest priority)

**Example: Override with environment variables**:

```bash
# Override database URL
export BASILICA_DATABASE__URL="postgresql://user:pass@localhost/validator"

# Override verification interval
export BASILICA_VERIFICATION__VERIFICATION_INTERVAL__SECS=300

# Run validator
./basilica-validator --config validator.toml start
```

### Complete Configuration Example

```toml
# === Bittensor Network Configuration ===
[bittensor]
# Wallet name (coldkey) - matches ~/.bittensor/wallets/{wallet_name}/
wallet_name = "validator"

# Hotkey name - matches ~/.bittensor/wallets/{wallet_name}/hotkeys/{hotkey_name}
hotkey_name = "default"

# Network selection: "finney" (mainnet), "test" (testnet), or "local"
network = "finney"

# Subnet ID: 39 for mainnet, 387 for testnet
netuid = 39

# Chain endpoint (auto-detected if not specified)
# Mainnet: wss://entrypoint-finney.opentensor.ai:443
# Testnet: wss://test.finney.opentensor.ai:443
# chain_endpoint = "wss://entrypoint-finney.opentensor.ai:443"

# Axon server port (for Bittensor network communication)
axon_port = 9090

# External IP address (required for proper network advertisement)
external_ip = "203.0.113.10"

# Optional: Override advertised axon endpoint
# advertised_axon_endpoint = "http://validator.example.com:9090"
# advertised_axon_tls = false

# === Database Configuration ===
[database]
# Database URL: SQLite or PostgreSQL
# SQLite (default): sqlite:./data/validator.db
# PostgreSQL: postgresql://user:pass@localhost:5432/validator
url = "sqlite:./data/validator.db"

# Connection pool settings
max_connections = 10
min_connections = 1

# Run database migrations on startup
run_migrations = true

# Connection timeout
[database.connect_timeout]
secs = 30
nanos = 0

# Idle connection timeout
[database.idle_timeout]
secs = 600
nanos = 0

# Maximum connection lifetime
[database.max_lifetime]
secs = 3600
nanos = 0

# === HTTP API Server Configuration ===
[server]
# API server bind address
host = "0.0.0.0"
port = 8080

# === Logging Configuration ===
[logging]
# Log level: trace, debug, info, warn, error
level = "info"

# Log format: json, pretty, compact
format = "pretty"

# Optional: Log to file
# file = "/var/log/basilica/validator.log"

# === Metrics Configuration ===
[metrics]
# Enable Prometheus metrics
enabled = true

# Metrics collection interval
[metrics.collection_interval]
secs = 30
nanos = 0

# Prometheus exporter settings
[metrics.prometheus]
host = "127.0.0.1"
port = 9090
path = "/metrics"

# Default labels for all metrics
[metrics.default_labels]
# env = "production"
# region = "us-east"

# Metrics retention period
[metrics.retention_period]
secs = 604800  # 7 days
nanos = 0

# === Verification Configuration ===
[verification]
# How often to run verification rounds
[verification.verification_interval]
secs = 600      # 10 minutes
nanos = 0

# Maximum concurrent lightweight verifications (SSH checks)
max_concurrent_verifications = 50

# Maximum concurrent full validations (binary executions)
max_concurrent_full_validations = 1024

# Timeout for individual verification challenges
[verification.challenge_timeout]
secs = 120
nanos = 0

# Minimum score threshold for miners (0.0 - 1.0)
min_score_threshold = 0.1

# Maximum miners to verify per round
max_miners_per_round = 20

# Minimum interval between verifying the same miner
[verification.min_verification_interval]
secs = 1800     # 30 minutes
nanos = 0

# Subnet ID (should match bittensor.netuid)
netuid = 39

# Use dynamic SSH endpoint discovery from miners
use_dynamic_discovery = true

# Timeout for miner discovery operations
[verification.discovery_timeout]
secs = 30
nanos = 0

# Fall back to static SSH config if dynamic discovery fails
fallback_to_static = true

# Cache miner endpoint info TTL
[verification.cache_miner_info_ttl]
secs = 300      # 5 minutes
nanos = 0

# gRPC port offset from miner's axon port (default: uses port 8080)
# grpc_port_offset = 1000  # Would use axon_port + 1000

# Collateral event scan interval (blockchain monitoring)
[verification.collateral_event_scan_interval]
secs = 12       # 1 Bittensor block
nanos = 0

# Interval between full binary validations per node
[verification.node_validation_interval]
secs = 21600    # 6 hours
nanos = 0

# Time period for cleaning up GPU assignments from offline nodes
[verification.gpu_assignment_cleanup_ttl]
secs = 7200     # 2 hours
nanos = 0

# Enable worker queue for decoupled validation execution
enable_worker_queue = false

# Binary validation settings
[verification.binary_validation]
# Path to validator-binary executable (excluded from docs per request)
validator_binary_path = "./validator-binary"
# Path to executor-binary for upload (excluded from docs per request)
executor_binary_path = "./executor-binary"
# Binary execution timeout
execution_timeout_secs = 1200  # 20 minutes
# Output format
output_format = "json"
# Enable binary validation
enabled = true
# Default node port for SSH tunnel cleanup
node_port = 3000

# Validation server mode configuration
[verification.binary_validation.server_mode]
bind_address = "127.0.0.1:4010"
remote_concurrency = 1024
verify_concurrency = 1
queue_capacity = 4096
health_check_interval_secs = 30
job_poll_interval_ms = 500
max_poll_attempts = 2400
server_ready_timeout_secs = 30
server_ready_check_interval_ms = 500

# Docker validation settings
[verification.docker_validation]
docker_image = "nvidia/cuda:12.8.0-runtime-ubuntu22.04"
pull_timeout_secs = 1800  # 30 minutes

# Storage validation settings
[verification.storage_validation]
min_required_storage_bytes = 1099511627776  # 1TB

# === Automatic Verification Configuration ===
[automatic_verification]
# Enable automatic verification during discovery
enabled = true

# Discovery verification interval in seconds
discovery_interval = 300  # 5 minutes

# Minimum time between verifications for same miner
min_verification_interval_hours = 1

# Maximum concurrent verifications
max_concurrent_verifications = 50

# Enable SSH session automation
enable_ssh_automation = true

# === Storage Configuration ===
[storage]
# Data directory for validator storage
data_dir = "./data"

# === API Configuration ===
[api]
# API server bind address (external services)
bind_address = "0.0.0.0:8080"

# Maximum request body size (bytes)
max_body_size = 1048576  # 1MB

# Optional: API key for authentication
# api_key = "your-secret-api-key"

# Default miner port for connections
miner_port = 8091

# === SSH Session Configuration ===
[ssh_session]
# Directory for ephemeral SSH keys
ssh_key_directory = "/tmp/validator_ssh_keys"

# SSH key algorithm: "ed25519" or "rsa"
key_algorithm = "ed25519"

# Optional: Persistent SSH private key path
# persistent_ssh_key_path = "/opt/basilica/keys/validator_persistent.pem"

# Default session duration (seconds)
default_session_duration = 300  # 5 minutes

# Maximum session duration (seconds)
max_session_duration = 3600  # 1 hour

# Rental session duration (0 = no predetermined duration)
rental_session_duration = 0

# Key cleanup interval
[ssh_session.key_cleanup_interval]
secs = 60
nanos = 0

# Enable automated SSH session management
enable_automated_sessions = true

# Maximum concurrent SSH sessions
max_concurrent_sessions = 5

# Session rate limit per hour
session_rate_limit = 20

# Enable SSH audit logging
enable_audit_logging = true

# Audit log file path
audit_log_path = "/var/log/basilica/ssh_audit.log"

# SSH connection timeout
[ssh_session.ssh_connection_timeout]
secs = 30
nanos = 0

# SSH command execution timeout
[ssh_session.ssh_command_timeout]
secs = 60
nanos = 0

# SSH retry attempts on connection failure
ssh_retry_attempts = 3

# Delay between retry attempts
[ssh_session.ssh_retry_delay]
secs = 2
nanos = 0

# Strict host key checking (false for dynamic node IPs)
strict_host_key_checking = false

# Known hosts file path (None = don't check)
# known_hosts_file = "/home/validator/.ssh/known_hosts"

# === Emission Configuration ===
[emission]
# Percentage of emissions to burn (0.0 - 100.0)
burn_percentage = 0.0

# UID to send burn weights to
burn_uid = 204

# Minimum miners required per GPU category to enable incentives
min_miners_per_category = 1

# Blocks between weight setting operations
weight_set_interval_blocks = 360

# Weight version key (for protocol upgrades)
weight_version_key = 0

# GPU model allocations with weights and minimum requirements
[emission.gpu_allocations]
A100 = { weight = 45.0, min_gpu_count = 8, min_gpu_vram = 1 }
H100 = { weight = 55.0, min_gpu_count = 8, min_gpu_vram = 1 }

# === Database Cleanup Configuration ===
[cleanup]
# Enable automatic database cleanup
enabled = true

# Cleanup interval
[cleanup.cleanup_interval]
secs = 3600     # 1 hour
nanos = 0

# Retention periods for different data types
[cleanup.verification_logs_retention]
secs = 2592000  # 30 days
nanos = 0

[cleanup.emission_metrics_retention]
secs = 7776000  # 90 days
nanos = 0

[cleanup.rental_logs_retention]
secs = 2592000  # 30 days
nanos = 0
```

### Configuration Validation

**Validate Before Starting**:

```bash
# Validate configuration file
./basilica-validator --config validator.toml config validate

# Example output:
# ✓ Configuration validation passed
# ✓ Database connection successful
# ✓ Bittensor wallet found: validator/default
# ✓ Network connectivity confirmed
# ! Warning: external_ip not set (auto-detection will be used)
# ! Warning: GPU allocations total weight is 100.0% (recommended)
```

**Common Validation Errors**:

1. **Invalid wallet path**:

   ```
   Error: Wallet not found at ~/.bittensor/wallets/validator/hotkeys/default
   Solution: Check wallet_name and hotkey_name match your Bittensor wallet
   ```

2. **Database connection failed**:

   ```
   Error: Failed to connect to database: Connection refused
   Solution: Ensure database is running and URL is correct
   ```

3. **Invalid GPU allocations**:

   ```
   Error: GPU allocation weights must sum to 100.0 (current: 95.0)
   Solution: Adjust gpu_allocations weights to total 100.0
   ```

4. **Network unreachable**:

   ```
   Error: Cannot reach Bittensor chain endpoint
   Solution: Check internet connectivity and chain_endpoint URL
   ```

---

## Deployment Methods

Four deployment methods for different use cases: Binary, Systemd, Docker, and Docker Compose.

### Method 1: Binary Deployment

**Best for**: Development, testing, manual control

**Step 1: Build the Validator**

```bash
# Clone repository
git clone https://github.com/your-org/basilica.git
cd basilica/basilica

# Build using the build script
./scripts/validator/build.sh

# Verify build
ls -lh basilica-validator
# Should show ~50MB binary
```

**Step 2: Prepare Configuration**

```bash
# Copy example config
cp config/validator.correct.toml config/validator.toml

# Edit configuration
nano config/validator.toml

# Set required fields:
# - bittensor.wallet_name
# - bittensor.hotkey_name
# - bittensor.external_ip
```

**Step 3: Create Data Directories**

```bash
# Create directories
mkdir -p data logs /tmp/validator_ssh_keys

# Set permissions
chmod 700 /tmp/validator_ssh_keys
```

**Step 4: Run Validator**

```bash
# Run in foreground (for testing)
./basilica-validator --config config/validator.toml start

# Run in background with nohup
nohup ./basilica-validator --config config/validator.toml start > logs/validator.log 2>&1 &

# Check process
ps aux | grep basilica-validator

# View logs
tail -f logs/validator.log
```

**Step 5: Verify Operation**

```bash
# Check health endpoint
curl http://localhost:8080/health

# Check miner discovery
curl http://localhost:8080/miners | jq

# Check metrics
curl http://localhost:9090/metrics
```

---

### Method 2: Systemd Service

**Best for**: Production, auto-restart, system integration

**Step 1: Build and Install**

```bash
# Build validator
./scripts/validator/build.sh

# Create installation directory
sudo mkdir -p /opt/basilica/{bin,config,data,logs}

# Copy binary
sudo cp basilica-validator /opt/basilica/bin/

# Copy configuration
sudo cp config/validator.toml /opt/basilica/config/

# Set ownership
sudo chown -R $USER:$USER /opt/basilica
```

**Step 2: Create Systemd Service File**

Create `/etc/systemd/system/basilica-validator.service`:

```ini
[Unit]
Description=Basilica Validator
After=network-online.target
Wants=network-online.target

[Service]
Type=simple
User=root
WorkingDirectory=/opt/basilica
ExecStart=/opt/basilica/bin/basilica-validator --config /opt/basilica/config/validator.toml start
Restart=on-failure
RestartSec=5
StandardOutput=journal
StandardError=journal
SyslogIdentifier=basilica-validator

# Security settings
NoNewPrivileges=yes
PrivateTmp=yes
ProtectSystem=strict
ReadWritePaths=/opt/basilica/data /opt/basilica/logs /tmp/validator_ssh_keys
ProtectHome=yes

# Resource limits
LimitNOFILE=65536
LimitNPROC=4096

[Install]
WantedBy=multi-user.target
```

**Step 3: Enable and Start Service**

```bash
# Reload systemd
sudo systemctl daemon-reload

# Enable service (start on boot)
sudo systemctl enable basilica-validator

# Start service
sudo systemctl start basilica-validator

# Check status
sudo systemctl status basilica-validator

# View logs
sudo journalctl -u basilica-validator -f

# View recent logs
sudo journalctl -u basilica-validator -n 100
```

**Step 4: Service Management**

```bash
# Stop service
sudo systemctl stop basilica-validator

# Restart service
sudo systemctl restart basilica-validator

# Disable service (don't start on boot)
sudo systemctl disable basilica-validator

# View service configuration
sudo systemctl cat basilica-validator
```

---

### Method 3: Docker Deployment

**Best for**: Containerized environments, easy updates, isolation

**Step 1: Build Docker Image**

```bash
# Build image using build script
cd scripts/validator
./build.sh --docker

# Or build manually
docker build -t basilica-validator:latest -f Dockerfile ../..

# Verify image
docker images | grep basilica-validator
```

**Step 2: Prepare Configuration and Volumes**

```bash
# Create host directories
mkdir -p /opt/basilica/{config,data,logs,wallets,ssh_keys}

# Copy configuration
cp ../../config/validator.toml /opt/basilica/config/

# Copy Bittensor wallet (or mount existing)
cp -r ~/.bittensor/wallets /opt/basilica/

# Set permissions
chmod 700 /opt/basilica/ssh_keys
chmod 700 /opt/basilica/wallets
```

**Step 3: Run Container**

```bash
# Run with Docker
docker run -d \
  --name basilica-validator \
  --restart unless-stopped \
  -p 9090:9090 \
  -p 8080:8080 \
  -p 9090:9090 \
  -v /opt/basilica/config:/opt/basilica/config:ro \
  -v /opt/basilica/data:/opt/basilica/data \
  -v /opt/basilica/logs:/opt/basilica/logs \
  -v /opt/basilica/wallets:/root/.bittensor/wallets:ro \
  -v /opt/basilica/ssh_keys:/tmp/validator_ssh_keys \
  basilica-validator:latest \
  --config /opt/basilica/config/validator.toml start

# Check container status
docker ps | grep basilica-validator

# View logs
docker logs -f basilica-validator

# View recent logs
docker logs --tail 100 basilica-validator
```

**Step 4: Container Management**

```bash
# Stop container
docker stop basilica-validator

# Start container
docker start basilica-validator

# Restart container
docker restart basilica-validator

# Remove container
docker rm -f basilica-validator

# Update to new version
docker pull basilica-validator:latest
docker stop basilica-validator
docker rm basilica-validator
# Re-run docker run command from Step 3
```

---

### Method 4: Docker Compose (Recommended for Production)

**Best for**: Production, monitoring stack, easy management

**Step 1: Prepare Compose File**

**Location**: `scripts/validator/compose.prod.yml`

```yaml
version: '3.8'

services:
  validator:
    image: basilica-validator:latest
    container_name: basilica-validator
    restart: unless-stopped
    command: --config /opt/basilica/config/validator.toml start
    ports:
      - "9090:9090"  # Bittensor axon
      - "8080:8080"  # API server
      - "9090:9090"  # Metrics
    volumes:
      - /opt/basilica/config:/opt/basilica/config:ro
      - /opt/basilica/data:/opt/basilica/data
      - /opt/basilica/logs:/opt/basilica/logs
      - ~/.bittensor/wallets:/root/.bittensor/wallets:ro
      - validator_ssh_keys:/tmp/validator_ssh_keys
    environment:
      - RUST_LOG=info
      - RUST_BACKTRACE=1
    networks:
      - basilica_network
    healthcheck:
      test: ["CMD", "curl", "-f", "http://localhost:8080/health"]
      interval: 30s
      timeout: 10s
      retries: 3

  prometheus:
    image: prom/prometheus:latest
    container_name: basilica-prometheus
    restart: unless-stopped
    command:
      - '--config.file=/etc/prometheus/prometheus.yml'
      - '--storage.tsdb.path=/prometheus'
      - '--storage.tsdb.retention.time=30d'
    ports:
      - "9091:9090"
    volumes:
      - ./prometheus.yml:/etc/prometheus/prometheus.yml:ro
      - prometheus_data:/prometheus
    networks:
      - basilica_network

  grafana:
    image: grafana/grafana:latest
    container_name: basilica-grafana
    restart: unless-stopped
    ports:
      - "3000:3000"
    environment:
      - GF_SECURITY_ADMIN_PASSWORD=admin
      - GF_USERS_ALLOW_SIGN_UP=false
    volumes:
      - grafana_data:/var/lib/grafana
      - ./grafana/dashboards:/etc/grafana/provisioning/dashboards:ro
      - ./grafana/datasources:/etc/grafana/provisioning/datasources:ro
    networks:
      - basilica_network
    depends_on:
      - prometheus

volumes:
  validator_ssh_keys:
  prometheus_data:
  grafana_data:

networks:
  basilica_network:
    driver: bridge
```

**Step 2: Create Prometheus Configuration**

**Location**: `scripts/validator/prometheus.yml`

```yaml
global:
  scrape_interval: 15s
  evaluation_interval: 15s

scrape_configs:
  - job_name: 'basilica-validator'
    static_configs:
      - targets: ['validator:9090']
        labels:
          service: 'validator'
```

**Step 3: Deploy Stack**

```bash
# Navigate to validator scripts
cd scripts/validator

# Ensure configuration exists
ls /opt/basilica/config/validator.toml

# Deploy with Docker Compose
docker compose -f compose.prod.yml up -d

# Check all services
docker compose -f compose.prod.yml ps

# View logs
docker compose -f compose.prod.yml logs -f validator

# View specific service logs
docker compose -f compose.prod.yml logs -f prometheus
docker compose -f compose.prod.yml logs -f grafana
```

**Step 4: Access Services**

```bash
# Validator API
curl http://localhost:8080/health

# Prometheus
open http://localhost:9091

# Grafana
open http://localhost:3000
# Login: admin / admin
```

**Step 5: Stack Management**

```bash
# Stop all services
docker compose -f compose.prod.yml down

# Stop but keep volumes
docker compose -f compose.prod.yml stop

# Start services
docker compose -f compose.prod.yml start

# Restart specific service
docker compose -f compose.prod.yml restart validator

# View resource usage
docker compose -f compose.prod.yml stats

# Remove everything including volumes
docker compose -f compose.prod.yml down -v
```

---

### Deployment Automation Script

**Use the provided deployment script for remote deployment**:

```bash
# Deploy to remote server with systemd
./scripts/validator/deploy.sh \
  --server user@validator.example.com:22 \
  --mode systemd \
  --sync-wallets \
  --health-check \
  --follow-logs

# Deploy with Docker
./scripts/validator/deploy.sh \
  --server user@validator.example.com:22 \
  --mode docker \
  --sync-wallets

# Deploy with Docker Compose (recommended)
./scripts/validator/deploy.sh \
  --server user@validator.example.com:22 \
  --mode docker-compose \
  --sync-wallets \
  --health-check
```

**Script Features** (code: `scripts/validator/deploy.sh`):

- Builds validator locally
- Uploads binary and configuration to remote server
- Optionally syncs Bittensor wallets
- Installs and starts service
- Performs health checks
- Can follow logs after deployment

---

## Verification Flow

Complete walkthrough of how validators verify miners and their GPU nodes.

### 6-Step Verification Process

#### Step 1: Miner Discovery from Metagraph

**Trigger**: Every `verification_interval` (default: 10 minutes)

**Code Flow** (`miner_prover/discovery.rs:40-122`):

```rust
// Query Bittensor metagraph
let neurons = subtensor.get_neurons(netuid).await?;

// Filter for miners (exclude validators)
let miners = neurons.iter()
    .filter(|n| !n.validator_permit)
    .map(|n| MinerInfo {
        uid: n.uid,
        hotkey: n.hotkey.to_ss58(),
        endpoint: format!("http://{}:{}", n.axon_info.ip, n.axon_info.port),
        stake: n.total_stake,
    })
    .collect();
```

**Result**: List of all miners on the subnet with their endpoints

**Example Discovery Log**:

```
2024-01-01T12:00:00Z INFO  Discovered 47 miners from metagraph
2024-01-01T12:00:00Z DEBUG Miner UID 5: 5G3qVa... at http://203.0.113.10:8080 (stake: 12500 TAO)
2024-01-01T12:00:00Z DEBUG Miner UID 12: 5HGjWa... at http://198.51.100.20:8080 (stake: 8300 TAO)
...
```

---

#### Step 2: gRPC Authentication with Miner

**Trigger**: For each miner selected for verification

**Code Flow** (`miner_prover/miner_client.rs:124-150`):

```rust
// Generate ephemeral SSH key
let ssh_keypair = Ed25519KeyPair::generate();
let ssh_public_key = ssh_keypair.public_key_openssh();

// Create authentication request
let auth_request = ValidatorAuthRequest {
    validator_hotkey: config.bittensor.hotkey.to_string(),
    timestamp: Utc::now().timestamp(),
    signature: sign_payload(&payload, &bittensor_keypair),
    ssh_public_key: Some(ssh_public_key),
    nonce: generate_nonce(),
};

// Send to miner's gRPC endpoint
let response = miner_client.authenticate_validator(auth_request).await?;
```

**Authentication Payload**:

```
BASILICA_AUTH_V1:123456:5FHneW...:1704067200
```

**Example Auth Log**:

```
2024-01-01T12:00:05Z INFO  Authenticating with miner UID 5 (5G3qVa...)
2024-01-01T12:00:05Z DEBUG Generated ephemeral SSH key: ssh-ed25519 AAAAC3Nza...
2024-01-01T12:00:06Z INFO  Authentication successful, session token: 550e8400-e29b-41d4-a716-446655440000
```

---

#### Step 3: Node Discovery via gRPC

**Trigger**: Immediately after successful authentication

**Code Flow** (`miner_prover/miner_client.rs:160-210`):

```rust
// Request node details from miner
let discover_request = DiscoverNodesRequest {
    validator_hotkey: config.bittensor.hotkey.to_string(),
};

// Receive streaming response
let mut node_stream = miner_client.discover_nodes(discover_request).await?;

while let Some(node_details) = node_stream.message().await? {
    nodes.push(NodeConnectionDetails {
        node_id: node_details.node_id,
        host: node_details.host,
        port: node_details.port,
        username: node_details.username,
        ssh_endpoint: node_details.ssh_endpoint,
    });
}
```

**Example Discovery Response**:

```json
{
  "node_id": "550e8400-e29b-41d4-a716-446655440000",
  "host": "192.168.1.100",
  "port": 22,
  "username": "basilica",
  "ssh_endpoint": "ssh://192.168.1.100:22"
}
```

**Example Discovery Log**:

```
2024-01-01T12:00:07Z INFO  Discovered 3 nodes from miner UID 5
2024-01-01T12:00:07Z DEBUG Node 550e8400: ssh://192.168.1.100:22 (basilica@192.168.1.100)
2024-01-01T12:00:07Z DEBUG Node 660f9511: ssh://192.168.1.101:22 (basilica@192.168.1.101)
2024-01-01T12:00:07Z DEBUG Node 770fa622: ssh://192.168.1.102:22 (basilica@192.168.1.102)
```

---

#### Step 4: Strategy Selection (Full vs Lightweight)

**Trigger**: For each node discovered

**Code Flow** (`miner_prover/validation_strategy.rs:10-50`):

```rust
// Check last validation time
let last_validation = db.get_last_validation(node_id).await?;

let strategy = match last_validation {
    None => {
        // Never validated
        ValidationStrategy::Full
    }
    Some(validation) if now - validation.timestamp > Duration::from_secs(6 * 3600) => {
        // More than 6 hours old
        ValidationStrategy::Full
    }
    Some(validation) if validation.failures > 0 => {
        // Previous failures
        ValidationStrategy::Full
    }
    Some(_) => {
        // Recently validated successfully
        ValidationStrategy::Lightweight
    }
};
```

**Example Strategy Log**:

```
2024-01-01T12:00:08Z DEBUG Node 550e8400: Last validated 2 hours ago → Lightweight
2024-01-01T12:00:08Z DEBUG Node 660f9511: Never validated → Full
2024-01-01T12:00:08Z DEBUG Node 770fa622: Last validated 8 hours ago → Full
```

---

#### Step 5a: Full Validation Execution

**Trigger**: Node requires full validation

**Detailed Flow** (`miner_prover/verification.rs:1583-1596`):

**5a.1: SSH Connection**:

```bash
ssh -i /tmp/validator_ssh_keys/ephemeral_550e8400.pem \
    -o ConnectTimeout=30 \
    -o StrictHostKeyChecking=no \
    basilica@192.168.1.100
```

**5a.2: Binary Upload**:

```bash
# Note: Binary upload/execution excluded from docs per user request
# Validator uploads verification binaries to node
# Executes GPU attestation and hardware profiling
# Downloads JSON results
```

**5a.3: Result Parsing**:

```json
{
  "gpu_attestation": {
    "gpus": [
      {
        "uuid": "GPU-550e8400-e29b-41d4-a716-446655440000",
        "model": "NVIDIA H100 PCIe",
        "vram_gb": 80,
        "cuda_version": "12.8",
        "driver_version": "550.54.15",
        "compute_capability": "9.0"
      }
    ],
    "validation_passed": true
  },
  "hardware_profile": {
    "cpu_model": "AMD EPYC 9654",
    "cpu_cores": 96,
    "ram_gb": 512,
    "disk_gb": 7680
  },
  "docker_validation": {
    "service_active": true,
    "docker_version": "24.0.7",
    "nvidia_runtime": true,
    "images_pulled": 1
  },
  "network_profile": {
    "download_mbps": 10000,
    "upload_mbps": 5000,
    "latency_ms": 15
  },
  "storage_validation": {
    "total_bytes": 8246337208320,
    "available_bytes": 6597069766656,
    "meets_requirement": true
  }
}
```

**5a.4: Score Calculation**:

```rust
let ssh_score = if ssh_connected { 0.5 } else { 0.0 };
let binary_score = if validation_passed { 0.5 } else { 0.0 };
let total_score = ssh_score + binary_score;
```

**5a.5: Database Storage**:

```sql
-- Store GPU UUID assignments
INSERT INTO gpu_uuid_assignments (gpu_uuid, node_id, miner_id, gpu_name, last_verified)
VALUES ('GPU-550e8400...', '550e8400...', 5, 'NVIDIA H100 PCIe', NOW());

-- Store hardware profile
INSERT INTO node_hardware_profile (miner_uid, node_id, cpu_model, cpu_cores, ram_gb, disk_gb)
VALUES (5, '550e8400...', 'AMD EPYC 9654', 96, 512, 7680);

-- Store verification result
INSERT INTO verification_logs (node_id, verification_type, score, success, details)
VALUES ('550e8400...', 'full', 1.0, 1, '{"gpu_count": 1, "model": "H100"}');

-- Update GPU profile for scoring
INSERT INTO miner_gpu_profiles (miner_uid, gpu_counts_json, total_score)
VALUES (5, '{"H100": 1}', 1.0)
ON CONFLICT (miner_uid) DO UPDATE
SET gpu_counts_json = '{"H100": 1}', total_score = 1.0, last_updated = NOW();
```

**Example Full Validation Log**:

```
2024-01-01T12:00:10Z INFO  [Full] Validating node 660f9511 (miner UID 5)
2024-01-01T12:00:11Z DEBUG [Full] SSH connected to 192.168.1.101:22
2024-01-01T12:00:12Z DEBUG [Full] Binary upload complete (15.2 MB in 1.2s)
2024-01-01T12:00:45Z DEBUG [Full] Binary execution complete (33.1s)
2024-01-01T12:00:46Z DEBUG [Full] Results downloaded (142 KB)
2024-01-01T12:00:46Z INFO  [Full] Validation passed: 1 GPU (H100), Docker: ✓, Storage: ✓
2024-01-01T12:00:47Z INFO  [Full] Node 660f9511 score: 1.00 (SSH: 0.50 + Binary: 0.50)
```

---

#### Step 5b: Lightweight Validation Execution

**Trigger**: Node recently validated (<6 hours, no failures)

**Detailed Flow** (`miner_prover/verification.rs:1566-1581`):

**5b.1: SSH Connection Test**:

```bash
ssh -i /tmp/validator_ssh_keys/ephemeral_550e8400.pem \
    -o ConnectTimeout=10 \
    basilica@192.168.1.100 \
    echo "ok"
```

**5b.2: Node Verification Timestamp Update**:

```sql
UPDATE miner_nodes
SET status = 'online',
    last_node_check = datetime('now')
WHERE node_id = '550e8400...' AND miner_id = 'miner_5';
```

**5b.3: Score Reuse**:

```rust
// Reuse previous validation score if SSH succeeds
let score = if ssh_connected {
    previous_validation.score
} else {
    0.0  // SSH failed, mark as down
};
```

**Example Lightweight Validation Log**:

```
2024-01-01T12:00:10Z INFO  [Lightweight] Checking node 550e8400 (miner UID 5)
2024-01-01T12:00:11Z DEBUG [Lightweight] SSH connected to 192.168.1.100:22
2024-01-01T12:00:11Z INFO  [Lightweight] Node 550e8400 is accessible, reusing score: 1.00
```

---

#### Step 6: Score Aggregation and Storage

**Trigger**: After all nodes verified for a miner

**Code Flow** (`miner_prover/verification.rs:280-313`):

```rust
// Aggregate scores across all nodes
let total_score = node_scores.iter().sum::<f64>();
let average_score = total_score / node_scores.len() as f64;

// Update miner's overall score
db.update_miner_score(miner_uid, average_score).await?;

// Log result
info!(
    "Miner UID {}: {} nodes verified, average score: {:.2}",
    miner_uid,
    node_scores.len(),
    average_score
);
```

**Example Aggregation Log**:

```
2024-01-01T12:01:00Z INFO  Verification round complete for miner UID 5
2024-01-01T12:01:00Z INFO  Nodes verified: 3 (2 full, 1 lightweight)
2024-01-01T12:01:00Z INFO  Node scores: [1.00, 1.00, 1.00]
2024-01-01T12:01:00Z INFO  Miner UID 5 average score: 1.00
2024-01-01T12:01:00Z INFO  GPU profile: H100=3
```

**Final Database State**:

```sql
-- Miner GPU profile for weight setting
SELECT * FROM miner_gpu_profiles WHERE miner_uid = 5;
-- miner_uid | gpu_counts_json    | total_score | last_updated
-- 5         | {"H100": 3}        | 1.00        | 2024-01-01 12:01:00

-- Individual node verification results
SELECT * FROM verification_logs WHERE node_id IN (SELECT id FROM miner_nodes WHERE miner_id = 5);
-- node_id   | verification_type | score | success | timestamp
-- 550e8400  | lightweight       | 1.00  | 1       | 2024-01-01 12:00:11
-- 660f9511  | full              | 1.00  | 1       | 2024-01-01 12:00:47
-- 770fa622  | full              | 1.00  | 1       | 2024-01-01 12:00:55
```

---

### Parallel Verification Execution

**Concurrency Management** (`miner_prover/scheduler.rs:268-338`):

```rust
// Create verification tasks for all selected miners
let tasks: Vec<_> = miners.iter()
    .map(|miner| {
        let engine = verification_engine.clone();
        async move {
            engine.verify_miner(miner).await
        }
    })
    .collect();

// Execute with concurrency limit
let results = futures::stream::iter(tasks)
    .buffer_unordered(config.max_concurrent_verifications)
    .collect::<Vec<_>>()
    .await;
```

**Resource Limits**:

- Lightweight verifications: Up to 50 concurrent
- Full validations: Up to 1024 concurrent
- Miners per round: Up to 20

**Example Parallel Execution Log**:

```
2024-01-01T12:00:00Z INFO  Starting verification round: 20 miners selected
2024-01-01T12:00:00Z DEBUG Spawning 20 concurrent verification tasks
2024-01-01T12:00:00Z DEBUG Active verifications: 20 (lightweight: 15, full: 5)
2024-01-01T12:00:15Z DEBUG Completed: 8 verifications (12 remaining)
2024-01-01T12:00:30Z DEBUG Completed: 16 verifications (4 remaining)
2024-01-01T12:00:45Z DEBUG Completed: 20 verifications (0 remaining)
2024-01-01T12:00:45Z INFO  Verification round complete: 20/20 successful (100%)
```

---

## Weight Setting and Emissions

How validators distribute TAO emissions based on rental revenue across GPU categories.

### Emission Algorithm Overview

**Goal**: Fairly distribute subnet emissions across miners based on rental revenue within each GPU category. Miners only receive weight when their nodes are actively rented and generating revenue.

**Configuration-Driven Allocation**:

```toml
[emission]
burn_percentage = 95.0
burn_uid = 204

[emission.gpu_allocations]
A100 = { weight = 45.0, min_gpu_count = 8, min_gpu_vram = 1 }
H100 = { weight = 55.0, min_gpu_count = 8, min_gpu_vram = 1 }
```

### Weight Calculation Process

#### Step 1: Sync Delivery Records

At each weight-setting epoch, the validator fetches delivery records from the billing API:

```rust
let deliveries = api_client.get_miner_delivery(epoch.period_start, epoch.period_end).await?;
```

Each delivery contains `miner_hotkey`, `node_id`, `gpu_category`, and `revenue_usd`.

---

#### Step 2: Group Deliveries by Category

Deliveries are grouped by GPU category. Only deliveries with `revenue_usd > 0` and recognized GPU categories contribute:

```
Miner 5:  H100 rentals → $800 revenue
Miner 12: H100 rentals → $400 revenue, A100 rentals → $600 revenue
Miner 23: A100 rentals → $300 revenue
```

---

#### Step 3: Burn Weight Calculation

```rust
let total_weight = u16::MAX;  // 65535

// Burn weight (95% of total)
let burn_weight = (total_weight as f64 * 0.95) as u16;  // 62258

weights[burn_uid] = burn_weight;

let remaining_weight = total_weight - burn_weight;  // 3277
```

---

#### Step 4: Category Weight Allocation

```rust
// A100 allocation (45% of remaining)
let a100_allocation = (remaining_weight as f64 * 0.45) as u16;  // 1475

// H100 allocation (55% of remaining)
let h100_allocation = (remaining_weight as f64 * 0.55) as u16;  // 1802
```

If a category has no active rentals, its allocation is added to burn.

---

#### Step 5: Miner Weights within Category

**H100 Category** (total revenue: $1200):

```rust
// Miner 5: $800 / $1200 = 0.667
let miner_5_h100_weight = (h100_allocation as f64 * 0.667) as u16;  // 1202

// Miner 12: $400 / $1200 = 0.333
let miner_12_h100_weight = (h100_allocation as f64 * 0.333) as u16;  // 600
```

**A100 Category** (total revenue: $900):

```rust
// Miner 12: $600 / $900 = 0.667
let miner_12_a100_weight = (a100_allocation as f64 * 0.667) as u16;  // 984

// Miner 23: $300 / $900 = 0.333
let miner_23_a100_weight = (a100_allocation as f64 * 0.333) as u16;  // 491
```

---

#### Step 6: Final Weight Vector

**Aggregate weights for miners with multiple categories**:

```rust
// Miner 5 (H100 only)
weights[5] = 1202

// Miner 12 (H100 + A100)
weights[12] = miner_12_h100_weight + miner_12_a100_weight  // 600 + 984 = 1584

// Miner 23 (A100 only)
weights[23] = 491

// Burn UID
weights[204] = 62258
```

**Final Weight Vector**:

```
UID   | Weight | Percentage | Revenue Source
------|--------|------------|------------------
5     | 1202   | 1.8%       | H100 rentals
12    | 1584   | 2.4%       | H100 + A100 rentals
23    | 491    | 0.7%       | A100 rentals
204   | 62258  | 95.0%      | BURN
------|--------|------------|------------------
Total | 65535  | 100.0%     |
```

---

### Weight Setting Frequency

**Block-Based Timing** (`bittensor_core/weight_setter.rs:80-120`):

```rust
// Check current block every 12 seconds
loop {
    let current_block = subtensor.get_current_block().await?;

    // Check if time to set weights
    let blocks_since_last = current_block - last_weight_set_block;

    if blocks_since_last >= config.weight_set_interval_blocks {
        // Time to set weights
        set_weights().await?;
        last_weight_set_block = current_block;
    }

    sleep(Duration::from_secs(12)).await;  // Bittensor block time
}
```

**Configuration**:

```toml
[emission]
weight_set_interval_blocks = 360  # ~72 minutes (360 * 12 seconds)
```

**Example Weight Setting Log**:

```
2024-01-01T12:00:00Z INFO  Current block: 1234567
2024-01-01T12:00:00Z INFO  Last weight set: block 1234207 (360 blocks ago)
2024-01-01T12:00:00Z INFO  Triggering weight set operation
2024-01-01T12:00:05Z INFO  Calculated weights for 52 miners
2024-01-01T12:00:05Z INFO  Burn allocation: 5.0% to UID 204
2024-01-01T12:00:10Z INFO  Submitted weights to chain (tx: 0xabcd...)
2024-01-01T12:00:15Z INFO  Weight set confirmed in block 1234568
```

---

### Emission Metrics Storage

**Database Tracking** (`persistence/emission_metrics.rs`):

```sql
-- Record emission event
INSERT INTO emission_metrics (
    timestamp,
    burn_amount,
    burn_percentage,
    category_distributions_json,
    total_miners,
    weight_set_block
) VALUES (
    NOW(),
    3277,
    5.0,
    '{"H100": 31129, "A100": 18677, "B200": 12452}',
    52,
    1234568
);

-- Record individual weight allocations
INSERT INTO weight_allocation_history (
    miner_uid,
    gpu_category,
    allocated_weight,
    miner_score,
    category_total_score,
    weight_set_block,
    emission_metrics_id
) VALUES
    (5, 'H100', 20987, 0.95, 11.28, 1234568, last_insert_id()),
    (12, 'H100', 10148, 0.92, 11.28, 1234568, last_insert_id()),
    (12, 'A100', 12644, 0.92, 21.76, 1234568, last_insert_id()),
    (23, 'A100', 6033, 0.88, 21.76, 1234568, last_insert_id()),
    (45, 'B200', 12452, 0.85, 1.7, 1234568, last_insert_id());
```

**Metrics Query Examples**:

```sql
-- Get emission history
SELECT
    timestamp,
    burn_percentage,
    total_miners,
    weight_set_block
FROM emission_metrics
ORDER BY timestamp DESC
LIMIT 10;

-- Get miner weight history
SELECT
    timestamp,
    gpu_category,
    allocated_weight,
    miner_score
FROM weight_allocation_history
WHERE miner_uid = 12
ORDER BY timestamp DESC
LIMIT 20;

-- Get category distribution over time
SELECT
    DATE(timestamp) as date,
    AVG(CAST(json_extract(category_distributions_json, '$.H100') AS REAL)) as avg_h100_weight,
    AVG(CAST(json_extract(category_distributions_json, '$.A100') AS REAL)) as avg_a100_weight
FROM emission_metrics
GROUP BY DATE(timestamp)
ORDER BY date DESC;
```

---

## Security & Best Practices

Critical security considerations and operational best practices for running a validator.

### SSH Security

#### Ephemeral Key Benefits

**Why Ephemeral Keys?**

1. **Limited Exposure**: Keys exist only during verification session
2. **Automatic Rotation**: New key generated for each verification
3. **No Long-Term Storage**: Validator doesn't store keys after use
4. **Miner Control**: Miner controls key lifetime on nodes

**Key Lifecycle**:

```
Generate → Send to Miner → Miner Deploys → Validation → Cleanup
  <1s         <1s           <5s           30-120s      <1s
```

#### SSH Audit Logging

**Enable Audit Logging**:

```toml
[ssh_session]
enable_audit_logging = true
audit_log_path = "/var/log/basilica/ssh_audit.log"
```

**Audit Log Analysis**:

```bash
# View recent SSH activity
tail -f /var/log/basilica/ssh_audit.log

# Count connections per node
grep "connected to" /var/log/basilica/ssh_audit.log | \
    awk '{print $6}' | sort | uniq -c | sort -rn

# Find failed connections
grep "connection failed" /var/log/basilica/ssh_audit.log

# Calculate average session duration
grep "disconnected from" /var/log/basilica/ssh_audit.log | \
    awk '{print $NF}' | sed 's/[^0-9]//g' | \
    awk '{sum+=$1; count++} END {print sum/count "s"}'
```

#### SSH Configuration Security

**Hardening SSH Sessions**:

```toml
[ssh_session]
# Connection timeouts prevent hanging connections
ssh_connection_timeout = { secs = 30, nanos = 0 }
ssh_command_timeout = { secs = 60, nanos = 0 }

# Retry logic for network issues
ssh_retry_attempts = 3
ssh_retry_delay = { secs = 2, nanos = 0 }

# Strict host key checking (false for dynamic nodes)
strict_host_key_checking = false

# Rate limiting prevents abuse
max_concurrent_sessions = 5
session_rate_limit = 20  # per hour
```

---

### Network Security

#### Firewall Configuration

**Minimal Firewall Rules**:

```bash
# Deny all incoming by default
sudo ufw default deny incoming
sudo ufw default allow outgoing

# Allow SSH for administration
sudo ufw allow 22/tcp

# Allow Bittensor axon
sudo ufw allow 9090/tcp

# Allow API server
sudo ufw allow 8080/tcp

# Allow metrics (if exposing externally)
sudo ufw allow 9090/tcp

# Enable firewall
sudo ufw enable

# Verify rules
sudo ufw status numbered
```

**Advanced Rules with Rate Limiting**:

```bash
# Rate limit SSH to prevent brute force
sudo ufw limit 22/tcp

# Rate limit API to prevent DoS
sudo ufw limit 8080/tcp

# Allow specific IP ranges for metrics
sudo ufw allow from 192.168.1.0/24 to any port 9090 proto tcp
```

#### DDoS Protection

**Application-Level Rate Limiting** (API server):

```rust
// Built into API server (code: api/mod.rs)
// - Request rate limiting per IP
// - Connection limits
// - Request size limits (1MB default)
```

**External Protection** (recommended for production):

- Use Cloudflare or similar CDN for API endpoints
- Deploy behind reverse proxy (nginx, traefik)
- Enable fail2ban for SSH protection

---

### Bittensor Security

#### Wallet Security

**Wallet Best Practices**:

1. **Separate Coldkey and Hotkey**:

   ```bash
   # Coldkey (high security, offline storage)
   ~/.bittensor/wallets/validator/coldkey

   # Hotkey (online, on validator server)
   ~/.bittensor/wallets/validator/hotkeys/default
   ```

2. **Coldkey Storage**:
   - Keep coldkey on offline, encrypted storage
   - Only use coldkey for staking/unstaking operations
   - Never store coldkey on validator server in production

3. **Hotkey Protection**:

   ```bash
   # Secure permissions
   chmod 600 ~/.bittensor/wallets/validator/hotkeys/default

   # Ownership
   chown validator:validator ~/.bittensor/wallets/validator/hotkeys/default
   ```

4. **Backup Strategy**:

   ```bash
   # Backup coldkey (offline storage)
   cp ~/.bittensor/wallets/validator/coldkey /secure/backup/location/

   # Backup hotkey (for disaster recovery)
   cp ~/.bittensor/wallets/validator/hotkeys/default /secure/backup/location/

   # Encrypt backups
   gpg --symmetric --cipher-algo AES256 /secure/backup/location/coldkey
   ```

#### Signature Verification

**Enabled by Default**:

- All gRPC requests from miners verified with Bittensor signatures
- Prevents impersonation attacks
- Uses sr25519 cryptography (Substrate standard)

**Verification Flow** (code: `crypto/core.rs:verify_bittensor_signature`):

```rust
// Verify signature on all incoming requests
let signature_valid = verify_bittensor_signature(
    &miner_hotkey,
    &signature_hex,
    &payload_bytes
)?;

if !signature_valid {
    return Err(Status::unauthenticated("Invalid signature"));
}
```

---

### Database Security

#### SQLite Security (Default)

**File Permissions**:

```bash
# Database file
chmod 600 data/validator.db

# Data directory
chmod 700 data/

# Ownership
chown validator:validator data/validator.db
```

**Backup Strategy**:

```bash
# Automated backups
*/15 * * * * sqlite3 /opt/basilica/data/validator.db ".backup '/opt/basilica/backups/validator_$(date +\%Y\%m\%d_\%H\%M\%S).db'"

# Retention policy (keep 7 days)
0 0 * * * find /opt/basilica/backups/ -name "validator_*.db" -mtime +7 -delete
```

#### PostgreSQL Security (Production)

**Connection Security**:

```toml
[database]
url = "postgresql://validator:STRONG_PASSWORD@localhost:5432/basilica?sslmode=require"
```

**PostgreSQL Configuration**:

```bash
# /etc/postgresql/15/main/postgresql.conf
ssl = on
ssl_cert_file = '/etc/ssl/certs/server.crt'
ssl_key_file = '/etc/ssl/private/server.key'

# /etc/postgresql/15/main/pg_hba.conf
# Require SSL connections
hostssl    basilica    validator    127.0.0.1/32    scram-sha-256
```

**User Privileges** (principle of least privilege):

```sql
-- Create validator database user
CREATE USER validator WITH PASSWORD 'STRONG_PASSWORD';

-- Grant only necessary privileges
GRANT CONNECT ON DATABASE basilica TO validator;
GRANT SELECT, INSERT, UPDATE, DELETE ON ALL TABLES IN SCHEMA public TO validator;
GRANT USAGE ON ALL SEQUENCES IN SCHEMA public TO validator;

-- Revoke dangerous privileges
REVOKE CREATE ON SCHEMA public FROM validator;
REVOKE DROP ON ALL TABLES IN SCHEMA public FROM validator;
```

---

### Monitoring and Alerting

#### Critical Metrics to Monitor

**Validator Health**:

```promql
# Up/down status
up{job="basilica-validator"}

# API response time
basilica_api_request_duration_seconds{quantile="0.95"}

# Verification success rate
rate(basilica_verification_success_total[5m]) / rate(basilica_verification_total[5m])
```

**Verification Performance**:

```promql
# Verifications per minute
rate(basilica_verification_total[1m]) * 60

# Average verification duration
rate(basilica_verification_duration_seconds_sum[5m]) / rate(basilica_verification_duration_seconds_count[5m])

# Failed verifications
rate(basilica_verification_failed_total[5m])
```

**Weight Setting**:

```promql
# Time since last weight set
time() - basilica_last_weight_set_timestamp

# Weight set errors
rate(basilica_weight_set_errors_total[1h])
```

#### Alerting Rules

**Example Prometheus Alerts** (`prometheus/alerts.yml`):

```yaml
groups:
  - name: basilica_validator
    interval: 30s
    rules:
      - alert: ValidatorDown
        expr: up{job="basilica-validator"} == 0
        for: 2m
        labels:
          severity: critical
        annotations:
          summary: "Validator is down"
          description: "Validator has been down for 2 minutes"

      - alert: VerificationFailureRate
        expr: |
          rate(basilica_verification_failed_total[5m]) /
          rate(basilica_verification_total[5m]) > 0.1
        for: 10m
        labels:
          severity: warning
        annotations:
          summary: "High verification failure rate"
          description: "More than 10% of verifications are failing"

      - alert: WeightSetStale
        expr: time() - basilica_last_weight_set_timestamp > 7200
        for: 5m
        labels:
          severity: critical
        annotations:
          summary: "Weights not set recently"
          description: "Weights haven't been set in over 2 hours"

      - alert: DatabaseErrors
        expr: rate(basilica_database_errors_total[5m]) > 0
        for: 5m
        labels:
          severity: warning
        annotations:
          summary: "Database errors detected"
          description: "Database is experiencing errors"
```

---

### Operational Best Practices

#### Regular Maintenance

**Daily Tasks**:

```bash
# Check validator health
curl http://localhost:8080/health

# Check recent verifications
journalctl -u basilica-validator --since "1 hour ago" | grep "Verification"

# Check database size
du -sh data/validator.db
```

**Weekly Tasks**:

```bash
# Review verification success rate
curl http://localhost:8080/verification/results | jq '.success_rate'

# Check disk space
df -h /opt/basilica

# Review error logs
journalctl -u basilica-validator -p err --since "1 week ago"

# Test backup restoration (on separate system)
cp backups/latest.db test/validator.db
sqlite3 test/validator.db "PRAGMA integrity_check;"
```

**Monthly Tasks**:

```bash
# Update validator software
git pull
./scripts/validator/build.sh
sudo systemctl restart basilica-validator

# Review and rotate logs
journalctl --vacuum-time=30d

# Database optimization (SQLite)
sqlite3 data/validator.db "VACUUM; ANALYZE;"

# Security audit
sudo apt update && sudo apt upgrade
sudo ufw status
```

#### Disaster Recovery Plan

**Backup Strategy**:

1. **Database**: Automated backups every 15 minutes, retain 7 days
2. **Configuration**: Version controlled, backed up daily
3. **Wallet**: Encrypted offline backup, multiple locations
4. **SSH Keys**: Ephemeral, no backup needed

**Recovery Procedure**:

1. **Server Failure**:

   ```bash
   # On new server
   git clone https://github.com/your-org/basilica.git
   ./scripts/validator/build.sh

   # Restore configuration
   cp backup/validator.toml /opt/basilica/config/

   # Restore wallet
   mkdir -p ~/.bittensor/wallets/validator/hotkeys/
   cp backup/hotkey ~/.bittensor/wallets/validator/hotkeys/default
   chmod 600 ~/.bittensor/wallets/validator/hotkeys/default

   # Restore database
   cp backup/validator.db /opt/basilica/data/

   # Start validator
   sudo systemctl start basilica-validator
   ```

2. **Database Corruption**:

   ```bash
   # Stop validator
   sudo systemctl stop basilica-validator

   # Restore from backup
   cp backups/validator_LATEST.db data/validator.db

   # Verify integrity
   sqlite3 data/validator.db "PRAGMA integrity_check;"

   # Restart validator
   sudo systemctl start basilica-validator
   ```

3. **Wallet Compromise**:

   ```bash
   # IMMEDIATE: Stop validator
   sudo systemctl stop basilica-validator

   # Create new hotkey
   btcli wallet new_hotkey --wallet.name validator --wallet.hotkey new_hotkey

   # Transfer stake to new hotkey
   btcli stake add --wallet.name validator --wallet.hotkey new_hotkey --amount ALL

   # Update configuration
   sed -i 's/hotkey_name = "default"/hotkey_name = "new_hotkey"/' config/validator.toml

   # Restart validator
   sudo systemctl start basilica-validator
   ```

---

## Monitoring

Comprehensive monitoring setup for validators using Prometheus and Grafana.

### Prometheus Metrics

**Built-in Metrics** (exported on port 9090 by default):

#### System Metrics

```promql
# CPU usage
basilica_cpu_usage_percent

# Memory usage
basilica_memory_usage_bytes
basilica_memory_available_bytes

# Disk usage
basilica_disk_usage_bytes{path="/opt/basilica/data"}
basilica_disk_available_bytes{path="/opt/basilica/data"}

# Network I/O
rate(basilica_network_received_bytes_total[1m])
rate(basilica_network_transmitted_bytes_total[1m])
```

#### Verification Metrics

```promql
# Total verifications
basilica_verification_total{type="full"}
basilica_verification_total{type="lightweight"}

# Verification success/failure
basilica_verification_success_total
basilica_verification_failed_total

# Verification duration
basilica_verification_duration_seconds{quantile="0.5"}
basilica_verification_duration_seconds{quantile="0.95"}
basilica_verification_duration_seconds{quantile="0.99"}

# Active verifications
basilica_verification_active

# Miner discovery
basilica_miners_discovered_total
basilica_miners_verified_total
```

#### Weight Setting Metrics

```promql
# Weight set operations
basilica_weight_set_total
basilica_weight_set_errors_total

# Last weight set timestamp
basilica_last_weight_set_timestamp

# Weight set duration
basilica_weight_set_duration_seconds
```

#### API Metrics

```promql
# HTTP requests
basilica_http_requests_total{method="GET",path="/miners"}
basilica_http_requests_total{status="200"}

# Request duration
basilica_http_request_duration_seconds{quantile="0.95"}

# Active connections
basilica_http_connections_active
```

#### Database Metrics

```promql
# Database connections
basilica_database_connections_active
basilica_database_connections_idle

# Query duration
basilica_database_query_duration_seconds{operation="select"}
basilica_database_query_duration_seconds{operation="insert"}

# Database size
basilica_database_size_bytes
```

### Grafana Dashboard

**Import Pre-built Dashboard**:

1. Access Grafana: `http://localhost:3000`
2. Login: admin / admin
3. Navigate: Dashboards → Import
4. Upload: `grafana/dashboards/validator.json`

**Key Panels**:

1. **Overview**:
   - Validator uptime
   - Total miners discovered
   - Verification success rate
   - Last weight set time

2. **Verification Performance**:
   - Verifications per minute (graph)
   - Verification duration (heatmap)
   - Success/failure ratio (gauge)
   - Active verifications (graph)

3. **Resource Usage**:
   - CPU usage (graph)
   - Memory usage (graph)
   - Disk usage (graph)
   - Network I/O (graph)

4. **API Performance**:
   - Requests per second (graph)
   - Request duration (heatmap)
   - Error rate (graph)
   - Active connections (graph)

5. **Weight Setting**:
   - Weight set history (table)
   - Burn percentage (stat)
   - Category allocations (pie chart)
   - Weight set duration (graph)

**Example Dashboard JSON** (abbreviated):

```json
{
  "dashboard": {
    "title": "Basilica Validator",
    "panels": [
      {
        "title": "Verification Success Rate",
        "targets": [
          {
            "expr": "rate(basilica_verification_success_total[5m]) / rate(basilica_verification_total[5m]) * 100"
          }
        ],
        "type": "gauge"
      },
      {
        "title": "Verifications per Minute",
        "targets": [
          {
            "expr": "rate(basilica_verification_total[1m]) * 60",
            "legendFormat": "{{type}}"
          }
        ],
        "type": "graph"
      }
    ]
  }
}
```

### Health Checks

**HTTP Health Endpoint**:

```bash
# Basic health check
curl http://localhost:8080/health

# Response (healthy)
{
  "status": "healthy",
  "timestamp": "2024-01-01T12:00:00Z",
  "checks": {
    "database": "ok",
    "bittensor": "ok",
    "verifications": "ok"
  },
  "metrics": {
    "miners_discovered": 47,
    "verifications_active": 12,
    "last_weight_set": "2024-01-01T11:00:00Z"
  }
}

# Response (unhealthy)
{
  "status": "unhealthy",
  "timestamp": "2024-01-01T12:00:00Z",
  "checks": {
    "database": "ok",
    "bittensor": "error",
    "verifications": "ok"
  },
  "errors": [
    "Cannot connect to Bittensor chain"
  ]
}
```

**Automated Health Monitoring**:

```bash
# Add to cron for periodic checks
*/5 * * * * curl -f http://localhost:8080/health || echo "Validator unhealthy!" | mail -s "Validator Alert" admin@example.com
```

**Docker Health Check**:

```yaml
# In docker-compose.yml
healthcheck:
  test: ["CMD", "curl", "-f", "http://localhost:8080/health"]
  interval: 30s
  timeout: 10s
  retries: 3
  start_period: 40s
```

### Log Analysis

**Structured Logging with `tracing`**:

**Log Levels**:

- `TRACE`: Very verbose, debugging internals
- `DEBUG`: Detailed operational information
- `INFO`: General operational information
- `WARN`: Warning conditions
- `ERROR`: Error conditions

**Key Log Patterns**:

```bash
# Verification activity
journalctl -u basilica-validator | grep "Verification"

# Weight setting events
journalctl -u basilica-validator | grep "Weight set"

# Error tracking
journalctl -u basilica-validator -p err

# Performance analysis (verification duration)
journalctl -u basilica-validator | grep "Validation complete" | \
    awk '{print $(NF-1)}' | sed 's/[^0-9.]//g' | \
    awk '{sum+=$1; count++} END {print "Average:", sum/count "s"}'

# Miner discovery trends
journalctl -u basilica-validator --since "24 hours ago" | \
    grep "Discovered.*miners" | \
    awk '{print $1, $2, $NF}' | sed 's/miners//'
```

**Centralized Logging** (optional):

```bash
# Ship logs to external service (e.g., Loki, Elasticsearch)

# Example: Promtail for Grafana Loki
# /etc/promtail/config.yml
server:
  http_listen_port: 9080

positions:
  filename: /tmp/positions.yaml

clients:
  - url: http://loki:3100/loki/api/v1/push

scrape_configs:
  - job_name: basilica-validator
    journal:
      json: false
      max_age: 12h
      labels:
        job: systemd-journal
        service: basilica-validator
    relabel_configs:
      - source_labels: ['__journal__systemd_unit']
        target_label: 'unit'
```

---

## Troubleshooting

Common issues and solutions for validator operation.

### Issue 1: Validator Not Discovering Miners

**Symptoms**:

```
2024-01-01T12:00:00Z INFO  Discovered 0 miners from metagraph
```

**Possible Causes and Solutions**:

1. **Wrong Netuid**:

   ```bash
   # Check configuration
   grep "netuid" config/validator.toml

   # Should be:
   # [bittensor] netuid = 39 (mainnet)
   # [verification] netuid = 39

   # Verify with Bittensor CLI
   btcli metagraph --netuid 39 | grep "MINER"
   ```

2. **Chain Connection Issues**:

   ```bash
   # Test chain connectivity
   curl -v wss://entrypoint-finney.opentensor.ai:443

   # Check logs for connection errors
   journalctl -u basilica-validator | grep "chain"

   # Try alternative chain endpoint
   # Edit config/validator.toml:
   # chain_endpoint = "wss://test.finney.opentensor.ai:443"
   ```

3. **Wallet Not Registered**:

   ```bash
   # Check wallet registration
   btcli wallet overview --wallet.name validator --wallet.hotkey default

   # Re-register if needed
   btcli subnet register --netuid 39 --wallet.name validator --wallet.hotkey default
   ```

---

### Issue 2: SSH Verification Failures

**Symptoms**:

```
2024-01-01T12:00:10Z ERROR [Full] SSH connection failed: Connection refused (192.168.1.100:22)
```

**Possible Causes and Solutions**:

1. **Miner Not Deploying SSH Keys**:

   ```bash
   # Check if validator's SSH key was provided in auth request
   journalctl -u basilica-validator | grep "ssh_public_key"

   # Verify ephemeral key generation
   ls -la /tmp/validator_ssh_keys/

   # Test manual SSH (won't work if key not deployed)
   ssh -i /tmp/validator_ssh_keys/latest.pem basilica@192.168.1.100
   ```

2. **Network Connectivity**:

   ```bash
   # Test network reachability
   ping 192.168.1.100

   # Test SSH port
   nc -zv 192.168.1.100 22

   # Traceroute to identify network issues
   traceroute 192.168.1.100
   ```

3. **Firewall Blocking Outbound SSH**:

   ```bash
   # Check if outbound SSH is allowed
   sudo ufw status | grep 22

   # Allow outbound SSH if blocked
   sudo ufw allow out 22/tcp
   ```

4. **SSH Timeout Too Short**:

   ```toml
   # Increase timeout in config/validator.toml
   [ssh_session]
   ssh_connection_timeout = { secs = 60, nanos = 0 }
   ssh_command_timeout = { secs = 120, nanos = 0 }
   ```

---

### Issue 3: Weight Setting Failing

**Symptoms**:

```
2024-01-01T12:00:00Z ERROR Failed to set weights: Insufficient stake
```

**Possible Causes and Solutions**:

1. **Insufficient Stake for Validator Permit**:

   ```bash
   # Check current stake
   btcli wallet overview --wallet.name validator

   # Check minimum required stake
   btcli metagraph --netuid 39 | grep "VALIDATOR" | head -1

   # Add more stake
   btcli stake add --wallet.name validator --wallet.hotkey default --amount 5000
   ```

2. **Weight Vector Invalid**:

   ```bash
   # Check logs for specific error
   journalctl -u basilica-validator | grep "set weights"

   # Common issues:
   # - Weights don't sum to 65535 (u16::MAX)
   # - Invalid UIDs in weight vector
   # - Empty weight vector

   # Verify GPU allocations total 100%
   grep "gpu_allocations" config/validator.toml -A 10
   ```

3. **Chain Transaction Failure**:

   ```bash
   # Check for chain errors
   journalctl -u basilica-validator | grep -i "transaction\|extrinsic"

   # Possible solutions:
   # - Wait for next block and retry
   # - Check chain health: https://telemetry.polkadot.io/
   # - Verify wallet has sufficient funds for transaction fees
   ```

---

### Issue 4: High Memory Usage

**Symptoms**:

```bash
# Memory usage over 90%
free -h
#               total        used        free
# Mem:           15Gi        14Gi       512Mi
```

**Possible Causes and Solutions**:

1. **Database Cache Too Large**:

   ```toml
   # Reduce database connections
   [database]
   max_connections = 5  # Reduce from 10
   min_connections = 1
   ```

2. **Too Many Concurrent Verifications**:

   ```toml
   # Reduce concurrency limits
   [verification]
   max_concurrent_verifications = 20  # Reduce from 50
   max_concurrent_full_validations = 512  # Reduce from 1024
   ```

3. **Memory Leak (rare)**:

   ```bash
   # Restart validator
   sudo systemctl restart basilica-validator

   # Monitor memory over time
   watch -n 5 free -h

   # If leak persists, report issue with logs
   ```

---

### Issue 5: Database Errors

**Symptoms**:

```
2024-01-01T12:00:00Z ERROR Database error: database is locked
```

**Possible Causes and Solutions**:

1. **SQLite Lock Contention**:

   ```bash
   # Check for other processes accessing database
   lsof data/validator.db

   # If using SQLite in production, consider PostgreSQL
   # SQLite not optimal for high concurrency
   ```

2. **Database Corruption**:

   ```bash
   # Check integrity
   sqlite3 data/validator.db "PRAGMA integrity_check;"

   # If corrupted, restore from backup
   sudo systemctl stop basilica-validator
   cp backups/validator_LATEST.db data/validator.db
   sudo systemctl start basilica-validator
   ```

3. **Disk Full**:

   ```bash
   # Check disk space
   df -h /opt/basilica

   # Clean up old logs
   journalctl --vacuum-time=7d

   # Enable database cleanup
   # In config/validator.toml:
   [cleanup]
   enabled = true
   ```

---

### Issue 6: API Not Responding

**Symptoms**:

```bash
curl http://localhost:8080/health
# curl: (7) Failed to connect to localhost port 8080: Connection refused
```

**Possible Causes and Solutions**:

1. **API Server Not Started**:

   ```bash
   # Check if validator is running
   sudo systemctl status basilica-validator

   # Check logs for API startup
   journalctl -u basilica-validator | grep "API server listening"

   # Should see: "API server listening on 0.0.0.0:8080"
   ```

2. **Port Already in Use**:

   ```bash
   # Check what's using port 8080
   sudo lsof -i :8080

   # Change API port in config
   [api]
   bind_address = "0.0.0.0:8081"
   ```

3. **Firewall Blocking**:

   ```bash
   # Check firewall rules
   sudo ufw status | grep 8080

   # Allow port if blocked
   sudo ufw allow 8080/tcp
   ```

---

### Issue 7: Verification Taking Too Long

**Symptoms**:

```
2024-01-01T12:00:00Z WARN Verification timeout: node 550e8400 exceeded 120s
```

**Possible Causes and Solutions**:

1. **Slow Network Connection**:

   ```bash
   # Test network speed to node
   iperf3 -c 192.168.1.100

   # Increase timeouts
   [verification]
   challenge_timeout = { secs = 300, nanos = 0 }  # Increase from 120s

   [ssh_session]
   ssh_command_timeout = { secs = 300, nanos = 0 }
   ```

2. **Binary Execution Slow on Node**:

   ```bash
   # Note: Binary-related troubleshooting excluded per user request
   # Check if node has sufficient resources
   # May need to exclude slow nodes from verification
   ```

3. **Too Many Verifications in Parallel**:

   ```toml
   # Reduce concurrency to avoid resource exhaustion
   [verification]
   max_concurrent_verifications = 25
   max_concurrent_full_validations = 256
   ```

---

### Issue 8: Metrics Not Showing in Prometheus

**Symptoms**:

- Grafana shows "No data"
- Prometheus shows target as "Down"

**Possible Causes and Solutions**:

1. **Metrics Not Enabled**:

   ```toml
   # Enable in config/validator.toml
   [metrics]
   enabled = true

   [metrics.prometheus]
   enabled = true
   port = 9090
   ```

2. **Prometheus Not Scraping**:

   ```bash
   # Check Prometheus targets
   curl http://localhost:9091/targets

   # Verify Prometheus config
   cat scripts/validator/prometheus.yml

   # Should have:
   scrape_configs:
     - job_name: 'basilica-validator'
       static_configs:
         - targets: ['validator:9090']
   ```

3. **Network Issue Between Prometheus and Validator**:

   ```bash
   # Test from Prometheus container
   docker exec basilica-prometheus curl http://validator:9090/metrics

   # Should return metrics output
   ```

---

### Issue 9: Wallet Not Found

**Symptoms**:

```
2024-01-01T12:00:00Z ERROR Wallet not found: ~/.bittensor/wallets/validator/hotkeys/default
```

**Possible Causes and Solutions**:

1. **Wrong Wallet Path**:

   ```bash
   # Check actual wallet location
   ls -la ~/.bittensor/wallets/

   # Update config to match
   [bittensor]
   wallet_name = "actual_wallet_name"
   hotkey_name = "actual_hotkey_name"
   ```

2. **Wallet Not Mounted (Docker)**:

   ```bash
   # Verify volume mount
   docker inspect basilica-validator | grep Mounts -A 20

   # Should show:
   # "Source": "/home/user/.bittensor/wallets",
   # "Destination": "/root/.bittensor/wallets"

   # Recreate container with correct mount
   docker rm basilica-validator
   docker run -v ~/.bittensor/wallets:/root/.bittensor/wallets:ro ...
   ```

3. **Permissions Issue**:

   ```bash
   # Check file permissions
   ls -la ~/.bittensor/wallets/validator/hotkeys/default

   # Fix if needed
   chmod 600 ~/.bittensor/wallets/validator/hotkeys/default
   chown $USER:$USER ~/.bittensor/wallets/validator/hotkeys/default
   ```

---

## Advanced Topics

Advanced configurations and optimizations for experienced operators.

### Custom Verification Strategies

**Implement Custom Strategy Selection**:

Current validator uses time-based strategy (6 hours). For advanced use cases, you can modify strategy selection logic.

**Example: Score-Based Strategy**:

```rust
// Pseudocode for custom strategy
fn determine_strategy_custom(node: &Node, history: &VerificationHistory) -> Strategy {
    // Use full validation for low-scoring nodes more frequently
    if history.average_score < 0.7 {
        // High-risk nodes: validate every 2 hours
        if now - history.last_validation > Duration::from_secs(2 * 3600) {
            return Strategy::Full;
        }
    }

    // Use lightweight for high-performing nodes
    if history.average_score > 0.95 && history.consecutive_successes > 10 {
        // Low-risk nodes: validate every 12 hours
        if now - history.last_validation > Duration::from_secs(12 * 3600) {
            return Strategy::Full;
        }
    }

    // Default: 6 hours
    if now - history.last_validation > Duration::from_secs(6 * 3600) {
        return Strategy::Full;
    }

    Strategy::Lightweight
}
```

**Configuration Location**: `crates/basilica-validator/src/miner_prover/validation_strategy.rs`

### High Availability Setup

**Multi-Validator Architecture**:

```
        ┌─────────────────┐
        │  Load Balancer  │
        │   (HAProxy)     │
        └────────┬────────┘
                 │
        ┌────────┴────────┐
        │                 │
  ┌─────▼─────┐     ┌────▼──────┐
  │Validator 1│     │Validator 2│
  │ (Active)  │     │ (Standby) │
  └─────┬─────┘     └────┬──────┘
        │                 │
        └────────┬────────┘
                 │
        ┌────────▼────────┐
        │   PostgreSQL    │
        │  (Shared DB)    │
        └─────────────────┘
```

**Configuration**:

1. **Shared Database** (required):

   ```toml
   # Both validators use same PostgreSQL database
   [database]
   url = "postgresql://validator:pass@db-server:5432/basilica"
   ```

2. **Distributed Locking** (prevent duplicate work):

   ```rust
   // Implemented in: crates/basilica-common/src/distributed/postgres_lock.rs
   // Ensures only one validator performs verification/weight-setting at a time
   ```

3. **Load Balancer Configuration** (HAProxy example):

   ```
   frontend validator_api
       bind *:8080
       default_backend validator_servers

   backend validator_servers
       balance roundrobin
       option httpchk GET /health
       server validator1 192.168.1.10:8080 check
       server validator2 192.168.1.11:8080 check fall 3 rise 2
   ```

**Failover Behavior**:

- Active validator performs verifications and weight setting
- Standby monitors health via database heartbeat
- On active failure, standby takes over within 30 seconds
- Both validators can serve API requests (load balanced)

### Performance Tuning

#### Database Optimization

**SQLite Tuning** (for moderate load):

```sql
-- Add to database initialization
PRAGMA journal_mode = WAL;           -- Write-Ahead Logging for concurrency
PRAGMA synchronous = NORMAL;         -- Balance safety and performance
PRAGMA cache_size = -64000;          -- 64MB cache
PRAGMA temp_store = MEMORY;          -- Temp tables in memory
PRAGMA mmap_size = 30000000000;      -- 30GB memory-mapped I/O
```

**PostgreSQL Tuning** (for high load):

```sql
-- /etc/postgresql/15/main/postgresql.conf

# Memory settings (adjust based on available RAM)
shared_buffers = 4GB
effective_cache_size = 12GB
maintenance_work_mem = 1GB
work_mem = 50MB

# Checkpointing
checkpoint_completion_target = 0.9
wal_buffers = 16MB
max_wal_size = 4GB

# Query planner
random_page_cost = 1.1  # For SSD storage
effective_io_concurrency = 200

# Parallelism
max_parallel_workers_per_gather = 4
max_parallel_workers = 8
max_worker_processes = 8

# Connection pooling (if using PgBouncer)
# Use transaction pooling mode for best performance
```

#### Verification Throughput Optimization

**Maximize Concurrent Verifications**:

```toml
[verification]
# Increase concurrent lightweight checks (low resource cost)
max_concurrent_verifications = 100

# Increase concurrent full validations (higher resource cost)
max_concurrent_full_validations = 2048

# Increase miners per round
max_miners_per_round = 50

# Reduce verification interval for faster discovery
verification_interval = { secs = 300, nanos = 0 }  # 5 minutes
```

**Worker Queue for Horizontal Scaling** (experimental):

```toml
[verification]
# Enable worker queue for distributed execution
enable_worker_queue = true
```

**Requirements**:

- Redis instance for queue management
- Multiple validator workers (separate processes/servers)
- Shared PostgreSQL database

**Architecture**:

```
Validator (Scheduler) → Redis Queue → Worker 1, Worker 2, Worker N
                                              ↓
                                        PostgreSQL (Shared)
```

#### Network Optimization

**Connection Pooling**:

```toml
[database]
# Optimize connection pool
max_connections = 20       # Increase from 10
min_connections = 5        # Increase from 1

[database.connect_timeout]
secs = 10                  # Reduce from 30 for faster failures
```

**SSH Connection Reuse**:

Current implementation creates new SSH connection per verification. For optimization, implement connection pooling:

```rust
// Pseudocode for SSH connection pool
struct SshConnectionPool {
    pools: HashMap<String, Vec<SshConnection>>,
    max_per_host: usize,
}

impl SshConnectionPool {
    fn get_connection(&mut self, host: &str) -> Result<SshConnection> {
        if let Some(conn) = self.pools.get_mut(host).and_then(|p| p.pop()) {
            if conn.is_alive() {
                return Ok(conn);
            }
        }

        // Create new connection if pool empty
        SshConnection::new(host)
    }

    fn return_connection(&mut self, host: String, conn: SshConnection) {
        self.pools.entry(host).or_default().push(conn);
    }
}
```

**Implementation Location**: `crates/basilica-validator/src/ssh/connection_pool.rs` (to be implemented)

### Custom GPU Categories

**Add New GPU Category**:

```toml
[emission.gpu_allocations]
# Existing categories
H100 = { weight = 35.0, min_gpu_count = 4, min_gpu_vram = 80 }
A100 = { weight = 25.0, min_gpu_count = 2, min_gpu_vram = 40 }
B200 = { weight = 25.0, min_gpu_count = 1, min_gpu_vram = 192 }
H200 = { weight = 10.0, min_gpu_count = 1, min_gpu_vram = 141 }

# Add new category (e.g., L40S)
L40S = { weight = 5.0, min_gpu_count = 2, min_gpu_vram = 48 }
```

**GPU Name Matching** (code: `scoring/gpu_categorization.rs`):

```rust
// GPU models are matched by string prefix
// Example: "NVIDIA H100 PCIe" matches category "H100"

fn categorize_gpu(model: &str) -> Option<String> {
    if model.contains("H100") {
        Some("H100".to_string())
    } else if model.contains("A100") {
        Some("A100".to_string())
    } else if model.contains("B200") {
        Some("B200".to_string())
    } else if model.contains("H200") {
        Some("H200".to_string())
    } else if model.contains("L40S") {
        Some("L40S".to_string())
    } else {
        None  // Unknown GPU, not eligible for weights
    }
}
```

**Weight Rebalancing**:

- Ensure all `gpu_allocations` weights sum to 100.0
- Weights are percentages of total emissions (after burn)
- Higher weight = more emissions allocated to that category

### API Authentication

**Enable API Key Authentication**:

```toml
[api]
api_key = "your-secret-api-key-here"
```

**Usage**:

```bash
# Without API key (fails if enabled)
curl http://localhost:8080/miners
# Response: 401 Unauthorized

# With API key
curl -H "X-API-Key: your-secret-api-key-here" http://localhost:8080/miners
# Response: 200 OK with miner data
```

**Advanced: JWT Authentication** (code: `api/auth.rs`):

For external service integration, implement JWT-based authentication:

```rust
// Pseudocode for JWT auth
struct JwtAuth {
    secret: String,
    issuer: String,
    audience: String,
}

impl JwtAuth {
    fn validate_token(&self, token: &str) -> Result<Claims> {
        let validation = Validation {
            iss: Some(self.issuer.clone()),
            aud: Some(self.audience.clone()),
            ..Default::default()
        };

        decode::<Claims>(token, &DecodingKey::from_secret(self.secret.as_ref()), &validation)
            .map(|data| data.claims)
    }
}
```

**Integration with Auth0** (constants in `common/src/auth_constants.rs`):

```rust
// Auth0 configuration
pub const AUTH0_DOMAIN: &str = env!("AUTH0_DOMAIN");
pub const AUTH0_CLIENT_ID: &str = env!("AUTH0_CLIENT_ID");
pub const AUTH0_AUDIENCE: &str = env!("AUTH0_AUDIENCE");
```

---

## Summary

This guide covered comprehensive validator operation from setup to advanced optimization. Key takeaways:

**Core Responsibilities**:

- Discover miners from Bittensor metagraph
- Verify GPU nodes via direct SSH access
- Score miners based on performance and reliability
- Distribute emissions via weight setting

**Two-Tier Verification**:

- Full validation: Binary execution + hardware profiling (every 6 hours)
- Lightweight validation: SSH accessibility check (every 10 minutes)

**Weight Setting**:

- GPU category-based allocation (H100, A100, B200, etc.)
- Score-weighted distribution within categories
- Configurable burn mechanism
- Block-based timing (default: every 360 blocks)

**Security**:

- Ephemeral SSH keys for verification
- Cryptographic authentication with miners
- Audit logging for all SSH operations
- Wallet security best practices

**Deployment**:

- Four methods: Binary, Systemd, Docker, Docker Compose
- Automated deployment scripts
- Production-ready monitoring with Prometheus/Grafana
- Health checks and disaster recovery

**Monitoring**:

- Comprehensive Prometheus metrics
- Pre-built Grafana dashboards
- Alerting rules for critical conditions
- Log analysis and troubleshooting

**Advanced Topics**:

- High availability setup with load balancing
- Performance tuning for database and network
- Custom verification strategies
- API authentication and external integration

For additional support, refer to specific sections or consult the codebase at `/root/workspace/spacejar/basilica/basilica/crates/basilica-validator/`.
