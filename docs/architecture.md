# Architecture Guide

This guide provides a comprehensive overview of Basilica's system architecture and design principles.

## System Overview

Basilica is a decentralized GPU compute platform built on the Bittensor network. It creates a trustless environment where GPU providers (miners) can offer compute resources, and validators ensure quality and reliability through **hardware verification**.

## Core Components

### 1. Validator

The validator is the quality assurance layer of the network:

- **Miner Discovery**: Discovers miners from Bittensor metagraph
- **SSH-Based Verification**: Direct SSH access to GPU nodes for verification
- **Ephemeral Key Management**: Generates short-lived SSH keys for each verification session
- **GPU Profile Management**: Maintains GPU performance profiles and benchmarks
- **Scoring System**: Calculates performance metrics using GPU categorization
- **Weight Setter**: Updates Bittensor network weights based on miner performance
- **REST API**: Provides external access to validation data and rentals
- **PostgreSQL/SQLite Storage**: Persists verification history, GPU profiles, and miner scores

**Key Files**:

- `crates/basilica-validator/src/miner_prover/` - Verification orchestration
- `crates/basilica-validator/src/bittensor_core/` - Weight setting
- `crates/basilica-validator/src/api/` - REST API
- `crates/basilica-validator/src/ssh/` - SSH session management

### 2. Miner

The miner acts as an SSH access orchestrator for GPU nodes:

- **Node Fleet Manager**: Manages configuration of multiple GPU nodes
- **Axon Server**: Serves on the Bittensor network for discovery
- **gRPC Server**: Provides validator authentication and node discovery API
- **SSH Key Deployment**: Automatically deploys validator SSH keys to nodes
- **Validator Assignment**: Routes validators to nodes based on stake-weighted strategies
- **Session Management**: Controls validator access duration and cleanup
- **SQLite Storage**: Persists node info, validator sessions, and assignments

**Key Files**:

- `crates/basilica-miner/src/node_manager.rs` - Node SSH orchestration
- `crates/basilica-miner/src/validator_comms.rs` - gRPC server for validators
- `crates/basilica-miner/src/validator_assignment.rs` - Validator routing logic

### 3. GPU Nodes

GPU nodes are standard servers with SSH access (no special software):

- **SSH Server**: Standard OpenSSH daemon
- **Docker Runtime**: NVIDIA Container Toolkit for GPU access
- **CUDA Drivers**: NVIDIA CUDA ≥12.8
- **Storage**: 1TB+ available disk space
- **NVIDIA GPUs**: Any model (A100, H100, B200, etc.)

### 4. Basilica API Gateway (Optional)

The smart HTTP gateway for external services:

- **Validator Discovery**: Automatic discovery of validators using Bittensor metagraph
- **Load Balancing**: Multiple strategies for distributing requests across validators
- **Request Aggregation**: Combines responses from multiple validators
- **Authentication**: API key and JWT-based authentication
- **Rate Limiting**: Configurable rate limits with different tiers
- **Caching**: Response caching with in-memory (Moka) or Redis backends
- **OpenAPI Documentation**: Auto-generated API documentation with Swagger UI

**Key Files**:

- `crates/basilica-api/` - Gateway implementation

## System Architecture

### Direct SSH Access Model

```text
┌─────────────────────────────────────────────────────────────┐
│                   BITTENSOR NETWORK                         │
│                     (Subnet Metagraph)                      │
└────────────────────────┬────────────────────────────────────┘
                         │
        ┌────────────────┼────────────────┐
        │                │                │
┌───────▼──────┐ ┌──────▼──────┐ ┌──────▼──────┐
│  VALIDATOR   │ │    MINER    │ │BASILICA API │
│              │ │             │ │  GATEWAY    │
│ ┌──────────┐ │ │ ┌─────────┐ │ │ ┌─────────┐ │
│ │ Miner    │ │ │ │  Axon   │ │ │ │  Load   │ │
│ │Discovery │ │ │ │ Server  │ │ │ │Balancer │ │
│ └──────────┘ │ │ └─────────┘ │ │ └─────────┘ │
│ ┌──────────┐ │ │ ┌─────────┐ │ │ ┌─────────┐ │
│ │Verif.    │ │ │ │ Node    │ │ │ │  Cache  │ │
│ │Engine    │ │ │ │ Manager │ │ │ │  Layer  │ │
│ └──────────┘ │ │ └─────────┘ │ │ └─────────┘ │
│ ┌──────────┐ │ │ ┌─────────┐ │ │ ┌─────────┐ │
│ │ Weight   │ │ │ │  gRPC   │ │ │ │  Auth   │ │
│ │ Setter   │ │ │ │ Server  │ │ │ │ Manager │ │
│ └──────────┘ │ │ └─────────┘ │ │ └─────────┘ │
│ ┌──────────┐ │ │ ┌─────────┐ │ │             │
│ │   SSH    │ │ │ │   SSH   │ │ │             │
│ │  Client  │ │ │ │  Client │ │ │             │
│ └──────────┘ │ │ └─────────┘ │ │             │
└──────┬───────┘ └──────┬──────┘ └─────────────┘
       │                │
       │ Direct SSH     │ SSH Key Deployment
       │                │
┌──────▼────────────────▼────────────────────────┐
│              GPU NODES (SSH Endpoints)         │
│  ┌──────────┐  ┌──────────┐  ┌──────────┐      │
│  │  Node 1  │  │  Node 2  │  │  Node N  │      │
│  │          │  │          │  │          │      │
│  │ SSH      │  │ SSH      │  │ SSH      │      │
│  │ Docker   │  │ Docker   │  │ Docker   │      │
│  │ NVIDIA   │  │ NVIDIA   │  │ NVIDIA   │      │
│  │  H100    │  │  A100    │  │  B200    │      │
│  └──────────┘  └──────────┘  └──────────┘      │
└────────────────────────────────────────────────┘
```

### Verification Flow Diagram

```text
┌──────────────┐                           ┌──────────────┐
│  VALIDATOR   │                           │    MINER     │
└──────┬───────┘                           └──────┬───────┘
       │                                          │
       │ 1. Query Metagraph                       │
       ├──────────────────────────────────────┐   │
       │  (Discover miners on subnet)         │   │
       │◄─────────────────────────────────────┘   │
       │                                          │
       │ 2. gRPC: Authenticate                    │
       │    (Send SSH public key + signature)     │
       ├─────────────────────────────────────────►│
       │                                          │
       │                                          │ 3. Deploy SSH key
       │                                          │    to all nodes
       │                                          ├───────────┐
       │                                          │           │
       │◄─────────────────────────────────────────┤           │
       │ 4. Return node SSH details               │◄──────────┘
       │    (username@host:port)                  │
       │                                          │
       │                                          │
       ├──────────────────────────────────────────┼───────────┐
       │ 5. SSH directly to node                              │
       │    (using ephemeral key)                             │
       │                                                      ▼
       │                                            ┌──────────────────┐
       │                                            │    GPU NODE      │
       │                                            │  (SSH Endpoint)  │
       │                                            └──────┬───────────┘
       │                                                   │
       │ 6. Execute verification                           │
       │    (upload binaries, run validation)              │
       ├──────────────────────────────────────────────────►│
       │                                                   │
       │◄──────────────────────────────────────────────────┤
       │ 7. Download results                               │
       │                                                   │
       │                                          ┌────────▼───────┐
       │ 8. Cleanup SSH key                       │                │
       │    (after session expiry)                │                │
       ├─────────────────────────────────────────►│                │
       │                                          └────────────────┘
       │
       │ 9. Store scores & Set weights
       ├───────────────────────────────────────► Bittensor Chain
       │
```

## Communication Protocols

### 1. Bittensor Protocol

- **Blockchain Integration**: Substrate-based chain communication
- **Metagraph Queries**: Discover miners and validators on subnet
- **Weight Updates**: On-chain consensus mechanism for emissions
- **Hotkey Verification**: Sr25519 signature verification

### 2. gRPC Protocol

**Used for**: Validator ↔ Miner communication

**Services** (defined in `crates/basilica-protocol/proto/`):

```protobuf
service MinerDiscovery {
  // Validator authenticates with miner
  rpc AuthenticateValidator(ValidatorAuthRequest) returns (MinerAuthResponse);

  // Validator discovers nodes from miner
  rpc DiscoverNodes(DiscoverNodesRequest) returns (stream NodeConnectionDetails);
}
```

**Key Messages**:

- `ValidatorAuthRequest`: Contains validator hotkey, signature, timestamp, SSH public key
- `NodeConnectionDetails`: Contains node_id, host, port, username, ssh_endpoint

### 3. SSH Protocol

**Purpose**: Direct validator→node verification

**Flow**:

1. Validator generates ephemeral ed25519 key pair
2. Validator sends public key to miner during authentication
3. Miner deploys public key to all nodes' `~/.ssh/authorized_keys`
4. Validator SSHs directly to nodes using private key
5. Validator executes verification commands remotely
6. Miner removes validator's key after session expiry

**Security**:

- Ephemeral keys (short-lived, auto-rotated)
- Tagged keys: `ssh-ed25519 AAAA... validator-{hotkey}`
- Miner controls access duration (typically 1 hour)
- Audit logging of all SSH operations

### 4. REST API

**Validator API** (external access):

```text
GET  /health              # Health check
GET  /miners              # List all miners
GET  /miners/:id/nodes    # List miner's nodes
GET  /gpu-profiles        # List GPU profiles
POST /rentals             # Start GPU rental
GET  /rentals/:id         # Rental status
GET  /rentals/:id/logs    # Stream rental logs
```

**Basilica Gateway API** (optional, for aggregation):

```text
GET  /api/v1/capacity     # Available GPU capacity
POST /api/v1/rentals      # Start rental (aggregated)
GET  /api/v1/rentals/:id  # Rental status
```

## Data Flows

### 1. Registration Flow

```text
Miner:
  1. Register hotkey on Bittensor subnet
  2. Configure GPU node IPs in miner.toml
  3. Deploy miner SSH key to nodes
  4. Start miner (advertises axon on chain)

Validator:
  1. Register hotkey on Bittensor subnet
  2. Acquire sufficient stake for validator permit
  3. Start validator
  4. Query metagraph → Discover all miners
```

### 2. Verification Flow (Detailed)

```text
Step 1: Miner Discovery
  Validator → Bittensor Chain: Query metagraph for subnet
  Validator ← Bittensor Chain: Return all neurons (validators + miners)
  Validator: Filter for miners (validator_permit = false)

Step 2: Authentication
  Validator: Generate ephemeral SSH key pair
  Validator → Miner (gRPC): ValidatorAuthRequest {
      validator_hotkey: "5G3qVa...",
      ssh_public_key: "ssh-ed25519 AAAA...",
      signature: "0xabcd...",  // Signs: BASILICA_AUTH_V1:{nonce}:{timestamp}
      timestamp: 1704067200
  }
  Miner: Verify signature with validator hotkey
  Miner: Check timestamp freshness (<5 minutes)
  Miner: Deploy validator's SSH public key to all nodes
  Miner → Validator: MinerAuthResponse { success: true, session_token: "uuid" }

Step 3: Node Discovery
  Validator → Miner (gRPC): DiscoverNodesRequest { validator_hotkey }
  Miner → Validator (stream): NodeConnectionDetails {
      node_id: "550e8400-...",
      host: "192.168.1.100",
      port: 22,
      username: "basilica",
      ssh_endpoint: "ssh://192.168.1.100:22"
  }

Step 4: Direct SSH Verification
  Validator: Select validation strategy (Full or Lightweight)

  If Full Validation:
    Validator → Node (SSH): Connect using ephemeral key
    Validator → Node (SSH): Upload verification binaries
    Validator → Node (SSH): Execute binary remotely
    Node: Run GPU attestation, Docker check, storage check
    Validator → Node (SSH): Download JSON results
    Validator: Parse and validate results
    Validator → Database: Store GPU UUIDs, hardware profile, scores

  If Lightweight Validation:
    Validator → Node (SSH): Quick connection test
    Validator → Database: Update last_seen timestamp
    Validator: Reuse previous validation score

Step 5: Score Aggregation
  Validator: Calculate average score across all nodes for miner
  Validator → Database: Update miner_gpu_profiles
  Validator: Store for weight setting

Step 6: Weight Setting
  Every N blocks (default: 360):
    Validator → Database: Query all miner GPU profiles
    Validator: Calculate weights per GPU category
    Validator: Apply burn percentage
    Validator → Bittensor Chain: Submit weight vector
```

### 3. Rental Flow

```text
External Service → Validator API: POST /rentals {
    node_id: "550e8400-...",
    duration_hours: 24,
    ssh_public_key: "ssh-rsa AAAA...",
    container_spec: {...}
}

Validator: Validate request
Validator → Miner (gRPC): Request rental session
Miner: Create extended SSH session for renter
Miner → Nodes (SSH): Deploy renter's SSH public key

Validator → Rental DB: Store rental record
Validator → External Service: Return {
    rental_id: "uuid",
    ssh_endpoint: "ssh://192.168.1.100:22",
    credentials: {...}
}

External Service → Node (SSH): Connect and use GPU
```

## Security Architecture

### 1. Cryptographic Framework

**Algorithms** (implemented in `crates/basilica-common/src/crypto/`):

- **Sr25519**: Bittensor signatures and hotkey verification
- **Ed25519**: SSH key generation and node authentication
- **P256 ECDSA**: GPU attestation signatures
- **Blake3**: High-performance hashing for data integrity
- **AES-256-GCM**: Encrypted storage and secure communication
- **Argon2**: Key derivation and password hashing

### 2. Trust Model

```text
┌──────────────────────────────────────────────────────┐
│  Validator Trust (SSH-Based Verification)            │
├──────────────────────────────────────────────────────┤
│  ✓ Validator uploads own binaries to nodes           │
│    → Prevents miners from faking verification code   │
│                                                      │
│  ✓ Direct SSH execution on hardware                  │
│    → No intermediary can intercept or modify         │
│                                                      │
│  ✓ Ephemeral SSH keys per session                    │
│    → Limited exposure window, auto-rotation          │
│                                                      │
│  ✓ Miner controls access duration                    │
│    → Validator cannot maintain persistent access     │
│                                                      │
│  ✓ Cryptographic GPU attestation                     │
│    → Hardware-signed proofs of GPU authenticity      │
└──────────────────────────────────────────────────────┘

┌──────────────────────────────────────────────────────┐
│  Network Trust (Bittensor Consensus)                 │
├──────────────────────────────────────────────────────┤
│  ✓ On-chain registration required                    │
│  ✓ Economic stake ensures validator honesty          │
│  ✓ Weight-based consensus for emissions              │
│  ✓ Hotkey signatures prevent impersonation           │
└──────────────────────────────────────────────────────┘
```

### 3. Security Layers

#### Network Layer

- **Firewall Rules**: Minimal open ports (Axon, API, SSH)
- **Rate Limiting**: Per-IP request limits on API
- **DDoS Protection**: Application-level throttling

#### Application Layer

- **Hotkey Verification**: All gRPC requests signed with Bittensor keys
- **Timestamp Validation**: Prevent replay attacks (5-minute window)
- **Nonce Tracking**: Ensure request uniqueness
- **API Key Authentication**: Optional API key for external access
- **SSH Rate Limiting**: Max concurrent sessions per validator

#### Data Layer

- **Encrypted Storage**: Sensitive data encrypted at rest
- **Secure Key Management**: SSH keys in protected directories (chmod 600)
- **Database Access Control**: Principle of least privilege
- **Audit Logging**: All SSH operations logged with validator identity

### 4. Attack Mitigation

**SSH Relay Attack Prevention**:

- Miner tags deployed keys with validator hotkey
- Validator must prove ownership of hotkey via signature
- Timestamp + nonce prevent replay of auth requests

**GPU Fraud Prevention**:

- Validators upload their own verification binaries
- GPU UUIDs tracked to prevent duplicate claims
- Hardware attestation with cryptographic signatures

**Access Control**:

- Miner removes validator SSH keys after session expiry
- No permanent validator access to nodes
- Session duration limits (default: 1 hour max)

## Scalability Considerations

### Horizontal Scaling

#### Validators

- **Independent Operation**: Each validator verifies independently
- **Parallel Verification**: Multiple validators verify different miners simultaneously
- **Load Distribution**: Subnet-wide verification load naturally distributed

#### Miners

- **Fleet Expansion**: Add nodes by updating miner config
- **Geographic Distribution**: Nodes can be globally distributed
- **No Fleet Limit**: Limited only by miner's SSH management capacity

#### GPU Nodes

- **Simple Addition**: Just needs SSH server + GPU
- **No Special Software**: No complex installation or setup
- **Heterogeneous Hardware**: Support any NVIDIA GPU model

### Performance Optimization

**Two-Tier Validation Strategy**:

```rust
// Lightweight validation: Every 10 minutes
if node.last_validated < 6_hours_ago {
    perform_full_validation();  // Binary execution
} else {
    perform_lightweight_validation();  // Quick SSH test
}
```

**Benefits**:

- **Efficiency**: Only upload binaries every 6 hours
- **Responsiveness**: Detect node failures within 10 minutes
- **Resource Savings**: Lightweight checks use minimal bandwidth

**Concurrency**:

- Validators verify up to 50 miners concurrently (lightweight)
- Up to 1024 concurrent full validations
- Miners handle multiple validator sessions simultaneously

### Resource Management

**Validator Resources**:

```toml
[verification]
max_concurrent_verifications = 50       # Lightweight checks
max_concurrent_full_validations = 1024  # Binary validations
max_miners_per_round = 20               # Per verification cycle
```

**Miner Resources**:

```toml
[ssh_session]
max_concurrent_sessions = 5    # Per validator
session_rate_limit = 20        # Per hour per validator
```

**Node Resources**:

- Docker manages container resource limits
- GPU allocated exclusively per container
- Storage requirements: 1TB+ available

## Deployment Patterns

### 1. Development Setup

**Single Machine** (for testing):

```bash
# Terminal 1: Run validator
./basilica-validator --config validator-dev.toml start

# Terminal 2: Run miner
./basilica-miner --config miner-dev.toml

# GPU Node: Just needs SSH server
sudo systemctl start ssh
```

### 2. Production: Distributed Deployment

**Validator Cluster**:

```text
Region: US-East
├── Validator 1: Primary (Active)
├── Validator 2: Standby (Failover)
└── Load Balancer: HAProxy
    └── Shared Database: PostgreSQL

Region: EU-West
├── Validator 3: Primary (Active)
└── Validator 4: Standby (Failover)
```

**Miner Fleet**:

```text
Miner Server (No GPU required):
├── Miner Binary (manages node fleet)
├── SQLite Database (node registry)
└── SSH Keys (access to nodes)

GPU Nodes (Distributed):
├── Datacenter A: 10x H100 nodes
├── Datacenter B: 20x A100 nodes
└── Datacenter C: 15x B200 nodes
```

**Benefits**:

- Geographic redundancy
- Load distribution
- Failure isolation
- Easy horizontal scaling

### 3. High Availability Pattern

```text
         ┌─────────────────┐
         │  Load Balancer  │
         │    (HAProxy)    │
         └────────┬────────┘
                  │
         ┌────────┴────────┐
         │                 │
   ┌─────▼─────┐     ┌────▼──────┐
   │Validator 1│     │Validator 2│
   │ (Active)  │     │ (Standby) │
   └─────┬─────┘     └────┬──────┘
         │                │
         └────────┬───────┘
                  │
         ┌────────▼─────────┐
         │   PostgreSQL     │
         │   (Shared DB +   │
         │ Distributed Lock)│
         └──────────────────┘
```

**Features**:

- Automatic failover within 30 seconds
- Distributed locking prevents duplicate work
- Both validators serve API requests
- Shared database for consistent state

## Monitoring and Observability

### 1. Metrics Collection

**Prometheus Integration** (all components):

```promql
# Validator metrics
basilica_verification_total{type="full|lightweight"}
basilica_verification_success_total
basilica_miners_discovered_total
basilica_weight_set_total

# Miner metrics
basilica_nodes_registered_total
basilica_validator_sessions_active
basilica_ssh_key_deployments_total

# Node metrics (collected during verification)
basilica_gpu_utilization{node_id, gpu_model}
basilica_node_uptime_seconds{node_id}
```

### 2. Logging Architecture

**Structured Logging** (using `tracing` crate):

```text
Component Logs
    ↓
Tracing Framework
    ↓
Log Aggregator (Loki/ELK)
    ↓
Query & Analysis
    ↓
Alerting (Prometheus Alertmanager)
```

**Key Log Targets**:

- `validator::discovery` - Miner discovery events
- `validator::verification` - Verification execution
- `validator::ssh` - SSH operations
- `miner::node_manager` - Node management
- `miner::validator_comms` - gRPC server

### 3. Health Monitoring

**Endpoints**:

```bash
# Validator health
curl http://localhost:8080/health
# Response: {"status": "healthy", "checks": {...}}

# Miner health
curl http://localhost:8091/health
# Response: {"status": "healthy", "nodes": 10, "active_sessions": 3}

# Prometheus metrics
curl http://localhost:9090/metrics
```

**Grafana Dashboards**:

- Validator Performance (verifications/min, success rate)
- Miner Fleet Status (nodes online, sessions active)
- GPU Utilization (per node, per category)
- Network Health (latency, error rates)

## Code Organization

### Workspace Structure

```text
basilica/
├── crates/
│   ├── basilica-common/        # Shared utilities
│   │   ├── src/
│   │   │   ├── crypto/         # Cryptographic operations
│   │   │   ├── ssh/            # SSH trait abstractions
│   │   │   ├── config/         # Configuration loading
│   │   │   ├── persistence/    # Database traits
│   │   │   └── identity.rs     # Core identity types
│   │   └── Cargo.toml
│   │
│   ├── basilica-protocol/      # gRPC definitions
│   │   ├── proto/              # Protobuf files
│   │   └── src/gen/            # Generated Rust code
│   │
│   ├── basilica-validator/     # Validator service
│   │   ├── src/
│   │   │   ├── miner_prover/   # Verification orchestration
│   │   │   ├── bittensor_core/ # Weight setting
│   │   │   ├── api/            # REST API
│   │   │   ├── ssh/            # SSH client implementation
│   │   │   └── service.rs      # Main service
│   │   ├── migrations/         # Database migrations
│   │   └── Cargo.toml
│   │
│   ├── basilica-miner/         # Miner service
│   │   ├── src/
│   │   │   ├── node_manager.rs     # Node SSH management
│   │   │   ├── validator_comms.rs  # gRPC server
│   │   │   ├── validator_assignment.rs # Routing logic
│   │   │   └── service.rs          # Main service
│   │   ├── migrations/         # Database migrations
│   │   └── Cargo.toml
│   │
│   └── basilica-api/           # API gateway (optional)
│       └── src/
│
├── scripts/
│   ├── validator/              # Validator deployment
│   │   ├── build.sh
│   │   ├── deploy.sh
│   │   └── systemd/
│   └── miner/                  # Miner deployment
│       ├── build.sh
│       ├── deploy.sh
│       └── systemd/
│
├── config/                     # Configuration examples
│   ├── validator.correct.toml
│   └── miner.correct.toml
│
└── docs/                       # Documentation
    ├── validator.md            # Validator guide
    ├── miner.md                # Miner guide
    ├── architecture.md         # This file
    └── README.md               # Documentation index
```

## Development Guidelines

### Testing Strategy

**1. Unit Tests** (component-level):

```bash
# Test individual modules
cargo test -p basilica-common
cargo test -p basilica-validator
cargo test -p basilica-miner
```

**2. Integration Tests** (cross-component):

```bash
# Test validator→miner communication
cargo test -p integration-tests --test validator_miner_integration

# Test SSH verification flow
cargo test -p integration-tests --test ssh_verification
```

**3. End-to-End Tests** (full system):

```bash
# Test complete verification cycle
cargo test -p integration-tests --test e2e_verification

# Test weight setting
cargo test -p integration-tests --test e2e_weights
```

### Code Quality

**Linting and Formatting**:

```bash
# Format all code
cargo fmt

# Check for issues
cargo clippy

# Fix common issues
just fix
```

**Security Checks**:

```bash
# Audit dependencies
cargo audit

# Check for unsafe code
cargo geiger
```

### Contributing Guidelines

1. **Follow Rust Best Practices**:
   - Use explicit error types (not `anyhow` in libraries)
   - Implement `BasilicaError` trait for custom errors
   - Prefer traits for abstractions (dependency injection)

2. **Documentation**:
   - Document all public APIs with examples
   - Include code references in docs (e.g., `miner_prover/discovery.rs:40-122`)
   - Update architecture docs when adding features

3. **Testing**:
   - Unit tests for all public functions
   - Integration tests for cross-component features
   - Performance tests for critical paths

4. **Security**:
   - Never log sensitive data (keys, signatures)
   - Validate all external inputs
   - Use constant-time comparisons for secrets

## Future Enhancements

### Planned Features

1. **Enhanced GPU Profiling**:
   - More granular performance benchmarks
   - GPU memory bandwidth testing
   - Multi-GPU optimization scoring

2. **Advanced Routing**:
   - Latency-based validator assignment
   - Geographic affinity routing
   - Load-aware node selection

3. **Rental Marketplace**:
   - Direct GPU rental API
   - Hourly pricing per GPU category
   - Automated billing and payments

4. **Federation**:
   - Cross-subnet resource sharing
   - Multi-chain support
   - Bridge to other GPU networks

5. **Performance Optimization**:
   - SSH connection pooling
   - Binary caching on nodes
   - Parallel verification pipelines

### Research Areas

- **Zero-Knowledge Proofs**: Verify computations without revealing data
- **Trusted Execution Environments**: TEE-based GPU attestation
- **Decentralized Storage**: IPFS integration for large datasets
- **ML Workload Optimization**: Specialized routing for training vs inference

## Conclusion

Basilica's SSH-based architecture provides:

✅ **Simplicity**: No intermediary agents, standard protocols
✅ **Security**: Direct verification prevents tampering
✅ **Scalability**: Easy horizontal scaling of validators and nodes
✅ **Trust**: Cryptographic proofs at every layer
✅ **Economics**: Bittensor provides incentive alignment

**Key Innovation**: Validators SSH directly to GPU nodes, eliminating intermediary trust requirements while maintaining security through cryptographic verification.

For implementation details:

- **Validators**: See [Validator Guide](validator.md)
- **Miners**: See [Miner Guide](miner.md)
- **Deployment**: See [Quick Start Guide](quickstart.md)
