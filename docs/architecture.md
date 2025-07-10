# Architecture Guide

This guide provides a comprehensive overview of Basilica's system architecture and design principles.

## System Overview

Basilica is a decentralized GPU compute marketplace built on the Bittensor network. It creates a trustless environment where GPU providers (miners) can offer compute resources, and validators ensure quality and reliability through cryptographic verification.

## Core Components

### 1. Validator

The validator is the quality assurance layer of the network:

- **Verification Engine**: Performs SSH-based remote validation of computational tasks
- **GPU Profile Management**: Maintains GPU performance profiles and benchmarks
- **Scoring System**: Maintains performance metrics for all miners using GPU categorization
- **Weight Setter**: Updates Bittensor network weights based on miner performance
- **REST API**: Provides external access to validation data
- **SQLite Storage**: Persists verification history, GPU profiles, and miner scores
- **Binary Validation**: Executes validation binaries for secure GPU verification

### 2. Miner

The miner acts as a fleet manager for GPU resources:

- **Executor Fleet Manager**: Orchestrates multiple GPU executor machines
- **Axon Server**: Serves compute requests on the Bittensor network
- **gRPC Client**: Manages communication with executors
- **Assignment Manager**: Routes computational tasks to appropriate executors
- **SSH Session Management**: Handles validator access and session orchestration
- **Stake Monitor**: Tracks stake levels and maintains service quality
- **SQLite Storage**: Persists executor assignments and registration data

### 3. Executor

The executor is the GPU machine agent:

- **gRPC Server**: Receives and processes task requests
- **Container Manager**: Orchestrates Docker containers for isolated execution
- **System Monitor**: Reports hardware status and resource utilization (CPU, GPU, memory, disk, network)
- **Validation Sessions**: Handles validator verification challenges with rate limiting
- **Security Layer**: Hotkey verification and access control
- **NVIDIA Integration**: Uses nvml-wrapper for GPU monitoring

### 4. Public API

The smart HTTP gateway for external access:

- **Validator Discovery**: Automatic discovery of validators using Bittensor metagraph
- **Load Balancing**: Multiple strategies for distributing requests across validators
- **Request Aggregation**: Combines responses from multiple validators
- **Authentication**: API key and JWT-based authentication
- **Rate Limiting**: Configurable rate limits with different tiers
- **Caching**: Response caching with in-memory (Moka) or Redis backends
- **OpenAPI Documentation**: Auto-generated API documentation with Swagger UI

## System Architecture

```text
┌─────────────────────────────────────────────────────────┐
│                   BITTENSOR NETWORK                     │
│                       (Subnet)                          │
└────────────────────────┬───────────────────────────────┘
                         │
        ┌────────────────┼────────────────┐
        │                │                │
┌───────▼──────┐ ┌──────▼──────┐ ┌──────▼──────┐
│  VALIDATOR   │ │    MINER    │ │ PUBLIC API  │
│              │ │             │ │  GATEWAY    │
│ ┌──────────┐ │ │ ┌─────────┐ │ │ ┌─────────┐ │
│ │   API    │ │ │ │  Axon   │ │ │ │  Load   │ │
│ │ Server   │ │ │ │ Server  │ │ │ │Balancer │ │
│ └──────────┘ │ │ └─────────┘ │ │ └─────────┘ │
│ ┌──────────┐ │ │ ┌─────────┐ │ │ ┌─────────┐ │
│ │  Binary  │ │ │ │  Fleet  │ │ │ │  Cache  │ │
│ │Validator │ │ │ │ Manager │ │ │ │  Layer  │ │
│ └──────────┘ │ │ └─────────┘ │ │ └─────────┘ │
│ ┌──────────┐ │ │ ┌─────────┐ │ │ ┌─────────┐ │
│ │ SQLite   │ │ │ │  gRPC   │ │ │ │  Auth   │ │
│ │   DB     │ │ │ │ Client  │ │ │ │ Manager │ │
│ └──────────┘ │ │ └─────────┘ │ │ └─────────┘ │
└──────┬───────┘ └──────┬──────┘ └─────────────┘
       │                │
       │ SSH            │ gRPC
       │                │
┌──────▼────────────────▼────────────────────────┐
│              EXECUTOR MACHINES                  │
│  ┌─────────┐  ┌─────────┐  ┌─────────┐        │
│  │ GPU-1   │  │ GPU-2   │  │ GPU-N   │  ...   │
│  │ NVIDIA  │  │ NVIDIA  │  │ NVIDIA  │        │
│  └─────────┘  └─────────┘  └─────────┘        │
└────────────────────────────────────────────────┘
```

## Communication Protocols

### 1. Bittensor Protocol

- **Blockchain Integration**: Substrate-based chain communication
- **Axon Protocol**: TCP-based compute request serving
- **Weight Updates**: On-chain consensus mechanism

### 2. gRPC Protocol

Used for miner-executor communication:

```proto
service Executor {
  rpc Execute(ExecuteRequest) returns (ExecuteResponse);
  rpc GetSystemInfo(Empty) returns (SystemInfo);
  rpc HealthCheck(Empty) returns (HealthStatus);
}
```

### 3. SSH Protocol

Validator verification mechanism:

- Secure remote command execution
- Hardware verification challenges
- Computational task validation

### 4. REST API

External interfaces for monitoring:

- Health endpoints
- Metrics collection
- Administrative APIs

## Data Flow

### 1. Registration Flow

```text
Miner → Bittensor Network: Register with stake
Validator → Bittensor Network: Query metagraph
Validator → Miner: Discover via metagraph
```

### 2. Verification Flow

```text
Validator → Miner (Axon): Request executor info
Miner → Validator: Return executor endpoints
Validator → Executor (SSH): Send verification challenge
Executor → Binary Validator: Execute validation binary
Binary Validator → Executor: Return GPU verification results
Executor → Validator: Return validation results
Validator → Storage: Update scores and GPU profiles
Validator → Bittensor: Set weights
```

### 3. Compute Request Flow

```text
Client → Miner (Axon): Submit compute task
Miner → Executor (gRPC): Forward task
Executor → Container: Execute in isolation
Executor → Miner: Return results
Miner → Client: Forward results
```

## Security Architecture

### 1. Cryptographic Framework

- **Ed25519**: Bittensor transaction signing and hotkey verification
- **P256 ECDSA**: Used in protocol for signatures
- **Blake3**: High-performance hashing
- **AES-GCM**: Encrypted storage and communication
- **TLS**: Secure communication channels
- **Argon2**: Password hashing for authentication

### 2. Trust Model

```text
Hardware Trust: Binary Validator → GPU Verification
Execution Trust: Validator → Verification Challenges
Network Trust: Bittensor → Consensus Mechanism
Access Control: Hotkey Verification → Authenticated Sessions
```

### 3. Security Layers

1. **Network Layer**
   - Firewall rules
   - VPN connections
   - Rate limiting

2. **Application Layer**
   - JWT authentication for Public API
   - API key validation
   - Hotkey verification for executor access
   - Rate limiting per identity

3. **Data Layer**
   - Encrypted storage
   - Secure key management
   - Database access control

## Scalability Considerations

### Horizontal Scaling

- **Validators**: Multiple validators increase verification coverage
- **Miners**: Fleet expansion through executor addition
- **Executors**: Distributed across geographic regions

### Performance Optimization

1. **Connection Pooling**: Reuse gRPC connections
2. **Batch Processing**: Group verification tasks
3. **Caching**: Store frequently accessed data
4. **Async Operations**: Non-blocking I/O throughout

### Resource Management

```toml
# Example resource limits
[executor.resources]
max_containers = 10
max_memory_per_container = "8Gi"
max_cpu_per_container = 4
gpu_allocation_mode = "exclusive"
```

## Deployment Patterns

### 1. Single-Node Development

```text
Local Machine:
├── Validator (Port 8081)
├── Miner (Port 8080)
└── Executor (Port 50051)
```

### 2. Distributed Production

```text
Validator Region:
├── Multiple Validators (Geographic distribution)
└── Load Balancer

Miner Region:
├── Miner Nodes (High availability)
└── Executor Fleet (GPU clusters)
```

## Monitoring and Observability

### 1. Metrics Collection

- **Prometheus Integration**: Time-series metrics
- **Custom Metrics**: Verification rates, GPU utilization
- **Network Metrics**: Latency, throughput

### 2. Logging Architecture

```text
Component Logs → Aggregator → Storage → Analysis
                     ↓
                Alerting System
```

### 3. Health Monitoring

```bash
# Health check endpoints
GET /health          # Basic health
GET /metrics         # Prometheus metrics
GET /api/v1/status   # Detailed status
```

## Development Guidelines

### Code Organization

```text
basilica/
├── crates/              # Rust workspace
│   ├── common/          # Shared utilities (crypto, config, storage, SSH)
│   ├── protocol/        # Protocol definitions (gRPC/protobuf)
│   ├── validator/       # Validator service
│   ├── miner/          # Miner service
│   ├── executor/        # Executor service
│   ├── bittensor/       # Network integration
│   ├── public-api/      # HTTP gateway service
│   └── integration-tests/ # End-to-end tests
├── scripts/             # Deployment scripts
│   └── gpu-attestor/    # GPU attestation Docker setup
├── bins/                # Precompiled binaries
└── docs/                # Documentation
```

### Testing Strategy

1. **Unit Tests**: Component-level testing
2. **Integration Tests**: Cross-component verification
3. **End-to-End Tests**: Full system validation
4. **Performance Tests**: Load and stress testing

### Contributing

1. Follow Rust best practices
2. Maintain comprehensive documentation
3. Include tests for new features

## Future Enhancements

### Planned Features

1. **Enhanced GPU Support**: Better GPU profiling and categorization
2. **Advanced Scheduling**: Intelligent task distribution based on GPU capabilities
3. **Federation**: Cross-subnet resource sharing
4. **Enhanced Security**: Improved binary validation and attestation
5. **Performance Optimization**: Better caching and load balancing strategies

## Conclusion

Basilica's architecture provides a robust foundation for decentralized GPU compute. The modular design enables independent scaling, security layers ensure trust, and Bittensor integration provides economic incentives for quality service.

For implementation details, see the [Miner Guide](miner.md) and [Validator Guide](validator.md).
