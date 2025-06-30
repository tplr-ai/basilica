# Architecture Guide

This guide provides a comprehensive overview of Basilica's system architecture and design principles.

## System Overview

Basilica is a decentralized GPU compute marketplace built on the Bittensor network. It creates a trustless environment where GPU providers (miners) can offer compute resources, and validators ensure quality and reliability through cryptographic verification.

## Core Components

### 1. Validator

The validator is the quality assurance layer of the network:

- **Verification Engine**: Performs SSH-based remote validation of computational tasks
- **Scoring System**: Maintains performance metrics for all miners
- **Weight Setter**: Updates Bittensor network weights based on miner performance
- **REST API**: Provides external access to validation data
- **SQLite Storage**: Persists verification history and miner scores

### 2. Miner

The miner acts as a fleet manager for GPU resources:

- **Executor Fleet Manager**: Orchestrates multiple GPU executor machines
- **Axon Server**: Serves compute requests on the Bittensor network
- **gRPC Server**: Manages communication with executors
- **Task Distributor**: Routes computational tasks to appropriate executors
- **Health Monitor**: Tracks executor status and availability

### 3. Executor

The executor is the GPU machine agent:

- **gRPC Server**: Receives and processes task requests
- **Container Manager**: Orchestrates Docker containers for isolated execution
- **System Monitor**: Reports hardware status and resource utilization
- **Validation Sessions**: Handles validator verification challenges
- **Security Layer**: JWT authentication and secure communication

### 4. GPU-Attestor

The hardware attestation component:

- **Hardware Detection**: Identifies GPU specifications and capabilities
- **VDF Computer**: Generates verifiable delay functions for proof-of-work
- **Crypto Signer**: Creates P256 ECDSA signatures for hardware proofs
- **Attestation Generator**: Produces cryptographically verifiable hardware claims

## System Architecture

```
┌─────────────────────────────────────────────────────────┐
│                   BITTENSOR NETWORK                     │
│                   (Subnet 39/387)                       │
└────────────────────────┬───────────────────────────────┘
                         │
        ┌────────────────┼────────────────┐
        │                │                │
┌───────▼──────┐ ┌──────▼──────┐ ┌──────▼──────┐
│  VALIDATOR   │ │    MINER    │ │    CLIENT   │
│              │ │             │ │   SYSTEMS   │
│ ┌──────────┐ │ │ ┌─────────┐ │ │             │
│ │   API    │ │ │ │  Axon   │ │ │             │
│ │ Server   │ │ │ │ Server  │ │ │             │
│ └──────────┘ │ │ └─────────┘ │ │             │
│ ┌──────────┐ │ │ ┌─────────┐ │ │             │
│ │   SSH    │ │ │ │  Fleet  │ │ │             │
│ │ Client   │ │ │ │ Manager │ │ │             │
│ └──────────┘ │ │ └─────────┘ │ │             │
│ ┌──────────┐ │ │ ┌─────────┐ │ │             │
│ │ SQLite   │ │ │ │  gRPC   │ │ │             │
│ │   DB     │ │ │ │ Server  │ │ │             │
│ └──────────┘ │ │ └─────────┘ │ │             │
└──────┬───────┘ └──────┬──────┘ └─────────────┘
       │                │
       │ SSH            │ gRPC
       │                │
┌──────▼────────────────▼────────────────────────┐
│              EXECUTOR MACHINES                  │
│  ┌─────────┐  ┌─────────┐  ┌─────────┐        │
│  │ GPU-1   │  │ GPU-2   │  │ GPU-N   │  ...   │
│  │ + Atts  │  │ + Atts  │  │ + Atts  │        │
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

```
Miner → Bittensor Network: Register with stake
Validator → Bittensor Network: Query metagraph
Validator → Miner: Discover via metagraph
```

### 2. Verification Flow

```
Validator → Miner (Axon): Request executor info
Miner → Validator: Return executor endpoints
Validator → Executor (SSH): Send verification challenge
Executor → Validator: Return signed results
Validator → Storage: Update scores
Validator → Bittensor: Set weights
```

### 3. Compute Request Flow

```
Client → Miner (Axon): Submit compute task
Miner → Executor (gRPC): Forward task
Executor → Container: Execute in isolation
Executor → Miner: Return results
Miner → Client: Forward results
```

## Security Architecture

### 1. Cryptographic Framework

- **P256 ECDSA**: Hardware attestation signatures
- **Ed25519**: Bittensor transaction signing
- **JWT**: Executor authentication tokens
- **TLS**: Secure communication channels

### 2. Trust Model

```
Hardware Trust: GPU-Attestor → Cryptographic Proof
Execution Trust: Validator → Verification Challenges
Network Trust: Bittensor → Consensus Mechanism
```

### 3. Security Layers

1. **Network Layer**
   - Firewall rules
   - VPN connections
   - Rate limiting

2. **Application Layer**
   - JWT authentication
   - API key validation
   - Request signing

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

```
Local Machine:
├── Validator (Port 8081)
├── Miner (Port 8080)
└── Executor (Port 50051)
```

### 2. Distributed Production

```
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

```
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

```
basilica/
├── crates/           # Rust workspace
│   ├── common/       # Shared utilities
│   ├── protocol/     # Protocol definitions
│   ├── validator/    # Validator service
│   ├── miner/        # Miner service
│   ├── executor/     # Executor service
│   └── bittensor/    # Network integration
├── scripts/          # Deployment scripts
├── docker/           # Container configs
└── docs/             # Documentation
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

1. **Multi-GPU Support**: Enhanced parallel processing
2. **Advanced Scheduling**: Intelligent task distribution
3. **Federation**: Cross-subnet resource sharing
4. **Enhanced Security**: Hardware security modules


## Conclusion

Basilica's architecture provides a robust foundation for decentralized GPU compute. The modular design enables independent scaling, security layers ensure trust, and Bittensor integration provides economic incentives for quality service.

For implementation details, see the [Miner Guide](miner.md) and [Validator Guide](validator.md).