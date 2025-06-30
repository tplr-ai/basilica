# GPU Attestor

Secure GPU hardware attestation for the Basilica decentralized compute network. This crate provides comprehensive hardware verification, cryptographic attestation, and integrity validation for GPU nodes.

## Features

- **Hardware Detection**: Multi-vendor GPU detection (NVIDIA, AMD, Intel)
- **System Profiling**: CPU, memory, storage, network, and Docker environment analysis
- **Network Benchmarking**: Latency, throughput, DNS resolution, and connectivity testing
- **VDF Computation**: Verifiable Delay Functions for proof-of-computation
- **Cryptographic Signing**: P256 ECDSA signatures with embedded key validation
- **Binary Integrity**: Self-verification with embedded public keys
- **Modular Architecture**: Clean separation between CLI, core logic, and attestation modules

## Architecture

### Core Modules

- **`cli`**: Command-line interface and configuration management
- **`attestation`**: Report building, signing, and verification
- **`gpu`**: Multi-vendor GPU detection and information collection
- **`hardware`**: System information collection and benchmarking
- **`network`**: Network performance testing and connectivity validation
- **`vdf`**: Verifiable Delay Function computation and verification
- **`integrity`**: Binary integrity checking with embedded keys

### Key Components

- **AttestationBuilder**: Fluent API for creating comprehensive attestation reports
- **AttestationSigner**: Cryptographic signing with ephemeral or embedded keys
- **AttestationVerifier**: Multi-layer verification (signature, content, timestamps)
- **GpuDetector**: Cross-platform GPU discovery and monitoring
- **SystemInfoCollector**: Hardware profiling and benchmark execution
- **NetworkBenchmarker**: Comprehensive network performance analysis
- **VdfComputer**: Secure proof-of-computation generation

## Quick Start

### 1. Generate Cryptographic Keys

```bash
# Generate new P256 key pair
./scripts/gen-key.sh

# This creates:
# - private_key.pem (private key, 600 permissions)
# - public_key.pem (public key in PEM format)
# - public_key.hex (compressed public key for builds)
```

### 2. Build the Attestor

```bash
# Auto-detect keys and build (recommended)
./scripts/gpu-attestor.build.sh

# Build release version
./scripts/gpu-attestor.build.sh --release

# Specify key explicitly
./scripts/gpu-attestor.build.sh 0279be667ef9dcbbac55a06295ce870b07029bfcdb2dce28d959f2815b16f81798 --release

# Build with specific public key from file
./scripts/gpu-attestor.build.sh $(cat ./public_key.hex) --release

# See all options
./scripts/gpu-attestor.build.sh --help
```

### 3. Run Attestation

```bash
# Full attestation with all features
./target/release/gpu-attestor

# Quick testing (skip slow operations)
./target/release/gpu-attestor --skip-vdf --skip-network-benchmark

# Custom output location
./target/release/gpu-attestor -o ./my-attestation

# See all CLI options
./target/release/gpu-attestor --help
```

## CLI Options

```bash
USAGE:
    gpu-attestor [OPTIONS]

OPTIONS:
    --executor-id <ID>              Unique identifier for this executor
    -o, --output <PATH>             Output path for attestation files [default: ./attestation]
    --skip-integrity-check          Skip binary integrity verification (testing only)
    --skip-network-benchmark        Skip network performance testing
    --skip-vdf                      Skip VDF computation
    --skip-hardware-collection      Skip detailed hardware collection
    --vdf-difficulty <NUMBER>       VDF computation difficulty [default: 1000]
    --vdf-algorithm <ALGORITHM>     VDF algorithm [default: simple] [possible: wesolowski, pietrzak, simple]
    --log-level <LEVEL>             Logging level [default: info] [possible: error, warn, info, debug, trace]
    -h, --help                      Print help information
    -V, --version                   Print version information
```

## Development

### Testing

```bash
# Run all tests
cargo test -p gpu-attestor

# Run specific test modules
cargo test -p gpu-attestor attestation::
cargo test -p gpu-attestor gpu::
cargo test -p gpu-attestor hardware::

# Run with environment variable for build tests
VALIDATOR_PUBLIC_KEY=0279be667ef9dcbbac55a06295ce870b07029bfcdb2dce28d959f2815b16f81798 cargo test -p gpu-attestor
```

### Code Quality

```bash
# Format code
cargo fmt

# Lint checks
cargo clippy

# Combined format and fix
just fix  # (if using justfile)
```

### Key Management

```bash
# Extract public key from existing private key
./scripts/extract-pubkey.sh private_key.pem

# Export public key to PEM file
./scripts/extract-pubkey.sh --export validator_key.pem

# Generate multiple key formats
./scripts/gen-key.sh  # Creates .pem and .hex files
```

## Output Files

After successful attestation, the following files are generated:

- **`attestation.json`**: Complete attestation report with all collected data
- **`attestation.sig`**: Cryptographic signature of the report
- **`attestation.pub`**: Public key used for signature verification

### Example Attestation Report Structure

```json
{
  "version": "1.0.0",
  "timestamp": "2024-06-17T14:00:00Z",
  "executor_id": "gpu-node-01",
  "binary_info": {
    "path": "/usr/local/bin/gpu-attestor",
    "signature_verified": true,
    "validator_public_key_fingerprint": "0279be...81798"
  },
  "gpu_info": [
    {
      "vendor": "Nvidia",
      "name": "NVIDIA GeForce RTX 4090",
      "memory_total": 25769803776,
      "driver_version": "525.60.11"
    }
  ],
  "system_info": {
    "cpu": { "cores": 16, "threads": 32, "brand": "AMD Ryzen 9 5950X" },
    "memory": { "total_bytes": 68719476736 },
    "docker": { "is_running": true, "version": "24.0.2" }
  },
  "network_benchmark": {
    "latency_tests": [...],
    "throughput_tests": [...],
    "dns_resolution_test": {...}
  },
  "vdf_proof": {
    "computation_time_ms": 1247,
    "algorithm": "SimpleSequential"
  }
}
```

## Security Features

### Binary Integrity

- **Embedded Keys**: Public keys embedded at build time for self-verification
- **Signature Validation**: Cryptographic verification of binary authenticity
- **Tamper Detection**: Detection of unauthorized modifications

### Cryptographic Security

- **P256 ECDSA**: Industry-standard elliptic curve cryptography
- **Compressed Keys**: Efficient 33-byte public key representation
- **Ephemeral Signing**: Secure key generation for attestation signatures

### Hardware Validation

- **Multi-layer Verification**: Comprehensive hardware and software validation
- **Time-bound Attestations**: Timestamp validation prevents replay attacks

## Troubleshooting

### Build Issues

**Missing VALIDATOR_PUBLIC_KEY:**

```bash
# Generate keys first
./scripts/gen-key.sh

# Then build
./scripts/gpu-attestor.build.sh
```

**OpenSSL Linking Errors:**

```bash
# Install system dependencies (Ubuntu/Debian)
sudo apt-get install libssl-dev pkg-config

# Or build library only (no binary)
cargo build -p gpu-attestor --lib
```

### Runtime Issues

**No GPUs Detected:**

- Ensure GPU drivers are installed
- Check system permissions for GPU access
- Use `--log-level debug` for detailed GPU detection logs

**Network Benchmark Failures:**

- Use `--skip-network-benchmark` for offline testing
- Check firewall and network connectivity
- Verify DNS resolution works

**VDF Computation Timeout:**

- Reduce difficulty: `--vdf-difficulty 100`
- Skip for testing: `--skip-vdf`
- Check available CPU resources

## Integration

### As a Library

```rust
use gpu_attestor::{
    AttestationBuilder, AttestationSigner, GpuDetector,
    SystemInfoCollector, NetworkBenchmarker
};

// Create attestation programmatically
let gpu_info = GpuDetector::query_gpu_info()?;
let system_info = SystemInfoCollector::collect_all()?;

let report = AttestationBuilder::new("my-executor".to_string())
    .with_gpu_info(gpu_info)
    .with_system_info(system_info)
    .build();

let signer = AttestationSigner::new();
let signed_attestation = signer.sign_attestation(report)?;
```

### Configuration Management

```rust
use gpu_attestor::cli::{Config, parse_args, setup_logging};

// Use CLI configuration in your application
let config = parse_args()?;
setup_logging(&config.log_level)?;

// Access all CLI options
println!("Executor ID: {}", config.executor_id);
println!("Skip VDF: {}", config.skip_vdf);
```

## Contributing

### Code Style

- Follow existing patterns and module organization
- Use the provided CLI module for configuration management
- Implement comprehensive error handling with `anyhow`
- Add tests for new functionality

### Module Guidelines

- **CLI**: Keep argument parsing and setup in `cli.rs`
- **Core Logic**: Business logic in appropriate domain modules
- **Tests**: Co-locate tests with implementation
- **Documentation**: Update README for user-facing changes

## License

This project is part of the Basilica network implementation.
