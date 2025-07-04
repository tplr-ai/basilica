# GPU Attestor Complete Guide

## Overview

The GPU Attestor is a secure hardware attestation system for the Basilisk network that validates GPU hardware capabilities and executes proof-of-work (PoW) challenges. It provides cryptographically signed attestations of GPU performance and ensures that miners cannot claim more computational resources than they actually possess.

## Architecture

### Core Components

```
crates/gpu-attestor/
├── src/
│   ├── main.rs              # CLI entry point
│   ├── lib.rs               # Library exports
│   ├── cli.rs               # Command-line interface
│   │
│   ├── attestation/         # Attestation generation & signing
│   │   ├── builder.rs       # Builds attestation documents
│   │   ├── signer.rs        # Cryptographic signing (RSA/ECDSA)
│   │   ├── verifier.rs      # Attestation verification
│   │   └── types.rs         # Attestation data structures
│   │
│   ├── challenge/           # GPU PoW challenge system
│   │   ├── handler.rs       # Main challenge handler
│   │   ├── matrix_pow.rs    # Matrix multiplication PoW
│   │   ├── kernel_ops.rs    # Shared CUDA kernel operations
│   │   ├── multi_gpu_handler.rs     # Multi-GPU execution
│   │   ├── multi_gpu_challenge.rs   # Challenge distribution
│   │   ├── sm_monitor.rs            # SM utilization monitoring
│   │   ├── core_saturation_validator.rs  # Core usage validation
│   │   └── anti_spoofing_validator.rs   # Anti-spoofing checks
│   │
│   ├── gpu/                 # GPU detection & management
│   │   ├── gpu_detector.rs  # Unified GPU detection (CUDA)
│   │   ├── cuda_driver/     # CUDA Driver API bindings
│   │   │   ├── cuda_driver_ffi.rs   # Low-level CUDA FFI
│   │   │   ├── cuda_wrapper.rs      # Safe CUDA wrappers
│   │   │   └── kernels/             # CUDA kernels
│   │   │       ├── matrix_multiply.cu/ptx
│   │   │       ├── prng.cu/ptx
│   │   │       └── sha256.cu/ptx
│   │   ├── benchmarks.rs    # GPU benchmarking
│   │   └── types.rs         # GPU-related types
│   │
│   ├── hardware/            # System hardware collection
│   ├── network/             # Network benchmarking
│   ├── os/                  # OS attestation
│   └── validation/          # Performance validation
```

## Key Features

### 1. GPU Detection and Enumeration

The unified `GpuDetector` automatically discovers all CUDA-capable GPUs:

```rust
// Detects all GPUs with detailed specifications
let detector = GpuDetector::detect_all()?;

// Each GPU includes:
// - Device ID, name, UUID
// - Compute capability
// - SM count and cores per SM
// - Memory (total/available)
// - Memory bandwidth
// - NVLink topology (if available)
```

### 2. GPU Proof-of-Work Challenge

The GPU PoW system uses bandwidth-intensive matrix multiplication to validate GPU capabilities:

#### Challenge Flow

1. **Challenge Reception**: Base64-encoded challenge parameters from validator
2. **Challenge Distribution**: Work divided across available GPUs
3. **Execution**: Parallel matrix multiplication with monitoring
4. **Validation**: Core saturation and anti-spoofing checks
5. **Result**: Cryptographic checksum of computation

#### Challenge Parameters

```rust
pub struct ChallengeParameters {
    pub challenge_type: String,              // "matrix_multiplication_pow"
    pub gpu_pow_seed: u64,                   // Deterministic seed
    pub matrix_dim: u32,                     // Matrix dimensions (e.g., 8192)
    pub num_matrices: u32,                   // Number of matrices
    pub num_iterations: u32,                 // Iteration count
    pub validator_nonce: String,             // Unique validator nonce
}
```

### 3. Matrix Multiplication PoW Algorithm

The core PoW algorithm performs bandwidth-intensive matrix operations:

```rust
// Simplified algorithm flow
for iteration in 0..num_iterations {
    for matrix_idx in 0..num_matrices {
        // Generate matrix indices deterministically
        let a_idx = philox_index(seed, iteration, matrix_idx * 2) % num_matrices;
        let b_idx = philox_index(seed, iteration, matrix_idx * 2 + 1) % num_matrices;
        
        // Generate matrices on GPU using PRNG
        prng.generate_matrix(&mut matrix_a, seed, a_idx, matrix_dim);
        prng.generate_matrix(&mut matrix_b, seed + 1, b_idx, matrix_dim);
        
        // Execute matrix multiplication on GPU
        kernel.execute(&matrix_a, &matrix_b, &result, matrix_dim);
        
        // Accumulate checksum
        accumulate_checksum(&mut checksum, &result);
    }
}
```

### 4. Multi-GPU Support

The system automatically distributes work across multiple GPUs:

- **Automatic Detection**: Discovers all available GPUs
- **Work Distribution**: Balances load based on GPU capabilities
- **Synchronization**: Barrier synchronization between GPUs
- **Topology Awareness**: Optimizes for NVLink/PCIe connections

### 5. Validation Mechanisms

#### Core Saturation Validation
- Monitors SM (Streaming Multiprocessor) utilization
- Ensures all GPU cores are actively engaged
- Prevents claiming unused hardware

#### Anti-Spoofing Protection
- Cross-GPU timing validation
- Memory bandwidth consistency checks
- Execution pattern analysis

### 6. Attestation Generation

The attestor generates comprehensive hardware attestations:

```rust
pub struct GpuAttestation {
    pub attestation_id: String,
    pub executor_id: String,
    pub timestamp: i64,
    pub gpu_info: Vec<GpuInfo>,
    pub benchmarks: Option<GpuBenchmarkResults>,
    pub challenge_result: Option<ChallengeResult>,
    pub signature: Option<AttestationSignature>,
}
```

## Usage

### Basic Attestation Mode
```bash
# Generate full hardware attestation
./gpu-attestor --executor-id mynode --output ./attestation
```

### GPU PoW Challenge Mode
```bash
# Execute GPU PoW challenge from validator
./gpu-attestor --challenge <base64_encoded_challenge>
```

### GPU Detection Mode
```bash
# Detect and display all GPUs
./gpu-attestor --detect
```

## Security Features

1. **Deterministic Execution**: All operations use deterministic algorithms for reproducibility
2. **Cryptographic Signing**: Attestations signed with RSA-2048 or ECDSA
3. **Binary Integrity**: Self-checksum validation
4. **Anti-Tampering**: Multiple validation layers prevent result manipulation

## Performance Optimization

1. **CUDA Kernel Optimization**:
   - Tiled matrix multiplication for cache efficiency
   - Optimized thread block dimensions per GPU architecture
   - Coalesced memory access patterns

2. **Memory Management**:
   - Pre-allocated CUDA buffers
   - Zero-copy operations where possible
   - Efficient buffer swapping

3. **Multi-GPU Scaling**:
   - Concurrent kernel execution
   - Minimal synchronization overhead
   - Load balancing based on GPU capabilities

## Supported GPUs

The attestor supports all NVIDIA GPUs with:
- CUDA Compute Capability 3.0+
- Tested on: V100, A100, H100, RTX series
- Multi-GPU configurations (DGX, HGX systems)

## Build Requirements

- CUDA Toolkit 11.0+
- Rust 1.70+
- Linux (Ubuntu 20.04+ recommended)
- NVIDIA drivers 450.0+

## API Integration

The attestor can be integrated as a library:

```rust
use gpu_attestor::{
    GpuDetector,
    ChallengeHandler,
    AttestationBuilder,
};

// Detect GPUs
let detector = GpuDetector::detect_all()?;

// Execute challenge
let mut handler = ChallengeHandler::new()?;
let result = handler.execute_challenge(&challenge_b64).await?;

// Build attestation
let attestation = AttestationBuilder::new()
    .with_executor_id("node-1")
    .with_gpu_info(detector.to_gpu_infos())
    .with_challenge_result(result)
    .build()?;
```

## Monitoring and Diagnostics

The attestor provides detailed logging and metrics:
- GPU utilization per SM
- Memory bandwidth achieved
- Execution timing breakdowns
- Validation results

## Future Enhancements

1. **Additional PoW Algorithms**: Support for different challenge types
2. **Vulkan/OpenCL Support**: Cross-vendor GPU support
3. **Remote Attestation**: TEE integration for secure attestation
4. **Performance Profiling**: Detailed performance analytics

## Troubleshooting

### Common Issues

1. **CUDA Not Found**: Ensure CUDA toolkit is installed and in PATH
2. **No GPUs Detected**: Check NVIDIA drivers and CUDA compatibility
3. **Out of Memory**: Reduce matrix dimensions or number of matrices
4. **Permission Denied**: May need to run with elevated privileges for hardware access

### Debug Mode

Enable detailed logging:
```bash
RUST_LOG=debug ./gpu-attestor --challenge <challenge>
```

## Contributing

The GPU Attestor is part of the Basilisk network. Contributions should focus on:
- Security improvements
- Performance optimizations
- Additional GPU vendor support
- Enhanced validation mechanisms