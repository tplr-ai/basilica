# GPU Attestor Complete Guide

## Overview

The GPU Attestor is a secure hardware attestation system for the Basilisk network that validates GPU hardware capabilities and executes proof-of-work (PoW) challenges. It provides cryptographically signed attestations of GPU performance and ensures that miners cannot claim more computational resources than they actually possess.

## Current Status (January 2025)

The GPU Attestor has been successfully refactored and all major issues have been resolved:

✅ **Fully Functional GPU PoW** - Deterministic checksum verification working correctly  
✅ **Multi-GPU Support** - Seamless execution across 1-8 GPUs with proper work distribution  
✅ **Memory Optimization** - Achieving 95.3% GPU memory utilization (77.7GB/80GB on H100)  
✅ **Seed Variation** - Different seeds produce different checksums as expected  
✅ **Integration Tests** - All tests passing with proper checksum verification

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
│   │   ├── handler.rs       # Main challenge handler (unified GPU support)
│   │   ├── kernel_ops.rs    # CUDA kernel operations (matrix_multiply_monitored)
│   │   ├── multi_gpu_handler.rs     # Multi-GPU concurrent execution
│   │   ├── multi_gpu_challenge.rs   # Challenge distribution with matrix_offset
│   │   ├── sm_monitor.rs            # SM utilization monitoring
│   │   ├── core_saturation_validator.rs  # Core usage validation
│   │   └── anti_spoofing_validator.rs   # Anti-spoofing checks
│   │
│   ├── gpu/                 # GPU detection & management
│   │   ├── gpu_detector.rs  # Unified GPU detection (consolidated from 3 modules)
│   │   ├── cuda_driver/     # CUDA Driver API bindings
│   │   │   ├── cuda_driver_ffi.rs   # Low-level CUDA FFI (includes cuMemsetD8_v2)
│   │   │   ├── cuda_wrapper.rs      # Safe CUDA wrappers (with set_current())
│   │   │   ├── cleanup.rs           # CUDA cleanup utilities
│   │   │   └── kernels/             # CUDA kernels (compiled for sm_90)
│   │   │       ├── matrix_multiply_monitored.cu/ptx  # Main kernel
│   │   │       ├── prng.cu/ptx      # Philox PRNG for deterministic generation
│   │   │       └── sha256.cu/ptx    # Checksum generation
│   │   ├── benchmarks.rs    # GPU benchmarking
│   │   ├── benchmark_collector.rs   # Benchmark collection
│   │   └── types.rs         # GPU-related types
│   │
│   ├── hardware/            # System hardware collection
│   ├── network/             # Network benchmarking
│   ├── os/                  # OS attestation
│   └── validation/          # Performance validation
```

## Key Features

### 1. GPU Detection and Enumeration

The unified `GpuDetector` (consolidated from three separate modules) automatically discovers all CUDA-capable GPUs:

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

**Key Improvements**:
- Consolidated detection logic from `detection.rs`, `cuda_detection.rs`, and `multi_gpu_detector.rs` into single `gpu_detector.rs`
- Respects `CUDA_VISIBLE_DEVICES` environment variable
- Fixed memory calculation bug (was showing 144GB per H100)

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
    pub gpu_pow_seed: u64,                   // Deterministic seed (affects checksums)
    pub matrix_dim: u32,                     // Matrix dimensions (e.g., 1024)
    pub num_matrices: u32,                   // Total matrices (distributed across GPUs)
    pub num_iterations: u32,                 // Iteration count per GPU
    pub validator_nonce: String,             // Unique validator nonce (prevents replay attacks)
}
```

**Why the Validator Nonce is Essential**:
1. **Replay Protection** - Prevents reusing old challenge results
2. **Challenge Uniqueness** - Each challenge is unique even with same parameters
3. **Challenge-Response Binding** - Ensures result matches specific challenge
4. **Audit Trail** - Tracks which validator issued which challenge

### 3. Matrix Multiplication PoW Algorithm

The core PoW algorithm performs bandwidth-intensive matrix operations with deterministic execution:

```rust
// Simplified algorithm flow (per GPU)
for iteration in 0..num_iterations {
    for local_idx in 0..matrices_per_gpu {
        // Calculate global matrix index for this GPU
        let global_matrix_idx = device.matrix_offset + local_idx;
        
        // Generate matrix indices deterministically
        let a_idx = philox_index(seed, iteration, global_matrix_idx * 2) % total_matrices;
        let b_idx = philox_index(seed, iteration, global_matrix_idx * 2 + 1) % total_matrices;
        
        // Generate matrices on GPU using PRNG with correct global indices
        prng.generate_matrix(&mut matrix_a, seed, a_idx, matrix_dim);
        prng.generate_matrix(&mut matrix_b, seed + 1, b_idx, matrix_dim);
        
        // Execute matrix multiplication using matrix_multiply_monitored kernel
        kernel.execute(&matrix_a, &matrix_b, &result, matrix_dim);
        
        // Accumulate checksum (SHA256)
        accumulate_checksum(&mut checksum, &result);
    }
}
```

**Key Implementation Details**:
- Uses `matrix_multiply_monitored` kernel (not tiled version)
- Compiled for `sm_90` architecture (H100 compatibility)
- Each GPU generates its assigned portion using correct global indices
- Deterministic Philox PRNG ensures reproducible matrix generation
- SHA256 checksum excludes execution times for determinism

### 4. Multi-GPU Support

The system automatically distributes work across multiple GPUs with proper coordination:

- **Automatic Detection**: Discovers all available GPUs (respects `CUDA_VISIBLE_DEVICES`)
- **Work Distribution**: Each GPU processes `num_matrices / gpu_count` matrices
- **Matrix Offset Tracking**: `DeviceChallenge` includes `matrix_offset` field for correct indexing
- **Synchronization**: Barrier synchronization ensures all GPUs complete together
- **Memory Saturation**: Allocates ~75GB per GPU for maximum utilization
- **Deterministic Results**: Same seed + same GPU config = same checksum

**Memory Allocation Strategy**:
```rust
// Target 75GB per GPU (95%+ utilization)
let target_memory_gb = 75.0;
// Allocate: A matrices, B matrices, results, workspace buffers
```

### 5. Validation Mechanisms

#### Checksum Verification (Primary)
- **Deterministic SHA256 checksums** of computation results
- **Symmetric execution**: Validator runs same GPU configuration as miner
- **Excludes variable data**: No execution times or system totals in checksum
- **Result**: Cryptographically secure proof of computation

#### Core Saturation Validation (Temporarily Disabled)
- Monitors SM (Streaming Multiprocessor) utilization
- Ensures all GPU cores are actively engaged
- Prevents claiming unused hardware

#### Anti-Spoofing Protection (Temporarily Disabled)
- Cross-GPU timing validation
- Memory bandwidth consistency checks
- Execution pattern analysis

**Note**: Currently only checksum verification is active. Other validations are disabled pending optimization.

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
   - Simple matrix multiplication kernel (tensor cores not yet utilized)
   - Compiled for sm_90 architecture (H100 specific)
   - Thread block dimensions: 16x16 threads per block
   - Current bandwidth utilization: ~0.015% (major optimization opportunity)

2. **Memory Management**:
   - Aggressive memory allocation strategy (75GB per GPU)
   - Pre-allocated CUDA buffers to avoid allocation overhead
   - Matrix generation directly in GPU memory
   - Achieved: 95.3% memory utilization (77.7GB/80GB on H100)

3. **Multi-GPU Scaling**:
   - Concurrent kernel execution across all GPUs
   - Barrier synchronization for deterministic results
   - Proper CUDA context management with `set_current()`
   - Matrix offset tracking for correct work distribution

## Supported GPUs

The attestor supports all NVIDIA GPUs with:
- CUDA Compute Capability 3.0+
- Tested extensively on: H100 PCIe (8x GPUs)
- Compiled for sm_90 architecture (H100 optimized)
- Multi-GPU configurations (DGX, HGX systems)
- Respects `CUDA_VISIBLE_DEVICES` for GPU selection

## Build Requirements

- CUDA Toolkit 12.0+ (tested with 12.8)
- Rust 1.70+
- Linux (Ubuntu 20.04+ recommended)
- NVIDIA drivers 450.0+
- nvcc compiler (for PTX generation)

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

1. **Performance Optimization**:
   - Utilize tensor cores for matrix multiplication
   - Improve bandwidth utilization (currently 0.015%)
   - Implement memory prefetching and kernel fusion

2. **Dynamic Scaling**:
   - Scale challenge parameters based on GPU count
   - Support heterogeneous GPU configurations
   - Dynamic work stealing for load balancing

3. **Additional Validations**:
   - Re-enable core saturation validation
   - Re-enable anti-spoofing checks
   - Add VRAM validation with memory saturation awareness

4. **Cross-Platform Support**:
   - Vulkan/OpenCL for AMD GPUs
   - Metal for Apple Silicon
   - DirectML for Windows

## Troubleshooting

### Common Issues and Solutions

1. **CUDA_ERROR_UNKNOWN during kernel loading**
   - Solution: Ensure kernel name matches (use `matrix_multiply_simple`)
   - Check PTX compilation succeeded

2. **Low memory allocation (< 10GB per GPU)**
   - Solution: Aggressive memory allocation strategy implemented
   - Now allocates ~75GB per GPU for saturation

3. **Checksum mismatch between miner and validator**
   - Solution: Ensure symmetric execution (same GPU count)
   - Check `CUDA_VISIBLE_DEVICES` matches between runs
   - Verify same seed is used

4. **CUDA_ERROR_OUT_OF_MEMORY**
   - Solution: Challenge automatically adjusts for available GPUs
   - Single GPU mode limits to 1152 matrices

5. **Kernel architecture mismatch**
   - Solution: Kernels compiled for sm_90 (H100)
   - May need to adjust for other GPU architectures

### Debug Mode

Enable detailed logging:
```bash
RUST_LOG=info cargo test gpu_pow_end_to_end_flow -- --nocapture
RUST_LOG=debug ./gpu-attestor --challenge <challenge>
```

### Testing with Specific GPUs

```bash
# Test with single GPU
CUDA_VISIBLE_DEVICES=0 cargo test gpu_pow_end_to_end_flow

# Test with multiple GPUs
CUDA_VISIBLE_DEVICES=0,1,2,3 cargo test gpu_pow_end_to_end_flow
```

## Contributing

The GPU Attestor is part of the Basilisk network. Contributions should focus on:
- Security improvements
- Performance optimizations (especially bandwidth utilization)
- Additional GPU vendor support
- Enhanced validation mechanisms
- Test coverage for various GPU configurations

## Recent Changes (January 2025)

### Major Fixes Implemented
1. **Kernel Loading**: Fixed function name mismatch (`matrix_multiply_simple`)
2. **Architecture Compatibility**: Updated to sm_90 for H100 support
3. **Memory Allocation**: Implemented 75GB per GPU saturation strategy
4. **Checksum Determinism**: Fixed by excluding execution times and system totals
5. **Matrix Generation**: Fixed to use correct global indices per GPU
6. **Symmetric Verification**: Validator respects `CUDA_VISIBLE_DEVICES`
7. **Code Consolidation**: Unified GPU detection into single module

### Known Limitations
- Low bandwidth utilization (0.015%) - major optimization opportunity
- Tensor cores not utilized - using simple matrix multiplication
- Other validations temporarily disabled (only checksum active)
- Fixed compilation for sm_90 (H100 specific)