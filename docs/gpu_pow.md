# GPU Proof-of-Work System

## Overview

The GPU Proof-of-Work (PoW) system in Basilisk ensures that miners actually possess the GPU hardware they claim to have. It works by issuing computational challenges that can only be completed efficiently on actual GPU hardware.

## How It Works

### 1. Challenge Generation (Validator Side)

The validator generates a challenge with the following parameters:
- **Random Seed**: A 64-bit random number that prevents precomputation
- **Matrix Dimension**: Size of square matrices (e.g., 256x256)
- **Number of Matrices**: How many matrices to generate (targets ~90% VRAM usage)
- **Matrix Indices**: Which two matrices to multiply (randomly selected)
- **Validator Nonce**: Unique identifier for replay protection

```rust
// Example challenge parameters
ChallengeParameters {
    challenge_type: "matrix_multiplication_pow",
    gpu_pow_seed: 12345678901234567890,
    matrix_dim: 256,
    num_matrices: 1000,
    matrix_a_index: 42,
    matrix_b_index: 873,
    validator_nonce: "val_9876543210",
}
```

### 2. Challenge Execution (Miner Side)

The miner's `gpu-attestor` executes the challenge:
1. **Matrix Generation**: Uses Philox4x32_10 PRNG with the seed to generate all matrices on GPU
2. **Matrix Selection**: Extracts the two specified matrices
3. **Matrix Multiplication**: Performs C = A × B using CUDA
4. **Checksum Computation**: Calculates SHA-256 of the result matrix on GPU
5. **Result Submission**: Returns checksum, execution time, and GPU info

### 3. Verification (Validator Side)

The validator verifies the result by:
1. **GPU Model Check**: Only validates if miner's GPU matches validator's GPU
2. **Local Execution**: Runs the exact same challenge on validator's GPU
3. **Checksum Comparison**: Verifies the checksums match exactly
4. **Timing Validation**: Ensures execution time is reasonable (within 1.5x of local time)
5. **VRAM Usage Check**: Confirms appropriate VRAM allocation

## Security Properties

### Prevents Precomputation
- Random seed makes it impossible to precompute results
- Seed is combined with matrix index for deterministic generation
- Each challenge is unique and unpredictable

### Requires Actual GPU
- No CPU fallback - GPU is mandatory
- Matrix operations are optimized for GPU parallelism
- Large memory requirements (GB of VRAM) make CPU execution impractical

### Same-Type Validation
- Validators can only verify GPUs they actually possess
- H100 validators verify H100 miners
- Prevents false claims about GPU models

### Timing Constraints
- Execution must complete within reasonable time
- Too fast indicates potential cheating
- Too slow indicates inferior hardware

## Implementation Details

### GPU Kernels

1. **PRNG Kernel** (`prng.cu`):
   - Philox4x32_10 algorithm for deterministic random generation
   - Generates matrices directly in GPU memory
   - Seed + matrix_index ensures unique matrices

2. **SHA-256 Kernel** (`sha256.cu`):
   - Parallel SHA-256 computation
   - Processes matrix in chunks for efficiency
   - Produces 32-byte checksum

3. **Matrix Multiply Kernel** (`matrix_multiply.cu`):
   - Standard CUDA matrix multiplication
   - Uses shared memory for performance
   - Handles large matrices efficiently

### Challenge Protocol

```
Validator                           Miner
    |                                 |
    |-------- Challenge Params ------>|
    |         (seed, indices)         |
    |                                 |
    |                          Execute on GPU
    |                          - Generate matrices
    |                          - Multiply A × B
    |                          - Compute SHA-256
    |                                 |
    |<-------- Challenge Result ------|
    |     (checksum, time, GPU)       |
    |                                 |
    Execute locally                    |
    Compare results                    |
    Verify timing                      |
    |                                 |
```

## Usage

### Running a Test Challenge

```bash
# Build gpu-attestor
cargo build --bin gpu-attestor

# Check GPU availability
./target/debug/gpu-attestor --info

# Run test script
./scripts/test_gpu_pow.sh
```

### Integration Test

```bash
# Run integration test (requires GPU)
cd crates/validator
cargo test gpu_pow_integration -- --ignored --nocapture
```

## Performance Characteristics

For an NVIDIA H100 with 80GB VRAM:
- Matrix dimension: 256×256
- Number of matrices: ~9,700 (using 90% of VRAM)
- Execution time: ~50-200ms depending on matrix indices
- Memory bandwidth: ~2TB/s utilized

## Limitations

1. **Same GPU Requirement**: Validators must have the same GPU model as miners
2. **CUDA Only**: Currently supports only NVIDIA GPUs with CUDA
3. **Memory Bound**: Performance limited by memory bandwidth
4. **No Partial Verification**: Cannot verify GPUs validator doesn't possess

## Future Improvements

1. **Multi-GPU Support**: Verify systems with multiple GPUs
2. **Cross-Model Verification**: Use performance ratios for different GPU models
3. **AMD ROCm Support**: Extend beyond NVIDIA GPUs
4. **Optimized Kernels**: Further optimize CUDA kernels for specific GPU architectures