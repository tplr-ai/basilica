# GPU Attestor System: End-to-End Architecture and Flow

## Overview

The GPU Attestor is a sophisticated system designed to verify GPU computational work through cryptographic proof-of-work challenges. It ensures that GPUs are performing genuine calculations and not spoofing results, while supporting both single-GPU and multi-GPU configurations.

## System Components

### 1. Core Components

#### GPU Attestor Binary (`gpu-attestor`)
- **Location**: `crates/gpu-attestor/src/main.rs`
- **Purpose**: Main executable that runs on miner nodes to execute GPU challenges
- **Key Functions**:
  - Hardware detection (GPU enumeration and capability assessment)
  - Challenge execution (matrix multiplication proof-of-work)
  - Result generation with cryptographic checksums
  - Multi-GPU orchestration

#### Validator Component
- **Location**: `crates/validator/src/validation/gpu_validator.rs`
- **Purpose**: Verifies challenge results on validator nodes
- **Key Functions**:
  - Challenge parameter generation
  - Statistical sampling verification
  - Checksum validation
  - Execution time verification

### 2. GPU Detection and Initialization

#### Multi-GPU Detector
- **Location**: `crates/gpu-attestor/src/gpu/multi_gpu_detector.rs`
- **Flow**:
  1. Enumerates all CUDA-capable devices using `cuDeviceGetCount`
  2. For each device:
     - Queries device properties (name, compute capability, memory)
     - Detects SM (Streaming Multiprocessor) count
     - Checks NVLink connectivity for multi-GPU topologies
  3. Creates system capability profile

#### CUDA Context Management
- **Location**: `crates/gpu-attestor/src/gpu/cuda_driver/`
- **Process**:
  1. Initializes CUDA driver API
  2. Creates context for each GPU
  3. Loads PTX kernels (compiled CUDA code)
  4. Manages memory allocation and transfers

## End-to-End Challenge Flow

### Phase 1: Challenge Generation (Validator Side)

1. **Validator Initialization**
   ```
   Validator → GpuValidator::initialize()
            → Detects validator's GPU model
            → Sets verification parameters
   ```

2. **Challenge Creation**
   ```
   Validator → ChallengeGenerator::generate_challenge()
            → Creates ChallengeParameters:
                - challenge_type: "matrix_multiplication_pow"
                - gpu_pow_seed: Random seed for deterministic matrix generation
                - matrix_dim: Matrix dimensions (e.g., 1024x1024)
                - num_matrices: Number of matrices to process
                - num_iterations: Iteration count for bandwidth testing
                - verification_sample_rate: Sampling rate (e.g., 0.1 for 10%)
                - validator_nonce: Unique challenge identifier
   ```

3. **Challenge Distribution**
   ```
   Validator → Serializes challenge to JSON
            → Base64 encodes
            → Sends to miner
   ```

### Phase 2: Challenge Execution (Miner Side)

1. **Challenge Reception**
   ```
   Miner → Receives base64-encoded challenge
        → gpu-attestor --challenge <base64_challenge>
        → ChallengeHandler::execute_challenge()
   ```

2. **Multi-GPU Detection and Setup**
   ```
   ChallengeHandler → Detects available GPUs
                   → If multi-GPU system:
                      → Creates MultiGpuChallengeHandler
                      → Initializes SM monitoring
                      → Sets up synchronization barriers
   ```

3. **Challenge Distribution (Multi-GPU)**
   ```
   MultiGpuChallengeHandler → Analyzes challenge parameters
                           → Distributes work across GPUs:
                              - Each GPU gets portion of matrices
                              - Synchronization points established
                              - Anti-spoofing measures activated
   ```

4. **Kernel Execution Flow**

   a. **Matrix Generation**
   ```
   For each GPU:
     → Load PRNG kernel (Philox4x32_10 algorithm)
     → Generate matrices deterministically from seed
     → Matrices stored in GPU memory
   ```

   b. **Matrix Multiplication with SM Monitoring**
   ```
   For each iteration:
     → Launch monitored kernel (matrix_multiply_monitored)
     → Kernel tracks:
        - SM utilization per multiprocessor
        - Thread participation counts
        - Block execution counts
     → Performs tiled matrix multiplication
     → Atomic operations track SM activity
   ```

   c. **Synchronization (Multi-GPU)**
   ```
   Barrier synchronization points:
     1. Start barrier - all GPUs begin together
     2. Mid barrier - anti-sharing validation
     3. End barrier - completion synchronization
   ```

5. **Result Generation**
   ```
   For each GPU:
     → Compute SHA256 checksum of final matrix
     → Collect performance metrics:
        - Execution time
        - Memory bandwidth utilization
        - SM utilization percentages
        - Compute throughput (TFLOPS)
   
   Global result:
     → Combine individual GPU checksums
     → Generate global checksum
     → Create metadata JSON with all metrics
   ```

### Phase 3: Result Verification (Validator Side)

1. **Sampled Verification**
   ```
   Validator → Receives ChallengeResult from miner
            → Extracts sampled_checksum from metadata
            → Generates same sample indices:
               - Uses same seed + nonce
               - Deterministic sampling algorithm
               - Typically 10% of iterations
   ```

2. **Local Execution**
   ```
   Validator → Executes only sampled iterations
            → gpu-attestor --challenge-sampled
            → Computes checksum for sampled work
            → Compares with miner's sampled checksum
   ```

3. **Validation Checks**
   ```
   Verification passes if:
     1. Sampled checksums match exactly
     2. Execution time is reasonable (within bounds)
     3. GPU model matches validator's GPU
     4. Memory usage matches expected values
     5. All metadata is consistent
   ```

## Technical Implementation Details

### CUDA Kernel Architecture

1. **Matrix Multiplication Kernel**
   - **Tiled algorithm**: 16x16 tile size for cache efficiency
   - **Shared memory**: Used for tile caching
   - **Grid configuration**: Dynamically sized based on matrix dimensions

2. **SM Monitoring Implementation**
   ```cuda
   // Get SM ID for current thread block
   unsigned int sm_id;
   asm volatile("mov.u32 %0, %%smid;" : "=r"(sm_id));
   
   // Track utilization
   atomicAdd(&sm_utilization_counter[sm_id], 1ULL);
   ```

3. **Memory Management**
   - **Single matrix buffers**: Avoids large allocations
   - **Buffer swapping**: Reuses memory between iterations
   - **Pinned memory**: For efficient CPU-GPU transfers

### Anti-Spoofing Measures

1. **SM Utilization Validation**
   - Ensures all SMs are active (>85% utilization)
   - Detects uneven work distribution
   - Validates execution patterns

2. **Cross-GPU Validation**
   - Synchronized execution timestamps
   - Bandwidth utilization consistency
   - Prevents result sharing between GPUs

3. **Timing Analysis**
   - Execution time must match hardware capabilities
   - Detects pre-computation attempts
   - Validates against theoretical limits

### Fallback Mechanisms

1. **SM Monitoring Fallback**
   - If %smid instruction fails, uses synthetic distribution
   - Maintains functionality on older architectures
   - Logs warnings for degraded monitoring

2. **Single-GPU Fallback**
   - Multi-GPU handler can operate with one GPU
   - Automatic detection and adaptation
   - No code changes required

## Performance Characteristics

### Typical Execution Times (H100 GPU)
- **Matrix Size**: 1024x1024 (double precision)
- **Single GPU**: ~1-2 seconds per 1000 iterations
- **8x GPU**: ~10-12 seconds for 2000 iterations per GPU
- **Bandwidth**: 0.1-1% utilization (compute-bound workload)

### Memory Usage
- **Per Matrix**: 8MB (1024x1024 doubles)
- **Total for 8 GPUs**: ~512GB across all devices
- **Per GPU**: ~64GB for typical multi-GPU challenge

### Verification Overhead
- **Sampling Rate**: 10% of iterations
- **Verification Time**: ~1.5 seconds for 200 samples
- **Overhead**: <15% compared to full execution

## Configuration and Deployment

### Environment Variables
- `GPU_ATTESTOR_PATH`: Override default binary location
- `GPU_COUNT`: Expected GPU count for testing
- `RUST_LOG`: Logging level (info/debug/trace)

### Build Requirements
- CUDA Toolkit 11.0+
- NVIDIA driver 450.0+
- Rust 1.70+
- C++ compiler for CUDA

### Compilation Process
```bash
# Build with CUDA kernels
cargo build --bin gpu-attestor --release

# Kernels are compiled by build.rs:
# - matrix_multiply.cu → matrix_multiply.ptx
# - matrix_multiply_monitored.cu → matrix_multiply_monitored.ptx
# - prng.cu → prng.ptx
# - sha256.cu → sha256.ptx
```

## Error Handling and Recovery

1. **GPU Initialization Failures**
   - Attempts alternate GPU if primary fails
   - Falls back to CPU verification if available
   - Comprehensive error reporting

2. **Memory Allocation Failures**
   - Reduces matrix count dynamically
   - Implements incremental allocation
   - Cleans up on failure

3. **Kernel Execution Failures**
   - Retries with different parameters
   - Falls back to non-monitored kernels
   - Maintains result integrity

## Security Considerations

1. **Cryptographic Integrity**
   - SHA256 checksums prevent tampering
   - Deterministic execution ensures reproducibility
   - Nonce prevents replay attacks

2. **Resource Isolation**
   - Each GPU has dedicated context
   - Memory isolation between executions
   - No shared state between challenges

3. **Verification Trust Model**
   - Validator must have same GPU model
   - Statistical sampling reduces computation
   - Cryptographic proofs ensure correctness

## Future Enhancements

1. **Planned Features**
   - Support for AMD GPUs (ROCm)
   - Dynamic difficulty adjustment
   - Network bandwidth verification
   - CPU+GPU hybrid challenges

2. **Optimization Opportunities**
   - Kernel fusion for reduced overhead
   - Adaptive sampling rates
   - Memory pool management
   - Multi-node coordination

## Debugging and Monitoring

### Key Log Messages
```
INFO: "Executing unified GPU challenge (single or multi-GPU)"
INFO: "Executing N iterations with SM monitoring"
WARN: "All SM monitoring counters are zero - using fallback"
INFO: "Challenge verified successfully with statistical sampling"
```

### Performance Metrics
- Execution time per GPU
- Memory bandwidth utilization
- SM utilization distribution
- Compute throughput (TFLOPS)
- Synchronization overhead

### Common Issues
1. **SM monitoring returns zeros**: Usually due to architecture incompatibility
2. **Low bandwidth utilization**: Normal for compute-bound workloads
3. **Synchronization timeouts**: Check GPU peer access configuration
4. **Memory allocation failures**: Reduce matrix count or size