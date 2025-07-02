# GPU Attestation Flow Documentation

## Overview

The GPU attestation system in Basilisk ensures that miners possess the GPU hardware they claim to have. It uses a challenge-response protocol where validators issue computational challenges that can only be completed efficiently on actual GPU hardware.

## System Architecture

```
┌─────────────┐                    ┌─────────────┐
│  Validator  │                    │    Miner    │
│             │                    │             │
│ ┌─────────┐ │                    │ ┌─────────┐ │
│ │Challenge│ │                    │ │Executor │ │
│ │Generator│ │                    │ │   GPU   │ │
│ └─────────┘ │                    │ └─────────┘ │
│             │                    │             │
│ ┌─────────┐ │                    │ ┌─────────┐ │
│ │   GPU   │ │                    │ │   gpu-  │ │
│ │Validator│ │                    │ │attestor │ │
│ └─────────┘ │                    │ └─────────┘ │
└─────────────┘                    └─────────────┘
```

## Detailed Flow

### Phase 1: Challenge Generation (Validator)

1. **Initialize Validator GPU**
   ```rust
   let mut gpu_validator = GpuValidator::new(gpu_attestor_path);
   gpu_validator.initialize().await?;
   ```
   - Runs gpu-attestor to detect validator's GPU model
   - Stores GPU information for later verification

2. **Generate Challenge Parameters**
   ```rust
   let challenge = ChallengeGenerator::new()
       .generate_challenge(gpu_model, vram_gb, nonce)?;
   ```
   
   Creates `ChallengeParameters` containing:
   - `challenge_type`: "matrix_multiplication_pow"
   - `gpu_pow_seed`: Random 64-bit seed
   - `matrix_dim`: Matrix dimension (default 256)
   - `num_matrices`: Number to fill ~90% VRAM (capped at 10,000)
   - `matrix_a_index`: Random index for first operand
   - `matrix_b_index`: Random index for second operand
   - `validator_nonce`: Unique ID for replay protection

### Phase 2: Challenge Transmission

1. **Serialize Challenge**
   ```bash
   challenge_json = serialize_to_json(challenge)
   challenge_b64 = base64_encode(challenge_json)
   ```

2. **Send to Miner**
   - Via SSH: `gpu-attestor --challenge <base64_challenge>`
   - Via gRPC: Future implementation
   - Via HTTP API: Alternative transport

### Phase 3: Challenge Execution (Miner/Executor)

1. **Parse Challenge**
   ```rust
   let challenge: ChallengeParameters = decode_and_parse(challenge_b64)?;
   ```

2. **Allocate GPU Memory**
   ```rust
   let buffer_size = num_matrices * matrix_dim * matrix_dim * sizeof(f64);
   let matrix_buffer = CudaBuffer::new(buffer_size)?;
   ```
   - Fails if insufficient VRAM available
   - Proves miner has claimed memory capacity

3. **Generate Matrices on GPU**
   ```cuda
   // CUDA kernel execution
   generate_matrices_philox<<<blocks, threads>>>(
       matrices, seed, num_matrices, matrix_dim
   );
   ```
   - Uses Philox4x32_10 PRNG with seed
   - Deterministic: same seed → same matrices
   - Direct GPU generation (no CPU transfer)

4. **Perform Matrix Multiplication**
   ```rust
   let result = multiply_f64(
       matrix_a_data,  // matrices[matrix_a_index]
       matrix_b_data,  // matrices[matrix_b_index]
       matrix_dim
   )?;
   ```

5. **Compute SHA-256 Checksum**
   ```cuda
   // CUDA kernel execution
   sha256_matrix_chunk<<<blocks, threads>>>(
       result_matrix, chunk_hashes, matrix_dim
   );
   sha256_combine_chunks<<<1, 1>>>(
       chunk_hashes, final_hash, num_chunks
   );
   ```
   - Parallel chunk processing
   - Final 32-byte hash computed on GPU

6. **Return Result**
   ```json
   {
     "result_checksum": "3e0f147355f59a2b2bc214a60e5c028107c3751660b6f11bfdbe35a72f667626",
     "execution_time_ms": 456,
     "gpu_model": "NVIDIA H100 PCIe",
     "vram_allocated_mb": 5000,
     "challenge_id": "validator_nonce_123",
     "success": true
   }
   ```

### Phase 4: Verification (Validator)

1. **Receive Result**
   ```rust
   let result: ChallengeResult = parse_miner_response(stdout)?;
   ```

2. **Check GPU Model**
   ```rust
   if result.gpu_model != validator.gpu_model {
       return Ok(false); // Cannot verify different GPU types
   }
   ```

3. **Execute Challenge Locally**
   ```rust
   // Validator runs exact same computation
   let local_result = execute_challenge_locally(&challenge)?;
   ```

4. **Compare Checksums**
   ```rust
   if result.result_checksum != local_result.checksum {
       return Ok(false); // Computation mismatch
   }
   ```

5. **Validate Timing**
   ```rust
   let time_ratio = result.execution_time_ms / local_result.execution_time_ms;
   if time_ratio > max_allowed_ratio {
       return Ok(false); // Too slow (inferior hardware)
   }
   ```

6. **Check Nonce**
   ```rust
   if result.challenge_id != challenge.validator_nonce {
       return Ok(false); // Replay attack or wrong challenge
   }
   ```

## Security Properties

### 1. Prevents Precomputation
- Random seed makes results unpredictable
- Matrix indices randomly selected
- Validator nonce prevents replay attacks

### 2. Requires Actual GPU
- No CPU fallback implemented
- Memory allocation proves VRAM capacity
- Computation time validates GPU performance

### 3. Deterministic Verification
- Same inputs always produce same outputs
- Validator can replicate exact computation
- Checksum provides cryptographic proof

### 4. Same-Type Validation
- Validators can only verify GPUs they possess
- Prevents false claims about GPU models
- Ensures accurate performance validation

## Data Structures

### ChallengeParameters (Protobuf)
```protobuf
message ChallengeParameters {
    string challenge_type = 1;
    string parameters_json = 2;  // Legacy, unused
    uint32 expected_duration_seconds = 3;
    uint32 difficulty_level = 4;
    string seed = 5;  // Legacy string format
    MachineInfo machine_info = 6;
    uint64 gpu_pow_seed = 7;  // Actual seed used
    uint32 matrix_dim = 8;
    uint32 num_matrices = 9;
    uint32 matrix_a_index = 10;
    uint32 matrix_b_index = 11;
    string validator_nonce = 12;
}
```

### ChallengeResult (Protobuf)
```protobuf
message ChallengeResult {
    string solution = 1;  // Unused for GPU PoW
    uint64 execution_time_ms = 2;
    repeated double gpu_utilization = 3;
    uint64 memory_usage_mb = 4;
    string error_message = 5;
    string metadata_json = 6;
    bytes result_checksum = 7;  // 32-byte SHA-256
    bool success = 8;
    string gpu_model = 9;
    uint64 vram_allocated_mb = 10;
    string challenge_id = 11;  // Must match validator_nonce
}
```

## Error Handling

### Miner-Side Errors
- **Insufficient VRAM**: Cannot allocate required memory
- **No GPU**: CUDA initialization fails
- **Kernel Launch Failure**: GPU computation errors
- **Invalid Challenge**: Malformed parameters

### Validator-Side Errors
- **No GPU Detected**: Validator cannot initialize
- **Execution Timeout**: Miner takes too long
- **Connection Failure**: Cannot reach miner
- **Invalid Result**: Malformed response

## Performance Considerations

### Memory Usage
- Matrix size: `N × N × 8 bytes` per matrix
- Total VRAM: `num_matrices × matrix_size`
- Example: 10,000 × 256×256 × 8 = 5GB

### Computation Time
- Matrix generation: O(num_matrices × N²)
- Matrix multiplication: O(N³)
- SHA-256: O(N²)
- Total: Dominated by matrix multiplication

### Network Overhead
- Challenge size: ~200 bytes
- Result size: ~300 bytes
- Minimal bandwidth requirement

## Integration Points

### 1. Validator Service
- `HardwareValidator`: Main validation orchestrator
- `ChallengeGenerator`: Creates challenges
- `GpuValidator`: Executes and verifies

### 2. Miner Service
- Executor registration and discovery
- Challenge routing to executors
- Result aggregation

### 3. gpu-attestor Binary
- Standalone executable on executor
- Handles challenge execution
- Returns JSON results

### 4. Scoring System
- Successful validation increases score
- Failed validation decreases score
- Timing affects performance rating

## Future Enhancements

1. **Cross-Model Verification**
   - Use performance ratios between GPU models
   - Enable validators to verify different GPUs

2. **Multi-GPU Support**
   - Distribute computation across multiple GPUs
   - Verify total system capacity

3. **Alternative Algorithms**
   - Support AMD ROCm
   - Add Intel GPU support
   - Implement WebGPU fallback

4. **Advanced Challenges**
   - Variable difficulty levels
   - Different computational patterns
   - Memory bandwidth tests