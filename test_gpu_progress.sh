#!/bin/bash
# Test script to run gpu-attestor with reduced workload and visible logs

# Create a test challenge with reduced iterations (10 instead of 4200)
CHALLENGE_JSON='{
  "challenge_type": "matrix_multiplication_pow",
  "parameters_json": "",
  "expected_duration_seconds": 10,
  "difficulty_level": 1,
  "seed": "10225113323962251674",
  "machine_info": null,
  "gpu_pow_seed": 10225113323962251674,
  "matrix_dim": 1024,
  "num_matrices": 9216,
  "matrix_a_index": 0,
  "matrix_b_index": 0,
  "validator_nonce": "test_progress_logging",
  "num_iterations": 10,
  "verification_sample_rate": 0.0
}'

# Base64 encode the challenge
CHALLENGE_B64=$(echo "$CHALLENGE_JSON" | base64 -w 0)

# Run gpu-attestor with RUST_LOG=info to see our logging
echo "Running GPU attestor with reduced workload (10 iterations instead of 4200)..."
echo "This should complete in ~10 seconds and show progress logging..."
echo ""

RUST_LOG=info /home/shadeform/bas-2/target/debug/gpu-attestor --challenge "$CHALLENGE_B64"