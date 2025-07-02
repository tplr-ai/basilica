#!/bin/bash
# Test H100 configuration with integration tests

echo "Testing H100 GPU PoW configuration..."
echo "Expected: 1024x1024 matrices, 9216 matrices, ~72GB memory"
echo ""

# Run a single GPU PoW test with specific parameters
cargo test -p integration-tests gpu_pow_end_to_end_flow -- --nocapture 2>&1 | grep -E "(Challenge params|Memory usage|matrix_dim|num_matrices|VRAM)"