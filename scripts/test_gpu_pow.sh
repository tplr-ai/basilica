#!/bin/bash
# Test script for GPU PoW end-to-end flow

set -e

echo "=== GPU PoW End-to-End Test ==="
echo ""

# Colors for output
GREEN='\033[0;32m'
RED='\033[0;31m'
YELLOW='\033[1;33m'
NC='\033[0m' # No Color

# Check if gpu-attestor binary exists
GPU_ATTESTOR_PATH="./target/debug/gpu-attestor"
if [ ! -f "$GPU_ATTESTOR_PATH" ]; then
    echo -e "${YELLOW}Building gpu-attestor...${NC}"
    cargo build --bin gpu-attestor
fi

# Step 1: Check GPU availability
echo -e "${YELLOW}Step 1: Checking GPU availability...${NC}"
# Run attestation to check if GPU is available
$GPU_ATTESTOR_PATH --skip-network-benchmark --skip-os-attestation --skip-docker-attestation --output gpu_test > gpu_detection.log 2>&1

if [ $? -eq 0 ]; then
    echo -e "${GREEN}✓ GPU detected${NC}"
    # Try to extract GPU info from attestation report
    if [ -f "gpu_test.json" ]; then
        cat gpu_test.json | jq -r '.report.gpu_info[0].name' 2>/dev/null || echo "GPU info saved to gpu_test.json"
    fi
else
    echo -e "${RED}✗ No GPU detected or gpu-attestor failed${NC}"
    echo "Error log:"
    tail -10 gpu_detection.log
    exit 1
fi

# Step 2: Create a test challenge
echo ""
echo -e "${YELLOW}Step 2: Creating test challenge...${NC}"

# Create a challenge JSON similar to what validator would generate
CHALLENGE_JSON=$(cat <<EOF
{
  "challenge_type": "matrix_multiplication_pow",
  "parameters_json": "",
  "expected_duration_seconds": 10,
  "difficulty_level": 1,
  "seed": "12345",
  "machine_info": null,
  "gpu_pow_seed": 12345,
  "matrix_dim": 256,
  "num_matrices": 100,
  "matrix_a_index": 42,
  "matrix_b_index": 73,
  "validator_nonce": "test_nonce_123"
}
EOF
)

# Base64 encode the challenge
CHALLENGE_B64=$(echo -n "$CHALLENGE_JSON" | base64 -w 0)
echo "Challenge created with:"
echo "  - Matrix dimension: 256x256"
echo "  - Number of matrices: 100"
echo "  - Seed: 12345"
echo "  - Matrix indices: A=42, B=73"

# Step 3: Execute the challenge
echo ""
echo -e "${YELLOW}Step 3: Executing GPU PoW challenge...${NC}"

START_TIME=$(date +%s.%N)
$GPU_ATTESTOR_PATH --challenge "$CHALLENGE_B64" > challenge_result.json 2>challenge_error.log
CHALLENGE_EXIT_CODE=$?
END_TIME=$(date +%s.%N)

if [ $CHALLENGE_EXIT_CODE -eq 0 ]; then
    EXECUTION_TIME=$(echo "$END_TIME - $START_TIME" | bc)
    echo -e "${GREEN}✓ Challenge completed in ${EXECUTION_TIME} seconds${NC}"
    
    # Parse result
    if command -v jq &> /dev/null; then
        echo ""
        echo "Challenge Result:"
        echo "  - Checksum: $(jq -r '.result_checksum' challenge_result.json 2>/dev/null || echo 'See challenge_result.json')"
        echo "  - Execution time: $(jq -r '.execution_time_ms' challenge_result.json 2>/dev/null || echo 'See challenge_result.json') ms"
        echo "  - VRAM allocated: $(jq -r '.vram_allocated_mb' challenge_result.json 2>/dev/null || echo 'See challenge_result.json') MB"
        echo "  - GPU model: $(jq -r '.gpu_model' challenge_result.json 2>/dev/null || echo 'See challenge_result.json')"
    else
        echo "Result saved to challenge_result.json"
    fi
else
    echo -e "${RED}✗ Challenge failed (exit code: $CHALLENGE_EXIT_CODE)${NC}"
    echo "Error log:"
    cat challenge_error.log
    exit 1
fi

# Step 4: Verify reproducibility
echo ""
echo -e "${YELLOW}Step 4: Verifying reproducibility...${NC}"

# Run the same challenge again
$GPU_ATTESTOR_PATH --challenge "$CHALLENGE_B64" > challenge_result2.json 2>/dev/null

if [ $? -eq 0 ]; then
    # Compare checksums
    if command -v jq &> /dev/null; then
        CHECKSUM1=$(jq -r '.result_checksum' challenge_result.json 2>/dev/null || echo "")
        CHECKSUM2=$(jq -r '.result_checksum' challenge_result2.json 2>/dev/null || echo "")
        
        if [ "$CHECKSUM1" = "$CHECKSUM2" ] && [ -n "$CHECKSUM1" ]; then
            echo -e "${GREEN}✓ Checksums match - deterministic execution verified${NC}"
        else
            echo -e "${RED}✗ Checksums don't match!${NC}"
            echo "  First:  $CHECKSUM1"
            echo "  Second: $CHECKSUM2"
        fi
    else
        echo "Install jq to verify checksums automatically"
    fi
else
    echo -e "${RED}✗ Second challenge execution failed${NC}"
fi

# Step 5: Test with different parameters
echo ""
echo -e "${YELLOW}Step 5: Testing with larger challenge...${NC}"

# Create a larger challenge
LARGE_CHALLENGE_JSON=$(cat <<EOF
{
  "challenge_type": "matrix_multiplication_pow",
  "parameters_json": "",
  "expected_duration_seconds": 10,
  "difficulty_level": 1,
  "seed": "98765",
  "machine_info": null,
  "gpu_pow_seed": 98765,
  "matrix_dim": 512,
  "num_matrices": 50,
  "matrix_a_index": 10,
  "matrix_b_index": 40,
  "validator_nonce": "test_nonce_456"
}
EOF
)

LARGE_CHALLENGE_B64=$(echo -n "$LARGE_CHALLENGE_JSON" | base64 -w 0)

echo "Testing with 512x512 matrices..."
START_TIME=$(date +%s.%N)
$GPU_ATTESTOR_PATH --challenge "$LARGE_CHALLENGE_B64" > large_challenge_result.json 2>/dev/null
if [ $? -eq 0 ]; then
    END_TIME=$(date +%s.%N)
    EXECUTION_TIME=$(echo "$END_TIME - $START_TIME" | bc)
    echo -e "${GREEN}✓ Large challenge completed in ${EXECUTION_TIME} seconds${NC}"
else
    echo -e "${RED}✗ Large challenge failed${NC}"
fi

echo ""
echo -e "${GREEN}=== GPU PoW Test Complete ===${NC}"
echo ""
echo "Output files:"
echo "  - gpu_info.json: GPU detection information"
echo "  - challenge_result.json: First challenge result"
echo "  - challenge_result2.json: Reproducibility test result"
echo "  - large_challenge_result.json: Large challenge result"
echo "  - challenge_error.log: Error log (if any)"