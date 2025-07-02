#!/bin/bash
# Test CUDA driver API GPU detection

set -e

echo "=== Testing CUDA Driver API GPU Detection ==="
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

# Test detection by running attestation with minimal options
echo -e "${YELLOW}Testing GPU detection via CUDA driver API...${NC}"

# Set a timeout to prevent hanging
timeout 10s $GPU_ATTESTOR_PATH \
    --skip-network-benchmark \
    --skip-os-attestation \
    --skip-docker-attestation \
    # --skip-gpu-benchmarks \
    --output cuda_test \
    --debug 2>&1 | tee cuda_detection.log

EXIT_CODE=${PIPESTATUS[0]}

if [ $EXIT_CODE -eq 0 ]; then
    echo -e "${GREEN}✓ GPU detection completed successfully${NC}"
    
    # Check if output file exists
    if [ -f "cuda_test.json" ]; then
        echo ""
        echo "GPU Information from attestation report:"
        if command -v jq &> /dev/null; then
            jq '.report.gpu_info' cuda_test.json 2>/dev/null || echo "See cuda_test.json for details"
        else
            echo "Install jq to view formatted output"
            echo "Raw output saved to cuda_test.json"
        fi
    fi
elif [ $EXIT_CODE -eq 124 ]; then
    echo -e "${RED}✗ GPU detection timed out after 10 seconds${NC}"
    echo "This usually indicates CUDA initialization is hanging"
    echo ""
    echo "Last 20 lines of log:"
    tail -20 cuda_detection.log
else
    echo -e "${RED}✗ GPU detection failed with exit code $EXIT_CODE${NC}"
    echo ""
    echo "Error details:"
    grep -E "ERROR|WARN|Failed" cuda_detection.log || tail -20 cuda_detection.log
fi

echo ""
echo "Full log saved to cuda_detection.log"

# Clean up
rm -f cuda_test.json cuda_test.sig cuda_test.pub 2>/dev/null || true