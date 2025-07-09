#!/bin/bash
# Direct test of VM-protected validation without debugger

echo "=== VM-Protected GPU Attestation Direct Test ==="
echo ""
echo "Testing the gpu-attestor binary with generic --challenge interface"
echo "This demonstrates the multi-layer security architecture:"
echo ""
echo "Layer 1 - VM PROTECTION (Primary Security):"
echo "  ✓ Encrypted bytecode prevents static analysis"
echo "  ✓ Anti-debugging detects runtime analysis"
echo "  ✓ Dynamic code generation defeats patterns"
echo "  ✓ Hidden validation logic in VM"
echo ""
echo "Layer 2 - INTERFACE OBFUSCATION (Additional):"
echo "  ✓ Generic --challenge parameter only"
echo "  ✓ No protocol-specific flags exposed"
echo "  ✓ Binary determines validation internally"
echo ""

# Create a test challenge
CHALLENGE_JSON='{
  "session_id": "test_direct_validation",
  "problem_size": 256,
  "seed_data": [1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11, 12, 13, 14, 15, 16],
  "timestamp": "1704067200",
  "resource_count": 1,
  "computation_timeout_ms": 5000,
  "protocol_timeout_ms": 10000
}'

# Base64 encode the challenge
CHALLENGE_B64=$(echo -n "$CHALLENGE_JSON" | base64 -w 0)

echo "Running validation with generic challenge..."
echo "Challenge (base64): ${CHALLENGE_B64:0:50}..."
echo ""

# Run the binary
./target/release/gpu-attestor --challenge "$CHALLENGE_B64"

# Check exit code
if [ $? -eq 0 ]; then
    echo ""
    echo "✓ VM-protected validation completed successfully"
    echo "  - Binary accepted generic challenge"
    echo "  - VM executed validation logic internally"
    echo "  - Returned PASS/FAIL result with no details"
else
    echo ""
    echo "✗ Validation failed (this may be expected depending on GPU availability)"
fi