#!/bin/bash
# Integration test script for Basilica Training SDK
set -e

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
PROJECT_ROOT="$(cd "$SCRIPT_DIR/.." && pwd)"

echo "=== Basilica Training SDK Integration Tests ==="
echo ""

# Activate venv if it exists
if [ -f "$PROJECT_ROOT/.venv/bin/activate" ]; then
    echo "--- Activating virtual environment ---"
    source "$PROJECT_ROOT/.venv/bin/activate"
    echo "Using Python: $(which python3)"
    echo ""
fi

# Check if API is running
echo "--- Checking API availability ---"
if ! curl -sf http://localhost:8000/health > /dev/null 2>&1; then
    echo "ERROR: API not running on localhost:8000"
    echo "Please start the API first with: just local-dev-up"
    exit 1
fi
echo "API is healthy"
echo ""

# Check for API token
if [ ! -f "$PROJECT_ROOT/build/api-token.txt" ]; then
    echo "ERROR: No API token found at build/api-token.txt"
    echo "Generate one with: ./scripts/api/gen-key.sh"
    exit 1
fi
TOKEN=$(cat "$PROJECT_ROOT/build/api-token.txt")
echo "Using API token: ${TOKEN:0:20}..."
echo ""

# Test capabilities endpoint
echo "--- Testing /capabilities endpoint ---"
CAPS=$(curl -sf -H "Authorization: Bearer $TOKEN" http://localhost:8000/capabilities)
if [ -z "$CAPS" ]; then
    echo "ERROR: Failed to fetch capabilities"
    exit 1
fi
echo "Capabilities: $(echo "$CAPS" | jq -c '.models[:2]')"
echo ""

# Test training_runs endpoint
echo "--- Testing /training_runs endpoint ---"
RUNS=$(curl -sf -H "Authorization: Bearer $TOKEN" http://localhost:8000/training_runs)
if [ -z "$RUNS" ]; then
    echo "ERROR: Failed to fetch training runs"
    exit 1
fi
echo "Training runs: $(echo "$RUNS" | jq -r '.total') total"
echo ""

# Test checkpoints endpoint
echo "--- Testing /checkpoints endpoint ---"
CHECKPOINTS=$(curl -sf -H "Authorization: Bearer $TOKEN" http://localhost:8000/checkpoints)
if [ -z "$CHECKPOINTS" ]; then
    echo "ERROR: Failed to fetch checkpoints"
    exit 1
fi
echo "Checkpoints: $(echo "$CHECKPOINTS" | jq -r '.total') total"
echo ""

# Run Python example with --rest flag
echo "--- Running RestClient Example ---"
cd "$PROJECT_ROOT"
python3 examples/training_example.py --rest
echo ""

# Run SDK unit tests (if pytest is available)
echo "--- Running SDK Unit Tests ---"
if command -v pytest &> /dev/null; then
    cd "$PROJECT_ROOT/crates/basilica-sdk-python"
    PYTHONPATH="python:$PYTHONPATH" pytest python/basilica/training/tests/ -v --tb=short 2>&1 || {
        echo "WARNING: Some unit tests failed"
    }
else
    echo "pytest not available, skipping unit tests"
    echo "Install with: pip install pytest pytest-asyncio"
fi
echo ""

echo "=== Integration Tests Complete ==="
