#!/bin/bash
#
# SDK Integration Tests Against Live K3d Cluster
#
# Runs Rust SDK and Python SDK integration tests against a running K3d cluster.
#
# Prerequisites:
#   - K3d cluster running with basilica-api (dev mode) and operator deployed
#     (use sandbox-k3d-e2e.sh setup if needed)
#   - Rust toolchain (for Rust SDK tests)
#   - Python 3.10+ with venv (for Python SDK tests)
#
# Usage:
#   ./sdk-k3d-test.sh [rust|python|all]
#
# Environment:
#   BASILICA_API_URL  - API endpoint (default: http://localhost:18082)
#   SANDBOX_IMAGE     - Sandbox container image

set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
REPO_DIR="$(cd "$SCRIPT_DIR/../.." && pwd)"
SDK_DIR="$REPO_DIR/crates/basilica-sdk"
PYTHON_SDK_DIR="$REPO_DIR/crates/basilica-sdk-python"
IMAGE_TAG_FILE="$SCRIPT_DIR/.sandbox-image-tag"

export BASILICA_API_URL="${BASILICA_API_URL:-http://localhost:18082}"
if [ -z "${SANDBOX_IMAGE_TAG:-}" ] && [ -f "$IMAGE_TAG_FILE" ]; then
    SANDBOX_IMAGE_TAG="$(cat "$IMAGE_TAG_FILE")"
fi
export SANDBOX_IMAGE="${SANDBOX_IMAGE:-k3d-basilica-registry:5050/basilica-exec-agent:${SANDBOX_IMAGE_TAG:-latest}}"

# Colors
RED='\033[0;31m'
GREEN='\033[0;32m'
YELLOW='\033[1;33m'
BLUE='\033[0;34m'
NC='\033[0m'

log_info() { echo -e "${BLUE}[INFO]${NC} $1"; }
log_success() { echo -e "${GREEN}[PASS]${NC} $1"; }
log_error() { echo -e "${RED}[FAIL]${NC} $1"; }
log_step() { echo -e "${YELLOW}[STEP]${NC} $1"; }

PASSED=0
FAILED=0
TEST_NAMESPACE="u-test-user"

check_cluster() {
    log_step "Checking K3d cluster health..."
    local health
    health=$(curl -s --max-time 5 "${BASILICA_API_URL}/health" 2>/dev/null || echo "")
    if echo "$health" | grep -q '"status":"healthy"'; then
        log_info "API is healthy at $BASILICA_API_URL"
    else
        log_error "API not reachable at $BASILICA_API_URL"
        echo "Run 'sandbox-k3d-e2e.sh setup' first."
        exit 1
    fi
}

cleanup_test_namespace() {
    log_step "Cleaning sandbox test namespace (${TEST_NAMESPACE})..."

    kubectl delete basilicasandbox -n "$TEST_NAMESPACE" --all --ignore-not-found --wait=true >/dev/null 2>&1 || true
    kubectl delete pods,services -n "$TEST_NAMESPACE" -l basilica.ai/type=sandbox --ignore-not-found >/dev/null 2>&1 || true

    while IFS= read -r secret; do
        kubectl delete -n "$TEST_NAMESPACE" "$secret" --ignore-not-found >/dev/null 2>&1 || true
    done < <(kubectl get secret -n "$TEST_NAMESPACE" -o name 2>/dev/null | grep '^secret/sandbox-.*-exec-secret$' || true)
}

run_rust_tests() {
    log_step "Running Rust SDK integration tests..."
    echo ""

    cleanup_test_namespace
    cd "$REPO_DIR"
    if cargo test --test sandbox_k3d -- --ignored --nocapture --test-threads=1 2>&1; then
        PASSED=$((PASSED + 1))
        log_success "Rust SDK integration tests passed"
    else
        FAILED=$((FAILED + 1))
        log_error "Rust SDK integration tests failed"
    fi
    echo ""
}

run_python_tests() {
    log_step "Running Python SDK integration tests..."
    echo ""

    cleanup_test_namespace
    cd "$PYTHON_SDK_DIR"

    # Set up venv if needed
    local venv_dir="$REPO_DIR/.venv-sdk-test"
    if [ ! -d "$venv_dir" ]; then
        log_info "Creating Python virtualenv at $venv_dir..."
        python3 -m venv "$venv_dir"
    fi

    source "$venv_dir/bin/activate"

    # Install dependencies
    pip install -q pytest maturin 2>/dev/null

    # Build and install the SDK
    log_info "Building Python SDK with maturin..."
    maturin develop --quiet 2>/dev/null || maturin develop

    # Run tests
    if python3 -m pytest tests/test_sandbox_k3d.py -v -m k3d -s 2>&1; then
        PASSED=$((PASSED + 1))
        log_success "Python SDK integration tests passed"
    else
        FAILED=$((FAILED + 1))
        log_error "Python SDK integration tests failed"
    fi

    deactivate 2>/dev/null || true
    echo ""
}

report() {
    echo "============================================"
    echo "  SDK Integration Test Results"
    echo "============================================"
    echo -e "  Passed: ${GREEN}${PASSED}${NC}"
    echo -e "  Failed: ${RED}${FAILED}${NC}"
    echo "============================================"

    if [ "$FAILED" -gt 0 ]; then
        exit 1
    fi
}

main() {
    local mode="${1:-all}"

    check_cluster

    case "$mode" in
        rust)
            run_rust_tests
            ;;
        python)
            run_python_tests
            ;;
        all)
            run_rust_tests
            run_python_tests
            ;;
        *)
            echo "Usage: $0 [rust|python|all]"
            exit 1
            ;;
    esac

    report
}

main "$@"
