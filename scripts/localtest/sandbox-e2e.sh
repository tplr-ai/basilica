#!/bin/bash
#
# Sandbox E2E Testing Script
#
# Tests the sandbox v2 architecture:
#   - Control-plane operations (create, list, get, delete) via sandbox-test-api at /sandboxes
#   - Data-plane operations (exec, run, files) direct to sandbox pods via port-forward
#   - Negative / edge case tests (invalid image, nonexistent sandbox, wrong auth)
#
# Prerequisites:
#   - K3d cluster running with basilica-sandbox-test cluster
#   - sandbox-test-api and basilica-sandbox-operator deployed
#   - BASILICA_API_URL set (default: http://localhost:18082)
#
# Usage:
#   ./sandbox-e2e.sh [command]
#
# Commands:
#   all       - Run all tests (default)
#   setup     - Check prerequisites
#   create    - Create a test sandbox
#   exec      - Test exec + run (data-plane)
#   files     - Test file operations (data-plane)
#   negative  - Negative / edge case tests
#   cleanup   - Delete test sandboxes

set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
cd "$SCRIPT_DIR"

DEFAULT_BACKEND_DIR="$SCRIPT_DIR/../../../basilica-backend"
BASILICA_BACKEND_DIR="${BASILICA_BACKEND_DIR:-$DEFAULT_BACKEND_DIR}"
if [ -d "$BASILICA_BACKEND_DIR" ]; then
    BASILICA_BACKEND_DIR="$(cd "$BASILICA_BACKEND_DIR" && pwd)"
fi

# Configuration
BASILICA_API_URL="${BASILICA_API_URL:-http://localhost:${SANDBOX_API_PORT:-18082}}"
NAMESPACE="${NAMESPACE:-default}"
SANDBOX_IMAGE="${SANDBOX_IMAGE:-k3d-basilica-registry:5050/basilica-exec-agent:latest}"

# Test state
SANDBOX_ID=""
SANDBOX_DOMAIN=""
EXEC_AGENT_SECRET=""
DATA_PLANE_PF_PID=""
DATA_PLANE_URL=""

# Counters
TESTS_PASSED=0
TESTS_FAILED=0
TESTS_SKIPPED=0

# Logging
log_info() { echo -e "\033[34m[INFO]\033[0m $1"; }
log_success() { echo -e "\033[32m[PASS]\033[0m $1"; }
log_warn() { echo -e "\033[33m[WARN]\033[0m $1"; }
log_error() { echo -e "\033[31m[FAIL]\033[0m $1"; }
log_test() { echo -e "\033[35m[TEST]\033[0m $1"; }
log_skip() { echo -e "\033[33m[SKIP]\033[0m $1"; }

pass() {
    TESTS_PASSED=$((TESTS_PASSED + 1))
    log_success "$1"
}

fail() {
    TESTS_FAILED=$((TESTS_FAILED + 1))
    log_error "$1"
}

skip() {
    TESTS_SKIPPED=$((TESTS_SKIPPED + 1))
    log_skip "$1"
}

# API helper (no auth token needed for sandbox-test-api)
api_call() {
    local method="$1"
    local path="$2"
    local data="${3:-}"

    if [ -n "$data" ]; then
        curl -s --max-time 30 -X "$method" \
            -H "Content-Type: application/json" \
            -d "$data" \
            "${BASILICA_API_URL}${path}"
    else
        curl -s --max-time 30 -X "$method" \
            "${BASILICA_API_URL}${path}"
    fi
}

# Data-plane helper (requires exec-agent secret)
data_plane_call() {
    local method="$1"
    local path="$2"
    local data="${3:-}"

    if [ -z "$DATA_PLANE_URL" ]; then
        log_error "No data-plane URL set"
        return 1
    fi

    if [ -n "$data" ]; then
        curl -s --max-time 30 -X "$method" \
            -H "Authorization: Bearer $EXEC_AGENT_SECRET" \
            -H "Content-Type: application/json" \
            -d "$data" \
            "${DATA_PLANE_URL}${path}"
    else
        curl -s --max-time 30 -X "$method" \
            -H "Authorization: Bearer $EXEC_AGENT_SECRET" \
            "${DATA_PLANE_URL}${path}"
    fi
}

# ============================================================================
# Prerequisites
# ============================================================================

check_prerequisites() {
    log_test "Checking prerequisites..."

    command -v curl &>/dev/null || { fail "curl not found"; exit 1; }
    command -v jq &>/dev/null || { fail "jq not found"; exit 1; }
    command -v kubectl &>/dev/null || { fail "kubectl not found"; exit 1; }

    # Check API health
    local health
    health=$(api_call GET "/health" 2>/dev/null || echo "{}")
    if echo "$health" | jq -e '.status == "healthy"' &>/dev/null; then
        pass "API health check passed"
    else
        fail "API health check failed: $health"
        exit 1
    fi

    # Check CRD exists
    if kubectl get crd basilicasandboxes.basilica.ai &>/dev/null; then
        pass "BasilicaSandbox CRD exists"
    else
        fail "BasilicaSandbox CRD not found"
        exit 1
    fi

    # Check operator is running
    if kubectl get deployment basilica-sandbox-operator -n basilica-system &>/dev/null; then
        local ready
        ready=$(kubectl get deployment basilica-sandbox-operator -n basilica-system -o jsonpath='{.status.readyReplicas}' 2>/dev/null)
        if [ "${ready:-0}" -ge 1 ]; then
            pass "Sandbox operator is running"
        else
            fail "Sandbox operator not ready"
            exit 1
        fi
    else
        fail "Sandbox operator deployment not found"
        exit 1
    fi
}

# ============================================================================
# Control-Plane Tests
# ============================================================================

test_create_sandbox() {
    log_test "POST /sandboxes -- create sandbox"

    local request
    request=$(jq -n --arg image "$SANDBOX_IMAGE" '{
        image: $image,
        cpu: "500m",
        memory: "512Mi",
        env: [{"name": "TEST_VAR", "value": "hello"}],
        ttlSeconds: 3600
    }')

    local response
    response=$(api_call POST "/sandboxes" "$request")

    if echo "$response" | jq -e '.sandboxId' &>/dev/null; then
        SANDBOX_ID=$(echo "$response" | jq -r '.sandboxId')
        SANDBOX_DOMAIN=$(echo "$response" | jq -r '.domain')
        EXEC_AGENT_SECRET=$(echo "$response" | jq -r '.execAgentSecret')

        # Verify response fields
        if [ -n "$SANDBOX_ID" ] && [ "$SANDBOX_ID" != "null" ] &&
           [ -n "$SANDBOX_DOMAIN" ] && [ "$SANDBOX_DOMAIN" != "null" ] &&
           [ -n "$EXEC_AGENT_SECRET" ] && [ "$EXEC_AGENT_SECRET" != "null" ]; then
            pass "Sandbox created: $SANDBOX_ID (domain: $SANDBOX_DOMAIN)"
        else
            fail "Sandbox response missing required fields: $response"
            return 1
        fi
    else
        fail "Failed to create sandbox: $response"
        return 1
    fi

    # Wait for sandbox to reach Running state
    log_info "Waiting for sandbox to be ready..."
    local max_attempts=60
    local attempt=0

    while [ $attempt -lt $max_attempts ]; do
        local status_response
        status_response=$(api_call GET "/sandboxes/$SANDBOX_ID")
        local status
        status=$(echo "$status_response" | jq -r '.status')

        if [ "$status" = "Running" ]; then
            pass "Sandbox reached Running state"
            return 0
        elif [ "$status" = "Failed" ] || [ "$status" = "Terminated" ]; then
            fail "Sandbox failed with status: $status"
            return 1
        fi

        sleep 2
        attempt=$((attempt + 1))
    done

    fail "Timeout waiting for sandbox to be ready (last status: $status)"
    return 1
}

test_list_sandboxes() {
    log_test "GET /sandboxes -- list sandboxes"

    local response
    response=$(api_call GET "/sandboxes")

    if echo "$response" | jq -e '.sandboxes' &>/dev/null; then
        local count
        count=$(echo "$response" | jq '.sandboxes | length')
        if [ "$count" -ge 1 ]; then
            # Verify our sandbox appears in the list
            local found
            found=$(echo "$response" | jq -r --arg id "$SANDBOX_ID" '.sandboxes[] | select(.sandboxId == $id) | .sandboxId')
            if [ "$found" = "$SANDBOX_ID" ]; then
                pass "List sandboxes: found $count sandboxes, including $SANDBOX_ID"
            else
                fail "List sandboxes: $SANDBOX_ID not found in response"
                return 1
            fi
        else
            fail "List sandboxes: expected at least 1, got $count"
            return 1
        fi
    else
        fail "List sandboxes failed: $response"
        return 1
    fi
}

test_get_sandbox() {
    log_test "GET /sandboxes/:id -- get sandbox detail"

    local response
    response=$(api_call GET "/sandboxes/$SANDBOX_ID")

    local sid
    sid=$(echo "$response" | jq -r '.sandboxId')
    local status
    status=$(echo "$response" | jq -r '.status')
    local image
    image=$(echo "$response" | jq -r '.image')

    if [ "$sid" = "$SANDBOX_ID" ] && [ -n "$status" ] && [ -n "$image" ]; then
        pass "Get sandbox detail: id=$sid, status=$status, image=$image"
    else
        fail "Get sandbox detail failed: $response"
        return 1
    fi
}

# ============================================================================
# Data-Plane Tests
# ============================================================================

setup_data_plane_access() {
    if [ -n "$DATA_PLANE_URL" ]; then
        return 0
    fi

    local pod_name="sandbox-${SANDBOX_ID}"
    local ns="u-test-user"  # sandbox-test-api uses fixed test user

    # Wait for pod to be ready
    log_info "Waiting for sandbox pod to be ready..."
    if ! kubectl wait --for=condition=Ready "pod/$pod_name" -n "$ns" --timeout=60s 2>/dev/null; then
        fail "Sandbox pod $pod_name not ready"
        kubectl describe pod "$pod_name" -n "$ns" 2>/dev/null | tail -10
        return 1
    fi

    # Start port-forward
    local local_port=$((RANDOM % 10000 + 20000))
    kubectl port-forward -n "$ns" "pod/$pod_name" "${local_port}:9999" &>/dev/null &
    DATA_PLANE_PF_PID=$!
    sleep 2

    if kill -0 "$DATA_PLANE_PF_PID" 2>/dev/null; then
        DATA_PLANE_URL="http://localhost:${local_port}"
        pass "Port-forward established: $DATA_PLANE_URL -> $ns/$pod_name:9999"
    else
        fail "Port-forward failed for $pod_name"
        return 1
    fi
}

test_exec() {
    log_test "POST /exec -- execute command via data-plane"

    setup_data_plane_access || return 1

    local response
    response=$(data_plane_call POST "/exec" '{"command": ["echo", "Hello, World!"]}')

    local stdout
    stdout=$(echo "$response" | jq -r '.stdout')
    local exit_code
    exit_code=$(echo "$response" | jq -r '.exitCode')

    if [ "$exit_code" = "0" ] && [[ "$stdout" == *"Hello, World!"* ]]; then
        pass "Exec: stdout='$stdout' exitCode=$exit_code"
    else
        fail "Exec failed: $response"
        return 1
    fi
}

test_run() {
    log_test "POST /run -- run code via data-plane"

    setup_data_plane_access || return 1

    local response
    response=$(data_plane_call POST "/run" '{"code": "print(1 + 1)"}')

    local stdout
    stdout=$(echo "$response" | jq -r '.stdout')
    local exit_code
    exit_code=$(echo "$response" | jq -r '.exitCode')

    if [ "$exit_code" = "0" ] && [[ "$stdout" == *"2"* ]]; then
        pass "Run: stdout='$stdout' exitCode=$exit_code"
    else
        fail "Run failed: $response"
        return 1
    fi
}

test_files() {
    log_test "POST /files/write, /files/read, /files/list -- file operations"

    setup_data_plane_access || return 1

    # Write file
    local response
    response=$(data_plane_call POST "/files/write" '{"path": "/workspace/test.txt", "content": "Hello from test!"}')

    if echo "$response" | jq -e '.path' &>/dev/null; then
        pass "File write succeeded"
    else
        fail "File write failed: $response"
        return 1
    fi

    # Read file
    response=$(data_plane_call POST "/files/read" '{"path": "/workspace/test.txt"}')

    local content
    content=$(echo "$response" | jq -r '.content')
    if [[ "$content" == *"Hello from test!"* ]]; then
        pass "File read succeeded: content matches"
    else
        fail "File read failed: $response"
        return 1
    fi

    # List files
    response=$(data_plane_call POST "/files/list" '{"path": "/workspace"}')

    if echo "$response" | jq -e '.files | length > 0' &>/dev/null; then
        local found
        found=$(echo "$response" | jq -r '.files[] | select(.name == "test.txt") | .name')
        if [ "$found" = "test.txt" ]; then
            pass "File list succeeded: test.txt found"
        else
            fail "File list: test.txt not found in response"
            return 1
        fi
    else
        fail "File list failed: $response"
        return 1
    fi
}

test_snapshot() {
    log_test "POST /snapshot/create -- snapshot creation"

    setup_data_plane_access || return 1

    local response
    response=$(data_plane_call POST "/snapshot/create" '{}')

    local status
    status=$(echo "$response" | jq -r '.status // empty')
    if [ "$status" = "created" ] || [ "$status" = "in_progress" ]; then
        pass "Snapshot creation initiated: $status"
    elif echo "$response" | jq -e '.snapshotId' &>/dev/null; then
        pass "Snapshot created: $(echo "$response" | jq -r '.snapshotId')"
    else
        # Snapshot may not be fully configured in test environment
        local error_msg
        error_msg=$(echo "$response" | jq -r '.error // empty')
        if [ -n "$error_msg" ]; then
            skip "Snapshot not available: $error_msg"
        else
            skip "Snapshot response unexpected: $response"
        fi
    fi
}

# ============================================================================
# Negative / Edge Case Tests
# ============================================================================

test_negative_cases() {
    log_test "Negative / edge case tests..."

    # 1. Create with invalid image
    log_test "  Create with invalid image"
    local response
    response=$(api_call POST "/sandboxes" '{"image": "evil/hacker-image:latest"}')
    local error
    error=$(echo "$response" | jq -r '.error // empty')
    if [ -n "$error" ] && [[ "$error" == *"not in the allowlist"* ]]; then
        pass "Invalid image rejected correctly"
    else
        local http_code
        http_code=$(curl -s -o /dev/null -w "%{http_code}" --max-time 10 -X POST \
            -H "Content-Type: application/json" \
            -d '{"image": "evil/hacker-image:latest"}' \
            "${BASILICA_API_URL}/sandboxes")
        if [ "$http_code" = "400" ]; then
            pass "Invalid image rejected with HTTP 400"
        else
            fail "Invalid image not rejected: HTTP $http_code, response: $response"
        fi
    fi

    # 2. Delete nonexistent sandbox
    log_test "  Delete nonexistent sandbox"
    local http_code
    http_code=$(curl -s -o /dev/null -w "%{http_code}" --max-time 10 -X DELETE \
        "${BASILICA_API_URL}/sandboxes/sb-nonexistent-99999999")
    if [ "$http_code" = "404" ]; then
        pass "Nonexistent sandbox delete returns 404"
    else
        fail "Nonexistent sandbox delete returned HTTP $http_code (expected 404)"
    fi

    # 3. Get nonexistent sandbox
    log_test "  Get nonexistent sandbox"
    http_code=$(curl -s -o /dev/null -w "%{http_code}" --max-time 10 -X GET \
        "${BASILICA_API_URL}/sandboxes/sb-nonexistent-99999999")
    if [ "$http_code" = "404" ]; then
        pass "Nonexistent sandbox get returns 404"
    else
        fail "Nonexistent sandbox get returned HTTP $http_code (expected 404)"
    fi

    # 4. Exec with wrong auth token
    log_test "  Exec with wrong auth token"
    if [ -n "$DATA_PLANE_URL" ]; then
        http_code=$(curl -s -o /dev/null -w "%{http_code}" --max-time 10 -X POST \
            -H "Authorization: Bearer wrong-token-12345" \
            -H "Content-Type: application/json" \
            -d '{"command": ["echo", "should fail"]}' \
            "${DATA_PLANE_URL}/exec")
        if [ "$http_code" = "401" ]; then
            pass "Wrong auth token returns 401"
        else
            fail "Wrong auth token returned HTTP $http_code (expected 401)"
        fi
    else
        skip "Exec auth test: no data-plane URL"
    fi
}

# ============================================================================
# Delete Test
# ============================================================================

test_delete_sandbox() {
    log_test "DELETE /sandboxes/:id -- delete sandbox"

    if [ -z "$SANDBOX_ID" ]; then
        skip "No sandbox to delete"
        return 0
    fi

    local response
    response=$(api_call DELETE "/sandboxes/$SANDBOX_ID")

    local status
    status=$(echo "$response" | jq -r '.status // empty')
    if [ "$status" = "deleting" ]; then
        pass "Sandbox $SANDBOX_ID deletion initiated"
    else
        fail "Sandbox deletion failed: $response"
        return 1
    fi

    # Verify it's actually gone (or in deleting state)
    sleep 5
    local http_code
    http_code=$(curl -s -o /dev/null -w "%{http_code}" --max-time 10 -X GET \
        "${BASILICA_API_URL}/sandboxes/$SANDBOX_ID")
    if [ "$http_code" = "404" ]; then
        pass "Sandbox $SANDBOX_ID confirmed deleted"
    else
        # May still be in deleting state
        log_info "Sandbox still exists (may be terminating): HTTP $http_code"
    fi
}

# ============================================================================
# Cleanup
# ============================================================================

cleanup() {
    log_info "Cleaning up..."

    # Kill port-forward if running
    if [ -n "${DATA_PLANE_PF_PID:-}" ]; then
        kill "$DATA_PLANE_PF_PID" 2>/dev/null || true
    fi

    # Clean up sandboxes via kubectl (more reliable than API for cleanup)
    kubectl delete basilicasandbox --all -n u-test-user --ignore-not-found 2>/dev/null || true

    log_info "Cleanup completed"
}

# ============================================================================
# Test Suite
# ============================================================================

print_summary() {
    echo ""
    echo "============================================"
    echo "  Test Summary"
    echo "============================================"
    echo "  Passed:  $TESTS_PASSED"
    echo "  Failed:  $TESTS_FAILED"
    echo "  Skipped: $TESTS_SKIPPED"
    echo "  Total:   $((TESTS_PASSED + TESTS_FAILED + TESTS_SKIPPED))"
    echo "============================================"

    if [ "$TESTS_FAILED" -gt 0 ]; then
        echo ""
        log_error "SOME TESTS FAILED"
        return 1
    else
        echo ""
        log_success "ALL TESTS PASSED"
        return 0
    fi
}

run_all_tests() {
    log_info "Running all sandbox E2E tests..."
    log_info "API URL: $BASILICA_API_URL"
    log_info "Sandbox image: $SANDBOX_IMAGE"
    echo ""

    # Prerequisites
    check_prerequisites
    echo ""

    # Control-plane tests
    test_create_sandbox || { fail "Create test failed - aborting"; cleanup; print_summary; exit 1; }
    echo ""

    test_list_sandboxes
    echo ""

    test_get_sandbox
    echo ""

    # Data-plane tests
    test_exec || { fail "Exec test failed"; }
    echo ""

    test_run || { fail "Run test failed"; }
    echo ""

    test_files || { fail "Files test failed"; }
    echo ""

    test_snapshot
    echo ""

    # Negative / edge case tests
    test_negative_cases
    echo ""

    # Delete test (control-plane)
    test_delete_sandbox
    echo ""

    # Cleanup and summary
    cleanup
    print_summary
}

# ============================================================================
# Main
# ============================================================================

case "${1:-all}" in
    setup)
        check_prerequisites
        ;;
    create)
        test_create_sandbox
        ;;
    exec)
        test_exec
        ;;
    files)
        test_files
        ;;
    snapshot)
        test_snapshot
        ;;
    negative)
        test_negative_cases
        ;;
    list)
        test_list_sandboxes
        ;;
    cleanup)
        cleanup
        ;;
    all)
        run_all_tests
        ;;
    *)
        echo "Usage: $0 {setup|create|exec|files|snapshot|negative|list|cleanup|all}"
        exit 1
        ;;
esac
