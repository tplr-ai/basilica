#!/bin/bash
#
# Sandbox E2E Testing Script
#
# Tests the sandbox v2 architecture:
#   - Control-plane operations (create, list, get, delete) via basilica-api at /sandboxes
#   - Data-plane operations (exec, run, files, snapshot) direct to sandbox domains
#
# Prerequisites:
#   - K3s cluster running (or minikube/kind with basilica CRDs applied)
#   - basilica-api and basilica-sandbox-operator deployed
#   - BASILICA_API_URL and BASILICA_API_TOKEN environment variables set
#
# Usage:
#   ./sandbox-e2e.sh [command]
#
# Commands:
#   setup     - Apply CRD and check prerequisites
#   create    - Create a test sandbox
#   exec      - Test command execution (data-plane, direct to sandbox domain)
#   files     - Test file operations (data-plane, direct to sandbox domain)
#   snapshot  - Test snapshot creation (data-plane, direct to sandbox domain)
#   cleanup   - Delete test sandboxes
#   all       - Run all tests (default)

set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
cd "$SCRIPT_DIR"

DEFAULT_BACKEND_DIR="$SCRIPT_DIR/../../../basilica-backend"
BASILICA_BACKEND_DIR="${BASILICA_BACKEND_DIR:-$DEFAULT_BACKEND_DIR}"
if [ -d "$BASILICA_BACKEND_DIR" ]; then
    BASILICA_BACKEND_DIR="$(cd "$BASILICA_BACKEND_DIR" && pwd)"
fi

# Load test configuration
if [ -f "test.conf" ]; then
    source test.conf
fi

# Default values
BASILICA_API_URL="${BASILICA_API_URL:-http://localhost:${SANDBOX_API_PORT:-18082}}"
BASILICA_API_TOKEN="${BASILICA_API_TOKEN:-test-token}"
NAMESPACE="${NAMESPACE:-default}"
SANDBOX_ID=""
SANDBOX_DOMAIN=""
EXEC_AGENT_SECRET=""

# Logging helpers
log_info() { echo -e "\033[34m[INFO]\033[0m $1"; }
log_success() { echo -e "\033[32m[SUCCESS]\033[0m $1"; }
log_warn() { echo -e "\033[33m[WARN]\033[0m $1"; }
log_error() { echo -e "\033[31m[ERROR]\033[0m $1"; }
log_test() { echo -e "\033[35m[TEST]\033[0m $1"; }

# Control-plane API helper (requests to basilica-api)
api_call() {
    local method="$1"
    local path="$2"
    local data="${3:-}"

    if [ -n "$data" ]; then
        curl -s -X "$method" \
            -H "Authorization: Bearer $BASILICA_API_TOKEN" \
            -H "Content-Type: application/json" \
            -d "$data" \
            "${BASILICA_API_URL}${path}"
    else
        curl -s -X "$method" \
            -H "Authorization: Bearer $BASILICA_API_TOKEN" \
            "${BASILICA_API_URL}${path}"
    fi
}

# Data-plane helper (requests direct to sandbox domain via exec-agent)
data_plane_call() {
    local method="$1"
    local path="$2"
    local data="${3:-}"

    if [ -z "$SANDBOX_DOMAIN" ]; then
        log_error "No sandbox domain set"
        return 1
    fi

    if [ -z "$EXEC_AGENT_SECRET" ]; then
        log_error "No exec-agent secret set"
        return 1
    fi

    local url="https://${SANDBOX_DOMAIN}${path}"

    # In local testing, the sandbox domain may not be resolvable via DNS.
    # Use the sandbox pod's cluster IP or port-forward instead.
    # If SANDBOX_DATA_PLANE_URL is set, use it as the base URL.
    if [ -n "${SANDBOX_DATA_PLANE_URL:-}" ]; then
        url="${SANDBOX_DATA_PLANE_URL}${path}"
    fi

    if [ -n "$data" ]; then
        curl -s -X "$method" \
            -H "Authorization: Bearer $EXEC_AGENT_SECRET" \
            -H "Content-Type: application/json" \
            -d "$data" \
            "$url"
    else
        curl -s -X "$method" \
            -H "Authorization: Bearer $EXEC_AGENT_SECRET" \
            "$url"
    fi
}

# Retrieve exec-agent secret from K8s for data-plane auth
retrieve_exec_agent_secret() {
    if ! command -v kubectl &>/dev/null; then
        log_warn "kubectl not available - cannot retrieve exec-agent secret for data-plane tests"
        return 1
    fi

    local secret_name="sandbox-${SANDBOX_ID}-exec-secret"
    # Secret is in the user's namespace (u-{user_id}).
    # In local testing with auth disabled, the user_id may vary.
    # Try common namespace patterns.
    local ns=""
    for candidate in "$NAMESPACE" "u-test-user" "u-test" "default"; do
        if kubectl get secret "$secret_name" -n "$candidate" &>/dev/null 2>&1; then
            ns="$candidate"
            break
        fi
    done

    if [ -z "$ns" ]; then
        log_warn "Could not find exec-agent secret $secret_name in any namespace"
        return 1
    fi

    EXEC_AGENT_SECRET=$(kubectl get secret "$secret_name" -n "$ns" -o jsonpath='{.data.EXEC_AGENT_SECRET}' | base64 -d)
    if [ -n "$EXEC_AGENT_SECRET" ]; then
        log_success "Retrieved exec-agent secret from $ns/$secret_name"
        return 0
    else
        log_warn "Exec-agent secret was empty"
        return 1
    fi
}

# Set up port-forward to sandbox pod for local data-plane access
setup_data_plane_access() {
    if [ -n "${SANDBOX_DATA_PLANE_URL:-}" ]; then
        log_info "Using pre-configured data-plane URL: $SANDBOX_DATA_PLANE_URL"
        return 0
    fi

    if ! command -v kubectl &>/dev/null; then
        log_warn "kubectl not available - data-plane tests will use sandbox domain directly"
        return 0
    fi

    local pod_name="sandbox-${SANDBOX_ID}"
    # Find the pod in the user namespace
    local ns=""
    for candidate in "$NAMESPACE" "u-test-user" "u-test" "default"; do
        if kubectl get pod "$pod_name" -n "$candidate" &>/dev/null 2>&1; then
            ns="$candidate"
            break
        fi
    done

    if [ -z "$ns" ]; then
        log_warn "Could not find sandbox pod $pod_name - data-plane tests will use domain directly"
        return 0
    fi

    # Start port-forward in background
    local local_port=$((RANDOM % 10000 + 20000))
    kubectl port-forward -n "$ns" "pod/$pod_name" "${local_port}:9999" &>/dev/null &
    local pf_pid=$!
    sleep 2

    if kill -0 "$pf_pid" 2>/dev/null; then
        SANDBOX_DATA_PLANE_URL="http://localhost:${local_port}"
        DATA_PLANE_PF_PID="$pf_pid"
        log_success "Port-forward established: $SANDBOX_DATA_PLANE_URL -> $ns/$pod_name:9999"
    else
        log_warn "Port-forward failed - data-plane tests will use domain directly"
    fi
}

# Check prerequisites
check_prerequisites() {
    log_info "Checking prerequisites..."

    # Check kubectl
    if command -v kubectl &>/dev/null; then
        log_success "kubectl found"
    else
        log_warn "kubectl not found - CRD operations will use API only"
    fi

    # Check curl
    command -v curl &>/dev/null || { log_error "curl not found"; exit 1; }

    # Check jq
    command -v jq &>/dev/null || { log_error "jq not found (required for JSON parsing)"; exit 1; }

    # Check API health
    local health=$(api_call GET "/health" 2>/dev/null || echo "{}")
    if echo "$health" | jq -e '.status == "ok" or .status == "healthy"' &>/dev/null; then
        log_success "API health check passed"
    else
        log_warn "API health check returned: $health"
    fi

    log_success "Prerequisites check completed"
}

# Apply CRD
setup_crd() {
    log_info "Setting up BasilicaSandbox CRD..."

    if ! command -v kubectl &>/dev/null; then
        log_warn "kubectl not available - skipping CRD setup"
        return 0
    fi

    # Check if CRD already exists
    if kubectl get crd basilicasandboxes.basilica.ai &>/dev/null; then
        log_success "CRD already exists"
        return 0
    fi

    # Try to use the generated CRD file from basilica-backend.
    local crd_file="$BASILICA_BACKEND_DIR/orchestrator/k8s/crds/basilica-sandbox.yaml"

    if [ -f "$crd_file" ]; then
        kubectl apply -f "$crd_file"
        log_success "CRD applied from $crd_file"
    else
        log_error "CRD not found and not already deployed."
        log_error "Run 'sandbox-k3d-e2e.sh deploy' first to set up infrastructure."
        return 1
    fi
}

# Create sandbox via control-plane API
test_create_sandbox() {
    log_test "Testing sandbox creation via control-plane API..."

    # CreateSandboxInput: image, cpu, memory, env, ttl_seconds
    # No language, resources, timeoutSeconds, idleTimeoutSeconds, autoSnapshot, networkIsolation
    local request='{
        "image": "basilica/sandbox-python:latest",
        "cpu": "500m",
        "memory": "512Mi",
        "env": [
            {"name": "TEST_VAR", "value": "hello"}
        ],
        "ttlSeconds": 3600
    }'

    local response=$(api_call POST "/sandboxes" "$request")

    if echo "$response" | jq -e '.sandboxId' &>/dev/null; then
        SANDBOX_ID=$(echo "$response" | jq -r '.sandboxId')
        SANDBOX_DOMAIN=$(echo "$response" | jq -r '.domain')
        log_success "Sandbox created: $SANDBOX_ID (domain: $SANDBOX_DOMAIN)"
    else
        log_error "Failed to create sandbox: $response"
        return 1
    fi

    # Wait for sandbox to be ready
    log_info "Waiting for sandbox to be ready..."
    local max_attempts=60
    local attempt=0

    while [ $attempt -lt $max_attempts ]; do
        local status_response=$(api_call GET "/sandboxes/$SANDBOX_ID")
        local status=$(echo "$status_response" | jq -r '.status')

        if [ "$status" = "Ready" ]; then
            # Update domain from detail response if available
            local detail_domain=$(echo "$status_response" | jq -r '.domain // empty')
            if [ -n "$detail_domain" ]; then
                SANDBOX_DOMAIN="$detail_domain"
            fi
            log_success "Sandbox is ready (domain: $SANDBOX_DOMAIN)"
            return 0
        elif [ "$status" = "Failed" ] || [ "$status" = "Terminated" ]; then
            log_error "Sandbox failed with status: $status"
            return 1
        fi

        sleep 2
        attempt=$((attempt + 1))
    done

    log_error "Timeout waiting for sandbox to be ready"
    return 1
}

# Test command execution via data-plane (direct to sandbox domain)
test_exec() {
    log_test "Testing command execution via data-plane..."

    if [ -z "$SANDBOX_ID" ]; then
        log_error "No sandbox ID - run create first"
        return 1
    fi

    # Retrieve exec-agent secret and set up data-plane access
    retrieve_exec_agent_secret || {
        log_warn "Skipping data-plane tests: could not retrieve exec-agent secret"
        return 1
    }
    setup_data_plane_access

    # Test simple command via exec-agent
    local request='{"command": ["echo", "Hello, World!"]}'
    local response=$(data_plane_call POST "/exec" "$request")

    local stdout=$(echo "$response" | jq -r '.stdout')
    local exit_code=$(echo "$response" | jq -r '.exitCode')

    if [ "$exit_code" = "0" ] && [[ "$stdout" == *"Hello, World!"* ]]; then
        log_success "Exec test passed: $stdout"
    else
        log_error "Exec test failed: exit_code=$exit_code, stdout=$stdout, response=$response"
        return 1
    fi

    # Test code run
    log_test "Testing code run via data-plane..."

    request='{"code": "print(1 + 1)"}'
    response=$(data_plane_call POST "/run" "$request")

    stdout=$(echo "$response" | jq -r '.stdout')
    exit_code=$(echo "$response" | jq -r '.exitCode')

    if [ "$exit_code" = "0" ] && [[ "$stdout" == *"2"* ]]; then
        log_success "Code run test passed: $stdout"
    else
        log_error "Code run test failed: exit_code=$exit_code, stdout=$stdout, response=$response"
        return 1
    fi
}

# Test file operations via data-plane (direct to sandbox domain)
test_files() {
    log_test "Testing file operations via data-plane..."

    if [ -z "$SANDBOX_ID" ]; then
        log_error "No sandbox ID - run create first"
        return 1
    fi

    if [ -z "$EXEC_AGENT_SECRET" ]; then
        retrieve_exec_agent_secret || {
            log_warn "Skipping data-plane tests: could not retrieve exec-agent secret"
            return 1
        }
        setup_data_plane_access
    fi

    # Write file
    local request='{"path": "/workspace/test.txt", "content": "Hello from test!"}'
    local response=$(data_plane_call POST "/files/write" "$request")

    if echo "$response" | jq -e '.path' &>/dev/null; then
        log_success "File write succeeded"
    else
        log_error "File write failed: $response"
        return 1
    fi

    # Read file
    request='{"path": "/workspace/test.txt"}'
    response=$(data_plane_call POST "/files/read" "$request")

    local content=$(echo "$response" | jq -r '.content')
    if [[ "$content" == *"Hello from test!"* ]]; then
        log_success "File read succeeded: $content"
    else
        log_error "File read failed: $response"
        return 1
    fi

    # List files
    request='{"path": "/workspace"}'
    response=$(data_plane_call POST "/files/list" "$request")

    if echo "$response" | jq -e '.files | length > 0' &>/dev/null; then
        log_success "File list succeeded: $(echo "$response" | jq '.files | length') files"
    else
        log_warn "File list returned empty or failed: $response"
    fi
}

# Test snapshot via data-plane (direct to sandbox domain)
test_snapshot() {
    log_test "Testing snapshot creation via data-plane..."

    if [ -z "$SANDBOX_ID" ]; then
        log_error "No sandbox ID - run create first"
        return 1
    fi

    if [ -z "$EXEC_AGENT_SECRET" ]; then
        retrieve_exec_agent_secret || {
            log_warn "Skipping snapshot test: could not retrieve exec-agent secret"
            return 1
        }
        setup_data_plane_access
    fi

    # Create snapshot archive
    local response=$(data_plane_call POST "/snapshot/create" '{}')

    local snapshot_status=$(echo "$response" | jq -r '.status // empty')
    if [ "$snapshot_status" = "created" ] || [ "$snapshot_status" = "in_progress" ]; then
        log_success "Snapshot creation initiated: $snapshot_status"
    elif echo "$response" | jq -e '.error' &>/dev/null; then
        local error_msg=$(echo "$response" | jq -r '.error')
        if [[ "$error_msg" == *"not configured"* ]]; then
            log_warn "Snapshot not yet configured (expected in local testing): $error_msg"
        else
            log_error "Snapshot creation failed: $response"
            return 1
        fi
    else
        log_error "Unexpected snapshot response: $response"
        return 1
    fi

    # Check snapshot status
    response=$(data_plane_call GET "/snapshot/status")
    log_info "Snapshot status: $response"
}

# List sandboxes via control-plane API
list_sandboxes() {
    log_info "Listing sandboxes via control-plane API..."

    local response=$(api_call GET "/sandboxes")

    if echo "$response" | jq -e '.sandboxes' &>/dev/null; then
        local count=$(echo "$response" | jq '.sandboxes | length')
        log_success "Found $count sandboxes:"
        echo "$response" | jq -r '.sandboxes[] | "  - \(.sandboxId): \(.status) (\(.image))"'
    else
        log_info "No sandboxes found or error: $response"
    fi
}

# Cleanup
cleanup() {
    log_info "Cleaning up test sandboxes..."

    # Kill port-forward if running
    if [ -n "${DATA_PLANE_PF_PID:-}" ]; then
        kill "$DATA_PLANE_PF_PID" 2>/dev/null || true
    fi

    if [ -n "$SANDBOX_ID" ]; then
        local response=$(api_call DELETE "/sandboxes/$SANDBOX_ID")
        log_success "Deleted sandbox: $SANDBOX_ID"
    fi

    # Clean up any other test sandboxes
    local sandboxes=$(api_call GET "/sandboxes" | jq -r '.sandboxes[]?.sandboxId // empty')
    for id in $sandboxes; do
        if [[ "$id" == sb-* ]]; then
            api_call DELETE "/sandboxes/$id" >/dev/null
            log_info "Deleted: $id"
        fi
    done

    log_success "Cleanup completed"
}

# Run all tests
run_all_tests() {
    log_info "Running all sandbox E2E tests..."
    echo ""

    check_prerequisites
    echo ""

    setup_crd
    echo ""

    test_create_sandbox || { log_error "Create test failed"; cleanup; exit 1; }
    echo ""

    test_exec || { log_error "Exec test failed"; cleanup; exit 1; }
    echo ""

    test_files || { log_error "Files test failed"; cleanup; exit 1; }
    echo ""

    test_snapshot || { log_error "Snapshot test failed"; cleanup; exit 1; }
    echo ""

    cleanup
    echo ""

    log_success "All sandbox E2E tests passed!"
}

# ============================================================================
# kubectl-only tests (when API is not available)
# ============================================================================

KUBECTL_SANDBOX_NAME=""

kubectl_create_sandbox() {
    log_test "Creating sandbox via kubectl..."

    KUBECTL_SANDBOX_NAME="test-sandbox-$(date +%s)"

    # Uses the new CRD schema: image (not language), no networkIsolation: none
    kubectl apply -f - <<EOF
apiVersion: basilica.ai/v1
kind: BasilicaSandbox
metadata:
  name: ${KUBECTL_SANDBOX_NAME}
  namespace: ${NAMESPACE}
spec:
  userId: "test-user"
  sandboxId: "${KUBECTL_SANDBOX_NAME}"
  image: "basilica/sandbox-python:latest"
  cpu: "500m"
  memory: "512Mi"
  ttlSeconds: 300
  networkIsolation: Egress
EOF

    log_success "Sandbox created: $KUBECTL_SANDBOX_NAME"

    # Wait for sandbox to be processed
    log_info "Waiting for sandbox..."
    sleep 5

    # Show status
    kubectl get basilicasandbox "$KUBECTL_SANDBOX_NAME" -n "$NAMESPACE" -o wide
}

kubectl_test_sandbox() {
    log_test "Testing sandbox via kubectl..."

    if [ -z "$KUBECTL_SANDBOX_NAME" ]; then
        KUBECTL_SANDBOX_NAME=$(kubectl get basilicasandbox -n "$NAMESPACE" -o jsonpath='{.items[0].metadata.name}' 2>/dev/null || echo "")
        if [ -z "$KUBECTL_SANDBOX_NAME" ]; then
            log_error "No sandbox found. Run 'kubectl-create' first."
            return 1
        fi
    fi

    # Get pod name from sandbox status
    local pod_name=$(kubectl get basilicasandbox "$KUBECTL_SANDBOX_NAME" -n "$NAMESPACE" -o jsonpath='{.status.podName}' 2>/dev/null || echo "")

    if [ -z "$pod_name" ]; then
        log_warn "Sandbox pod not yet created, checking for matching pods..."
        pod_name=$(kubectl get pods -n "$NAMESPACE" -l "basilica.ai/type=sandbox" -o jsonpath='{.items[0].metadata.name}' 2>/dev/null || echo "")
    fi

    if [ -z "$pod_name" ]; then
        log_error "No sandbox pod found"
        return 1
    fi

    log_info "Using pod: $pod_name"

    # Wait for pod to be ready
    log_info "Waiting for pod to be ready..."
    kubectl wait --for=condition=Ready pod/"$pod_name" -n "$NAMESPACE" --timeout=60s || {
        log_warn "Pod not ready, checking status..."
        kubectl describe pod "$pod_name" -n "$NAMESPACE" | tail -20
        return 1
    }

    # Test exec in sandbox container
    log_info "Testing exec in sandbox..."
    local result=$(kubectl exec -n "$NAMESPACE" "$pod_name" -c sandbox -- python3 -c "print('Hello from sandbox!')" 2>&1)

    if [[ "$result" == *"Hello from sandbox!"* ]]; then
        log_success "Exec test passed: $result"
    else
        log_error "Exec test failed: $result"
        return 1
    fi

    # Test file operations
    log_info "Testing file operations..."
    kubectl exec -n "$NAMESPACE" "$pod_name" -c sandbox -- sh -c "echo 'test content' > /workspace/test.txt"
    local content=$(kubectl exec -n "$NAMESPACE" "$pod_name" -c sandbox -- cat /workspace/test.txt 2>&1)

    if [[ "$content" == *"test content"* ]]; then
        log_success "File operations test passed"
    else
        log_error "File operations test failed: $content"
        return 1
    fi

    log_success "All kubectl tests passed"
}

kubectl_cleanup() {
    log_info "Cleaning up kubectl-created sandboxes..."

    kubectl delete basilicasandbox --all -n "$NAMESPACE" --ignore-not-found

    log_success "Cleanup completed"
}

kubectl_list() {
    log_info "Listing sandboxes via kubectl..."

    echo "=== BasilicaSandbox CRs ==="
    kubectl get basilicasandbox -n "$NAMESPACE" -o wide 2>/dev/null || echo "No sandboxes found"
    echo ""

    echo "=== Sandbox Pods ==="
    kubectl get pods -n "$NAMESPACE" -l "basilica.ai/type=sandbox" -o wide 2>/dev/null || echo "No sandbox pods"
}

kubectl_run_all() {
    log_info "Running kubectl-only E2E tests..."
    echo ""

    check_prerequisites
    echo ""

    setup_crd
    echo ""

    kubectl_create_sandbox || { log_error "Create failed"; kubectl_cleanup; exit 1; }
    echo ""

    kubectl_test_sandbox || { log_error "Test failed"; kubectl_cleanup; exit 1; }
    echo ""

    kubectl_cleanup
    echo ""

    log_success "All kubectl E2E tests passed!"
}

# Main
case "${1:-all}" in
    setup)
        check_prerequisites
        setup_crd
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
    list)
        list_sandboxes
        ;;
    cleanup)
        cleanup
        ;;
    all)
        run_all_tests
        ;;
    # kubectl-only commands
    kubectl-create)
        kubectl_create_sandbox
        ;;
    kubectl-test)
        kubectl_test_sandbox
        ;;
    kubectl-list)
        kubectl_list
        ;;
    kubectl-cleanup)
        kubectl_cleanup
        ;;
    kubectl-all)
        kubectl_run_all
        ;;
    *)
        echo "Usage: $0 {setup|create|exec|files|snapshot|list|cleanup|all}"
        echo ""
        echo "API-based tests (control-plane via API, data-plane via sandbox domain):"
        echo "  setup     - Apply CRD and check prerequisites"
        echo "  create    - Create sandbox via API (POST /sandboxes)"
        echo "  exec      - Test exec via data-plane (POST https://<domain>/exec)"
        echo "  files     - Test files via data-plane (POST https://<domain>/files/*)"
        echo "  snapshot  - Test snapshot via data-plane (POST https://<domain>/snapshot/*)"
        echo "  list      - List sandboxes via API (GET /sandboxes)"
        echo "  cleanup   - Delete test sandboxes via API"
        echo "  all       - Run all tests"
        echo ""
        echo "kubectl-only commands (no API required):"
        echo "  kubectl-create   - Create sandbox via kubectl"
        echo "  kubectl-test     - Test sandbox via kubectl exec"
        echo "  kubectl-list     - List sandboxes via kubectl"
        echo "  kubectl-cleanup  - Delete all sandboxes"
        echo "  kubectl-all      - Run all kubectl tests"
        exit 1
        ;;
esac
