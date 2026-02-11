#!/bin/bash
#
# Sandbox E2E Testing Script
#
# Tests the BasilicaSandbox CRD, controller, and API endpoints.
# Run this after the core services are up (test-workflow.sh start).
#
# Prerequisites:
#   - K3s cluster running (or minikube/kind with basilica CRDs applied)
#   - basilica-api and basilica-operator deployed
#   - BASILICA_API_URL and BASILICA_API_TOKEN environment variables set
#
# Usage:
#   ./sandbox-e2e.sh [command]
#
# Commands:
#   setup     - Apply CRD and check prerequisites
#   create    - Create a test sandbox
#   exec      - Test command execution
#   files     - Test file operations
#   snapshot  - Test snapshot creation
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
BASILICA_API_URL="${BASILICA_API_URL:-http://localhost:8080}"
BASILICA_API_TOKEN="${BASILICA_API_TOKEN:-test-token}"
NAMESPACE="${NAMESPACE:-default}"
SANDBOX_ID=""

# Logging helpers
log_info() { echo -e "\033[34m[INFO]\033[0m $1"; }
log_success() { echo -e "\033[32m[SUCCESS]\033[0m $1"; }
log_warn() { echo -e "\033[33m[WARN]\033[0m $1"; }
log_error() { echo -e "\033[31m[ERROR]\033[0m $1"; }
log_test() { echo -e "\033[35m[TEST]\033[0m $1"; }

# API helper
api_call() {
    local method="$1"
    local path="$2"
    local data="$3"
    
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
    
    if command -v kubectl &>/dev/null; then
        # Try to use the generated CRD file from basilica-backend.
        local crd_file="$BASILICA_BACKEND_DIR/orchestrator/k8s/crds/basilica-sandbox.yaml"
        
        if [ -f "$crd_file" ]; then
            kubectl apply -f "$crd_file"
            log_success "CRD applied from $crd_file"
        else
            log_warn "CRD file not found at $crd_file - trying to generate"
            # Fallback: try to generate from operator
            local operator_dir="$BASILICA_BACKEND_DIR"
            if [ -d "$operator_dir" ] && [ -f "$operator_dir/Cargo.toml" ]; then
                (cd "$operator_dir" && cargo run --package basilica-operator --bin crdgen 2>/dev/null | \
                    sed -n '/kind: CustomResourceDefinition/,/---/{/---/!p}' | \
                    grep -A 1000 "basilicasandboxes.basilica.ai" | kubectl apply -f -)
                log_success "CRD generated and applied"
            else
                log_error "Could not find CRD file or operator to generate it."
                log_error "Set BASILICA_BACKEND_DIR to a local basilica-backend checkout."
                log_error "Example: export BASILICA_BACKEND_DIR=\$HOME/code/basilica-backend"
                return 1
            fi
        fi
        
        # Apply RBAC for sandboxes
        local rbac_file="$BASILICA_BACKEND_DIR/orchestrator/k8s/services/sandbox-rbac.yaml"
        if [ -f "$rbac_file" ]; then
            kubectl apply -f "$rbac_file"
            log_success "RBAC applied"
        fi
    else
        log_warn "kubectl not available - skipping CRD setup"
    fi
}

# Create sandbox
test_create_sandbox() {
    log_test "Testing sandbox creation..."
    
    local request='{
        "language": "python",
        "resources": {
            "cpu": "500m",
            "memory": "512Mi"
        },
        "env": [
            {"name": "TEST_VAR", "value": "hello"}
        ],
        "timeoutSeconds": 3600,
        "idleTimeoutSeconds": 600,
        "autoSnapshot": false,
        "networkIsolation": "none"
    }'
    
    local response=$(api_call POST "/api/v1/sandboxes" "$request")
    
    if echo "$response" | jq -e '.sandboxId' &>/dev/null; then
        SANDBOX_ID=$(echo "$response" | jq -r '.sandboxId')
        log_success "Sandbox created: $SANDBOX_ID"
    else
        log_error "Failed to create sandbox: $response"
        return 1
    fi
    
    # Wait for sandbox to be ready
    log_info "Waiting for sandbox to be ready..."
    local max_attempts=60
    local attempt=0
    
    while [ $attempt -lt $max_attempts ]; do
        local status=$(api_call GET "/api/v1/sandboxes/$SANDBOX_ID" | jq -r '.state')
        
        if [ "$status" = "Ready" ]; then
            log_success "Sandbox is ready"
            return 0
        elif [ "$status" = "Failed" ] || [ "$status" = "Terminated" ]; then
            log_error "Sandbox failed with state: $status"
            return 1
        fi
        
        sleep 2
        attempt=$((attempt + 1))
    done
    
    log_error "Timeout waiting for sandbox to be ready"
    return 1
}

# Test command execution
test_exec() {
    log_test "Testing command execution..."
    
    if [ -z "$SANDBOX_ID" ]; then
        log_error "No sandbox ID - run create first"
        return 1
    fi
    
    # Test simple command
    local request='{"command": ["echo", "Hello, World!"]}'
    local response=$(api_call POST "/api/v1/sandboxes/$SANDBOX_ID/exec" "$request")
    
    local stdout=$(echo "$response" | jq -r '.stdout')
    local exit_code=$(echo "$response" | jq -r '.exitCode')
    
    if [ "$exit_code" = "0" ] && [[ "$stdout" == *"Hello, World!"* ]]; then
        log_success "Exec test passed: $stdout"
    else
        log_error "Exec test failed: exit_code=$exit_code, stdout=$stdout"
        return 1
    fi
    
    # Test code run
    log_test "Testing code run..."
    
    request='{"code": "print(1 + 1)"}'
    response=$(api_call POST "/api/v1/sandboxes/$SANDBOX_ID/run" "$request")
    
    stdout=$(echo "$response" | jq -r '.stdout')
    exit_code=$(echo "$response" | jq -r '.exitCode')
    
    if [ "$exit_code" = "0" ] && [[ "$stdout" == *"2"* ]]; then
        log_success "Code run test passed: $stdout"
    else
        log_error "Code run test failed: exit_code=$exit_code, stdout=$stdout"
        return 1
    fi
}

# Test file operations
test_files() {
    log_test "Testing file operations..."
    
    if [ -z "$SANDBOX_ID" ]; then
        log_error "No sandbox ID - run create first"
        return 1
    fi
    
    # Write file
    local request='{"path": "/workspace/test.txt", "content": "Hello from test!"}'
    local response=$(api_call POST "/api/v1/sandboxes/$SANDBOX_ID/files/write" "$request")
    
    if echo "$response" | jq -e '.path' &>/dev/null; then
        log_success "File write succeeded"
    else
        log_error "File write failed: $response"
        return 1
    fi
    
    # Read file
    request='{"path": "/workspace/test.txt"}'
    response=$(api_call POST "/api/v1/sandboxes/$SANDBOX_ID/files/read" "$request")
    
    local content=$(echo "$response" | jq -r '.content')
    if [[ "$content" == *"Hello from test!"* ]]; then
        log_success "File read succeeded: $content"
    else
        log_error "File read failed: $response"
        return 1
    fi
    
    # List files
    request='{"path": "/workspace"}'
    response=$(api_call POST "/api/v1/sandboxes/$SANDBOX_ID/files/list" "$request")
    
    if echo "$response" | jq -e '.files | length > 0' &>/dev/null; then
        log_success "File list succeeded: $(echo "$response" | jq '.files | length') files"
    else
        log_warn "File list returned empty or failed: $response"
    fi
}

# Test snapshot
test_snapshot() {
    log_test "Testing snapshot creation..."
    
    if [ -z "$SANDBOX_ID" ]; then
        log_error "No sandbox ID - run create first"
        return 1
    fi
    
    local request='{"name": "test-snapshot"}'
    local response=$(api_call POST "/api/v1/sandboxes/$SANDBOX_ID/snapshot" "$request")
    
    local snapshot_id=$(echo "$response" | jq -r '.snapshotId')
    if [ -n "$snapshot_id" ] && [ "$snapshot_id" != "null" ]; then
        log_success "Snapshot created: $snapshot_id"
    else
        log_warn "Snapshot creation may have failed: $response"
    fi
}

# List sandboxes
list_sandboxes() {
    log_info "Listing sandboxes..."
    
    local response=$(api_call GET "/api/v1/sandboxes")
    
    if echo "$response" | jq -e '.sandboxes' &>/dev/null; then
        local count=$(echo "$response" | jq '.sandboxes | length')
        log_success "Found $count sandboxes:"
        echo "$response" | jq -r '.sandboxes[] | "  - \(.sandboxId): \(.state) (\(.language))"'
    else
        log_info "No sandboxes found or error: $response"
    fi
}

# Cleanup
cleanup() {
    log_info "Cleaning up test sandboxes..."
    
    if [ -n "$SANDBOX_ID" ]; then
        local response=$(api_call DELETE "/api/v1/sandboxes/$SANDBOX_ID")
        log_success "Deleted sandbox: $SANDBOX_ID"
    fi
    
    # Clean up any other test sandboxes
    local sandboxes=$(api_call GET "/api/v1/sandboxes" | jq -r '.sandboxes[]?.sandboxId // empty')
    for id in $sandboxes; do
        if [[ "$id" == sandbox-* ]]; then
            api_call DELETE "/api/v1/sandboxes/$id" >/dev/null
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
    
    test_snapshot || log_warn "Snapshot test had issues"
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
    
    kubectl apply -f - <<EOF
apiVersion: basilica.ai/v1
kind: BasilicaSandbox
metadata:
  name: ${KUBECTL_SANDBOX_NAME}
  namespace: ${NAMESPACE}
spec:
  userId: "test-user"
  language: python
  resources:
    cpu: "500m"
    memory: "512Mi"
  timeoutSeconds: 300
  idleTimeoutSeconds: 60
  networkIsolation: none
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
        echo "kubectl-only commands (no API required):"
        echo "  kubectl-create   - Create sandbox via kubectl"
        echo "  kubectl-test     - Test sandbox via kubectl exec"
        echo "  kubectl-list     - List sandboxes via kubectl"
        echo "  kubectl-cleanup  - Delete all sandboxes"
        echo "  kubectl-all      - Run all kubectl tests"
        exit 1
        ;;
esac

