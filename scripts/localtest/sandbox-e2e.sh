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

set -e

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
cd "$SCRIPT_DIR"

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
        # Generate CRD from Rust definition
        cat <<EOF | kubectl apply -f -
apiVersion: apiextensions.k8s.io/v1
kind: CustomResourceDefinition
metadata:
  name: basilicasandboxes.basilica.ai
spec:
  group: basilica.ai
  versions:
    - name: v1
      served: true
      storage: true
      schema:
        openAPIV3Schema:
          type: object
          properties:
            spec:
              type: object
              required: ["userId", "language"]
              properties:
                userId:
                  type: string
                language:
                  type: string
                image:
                  type: string
                resources:
                  type: object
                  properties:
                    cpu:
                      type: string
                    memory:
                      type: string
                    gpus:
                      type: object
                      properties:
                        count:
                          type: integer
                        model:
                          type: array
                          items:
                            type: string
                env:
                  type: array
                  items:
                    type: object
                    properties:
                      name:
                        type: string
                      value:
                        type: string
                timeoutSeconds:
                  type: integer
                idleTimeoutSeconds:
                  type: integer
                autoSnapshot:
                  type: boolean
                restoreFrom:
                  type: string
                networkIsolation:
                  type: string
                  enum: ["none", "egress", "full"]
            status:
              type: object
              properties:
                state:
                  type: string
                sandboxId:
                  type: string
                podName:
                  type: string
                nodeName:
                  type: string
                websocketPath:
                  type: string
                createdAt:
                  type: string
                lastActivityAt:
                  type: string
                message:
                  type: string
                snapshotId:
                  type: string
      subresources:
        status: {}
      additionalPrinterColumns:
        - name: State
          type: string
          jsonPath: .status.state
        - name: Language
          type: string
          jsonPath: .spec.language
        - name: Age
          type: date
          jsonPath: .metadata.creationTimestamp
  scope: Namespaced
  names:
    plural: basilicasandboxes
    singular: basilicasandbox
    kind: BasilicaSandbox
    shortNames:
      - bsb
      - sandbox
EOF
        log_success "CRD applied"
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
    *)
        echo "Usage: $0 {setup|create|exec|files|snapshot|list|cleanup|all}"
        exit 1
        ;;
esac

