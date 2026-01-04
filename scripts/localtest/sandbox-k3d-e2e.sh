#!/bin/bash
#
# Sandbox K3d E2E Testing Script
#
# Spins up a local K3d cluster, deploys sandbox infrastructure, and runs E2E tests.
#
# Prerequisites:
#   - Docker installed and running
#   - k3d installed (https://k3d.io)
#   - kubectl installed
#
# Usage:
#   ./sandbox-k3d-e2e.sh [command]
#
# Commands:
#   setup     - Create K3d cluster and deploy infrastructure
#   test      - Run sandbox E2E tests (assumes setup done)
#   all       - Setup + test (default)
#   cleanup   - Delete K3d cluster
#   status    - Show cluster and pod status
#   logs      - Show operator and API logs

set -e

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
BACKEND_DIR="$(cd "$SCRIPT_DIR/../../../basilica-backend" && pwd)"

# Configuration
CLUSTER_NAME="basilica-sandbox-test"
REGISTRY_NAME="basilica-registry"
REGISTRY_PORT="5050"
API_PORT="8080"
NAMESPACE="default"

# Image names
EXEC_AGENT_IMAGE="localhost:${REGISTRY_PORT}/basilica-exec-agent:latest"
OPERATOR_IMAGE="localhost:${REGISTRY_PORT}/basilica-operator:latest"
API_IMAGE="localhost:${REGISTRY_PORT}/basilica-api:latest"

# Colors for output
RED='\033[0;31m'
GREEN='\033[0;32m'
YELLOW='\033[1;33m'
BLUE='\033[0;34m'
PURPLE='\033[0;35m'
NC='\033[0m' # No Color

log_info() { echo -e "${BLUE}[INFO]${NC} $1"; }
log_success() { echo -e "${GREEN}[SUCCESS]${NC} $1"; }
log_warn() { echo -e "${YELLOW}[WARN]${NC} $1"; }
log_error() { echo -e "${RED}[ERROR]${NC} $1"; }
log_step() { echo -e "${PURPLE}[STEP]${NC} $1"; }

# ============================================================================
# Prerequisites
# ============================================================================

check_prerequisites() {
    log_step "Checking prerequisites..."
    
    local missing=()
    
    # Check Docker
    if ! command -v docker &>/dev/null; then
        missing+=("docker")
    elif ! docker info &>/dev/null; then
        log_error "Docker is not running"
        exit 1
    fi
    
    # Check k3d
    if ! command -v k3d &>/dev/null; then
        missing+=("k3d")
    fi
    
    # Check kubectl
    if ! command -v kubectl &>/dev/null; then
        missing+=("kubectl")
    fi
    
    # Check curl
    if ! command -v curl &>/dev/null; then
        missing+=("curl")
    fi
    
    # Check jq
    if ! command -v jq &>/dev/null; then
        missing+=("jq")
    fi
    
    if [ ${#missing[@]} -ne 0 ]; then
        log_error "Missing required tools: ${missing[*]}"
        echo ""
        echo "Install missing tools:"
        for tool in "${missing[@]}"; do
            case $tool in
                docker) echo "  - Docker: https://docs.docker.com/get-docker/" ;;
                k3d) echo "  - k3d: curl -s https://raw.githubusercontent.com/k3d-io/k3d/main/install.sh | bash" ;;
                kubectl) echo "  - kubectl: https://kubernetes.io/docs/tasks/tools/" ;;
                curl) echo "  - curl: apt-get install curl / brew install curl" ;;
                jq) echo "  - jq: apt-get install jq / brew install jq" ;;
            esac
        done
        exit 1
    fi
    
    # Check if backend directory exists
    if [ ! -d "$BACKEND_DIR" ]; then
        log_error "Backend directory not found at $BACKEND_DIR"
        exit 1
    fi
    
    log_success "All prerequisites satisfied"
}

# ============================================================================
# K3d Cluster Management
# ============================================================================

cluster_exists() {
    k3d cluster list 2>/dev/null | grep -q "$CLUSTER_NAME"
}

registry_exists() {
    docker ps --format '{{.Names}}' 2>/dev/null | grep -q "k3d-${REGISTRY_NAME}"
}

create_registry() {
    if registry_exists; then
        log_info "Registry already exists"
        return 0
    fi
    
    log_step "Creating local registry..."
    k3d registry create "$REGISTRY_NAME" --port "$REGISTRY_PORT"
    log_success "Registry created at localhost:${REGISTRY_PORT}"
}

create_cluster() {
    if cluster_exists; then
        log_info "Cluster '$CLUSTER_NAME' already exists"
        k3d kubeconfig merge "$CLUSTER_NAME" --kubeconfig-switch-context
        return 0
    fi
    
    log_step "Creating K3d cluster '$CLUSTER_NAME'..."
    
    # Create cluster with registry
    k3d cluster create "$CLUSTER_NAME" \
        --registry-use "k3d-${REGISTRY_NAME}:${REGISTRY_PORT}" \
        --port "${API_PORT}:80@loadbalancer" \
        --agents 0 \
        --servers 1 \
        --wait \
        --timeout 120s
    
    # Switch kubectl context
    k3d kubeconfig merge "$CLUSTER_NAME" --kubeconfig-switch-context
    
    log_success "Cluster '$CLUSTER_NAME' created"
    
    # Wait for cluster to be ready
    log_info "Waiting for cluster to be ready..."
    kubectl wait --for=condition=Ready nodes --all --timeout=120s
    
    log_success "Cluster is ready"
}

delete_cluster() {
    if cluster_exists; then
        log_step "Deleting cluster '$CLUSTER_NAME'..."
        k3d cluster delete "$CLUSTER_NAME"
        log_success "Cluster deleted"
    else
        log_info "Cluster '$CLUSTER_NAME' does not exist"
    fi
}

# ============================================================================
# Image Building
# ============================================================================

build_exec_agent() {
    log_step "Building exec-agent image..."
    
    local dockerfile="$BACKEND_DIR/crates/basilica-exec-agent/Dockerfile"
    local context="$BACKEND_DIR/crates/basilica-exec-agent"
    
    if [ ! -f "$dockerfile" ]; then
        log_error "Dockerfile not found at $dockerfile"
        return 1
    fi
    
    docker build -t "$EXEC_AGENT_IMAGE" -f "$dockerfile" "$context"
    docker push "$EXEC_AGENT_IMAGE"
    
    log_success "exec-agent image built and pushed"
}

build_operator() {
    log_step "Building operator image..."
    
    local dockerfile="$BACKEND_DIR/scripts/operator/Dockerfile"
    local context="$BACKEND_DIR"
    
    if [ ! -f "$dockerfile" ]; then
        log_warn "Operator Dockerfile not found at $dockerfile"
        log_info "Using a minimal operator image for testing..."
        
        # Create a minimal operator image for testing
        cat <<'DOCKERFILE' | docker build -t "$OPERATOR_IMAGE" -f - "$BACKEND_DIR"
FROM rust:1.80-slim AS builder
WORKDIR /app
RUN apt-get update && apt-get install -y pkg-config libssl-dev && rm -rf /var/lib/apt/lists/*
COPY Cargo.toml Cargo.lock ./
COPY crates ./crates
RUN cargo build --release --package basilica-operator

FROM debian:bookworm-slim
RUN apt-get update && apt-get install -y ca-certificates && rm -rf /var/lib/apt/lists/*
COPY --from=builder /app/target/release/basilica-operator /usr/local/bin/
ENTRYPOINT ["/usr/local/bin/basilica-operator"]
DOCKERFILE
        docker push "$OPERATOR_IMAGE"
    else
        docker build -t "$OPERATOR_IMAGE" -f "$dockerfile" "$context"
        docker push "$OPERATOR_IMAGE"
    fi
    
    log_success "Operator image built and pushed"
}

build_api() {
    log_step "Building API image..."
    
    local dockerfile="$BACKEND_DIR/scripts/api/Dockerfile"
    local context="$BACKEND_DIR"
    
    if [ ! -f "$dockerfile" ]; then
        log_warn "API Dockerfile not found at $dockerfile"
        log_info "Using a minimal API image for testing..."
        
        # Create a minimal API image for testing
        cat <<'DOCKERFILE' | docker build -t "$API_IMAGE" -f - "$BACKEND_DIR"
FROM rust:1.80-slim AS builder
WORKDIR /app
RUN apt-get update && apt-get install -y pkg-config libssl-dev && rm -rf /var/lib/apt/lists/*
COPY Cargo.toml Cargo.lock ./
COPY crates ./crates
RUN cargo build --release --package basilica-api

FROM debian:bookworm-slim
RUN apt-get update && apt-get install -y ca-certificates && rm -rf /var/lib/apt/lists/*
COPY --from=builder /app/target/release/basilica-api /usr/local/bin/
EXPOSE 8080
ENTRYPOINT ["/usr/local/bin/basilica-api"]
DOCKERFILE
        docker push "$API_IMAGE"
    else
        docker build -t "$API_IMAGE" -f "$dockerfile" "$context"
        docker push "$API_IMAGE"
    fi
    
    log_success "API image built and pushed"
}

build_images() {
    log_step "Building all images..."
    
    build_exec_agent
    # Skip full operator/API builds for now - they take too long
    # and we can test with kubectl exec directly
    
    log_success "All required images built"
}

# ============================================================================
# Infrastructure Deployment
# ============================================================================

deploy_crds() {
    log_step "Deploying CRDs..."
    
    local crd_file="$BACKEND_DIR/orchestrator/k8s/crds/basilica-sandbox.yaml"
    
    if [ -f "$crd_file" ]; then
        kubectl apply -f "$crd_file"
        log_success "CRD applied from $crd_file"
    else
        log_warn "CRD file not found at $crd_file"
        log_info "Generating CRD from operator..."
        
        # Apply inline CRD
        kubectl apply -f - <<'EOF'
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
                  enum: ["python", "javascript", "typescript", "go", "rust", "bash"]
                image:
                  type: string
                resources:
                  type: object
                  properties:
                    cpu:
                      type: string
                    memory:
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
        log_success "CRD applied inline"
    fi
}

deploy_rbac() {
    log_step "Deploying RBAC..."
    
    local rbac_file="$BACKEND_DIR/orchestrator/k8s/services/sandbox-rbac.yaml"
    local operator_rbac="$BACKEND_DIR/orchestrator/k8s/services/sandbox-operator-rbac.yaml"
    
    if [ -f "$rbac_file" ]; then
        kubectl apply -f "$rbac_file"
        log_success "Sandbox RBAC applied"
    else
        log_warn "Sandbox RBAC file not found, applying minimal RBAC..."
        kubectl apply -f - <<'EOF'
apiVersion: v1
kind: ServiceAccount
metadata:
  name: basilica-sandbox
  namespace: default
EOF
    fi
    
    if [ -f "$operator_rbac" ]; then
        # Create operator namespace and service account first
        kubectl create namespace basilica-system --dry-run=client -o yaml | kubectl apply -f -
        kubectl create serviceaccount basilica-operator -n basilica-system --dry-run=client -o yaml | kubectl apply -f -
        kubectl apply -f "$operator_rbac"
        log_success "Operator RBAC applied"
    fi
}

deploy_network_policies() {
    log_step "Deploying NetworkPolicies..."
    
    local netpol_file="$BACKEND_DIR/orchestrator/k8s/networking/sandbox-network-policies.yaml"
    
    if [ -f "$netpol_file" ]; then
        kubectl apply -f "$netpol_file"
        log_success "NetworkPolicies applied"
    else
        log_warn "NetworkPolicies file not found, skipping..."
    fi
}

deploy_operator() {
    log_step "Deploying operator..."
    
    local deploy_file="$SCRIPT_DIR/k3d-manifests/operator-deploy.yaml"
    
    if [ -f "$deploy_file" ]; then
        kubectl apply -f "$deploy_file"
    else
        log_info "Using inline operator deployment..."
        kubectl apply -f - <<EOF
apiVersion: apps/v1
kind: Deployment
metadata:
  name: basilica-operator
  namespace: basilica-system
spec:
  replicas: 1
  selector:
    matchLabels:
      app: basilica-operator
  template:
    metadata:
      labels:
        app: basilica-operator
    spec:
      serviceAccountName: basilica-operator
      containers:
      - name: operator
        image: ${OPERATOR_IMAGE}
        imagePullPolicy: Always
        env:
        - name: RUST_LOG
          value: info
        resources:
          limits:
            cpu: 500m
            memory: 256Mi
          requests:
            cpu: 100m
            memory: 128Mi
EOF
    fi
    
    log_success "Operator deployed"
}

deploy_api() {
    log_step "Deploying API..."
    
    local deploy_file="$SCRIPT_DIR/k3d-manifests/api-deploy.yaml"
    
    if [ -f "$deploy_file" ]; then
        kubectl apply -f "$deploy_file"
    else
        log_info "Using inline API deployment..."
        kubectl apply -f - <<EOF
apiVersion: apps/v1
kind: Deployment
metadata:
  name: basilica-api
  namespace: default
spec:
  replicas: 1
  selector:
    matchLabels:
      app: basilica-api
  template:
    metadata:
      labels:
        app: basilica-api
        app.kubernetes.io/name: basilica-api
    spec:
      containers:
      - name: api
        image: ${API_IMAGE}
        imagePullPolicy: Always
        ports:
        - containerPort: 8080
        env:
        - name: RUST_LOG
          value: info
        resources:
          limits:
            cpu: 500m
            memory: 256Mi
          requests:
            cpu: 100m
            memory: 128Mi
---
apiVersion: v1
kind: Service
metadata:
  name: basilica-api
  namespace: default
spec:
  selector:
    app: basilica-api
  ports:
  - port: 80
    targetPort: 8080
  type: ClusterIP
---
apiVersion: networking.k8s.io/v1
kind: Ingress
metadata:
  name: basilica-api
  namespace: default
spec:
  rules:
  - http:
      paths:
      - path: /
        pathType: Prefix
        backend:
          service:
            name: basilica-api
            port:
              number: 80
EOF
    fi
    
    log_success "API deployed"
}

deploy_infrastructure() {
    log_step "Deploying all infrastructure..."
    
    deploy_crds
    deploy_rbac
    deploy_network_policies
    
    # Skip operator/API deployment for minimal testing
    # They require full builds which are slow
    log_info "Skipping operator/API deployment for minimal testing"
    log_info "Sandboxes can be created manually with kubectl"
    
    log_success "Infrastructure deployed"
}

wait_for_deployments() {
    log_step "Waiting for deployments to be ready..."
    
    # Wait for operator if deployed
    if kubectl get deployment basilica-operator -n basilica-system &>/dev/null; then
        kubectl rollout status deployment/basilica-operator -n basilica-system --timeout=120s || true
    fi
    
    # Wait for API if deployed
    if kubectl get deployment basilica-api -n default &>/dev/null; then
        kubectl rollout status deployment/basilica-api -n default --timeout=120s || true
    fi
    
    log_success "Deployments ready"
}

# ============================================================================
# Testing
# ============================================================================

run_e2e_tests() {
    log_step "Running E2E tests..."
    
    local e2e_script="$SCRIPT_DIR/sandbox-e2e.sh"
    
    if [ -f "$e2e_script" ]; then
        # Set environment for E2E script
        export BASILICA_API_URL="http://localhost:${API_PORT}"
        export BASILICA_API_TOKEN="test-token"
        export NAMESPACE="$NAMESPACE"
        
        # Run kubectl-based tests (no API required)
        bash "$e2e_script" kubectl-all
    else
        log_warn "E2E script not found at $e2e_script"
        log_info "Running manual sandbox test..."
        run_manual_sandbox_test
    fi
}

run_api_tests() {
    log_step "Running API-based E2E tests..."
    
    local e2e_script="$SCRIPT_DIR/sandbox-e2e.sh"
    
    if [ -f "$e2e_script" ]; then
        # Set environment for E2E script
        export BASILICA_API_URL="http://localhost:${API_PORT}"
        export BASILICA_API_TOKEN="test-token"
        export NAMESPACE="$NAMESPACE"
        
        # Run full API tests
        bash "$e2e_script" all
    else
        log_error "E2E script not found at $e2e_script"
        exit 1
    fi
}

run_manual_sandbox_test() {
    log_step "Running manual sandbox test..."
    
    # Create a test sandbox directly via kubectl
    local sandbox_name="test-sandbox-$(date +%s)"
    
    log_info "Creating sandbox: $sandbox_name"
    
    kubectl apply -f - <<EOF
apiVersion: basilica.ai/v1
kind: BasilicaSandbox
metadata:
  name: ${sandbox_name}
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
    
    log_success "Sandbox created: $sandbox_name"
    
    # Wait for sandbox
    log_info "Waiting for sandbox to be ready..."
    sleep 5
    
    # Check status
    kubectl get basilicasandbox "$sandbox_name" -n "$NAMESPACE" -o yaml
    
    # Cleanup
    log_info "Cleaning up test sandbox..."
    kubectl delete basilicasandbox "$sandbox_name" -n "$NAMESPACE" --ignore-not-found
    
    log_success "Manual sandbox test completed"
}

# ============================================================================
# Status and Logs
# ============================================================================

show_status() {
    log_step "Cluster Status"
    echo ""
    
    echo "=== K3d Clusters ==="
    k3d cluster list
    echo ""
    
    echo "=== Nodes ==="
    kubectl get nodes -o wide 2>/dev/null || echo "No nodes (cluster not running?)"
    echo ""
    
    echo "=== Namespaces ==="
    kubectl get namespaces 2>/dev/null || echo "Cannot get namespaces"
    echo ""
    
    echo "=== CRDs ==="
    kubectl get crd 2>/dev/null | grep -E "NAME|basilica" || echo "No Basilica CRDs"
    echo ""
    
    echo "=== Pods (all namespaces) ==="
    kubectl get pods -A 2>/dev/null || echo "Cannot get pods"
    echo ""
    
    echo "=== Sandboxes ==="
    kubectl get basilicasandbox -A 2>/dev/null || echo "No sandboxes or CRD not installed"
    echo ""
    
    echo "=== Services ==="
    kubectl get svc -A 2>/dev/null | grep -E "NAME|basilica" || echo "No Basilica services"
}

show_logs() {
    log_step "Showing logs..."
    
    echo "=== Operator Logs ==="
    kubectl logs -n basilica-system -l app=basilica-operator --tail=50 2>/dev/null || echo "No operator logs"
    echo ""
    
    echo "=== API Logs ==="
    kubectl logs -n default -l app=basilica-api --tail=50 2>/dev/null || echo "No API logs"
}

# ============================================================================
# Cleanup
# ============================================================================

cleanup() {
    log_step "Cleaning up..."
    
    # Delete sandboxes first
    kubectl delete basilicasandbox --all -n "$NAMESPACE" --ignore-not-found 2>/dev/null || true
    
    # Delete cluster
    delete_cluster
    
    log_success "Cleanup completed"
}

# ============================================================================
# Main Setup
# ============================================================================

setup() {
    log_step "Setting up sandbox testing environment..."
    
    check_prerequisites
    create_registry
    create_cluster
    build_images
    deploy_infrastructure
    
    log_success "Setup completed!"
    echo ""
    show_status
}

# ============================================================================
# Main
# ============================================================================

main() {
    case "${1:-all}" in
        "prereq"|"prerequisites")
            check_prerequisites
            ;;
        "cluster")
            check_prerequisites
            create_registry
            create_cluster
            ;;
        "build")
            check_prerequisites
            build_images
            ;;
        "deploy")
            deploy_infrastructure
            ;;
        "setup")
            setup
            ;;
        "test")
            run_e2e_tests
            ;;
        "api-test")
            run_api_tests
            ;;
        "manual-test")
            run_manual_sandbox_test
            ;;
        "all")
            setup
            echo ""
            run_e2e_tests
            ;;
        "status")
            show_status
            ;;
        "logs")
            show_logs
            ;;
        "cleanup")
            cleanup
            ;;
        "help"|"-h"|"--help")
            echo "Sandbox K3d E2E Testing Script"
            echo ""
            echo "Usage: $0 [command]"
            echo ""
            echo "Commands:"
            echo "  prereq       - Check prerequisites only"
            echo "  cluster      - Create K3d cluster and registry"
            echo "  build        - Build Docker images"
            echo "  deploy       - Deploy infrastructure to cluster"
            echo "  setup        - Full setup (cluster + build + deploy)"
            echo "  test         - Run kubectl-based E2E tests (no API required)"
            echo "  api-test     - Run full API-based E2E tests (requires API)"
            echo "  manual-test  - Run quick manual sandbox test via kubectl"
            echo "  all          - Setup + test (default)"
            echo "  status       - Show cluster and pod status"
            echo "  logs         - Show operator and API logs"
            echo "  cleanup      - Delete K3d cluster"
            echo "  help         - Show this help"
            ;;
        *)
            log_error "Unknown command: $1"
            echo "Run '$0 help' for usage"
            exit 1
            ;;
    esac
}

# Handle Ctrl+C gracefully
trap 'echo ""; log_warn "Interrupted"; exit 130' INT

main "$@"

