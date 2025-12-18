#!/usr/bin/env bash
# Local E2E Testing for Training Service MVP
#
# Architecture:
#   - Basilica API: runs locally (cargo run)
#   - Operator, Training Pods, Envoy Gateway: run in k3d cluster
#
# Prerequisites:
#   - Docker Desktop
#   - k3d (will be installed if missing)
#   - kubectl
#   - Rust toolchain (for API)
#
# Usage:
#   ./scripts/local-training-e2e.sh cluster-up     # Start k3d cluster
#   ./scripts/local-training-e2e.sh deploy         # Deploy operator + gateway
#   ./scripts/local-training-e2e.sh api            # Run API locally
#   ./scripts/local-training-e2e.sh test           # Run e2e test
#   ./scripts/local-training-e2e.sh cluster-down   # Clean up

set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
ROOT_DIR="$(cd "$SCRIPT_DIR/.." && pwd)"
CLUSTER_NAME="basilica-training-local"
KUBECONFIG_PATH="$ROOT_DIR/build/k3s-training.yaml"

# Colors
RED='\033[0;31m'
GREEN='\033[0;32m'
YELLOW='\033[1;33m'
BLUE='\033[0;34m'
NC='\033[0m'

log_info() { echo -e "${GREEN}[INFO]${NC} $1"; }
log_warn() { echo -e "${YELLOW}[WARN]${NC} $1"; }
log_error() { echo -e "${RED}[ERROR]${NC} $1"; }
log_step() { echo -e "${BLUE}[STEP]${NC} $1"; }

check_prerequisites() {
    log_info "Checking prerequisites..."

    if ! command -v docker &> /dev/null; then
        log_error "Docker is required. Please install Docker Desktop."
        exit 1
    fi

    if ! docker info &> /dev/null; then
        log_error "Docker is not running. Please start Docker Desktop."
        exit 1
    fi

    if ! command -v kubectl &> /dev/null; then
        log_error "kubectl is required. Install with: brew install kubectl"
        exit 1
    fi

    # Install k3d if needed
    if ! command -v k3d &> /dev/null; then
        log_info "Installing k3d..."
        curl -s https://raw.githubusercontent.com/k3d-io/k3d/main/install.sh | bash
    fi

    log_info "Prerequisites OK"
}

cluster_up() {
    check_prerequisites

    log_step "Creating k3d cluster: $CLUSTER_NAME"

    if k3d cluster list 2>/dev/null | grep -q "$CLUSTER_NAME"; then
        log_warn "Cluster already exists, reusing it"
    else
        k3d cluster create "$CLUSTER_NAME" \
            --api-port 6550 \
            --port "9080:80@loadbalancer" \
            --port "9443:443@loadbalancer" \
            --k3s-arg "--disable=traefik@server:0" \
            --wait
    fi

    # Export kubeconfig
    mkdir -p "$ROOT_DIR/build"
    k3d kubeconfig get "$CLUSTER_NAME" > "$KUBECONFIG_PATH"

    # Wait for cluster to be ready
    export KUBECONFIG="$KUBECONFIG_PATH"
    kubectl wait --for=condition=Ready nodes --all --timeout=120s

    log_info "Cluster ready!"
    log_info "Kubeconfig: $KUBECONFIG_PATH"
    echo ""
    echo "To use this cluster:"
    echo "  export KUBECONFIG=$KUBECONFIG_PATH"
}

install_gateway() {
    export KUBECONFIG="$KUBECONFIG_PATH"

    log_step "Installing Gateway API CRDs..."
    kubectl apply -f https://github.com/kubernetes-sigs/gateway-api/releases/download/v1.2.0/standard-install.yaml

    log_step "Installing Envoy Gateway..."
    # Try helm first, fall back to kubectl
    if command -v helm &> /dev/null; then
        helm upgrade --install eg oci://docker.io/envoyproxy/gateway-helm \
            --version v1.2.1 \
            --namespace envoy-gateway-system \
            --create-namespace \
            --wait 2>/dev/null || {
                log_warn "Helm install failed, trying kubectl..."
                kubectl apply -f https://github.com/envoyproxy/gateway/releases/download/v1.2.1/install.yaml
            }
    else
        kubectl apply -f https://github.com/envoyproxy/gateway/releases/download/v1.2.1/install.yaml
    fi

    # Wait for Envoy Gateway
    log_info "Waiting for Envoy Gateway to be ready..."
    sleep 10
    kubectl wait --namespace envoy-gateway-system \
        --for=condition=Available deployment/envoy-gateway \
        --timeout=180s || log_warn "Timeout waiting for envoy-gateway, continuing..."

    log_info "Gateway installed"
}

build_and_load_images() {
    export KUBECONFIG="$KUBECONFIG_PATH"
    cd "$ROOT_DIR"

    log_step "Building training service image (CPU mode)..."
    docker build \
        -t basilica-training:local \
        -f services/training-service/Dockerfile \
        services/training-service

    log_step "Building operator image..."
    docker build \
        -t basilica-operator:local \
        -f scripts/operator/Dockerfile \
        .

    log_step "Loading images into k3d..."
    k3d image import basilica-training:local -c "$CLUSTER_NAME"
    k3d image import basilica-operator:local -c "$CLUSTER_NAME"

    log_info "Images ready"
}

deploy_operator() {
    export KUBECONFIG="$KUBECONFIG_PATH"
    cd "$ROOT_DIR"

    log_step "Applying CRDs and manifests..."

    # Apply TrainingSession CRD
    kubectl apply -f orchestrator/k8s/training/training-session-crd.yaml

    # Apply namespaces
    kubectl apply -f orchestrator/k8s/training/local-dev/namespace.yaml

    # Apply gateway resources
    kubectl apply -f orchestrator/k8s/training/local-dev/gateway.yaml

    # Apply operator
    kubectl apply -f orchestrator/k8s/training/local-dev/operator-deployment.yaml

    log_info "Waiting for operator..."
    kubectl wait --namespace basilica-system \
        --for=condition=Available deployment/basilica-operator \
        --timeout=120s

    log_info "Operator deployed"
}

deploy() {
    install_gateway
    build_and_load_images
    deploy_operator
    show_status
}

setup_postgres() {
    log_step "Starting Postgres for API..."
    cd "$ROOT_DIR"

    # Check if postgres container exists but user is missing (stale volume)
    if docker compose -f scripts/api/compose.dev.yml ps postgres 2>/dev/null | grep -q "running"; then
        # Test if the user exists
        if ! docker compose -f scripts/api/compose.dev.yml exec -T postgres psql -U api -d basilica_api -c "SELECT 1" &>/dev/null; then
            log_warn "Postgres has stale data, recreating..."
            docker compose -f scripts/api/compose.dev.yml down -v
        fi
    fi

    docker compose -f scripts/api/compose.dev.yml up -d postgres

    # Wait for Postgres to be ready
    log_info "Waiting for Postgres to be ready..."
    for i in {1..30}; do
        if docker compose -f scripts/api/compose.dev.yml exec -T postgres pg_isready -U api -d basilica_api &>/dev/null; then
            log_info "Postgres is ready"
            return 0
        fi
        sleep 1
    done
    log_error "Postgres failed to start"
    exit 1
}

reset_db() {
    log_step "Resetting Postgres database..."
    cd "$ROOT_DIR"

    docker compose -f scripts/api/compose.dev.yml down -v
    log_info "Database volume removed. Run '$0 api' to recreate."
}

setup_api_key() {
    log_step "Generating API key..."
    cd "$ROOT_DIR"

    # Generate key and capture output (include training scopes)
    KEY_OUTPUT=$(cargo run -p basilica-api --bin gen-api-key -- --user testuser --name training-e2e --scopes "training:*" 2>/dev/null)

    # Extract the token and SQL
    API_TOKEN=$(echo "$KEY_OUTPUT" | grep "Token (Authorization):" | sed 's/.*Bearer //')
    SQL_INSERT=$(echo "$KEY_OUTPUT" | grep "^INSERT INTO")

    if [ -z "$API_TOKEN" ] || [ -z "$SQL_INSERT" ]; then
        log_error "Failed to generate API key"
        echo "$KEY_OUTPUT"
        exit 1
    fi

    log_info "Generated API token: ${API_TOKEN:0:20}..."

    # Delete any existing key for this user/name, then insert new one
    docker compose -f scripts/api/compose.dev.yml exec -T postgres psql -U api -d basilica_api -c "DELETE FROM api_keys WHERE user_id='testuser' AND name='training-e2e';" 2>/dev/null || true
    docker compose -f scripts/api/compose.dev.yml exec -T postgres psql -U api -d basilica_api -c "$SQL_INSERT" 2>/dev/null || {
        log_error "Failed to insert API key into database"
        exit 1
    }

    # Save token to file for test script
    echo "$API_TOKEN" > "$ROOT_DIR/build/api-token.txt"
    log_info "Token saved to build/api-token.txt"
}

run_api() {
    log_step "Starting Basilica API locally..."
    log_info "API will connect to k3d cluster using kubeconfig"

    cd "$ROOT_DIR"

    # Ensure Postgres is running
    setup_postgres

    # Set environment for local API
    export KUBECONFIG="$KUBECONFIG_PATH"
    export RUST_LOG="info,basilica_api=debug"

    log_info "Starting API on http://localhost:8000 (DEV MODE)"
    log_info "Using config: config/api.local.toml"
    log_info "Press Ctrl+C to stop"
    echo ""
    echo "To get an API key, run in another terminal:"
    echo "  $0 gen-key"
    echo ""

    cargo run -p basilica-api --bin basilica-api -- --config config/api.local.toml
}

gen_key() {
    setup_postgres
    setup_api_key

    echo ""
    log_info "=== API Key Ready ==="
    echo "Token saved to: build/api-token.txt"
    echo ""
    echo "Use in requests:"
    echo "  curl -H \"Authorization: Bearer \$(cat build/api-token.txt)\" http://localhost:8000/sessions"
}

cleanup_existing_sessions() {
    log_step "Cleaning up existing training sessions..."

    export KUBECONFIG="$KUBECONFIG_PATH"

    # Delete all training sessions in test namespace
    EXISTING=$(kubectl get trainingsessions -n u-testuser -o name 2>/dev/null)
    if [ -n "$EXISTING" ]; then
        log_info "Found existing sessions, deleting..."
        kubectl delete trainingsessions --all -n u-testuser --wait=true --timeout=60s 2>/dev/null || true

        # Wait for pods to terminate
        log_info "Waiting for training pods to terminate..."
        for i in {1..30}; do
            PODS=$(kubectl get pods -n u-testuser -l app=basilica-training --no-headers 2>/dev/null | wc -l)
            if [ "$PODS" -eq 0 ]; then
                log_info "All training pods terminated"
                break
            fi
            sleep 2
        done
    else
        log_info "No existing sessions found"
    fi
}

run_test() {
    log_step "Running E2E test..."

    # Clean up any existing test sessions first
    cleanup_existing_sessions

    API_URL="http://localhost:8000"

    # Load API token if available
    AUTH_HEADER=""
    if [ -f "$ROOT_DIR/build/api-token.txt" ]; then
        API_TOKEN=$(cat "$ROOT_DIR/build/api-token.txt")
        AUTH_HEADER="Authorization: Bearer $API_TOKEN"
        log_info "Using API token from build/api-token.txt"
    else
        log_warn "No API token found. Run '$0 gen-key' to create one."
        log_warn "Continuing without authentication..."
    fi

    # Health check
    log_info "Checking API health..."
    if ! curl -sf "$API_URL/health" > /dev/null; then
        log_error "API not running. Start it with: $0 api"
        exit 1
    fi
    curl -s "$API_URL/health" | jq .

    # Create a training session (CPU mode with gpu_count: 0)
    log_info "Creating training session (CPU mode)..."
    if [ -n "$AUTH_HEADER" ]; then
        RESPONSE=$(curl -s -X POST "$API_URL/sessions" \
            -H "Content-Type: application/json" \
            -H "$AUTH_HEADER" \
            -d '{
                "baseModel": "facebook/opt-125m",
                "checkpointStorage": {
                    "backend": "r2",
                    "bucket": "test-bucket",
                    "path": "test/checkpoints"
                },
                "loraConfig": {
                    "rank": 8,
                    "alpha": 16
                },
                "gpuResources": {
                    "count": 0
                }
            }')
    else
        RESPONSE=$(curl -s -X POST "$API_URL/sessions" \
            -H "Content-Type: application/json" \
            -d '{
                "baseModel": "facebook/opt-125m",
                "checkpointStorage": {
                    "backend": "r2",
                    "bucket": "test-bucket",
                    "path": "test/checkpoints"
                },
                "loraConfig": {
                    "rank": 8,
                    "alpha": 16
                },
                "gpuResources": {
                    "count": 0
                }
            }')
    fi

    echo "$RESPONSE" | jq .

    SESSION_ID=$(echo "$RESPONSE" | jq -r '.sessionId // empty')

    if [ -z "$SESSION_ID" ]; then
        log_error "Failed to create session"
        echo "Response: $RESPONSE"
        return 1
    fi

    log_info "Session created: $SESSION_ID"

    # Check CRD in cluster
    export KUBECONFIG="$KUBECONFIG_PATH"
    log_info "Checking TrainingSession CRD in cluster..."
    kubectl get trainingsessions -A

    # Wait for pod to be ready
    log_info "Waiting for training pod to be ready..."
    for i in {1..60}; do
        POD_STATUS=$(kubectl get pods -A -l app=basilica-training -o jsonpath='{.items[0].status.phase}' 2>/dev/null)
        if [ "$POD_STATUS" = "Running" ]; then
            log_info "Training pod is running"
            break
        fi
        echo -n "."
        sleep 2
    done
    echo ""
    kubectl get pods -A -l app=basilica-training

    # Get the namespace for the session
    NAMESPACE="u-testuser"
    POD_NAME="training-$SESSION_ID"

    # Wait a bit more for the service to be fully ready
    log_info "Waiting for training service to initialize..."
    sleep 5

    # Run full training test
    run_training_steps "$SESSION_ID" "$NAMESPACE" "$POD_NAME"

    # Get session status
    log_info "Getting final session status..."
    if [ -n "$AUTH_HEADER" ]; then
        curl -s -H "$AUTH_HEADER" "$API_URL/sessions/$SESSION_ID" | jq .
    else
        curl -s "$API_URL/sessions/$SESSION_ID" | jq .
    fi

    log_info "E2E test completed!"
    echo ""
    echo "Next steps:"
    echo "  - Check operator logs: $0 logs operator"
    echo "  - Check training pod logs: $0 logs training"
    echo "  - Delete session: curl -X DELETE -H \"Authorization: Bearer \$(cat build/api-token.txt)\" $API_URL/sessions/$SESSION_ID"
}

run_training_steps() {
    local SESSION_ID=$1
    local NAMESPACE=$2
    local POD_NAME=$3

    log_step "Running training steps via port-forward..."

    export KUBECONFIG="$KUBECONFIG_PATH"

    # Start port-forward in background
    log_info "Starting port-forward to training pod..."
    kubectl port-forward -n "$NAMESPACE" "pod/$POD_NAME" 8001:8000 &
    PF_PID=$!
    sleep 3

    # Cleanup on exit
    trap "kill $PF_PID 2>/dev/null" EXIT

    TRAINING_URL="http://localhost:8001"

    # Check training service health
    log_info "Checking training service health..."
    if ! curl -sf "$TRAINING_URL/health" > /dev/null 2>&1; then
        log_error "Training service not accessible via port-forward"
        kill $PF_PID 2>/dev/null
        return 1
    fi
    curl -s "$TRAINING_URL/health" | jq .

    # Create a session in the training backend
    # Note: The session ID here is internal to the training service
    INTERNAL_SESSION="train-session-1"
    log_info "Creating training session in backend: $INTERNAL_SESSION"

    RESPONSE=$(curl -s -X POST "$TRAINING_URL/sessions" \
        -H "Content-Type: application/json" \
        -d "{
            \"session_id\": \"$INTERNAL_SESSION\",
            \"base_model\": \"facebook/opt-125m\",
            \"lora_config\": {
                \"rank\": 8,
                \"alpha\": 16,
                \"dropout\": 0.05
            },
            \"optimizer_config\": {
                \"learning_rate\": 0.0001,
                \"weight_decay\": 0.01
            }
        }")
    echo "$RESPONSE" | jq .

    if echo "$RESPONSE" | jq -e '.session_id' > /dev/null 2>&1; then
        log_info "Training session created successfully"
    else
        log_error "Failed to create training session"
        echo "$RESPONSE"
        kill $PF_PID 2>/dev/null
        return 1
    fi

    # Wait for model to load (first time takes a while)
    log_info "Waiting for model to load (this may take a minute on first run)..."
    sleep 10

    # Run a few training steps with sample data
    # Using pre-tokenized data for facebook/opt-125m
    # This is a simple "Hello world" style sequence
    log_info "Running training steps..."

    NUM_STEPS=3
    for step in $(seq 1 $NUM_STEPS); do
        log_info "Training step $step/$NUM_STEPS"

        # Forward-backward pass with sample tokens
        # These are valid token IDs for OPT-125m
        # Sequence: "The quick brown fox jumps over the lazy dog"
        RESPONSE=$(curl -s -X POST "$TRAINING_URL/sessions/$INTERNAL_SESSION/forward_backward" \
            -H "Content-Type: application/json" \
            -d '{
                "input_ids": [[2, 133, 2119, 6219, 23602, 13855, 81, 5, 22414, 2335]],
                "attention_mask": [[1, 1, 1, 1, 1, 1, 1, 1, 1, 1]],
                "labels": [[2, 133, 2119, 6219, 23602, 13855, 81, 5, 22414, 2335]]
            }')

        LOSS=$(echo "$RESPONSE" | jq -r '.loss // "error"')
        TOKENS=$(echo "$RESPONSE" | jq -r '.tokens_processed // 0')

        if [ "$LOSS" = "error" ]; then
            log_error "Forward-backward failed"
            echo "$RESPONSE" | jq .
            kill $PF_PID 2>/dev/null
            return 1
        fi

        log_info "  Loss: $LOSS, Tokens: $TOKENS"

        # Optimizer step
        RESPONSE=$(curl -s -X POST "$TRAINING_URL/sessions/$INTERNAL_SESSION/optim_step")
        STEP_NUM=$(echo "$RESPONSE" | jq -r '.step // "error"')

        if [ "$STEP_NUM" = "error" ]; then
            log_error "Optimizer step failed"
            echo "$RESPONSE" | jq .
            kill $PF_PID 2>/dev/null
            return 1
        fi

        log_info "  Completed step: $STEP_NUM"
    done

    # Get final status
    log_info "Getting training session status..."
    curl -s "$TRAINING_URL/sessions/$INTERNAL_SESSION" | jq .

    # Test text generation
    log_info "Testing text generation..."
    RESPONSE=$(curl -s -X POST "$TRAINING_URL/sessions/$INTERNAL_SESSION/sample" \
        -H "Content-Type: application/json" \
        -d '{
            "prompt": "The quick brown",
            "max_tokens": 20,
            "temperature": 0.7
        }')
    echo "$RESPONSE" | jq .

    log_info "Training test completed successfully!"
    log_info "Steps completed: $NUM_STEPS"

    # Cleanup port-forward
    kill $PF_PID 2>/dev/null
    trap - EXIT
}

show_status() {
    export KUBECONFIG="$KUBECONFIG_PATH"

    echo ""
    log_info "=== Cluster Status ==="
    kubectl get nodes

    echo ""
    log_info "=== Deployments ==="
    kubectl get deployments -A | grep -E "NAME|basilica|envoy"

    echo ""
    log_info "=== Pods ==="
    kubectl get pods -A | grep -E "NAME|basilica|envoy|training"

    echo ""
    log_info "=== TrainingSessions ==="
    kubectl get trainingsessions -A 2>/dev/null || echo "No TrainingSessions found"

    echo ""
    log_info "=== HTTPRoutes ==="
    kubectl get httproutes -A 2>/dev/null || echo "No HTTPRoutes found"

    echo ""
    log_info "=== Gateway ==="
    kubectl get gateway -A 2>/dev/null || echo "No Gateway found"
}

show_logs() {
    export KUBECONFIG="$KUBECONFIG_PATH"
    COMPONENT=${1:-operator}

    case $COMPONENT in
        operator)
            kubectl logs -n basilica-system -l app=basilica-operator -f --tail=100
            ;;
        training)
            kubectl logs -A -l app=basilica-training -f --tail=100
            ;;
        gateway)
            kubectl logs -n envoy-gateway-system -l app.kubernetes.io/name=envoy-gateway -f --tail=100
            ;;
        *)
            log_error "Unknown component: $COMPONENT"
            echo "Usage: $0 logs [operator|training|gateway]"
            ;;
    esac
}

cluster_down() {
    log_step "Cleaning up..."

    # Stop Postgres
    log_info "Stopping Postgres..."
    docker compose -f "$ROOT_DIR/scripts/api/compose.dev.yml" down 2>/dev/null || true

    if k3d cluster list 2>/dev/null | grep -q "$CLUSTER_NAME"; then
        k3d cluster delete "$CLUSTER_NAME"
        rm -f "$KUBECONFIG_PATH"
        log_info "Cluster deleted"
    else
        log_warn "Cluster not found"
    fi

    # Clean up generated files
    rm -f "$ROOT_DIR/build/api-token.txt"
    log_info "Cleanup complete"
}

show_help() {
    cat << EOF
Local E2E Testing for Training Service MVP

Architecture:
  - API runs locally (cargo run) with Postgres in Docker
  - Operator, Training Pods run in k3d cluster
  - Envoy Gateway handles training operation routing

Usage: $0 <command>

Commands:
  cluster-up    Create k3d cluster
  deploy        Install gateway + build images + deploy operator
  api           Run Basilica API locally (starts Postgres, in foreground)
  gen-key       Generate API key and insert into Postgres
  reset-db      Reset Postgres database (delete volume)
  test          Run full E2E test with actual training steps
  cleanup       Delete all existing training sessions
  status        Show cluster status
  logs          Show logs [operator|training|gateway]
  cluster-down  Delete the k3d cluster

Quick Start:
  1. $0 cluster-up     # Create cluster (one time)
  2. $0 deploy         # Deploy operator + gateway
  3. $0 api            # Run API (in one terminal) - starts Postgres
  4. $0 gen-key        # Generate API key (in another terminal)
  5. $0 test           # Run full training test (in another terminal)
  6. $0 cluster-down   # Clean up when done

What the test does:
  1. Creates a TrainingSession via API (creates K8s CRD)
  2. Waits for training pod to be ready
  3. Port-forwards to training pod
  4. Creates a training session in the backend
  5. Loads facebook/opt-125m model with LoRA adapter
  6. Runs 3 training steps (forward-backward + optim_step)
  7. Tests text generation with the fine-tuned model

Environment:
  KUBECONFIG will be set to: $KUBECONFIG_PATH

Database:
  Postgres: localhost:5433 (user: api, db: basilica_api)
  Started via: docker compose -f scripts/api/compose.dev.yml

Config:
  API config: config/api.local.toml (dev mode enabled)
EOF
}

# Main
case "${1:-help}" in
    cluster-up)
        cluster_up
        ;;
    deploy)
        deploy
        ;;
    api)
        run_api
        ;;
    gen-key)
        gen_key
        ;;
    reset-db)
        reset_db
        ;;
    test)
        run_test
        ;;
    cleanup)
        cleanup_existing_sessions
        ;;
    status)
        show_status
        ;;
    logs)
        show_logs "${2:-operator}"
        ;;
    cluster-down|teardown)
        cluster_down
        ;;
    help|--help|-h)
        show_help
        ;;
    *)
        log_error "Unknown command: $1"
        show_help
        exit 1
        ;;
esac
