# Basilica Justfile - Service-scoped commands matching CI expectations
# Run `just --list` to see all available commands

# Install development tools
install-dev-tools:
    cargo install cargo-audit cargo-deny cargo-license


# =============================================================================
# FORMATTING & LINTING
# =============================================================================

# Format all code
fmt:
    cargo fmt --all

# Check if code is formatted (CI style)
fmt-check:
    cargo fmt --all -- --check

# Fix linting issues and format code
fix:
    #!/usr/bin/env bash
    # First run with --fix to auto-fix what we can
    cargo clippy --fix --allow-dirty --workspace --all-targets
    # Then run without --fix to catch remaining issues (like CI does)
    cargo clippy --workspace --all-targets -- -D warnings
    cargo fmt --all

# Fix linting issues including basilica-storage (requires libfuse3-dev system package)
fix-all:
    #!/usr/bin/env bash
    # First run with --fix to auto-fix what we can
    cargo clippy --fix --allow-dirty --workspace --all-targets --all-features
    # Then run without --fix to catch remaining issues (like CI does)
    cargo clippy --workspace --all-targets --all-features -- -D warnings
    cargo fmt --all

# Lint workspace packages
lint: fmt-check
    #!/usr/bin/env bash
    cargo clippy --workspace --all-targets --all-features -- -D warnings

# Full lint check (matches CI format-and-lint job)
lint-ci: fmt-check
    #!/usr/bin/env bash
    cargo clippy -p common -p protocol -p executor -p gpu-attestor -p bittensor --all-targets --all-features -- -D warnings

# =============================================================================
# TEST COMMANDS
# =============================================================================

# Run tests
test-run *ARGS:
    #!/usr/bin/env bash
    chmod +x scripts/test/run.sh
    ./scripts/test/run.sh {{ARGS}}

# Verify test implementation
test-verify:
    #!/usr/bin/env bash
    chmod +x scripts/test/verify.sh
    ./scripts/test/verify.sh

# Show test statistics
test-stats *ARGS:
    #!/usr/bin/env bash
    chmod +x scripts/test/stats.sh
    ./scripts/test/stats.sh {{ARGS}}

# =============================================================================
# WORKSPACE COMMANDS
# =============================================================================

# Build workspace
build:
    #!/usr/bin/env bash
    cargo build --workspace

# Build workspace (release)
build-release:
    #!/usr/bin/env bash
    cargo build --release --workspace

# Test workspace
test:
    #!/usr/bin/env bash
    cargo test --workspace

# Check workspace
check:
    #!/usr/bin/env bash
    cargo check --workspace

# Test with coverage
cov:
    #!/usr/bin/env bash
    cargo install cargo-tarpaulin 2>/dev/null || true
    cargo tarpaulin --workspace --out Html --output-dir target/coverage

# Clean workspace
clean:
    cargo clean
    rm -f executor.db*
    rm -f *.log

# =============================================================================
# SECURITY & QUALITY
# =============================================================================

# Run security audit
audit:
    cargo audit

# =============================================================================
# DOCKER BUILDS
# =============================================================================

# Build executor Docker image and extract binary
docker-build-executor:
    chmod +x scripts/executor/build.sh
    ./scripts/executor/build.sh

# Build gpu-attestor Docker image and extract binary
docker-build-gpu-attestor:
    chmod +x scripts/gpu-attestor/build.sh
    ./scripts/gpu-attestor/build.sh

# Build all Docker images
docker-build: docker-build-executor docker-build-gpu-attestor

# Build operator Docker image locally
docker-build-operator TAG="latest":
    #!/usr/bin/env bash
    set -euo pipefail
    chmod +x scripts/operator/build.sh
    # Sanitize accidental "TAG=..." input
    CLEAN_TAG="{{TAG}}"
    if [[ "$CLEAN_TAG" == TAG=* ]]; then CLEAN_TAG="${CLEAN_TAG#TAG=}"; fi
    echo "Building operator image with tag: $CLEAN_TAG"
    ./scripts/operator/build.sh --image-name ghcr.io/one-covenant/basilica-operator --image-tag "$CLEAN_TAG"
    echo "✅ Operator image built: ghcr.io/one-covenant/basilica-operator:$CLEAN_TAG"
    echo ""
    echo "To push to registry, run:"
    echo "  docker push ghcr.io/one-covenant/basilica-operator:$CLEAN_TAG"

# Build and push operator Docker image
docker-push-operator TAG="latest":
    #!/usr/bin/env bash
    set -euo pipefail
    # Sanitize accidental "TAG=..." input
    CLEAN_TAG="{{TAG}}"
    if [[ "$CLEAN_TAG" == TAG=* ]]; then CLEAN_TAG="${CLEAN_TAG#TAG=}"; fi
    # Build first
    just docker-build-operator "$CLEAN_TAG"
    # Push
    echo "Pushing operator image to registry..."
    docker push ghcr.io/one-covenant/basilica-operator:$CLEAN_TAG
    echo "✅ Operator image pushed: ghcr.io/one-covenant/basilica-operator:$CLEAN_TAG"

# Build storage daemon Docker image locally
docker-build-storage TAG="latest":
    #!/usr/bin/env bash
    set -euo pipefail
    chmod +x scripts/storage-daemon/build.sh
    # Sanitize accidental "TAG=..." input
    CLEAN_TAG="{{TAG}}"
    if [[ "$CLEAN_TAG" == TAG=* ]]; then CLEAN_TAG="${CLEAN_TAG#TAG=}"; fi
    echo "Building storage daemon image with tag: $CLEAN_TAG"
    ./scripts/storage-daemon/build.sh --image-name ghcr.io/one-covenant/basilica/storage-daemon --image-tag "$CLEAN_TAG"
    echo "✅ Storage daemon image built: ghcr.io/one-covenant/basilica/storage-daemon:$CLEAN_TAG"
    echo ""
    echo "To push to registry, run:"
    echo "  just docker-push-storage $CLEAN_TAG"

# Build and push storage daemon Docker image
docker-push-storage TAG="latest":
    #!/usr/bin/env bash
    set -euo pipefail
    chmod +x scripts/storage-daemon/push.sh
    # Sanitize accidental "TAG=..." input
    CLEAN_TAG="{{TAG}}"
    if [[ "$CLEAN_TAG" == TAG=* ]]; then CLEAN_TAG="${CLEAN_TAG#TAG=}"; fi
    # Build first
    just docker-build-storage "$CLEAN_TAG"
    # Push
    echo "Pushing storage daemon image to registry..."
    ./scripts/storage-daemon/push.sh \
      --source-image ghcr.io/one-covenant/basilica/storage-daemon \
      --target-image ghcr.io/one-covenant/basilica/storage-daemon \
      --tag "$CLEAN_TAG"
    echo "✅ Storage daemon image pushed: ghcr.io/one-covenant/basilica/storage-daemon:$CLEAN_TAG"

# =============================================================================
# DEPLOYMENT COMMANDS
# =============================================================================

# Deploy miner to remote server
deploy-miner HOST PORT="22":
    #!/usr/bin/env bash
    chmod +x scripts/miner/deploy.sh
    ./scripts/miner/deploy.sh {{HOST}} {{PORT}}

# Deploy executor to remote server
deploy-executor HOST PORT="22":
    #!/usr/bin/env bash
    chmod +x scripts/executor/deploy.sh
    ./scripts/executor/deploy.sh {{HOST}} {{PORT}}

# Deploy validator to remote server
deploy-validator HOST PORT="22":
    #!/usr/bin/env bash
    chmod +x scripts/validator/deploy.sh
    ./scripts/validator/deploy.sh {{HOST}} {{PORT}}

# Deploy basilica-api to remote server
deploy-basilica-api HOST PORT="22":
    #!/usr/bin/env bash
    chmod +x scripts/basilica-api/deploy.sh
    ./scripts/basilica-api/deploy.sh {{HOST}} {{PORT}}

# Set docker compose command (use v2 by default)
docker_compose := "docker compose"

# Build docker compose services
docker-compose-build:
    docker compose -f docker/docker-compose.yml build

# Start miner and executor with GPU support
docker-up:
    docker compose -f docker/docker-compose.yml up -d

# Start development environment with monitoring (PostgreSQL)
docker-dev-postgres:
    docker compose -f docker/docker-compose.dev.yml up -d

# Start development environment with SQLite (recommended)
docker-dev:
    docker compose -f docker/docker-compose.dev-sqlite.yml up -d

# Start development with remote executors (NEW)
dev:
    #!/usr/bin/env bash
    set -e
    echo "🚀 Starting Basilica with remote executors..."
    # Build executor binary first
    cargo build --release -p executor
    # Start services (rebuild to get latest miner binary)
    cd docker && docker compose -f docker-compose.dev-remote.yml up -d --build
    # Wait for startup
    echo "⏳ Waiting for services to start..."
    sleep 5
    # Deploy executors
    echo "🚀 Deploying to remote machines..."
    docker exec basilica-miner-dev miner -c /config/miner-local.toml deploy-executors || echo "⚠️  Deployment failed - check your SSH config"
    echo "✅ Started! Use 'just dev-status' to check executor status"

# Check status of remote executors
dev-status:
    @docker exec basilica-miner-dev miner -c /config/miner-local.toml deploy-executors --status-only 2>/dev/null || echo "Miner not running"

# View logs for development environment
dev-logs:
    cd docker && docker compose -f docker-compose.dev-remote.yml logs -f

# Stop development environment with remote executors
dev-down:
    cd docker && docker compose -f docker-compose.dev-remote.yml down

# Stop all services
docker-down:
    docker compose -f docker/docker-compose.yml down

# Stop development services
docker-dev-down:
    docker compose -f docker/docker-compose.dev-sqlite.yml down

# Stop PostgreSQL development services
docker-dev-postgres-down:
    docker compose -f docker/docker-compose.dev.yml down

# =============================================================================
# CI SHORTCUTS
# =============================================================================

# Trigger the manual workflow to build and push all images with a tag,
# then print the Ansible one-liner to deploy Operator + API using that tag.
ci-build-images TAG="k3_test":
    #!/usr/bin/env bash
    if ! command -v gh >/dev/null 2>&1; then
        echo "GitHub CLI (gh) is not installed. Install from https://cli.github.com/" >&2
        exit 1
    fi
    REF=$(git rev-parse --abbrev-ref HEAD)
    echo "Triggering CI on ref $REF to build and push images with tag={{TAG}}"
    if gh workflow run ci.yml -r "$REF" -f build_images=true -f image_tag={{TAG}}; then
        echo "Dispatched workflow_dispatch successfully."
    else
        echo "workflow_dispatch not available on default branch; falling back to push-triggered build workflow..."
        # Touch the trigger file and commit with TAG=<value> to pass tag to the workflow
        mkdir -p .github/triggers
        date +%s > .github/triggers/build-k3-test-images
        git add .github/triggers/build-k3-test-images
        git commit -m "trigger build-k3-test-images TAG={{TAG}}" --allow-empty
        git push origin "$REF"
        echo "Pushed trigger commit. The build-k3-test-images workflow will run on branch $REF."
    fi
    echo
    echo "When the workflow finishes, deploy with Ansible:"
    echo "  cd orchestrator/ansible && ansible-playbook -i inventories/example.ini playbooks/e2e-apply.yml \"-e operator_image=ghcr.io/one-covenant/basilica-operator:{{TAG}}\" \"-e api_image=ghcr.io/one-covenant/basilica-api:{{TAG}}\""

# Manually run CI on the current branch (or a given ref)
ci-run REF="":
    #!/usr/bin/env bash
    if ! command -v gh >/dev/null 2>&1; then
        echo "GitHub CLI (gh) is not installed. Install from https://cli.github.com/" >&2
        exit 1
    fi
    if [[ -z "{{REF}}" ]]; then
        REF=$(git rev-parse --abbrev-ref HEAD)
    else
        REF={{REF}}
    fi
    echo "Triggering CI workflow on ref $REF"
    gh workflow run ci.yml -r "$REF"

# View logs for all services
docker-logs:
    docker compose -f docker/docker-compose.yml logs -f

# View logs for specific service
docker-logs-service service:
    docker compose -f docker/docker-compose.yml logs -f {{service}}

# Check GPU availability in executor
docker-gpu-check:
    docker compose -f docker/docker-compose.yml exec executor nvidia-smi

# Rebuild and restart services
docker-restart:
    @just docker-down
    @just docker-compose-build
    @just docker-up

# Clean up docker resources
docker-clean:
    docker compose -f docker/docker-compose.yml down -v
    docker system prune -f

# =============================================================================
# DOCUMENTATION
# =============================================================================

# Build documentation
docs:
    cargo doc --workspace --no-deps --document-private-items

# Open documentation
docs-open:
    cargo doc --workspace --no-deps --document-private-items --open

# =============================================================================
# E2E APPLY HELPERS
# =============================================================================

# Configure R2 storage credentials securely using Ansible Vault
r2-setup:
    #!/usr/bin/env bash
    set -euo pipefail
    echo "━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━"
    echo "🔐 Secure R2 Storage Credentials Setup"
    echo "━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━"
    echo ""
    echo "This will:"
    echo "  1. Prompt for vault password (min 12 chars)"
    echo "  2. Collect R2 credentials interactively"
    echo "  3. Encrypt credentials with AES-256"
    echo "  4. Save to orchestrator/ansible/group_vars/all/vault.yml"
    echo ""
    cd orchestrator/ansible
    chmod +x scripts/05-secure-r2-setup.sh
    ./scripts/05-secure-r2-setup.sh
    cd ../..
    echo ""
    echo "✅ R2 credentials configured!"
    echo ""
    echo "To deploy credentials to your cluster:"
    echo "  just e2e-apply TAG=k3_test"

# Upload R2 credentials directly to cluster (for already-configured credentials)
r2-upload:
    #!/usr/bin/env bash
    set -euo pipefail
    echo "━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━"
    echo "📤 Upload R2 Credentials to Cluster"
    echo "━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━"
    echo ""
    echo "This will upload your R2 credentials directly to the cluster."
    echo "Provide credentials via environment variables or interactively."
    echo ""
    cd orchestrator/ansible
    chmod +x scripts/06-upload-r2-credentials.sh
    ./scripts/06-upload-r2-credentials.sh
    cd ../..
    echo ""
    echo "✅ Credentials uploaded!"

# Apply the full stack on the remote K3s server with images by tag, and a test Postgres URL.
# Defaults match config/deploy/postgres.yaml (user=basilica, db=basilica, password=devpassword).
# Optional: Set SETUP_R2=true to configure R2 storage credentials first
e2e-apply TAG="k3_test" DB_USER="basilica" DB_PASS="devpassword" DB_NAME="basilica" SETUP_R2="false":
    #!/usr/bin/env bash
    set -euo pipefail

    # Check if R2 setup is requested
    if [ "{{SETUP_R2}}" = "true" ]; then
        echo "━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━"
        echo "🔐 R2 Storage Credentials Setup"
        echo "━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━"
        echo ""
        echo "Running secure R2 setup before deployment..."
        echo ""
        cd orchestrator/ansible
        chmod +x scripts/05-secure-r2-setup.sh
        ./scripts/05-secure-r2-setup.sh
        cd ../..
        echo ""
        echo "━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━"
        echo ""
    fi

    DB_HOST="basilica-postgres.basilica-system.svc.cluster.local"
    DB_URL="postgres://{{DB_USER}}:{{DB_PASS}}@${DB_HOST}:5432/{{DB_NAME}}"
    echo "Using DB_URL=${DB_URL}"
    cd orchestrator/ansible

    # Check if vault file exists and use it
    VAULT_ARGS=""
    if [ -f group_vars/all/vault.yml ]; then
        echo "✅ Found encrypted R2 credentials (group_vars/all/vault.yml)"
        if [ -f .vault_password ]; then
            echo "✅ Using vault password file (.vault_password)"
            VAULT_ARGS="--vault-password-file=.vault_password"
        else
            echo "🔑 Enter vault password to decrypt R2 credentials:"
            VAULT_ARGS="--ask-vault-pass"
        fi
    else
        echo "ℹ️  No R2 credentials found (group_vars/all/vault.yml)"
        echo "   To set up R2 storage, run: just e2e-apply SETUP_R2=true"
        echo "   Or run manually: cd orchestrator/ansible && ./scripts/05-secure-r2-setup.sh"
    fi

    ansible-playbook -i inventories/example.ini playbooks/e2e-apply.yml \
      -e install_local_subtensor_k8s=true \
      -e operator_image=ghcr.io/one-covenant/basilica-operator:{{TAG}} \
      -e api_image=ghcr.io/one-covenant/basilica-api:{{TAG}} \
      -e api_database_url="${DB_URL}" \
      $VAULT_ARGS
    echo
    echo "Look for 'Generated API token' in the output above."

# Fresh reinstall helper: deletes namespaces + CRDs locally, then runs e2e-apply.
e2e-reinstall TAG="k3_test" DB_USER="basilica" DB_PASS="devpassword" DB_NAME="basilica" TENANT_NS="u-test":
    #!/usr/bin/env bash
    set -euo pipefail
    echo "Teardown remote cluster resources via Ansible..."
    cd orchestrator/ansible
    ansible-playbook -i inventories/example.ini playbooks/e2e-teardown.yml -e tenant_namespace={{TENANT_NS}} || true
    cd - >/dev/null
    just e2e-apply TAG={{TAG}} DB_USER={{DB_USER}} DB_PASS={{DB_PASS}} DB_NAME={{DB_NAME}}

# =============================================================================
# SUBTENSOR (LOCALNET) HELPERS
# =============================================================================

# Spin up Subtensor local devnet (Alice + Bob) in Kubernetes and run init script
subtensor-up:
    #!/usr/bin/env bash
    set -euo pipefail
    cd orchestrator/ansible
    ansible-playbook -i inventories/example.ini playbooks/subtensor-up.yml

# Tear down Subtensor local devnet (delete Alice/Bob/ConfigMap/Job)
subtensor-down:
    #!/usr/bin/env bash
    set -euo pipefail
    cd orchestrator/ansible
    ansible-playbook -i inventories/example.ini playbooks/subtensor-down.yml || true

# Show Subtensor resources and recent events
subtensor-status:
    #!/usr/bin/env bash
    set -euo pipefail
    cd orchestrator/ansible
    ansible-playbook -i inventories/example.ini playbooks/subtensor-status.yml

# Tail Subtensor init job logs
subtensor-logs:
    #!/usr/bin/env bash
    set -euo pipefail
    export KUBECONFIG=../build/k3s.yaml
    POD=$(kubectl -n basilica-system get pod -l job-name=subtensor-init -o jsonpath='{.items[0].metadata.name}' 2>/dev/null || true)
    if [ -z "$POD" ]; then
        echo "No subtensor-init pod found; have you run 'just subtensor-up' or e2e-apply with install_local_subtensor_k8s=true?" >&2
        exit 1
    fi
    echo "Streaming logs from $POD... (Ctrl-C to stop)"
    kubectl -n basilica-system logs -f "$POD" || kubectl -n basilica-system logs job/subtensor-init --tail=-1 || true

# API logs and describe (runs on server)
api-status:
    #!/usr/bin/env bash
    set -euo pipefail
    cd orchestrator/ansible
    ansible-playbook -i inventories/example.ini playbooks/api-status.yml

# API logs locally using KUBECONFIG=build/k3s.yaml
api-logs:
    #!/usr/bin/env bash
    set -euo pipefail
    export KUBECONFIG=build/k3s.yaml
    kubectl -n basilica-system logs deploy/basilica-api --tail=200 -f

# Start local Subtensor (Alice/Bob) + Envoy WSS
local-subtensor-up:
    #!/usr/bin/env bash
    set -euo pipefail
    ./scripts/subtensor-local/start.sh

# Stop local Subtensor
local-subtensor-down:
    #!/usr/bin/env bash
    set -euo pipefail
    docker compose -f scripts/subtensor-local/docker-compose.yml down -v

# Setup remote K3s cluster via Ansible (run ONCE to provision the cluster)
k3s-provision TAG="k3_test":
    #!/usr/bin/env bash
    set -euo pipefail
    echo "🚀 Provisioning remote K3s cluster via Ansible..."
    cd orchestrator/ansible
    ansible-playbook -i inventories/example.ini playbooks/k3s-setup.yml
    ansible-playbook -i inventories/example.ini playbooks/e2e-apply.yml \
      -e operator_image=ghcr.io/one-covenant/basilica-operator:{{TAG}} \
      -e api_image=ghcr.io/one-covenant/basilica-api:{{TAG}}
    cd ../..
    echo "✅ Remote K3s cluster provisioned"
    echo "   Kubeconfig: build/k3s.yaml"
    echo ""
    export KUBECONFIG="$(pwd)/build/k3s.yaml"
    kubectl get nodes -o wide

# Setup k3d cluster for local development (alternative to remote K3s)
local-k3d-up:
    #!/usr/bin/env bash
    set -euo pipefail

    # Install k3d if needed
    if ! command -v k3d &> /dev/null; then
        echo "Installing k3d..."
        curl -s https://raw.githubusercontent.com/k3d-io/k3d/main/install.sh | bash
    fi

    # Create cluster if it doesn't exist
    if ! k3d cluster list | grep -q basilica-local; then
        echo "Creating k3d cluster..."
        k3d cluster create basilica-local \
            --api-port 6443 \
            --port "8000:80@loadbalancer" \
            --port "8443:443@loadbalancer" \
            --k3s-arg "--disable=traefik@server:0"
    fi

    # Export kubeconfig
    mkdir -p build
    k3d kubeconfig get basilica-local > build/k3s.yaml
    echo "✅ k3d cluster ready, kubeconfig at: build/k3s.yaml"
    echo "   export KUBECONFIG=\$(pwd)/build/k3s.yaml"

# Teardown k3d cluster
local-k3d-down:
    #!/usr/bin/env bash
    set -euo pipefail
    if k3d cluster list | grep -q basilica-local; then
        k3d cluster delete basilica-local
        rm -f build/k3s.yaml
        echo "✅ k3d cluster deleted"
    else
        echo "No k3d cluster found"
    fi

# Start local API (uses scripts/api/.env.local if present)
local-api-up:
    #!/usr/bin/env bash
    set -euo pipefail

    # Check if kubeconfig already exists (from Ansible or previous k3d setup)
    if [ -f build/k3s.yaml ]; then
        echo "✅ Using existing kubeconfig from build/k3s.yaml"
        # Check if it's from k3d or remote K3s
        if grep -q "k3d-basilica-local" build/k3s.yaml; then
            echo "   (k3d cluster detected)"
            # Ensure k3d cluster is running
            if ! k3d cluster list 2>/dev/null | grep -q basilica-local; then
                echo "⚠️  k3d cluster not running, starting it..."
                just local-k3d-up
            fi
        else
            echo "   (Remote K3s cluster detected - from Ansible setup)"
        fi
    else
        echo "⚠️  No kubeconfig found. Setting up local k3d cluster..."
        just local-k3d-up
    fi

    echo "Creating scripts/api/.env.local for local API..."
    {
        printf 'RUST_LOG=basilica_api=debug\n'
        printf 'RUST_BACKTRACE=1\n'
        printf 'BASILICA_API_BITTENSOR__NETWORK=local\n'
        printf 'BASILICA_API_BITTENSOR__NETUID=2\n'
        printf 'BASILICA_API_BITTENSOR__CHAIN_ENDPOINT=wss://host.docker.internal:9944\n'
        printf 'BASILICA_API_BITTENSOR__VALIDATOR_HOTKEY=5DJBmrfyRqe6eUUHLaWSho3Wgr5i8gDTWKxxWEmXvFhHvWTM\n'
        printf 'SSL_CERT_FILE=/etc/ssl/certs/subtensor-ca.crt\n'
    } > scripts/api/.env.local

    docker compose -f scripts/api/compose.dev.yml up -d
    echo "✅ API started with k8s backend enabled"

# Stop local API
local-api-down:
    #!/usr/bin/env bash
    set -euo pipefail
    docker compose -f scripts/api/compose.dev.yml down

# Start local Validator against local Subtensor
local-validator-up:
    #!/usr/bin/env bash
    set -euo pipefail
    # Detect external IP
    EXTERNAL_IP=$(curl -s https://api.ipify.org || curl -s https://ifconfig.me || echo "127.0.0.1")
    echo "Detected external IP: $EXTERNAL_IP"

    # Ensure build directory exists
    mkdir -p build

    # Check for K3s configuration and set environment variables
    if [ -f build/k3s.yaml ] && [ -f build/k3s-node-token.txt ]; then
        echo "✅ K3s configuration found - enabling node onboarding"
        export BASILICA_ENABLE_K3S_JOIN=true
        # Extract K3s server URL from kubeconfig
        export BASILICA_K3S_URL=$(grep -oP 'server:\s*\K[^\s]+' build/k3s.yaml | head -1)
        export BASILICA_K3S_TOKEN=$(cat build/k3s-node-token.txt)
        export BASILICA_K3S_CHANNEL=stable
        export BASILICA_TAINT_EXCLUSIVE=false
        export BASILICA_NAMESPACE=default
        echo "   K3s URL: $BASILICA_K3S_URL"
        echo "   Token length: ${#BASILICA_K3S_TOKEN} chars"
    else
        echo "ℹ️  No K3s configuration found - node onboarding disabled"
        echo "   To enable: run 'just e2e-apply' to fetch K3s token"
        export BASILICA_ENABLE_K3S_JOIN=false
        export BASILICA_K3S_URL=""
        export BASILICA_K3S_TOKEN=""
        # Create placeholder kubeconfig if it doesn't exist (for docker volume mount)
        if [ ! -f build/k3s.yaml ]; then
            echo "# Placeholder kubeconfig - run 'just e2e-apply' to fetch real config" > build/k3s.yaml
        fi
    fi

    # Always regenerate validator.local.toml with current external IP
    echo "Creating config/validator.local.toml for local validator..."
    {
        printf '[database]\n'
        printf 'url = "sqlite:/app/data/validator.db?mode=rwc"\n'
        printf 'max_connections = 10\n'
        printf 'run_migrations = true\n\n'
        printf '[server]\n'
        printf 'host = "0.0.0.0"\n'
        printf 'port = 8080\n'
        printf 'advertised_host = "%s"\n' "$EXTERNAL_IP"
        printf 'advertised_port = 8080\n'
        printf 'advertised_tls = false\n'
        printf 'max_connections = 1000\n'
        printf 'request_timeout = { secs = 30 }\n\n'
        printf '[bittensor]\n'
        printf 'wallet_name = "Alice"\n'
        printf 'hotkey_name = "default"\n'
        printf 'network = "local"\n'
        printf 'netuid = 2\n'
        printf 'chain_endpoint = "wss://host.docker.internal:9944"\n'
        printf 'weight_interval_secs = 300\n'
        printf 'axon_port = 8080\n'
        printf 'external_ip = "%s"\n\n' "$EXTERNAL_IP"
        printf '[metrics]\n'
        printf 'enabled = true\n\n'
        printf '[logging]\n'
        printf 'level = "debug"\n'
        printf 'format = "json"\n'
        printf 'output = "./validator.log"\n'
        printf '\n[emission]\n'
        printf 'burn_percentage = 0.0\n'
        printf 'burn_uid = 0\n'
        printf 'weight_set_interval_blocks = 360\n'
        printf '\n[emission.gpu_allocations]\n'
        printf 'A100 = { weight = 50.0, min_gpu_count = 1, min_gpu_vram = 1 }\n'
        printf 'H100 = { weight = 30.0, min_gpu_count = 1, min_gpu_vram = 1 }\n'
        printf 'B200 = { weight = 20.0, min_gpu_count = 1, min_gpu_vram = 1 }\n'

        # Add verification section with K8s profile publishing if K3s is configured
        if [ "${BASILICA_ENABLE_K3S_JOIN:-false}" = "true" ]; then
            printf '\n[verification]\n'
            printf 'max_concurrent_verifications = 10\n'
            printf 'verification_interval = { secs = 60 }  # Verify every 60 seconds\n'
            printf 'min_score_threshold = 0.1\n'
            printf 'challenge_timeout = { secs = 120 }\n'
            printf 'retry_attempts = 3\n'
            printf 'retry_delay = { secs = 5 }\n'
            printf 'enable_k8s_profile_publishing = true\n'
            printf 'k8s_profile_namespace = "default"\n'
            printf '\n[verification.node_groups]\n'
            printf 'strategy = "all-jobs"  # Options: round-robin, all-jobs, all-rentals\n'
            printf '# jobs_percentage = 30  # For round-robin: 30%% jobs, 70%% rentals\n'
            printf '# force_group = "jobs" # Optional: override all assignments\n'
        fi
    } > config/validator.local.toml
    # Run validator container
    docker compose -f scripts/validator/compose.local.yml up -d

local-validator-down:
    #!/usr/bin/env bash
    set -euo pipefail
    docker compose -f scripts/validator/compose.local.yml down -v

# Start local Miner against local Subtensor (uses Alice/M1 hotkey)
local-miner-up:
    #!/usr/bin/env bash
    set -euo pipefail
    # Detect external IP
    EXTERNAL_IP=$(curl -s https://api.ipify.org || curl -s https://ifconfig.me || echo "127.0.0.1")
    echo "Detected external IP: $EXTERNAL_IP"

    # Always regenerate miner.local.toml with current external IP
    echo "Creating config/miner.local.toml for local miner..."
    {
        printf '[database]\n'
        printf 'url = "sqlite:/app/data/miner.db?mode=rwc"\n'
        printf 'max_connections = 10\n'
        printf 'run_migrations = true\n\n'
        printf '[bittensor]\n'
        printf 'wallet_name = "Alice"\n'
        printf 'hotkey_name = "M1"\n'
        printf 'network = "local"\n'
        printf 'netuid = 2\n'
        printf 'chain_endpoint = "wss://host.docker.internal:9944"\n'
        printf 'weight_interval_secs = 300\n'
        printf 'axon_port = 8091\n'
        printf 'external_ip = "%s"\n' "$EXTERNAL_IP"
        printf 'skip_registration = false\n\n'
        printf '[validator_comms]\n'
        printf 'host = "0.0.0.0"\n'
        printf 'port = 8080\n\n'
        printf '[node_management]\n'
        printf 'nodes = [\n'
        printf '  { host = "69.19.137.104", port = 22, username = "shadeform" },\n'
        printf ']\n\n'
        printf '[ssh_session]\n'
        printf 'miner_node_key_path = "/root/.ssh/tplr"\n'
        printf 'default_node_username = "shadeform"\n\n'
        printf '[security]\n'
        printf 'verify_signatures = false\n\n'
        printf '[metrics]\n'
        printf 'enabled = true\n\n'
        printf '[metrics.prometheus]\n'
        printf 'host = "0.0.0.0"\n'
        printf 'port = 9090\n\n'
        printf '[validator_assignment]\n'
        printf 'strategy = "fixed_assignment"\n'
        printf 'validator_hotkey = "5DJBmrfyRqe6eUUHLaWSho3Wgr5i8gDTWKxxWEmXvFhHvWTM"\n'
    } > config/miner.local.toml
    # Run miner container
    docker compose -f scripts/miner/compose.local.yml up -d

local-miner-down:
    #!/usr/bin/env bash
    set -euo pipefail
    docker compose -f scripts/miner/compose.local.yml down -v

# Start local Subtensor + API together
local-dev-up TAG="k3_test":
    #!/usr/bin/env bash
    set -euo pipefail
    echo "🔧 Starting local Subtensor (Alice/Bob + Envoy WSS)..."
    just local-subtensor-up
    echo "⏳ Waiting a few seconds for WS/WSS to settle..."
    sleep 5
    echo "🧠 Starting local Validator (Alice/default)..."
    just local-validator-up
    echo "⏳ Waiting a few seconds for Validator to initialize..."
    sleep 5
    echo "🚀 Deploying operator to the cluster (TAG={{TAG}})..."
    just deploy-operator-api TAG={{TAG}}
    echo "🌐 Starting local API (compose.dev)..."
    just local-api-up
    echo "✅ Local dev is up: API on http://localhost:8000, WSS ws(s)://localhost:9944"

local-dev-down:
    #!/usr/bin/env bash
    set -euo pipefail
    echo "🛑 Stopping local API..."
    just local-api-down
    echo "🛑 Stopping local Subtensor (Alice/Bob + Envoy)..."
    docker compose -f scripts/subtensor-local/docker-compose.yml down -v
    echo "🛑 Stopping local Validator..."
    just local-validator-down
    echo "🧹 Removing operator from cluster..."
    just operator-down
    echo "✅ Local dev is down and operator removed."

# Optional: remove the operator from the cluster as well
operator-down:
    #!/usr/bin/env bash
    set -euo pipefail
    export KUBECONFIG=build/k3s.yaml
    echo "🧹 Deleting operator deployment/service (if present)..."
    kubectl -n basilica-system delete deploy/basilica-operator svc/basilica-operator --ignore-not-found
    echo "✅ Operator removed."

# Redeploy operator with new image (updates image and restarts)
operator-redeploy TAG="k3_test":
    #!/usr/bin/env bash
    set -euo pipefail
    # Sanitize accidental "TAG=..." input
    CLEAN_TAG="{{TAG}}"
    if [[ "$CLEAN_TAG" == TAG=* ]]; then CLEAN_TAG="${CLEAN_TAG#TAG=}"; fi
    export KUBECONFIG=build/k3s.yaml

    echo "🔄 Redeploying operator with image: ghcr.io/one-covenant/basilica-operator:$CLEAN_TAG"

    # Check if deployment exists
    if ! kubectl get deployment/basilica-operator -n basilica-system &>/dev/null; then
        echo "❌ Operator deployment not found in basilica-system namespace"
        echo "Run 'just deploy-operator-api $CLEAN_TAG' to deploy it first"
        exit 1
    fi

    # Get the container name from the deployment
    CONTAINER_NAME=$(kubectl get deployment/basilica-operator -n basilica-system -o jsonpath='{.spec.template.spec.containers[0].name}')
    echo "📦 Container name: $CONTAINER_NAME"

    # Update the image
    kubectl set image deployment/basilica-operator \
      "$CONTAINER_NAME=ghcr.io/one-covenant/basilica-operator:$CLEAN_TAG" \
      -n basilica-system

    echo "⏳ Waiting for rollout to complete..."
    kubectl rollout status deployment/basilica-operator -n basilica-system --timeout=120s
    echo ""
    echo "✅ Operator redeployed successfully"
    echo ""
    echo "📋 Operator pods:"
    kubectl get pods -n basilica-system -l app=basilica-operator
    echo ""
    echo "📝 Recent logs:"
    kubectl logs -n basilica-system -l app=basilica-operator --tail=20 --prefix=true || echo "  (No logs yet)"

# Deploy only Operator and API (templates + rollout)
deploy-operator-api TAG="k3_test":
    #!/usr/bin/env bash
    set -euo pipefail
    cd orchestrator/ansible
    ansible-playbook -i inventories/example.ini playbooks/e2e-apply.yml \
      -e operator_image=ghcr.io/one-covenant/basilica-operator:{{TAG}} \
      -e api_image=ghcr.io/one-covenant/basilica-api:{{TAG}} \
      --tags deploy_api_operator

# Resume only the Subtensor WSS setup inside e2e-apply
wss-enable:
    #!/usr/bin/env bash
    set -euo pipefail
    cd orchestrator/ansible
    ansible-playbook -i inventories/example.ini playbooks/e2e-apply.yml --tags subtensor_wss

# Resume only the public WSS (Ingress + cert-manager) setup
wss-public-enable:
    #!/usr/bin/env bash
    set -euo pipefail
    cd orchestrator/ansible
    ansible-playbook -i inventories/example.ini playbooks/e2e-apply.yml --tags subtensor_public_wss

# Bootstrap only the cluster networking (ingress-nginx + cert-manager + ClusterIssuer)
networking-bootstrap:
    #!/usr/bin/env bash
    set -euo pipefail
    cd orchestrator/ansible
    ansible-playbook -i inventories/example.ini playbooks/e2e-apply.yml --tags networking

# Run only WSS setup via subtensor-up playbook
subtensor-wss:
    #!/usr/bin/env bash
    set -euo pipefail
    cd orchestrator/ansible
    ansible-playbook -i inventories/example.ini playbooks/subtensor-up.yml --tags subtensor_wss

# Regenerate local kubeconfig (and install kubectl if missing)
k3s-kubeconfig:
    #!/usr/bin/env bash
    set -euo pipefail
    rm -f build/k3s.yaml || true
    cd orchestrator/ansible
    ansible-playbook -i inventories/example.ini playbooks/e2e-apply.yml --tags kubeconfig

# Regenerate local kubeconfig for SSH tunnel (server=127.0.0.1)
k3s-kubeconfig-local:
    #!/usr/bin/env bash
    set -euo pipefail
    rm -f build/k3s.yaml || true
    cd orchestrator/ansible
    ansible-playbook -i inventories/example.ini playbooks/e2e-apply.yml --tags kubeconfig -e kubeconfig_server=127.0.0.1

# # Open an SSH tunnel to the K3s API (local 6443 -> remote 127.0.0.1:6443)
# # Usage: just k3s-tunnel HOST KEY=~/.ssh/sam.pem USER=ubuntu
# k3s-tunnel HOST KEY=~/.ssh/sam.pem USER=ubuntu:
#     #!/usr/bin/env bash
#     set -euo pipefail
#     echo "Opening SSH tunnel on localhost:6443 to {{HOST}} (Ctrl-C to close)"
#     ssh -i {{KEY}} -N -L 6443:127.0.0.1:6443 {{USER}}@{{HOST}}

# =============================================================================
# PYTHON SDK
# =============================================================================

# Develop Python SDK (install in editable mode with auto-generated stubs)
develop-python:
    #!/usr/bin/env bash
    
    # Create venv if needed
    if [ ! -d ".venv" ]; then
        echo "Creating virtual environment..."
        uv venv
    fi
    
    # Install Python SDK in editable mode
    echo "Installing Python SDK..."
    uv pip install -e crates/basilica-sdk-python
    
    # Generate type stubs
    echo "Generating type stubs..."
    cd crates/basilica-sdk-python
    cargo run --bin stub_gen --features stub-gen
    
    echo "✓ Python SDK installed with type stubs"
    echo "✓ Stub file generated at: python/basilica/_basilica.pyi"
    echo "✓ Virtual environment: .venv (root directory)"

# =============================================================================
# INTEGRATION TESTS
# =============================================================================

# Clean remote binaries and data
clean-remote:
    #!/usr/bin/env bash
    echo "Cleaning remote binaries and data..."
    echo "===================================="

    # Stop all services first
    echo "Stopping services..."
    ssh -i ~/.ssh/tplr shadeform@185.26.8.109 -p 22 "pkill -f executor || true" 2>/dev/null || true
    ssh -i ~/.ssh/tplr root@51.159.160.71 -p 55960 "pkill -f miner || true" 2>/dev/null || true
    ssh -i ~/.ssh/tplr root@51.159.130.131 -p 41199 "pkill -f validator || true" 2>/dev/null || true
    sleep 2

    # Clean executor
    echo "Cleaning executor at 185.26.8.109..."
    ssh -i ~/.ssh/tplr shadeform@185.26.8.109 -p 22 "rm -rf /opt/basilica/bin/* /opt/basilica/config/* /opt/basilica/data/* /opt/basilica/logs/*" 2>/dev/null || echo "  Warning: Could not clean executor"

    # Clean miner
    echo "Cleaning miner at 51.159.160.71..."
    ssh -i ~/.ssh/tplr root@51.159.160.71 -p 55960 "rm -rf /opt/basilica/bin/* /opt/basilica/config/* /opt/basilica/data/* /opt/basilica/logs/*" 2>/dev/null || echo "  Warning: Could not clean miner"

    # Clean validator
    echo "Cleaning validator at 51.159.130.131..."
    ssh -i ~/.ssh/tplr root@51.159.130.131 -p 41199 "rm -rf /opt/basilica/bin/* /opt/basilica/config/* /opt/basilica/data/* /opt/basilica/logs/*" 2>/dev/null || echo "  Warning: Could not clean validator"

    echo "Remote cleanup complete!"

# Run integration tests (use 'just int clean' to wipe remote servers first)
int MODE="":
    #!/usr/bin/env bash
    set -euo pipefail

    # Check if clean mode is requested
    if [ "{{ MODE }}" = "clean" ]; then
        echo "Clean mode: Full cleanup and rebuild"
        echo "===================================="

        # Delete local binaries
        echo "Deleting local binaries..."
        rm -f validator miner executor gpu-attestor
        echo "Local binaries deleted"

        # Stop all services
        echo "Stopping all remote services..."
        ssh -i ~/.ssh/tplr root@51.159.130.131 -p 41199 "pkill -f executor || true" 2>/dev/null || true
        ssh -i ~/.ssh/tplr root@51.159.160.71 -p 55960 "pkill -f miner || true" 2>/dev/null || true
        ssh -i ~/.ssh/tplr root@51.159.183.42 -p 61993 "pkill -f validator || true" 2>/dev/null || true
        sleep 2

        # Clean all remote directories
        echo "Cleaning remote directories..."
        ssh -i ~/.ssh/tplr root@51.159.130.131 -p 41199 "rm -rf /opt/basilica/{bin,config,data,logs}/*" 2>/dev/null || true
        ssh -i ~/.ssh/tplr root@51.159.160.71 -p 55960 "rm -rf /opt/basilica/{bin,config,data,logs}/*" 2>/dev/null || true
        ssh -i ~/.ssh/tplr root@51.159.183.42 -p 61993 "rm -rf /opt/basilica/{bin,config,data,logs}/*" 2>/dev/null || true

        echo "Remote cleanup complete!"
        echo ""

        # Force rebuild
        NEED_BUILD=true
    fi

    echo "Starting smart integration tests..."
    echo "================================"

    # Check if binaries exist locally
    NEED_BUILD=false
    for binary in validator miner executor gpu-attestor; do
        if [ ! -f "$binary" ]; then
            echo "Missing binary: $binary"
            NEED_BUILD=true
        fi
    done

    if [ "$NEED_BUILD" = "true" ]; then
        echo "Building missing binaries..."
        [ ! -f validator ] && ./scripts/validator/build.sh
        [ ! -f miner ] && ./scripts/miner/build.sh
        [ ! -f executor ] && ./scripts/executor/build.sh
        [ ! -f gpu-attestor ] && ./scripts/gpu-attestor/build.sh
    else
        echo "All binaries exist, skipping build"
    fi

    # Check configurations
    if [ ! -f executor.toml ] || [ ! -f miner.toml ] || [ ! -f validator.toml ]; then
        echo "Generating configurations..."
        ./scripts/basilica.sh provision config production
    else
        echo "Configurations exist"
    fi

    # Deploy only if needed
    echo "Checking deployment status..."
    NEED_DEPLOY=false

    # Check each server
    if ! ssh -i ~/.ssh/tplr root@51.159.130.131 -p 41199 "test -f /opt/basilica/bin/executor" 2>/dev/null; then
        NEED_DEPLOY=true
    fi
    if ! ssh -i ~/.ssh/tplr root@51.159.160.71 -p 55960 "test -f /opt/basilica/bin/miner" 2>/dev/null; then
        NEED_DEPLOY=true
    fi
    if ! ssh -i ~/.ssh/tplr root@51.159.183.42 -p 61993 "test -f /opt/basilica/bin/validator" 2>/dev/null; then
        NEED_DEPLOY=true
    fi

    if [ "$NEED_DEPLOY" = "true" ]; then
        echo "Deploying binaries..."
        ./scripts/basilica.sh deploy binaries production
    else
        echo "Binaries already deployed, skipping"
    fi

    # Check if services are running
    echo "Checking service status..."

    # Stop services if needed
    echo "Stopping any running services..."
    ssh -i ~/.ssh/tplr root@51.159.130.131 -p 41199 "pkill -f executor || true" 2>/dev/null || true
    ssh -i ~/.ssh/tplr root@51.159.160.71 -p 55960 "pkill -f miner || true" 2>/dev/null || true
    ssh -i ~/.ssh/tplr root@51.159.183.42 -p 61993 "pkill -f validator || true" 2>/dev/null || true

    sleep 2

    # Note about executor environment
    echo "Note: Executor at 51.159.130.131 is running in a Docker container"
    echo "      Port 50051 may not be accessible externally without port mapping"
    echo "      Hardware attestation requires --privileged flag on the container"

    # Start services
    echo "Starting executor..."
    ssh -i ~/.ssh/tplr -f root@51.159.130.131 -p 41199 'cd /opt/basilica && /opt/basilica/bin/executor --server --config /opt/basilica/config/executor.toml > /opt/basilica/logs/executor.log 2>&1 &'
    echo "Executor started"

    echo "Starting miner..."
    ssh -i ~/.ssh/tplr -f root@51.159.160.71 -p 55960 'cd /opt/basilica && /opt/basilica/bin/miner --config /opt/basilica/config/miner.toml > /opt/basilica/logs/miner.log 2>&1 &'
    echo "Miner started"

    echo "Starting validator..."
    ssh -i ~/.ssh/tplr -f root@51.159.183.42 -p 61993 'cd /opt/basilica && /opt/basilica/bin/validator start --config /opt/basilica/config/validator.toml > /opt/basilica/logs/validator.log 2>&1 &'
    echo "Validator started"

    # Give services time to start
    echo "Waiting for services to start..."
    sleep 5

    # Check status
    echo "Checking miner status..."
    ssh -i ~/.ssh/tplr root@51.159.160.71 -p 55960 "ps aux | grep 'miner --config' | grep -v grep || echo 'Miner process not found'"

    echo "Checking validator status..."
    ssh -i ~/.ssh/tplr root@51.159.183.42 -p 61993 "ps aux | grep 'validator start' | grep -v grep || echo 'Validator process not found'"

    echo "Checking executor status..."
    ssh -i ~/.ssh/tplr root@51.159.130.131 -p 41199 "ps aux | grep 'executor --server' | grep -v grep || echo 'Executor process not found'"

    # Check logs
    echo "Checking miner logs..."
    ssh -i ~/.ssh/tplr root@51.159.160.71 -p 55960 "tail -20 /opt/basilica/logs/miner.log 2>/dev/null || echo 'No miner logs found'"

    echo "Checking validator logs..."
    ssh -i ~/.ssh/tplr root@51.159.183.42 -p 61993 "tail -20 /opt/basilica/logs/validator.log 2>/dev/null || echo 'No validator logs found'"

    echo "Checking executor logs..."
    ssh -i ~/.ssh/tplr root@51.159.130.131 -p 41199 "tail -20 /opt/basilica/logs/executor.log 2>/dev/null || echo 'No executor logs found'"

    # Health checks
    echo ""
    echo "Running health checks..."

    echo "Testing Executor (gRPC port 50051):"
    # Executor only has gRPC, not HTTP health
    timeout 2 bash -c "</dev/tcp/51.159.130.131/50051" 2>/dev/null && echo " ✓ Executor gRPC port 50051 is accessible" || echo " ✗ Executor gRPC port 50051 is NOT accessible"

    echo ""
    echo "Testing Miner health (HTTP port 8080):"
    curl -s --max-time 5 http://51.159.160.71:8080/health && echo " ✓ Miner health check passed" || echo " ✗ Miner health check failed"

    echo ""
    echo "Testing Validator health (HTTP port 8081):"
    curl -s --max-time 5 http://51.159.183.42:8081/health && echo " ✓ Validator health check passed" || echo " ✗ Validator health check failed"

    # Test miner-executor connectivity
    echo ""
    echo "Testing Miner -> Executor connectivity:"
    # First check if executor is listening
    echo " Checking if executor is listening on port 50051..."
    ssh -i ~/.ssh/tplr root@51.159.130.131 -p 41199 "grep -q 'Starting gRPC server on 0.0.0.0:50051' /opt/basilica/logs/executor.log && echo '  ✓ Executor says it is listening on port 50051' || echo '  ✗ Executor not listening'"

    # Test connectivity from miner
    if ssh -i ~/.ssh/tplr root@51.159.160.71 -p 55960 "timeout 2 bash -c '</dev/tcp/51.159.130.131/50051' 2>/dev/null"; then
        echo " ✓ Miner can reach Executor on port 50051"
    else
        echo " ✗ Miner CANNOT reach Executor on port 50051"
        echo " Note: The executor container may need port 50051 exposed/mapped"
        echo " This is expected if executor is in a container without port mapping"
        # Check local connectivity from within executor machine
        echo " Checking internal connectivity..."
        if ssh -i ~/.ssh/tplr root@51.159.130.131 -p 41199 "timeout 2 bash -c '</dev/tcp/localhost/50051' 2>/dev/null"; then
            echo " ✓ Executor is accessible locally on port 50051"
            echo " ℹ️  External access requires container port mapping (e.g., -p 50051:50051)"
        else
            echo " ✗ Executor not accessible even locally"
        fi
    fi

    # GPU attestation test
    echo ""
    echo "Testing GPU attestation:"

    # Check container status
    echo "Checking executor environment..."
    if ssh -i ~/.ssh/tplr root@51.159.130.131 -p 41199 "[ -f /.dockerenv ] || grep -q docker /proc/1/cgroup 2>/dev/null"; then
        echo " ℹ️  Executor is running in a Docker container"

        # Check if we have privileged access
        if ssh -i ~/.ssh/tplr root@51.159.130.131 -p 41199 "[ -r /dev/mem ] && dmidecode -t system >/dev/null 2>&1"; then
            echo " ✓ Container has privileged access"
            # Run full attestation
            echo "Running GPU attestation with full hardware collection..."
            ssh -i ~/.ssh/tplr root@51.159.130.131 -p 41199 "/opt/basilica/bin/gpu-attestor --executor-id prod-executor-1 --output /opt/basilica/data/attestations/test" && echo " ✓ GPU attestation completed successfully" || echo " ✗ GPU attestation failed"
        else
            echo " ⚠️  Container lacks privileged access - cannot read hardware information"
            echo " Note: The container at 51.159.130.131 needs to be restarted with --privileged flag"
            echo " For now, running attestation with limited hardware collection..."
            ssh -i ~/.ssh/tplr root@51.159.130.131 -p 41199 "/opt/basilica/bin/gpu-attestor --executor-id prod-executor-1 --output /opt/basilica/data/attestations/test --skip-hardware-collection --skip-os-attestation" && echo " ✓ GPU attestation completed (limited mode)" || echo " ✗ GPU attestation failed"
        fi
    else
        echo " ✓ Executor is running on bare metal"
        # Run full attestation
        echo "Running GPU attestation with full hardware collection..."
        ssh -i ~/.ssh/tplr root@51.159.130.131 -p 41199 "/opt/basilica/bin/gpu-attestor --executor-id prod-executor-1 --output /opt/basilica/data/attestations/test" && echo " ✓ GPU attestation completed successfully" || echo " ✗ GPU attestation failed"
    fi

    # Docker container test
    echo ""
    echo "Testing Docker container creation:"
    ssh -i ~/.ssh/tplr root@51.159.130.131 -p 41199 "docker rm -f cpu-rental-test 2>/dev/null || true"
    ssh -i ~/.ssh/tplr root@51.159.130.131 -p 41199 "docker run -d --name cpu-rental-test --restart unless-stopped ubuntu:22.04 bash -c 'echo Container is running && sleep infinity'" || echo "Docker container creation failed"
    ssh -i ~/.ssh/tplr root@51.159.130.131 -p 41199 "docker ps | grep rental" || echo "Docker container not running"

    echo "================================"
    echo "Integration tests completed!"
    echo ""
    echo "Summary:"
    echo "- Binaries: Built and deployed"
    echo "- Services: Started on all servers"
    echo "- Health checks: HTTP endpoints available for miner/validator"
    echo "- Known limitations:"
    echo "  * Executor gRPC port may not be externally accessible (container port mapping needed)"
    echo "  * GPU attestation limited without container --privileged flag"
    echo "  * To fix: Restart executor container from host with: docker run -d --privileged -p 50051:50051 ..."


# Run integration tests on testnet (subnet 387) - use 'just int-testnet clean' to wipe remote servers first
int-testnet MODE="":
    #!/usr/bin/env bash
    set -euo pipefail

    # Check if clean mode is requested
    if [ "{{ MODE }}" = "clean" ]; then
        echo "Clean mode: Full cleanup and rebuild"
        echo "===================================="

        # Delete local binaries
        echo "Deleting local binaries..."
        rm -f validator miner executor gpu-attestor
        echo "Local binaries deleted"

        # Stop all services
        echo "Stopping all remote services..."
        ssh -i ~/.ssh/tplr shadeform@185.26.8.109 -p 22 "pkill -f executor || true" 2>/dev/null || true
        ssh -i ~/.ssh/tplr root@51.159.160.71 -p 55960 "pkill -f miner || true" 2>/dev/null || true
        ssh -i ~/.ssh/tplr root@51.159.130.131 -p 41199 "pkill -f validator || true" 2>/dev/null || true
        sleep 2

        # Clean all remote directories
        echo "Cleaning remote directories..."
        ssh -i ~/.ssh/tplr shadeform@185.26.8.109 -p 22 "rm -rf /opt/basilica/{bin,config,data,logs}/*" 2>/dev/null || true
        ssh -i ~/.ssh/tplr root@51.159.160.71 -p 55960 "rm -rf /opt/basilica/{bin,config,data,logs}/*" 2>/dev/null || true
        ssh -i ~/.ssh/tplr root@51.159.130.131 -p 41199 "rm -rf /opt/basilica/{bin,config,data,logs}/*" 2>/dev/null || true

        echo "Remote cleanup complete!"
        echo ""

        # Force rebuild with testnet metadata
        echo "Forcing rebuild with testnet metadata..."
        export BITTENSOR_NETWORK=test
        NEED_BUILD=true
    fi

    echo "Starting Basilica testnet integration tests..."
    echo "============================================"
    echo "Target: Bittensor Testnet Subnet 387"
    echo "Chain: wss://test.finney.opentensor.ai:443"
    echo ""

    # Check if binaries exist locally
    NEED_BUILD=false
    for binary in validator miner executor gpu-attestor; do
        if [ ! -f "$binary" ]; then
            echo "Missing binary: $binary"
            NEED_BUILD=true
        fi
    done

    if [ "$NEED_BUILD" = "true" ]; then
        echo "Building missing binaries for TESTNET..."
        echo "Setting BITTENSOR_NETWORK=test for metadata generation"
        export BITTENSOR_NETWORK=test
        export METADATA_CHAIN_ENDPOINT="wss://test.finney.opentensor.ai:443"

        [ ! -f validator ] && BITTENSOR_NETWORK=test ./scripts/validator/build.sh
        [ ! -f miner ] && BITTENSOR_NETWORK=test ./scripts/miner/build.sh
        [ ! -f executor ] && BITTENSOR_NETWORK=test ./scripts/executor/build.sh
        [ ! -f gpu-attestor ] && BITTENSOR_NETWORK=test ./scripts/gpu-attestor/build.sh
    else
        echo "All binaries exist, skipping build"
    fi

    # Generate testnet configurations
    echo "Generating testnet configurations..."
    ./scripts/basilica.sh provision config testnet

    # Check deployment status
    echo "Checking deployment status..."
    NEED_DEPLOY=false

    # Check each server
    if ! ssh -i ~/.ssh/tplr root@185.26.8.109 -p 9001 "test -f /opt/basilica/bin/executor" 2>/dev/null; then
        NEED_DEPLOY=true
    fi
    if ! ssh -i ~/.ssh/tplr root@51.159.160.71 -p 55960 "test -f /opt/basilica/bin/miner" 2>/dev/null; then
        NEED_DEPLOY=true
    fi
    if ! ssh -i ~/.ssh/tplr root@51.159.130.131 -p 41199 "test -f /opt/basilica/bin/validator" 2>/dev/null; then
        NEED_DEPLOY=true
    fi

    if [ "$NEED_DEPLOY" = "true" ]; then
        echo "Deploying binaries to testnet servers..."
        ./scripts/basilica.sh deploy binaries testnet
    else
        echo "Binaries already deployed, skipping"
    fi

    # Stop any running services
    echo "Stopping any running services..."
    ssh -i ~/.ssh/tplr shadeform@185.26.8.109 -p 22 "pkill -f executor || true" 2>/dev/null || true
    ssh -i ~/.ssh/tplr root@51.159.160.71 -p 55960 "pkill -f miner || true" 2>/dev/null || true
    ssh -i ~/.ssh/tplr root@51.159.130.131 -p 41199 "pkill -f validator || true" 2>/dev/null || true

    sleep 2

    # Start services with testnet configs
    echo "Starting executor (testnet mode)..."
    ssh -i ~/.ssh/tplr -f shadeform@185.26.8.109 -p 22 'cd /opt/basilica && /opt/basilica/bin/executor --server --config /opt/basilica/config/executor.toml > /opt/basilica/logs/executor-testnet.log 2>&1 &'
    echo "Executor started"

    echo "Starting miner (testnet mode)..."
    ssh -i ~/.ssh/tplr -f root@51.159.160.71 -p 55960 'cd /opt/basilica && /opt/basilica/bin/miner --config /opt/basilica/config/miner.toml > /opt/basilica/logs/miner-testnet.log 2>&1 &'
    echo "Miner started"

    echo "Starting validator (testnet mode)..."
    ssh -i ~/.ssh/tplr -f root@51.159.130.131 -p 41199 'cd /opt/basilica && /opt/basilica/bin/validator start --config /opt/basilica/config/validator.toml > /opt/basilica/logs/validator-testnet.log 2>&1 &'
    echo "Validator started"

    # Give services time to start and register
    echo "Waiting for services to start and register on testnet..."
    sleep 10

    # Check status
    echo ""
    echo "Checking testnet service status..."
    echo "================================"

    echo "Miner process:"
    ssh -i ~/.ssh/tplr root@51.159.160.71 -p 55960 "ps aux | grep 'miner --config' | grep -v grep || echo 'Miner process not found'"

    echo ""
    echo "Validator process:"
    ssh -i ~/.ssh/tplr root@51.159.130.131 -p 41199 "ps aux | grep 'validator start' | grep -v grep || echo 'Validator process not found'"

    echo ""
    echo "Executor process:"
    ssh -i ~/.ssh/tplr shadeform@185.26.8.109 -p 22 "ps aux | grep 'executor --server' | grep -v grep || echo 'Executor process not found'"

    # Check testnet registration
    echo ""
    echo "Checking Bittensor testnet registration..."
    echo "========================================="

    echo "Miner logs (checking for testnet registration):"
    ssh -i ~/.ssh/tplr root@51.159.160.71 -p 55960 "grep -E '(test|387|finney)' /opt/basilica/logs/miner-testnet.log 2>/dev/null | tail -10 || echo 'No testnet logs found yet'"

    echo ""
    echo "Validator logs (checking for testnet registration):"
    ssh -i ~/.ssh/tplr root@51.159.130.131 -p 41199 "grep -E '(test|387|finney)' /opt/basilica/logs/validator-testnet.log 2>/dev/null | tail -10 || echo 'No testnet logs found yet'"

    # Health checks
    echo ""
    echo "Running testnet health checks..."
    echo "================================"

    echo "Testing Miner health (HTTP port 8080):"
    curl -s --max-time 5 http://51.159.160.71:8080/health && echo " ✓ Miner health check passed" || echo " ✗ Miner health check failed"

    echo ""
    echo "Testing Validator health (HTTP port 8081):"
    curl -s --max-time 5 http://51.159.130.131:8081/health && echo " ✓ Validator health check passed" || echo " ✗ Validator health check failed"

    echo ""
    echo "============================================"
    echo "Testnet integration test setup completed!"
    echo ""
    echo "Configuration:"
    echo "- Network: Bittensor Testnet"
    echo "- Subnet ID: 387"
    echo "- Chain Endpoint: wss://test.finney.opentensor.ai:443"
    echo "- Wallets:"
    echo "  * Validator: ~/.bittensor/wallets/validator/ (hotkey: validator)"
    echo "  * Miner: ~/.bittensor/wallets/test_miner/ (hotkey: default)"
    echo ""
    echo "Note: Using existing wallets. Make sure they are:"
    echo "1. Already created at the paths above (DO NOT create new ones)"
    echo "2. Funded with testnet TAO"
    echo "3. Registered on subnet 387"
    echo ""
    echo "To check registration status:"
    echo "btcli subnet metagraph --netuid 387 --subtensor.network test"
    echo ""
    echo "To view logs:"
    echo "- Miner: ssh -i ~/.ssh/tplr root@51.159.160.71 -p 55960 'tail -f /opt/basilica/logs/miner-testnet.log'"
    echo "- Validator: ssh -i ~/.ssh/tplr root@51.159.130.131 -p 41199 'tail -f /opt/basilica/logs/validator-testnet.log'"
    echo "- Executor: ssh -i ~/.ssh/tplr shadeform@185.26.8.109 -p 22 'tail -f /opt/basilica/logs/executor-testnet.log'"


# =============================================================================
# LOCALNET COMMANDS
# =============================================================================

# Start local Subtensor network with all Basilica services
localnet:
    #!/usr/bin/env bash
    cd scripts/localnet && ./setup.sh

# Restart localnet services (rebuilds containers)
localnet-restart:
    #!/usr/bin/env bash
    cd scripts/localnet && ./restart.sh

# =============================================================================
# E2E LOCAL DEVELOPMENT ENVIRONMENT
# =============================================================================

# Setup complete E2E environment (local subtensor + remote K3s + operator + local validator + local API)
# Prerequisites: Run 'just ci-build-images' first to build and push images with k3_test tag
# R2 storage: Automatically deployed if vault credentials exist at orchestrator/ansible/group_vars/all/vault.yml
e2e-up TAG="k3_test":
    #!/usr/bin/env bash
    set -euo pipefail
    echo "🚀 Setting up complete E2E environment..."
    echo "Using image tag: {{TAG}}"
    echo ""

    # 1. Start local Subtensor
    echo "📡 Starting local Subtensor..."
    just local-subtensor-up

    # 1.5 Verify subnet has neurons registered (sometimes needs re-run)
    echo "🔍 Verifying subnet initialization..."
    NEURON_COUNT=$(python3 scripts/subtensor-local/check_neurons.py 2>/dev/null || echo "0")

    if [ "$NEURON_COUNT" -lt 3 ]; then
        echo "⚠️  Subnet has only $NEURON_COUNT neurons, re-running initialization..."
        cd scripts/subtensor-local
        CHAIN_ENDPOINT="wss://localhost:9944" NETUID=2 WALLET_PATH="$HOME/.bittensor/wallets" python3 init.py
        cd ../..
    else
        echo "✅ Subnet has $NEURON_COUNT neurons registered"
    fi

    # 2. Provision remote K3s cluster and deploy operator via Ansible
    echo "☸️  Provisioning remote K3s cluster and deploying operator..."
    just k3s-provision {{TAG}}

    # 3. Start local validator
    echo "🔍 Starting local validator (Alice/default)..."
    just local-validator-up

    # 3.5. Start local miner
    echo "⛏️  Starting local miner (Alice/M1)..."
    just local-miner-up

    # 4. Start local API (connects to remote K3s and local Subtensor)
    echo "🌐 Starting local API..."
    just local-api-up

    echo "✅ E2E environment is ready!"
    echo ""
    echo "📊 Environment status:"
    echo "  - Local Subtensor: wss://localhost:9944"
    echo "  - Remote K3s cluster: see build/k3s.yaml"
    echo "  - Operator image: ghcr.io/one-covenant/basilica-operator:{{TAG}}"
    echo "  - API image: ghcr.io/one-covenant/basilica-api:{{TAG}}"
    echo "  - Local Validator (Alice/default): running on port 8080"
    echo "  - Local Miner (Alice/M1): running on port 8081"
    echo "  - Local API: http://localhost:8000"
    echo ""
    echo "Run 'just e2e-status' to check all components"

# Tear down E2E environment (remote K3s deployments + local services)
e2e-down:
    #!/usr/bin/env bash
    set -euo pipefail
    echo "🔄 Tearing down E2E environment..."

    # 1. Teardown remote K3s deployments via Ansible
    echo "🧹 Cleaning up remote K3s deployments..."
    cd orchestrator/ansible
    ansible-playbook -i inventories/example.ini playbooks/e2e-teardown.yml || true
    cd ../..

    # 2. Stop local API
    echo "🛑 Stopping local API..."
    just local-api-down || true

    # 3. Stop local miner
    echo "🛑 Stopping local miner..."
    just local-miner-down || true

    # 4. Stop local validator
    echo "🛑 Stopping local validator..."
    just local-validator-down || true

    # 5. Stop local Subtensor
    echo "🛑 Stopping local Subtensor..."
    cd scripts/subtensor-local && docker compose down -v || true
    cd ../..

    echo "✅ E2E environment torn down"
    echo ""
    echo "Note: Remote K3s cluster is still running (use 'just k3s-teardown' to remove it completely)"

# Show which K3s cluster is configured
k3s-status:
    #!/usr/bin/env bash
    set -euo pipefail

    if [ ! -f build/k3s.yaml ]; then
        echo "❌ No kubeconfig found at build/k3s.yaml"
        echo ""
        echo "To set up a cluster, run ONE of:"
        echo "  just k3s-provision      # Remote K3s via Ansible"
        echo "  just local-k3d-up       # Local k3d cluster"
        exit 1
    fi

    export KUBECONFIG=build/k3s.yaml
    SERVER=$(kubectl config view -o jsonpath='{.clusters[0].cluster.server}')

    echo "━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━"
    echo "☸️  K3s Cluster Status"
    echo "━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━"
    echo ""
    echo "Server: $SERVER"

    if echo "$SERVER" | grep -q "k3d"; then
        echo "Type: k3d (local)"
    else
        echo "Type: Remote K3s (Ansible-managed)"
    fi
    echo ""
    echo "Nodes:"
    kubectl get nodes -o wide || echo "  ❌ Cannot reach cluster"
    echo ""
    echo "Basilica Pods:"
    kubectl get pods -n basilica-system 2>/dev/null || echo "  No basilica-system namespace found"

# Show status of E2E environment
e2e-status:
    #!/usr/bin/env bash
    set -euo pipefail

    echo "━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━"
    echo "📊 E2E Environment Status"
    echo "━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━"
    echo ""

    # Local Subtensor
    echo "📡 Local Subtensor:"
    if docker ps --filter "name=subtensor" --format "{{{{.Names}}}}" | grep -q subtensor; then
        docker ps --filter "name=subtensor" --format "table {{{{.Names}}}}\t{{{{.Status}}}}"
    else
        echo "  ❌ Not running"
    fi
    echo ""

    # Remote K3s cluster
    echo "☸️  Remote K3s Cluster:"
    if [ -f build/k3s.yaml ]; then
        export KUBECONFIG=build/k3s.yaml
        SERVER=$(kubectl config view -o jsonpath='{.clusters[0].cluster.server}' 2>/dev/null || echo "")
        if [ -n "$SERVER" ]; then
            echo "  ✅ Configured: $SERVER"
            kubectl get nodes -o wide 2>/dev/null || echo "  ⚠️  Cannot reach cluster"
        else
            echo "  ❌ Invalid kubeconfig"
        fi
    else
        echo "  ❌ No kubeconfig (run 'just k3s-provision')"
    fi
    echo ""

    # K8s deployments
    if command -v kubectl &> /dev/null && [ -f build/k3s.yaml ]; then
        export KUBECONFIG=build/k3s.yaml
        echo "🎯 Remote Basilica Services:"
        kubectl get pods -n basilica-system 2>/dev/null || echo "  ❌ No pods found"
        echo ""
    fi

    # Local Validator
    echo "🔍 Local Validator (Alice/default):"
    if docker ps --filter "name=basilica-validator" --format "{{{{.Names}}}}" | grep -q validator; then
        docker ps --filter "name=basilica-validator" --format "table {{{{.Names}}}}\t{{{{.Status}}}}"
        echo "  Port: 8080"
    else
        echo "  ❌ Not running"
    fi
    echo ""

    # Local Miner
    echo "⛏️  Local Miner (Alice/M1):"
    if docker ps --filter "name=basilica-miner" --format "{{{{.Names}}}}" | grep -q miner; then
        docker ps --filter "name=basilica-miner" --format "table {{{{.Names}}}}\t{{{{.Status}}}}"
        echo "  Port: 8081"
    else
        echo "  ❌ Not running"
    fi
    echo ""

    # Local API
    echo "🌐 Local API:"
    if docker ps --filter "name=basilica-api" --format "{{{{.Names}}}}" | grep -q api; then
        docker ps --filter "name=basilica-api" --format "table {{{{.Names}}}}\t{{{{.Status}}}}"
        echo "  URL: http://localhost:8000"
    else
        echo "  ❌ Not running"
    fi

# Run complete E2E validation test suite (RBAC + API key + smoke tests)
e2e-validate:
    #!/usr/bin/env bash
    set -euo pipefail
    chmod +x orchestrator/scripts/e2e/run-validation.sh
    ./orchestrator/scripts/e2e/run-validation.sh

# =============================================================================
# SHOW HELP
# =============================================================================

# Show help
default:
    @just --list
