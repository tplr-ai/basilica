# Basilica Justfile - Public repo (miner, validator, cli, sdk)
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
    cargo clippy --fix --allow-dirty --workspace --all-targets
    cargo clippy --workspace --all-targets -- -D warnings
    cargo fmt --all

# Lint workspace packages
lint: fmt-check
    cargo clippy --workspace --all-targets -- -D warnings

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
    cargo build --workspace

# Build workspace (release)
build-release:
    cargo build --release --workspace

# Test workspace
test:
    cargo test --workspace

# Check workspace
check:
    cargo check --workspace

# Test with coverage
cov:
    #!/usr/bin/env bash
    cargo install cargo-tarpaulin 2>/dev/null || true
    cargo tarpaulin --workspace --out Html --output-dir target/coverage

# Clean workspace
clean:
    cargo clean
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

# Build all public Docker images
docker-build-all TAG="latest":
    #!/usr/bin/env bash
    TAG="{{TAG}}" ./scripts/build-images.sh

# Build miner Docker image
docker-build-miner TAG="latest":
    #!/usr/bin/env bash
    set -euo pipefail
    chmod +x scripts/miner/build.sh
    CLEAN_TAG="{{TAG}}"
    if [[ "$CLEAN_TAG" == TAG=* ]]; then CLEAN_TAG="${CLEAN_TAG#TAG=}"; fi
    ./scripts/miner/build.sh --image-name ghcr.io/one-covenant/basilica/miner --image-tag "$CLEAN_TAG"

# Build validator Docker image
docker-build-validator TAG="latest":
    #!/usr/bin/env bash
    set -euo pipefail
    chmod +x scripts/validator/build.sh
    CLEAN_TAG="{{TAG}}"
    if [[ "$CLEAN_TAG" == TAG=* ]]; then CLEAN_TAG="${CLEAN_TAG#TAG=}"; fi
    BITTENSOR_NETWORK=finney ./scripts/validator/build.sh --image-name ghcr.io/one-covenant/basilica/validator --image-tag "$CLEAN_TAG"

# Build CLI Docker image
docker-build-cli TAG="latest":
    #!/usr/bin/env bash
    set -euo pipefail
    chmod +x scripts/cli/build.sh
    CLEAN_TAG="{{TAG}}"
    if [[ "$CLEAN_TAG" == TAG=* ]]; then CLEAN_TAG="${CLEAN_TAG#TAG=}"; fi
    ./scripts/cli/build.sh --image-name ghcr.io/one-covenant/basilica/cli --image-tag "$CLEAN_TAG"

# =============================================================================
# DEPLOYMENT COMMANDS
# =============================================================================

# Deploy miner to remote server
deploy-miner HOST PORT="22":
    #!/usr/bin/env bash
    chmod +x scripts/miner/deploy.sh
    ./scripts/miner/deploy.sh {{HOST}} {{PORT}}

# Deploy validator to remote server
deploy-validator HOST PORT="22":
    #!/usr/bin/env bash
    chmod +x scripts/validator/deploy.sh
    ./scripts/validator/deploy.sh {{HOST}} {{PORT}}

# Set docker compose command (use v2 by default)
docker_compose := "docker compose"

# Build docker compose services
docker-compose-build:
    docker compose -f docker/docker-compose.yml build

# Start miner with GPU support
docker-up:
    docker compose -f docker/docker-compose.yml up -d

# Stop all services
docker-down:
    docker compose -f docker/docker-compose.yml down

# =============================================================================
# CI SHORTCUTS
# =============================================================================

# Trigger CI on current branch
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
# SUBTENSOR (LOCAL DEVELOPMENT)
# =============================================================================

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

# Start local Validator against local Subtensor
local-validator-up:
    #!/usr/bin/env bash
    set -euo pipefail
    # Detect external IP
    EXTERNAL_IP=$(curl -s https://api.ipify.org || curl -s https://ifconfig.me || echo "127.0.0.1")
    echo "Detected external IP: $EXTERNAL_IP"

    # Ensure build directory exists
    mkdir -p build

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
        printf 'nodes = []\n\n'
        printf '[ssh_session]\n'
        printf 'miner_node_key_path = "/root/.ssh/id_rsa"\n'
        printf 'default_node_username = "root"\n\n'
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

# Start local dev environment (subtensor + validator + miner)
local-dev-up:
    #!/usr/bin/env bash
    set -euo pipefail
    echo "Starting local Subtensor (Alice/Bob + Envoy WSS)..."
    just local-subtensor-up
    echo "Waiting for WS/WSS to settle..."
    sleep 5
    echo "Starting local Validator (Alice/default)..."
    just local-validator-up
    echo "Waiting for Validator to initialize..."
    sleep 3
    echo "Starting local Miner (Alice/M1)..."
    just local-miner-up
    echo "Local dev is up:"
    echo "  - Subtensor: wss://localhost:9944"
    echo "  - Validator: port 8080"
    echo "  - Miner: port 8091"

local-dev-down:
    #!/usr/bin/env bash
    set -euo pipefail
    echo "Stopping local Miner..."
    just local-miner-down || true
    echo "Stopping local Validator..."
    just local-validator-down || true
    echo "Stopping local Subtensor..."
    docker compose -f scripts/subtensor-local/docker-compose.yml down -v || true
    echo "Local dev is down."

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

    # Set PYTHONHOME for pyo3 stub generator to find Python's standard library
    export PYTHONHOME=$(python3 -c "import sys; print(sys.base_prefix)")
    cargo run --bin stub_gen --features stub-gen

    echo "Python SDK installed with type stubs"
    echo "Stub file generated at: python/basilica/_basilica.pyi"
    echo "Virtual environment: .venv (root directory)"

# =============================================================================
# SHOW HELP
# =============================================================================

# Show help
default:
    @just --list
