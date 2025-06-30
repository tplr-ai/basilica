# Basilica Justfile - Service-scoped commands matching CI expectations
# Run `just --list` to see all available commands

# Install development tools
install-dev-tools:
    cargo install cargo-audit cargo-deny cargo-license

# Generate key for gpu-attestor (required for builds)
gen-key:
    #!/usr/bin/env bash
    if [ ! -f "public_key.hex" ]; then
        echo "Generating key for gpu-attestor..."
        chmod +x scripts/gen-key.sh
        ./scripts/gen-key.sh
    else
        echo "Key already exists"
    fi

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
fix: gen-key
    #!/usr/bin/env bash
    export VALIDATOR_PUBLIC_KEY=$(cat public_key.hex)
    cargo clippy --fix --allow-dirty --workspace --all-targets --exclude integration-tests -- -A clippy::too_many_arguments -A clippy::ptr_arg -A dead_code
    cargo fmt --all

# Lint workspace packages
lint: gen-key fmt-check
    #!/usr/bin/env bash
    export VALIDATOR_PUBLIC_KEY=$(cat public_key.hex)
    cargo clippy --workspace --all-targets --all-features -- -D warnings -A clippy::result_large_err -A clippy::type_complexity -A clippy::manual_clamp -A clippy::too_many_arguments -A clippy::ptr_arg -A unused_variables

# Full lint check (matches CI format-and-lint job)
lint-ci: gen-key fmt-check
    #!/usr/bin/env bash
    export VALIDATOR_PUBLIC_KEY=$(cat public_key.hex)
    cargo clippy -p common -p protocol -p executor -p gpu-attestor -p bittensor --all-targets --all-features -- -D warnings -A clippy::result_large_err -A clippy::type_complexity -A clippy::manual_clamp -A clippy::too_many_arguments -A clippy::ptr_arg -A unused_variables -A clippy::manual_async_fn

# =============================================================================
# WORKSPACE COMMANDS
# =============================================================================

# Build workspace
build: gen-key
    #!/usr/bin/env bash
    export VALIDATOR_PUBLIC_KEY=$(cat public_key.hex)
    cargo build --workspace

# Build workspace (release)
build-release: gen-key
    #!/usr/bin/env bash
    export VALIDATOR_PUBLIC_KEY=$(cat public_key.hex)
    cargo build --release --workspace

# Test workspace
test: gen-key
    #!/usr/bin/env bash
    export VALIDATOR_PUBLIC_KEY=$(cat public_key.hex)
    cargo test --workspace

# Check workspace
check: gen-key
    #!/usr/bin/env bash
    export VALIDATOR_PUBLIC_KEY=$(cat public_key.hex)
    cargo check --workspace

# Test with coverage
cov: gen-key
    #!/usr/bin/env bash
    export VALIDATOR_PUBLIC_KEY=$(cat public_key.hex)
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
docker-build-gpu-attestor: gen-key
    chmod +x scripts/gpu-attestor/build.sh
    ./scripts/gpu-attestor/build.sh

# Build all Docker images
docker-build: docker-build-executor docker-build-gpu-attestor

# =============================================================================
# DOCKER COMPOSE COMMANDS
# =============================================================================

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
dev: gen-key
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
# INTEGRATION TESTS
# =============================================================================

# Clean remote binaries and data
clean-remote:
    #!/usr/bin/env bash
    echo "Cleaning remote binaries and data..."
    echo "===================================="
    
    # Stop all services first
    echo "Stopping services..."
    ssh -i ~/.ssh/tplr root@64.247.196.98 -p 9001 "pkill -f executor || true" 2>/dev/null || true
    ssh -i ~/.ssh/tplr root@51.159.160.71 -p 55960 "pkill -f miner || true" 2>/dev/null || true
    ssh -i ~/.ssh/tplr root@51.159.130.131 -p 41199 "pkill -f validator || true" 2>/dev/null || true
    sleep 2
    
    # Clean executor
    echo "Cleaning executor at 64.247.196.98..."
    ssh -i ~/.ssh/tplr root@64.247.196.98 -p 9001 "rm -rf /opt/basilica/bin/* /opt/basilica/config/* /opt/basilica/data/* /opt/basilica/logs/*" 2>/dev/null || echo "  Warning: Could not clean executor"
    
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
        echo "Configurations exist, skipping generation"
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
        ssh -i ~/.ssh/tplr root@64.247.196.98 -p 9001 "pkill -f executor || true" 2>/dev/null || true
        ssh -i ~/.ssh/tplr root@51.159.160.71 -p 55960 "pkill -f miner || true" 2>/dev/null || true
        ssh -i ~/.ssh/tplr root@51.159.130.131 -p 41199 "pkill -f validator || true" 2>/dev/null || true
        sleep 2
        
        # Clean all remote directories
        echo "Cleaning remote directories..."
        ssh -i ~/.ssh/tplr root@64.247.196.98 -p 9001 "rm -rf /opt/basilica/{bin,config,data,logs}/*" 2>/dev/null || true
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
    if ! ssh -i ~/.ssh/tplr root@64.247.196.98 -p 9001 "test -f /opt/basilica/bin/executor" 2>/dev/null; then
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
    ssh -i ~/.ssh/tplr root@64.247.196.98 -p 9001 "pkill -f executor || true" 2>/dev/null || true
    ssh -i ~/.ssh/tplr root@51.159.160.71 -p 55960 "pkill -f miner || true" 2>/dev/null || true
    ssh -i ~/.ssh/tplr root@51.159.130.131 -p 41199 "pkill -f validator || true" 2>/dev/null || true
    
    sleep 2
    
    # Deploy testnet configurations
    echo "Deploying testnet configurations..."
    ./scripts/basilica.sh provision config testnet
    
    # Start services with testnet configs
    echo "Starting executor (testnet mode)..."
    ssh -i ~/.ssh/tplr -f root@64.247.196.98 -p 9001 'cd /opt/basilica && /opt/basilica/bin/executor --server --config /opt/basilica/config/executor.toml > /opt/basilica/logs/executor-testnet.log 2>&1 &'
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
    ssh -i ~/.ssh/tplr root@64.247.196.98 -p 9001 "ps aux | grep 'executor --server' | grep -v grep || echo 'Executor process not found'"
    
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
    echo "- Executor: ssh -i ~/.ssh/tplr root@64.247.196.98 -p 9001 'tail -f /opt/basilica/logs/executor-testnet.log'"


# =============================================================================
# SHOW HELP
# =============================================================================

# Show help
default:
    @just --list