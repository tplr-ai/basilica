#!/bin/bash
# Basilica Localnet - Start Services
# Usage: ./start.sh [profile] [--build]
#
# Profiles:
#   network     - Subtensor only
#   validator   - Subtensor + Validator
#   miner       - Above + Miner
#   all         - Everything (default)

set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
cd "${SCRIPT_DIR}"

show_help() {
    echo "Basilica Localnet - Start Services"
    echo ""
    echo "Usage: ./start.sh [profile] [--build] [-h|--help]"
    echo ""
    echo "Starts localnet services using Docker Compose profiles."
    echo "Automatically initializes the subnet (wallets, funding, registration)"
    echo "after Subtensor is ready."
    echo ""
    echo "Profiles:"
    echo "  network     Subtensor only"
    echo "  validator   Subtensor + Validator"
    echo "  miner       Above + Miner"
    echo "  all         Everything (default)"
    echo ""
    echo "Options:"
    echo "  --build      Rebuild Docker images before starting"
    echo "  -h, --help   Show this help"
    echo ""
    echo "Examples:"
    echo "  ./start.sh                  # Start all services"
    echo "  ./start.sh network          # Start Subtensor only"
    echo "  ./start.sh miner --build    # Rebuild and start up to miner"
    echo ""
    echo "Endpoints (when fully started):"
    echo "  Subtensor:   ws://localhost:9944"
    echo "  Validator:   http://localhost:8080 (API), :9090/metrics"
    echo "  Miner:       localhost:8092 (gRPC), :9091/metrics"
    echo ""
    echo "See also: ./stop.sh, ./restart.sh, ./test.sh"
}

# Parse arguments
PROFILE="${1:-all}"
BUILD_FLAG=""

for arg in "$@"; do
    case $arg in
        -h|--help)
            show_help
            exit 0
            ;;
        --build)
            BUILD_FLAG="--build"
            shift
            ;;
        -*)
            echo "Unknown option: $arg"
            echo "Run './start.sh --help' for usage"
            exit 1
            ;;
    esac
done

# Normalize profile name
case "${PROFILE}" in
    network|subtensor)
        PROFILE="network"
        ;;
    validator|val)
        PROFILE="validator"
        ;;
    miner|min|all|"")
        PROFILE="miner"
        ;;
    *)
        echo "Unknown profile: ${PROFILE}"
        echo ""
        echo "Available profiles:"
        echo "  network     - Subtensor only"
        echo "  validator   - Subtensor + Validator"
        echo "  miner       - Above + Miner"
        echo "  all         - Everything (default)"
        exit 1
        ;;
esac

echo "========================================"
echo "  Starting Basilica Localnet"
echo "  Profile: ${PROFILE}"
echo "========================================"
echo ""

# Check prerequisites
if ! command -v curl &> /dev/null; then
    echo "ERROR: curl is not installed"
    echo "Install: brew install curl (macOS) or apt install curl (Linux)"
    exit 1
fi

if ! command -v nc &> /dev/null; then
    echo "ERROR: netcat (nc) is not installed"
    echo "Install: brew install netcat (macOS) or apt install netcat-openbsd (Linux)"
    exit 1
fi

# Generate SSH keys if they don't exist
SSH_KEY_DIR="${SCRIPT_DIR}/ssh-keys"
mkdir -p "${SSH_KEY_DIR}"
MINER_KEY="${SSH_KEY_DIR}/miner_node_key"
if [ ! -f "${MINER_KEY}" ]; then
    echo "Generating miner SSH key..."
    ssh-keygen -t ed25519 -f "${MINER_KEY}" -N "" -C "basilica-miner-localnet"
    chmod 600 "${MINER_KEY}"
    chmod 644 "${MINER_KEY}.pub"
    echo ""
fi

# Copy example configs if local configs don't exist
CONFIG_DIR="$SCRIPT_DIR/configs"
NEW_CONFIGS=()
if [ ! -f "$CONFIG_DIR/validator.toml" ]; then
    echo "Creating validator.toml from example..."
    cp "$CONFIG_DIR/validator.example.toml" "$CONFIG_DIR/validator.toml"
    NEW_CONFIGS+=("$CONFIG_DIR/validator.toml")
fi
if [ ! -f "$CONFIG_DIR/miner.toml" ]; then
    echo "Creating miner.toml from example..."
    cp "$CONFIG_DIR/miner.example.toml" "$CONFIG_DIR/miner.toml"
    NEW_CONFIGS+=("$CONFIG_DIR/miner.toml")
fi

if [ ${#NEW_CONFIGS[@]} -gt 0 ]; then
    echo ""
    echo "New config files created:"
    for f in "${NEW_CONFIGS[@]}"; do
        echo "  - $(basename "$f")"
    done
    echo ""
    echo "Please review and edit them if needed, then press Enter to continue."
    read -r
fi

# Function to wait for a service
wait_for_service() {
    local name=$1
    local url=$2
    local max_attempts=${3:-30}
    local attempt=1

    echo -n "  Waiting for ${name}..."
    while [ $attempt -le $max_attempts ]; do
        if curl -sf "${url}" > /dev/null 2>&1; then
            echo " Ready!"
            return 0
        fi
        echo -n "."
        sleep 2
        ((attempt++))
    done
    echo " Timeout!"
    return 1
}

# Function to wait for a TCP port (for gRPC services without HTTP endpoints)
wait_for_port() {
    local name=$1
    local host=$2
    local port=$3
    local max_attempts=${4:-30}
    local attempt=1

    echo -n "  Waiting for ${name}..."
    while [ $attempt -le $max_attempts ]; do
        if nc -z "${host}" "${port}" 2>/dev/null; then
            echo " Ready!"
            return 0
        fi
        echo -n "."
        sleep 2
        ((attempt++))
    done
    echo " Timeout!"
    return 1
}

# Function to initialize wallets and subnet
# Always runs init-subnet.sh because:
# - init-subnet.sh is fully idempotent
# - It checks wallet existence internally and skips creation if they exist
# - It handles "already registered" errors gracefully
# - Funding checks balance before transferring
init_subnet() {
    echo "  Initializing subnet (creating wallets, funding, registering)..."

    # Check prerequisites
    if ! command -v uvx &> /dev/null; then
        echo "  ERROR: uv is not installed (required for btcli)"
        echo "  Install: curl -LsSf https://astral.sh/uv/install.sh | sh"
        return 1
    fi

    if ! command -v jq &> /dev/null; then
        echo "  ERROR: jq is not installed"
        echo "  Install: brew install jq (macOS) or apt install jq (Linux)"
        return 1
    fi

    # Run init-subnet.sh
    "${SCRIPT_DIR}/init-subnet.sh"
}

# Create shared network if it doesn't exist
docker network create basilica-localnet --subnet 172.28.0.0/16 2>/dev/null && \
    echo "Created shared network: basilica-localnet" || true

# Two-phase startup: Start subtensor first, init wallets, then start remaining services
# This prevents race conditions where validator starts before wallets exist

echo "[1/4] Starting subtensor (network profile)..."
docker compose --profile network up -d ${BUILD_FLAG}

echo ""
echo "[2/4] Waiting for Subtensor..."
wait_for_service "Subtensor" "http://localhost:9944/health"

echo ""
echo "[3/4] Initializing subnet..."
init_subnet

# For network-only profile, we're done
if [ "${PROFILE}" = "network" ]; then
    echo ""
    echo "[4/4] Network profile complete."
else
    echo ""
    echo "[4/4] Starting remaining services for profile: ${PROFILE}..."

    docker compose --profile "${PROFILE}" up -d ${BUILD_FLAG}

    echo ""
    echo "Waiting for services..."

    # Wait for services based on profile
    case "${PROFILE}" in
        validator)
            wait_for_service "Validator" "http://localhost:8080/health" 60
            ;;
        miner)
            wait_for_service "Validator" "http://localhost:8080/health" 60
            wait_for_port "Miner" "localhost" 8092 60
            ;;
    esac
fi

echo ""
echo "========================================"
echo "  Services Started!"
echo "========================================"
echo ""

# Display running services
echo "Running services:"
docker compose ps --format "table {{.Name}}\t{{.Status}}\t{{.Ports}}" 2>/dev/null | head -20

echo ""
echo "Endpoints:"
case "${PROFILE}" in
    network)
        echo "  Subtensor:   ws://localhost:9944"
        ;;
    validator)
        echo "  Subtensor:   ws://localhost:9944"
        echo "  Validator:   http://localhost:8080 (API)"
        echo "               http://localhost:9090/metrics"
        ;;
    miner)
        echo "  Subtensor:   ws://localhost:9944"
        echo "  Validator:   http://localhost:8080 (API)"
        echo "               http://localhost:9090/metrics"
        echo "  Miner:       localhost:8092 (gRPC)"
        echo "               http://localhost:9091/metrics"
        ;;
esac

echo ""
echo "Commands:"
echo "  Check health:  ./test.sh"
echo "  View logs:     docker compose logs -f [service]"
echo "  Stop:          docker compose down"
echo ""
