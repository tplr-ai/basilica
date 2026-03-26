#!/bin/bash
# Basilica Localnet - Restart Services
# Usage: ./restart.sh [services...] [options]
#
# Arguments:
#   services     Service names to restart (default: all)
#                Available: subtensor, validator, miner
#
# Options:
#   --profile    Restart services by profile (network, validator, miner)
#   -f, --logs   Follow logs after restart
#   -h, --help   Show help

set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
cd "${SCRIPT_DIR}"

# Parse arguments
SERVICES=()
PROFILE=""
FOLLOW_LOGS=false

show_help() {
    echo "Basilica Localnet - Restart Services"
    echo ""
    echo "Usage: ./restart.sh [services...] [options]"
    echo ""
    echo "Arguments:"
    echo "  services       Service names to restart (default: all running)"
    echo "                 Available: subtensor, validator, miner"
    echo ""
    echo "Options:"
    echo "  --profile NAME Restart services by profile"
    echo "                 Profiles: network, validator, miner"
    echo "  -f, --logs     Follow logs after restart"
    echo "  -h, --help     Show this help"
    echo ""
    echo "Examples:"
    echo "  ./restart.sh                      # Restart all running services"
    echo "  ./restart.sh miner                # Restart only the miner"
    echo "  ./restart.sh validator miner      # Restart validator and miner"
    echo "  ./restart.sh --profile miner      # Restart all services in miner profile"
    echo "  ./restart.sh miner -f             # Restart miner and follow logs"
}

while [[ $# -gt 0 ]]; do
    case $1 in
        -h|--help)
            show_help
            exit 0
            ;;
        -f|--logs)
            FOLLOW_LOGS=true
            shift
            ;;
        --profile)
            if [[ -n "${2:-}" ]]; then
                PROFILE="$2"
                shift 2
            else
                echo "Error: --profile requires a profile name"
                exit 1
            fi
            ;;
        -*)
            echo "Unknown option: $1"
            echo "Run './restart.sh --help' for usage"
            exit 1
            ;;
        *)
            SERVICES+=("$1")
            shift
            ;;
    esac
done

# Normalize profile name if specified
if [[ -n "${PROFILE}" ]]; then
    case "${PROFILE}" in
        network|subtensor)
            PROFILE="network"
            ;;
        validator|val)
            PROFILE="validator"
            ;;
        miner|min|all)
            PROFILE="miner"
            ;;
        *)
            echo "Unknown profile: ${PROFILE}"
            echo ""
            echo "Available profiles:"
            echo "  network     - Subtensor only"
            echo "  validator   - Subtensor + Validator"
            echo "  miner       - All services"
            exit 1
            ;;
    esac
fi

echo "========================================"
echo "  Restarting Basilica Localnet"
if [[ -n "${PROFILE}" ]]; then
    echo "  Profile: ${PROFILE}"
elif [[ ${#SERVICES[@]} -gt 0 ]]; then
    echo "  Services: ${SERVICES[*]}"
else
    echo "  Services: all running"
fi
echo "========================================"
echo ""

# Build the service/profile arguments
if [[ -n "${PROFILE}" ]]; then
    # Profile-based restart
    echo "[1/3] Stopping services in profile: ${PROFILE}..."
    docker compose --profile "${PROFILE}" stop

    echo ""
    echo "[2/3] Rebuilding services (with cache)..."
    docker compose --profile "${PROFILE}" build

    echo ""
    echo "[3/3] Starting services..."
    docker compose --profile "${PROFILE}" up -d

    LOG_TARGET="--profile ${PROFILE}"
elif [[ ${#SERVICES[@]} -gt 0 ]]; then
    # Service-based restart
    echo "[1/3] Stopping services: ${SERVICES[*]}..."
    docker compose stop "${SERVICES[@]}"

    echo ""
    echo "[2/3] Rebuilding services (with cache)..."
    docker compose build "${SERVICES[@]}"

    echo ""
    echo "[3/3] Starting services..."
    docker compose up -d "${SERVICES[@]}"

    LOG_TARGET="${SERVICES[*]}"
else
    # Restart all running services
    echo "[1/3] Stopping all services..."
    docker compose stop

    echo ""
    echo "[2/3] Rebuilding services (with cache)..."
    docker compose build

    echo ""
    echo "[3/3] Starting services..."
    docker compose up -d

    LOG_TARGET=""
fi

echo ""
echo "========================================"
echo "  Restart Complete!"
echo "========================================"
echo ""

# Display running services
echo "Running services:"
docker compose ps --format "table {{.Name}}\t{{.Status}}\t{{.Ports}}" 2>/dev/null | head -20

if [[ "${FOLLOW_LOGS}" == true ]]; then
    echo ""
    echo "Following logs (Ctrl+C to stop)..."
    echo ""
    if [[ -n "${LOG_TARGET}" && "${LOG_TARGET}" != "--profile"* ]]; then
        docker compose logs -f ${LOG_TARGET}
    else
        docker compose logs -f
    fi
fi
