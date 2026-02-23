#!/bin/bash
# Basilica Localnet - Stop Services
# Usage: ./stop.sh [--clean]
#
# Options:
#   --clean   Reset blockchain state (volumes + network) for fresh start
#             Note: Wallets are preserved (can be re-registered on fresh chain)

set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
cd "${SCRIPT_DIR}"

# Parse arguments
CLEAN=false

for arg in "$@"; do
    case $arg in
        --clean|--purge|--reset)
            CLEAN=true
            ;;
        -h|--help)
            echo "Usage: ./stop.sh [--clean]"
            echo ""
            echo "Options:"
            echo "  --clean   Reset blockchain state (volumes + network) for fresh start"
            echo "            Note: Wallets are preserved (can be re-registered on fresh chain)"
            echo ""
            exit 0
            ;;
        -*)
            echo "Unknown option: $arg"
            echo "Use --help for usage"
            exit 1
            ;;
    esac
done

echo "========================================"
echo "  Stopping Basilica Localnet"
if [ "$CLEAN" = true ]; then
    echo "  Mode: Clean (reset blockchain state)"
fi
echo "========================================"
echo ""

# Stop containers (include all profiles so compose sees every service)
if [ "$CLEAN" = true ]; then
    echo "[1/3] Stopping containers and removing volumes..."
    docker compose --profile network --profile validator --profile miner down -v 2>/dev/null || true

    echo ""
    echo "[2/3] Removing orphaned volumes..."
    VOLUMES=$(docker volume ls --filter "label=com.docker.compose.project=localnet" -q 2>/dev/null)
    if [ -n "$VOLUMES" ]; then
        docker volume rm $VOLUMES 2>/dev/null || true
    fi

    echo ""
    echo "[3/3] Removing Docker network..."
    docker network rm basilica-localnet 2>/dev/null || true
else
    echo "[1/3] Stopping containers..."
    docker compose --profile network --profile validator --profile miner down 2>/dev/null || true

    echo ""
    echo "[2/3] Skipping volume removal (use --clean to remove)"
    echo "[3/3] Skipping network removal (use --clean to remove)"
fi

echo ""
echo "========================================"
echo "  Localnet Stopped"
echo "========================================"
echo ""

# Show status
RUNNING=$(docker compose ps -q 2>/dev/null | wc -l | tr -d ' ')
if [ "$RUNNING" -gt 0 ]; then
    echo "Warning: Some containers may still be running:"
    docker compose ps 2>/dev/null
else
    echo "All containers stopped."
fi

if [ "$CLEAN" = true ]; then
    echo ""
    echo "Blockchain state reset. Wallets preserved for re-registration."
    echo "Ready for fresh start:"
    echo "  1. ./start.sh network"
    echo "  2. ./init-subnet.sh"
else
    echo ""
    echo "Data preserved. To start again: ./start.sh"
    echo "To remove all data: ./stop.sh --clean"
fi
echo ""
