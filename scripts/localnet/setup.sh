#!/bin/bash
# Basilica Localnet - One-time Setup
# Creates SSH keys and prepares the environment
# Wallet creation is handled by init-subnet.sh

set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"

show_help() {
    echo "Basilica Localnet - One-time Setup"
    echo ""
    echo "Usage: ./setup.sh [-h|--help]"
    echo ""
    echo "Creates SSH keys and pulls Docker images needed for the localnet."
    echo "Run this once before starting services for the first time."
    echo ""
    echo "Prerequisites:"
    echo "  - Docker and Docker Compose installed"
    echo "  - uv (optional, needed later for init-subnet.sh)"
    echo ""
    echo "What it does:"
    echo "  1. Checks prerequisites (Docker, Docker Compose, uv)"
    echo "  2. Generates SSH keys for miner and validator"
    echo "  3. Pulls required Docker images"
    echo ""
    echo "Options:"
    echo "  -h, --help   Show this help"
    echo ""
    echo "Next steps after setup:"
    echo "  1. ./start.sh network    Start Subtensor"
    echo "  2. ./init-subnet.sh      Create wallets and register neurons"
    echo "  3. ./start.sh miner      Start all services"
    echo "  4. ./test.sh             Check health"
}

[[ "${1:-}" =~ ^(-h|--help)$ ]] && show_help && exit 0

echo "========================================"
echo "  Basilica Localnet Setup"
echo "========================================"
echo ""

# =============================================================================
# Check Prerequisites
# =============================================================================
echo "[1/3] Checking prerequisites..."

# Check Docker
if ! command -v docker &> /dev/null; then
    echo "  ERROR: Docker is not installed"
    echo "  Install Docker: https://docs.docker.com/get-docker/"
    exit 1
fi
echo "  Docker: OK"

# Check Docker Compose
if ! docker compose version &> /dev/null; then
    echo "  ERROR: Docker Compose is not available"
    echo "  Docker Compose should be included with Docker Desktop"
    exit 1
fi
echo "  Docker Compose: OK"

# Check uv (required for btcli)
if command -v uvx &> /dev/null; then
    echo "  uv: OK"
else
    echo "  uv: Not found (required for init-subnet.sh)"
    echo "  Install uv: curl -LsSf https://astral.sh/uv/install.sh | sh"
fi

echo ""

# =============================================================================
# Generate SSH Keys
# =============================================================================
echo "[2/3] Setting up SSH keys..."

SSH_KEY_DIR="${SCRIPT_DIR}/ssh-keys"
mkdir -p "${SSH_KEY_DIR}"

# Generate miner SSH key for node access
MINER_KEY="${SSH_KEY_DIR}/miner_node_key"
if [ -f "${MINER_KEY}" ]; then
    echo "  Miner SSH key already exists"
else
    echo "  Generating miner SSH key..."
    ssh-keygen -t ed25519 -f "${MINER_KEY}" -N "" -C "basilica-miner-localnet"
    chmod 600 "${MINER_KEY}"
    chmod 644 "${MINER_KEY}.pub"
fi

# Generate validator SSH key
VALIDATOR_KEY="${SSH_KEY_DIR}/validator_key"
if [ -f "${VALIDATOR_KEY}" ]; then
    echo "  Validator SSH key already exists"
else
    echo "  Generating validator SSH key..."
    ssh-keygen -t ed25519 -f "${VALIDATOR_KEY}" -N "" -C "basilica-validator-localnet"
    chmod 600 "${VALIDATOR_KEY}"
    chmod 644 "${VALIDATOR_KEY}.pub"
fi

echo "  SSH keys stored in: ${SSH_KEY_DIR}"
echo ""

# =============================================================================
# Pull Docker Images
# =============================================================================
echo "[3/3] Pulling Docker images..."

cd "${SCRIPT_DIR}"
docker compose pull subtensor 2>/dev/null || true

echo ""
echo "========================================"
echo "  Setup Complete!"
echo "========================================"
echo ""
echo "Next steps:"
echo "  1. Start Subtensor:     ./start.sh network"
echo "  2. Initialize subnet:   ./init-subnet.sh"
echo "  3. Start services:      ./start.sh miner"
echo "  4. Check health:        ./test.sh"
echo ""
echo "Available profiles:"
echo "  network     - Subtensor only"
echo "  validator   - Subtensor + Validator"
echo "  miner       - Above + Miner"
echo "  all         - Everything (default)"
echo ""
