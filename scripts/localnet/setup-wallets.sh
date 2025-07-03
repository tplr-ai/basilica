#!/bin/bash
# Setup wallets for local Bittensor testing
# This script creates the necessary wallets for owner, validator, and miners

set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
WALLETS_DIR="${HOME}/.bittensor/wallets"

echo "=== Basilica Local Wallet Setup ==="
echo "Creating wallets in: ${WALLETS_DIR}"

# Check if uv is installed
if ! command -v uvx &> /dev/null; then
    echo "Installing uv package manager..."
    curl -LsSf https://astral.sh/uv/install.sh | sh
    # Source the shell profile to get uv in PATH
    if [ -f "$HOME/.bashrc" ]; then
        source "$HOME/.bashrc"
    elif [ -f "$HOME/.zshrc" ]; then
        source "$HOME/.zshrc"
    fi
fi

# Function to create wallet with error handling
create_wallet() {
    local wallet_name=$1
    local wallet_type=$2
    local hotkey_name=${3:-default}

    echo ""
    echo "Creating ${wallet_type} for wallet: ${wallet_name}"

    if [ "$wallet_type" = "coldkey" ]; then
        if [ -d "${WALLETS_DIR}/${wallet_name}/coldkey" ]; then
            echo "Coldkey already exists for ${wallet_name}"
        else
            echo "  Creating new coldkey..."
            uvx --from bittensor-cli btcli wallet new_coldkey \
                --wallet.name "${wallet_name}" \
                -p "${WALLETS_DIR}" \
                --n-words 24 \
                --no-use-password
        fi
    else
        if [ -f "${WALLETS_DIR}/${wallet_name}/hotkeys/${hotkey_name}" ]; then
            echo "Hotkey '${hotkey_name}' already exists for ${wallet_name}"
        else
            echo "  Creating new hotkey '${hotkey_name}'..."
            uvx --from bittensor-cli btcli wallet new_hotkey \
                --wallet.name "${wallet_name}" \
                --wallet.hotkey "${hotkey_name}" \
                --n-words 24
        fi
    fi
}

# Create wallets
echo ""
echo "=== Creating Wallets ==="

# Owner wallet
create_wallet "owner" "coldkey"
create_wallet "owner" "hotkey" "default"

# Validator wallet
create_wallet "validator" "coldkey"
create_wallet "validator" "hotkey" "default"

# Miner 1 wallet
create_wallet "miner_1" "coldkey"
create_wallet "miner_1" "hotkey" "default"

# Miner 2 wallet
create_wallet "miner_2" "coldkey"
create_wallet "miner_2" "hotkey" "default"

# Create wallet for test configurations
create_wallet "test_validator" "coldkey"
create_wallet "test_validator" "hotkey" "test_hotkey"

create_wallet "test_miner" "coldkey"
create_wallet "test_miner" "hotkey" "default"

# Public API wallet
create_wallet "public-api" "coldkey"
create_wallet "public-api" "hotkey" "default"

echo ""
echo "=== Wallet Setup Complete ==="
echo ""
echo "Created wallets:"
echo "  - owner (coldkey + hotkey)"
echo "  - validator (coldkey + hotkey)"
echo "  - miner_1 (coldkey + hotkey)"
echo "  - miner_2 (coldkey + hotkey)"
echo "  - test_validator (coldkey + test_hotkey)"
echo "  - test_miner (coldkey + hotkey)"
echo "  - public-api (coldkey + hotkey)"
echo ""
echo "Wallets are stored in: ${WALLETS_DIR}"
echo ""

# Display wallet addresses
echo "=== Wallet Addresses ==="
for wallet in owner validator miner_1 miner_2 test_validator test_miner public-api; do
    if [ -d "${WALLETS_DIR}/${wallet}" ]; then
        echo ""
        echo "${wallet}:"
        # Show coldkey address if possible
        if command -v btcli &> /dev/null; then
            btcli wallet overview --wallet.name "${wallet}" --wallet.path "${WALLETS_DIR}" 2>/dev/null || true
        fi
    fi
done

# Copy wallets to localnet directory with proper permissions for containers
echo ""
echo "=== Preparing Wallets for Container Access ==="
LOCALNET_WALLETS_DIR="${SCRIPT_DIR}/wallets"
mkdir -p "${LOCALNET_WALLETS_DIR}"

# Copy wallets with proper permissions
if [ -d "${WALLETS_DIR}" ]; then
    cp -r "${WALLETS_DIR}"/* "${LOCALNET_WALLETS_DIR}/" 2>/dev/null || true
    chmod -R 755 "${LOCALNET_WALLETS_DIR}"
    echo "Wallets copied to ${LOCALNET_WALLETS_DIR} with read permissions"
else
    echo "No wallets found at ${WALLETS_DIR}"
fi

echo ""
echo "Next steps:"
echo "1. Start the local Subtensor network: docker compose -f compose.yml up -d"
echo "2. Services will run without registration (skip_registration=true for miner, --local-test for validator)"
echo ""
