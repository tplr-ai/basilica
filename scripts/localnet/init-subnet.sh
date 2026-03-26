#!/bin/bash
# Basilica Localnet - Initialize Subnet
# Creates wallets, funds them, and registers neurons on pre-existing subnet 1 ("apex")
#
# Run after: ./start.sh network

set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
WALLETS_DIR="${SCRIPT_DIR}/wallets"
NETUID=1

show_help() {
    echo "Basilica Localnet - Initialize Subnet"
    echo ""
    echo "Usage: ./init-subnet.sh [-h|--help]"
    echo ""
    echo "Creates wallets, funds them via Alice transfer, and registers"
    echo "validator and miner neurons on the local Subtensor chain (netuid=1)."
    echo ""
    echo "Prerequisites:"
    echo "  - uv (for btcli):  curl -LsSf https://astral.sh/uv/install.sh | sh"
    echo "  - jq:              brew install jq (macOS) or apt install jq (Linux)"
    echo "  - bc:              brew install bc (macOS) or apt install bc (Linux)"
    echo "  - Subtensor running: ./start.sh network"
    echo ""
    echo "What it does:"
    echo "  1. Creates validator and miner_1 wallets (coldkey + hotkey)"
    echo "  2. Creates Alice wallet from known dev seed"
    echo "  3. Funds wallets via Alice transfer (10,000 TAO each)"
    echo "  4. Registers validator on netuid=${NETUID}"
    echo "  5. Registers miner on netuid=${NETUID}"
    echo ""
    echo "Options:"
    echo "  -h, --help   Show this help"
    echo ""
    echo "Notes:"
    echo "  - Idempotent: safe to run multiple times"
    echo "  - Wallets stored in: ${WALLETS_DIR}"
    echo ""
    echo "See also: ./setup.sh, ./start.sh"
}

[[ "${1:-}" =~ ^(-h|--help)$ ]] && show_help && exit 0

echo "========================================"
echo "  Basilica Localnet - Subnet Init"
echo "========================================"
echo ""

# =============================================================================
# Check Prerequisites
# =============================================================================
echo "[1/5] Checking prerequisites..."

# Check uv
if ! command -v uvx &> /dev/null; then
    echo "  ERROR: uv is not installed"
    echo "  Install uv: curl -LsSf https://astral.sh/uv/install.sh | sh"
    exit 1
fi
echo "  uv: OK"

# Check jq (needed for parsing wallet addresses)
if ! command -v jq &> /dev/null; then
    echo "  ERROR: jq is not installed"
    echo "  Install jq: brew install jq (macOS) or apt install jq (Linux)"
    exit 1
fi
echo "  jq: OK"

# Check bc (needed for balance comparison in fund_wallet)
if ! command -v bc &> /dev/null; then
    echo "  ERROR: bc is not installed"
    echo "  Install bc: brew install bc (macOS) or apt install bc (Linux)"
    exit 1
fi
echo "  bc: OK"

# Check Subtensor is running
if ! curl -sf "http://localhost:9944/health" > /dev/null 2>&1; then
    echo "  ERROR: Subtensor is not running"
    echo "  Start it first: ./start.sh network"
    exit 1
fi
echo "  Subtensor: OK"

echo ""

# =============================================================================
# Create Wallets
# =============================================================================
echo "[2/5] Creating wallets in ${WALLETS_DIR}..."

mkdir -p "${WALLETS_DIR}"

create_wallet() {
    local wallet_name=$1
    local key_type=$2
    local hotkey_name=${3:-default}

    if [ "$key_type" = "coldkey" ]; then
        if [ -f "${WALLETS_DIR}/${wallet_name}/coldkey" ]; then
            echo "  Coldkey '${wallet_name}' already exists"
        else
            echo "  Creating coldkey '${wallet_name}'..."
            uvx --from bittensor-cli btcli wallet new_coldkey \
                --wallet.name "${wallet_name}" \
                --wallet.path "${WALLETS_DIR}" \
                --n-words 24 \
                --no-use-password
        fi
    else
        if [ -f "${WALLETS_DIR}/${wallet_name}/hotkeys/${hotkey_name}" ]; then
            echo "  Hotkey '${wallet_name}/${hotkey_name}' already exists"
        else
            echo "  Creating hotkey '${wallet_name}/${hotkey_name}'..."
            uvx --from bittensor-cli btcli wallet new_hotkey \
                --wallet.name "${wallet_name}" \
                --wallet.hotkey "${hotkey_name}" \
                --wallet.path "${WALLETS_DIR}" \
                --n-words 24
        fi
    fi
}

# Create validator wallet
create_wallet "validator" "coldkey"
create_wallet "validator" "hotkey" "default"

# Create miner wallet
create_wallet "miner_1" "coldkey"
create_wallet "miner_1" "hotkey" "default"

echo ""

# =============================================================================
# Create Alice Wallet (pre-funded account for transfers)
# =============================================================================
echo "[2.5/5] Creating Alice wallet from known seed..."

# Alice's well-known seed for Substrate dev chains
ALICE_SEED="0xe5be9a5092b81bca64be81d212e7f2f9eba183bb7a90954f7b76361f6edb5c0a"

if [ -f "${WALLETS_DIR}/alice/coldkey" ]; then
    echo "  Alice coldkey already exists"
else
    echo "  Regenerating Alice coldkey from seed..."
    uvx --from bittensor-cli btcli wallet regen_coldkey \
        --wallet.path "${WALLETS_DIR}" \
        --wallet.name alice \
        --seed "${ALICE_SEED}" \
        --no-use-password
fi

echo ""

# =============================================================================
# Fund Wallets (using Alice transfer instead of faucet - no torch required)
# =============================================================================
echo "[3/5] Funding wallets via Alice transfer..."

fund_wallet() {
    local wallet_name=$1
    local amount=${2:-10000}

    # Check current balance first
    local current_balance
    current_balance=$(uvx --from bittensor-cli btcli wallet balance \
        --wallet-path "${WALLETS_DIR}" \
        --wallet-name "${wallet_name}" \
        --network local \
        --json-output 2>/dev/null | jq -r '.balance.free // "0"' | sed 's/[^0-9.]//g')

    # If balance > 0, skip funding
    if [ -n "$current_balance" ] && [ "$(echo "$current_balance > 0" | bc -l 2>/dev/null)" = "1" ]; then
        echo "  '${wallet_name}' already has ${current_balance} TAO, skipping transfer"
        return 0
    fi

    echo "  Getting address for '${wallet_name}'..."
    local dest_addr
    dest_addr=$(uvx --from bittensor-cli btcli wallet list \
        --wallet-path "${WALLETS_DIR}" \
        --wallet-name "${wallet_name}" \
        --json-output 2>/dev/null | jq -r '.wallets[0].ss58_address')

    if [ -z "$dest_addr" ] || [ "$dest_addr" = "null" ]; then
        echo "  ERROR: Could not get address for ${wallet_name}"
        return 1
    fi

    echo "  Transferring ${amount} TAO to '${wallet_name}' (${dest_addr})..."
    uvx --from bittensor-cli btcli wallet transfer \
        --wallet.name alice \
        --wallet.path "${WALLETS_DIR}" \
        --destination "${dest_addr}" \
        --amount "${amount}" \
        --network local \
        --no-prompt
}

fund_wallet "validator"
fund_wallet "miner_1"

echo ""

# =============================================================================
# Register Validator (using pre-registered subnet 1 "apex")
# =============================================================================
echo "[4/5] Registering validator on netuid=${NETUID}..."

reg_out=$(uvx --from bittensor-cli btcli subnet register \
    --wallet.name "validator" \
    --wallet.hotkey "default" \
    --wallet.path "${WALLETS_DIR}" \
    --netuid "${NETUID}" \
    --network local \
    --no-prompt 2>&1)
reg_exit=$?

if [ $reg_exit -ne 0 ]; then
    if echo "$reg_out" | grep -qi "already.*registered\|already.*exist"; then
        echo "  Validator may already be registered"
    else
        echo "  ERROR: Validator registration failed"
        echo "$reg_out"
        exit 1
    fi
fi

echo ""

# =============================================================================
# Stake TAO to Validator (required for validator permit)
# =============================================================================
echo "[4.5/5] Staking TAO to validator hotkey for validator permit..."

stake_out=$(uvx --from bittensor-cli btcli stake add \
    --wallet.name "validator" \
    --wallet.hotkey "default" \
    --wallet.path "${WALLETS_DIR}" \
    --netuid "${NETUID}" \
    --amount 1000 \
    --no-safe-staking \
    --no-mev-protection \
    --network local \
    --no-prompt 2>&1)
stake_exit=$?

if [ $stake_exit -ne 0 ]; then
    echo "  WARNING: Staking failed (may already be staked)"
    echo "$stake_out"
else
    echo "  Staked 1000 TAO to validator hotkey"
fi

echo ""

# =============================================================================
# Register Miner
# =============================================================================
echo "[5/5] Registering miner on netuid=${NETUID}..."

reg_out=$(uvx --from bittensor-cli btcli subnet register \
    --wallet.name "miner_1" \
    --wallet.hotkey "default" \
    --wallet.path "${WALLETS_DIR}" \
    --netuid "${NETUID}" \
    --network local \
    --no-prompt 2>&1)
reg_exit=$?

if [ $reg_exit -ne 0 ]; then
    if echo "$reg_out" | grep -qi "already.*registered\|already.*exist"; then
        echo "  Miner may already be registered"
    else
        echo "  ERROR: Miner registration failed"
        echo "$reg_out"
        exit 1
    fi
fi

echo ""

echo ""

# =============================================================================
# Summary
# =============================================================================
echo "========================================"
echo "  Subnet Initialization Complete!"
echo "========================================"
echo ""
echo "Wallets created in: ${WALLETS_DIR}"
ls -la "${WALLETS_DIR}/"
echo ""
echo "Subnet info:"
uvx --from bittensor-cli btcli subnet list --network local 2>/dev/null | head -20 || true
echo ""
echo "Metagraph (netuid=${NETUID}):"
uvx --from bittensor-cli btcli subnet metagraph --netuid "${NETUID}" --network local 2>/dev/null | head -20 || true
echo ""
echo "Next steps:"
echo "  1. Start remaining services: ./start.sh miner"
echo "  2. Check health:             ./test.sh"
echo ""
