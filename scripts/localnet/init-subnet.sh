#!/bin/bash
# Basilica Localnet - Initialize Subnet
# Dissolves genesis subnet 1, re-registers with deep AMM pool (10,000 TAO),
# creates wallets, funds them, and registers neurons.
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
    echo "  1. Seeds subnet AMM pool (dissolve genesis subnet, re-register with 10,000 TAO)"
    echo "  2. Creates validator and miner_1 wallets (coldkey + hotkey)"
    echo "  3. Creates Alice wallet from known dev seed"
    echo "  4. Funds wallets via Alice transfer (10,000 TAO each)"
    echo "  5. Registers validator and miner on netuid=${NETUID}"
    echo "  6. Stakes 1 TAO to validator, starts subnet (emissions + alpha)"
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
echo "[1/6] Checking prerequisites..."

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
# Disable EVM Deployment Whitelist (matching mainnet behavior)
# =============================================================================
echo "[1.5/6] Disabling EVM deployment whitelist..."

uv run --python 3.12 --with substrate-interface python3 -c "
from substrateinterface import SubstrateInterface, Keypair
substrate = SubstrateInterface(url='ws://localhost:9944')
alice = Keypair.create_from_uri('//Alice')
inner = substrate.compose_call('EVM', 'disable_whitelist', {'disabled': True})
sudo = substrate.compose_call('Sudo', 'sudo', {'call': inner})
extrinsic = substrate.create_signed_extrinsic(call=sudo, keypair=alice)
receipt = substrate.submit_extrinsic(extrinsic, wait_for_inclusion=True)
print('  EVM whitelist disabled' if receipt.is_success else f'  WARNING: {receipt.error_message}')
" || echo "  WARNING: Could not disable EVM whitelist (continuing anyway)"

echo ""

# =============================================================================
# Seed Subnet AMM Pool
# Dissolves genesis subnet 1 (tiny pool), re-registers with 10,000 TAO initial
# liquidity so integration tests get predictable ~1:1 TAO:alpha swap rates.
# =============================================================================
echo "[1.8/6] Seeding subnet AMM pool with initial liquidity..."

uv run --python 3.12 --with substrate-interface python3 -c "
from substrateinterface import SubstrateInterface, Keypair
import sys

substrate = SubstrateInterface(url='ws://localhost:9944')
alice = Keypair.create_from_uri('//Alice')

POOL_TAO_RAO = 10_000_000_000_000  # 10,000 TAO in RAO

# Set NetworkMinLockCost to 10,000 TAO so register_network seeds a deep pool
inner = substrate.compose_call('AdminUtils', 'sudo_set_network_min_lock_cost', {'lock_cost': POOL_TAO_RAO})
sudo = substrate.compose_call('Sudo', 'sudo', {'call': inner})
receipt = substrate.submit_extrinsic(substrate.create_signed_extrinsic(call=sudo, keypair=alice), wait_for_inclusion=True)
if not receipt.is_success:
    print(f'  ERROR: Failed to set NetworkMinLockCost: {receipt.error_message}')
    sys.exit(1)
print('  NetworkMinLockCost set to 10,000 TAO')

# Dissolve genesis subnet 1 (has tiny pool from chain genesis)
inner = substrate.compose_call('SubtensorModule', 'root_dissolve_network', {'netuid': 1})
sudo = substrate.compose_call('Sudo', 'sudo', {'call': inner})
receipt = substrate.submit_extrinsic(substrate.create_signed_extrinsic(call=sudo, keypair=alice), wait_for_inclusion=True)
if not receipt.is_success:
    print(f'  ERROR: Failed to dissolve subnet 1: {receipt.error_message}')
    sys.exit(1)
print('  Genesis subnet 1 dissolved')

# Re-register subnet (gets netuid=1, Alice pays 10,000 TAO lock -> pool seeded 10,000:10,000)
call = substrate.compose_call('SubtensorModule', 'register_network', {'hotkey': alice.ss58_address})
receipt = substrate.submit_extrinsic(substrate.create_signed_extrinsic(call=call, keypair=alice), wait_for_inclusion=True)
if not receipt.is_success:
    print(f'  ERROR: Failed to register subnet: {receipt.error_message}')
    sys.exit(1)
print('  Subnet re-registered with 10,000 TAO : 10,000 alpha pool')

# Reset NetworkMinLockCost to 1 TAO for future registrations
inner = substrate.compose_call('AdminUtils', 'sudo_set_network_min_lock_cost', {'lock_cost': 1_000_000_000})
sudo = substrate.compose_call('Sudo', 'sudo', {'call': inner})
receipt = substrate.submit_extrinsic(substrate.create_signed_extrinsic(call=sudo, keypair=alice), wait_for_inclusion=True)
if not receipt.is_success:
    print(f'  WARNING: Failed to reset NetworkMinLockCost: {receipt.error_message}')
print('  NetworkMinLockCost reset to 1 TAO')

# Set tempo to 20 (default 10 is too short for btcli's mortal-era transaction calculation)
inner = substrate.compose_call('AdminUtils', 'sudo_set_tempo', {'netuid': 1, 'tempo': 20})
sudo = substrate.compose_call('Sudo', 'sudo', {'call': inner})
receipt = substrate.submit_extrinsic(substrate.create_signed_extrinsic(call=sudo, keypair=alice), wait_for_inclusion=True)
if not receipt.is_success:
    print(f'  WARNING: Failed to set tempo: {receipt.error_message}')
print('  Tempo set to 20')

# Verify pool state
tao = substrate.query('SubtensorModule', 'SubnetTAO', [1])
alpha_in = substrate.query('SubtensorModule', 'SubnetAlphaIn', [1])
print(f'  Pool: SubnetTAO={tao}, SubnetAlphaIn={alpha_in}')
" || { echo "  ERROR: Pool seeding failed"; exit 1; }

echo ""

# =============================================================================
# Create Wallets
# =============================================================================
echo "[2/6] Creating wallets in ${WALLETS_DIR}..."

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
            uvx --python 3.12 --from bittensor-cli btcli wallet new-coldkey \
                --wallet-name "${wallet_name}" \
                --wallet-path "${WALLETS_DIR}" \
                --n-words 24 \
                --no-use-password
        fi
    else
        if [ -f "${WALLETS_DIR}/${wallet_name}/hotkeys/${hotkey_name}" ]; then
            echo "  Hotkey '${wallet_name}/${hotkey_name}' already exists"
        else
            echo "  Creating hotkey '${wallet_name}/${hotkey_name}'..."
            uvx --python 3.12 --from bittensor-cli btcli wallet new-hotkey \
                --wallet-name "${wallet_name}" \
                --hotkey "${hotkey_name}" \
                --wallet-path "${WALLETS_DIR}" \
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
echo "[2.5/6] Creating Alice wallet from known seed..."

# Alice's well-known seed for Substrate dev chains
ALICE_SEED="0xe5be9a5092b81bca64be81d212e7f2f9eba183bb7a90954f7b76361f6edb5c0a"

if [ -f "${WALLETS_DIR}/alice/coldkey" ]; then
    echo "  Alice coldkey already exists"
else
    echo "  Regenerating Alice coldkey from seed..."
    uvx --python 3.12 --from bittensor-cli btcli wallet regen-coldkey \
        --wallet-path "${WALLETS_DIR}" \
        --wallet-name alice \
        --seed "${ALICE_SEED}" \
        --no-use-password
fi

echo ""

# =============================================================================
# Fund Wallets (using Alice transfer instead of faucet - no torch required)
# =============================================================================
echo "[3/6] Funding wallets via Alice transfer..."

fund_wallet() {
    local wallet_name=$1
    local amount=${2:-10000}

    # Check current balance first
    local current_balance
    local balance_json
    balance_json=$(uvx --python 3.12 --from bittensor-cli btcli wallet balance \
        --wallet-path "${WALLETS_DIR}" \
        --wallet-name "${wallet_name}" \
        --network local \
        --json-output 2>&1) || {
        echo "  WARNING: balance check failed for '${wallet_name}':"
        echo "  ${balance_json}"
        echo "  Proceeding with funding..."
    }
    current_balance=$(echo "$balance_json" | jq -r '.balance.free // "0"' 2>/dev/null | sed 's/[^0-9.]//g')

    # If balance > 0, skip funding
    if [ -n "$current_balance" ] && [ "$(echo "$current_balance > 0" | bc -l 2>/dev/null)" = "1" ]; then
        echo "  '${wallet_name}' already has ${current_balance} TAO, skipping transfer"
        return 0
    fi

    echo "  Getting address for '${wallet_name}'..."
    local dest_addr
    local list_json
    list_json=$(uvx --python 3.12 --from bittensor-cli btcli wallet list \
        --wallet-path "${WALLETS_DIR}" \
        --json-output 2>&1) || {
        echo "  ERROR: wallet list failed for '${wallet_name}':"
        echo "  ${list_json}"
        return 1
    }
    dest_addr=$(echo "$list_json" | jq -r --arg name "${wallet_name}" '.wallets[] | select(.name == $name) | .ss58_address' 2>/dev/null)

    if [ -z "$dest_addr" ] || [ "$dest_addr" = "null" ]; then
        echo "  ERROR: Could not get address for ${wallet_name}"
        return 1
    fi

    echo "  Transferring ${amount} TAO to '${wallet_name}' (${dest_addr})..."
    uvx --python 3.12 --from bittensor-cli btcli wallet transfer \
        --wallet-name alice \
        --wallet-path "${WALLETS_DIR}" \
        --destination "${dest_addr}" \
        --amount "${amount}" \
        --network local \
        --no-prompt
}

fund_wallet "validator"
fund_wallet "miner_1"

echo ""

# =============================================================================
# Register Validator + Miner
# =============================================================================
echo "[4/6] Registering validator and miner..."

echo "  Registering validator on netuid ${NETUID}..."
uvx --python 3.12 --from bittensor-cli btcli subnet register \
    --wallet-name validator \
    --hotkey default \
    --wallet-path "${WALLETS_DIR}" \
    --netuid "${NETUID}" \
    --network local \
    --era 100 \
    --no-prompt

echo "  Registering miner on netuid ${NETUID}..."
uvx --python 3.12 --from bittensor-cli btcli subnet register \
    --wallet-name miner_1 \
    --hotkey default \
    --wallet-path "${WALLETS_DIR}" \
    --netuid "${NETUID}" \
    --network local \
    --era 100 \
    --no-prompt

echo ""

# =============================================================================
# Start Subnet (activates emission schedule and enables subtokens/alpha)
# Must happen before staking — add_stake requires subtokens to be enabled.
# =============================================================================
echo "[5/6] Starting subnet (enabling emissions and subtokens)..."

start_out=$(uvx --python 3.12 --from bittensor-cli btcli subnet start \
    --netuid "${NETUID}" \
    --wallet-name "alice" \
    --wallet-path "${WALLETS_DIR}" \
    --network local \
    --no-prompt 2>&1)
start_exit=$?

if [ $start_exit -ne 0 ]; then
    if echo "$start_out" | grep -qi "already.*started\|already.*enabled\|already.*active"; then
        echo "  Subnet may already be started"
    else
        echo "  WARNING: Subnet start failed (subtokens/alpha may not work)"
        echo "$start_out"
    fi
else
    echo "  Subnet ${NETUID} started (emissions and subtokens enabled)"
fi

echo "  Waiting for next epoch to activate subtokens (tempo=20, ~40s)..."
sleep 45

echo ""

# =============================================================================
# Stake to Validator
# btcli stake add is broken on this chain version (queries Swap.AlphaSqrtPrice
# which doesn't exist on spec_version 380), so we use substrate-interface.
# =============================================================================
echo "[6/6] Staking 1 TAO to validator..."

uv run --python 3.12 --with substrate-interface python3 -c "
from substrateinterface import SubstrateInterface, Keypair
import json, sys
substrate = SubstrateInterface(url='ws://localhost:9944')
with open('${WALLETS_DIR}/validator/coldkey') as f:
    ck = json.load(f)
coldkey = Keypair.create_from_seed(ck['secretSeed'])
with open('${WALLETS_DIR}/validator/hotkeys/default') as f:
    hk = json.load(f)
call = substrate.compose_call('SubtensorModule', 'add_stake', {
    'hotkey': hk['ss58Address'], 'netuid': ${NETUID}, 'amount_staked': 1_000_000_000,
})
receipt = substrate.submit_extrinsic(substrate.create_signed_extrinsic(call=call, keypair=coldkey), wait_for_inclusion=True)
if receipt.is_success:
    print('  Staked 1 TAO to validator')
else:
    print(f'  ERROR: Staking failed: {receipt.error_message}')
    sys.exit(1)
" || { echo "  ERROR: Staking failed"; exit 1; }

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
uvx --python 3.12 --from bittensor-cli btcli subnet list --network local 2>/dev/null | head -20 || true
echo ""
echo "Metagraph (netuid=${NETUID}):"
uvx --python 3.12 --from bittensor-cli btcli subnet metagraph --netuid "${NETUID}" --network local 2>/dev/null | head -20 || true
echo ""
echo "Next steps:"
echo "  1. Start remaining services: ./start.sh miner"
echo "  2. Check health:             ./test.sh"
echo ""
