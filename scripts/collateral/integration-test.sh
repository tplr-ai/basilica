#!/usr/bin/env bash
# Collateral Contract Integration Test
#
# Deploys a fresh collateral contract to the local Substrate testnet and
# exercises TAO deposit, reclaim (with finalization delay), and slash flows.
#
# Prerequisites:
#   - Local Subtensor running at http://localhost:9944
#   - cast and forge installed (Foundry)
#   - python3 available
#
# Usage:
#   ./scripts/collateral/integration-test.sh
set -euo pipefail

# ─── Paths ───────────────────────────────────────────────────────────────────

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
REPO_ROOT="$(cd "${SCRIPT_DIR}/../.." && pwd)"
CONTRACT_DIR="${REPO_ROOT}/crates/collateral-contract"

source "${REPO_ROOT}/scripts/lib/common.sh"

# ─── Constants ───────────────────────────────────────────────────────────────

RPC_URL="http://localhost:9944"
FAUCET_KEY="0x5fb92d6e98884f76de468fa3f6278f8807c48bebc13595d45af5bdc4da702133"
DECISION_TIMEOUT=5        # seconds — short for fast test cycles
MIN_COLLATERAL=1          # 1 wei
NETUID=1

# Use the real localnet validator hotkey so alpha staking works with precompiles.
LOCALNET_WALLETS_DIR="${REPO_ROOT}/scripts/localnet/wallets"
VALIDATOR_HOTKEY_FILE="${LOCALNET_WALLETS_DIR}/validator/hotkeys/defaultpub.txt"
if [[ -f "$VALIDATOR_HOTKEY_FILE" ]]; then
    VALIDATOR_HOTKEY="0x$(awk -F'"' '/"publicKey"/ {print $4; exit}' "$VALIDATOR_HOTKEY_FILE" | sed 's/^0x//')"
else
    VALIDATOR_HOTKEY="0x0000000000000000000000000000000000000000000000000000000000000001"
fi

HOTKEY_1="0x0000000000000000000000000000000000000000000000000000000000000064"
HOTKEY_2="0x0000000000000000000000000000000000000000000000000000000000000065"
HOTKEY_3="0x0000000000000000000000000000000000000000000000000000000000000066"
HOTKEY_4="0x0000000000000000000000000000000000000000000000000000000000000067"
NODE_ID_1="0x00000000000000000000000000000001"  # bytes16
NODE_ID_2="0x00000000000000000000000000000002"  # bytes16
NODE_ID_3="0x00000000000000000000000000000003"  # bytes16
NODE_ID_4="0x00000000000000000000000000000004"  # bytes16

STAKING_PRECOMPILE="0x0000000000000000000000000000000000000805"
ADDRESS_MAPPING="0x000000000000000000000000000000000000080C"

ZERO_BYTES32="0x0000000000000000000000000000000000000000000000000000000000000000"
ZERO_ADDR="0x0000000000000000000000000000000000000000"

# TAO amounts in wei (1 TAO = 1e18 wei)
TAO_10="10000000000000000000"
TAO_8="8000000000000000000"
TAO_5="5000000000000000000"
TAO_3="3000000000000000000"
TAO_50="50000000000000000000"
TAO_100="100000000000000000000"

# RAO amounts for staking precompile (1 TAO = 1e9 RAO)
RAO_30_TAO="30000000000"  # 30 TAO in RAO — one-time stake during setup

TEST_URL="http://localhost:8080/evidence/test.json"
TEST_SHA="0xd41d8cd98f00b204e9800998ecf8427ed41d8cd98f00b204e9800998ecf8427e"

# ─── Test Counters ───────────────────────────────────────────────────────────

TESTS_PASSED=0
TESTS_FAILED=0
TESTS_TOTAL=0
NEXT_RECLAIM_ID=0  # mirrors contract's nextReclaimId — increment after each reclaimCollateral call

# ─── Helper Functions ────────────────────────────────────────────────────────

wei_compare() {
    python3 "${SCRIPT_DIR}/wei_compare.py" "$@"
}

banner() {
    echo ""
    echo -e "${PURPLE}════════════════════════════════════════════════════════════════${NC}"
    echo -e "${PURPLE}  $1${NC}"
    echo -e "${PURPLE}════════════════════════════════════════════════════════════════${NC}"
}

section() {
    echo ""
    log_header "$1"
}

pass() {
    local desc="$1"
    TESTS_PASSED=$((TESTS_PASSED + 1))
    TESTS_TOTAL=$((TESTS_TOTAL + 1))
    echo -e "  ${GREEN}[PASS]${NC}  $desc"
}

fail() {
    local desc="$1"
    TESTS_FAILED=$((TESTS_FAILED + 1))
    TESTS_TOTAL=$((TESTS_TOTAL + 1))
    echo -e "  ${RED}[FAIL]${NC}  $desc"
}

assert_eq() {
    local actual="${1,,}"   # lowercase
    local expected="${2,,}" # lowercase
    local desc="$3"
    if [[ "$actual" == "$expected" ]]; then
        pass "$desc"
    else
        fail "$desc (got: $actual, expected: $expected)"
    fi
}

assert_gt() {
    local actual="$1"
    local threshold="$2"
    local desc="$3"
    if wei_compare gt "$actual" "$threshold"; then
        pass "$desc"
    else
        fail "$desc (got: $actual, expected > $threshold)"
    fi
}

assert_revert() {
    local desc="$1"
    shift
    if "$@" >/dev/null 2>&1; then
        fail "$desc -- expected revert but succeeded"
    else
        pass "$desc"
    fi
}

# Wraps cast send with standard flags. Extra args (e.g. --value) pass through.
cast_send() {
    local key="$1" contract="$2" sig="$3"
    shift 3
    cast send --rpc-url "$RPC_URL" --private-key "$key" --legacy \
        "$contract" "$sig" "$@" >/dev/null
}

# Wraps cast call. Returns decoded value to stdout.
# Strips the "[1e19]"-style annotation that newer foundry versions append to large numbers.
cast_query() {
    local contract="$1" sig="$2"
    shift 2
    cast call --rpc-url "$RPC_URL" "$contract" "$sig" "$@" | awk '{print $1}'
}

get_balance() {
    cast balance --rpc-url "$RPC_URL" "$1"
}

# ═════════════════════════════════════════════════════════════════════════════
#  Prerequisites
# ═════════════════════════════════════════════════════════════════════════════

banner "Collateral Contract Integration Test"

section "Prerequisites"

for cmd in cast forge python3; do
    if command -v "$cmd" >/dev/null 2>&1; then
        log_info "$cmd ... found"
    else
        log_error "$cmd not found -- install it and retry"
        exit 1
    fi
done

if block_num="$(cast block-number --rpc-url "$RPC_URL" 2>/dev/null)"; then
    log_info "RPC reachable at $RPC_URL (block #${block_num})"
else
    log_error "RPC not reachable at $RPC_URL -- start localnet first"
    exit 1
fi

# ═════════════════════════════════════════════════════════════════════════════
#  Setup: Deploy & Fund
# ═════════════════════════════════════════════════════════════════════════════

section "Setup: Deploy & Fund"

DEPLOYER_ADDR="$(cast wallet address --private-key "$FAUCET_KEY")"
log_info "Deployer/Trustee: $DEPLOYER_ADDR"

log_info "Creating miner wallet..."
wallet_json="$(cast wallet new --json)"
MINER_KEY="$(printf '%s\n' "$wallet_json" | awk -F'"' '/"private_key"/ {print $4; exit}')"
MINER_ADDR="$(printf '%s\n' "$wallet_json" | awk -F'"' '/"address"/ {print $4; exit}')"
[[ -n "$MINER_KEY" ]]  || { log_error "Failed to parse miner private key"; exit 1; }
[[ -n "$MINER_ADDR" ]] || { log_error "Failed to parse miner address"; exit 1; }
log_info "  Miner address: $MINER_ADDR"

log_info "Funding miner with 100 TAO from faucet..."
cast send --rpc-url "$RPC_URL" --private-key "$FAUCET_KEY" --legacy \
    "$MINER_ADDR" --value "$TAO_100" >/dev/null
log_success "Miner funded ($(wei_compare fmt "$TAO_100"))"

log_info "Staking 30 TAO as alpha for later alpha deposit tests..."
cast_send "$MINER_KEY" "$STAKING_PRECOMPILE" \
    "addStake(bytes32,uint256,uint256)" \
    "$VALIDATOR_HOTKEY" "$RAO_30_TAO" "$NETUID"

# Query how much alpha the miner actually received (depends on AMM pool state).
MINER_COLDKEY="$(cast_query "$ADDRESS_MAPPING" "addressMapping(address)(bytes32)" "$MINER_ADDR")"
MINER_ALPHA="$(cast_query "$STAKING_PRECOMPILE" \
    "getStake(bytes32,bytes32,uint256)(uint256)" \
    "$VALIDATOR_HOTKEY" "$MINER_COLDKEY" "$NETUID")"
log_success "Miner staked 30 TAO, received ${MINER_ALPHA} alpha RAO"

# Derive deposit amounts: split miner's alpha into 4 equal parts for T9, T10, T13, spare.
ALPHA_DEPOSIT_RAO="$(python3 -c "print(int('${MINER_ALPHA}') // 4)")"
log_info "Alpha deposit amount per test: ${ALPHA_DEPOSIT_RAO} RAO (1/4 of miner's alpha)"

log_info "Deploying contract..."
deploy_output="$(
    cd "$CONTRACT_DIR"
    NETUID="$NETUID" \
    TRUSTEE_ADDRESS="$DEPLOYER_ADDR" \
    MIN_COLLATERAL="$MIN_COLLATERAL" \
    DECISION_TIMEOUT="$DECISION_TIMEOUT" \
    ADMIN_ADDRESS="$DEPLOYER_ADDR" \
    VALIDATOR_HOTKEY="$VALIDATOR_HOTKEY" \
    PRIVATE_KEY="$FAUCET_KEY" \
    RPC_URL="$RPC_URL" \
    TAO_DEPOSITS_ENABLED=true \
    ALPHA_DEPOSITS_ENABLED=true \
    bash ./deploy.sh 2>&1
)"

PROXY="$(printf '%s\n' "$deploy_output" | awk '/^Proxy:/ {print $2}' | tail -n1)"
if [[ -z "$PROXY" ]]; then
    log_error "Failed to parse proxy address from deploy output:"
    echo "$deploy_output"
    exit 1
fi
log_info "  Proxy: $PROXY"
log_success "Contract deployed"

log_info "Verifying deployment..."
timeout_val="$(cast_query "$PROXY" "decisionTimeout()(uint64)")"
assert_eq "$timeout_val" "$DECISION_TIMEOUT" "decisionTimeout() == $DECISION_TIMEOUT"

trustee_val="$(cast_query "$PROXY" "trustee()(address)")"
assert_eq "$trustee_val" "$DEPLOYER_ADDR" "trustee() == deployer"

# ═════════════════════════════════════════════════════════════════════════════
#  T1: Deposit 10 TAO — Node 1
# ═════════════════════════════════════════════════════════════════════════════

section "T1: Deposit 10 TAO -- Node 1"

log_info "Miner deposits 10 TAO on (HOTKEY_1, NODE_ID_1)..."
cast_send "$MINER_KEY" "$PROXY" \
    "deposit(bytes32,bytes16,bytes32,uint256)" \
    "$HOTKEY_1" "$NODE_ID_1" "$ZERO_BYTES32" 0 \
    --value "$TAO_10"

col="$(cast_query "$PROXY" "taoCollaterals(bytes32,bytes16)(uint256)" "$HOTKEY_1" "$NODE_ID_1")"
assert_eq "$col" "$TAO_10" "taoCollaterals(HOTKEY_1, NODE_ID_1) == 10 TAO"

owner="$(cast_query "$PROXY" "nodeToMiner(bytes32,bytes16)(address)" "$HOTKEY_1" "$NODE_ID_1")"
assert_eq "$owner" "$MINER_ADDR" "nodeToMiner(HOTKEY_1, NODE_ID_1) == miner"

# ═════════════════════════════════════════════════════════════════════════════
#  T2: Deposit 5 TAO — Node 2
# ═════════════════════════════════════════════════════════════════════════════

section "T2: Deposit 5 TAO -- Node 2"

log_info "Miner deposits 5 TAO on (HOTKEY_2, NODE_ID_2)..."
cast_send "$MINER_KEY" "$PROXY" \
    "deposit(bytes32,bytes16,bytes32,uint256)" \
    "$HOTKEY_2" "$NODE_ID_2" "$ZERO_BYTES32" 0 \
    --value "$TAO_5"

col="$(cast_query "$PROXY" "taoCollaterals(bytes32,bytes16)(uint256)" "$HOTKEY_2" "$NODE_ID_2")"
assert_eq "$col" "$TAO_5" "taoCollaterals(HOTKEY_2, NODE_ID_2) == 5 TAO"

owner="$(cast_query "$PROXY" "nodeToMiner(bytes32,bytes16)(address)" "$HOTKEY_2" "$NODE_ID_2")"
assert_eq "$owner" "$MINER_ADDR" "nodeToMiner(HOTKEY_2, NODE_ID_2) == miner"

# ═════════════════════════════════════════════════════════════════════════════
#  T3: Early Finalize Reverts
# ═════════════════════════════════════════════════════════════════════════════

section "T3: Early Finalize Reverts"

RECLAIM_ID=$NEXT_RECLAIM_ID
log_info "Miner starts reclaim on (HOTKEY_1, NODE_ID_1) [reclaimId=$RECLAIM_ID]..."
cast_send "$MINER_KEY" "$PROXY" \
    "reclaimCollateral(bytes32,bytes16,string,bytes32)" \
    "$HOTKEY_1" "$NODE_ID_1" "$TEST_URL" "$TEST_SHA"
NEXT_RECLAIM_ID=$((NEXT_RECLAIM_ID + 1))

log_info "Immediately calling finalizeReclaim($RECLAIM_ID) -- should revert..."
assert_revert "finalizeReclaim($RECLAIM_ID) reverts before timeout" \
    cast_send "$MINER_KEY" "$PROXY" "finalizeReclaim(uint256)" "$RECLAIM_ID"

# ═════════════════════════════════════════════════════════════════════════════
#  T4: Wait + Successful Finalize
# ═════════════════════════════════════════════════════════════════════════════

section "T4: Wait + Successful Finalize"

SLEEP_SECS=$((DECISION_TIMEOUT + 3))
log_info "Sleeping ${SLEEP_SECS}s for timeout to expire..."
sleep "$SLEEP_SECS"

bal_before="$(get_balance "$MINER_ADDR")"
log_info "Miner balance before: $(wei_compare fmt "$bal_before")"

log_info "Calling finalizeReclaim($RECLAIM_ID)..."
cast_send "$MINER_KEY" "$PROXY" "finalizeReclaim(uint256)" "$RECLAIM_ID"

bal_after="$(get_balance "$MINER_ADDR")"
log_info "Miner balance after:  $(wei_compare fmt "$bal_after")"

col="$(cast_query "$PROXY" "taoCollaterals(bytes32,bytes16)(uint256)" "$HOTKEY_1" "$NODE_ID_1")"
assert_eq "$col" "0" "taoCollaterals(HOTKEY_1, NODE_ID_1) == 0"

owner="$(cast_query "$PROXY" "nodeToMiner(bytes32,bytes16)(address)" "$HOTKEY_1" "$NODE_ID_1")"
assert_eq "$owner" "$ZERO_ADDR" "nodeToMiner(HOTKEY_1, NODE_ID_1) == address(0)"

assert_gt "$bal_after" "$bal_before" "miner balance increased after finalize"

# ═════════════════════════════════════════════════════════════════════════════
#  T5: Re-deposit 8 TAO — Node 1
# ═════════════════════════════════════════════════════════════════════════════

section "T5: Re-deposit 8 TAO -- Node 1"

log_info "Miner deposits 8 TAO on (HOTKEY_1, NODE_ID_1)..."
cast_send "$MINER_KEY" "$PROXY" \
    "deposit(bytes32,bytes16,bytes32,uint256)" \
    "$HOTKEY_1" "$NODE_ID_1" "$ZERO_BYTES32" 0 \
    --value "$TAO_8"

col="$(cast_query "$PROXY" "taoCollaterals(bytes32,bytes16)(uint256)" "$HOTKEY_1" "$NODE_ID_1")"
assert_eq "$col" "$TAO_8" "taoCollaterals(HOTKEY_1, NODE_ID_1) == 8 TAO"

owner="$(cast_query "$PROXY" "nodeToMiner(bytes32,bytes16)(address)" "$HOTKEY_1" "$NODE_ID_1")"
assert_eq "$owner" "$MINER_ADDR" "nodeToMiner(HOTKEY_1, NODE_ID_1) == miner"

# ═════════════════════════════════════════════════════════════════════════════
#  T6: Partial Slash — 3 TAO on Node 1
# ═════════════════════════════════════════════════════════════════════════════

section "T6: Partial Slash -- 3 TAO on Node 1"

trustee_bal_before="$(get_balance "$DEPLOYER_ADDR")"
log_info "Trustee balance before: $(wei_compare fmt "$trustee_bal_before")"

log_info "Trustee slashes 3 TAO on (HOTKEY_1, NODE_ID_1)..."
cast_send "$FAUCET_KEY" "$PROXY" \
    "slashCollateral(bytes32,bytes16,uint256,uint256,string,bytes32)" \
    "$HOTKEY_1" "$NODE_ID_1" "$TAO_3" 0 "$TEST_URL" "$TEST_SHA"

trustee_bal_after="$(get_balance "$DEPLOYER_ADDR")"
log_info "Trustee balance after:  $(wei_compare fmt "$trustee_bal_after")"

assert_gt "$trustee_bal_after" "$trustee_bal_before" "trustee balance increased after slash"

col="$(cast_query "$PROXY" "taoCollaterals(bytes32,bytes16)(uint256)" "$HOTKEY_1" "$NODE_ID_1")"
assert_eq "$col" "$TAO_5" "taoCollaterals(HOTKEY_1, NODE_ID_1) == 5 TAO (8 - 3)"

owner="$(cast_query "$PROXY" "nodeToMiner(bytes32,bytes16)(address)" "$HOTKEY_1" "$NODE_ID_1")"
assert_eq "$owner" "$MINER_ADDR" "nodeToMiner preserved after partial slash"

# ═════════════════════════════════════════════════════════════════════════════
#  T7: Full Slash — remaining 5 TAO on Node 1
# ═════════════════════════════════════════════════════════════════════════════

section "T7: Full Slash -- 5 TAO on Node 1"

trustee_bal_before="$(get_balance "$DEPLOYER_ADDR")"
log_info "Trustee balance before: $(wei_compare fmt "$trustee_bal_before")"

log_info "Trustee slashes remaining 5 TAO on (HOTKEY_1, NODE_ID_1)..."
cast_send "$FAUCET_KEY" "$PROXY" \
    "slashCollateral(bytes32,bytes16,uint256,uint256,string,bytes32)" \
    "$HOTKEY_1" "$NODE_ID_1" "$TAO_5" 0 "$TEST_URL" "$TEST_SHA"

trustee_bal_after="$(get_balance "$DEPLOYER_ADDR")"
log_info "Trustee balance after:  $(wei_compare fmt "$trustee_bal_after")"

assert_gt "$trustee_bal_after" "$trustee_bal_before" "trustee balance increased after slash"

col="$(cast_query "$PROXY" "taoCollaterals(bytes32,bytes16)(uint256)" "$HOTKEY_1" "$NODE_ID_1")"
assert_eq "$col" "0" "taoCollaterals(HOTKEY_1, NODE_ID_1) == 0"

owner="$(cast_query "$PROXY" "nodeToMiner(bytes32,bytes16)(address)" "$HOTKEY_1" "$NODE_ID_1")"
assert_eq "$owner" "$ZERO_ADDR" "nodeToMiner cleared after full slash"

# ═════════════════════════════════════════════════════════════════════════════
#  T8: Full Slash — 5 TAO on Node 2
# ═════════════════════════════════════════════════════════════════════════════

section "T8: Full Slash -- 5 TAO on Node 2"

trustee_bal_before="$(get_balance "$DEPLOYER_ADDR")"
log_info "Trustee balance before: $(wei_compare fmt "$trustee_bal_before")"

log_info "Trustee slashes 5 TAO on (HOTKEY_2, NODE_ID_2)..."
cast_send "$FAUCET_KEY" "$PROXY" \
    "slashCollateral(bytes32,bytes16,uint256,uint256,string,bytes32)" \
    "$HOTKEY_2" "$NODE_ID_2" "$TAO_5" 0 "$TEST_URL" "$TEST_SHA"

trustee_bal_after="$(get_balance "$DEPLOYER_ADDR")"
log_info "Trustee balance after:  $(wei_compare fmt "$trustee_bal_after")"

assert_gt "$trustee_bal_after" "$trustee_bal_before" "trustee balance increased after slash"

col="$(cast_query "$PROXY" "taoCollaterals(bytes32,bytes16)(uint256)" "$HOTKEY_2" "$NODE_ID_2")"
assert_eq "$col" "0" "taoCollaterals(HOTKEY_2, NODE_ID_2) == 0"

owner="$(cast_query "$PROXY" "nodeToMiner(bytes32,bytes16)(address)" "$HOTKEY_2" "$NODE_ID_2")"
assert_eq "$owner" "$ZERO_ADDR" "nodeToMiner cleared after full slash"

# ═════════════════════════════════════════════════════════════════════════════
#  T9: Combined TAO + Alpha Deposit — Node 3
# ═════════════════════════════════════════════════════════════════════════════

section "T9: Combined TAO + Alpha Deposit -- Node 3"

# Deposit both TAO (5 TAO via msg.value) and alpha (fixed ALPHA_DEPOSIT_RAO).
# The contract's transferAlpha() uses delegatecall so the precompile sees the miner
# as origin, transferring alpha from miner's coldkey to the contract's coldkey.
log_info "Miner deposits 5 TAO + ${ALPHA_DEPOSIT_RAO} alpha RAO on (HOTKEY_3, NODE_ID_3)..."
cast_send "$MINER_KEY" "$PROXY" \
    "deposit(bytes32,bytes16,bytes32,uint256)" \
    "$HOTKEY_3" "$NODE_ID_3" "$VALIDATOR_HOTKEY" "$ALPHA_DEPOSIT_RAO" \
    --value "$TAO_5"

col="$(cast_query "$PROXY" "taoCollaterals(bytes32,bytes16)(uint256)" "$HOTKEY_3" "$NODE_ID_3")"
assert_eq "$col" "$TAO_5" "taoCollaterals(HOTKEY_3, NODE_ID_3) == 5 TAO"

alpha_col="$(cast_query "$PROXY" "alphaCollaterals(bytes32,bytes16)(uint256)" "$HOTKEY_3" "$NODE_ID_3")"
log_info "  alphaCollaterals = ${alpha_col} RAO (requested ${ALPHA_DEPOSIT_RAO})"
assert_gt "$alpha_col" "0" "alphaCollaterals(HOTKEY_3, NODE_ID_3) > 0"
owner="$(cast_query "$PROXY" "nodeToMiner(bytes32,bytes16)(address)" "$HOTKEY_3" "$NODE_ID_3")"
assert_eq "$owner" "$MINER_ADDR" "nodeToMiner(HOTKEY_3, NODE_ID_3) == miner"

# ═════════════════════════════════════════════════════════════════════════════
#  T10: Alpha-only Deposit — Node 4
# ═════════════════════════════════════════════════════════════════════════════

section "T10: Alpha-only Deposit -- Node 4"

# Deposit alpha-only: no --value flag, just alpha amount.
log_info "Miner deposits ${ALPHA_DEPOSIT_RAO} alpha RAO (no TAO) on (HOTKEY_4, NODE_ID_4)..."
cast_send "$MINER_KEY" "$PROXY" \
    "deposit(bytes32,bytes16,bytes32,uint256)" \
    "$HOTKEY_4" "$NODE_ID_4" "$VALIDATOR_HOTKEY" "$ALPHA_DEPOSIT_RAO"

tao_col="$(cast_query "$PROXY" "taoCollaterals(bytes32,bytes16)(uint256)" "$HOTKEY_4" "$NODE_ID_4")"
assert_eq "$tao_col" "0" "taoCollaterals(HOTKEY_4, NODE_ID_4) == 0 (alpha-only deposit)"

alpha_col="$(cast_query "$PROXY" "alphaCollaterals(bytes32,bytes16)(uint256)" "$HOTKEY_4" "$NODE_ID_4")"
log_info "  alphaCollaterals = ${alpha_col} RAO"
assert_gt "$alpha_col" "0" "alphaCollaterals(HOTKEY_4, NODE_ID_4) > 0"

owner="$(cast_query "$PROXY" "nodeToMiner(bytes32,bytes16)(address)" "$HOTKEY_4" "$NODE_ID_4")"
assert_eq "$owner" "$MINER_ADDR" "nodeToMiner(HOTKEY_4, NODE_ID_4) == miner"

# ═════════════════════════════════════════════════════════════════════════════
#  T11: Alpha Reclaim — Early Finalize Reverts
# ═════════════════════════════════════════════════════════════════════════════

section "T11: Alpha Reclaim -- Early Finalize Reverts"

RECLAIM_ID=$NEXT_RECLAIM_ID
log_info "Miner starts reclaim on (HOTKEY_4, NODE_ID_4) [reclaimId=$RECLAIM_ID]..."
cast_send "$MINER_KEY" "$PROXY" \
    "reclaimCollateral(bytes32,bytes16,string,bytes32)" \
    "$HOTKEY_4" "$NODE_ID_4" "$TEST_URL" "$TEST_SHA"
NEXT_RECLAIM_ID=$((NEXT_RECLAIM_ID + 1))

log_info "Immediately calling finalizeReclaim($RECLAIM_ID) -- should revert..."
assert_revert "finalizeReclaim($RECLAIM_ID) reverts before timeout" \
    cast_send "$MINER_KEY" "$PROXY" "finalizeReclaim(uint256)" "$RECLAIM_ID"

# ═════════════════════════════════════════════════════════════════════════════
#  T12: Wait + Successful Alpha Finalize
# ═════════════════════════════════════════════════════════════════════════════

section "T12: Wait + Successful Alpha Finalize"

log_info "Sleeping ${SLEEP_SECS}s for timeout to expire..."
sleep "$SLEEP_SECS"

log_info "Calling finalizeReclaim($RECLAIM_ID)..."
cast_send "$MINER_KEY" "$PROXY" "finalizeReclaim(uint256)" "$RECLAIM_ID"

tao_col="$(cast_query "$PROXY" "taoCollaterals(bytes32,bytes16)(uint256)" "$HOTKEY_4" "$NODE_ID_4")"
assert_eq "$tao_col" "0" "taoCollaterals(HOTKEY_4, NODE_ID_4) == 0"

alpha_col="$(cast_query "$PROXY" "alphaCollaterals(bytes32,bytes16)(uint256)" "$HOTKEY_4" "$NODE_ID_4")"
assert_eq "$alpha_col" "0" "alphaCollaterals(HOTKEY_4, NODE_ID_4) == 0 after finalize"

owner="$(cast_query "$PROXY" "nodeToMiner(bytes32,bytes16)(address)" "$HOTKEY_4" "$NODE_ID_4")"
assert_eq "$owner" "$ZERO_ADDR" "nodeToMiner(HOTKEY_4, NODE_ID_4) == address(0)"

# ═════════════════════════════════════════════════════════════════════════════
#  T13: Alpha Re-deposit + Partial Alpha Slash
# ═════════════════════════════════════════════════════════════════════════════

section "T13: Alpha Re-deposit + Partial Alpha Slash"

# Re-deposit alpha-only on the now-cleared node using fixed amount.
ALPHA_DEPOSIT_RAO_T13="$ALPHA_DEPOSIT_RAO"
log_info "Miner deposits ${ALPHA_DEPOSIT_RAO_T13} alpha RAO on (HOTKEY_4, NODE_ID_4)..."
cast_send "$MINER_KEY" "$PROXY" \
    "deposit(bytes32,bytes16,bytes32,uint256)" \
    "$HOTKEY_4" "$NODE_ID_4" "$VALIDATOR_HOTKEY" "$ALPHA_DEPOSIT_RAO_T13"

alpha_col_full="$(cast_query "$PROXY" "alphaCollaterals(bytes32,bytes16)(uint256)" "$HOTKEY_4" "$NODE_ID_4")"
log_info "  alphaCollaterals after deposit = ${alpha_col_full} RAO"

# Slash half the alpha (slashAmount=0 for TAO, slashAlphaAmount=half).
slash_alpha="$(python3 -c "print(int('${alpha_col_full}') // 2)")"
expected_remaining="$(python3 -c "print(int('${alpha_col_full}') - int('${slash_alpha}'))")"
log_info "Trustee slashes ${slash_alpha} alpha RAO (half)..."
cast_send "$FAUCET_KEY" "$PROXY" \
    "slashCollateral(bytes32,bytes16,uint256,uint256,string,bytes32)" \
    "$HOTKEY_4" "$NODE_ID_4" 0 "$slash_alpha" "$TEST_URL" "$TEST_SHA"

alpha_col_after="$(cast_query "$PROXY" "alphaCollaterals(bytes32,bytes16)(uint256)" "$HOTKEY_4" "$NODE_ID_4")"
log_info "  alphaCollaterals after slash = ${alpha_col_after} RAO"
assert_eq "$alpha_col_after" "$expected_remaining" "alphaCollaterals == expected after partial slash"

owner="$(cast_query "$PROXY" "nodeToMiner(bytes32,bytes16)(address)" "$HOTKEY_4" "$NODE_ID_4")"
assert_eq "$owner" "$MINER_ADDR" "nodeToMiner preserved after partial alpha slash"

# ═════════════════════════════════════════════════════════════════════════════
#  T14: Full Alpha Slash — Node 4
# ═════════════════════════════════════════════════════════════════════════════

section "T14: Full Alpha Slash -- Node 4"

remaining_alpha="$(cast_query "$PROXY" "alphaCollaterals(bytes32,bytes16)(uint256)" "$HOTKEY_4" "$NODE_ID_4")"
log_info "Trustee slashes remaining ${remaining_alpha} alpha RAO..."
cast_send "$FAUCET_KEY" "$PROXY" \
    "slashCollateral(bytes32,bytes16,uint256,uint256,string,bytes32)" \
    "$HOTKEY_4" "$NODE_ID_4" 0 "$remaining_alpha" "$TEST_URL" "$TEST_SHA"

alpha_col="$(cast_query "$PROXY" "alphaCollaterals(bytes32,bytes16)(uint256)" "$HOTKEY_4" "$NODE_ID_4")"
assert_eq "$alpha_col" "0" "alphaCollaterals(HOTKEY_4, NODE_ID_4) == 0"

owner="$(cast_query "$PROXY" "nodeToMiner(bytes32,bytes16)(address)" "$HOTKEY_4" "$NODE_ID_4")"
assert_eq "$owner" "$ZERO_ADDR" "nodeToMiner cleared after full alpha slash"

# ═════════════════════════════════════════════════════════════════════════════
#  Summary
# ═════════════════════════════════════════════════════════════════════════════

echo ""
echo -e "${PURPLE}════════════════════════════════════════════════════════════════${NC}"
if [[ "$TESTS_FAILED" -eq 0 ]]; then
    echo -e "  ${GREEN}Results: ${TESTS_PASSED} passed, ${TESTS_FAILED} failed (${TESTS_TOTAL} total)${NC}"
    echo -e "  ${GREEN}ALL PASSED${NC}"
else
    echo -e "  ${RED}Results: ${TESTS_PASSED} passed, ${TESTS_FAILED} failed (${TESTS_TOTAL} total)${NC}"
    echo -e "  ${RED}SOME TESTS FAILED${NC}"
fi
echo -e "${PURPLE}════════════════════════════════════════════════════════════════${NC}"

if [[ "$TESTS_FAILED" -gt 0 ]]; then
    exit 1
fi
