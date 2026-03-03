#!/usr/bin/env bash
# E2E Validator Collateral Event Parsing & Slashing Test
#
# Sends transactions to the validator's own collateral contract
# (the one it is actively scanning) and verifies the validator's DB state
# after each operation. This tests the full pipeline:
#   on-chain tx → EVM event → validator scan → decode → SQLite persist
#
# Prerequisites:
#   - Localnet running (subtensor + validator container)
#   - cast installed (Foundry)
#   - python3, docker, sqlite3 available
#
# Usage:
#   ./scripts/collateral/e2e-validator-test.sh
set -euo pipefail

# ─── Paths ───────────────────────────────────────────────────────────────────

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
REPO_ROOT="$(cd "${SCRIPT_DIR}/../.." && pwd)"

source "${REPO_ROOT}/scripts/lib/common.sh"

# ─── Load .env.local ─────────────────────────────────────────────────────────

ENV_FILE="${SCRIPT_DIR}/.env.local"
if [[ ! -f "$ENV_FILE" ]]; then
    log_error ".env.local not found at $ENV_FILE — run setup-localnet-env.sh first"
    exit 1
fi
# shellcheck disable=SC1090
source "$ENV_FILE"

# CONTRACT_ADDRESS and PRIVATE_KEY come from .env.local
PROXY="${CONTRACT_ADDRESS:?CONTRACT_ADDRESS not set in .env.local}"
DEPLOYER_KEY="${PRIVATE_KEY:?PRIVATE_KEY not set in .env.local}"

# ─── Constants ───────────────────────────────────────────────────────────────

RPC_URL="http://localhost:9944"
FAUCET_KEY="0x5fb92d6e98884f76de468fa3f6278f8807c48bebc13595d45af5bdc4da702133"
VALIDATOR_CONTAINER="basilica-validator"
VALIDATOR_DB_PATH="/opt/basilica/data/validator.db"
SCAN_WAIT=20  # seconds to wait for validator scan cycle (12s interval + buffer)

NETUID=1
STAKING_PRECOMPILE="0x0000000000000000000000000000000000000805"
ADDRESS_MAPPING="0x000000000000000000000000000000000000080C"

# Use the real localnet validator hotkey so alpha staking works with precompiles.
LOCALNET_WALLETS_DIR="${REPO_ROOT}/scripts/localnet/wallets"
VALIDATOR_HOTKEY_FILE="${LOCALNET_WALLETS_DIR}/validator/hotkeys/defaultpub.txt"
if [[ -f "$VALIDATOR_HOTKEY_FILE" ]]; then
    VALIDATOR_HOTKEY="0x$(awk -F'"' '/"publicKey"/ {print $4; exit}' "$VALIDATOR_HOTKEY_FILE" | sed 's/^0x//')"
else
    VALIDATOR_HOTKEY="0x0000000000000000000000000000000000000000000000000000000000000001"
fi

ZERO_BYTES32="0x0000000000000000000000000000000000000000000000000000000000000000"
ZERO_ADDR="0x0000000000000000000000000000000000000000"

# Generate random hotkey/node_id suffixes so each run uses fresh identifiers.
# This avoids NodeNotOwned errors from previous runs' stale deposits.
RAND_SUFFIX="$(openssl rand -hex 4)"
# bytes32 = 64 hex chars; bytes16 = 32 hex chars
HOTKEY_1="0x0000000000000000000000000000000000000000000000000000e2e1${RAND_SUFFIX}"
HOTKEY_2="0x0000000000000000000000000000000000000000000000000000e2e2${RAND_SUFFIX}"
NODE_ID_1="0x00000000000000000000e2e1${RAND_SUFFIX}"  # bytes16
NODE_ID_2="0x00000000000000000000e2e2${RAND_SUFFIX}"  # bytes16

# Hex with 0x prefix — how the validator stores them in SQLite (RPC snapshot: format!("0x{}", hex::encode(...)))
HEX_HOTKEY_1="${HOTKEY_1}"
HEX_HOTKEY_2="${HOTKEY_2}"
HEX_NODE_ID_1="${NODE_ID_1}"
HEX_NODE_ID_2="${NODE_ID_2}"

# TAO amounts in wei (1 TAO = 1e18 wei)
TAO_1="1000000000000000000"
TAO_2="2000000000000000000"
TAO_5="5000000000000000000"
TAO_10="10000000000000000000"

# RAO amounts for staking precompile (1 TAO = 1e9 RAO)
# minAlphaCollateralIncrease on localnet is 5e9 RAO (5 alpha).
# Stake 10 TAO to ensure the miner receives >= 5 alpha after AMM conversion.
RAO_5_TAO="10000000000"   # 10 TAO in RAO — stake during setup (buffer for AMM rate)
RAO_1_ALPHA="5000000000"  # 5 alpha in RAO — deposit amount (matches minAlphaCollateralIncrease)

TEST_URL="http://localhost:8080/evidence/e2e-test.json"
TEST_SHA="0xd41d8cd98f00b204e9800998ecf8427ed41d8cd98f00b204e9800998ecf8427e"

# ─── Test Counters ───────────────────────────────────────────────────────────

TESTS_PASSED=0
TESTS_FAILED=0
TESTS_TOTAL=0
ORIGINAL_DECISION_TIMEOUT=""

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

assert_not_eq() {
    local actual="${1,,}"   # lowercase
    local expected="${2,,}" # lowercase
    local desc="$3"
    if [[ "$actual" != "$expected" ]]; then
        pass "$desc"
    else
        fail "$desc (got: $actual, expected != $expected)"
    fi
}

# Wraps cast send with standard flags. Extra args (e.g. --value) pass through.
cast_send() {
    local key="$1" contract="$2" sig="$3"
    shift 3
    cast send --rpc-url "$RPC_URL" --private-key "$key" --legacy \
        "$contract" "$sig" "$@" >/dev/null
}

# Like cast_send but returns the JSON receipt (for extracting event logs).
cast_send_json() {
    local key="$1" contract="$2" sig="$3"
    shift 3
    cast send --rpc-url "$RPC_URL" --private-key "$key" --legacy --json \
        "$contract" "$sig" "$@"
}

# Extract the reclaimRequestId from a ReclaimProcessStarted event in a tx receipt.
# ReclaimProcessStarted has reclaimRequestId as topic[1] (first indexed param).
extract_reclaim_id() {
    local receipt_json="$1"
    # ReclaimProcessStarted event signature topic[0]
    local event_sig_hash
    event_sig_hash="$(cast keccak "ReclaimProcessStarted(uint256,bytes32,bytes16,address,uint256,bytes32,uint256,uint64,string,bytes32)")"
    local reclaim_id_hex
    reclaim_id_hex="$(printf '%s' "$receipt_json" | python3 -c "
import json, sys
receipt = json.load(sys.stdin)
sig = '${event_sig_hash}'.lower()
for log in receipt.get('logs', []):
    topics = [t.lower() for t in log.get('topics', [])]
    if len(topics) >= 2 and topics[0] == sig:
        print(int(topics[1], 16))
        sys.exit(0)
print('__NOT_FOUND__')
")"
    echo "$reclaim_id_hex"
}

# Wraps cast call. Returns decoded value to stdout.
cast_query() {
    local contract="$1" sig="$2"
    shift 2
    cast call --rpc-url "$RPC_URL" "$contract" "$sig" "$@" | awk '{print $1}'
}

get_balance() {
    cast balance --rpc-url "$RPC_URL" "$1"
}

# Query the validator's SQLite DB by copying it to a temp dir on the host.
# The container doesn't have sqlite3 installed, so we copy the DB + WAL files
# and query locally.
DB_TEMP_DIR=""
db_query() {
    if [[ -z "$DB_TEMP_DIR" ]]; then
        DB_TEMP_DIR="$(mktemp -d)"
    fi
    # Copy all DB files (db + WAL + SHM) for a consistent read
    docker cp "${VALIDATOR_CONTAINER}:${VALIDATOR_DB_PATH}" "${DB_TEMP_DIR}/validator.db" >/dev/null 2>&1
    docker cp "${VALIDATOR_CONTAINER}:${VALIDATOR_DB_PATH}-wal" "${DB_TEMP_DIR}/validator.db-wal" >/dev/null 2>&1 || true
    docker cp "${VALIDATOR_CONTAINER}:${VALIDATOR_DB_PATH}-shm" "${DB_TEMP_DIR}/validator.db-shm" >/dev/null 2>&1 || true
    sqlite3 -noheader -batch "${DB_TEMP_DIR}/validator.db" "$1" | tr -d '[:space:]'
}

# Wait for the validator scan cycle to pick up new events.
wait_for_scan() {
    local msg="${1:-validator scan cycle}"
    echo -n "  Waiting ${SCAN_WAIT}s for ${msg}"
    for ((i=0; i<SCAN_WAIT; i++)); do
        sleep 1
        echo -n "."
    done
    echo " done"
}

# Assert a DB query returns an expected value.
assert_db_eq() {
    local query="$1"
    local expected="${2,,}"  # lowercase
    local desc="$3"
    local actual
    actual="$(db_query "$query" 2>/dev/null || echo "__DB_ERROR__")"
    actual="${actual,,}"     # lowercase
    if [[ "$actual" == "$expected" ]]; then
        pass "$desc"
    else
        fail "$desc (got: $actual, expected: $expected)"
    fi
}

# Cleanup: restore original decision timeout + remove temp dir
cleanup() {
    if [[ -n "$ORIGINAL_DECISION_TIMEOUT" ]]; then
        log_info "Restoring decision timeout to ${ORIGINAL_DECISION_TIMEOUT}..."
        cast_send "$DEPLOYER_KEY" "$PROXY" "updateDecisionTimeout(uint64)" "$ORIGINAL_DECISION_TIMEOUT" 2>/dev/null || true
    fi
    if [[ -n "$DB_TEMP_DIR" && -d "$DB_TEMP_DIR" ]]; then
        rm -rf "$DB_TEMP_DIR"
    fi
}
trap cleanup EXIT

# ═════════════════════════════════════════════════════════════════════════════
#  Prerequisites
# ═════════════════════════════════════════════════════════════════════════════

banner "E2E Validator Collateral Event Parsing Test"

section "Prerequisites"

for cmd in cast python3 docker sqlite3; do
    if command -v "$cmd" >/dev/null 2>&1; then
        log_info "$cmd ... found"
    else
        log_error "$cmd not found -- install it and retry"
        exit 1
    fi
done

if ! docker ps --format '{{.Names}}' | grep -q "^${VALIDATOR_CONTAINER}$"; then
    log_error "Validator container '${VALIDATOR_CONTAINER}' not running -- start localnet first"
    exit 1
fi
log_info "Validator container ... running"

if block_num="$(cast block-number --rpc-url "$RPC_URL" 2>/dev/null)"; then
    log_info "RPC reachable at $RPC_URL (block #${block_num})"
else
    log_error "RPC not reachable at $RPC_URL -- start localnet first"
    exit 1
fi

# Verify we can query the validator DB (copies DB files to host temp dir)
if db_query "SELECT 1" >/dev/null 2>&1; then
    log_info "Validator DB ... accessible (via docker cp + host sqlite3)"
else
    log_error "Cannot access validator DB -- check docker cp and sqlite3"
    exit 1
fi

log_info "Contract: $PROXY"

# Verify the contract actually has code deployed
CONTRACT_CODE="$(cast code --rpc-url "$RPC_URL" "$PROXY" 2>/dev/null || echo "")"
if [[ -z "$CONTRACT_CODE" || "$CONTRACT_CODE" == "0x" ]]; then
    log_error "Contract $PROXY has no code on-chain (chain was likely reset)"
    log_error "Redeploy: rm scripts/collateral/.env.local && ./scripts/localnet/start.sh validator"
    exit 1
fi
log_info "Contract code ... verified on-chain"

# ═════════════════════════════════════════════════════════════════════════════
#  Setup
# ═════════════════════════════════════════════════════════════════════════════

section "Setup"

DEPLOYER_ADDR="$(cast wallet address --private-key "$DEPLOYER_KEY")"
log_info "Deployer/Trustee/Admin: $DEPLOYER_ADDR"

# Read and save current decision timeout for restore
ORIGINAL_DECISION_TIMEOUT="$(cast_query "$PROXY" "decisionTimeout()(uint64)")"
log_info "Current decision timeout: ${ORIGINAL_DECISION_TIMEOUT}s"

# Shorten decision timeout for fast test cycles
DECISION_TIMEOUT=5
log_info "Setting decision timeout to ${DECISION_TIMEOUT}s..."
cast_send "$DEPLOYER_KEY" "$PROXY" "updateDecisionTimeout(uint64)" "$DECISION_TIMEOUT"
log_success "Decision timeout updated"

# Create fresh miner wallet
log_info "Creating miner wallet..."
wallet_json="$(cast wallet new --json)"
MINER_KEY="$(printf '%s\n' "$wallet_json" | awk -F'"' '/"private_key"/ {print $4; exit}')"
MINER_ADDR="$(printf '%s\n' "$wallet_json" | awk -F'"' '/"address"/ {print $4; exit}')"
[[ -n "$MINER_KEY" ]]  || { log_error "Failed to parse miner private key"; exit 1; }
[[ -n "$MINER_ADDR" ]] || { log_error "Failed to parse miner address"; exit 1; }
log_info "Miner address: $MINER_ADDR"

# Miner address as stored in DB (lowercase with 0x prefix, from address_to_string)
MINER_ADDR_DB="${MINER_ADDR,,}"

MINER_FUND="20000000000000000000"  # 20 TAO in wei
log_info "Funding miner with 20 TAO from faucet..."
cast send --rpc-url "$RPC_URL" --private-key "$FAUCET_KEY" --legacy \
    "$MINER_ADDR" --value "$MINER_FUND" >/dev/null
log_success "Miner funded ($(wei_compare fmt "$MINER_FUND"))"

log_info "Staking 5 TAO as alpha for later alpha deposit tests..."
cast_send "$MINER_KEY" "$STAKING_PRECOMPILE" \
    "addStake(bytes32,uint256,uint256)" \
    "$VALIDATOR_HOTKEY" "$RAO_5_TAO" "$NETUID"
log_success "Miner staked 5 TAO as alpha"

# Record last_scanned_block_number before our transactions
INITIAL_SCAN_BLOCK="$(db_query "SELECT last_scanned_block_number FROM collateral_scan_status WHERE id = 1")"
log_info "Initial last_scanned_block_number: $INITIAL_SCAN_BLOCK"

# ═════════════════════════════════════════════════════════════════════════════
#  T1: TAO Deposit (1 TAO) → Verify DB
# ═════════════════════════════════════════════════════════════════════════════

section "T1: TAO Deposit (1 TAO) -- Node 1"

log_info "Miner deposits 1 TAO on (HOTKEY_1, NODE_ID_1)..."
cast_send "$MINER_KEY" "$PROXY" \
    "deposit(bytes32,bytes16,bytes32,uint256)" \
    "$HOTKEY_1" "$NODE_ID_1" "$ZERO_BYTES32" 0 \
    --value "$TAO_1"

wait_for_scan "T1 deposit"

assert_db_eq \
    "SELECT tao_collateral FROM collateral_status WHERE hotkey = '${HEX_HOTKEY_1}' AND node_id = '${HEX_NODE_ID_1}'" \
    "$TAO_1" \
    "DB: tao_collateral == 1 TAO for node 1"

assert_db_eq \
    "SELECT miner FROM collateral_status WHERE hotkey = '${HEX_HOTKEY_1}' AND node_id = '${HEX_NODE_ID_1}'" \
    "$MINER_ADDR_DB" \
    "DB: miner matches depositor for node 1"

# Verify scan block advanced
CURRENT_SCAN_BLOCK="$(db_query "SELECT last_scanned_block_number FROM collateral_scan_status WHERE id = 1")"
if [[ "$CURRENT_SCAN_BLOCK" -gt "$INITIAL_SCAN_BLOCK" ]]; then
    pass "last_scanned_block_number advanced ($INITIAL_SCAN_BLOCK → $CURRENT_SCAN_BLOCK)"
else
    fail "last_scanned_block_number did not advance (still $CURRENT_SCAN_BLOCK)"
fi

# ═════════════════════════════════════════════════════════════════════════════
#  T2: Combined TAO + Alpha Deposit → Verify DB
# ═════════════════════════════════════════════════════════════════════════════

section "T2: Combined TAO + Alpha Deposit -- Node 2"

log_info "Miner deposits 1 TAO + ${RAO_1_ALPHA} alpha RAO on (HOTKEY_2, NODE_ID_2)..."
cast_send "$MINER_KEY" "$PROXY" \
    "deposit(bytes32,bytes16,bytes32,uint256)" \
    "$HOTKEY_2" "$NODE_ID_2" "$VALIDATOR_HOTKEY" "$RAO_1_ALPHA" \
    --value "$TAO_1"

wait_for_scan "T2 combined deposit"

assert_db_eq \
    "SELECT tao_collateral FROM collateral_status WHERE hotkey = '${HEX_HOTKEY_2}' AND node_id = '${HEX_NODE_ID_2}'" \
    "$TAO_1" \
    "DB: tao_collateral == 1 TAO for node 2"

ALPHA_COL="$(db_query "SELECT alpha_collateral FROM collateral_status WHERE hotkey = '${HEX_HOTKEY_2}' AND node_id = '${HEX_NODE_ID_2}'")"
if wei_compare gt "$ALPHA_COL" "0"; then
    pass "DB: alpha_collateral > 0 for node 2 (got: $ALPHA_COL)"
else
    fail "DB: alpha_collateral > 0 for node 2 (got: $ALPHA_COL)"
fi

# ═════════════════════════════════════════════════════════════════════════════
#  T3: Reclaim Start → Verify Pending State
# ═════════════════════════════════════════════════════════════════════════════

section "T3: Reclaim Start -- Node 1"

log_info "Miner starts reclaim on (HOTKEY_1, NODE_ID_1)..."
RECLAIM_RECEIPT="$(cast_send_json "$MINER_KEY" "$PROXY" \
    "reclaimCollateral(bytes32,bytes16,string,bytes32)" \
    "$HOTKEY_1" "$NODE_ID_1" "$TEST_URL" "$TEST_SHA")"
RECLAIM_ID="$(extract_reclaim_id "$RECLAIM_RECEIPT")"
if [[ "$RECLAIM_ID" == "__NOT_FOUND__" ]]; then
    log_error "Could not extract reclaimRequestId from tx receipt"
    exit 1
fi
log_info "Extracted reclaimRequestId=$RECLAIM_ID from event log"

wait_for_scan "T3 reclaim start"

# RPC snapshot approach: pending reclaim state is tracked in collateral_reclaims (not collateral_status.pending_tao_reclaim)
# Check collateral_reclaims row exists
RECLAIM_ROW="$(db_query "SELECT reclaim_request_id FROM collateral_reclaims WHERE hotkey = '${HEX_HOTKEY_1}' AND node_id = '${HEX_NODE_ID_1}'")"
assert_eq "$RECLAIM_ROW" "$RECLAIM_ID" "DB: collateral_reclaims row exists with correct reclaim_request_id"

# ═════════════════════════════════════════════════════════════════════════════
#  T4: Wait + Finalize Reclaim → Verify Cleared
# ═════════════════════════════════════════════════════════════════════════════

section "T4: Wait + Finalize Reclaim -- Node 1"

SLEEP_SECS=$((DECISION_TIMEOUT + 3))
log_info "Sleeping ${SLEEP_SECS}s for decision timeout to expire..."
sleep "$SLEEP_SECS"

log_info "Calling finalizeReclaim($RECLAIM_ID)..."
cast_send "$MINER_KEY" "$PROXY" "finalizeReclaim(uint256)" "$RECLAIM_ID"

wait_for_scan "T4 finalize reclaim"

# RPC snapshot approach: nodes with zero collateral are removed from activeNodeKeys[] on-chain
# and subsequently deleted from collateral_status DB (not zeroed).
assert_db_eq \
    "SELECT COUNT(*) FROM collateral_status WHERE hotkey = '${HEX_HOTKEY_1}' AND node_id = '${HEX_NODE_ID_1}'" \
    "0" \
    "DB: collateral_status row deleted after full reclaim for node 1 (new behavior: no zero-balance rows)"

# Reclaim record should be deleted
RECLAIM_COUNT="$(db_query "SELECT COUNT(*) FROM collateral_reclaims WHERE reclaim_request_id = '$RECLAIM_ID'")"
assert_eq "$RECLAIM_COUNT" "0" "DB: collateral_reclaims row deleted after finalize"

# ═════════════════════════════════════════════════════════════════════════════
#  T5: Partial Slash → Verify DB
# ═════════════════════════════════════════════════════════════════════════════

section "T5: Partial Slash -- Node 1"

log_info "Miner deposits 2 TAO on (HOTKEY_1, NODE_ID_1)..."
cast_send "$MINER_KEY" "$PROXY" \
    "deposit(bytes32,bytes16,bytes32,uint256)" \
    "$HOTKEY_1" "$NODE_ID_1" "$ZERO_BYTES32" 0 \
    --value "$TAO_2"

wait_for_scan "T5 deposit"

assert_db_eq \
    "SELECT tao_collateral FROM collateral_status WHERE hotkey = '${HEX_HOTKEY_1}' AND node_id = '${HEX_NODE_ID_1}'" \
    "$TAO_2" \
    "DB: tao_collateral == 2 TAO after re-deposit"

log_info "Trustee slashes 1 TAO on (HOTKEY_1, NODE_ID_1)..."
cast_send "$DEPLOYER_KEY" "$PROXY" \
    "slashCollateral(bytes32,bytes16,uint256,uint256,string,bytes32)" \
    "$HOTKEY_1" "$NODE_ID_1" "$TAO_1" 0 "$TEST_URL" "$TEST_SHA"

wait_for_scan "T5 partial slash"

assert_db_eq \
    "SELECT tao_collateral FROM collateral_status WHERE hotkey = '${HEX_HOTKEY_1}' AND node_id = '${HEX_NODE_ID_1}'" \
    "$TAO_1" \
    "DB: tao_collateral == 1 TAO after partial slash (2 - 1)"

# Miner still set after partial slash
assert_db_eq \
    "SELECT miner FROM collateral_status WHERE hotkey = '${HEX_HOTKEY_1}' AND node_id = '${HEX_NODE_ID_1}'" \
    "$MINER_ADDR_DB" \
    "DB: miner preserved after partial slash"

# ═════════════════════════════════════════════════════════════════════════════
#  T6: Full Slash → Verify Ownership Cleared
# ═════════════════════════════════════════════════════════════════════════════

section "T6: Full Slash -- Node 1"

log_info "Trustee slashes remaining 1 TAO on (HOTKEY_1, NODE_ID_1)..."
cast_send "$DEPLOYER_KEY" "$PROXY" \
    "slashCollateral(bytes32,bytes16,uint256,uint256,string,bytes32)" \
    "$HOTKEY_1" "$NODE_ID_1" "$TAO_1" 0 "$TEST_URL" "$TEST_SHA"

wait_for_scan "T6 full slash"

# RPC snapshot approach: nodes with zero collateral are removed from activeNodeKeys[] on-chain
# and subsequently deleted from collateral_status DB (not zeroed).
assert_db_eq \
    "SELECT COUNT(*) FROM collateral_status WHERE hotkey = '${HEX_HOTKEY_1}' AND node_id = '${HEX_NODE_ID_1}'" \
    "0" \
    "DB: collateral_status row deleted after full slash (new behavior: no zero-balance rows)"

# ═════════════════════════════════════════════════════════════════════════════
#  T7: Validator Logs Verification
# ═════════════════════════════════════════════════════════════════════════════

section "T7: Validator Logs Verification"

# grep -q with pipefail causes SIGPIPE (exit 141) when grep exits early.
# Use a helper that disables pipefail for the pipeline.
log_grep() {
    (set +o pipefail; docker logs "$VALIDATOR_CONTAINER" 2>&1 | grep -q "$1")
}

if log_grep "Starting collateral sync loop (RPC-based)"; then
    pass "Validator log: 'Starting collateral sync loop (RPC-based)' found"
else
    fail "Validator log: 'Starting collateral sync loop (RPC-based)' not found"
fi

if log_grep "Syncing collateral state from contract"; then
    pass "Validator log: 'Syncing collateral state from contract' messages found"
else
    fail "Validator log: 'Syncing collateral state from contract' messages not found"
fi

# Only flag persistent/real sync failures — exclude transient "database is locked"
# errors which our db_query (docker cp) can cause.
if (set +o pipefail; docker logs "$VALIDATOR_CONTAINER" 2>&1 \
    | grep "Collateral sync failed" \
    | grep -v "database is locked" \
    | grep -q .); then
    fail "Validator log: 'Collateral sync failed' found (non-transient)"
else
    pass "Validator log: no persistent 'Collateral sync failed' errors"
fi

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
