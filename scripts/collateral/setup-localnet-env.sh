#!/usr/bin/env bash
# Deploy collateral contract on localnet and write scripts/collateral/.env.local
set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
REPO_ROOT="$(cd "${SCRIPT_DIR}/../.." && pwd)"
CONTRACT_DIR="${REPO_ROOT}/crates/collateral-contract"
LOCALNET_WALLETS_DIR="${REPO_ROOT}/scripts/localnet/wallets"

ENV_FILE="${SCRIPT_DIR}/.env.local"
RPC_URL="http://localhost:9944"

PRIVATE_KEY="${PRIVATE_KEY:-}"
HOTKEY="${HOTKEY:-}"
ALPHA_HOTKEY="${ALPHA_HOTKEY:-}"
NODE_ID="${NODE_ID:-6339ba4f-60f9-45c2-9d95-2b755bb57ca6}"
ALPHA_AMOUNT_WEI="${ALPHA_AMOUNT_WEI:-1000000000000000000}"
NETUID="${NETUID:-1}"
MIN_COLLATERAL="${MIN_COLLATERAL:-1}"
DECISION_TIMEOUT="${DECISION_TIMEOUT:-3600}"
URL="${URL:-http://localhost:8080/evidence/localnet-reclaim-1.json}"
URL_CONTENT_SHA256="${URL_CONTENT_SHA256:-d41d8cd98f00b204e9800998ecf8427ed41d8cd98f00b204e9800998ecf8427e}"

VALIDATOR_HOTKEY_FILE="${VALIDATOR_HOTKEY_FILE:-${LOCALNET_WALLETS_DIR}/validator/hotkeys/defaultpub.txt}"
DEPLOYER_WALLET_FILE="${DEPLOYER_WALLET_FILE:-${LOCALNET_WALLETS_DIR}/contract_deployer_evm.env}"

# Localnet dev faucet (Alith): used only to fund generated deployer wallets.
FAUCET_PRIVATE_KEY="${FAUCET_PRIVATE_KEY:-0x5fb92d6e98884f76de468fa3f6278f8807c48bebc13595d45af5bdc4da702133}"
MIN_DEPLOYER_BALANCE_WEI="${MIN_DEPLOYER_BALANCE_WEI:-1000000000000000000}"
FAUCET_FUND_AMOUNT_WEI="${FAUCET_FUND_AMOUNT_WEI:-5000000000000000000}"
SKIP_FUNDING=false

deployer_address=""
wallet_file_used=""
wallet_file_created=false

die() {
  echo "ERROR: $*" >&2
  exit 1
}

usage() {
  cat <<USAGE
Usage: $(basename "$0") [options]

Deploy localnet collateral contract and generate scripts/collateral/.env.local.

Options:
  --env <path>                        Output env file path (default: scripts/collateral/.env.local)
  --rpc-url <url>                     RPC URL (default: http://localhost:9944)
  --private-key <hex>                 Explicit deployer key (skip wallet auto-create/load)
  --deployer-wallet <path>            Deployer wallet file (default: scripts/localnet/wallets/contract_deployer_evm.env)
  --validator-hotkey-file <path>      Validator hotkey JSON file (default: scripts/localnet/wallets/validator/hotkeys/defaultpub.txt)
  --faucet-private-key <hex>          Funder key for generated wallet top-up (default: localnet Alith key)
  --min-deployer-balance-wei <uint>   Min wallet balance before top-up (default: 1000000000000000000)
  --faucet-fund-amount-wei <uint>     Top-up amount when balance is below minimum (default: 5000000000000000000)
  --skip-funding                      Do not auto-fund deployer wallet
  --hotkey <64-hex>                   HOTKEY for collateral ops (defaults from validator wallet file)
  --alpha-hotkey <64-hex>             ALPHA_HOTKEY for deposit ops (default: HOTKEY)
  --node-id <uuid>                    Node UUID for ops scripts
  --alpha-amount-wei <uint>           Alpha amount for deposits
  --netuid <uint>                     Contract netuid (default: 1)
  --min-collateral <uint>             Minimum collateral increase
  --decision-timeout <secs>           Reclaim decision timeout seconds
  -h, --help                          Show this help

Example:
  scripts/collateral/setup-localnet-env.sh
USAGE
}

require_cmd() {
  command -v "$1" >/dev/null 2>&1 || die "required command not found: $1"
}

normalize_private_key() {
  local key="$1"
  [[ -n "$key" ]] || die "private key cannot be empty"
  key="${key#0x}"
  key="${key,,}"
  [[ "$key" =~ ^[0-9a-f]{64}$ ]] || die "private key must be 64 hex chars"
  echo "0x${key}"
}

normalize_hotkey() {
  local hotkey="$1"
  hotkey="${hotkey#0x}"
  hotkey="${hotkey,,}"
  [[ "$hotkey" =~ ^[0-9a-f]{64}$ ]] || die "hotkey must be 64 hex chars"
  echo "$hotkey"
}

require_uint() {
  local name="$1"
  local value="$2"
  [[ "$value" =~ ^[0-9]+$ ]] || die "${name} must be an unsigned integer"
}

uint_lt() {
  local left="${1#0}"
  local right="${2#0}"

  [[ -z "$left" ]] && left="0"
  [[ -z "$right" ]] && right="0"

  if (( ${#left} < ${#right} )); then
    return 0
  fi

  if (( ${#left} > ${#right} )); then
    return 1
  fi

  [[ "$left" < "$right" ]]
}

read_wallet_var() {
  local file="$1"
  local key="$2"
  awk -F= -v target="$key" '
    $1 == target {
      gsub(/^[[:space:]]+|[[:space:]]+$/, "", $2);
      print $2;
      exit
    }
  ' "$file"
}

load_or_create_deployer_wallet() {
  local wallet_json
  local current_umask

  if [[ -n "$PRIVATE_KEY" ]]; then
    PRIVATE_KEY="$(normalize_private_key "$PRIVATE_KEY")"
    deployer_address="$(cast wallet address --private-key "$PRIVATE_KEY")"
    [[ -n "$deployer_address" ]] || die "failed to derive deployer address from --private-key"
    return
  fi

  if [[ -f "$DEPLOYER_WALLET_FILE" ]]; then
    PRIVATE_KEY="$(read_wallet_var "$DEPLOYER_WALLET_FILE" "DEPLOYER_PRIVATE_KEY")"
    deployer_address="$(read_wallet_var "$DEPLOYER_WALLET_FILE" "DEPLOYER_ADDRESS")"
    [[ -n "$PRIVATE_KEY" ]] || die "DEPLOYER_PRIVATE_KEY missing in ${DEPLOYER_WALLET_FILE}"
    PRIVATE_KEY="$(normalize_private_key "$PRIVATE_KEY")"
    if [[ -z "$deployer_address" ]]; then
      deployer_address="$(cast wallet address --private-key "$PRIVATE_KEY")"
    fi
    wallet_file_used="$DEPLOYER_WALLET_FILE"
    return
  fi

  mkdir -p "$(dirname "$DEPLOYER_WALLET_FILE")"

  wallet_json="$(cast wallet new --json)"
  PRIVATE_KEY="$(printf '%s\n' "$wallet_json" | awk -F'"' '/"private_key"/ {print $4; exit}')"
  deployer_address="$(printf '%s\n' "$wallet_json" | awk -F'"' '/"address"/ {print $4; exit}')"

  [[ -n "$PRIVATE_KEY" ]] || die "failed to parse private key from cast wallet output"
  [[ -n "$deployer_address" ]] || die "failed to parse address from cast wallet output"

  PRIVATE_KEY="$(normalize_private_key "$PRIVATE_KEY")"
  wallet_file_used="$DEPLOYER_WALLET_FILE"

  current_umask="$(umask)"
  umask 077
  cat > "$DEPLOYER_WALLET_FILE" <<EOF
# Generated by scripts/collateral/setup-localnet-env.sh on $(date -u +"%Y-%m-%dT%H:%M:%SZ")
DEPLOYER_ADDRESS=$deployer_address
DEPLOYER_PRIVATE_KEY=$PRIVATE_KEY
EOF
  umask "$current_umask"
  wallet_file_created=true
}

ensure_deployer_funded() {
  local faucet_address
  local current_balance

  if [[ "$SKIP_FUNDING" == true ]]; then
    echo "Skipping deployer wallet funding (--skip-funding)."
    return
  fi

  FAUCET_PRIVATE_KEY="$(normalize_private_key "$FAUCET_PRIVATE_KEY")"
  require_uint "MIN_DEPLOYER_BALANCE_WEI" "$MIN_DEPLOYER_BALANCE_WEI"
  require_uint "FAUCET_FUND_AMOUNT_WEI" "$FAUCET_FUND_AMOUNT_WEI"

  faucet_address="$(cast wallet address --private-key "$FAUCET_PRIVATE_KEY")"
  [[ -n "$faucet_address" ]] || die "failed to derive faucet address from faucet private key"

  if [[ "${faucet_address,,}" == "${deployer_address,,}" ]]; then
    echo "Deployer wallet is faucet wallet; skipping top-up."
    return
  fi

  current_balance="$(cast balance --rpc-url "$RPC_URL" "$deployer_address")"
  [[ "$current_balance" =~ ^[0-9]+$ ]] || die "failed to read deployer balance"

  if uint_lt "$current_balance" "$MIN_DEPLOYER_BALANCE_WEI"; then
    echo "Funding deployer wallet (${deployer_address}) with ${FAUCET_FUND_AMOUNT_WEI} wei..."
    cast send \
      --rpc-url "$RPC_URL" \
      --private-key "$FAUCET_PRIVATE_KEY" \
      --legacy \
      "$deployer_address" \
      --value "$FAUCET_FUND_AMOUNT_WEI" >/dev/null
    current_balance="$(cast balance --rpc-url "$RPC_URL" "$deployer_address")"
    echo "Deployer wallet balance is now ${current_balance} wei."
  else
    echo "Deployer wallet already funded (${current_balance} wei)."
  fi
}

load_hotkey_defaults() {
  if [[ -n "$HOTKEY" ]]; then
    HOTKEY="$(normalize_hotkey "$HOTKEY")"
  else
    [[ -f "$VALIDATOR_HOTKEY_FILE" ]] || die "validator hotkey file not found: ${VALIDATOR_HOTKEY_FILE}"
    HOTKEY="$(awk -F'"' '/"publicKey"/ {print $4; exit}' "$VALIDATOR_HOTKEY_FILE")"
    [[ -n "$HOTKEY" ]] || die "failed to read publicKey from ${VALIDATOR_HOTKEY_FILE}"
    HOTKEY="$(normalize_hotkey "$HOTKEY")"
  fi

  if [[ -z "$ALPHA_HOTKEY" ]]; then
    ALPHA_HOTKEY="$HOTKEY"
  else
    ALPHA_HOTKEY="$(normalize_hotkey "$ALPHA_HOTKEY")"
  fi
}

while [[ $# -gt 0 ]]; do
  case "$1" in
    --env)
      shift
      [[ $# -gt 0 ]] || die "--env requires a value"
      ENV_FILE="$1"
      ;;
    --rpc-url)
      shift
      [[ $# -gt 0 ]] || die "--rpc-url requires a value"
      RPC_URL="$1"
      ;;
    --private-key)
      shift
      [[ $# -gt 0 ]] || die "--private-key requires a value"
      PRIVATE_KEY="$1"
      ;;
    --deployer-wallet)
      shift
      [[ $# -gt 0 ]] || die "--deployer-wallet requires a value"
      DEPLOYER_WALLET_FILE="$1"
      ;;
    --validator-hotkey-file)
      shift
      [[ $# -gt 0 ]] || die "--validator-hotkey-file requires a value"
      VALIDATOR_HOTKEY_FILE="$1"
      ;;
    --faucet-private-key)
      shift
      [[ $# -gt 0 ]] || die "--faucet-private-key requires a value"
      FAUCET_PRIVATE_KEY="$1"
      ;;
    --min-deployer-balance-wei)
      shift
      [[ $# -gt 0 ]] || die "--min-deployer-balance-wei requires a value"
      MIN_DEPLOYER_BALANCE_WEI="$1"
      ;;
    --faucet-fund-amount-wei)
      shift
      [[ $# -gt 0 ]] || die "--faucet-fund-amount-wei requires a value"
      FAUCET_FUND_AMOUNT_WEI="$1"
      ;;
    --skip-funding)
      SKIP_FUNDING=true
      ;;
    --hotkey)
      shift
      [[ $# -gt 0 ]] || die "--hotkey requires a value"
      HOTKEY="$1"
      ;;
    --alpha-hotkey)
      shift
      [[ $# -gt 0 ]] || die "--alpha-hotkey requires a value"
      ALPHA_HOTKEY="$1"
      ;;
    --node-id)
      shift
      [[ $# -gt 0 ]] || die "--node-id requires a value"
      NODE_ID="$1"
      ;;
    --alpha-amount-wei)
      shift
      [[ $# -gt 0 ]] || die "--alpha-amount-wei requires a value"
      ALPHA_AMOUNT_WEI="$1"
      ;;
    --netuid)
      shift
      [[ $# -gt 0 ]] || die "--netuid requires a value"
      NETUID="$1"
      ;;
    --min-collateral)
      shift
      [[ $# -gt 0 ]] || die "--min-collateral requires a value"
      MIN_COLLATERAL="$1"
      ;;
    --decision-timeout)
      shift
      [[ $# -gt 0 ]] || die "--decision-timeout requires a value"
      DECISION_TIMEOUT="$1"
      ;;
    -h|--help)
      usage
      exit 0
      ;;
    *)
      die "unknown argument: $1"
      ;;
  esac
  shift
done

require_cmd forge
require_cmd cast
require_cmd awk

[[ -d "$CONTRACT_DIR" ]] || die "contract directory not found: $CONTRACT_DIR"

# Validate localnet RPC early (user requested separate network startup step).
if ! cast block-number --rpc-url "$RPC_URL" >/dev/null 2>&1; then
  die "RPC is not reachable at ${RPC_URL}. Start localnet first: (cd scripts/localnet && ./start.sh network)"
fi

load_hotkey_defaults
load_or_create_deployer_wallet
ensure_deployer_funded

echo "Deploying collateral contract (atomic proxy init)..."
deploy_output="$(
  cd "$CONTRACT_DIR"
  NETUID="$NETUID" \
  TRUSTEE_ADDRESS="$deployer_address" \
  MIN_COLLATERAL="$MIN_COLLATERAL" \
  DECISION_TIMEOUT="$DECISION_TIMEOUT" \
  ADMIN_ADDRESS="$deployer_address" \
  VALIDATOR_HOTKEY="$HOTKEY" \
  PRIVATE_KEY="$PRIVATE_KEY" \
  RPC_URL="$RPC_URL" \
  bash ./deploy.sh
)"
echo "$deploy_output"

implementation_address="$(printf '%s\n' "$deploy_output" | awk '/^Implementation:/ {print $2}' | tail -n1)"
proxy_address="$(printf '%s\n' "$deploy_output" | awk '/^Proxy:/ {print $2}' | tail -n1)"

[[ -n "$implementation_address" ]] || die "failed to parse implementation address from deploy output"
[[ -n "$proxy_address" ]] || die "failed to parse proxy address from deploy output"

mkdir -p "$(dirname "$ENV_FILE")"

if [[ -f "$ENV_FILE" ]]; then
  backup_file="${ENV_FILE}.bak.$(date +%Y%m%d_%H%M%S)"
  cp "$ENV_FILE" "$backup_file"
  echo "Backed up existing env file to: $backup_file"
fi

cat > "$ENV_FILE" <<EOF
# Generated by scripts/collateral/setup-localnet-env.sh on $(date -u +"%Y-%m-%dT%H:%M:%SZ")

NETWORK=local
CONTRACT_ADDRESS=$proxy_address

# Local signer key for collateral tx scripts (localnet only)
PRIVATE_KEY=$PRIVATE_KEY

HOTKEY=$HOTKEY
NODE_ID=$NODE_ID
ALPHA_HOTKEY=$ALPHA_HOTKEY
ALPHA_AMOUNT_WEI=$ALPHA_AMOUNT_WEI

URL=$URL
URL_CONTENT_SHA256=$URL_CONTENT_SHA256

RECLAIM_REQUEST_ID=

EVENTS_FORMAT=pretty
EOF

echo
echo "Wrote env file: $ENV_FILE"
echo "Implementation: $implementation_address"
echo "Proxy:          $proxy_address"
echo "Deployer:       $deployer_address"
if [[ -n "$wallet_file_used" ]]; then
  if [[ "$wallet_file_created" == true ]]; then
    echo "Deployer wallet file created: $wallet_file_used"
  else
    echo "Deployer wallet file reused:  $wallet_file_used"
  fi
fi
echo
echo "Next:"
echo "  source \"$ENV_FILE\""
echo "  cc() { collateral-cli --network \"\$NETWORK\" --contract-address \"\$CONTRACT_ADDRESS\" \"\$@\"; }"
echo "  cc query trustee"
echo "  cc tx deposit --private-key \"\$PRIVATE_KEY\" --hotkey \"\$HOTKEY\" --node-id \"\$NODE_ID\" --alpha-hotkey \"\$ALPHA_HOTKEY\" --alpha-amount \"\$ALPHA_AMOUNT_WEI\""
