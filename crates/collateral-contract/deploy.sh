#!/usr/bin/env bash
set -euo pipefail

NETUID="${NETUID:-39}"
TRUSTEE_ADDRESS="${TRUSTEE_ADDRESS:-0xf24FF3a9CF04c71Dbc94D0b566f7A27B94566cac}"
MIN_COLLATERAL="${MIN_COLLATERAL:-1}"
DECISION_TIMEOUT="${DECISION_TIMEOUT:-1}"
ADMIN_ADDRESS="${ADMIN_ADDRESS:-0xf24FF3a9CF04c71Dbc94D0b566f7A27B94566cac}"
VALIDATOR_HOTKEY="${VALIDATOR_HOTKEY:-0x900dd1d8d4d94772b09fc1c82a74ea4af1471ba5594371ccc10632a1611b1945}"
PRIVATE_KEY="${PRIVATE_KEY:-0x}"
# export RPC_URL=https://lite.chain.opentensor.ai:443
# export RPC_URL=https://test.finney.opentensor.ai
RPC_URL="${RPC_URL:-http://localhost:9944}"

if [ "$PRIVATE_KEY" = "0x" ]; then
  echo "set PRIVATE_KEY to a real key before deploying" >&2
  exit 1
fi

echo "Deploying CollateralUpgradeable implementation..."
impl_output="$(FOUNDRY_PROFILE=local forge create src/CollateralUpgradeable.sol:CollateralUpgradeable \
  --rpc-url "$RPC_URL" \
  --private-key "$PRIVATE_KEY" \
  --legacy \
  --broadcast)"
echo "$impl_output"

IMPLEMENTATION_ADDRESS="$(echo "$impl_output" | awk '/Deployed to:/ {print $3}')"
if [ -z "$IMPLEMENTATION_ADDRESS" ]; then
  echo "failed to parse implementation address" >&2
  exit 1
fi

INIT_DATA="$(cast calldata \
  "initialize(uint16,address,uint256,uint64,address,bytes32)" \
  "$NETUID" \
  "$TRUSTEE_ADDRESS" \
  "$MIN_COLLATERAL" \
  "$DECISION_TIMEOUT" \
  "$ADMIN_ADDRESS" \
  "$VALIDATOR_HOTKEY")"

echo "Deploying ERC1967Proxy with initialize calldata..."
proxy_output="$(FOUNDRY_PROFILE=local forge create lib/openzeppelin-contracts/contracts/proxy/ERC1967/ERC1967Proxy.sol:ERC1967Proxy \
  --rpc-url "$RPC_URL" \
  --private-key "$PRIVATE_KEY" \
  --legacy \
  --broadcast \
  --constructor-args "$IMPLEMENTATION_ADDRESS" "$INIT_DATA")"
echo "$proxy_output"

PROXY_ADDRESS="$(echo "$proxy_output" | awk '/Deployed to:/ {print $3}')"
if [ -z "$PROXY_ADDRESS" ]; then
  echo "failed to parse proxy address" >&2
  exit 1
fi

echo
echo "Implementation: $IMPLEMENTATION_ADDRESS"
echo "Proxy:          $PROXY_ADDRESS"
