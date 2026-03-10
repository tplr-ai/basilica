#!/usr/bin/env bash
set -euo pipefail

NETUID="${NETUID:-39}"
TRUSTEE_ADDRESS="${TRUSTEE_ADDRESS:?must be set to the trustee EVM address}"
MIN_COLLATERAL="${MIN_COLLATERAL:-100000000000000000}"
MIN_ALPHA_COLLATERAL="${MIN_ALPHA_COLLATERAL:-5000000000}"
DECISION_TIMEOUT="${DECISION_TIMEOUT:-1}"
ADMIN_ADDRESS="${ADMIN_ADDRESS:?must be set to the admin EVM address}"
VALIDATOR_HOTKEY="${VALIDATOR_HOTKEY:?must be set to the validator hotkey (0x-prefixed 64 hex chars)}"
PRIVATE_KEY="${PRIVATE_KEY:?must be set to the deployer private key}"
RPC_URL="${RPC_URL:?must be set to the target RPC URL}"

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

TAO_DEPOSITS_ENABLED="${TAO_DEPOSITS_ENABLED:-true}"
ALPHA_DEPOSITS_ENABLED="${ALPHA_DEPOSITS_ENABLED:-true}"

INIT_DATA="$(cast calldata \
  "initialize(uint16,address,uint256,uint256,uint64,address,bytes32,bool,bool)" \
  "$NETUID" \
  "$TRUSTEE_ADDRESS" \
  "$MIN_COLLATERAL" \
  "$MIN_ALPHA_COLLATERAL" \
  "$DECISION_TIMEOUT" \
  "$ADMIN_ADDRESS" \
  "$VALIDATOR_HOTKEY" \
  "$TAO_DEPOSITS_ENABLED" \
  "$ALPHA_DEPOSITS_ENABLED")"

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
