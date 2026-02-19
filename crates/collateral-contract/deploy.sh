#!/usr/bin/env bash
set -euo pipefail

export NETUID=39
export TRUSTEE_ADDRESS=0xf24FF3a9CF04c71Dbc94D0b566f7A27B94566cac
export MIN_COLLATERAL=1
export DECISION_TIMEOUT=1
export ADMIN_ADDRESS=0xf24FF3a9CF04c71Dbc94D0b566f7A27B94566cac
export PRIVATE_KEY=0x
# export RPC_URL=https://lite.chain.opentensor.ai:443
# export RPC_URL=https://test.finney.opentensor.ai
export RPC_URL=http://localhost:9944
forge script script/DeployUpgradeable.s.sol \
 --rpc-url "$RPC_URL" \
 --private-key "$PRIVATE_KEY" \
 --broadcast
