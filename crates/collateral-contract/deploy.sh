+#!/usr/bin/env bash
+set -euo pipefail

export NETUID=39
export TRUSTEE_ADDRESS=0xf24FF3a9CF04c71Dbc94D0b566f7A27B94566cac
export MIN_COLLATERAL=1
export DECISION_TIMEOUT=1
export ADMIN_ADDRESS=0xf24FF3a9CF04c71Dbc94D0b566f7A27B94566cac
export ALPHA_HOTKEY=0x0000000000000000000000000000000000000000000000000000000000000001
export PRIVATE_KEY=0x5fb92d6e98884f76de468fa3f6278f8807c48bebc13595d45af5bdc4da702133
# export RPC_URL=https://lite.chain.opentensor.ai:443
# export RPC_URL=https://test.finney.opentensor.ai
export RPC_URL=http://localhost:9944
forge script script/DeployUpgradeable.s.sol \
 --rpc-url "$RPC_URL" \
 --private-key "$PRIVATE_KEY" \
 --broadcast

