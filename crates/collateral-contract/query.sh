#!/usr/bin/env bash
set -euo pipefail

# basic query to verify the contract is deployed and initialized
export CONTRACT_ADDRESS=0x970951a12F975E6762482ACA81E57D5A2A4e73F4
export NETWORK=local

collateral-cli --network "$NETWORK" --contract-address "$CONTRACT_ADDRESS" query trustee
collateral-cli --network "$NETWORK" --contract-address "$CONTRACT_ADDRESS" query min-collateral-increase
collateral-cli --network "$NETWORK" --contract-address "$CONTRACT_ADDRESS" query decision-timeout
collateral-cli --network "$NETWORK" --contract-address "$CONTRACT_ADDRESS" query netuid

