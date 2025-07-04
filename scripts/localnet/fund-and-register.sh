#!/bin/bash

set -euo pipefail

# The following code is kept for reference but not executed in localnet
NETWORK="local"
NETUID=1  # Default subnet

# Check Subtensor
if ! nc -z localhost 9944 2>/dev/null; then
    echo "ERROR: Subtensor not running on localhost:9944"
    echo "Start it with: docker compose -f compose.yml up -d alice"
    exit 1
fi

# Wait for chain
echo "Waiting for chain..."
while ! uvx --from bittensor-cli btcli subnet list --network ${NETWORK} 2>/dev/null | grep -q "Subnets"; do
    sleep 2
done

# Fund wallets
for wallet in owner validator miner_1 miner_2 test_validator test_miner; do
    echo "Funding ${wallet}..."
    uvx --from bittensor-cli --with torch btcli wallet faucet --wallet.name "${wallet}" --network ${NETWORK} -y
done

# Create subnet
echo "Creating subnet..."
uvx --from bittensor-cli btcli subnet create --wallet.name owner --wallet.hotkey default --network ${NETWORK} -y

# Register neurons
for wallet in validator miner_1 miner_2 test_validator test_miner; do
    echo "Registering ${wallet}..."
    uvx --from bittensor-cli btcli subnet register --wallet.name "${wallet}" --wallet.hotkey default --netuid ${NETUID} --network ${NETWORK} -y
done

# Add stake
echo "Adding stake to validators..."
uvx --from bittensor-cli btcli stake add --wallet.name validator --wallet.hotkey default --amount 100 --network ${NETWORK} --unsafe -y
uvx --from bittensor-cli btcli stake add --wallet.name test_validator --wallet.hotkey test_hotkey --amount 50 --network ${NETWORK} --unsafe -y

echo "Done. Check metagraph: uvx --from bittensor-cli btcli subnet metagraph --netuid ${NETUID} --network ${NETWORK}"
