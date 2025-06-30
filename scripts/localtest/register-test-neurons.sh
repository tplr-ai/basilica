#!/bin/bash
# Register test neurons on local subtensor for testing

echo "Registering test neurons on local subtensor..."

# Wait for subtensor to be ready
echo "Waiting for subtensor to be ready..."
for i in {1..30}; do
    if nc -z localhost 9944 2>/dev/null; then
        echo "Subtensor is ready!"
        break
    fi
    echo "Waiting for subtensor... ($i/30)"
    sleep 2
done

# Use docker to run btcli commands against local chain
# The devnet-ready image should have test wallets pre-funded

# Register validator
echo "Registering validator..."
docker run --rm --network host \
    -v ~/.bittensor:/root/.bittensor \
    ghcr.io/opentensor/bittensor:latest \
    btcli subnet register \
    --netuid 1 \
    --wallet.name test_validator \
    --wallet.hotkey test_hotkey \
    --subtensor.network local \
    --subtensor.chain_endpoint ws://localhost:9944 \
    --no_prompt || echo "Validator registration failed or already registered"

# Register miner
echo "Registering miner..."
docker run --rm --network host \
    -v ~/.bittensor:/root/.bittensor \
    ghcr.io/opentensor/bittensor:latest \
    btcli subnet register \
    --netuid 1 \
    --wallet.name test_miner \
    --wallet.hotkey default \
    --subtensor.network local \
    --subtensor.chain_endpoint ws://localhost:9944 \
    --no_prompt || echo "Miner registration failed or already registered"

echo "Registration complete!"