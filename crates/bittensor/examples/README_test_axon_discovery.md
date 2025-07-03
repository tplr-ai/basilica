# Axon Discovery Test (Testnet)

This test compares two methods of reading axon information from the Bittensor testnet chain:

1. **Metagraph Method**: Uses the runtime API to fetch the entire metagraph
2. **Direct Storage Method**: Queries the Axons storage map directly

## Running the Test

```bash
cd crates/bittensor
cargo run --example test_axon_discovery
```

This connects to the testnet at `wss://test.finney.opentensor.ai:443/` on netuid 387.

## What to Look For

The test will output:
- Total neurons and axons in the metagraph
- Raw IP values (u128) stored on chain
- Decoded IPs using both lower 32 bits and upper 32 bits
- Comparison between metagraph data and direct storage queries

This helps debug whether:
1. IPs are being stored correctly on chain
2. The encoding/decoding mismatch is causing issues
3. The metagraph is properly aggregating axon data

## Expected Output

You should see IP addresses decoded in one of two ways:
- **Lower 32 bits**: If IPs are stored in bits 0-31 (current serve_axon implementation)
- **Upper 32 bits**: If IPs are stored in bits 96-127 (discovery.rs expectation)

The correct decoded IPs should match actual neuron IPs, not 0.0.0.0.