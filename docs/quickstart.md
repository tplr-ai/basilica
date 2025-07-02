# Quick Start Guide - Updated Instructions

This guide provides updated instructions for building and running Basilica miner and validator nodes with all recent changes.

## Key Changes

1. **Dynamic UID Discovery**: UIDs are no longer hardcoded in configuration files. Both miner and validator automatically discover their UID from the Bittensor metagraph based on their hotkey.

2. **Chain Endpoint Auto-Detection**: The chain endpoint is automatically detected based on the network type if not explicitly specified:
   - `finney`: wss://entrypoint-finney.opentensor.ai:443
   - `test`: wss://test.finney.opentensor.ai:443
   - `local`: ws://127.0.0.1:9944

3. **Wallet Format Support**: Both JSON wallet files (new format) and raw seed phrases (old format) are supported. JSON wallets include fields like `secretPhrase`, `publicKey`, `accountId`, etc.

4. **Build Requirements**: A validator public key is required when building the validator from source (stored in `public_key.hex`). The miner does not require this key.

## Prerequisites

- Rust and Cargo installed
- Bittensor wallet with TAO tokens
- Validator public key file (`public_key.hex`) - only needed for building validator
- Linux/Unix environment

## Quick Setup

### 1. Generate Validator Key (only if building validator)

```bash
./scripts/gen-key.sh  # Creates public_key.hex
```

### 2. Create Configuration Files

Create `config/miner.toml`:
```toml
[server]
host = "0.0.0.0"
port = 8092

[database]
url = "sqlite:./data/miner.db"
max_connections = 5
min_connections = 1
run_migrations = true

[bittensor]
wallet_name = "test_miner"
hotkey_name = "default"
network = "test"  # or "finney" for mainnet
netuid = 387      # use 39 for mainnet
coldkey_name = "default"
axon_port = 8091

[executor_management]
health_check_interval = { secs = 30, nanos = 0 }
health_check_timeout = { secs = 10, nanos = 0 }
max_retry_attempts = 3
auto_recovery = true

[[executor_management.executors]]
id = "test-executor-1"
name = "Test Executor"
grpc_address = "127.0.0.1:50051"

# ... rest of configuration
```

Create `config/validator.toml`:
```toml
[server]
host = "0.0.0.0"
port = 8080

[database]
url = "sqlite:./data/validator.db"
max_connections = 5
min_connections = 1
run_migrations = true

[bittensor]
wallet_name = "test_validator"
hotkey_name = "default"
network = "test"  # or "finney" for mainnet
netuid = 387      # use 39 for mainnet
weight_interval_secs = 300
axon_port = 9091

[verification]
verification_interval = { secs = 600, nanos = 0 }
max_concurrent_verifications = 50
challenge_timeout = { secs = 120, nanos = 0 }
min_score_threshold = 0.1
min_stake_threshold = 1.0
max_miners_per_round = 20
min_verification_interval = { secs = 1800, nanos = 0 }
netuid = 387

# ... rest of configuration
```

### 3. Create Data Directory

```bash
mkdir -p data
```

### 4. Build Components

```bash
# Build miner (no validator key needed)
cargo build -p miner

# Build validator (requires validator public key)
VALIDATOR_PUBLIC_KEY=$(cat public_key.hex | tr -d '\n') cargo build -p validator

# Or set environment variable for validator build
export VALIDATOR_PUBLIC_KEY=$(cat public_key.hex | tr -d '\n')
cargo build -p validator
```

### 5. Set Up Wallets

```bash
# Create test wallets (if they don't exist)
mkdir -p ~/.bittensor/wallets/test_miner/hotkeys
mkdir -p ~/.bittensor/wallets/test_validator/hotkeys

# Copy existing wallet or create new ones
# For testing, you can use the same wallet for both:
cp ~/.bittensor/wallets/your_wallet/hotkeys/default ~/.bittensor/wallets/test_miner/hotkeys/default
cp ~/.bittensor/wallets/your_wallet/hotkeys/default ~/.bittensor/wallets/test_validator/hotkeys/default
```

### 6. Run Services

Run Miner:
```bash
./target/debug/miner --config config/miner.toml
```

Run Validator:
```bash
./target/debug/validator start --config config/validator.toml
```

## Common Issues and Solutions

### Database Connection Error
```
Error: unable to open database file
```
**Solution**: Ensure the data directory exists and has proper permissions:
```bash
mkdir -p data
chmod 755 data
```

### Wallet Loading Error
```
Error: Failed to load hotkey: Invalid format
```
**Solution**: Ensure wallet files exist and are in the correct format. The system supports both JSON wallets and raw seed phrases.

### GPU Attestor Build
The gpu-attestor binary no longer requires a validator public key to build. Simply run:
```bash
cargo build --bin gpu-attestor
```

### Executor Configuration Error
```
Error: At least one executor must be configured
```
**Solution**: Ensure at least one executor is defined in the `[[executor_management.executors]]` section of miner.toml.

### Metadata Compatibility Error
```
Error: The generated code is not compatible with the node
```
**Solution**: The system now uses its own generated metadata instead of cached types. Ensure you're building with the correct network:
```bash
BITTENSOR_NETWORK=test cargo build -p miner
```

## Network Configuration

The system supports three networks:
- `finney`: Main Bittensor network
- `test`: Test network (test.finney.opentensor.ai)
- `local`: Local development network

The chain endpoint is automatically selected based on the network unless explicitly specified in the configuration.

## Monitoring

Check service health:
```bash
# Miner
curl http://localhost:8092/health

# Validator
curl http://localhost:8080/health
```

View metrics:
```bash
# Miner metrics
curl http://localhost:9091/metrics

# Validator metrics
curl http://localhost:9090/metrics
```

## Next Steps

- Review the detailed [Miner Guide](miner.md) for advanced miner configuration
- Check the [Validator Guide](validator.md) for validator-specific details
- See the [Architecture Guide](architecture.md) for system design overview