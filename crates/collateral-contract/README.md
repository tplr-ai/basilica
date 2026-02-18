# Collateral Contract

This package contains the Collateral smart contract and a comprehensive CLI tool for interacting with it.

## Components

- **Smart Contracts**: Solidity contracts for collateral management
- **Rust Library**: Contract bindings and interaction functions
- **CLI Tool**: Command-line interface for all contract operations

## Development Setup

### Smart Contract Development

```bash
# Install Foundry
curl -L https://foundry.paradigm.xyz | bash
forge init
forge install OpenZeppelin/openzeppelin-contracts
forge install OpenZeppelin/openzeppelin-contracts-upgradeable

# Run contract tests
forge test
```

### deploy to testnet

```bash
export NETUID=39
export TRUSTEE_ADDRESS=0xABCaD56aa87f3718C8892B48cB443c017Cd632BB
export MIN_COLLATERAL=1000000000000000000
export DECISION_TIMEOUT=3600
export ADMIN_ADDRESS=0xABCaD56aa87f3718C8892B48cB443c017Cd632BB
export PRIVATE_KEY=0x0000000000000000000000000000000000000000000000000000000000000000

forge script script/DeployUpgradeable.s.sol \
 --rpc-url https://test.chain.opentensor.ai \
 --private-key YOUR_PRIVATE_KEY \
 --broadcast

# Output like
# CollateralUpgradeable

✅ [Success] Hash: 0xb727d00872419766cd274f5c15b764bb010e74728720925cc5d5c85405dcea31
Contract Address: 0x567E4c231AB946CdEf1C48eFA154BB8790Ae58Ba
Block: 5232407
Paid: 0.005569686466097676 ETH (276627 gas \* 20.134283588 gwei)

# ERC1967Proxy

✅ [Success] Hash: 0xcd91c195019ec4373be318554e4dbfbb95a59a8a823016253c3b4e673310ffa6
Contract Address: 0x22ffee3f67E476870C1A79059C3c51AeD096dF92
Block: 5232407
Paid: 0.068712128660782484 ETH (3412693 gas \* 20.134283588 gwei)

```

### CLI Development

````bash
# Build the CLI
cargo build --bin collateral-cli

# Run library tests
cargo test --lib

# Run all tests
cargo test


## CLI Tool Usage

The `collateral-cli` provides a comprehensive interface for interacting with the Collateral contract.

### Installation

```bash
# Build the CLI tool
cargo build --release --bin collateral-cli

# The binary will be available at target/release/collateral-cli
````

```bash
# Install the CLI tool
cd crates/collateral-contract
cargo install --path .
# The binary will be available at $HOME/.cargo/bin/collateral-cli
```

### Global Options

```bash
# Show help
collateral-cli --help

# Show version
collateral-cli --version

# Use different networks
collateral-cli --network mainnet    # Default
collateral-cli --network testnet    # Test network
collateral-cli --network local      # Local development

# Override contract address
collateral-cli --contract-address 0x1234567890123456789012345678901234567890
```

## Command Examples

### Transaction Commands

#### Deposit Collateral

```bash
# Basic deposit on mainnet
collateral-cli tx deposit \
  --private-key $PRIVATE_KEY \
  --hotkey 0123456789abcdef0123456789abcdef0123456789abcdef0123456789abcdef \
  --node-id 123 \
  --amount 1000000000000000000

# Deposit on testnet
collateral-cli --network testnet tx deposit \
  --private-key $PRIVATE_KEY \
  --hotkey 0123456789abcdef0123456789abcdef0123456789abcdef0123456789abcdef \
  --node-id 456 \
  --amount 5000000000000000000

# Deposit with custom contract address
collateral-cli --contract-address 0x5FbDB2315678afecb367f032d93F642f64180aa3 tx deposit \
  --private-key $PRIVATE_KEY \
  --hotkey 0123456789abcdef0123456789abcdef0123456789abcdef0123456789abcdef \
  --node-id 789 \
  --amount 2000000000000000000

# Using environment variable for private key
export PRIVATE_KEY=0x1234567890abcdef1234567890abcdef1234567890abcdef1234567890abcdef12
collateral-cli tx deposit \
  --hotkey 0123456789abcdef0123456789abcdef0123456789abcdef0123456789abcdef \
  --node-id 101 \
  --amount 1500000000000000000
```

#### Reclaim Collateral

```bash
# Basic reclaim
collateral-cli tx reclaim-collateral \
  --private-key $PRIVATE_KEY \
  --hotkey 0123456789abcdef0123456789abcdef0123456789abcdef0123456789abcdef \
  --node-id 123 \
  --url "https://example.com/reclaim-proof" \
  --url-content-sha256 abcdef1234567890abcdef1234567890abcdef1234567890abcdef1234567890

# Reclaim on testnet
collateral-cli --network testnet tx reclaim-collateral \
  --private-key $PRIVATE_KEY \
  --hotkey 0123456789abcdef0123456789abcdef0123456789abcdef0123456789abcdef \
  --node-id 456 \
  --url "https://proof-server.testnet.com/evidence/456" \
  --url-content-sha256 d41d8cd98f00b204e9800998ecf8427ed41d8cd98f00b204e9800998ecf8427e
```

#### Finalize Reclaim

```bash
# Finalize reclaim request
collateral-cli tx finalize-reclaim \
  --private-key $PRIVATE_KEY \
  --reclaim-request-id 42

# Finalize with hex request ID
collateral-cli tx finalize-reclaim \
  --private-key $PRIVATE_KEY \
  --reclaim-request-id 0x2a
```

#### Deny Reclaim

```bash
# Deny reclaim request
collateral-cli tx deny-reclaim \
  --private-key $PRIVATE_KEY \
  --reclaim-request-id 42 \
  --url "https://example.com/denial-proof" \
  --url-content-sha256 5d41402abc4b2a76b9719d911017c5925d41402abc4b2a76b9719d911017c592
```

#### Slash Collateral

```bash
# Slash collateral for misconduct
collateral-cli tx slash-collateral \
  --private-key $PRIVATE_KEY \
  --hotkey 0123456789abcdef0123456789abcdef0123456789abcdef0123456789abcdef \
  --node-id 123 \
  --url "https://evidence.example.com/slash-proof" \
  --url-content-sha256 aab03e786183b16c8a0b15f6b40ff607aab03e786183b16c8a0b15f6b40ff607

# Slash on testnet with detailed proof
collateral-cli --network testnet tx slash-collateral \
  --private-key $PRIVATE_KEY \
  --hotkey fedcba9876543210fedcba9876543210fedcba9876543210fedcba9876543210 \
  --node-id 999 \
  --url "https://audit.testnet.com/violations/999" \
  --url-content-sha256 098f6bcd4621d373cade4e832627b4f6098f6bcd4621d373cade4e832627b4f6
```

### Query Commands

#### Basic Queries

```bash
# Get network UID
collateral-cli query netuid

# Get trustee address
collateral-cli query trustee

# Get decision timeout (in seconds)
collateral-cli query decision-timeout

# Get minimum collateral increase
collateral-cli query min-collateral-increase
```

#### Node-Specific Queries

```bash
# Get miner address for node
collateral-cli query node-to-miner \
  --hotkey 0123456789abcdef0123456789abcdef0123456789abcdef0123456789abcdef \
  --node-id 123

# Get collateral amount for node
collateral-cli query collaterals \
  --hotkey 0123456789abcdef0123456789abcdef0123456789abcdef0123456789abcdef \
  --node-id 123

# Get reclaim details
collateral-cli query reclaims \
  --reclaim-request-id 42

# Query on different networks
collateral-cli --network testnet query collaterals \
  --hotkey fedcba9876543210fedcba9876543210fedcba9876543210fedcba9876543210 \
  --node-id 456

# Query with custom contract
collateral-cli --contract-address 0x9fE46736679d2D9a65F0992F2272dE9f3c7fa6e0 query netuid
```

### Event Scanning Commands

```bash
# Scan events with pretty output (default)
collateral-cli events scan --network testnet --from-block 0 --to-block 1000

# Scan events with JSON output
collateral-cli events scan --network testnet --from-block 0 --to-block 1000 --format json

# Scan recent events (last 100 blocks from current)
collateral-cli events scan --from-block $(echo "$(curl -s -X POST -H 'Content-Type: application/json' --data '{\"jsonrpc\":\"2.0\",\"method\":\"eth_blockNumber\",\"params\":[],\"id\":1}' https://lite.chain.opentensor.ai:443 | jq -r .result | sed 's/0x//' | awk '{print strtonum(\"0x\" $0)}') - 100" | bc)

# Scan events on testnet
collateral-cli --network testnet events scan --from-block 0 --to-block 1000 --format json

# Scan events with custom contract
collateral-cli --contract-address 0x8464135c8F25Da09e49BC8782676a84730C318bC events scan --from-block 0 --to-block 1000
```

## Testing Commands

### Unit Tests

```bash
# Run all tests
cargo test

# Run library tests only
cargo test --lib

# Run CLI binary tests only
cargo test --bin collateral-cli

# Run specific test
cargo test test_parse_hotkey

# Run tests with output
cargo test -- --nocapture

# Run tests in release mode
cargo test --release
```

### Integration Tests

```bash
# Test CLI help system
cargo run --bin collateral-cli -- --help
cargo run --bin collateral-cli -- tx --help
cargo run --bin collateral-cli -- query --help
cargo run --bin collateral-cli -- events --help

# Test CLI argument validation
cargo run --bin collateral-cli -- --network invalid_network query netuid  # Should fail
cargo run --bin collateral-cli -- --contract-address "invalid" query netuid  # Should fail
cargo run --bin collateral-cli -- tx deposit  # Should fail (missing args)

# Test different networks
cargo run --bin collateral-cli -- --network mainnet query netuid   # Default network
cargo run --bin collateral-cli -- --network testnet query netuid   # Testnet
cargo run --bin collateral-cli -- --network local query netuid     # Local (will fail without local node)
```

### Contract Tests

```bash
# Run Solidity tests
forge test

# Run specific contract test
forge test --match-test testDeposit

# Run tests with verbosity
forge test -vvv

# Run tests with gas reporting
forge test --gas-report

# Test contract deployment
forge script script/DeployUpgradeable.s.sol
```
