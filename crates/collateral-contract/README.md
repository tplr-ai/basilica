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

### Deploying the Contract

Deployment is handled by [`deploy.sh`](./deploy.sh), which deploys the implementation contract and an ERC1967 proxy in one step.

**Required environment variables:**

| Variable | Description |
|---|---|
| `PRIVATE_KEY` | Deployer's private key (only needs gas, no on-chain privilege after deploy) |
| `RPC_URL` | Target network RPC endpoint |
| `TRUSTEE_ADDRESS` | EVM address that can slash collateral and deny reclaims |
| `ADMIN_ADDRESS` | EVM address that can upgrade the contract and change config |
| `VALIDATOR_HOTKEY` | Substrate validator hotkey (0x-prefixed, 64 hex chars) |

**Optional variables (have sensible defaults):**

| Variable | Default | Description |
|---|---|---|
| `NETUID` | `39` | Subnet ID |
| `MIN_COLLATERAL` | `100000000000000000` | Minimum TAO collateral increase (wei) |
| `MIN_ALPHA_COLLATERAL` | `5000000000` | Minimum alpha collateral increase (RAO) |
| `DECISION_TIMEOUT` | `86400` | Reclaim decision window (seconds; default = 1 day) |
| `TAO_DEPOSITS_ENABLED` | `false` | Whether TAO deposits are accepted |
| `ALPHA_DEPOSITS_ENABLED` | `true` | Whether alpha deposits are accepted |

**Example — testnet:**

```bash
export PRIVATE_KEY="0x<deployer-key>"
export RPC_URL="https://test.chain.opentensor.ai"
export TRUSTEE_ADDRESS="0x<trustee-address>"
export ADMIN_ADDRESS="0x<admin-address>"
export VALIDATOR_HOTKEY="0x<validator-hotkey>"
export DECISION_TIMEOUT=86400

bash ./deploy.sh
```

**Example — mainnet:**

```bash
export PRIVATE_KEY="0x<deployer-key>"
export RPC_URL="https://lite.chain.opentensor.ai"
export TRUSTEE_ADDRESS="0x<trustee-address>"
export ADMIN_ADDRESS="0x<admin-address>"
export VALIDATOR_HOTKEY="0x<validator-hotkey>"
export DECISION_TIMEOUT=86400

bash ./deploy.sh
```

**Localnet** deployment is automated by `scripts/collateral/setup-localnet-env.sh`, which sets the deployer as both trustee and admin.

### Upgrading the Implementation

After the initial deployment, the implementation can be upgraded via [`upgrade.sh`](./upgrade.sh). Only the admin (`UPGRADER_ROLE` holder) can perform upgrades.

**Required environment variables:**

| Variable | Description |
|---|---|
| `PROXY_ADDRESS` | The existing proxy contract address |
| `PRIVATE_KEY` | Admin's private key (must have `UPGRADER_ROLE`) |
| `RPC_URL` | Target network RPC endpoint |

**Optional:**

| Variable | Description |
|---|---|
| `MIGRATE_CALLDATA` | Calldata for a migration function on the new implementation (default: `0x` = no migration) |

**Example:**

```bash
export PROXY_ADDRESS="0x<proxy-address>"
export PRIVATE_KEY="0x<admin-key>"
export RPC_URL="https://test.chain.opentensor.ai"

bash ./upgrade.sh
```

The script performs pre-flight checks (RPC reachability, proxy existence, UPGRADER_ROLE verification, version comparison), deploys the new implementation, prompts for confirmation, then verifies the upgrade succeeded.

**Upgrade safety rules:**

*Storage layout:*

1. **Never reorder or remove existing storage variables** — the proxy's storage layout must stay compatible. New variables go after existing ones (before the `_gap`).
2. **Shrink `_gap` when adding new storage** — if you add N new storage slots, reduce `_gap` from `[49]` to `[49-N]` to keep total slot count stable.
3. **Never change the type of an existing variable** — e.g. changing `uint256` to `address` corrupts storage.
4. **Never change inheritance order** — parent contracts occupy storage slots too; reordering them shifts everything.
5. **Do not set initial values on state variable declarations** — e.g. `uint256 public x = 42` runs in the implementation's constructor context, not the proxy's. The proxy never sees it. Set all initial values in `initialize()` or a migration function.
6. **Do not add state variables in parent/base contracts without shrinking their gap** — adding a stateful base contract mid-inheritance chain shifts all downstream storage slots.

*Initialization:*

7. **No constructors with state** — the existing `constructor() { _disableInitializers(); }` is correct. Never add state-setting logic to constructors since it won't affect the proxy.
8. **Use a migration function for new state** — if the new version needs to initialize new variables, add a `migrateVN()` function and pass its calldata via `MIGRATE_CALLDATA`. Guard it with `reinitializer(N)` to ensure it runs exactly once.
9. **Call every parent initializer exactly once** — Solidity does not auto-call parent initializers. Missing one silently leaves that module uninitialized.

*UUPS-specific:*

10. **Never remove or break the upgrade mechanism** — every new implementation MUST inherit `UUPSUpgradeable` and include a working `_authorizeUpgrade`. If a new implementation is deployed without upgrade logic, the proxy is permanently bricked with no recovery path.
11. **Bump `getVersion()`** — the upgrade script checks that the new version is greater than the current one.
12. **Never use `selfdestruct`** — if executed in the implementation's context via `delegatecall`, it destroys the proxy's state. Post-Cancun this is less dangerous but the rule stands as defense-in-depth.
13. **Never use unrestricted `delegatecall` to user-supplied addresses** — the target executes in the proxy's storage context. Our contract only `delegatecall`s to hardcoded precompile addresses, which is safe.

*ABI compatibility:*

14. **Do not remove or change the signature of existing external/public functions** — other contracts and off-chain systems call the proxy by selector. You can add new functions freely.

*Operational:*

15. **Constants and immutables are safe to change** — they're stored in bytecode, not storage slots.
16. **Test upgrades against a fork of production state** — dry-run `upgradeToAndCall` on a mainnet/testnet fork to verify existing state is preserved. Use `forge inspect` to diff storage layouts between old and new implementations.
17. **Use a multisig or timelock for `UPGRADER_ROLE` in production** — a compromised single EOA with upgrade authority can replace the implementation with a malicious contract.

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

> Transaction policy in this CLI is alpha-primary: deposit and slash commands only expose alpha inputs.
> TAO state remains available via query and event sync.
> Alpha amount arguments are in RAO (1e9 = 1 alpha).

#### Deposit Collateral

```bash
# Basic deposit on mainnet (alpha-only tx path)
collateral-cli tx deposit \
  --private-key $PRIVATE_KEY \
  --hotkey 0123456789abcdef0123456789abcdef0123456789abcdef0123456789abcdef \
  --node-id 6339ba4f-60f9-45c2-9d95-2b755bb57ca6 \
  --alpha-hotkey fedcba9876543210fedcba9876543210fedcba9876543210fedcba9876543210 \
  --alpha-amount 5000000000

# Deposit on testnet
collateral-cli --network testnet tx deposit \
  --private-key $PRIVATE_KEY \
  --hotkey 0123456789abcdef0123456789abcdef0123456789abcdef0123456789abcdef \
  --node-id 9e0a4d34-3110-48d1-b3c5-580f44270f13 \
  --alpha-hotkey fedcba9876543210fedcba9876543210fedcba9876543210fedcba9876543210 \
  --alpha-amount 10000000000

# Deposit with custom contract address
collateral-cli --contract-address 0x5FbDB2315678afecb367f032d93F642f64180aa3 tx deposit \
  --private-key $PRIVATE_KEY \
  --hotkey 0123456789abcdef0123456789abcdef0123456789abcdef0123456789abcdef \
  --node-id 1f4b20b4-1fdd-4fbb-8904-4310ec6df456 \
  --alpha-hotkey fedcba9876543210fedcba9876543210fedcba9876543210fedcba9876543210 \
  --alpha-amount 7000000000

# Using environment variable for private key
export PRIVATE_KEY=0x<your-private-key>
collateral-cli tx deposit \
  --hotkey 0123456789abcdef0123456789abcdef0123456789abcdef0123456789abcdef \
  --node-id 12c61943-7ce0-470f-a3aa-14df501f15e2 \
  --alpha-hotkey fedcba9876543210fedcba9876543210fedcba9876543210fedcba9876543210 \
  --alpha-amount 6000000000
```

#### Reclaim Collateral

```bash
# Alpha reclaim destination is derived from the node owner's mapped coldkey.
# Basic reclaim
collateral-cli tx reclaim-collateral \
  --private-key $PRIVATE_KEY \
  --hotkey 0123456789abcdef0123456789abcdef0123456789abcdef0123456789abcdef \
  --node-id 6339ba4f-60f9-45c2-9d95-2b755bb57ca6 \
  --url "https://example.com/reclaim-proof" \
  --url-content-sha256 abcdef1234567890abcdef1234567890abcdef1234567890abcdef1234567890

# Reclaim on testnet
collateral-cli --network testnet tx reclaim-collateral \
  --private-key $PRIVATE_KEY \
  --hotkey 0123456789abcdef0123456789abcdef0123456789abcdef0123456789abcdef \
  --node-id 9e0a4d34-3110-48d1-b3c5-580f44270f13 \
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
# Slash collateral for misconduct (alpha-only tx path)
collateral-cli tx slash-collateral \
  --private-key $PRIVATE_KEY \
  --hotkey 0123456789abcdef0123456789abcdef0123456789abcdef0123456789abcdef \
  --node-id 6339ba4f-60f9-45c2-9d95-2b755bb57ca6 \
  --slash-alpha-amount 5000000000 \
  --url "https://evidence.example.com/slash-proof" \
  --url-content-sha256 aab03e786183b16c8a0b15f6b40ff607aab03e786183b16c8a0b15f6b40ff607

# Slash on testnet with detailed proof
collateral-cli --network testnet tx slash-collateral \
  --private-key $PRIVATE_KEY \
  --hotkey fedcba9876543210fedcba9876543210fedcba9876543210fedcba9876543210 \
  --node-id 9e0a4d34-3110-48d1-b3c5-580f44270f13 \
  --slash-alpha-amount 10000000000 \
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

# Get minimum alpha collateral increase
collateral-cli query min-alpha-collateral-increase

# Check deposit toggles
collateral-cli query tao-deposits-enabled
collateral-cli query alpha-deposits-enabled
```

#### Node-Specific Queries

```bash
# Get miner address for node
collateral-cli query node-to-miner \
  --hotkey 0123456789abcdef0123456789abcdef0123456789abcdef0123456789abcdef \
  --node-id 6339ba4f-60f9-45c2-9d95-2b755bb57ca6

# Get TAO + alpha collateral amounts for node
collateral-cli query collaterals \
  --hotkey 0123456789abcdef0123456789abcdef0123456789abcdef0123456789abcdef \
  --node-id 6339ba4f-60f9-45c2-9d95-2b755bb57ca6

# Get reclaim details
collateral-cli query reclaims \
  --reclaim-request-id 42

# Query on different networks
collateral-cli --network testnet query collaterals \
  --hotkey fedcba9876543210fedcba9876543210fedcba9876543210fedcba9876543210 \
  --node-id 9e0a4d34-3110-48d1-b3c5-580f44270f13

# List active nodes/reclaims
collateral-cli query all-collaterals
collateral-cli query all-reclaims

# Query with custom contract
collateral-cli --contract-address 0x9fE46736679d2D9a65F0992F2272dE9f3c7fa6e0 query netuid
```

### Event Scanning Commands

```bash
# Scan events with pretty output (default range: 0 to current block)
collateral-cli --network testnet events scan

# Scan events with JSON output
collateral-cli --network testnet events scan --format json

# Scan recent events (last 100 blocks from current)
collateral-cli events scan --from-block $(echo "$(curl -s -X POST -H 'Content-Type: application/json' --data '{\"jsonrpc\":\"2.0\",\"method\":\"eth_blockNumber\",\"params\":[],\"id\":1}' https://lite.chain.opentensor.ai:443 | jq -r .result | sed 's/0x//' | awk '{print strtonum(\"0x\" $0)}') - 100" | bc)

# Scan events on testnet
collateral-cli --network testnet events scan --format json

# Scan events with custom contract
collateral-cli --contract-address 0x8464135c8F25Da09e49BC8782676a84730C318bC events scan
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

# Test contract deployment flow (see "Deploying the Contract" section for required env vars)
bash ./deploy.sh
```
