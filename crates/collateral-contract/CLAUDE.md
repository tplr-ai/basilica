# Collateral Contract

## Bittensor EVM Compatibility Layer

Bittensor runs on **Subtensor**, a Substrate-based blockchain. The EVM layer is built using Frontier (Substrate's EVM compatibility framework) and runs **on top of** Subtensor as an application layer. All execution happens on the Bittensor blockchain, not Ethereum.

### Two Address Worlds

| Property | Substrate Side | EVM Side |
|---|---|---|
| **Format** | SS58 (starts with `5`) | H160 (starts with `0x`) |
| **Size** | 32-byte public key (AccountId32) | 20-byte address |
| **Key type** | Ed25519/Sr25519 | Secp256k1 |
| **Wallet tools** | btcli, Bittensor SDK, Polkadot.js | MetaMask, Hardhat, ethers.js |
| **Can do** | Subtensor extrinsics (staking, registration, transfers) | EVM smart contract calls |
| **Cannot do** | Sign EVM smart contracts | Sign Substrate extrinsics |

### HashedAddressMapping (H160 <-> SS58)

Bittensor uses Frontier's `HashedAddressMapping` for deterministic, one-way address derivation:

**H160 -> SS58 (EVM address -> Substrate mirror):**
```rust
fn into_account_id(address: H160) -> AccountId32 {
    let mut data = [0u8; 24];
    data[0..4].copy_from_slice(b"evm:");
    data[4..24].copy_from_slice(&address[..]);
    let hash = blake2_256(&data);
    AccountId32::from(hash)
}
```

**SS58 -> H160 (Substrate address -> EVM mirror):**
Take the first 20 bytes of the 32-byte Substrate public key.

**Critical:** Neither direction yields a usable private key. The derived "mirror" addresses are accounting-only.

### Four Addresses, Two Keypairs

```
Keypair A (Ed25519/Sr25519 - Bittensor native):
  +-- #1: Native SS58 address (you control, signs extrinsics)
  +-- #4: EVM mirror (first 20 bytes of pubkey, NO private key)

Keypair B (Secp256k1 - Ethereum native):
  +-- #3: Native H160 address (you control, signs EVM txns)
  +-- #2: SS58 mirror (blake2("evm:" ++ h160_bytes), NO private key)
```

An SS58 wallet CANNOT sign EVM transactions. An EVM wallet CANNOT sign Substrate extrinsics. They are separate identity domains on the same chain.

### Balance Flow Between Layers

- Sending TAO from Substrate wallet (#1) to EVM mirror SS58 address (#2) makes it appear in the EVM wallet (#3) in MetaMask.
- Going EVM -> Substrate uses the `BalanceTransfer` precompile or `evm.withdraw()` extrinsic.
- Same TAO token, different account format, no wrapping involved.

### Precompiles (EVM -> Substrate Bridge)

| Precompile | Address | Purpose |
|---|---|---|
| Ed25519Verify | `0x...0402` | Verify Ed25519 signatures (prove SS58 key ownership from EVM) |
| StakingV2 | `0x...0805` | Add/remove stake, move stake between hotkeys |
| BalanceTransfer | custom | Transfer TAO between accounts |
| SubnetPrecompile | custom | Subnet operations |

**Staking caveat:** When a smart contract calls the staking precompile, the **contract's address** is the coldkey (not the original caller).

### Network Config

| Network | RPC URL | Chain ID |
|---|---|---|
| Mainnet | `https://lite.chain.opentensor.ai` | 964 |
| Testnet | `https://test.finney.opentensor.ai` | 945 |
| Localnet | `http://localhost:9944` | 42 |

### Unit Conversion

- EVM side: 1 TAO = 1e18 (like ETH wei)
- Substrate staking (RAO): 1 TAO = 1e9 RAO
- When calling staking precompile from EVM with `msg.value`: `amount_rao = msg.value / 1e9`

## This Crate

### Architecture

- **Solidity contracts** (`src/Collateral.sol`, `src/CollateralUpgradeableV2.sol`): Upgradeable ERC1967 proxy pattern via OpenZeppelin
- **Rust library** (`src/lib.rs`): Alloy-based contract bindings generated via `sol!` macro from ABI JSON
- **CLI** (`src/main.rs`): `collateral-cli` for all contract operations (deposit, reclaim, slash, query, events)

### Contract Identity Model

The contract uses **H160 addresses** for miners and **bytes32** for hotkeys/coldkeys (which are Substrate public keys passed as raw 32-byte values). The `nodeId` is `bytes16` (UUID). The trustee is an H160 address.

Both miners and validators need TAO in their H160 wallets for gas (~0.01 TAO minimum).

### Key Contract State

- `nodeToMiner[hotkey][nodeId] -> address`: Maps (hotkey, nodeId) to the miner's H160 address
- `collaterals[hotkey][nodeId] -> uint256`: TAO collateral amount (in wei, 1e18)
- `alphaCollaterals[hotkey][nodeId] -> uint256`: Alpha token collateral
- `reclaims[reclaimId] -> Reclaim`: Pending reclaim requests with deny timeout
- `CONTRACT_COLDKEY`: bytes32 Substrate coldkey associated with the contract (for staking precompile calls)

### Contract Operations

- **Deposit**: Miner sends TAO + optional alpha to lock as collateral for a (hotkey, nodeId) pair
- **Reclaim**: Miner requests collateral back, starts a timeout window for trustee to deny
- **Finalize Reclaim**: After timeout passes without denial, miner withdraws
- **Deny Reclaim**: Trustee rejects a reclaim within the timeout window
- **Slash**: Trustee slashes a miner's collateral for misconduct
- **Burn Register**: Calls the `0x0804` precompile to burn-register the contract on-chain

### Deployment

- Mainnet deployment is **whitelisted only** (request via Bittensor Discord `#evm-bittensor`)
- Uses `forge script script/DeployUpgradeable.s.sol` with ERC1967 proxy
- After deployment, update ABI via `update_abi.py`

### Testing

```bash
forge test           # Solidity contract tests
cargo test --lib     # Rust library unit tests
cargo test           # All tests
```
