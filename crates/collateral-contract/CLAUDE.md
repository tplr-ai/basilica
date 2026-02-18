# Collateral Contract

## Bittensor EVM Compatibility Layer

Bittensor runs on **Subtensor**, a Substrate-based blockchain. The EVM layer is built using Frontier (Substrate's EVM compatibility framework) and runs **on top of** Subtensor as an application layer. All execution happens on the Bittensor blockchain, not Ethereum.

### Bittensor Key Model (Coldkey / Hotkey)

- **Coldkey**: Primary ownership key (SS58, sr25519/ed25519). Controls funds, staking, unstaking, transfers, subnet registration. Keep offline.
- **Hotkey**: Operational key (same key type, independently generated -- NOT derived from coldkey). Runs miners/validators, signs weight-setting extrinsics.
- **Link**: On-chain mapping (`hotkey -> coldkey`) created when coldkey signs a registration extrinsic. One coldkey can own multiple hotkeys.
- **Fees**: Coldkey pays registration costs. Weight-related hotkey extrinsics (`set_weights`, `commit_weights`, `reveal_weights`) are fee-free. Do NOT send TAO to a hotkey -- it's not designed to hold funds.

### Two Address Worlds

| Property | Substrate Side | EVM Side |
|---|---|---|
| **Format** | SS58 (starts with `5`) | H160 (starts with `0x`) |
| **Size** | 32-byte public key (AccountId32) | 20-byte address |
| **Key type** | Ed25519/Sr25519 | Secp256k1 |
| **Wallet tools** | btcli, Bittensor SDK, Polkadot.js | MetaMask, Hardhat, ethers.js |
| **Can do** | Subtensor extrinsics (staking, registration, transfers) | EVM smart contract calls |
| **Cannot do** | Sign EVM smart contracts | Sign Substrate extrinsics |

### HashedAddressMapping (H160 -> SS58, One-Way Only)

Bittensor uses Frontier's `HashedAddressMapping` for deterministic, **one-way** address derivation from EVM to Substrate:

```rust
fn into_account_id(address: H160) -> AccountId32 {
    let mut data = [0u8; 24];
    data[0..4].copy_from_slice(b"evm:");      // literal 4 bytes
    data[4..24].copy_from_slice(&address[..]);  // 20-byte H160 address
    let hash = blake2b_256(&data);              // blake2b with 256-bit output
    AccountId32::from(hash)
}
```

There is **no general reverse mapping** from SS58 back to H160. The resulting SS58 address is a "mirror" with no known private key. An EVM contract's on-chain Substrate identity is derived this way.

### Balance Flow Between Layers

- Sending TAO from Substrate wallet to an EVM mirror SS58 address makes it appear in the corresponding EVM wallet in MetaMask.
- Going EVM -> Substrate uses the `BalanceTransfer` precompile or `evm.withdraw()` extrinsic.
- Same TAO token, different account format, no wrapping involved.

### Precompiles Used by This Contract

| Precompile | Address | Purpose |
|---|---|---|
| Ed25519Verify | `0x0000000000000000000000000000000000000402` | Verify Ed25519 signatures (prove SS58 key ownership from EVM) |
| INeuron | `0x0000000000000000000000000000000000000804` | `burnedRegister` -- register the contract on-chain |
| StakingV2 | `0x0000000000000000000000000000000000000805` | `addStake`, `removeStake`, `getStake`, `transferStake`, `moveStake`, `burnAlpha` |

**Staking precompile caveat:** When a smart contract calls the staking precompile via `call`, the **contract's EVM address** (mapped to its Substrate mirror via HashedAddressMapping) is the coldkey. Use `delegatecall` to preserve the original caller's identity.

### StakingV2 Key Functions

| Function | Signature | What It Does |
|---|---|---|
| `addStake` | `(bytes32 hotkey, uint256 amount, uint256 netuid)` payable | Stake TAO into subnet, receive alpha |
| `removeStake` | `(bytes32 hotkey, uint256 amount, uint256 netuid)` payable | Unstake alpha, receive TAO back |
| `getStake` | `(bytes32 hotkey, bytes32 coldkey, uint256 netuid) -> uint256` view | Query alpha balance |
| `transferStake` | `(bytes32 destination_coldkey, bytes32 hotkey, uint256 origin_netuid, uint256 destination_netuid, uint256 amount)` payable | Transfer alpha ownership to another coldkey |
| `moveStake` | `(bytes32 origin_hotkey, bytes32 destination_hotkey, uint256 origin_netuid, uint256 destination_netuid, uint256 amount)` payable | Re-delegate alpha to another hotkey (same coldkey) |
| `burnAlpha` | `(bytes32 hotkey, uint256 amount, uint256 netuid)` payable | Burn alpha tokens |

### Network Config

| Network | RPC URL | Chain ID |
|---|---|---|
| Mainnet | `https://lite.chain.opentensor.ai` | 964 |
| Testnet | `https://test.chain.opentensor.ai` | 945 |
| Localnet | `http://localhost:9944` | 42 |

### Unit Conversion

- EVM side: 1 TAO = 1e18 (18 decimals, like ETH wei)
- Substrate side: 1 TAO = 1e9 RAO

## Alpha Tokens

Each Bittensor subnet has its own **alpha token**. Alpha is NOT an ERC-20 -- it exists as **staked balances on the Substrate layer**, tracked per `(hotkey, coldkey, netuid)` tuple.

- Each subnet is an **AMM** with two reserve pools: TAO and Alpha
- **Price** = TAO_in_reserve / Alpha_in_reserve
- Staking TAO into a subnet swaps TAO for alpha via the AMM
- Unstaking swaps alpha back to TAO (with slippage)
- Each alpha has a **21M max supply** (same as TAO), with its own halving schedule starting from subnet creation
- Alpha emission starts at ~1 alpha/block per subnet, halving independently
- **Coldkey controls alpha** -- to give someone ownership of alpha, you must `transferStake` to their coldkey. `moveStake` only changes the hotkey (validator delegation) but keeps the same coldkey owner.

## This Crate

### Architecture

- **Solidity contracts** (`src/CollateralUpgradeable.sol`): Upgradeable ERC1967 proxy pattern via OpenZeppelin
- **Rust library** (`src/lib.rs`): Alloy-based contract bindings generated via `sol!` macro from ABI JSON
- **CLI** (`src/main.rs`): `collateral-cli` for all contract operations (deposit, reclaim, slash, query, events)

### Contract Identity Model

| Parameter | Type | What It Is |
|---|---|---|
| `CONTRACT_COLDKEY` | `bytes32` | Substrate coldkey. The contract's owner identity on the Substrate staking side. Derived once in `initialize()` from AddressMapping precompile (`0x...080C`) using `address(this)`. |
| `VALIDATOR_HOTKEY` | `bytes32` | Substrate validator hotkey. Where the contract consolidates all alpha collateral. Set at `initialize()`. |
| `TRUSTEE` | `address` (H160) | EVM address with admin powers: slash, deny reclaims, burn-register. |
| `NETUID` | `uint16` | The Basilica subnet ID. All alpha operations use this netuid. |

Both miners and the trustee need TAO in their H160 wallets for gas.

### Dual-Mode Collateral State

Per `(hotkey, nodeId)` the contract tracks:
- `collaterals` -- total TAO locked (in wei, 1e18 = 1 TAO)
- `alphaCollaterals` -- total alpha locked (in alpha units from staking precompile)
- `collateralUnderPendingReclaims` -- TAO reserved for pending reclaims
- `alphaCollateralUnderPendingReclaims` -- alpha reserved for pending reclaims
- `nodeToMiner` -- the miner's H160 address (set on first deposit, cleared when all four balances are zero)

### TAO Collateral Flow

- **Deposit:** Miner sends TAO as `msg.value`. Recorded in `collaterals[hotkey][nodeId]`.
- **Reclaim:** TAO sent back to miner via `payable(miner).call{value: amount}`.
- **Slash:** TAO sent to `address(0)` (burned).

### Alpha Collateral Flow

**Deposit (`transferAlpha`):**
1. Miner has alpha staked to `alphaHotkey` under their own coldkey on the Basilica subnet.
2. Contract calls `IStaking.transferStake` via **`delegatecall`** -- preserves miner's identity as origin, so precompile sees the miner's coldkey. Alpha moves from miner's coldkey to `CONTRACT_COLDKEY` under `alphaHotkey`.
3. Actual amount received = `newContractStake - oldContractStake` (swap fees may reduce it).
4. If `alphaHotkey != VALIDATOR_HOTKEY`, calls `IStaking.moveStake` via **`call`** to consolidate alpha from `alphaHotkey` to `VALIDATOR_HOTKEY` (uses contract's identity as coldkey).
5. Recorded in `alphaCollaterals[hotkey][nodeId]`.

**Reclaim (`withdrawAlpha`):**
1. Calls `IStaking.transferStake(alphaColdkey, VALIDATOR_HOTKEY, NETUID, NETUID, alphaAmount)` via **`call`**.
2. This changes **only the coldkey ownership** — alpha moves from `(VALIDATOR_HOTKEY, CONTRACT_COLDKEY, NETUID)` to `(VALIDATOR_HOTKEY, miner's alphaColdkey, NETUID)`.
3. The alpha remains staked under `VALIDATOR_HOTKEY`. It is **not** unstaked or converted back to TAO. The miner must separately call `removeStake` to convert to TAO, or `moveStake` to re-delegate to a different hotkey.

**Slash:**
- Alpha transferred to trustee's coldkey (`TRUSTEE_COLDKEY`) via `transferStake` (not left locked in the contract). Same as reclaim — only coldkey ownership changes, alpha stays staked under `VALIDATOR_HOTKEY`.

### `delegatecall` vs `call` (Critical)

| Call Type | Precompile sees as origin | Used When |
|---|---|---|
| `delegatecall` | Original `msg.sender` (miner's EVM address -> miner's Substrate mirror) | `transferAlpha` step 1: moving alpha FROM miner's coldkey TO contract's coldkey |
| `call` | Proxy contract's EVM address (-> contract's Substrate mirror) | `moveStake`: consolidating alpha under `VALIDATOR_HOTKEY`; `withdrawAlpha`: transferring alpha ownership from contract's coldkey to miner's coldkey |

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
