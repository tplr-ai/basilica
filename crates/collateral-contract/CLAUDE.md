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
| AddressMapping | `0x000000000000000000000000000000000000080C` | `addressMapping` -- derive Substrate AccountId32 from EVM H160 address |
| INeuron | `0x0000000000000000000000000000000000000804` | `burnedRegister` -- register the contract on-chain |
| StakingV2 | `0x0000000000000000000000000000000000000805` | Staking operations: stake, unstake, query, transfer, and move alpha |

**Staking precompile caveat:** When a smart contract calls the staking precompile via `call`, the **contract's EVM address** (mapped to its Substrate mirror via HashedAddressMapping) is the coldkey. Use `delegatecall` to preserve the original caller's identity.

### StakingV2 Key Functions

**CRITICAL: All StakingV2 precompile `amount` parameters use RAO (1e9 per TAO), NOT wei (1e18).** Despite being called from the EVM side, these precompiles bridge to Substrate which uses RAO. Passing wei values will either revert with `NotEnoughBalanceToStake` (amount too large) or stake a negligible amount (amount too small). Do NOT pass `msg.value` -- these functions deduct from the caller's free balance directly.

| Function | Signature | What It Does |
|---|---|---|
| `addStake` | `(bytes32 hotkey, uint256 amount, uint256 netuid)` payable | Stake TAO into subnet, receive alpha. `amount` in **RAO**. |
| `removeStake` | `(bytes32 hotkey, uint256 amount, uint256 netuid)` payable | Unstake alpha, receive TAO back. `amount` in **RAO**. |
| `getStake` | `(bytes32 hotkey, bytes32 coldkey, uint256 netuid) -> uint256` view | Query alpha balance. Returns **RAO**. |
| `transferStake` | `(bytes32 destination_coldkey, bytes32 hotkey, uint256 origin_netuid, uint256 destination_netuid, uint256 amount)` payable | Transfer alpha ownership to another coldkey. `amount` in **RAO**. |
| `moveStake` | `(bytes32 origin_hotkey, bytes32 destination_hotkey, uint256 origin_netuid, uint256 destination_netuid, uint256 amount)` payable | Re-delegate alpha to another hotkey (same coldkey). `amount` in **RAO**. |
| `burnAlpha` | `(bytes32 hotkey, uint256 amount, uint256 netuid)` payable | Burn alpha tokens. `amount` in **RAO**. |

### StakingV2 Transfer Guarantees (Same-Subnet)

**Both `transferStake` and `moveStake` are 1:1 when `origin_netuid == destination_netuid`.** This contract always uses the same `netuid` for both parameters, so these guarantees apply to all contract operations.

**Code path** (verified in subtensor source):
1. EVM precompile (`precompiles/src/staking.rs`) — pure pass-through to Substrate pallet, no transformations
2. Pallet (`pallets/subtensor/src/staking/move_stake.rs`) — `transition_stake_internal` branches on netuid equality
3. Same-subnet branch calls `transfer_stake_within_subnet` (`pallets/subtensor/src/staking/stake_utils.rs:850`) — direct share pool debit/credit, no AMM

In `transfer_stake_within_subnet`, the debited amount feeds directly into the credit:
```
actual_alpha_decrease = decrease_stake_for_hotkey_and_coldkey_on_subnet(origin, alpha)
actual_alpha_moved = increase_stake_for_hotkey_and_coldkey_on_subnet(destination, actual_alpha_decrease)
```
Events explicitly emit `0_u64` for the fee field. Source comments confirm: *"no slippage in this move"*.

| Property | Same-subnet (this contract) | Cross-subnet |
|---|---|---|
| AMM swap | **No** | Yes (alpha->TAO->alpha) |
| 0.05% liquidity fee | **No** | Yes |
| Slippage | **No** | Yes |
| Destination = Origin amount | **Yes** | No |

**Edge cases (negligible):** Share pool uses `U64F64` fixed-point arithmetic, which can introduce sub-RAO (< 1e-9 TAO) rounding. Transaction fees are paid separately in TAO, not deducted from the moved alpha.

### No Runtime-Level Validator Slashing

Bittensor does **not** slash validator stake at the protocol level. There is no mechanism in Subtensor that automatically reduces staked alpha/TAO as a penalty. Verified by inspecting `pallets/subtensor/src/staking/` — no slash functions exist for staking balances.

The only ways stake can decrease:
- Explicit `removeStake` (unstaking)
- Explicit `transferStake` or `moveStake`
- Deregistration removes the validation *permit* but **does not touch stake balances**

**Implication for this contract:** The internal `alphaCollaterals` accounting will not desync from the live precompile stake due to external events. The `withdrawAlpha` revert path (`contractStake < alphaAmount`) should never trigger from runtime-side balance reduction, only from contract-side slashing that reduces `alphaCollaterals` without a corresponding `transferStake`.

### Network Config

| Network | RPC URL | Chain ID |
|---|---|---|
| Mainnet | `https://lite.chain.opentensor.ai` | 964 |
| Testnet | `https://test.chain.opentensor.ai` | 945 |
| Localnet | `http://localhost:9944` | 42 |

### Unit Conversion

- **TAO on EVM**: 1 TAO = 1e18 (18 decimals, like ETH wei). Used by: `msg.value`, `address.balance`, ERC-20 amounts, Solidity contract storage (e.g. `taoCollaterals`).
- **TAO on Substrate**: 1 TAO = 1e9 RAO. Used by: **all StakingV2 precompile parameters**, btcli commands, Substrate RPC.
- **Alpha (everywhere)**: 1 Alpha = 1e9 RAO. Alpha has **no 1e18 representation** — it is always 1e9 RAO whether accessed from EVM precompiles or Substrate. Both `TaoCurrency` and `AlphaCurrency` are `u64` wrappers on-chain.

**The precompile boundary is the conversion point.** Solidity code that stores TAO in wei (1e18) must convert to RAO (1e9) before calling staking precompiles, and convert back when reading `getStake` results. The collateral contract's `alphaCollaterals` stores values in RAO (the unit returned by staking precompiles).

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

| State Variable | Type | What It Is |
|---|---|---|
| `contractColdkey` | `bytes32` | Substrate coldkey. The contract's owner identity on the Substrate staking side. Derived once in `initialize()` from AddressMapping precompile (`0x...080C`) using `address(this)`. |
| `validatorHotkey` | `bytes32` | Substrate validator hotkey. Where the contract consolidates all alpha collateral. Set at `initialize()`. |
| `trustee` | `address` (H160) | EVM address with admin powers: slash, deny reclaims, burn-register. Updatable via `updateTrustee()`. |
| `netuid` | `uint16` | The Basilica subnet ID. All alpha operations use this netuid. |

Both miners and the trustee need TAO in their H160 wallets for gas.

### Dual-Mode Collateral State

Per `(hotkey, nodeId)` the contract tracks:
- `taoCollaterals` -- total TAO locked (in wei, 1e18 = 1 TAO)
- `alphaCollaterals` -- total alpha locked (in RAO, 1e9 = 1 alpha, same scale as TAO)
- `taoCollateralUnderPendingReclaims` -- TAO reserved for pending reclaims
- `alphaCollateralUnderPendingReclaims` -- alpha reserved for pending reclaims
- `nodeToMiner` -- the miner's H160 address (set on first deposit, cleared when all four balances are zero)

### TAO Collateral Flow

- **Deposit:** Miner sends TAO as `msg.value`. Recorded in `taoCollaterals[hotkey][nodeId]`.
- **Reclaim:** TAO sent back to miner via `payable(miner).call{value: amount}`.
- **Slash:** TAO sent to the trustee's EVM address.

### Alpha Collateral Flow

**Deposit (`transferAlpha`):**
1. Miner has alpha staked to `alphaHotkey` under their own coldkey on the Basilica subnet.
2. Contract calls `IStaking.transferStake` via **`delegatecall`** -- preserves miner's identity as origin, so precompile sees the miner's coldkey. Alpha moves from miner's coldkey to `contractColdkey` under `alphaHotkey`.
3. Actual amount received = `newContractStake - oldContractStake` (swap fees may reduce it).
4. If `alphaHotkey != validatorHotkey`, calls `IStaking.moveStake` via **`call`** to consolidate alpha from `alphaHotkey` to `validatorHotkey` (uses contract's identity as coldkey).
5. Recorded in `alphaCollaterals[hotkey][nodeId]`.

**Reclaim (`withdrawAlpha`):**
1. Calls `IStaking.transferStake(alphaColdkey, validatorHotkey, netuid, netuid, alphaAmount)` via **`call`**.
2. This changes **only the coldkey ownership** — alpha moves from `(validatorHotkey, contractColdkey, netuid)` to `(validatorHotkey, miner's alphaColdkey, netuid)`.
3. The alpha remains staked under `validatorHotkey`. It is **not** unstaked or converted back to TAO. The miner must separately call `removeStake` to convert to TAO, or `moveStake` to re-delegate to a different hotkey.

**Slash:**
- Alpha transferred to the trustee's derived coldkey via `transferStake` (not left locked in the contract). Same as reclaim — only coldkey ownership changes, alpha stays staked under `validatorHotkey`.

### `delegatecall` vs `call` (Critical)

| Call Type | Precompile sees as origin | Used When |
|---|---|---|
| `delegatecall` | Original `msg.sender` (miner's EVM address -> miner's Substrate mirror) | `transferAlpha` step 1: moving alpha FROM miner's coldkey TO contract's coldkey |
| `call` | Proxy contract's EVM address (-> contract's Substrate mirror) | `moveStake`: consolidating alpha under `validatorHotkey`; `withdrawAlpha`: transferring alpha ownership from contract's coldkey to miner's coldkey |

### Deployment

- Mainnet EVM deployment whitelist is **disabled** (open to all). Localnet init scripts also disable it automatically via sudo.
- Uses `bash ./deploy.sh` (implementation + ERC1967 proxy via `forge create`)
- Proxy initialization is passed as constructor calldata when deploying `ERC1967Proxy`
- After deployment, update ABI via `update_abi.py`

### Testing

```bash
forge test           # Solidity contract tests
cargo test --lib     # Rust library unit tests
cargo test           # All tests
```
