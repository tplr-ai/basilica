# Collateral Contract

This contract is derived from the upstream project at [Datura-ai/celium-collateral-contracts](https://github.com/Datura-ai/celium-collateral-contracts/) and adapted for Basilica. It enables subnet owners to require miners to lock collateral as assurance of service quality.

> **Terminology Note**: This contract uses "executor" terminology to refer to **GPU nodes** (individual GPU machines). The Basilica executor binary has been deprecated and replaced with direct SSH access to GPU nodes. When you see "executor UUID" or "executor ID" in this document, it refers to the UUID/ID of a specific GPU node, not the deprecated executor software.

> **Purpose**: Manage miner collateral in the Bittensor ecosystem and enable the subnet owner (contract admin)—or an explicitly authorized slasher—to penalize misbehavior. Slashing authority is currently centralized to the admin; future upgrades may delegate or decentralize this capability.

> **Design**: One collateral contract per one subnet.

This smart contract is **generic** and works with **any Bittensor subnet**.

We provide the CLI to interact with collateral contract, the details could be found in [`README.md`](/crates/collateral-contract/README.md)

## ⚖️ A Note on Slashing Philosophy

The power to slash collateral carries weight — it protects subnet quality, but also risks abuse if unchecked.  
This contract encourages **automated enforcement** wherever possible, ensuring consistency and fairness across validators.

Manual slashing is supported for now where misbehavior is clear. For slash transaction, there is a link to explain why it is slashed. It is clear and transparent.

However, subnet owner should approach this capability **with restraint and responsibility**.  
Every manual slash must be:

- **Justified** — supported by strong evidence (logs, signatures, links).
- **Transparent** — the justification URL and content hash are stored on-chain.
- **Proportional** — reflecting the severity and intent of the violation.

Whenever possible in the future, validators are encouraged to **automate detection and slashing logic** so that actions are data-driven and reproducible.
Automation helps ensure miners are treated consistently across validators — and enables **retroactive enforcement** without requiring on-the-spot judgment.

Slashing is a **last-resort accountability tool**, not a convenience.

This model is designed for **trust-minimized collaboration**, not permissionless aggression.  
Use slashing to **protect the network**, not to punish disagreement.

## Overview

This contract creates a **trust-minimized interaction** between miners and validators in the Bittensor ecosystem.

- **Miners Lock Collateral**

  Miners demonstrate their commitment by staking collateral into the validator's contract. Miners can now specify a **GPU node UUID** (referred to as "executor UUID" in the contract) during deposit to associate their collateral with specific GPU nodes.

- **Collateral-Based Prioritization**

  Validators may choose to favor miners with higher collateral when assigning tasks, incentivizing greater stakes for reliable performance.

- **Admin Slashing**
  The subnet owner (contract admin) or an explicitly authorized slasher can penalize a misbehaving miner by slashing some or all of the miner's collateral.

- **Automatic Release**

  If the authorized slasher/admin does not respond to a miner's reclaim request within a configured deadline, the miner can reclaim their stake, preventing indefinite lock-ups.

- **Trustless & Auditable**

  All operations (deposits, reclaims, slashes) are publicly logged on-chain, enabling transparent oversight for both validators and miners.

- **Off-Chain Justifications**

  Functions `slashCollateral`, `reclaimCollateral`, and `denyReclaim` include URL fields (and content MD5 checksums) to reference off-chain
  explanations or evidence for each action, ensuring decisions are transparent and auditable.

- **Configurable Minimum Bond & Decision Deadline**
  Defines a minimum stake requirement and a strict timeline for admin/slasher responses.

> **Important notice on addressing**
>
> This contract uses **H160 (Ethereum) addresses** for miner and validator identities.
>
> - Before interacting with the contract, participants must control an Ethereum-compatible wallet (H160) to sign transactions.
> - We recommend associating your H160 wallet with your **SS58 hotkey** to help validators reliably identify miners.
> - Converting an H160 to an SS58 representation does not grant control of funds or keys; it is a mapping for identification. To formally link identities, use the Subtensor extrinsic `associate_evm_key` (see the Subtensor source: [associate_evm_key](https://github.com/opentensor/subtensor/blob/main/pallets/subtensor/src/macros/dispatches.rs#L2001)).

> - Note: deriving an SS58 from an H160 does not imply key control. To assert linkage on-chain, use `associate_evm_key` as referenced above.

> **Transaction Fees**
>
> All on-chain actions (deposits, slashes, reclaims, etc.) consume gas, so **both miners and validators must hold enough TAO in their Ethereum (H160) wallets** to cover transaction fees.
>
> - Make sure to keep a sufficient balance to handle any deposits, reclaims, or slashes you need to perform.
> - Convert H160 to SS58 [`convertH160ToSS58`](https://github.com/opentensor/subtensor/blob/main/evm-tests/src/address-utils.ts#L14) to transfer TAO to it.
> - You can transfer TAO back to your SS58 wallet when no more contract interactions are required. See [`transfer token from EVM to Substrate`](https://github.com/opentensor/subtensor/blob/main/evm-tests/test/eth.substrate-transfer.test.ts#L78).

## Demo

## Collateral Smart Contract Lifecycle

Below is a typical sequence for integrating and using this collateral contract within a Bittensor subnet:

- **Subnet Integration**

  - The subnet owner **updates validator software** to prioritize miners with higher collateral when assigning tasks.
  - Validators adopt this updated code and prepare to enforce collateral requirements.

- **Owner Deployment**

  - The owner **creates an Ethereum (H160) wallet**, links it to their hotkey, and funds it with enough TAO to cover transaction fees.
  - The owner **deploys the contract**, requiring participating miners to stake collateral.
  - The owner **publishes the contract address** on-chain, allowing miners to discover and verify it.
  - Once ready, the owner **enables collateral-required mode** and prioritizes miners based on their locked amounts.

- **Miner Deposit**

  - Each miner **creates an Ethereum (H160) wallet**, links it to their hotkey, and funds it with enough TAO for transaction fees.
  - Miners **retrieve** the owner's contract address from the chain or another trusted source.
  - Upon confirmation, miners **deposit** collateral by calling the contract's `deposit(executorUuid)` function, specifying the **GPU node UUID** (labeled as "executor UUID" in the contract interface) to associate the collateral with specific GPU nodes.
  - Confirm on-chain that your collateral has been successfully locked for that GPU node

- **Slashing Misbehaving Miners**
  If a miner is found violating subnet rules (e.g., returning invalid responses), the subnet owner (admin) or an authorized slasher **calls** `slashCollateral()` for a full slash or `slashCollateralAmount()` for a partial slash, providing the `hotkey`, `executorUuid`, `amount` (partial only), and justification details to reduce the miner’s collateral.
  Partial slashing requires the upgraded contract that includes `slashCollateralAmount` plus regenerated ABI/CLI bindings.

- **Reclaiming Collateral**
  - When miners wish to withdraw their stake, they **initiate a reclaim** by calling `reclaimCollateral()`, specifying the **GPU node UUID** (labeled as "executor UUID" in the contract) associated with the collateral.
  - If the validator does not deny the request before the deadline, miners (or anyone) can **finalize** it using `finalizeReclaim()`, thus unlocking and returning the collateral.

## Usage Guides

Below are step-by-step instructions tailored to **miners**, **validators**, and **subnet owners**.
Refer to the repository's [`collateral_contract/`](/crates/collateral-contract/) folder for sample implementations and helper scripts.
You need replace the variable with the correct value like contract address.

## As a Miner, you can:

- **Deposit Collateral**
  If you plan to stake for multiple validators, simply repeat these steps for each one:

  - Obtain the validator's contract address (usually via tools provided by the subnet owner).
  - Verify that code deployed at the address is indeed the collateral smart contract, the trustee and netuid kept inside are as expected - see [`query.sh`](/crates/collateral-contract/query.sh).

  ```shell
    #!/usr/bin/env bash
    set -euo pipefail

    # basic query to verify the contract is deployed and initialized
    export CONTRACT_ADDRESS=0x
    export NETWORK=mainnet

    collateral-cli --network "$NETWORK" --contract-address "$CONTRACT_ADDRESS" query trustee
    collateral-cli --network "$NETWORK" --contract-address "$CONTRACT_ADDRESS" query min-collateral-increase
    collateral-cli --network "$NETWORK" --contract-address "$CONTRACT_ADDRESS" query decision-timeout
    collateral-cli --network "$NETWORK" --contract-address "$CONTRACT_ADDRESS" query netuid
  ```

  - Run deposit command to initiate the deposit transaction with your specified amount of $TAO. running `collateral-cli tx deposit`, reference in [`flow.sh`](/crates/collateral-contract/flow.sh).

  ```shell
  #!/usr/bin/env bash
  set -euo pipefail

  # the whole collateral flow to verify everything
  export NETWORK=local
  export CONTRACT_ADDRESS=0x
  export HOTKEY=0x
  export EXECUTOR_ID=6339ba4f-60f9-45c2-9d95-2b755bb57ca6
  # WARNING: never commit or paste real keys in scripts
  export PRIVATE_KEY=0x
  # deposit
  collateral-cli --network "$NETWORK" --contract-address "$CONTRACT_ADDRESS" tx deposit \
  --private-key "$PRIVATE_KEY" \
  --hotkey "$HOTKEY" \
  --amount 10 \
  --executor-id "$EXECUTOR_ID"
  ```

  - Confirm on-chain that your collateral has been successfully locked for that validator. running `collateral-cli query executor-to-miner` and `collateral-cli query collaterals`, reference in [`flow.sh`](/crates/collateral-contract/flow.sh)

  ```shell
  #!/usr/bin/env bash
  set -euo pipefail
  export NETWORK=local
  export CONTRACT_ADDRESS=0x
  export HOTKEY=0x
  export EXECUTOR_ID=6339ba4f-60f9-45c2-9d95-2b755bb57ca6


  # check the executor to miner, miner is not zero if deposit is successful

  collateral-cli --network "$NETWORK" --contract-address "$CONTRACT_ADDRESS" query executor-to-miner \
  --hotkey "$HOTKEY" \
  --executor-id "$EXECUTOR_ID"

  # check the collaterals should be amount you deposit

  collateral-cli --network "$NETWORK" --contract-address "$CONTRACT_ADDRESS" query collaterals \
  --hotkey "$HOTKEY" \
  --executor-id "$EXECUTOR_ID"
  ```

- **Reclaim Collateral**
- Initiate the reclaim process by running `collateral-cli tx reclaim-collateral`. reference in [`flow.sh`](/crates/collateral-contract/flow.sh).
- Wait for the validator's response or for the configured inactivity timeout to pass.
- If the validator does not deny your request by the deadline, running `collateral-cli tx finalize-reclaim`. reference in [`flow.sh`](/crates/collateral-contract/flow.sh).
- Verify on-chain that your balance has been updated accordingly.

### As a validator.

The validators won't evaluate or list the miners' executors as available, if the miner hasn't backed their executors with a collateral stake. Thus, the miner will only receive weights, based on the available executors.

### As a Owner, you can:

- **Deploy the Contract**
  This contract uses the **UUPS (Universal Upgradeable Proxy Standard) proxy pattern** to enable seamless upgrades without losing contract state.
  With UUPS, the proxy contract holds all storage and delegates logic to an implementation contract. When you upgrade, you deploy a new implementation and point the proxy to it—**all balances and mappings are preserved**.

- Install [Foundry](https://book.getfoundry.sh/).
  ```shell
  # Install Forge
  curl -L https://foundry.paradigm.xyz | bash
  source /home/ubuntu/.bashrc  # Or start a new terminal session
  foundryup
  forge --version
  forge init
  # Add upgradeable proxy dependencies
  forge install OpenZeppelin/openzeppelin-contracts
  forge install OpenZeppelin/openzeppelin-contracts-upgradeable
  forge build
  ```
- Clone this repository.
- Install project dependencies:
  ```shell
  cargo build --release
  ```
- Compile and deploy the contract, use [`deploy.sh`](/crates/collateral-contract/deploy.sh) with your details as arguments.

```shell
#!/usr/bin/env bash
set -euo pipefail

export NETUID=39
export TRUSTEE_ADDRESS=0xf24FF3a9CF04c71Dbc94D0b566f7A27B94566cac
export MIN_COLLATERAL=1
export DECISION_TIMEOUT=1
export ADMIN_ADDRESS=0xf24FF3a9CF04c71Dbc94D0b566f7A27B94566cac
# WARNING: never commit or paste real keys in scripts
export PRIVATE_KEY=0x
# export RPC_URL=https://lite.chain.opentensor.ai:443
# export RPC_URL=https://test.finney.opentensor.ai
export RPC_URL=http://localhost:9944
forge script script/DeployUpgradeable.s.sol \
 --rpc-url "$RPC_URL" \
 --private-key "$PRIVATE_KEY" \
 --broadcast
```

- Record the deployed contract address and publish it.

## FAQ

### Why should miners deposit into the smart contract?

Depositing collateral not only demonstrates a miner's commitment to the network and ensures accountability but also enables them to become eligible for mining rewards. The miners who didn't deposit collateral or penalized won't get any rewards.

### When will a miner's deposit be slashed?

Subnet owner will slash when miner stops providing GPU node access or fails to meet service commitments, such as when a customer loses SSH access to the rental container. In the future, all validators will take the responsibility and privilege to slash.

### When will a miner's reclaim request be declined?

Miner's reclaim request will be declined when their GPU node is rented by a customer on the platform.

### What will happen when a miner's deposit is slashed?

Miner will lose the deposited amount for the violating GPU node; the miner needs to deposit for that GPU node again if they want to keep getting rewards for it.
