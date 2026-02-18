## Collateral Contract TODOs (feat/incentive_revamp)

### Security Audit (2026-02-06): Critical/High Collateral-Loss Findings

1) ~~Critical - Alpha pending-reclaim counter is never decremented on finalize~~ **DONE**
- Problem: `reclaimCollateral` increments `alphaCollateralUnderPendingReclaims`, but `finalizeReclaim` only decrements `collateralUnderPendingReclaims` and never clears the alpha pending counter. After one successful alpha reclaim, later alpha deposits on the same `(hotkey, nodeId)` can become non-reclaimable due to stale pending alpha accounting.
- Impact: legitimate miner alpha collateral can become permanently locked.
- Fix:
  - In `finalizeReclaim`, decrement `alphaCollateralUnderPendingReclaims[hotkey][nodeId]` by `alphaAmount` before external calls.
  - Enforce invariant checks that pending alpha never exceeds tracked alpha.
- Regression test:
  - alpha deposit -> reclaim -> finalize -> alpha deposit again -> reclaim/finalize must still succeed.
- Files: `crates/collateral-contract/src/CollateralUpgradeable.sol`

2) ~~High - Pending reclaim plus slash can deadlock collateral (non-finalizable and non-deniable)~~ **DONE**
- Problem: `slashCollateral` mutates live balances while pending reclaim keeps snapshot amounts. Later `finalizeReclaim` can revert when snapshot reclaim amount exceeds remaining balance. For alpha-only requests, `denyReclaimRequest` currently treats `reclaim.amount == 0` as "not found", preventing trustee cleanup.
- Impact: remaining collateral can become permanently stuck (cannot be finalized by miner and cannot be denied by trustee).
- Fix:
  - Change reclaim existence check to: `if (reclaim.amount == 0 && reclaim.alphaAmount == 0) revert ReclaimNotFound();`
  - Implemented partial finalize: `finalizeReclaim` caps transfer to available balance instead of reverting. Slash and reclaim are now fully independent operations.
- Regression tests:
  - pending alpha reclaim + partial alpha slash must not leave unrecoverable remainder.
  - alpha-only reclaim must be deny-able before timeout.
- Files: `crates/collateral-contract/src/CollateralUpgradeable.sol`, `crates/collateral-contract/src/lib.rs`, `crates/basilica-validator/src/collateral/slash_executor.rs`

### ~~Upgrade Safety (Critical)~~ **DONE**
- Problem: Storage layout is not append‑only. New fields (`CONTRACT_COLDKEY`, `VALIDATOR_HOTKEY`, `alphaCollaterals`, `alphaCollateralUnderPendingReclaims`, plus new fields in `Reclaim`) are inserted before existing mappings/struct fields, which will corrupt state on upgrade for any already‑deployed proxy.
  Fix:
  - Contract is not yet deployed, so current layout becomes canonical V1 (no reordering needed).
  - Added `uint256[50] private __gap` storage gap after `nextReclaimId` for future upgrade safety.
  Files: `crates/collateral-contract/src/CollateralUpgradeable.sol`

### ~~Reclaim Deny Logic (High)~~ **DONE**
- Problem: `denyReclaimRequest` treats `reclaim.amount == 0` as "not found". Alpha‑only reclaims have `amount == 0`, so they cannot be denied.
  Fix:
  - Change existence check to `if (reclaim.amount == 0 && reclaim.alphaAmount == 0) revert ReclaimNotFound();`
  Files: `crates/collateral-contract/src/CollateralUpgradeable.sol`

### ~~Pending Alpha Accounting + Underflow (High)~~ **DONE**
- Problem: `finalizeReclaim` never decrements `alphaCollateralUnderPendingReclaims`. Also no alpha‑side sufficiency check is performed before subtracting `alphaAmount`, so slashes during the pending window can cause underflow/revert.
  Fix:
  - Decrement `alphaCollateralUnderPendingReclaims[hotkey][nodeId]` in `finalizeReclaim`.
  - Instead of reverting on insufficiency, `finalizeReclaim` now caps the transfer to the available balance (partial finalize), preventing both underflow and deadlock.
  Files: `crates/collateral-contract/src/CollateralUpgradeable.sol`

### ~~Zero‑Value Deposit Ownership (Medium)~~ **DONE**
- Problem: `deposit` allows `msg.value == 0` and `alphaAmount == 0`, yet still claims `nodeToMiner`, enabling ownership griefing without collateral.
  Fix:
  - Added `if (msg.value == 0 && alphaAmount == 0) revert AmountZero()` guard at top of `deposit()`.
  Files: `crates/collateral-contract/src/CollateralUpgradeable.sol`

### ~~Contract Coldkey Mutability (Medium)~~ **DONE**
- Problem: `setContractColdkey` can change the coldkey even if the contract already holds stake, potentially orphaning existing alpha collateral.
  Fix:
  - Removed `setContractColdkey` entirely.
  - `CONTRACT_COLDKEY` is now derived once in `initialize()` via AddressMapping precompile (`0x...080C`) using `address(this)` and is never externally mutable.
  Files: `crates/collateral-contract/src/CollateralUpgradeable.sol`, `crates/collateral-contract/src/lib.rs`, `crates/collateral-contract/src/main.rs`

### ~~Partial Slash Persistence Desync (Medium)~~ **DONE**
- Problem: Validator persistence set `miner=0` on every slash, but on‑chain clears `nodeToMiner` only when collateral ownership is truly exhausted, causing desync after partial slashes.
  Fix:
  - Ownership clearing is now rule-based: only set `miner=0` when live and pending collateral are all zero (`tao_collateral == 0 && alpha_collateral == 0 && pending_tao_reclaim == 0 && pending_alpha_reclaim == 0`).
  - Added regression coverage for full slash while pending reclaim exists; miner remains set until pending is resolved.
  Files: `crates/basilica-validator/src/persistence/collateral_persistence.rs`

### ~~Evidence Checksum Consistency (Integration)~~ **DONE**
- Problem: On-chain checksum fields previously used MD5 naming while validator slash execution submitted a truncated SHA-256 value.
  Fix:
  - Standardized all checksum interfaces on full SHA-256 (`bytes32` / `[u8; 32]`) and renamed fields to `urlContentSha256`.
  - Updated contract ABI/events, CLI parsing/flags (`--url-content-sha256`), validator slash checksum computation, and persistence column naming to `url_content_sha256`.
  - Added migration `018_rename_md5_checksum_to_sha256.sql` to rename the persistence column and null out non-64-hex legacy values.
  Files: `crates/collateral-contract/src/CollateralUpgradeable.sol`, `crates/collateral-contract/src/main.rs`, `crates/collateral-contract/src/lib.rs`, `crates/basilica-validator/src/collateral/slash_executor.rs`, `crates/basilica-validator/src/persistence/collateral_persistence.rs`

### ABI/CLI Updates (Integration) **PARTIALLY RESOLVED**
- Problem: CLI and Rust bindings now send only alpha collateral and ignore TAO `msg.value`, while contract still supports TAO. This can cause mixed‑collateral state drift.
  Done:
  - Validator-side drift is resolved: event sync now tracks both TAO and alpha state, includes reclaim lifecycle events (`ReclaimProcessStarted`, `Denied`), and persists pending reclaim amounts to mirror on-chain lifecycle semantics.
  - Added event-coverage and lifecycle tests so newly supported events must remain wired in persistence dispatch.
  Still to do:
  - Decide product direction explicitly:
    - If TAO is deprecated: remove TAO paths from contract ABI and bindings (or gate/disable them), and document alpha-only behavior as canonical.
    - If TAO remains supported: extend CLI/Rust tx interfaces to accept TAO inputs and set nonzero `msg.value` for TAO deposits; keep slash/reclaim tooling consistent across both assets.
  - Deferred in this pass: no TAO/alpha product-direction changes were made while standardizing evidence checksums.
  - Align docs/operator playbooks with whichever direction is chosen so tx behavior and validator sync guarantees are unambiguous.
  Files: `crates/collateral-contract/src/lib.rs`, `crates/collateral-contract/src/main.rs`
