# PR #372: feat(collateral): harden dual-state collateral contract and validator integration

## Summary

This PR is a substantial change (+8994/-5933 across 74 files) that hardens Basilica's collateral integration across three layers:
1. **Solidity contract** (`CollateralUpgradeable.sol`) -- dual-asset (TAO + alpha) deposit/reclaim/slash lifecycle
2. **Rust validator** -- event ingestion redesigned from block-based scanning to RPC-based state snapshotting, persistence, preference evaluation
3. **CLI/tooling** -- updated bindings, interactive prompts, localnet deployment scripts

A previous review on commit `4a47f3f6` raised 10 critical/high findings, 6 of which were retracted as factually incorrect. Since then, **85+ additional commits** have been pushed addressing the remaining issues and adding new features (paginated snapshots, preference precomputation, graceful shutdown, etc.).

### Testing Evidence

| Suite | Result |
|-------|--------|
| Rust clippy | 0 errors, 1 warning (unused import in SDK, unrelated) |
| Rust unit tests | 308 passed, 8 ignored, 0 failed |
| Rust integration tests | All passed (collateral e2e, slash flow) |
| Solidity tests (Foundry) | **97 passed**, 0 failed, 0 skipped |
| Rust doctests | 1 failure (stale SDK doctest, unrelated to this PR) |

---

## Status of Previously Identified Issues

| ID | Original Claim | Current Status | Evidence |
|----|---------------|----------------|----------|
| **Slash precision bug** (H-6) | Sub-1% fractions cause 100% slash | **FIXED** | `slash_executor.rs:267-287` -- returns `U256::from(1u64)` floor when numerator rounds to 0. Config validation at `collateral.rs:93-96` enforces `slash_fraction >= 0.01` |
| **No graceful shutdown** (C-4) | Scan loop has no CancellationToken | **FIXED** | `collateral_scan.rs:42,63-78` -- CancellationToken with `tokio::select!` branch. `service.rs:480-482` calls `scanner.stop()` on shutdown |
| **MAX_BLOCKS_PER_SCAN unused** (H-5) | Constant defined but never referenced | **FIXED (removed)** | Architecture redesigned from event scanning to RPC-based state snapshotting. Constant no longer exists |
| **DB-chain reconciliation** | Eligibility uses DB only, slash uses chain | **PARTIALLY ADDRESSED** | Redesigned to full-state snapshots from chain (`sync_all_collateral_nodes` with upsert + stale row cleanup). Gap reduced to scan interval window. Slash still verifies on-chain before execution |
| **Deposit idempotency** | Deposit handler lacks duplicate detection | **NO LONGER APPLICABLE** | Redesigned to snapshot-based sync -- each cycle replaces DB state with chain state via `INSERT ... ON CONFLICT ... DO UPDATE`. Inherently idempotent |

---

## Issues Found

### HIGH Severity Issues (Advised to Fix Before Merge)

#### ~~H-1: `deposit_with_config` is an exact duplicate of `deposit`~~ **FIXED**

**Classification**: DRY Violation / Dead Code
**File**: `crates/collateral-contract/src/lib.rs` (removed)

**Resolution**: `deposit_with_config` was removed. It was character-for-character identical to `deposit` with zero callers.

---

#### ~~H-2: `block_in_place` wrapping `block_on` in async context~~ **FIXED**

**Classification**: Concurrency Anti-pattern
**File**: `crates/basilica-validator/src/rental/mod.rs:754-765`

**Resolution**: Replaced `block_in_place`/`block_on` wrapper with a direct `.await` call, since `start_rental_inner` is already `async`.

---

### MEDIUM Severity Issues (Optional to Fix Before Merge)

#### ~~M-4: Two independent `CollateralEvaluator` instances~~ **FIXED**

**File**: `crates/basilica-validator/src/service.rs:312, 444`

**Resolution**: Single `Arc<CollateralEvaluator>` created in `init_collateral_components`, cloned for both `CollateralManager` and `Collateral` scanner via `TaskInputs`.

---

#### M-5: TODO comments in production code

| File | Line | TODO |
|------|------|------|
| `slash_executor.rs` | 147 | `// TODO: Support structured secret payloads (e.g., JSON) and rotation metadata.` |
| `rental/mod.rs` | 186 | `// TODO: Wire this from config for callers using 'new'.` |

---

#### ~~M-6: `f64` used for RAO/alpha conversion in CLI~~ **FIXED**

**File**: `crates/basilica-cli/src/cli/handlers/collateral.rs:118-132`

**Resolution**: `alpha_to_rao` and `rao_to_alpha` rewritten using `rust_decimal::Decimal`. Call sites convert CLI `f64` args via `Decimal::try_from()`.

---

#### ~~M-7: Regex compiled on every call~~ **FIXED**

**File**: `crates/basilica-validator/src/persistence/miner_nodes.rs:48`

**Resolution**: Regex compiled once using `once_cell::sync::Lazy` static (MSRV-compatible alternative to `LazyLock`).

---

#### ~~M-8: Hardcoded multiplier in warning message~~ **FIXED**

**File**: `crates/basilica-validator/src/collateral/evaluator.rs:195`

**Resolution**: Warning message now uses `self.config.warning_threshold_multiplier` instead of hardcoded `"1.5x"`.

---

### LOW Severity Issues (Minor Improvements)

| # | Issue | File | Detail |
|---|-------|------|--------|
| ~~L-1~~ | ~~`evaluate()` is `async` but does no async work~~ | `evaluator.rs:56` | **FIXED**: Removed `async` keyword and `.await` at all 6 call sites |
| ~~L-2~~ | ~~`#[allow(unused_imports)]` on slash_executor exports~~ | `collateral/mod.rs:9` | **FIXED**: Removed allow, removed `CollateralChainClient` from re-export, updated e2e test import |

---

## Security Assessment

### Threat Model

| Attack Vector | Risk | Assessment |
|--------------|------|------------|
| Malicious miner avoiding slash | **Mitigated** | Trustee can deny reclaims; slash operates on total balance, not available amount |
| Compromised trustee key | **Accepted risk** | Full slash authority by design. AWS Secrets Manager, shadow mode, and `updateTrustee` provide mitigations |
| Front-running transactions | **Low** | Bittensor EVM uses AURA/BABE consensus, not MEV-susceptible Ethereum mempool |
| Chain reorg exploits | **Mitigated** | Scanner uses GRANDPA-finalized blocks with cryptographic finality |
| RPC manipulation | **Operational** | Mitigated by secure RPC configuration; slash verifies on-chain state |
| DB tampering | **Mitigated** | DB is local; slash decisions verify on-chain before execution |

### Security Strengths

- CEI pattern correctly followed in all state-modifying contract functions
- `ReentrancyGuard` on all mutative functions
- `receive()` and `fallback()` reject direct ETH with `InvalidDepositMethod()`
- TRUSTEE_ROLE lockdown via `grantRole`/`revokeRole`/`renounceRole` overrides
- EOA-only first deposit enforcement (blocks contract-based and constructor bypass)
- Storage gap (`uint256[49] private _gap`) for upgrade safety
- SQL injection prevention via parameterized `sqlx` bindings
- SHA-256 evidence audit trail for slash decisions

---

## Positive Observations

1. **Architecture redesign from event scanning to RPC snapshots**: The move to `get_all_collaterals_at_block` with pagination and block-pinning eliminates entire classes of bugs (missed events, reorg handling, idempotency). Significant improvement.

2. **Comprehensive Solidity test suite**: 97 tests covering deposits, reclaims, slashes, denials, access control, edge cases, upgrades, and node ownership. The `NodeOwnershipEdgeCases.t.sol` is particularly thorough.

3. **Atomic per-block state sync**: `sync_all_collateral_nodes` uses DB transactions with upsert + stale row cleanup.

4. **Preference precomputation in background**: Moved from rental hot path to background scan loop, improving request latency.

5. **Shadow mode default**: `shadow_mode = true` prevents accidental on-chain slashing during initial deployment.

6. **Config validation**: Comprehensive `validate()` covering R2 evidence config, slash fraction bounds, key source requirements.

7. **Trait-based chain client**: `CollateralChainClient` trait enables clean test mocking.

8. **Finalized block reads**: Scanner uses GRANDPA-finalized blocks, making reorg attacks impossible on Substrate chains.

9. **SQL injection prevention**: All queries use parameterized `sqlx` bindings. Column selection uses `match` not string interpolation.

10. **Evidence audit trail**: SHA-256 checksums on slash evidence with R2 storage, creating immutable records.

11. **Saturating arithmetic throughout**: Both Solidity (OpenZeppelin Math) and Rust (U256 `saturating_add`/`saturating_sub`) prevent over/underflow.

12. **Clean separation of concerns**: Evaluator, manager, executor, scanner each have single responsibilities following SOLID principles.

---

## Recommendation and Next Steps

**The PR is in strong shape for merge**, with the following recommended action items:

### Before merge (2 items):
1. ~~**H-1**: Remove the dead `deposit_with_config` duplicate function~~ **DONE**
2. ~~**H-2**: Replace `block_in_place`/`block_on` with direct `.await` in `rental/mod.rs:754-765`~~ **DONE**

### Recommended follow-up (tracked as issues):
- ~~M-4: Share `CollateralEvaluator` via `Arc` instead of creating two instances~~ **DONE**
- M-5: Resolve TODO comments (left as-is -- legitimate future work items)
- ~~M-6: Switch CLI from `f64` to `Decimal` for RAO conversions~~ **DONE**
- ~~M-7: Use `LazyLock` for regex in `miner_nodes.rs`~~ **DONE** (used `once_cell::sync::Lazy` for MSRV compatibility)
- ~~M-8: Use config value in warning message instead of hardcoded "1.5x"~~ **DONE**
- ~~L-1: Remove unnecessary `async` from `evaluate()`~~ **DONE**
- ~~L-2: Replace `#[allow(unused_imports)]` with `#[cfg(test)]` on slash_executor exports~~ **DONE**

### Architectural note:
The DB-chain reconciliation gap is acceptable given the snapshot-based redesign. The gap is bounded by `collateral_event_scan_interval` (configurable), and slash execution always verifies on-chain state. For production hardening, consider adding a Prometheus metric for snapshot staleness as the team previously suggested.

The overall architecture is sound, follows SOLID principles, and demonstrates strong engineering fundamentals. The dual-asset collateral model is well-designed with proper separation of concerns. The test suite (97 Solidity + 308 Rust tests) provides excellent coverage.
