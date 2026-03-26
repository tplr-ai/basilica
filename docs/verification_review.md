# Verification Feature Review

This document summarizes the end-to-end verification flow (veritas → basilica-2) and the changes
implemented to enforce strict binary outcomes, propagate failure reasons, and harden proofs.

## Scope
- veritas: executor + validator binaries, PoW proofs (CPU/storage), and verification logic.
- basilica-2: validator orchestration, parsing, persistence, and node profile publication.

## Strict Binary Outcome
**Goal:** Any failed check yields overall failure; no partial success or scoring.

Implemented changes:
- `ValidatorBinaryOutput.success` is **strict**: `success && failure_reasons.is_empty()`.
- Validation score is binary: `1.0` for success, `0.0` for failure.
- Failure reasons are captured and propagated to:
  - DB `verification_logs.details.failure_reasons`
  - Node profile CR `status.failureReasons`
  - Error message surface (joined reasons)

## Failure Reasons
Failure reasons are now carried end-to-end:
- veritas emits `failure_reasons` in JSON.
- basilica-2 parses and persists them.
- Node profile CR includes them for external visibility.

## Remaining Assumptions / Risks
1) **Failure reasons on pre-validations** are still limited to high-level reasons unless explicitly
   added for Docker/NAT/storage.
2) **Node profile on failure** is only updated if a successful node_result exists; failures won't
   publish CRs unless explicitly added.

## Removed Features
- **Bandwidth PoW**: Removed due to unreliable verification - network conditions are too variable
  for consistent proof-of-work verification.

## References
- `veritas` executor: `crates/executor-binary/src/server/handlers.rs`
- `veritas` validator: `crates/validator-binary/src/executor_client.rs`, `executor_manager.rs`
- `basilica-2` validator: `crates/basilica-validator/src/miner_prover/validation_binary.rs`

