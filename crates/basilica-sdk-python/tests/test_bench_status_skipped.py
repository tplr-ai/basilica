"""
Tests for `BenchStatus(phase="Skipped")` terminal-phase recognition.

Closes #480. Cross-repo reference: `one-covenant/basilica-backend#419`
Stage 4 take-5 Cell B and the basilica-backend operator X2 fix
(`one-covenant/basilica-backend#650 / #653`), which emits a terminal
`BenchStatus{phase=Skipped, lastAttemptOutcome="skipped",
lastAttemptAt=...}` on the UD CR when workers exit before the
bench-controller observes them. The SDK must recognise this as
terminal and return cleanly from `wait_until_bench_complete` instead
of polling until the user-supplied timeout and raising `TimeoutError`.

Coverage
--------
- `_BENCH_TERMINAL_PHASES` contains all four terminal phases
  (`Succeeded`, `Failed`, `TimedOut`, `Skipped`).
- `BENCH_PHASE_SKIPPED` is importable from the top-level `basilica`
  package (export check).
- `BenchStatus(phase="Skipped")` properties:
  `is_terminal=True`, `is_skipped=True`, `is_successful=False`,
  `is_failed=False`.
- The matching properties for the other five phases (`Succeeded`,
  `Failed`, `TimedOut`, `Pending`, `Running`) are consistent (e.g.
  `Failed` -> `is_failed=True, is_successful=False, is_skipped=False`).
- `wait_until_bench_complete` returns the `BenchStatus` on `Skipped`
  without raising, mirroring the existing `Succeeded` / `Failed`
  / `TimedOut` paths.
- `wait_until_bench_complete_async` honours the same contract.
- `BenchStatus.from_status_dict` round-trips the operator's
  `phase=Skipped, lastAttemptOutcome=skipped` shape verbatim.
"""

import asyncio
from typing import Any, Dict
from unittest.mock import MagicMock

import basilica
from basilica import BenchStatus, DistributedTraining
from basilica.distributed import (
    BENCH_PHASE_FAILED,
    BENCH_PHASE_PENDING,
    BENCH_PHASE_RUNNING,
    BENCH_PHASE_SKIPPED,
    BENCH_PHASE_SUCCEEDED,
    BENCH_PHASE_TIMED_OUT,
    _BENCH_TERMINAL_PHASES,
)


def _make_bench_status(
    phase: str,
    message: str = "",
    last_attempt_outcome: str = "",
) -> BenchStatus:
    """Build a `BenchStatus` mirroring the operator's wire shape."""
    return BenchStatus(
        mode="on-start",
        phase=phase,
        result=None,
        started_at=None,
        completed_at=None,
        message=message or None,
        last_attempt_at=None,
        last_attempt_outcome=last_attempt_outcome or None,
    )


def _make_status_dict_for_phase(phase: str, message: str = "") -> Any:
    """PyO3-shape fake response with a bench block at `phase`."""
    bench: Dict[str, Any] = {
        "mode": "on-start",
        "phase": phase,
    }
    if phase != "Pending":
        bench["startedAt"] = "2026-05-18T00:46:25Z"
    if phase in {"Succeeded", "Failed", "TimedOut"}:
        bench["completedAt"] = "2026-05-18T01:01:31Z"
    if phase == "Skipped":
        bench["lastAttemptAt"] = "2026-05-18T01:15:50Z"
        bench["lastAttemptOutcome"] = "skipped"
    if message:
        bench["message"] = message

    class FakeDeployment:
        instance_name = "ud-bench-skipped"
        user_id = "u-test"
        namespace = "u-test"
        image = "pytorch/pytorch"
        state = "running"
        url = "https://x"
        created_at = "2026-05-18T00:46:25Z"
        updated_at = "2026-05-18T01:15:50Z"
        phase = "succeeded"
        message = None
        share_token = None
        share_url = None
        public_metadata = False
        distributed = {
            "worldSize": {
                "ready": 0,
                "target": 4,
                "min": 2,
                "max": 4,
                "belowMinimum": True,
            },
            "ranks": [],
            "transport": "hub-relay",
            "bench": bench,
        }

    return FakeDeployment()


class TestTerminalPhaseFrozenset:
    def test_skipped_is_in_terminal_frozenset(self) -> None:
        assert BENCH_PHASE_SKIPPED in _BENCH_TERMINAL_PHASES

    def test_frozenset_contains_all_four_terminal_phases(self) -> None:
        assert _BENCH_TERMINAL_PHASES == frozenset(
            {
                BENCH_PHASE_SUCCEEDED,
                BENCH_PHASE_FAILED,
                BENCH_PHASE_TIMED_OUT,
                BENCH_PHASE_SKIPPED,
            }
        )

    def test_non_terminal_phases_excluded(self) -> None:
        assert BENCH_PHASE_PENDING not in _BENCH_TERMINAL_PHASES
        assert BENCH_PHASE_RUNNING not in _BENCH_TERMINAL_PHASES


class TestSkippedConstantExport:
    def test_bench_phase_skipped_top_level_export(self) -> None:
        """`from basilica import BENCH_PHASE_SKIPPED` resolves."""
        assert basilica.BENCH_PHASE_SKIPPED == "Skipped"

    def test_all_phase_constants_consistent_across_exports(self) -> None:
        """Top-level re-exports match the distributed-module constants."""
        assert basilica.BENCH_PHASE_SKIPPED == BENCH_PHASE_SKIPPED
        assert basilica.BENCH_PHASE_SUCCEEDED == BENCH_PHASE_SUCCEEDED
        assert basilica.BENCH_PHASE_FAILED == BENCH_PHASE_FAILED
        assert basilica.BENCH_PHASE_TIMED_OUT == BENCH_PHASE_TIMED_OUT


class TestBenchStatusSkippedProperties:
    def test_skipped_is_terminal(self) -> None:
        bs = _make_bench_status(
            "Skipped",
            message="bench skipped: workers exited before bench-controller observed them",
            last_attempt_outcome="skipped",
        )
        assert bs.is_terminal is True

    def test_skipped_is_not_successful(self) -> None:
        bs = _make_bench_status("Skipped")
        assert bs.is_successful is False

    def test_skipped_is_not_failed(self) -> None:
        bs = _make_bench_status("Skipped")
        assert bs.is_failed is False

    def test_skipped_is_skipped(self) -> None:
        bs = _make_bench_status("Skipped")
        assert bs.is_skipped is True


class TestOtherPhasePropertyMatrix:
    """Property invariants across the full phase enum -- guards against
    Skipped semantics accidentally bleeding into the other phases."""

    def test_succeeded(self) -> None:
        bs = _make_bench_status("Succeeded")
        assert bs.is_terminal is True
        assert bs.is_successful is True
        assert bs.is_failed is False
        assert bs.is_skipped is False

    def test_failed(self) -> None:
        bs = _make_bench_status("Failed", message="err")
        assert bs.is_terminal is True
        assert bs.is_successful is False
        assert bs.is_failed is True
        assert bs.is_skipped is False

    def test_timed_out(self) -> None:
        bs = _make_bench_status("TimedOut", message="deadline elapsed")
        assert bs.is_terminal is True
        assert bs.is_successful is False
        assert bs.is_failed is True
        assert bs.is_skipped is False

    def test_pending(self) -> None:
        bs = _make_bench_status("Pending")
        assert bs.is_terminal is False
        assert bs.is_successful is False
        assert bs.is_failed is False
        assert bs.is_skipped is False

    def test_running(self) -> None:
        bs = _make_bench_status("Running")
        assert bs.is_terminal is False
        assert bs.is_successful is False
        assert bs.is_failed is False
        assert bs.is_skipped is False


class TestFromStatusDictSkipped:
    def test_roundtrip_operator_skipped_shape(self) -> None:
        raw: Dict[str, Any] = {
            "mode": "on-start",
            "phase": "Skipped",
            "message": "bench skipped: workers exited before bench-controller observed them",
            "lastAttemptAt": "2026-05-18T01:15:50.448532222+00:00",
            "lastAttemptOutcome": "skipped",
        }
        bs = BenchStatus.from_status_dict(raw)
        assert bs.phase == "Skipped"
        assert bs.is_terminal is True
        assert bs.is_skipped is True
        assert bs.is_successful is False
        assert bs.is_failed is False
        assert bs.mode == "on-start"
        assert bs.result is None
        assert bs.last_attempt_outcome == "skipped"
        assert (
            bs.message
            == "bench skipped: workers exited before bench-controller observed them"
        )
        assert bs.last_attempt_at is not None


class TestWaitUntilBenchCompleteSkipped:
    """Closes #480: `wait_until_bench_complete` returns the `Skipped`
    BenchStatus rather than polling until the user-supplied timeout and
    raising `TimeoutError`."""

    def test_returns_skipped_status_without_raising(self) -> None:
        client = MagicMock()
        client.get.return_value = _make_status_dict_for_phase(
            "Skipped",
            message="bench skipped: workers exited before bench-controller observed them",
        )
        training = DistributedTraining(client, "ud-bench-skipped")
        bs = training.wait_until_bench_complete(timeout=10)
        assert bs is not None
        assert bs.phase == "Skipped"
        assert bs.is_terminal is True
        assert bs.is_skipped is True
        assert bs.is_successful is False
        assert bs.is_failed is False
        assert bs.last_attempt_outcome == "skipped"
        assert bs.message is not None
        assert "workers exited" in bs.message

    def test_async_returns_skipped_status_without_raising(self) -> None:
        client = MagicMock()
        client.get.return_value = _make_status_dict_for_phase(
            "Skipped",
            message="bench skipped: workers exited before bench-controller observed them",
        )
        training = DistributedTraining(client, "ud-bench-skipped")

        async def _run() -> BenchStatus:
            result = await training.wait_until_bench_complete_async(timeout=10)
            assert result is not None
            return result

        bs = asyncio.run(_run())
        assert bs.phase == "Skipped"
        assert bs.is_terminal is True
        assert bs.is_skipped is True

    def test_other_terminal_phases_unaffected(self) -> None:
        """Regression guard: Succeeded / Failed / TimedOut still terminate."""
        for phase in ("Succeeded", "Failed", "TimedOut"):
            client = MagicMock()
            client.get.return_value = _make_status_dict_for_phase(phase)
            training = DistributedTraining(client, "ud-bench-" + phase.lower())
            bs = training.wait_until_bench_complete(timeout=10)
            assert bs is not None, f"phase={phase} returned None"
            assert bs.phase == phase
            assert bs.is_terminal is True
