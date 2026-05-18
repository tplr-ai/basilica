"""
Unit tests pinning the simplified bench API surface
(basilica-backend issue 661 / SDK-S2).

WHY this file exists (read the issue body for the full plan):

Today the SDK exposes too much state for an opt-in measurement helper:
- ``bench: str`` modes (``"on-start"`` / ``"off"``) -- two string tokens
  to memorize for a binary opt-in.
- ``training.wait_until_bench_complete(timeout=...)`` -- raises
  ``TimeoutError`` or returns a four-phase ``BenchStatus``.
- ``BenchStatus`` with four terminal phases (``Succeeded`` / ``Failed`` /
  ``TimedOut`` / ``Skipped``) and ``is_terminal`` / ``is_successful`` /
  ``is_failed`` / ``is_skipped`` properties.
- Two access paths to the result (``bench_status.result`` vs
  ``training.bench``).

Target after S2 (per
``docs/plans/SDK-API-SIMPLIFICATION-PLAN.md`` on basilica-backend
``main``):

- ``bench: bool`` -- ``True`` opts in, ``False`` opts out. The string
  values ``"on-start"`` / ``"off"`` remain accepted for backward-compat
  with a ``DeprecationWarning`` pointing at the bool form.
- ``training.bench`` returns ``BenchResult | None`` (unchanged; lazy).
- ``training.bench_diagnostics`` returns ``Optional[Dict[str, Any]]`` --
  a small dict with ``phase``, ``message``, ``mode``,
  ``started_at`` / ``completed_at`` / ``last_attempt_at`` /
  ``last_attempt_outcome`` for the rare caller who needs to know WHY
  the probe did not measure. Most users only read ``training.bench``.
- ``wait_until_bench_complete[_async]`` and direct ``BenchStatus`` use
  remain functional but emit ``DeprecationWarning`` pointing at
  ``training.bench`` / ``training.bench_diagnostics``.

These tests:
1. PRE-FIX: fail (today's SDK rejects ``bench=True`` as an invalid
   string, has no ``bench_diagnostics`` attribute, and emits no
   deprecation warning on the string-mode form).
2. POST-FIX: pass.

Stubbing pattern mirrors ``test_deploy_distributed_managed.py``: bypass
``BasilicaClient.__init__`` and stub the PyO3 binding so no auth /
network calls fire.
"""

import warnings
from typing import Any, Dict
from unittest.mock import MagicMock

import pytest

from basilica import (
    BasilicaClient,
    BenchResult,
    DistributedTraining,
    WorldSize,
)


# =============================================================================
# Helpers.
# =============================================================================


def _make_client_with_stub(
    name: str = "dlc-bench-bool-test",
    namespace: str = "u-test",
) -> BasilicaClient:
    """Build a BasilicaClient whose PyO3 binding is fully stubbed.

    Bypasses ``BasilicaClient.__init__`` to avoid the auth bootstrap.
    """
    client = BasilicaClient.__new__(BasilicaClient)
    inner = MagicMock()

    create_response = MagicMock()
    create_response.instance_name = name
    inner.create_distributed_deployment = MagicMock(return_value=create_response)

    get_response = MagicMock()
    get_response.namespace = namespace
    get_response.instance_name = name
    get_response.image = "ghcr.io/example/trainer:latest"
    get_response.phase = "ready"
    get_response.message = None
    get_response.share_token = None
    get_response.share_url = None
    get_response.public_metadata = None
    # Workers Ready immediately so wait_until_min_world returns
    # without blocking the unit test.
    get_response.distributed = {
        "worldSize": {
            "ready": 2,
            "target": 2,
            "min": 2,
            "max": 4,
            "belowMinimum": False,
        },
    }
    inner.get_deployment = MagicMock(return_value=get_response)
    inner.delete_deployment = MagicMock(return_value=None)
    inner.scale_distributed_deployment = MagicMock(return_value=None)

    client._client = inner
    return client


def _deploy_kwargs() -> Dict[str, Any]:
    """Minimum kwargs that exercise the ``bench=`` parameter."""
    return {
        "name": "dlc-bench-bool-test",
        "image": "ghcr.io/example/trainer:latest",
        "world_size": WorldSize(min=2, target=2, max=4),
        "command": ["python3", "/workspace/noop.py"],
        # `timeout=0` is fine: the stubbed status reports min ranks ready,
        # so wait_until_min_world returns immediately.
        "timeout": 0,
    }


def _stub_bench_block(
    phase: str,
    *,
    mode: str = "on-start",
    message: str | None = None,
    with_result: bool = False,
) -> Dict[str, Any]:
    """Build a ``status.distributed.bench`` block mirroring the operator's
    wire shape. Used by tests that exercise the lazy ``training.bench`` and
    ``training.bench_diagnostics`` surfaces."""
    bench: Dict[str, Any] = {"mode": mode, "phase": phase}
    if phase != "Pending":
        bench["startedAt"] = "2026-05-18T00:46:25Z"
    if phase in {"Succeeded", "Failed", "TimedOut"}:
        bench["completedAt"] = "2026-05-18T01:01:31Z"
    if phase == "Skipped":
        bench["lastAttemptAt"] = "2026-05-18T01:15:50Z"
        bench["lastAttemptOutcome"] = "skipped"
    if message is not None:
        bench["message"] = message
    if with_result:
        bench["result"] = {
            "measuredAt": "2026-05-18T01:00:00Z",
            "busbwGbpsP50": 12.345,
            "busbwGbpsP10": 10.0,
            "busbwGbpsP90": 15.0,
            "algbwGbpsP50": 10.0,
            "latencyUsAt1mib": 50.0,
            "sizeBytesSwept": [1048576, 16777216],
            "probeNodeA": "node-a",
            "probeNodeB": "node-b",
        }
    return bench


def _stub_response_with_bench(
    name: str,
    namespace: str,
    bench_block: Dict[str, Any] | None,
) -> Any:
    """PyO3-shape fake DeploymentResponse with optional bench block."""

    class FakeDeployment:
        instance_name = name
        user_id = namespace
        image = "ghcr.io/example/trainer:latest"
        state = "running"
        url = "https://x"
        created_at = "2026-05-18T00:46:25Z"
        updated_at = "2026-05-18T01:15:50Z"
        phase = "succeeded"
        message = None
        share_token = None
        share_url = None
        public_metadata = False
        distributed: Dict[str, Any] = {
            "worldSize": {
                "ready": 2,
                "target": 2,
                "min": 2,
                "max": 2,
                "belowMinimum": False,
            },
            "ranks": [],
        }

    fake = FakeDeployment()
    fake.namespace = namespace
    if bench_block is not None:
        fake.distributed = {**fake.distributed, "bench": bench_block}
    return fake


# =============================================================================
# A. ``bench=bool`` acceptance.
# =============================================================================


class TestBenchBoolAcceptance:
    """``bench=True`` and ``bench=False`` must be accepted without warnings or errors."""

    def test_bench_true_accepted_without_warning(self) -> None:
        client = _make_client_with_stub()
        with warnings.catch_warnings(record=True) as recorded:
            warnings.simplefilter("always")
            training = client._deploy_distributed_impl(bench=True, **_deploy_kwargs())
        assert isinstance(training, DistributedTraining)
        deprecations = [w for w in recorded if issubclass(w.category, DeprecationWarning)]
        assert deprecations == [], (
            f"bench=True (canonical) must NOT raise DeprecationWarning, "
            f"got {[str(w.message) for w in deprecations]!r}"
        )

    def test_bench_false_accepted_without_warning(self) -> None:
        client = _make_client_with_stub()
        with warnings.catch_warnings(record=True) as recorded:
            warnings.simplefilter("always")
            training = client._deploy_distributed_impl(bench=False, **_deploy_kwargs())
        assert isinstance(training, DistributedTraining)
        deprecations = [w for w in recorded if issubclass(w.category, DeprecationWarning)]
        assert deprecations == [], (
            f"bench=False (canonical) must NOT raise DeprecationWarning, "
            f"got {[str(w.message) for w in deprecations]!r}"
        )

    def test_bench_true_emits_on_start_on_the_wire(self) -> None:
        """``bench=True`` -> request body has ``distributed.bench.mode='on-start'``."""
        client = _make_client_with_stub()
        client._deploy_distributed_impl(bench=True, **_deploy_kwargs())
        sent_payload = client._client.create_distributed_deployment.call_args.args[0]
        assert sent_payload["distributed"]["bench"]["mode"] == "on-start"

    def test_bench_false_emits_off_on_the_wire(self) -> None:
        """``bench=False`` -> request body has ``distributed.bench.mode='off'``."""
        client = _make_client_with_stub()
        client._deploy_distributed_impl(bench=False, **_deploy_kwargs())
        sent_payload = client._client.create_distributed_deployment.call_args.args[0]
        assert sent_payload["distributed"]["bench"]["mode"] == "off"

    def test_bench_default_is_off(self) -> None:
        """Omitting ``bench`` (default) emits ``mode=off`` -- no probe scheduled."""
        client = _make_client_with_stub()
        client._deploy_distributed_impl(**_deploy_kwargs())
        sent_payload = client._client.create_distributed_deployment.call_args.args[0]
        assert sent_payload["distributed"]["bench"]["mode"] == "off"


# =============================================================================
# B. ``@basilica.distributed`` decorator accepts bench=bool.
# =============================================================================


class TestDecoratorAcceptsBenchBool:
    def test_decorator_factory_accepts_bench_true(self) -> None:
        """``@basilica.distributed(bench=True)`` is constructible without raising."""
        from basilica import distributed
        decorator = distributed(
            name="dlc-decorator-bench-test",
            world_size=WorldSize(min=2, target=2, max=2),
            bench=True,
        )
        # The decorator returns a callable that wraps a function; we
        # do not need to deploy here -- the kwargs were captured.
        assert callable(decorator)

    def test_decorator_factory_accepts_bench_false(self) -> None:
        from basilica import distributed
        decorator = distributed(
            name="dlc-decorator-bench-test",
            world_size=WorldSize(min=2, target=2, max=2),
            bench=False,
        )
        assert callable(decorator)


# =============================================================================
# C. ``bench=str`` form is REMOVED in 0.30.0 (S7).
# =============================================================================


class TestBenchStrRemoved:
    """Post-S7: passing a string for bench raises ValidationError, not a
    DeprecationWarning. Migration path: ``bench=True`` / ``bench=False``."""

    def test_bench_on_start_str_raises_validation_error(self) -> None:
        from basilica.exceptions import ValidationError
        client = _make_client_with_stub()
        with pytest.raises(ValidationError, match=r"bench must be bool"):
            client._deploy_distributed_impl(bench="on-start", **_deploy_kwargs())

    def test_bench_off_str_raises_validation_error(self) -> None:
        from basilica.exceptions import ValidationError
        client = _make_client_with_stub()
        with pytest.raises(ValidationError, match=r"bench must be bool"):
            client._deploy_distributed_impl(bench="off", **_deploy_kwargs())


# =============================================================================
# D. ``training.bench_diagnostics`` (new simplified debug surface).
# =============================================================================


class TestBenchDiagnostics:
    """``training.bench_diagnostics`` is the rarely-needed debug accessor.

    Returns ``None`` when bench wasn't requested (mode=off) OR no
    operator status block; otherwise a dict with ``phase`` / ``message`` /
    timestamp keys. Replaces the four-property ``BenchStatus`` enum
    ceremony for the common case.
    """

    def test_diagnostics_attribute_exists(self) -> None:
        """``DistributedTraining.bench_diagnostics`` is a public attribute."""
        assert hasattr(DistributedTraining, "bench_diagnostics"), (
            "training.bench_diagnostics is the simplified debug surface "
            "for SDK-S2; it must exist on the class even before instance "
            "creation. See basilica-backend#661."
        )

    def test_diagnostics_returns_none_when_bench_off(self) -> None:
        client = MagicMock()
        client.get.return_value = _stub_response_with_bench(
            "ud-bench-off", "u-test", bench_block={"mode": "off", "phase": "Skipped"}
        )
        training = DistributedTraining(client, "ud-bench-off")
        # With mode=off, the diagnostics surface should report None to
        # the user (the operator publishes a Skipped block for bookkeeping
        # but the user didn't ask for the probe).
        diag = training.bench_diagnostics
        assert diag is None, (
            f"bench_diagnostics must be None when mode=off, got {diag!r}"
        )

    def test_diagnostics_returns_none_when_no_bench_block(self) -> None:
        client = MagicMock()
        client.get.return_value = _stub_response_with_bench(
            "ud-no-bench", "u-test", bench_block=None
        )
        training = DistributedTraining(client, "ud-no-bench")
        assert training.bench_diagnostics is None

    def test_diagnostics_returns_dict_when_bench_on_start_skipped(self) -> None:
        client = MagicMock()
        client.get.return_value = _stub_response_with_bench(
            "ud-bench-skipped",
            "u-test",
            bench_block=_stub_bench_block(
                "Skipped",
                message="workers exited before bench-controller observed them",
            ),
        )
        training = DistributedTraining(client, "ud-bench-skipped")
        diag = training.bench_diagnostics
        assert diag is not None, "bench=on-start + Skipped -> diagnostics must be non-None"
        assert isinstance(diag, dict)
        assert diag["phase"] == "Skipped"
        assert diag["mode"] == "on-start"
        assert "workers exited" in diag["message"]
        assert "last_attempt_at" in diag
        assert "last_attempt_outcome" in diag
        assert diag["last_attempt_outcome"] == "skipped"

    def test_diagnostics_returns_dict_when_bench_succeeded(self) -> None:
        client = MagicMock()
        client.get.return_value = _stub_response_with_bench(
            "ud-bench-succ",
            "u-test",
            bench_block=_stub_bench_block("Succeeded", with_result=True),
        )
        training = DistributedTraining(client, "ud-bench-succ")
        diag = training.bench_diagnostics
        assert diag is not None
        assert diag["phase"] == "Succeeded"
        assert diag["mode"] == "on-start"
        assert "started_at" in diag
        assert "completed_at" in diag


# =============================================================================
# E. ``training.bench`` (unchanged) — lazy ``BenchResult | None``.
# =============================================================================


class TestTrainingBenchLazyResult:
    """``training.bench`` collapses all four non-Succeeded terminal phases
    to ``None``. The user reads "did we measure?" with a single
    ``if training.bench is not None`` check; the four-phase ceremony
    moves to ``bench_diagnostics`` for debugging."""

    def test_bench_returns_result_on_succeeded(self) -> None:
        client = MagicMock()
        client.get.return_value = _stub_response_with_bench(
            "ud-bench-succ",
            "u-test",
            bench_block=_stub_bench_block("Succeeded", with_result=True),
        )
        training = DistributedTraining(client, "ud-bench-succ")
        assert training.bench is not None
        assert isinstance(training.bench, BenchResult)
        assert training.bench.busbw_gbps_p50 == 12.345

    def test_bench_returns_none_on_skipped(self) -> None:
        """Skipped means "no measurement" -- the user reads ``None``."""
        client = MagicMock()
        client.get.return_value = _stub_response_with_bench(
            "ud-bench-skipped",
            "u-test",
            bench_block=_stub_bench_block("Skipped"),
        )
        training = DistributedTraining(client, "ud-bench-skipped")
        assert training.bench is None

    def test_bench_returns_none_on_failed(self) -> None:
        client = MagicMock()
        client.get.return_value = _stub_response_with_bench(
            "ud-bench-failed",
            "u-test",
            bench_block=_stub_bench_block("Failed", message="probe crashed"),
        )
        training = DistributedTraining(client, "ud-bench-failed")
        assert training.bench is None

    def test_bench_returns_none_on_timed_out(self) -> None:
        client = MagicMock()
        client.get.return_value = _stub_response_with_bench(
            "ud-bench-timeout",
            "u-test",
            bench_block=_stub_bench_block("TimedOut", message="deadline elapsed"),
        )
        training = DistributedTraining(client, "ud-bench-timeout")
        assert training.bench is None


# =============================================================================
# F. ``wait_until_bench_complete[_async]`` and ``bench_status`` are
#    REMOVED in 0.30.0 (S7). Migration: read ``training.bench``
#    (``BenchResult | None``) and ``training.bench_diagnostics``
#    (``dict | None``).
# =============================================================================


class TestRemovedBenchAccessors:
    def test_wait_until_bench_complete_is_removed(self) -> None:
        assert not hasattr(DistributedTraining, "wait_until_bench_complete"), (
            "wait_until_bench_complete must be removed in 0.30.0 (SDK-S7); "
            "use training.bench / training.bench_diagnostics."
        )

    def test_wait_until_bench_complete_async_is_removed(self) -> None:
        assert not hasattr(
            DistributedTraining, "wait_until_bench_complete_async"
        ), (
            "wait_until_bench_complete_async must be removed in 0.30.0 "
            "(SDK-S7); use training.bench / training.bench_diagnostics."
        )

    def test_bench_status_property_is_removed(self) -> None:
        assert not hasattr(DistributedTraining, "bench_status"), (
            "bench_status must be removed in 0.30.0 (SDK-S7); use "
            "training.bench (BenchResult | None) or "
            "training.bench_diagnostics (dict | None)."
        )

    def test_basilica_does_not_re_export_BenchStatus(self) -> None:
        import basilica
        assert not hasattr(basilica, "BenchStatus"), (
            "BenchStatus must not be re-exported from basilica in 0.30.0 "
            "(SDK-S7); use BenchResult plus the dict from "
            "training.bench_diagnostics."
        )
