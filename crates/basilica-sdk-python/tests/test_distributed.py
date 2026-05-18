"""
Unit tests for the distributed-training SDK surface (Phase 5b).

Coverage:
- Dataclass __post_init__ validation (WorldSize bounds).
- DistributedTraining.scale rejects target<1 BEFORE the network call.
- wait_until_min_world(timeout=0) raises BelowMinimumWorld immediately.
- NEGATIVE TEST: BasilicaClient has NO preflight / nccl_baseline. This
  pins the SDK arch § 7 tenancy contract -- a future agent restoring
  those helpers would not break tests without explicit removal of the
  assertion.
- Build-helper produces correct camelCase JSON for POST /deployments.

These tests do NOT require a running cluster. The Phase 5b prompt's
"Test E" (live e2e on K3s) is a separate manual run, deferred until the
DNS-1035 fix lands in basilica-api.
"""

import asyncio
from typing import Any, Dict
from unittest.mock import MagicMock

import pytest

from basilica import (
    BasilicaClient,
    BelowMinimumWorld,
    BenchResult,
    DistributedError,
    DistributedFunction,
    DistributedTraining,
    ProviderFilter,
    QuotaExceeded,
    RankExit,
    RankStatus,
    UDTerminalState,
    ValidationError,
    WorldSize,
    WorldSizeOutOfBounds,
    WorldStatus,
    distributed,
)


# =============================================================================
# Dataclass validation (SDK arch § 8).
# =============================================================================


class TestWorldSizeValidation:
    def test_valid_triples_accepted(self) -> None:
        WorldSize(min=1, target=1, max=1)
        WorldSize(min=2, target=4, max=8)
        WorldSize(min=4, target=4, max=4)

    def test_min_zero_rejected(self) -> None:
        with pytest.raises(WorldSizeOutOfBounds) as exc_info:
            WorldSize(min=0, target=1, max=1)
        assert exc_info.value.requested == 0
        assert "min must be >= 1" in str(exc_info.value)

    def test_target_below_min_rejected(self) -> None:
        with pytest.raises(WorldSizeOutOfBounds) as exc_info:
            WorldSize(min=4, target=2, max=8)
        assert exc_info.value.requested == 2
        assert exc_info.value.min == 4
        assert "must be >= WorldSize.min" in str(exc_info.value)

    def test_max_below_target_rejected(self) -> None:
        with pytest.raises(WorldSizeOutOfBounds) as exc_info:
            WorldSize(min=2, target=8, max=4)
        assert exc_info.value.requested == 8
        assert exc_info.value.max == 4
        assert "must be >= WorldSize.target" in str(exc_info.value)


class TestProviderFilterDefaults:
    def test_empty_lists_default(self) -> None:
        pf = ProviderFilter()
        assert pf.include == []
        assert pf.exclude == []


# =============================================================================
# NEGATIVE TEST -- locks SDK arch § 7 tenancy contract.
#
# A future agent restoring `client.preflight(...)` or
# `client.nccl_baseline(...)` would not break tests without explicit
# removal of these assertions. That is exactly the property the Phase 5b
# prompt asks for: do not let those user-facing helpers come back as a
# silent regression.
# =============================================================================


class TestNoStandaloneBenchHelpers:
    def test_basilica_client_has_no_preflight(self) -> None:
        assert not hasattr(BasilicaClient, "preflight"), (
            "BasilicaClient.preflight was removed in Phase 5b -- it implied "
            "a shared cross-tenant cache, violating SDK arch § 7. Bench data "
            "is per-UD via training.bench. Do NOT restore this method."
        )

    def test_basilica_client_has_no_nccl_baseline(self) -> None:
        assert not hasattr(BasilicaClient, "nccl_baseline"), (
            "BasilicaClient.nccl_baseline was removed in Phase 5b for the "
            "same SDK arch § 7 reason as preflight. Per-UD bench probes "
            "(bench=True + training.bench) are the supported path."
        )

    def test_basilica_client_has_no_preflight_async(self) -> None:
        assert not hasattr(BasilicaClient, "preflight_async")
        assert not hasattr(BasilicaClient, "nccl_baseline_async")

    def test_basilica_module_does_not_expose_preflight(self) -> None:
        import basilica
        assert not hasattr(basilica, "preflight")
        assert not hasattr(basilica, "nccl_baseline")


# =============================================================================
# DistributedTraining facade behaviour (SDK arch § 6).
# =============================================================================


def _make_mock_training(
    name: str = "dlc-test",
    namespace: str = "u-test",
    world: Dict[str, Any] = None,
) -> DistributedTraining:
    """Build a DistributedTraining whose backing client is fully mocked.

    Uses a plain MagicMock (no spec) because BasilicaClient's `_client`
    is a private attribute holding the PyO3 binding -- spec-based mocks
    refuse to materialize private attrs lazily, and tests need to read
    `training._client._client.scale_distributed_deployment` to assert
    the SDK's pre-network rejection contract.
    """
    client = MagicMock()
    # Explicitly install the inner PyO3-bound `_client` attribute so the
    # facade's `self._client._client.scale_distributed_deployment(...)`
    # call resolves without surfacing a spec error.
    client._client = MagicMock()
    fake_response = MagicMock()
    fake_response.namespace = namespace
    fake_response._distributed_status = {
        "worldSize": world or {"ready": 0, "target": 0, "min": 2, "max": 4, "belowMinimum": True},
    }
    client.get.return_value = fake_response
    training = DistributedTraining(client, name)
    # Pre-populate cache so methods don't try to refresh.
    training._cached_status = {
        "namespace": namespace,
        "distributed": {
            "worldSize": world or {
                "ready": 0,
                "target": 0,
                "min": 2,
                "max": 4,
                "belowMinimum": True,
            },
        },
    }
    training.namespace = namespace
    training.rendezvous_endpoint = (
        f"{name}-rendezvous.{namespace}.svc.cluster.local:2379"
    )
    return training


class TestDistributedTrainingScale:
    def test_scale_zero_rejected_before_network(self) -> None:
        training = _make_mock_training()
        with pytest.raises(WorldSizeOutOfBounds) as exc_info:
            training.scale(target=0)
        assert exc_info.value.requested == 0
        # The PyO3 client method must NOT have been called -- the rejection
        # is local. This is the SDK arch § 11 contract: target < 1 fails
        # synchronously before any HTTP call.
        training._client._client.scale_distributed_deployment.assert_not_called()

    def test_scale_negative_rejected_before_network(self) -> None:
        training = _make_mock_training()
        with pytest.raises(WorldSizeOutOfBounds):
            training.scale(target=-1)
        training._client._client.scale_distributed_deployment.assert_not_called()

    def test_scale_passes_target_through_to_client(self) -> None:
        training = _make_mock_training()
        # First call returns; subsequent refresh re-reads cached status.
        training._client._client.scale_distributed_deployment = MagicMock()
        # Stub get() so refresh() after the scale doesn't blow up.
        training._client.get = MagicMock(
            return_value=MagicMock(
                namespace="u-test",
                _distributed_status={
                    "worldSize": {
                        "ready": 0,
                        "target": 3,
                        "min": 2,
                        "max": 4,
                        "belowMinimum": True,
                    }
                },
            )
        )
        ws = training.scale(target=3)
        training._client._client.scale_distributed_deployment.assert_called_once_with(
            "dlc-test", 3
        )
        assert ws.target == 3


class TestIssue495ScaleStaleTargetGate:
    """
    Issue #495: `scale(target=N)` patches `spec.distributed.worldSize.target`
    only; the operator mirrors `spec → status` on its next reconcile. The
    SDK's `wait_until_target_world` must NOT short-circuit while the
    operator-observed `status.worldSize.target` still carries the
    pre-patch value.
    """

    def test_issue_495_scale_returns_world_status_with_requested_target(
        self,
    ) -> None:
        # Stub the API so refresh() after scale() still returns the OLD
        # `status.worldSize.target` (operator hasn't reconciled yet). The
        # SDK must NOT report the stale value back to the caller.
        training = _make_mock_training(
            world={
                "ready": 2,
                "target": 2,
                "min": 2,
                "max": 4,
                "belowMinimum": False,
            },
        )
        training._client._client.scale_distributed_deployment = MagicMock()
        training._client.get = MagicMock(
            return_value=MagicMock(
                namespace="u-test",
                _distributed_status={
                    "worldSize": {
                        "ready": 2,
                        "target": 2,  # stale: operator hasn't observed scale yet
                        "min": 2,
                        "max": 4,
                        "belowMinimum": False,
                    }
                },
            )
        )

        ws = training.scale(target=3)

        # Reported target must reflect the request, not the stale operator
        # snapshot. Otherwise `Scaled to target=3; world now: ...target=2`
        # is a lie that misleads operators reviewing example output.
        assert ws.target == 3, (
            "issue #495: scale() must report the REQUESTED target, not the "
            "stale `status.worldSize.target` from before the operator's "
            "next reconcile."
        )

    def test_issue_495_wait_until_target_world_does_not_shortcircuit_on_stale_target(
        self, monkeypatch: pytest.MonkeyPatch
    ) -> None:
        # Reproduces the ex21-r2 silent-failure: scale(target=3) succeeds,
        # but `status.worldSize.target` stays at 2 because the operator
        # hasn't reconciled yet. wait_until_target_world MUST keep waiting.
        training = _make_mock_training(
            world={
                "ready": 2,
                "target": 2,
                "min": 2,
                "max": 4,
                "belowMinimum": False,
            },
        )
        training._client._client.scale_distributed_deployment = MagicMock()

        # Three refresh ticks before status reflects the patch:
        #   tick 0 (the refresh inside scale()): stale target=2 ready=2
        #   tick 1 (first wait iteration): stale target=2 ready=2
        #   tick 2 (second wait iteration): operator reconciled, target=3 ready=2
        #   tick 3 (third wait iteration): rank joined, target=3 ready=3 -- DONE
        responses = [
            MagicMock(
                namespace="u-test",
                _distributed_status={
                    "worldSize": {
                        "ready": 2,
                        "target": 2,
                        "min": 2,
                        "max": 4,
                        "belowMinimum": False,
                    }
                },
            ),
            MagicMock(
                namespace="u-test",
                _distributed_status={
                    "worldSize": {
                        "ready": 2,
                        "target": 2,
                        "min": 2,
                        "max": 4,
                        "belowMinimum": False,
                    }
                },
            ),
            MagicMock(
                namespace="u-test",
                _distributed_status={
                    "worldSize": {
                        "ready": 2,
                        "target": 3,
                        "min": 2,
                        "max": 4,
                        "belowMinimum": False,
                    }
                },
            ),
            MagicMock(
                namespace="u-test",
                _distributed_status={
                    "worldSize": {
                        "ready": 3,
                        "target": 3,
                        "min": 2,
                        "max": 4,
                        "belowMinimum": False,
                    }
                },
            ),
        ]
        training._client.get = MagicMock(side_effect=responses)
        # Avoid real sleeps -- the loop's inter-poll delay is irrelevant.
        sleeps: list = []
        import sys

        _dist_mod = sys.modules["basilica.distributed"]
        monkeypatch.setattr(_dist_mod.time, "sleep", lambda s: sleeps.append(s))

        training.scale(target=3)
        # The pre-fix bug: this returned ~immediately because ws.target==2
        # and ws.ready==2 (stale) satisfied `ready >= target`. Post-fix,
        # the pending-scale gate forces the loop to wait until status
        # reflects target=3 AND ready>=3.
        training.wait_until_target_world(timeout=120)

        ws_final = training.world
        assert ws_final.ready == 3
        assert ws_final.target == 3
        # Loop must have iterated at least twice (i.e. slept at least once).
        assert sleeps, (
            "issue #495: wait_until_target_world short-circuited on the "
            "stale `status.worldSize.target` instead of waiting for the "
            "operator to reflect the requested target."
        )
        # Pending-scale flag must be cleared on success.
        assert training._pending_scale_target is None

    def test_issue_495_wait_until_target_world_calls_refresh_before_first_check(
        self,
    ) -> None:
        # Lock the SDK invariant: even with no pending scale, the first
        # comparison in `wait_until_target_world` sees a fresh refresh.
        training = _make_mock_training(
            world={
                "ready": 2,
                "target": 2,
                "min": 2,
                "max": 4,
                "belowMinimum": False,
            },
        )
        # Stub the API so the first refresh() call returns ready==target.
        training._client.get = MagicMock(
            return_value=MagicMock(
                namespace="u-test",
                _distributed_status={
                    "worldSize": {
                        "ready": 2,
                        "target": 2,
                        "min": 2,
                        "max": 4,
                        "belowMinimum": False,
                    }
                },
            )
        )
        # Should return without raising.
        training.wait_until_target_world(timeout=10)
        # `client.get` is the underlying refresh path; it MUST have been
        # called at least once before the success-predicate check.
        assert training._client.get.call_count >= 1, (
            "issue #495: wait_until_target_world must refresh from the API "
            "before its first ready/target comparison."
        )

    def test_issue_495_wait_until_target_world_times_out_with_pending_required_min(
        self, monkeypatch: pytest.MonkeyPatch
    ) -> None:
        # If the operator never reflects the patch, the timeout error
        # message must report the REQUESTED target as required_min, not
        # the stale operator-observed target.
        training = _make_mock_training(
            world={
                "ready": 2,
                "target": 2,
                "min": 2,
                "max": 4,
                "belowMinimum": False,
            },
        )
        training._client._client.scale_distributed_deployment = MagicMock()
        # Status never advances past the stale snapshot.
        training._client.get = MagicMock(
            return_value=MagicMock(
                namespace="u-test",
                _distributed_status={
                    "worldSize": {
                        "ready": 2,
                        "target": 2,
                        "min": 2,
                        "max": 4,
                        "belowMinimum": False,
                    }
                },
            )
        )
        import sys

        _dist_mod = sys.modules["basilica.distributed"]
        monkeypatch.setattr(_dist_mod.time, "sleep", lambda s: None)

        training.scale(target=4)
        with pytest.raises(BelowMinimumWorld) as exc_info:
            training.wait_until_target_world(timeout=0)
        assert exc_info.value.ready == 2
        assert exc_info.value.required_min == 4, (
            "issue #495: timeout exception must surface the REQUESTED "
            "target as required_min (not stale `status.worldSize.target`)."
        )


class TestWaitUntilMinWorld:
    def test_timeout_zero_raises_below_minimum_immediately(self) -> None:
        # World is below min (ready=0, min=2). With timeout=0, the call
        # must raise BelowMinimumWorld synchronously without spinning.
        training = _make_mock_training(
            world={"ready": 0, "target": 2, "min": 2, "max": 4, "belowMinimum": True},
        )
        # Stub refresh-time get() so the final refresh inside wait sees
        # the same below-min state.
        training._client.get = MagicMock(
            return_value=MagicMock(
                namespace="u-test",
                _distributed_status={
                    "worldSize": {
                        "ready": 0,
                        "target": 2,
                        "min": 2,
                        "max": 4,
                        "belowMinimum": True,
                    }
                },
            )
        )
        with pytest.raises(BelowMinimumWorld) as exc_info:
            training.wait_until_min_world(timeout=0)
        assert exc_info.value.ready == 0
        assert exc_info.value.required_min == 2

    def test_returns_when_world_already_at_min(self) -> None:
        training = _make_mock_training(
            world={"ready": 4, "target": 4, "min": 2, "max": 8, "belowMinimum": False},
        )
        training._client.get = MagicMock(
            return_value=MagicMock(
                namespace="u-test",
                _distributed_status={
                    "worldSize": {
                        "ready": 4,
                        "target": 4,
                        "min": 2,
                        "max": 8,
                        "belowMinimum": False,
                    }
                },
            )
        )
        # Should return without raising.
        training.wait_until_min_world(timeout=10)


class TestAsyncParity:
    """SDK arch § 9: every facade method has an _async counterpart."""

    def test_distributed_training_has_async_counterparts(self) -> None:
        for sync_name in [
            "scale",
            "wait_until_min_world",
            "wait_until_target_world",
            "metrics",
            "events",
            "logs",
            "stream_logs",
            "refresh",
            "delete",
        ]:
            async_name = f"{sync_name}_async"
            assert hasattr(DistributedTraining, async_name), (
                f"DistributedTraining.{async_name} missing -- SDK arch § 9 "
                f"requires every method to have an _async counterpart."
            )

    def test_basilica_client_has_private_distributed_impl(self) -> None:
        # Post-S7 (0.30.0): the public deploy_distributed* methods are
        # removed; the decorator (`@basilica.distributed`) routes
        # through the private `_deploy_distributed_impl[_async]` pair.
        assert hasattr(BasilicaClient, "_deploy_distributed_impl")
        assert hasattr(BasilicaClient, "_deploy_distributed_impl_async")
        assert not hasattr(BasilicaClient, "deploy_distributed"), (
            "deploy_distributed must be removed in 0.30.0 (SDK-S7)"
        )
        assert not hasattr(BasilicaClient, "deploy_distributed_async"), (
            "deploy_distributed_async must be removed in 0.30.0 (SDK-S7)"
        )


# =============================================================================
# BenchResult parsing (SDK arch § 7 / § 8).
# =============================================================================


class TestBenchResult:
    def test_parses_full_status_dict(self) -> None:
        raw = {
            "measuredAt": "2026-05-02T10:00:00Z",
            "busbwGbpsP10": 0.045,
            "busbwGbpsP50": 0.063,
            "busbwGbpsP90": 0.072,
            "algbwGbpsP50": 0.058,
            "latencyUsAt1mib": 1850.0,
            "sizeBytesSwept": [1048576, 16777216, 268435456],
            "probeNodeA": "basilica-verda-fin-03",
            "probeNodeB": "basilica-verda-fin-04",
        }
        result = BenchResult.from_status_dict(raw)
        assert result.busbw_gbps_p50 == 0.063
        assert result.size_bytes_swept == [1048576, 16777216, 268435456]
        assert result.probe_node_a == "basilica-verda-fin-03"
        assert result.probe_node_b == "basilica-verda-fin-04"

    def test_handles_missing_optionals(self) -> None:
        # Probe failed mid-sweep -- some percentile fields absent.
        raw = {
            "measuredAt": "2026-05-02T10:00:00Z",
            "probeNodeA": "node-a",
            "probeNodeB": "node-b",
        }
        result = BenchResult.from_status_dict(raw)
        assert result.busbw_gbps_p50 is None
        assert result.latency_us_at_1mib is None
        assert result.size_bytes_swept == []


# =============================================================================
# CreateDistributedDeploymentRequest build path (BasilicaClient internal).
# =============================================================================


class TestBuildDistributedRequest:
    def _client(self) -> BasilicaClient:
        # Construct a client without trying to authenticate. We only
        # exercise the internal `_build_distributed_request` helper.
        client = BasilicaClient.__new__(BasilicaClient)
        client._base_url = "https://api.basilica.ai"
        return client

    def test_request_shape_camelcase_kebab_enums(self) -> None:
        client = self._client()
        req = client._build_distributed_request(
            name="dlc-test",
            source=None,
            image="my-image",
            port=18789,
            env=None,
            cpu="8",
            memory="32Gi",
            gpu_count=1,
            gpu_models=["A100"],
            min_gpu_memory_gb=40,
            world_size=WorldSize(min=2, target=4, max=8),
            provider_filter=ProviderFilter(include=["verda"], exclude=[]),
            topology_spread="provider-aware",
            nccl_env={"NCCL_DEBUG": "WARN"},
            bench="on-start",
            rendezvous_backend="etcd-v2",
            command=["python", "train.py"],
            args=["--epochs", "10"],
            pip_packages=None,
            ttl_seconds=3600,
            enable_billing=True,
        )
        assert req["instanceName"] == "dlc-test"
        assert req["replicas"] == 4  # mirrors target
        d = req["distributed"]
        assert d["enabled"] is True
        assert d["worldSize"] == {"min": 2, "target": 4, "max": 8}
        assert d["rendezvous"]["backend"] == "etcd-v2"
        assert d["providerFilter"]["include"] == ["verda"]
        assert d["providerFilter"]["exclude"] == []
        assert d["topologySpread"]["strategy"] == "provider-aware"
        assert d["bench"]["mode"] == "on-start"
        assert d["nccl"]["env"]["NCCL_DEBUG"] == "WARN"
        # BYO command: shlex-joined, NOT auto-mapped to "auto" because
        # `source` was None (regression guard for review-comment fix).
        assert d["command"] != "auto"
        assert "python" in d["command"]
        assert "train.py" in d["command"]

    def test_neither_source_nor_command_raises(self) -> None:
        from basilica.exceptions import ValidationError

        client = self._client()
        with pytest.raises(ValidationError) as exc_info:
            client._build_distributed_request(
                name="dlc-test",
                source=None,
                image="x",
                port=80,
                env=None,
                cpu="1",
                memory="1Gi",
                gpu_count=1,
                gpu_models=None,
                min_gpu_memory_gb=None,
                world_size=WorldSize(min=1, target=1, max=1),
                provider_filter=None,
                topology_spread="provider-aware",
                nccl_env=None,
                bench="off",
                rendezvous_backend="etcd-v2",
                command=None,
                args=None,
                pip_packages=None,
                ttl_seconds=None,
                enable_billing=True,
            )
        # Reviewer concern: with neither `source` nor `command`, we used
        # to silently default `distributed.command` to "auto", breaking
        # the BYO-launcher example. Now we hard-fail.
        msg = str(exc_info.value).lower()
        assert "source" in msg or "command" in msg

    def test_source_callable_ships_via_b64(self) -> None:
        # Source-shipping path: writes /tmp/__basilica_source.py via
        # base64-decoded bash one-liner. Operator wraps in `sh -c`.
        # /tmp/ (not /workspace/) because the operator pod template
        # runs as uid=1000 with no writable /workspace mount. See #448.
        # Post-S7 (0.30.0): source is Callable-only (the str/Path shapes
        # were removed). The Callable IS the per-rank entrypoint.
        client = self._client()

        def _train() -> None:
            print("hello")

        req = client._build_distributed_request(
            name="dlc-source-test",
            source=_train,
            image="my-image",
            port=18789,
            env=None,
            cpu="1",
            memory="1Gi",
            gpu_count=1,
            gpu_models=None,
            min_gpu_memory_gb=None,
            world_size=WorldSize(min=1, target=1, max=1),
            provider_filter=None,
            topology_spread="provider-aware",
            nccl_env=None,
            bench="off",
            rendezvous_backend="etcd-v2",
            command=None,
            args=None,
            pip_packages=None,
            ttl_seconds=None,
            enable_billing=True,
        )
        d_cmd = req["distributed"]["command"]
        assert "base64 -d > /tmp/__basilica_source.py" in d_cmd
        assert "exec torchrun" in d_cmd
        assert "$BASILICA_RDZV_ENDPOINT" in d_cmd
        assert d_cmd != "auto"

    def test_source_shipping_writes_to_tmp_not_workspace(self) -> None:
        # Issue #448 regression guard. The pytorch base image has
        # /workspace owned by uid=0; pods run as uid=1000 with no
        # writable /workspace mount, so writing the source there
        # crashes every rank with "Permission denied" and CrashLoopBackoff.
        # /tmp/ is writable by any uid in any standard base image.
        # NEGATIVE assertion locks the contract: a regression that
        # restores /workspace/ would fail this test loudly.
        client = self._client()

        def _rank_up() -> None:
            print("rank-up")

        req = client._build_distributed_request(
            name="dlc-issue-448",
            source=_rank_up,
            image="pytorch/pytorch:2.1.0-cuda12.1-cudnn8-runtime",
            port=18789,
            env=None,
            cpu="1",
            memory="1Gi",
            gpu_count=1,
            gpu_models=None,
            min_gpu_memory_gb=None,
            world_size=WorldSize(min=2, target=2, max=2),
            provider_filter=None,
            topology_spread="provider-aware",
            nccl_env=None,
            bench="off",
            rendezvous_backend="etcd-v2",
            command=None,
            args=None,
            pip_packages=None,
            ttl_seconds=None,
            enable_billing=True,
        )
        d_cmd = req["distributed"]["command"]
        # Positive: write target + torchrun arg both reference /tmp/.
        assert d_cmd.count("/tmp/__basilica_source.py") == 2, (
            f"expected /tmp/__basilica_source.py to appear twice (write "
            f"target + torchrun arg), got command: {d_cmd!r}"
        )
        # Negative: no /workspace/ anywhere -- locks the #448 fix.
        assert "/workspace/" not in d_cmd, (
            f"command must not write to /workspace/ -- pytorch base "
            f"image's /workspace is root-owned and pods run as uid=1000. "
            f"See issue #448. command: {d_cmd!r}"
        )

    def test_bench_placement_default_omits_field(self) -> None:
        # Architecture doc § 11.1 placement knob: default `preferred`
        # MUST NOT appear on the wire. The operator's serde default is
        # also `Preferred`, so omitting the field keeps wire-compat with
        # operators that don't yet know about the field. Locks the
        # `bench_placement="preferred"` -> no `placement` key contract.
        client = self._client()
        req = client._build_distributed_request(
            name="dlc-pref",
            source=None,
            image="x",
            port=80,
            env=None,
            cpu="1",
            memory="1Gi",
            gpu_count=1,
            gpu_models=None,
            min_gpu_memory_gb=None,
            world_size=WorldSize(min=2, target=2, max=2),
            provider_filter=None,
            topology_spread="provider-aware",
            nccl_env=None,
            bench="on-start",
            rendezvous_backend="etcd-v2",
            command=["python", "x.py"],
            args=None,
            pip_packages=None,
            ttl_seconds=None,
            enable_billing=True,
            # explicit default — same as omitting the kwarg.
            bench_placement="preferred",
        )
        bench_dict = req["distributed"]["bench"]
        assert bench_dict["mode"] == "on-start"
        assert "placement" not in bench_dict, (
            "default bench_placement='preferred' must NOT emit the field "
            f"on the wire (keeps wire-compat with operators that don't "
            f"know the placement enum yet); got: {bench_dict!r}"
        )

    def test_bench_placement_strict_emits_lowercase_token(self) -> None:
        # Architecture doc § 11.1 placement knob: opt-in `strict` lands
        # on the wire as `placement: "strict"` (lowercase, matches the
        # operator's serde rename).
        client = self._client()
        req = client._build_distributed_request(
            name="dlc-strict",
            source=None,
            image="x",
            port=80,
            env=None,
            cpu="1",
            memory="1Gi",
            gpu_count=1,
            gpu_models=None,
            min_gpu_memory_gb=None,
            world_size=WorldSize(min=2, target=2, max=2),
            provider_filter=None,
            topology_spread="provider-aware",
            nccl_env=None,
            bench="on-start",
            rendezvous_backend="etcd-v2",
            command=["python", "x.py"],
            args=None,
            pip_packages=None,
            ttl_seconds=None,
            enable_billing=True,
            bench_placement="strict",
        )
        bench_dict = req["distributed"]["bench"]
        assert bench_dict["mode"] == "on-start"
        assert bench_dict["placement"] == "strict", (
            f"bench_placement='strict' must emit lowercase 'strict' on "
            f"the wire; got: {bench_dict!r}"
        )

    def test_invalid_bench_placement_rejected(self) -> None:
        # Negative: only "preferred" / "strict" are accepted. Anything
        # else raises ValidationError before the request leaves the SDK.
        from basilica.exceptions import ValidationError

        client = self._client()
        with pytest.raises(ValidationError) as exc_info:
            client._build_distributed_request(
                name="dlc-bad",
                source=None,
                image="x",
                port=80,
                env=None,
                cpu="1",
                memory="1Gi",
                gpu_count=1,
                gpu_models=None,
                min_gpu_memory_gb=None,
                world_size=WorldSize(min=1, target=1, max=1),
                provider_filter=None,
                topology_spread="provider-aware",
                nccl_env=None,
                bench="on-start",
                rendezvous_backend="etcd-v2",
                command=["python", "x.py"],
                args=None,
                pip_packages=None,
                ttl_seconds=None,
                enable_billing=True,
                bench_placement="invalid-placement",
            )
        assert exc_info.value.field == "bench_placement"

    def test_invalid_bench_mode_rejected(self) -> None:
        from basilica.exceptions import ValidationError

        client = self._client()
        with pytest.raises(ValidationError) as exc_info:
            client._build_distributed_request(
                name="dlc-test",
                source=None,
                image="x",
                port=80,
                env=None,
                cpu="1",
                memory="1Gi",
                gpu_count=1,
                gpu_models=None,
                min_gpu_memory_gb=None,
                world_size=WorldSize(min=1, target=1, max=1),
                provider_filter=None,
                topology_spread="provider-aware",
                nccl_env=None,
                bench="invalid-mode",
                rendezvous_backend="etcd-v2",
                command=["python", "x.py"],
                args=None,
                pip_packages=None,
                ttl_seconds=None,
                enable_billing=True,
            )
        assert exc_info.value.field == "bench"

    # -------------------------------------------------------------------------
    # Issue #452: $VAR expansion in `command=` must survive the `sh -c` wrap.
    #
    # The operator wraps `distributed.command` in `["/bin/sh", "-c", <cmd>]`
    # (operator distributed.rs build_worker_command). The pre-fix
    # implementation used plain `shlex.join`, which single-quotes any token
    # containing `$`, killing shell expansion. Live failure on UD f01dd43a
    # (2026-05-02 smoke): `int('$BASILICA_WORLD_TARGET')` raised ValueError.
    # -------------------------------------------------------------------------

    def _build_with_command(
        self, client: BasilicaClient, command: list
    ) -> Dict[str, Any]:
        return client._build_distributed_request(
            name="dlc-452",
            source=None,
            image="my-image",
            port=18789,
            env=None,
            cpu="1",
            memory="1Gi",
            gpu_count=1,
            gpu_models=None,
            min_gpu_memory_gb=None,
            world_size=WorldSize(min=1, target=1, max=1),
            provider_filter=None,
            topology_spread="provider-aware",
            nccl_env=None,
            bench="off",
            rendezvous_backend="etcd-v2",
            command=command,
            args=None,
            pip_packages=None,
            ttl_seconds=None,
            enable_billing=True,
        )

    def test_command_bash_dash_c_emits_script_verbatim(self) -> None:
        # Canonical "I am a shell script" shape: the script string lands
        # on `distributed.command` byte-for-byte, no surrounding quotes,
        # no escape of `$`. Issue #452.
        client = self._client()
        script = "echo $BASILICA_WORLD_TARGET"
        req = self._build_with_command(client, ["bash", "-c", script])
        assert req["distributed"]["command"] == script
        # Belt-and-braces: no single-quotes wrapping the env-var ref.
        assert "'$BASILICA_WORLD_TARGET'" not in req["distributed"]["command"]

    def test_command_sh_dash_c_also_recognised(self) -> None:
        # All four launcher spellings are recognised; the wrapper layer
        # is what matters, not which busybox/bash flavour the user typed.
        client = self._client()
        for launcher in ("sh", "bash", "/bin/sh", "/bin/bash"):
            req = self._build_with_command(
                client, [launcher, "-c", "true && echo $X"]
            )
            assert req["distributed"]["command"] == "true && echo $X", launcher

    def test_command_torchrun_argv_preserves_dollar_vars(self) -> None:
        # Real shape from examples/21_distributed_torchrun.py: a flat argv
        # list containing `$BASILICA_*` tokens. Pre-fix `shlex.join` wrote
        # `'--nnodes=$BASILICA_WORLD_TARGET'` (single-quoted), the operator
        # passed THAT to `sh -c`, the literal string reached the user's
        # `int(...)`, ValueError. Post-fix: tokens stay verbatim.
        client = self._client()
        req = self._build_with_command(
            client,
            [
                "torchrun",
                "--rdzv-backend=etcd-v2",
                "--rdzv-endpoint=$BASILICA_RDZV_ENDPOINT",
                "--rdzv-id=$BASILICA_RDZV_ID",
                "--nnodes=$BASILICA_WORLD_TARGET",
                "--nproc-per-node=$BASILICA_GPUS_PER_POD",
                "--max-restarts=10",
                "/workspace/all_reduce_smoke.py",
            ],
        )
        cmd = req["distributed"]["command"]
        # Positive: every $VAR ref appears unquoted.
        for var in (
            "$BASILICA_RDZV_ENDPOINT",
            "$BASILICA_RDZV_ID",
            "$BASILICA_WORLD_TARGET",
            "$BASILICA_GPUS_PER_POD",
        ):
            assert var in cmd, f"missing {var} in {cmd!r}"
        # Negative regression guard: the OLD `shlex.join` wrapped any
        # token containing `$` in single quotes (e.g.
        # `'--rdzv-endpoint=$BASILICA_RDZV_ENDPOINT'`). Locks the #452
        # fix in place -- if a future refactor restores `shlex.join`,
        # these substrings would re-appear and the test would fail.
        assert "'--rdzv-endpoint=$BASILICA_RDZV_ENDPOINT'" not in cmd, (
            f"regression: `shlex.join` quoting of $VAR token returned. "
            f"command: {cmd!r}"
        )
        assert "'--nnodes=$BASILICA_WORLD_TARGET'" not in cmd, (
            f"regression: `shlex.join` quoting of $VAR token returned. "
            f"command: {cmd!r}"
        )
        # Structure preserved.
        assert cmd.startswith("torchrun ")
        assert cmd.endswith(" /workspace/all_reduce_smoke.py")

    def test_command_with_whitespace_token_falls_back_to_quote(self) -> None:
        # Safety: a token with embedded whitespace MUST be quoted, otherwise
        # sh would split it into multiple argv elements. The fallback path
        # uses `shlex.quote` to preserve argv shape.
        client = self._client()
        req = self._build_with_command(
            client, ["my-binary", "--flag", "value with spaces"]
        )
        cmd = req["distributed"]["command"]
        # Quoted somehow (single-quote is shlex.quote's choice).
        assert "'value with spaces'" in cmd
        # Bare tokens stay bare.
        assert cmd.startswith("my-binary --flag ")

    def test_command_simple_invocation_no_quoting_overhead(self) -> None:
        # `command=["my-binary"]` produces just `my-binary` -- no quoting
        # added, no list-bracket leakage. Smoke that the trivial case
        # works after the helper refactor.
        client = self._client()
        req = self._build_with_command(client, ["my-binary"])
        assert req["distributed"]["command"] == "my-binary"


# =============================================================================
# Decorator (SDK arch § 5).
# =============================================================================


class TestDistributedDecorator:
    def test_returns_distributed_function_wrapper(self) -> None:
        @distributed(
            name="dlc-decorator-test",
            world_size=WorldSize(min=2, target=2, max=2),
            gpu_count=1,
        )
        def train_fn() -> None:
            pass

        assert isinstance(train_fn, DistributedFunction)
        assert train_fn._kwargs["name"] == "dlc-decorator-test"
        assert train_fn._kwargs["world_size"].target == 2

    def test_local_runs_function_in_process(self) -> None:
        executed = []

        @distributed(
            name="dlc-decorator-local",
            world_size=WorldSize(min=1, target=1, max=1),
            gpu_count=1,
        )
        def train_fn() -> None:
            executed.append(1)

        train_fn.local()
        assert executed == [1]

    def test_decorator_normalizes_provider_filter_dict(self) -> None:
        @distributed(
            name="dlc-pf-dict",
            world_size=WorldSize(min=1, target=1, max=1),
            provider_filter={"include": ["hyperstack"], "exclude": ["masscompute"]},
            gpu_count=1,
        )
        def train_fn() -> None:
            pass

        pf = train_fn._kwargs["provider_filter"]
        assert isinstance(pf, ProviderFilter)
        assert pf.include == ["hyperstack"]
        assert pf.exclude == ["masscompute"]

    def test_decorator_requires_world_size(self) -> None:
        with pytest.raises(ValueError, match="world_size"):
            @distributed(name="dlc-missing-ws")
            def _(): pass


# =============================================================================
# Exception attribute round-trip (SDK arch § 8 -- structured context).
# =============================================================================


class TestExceptionAttributes:
    def test_quota_exceeded_carries_three_numbers(self) -> None:
        e = QuotaExceeded(
            "namespace rank budget exceeded: current=8, requested=6, limit=10",
            current=8, requested=6, limit=10,
        )
        assert e.current == 8
        assert e.requested == 6
        assert e.limit == 10

    def test_below_minimum_world_carries_ready_required_and_timeout(self) -> None:
        e = BelowMinimumWorld(
            "ready=2, required_min=4 (timeout 300s)",
            ready=2,
            required_min=4,
            timeout=300,
        )
        assert e.ready == 2
        assert e.required_min == 4
        assert e.timeout == 300

    def test_below_minimum_world_timeout_optional(self) -> None:
        # Outside a wait context (e.g. raised from event-stream parsing),
        # timeout may be None.
        e = BelowMinimumWorld("inline", ready=0, required_min=2)
        assert e.timeout is None

    def test_world_size_out_of_bounds_carries_full_triple(self) -> None:
        e = WorldSizeOutOfBounds(
            "requested=12, min=4, max=8",
            requested=12, min=4, max=8,
        )
        assert e.requested == 12
        assert e.min == 4
        assert e.max == 8


# =============================================================================
# Issue #449 regression tests: status.distributed is read end-to-end.
#
# Before #449 the PyO3 binding had no `distributed` getter; every read
# property on `DistributedTraining` returned zeros / empty / None on a
# healthy cluster, and `wait_until_min_world(timeout=N)` raised
# BelowMinimumWorld immediately. These tests lock the contract: a
# regression that drops `distributed` from the PyO3 binding (or from
# `_coerce_to_dict`) would fail the negative test loudly.
# =============================================================================


class TestIssue449DeploymentResponseDistributed:
    """Mocked end-to-end: `client.get` -> `_coerce_to_dict` -> facade reads."""

    def _fake_pyo3_response(self) -> Any:
        """Build a stand-in for the PyO3 `DeploymentResponse` that exposes
        the full attribute set landed by issue #449.

        Uses an explicit instance with declared attributes (rather than a
        plain MagicMock) because the post-#449 `_coerce_to_dict` checks
        `isinstance(d, dict)` to distinguish a real PyO3 `distributed`
        attribute from a MagicMock auto-generated attribute.
        """

        class FakeDeployment:
            instance_name = "dlc-449-mock"
            user_id = "u-test"
            namespace = "u-test"
            image = "pytorch/pytorch:2.1.0-cuda12.1-cudnn8-runtime"
            state = "running"
            url = "https://dlc-449-mock.deployments.basilica.ai"
            created_at = "2026-05-02T10:00:00Z"
            updated_at = "2026-05-02T10:05:00Z"
            phase = "Running"
            message = None
            share_token = None
            share_url = None
            public_metadata = False
            distributed = {
                "worldSize": {
                    "ready": 2,
                    "target": 2,
                    "min": 2,
                    "max": 3,
                    "belowMinimum": False,
                },
                "ranks": [
                    {
                        "rank": 0,
                        "podName": "dlc-449-mock-0",
                        "nodeName": "basilica-verda-fin-03",
                        "provider": "verda",
                        "region": "FIN-03",
                        "phase": "Running",
                        "restarts": 0,
                    },
                    {
                        "rank": 1,
                        "podName": "dlc-449-mock-1",
                        "nodeName": "basilica-verda-fin-04",
                        "provider": "verda",
                        "region": "FIN-04",
                        "phase": "Running",
                        "restarts": 0,
                    },
                ],
                "transport": "hub-relay",
                "bench": {
                    "mode": "on-start",
                    "result": {
                        "measuredAt": "2026-05-02T10:00:30Z",
                        "busbwGbpsP50": 0.00897,
                        "sizeBytesSwept": [1048576, 16777216],
                        "probeNodeA": "basilica-verda-fin-03",
                        "probeNodeB": "basilica-verda-fin-04",
                    },
                    "lastAttemptOutcome": "success",
                },
            }

        return FakeDeployment()

    def test_world_returns_real_values_not_zeros(self) -> None:
        # The original bug: facade read zeros on a healthy cluster.
        client = MagicMock()
        client.get.return_value = self._fake_pyo3_response()
        training = DistributedTraining(client, "dlc-449-mock")
        ws = training.world
        assert ws.ready == 2
        assert ws.target == 2
        assert ws.min == 2
        assert ws.max == 3
        assert ws.below_minimum is False

    def test_ranks_returns_two_pods_with_provider_and_region(self) -> None:
        client = MagicMock()
        client.get.return_value = self._fake_pyo3_response()
        training = DistributedTraining(client, "dlc-449-mock")
        ranks = training.ranks
        assert len(ranks) == 2
        assert ranks[0].rank == 0
        assert ranks[0].pod_name == "dlc-449-mock-0"
        assert ranks[0].node == "basilica-verda-fin-03"
        assert ranks[0].provider == "verda"
        assert ranks[0].region == "FIN-03"
        assert ranks[0].phase == "Running"
        assert ranks[1].rank == 1
        assert ranks[1].pod_name == "dlc-449-mock-1"
        assert ranks[1].provider == "verda"

    def test_bench_returns_populated_result(self) -> None:
        client = MagicMock()
        client.get.return_value = self._fake_pyo3_response()
        training = DistributedTraining(client, "dlc-449-mock")
        bench = training.bench
        assert bench is not None
        assert bench.busbw_gbps_p50 == 0.00897
        assert bench.size_bytes_swept == [1048576, 16777216]
        assert bench.probe_node_a == "basilica-verda-fin-03"
        assert bench.probe_node_b == "basilica-verda-fin-04"

    def test_metrics_returns_real_world_size(self) -> None:
        client = MagicMock()
        client.get.return_value = self._fake_pyo3_response()
        training = DistributedTraining(client, "dlc-449-mock")
        m = training.metrics()
        assert m.world_ready == 2
        assert m.world_target == 2
        assert m.rank_restarts_total == 0

    def test_wait_until_min_world_returns_immediately_when_at_min(self) -> None:
        # Before #449, this raised BelowMinimumWorld(ready=0, required_min=0)
        # because both zeros came from the dropped `distributed` block.
        client = MagicMock()
        client.get.return_value = self._fake_pyo3_response()
        training = DistributedTraining(client, "dlc-449-mock")
        # Should NOT raise (ready=2 >= min=2).
        training.wait_until_min_world(timeout=10)

    def test_NEGATIVE_coerce_to_dict_carries_distributed_and_image(self) -> None:
        """Locks the bug from regressing.

        Before #449: `_coerce_to_dict(client.get(name))` returned a dict
        with only `[namespace, state, url, userId]` (4 keys). After #449
        the dict MUST also carry `image`, `phase`, `createdAt`, and
        crucially `distributed`. A regression that drops the PyO3
        `distributed` getter (or stops walking it in `_coerce_to_dict`)
        would fail this test loudly.
        """
        from basilica.distributed import _coerce_to_dict

        d = _coerce_to_dict(self._fake_pyo3_response())
        # The fields the original bug-report logged as missing:
        for required_key in (
            "instanceName",
            "userId",
            "namespace",
            "image",
            "state",
            "url",
            "createdAt",
            "updatedAt",
            "phase",
            "distributed",
        ):
            assert required_key in d, (
                f"_coerce_to_dict must carry '{required_key}' (issue #449); "
                f"got keys: {sorted(d.keys())}"
            )
        # And the distributed block carries the operator's camelCase shape.
        dist = d["distributed"]
        assert dist["worldSize"]["ready"] == 2
        assert dist["worldSize"]["belowMinimum"] is False
        assert len(dist["ranks"]) == 2
        assert dist["bench"]["mode"] == "on-start"
        assert dist["bench"]["result"]["busbwGbpsP50"] == 0.00897


# =============================================================================
# Issue #454 / #453 regression tests: the `Deployment` wrapper preserves
# `distributed` (and other PyO3 fields) end-to-end through `client.get(name)`.
#
# PR #451 fixed the PyO3 binding (`DeploymentResponse.distributed` getter)
# but the lock-down test only covered `_coerce_to_dict(pyo3_response)`,
# NOT `_coerce_to_dict(client.get(name))`. The high-level Python wrapper
# `Deployment._from_response(...)` continued to drop `distributed`,
# `image`, `phase`, `message`, `share_token`, `share_url`,
# `public_metadata`, so the wrapper-path was still broken on `main` after
# #451 merged. Issues #453 (elasticity-demo agent) and #454 (this fix)
# both reported the same wrapper-strip symptom.
# =============================================================================


class TestIssue454DeploymentWrapperCarriesDistributed:
    """End-to-end: `BasilicaClient(...).get(name)` -> `Deployment` wrapper
    -> `_coerce_to_dict` -> facade reads. The mock injects at the lowest
    layer (`_BasilicaClient.get_deployment`) so the assertion exercises
    the real `Deployment._from_response` factory.
    """

    def _fake_pyo3_response(self) -> Any:
        """Mirror of the issue #449 fixture but used through the wrapper path.

        Must use a class with declared attributes (not MagicMock) for the
        same reason as `TestIssue449...`: `_coerce_to_dict` does an
        `isinstance(d, dict)` check on `obj.distributed`, which a
        MagicMock auto-attr would falsely truthy past.
        """

        class FakeReplicas:
            ready = 2
            desired = 2

        class FakeDeployment:
            instance_name = "dlc-454-mock"
            user_id = "u-test"
            namespace = "u-test"
            image = "pytorch/pytorch:2.1.0-cuda12.1-cudnn8-runtime"
            state = "running"
            url = "https://dlc-454-mock.deployments.basilica.ai"
            created_at = "2026-05-02T10:00:00Z"
            updated_at = "2026-05-02T10:05:00Z"
            phase = "Running"
            message = None
            share_token = None
            share_url = None
            public_metadata = False
            replicas = FakeReplicas()
            distributed = {
                "worldSize": {
                    "ready": 2,
                    "target": 2,
                    "min": 2,
                    "max": 3,
                    "belowMinimum": False,
                },
                "ranks": [
                    {
                        "rank": 0,
                        "podName": "dlc-454-mock-0",
                        "nodeName": "basilica-verda-fin-03",
                        "provider": "verda",
                        "region": "FIN-03",
                        "phase": "Running",
                        "restarts": 0,
                    },
                    {
                        "rank": 1,
                        "podName": "dlc-454-mock-1",
                        "nodeName": "basilica-verda-fin-04",
                        "provider": "verda",
                        "region": "FIN-04",
                        "phase": "Running",
                        "restarts": 0,
                    },
                ],
                "transport": "hub-relay",
                "bench": {
                    "mode": "on-start",
                    "result": {
                        "measuredAt": "2026-05-02T10:00:30Z",
                        "busbwGbpsP50": 0.00897,
                        "sizeBytesSwept": [1048576, 16777216],
                        "probeNodeA": "basilica-verda-fin-03",
                        "probeNodeB": "basilica-verda-fin-04",
                    },
                    "lastAttemptOutcome": "success",
                },
            }

        return FakeDeployment()

    def _build_client(self) -> BasilicaClient:
        """Construct a real `BasilicaClient` with a mocked `_BasilicaClient`
        underneath so the wrapper path (`get` -> `get_deployment` ->
        `Deployment._from_response`) executes for real.
        """
        client = BasilicaClient(
            base_url="https://api.test.invalid",
            api_key="fake-test-token",
        )
        # Replace the PyO3 inner client with a mock that returns our fixture.
        # `BasilicaClient.get_deployment` delegates to `self._client.get_deployment`.
        client._client = MagicMock()
        client._client.get_deployment.return_value = self._fake_pyo3_response()
        return client

    def test_wrapper_carries_distributed(self) -> None:
        """The load-bearing assertion: `client.get(name).distributed` must
        be the operator's camelCase dict, not None and not stripped.
        """
        client = self._build_client()
        deployment = client.get("dlc-454-mock")
        assert deployment.distributed is not None, (
            "Deployment wrapper dropped `distributed` (issue #454). "
            "PR #451 fixed PyO3; the wrapper was still broken on main."
        )
        assert deployment.distributed["worldSize"]["ready"] == 2
        assert deployment.distributed["worldSize"]["target"] == 2
        assert deployment.distributed["worldSize"]["min"] == 2
        assert deployment.distributed["worldSize"]["belowMinimum"] is False
        assert len(deployment.distributed["ranks"]) == 2

    def test_wrapper_carries_other_previously_dropped_fields(self) -> None:
        """`image`, `phase`, `message`, `share_token`, `share_url`,
        `public_metadata` were all dropped by `Deployment._from_response`
        before this fix. They are part of the PyO3 binding's surface
        (see `crates/basilica-sdk-python/src/types.rs::DeploymentResponse`)
        and must reach the user.
        """
        client = self._build_client()
        deployment = client.get("dlc-454-mock")
        assert deployment.image == "pytorch/pytorch:2.1.0-cuda12.1-cudnn8-runtime"
        assert deployment.phase == "Running"
        assert deployment.message is None
        assert deployment.share_token is None
        assert deployment.share_url is None
        assert deployment.public_metadata is False
        # Pre-existing fields should still survive.
        assert deployment.name == "dlc-454-mock"
        assert deployment.namespace == "u-test"
        assert deployment.user_id == "u-test"
        assert deployment.state == "running"
        assert deployment.created_at == "2026-05-02T10:00:00Z"
        assert deployment.updated_at == "2026-05-02T10:05:00Z"

    def test_NEGATIVE_coerce_to_dict_after_wrapper_carries_distributed(self) -> None:
        """The test PR #451 should have had.

        PR #451's `test_NEGATIVE_coerce_to_dict_carries_distributed_and_image`
        passed a raw PyO3 fixture into `_coerce_to_dict` and asserted the
        keys round-trip. That test did NOT exercise
        `Deployment._from_response`, so the wrapper-strip bug went
        undetected and shipped to users.

        This test goes one layer deeper: the input to `_coerce_to_dict`
        is the `Deployment` wrapper produced by `client.get(name)`. If
        the wrapper drops a field, this assertion fails.
        """
        from basilica.distributed import _coerce_to_dict

        client = self._build_client()
        deployment = client.get("dlc-454-mock")
        d = _coerce_to_dict(deployment)
        for required_key in (
            "instanceName",
            "userId",
            "namespace",
            "image",
            "state",
            "url",
            "createdAt",
            "updatedAt",
            "phase",
            "distributed",
        ):
            assert required_key in d, (
                f"_coerce_to_dict(client.get(name)) must carry "
                f"'{required_key}' (issue #454); got keys: {sorted(d.keys())}"
            )
        # The distributed block survives the wrapper round-trip.
        dist = d["distributed"]
        assert dist["worldSize"]["ready"] == 2
        assert dist["worldSize"]["belowMinimum"] is False
        assert len(dist["ranks"]) == 2
        assert dist["bench"]["mode"] == "on-start"

    def test_distributed_training_facade_reads_via_wrapper(self) -> None:
        """The user-visible symptom: `DistributedTraining.world` must
        return real values when driven through `client.get(name)`. This
        is the path the elasticity-demo agent (#453) exercised before
        having to subclass `refresh()` to bypass the wrapper.
        """
        client = self._build_client()
        training = DistributedTraining(client, "dlc-454-mock")
        ws = training.world
        assert ws.ready == 2, (
            f"DistributedTraining.world.ready must be 2 (got {ws.ready}); "
            f"this is the symptom #453 / #454 reported on a healthy UD."
        )
        assert ws.target == 2
        assert ws.min == 2
        assert ws.max == 3
        assert ws.below_minimum is False
        # `wait_until_min_world` must NOT raise BelowMinimumWorld here:
        # ready=2 >= min=2.
        training.wait_until_min_world(timeout=1)


# =============================================================================
# Phase 5b (#445) -- distributed UD completion semantics.
# =============================================================================


def _make_terminal_training(
    name: str = "dlc-done",
    namespace: str = "u-test",
    phase: str = "succeeded",
    rank_exits: list = None,
) -> DistributedTraining:
    """Build a `DistributedTraining` with a pre-populated terminal status.

    The phase is at the top level of `_cached_status` (mirroring
    `DeploymentResponse.phase`). `rank_exits` lives under
    `distributed.rankExits` (camelCase JSON keys).
    """
    client = MagicMock()
    client._client = MagicMock()
    fake_response = MagicMock()
    fake_response.namespace = namespace
    fake_response.phase = phase
    fake_response._distributed_status = {
        "worldSize": {
            "ready": 0,
            "target": 2,
            "min": 2,
            "max": 2,
            "belowMinimum": True,
        },
        "rankExits": rank_exits or [
            {
                "rank": 0,
                "exitCode": 0,
                "terminationReason": "Completed",
                "restartCount": 1,
            },
            {
                "rank": 1,
                "exitCode": 0,
                "terminationReason": "Completed",
                "restartCount": 1,
            },
        ],
    }
    client.get.return_value = fake_response
    training = DistributedTraining(client, name)
    training._cached_status = {
        "namespace": namespace,
        "phase": phase,
        "distributed": fake_response._distributed_status,
    }
    training.namespace = namespace
    training.rendezvous_endpoint = (
        f"{name}-rendezvous.{namespace}.svc.cluster.local:2379"
    )
    return training


class TestPhase5bCompletionSemantics:
    """Phase 5b (#445) SDK-side terminal-state contracts."""

    def test_phase_property_reads_succeeded_from_cached_status(self) -> None:
        training = _make_terminal_training(phase="succeeded")
        assert training.phase == "succeeded"
        assert training.is_terminal is True

    def test_phase_property_reads_running_phase(self) -> None:
        # Non-terminal phase: `is_terminal` False, scale path open.
        client = MagicMock()
        client._client = MagicMock()
        client.get.return_value = MagicMock(
            namespace="u-test",
            phase="ready",
            _distributed_status={
                "worldSize": {
                    "ready": 2,
                    "target": 2,
                    "min": 2,
                    "max": 2,
                    "belowMinimum": False,
                }
            },
        )
        training = DistributedTraining(client, "dlc-running")
        training._cached_status = {
            "namespace": "u-test",
            "phase": "ready",
            "distributed": {
                "worldSize": {
                    "ready": 2,
                    "target": 2,
                    "min": 2,
                    "max": 2,
                    "belowMinimum": False,
                }
            },
        }
        training.namespace = "u-test"
        assert training.phase == "ready"
        assert training.is_terminal is False

    def test_rank_exits_property_reads_camelcase_json(self) -> None:
        training = _make_terminal_training(phase="succeeded")
        exits = training.rank_exits
        assert len(exits) == 2
        assert all(isinstance(e, RankExit) for e in exits)
        assert exits[0].rank == 0
        assert exits[0].exit_code == 0
        assert exits[0].termination_reason == "Completed"
        assert exits[0].restart_count == 1
        assert exits[1].rank == 1
        assert exits[1].exit_code == 0

    def test_rank_exits_empty_when_running(self) -> None:
        # No `rankExits` key in distributed status -> empty list.
        client = MagicMock()
        client._client = MagicMock()
        training = DistributedTraining(client, "dlc-running")
        training._cached_status = {
            "namespace": "u-test",
            "phase": "ready",
            "distributed": {
                "worldSize": {
                    "ready": 2,
                    "target": 2,
                    "min": 2,
                    "max": 2,
                    "belowMinimum": False,
                }
            },
        }
        training.namespace = "u-test"
        assert training.rank_exits == []

    def test_scale_on_succeeded_raises_ud_terminal_state(self) -> None:
        """SDK-side rejection BEFORE the API call. The operator's
        `UDTerminalState` Warning Event is the defense in depth for
        kubectl-edit mutations that bypass the SDK."""
        training = _make_terminal_training(phase="succeeded")
        # Stub the refresh fetch so the scale path has the latest phase.
        training._client.get = MagicMock(
            return_value=MagicMock(
                namespace="u-test",
                phase="succeeded",
                _distributed_status={
                    "worldSize": {
                        "ready": 0,
                        "target": 2,
                        "min": 2,
                        "max": 2,
                        "belowMinimum": True,
                    }
                },
            )
        )

        with pytest.raises(UDTerminalState) as exc_info:
            training.scale(target=4)
        assert exc_info.value.phase == "succeeded"
        assert exc_info.value.requested_target == 4
        # The PyO3 patch MUST NOT have been issued -- the SDK rejection
        # is local.
        training._client._client.scale_distributed_deployment.assert_not_called()

    def test_scale_on_failed_raises_ud_terminal_state(self) -> None:
        """`failed` is also terminal; mirrors the operator's
        `is_terminal` set."""
        training = _make_terminal_training(phase="failed")
        training._client.get = MagicMock(
            return_value=MagicMock(
                namespace="u-test",
                phase="failed",
                _distributed_status={
                    "worldSize": {
                        "ready": 1,
                        "target": 2,
                        "min": 2,
                        "max": 2,
                        "belowMinimum": True,
                    }
                },
            )
        )

        with pytest.raises(UDTerminalState) as exc_info:
            training.scale(target=4)
        assert exc_info.value.phase == "failed"
        training._client._client.scale_distributed_deployment.assert_not_called()

    def test_wait_until_complete_returns_world_when_already_succeeded_at_entry_raises(
        self,
    ) -> None:
        """`wait_until_complete` distinguishes 'I waited and it
        completed' from 'I called this on an already-complete UD' --
        the latter raises `UDTerminalState` so the caller switches to
        reading `t.world` / `t.bench` / `t.rank_exits` directly."""
        training = _make_terminal_training(phase="succeeded")
        training._client.get = MagicMock(
            return_value=MagicMock(
                namespace="u-test",
                phase="succeeded",
                _distributed_status={
                    "worldSize": {
                        "ready": 0,
                        "target": 2,
                        "min": 2,
                        "max": 2,
                        "belowMinimum": True,
                    }
                },
            )
        )

        with pytest.raises(UDTerminalState) as exc_info:
            training.wait_until_complete(timeout=1)
        assert exc_info.value.phase == "succeeded"
        assert exc_info.value.requested_target is None

    def test_wait_until_complete_returns_world_when_phase_flips_to_succeeded(
        self,
    ) -> None:
        """Real waiting path: phase starts non-terminal, flips to
        `succeeded` mid-wait, returns `WorldStatus`. We simulate two
        polls: first sees `pending`, second sees `succeeded`."""
        # Counter to mutate the response across refresh() calls.
        call_count = {"n": 0}

        def fake_get(name: str) -> MagicMock:
            call_count["n"] += 1
            phase = "pending" if call_count["n"] <= 1 else "succeeded"
            return MagicMock(
                namespace="u-test",
                phase=phase,
                _distributed_status={
                    "worldSize": {
                        "ready": 0,
                        "target": 2,
                        "min": 2,
                        "max": 2,
                        "belowMinimum": True,
                    },
                    "rankExits": [],
                },
            )

        client = MagicMock()
        client._client = MagicMock()
        client.get.side_effect = fake_get
        training = DistributedTraining(client, "dlc-flip")
        # Start the wait. timeout=1 so the test is fast; we expect the
        # second poll to flip the phase, returning WorldStatus.
        ws = training.wait_until_complete(timeout=1)
        assert isinstance(ws, WorldStatus)

    def test_wait_until_complete_raises_below_minimum_on_failed(self) -> None:
        """`failed` mid-wait raises `BelowMinimumWorld` with per-rank
        exit detail. The first refresh sees `pending` (so the entry
        guard does not fire); the second sees `failed`."""
        call_count = {"n": 0}

        def fake_get(name: str) -> MagicMock:
            call_count["n"] += 1
            phase = "pending" if call_count["n"] <= 1 else "failed"
            return MagicMock(
                namespace="u-test",
                phase=phase,
                _distributed_status={
                    "worldSize": {
                        "ready": 1,
                        "target": 2,
                        "min": 2,
                        "max": 2,
                        "belowMinimum": True,
                    },
                    "rankExits": [
                        {
                            "rank": 0,
                            "exitCode": 0,
                            "terminationReason": "Completed",
                            "restartCount": 0,
                        },
                        {
                            "rank": 1,
                            "exitCode": 137,
                            "terminationReason": "OOMKilled",
                            "restartCount": 1,
                        },
                    ],
                },
            )

        client = MagicMock()
        client._client = MagicMock()
        client.get.side_effect = fake_get
        training = DistributedTraining(client, "dlc-fail")

        with pytest.raises(BelowMinimumWorld) as exc_info:
            training.wait_until_complete(timeout=1)
        # The error message must mention the OOMKilled rank so the
        # caller has actionable context (per CLAUDE.md
        # `feedback_evidence_backed_issues`).
        assert "rank 1" in str(exc_info.value)
        assert "137" in str(exc_info.value)

    def test_wait_until_target_world_returns_clean_on_succeeded(self) -> None:
        """Phase 5b: `wait_until_target_world` returns cleanly when the
        UD has reached `succeeded` (training is done by definition).
        Was previously timing out on `ready=0` after the
        StatefulSet scaled to 0."""
        # First poll observes `succeeded`; with `ready=0` (workers
        # scaled to 0), the legacy `ready >= target` check would have
        # timed out.
        client = MagicMock()
        client._client = MagicMock()
        client.get.return_value = MagicMock(
            namespace="u-test",
            phase="succeeded",
            _distributed_status={
                "worldSize": {
                    "ready": 0,
                    "target": 2,
                    "min": 2,
                    "max": 2,
                    "belowMinimum": True,
                },
                "rankExits": [
                    {
                        "rank": 0,
                        "exitCode": 0,
                        "terminationReason": "Completed",
                        "restartCount": 1,
                    },
                    {
                        "rank": 1,
                        "exitCode": 0,
                        "terminationReason": "Completed",
                        "restartCount": 1,
                    },
                ],
            },
        )
        training = DistributedTraining(client, "dlc-finish")
        # Pre-populate the cache so the wait sees `succeeded` on its
        # first refresh.
        training._cached_status = {
            "namespace": "u-test",
            "phase": "succeeded",
            "distributed": client.get.return_value._distributed_status,
        }
        training.namespace = "u-test"

        # Should return without raising; explicit None result.
        result = training.wait_until_target_world(timeout=1)
        assert result is None

    def test_async_phase_5b_methods_exist(self) -> None:
        """Phase 5b: `wait_until_complete_async` exists for parity with
        the sync method (SDK arch § 9)."""
        training = _make_terminal_training()
        assert hasattr(training, "wait_until_complete")
        assert hasattr(training, "wait_until_complete_async")
        assert asyncio.iscoroutinefunction(training.wait_until_complete_async)


# =============================================================================
# Issue #486: deploy_distributed must clean up the UD on wait_until_min_world
# failure so the documented `deploy -> use -> delete()` flow doesn't leak
# cluster state on every distributed-UD failure path.
# =============================================================================


def _pyo3_response(
    instance_name: str,
    namespace: str,
    distributed: Dict[str, Any],
) -> MagicMock:
    """A PyO3-shaped DeploymentResponse with a real `distributed` dict.

    The `distributed` attribute is set to a real dict (not a MagicMock
    auto-attr) so `Deployment._from_response` picks it up and
    `_coerce_to_dict` later reads it via `isinstance(..., dict)`.
    """
    response = MagicMock()
    response.instance_name = instance_name
    response.friendly_name = ""
    response.user_id = namespace
    response.namespace = namespace
    response.image = "pytorch/pytorch:2.4.0-cuda12.4-cudnn9-runtime"
    response.state = "creating"
    response.url = ""
    response.created_at = "2026-05-08T00:00:00Z"
    response.updated_at = "2026-05-08T00:00:00Z"
    response.phase = "running"
    response.message = None
    response.share_token = None
    response.share_url = None
    response.public_metadata = False
    response.distributed = distributed
    return response


def _below_min_pyo3_response(
    instance_name: str = "dlc-486-mock",
    namespace: str = "u-test",
) -> MagicMock:
    """A response stuck below `world_size.min`. Makes
    `wait_until_min_world(timeout=0)` raise BelowMinimumWorld
    synchronously."""
    return _pyo3_response(
        instance_name,
        namespace,
        {
            "worldSize": {
                "ready": 0,
                "target": 2,
                "min": 2,
                "max": 2,
                "belowMinimum": True,
            },
        },
    )


def _noop_train() -> None:
    """Per-rank no-op for tests that exercise the deploy path."""
    print("x")


def _build_minimal_client() -> BasilicaClient:
    """Construct a real BasilicaClient with a mocked PyO3 inner client.

    Bypasses real auth via `__new__` so tests don't need credentials.
    """
    client = BasilicaClient.__new__(BasilicaClient)
    client._base_url = "https://api.test.invalid"
    client._client = MagicMock()
    return client


class TestDeployDistributedCleanupOnFailure:
    """Issue #486: any failure inside `deploy_distributed` must
    best-effort `delete()` the UD before re-raising so the documented
    `deploy -> use -> delete()` flow doesn't leak the UD on the failure
    path."""

    def _wire_below_min(self, client: BasilicaClient) -> MagicMock:
        response = _below_min_pyo3_response()
        client._client.create_distributed_deployment.return_value = response
        client._client.get_deployment.return_value = response
        client._client.delete_deployment.return_value = MagicMock()
        return response

    def test_below_minimum_triggers_delete_and_reraises(self) -> None:
        client = _build_minimal_client()
        response = self._wire_below_min(client)

        with pytest.raises(BelowMinimumWorld):
            client._deploy_distributed_impl(
                name="dlc-486",
                source=_noop_train,
                world_size=WorldSize(min=2, target=2, max=2),
                timeout=0,
            )

        client._client.delete_deployment.assert_called_once_with(
            response.instance_name
        )

    def test_delete_failure_does_not_swallow_original(self) -> None:
        """If best-effort cleanup itself raises, the ORIGINAL exception
        must still reach the caller. This is the contract that lets
        users still observe (and act on) the deploy failure even when
        cluster cleanup is broken."""
        client = _build_minimal_client()
        response = self._wire_below_min(client)
        client._client.delete_deployment.side_effect = RuntimeError(
            "API gateway 503 — cleanup attempt failed"
        )

        with pytest.raises(BelowMinimumWorld):
            client._deploy_distributed_impl(
                name="dlc-486",
                source=_noop_train,
                world_size=WorldSize(min=2, target=2, max=2),
                timeout=0,
            )

        # Cleanup was attempted exactly once; the secondary failure was
        # swallowed in favor of the original BelowMinimumWorld.
        client._client.delete_deployment.assert_called_once_with(
            response.instance_name
        )

    def test_arbitrary_exception_in_wait_triggers_cleanup(self) -> None:
        """The wrapper must clean up on ANY exception from
        `wait_until_min_world`, not just `BelowMinimumWorld` (issue #486
        explicitly calls out timeout/server-error/network glitch paths
        as also leaking the UD)."""
        client = _build_minimal_client()
        response = self._wire_below_min(client)
        # Force the next get_deployment call (inside wait_until_min_world's
        # final refresh) to blow up with a non-BelowMinimumWorld exception.
        # The first call (during deploy_distributed's `training.refresh()`)
        # succeeds; the second call (during the wait) raises.
        first_response = response
        client._client.get_deployment.side_effect = [
            first_response,
            ConnectionError("network glitch"),
        ]

        with pytest.raises(ConnectionError):
            client._deploy_distributed_impl(
                name="dlc-486",
                source=_noop_train,
                world_size=WorldSize(min=2, target=2, max=2),
                timeout=0,
            )

        client._client.delete_deployment.assert_called_once_with(
            response.instance_name
        )

    def test_success_path_does_not_call_delete(self) -> None:
        """Regression guard: the cleanup wrapper must not fire when
        `wait_until_min_world` returns cleanly. Issue #486 fix is
        purely an exception-cleanup wrapper; the success path is
        unchanged."""
        client = _build_minimal_client()
        response = _pyo3_response(
            "dlc-486-ok",
            "u-test",
            {
                "worldSize": {
                    "ready": 2,
                    "target": 2,
                    "min": 2,
                    "max": 2,
                    "belowMinimum": False,
                },
            },
        )
        client._client.create_distributed_deployment.return_value = response
        client._client.get_deployment.return_value = response

        training = client._deploy_distributed_impl(
            name="dlc-486",
            source=_noop_train,
            world_size=WorldSize(min=2, target=2, max=2),
            timeout=10,
        )

        assert training.name == "dlc-486-ok"
        client._client.delete_deployment.assert_not_called()


class TestDeployDistributedAsyncCleanupOnFailure:
    """Async parity for issue #486 (SDK arch § 9 — every facade method
    has an _async counterpart, and the cleanup contract must hold for
    both)."""

    def test_async_below_minimum_triggers_delete_and_reraises(self) -> None:
        client = _build_minimal_client()
        response = _below_min_pyo3_response()
        client._client.create_distributed_deployment.return_value = response
        client._client.get_deployment.return_value = response
        client._client.delete_deployment.return_value = MagicMock()

        async def run() -> None:
            await client._deploy_distributed_impl_async(
                name="dlc-486",
                source=_noop_train,
                world_size=WorldSize(min=2, target=2, max=2),
                timeout=0,
            )

        with pytest.raises(BelowMinimumWorld):
            asyncio.run(run())

        client._client.delete_deployment.assert_called_once_with(
            response.instance_name
        )

    def test_async_delete_failure_does_not_swallow_original(self) -> None:
        client = _build_minimal_client()
        response = _below_min_pyo3_response()
        client._client.create_distributed_deployment.return_value = response
        client._client.get_deployment.return_value = response
        client._client.delete_deployment.side_effect = RuntimeError(
            "API gateway 503 during cleanup"
        )

        async def run() -> None:
            await client._deploy_distributed_impl_async(
                name="dlc-486",
                source=_noop_train,
                world_size=WorldSize(min=2, target=2, max=2),
                timeout=0,
            )

        with pytest.raises(BelowMinimumWorld):
            asyncio.run(run())

        client._client.delete_deployment.assert_called_once_with(
            response.instance_name
        )


# =============================================================================
# Issue B/N (refs #506): bench-decoupling SDK tests.
#
# These codify that `wait_until_min_world` polls workers ONLY and is
# unaffected by `status.distributed.bench.phase`. They also exercise
# the new `bench_status` accessor and `wait_until_bench_complete`
# opt-in waiter.
# =============================================================================


class TestBenchPhaseDecoupling:
    """Bench probe lifecycle is published asynchronously; never gates `world.ready`."""

    def _make_status_dict(
        self, ready: int, min_: int, bench_phase: str, bench_result: bool = False
    ) -> Any:
        bench: Dict[str, Any] = {
            "mode": "on-start",
            "phase": bench_phase,
        }
        if bench_phase != "Pending":
            bench["startedAt"] = "2026-05-08T12:00:00Z"
        if bench_phase in {"Succeeded", "Failed", "TimedOut"}:
            bench["completedAt"] = "2026-05-08T12:10:00Z"
        if bench_phase == "Succeeded":
            bench["lastAttemptOutcome"] = "success"
        if bench_phase == "Failed":
            bench["lastAttemptOutcome"] = "error"
        if bench_phase == "TimedOut":
            bench["lastAttemptOutcome"] = "timeout"
        if bench_phase == "Skipped":
            bench["lastAttemptOutcome"] = "skipped"
            bench["lastAttemptAt"] = "2026-05-08T12:01:00Z"
        if bench_result:
            bench["result"] = {
                "measuredAt": "2026-05-08T12:00:00Z",
                "busbwGbpsP50": 12.5,
                "sizeBytesSwept": [1048576],
                "probeNodeA": "n-a",
                "probeNodeB": "n-b",
            }

        class FakeDeployment:
            instance_name = "ud-bench-decouple"
            user_id = "u-test"
            namespace = "u-test"
            image = "pytorch/pytorch"
            state = "running"
            url = "https://x"
            created_at = "2026-05-08T12:00:00Z"
            updated_at = "2026-05-08T12:00:00Z"
            phase = "Running"
            message = None
            share_token = None
            share_url = None
            public_metadata = False
            distributed = {
                "worldSize": {
                    "ready": ready,
                    "target": min_,
                    "min": min_,
                    "max": min_,
                    "belowMinimum": ready < min_,
                },
                "ranks": [],
                "transport": "hub-relay",
                "bench": bench,
            }

        return FakeDeployment()

    def test_wait_until_min_world_returns_when_workers_ready_bench_pending(self) -> None:
        """LOAD-BEARING: workers Ready -> world.ready >= min, even if bench is Pending."""
        client = MagicMock()
        client.get.return_value = self._make_status_dict(2, 2, "Pending")
        training = DistributedTraining(client, "ud-bench-decouple")
        # Should NOT raise even though bench is Pending.
        training.wait_until_min_world(timeout=10)

    def test_wait_until_min_world_returns_when_workers_ready_bench_failed(self) -> None:
        """LOAD-BEARING: bench Failed does NOT propagate to world.ready.

        Post-S7: ``bench_status`` public accessor is gone; verify via
        ``training.bench`` (``None`` on Failed) + ``bench_diagnostics``
        (dict carrying ``phase``).
        """
        client = MagicMock()
        client.get.return_value = self._make_status_dict(2, 2, "Failed")
        training = DistributedTraining(client, "ud-bench-decouple")
        training.wait_until_min_world(timeout=10)
        # bench=None on non-Succeeded outcomes (S7 simplification).
        assert training.bench is None
        diag = training.bench_diagnostics
        assert diag is not None
        assert diag["phase"] == "Failed"

    def test_wait_until_min_world_returns_when_workers_ready_bench_timed_out(
        self,
    ) -> None:
        """LOAD-BEARING: bench TimedOut does NOT propagate to world.ready."""
        client = MagicMock()
        client.get.return_value = self._make_status_dict(2, 2, "TimedOut")
        training = DistributedTraining(client, "ud-bench-decouple")
        training.wait_until_min_world(timeout=10)
        assert training.bench is None
        diag = training.bench_diagnostics
        assert diag is not None
        assert diag["phase"] == "TimedOut"

    def test_bench_diagnostics_reflects_each_lifecycle_state(self) -> None:
        """Bench lifecycle is observable through ``bench_diagnostics``
        (dict) independent of UD phase. ``training.bench`` collapses
        non-success to ``None``."""
        for phase in ["Pending", "Running", "Succeeded", "Failed", "TimedOut"]:
            client = MagicMock()
            client.get.return_value = self._make_status_dict(
                2, 2, phase, bench_result=(phase == "Succeeded")
            )
            training = DistributedTraining(client, "ud-bench-decouple")
            diag = training.bench_diagnostics
            assert diag is not None, f"bench_diagnostics absent for phase={phase}"
            assert diag["phase"] == phase
            assert diag["mode"] == "on-start"

    def test_bench_diagnostics_returns_none_when_bench_off(self) -> None:
        class FakeDeployment:
            instance_name = "ud"
            user_id = "u"
            namespace = "u"
            image = "x"
            state = "running"
            url = "https://x"
            created_at = "t"
            updated_at = "t"
            phase = "Running"
            message = None
            share_token = None
            share_url = None
            public_metadata = False
            distributed = {
                "worldSize": {
                    "ready": 2,
                    "target": 2,
                    "min": 2,
                    "max": 2,
                    "belowMinimum": False,
                },
                "ranks": [],
                "transport": "hub-relay",
            }

        client = MagicMock()
        client.get.return_value = FakeDeployment()
        training = DistributedTraining(client, "ud")
        assert training.bench_diagnostics is None
        assert training.bench is None

    def test_bench_accessor_returns_result_payload_on_succeeded(self) -> None:
        """``training.bench -> Optional[BenchResult]`` (canonical surface)."""
        client = MagicMock()
        client.get.return_value = self._make_status_dict(
            2, 2, "Succeeded", bench_result=True
        )
        training = DistributedTraining(client, "ud-bench-decouple")
        r = training.bench
        assert r is not None
        assert r.busbw_gbps_p50 == 12.5
        assert r.probe_node_a == "n-a"

    def test_bench_returns_none_on_non_success_phases(self) -> None:
        """Post-S7: ``training.bench`` collapses Skipped/Failed/TimedOut/Pending to None."""
        for phase in ["Pending", "Running", "Skipped", "Failed", "TimedOut"]:
            client = MagicMock()
            client.get.return_value = self._make_status_dict(2, 2, phase)
            training = DistributedTraining(client, f"ud-bench-{phase.lower()}")
            assert training.bench is None, (
                f"training.bench must be None for phase={phase}; got non-None"
            )


# =============================================================================
# Issue B/N (refs #506) follow-up: deploy_distributed default does NOT
# wait for the bench probe; bench is opt-in via `wait_for_bench=` modes.
#
# Pre-this-change behaviour: deploy_distributed implicitly blocked on
# bench (20-min budget). The chained #510-#515 bench init-container
# regressions made `deploy_distributed` look failed even when workers
# were Ready. New contract: deploy returns when workers Ready; bench
# is observed via `training.wait_until_bench_complete()` or the
# `wait_for_bench=` parameter.
# =============================================================================


def _ready_pyo3_response_with_bench(
    bench_phase: str,
    bench_result: bool = False,
    bench_message: str = "",
    instance_name: str = "dlc-bench-opt-in",
    namespace: str = "u-test",
) -> MagicMock:
    """Workers Ready (2/2). Bench published with the supplied phase.

    Built on the same PyO3-shape used by the issue #486 tests so the
    BasilicaClient.deploy_distributed -> training.refresh path works
    identically.
    """
    bench: Dict[str, Any] = {"mode": "on-start", "phase": bench_phase}
    if bench_phase != "Pending":
        bench["startedAt"] = "2026-05-08T12:00:00Z"
    if bench_phase in {"Succeeded", "Failed", "TimedOut"}:
        bench["completedAt"] = "2026-05-08T12:10:00Z"
    if bench_message:
        bench["message"] = bench_message
    if bench_result and bench_phase == "Succeeded":
        bench["result"] = {
            "measuredAt": "2026-05-08T12:10:00Z",
            "busbwGbpsP50": 42.0,
            "sizeBytesSwept": [1048576],
            "probeNodeA": "n-a",
            "probeNodeB": "n-b",
        }
    return _pyo3_response(
        instance_name,
        namespace,
        {
            "worldSize": {
                "ready": 2,
                "target": 2,
                "min": 2,
                "max": 2,
                "belowMinimum": False,
            },
            "ranks": [],
            "bench": bench,
        },
    )


class TestDeployDistributedDefaultDoesNotWaitForBench:
    """Refs #506: `deploy_distributed` default `wait_for_bench='never'`
    must return as soon as `wait_until_min_world` succeeds; it must NOT
    block on bench probe completion."""

    def test_default_returns_with_bench_pending(self) -> None:
        """LOAD-BEARING: workers Ready + bench Pending -> default deploy
        returns; ``training.bench_diagnostics['phase'] == 'Pending'``."""
        client = _build_minimal_client()
        response = _ready_pyo3_response_with_bench("Pending")
        client._client.create_distributed_deployment.return_value = response
        client._client.get_deployment.return_value = response

        training = client._deploy_distributed_impl(
            name="dlc-default-bench-pending",
            source=_noop_train,
            world_size=WorldSize(min=2, target=2, max=2),
            bench=True,
            timeout=10,
        )

        # Deploy succeeded; bench is observably still Pending. The SDK
        # did NOT block on bench (otherwise the default 20-min budget
        # would have either succeeded or thrown TimeoutError -- with
        # the mock, "Pending" never advances).
        assert training.name == response.instance_name
        diag = training.bench_diagnostics
        assert diag is not None
        assert diag["phase"] == "Pending"
        # Pending is not terminal; training.bench is None until terminal.
        assert training.bench is None
        client._client.delete_deployment.assert_not_called()

    def test_default_returns_with_bench_failed(self) -> None:
        """The whole point of the change: bench Failed must not make
        the deploy look failed when workers are Ready."""
        client = _build_minimal_client()
        response = _ready_pyo3_response_with_bench("Failed")
        client._client.create_distributed_deployment.return_value = response
        client._client.get_deployment.return_value = response

        training = client._deploy_distributed_impl(
            name="dlc-default-bench-failed",
            source=_noop_train,
            world_size=WorldSize(min=2, target=2, max=2),
            bench=True,
            timeout=10,
        )

        assert training.name == response.instance_name
        diag = training.bench_diagnostics
        assert diag is not None
        assert diag["phase"] == "Failed"
        # The UD was NOT auto-deleted just because the bench failed.
        client._client.delete_deployment.assert_not_called()

    def test_default_invalid_wait_for_bench_value_rejected(self) -> None:
        """Typo defense: only the three documented modes are accepted."""
        client = _build_minimal_client()
        with pytest.raises(ValidationError):
            client._deploy_distributed_impl(
                name="dlc",
                source=_noop_train,
                world_size=WorldSize(min=2, target=2, max=2),
                wait_for_bench="best-effort",  # hyphen, not underscore
                timeout=0,
            )


class TestDeployDistributedWaitForBenchBestEffort:
    """`wait_for_bench='best_effort'`: also wait for bench, but never
    raise on a non-Succeeded outcome."""

    def test_best_effort_returns_on_bench_failed_without_raising(self) -> None:
        client = _build_minimal_client()
        response = _ready_pyo3_response_with_bench(
            "Failed", bench_message="bench worker pod OOMKilled"
        )
        client._client.create_distributed_deployment.return_value = response
        client._client.get_deployment.return_value = response

        with pytest.warns(UserWarning, match="best_effort"):
            training = client._deploy_distributed_impl(
                name="dlc-best-effort-failed",
                source=_noop_train,
                world_size=WorldSize(min=2, target=2, max=2),
                bench=True,
                timeout=10,
                wait_for_bench="best_effort",
                bench_timeout=10,
            )

        assert training.name == response.instance_name
        client._client.delete_deployment.assert_not_called()

    def test_best_effort_returns_on_bench_succeeded(self) -> None:
        client = _build_minimal_client()
        response = _ready_pyo3_response_with_bench("Succeeded", bench_result=True)
        client._client.create_distributed_deployment.return_value = response
        client._client.get_deployment.return_value = response

        training = client._deploy_distributed_impl(
            name="dlc-best-effort-success",
            source=_noop_train,
            world_size=WorldSize(min=2, target=2, max=2),
            bench=True,
            timeout=10,
            wait_for_bench="best_effort",
            bench_timeout=10,
        )

        assert training.name == response.instance_name
        assert training.bench is not None
        assert training.bench.busbw_gbps_p50 == 42.0


class TestDeployDistributedWaitForBenchRequired:
    """`wait_for_bench='required'`: raise on a non-Succeeded outcome."""

    def test_required_raises_on_bench_failed(self) -> None:
        client = _build_minimal_client()
        response = _ready_pyo3_response_with_bench(
            "Failed", bench_message="OOMKilled"
        )
        client._client.create_distributed_deployment.return_value = response
        client._client.get_deployment.return_value = response

        with pytest.raises(DistributedError, match="phase=Failed"):
            client._deploy_distributed_impl(
                name="dlc-required-failed",
                source=_noop_train,
                world_size=WorldSize(min=2, target=2, max=2),
                bench=True,
                timeout=10,
                wait_for_bench="required",
                bench_timeout=10,
            )
        # The UD itself is still up: required mode does not auto-delete
        # because the workers are healthy. The caller decides.
        client._client.delete_deployment.assert_not_called()

    def test_required_raises_on_bench_timed_out(self) -> None:
        client = _build_minimal_client()
        response = _ready_pyo3_response_with_bench("TimedOut")
        client._client.create_distributed_deployment.return_value = response
        client._client.get_deployment.return_value = response

        with pytest.raises(DistributedError, match="phase=TimedOut"):
            client._deploy_distributed_impl(
                name="dlc-required-timedout",
                source=_noop_train,
                world_size=WorldSize(min=2, target=2, max=2),
                bench=True,
                timeout=10,
                wait_for_bench="required",
                bench_timeout=10,
            )

    def test_required_raises_on_wait_timeout(self) -> None:
        """When bench never reaches a terminal phase within
        `bench_timeout`, required mode wraps the inner TimeoutError
        in DistributedError."""
        client = _build_minimal_client()
        # Bench stays Pending -> wait_until_bench_complete will hit
        # its own TimeoutError, which the helper re-raises as
        # DistributedError.
        response = _ready_pyo3_response_with_bench("Pending")
        client._client.create_distributed_deployment.return_value = response
        client._client.get_deployment.return_value = response

        with pytest.raises(DistributedError, match="terminal phase"):
            client._deploy_distributed_impl(
                name="dlc-required-timeout",
                source=_noop_train,
                world_size=WorldSize(min=2, target=2, max=2),
                bench=True,
                timeout=10,
                wait_for_bench="required",
                bench_timeout=0,
            )

    def test_required_succeeds_on_bench_succeeded(self) -> None:
        client = _build_minimal_client()
        response = _ready_pyo3_response_with_bench("Succeeded", bench_result=True)
        client._client.create_distributed_deployment.return_value = response
        client._client.get_deployment.return_value = response

        training = client._deploy_distributed_impl(
            name="dlc-required-success",
            source=_noop_train,
            world_size=WorldSize(min=2, target=2, max=2),
            bench=True,
            timeout=10,
            wait_for_bench="required",
            bench_timeout=10,
        )

        assert training.name == response.instance_name
        assert training.bench is not None
