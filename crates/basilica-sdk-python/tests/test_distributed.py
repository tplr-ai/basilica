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
    DistributedFunction,
    DistributedTraining,
    ProviderFilter,
    QuotaExceeded,
    RankStatus,
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
            "(bench='on-start' + training.bench) are the supported path."
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

    def test_basilica_client_has_deploy_distributed_async(self) -> None:
        assert hasattr(BasilicaClient, "deploy_distributed")
        assert hasattr(BasilicaClient, "deploy_distributed_async")


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

    def test_source_string_ships_via_b64(self) -> None:
        # Source-shipping path: writes /workspace/__basilica_source.py via
        # base64-decoded bash one-liner. Operator wraps in `sh -c`.
        client = self._client()
        req = client._build_distributed_request(
            name="dlc-source-test",
            source="print('hello')\n",
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
        assert "base64 -d > /workspace/__basilica_source.py" in d_cmd
        assert "exec torchrun" in d_cmd
        assert "$BASILICA_RDZV_ENDPOINT" in d_cmd
        assert d_cmd != "auto"

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
