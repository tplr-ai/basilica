"""
Unit tests for `deploy_distributed_managed` / `deploy_distributed_managed_async`.

Refs: basilica-backend#538 (defensive sibling of basilica-backend#486).

The managed variant returns a context-manager that calls
`training.delete()` on scope exit (success or exception). Issue #538
covers the caller-side leak: when an intermediate wait such as
`wait_until_target_world(timeout=...)` after `scale()` raises, the
script aborts before reaching `delete()` and the UD leaks. These
tests pin the auto-cleanup contract.

Test stubbing strategy mirrors `tests/test_distributed.py`: stub the
PyO3 binding at `BasilicaClient._client.create_distributed_deployment`
and `BasilicaClient._client.delete_deployment`. We bypass
`BasilicaClient.__init__` (the real constructor needs auth env / CLI
tokens) by allocating an instance via `__new__`, which is the same
pattern the existing distributed tests already rely on indirectly via
`DistributedTraining(MagicMock(), name)`.
"""

from typing import Any, Dict, List
from unittest.mock import MagicMock, AsyncMock

import pytest

from basilica import (
    BasilicaClient,
    DistributedTraining,
    DistributedTrainingManaged,
    DistributedTrainingManagedAsync,
    WorldSize,
)


# =============================================================================
# Helpers.
# =============================================================================


def _make_client_with_stub(
    name: str = "dlc-managed-test",
    namespace: str = "u-test",
) -> BasilicaClient:
    """Build a BasilicaClient whose PyO3 binding is fully stubbed.

    Bypasses BasilicaClient.__init__ to avoid the auth bootstrap
    (env-var / CLI tokens), since these tests do not exercise auth.
    """
    client = BasilicaClient.__new__(BasilicaClient)
    inner = MagicMock()

    # `create_distributed_deployment` returns a `DeploymentResponse` whose
    # only attribute the SDK reads here is `instance_name`.
    create_response = MagicMock()
    create_response.instance_name = name
    inner.create_distributed_deployment = MagicMock(return_value=create_response)

    # `get_deployment` is invoked indirectly by `training.refresh()` /
    # `wait_until_min_world` via `BasilicaClient.get(name)`. The path is
    # PyO3 response -> `Deployment._from_response` (real Python object)
    # -> `_coerce_to_dict(deployment)` -> reads `deployment.distributed`
    # ONLY if it is a real dict. Provide a response whose `distributed`
    # attribute is a literal dict already showing min ranks ready so
    # `wait_until_min_world` returns on the first poll.
    get_response = MagicMock()
    get_response.namespace = namespace
    get_response.instance_name = name
    get_response.distributed = {
        "worldSize": {
            "ready": 2,
            "target": 2,
            "min": 2,
            "max": 4,
            "belowMinimum": False,
        },
    }
    # `Deployment.__init__` reads several scalar fields straight off the
    # PyO3 response object; explicitly pin them so MagicMock does not
    # produce a non-string auto-attr that downstream type checks reject.
    get_response.image = "ghcr.io/example/trainer:latest"
    get_response.phase = "ready"
    get_response.message = None
    get_response.share_token = None
    get_response.share_url = None
    get_response.public_metadata = None
    inner.get_deployment = MagicMock(return_value=get_response)

    # Track `delete_deployment` calls so tests can assert auto-cleanup.
    inner.delete_deployment = MagicMock(return_value=None)

    client._client = inner
    return client


def _deploy_kwargs() -> Dict[str, Any]:
    """Minimum kwargs for `deploy_distributed_managed[_async]`."""
    return {
        "name": "dlc-managed-test",
        "image": "ghcr.io/example/trainer:latest",
        "world_size": WorldSize(min=2, target=2, max=4),
        # `command=` (BYO launcher) keeps the request build path simple
        # and avoids the source-packager codepath.
        "command": ["python3", "/workspace/noop.py"],
        # `timeout=0` is fine here: the stubbed status already reports
        # min ranks ready, so `wait_until_min_world` returns immediately.
        "timeout": 0,
    }


# =============================================================================
# Sync managed deploy.
# =============================================================================


class TestDeployDistributedManagedSync:
    def test_managed_calls_delete_on_normal_exit(self) -> None:
        """No exception inside the `with` block: `delete_deployment` runs once."""
        client = _make_client_with_stub()

        with client.deploy_distributed_managed(**_deploy_kwargs()) as training:
            assert isinstance(training, DistributedTraining)
            assert training.name == "dlc-managed-test"

        # delete fired exactly once on scope exit.
        assert client._client.delete_deployment.call_count == 1
        client._client.delete_deployment.assert_called_with("dlc-managed-test")

    def test_managed_calls_delete_on_exception_propagation(self) -> None:
        """Exception inside `with` body: delete still runs AND the original error propagates."""
        client = _make_client_with_stub()

        class _MyError(RuntimeError):
            pass

        with pytest.raises(_MyError, match="boom-from-caller"):
            with client.deploy_distributed_managed(**_deploy_kwargs()) as training:
                # Simulate the issue #538 scenario: an intermediate wait
                # (e.g. wait_until_target_world after scale) raises and
                # the caller would otherwise leak the UD.
                raise _MyError("boom-from-caller")

        assert client._client.delete_deployment.call_count == 1
        client._client.delete_deployment.assert_called_with("dlc-managed-test")

    def test_managed_swallows_delete_failure(self) -> None:
        """delete() failing must not produce a NEW exception out of __exit__."""
        client = _make_client_with_stub()
        client._client.delete_deployment = MagicMock(
            side_effect=RuntimeError("delete failed")
        )

        # Normal exit: no caller exception, delete raises. The contract
        # says __exit__ swallows it (best-effort cleanup).
        with client.deploy_distributed_managed(**_deploy_kwargs()) as training:
            assert isinstance(training, DistributedTraining)

        assert client._client.delete_deployment.call_count == 1

    def test_managed_swallows_delete_failure_during_exception(self) -> None:
        """delete() raising must NOT mask the propagating exception."""
        client = _make_client_with_stub()
        client._client.delete_deployment = MagicMock(
            side_effect=RuntimeError("delete failed")
        )

        class _CallerError(RuntimeError):
            pass

        with pytest.raises(_CallerError, match="caller-error"):
            with client.deploy_distributed_managed(**_deploy_kwargs()):
                raise _CallerError("caller-error")

        assert client._client.delete_deployment.call_count == 1

    def test_managed_returns_distributed_training_managed_handle(self) -> None:
        """Outside `with`, the managed object exposes `.training` (handle access)."""
        client = _make_client_with_stub()

        managed = client.deploy_distributed_managed(**_deploy_kwargs())
        assert isinstance(managed, DistributedTrainingManaged)
        assert isinstance(managed.training, DistributedTraining)
        assert managed.training.name == "dlc-managed-test"

        # Manual __enter__/__exit__ also works.
        entered = managed.__enter__()
        assert entered is managed.training
        managed.__exit__(None, None, None)
        assert client._client.delete_deployment.call_count == 1


# =============================================================================
# Async managed deploy.
# =============================================================================


class TestDeployDistributedManagedAsync:
    @pytest.mark.asyncio
    async def test_managed_async_calls_delete_async(self) -> None:
        """`async with` triggers `delete()` on scope exit.

        `delete_async` defers to `delete` via run_in_executor, which
        ultimately invokes `_client.delete_deployment`. Asserting the
        delete_deployment counter covers both sync and async paths
        without coupling to the executor plumbing.
        """
        client = _make_client_with_stub()

        async with client.deploy_distributed_managed_async(**_deploy_kwargs()) as training:
            assert isinstance(training, DistributedTraining)
            assert training.name == "dlc-managed-test"

        assert client._client.delete_deployment.call_count == 1
        client._client.delete_deployment.assert_called_with("dlc-managed-test")

    @pytest.mark.asyncio
    async def test_managed_async_calls_delete_on_exception(self) -> None:
        """Exception inside async body: delete still runs, original propagates."""
        client = _make_client_with_stub()

        class _MyAsyncError(RuntimeError):
            pass

        with pytest.raises(_MyAsyncError, match="async-boom"):
            async with client.deploy_distributed_managed_async(
                **_deploy_kwargs()
            ) as training:
                _ = training  # keep handle in scope
                raise _MyAsyncError("async-boom")

        assert client._client.delete_deployment.call_count == 1

    @pytest.mark.asyncio
    async def test_managed_async_swallows_delete_failure(self) -> None:
        """Async delete failure does not escape `__aexit__`."""
        client = _make_client_with_stub()
        client._client.delete_deployment = MagicMock(
            side_effect=RuntimeError("async delete failed")
        )

        async with client.deploy_distributed_managed_async(**_deploy_kwargs()):
            pass  # normal exit; cleanup will fail but be swallowed

        assert client._client.delete_deployment.call_count == 1


# =============================================================================
# API surface assertions: pin the new public symbols.
# =============================================================================


class TestManagedSurfaceExposed:
    def test_basilica_client_exposes_deploy_distributed_managed(self) -> None:
        assert hasattr(BasilicaClient, "deploy_distributed_managed")
        assert hasattr(BasilicaClient, "deploy_distributed_managed_async")

    def test_distributed_training_managed_is_importable(self) -> None:
        # Re-exported from `basilica` so callers can type-annotate the
        # managed return value.
        from basilica import DistributedTrainingManaged as Managed
        assert Managed is DistributedTrainingManaged

    def test_managed_class_implements_sync_protocol(self) -> None:
        for method in ("__enter__", "__exit__"):
            assert hasattr(DistributedTrainingManaged, method), (
                f"DistributedTrainingManaged missing {method}"
            )

    def test_async_managed_class_implements_async_protocol(self) -> None:
        for method in ("__aenter__", "__aexit__"):
            assert hasattr(DistributedTrainingManagedAsync, method), (
                f"DistributedTrainingManagedAsync missing {method}"
            )
