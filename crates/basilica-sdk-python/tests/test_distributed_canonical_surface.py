"""
Unit tests pinning the canonical distributed-training SDK surface
(basilica-backend issue 660 / SDK-S1).

WHY this file exists (read the issue body for the full plan):

Today the SDK exposes THREE deploy paths for a distributed UD:
- `@basilica.distributed(...)` -- decorator-call returns DistributedTraining
   (fire-and-forget) but the returned object is NOT itself a
   context-manager, so the user cannot write
   `with train() as training:` to orchestrate mid-run.
- `client.deploy_distributed_managed(...)` -- ceremony wrapper returning a
   `DistributedTrainingManaged` shim whose `__enter__` yields the
   underlying Training. Different name, different shape, same outcome.
- `client.deploy_distributed(...)` -- explicit-cleanup factory (still
   used internally; callers must `.delete()` themselves).

The user has to learn the trade-off before they pick. The
SDK-API-SIMPLIFICATION-PLAN (`docs/plans/SDK-API-SIMPLIFICATION-PLAN.md`
on basilica-backend main) calls this an "engineer around the same
problem 1000 different ways" anti-pattern.

Target after S1:
- `DistributedTraining` IS itself context-manager-able (defines
  `__enter__` / `__exit__` and the async variants).
- The decorator-call returns `DistributedTraining` directly -- bare call
  is fire-and-forget (callers can `.delete()` explicitly OR let TTL
  expire), `with train() as training:` opens the context.
- `deploy_distributed_managed` + `deploy_distributed_managed_async`
  remain available for two minor versions but emit
  `DeprecationWarning` on use, pointing at `@basilica.distributed`.
- `deploy_distributed` + `deploy_distributed_async` remain available
  for two minor versions but emit `DeprecationWarning` on use,
  pointing at `@basilica.distributed`.

These tests:
1. PRE-FIX: fail (they assert the post-fix shape, which today's SDK
   does not provide).
2. POST-FIX: pass.

Stubbing pattern mirrors `test_deploy_distributed_managed.py`: bypass
`BasilicaClient.__init__` and stub the PyO3 binding so no auth /
network calls fire.
"""

from typing import Any, Dict
from unittest.mock import MagicMock

import pytest

import basilica
from basilica import (
    BasilicaClient,
    DistributedTraining,
    WorldSize,
)


# =============================================================================
# Shared stub helpers (intentionally a near-clone of
# test_deploy_distributed_managed.py's helpers so the two test files exercise
# the same client wiring shape).
# =============================================================================


def _make_client_with_stub(
    name: str = "dlc-s1-canonical-test",
    namespace: str = "u-test",
) -> BasilicaClient:
    """BasilicaClient with PyO3 binding fully stubbed; bypasses __init__."""
    client = BasilicaClient.__new__(BasilicaClient)
    inner = MagicMock()

    create_response = MagicMock()
    create_response.instance_name = name
    inner.create_distributed_deployment = MagicMock(return_value=create_response)

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
    get_response.image = "ghcr.io/example/trainer:latest"
    get_response.phase = "ready"
    get_response.message = None
    get_response.share_token = None
    get_response.share_url = None
    get_response.public_metadata = None
    inner.get_deployment = MagicMock(return_value=get_response)

    inner.delete_deployment = MagicMock(return_value=None)

    client._client = inner
    return client


def _deploy_kwargs() -> Dict[str, Any]:
    """Minimum kwargs accepted by deploy_distributed[_managed]."""
    return {
        "name": "dlc-s1-canonical-test",
        "image": "ghcr.io/example/trainer:latest",
        "world_size": WorldSize(min=2, target=2, max=4),
        "command": ["python3", "/workspace/noop.py"],
        "timeout": 0,
    }


# =============================================================================
# Target 1: DistributedTraining itself is context-manager-able.
#
# Today: DistributedTraining has neither __enter__ nor __exit__. Users who
# want auto-cleanup-on-scope-exit go through deploy_distributed_managed.
# Post-S1: the Training handle is the ONLY thing the user touches, with or
# without `with`.
# =============================================================================


class TestDistributedTrainingIsContextManagerable:
    def test_distributed_training_defines_enter(self) -> None:
        assert hasattr(DistributedTraining, "__enter__"), (
            "issue 660: DistributedTraining must define __enter__ so callers "
            "can write `with training:` directly (no wrapper class). "
            "Today they go through deploy_distributed_managed; that "
            "ceremony is the anti-pattern S1 collapses."
        )

    def test_distributed_training_defines_exit(self) -> None:
        assert hasattr(DistributedTraining, "__exit__"), (
            "issue 660: DistributedTraining must define __exit__ so the "
            "`with` block triggers auto-cleanup."
        )

    def test_distributed_training_defines_aenter(self) -> None:
        assert hasattr(DistributedTraining, "__aenter__"), (
            "issue 660: DistributedTraining must define __aenter__ so "
            "callers can write `async with training:` directly."
        )

    def test_distributed_training_defines_aexit(self) -> None:
        assert hasattr(DistributedTraining, "__aexit__"), (
            "issue 660: DistributedTraining must define __aexit__."
        )

    def test_with_training_runs_delete_on_normal_exit(self) -> None:
        """`with training: ...` cleans up the UD on scope exit."""
        client = _make_client_with_stub()
        training = DistributedTraining(client, "dlc-s1-canonical-test")
        training.refresh()
        with training as t:
            assert t is training
        assert client._client.delete_deployment.call_count == 1
        client._client.delete_deployment.assert_called_with(
            "dlc-s1-canonical-test"
        )

    def test_with_training_runs_delete_on_exception(self) -> None:
        """Exception inside `with` propagates; delete still fires once."""
        client = _make_client_with_stub()
        training = DistributedTraining(client, "dlc-s1-canonical-test")
        training.refresh()

        class _Boom(RuntimeError):
            pass

        with pytest.raises(_Boom, match="caller-error"):
            with training:
                raise _Boom("caller-error")

        assert client._client.delete_deployment.call_count == 1

    def test_with_training_swallows_delete_failure(self) -> None:
        """If delete itself raises, the failure is swallowed (best-effort)."""
        client = _make_client_with_stub()
        client._client.delete_deployment = MagicMock(
            side_effect=RuntimeError("delete failed")
        )
        training = DistributedTraining(client, "dlc-s1-canonical-test")
        training.refresh()
        # Normal exit + delete raises -> __exit__ must swallow it.
        with training:
            pass
        assert client._client.delete_deployment.call_count == 1

    def test_with_training_does_not_mask_caller_exception_on_delete_failure(
        self,
    ) -> None:
        """delete() raising must NOT replace the propagating exception."""
        client = _make_client_with_stub()
        client._client.delete_deployment = MagicMock(
            side_effect=RuntimeError("delete failed")
        )
        training = DistributedTraining(client, "dlc-s1-canonical-test")
        training.refresh()

        class _CallerErr(RuntimeError):
            pass

        with pytest.raises(_CallerErr, match="primary"):
            with training:
                raise _CallerErr("primary")

        assert client._client.delete_deployment.call_count == 1


# =============================================================================
# Target 2: @basilica.distributed decorator-call returns a DistributedTraining
# directly (the wrapper class -- DistributedFunction -- still exists, but the
# returned object FROM __call__ is the Training, not a separate wrapper).
#
# Today: DistributedFunction.__call__ already returns DistributedTraining,
# so this part is mostly a pin against regression. The behaviour change is
# that the Training is context-manager-able (Target 1) -- the decorator
# itself does not gain new return type.
# =============================================================================


class TestDecoratorReturnsContextManagerableTraining:
    def test_decorator_call_returns_distributed_training(self) -> None:
        """Decorator-call returns Training (not a wrapper) -- pin against regression."""
        # We do NOT actually invoke the decorator-call here (that would
        # require a stubbed BasilicaClient + the whole deploy plumbing).
        # We pin the return-type annotation instead. The behavioural test
        # lives in the post-fix examples and the runtime verification.
        import inspect

        from basilica.decorators import DistributedFunction

        sig = inspect.signature(DistributedFunction.__call__)
        # The return annotation IS DistributedTraining today; this test
        # just locks it.
        annotation = sig.return_annotation
        # When the annotation is a string (forward reference), accept that
        # too; we only care that it isn't a managed wrapper.
        if isinstance(annotation, str):
            assert "DistributedTraining" in annotation, (
                f"DistributedFunction.__call__ must return DistributedTraining, "
                f"got annotation={annotation!r}"
            )
        else:
            assert annotation is DistributedTraining, (
                f"DistributedFunction.__call__ must return DistributedTraining, "
                f"got {annotation!r}"
            )


# =============================================================================
# Post-S7 (0.30.0): the deprecated factories from S1 are REMOVED, not
# just warn-and-still-callable. Per basilica-backend issue 666 / SDK-S7,
# the public surfaces below are gone; users must use the
# @basilica.distributed decorator (canonical) or the
# basilica.distributed(command=[...]) factory.
# =============================================================================


class TestRemovedManagedSurfaceIsAttributeError:
    def test_deploy_distributed_managed_is_removed(self) -> None:
        client = _make_client_with_stub()
        assert not hasattr(client, "deploy_distributed_managed"), (
            "deploy_distributed_managed must be removed in 0.30.0 "
            "(SDK-S7); use @basilica.distributed instead."
        )

    def test_deploy_distributed_managed_async_is_removed(self) -> None:
        client = _make_client_with_stub()
        assert not hasattr(client, "deploy_distributed_managed_async"), (
            "deploy_distributed_managed_async must be removed in 0.30.0 "
            "(SDK-S7); use @basilica.distributed instead."
        )


class TestRemovedExplicitDeploySurfaceIsAttributeError:
    def test_deploy_distributed_is_removed(self) -> None:
        client = _make_client_with_stub()
        assert not hasattr(client, "deploy_distributed"), (
            "deploy_distributed must be removed in 0.30.0 (SDK-S7); use "
            "@basilica.distributed instead."
        )

    def test_deploy_distributed_async_is_removed(self) -> None:
        client = _make_client_with_stub()
        assert not hasattr(client, "deploy_distributed_async"), (
            "deploy_distributed_async must be removed in 0.30.0 "
            "(SDK-S7); use @basilica.distributed instead."
        )


# =============================================================================
# Target 5: internal callers (the decorator path) MUST NOT trip the
# DeprecationWarning. The decorator deploys through `deploy_distributed`
# internally, but users see the decorator -- not the underlying method.
#
# We pin an internal escape hatch (`_skip_deprecation_warning=True` or
# similar) so the decorator path stays silent while user-facing calls warn.
# =============================================================================


class TestDecoratorInternalPathDoesNotWarn:
    def test_decorator_deploys_without_emitting_deprecation_warning(self) -> None:
        """
        The decorator class (`DistributedFunction`) deploys through the
        underlying client method, but the user did not call that method
        directly -- they used the canonical decorator surface. So no
        DeprecationWarning should leak to them.
        """
        client = _make_client_with_stub()

        @basilica.distributed(
            name="dlc-s1-canonical-test",
            image="ghcr.io/example/trainer:latest",
            world_size=WorldSize(min=2, target=2, max=4),
            timeout=0,
        )
        def train() -> None:
            """Per-rank entrypoint -- body is irrelevant for the warning test."""
            pass

        # The decorator's deploy(client=...) is the internal path; the user
        # invoked it via the decorator -- no warning should fire.
        import warnings as _warnings

        with _warnings.catch_warnings(record=True) as caught:
            _warnings.simplefilter("always")
            training = train.deploy(client=client)
            training.delete()

        deprecation_warnings = [
            w for w in caught if issubclass(w.category, DeprecationWarning)
        ]
        assert not deprecation_warnings, (
            f"Decorator internal deploy path emitted DeprecationWarning(s): "
            f"{[str(w.message) for w in deprecation_warnings]}. "
            f"Users of @basilica.distributed must not see deprecation warnings "
            f"from the underlying deploy_distributed call -- only direct "
            f"callers of deploy_distributed[_managed] should."
        )
