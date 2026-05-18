"""
Unit tests pinning the SDK-S3 simplification surface
(basilica-backend issue 662): ``basilica.distributed(command=[...], ...)``
collapses the BYO-launcher path into the same ``@basilica.distributed``
surface used for function-body deploys, dropping the ``_managed``
suffix as the canonical entry point.

WHY this file exists (read the issue body for the full plan):

Today BYO-launcher deploys go through
``client.deploy_distributed_managed(command=[...])``. The ``_managed``
suffix signals SDK plumbing (back-compat shim returning the cleanup
wrapper) rather than a user-facing semantic. After SDK-S3 the same
``basilica.distributed`` symbol handles both shapes:

- Function body: ``@basilica.distributed(...) def train(): ...`` ->
  decorator returning ``DistributedFunction``; ``train()`` deploys.
- BYO launcher: ``basilica.distributed(command=[...], ...)`` invoked
  WITHOUT a decorated function -> factory returning ``DistributedTraining``
  directly. ``with training:`` opens the context manager (already wired
  in S1).

The decorator form gates on whether a ``Callable`` is supplied. The
factory form gates on ``command`` being present. The two shapes share
configuration semantics and produce the same end-state object
(``DistributedTraining`` is the canonical handle).

Stubbing pattern mirrors ``test_distributed_canonical_surface.py``
and ``test_deploy_distributed_managed.py``: bypass ``__init__`` and
stub the PyO3 binding so no auth / network calls fire.
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
# Shared stub helpers (near-clone of
# ``test_distributed_canonical_surface.py``'s helpers so the two test
# files exercise the same client wiring shape).
# =============================================================================


def _make_client_with_stub(
    name: str = "dlc-s3-factory-test",
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


def _factory_kwargs() -> Dict[str, Any]:
    """Minimum kwargs accepted by the BYO-launcher factory shape."""
    return {
        "name": "dlc-s3-factory-test",
        "image": "ghcr.io/example/trainer:latest",
        "world_size": WorldSize(min=2, target=2, max=4),
        "command": ["python3", "/workspace/noop.py"],
        "timeout": 0,
    }


# =============================================================================
# Target 1: ``basilica.distributed(command=[...], ...)`` invoked WITHOUT a
# decorated function returns ``DistributedTraining`` directly.
#
# Today: ``basilica.distributed(...)`` always returns a decorator factory
# (``Callable[[Callable], DistributedFunction]``), so calling it without
# subsequently decorating a function yields a closure -- not a Training.
# Post-S3: when ``command=`` is set, the call short-circuits the decorator
# path and produces a ``DistributedTraining`` immediately, using a default
# client (or the supplied ``client=``).
# =============================================================================


class TestDistributedCommandFactory:
    def test_factory_call_with_command_returns_distributed_training(
        self,
    ) -> None:
        """``basilica.distributed(command=[...], ...)`` -> Training (no decorator)."""
        client = _make_client_with_stub()
        training = basilica.distributed(client=client, **_factory_kwargs())
        try:
            assert isinstance(training, DistributedTraining), (
                f"issue 662: basilica.distributed(command=[...], ...) must "
                f"return DistributedTraining directly (factory shape), got "
                f"{type(training).__name__}. Today this call returns a "
                f"decorator closure -- the _managed-suffix anti-pattern."
            )
            assert training.name == "dlc-s3-factory-test"
        finally:
            # Cleanup so the stubbed UD does not stay "alive".
            training.delete()

    def test_factory_training_is_context_manager_able(self) -> None:
        """Returned Training opens via ``with`` (relies on S1 wiring)."""
        client = _make_client_with_stub()
        training = basilica.distributed(client=client, **_factory_kwargs())
        with training as t:
            assert t is training
        # __exit__ ran delete exactly once.
        assert client._client.delete_deployment.call_count == 1
        client._client.delete_deployment.assert_called_with(
            "dlc-s3-factory-test"
        )

    def test_factory_call_does_not_emit_deprecation_warning(self) -> None:
        """
        Canonical surface MUST stay silent: the user opted into the
        post-S3 form, so the underlying ``deploy_distributed`` call must
        be invoked with ``_emit_deprecation=False`` (same contract as the
        decorator path).
        """
        client = _make_client_with_stub()
        import warnings as _warnings
        with _warnings.catch_warnings(record=True) as caught:
            _warnings.simplefilter("always")
            training = basilica.distributed(client=client, **_factory_kwargs())
            training.delete()
        deprecation_warnings = [
            w for w in caught if issubclass(w.category, DeprecationWarning)
        ]
        assert not deprecation_warnings, (
            f"basilica.distributed(command=[...]) factory path emitted "
            f"DeprecationWarning(s): "
            f"{[str(w.message) for w in deprecation_warnings]}. "
            f"The factory form IS the canonical surface and must not warn."
        )

    def test_factory_uses_default_client_when_none_supplied(self) -> None:
        """
        With no ``client=`` kwarg, the factory must build a
        ``BasilicaClient()`` lazily. Mirror the decorator-call pattern.
        """
        # We patch the BasilicaClient constructor so no auth bootstrap fires.
        import basilica as _basilica_pkg

        stub_client = _make_client_with_stub()
        orig_init = BasilicaClient.__init__

        def _stub_init(self: BasilicaClient, *args: Any, **kwargs: Any) -> None:
            # Inherit the already-stubbed inner from `stub_client` so this
            # test does not actually hit the auth bootstrap. We mutate
            # `self` to look like `stub_client` without touching globals.
            self._client = stub_client._client

        try:
            BasilicaClient.__init__ = _stub_init  # type: ignore[method-assign]
            training = _basilica_pkg.distributed(**_factory_kwargs())
            try:
                assert isinstance(training, DistributedTraining)
            finally:
                training.delete()
        finally:
            BasilicaClient.__init__ = orig_init  # type: ignore[method-assign]


# =============================================================================
# Target 2: Decorator form is unchanged -- callable-on-function continues to
# return a ``DistributedFunction`` wrapper. Pin against regression.
# =============================================================================


class TestDecoratorFormStillWorks:
    def test_decorator_on_function_returns_distributed_function(self) -> None:
        """``@basilica.distributed(...) def train(): ...`` -> wrapper."""
        from basilica.decorators import DistributedFunction

        @basilica.distributed(
            name="dlc-s3-decorator-test",
            image="ghcr.io/example/trainer:latest",
            world_size=WorldSize(min=2, target=2, max=4),
            timeout=0,
        )
        def train() -> None:
            pass

        assert isinstance(train, DistributedFunction), (
            f"issue 662 regression: @basilica.distributed must continue to "
            f"return DistributedFunction when decorating a function; got "
            f"{type(train).__name__}."
        )

    def test_decorator_call_still_deploys_via_wrapper(self) -> None:
        """Decorator path delegates to ``deploy_distributed`` as before."""
        client = _make_client_with_stub(name="dlc-s3-decorator-deploy")

        @basilica.distributed(
            name="dlc-s3-decorator-deploy",
            image="ghcr.io/example/trainer:latest",
            world_size=WorldSize(min=2, target=2, max=4),
            timeout=0,
        )
        def train() -> None:
            pass

        training = train.deploy(client=client)
        try:
            assert isinstance(training, DistributedTraining)
            assert training.name == "dlc-s3-decorator-deploy"
        finally:
            training.delete()


# =============================================================================
# Target 3: ``deploy_distributed_managed`` continues to emit
# DeprecationWarning (re-pin of S1's contract; the warning message must
# steer callers at the canonical ``basilica.distributed`` surface for
# BYO-launcher use cases too -- not just the decorator form).
# =============================================================================


class TestManagedBYORedirection:
    def test_managed_with_command_emits_deprecation_warning(self) -> None:
        """
        BYO launcher through ``deploy_distributed_managed(command=...)``
        warns just like S1's decorator-replacement contract. The warning
        text must mention ``basilica.distributed`` as the canonical
        replacement (already covered by the S1 wording; this test pins
        that BYO callers see the same redirection).
        """
        client = _make_client_with_stub(name="dlc-s3-managed-byo")
        with pytest.warns(DeprecationWarning, match=r"@basilica\.distributed"):
            with client.deploy_distributed_managed(
                name="dlc-s3-managed-byo",
                image="ghcr.io/example/trainer:latest",
                world_size=WorldSize(min=2, target=2, max=4),
                command=["torchrun", "/workspace/noop.py"],
                timeout=0,
            ):
                pass


# =============================================================================
# Target 4: argument validation -- the factory must reject calls that pass
# neither ``command`` nor a decoratable function. Today ``basilica.distributed()``
# always returns a decorator closure; without ``command=`` the caller MUST
# subsequently decorate a function. The new factory shape only activates
# when ``command`` is set.
# =============================================================================


class TestFactoryArgumentValidation:
    def test_factory_without_command_returns_decorator_closure(self) -> None:
        """
        Backwards-compat: when ``command`` is NOT set, the symbol
        behaves as today -- returns a decorator. This pins that the
        S3 change is additive (gated on ``command``) and does not
        silently break callers that decorate a function.
        """
        from basilica.decorators import DistributedFunction

        decorator_closure = basilica.distributed(
            name="dlc-s3-no-command-test",
            image="ghcr.io/example/trainer:latest",
            world_size=WorldSize(min=2, target=2, max=4),
            timeout=0,
        )
        assert callable(decorator_closure), (
            "issue 662: basilica.distributed(...) with no command= must "
            "still return a decorator closure for the function-body path."
        )

        def train() -> None:
            pass

        wrapped = decorator_closure(train)
        assert isinstance(wrapped, DistributedFunction)
