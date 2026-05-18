"""
Unit tests pinning the SDK-S4 simplification surface
(basilica-backend issue 663): ``source: Union[str, Path]`` deprecated in
favour of the ``Callable``-via-decorator path.

WHY this file exists (read the issue body for the full plan):

Today ``deploy_distributed(source=...)`` accepts THREE shapes:
- ``source='<inline python>'`` -- string literal (or triple-quoted)
- ``source=Path('./train.py')`` -- file path
- ``source=my_function`` -- a Callable

Three input shapes for "the code workers should run". The Callable form
is what the decorator already uses. Strings and Paths add maintenance
surface without clear product value (file IO + base64 edge cases + AST
quirks). The plan
(``docs/plans/SDK-API-SIMPLIFICATION-PLAN.md`` on basilica-backend main,
Problem 2) calls for collapsing to ``Callable``-only via the decorator.

Target after S4:
- ``source: str`` -> ``DeprecationWarning`` pointing at the decorator
  (or the ``runpy.run_path(...)`` pattern for users genuinely shipping a
  script file).
- ``source: pathlib.Path`` -> ``DeprecationWarning`` (same target).
- ``source: Callable`` -> silent (canonical; what the decorator passes).
- Decorator path stays silent: ``DistributedFunction.deploy(...)`` passes
  ``_emit_deprecation=False`` (the existing gate that already silences
  the S1 ``deploy_distributed`` deprecation when called from the
  canonical surface). The same gate also silences this source-shape
  deprecation so users of ``@basilica.distributed`` see NO warnings.

These tests:
1. PRE-FIX: fail (no DeprecationWarning is emitted today for str/Path
   source inputs).
2. POST-FIX: pass.

Stubbing pattern mirrors ``test_distributed_canonical_surface.py`` and
``test_distributed_command_factory.py``: bypass
``BasilicaClient.__init__`` and stub the PyO3 binding so no auth /
network calls fire.
"""

import os
import tempfile
import warnings
from pathlib import Path
from typing import Any, Dict
from unittest.mock import MagicMock

import pytest

import basilica
from basilica import (
    BasilicaClient,
    WorldSize,
)


# =============================================================================
# Shared stub helpers (near-clone of ``test_distributed_canonical_surface.py``
# and ``test_distributed_command_factory.py`` so the three test files exercise
# the same client wiring shape).
# =============================================================================


def _make_client_with_stub(
    name: str = "dlc-s4-source-test",
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


def _common_kwargs() -> Dict[str, Any]:
    """Shared minimum kwargs; tests fill in ``source`` per case."""
    return {
        "name": "dlc-s4-source-test",
        "image": "ghcr.io/example/trainer:latest",
        "world_size": WorldSize(min=2, target=2, max=4),
        "timeout": 0,
    }


@pytest.fixture
def tmp_script_path() -> Path:
    """A real, readable Python file the SDK's SourcePackager can ingest."""
    fd, path = tempfile.mkstemp(suffix=".py")
    try:
        with os.fdopen(fd, "w", encoding="utf-8") as f:
            f.write("print('hello from a script file')\n")
        yield Path(path)
    finally:
        try:
            os.unlink(path)
        except OSError:
            pass


# =============================================================================
# Target 1: ``source`` as a ``str`` (inline code) emits DeprecationWarning.
#
# Today: silently accepted. After S4: ``DeprecationWarning`` pointing at
# the ``@basilica.distributed`` decorator (or the ``runpy.run_path(...)``
# pattern for users who must ship an external script).
# =============================================================================


def _source_shape_warnings(caught: list) -> list:
    """
    Filter ``caught`` warnings down to just the source-shape deprecations.

    The S1 ``deploy_distributed``-itself deprecation also mentions
    ``@basilica.distributed`` (the canonical surface it points users
    at), so a plain ``match=r"@basilica\\.distributed"`` matches both
    the S1 warning AND the new S4 source-shape warning. To distinguish,
    we look for ``"source"`` in the message text (case-insensitive) --
    only S4 talks about the source parameter.
    """
    return [
        w for w in caught
        if issubclass(w.category, DeprecationWarning)
        and "source" in str(w.message).lower()
    ]


class TestSourceStringEmitsDeprecation:
    def test_source_inline_string_emits_deprecation_warning(self) -> None:
        client = _make_client_with_stub()
        with warnings.catch_warnings(record=True) as caught:
            warnings.simplefilter("always")
            training = client.deploy_distributed(
                source="print('inline hello')\n",
                **_common_kwargs(),
            )
            training.delete()
        sw = _source_shape_warnings(caught)
        assert sw, (
            f"Expected a source-shape DeprecationWarning. Got: "
            f"{[str(w.message) for w in caught]}"
        )

    def test_source_inline_string_warning_points_at_decorator(self) -> None:
        """
        The warning should point users at the canonical alternatives:
        decorator + Callable OR ``runpy.run_path`` for external scripts.
        We assert the message mentions the decorator name. The runpy hint
        is recommended-but-not-strictly-asserted to keep the test stable
        under future rewordings.
        """
        client = _make_client_with_stub()
        with warnings.catch_warnings(record=True) as caught:
            warnings.simplefilter("always")
            training = client.deploy_distributed(
                source="print('inline hello')\n",
                **_common_kwargs(),
            )
            training.delete()
        sw = _source_shape_warnings(caught)
        assert sw, (
            f"Expected a source-shape DeprecationWarning. Got: "
            f"{[str(w.message) for w in caught]}"
        )
        assert any(
            "@basilica.distributed" in str(w.message)
            for w in sw
        ), (
            f"Source deprecation warning must mention the decorator. "
            f"Got: {[str(w.message) for w in sw]}"
        )

    @pytest.mark.asyncio
    async def test_source_inline_string_emits_deprecation_warning_async(self) -> None:
        client = _make_client_with_stub()
        with warnings.catch_warnings(record=True) as caught:
            warnings.simplefilter("always")
            training = await client.deploy_distributed_async(
                source="print('inline hello')\n",
                **_common_kwargs(),
            )
            await training.delete_async()
        sw = _source_shape_warnings(caught)
        assert sw, (
            f"Expected a source-shape DeprecationWarning (async). Got: "
            f"{[str(w.message) for w in caught]}"
        )


# =============================================================================
# Target 2: ``source`` as a ``pathlib.Path`` emits DeprecationWarning.
#
# Today: silently accepted (``SourcePackager`` reads the file). After S4:
# same DeprecationWarning as the ``str`` form, since both shapes are
# being collapsed into the Callable-via-decorator surface.
# =============================================================================


class TestSourcePathEmitsDeprecation:
    def test_source_pathlib_path_emits_deprecation_warning(
        self, tmp_script_path: Path
    ) -> None:
        client = _make_client_with_stub()
        with warnings.catch_warnings(record=True) as caught:
            warnings.simplefilter("always")
            training = client.deploy_distributed(
                source=tmp_script_path,
                **_common_kwargs(),
            )
            training.delete()
        sw = _source_shape_warnings(caught)
        assert sw, (
            f"Expected a source-shape DeprecationWarning for Path input. "
            f"Got: {[str(w.message) for w in caught]}"
        )

    def test_source_str_filepath_emits_deprecation_warning(
        self, tmp_script_path: Path
    ) -> None:
        """
        A ``str`` that resolves to a file goes through the same
        ``SourcePackager`` file-reading path as ``Path``. Both are the
        anti-pattern S4 collapses; both must warn.
        """
        client = _make_client_with_stub()
        with warnings.catch_warnings(record=True) as caught:
            warnings.simplefilter("always")
            training = client.deploy_distributed(
                source=str(tmp_script_path),
                **_common_kwargs(),
            )
            training.delete()
        sw = _source_shape_warnings(caught)
        assert sw, (
            f"Expected a source-shape DeprecationWarning for str filepath. "
            f"Got: {[str(w.message) for w in caught]}"
        )

    @pytest.mark.asyncio
    async def test_source_pathlib_path_emits_deprecation_warning_async(
        self, tmp_script_path: Path
    ) -> None:
        client = _make_client_with_stub()
        with warnings.catch_warnings(record=True) as caught:
            warnings.simplefilter("always")
            training = await client.deploy_distributed_async(
                source=tmp_script_path,
                **_common_kwargs(),
            )
            await training.delete_async()
        sw = _source_shape_warnings(caught)
        assert sw, (
            f"Expected a source-shape DeprecationWarning (async, Path). "
            f"Got: {[str(w.message) for w in caught]}"
        )


# =============================================================================
# Target 3: ``source`` as a ``Callable`` does NOT emit DeprecationWarning.
#
# The Callable form is what the decorator already uses; it is the
# canonical input shape going forward. Users who construct a
# ``deploy_distributed(source=callable)`` call directly should NOT see
# the source-shape warning -- only the S1 ``deploy_distributed``-itself
# deprecation (which is a separate concern handled by SDK-S1).
# =============================================================================


def _callable_for_source_tests() -> None:
    """
    Fixture for the Callable-form source-deprecation tests.

    Defined at module scope so ``inspect.getsource`` can read the body
    (functions defined inside test methods raise ``OSError`` from
    ``inspect.getsourcelines`` when not in a frame). The body itself is
    irrelevant -- the stubbed deploy never executes it.
    """
    print("rank-up via callable")


class TestSourceCallableStaysSilent:
    def test_source_callable_does_not_emit_source_deprecation(self) -> None:
        """
        The Callable form is the canonical input. ``deploy_distributed``
        is itself S1-deprecated, so a direct call still warns about that,
        but NOT about the source shape -- the Callable is the target shape.
        """
        client = _make_client_with_stub()
        with warnings.catch_warnings(record=True) as caught:
            warnings.simplefilter("always")
            training = client.deploy_distributed(
                source=_callable_for_source_tests,
                **_common_kwargs(),
            )
            training.delete()
        sw = _source_shape_warnings(caught)
        assert not sw, (
            f"Callable source must NOT trigger a source-shape "
            f"DeprecationWarning. Got: {[str(w.message) for w in sw]}"
        )

    @pytest.mark.asyncio
    async def test_source_callable_does_not_emit_source_deprecation_async(
        self,
    ) -> None:
        client = _make_client_with_stub()
        with warnings.catch_warnings(record=True) as caught:
            warnings.simplefilter("always")
            training = await client.deploy_distributed_async(
                source=_callable_for_source_tests,
                **_common_kwargs(),
            )
            await training.delete_async()
        sw = _source_shape_warnings(caught)
        assert not sw, (
            f"Callable source (async) must NOT trigger a source-shape "
            f"DeprecationWarning. Got: {[str(w.message) for w in sw]}"
        )


# =============================================================================
# Target 4: the decorator path stays silent.
#
# Users using ``@basilica.distributed`` see NO warnings -- including the
# new source-shape one. The decorator internally extracts the function
# body to a string and calls ``deploy_distributed(source=<str>, ...,
# _emit_deprecation=False)``. The existing ``_emit_deprecation=False``
# gate already silences the S1 deprecation; the same gate must also
# silence the new source-shape deprecation.
# =============================================================================


class TestDecoratorPathDoesNotEmitSourceDeprecation:
    def test_decorator_deploys_without_emitting_source_deprecation(self) -> None:
        """
        The decorator extracts the function body into a ``str`` and
        forwards it as ``source=<str>``. Without the silencing gate, S4
        would warn here -- the user opted into the canonical surface
        and must not be warned about an internal plumbing detail.
        """
        client = _make_client_with_stub()

        @basilica.distributed(
            name="dlc-s4-source-test",
            image="ghcr.io/example/trainer:latest",
            world_size=WorldSize(min=2, target=2, max=4),
            timeout=0,
        )
        def train() -> None:
            """Per-rank entrypoint; body irrelevant for the warning test."""
            pass

        with warnings.catch_warnings(record=True) as caught:
            warnings.simplefilter("always")
            training = train.deploy(client=client)
            training.delete()

        deprecation_warnings = [
            w for w in caught if issubclass(w.category, DeprecationWarning)
        ]
        assert not deprecation_warnings, (
            f"Decorator internal deploy path emitted DeprecationWarning(s): "
            f"{[str(w.message) for w in deprecation_warnings]}. "
            f"@basilica.distributed users must not see any "
            f"DeprecationWarnings -- the decorator IS the canonical "
            f"surface; the source=<str> internal call is plumbing."
        )

    def test_command_factory_does_not_emit_source_deprecation(self) -> None:
        """
        The S3 ``basilica.distributed(command=[...])`` factory path is the
        canonical BYO-launcher surface; it has NO source= argument at all.
        Pin that the factory path does not surface a source-shape warning
        (the factory path forwards ``source=None`` to ``deploy_distributed``
        with ``_emit_deprecation=False``, so it must stay silent).
        """
        client = _make_client_with_stub()

        with warnings.catch_warnings(record=True) as caught:
            warnings.simplefilter("always")
            training = basilica.distributed(
                name="dlc-s4-source-test",
                image="ghcr.io/example/trainer:latest",
                world_size=WorldSize(min=2, target=2, max=4),
                command=["python3", "/workspace/noop.py"],
                timeout=0,
                client=client,
            )
            training.delete()

        deprecation_warnings = [
            w for w in caught
            if issubclass(w.category, DeprecationWarning)
            and "source" in str(w.message).lower()
        ]
        assert not deprecation_warnings, (
            f"S3 command-factory path emitted source-shape "
            f"DeprecationWarning(s): "
            f"{[str(w.message) for w in deprecation_warnings]}."
        )
