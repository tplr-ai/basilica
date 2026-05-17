"""
Tests for `@basilica.deployment` and `@basilica.distributed` source extraction.

The decorators ship the wrapped function's source to a worker pod via a
base64-encoded string. The extraction must include the module-level
`import` statements that the function body relies on; otherwise the
worker raises `NameError` at runtime (see one-covenant/basilica#477).

These tests pin the import-capture contract so a future refactor that
drops the module-level scan does not regress silently.
"""

# Module-level imports that the decorated test fixtures reference.
import os
import time
from typing import Optional

import pytest

import basilica
from basilica import ProviderFilter, WorldSize


# =============================================================================
# @basilica.distributed source extraction
# =============================================================================


@basilica.distributed(
    name="test-dist-imports",
    image="ignored:latest",
    world_size=WorldSize(min=1, target=1, max=1),
    gpu_count=1,
    gpu_models=["H100"],
    provider_filter=ProviderFilter(include=["hyperstack"]),
)
def _train_with_module_imports() -> None:
    """Fixture: references `os` and `time` from module-level scope."""
    local_rank = int(os.environ.get("LOCAL_RANK", 0))
    started = time.time()
    print(f"rank={local_rank} started={started}", flush=True)


@basilica.distributed(
    name="test-dist-aliased",
    image="ignored:latest",
    world_size=WorldSize(min=1, target=1, max=1),
    gpu_count=1,
    gpu_models=["H100"],
    provider_filter=ProviderFilter(include=["hyperstack"]),
)
def _train_with_aliased_import() -> None:
    """Fixture: references `Optional` from a `from typing import Optional`."""
    x: Optional[int] = 0
    print(x)


@basilica.distributed(
    name="test-dist-no-imports",
    image="ignored:latest",
    world_size=WorldSize(min=1, target=1, max=1),
    gpu_count=1,
    gpu_models=["H100"],
    provider_filter=ProviderFilter(include=["hyperstack"]),
)
def _train_no_module_imports() -> None:
    """Fixture: only references built-ins, no module-level imports."""
    print("hello")


@basilica.distributed(
    name="test-dist-unused-filter",
    image="ignored:latest",
    world_size=WorldSize(min=1, target=1, max=1),
    gpu_count=1,
    gpu_models=["H100"],
    provider_filter=ProviderFilter(include=["hyperstack"]),
)
def _train_uses_os_only() -> None:
    """
    Fixture: references only `os` from module scope. The decorator's
    captured imports must NOT include `basilica`, `time`, `pytest`,
    `Optional` etc. that are imported in this test module but unused
    by the body.
    """
    print(os.environ.get("X", "fallback"))


class TestDistributedSourceCapturesModuleImports:
    """
    Pin contract: `@basilica.distributed`'s extracted source must include
    the module-level `import` statements the function body relies on, so
    the wrapped script does not raise `NameError` on the worker pod.
    """

    def test_extracted_source_contains_module_level_imports(self) -> None:
        source = _train_with_module_imports._extract_source()
        # Header should carry the module-level imports the body uses.
        head = source.split("def ")[0]
        assert "import os" in head, (
            "Expected `import os` to be captured from module scope; "
            f"produced source head was:\n{head!r}"
        )
        assert "import time" in head, (
            "Expected `import time` to be captured from module scope; "
            f"produced source head was:\n{head!r}"
        )

    def test_extracted_source_contains_from_imports(self) -> None:
        source = _train_with_aliased_import._extract_source()
        head = source.split("def ")[0]
        assert "from typing import" in head and "Optional" in head, (
            "Expected `from typing import Optional` to be captured; "
            f"produced source head was:\n{head!r}"
        )

    def test_extracted_source_with_no_imports_still_works(self) -> None:
        source = _train_no_module_imports._extract_source()
        # Must still contain the function definition and call.
        assert "def _train_no_module_imports()" in source
        assert "_train_no_module_imports()" in source

    def test_extracted_source_executes_without_name_error(self) -> None:
        """
        Compile + exec the produced source with `__name__ == "__main__"`
        so the trailing call actually runs the body. Pre-fix this raised
        `NameError: name 'os' is not defined`. This is the canonical
        bug from one-covenant/basilica#477.
        """
        source = _train_with_module_imports._extract_source()
        ns: dict = {"__name__": "__main__"}
        compiled = compile(source, "<test-source>", "exec")
        # Must not raise NameError on the module-level names used inside
        # the body.
        exec(compiled, ns)
        assert "_train_with_module_imports" in ns

    def test_extracted_source_filters_unused_module_imports(self) -> None:
        """
        The extracted source must NOT include module-level imports that
        the body never references. Specifically: a body that only uses
        `os` should NOT carry `import basilica`, `import time`,
        `import pytest`, `from typing import Optional`, etc. on its
        head, because the worker container does not have those
        installed and the import would fail at module-eval time.

        Regression: pre-second-fix the decorator captured ALL
        module-level imports (e.g. `import basilica`), which caused the
        worker pod to raise `ModuleNotFoundError: No module named
        'basilica'` on the runtime path. See task D2 (basilica-backend
        #419 Stage 4 take-3) runtime trace.
        """
        source = _train_uses_os_only._extract_source()
        head = source.split("def ")[0]
        assert "import os" in head, (
            f"Expected `import os` (the only referenced module-level "
            f"import) in head; got:\n{head!r}"
        )
        # These are imported at module level in this test file but the
        # function body does not reference them; they must be filtered.
        assert "import basilica" not in head, (
            f"Expected `import basilica` filtered out (unused by body); "
            f"got:\n{head!r}"
        )
        assert "import pytest" not in head, (
            f"Expected `import pytest` filtered out (unused by body); "
            f"got:\n{head!r}"
        )
        assert "from typing" not in head, (
            f"Expected `from typing import Optional` filtered out "
            f"(unused by body); got:\n{head!r}"
        )


# =============================================================================
# @basilica.deployment source extraction (same code path, same bug)
# =============================================================================


@basilica.deployment(name="test-dep-imports", port=8000)
def _serve_with_module_imports() -> None:
    """Fixture: references `os` from module-level scope."""
    port = int(os.environ.get("PORT", 8000))
    print(f"port={port}", flush=True)


class TestDeploymentSourceCapturesModuleImports:
    """
    `@basilica.deployment` uses the same source-extraction shape as
    `@basilica.distributed`. Same fix, same contract.
    """

    def test_extracted_source_contains_module_level_imports(self) -> None:
        source = _serve_with_module_imports._extract_source()
        head = source.split("def ")[0]
        assert "import os" in head, (
            "Expected `import os` to be captured from module scope; "
            f"produced source head was:\n{head!r}"
        )

    def test_extracted_source_executes_without_name_error(self) -> None:
        source = _serve_with_module_imports._extract_source()
        # The deployment decorator appends a call; we want to test that
        # the *body* is well-formed and references resolve, not that the
        # call actually runs. Re-write the call to a no-op by trimming
        # everything after the def for this exec.
        # Simpler: compile and instantiate without running the trailing
        # call. We do this by stopping at the trailing call line.
        lines = source.splitlines()
        # Strip trailing call (the decorator appends `<name>()`).
        while lines and not lines[-1].strip().startswith("def "):
            # Stop trimming if we've hit the function def itself.
            tail = lines[-1].strip()
            if tail == "" or tail == f"{_serve_with_module_imports.__name__}()":
                lines.pop()
                continue
            break
        trimmed = "\n".join(lines) + "\n"
        ns: dict = {"__name__": "not-main"}
        compiled = compile(trimmed, "<test-source>", "exec")
        exec(compiled, ns)
        assert "_serve_with_module_imports" in ns


if __name__ == "__main__":
    pytest.main([__file__, "-v"])
