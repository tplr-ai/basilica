# Changelog

All notable changes to the Basilica Python SDK will be documented in this file.

The format is based on [Keep a Changelog](https://keepachangelog.com/en/1.0.0/),
and this project adheres to [Semantic Versioning](https://semver.org/spec/v2.0.0.html).

## [Unreleased]

### Changed
- `GpuOffering.provider` now accepts any provider/availability-zone value the
  API emits (e.g. AZ-root codenames `cyan`, `plum`, `opal`), with the region in
  the separate `region` field. It is still surfaced to Python as a string — no
  Python API change — but the underlying decode no longer goes through a fixed
  enum, so new providers/AZs work without an SDK release. Previously
  `list_secure_cloud_gpus()` failed against a newer API with
  `unknown variant ...`.

## [0.30.1] - 2026-05-22

### Fixed

- **Inject `--rdzv-conf=timeout=1500`** into the BYO launcher command for
  distributed runs. Earlier versions relied on torchelastic's 600s default
  rendezvous join timeout, which was too short for cold-start image-pull
  windows on freshly provisioned nodes. Refs basilica-backend#419 (PR #492).
- **Rewrite `--rdzv-backend=etcd-v2`→`etcd`** in the BYO launcher command.
  The `etcd-v2` backend has a known regression in torch 2.5.0a0; the
  operator's auto-path workaround did not cover BYO commands. Refs
  basilica-backend#419 (PR #492).
- **Inject `--rdzv-conf=last_call_timeout=900`** alongside the existing
  `timeout=1500`. torchelastic's default `last_call_timeout=30s` was
  insufficient for autoscaler-provisioned late ranks under capacity
  pressure. Refs basilica-backend#419 (PR #493).

## [0.30.0] - 2026-05-18

This is the major-equivalent (pre-1.0) bump that REMOVES every surface
deprecated by SDK-S1 through SDK-S4. The canonical surface is
``@basilica.distributed`` (decorator on a function) or
``basilica.distributed(command=[...])`` (BYO-launcher factory); both
return a ``DistributedTraining`` context manager. Read back bench data
via ``training.bench`` (``BenchResult | None``) and
``training.bench_diagnostics`` (``dict | None``).

### Removed

BREAKING CHANGE: the following public surfaces are removed; users still
on 0.29.x must migrate to ``@basilica.distributed`` before upgrading.

- ``BasilicaClient.deploy_distributed`` and
  ``BasilicaClient.deploy_distributed_async`` -- use
  ``@basilica.distributed`` on a function (the decorated function's
  ``__call__`` deploys and returns a ``DistributedTraining``).
- ``BasilicaClient.deploy_distributed_managed`` and
  ``BasilicaClient.deploy_distributed_managed_async`` -- subsumed by
  ``DistributedTraining``'s own ``__enter__`` / ``__aenter__``. Use
  ``with train() as training:`` or ``async with`` directly on the
  decorator-returned object.
- ``DistributedTrainingManaged`` and ``DistributedTrainingManagedAsync``
  classes (the wrappers the removed factories returned).
- The ``source: Union[str, Path]`` shapes on the (now-private)
  distributed deploy path. Only ``Callable`` is accepted, via the
  decorator. Wrap external scripts via
  ``runpy.run_path("/workspace/...")`` inside a decorated function.
- The ``bench: str`` modes ``"on-start"`` and ``"off"`` -- use
  ``bench=True`` / ``bench=False`` instead.
- ``DistributedTraining.wait_until_bench_complete`` and
  ``DistributedTraining.wait_until_bench_complete_async`` -- read
  ``training.bench`` (``BenchResult | None``) after the UD reaches a
  terminal state. The context manager's ``__exit__`` blocks until the
  UD is gone; ``training.bench`` is the final answer at that point. For
  the rare debug case where bench is ``None`` and you need to know why,
  read ``training.bench_diagnostics`` (``dict | None``).
- ``DistributedTraining.bench_status`` property and the public
  ``BenchStatus`` re-export from the ``basilica`` package. Same
  migration: ``training.bench`` for the result;
  ``training.bench_diagnostics`` for the debug dict.
- The internal ``_emit_deprecation`` kwarg on the (now-private) deploy
  impl -- the deprecation-gating plumbing it controlled no longer has
  any deprecation paths to suppress.

### Migration matrix (legacy -> canonical)

| Removed (0.29.x) | Replacement (0.30.0) |
|------------------|----------------------|
| ``client.deploy_distributed(source=fn, ...)`` | ``@basilica.distributed(...)\ndef fn(): ...\ntraining = fn()`` |
| ``client.deploy_distributed(source="<inline>", ...)`` | wrap the inline as a function body; decorate it |
| ``client.deploy_distributed(source=Path("./train.py"), ...)`` | ``runpy.run_path('/workspace/train.py')`` inside a decorated function |
| ``client.deploy_distributed_managed(command=[...], ...)`` | ``basilica.distributed(command=[...], ...)`` |
| ``client.deploy_distributed_managed(source=fn, ...)`` | ``with fn() as training:`` after ``@basilica.distributed`` on ``fn`` |
| ``bench="on-start"`` | ``bench=True`` |
| ``bench="off"`` | ``bench=False`` |
| ``training.wait_until_bench_complete(timeout=t)`` | block via ``with training:`` then read ``training.bench`` |
| ``training.bench_status.phase`` | ``training.bench_diagnostics["phase"]`` |
| ``BenchStatus`` (typed enum) | ``BenchResult`` (result payload) or the ``dict`` from ``bench_diagnostics`` |

### Internal

- The deploy logic that previously lived behind ``deploy_distributed``
  now lives on the private ``BasilicaClient._deploy_distributed_impl``
  and ``_deploy_distributed_impl_async`` methods. The decorator
  (``@basilica.distributed``) and the BYO-launcher factory
  (``basilica.distributed(command=...)``) both route through these
  private methods. There is no public API change beyond the removals
  listed above.
- ``BasilicaClient._handle_post_deploy_bench_wait[_async]`` now polls
  ``training._bench_status_raw`` directly instead of routing through
  the removed ``wait_until_bench_complete`` wrapper. The
  ``wait_for_bench`` / ``bench_timeout`` kwargs on ``@basilica.distributed``
  keep their existing semantics.
- The ``BenchStatus`` dataclass remains in ``basilica.distributed`` as
  an internal type backing ``_bench_status_raw`` (and therefore
  ``bench_diagnostics``); it is no longer re-exported from the top-level
  ``basilica`` package.

Closes basilica-backend#666. Refs the SDK API simplification plan
(``docs/plans/SDK-API-SIMPLIFICATION-PLAN.md`` on basilica-backend
main) ticket SDK-S7 ("cut major version when deprecations are
removed").

## [0.29.7] - 2026-05-18

### Deprecated
- `BasilicaClient.deploy_distributed(source=...)` (and its async sibling)
  emits a `DeprecationWarning` when `source` is a `str` or
  `pathlib.Path`. The `Callable` shape -- what the
  `@basilica.distributed` decorator already passes internally -- stays
  silent. The canonical input shape is now "decorate a function", which
  the SDK extracts via `inspect.getsource(...)`; the `str`/`Path`
  variants add maintenance surface (file IO + base64 edge cases + AST
  quirks) without product value. Users who need to ship an external
  script wrap it via `runpy.run_path("/workspace/...")` inside a
  decorated function. Both deprecated input shapes remain functional
  for two minor versions; remove at the next major alongside
  `deploy_distributed*` itself.
- The decorator path stays silent because
  `DistributedFunction.deploy(...)` passes `_emit_deprecation=False` to
  the underlying call -- the same gate that already silenced the S1
  `deploy_distributed`-itself deprecation now also silences the S4
  source-shape deprecation.

Closes basilica-backend#663. Refs the SDK API simplification plan
(`docs/plans/SDK-API-SIMPLIFICATION-PLAN.md` on basilica-backend main)
ticket SDK-S4 ("source parameter accepts Callable only; deprecate
Union[str, Path]").

## [0.29.6] - 2026-05-18

### Added
- `basilica.distributed(command=[...], ...)` now works as a factory and
  returns a `DistributedTraining` directly (no decorator wrapping). The
  same `basilica.distributed` symbol handles both shapes: decorator on
  a function (per-rank entrypoint) and factory with BYO launcher. The
  factory short-circuits when `command` is set, deploys immediately
  through `deploy_distributed(_emit_deprecation=False)`, and returns
  the canonical context-manager handle. Pass `client=` to inject an
  existing `BasilicaClient`; otherwise a default one is built lazily.

Closes basilica-backend#662. Refs the SDK API simplification plan
(`docs/plans/SDK-API-SIMPLIFICATION-PLAN.md` on basilica-backend main)
ticket SDK-S3 ("command= parameter on @basilica.distributed for BYO
launcher; drop the _managed suffix as the canonical entry point").

## [0.29.5] - 2026-05-18

### Added
- `DistributedTraining` is now itself a context manager (sync and async).
  `__enter__` / `__exit__` return the handle and best-effort `delete()`
  the UD on scope exit; `__aenter__` / `__aexit__` are the async
  counterparts. Replaces the prior `DistributedTrainingManaged` ceremony
  wrapper -- callers now write `with train() as training:` directly on
  the decorator-returned object.

### Deprecated
- `BasilicaClient.deploy_distributed` and `deploy_distributed_async`
  emit `DeprecationWarning` on direct calls. The decorator
  `@basilica.distributed` remains the canonical surface; the decorator
  itself does NOT trip the warning (it passes `_emit_deprecation=False`
  to the underlying call).
- `BasilicaClient.deploy_distributed_managed` and
  `deploy_distributed_managed_async` emit `DeprecationWarning`. The
  ceremony wrapper they returned is redundant now that
  `DistributedTraining` is itself context-manager-able. Both methods
  remain functional for two minor versions; remove at the next major
  bump.

Closes basilica-backend#660. Refs the SDK API simplification plan
(`docs/plans/SDK-API-SIMPLIFICATION-PLAN.md` on basilica-backend main)
ticket SDK-S1.

## [0.29.4] - 2026-05-18

### Fixed
- `BenchStatus` recognises `phase=Skipped` as a terminal state.
  `_BENCH_TERMINAL_PHASES` now contains all four operator-side terminal
  phases (`Succeeded`, `Failed`, `TimedOut`, `Skipped`), so
  `BenchStatus.is_terminal` returns `True` on `Skipped` and
  `wait_until_bench_complete` / `wait_until_bench_complete_async` return
  the terminal `BenchStatus` instead of polling until the user-supplied
  timeout and raising `TimeoutError`. Pre-fix, the SDK had the data
  (the operator wrote terminal `BenchStatus{phase=Skipped,
  lastAttemptOutcome="skipped"}` to the UD CR) but did not act on it
  -- the `TimeoutError` message literally contained `(phase=Skipped)`.
  Closes #480. Cross-repo reference:
  `one-covenant/basilica-backend#419` Stage 4 take-5 Cell B and the
  basilica-backend operator X2 fix (`one-covenant/basilica-backend#650
  / #653`).

### Added
- `BenchStatus.is_successful` / `is_failed` / `is_skipped` properties.
  Pin the semantic that `Skipped` is terminal but neither success nor
  failure -- the bench probe was not run (e.g. workers exited before
  the bench-controller observed them). The workload's own exit codes
  remain the source of truth for run success; bench is an opt-in,
  best-effort measurement.

## [0.29.3] - 2026-05-17

### Fixed
- `@basilica.distributed` / `@basilica.deployment` / `SourcePackager.from_function`
  now filter the captured module-level imports down to those whose
  bound names are actually referenced by the function body. Without
  this filter, the v0.29.2 fix shipped every module-level import to
  the worker pod — including `import basilica` and
  `from basilica import ...` that are only used by the decorator
  itself — which caused the worker to fail with
  `ModuleNotFoundError: No module named 'basilica'` at runtime
  (`basilica-sdk` is not installed in the trainer image). The filter
  uses AST walking of the function body to collect referenced `Name`
  / leftmost `Attribute` identifiers and emits only the matching
  imports. Refs #477 follow-up. Cross-repo reference:
  `one-covenant/basilica-backend#419` Stage 4 take-3 Cell B runtime
  trace.

## [0.29.2] - 2026-05-16

### Fixed
- `@basilica.distributed` and `@basilica.deployment` now capture the
  defining module's top-level `import` and `from ... import ...`
  statements and prepend them to the source shipped to the worker pod.
  Before this fix, only the function body was shipped; module-level
  names referenced inside the body (e.g. the `import os` in
  `examples/20_distributed_diloco.py`) raised `NameError` at worker
  runtime. Closes #477. Cross-repo reference:
  `one-covenant/basilica-backend#419` Stage 4 take-3 Cell B. The same
  capture is applied in `SourcePackager.from_function()` for the
  lower-level packaging path.

## [0.29.1] - 2026-05-09

### Fixed
- `DistributedTraining.bench` and `wait_until_bench_complete()` now
  surface the four PR #517 lifecycle fields (`phase`, `startedAt`,
  `completedAt`, `message`) end-to-end. The SDK's PyO3 binding
  deserialises the API JSON into the strongly-typed
  `DistributedBenchStatus` and re-serialises it back to a Python
  dict via `pythonize`; the four new fields were absent from that
  type and were silently dropped, so `wait_until_bench_complete()`
  could never observe the operator's terminal `phase` and would
  always fall through to `TimeoutError` even on a successful probe.
  Closes #521. The companion basilica-backend fix
  (one-covenant/basilica-backend#522) widens the wire mirror on the
  basilica-api side; both releases are required for the fix to be
  visible end-to-end.

## [0.29.0] - 2026-05-04

### Added
- Distributed training via `deploy_distributed()` and the
  `@distributed` decorator, returning a `DistributedTraining` facade
  with `scale()`, `wait_until_min_world()`, `logs()`, `events()`,
  `metrics()`, and `bench()`, plus rank/world status reporting and
  bench results. Full `_async` parity.
- `BasilicaClient.get_by_name()` looks up a `Deployment` by the
  user-supplied display name instead of the UUID `instance_name`.
- `friendly_name` property on the `Deployment` wrapper and on
  deployment response objects.

### Fixed
- `client.get()` and the `Deployment` wrapper now expose `image`,
  `phase`, `message`, `share_token`, `share_url`, and
  `public_metadata`, which the SDK had been silently dropping from
  the API response.

## [0.28.0] - 2026-04-27

### Added
- Denvr Data cloud provider support. Secure cloud GPU listings that
  include Denvr offerings now deserialize successfully instead of
  failing the entire response.

### Fixed
- `basilica.__version__` now reflects the installed package version.
  Previously the module exposed a hardcoded literal that drifted from
  `pyproject.toml` across releases (e.g. 0.27.0 wheels reported
  `__version__ == "0.17.0"`). The attribute is now resolved at import
  time via `importlib.metadata.version("basilica-sdk")` so it can no
  longer drift.

### Security
- Bumped rustls-webpki to 0.103.13 in the bundled native extension to
  address RUSTSEC-2026-0104 (reachable panic in CRL parsing).

## [0.27.0] - 2026-04-20

### Added
- Shadeform cloud provider support. Secure cloud GPU listings that
  include Shadeform offerings now deserialize successfully instead of
  failing the entire response.

## [0.26.0] - 2026-04-18

### Added
- Optional `name` parameter on `start_rental()`,
  `start_secure_cloud_rental()`, and `start_cpu_rental()`.
- `stop_rental()` and `get_rental_status()` accept a rental name or
  rental ID as the target.
- Rental response objects expose a `name` property.
- `start_rental.py`, `start_secure_cloud_gpu_rental.py`, and
  `start_cpu_rental.py` examples updated to demonstrate naming rentals
  and displaying names in status output.

### Changed
- `stop_rental()` now returns `None` to match the backend's HTTP 204 No
  Content response.

## [0.25.2] - 2026-03-11

### Fixed
- Linux wheels now correctly tagged `cp310-abi3` instead of `cp38-cp38`
- Maturin builds now explicitly use Python 3.10 interpreter inside manylinux containers

## [0.25.1] - 2026-03-11

### Fixed
- Enable abi3 stable ABI for wheel builds so Linux wheels are compatible with Python 3.10+
- Add Linux aarch64 pre-built wheels
- Previously Linux x86_64 wheels were tagged cp38-only, forcing source compilation on Python 3.10+

### Added
- `health_check` parameter on `deploy()`, `deploy_async()`, `deploy_vllm()`, `deploy_sglang()`, `create_deployment()`, `create_deployment_async()`
- `HealthCheckConfig` and `ProbeConfig` types for configuring startup, liveness, and readiness probes
- Default health checks for `deploy_vllm()` (port 8000) and `deploy_sglang()` (port 30000) with 10-minute startup tolerance
- `health_check` support in `@deployment` decorator and `DeploymentSpec`
- `deploy_sglang_health_check.py` example demonstrating custom health probes for large model deployments

## [0.16.0] - 2026-02-02
### Added
- OpenClaw summon template support and provider preset defaults.

## [0.15.0] - 2026-01-30

### Added
- Share token management: `regenerate_share_token()`, `get_share_token_status()`, `revoke_share_token()`
- Health check binding: `health_check()`
- `is_spot` field on `GpuOffering`, `SecureCloudRentalResponse`, `CpuRentalResponse`, `SecureCloudRentalListItem`, `CpuRentalListItem`

## [0.14.0] - 2026-01-26

### Added
- Topology spread support for pod distribution across nodes via `topology_spread` parameter
- `TopologySpreadConfig` and `SpreadMode` types for configuring pod spread constraints
- Secure cloud GPU rental API: `list_secure_cloud_gpus()`, `start_secure_cloud_rental()`, `stop_secure_cloud_rental()`, `list_secure_cloud_rentals()`
- `GpuOffering`, `SecureCloudRentalResponse`, `SecureCloudRentalListItem` types for GPU rentals

### Changed
- Refactored SSH utilities to shared `rental_utils` module
- Reorganized type stubs and removed obsolete classes
- Removed `container_image`, `environment`, `ports` from secure cloud rental requests
- Made `estimated_hourly_cost` optional in offerings

### Fixed
- Topology spread now available in `create_deployment_async()` and all deployment methods

## [0.13.0] - 2026-01-20

### Changed
- Remove pre-flight node availability check from deploy methods
- SDK no longer calls `list_nodes` to auto-detect GPU models before deployment
- Deployments now rely on `min_gpu_memory_gb` for GPU scheduling instead of specific models
- Let the API/scheduler handle GPU selection and autoscaling

### Removed
- `_extract_gpu_model_id()` function (no longer needed)
- GPU model auto-detection logic that blocked deployments when no nodes were immediately available

### Fixed
- Deployments no longer fail with "No GPU nodes available" when cluster is empty
- Autoscaler can now provision nodes for pending GPU workloads

## [0.12.0] - 2026-01-13

### Added
- Async API methods: `deploy_async()`, `get_async()`, `list_async()` for concurrent operations
- Async low-level methods: `create_deployment_async()`, `get_deployment_async()`, `delete_deployment_async()`
- Async utility methods: `health_check_async()`, `list_nodes_async()`, `get_balance_async()`
- GPU model auto-detection from available nodes when `gpu_models` not specified
- `_extract_gpu_model_id()` for NVML name to K8s label conversion
- Callable source support in `deploy()` via `SourcePackager.from_function()`
- HTTP endpoint readiness verification in `wait_until_ready()`
- Async DNS resolution and HTTP readiness checks
- Comprehensive async test suite (`test_async_methods.py`)
- GPU model extraction test suite (`test_gpu_model_extraction.py`)
- Async concurrent deployment example (`21_async_concurrent.py`)

### Changed
- Rename `deployment.py` to `_deployment.py` (internal module)
- `deploy()` now accepts `Callable` source in addition to file paths and inline code
- Improved error handling in `list()` with warnings instead of silent failures
- Examples updated to use `min_gpu_memory_gb` instead of hardcoded `gpu_models`

## [0.11.0] - 2025-12-31

### Fixed
- Fix DNS propagation race condition in `wait_until_ready()` where the method could return before the deployment URL was DNS-resolvable
- Add `_is_dns_resolvable()` helper to verify DNS resolution before returning ready status

### Added
- Unit tests for DNS resolution verification in `wait_until_ready()`
- End-to-end integration test for DNS propagation fix

## [0.10.0] - 2025-12-19

### Added
- `deploy_vllm()` method for one-line vLLM inference server deployments
- `deploy_sglang()` method for one-line SGLang inference server deployments
- GPU requirements auto-detection based on model size via `templates/model_size.py`
- Support for all vLLM options: `tensor_parallel_size`, `dtype`, `quantization`, `gpu_memory_utilization`, etc.
- Support for all SGLang options: `context_length`, `mem_fraction_static`, etc.
- Auto-configured persistent storage for HuggingFace model caching
- Auto-generated deployment names from model identifiers

## [0.9.0] - 2025-12-09

### Added
- Progress callback support in `wait_until_ready()` with `on_progress` and `silent` parameters
- `wait_for_ready()` function with progress callback support in Rust SDK
- Deployment events, scaling, and health check types

### Changed
- Improved `wait_for_ready` state tracking and ready condition logic

## [0.8.0] - 2025-12-06

### Added
- Deployment progress and resource request bindings

## [0.7.0] - 2025-12-05

### Added
- Deployment progress tracking with `ProgressInfo` dataclass
- Progress callbacks in `wait_until_ready()` via `on_progress` parameter
- `DeploymentStatus.progress` field for tracking sync and startup progress

### Changed
- Improved deployment status reporting with detailed phase information

## [0.6.0] - 2025-11-13

### Added
- `@deployment` decorator for declarative function-based deployments
- `DeployedFunction` wrapper class for decorator API
- `DeploymentSpec` frozen dataclass for immutable deployment configuration
- High-level `deploy()` method on `BasilicaClient` for one-line deployments
- `Deployment` facade class with `url`, `logs()`, `delete()`, `status()` methods
- `DeploymentStatus` dataclass with `is_ready`, `is_failed`, `is_pending` properties
- Core facade modules: `deployment.py`, `decorators.py`, `spec.py`
- `Volume` class for persistent storage with `from_name()` factory method
- `SourcePackager` class for automatic source code packaging
- Framework auto-detection (FastAPI, Flask, Django) in `SourcePackager`
- GPU requirements support: `gpu_count`, `gpu_models`, `min_cuda_version`, `min_gpu_memory_gb`
- Storage support: `storage=True` or `storage="/path"` parameter

### Changed
- `BasilicaClient` now exposes both high-level (`deploy()`, `get()`, `list()`) and low-level APIs
- Bucket parameter is now optional in `PersistentStorageSpec`

## [0.5.0] - 2025-11-11

### Added
- Python SDK bindings for deployment operations via PyO3
- `create_deployment()`, `get_deployment()`, `delete_deployment()`, `list_deployments()` methods
- `get_deployment_logs()` for streaming container logs
- Deployment response types: `DeploymentResponse`, `DeploymentSummary`, `DeploymentListResponse`

## [0.4.0] - 2025-11-11

### Added
- GPU and storage types exposed to Python SDK
- `GpuRequirementsSpec` for GPU resource specifications
- `StorageSpec`, `PersistentStorageSpec`, `StorageBackend` types
- `ResourceRequirements` with GPU support

## [0.3.0] - 2025-11-11

### Added
- Public deployment parameter for creating public URLs
- Improved delete response serialization

### Fixed
- Environment variable serialization type handling

## [0.2.0] - 2025-11-11

### Added
- Comprehensive exception hierarchy with 12 exception types
- `BasilicaError`, `AuthenticationError`, `AuthorizationError`, `ValidationError`
- `DeploymentError`, `DeploymentNotFound`, `DeploymentTimeout`, `DeploymentFailed`
- `ResourceError`, `StorageError`, `NetworkError`, `RateLimitError`, `SourceError`

### Changed
- Migrated from JWT tokens to API key authentication
- Standardized API token environment variable naming to `BASILICA_API_TOKEN`

### Fixed
- PyO3 signature default value syntax for v0.26 compatibility
- pyo3-stub-gen build errors with extension-module feature

## [0.1.0] - 2025-10-10

### Added
- Initial release of Basilica Python SDK
- Support for GPU rental management via Basilica API
- Client authentication via API keys (environment variable or direct)
- Health check functionality for API monitoring
- Node listing and filtering with query parameters
- Rental lifecycle management:
  - Start rentals with flexible node selection
  - Get rental status with SSH access information
  - Stop active rentals
  - List all rentals with optional filtering
- SSH access utilities for easy connection to rental instances
- Auto-configuration from environment variables:
  - `BASILICA_API_URL` for API endpoint
  - `BASILICA_API_TOKEN` for authentication
- Type hints via `.pyi` stub files for IDE support
- PyO3-based Rust bindings for performance
- Cross-platform support (Linux, macOS, Windows)

### Documentation
- README with installation and usage instructions
- Inline API documentation
- Example code for common workflows

[Unreleased]: https://github.com/one-covenant/basilica/compare/basilica-sdk-python-v0.14.0...HEAD
[0.14.0]: https://github.com/one-covenant/basilica/compare/basilica-sdk-python-v0.13.0...basilica-sdk-python-v0.14.0
[0.13.0]: https://github.com/one-covenant/basilica/compare/basilica-sdk-python-v0.12.0...basilica-sdk-python-v0.13.0
[0.12.0]: https://github.com/one-covenant/basilica/compare/basilica-sdk-python-v0.11.0...basilica-sdk-python-v0.12.0
[0.11.0]: https://github.com/one-covenant/basilica/compare/basilica-sdk-python-v0.10.0...basilica-sdk-python-v0.11.0
[0.10.0]: https://github.com/one-covenant/basilica/compare/basilica-sdk-python-v0.9.0...basilica-sdk-python-v0.10.0
[0.9.0]: https://github.com/one-covenant/basilica/compare/basilica-sdk-python-v0.8.0...basilica-sdk-python-v0.9.0
[0.8.0]: https://github.com/one-covenant/basilica/compare/basilica-sdk-python-v0.7.0...basilica-sdk-python-v0.8.0
[0.7.0]: https://github.com/one-covenant/basilica/compare/basilica-sdk-python-v0.6.0...basilica-sdk-python-v0.7.0
[0.6.0]: https://github.com/one-covenant/basilica/compare/basilica-sdk-python-v0.5.0...basilica-sdk-python-v0.6.0
[0.5.0]: https://github.com/one-covenant/basilica/compare/basilica-sdk-python-v0.4.0...basilica-sdk-python-v0.5.0
[0.4.0]: https://github.com/one-covenant/basilica/compare/basilica-sdk-python-v0.3.0...basilica-sdk-python-v0.4.0
[0.3.0]: https://github.com/one-covenant/basilica/compare/basilica-sdk-python-v0.2.0...basilica-sdk-python-v0.3.0
[0.2.0]: https://github.com/one-covenant/basilica/compare/basilica-sdk-python-v0.1.0...basilica-sdk-python-v0.2.0
[0.1.0]: https://github.com/one-covenant/basilica/releases/tag/basilica-sdk-python-v0.1.0
