"""
Decorator-based deployment API.

Provides @deployment decorator for declarative function deployments and
@distributed decorator for distributed-training (NCCL collective) jobs.
"""
import ast
import functools
import inspect
import sys
import textwrap
from typing import Any, Callable, Dict, List, Optional, TYPE_CHECKING, Union

from .distributed import DistributedTraining, ProviderFilter, WorldSize
from .spec import DeploymentSpec
from .volume import Volume

if TYPE_CHECKING:
    from basilica._basilica import HealthCheckConfig


def _collect_referenced_names(func_source: str) -> set:
    """
    Walk the AST of `func_source` and collect every free `Name` and the
    leftmost identifier of every `Attribute` chain (e.g. `os.environ.get`
    contributes `os`). Local assignments and inside-body imports are NOT
    excluded — that's fine because the import-prepending logic just
    short-circuits on names the body genuinely never mentions.
    """
    names: set = set()
    try:
        tree = ast.parse(func_source)
    except SyntaxError:
        return names
    for node in ast.walk(tree):
        if isinstance(node, ast.Name):
            names.add(node.id)
        elif isinstance(node, ast.Attribute):
            # Find leftmost Name in the Attribute chain (e.g. `os` in `os.environ.get`).
            cursor: ast.AST = node
            while isinstance(cursor, ast.Attribute):
                cursor = cursor.value
            if isinstance(cursor, ast.Name):
                names.add(cursor.id)
    return names


def _extract_module_level_imports(func: Callable, func_source: str) -> str:
    """
    Extract the subset of module-level `import` / `from ... import ...`
    statements from the module that defines `func` whose bound names
    are referenced inside `func_source`.

    The decorator-based deploy path ships only the function source via
    `inspect.getsource(func)`; module-level imports the body references
    would otherwise be missing on the worker. Filtering by actual usage
    keeps unrelated module-level imports (e.g. `import basilica` used
    only by the decorator) OUT of the worker source, so the worker pod
    does not have to install packages it never references.

    Returns an empty string if the module source is unavailable
    (interactive sessions, exec/eval contexts, frozen modules) or if no
    module-level import binding is referenced by the function body.

    Refs: one-covenant/basilica#477,
    one-covenant/basilica-backend#419 (Stage 4 take-3 Cell B).
    """
    module = sys.modules.get(getattr(func, "__module__", "") or "")
    if module is None:
        return ""
    try:
        module_source = inspect.getsource(module)
    except (OSError, TypeError):
        return ""
    try:
        tree = ast.parse(module_source)
    except SyntaxError:
        return ""

    referenced = _collect_referenced_names(func_source)
    if not referenced:
        return ""

    lines: List[str] = []
    for node in tree.body:
        if isinstance(node, ast.Import):
            kept = [
                alias for alias in node.names
                if _import_binds_referenced_name(alias, referenced)
            ]
            if kept:
                filtered = ast.Import(names=kept)
                lines.append(ast.unparse(filtered))
        elif isinstance(node, ast.ImportFrom):
            kept = [
                alias for alias in node.names
                if _from_import_binds_referenced_name(alias, referenced)
            ]
            if kept:
                filtered = ast.ImportFrom(
                    module=node.module, names=kept, level=node.level
                )
                lines.append(ast.unparse(filtered))
    if not lines:
        return ""
    return "\n".join(lines) + "\n"


def _import_binds_referenced_name(alias: ast.alias, referenced: set) -> bool:
    """
    `import foo` binds `foo`. `import foo as bar` binds `bar`.
    `import foo.bar` binds the top-level name `foo` (Python module
    resolution).
    """
    if alias.asname is not None:
        return alias.asname in referenced
    # `import foo.bar` → binds `foo`.
    return alias.name.split(".")[0] in referenced


def _from_import_binds_referenced_name(alias: ast.alias, referenced: set) -> bool:
    """
    `from x import foo` binds `foo`. `from x import foo as bar` binds `bar`.
    `from x import *` is treated as referenced (we cannot know what names
    it exposes without executing the import).
    """
    if alias.name == "*":
        return True
    if alias.asname is not None:
        return alias.asname in referenced
    return alias.name in referenced


class DeployedFunction:
    """
    Wrapper around a decorated function with deployment capabilities.

    Provides methods to deploy the function to Basilica cloud or run locally.
    Calling the wrapper directly triggers deployment.
    """

    def __init__(self, func: Callable, spec: DeploymentSpec):
        self._func = func
        self._spec = spec
        self._deployment = None
        functools.update_wrapper(self, func)

    @property
    def spec(self) -> DeploymentSpec:
        """Return the deployment specification."""
        return self._spec

    @property
    def deployment(self):
        """Return the current deployment if deployed, else None."""
        return self._deployment

    def local(self, *args, **kwargs):
        """Execute the function locally for testing."""
        return self._func(*args, **kwargs)

    def deploy(self, client=None):
        """
        Deploy the function to Basilica cloud.

        Args:
            client: Optional BasilicaClient instance. If not provided,
                   creates a new client using environment credentials.

        Returns:
            Deployment instance with url, logs(), delete(), etc.
        """
        from . import BasilicaClient

        client = client or BasilicaClient()
        source = self._extract_source()
        storage = self._resolve_storage()
        gpu_models = self._resolve_gpu_models()

        self._deployment = client.deploy(
            name=self._spec.name,
            source=source,
            image=self._spec.image,
            port=self._spec.port,
            cpu=self._spec.cpu,
            memory=self._spec.memory,
            gpu_count=self._spec.gpu_count,
            gpu_models=gpu_models,
            min_cuda_version=self._spec.min_cuda_version,
            min_gpu_memory_gb=self._spec.min_gpu_memory_gb,
            interconnect=self._spec.interconnect,
            geo=self._spec.geo,
            spot=self._spec.spot,
            infiniband=self._spec.infiniband,
            storage=storage,
            env=self._spec.env,
            pip_packages=self._spec.pip_packages,
            replicas=self._spec.replicas,
            ttl_seconds=self._spec.ttl_seconds,
            public=self._spec.public,
            timeout=self._spec.timeout,
            health_check=self._spec.health_check,
        )
        return self._deployment

    def _extract_source(self) -> str:
        """Extract function body as executable source code.

        Captures the defining module's top-level `import` statements
        (via :func:`_extract_module_level_imports`) so the body's
        references to module-level names resolve on the worker.
        """
        full_source = inspect.getsource(self._func)
        lines = full_source.split('\n')

        # Find the 'def' line (skip decorator lines)
        def_idx = 0
        for i, line in enumerate(lines):
            stripped = line.lstrip()
            if stripped.startswith('def '):
                def_idx = i
                break

        # Extract from def line onwards
        func_lines = lines[def_idx:]
        func_source = '\n'.join(func_lines)
        func_source = textwrap.dedent(func_source)

        # Prepend the defining module's top-level imports whose bound
        # names appear in the function body, so references like `os`,
        # `time`, `Optional` resolve on the worker pod without dragging
        # in unrelated imports (e.g. `import basilica` used only by the
        # decorator). Refs one-covenant/basilica#477.
        imports = _extract_module_level_imports(self._func, func_source)

        # Generate entry point that calls the function
        func_name = self._func.__name__
        return f'''{imports}{func_source}

{func_name}()
'''

    def _resolve_storage(self) -> Union[bool, str, None]:
        """Convert volumes dict to storage parameter."""
        if not self._spec.volumes:
            return None

        # Get first (and only) volume mount
        mount_path, volume = next(iter(self._spec.volumes.items()))
        return mount_path

    def _resolve_gpu_models(self) -> Optional[List[str]]:
        """Resolve GPU models from shorthand or explicit list."""
        if self._spec.gpu_models:
            return self._spec.gpu_models
        if self._spec.gpu:
            return [self._spec.gpu]
        return None

    def __call__(self, *args, **kwargs):
        """Calling the function deploys it and returns the deployment."""
        return self.deploy()


def deployment(
    name: str,
    image: str = "python:3.11-slim",
    port: int = 8000,
    cpu: str = "500m",
    memory: str = "512Mi",
    gpu: Optional[str] = None,
    gpu_count: Optional[int] = None,
    gpu_models: Optional[List[str]] = None,
    min_cuda_version: Optional[str] = None,
    min_gpu_memory_gb: Optional[int] = None,
    interconnect: Optional[str] = None,
    geo: Optional[str] = None,
    spot: Optional[bool] = None,
    infiniband: Optional[bool] = None,
    volumes: Optional[Dict[str, Volume]] = None,
    env: Optional[Dict[str, str]] = None,
    pip_packages: Optional[List[str]] = None,
    replicas: int = 1,
    ttl_seconds: Optional[int] = None,
    public: bool = True,
    timeout: int = 300,
    health_check: Optional["HealthCheckConfig"] = None,
) -> Callable[[Callable], DeployedFunction]:
    """
    Decorator to mark a function for deployment to Basilica.

    The decorated function can be deployed by calling it directly,
    or via the .deploy() method for more control.

    Args:
        name: Deployment name (DNS-safe: lowercase, numbers, hyphens)
        image: Container image. Default: python:3.11-slim
        port: Port your application listens on. Default: 8000
        cpu: CPU allocation. Default: "500m"
        memory: Memory allocation. Default: "512Mi"
        gpu: GPU model shorthand. Example: "NVIDIA-RTX-A4000"
        gpu_count: Number of GPUs (1-8)
        gpu_models: Acceptable GPU models list. Example: ["A100", "H100"]
        min_cuda_version: Minimum CUDA version
        min_gpu_memory_gb: Minimum GPU VRAM in GB
        interconnect: GPU interconnect type. "SXM" or "PCIe"
        geo: Geographic region preference. "US", "EU", "CA", "APAC"
        spot: Spot instance preference. True=prefer spot, False=exclude spot
        infiniband: Require InfiniBand networking
        volumes: Volume mounts. Example: {"/data": Volume.from_name("cache")}
        env: Environment variables
        pip_packages: Additional pip packages to install
        replicas: Number of instances. Default: 1
        ttl_seconds: Auto-delete after N seconds
        public: Create public URL. Default: True
        timeout: Seconds to wait for deployment. Default: 300
        health_check: Custom health check configuration (HealthCheckConfig).
                     Use HealthCheckConfig(liveness=..., readiness=..., startup=...)
                     with ProbeConfig for each probe.

    Returns:
        DeployedFunction wrapper

    Example:
        >>> @basilica.deployment(name="hello", port=8000)
        ... def serve():
        ...     from http.server import HTTPServer, BaseHTTPRequestHandler
        ...     HTTPServer(('', 8000), BaseHTTPRequestHandler).serve_forever()
        >>>
        >>> deployment = serve()  # Deploys and returns Deployment
        >>> print(deployment.url)
    """

    def decorator(func: Callable) -> DeployedFunction:
        spec = DeploymentSpec(
            name=name,
            image=image,
            port=port,
            cpu=cpu,
            memory=memory,
            gpu=gpu,
            gpu_count=gpu_count,
            gpu_models=gpu_models,
            min_cuda_version=min_cuda_version,
            min_gpu_memory_gb=min_gpu_memory_gb,
            interconnect=interconnect,
            geo=geo,
            spot=spot,
            infiniband=infiniband,
            volumes=volumes,
            env=env,
            pip_packages=pip_packages,
            replicas=replicas,
            ttl_seconds=ttl_seconds,
            public=public,
            timeout=timeout,
            health_check=health_check,
        )
        return DeployedFunction(func, spec)

    return decorator


# =============================================================================
# Distributed-training decorator (SDK arch § 5).
#
# Mirrors @deployment: source-introspection, returns a wrapper, calling the
# wrapper deploys. Differences:
#   1. The decorated function body is the PER-RANK entrypoint -- the SDK
#      wraps it under torchrun via the operator's `command="auto"` rendering.
#   2. The wrapper returns DistributedTraining (not Deployment).
#   3. Resource flags are distributed-shaped: world_size, provider_filter,
#      topology_spread, bench, rendezvous_backend, nccl_env.
#
# Resolves SDK arch § 13 open question 1 in the simplest direction: rank /
# world_size / provider / region are accessible inside the decorated body
# only via env vars (BASILICA_RANK, BASILICA_WORLD_TARGET, BASILICA_PROVIDER,
# etc.) -- no decorator-injected globals. Matches PyTorch's idiomatic
# `os.environ['RANK']` pattern.
# =============================================================================


class DistributedFunction:
    """
    Wrapper around a function decorated with `@basilica.distributed`.

    Calling the wrapper deploys; `.deploy(client=...)` for explicit-client
    usage; `.local()` for in-process single-rank testing (mirrors the
    existing `@deployment.local()` pattern).

    Returns a `DistributedTraining` from `.deploy()` -- NOT a `Deployment`.
    """

    def __init__(self, func: Callable, kwargs: Dict[str, Any]):
        self._func = func
        self._kwargs = kwargs
        self._training: Optional[DistributedTraining] = None
        functools.update_wrapper(self, func)

    @property
    def training(self) -> Optional[DistributedTraining]:
        """Most recent `DistributedTraining` from `.deploy()`, or None if not yet deployed."""
        return self._training

    def local(self, *args, **kwargs):
        """
        Execute the function locally (single rank, no rendezvous, no NCCL).
        Matches the @deployment .local() escape hatch for unit testing the
        function body without a deploy round-trip.
        """
        return self._func(*args, **kwargs)

    def deploy(self, client=None) -> DistributedTraining:
        """
        Deploy the decorated function as a distributed training job.

        Args:
            client: Optional BasilicaClient instance. If None, uses default.

        Returns:
            DistributedTraining: facade with scale/wait/logs/bench/delete
                and `_async` counterparts. Is itself context-manager-able
                (basilica-backend issue 660 / SDK-S1) -- use
                ``with train() as training:`` for mid-run orchestration
                with auto-cleanup.
        """
        from . import BasilicaClient

        client = client or BasilicaClient()
        # Pass the user's Callable directly to the private impl; the
        # impl packages the body via
        # `basilica.source._package_function_for_torchrun(...)`. The
        # public `deploy_distributed*` methods were removed in 0.30.0
        # (SDK-S7); the decorator is the canonical surface and routes
        # through the private impl directly.
        self._training = client._deploy_distributed_impl(
            source=self._func,
            **self._kwargs,
        )
        return self._training

    def __call__(self, *args, **kwargs) -> DistributedTraining:
        """Calling the wrapped function deploys it. SDK arch § 5 example."""
        return self.deploy()

    def _extract_source(self) -> str:
        """
        Extract the per-rank entrypoint module text for this function.

        Thin shim over
        :func:`basilica.source._package_function_for_torchrun`. The
        implementation lives there (DRY with the deploy path inside
        ``BasilicaClient._build_distributed_request``); this method
        exists so decorator-introspection tests can pin the same module
        text the deploy path ships to the worker pod.

        Captures the defining module's top-level ``import`` statements
        (via :func:`_extract_module_level_imports`) so the body's
        references to module-level names resolve on the worker pod.
        Refs one-covenant/basilica#477,
        one-covenant/basilica-backend#419 (Stage 4 take-3 Cell B).
        """
        from .source import _package_function_for_torchrun

        return _package_function_for_torchrun(self._func)


def distributed(
    name: str,
    image: str = "pytorch/pytorch:2.4.0-cuda12.4-cudnn9-runtime",
    port: int = 18789,
    cpu: str = "8",
    memory: str = "32Gi",
    gpu_count: int = 1,
    gpu_models: Optional[List[str]] = None,
    min_gpu_memory_gb: Optional[int] = None,
    world_size: Optional[WorldSize] = None,
    provider_filter: Optional[Union[ProviderFilter, Dict[str, List[str]]]] = None,
    topology_spread: str = "provider-aware",
    nccl_env: Optional[Dict[str, str]] = None,
    bench: bool = False,
    rendezvous_backend: str = "etcd-v2",
    env: Optional[Dict[str, str]] = None,
    pip_packages: Optional[List[str]] = None,
    ttl_seconds: Optional[int] = 86400,
    timeout: int = 600,
    enable_billing: bool = True,
    wait_for_bench: str = "never",
    bench_timeout: int = 1500,
    command: Optional[List[str]] = None,
    args: Optional[List[str]] = None,
    client: Optional[Any] = None,
) -> Union[Callable[[Callable], DistributedFunction], DistributedTraining]:
    """
    Decorator-or-factory marking the per-rank entrypoint of a
    distributed-training UserDeployment. SDK arch § 5.

    Two shapes share the same call site:

    1. Decorator (function body): the decorated function is the per-rank
       entrypoint. The standard PyTorch idiom applies --
       ``dist.init_process_group(backend="nccl")`` then your training
       loop. torchrun + the operator handle fan-out; you do NOT branch
       on ``rank == 0``::

           @basilica.distributed(name="dlc-llama-7b", world_size=...,
                                 gpu_count=1, gpu_models=["H100"])
           def train():
               import torch.distributed as dist
               dist.init_process_group(backend="nccl")
               ...

           training = train()  # deploys, returns DistributedTraining

    2. Factory (BYO launcher): passing ``command=[...]`` deploys
       immediately (no function to decorate) and returns a
       :class:`DistributedTraining` directly. Mirrors the prior
       ``deploy_distributed_managed(command=...)`` call site without
       the ``_managed`` suffix::

           training = basilica.distributed(
               name="dlc-torchrun",
               command=["torchrun", "--rdzv-backend=etcd",
                        "--rdzv-endpoint=$BASILICA_RDZV_ENDPOINT", ...],
               world_size=WorldSize(min=2, target=4, max=8),
               gpu_count=1, gpu_models=["A100", "H100"],
           )
           with training:
               training.scale(target=4)

       The factory short-circuits when ``command`` is set: the call
       returns :class:`DistributedTraining`, NOT a decorator closure.
       Use ``client=`` to inject an existing :class:`BasilicaClient`
       (otherwise a fresh one is built lazily, matching the decorator's
       ``.deploy()`` semantics). See basilica-backend issue 662 / SDK-S3.

    Args:
        name: Deployment name. DNS-safe (lowercase, digits, hyphens). The
            UD's K8s resources and the user-visible handle name both
            derive from this; pick something that means something to you
            later when you scan ``basilica deploy ls``.
        image: Container image for every rank pod. Defaults to a pytorch
            + CUDA runtime; for distributed NCCL workloads the
            canonical base is
            ``ghcr.io/one-covenant/basilica/basilica-distributed-trainer:latest``
            because stock NGC pytorch images don't include
            ``python-etcd`` (required by torchelastic's etcd-v2
            rendezvous backend). See ``BRINGING-YOUR-OWN-TRAINER-IMAGE.md``
            on basilica-backend for the BYO-image contract.
        port: Worker container port. Default 18789 matches the operator's
            headless governance Service; override only if your launcher
            binds something else.
        cpu, memory: Resource requests per rank pod. Sized for the
            decorated function body's needs, not the training step's
            compute (the GPU does the compute).
        gpu_count: GPUs per rank pod (``1`` means 1 GPU per rank; set
            higher for NVLink ranks). Aggregate fleet GPU count is
            ``world_size.target * gpu_count``.
        gpu_models: Acceptable GPU models. The autoscaler matches an
            offering against this AND ``min_gpu_memory_gb``;
            empty means "any model".
        min_gpu_memory_gb: Minimum GPU VRAM in GB. Use this to fall back
            across multiple GPU SKUs (e.g. accept A100-40GB OR H100-80GB
            on a 30GB workload).
        world_size: ``WorldSize(min, target, max)``. REQUIRED. ``min`` is
            the floor below which the run pauses (torchelastic
            MIN_NODES); ``target`` is the steady-state replica count;
            ``max`` is the hard ceiling enforced by ``scale()``. The SDK
            short-circuits on ``WorldSize(...)``-construction if the
            triple violates ``1 <= min <= target <= max``.
        provider_filter: ``ProviderFilter(include=[...], exclude=[...])``
            or the dict equivalent. Values are Basilica public AZ-root
            names such as ``cyan``, ``plum``, or ``opal``. With
            ``topology_spread="pack"``, the autoscaler picks ONE AZ root
            from ``include`` and packs all workers on it; the include-list
            is a fallback set, not a multi-AZ requirement.
        topology_spread: One of ``pack | provider-aware | region-aware |
            none``. ``"pack"`` is recommended for NCCL-collective
            workloads -- it forces same-AZ-root direct WireGuard mesh,
            which is 5-10x faster than the hub-relay fallback. Default
            ``"provider-aware"``.
        nccl_env: NCCL environment variables merged on top of operator
            defaults. User values win on collision. Common pattern:
            ``{"NCCL_DEBUG": "WARN"}`` to surface NCCL diagnostics
            without flooding logs.
        bench: ``True`` opts in to the per-UD NCCL bench probe -- a
            2-rank ``all_reduce_perf`` measurement on the same AZ-root
            mesh as your workers. Read back as ``training.bench``
            (``BenchResult | None`` -- ``None`` means "no measurement",
            regardless of why). Use ``training.bench_diagnostics`` for
            the rare debug detail. Default ``False`` because the probe
            costs 2 extra GPU ranks for its duration; users who don't
            need the measurement shouldn't pay for it. The legacy str
            modes ``"on-start"`` / ``"off"`` were removed in 0.30.0; see
            ``basilica-backend`` issue 661 / SDK-S2 for migration.
        rendezvous_backend: ``etcd-v2`` (default) | ``c10d`` | ``static``.
            Default ``etcd-v2`` is the only backend with full
            elasticity (mid-run scale, rank join/drain). Pick a
            non-default only if you know you need it.
        env: Environment variables passed to the worker pods. The
            operator wires ``RANK`` / ``WORLD_SIZE`` / ``LOCAL_RANK`` /
            ``MASTER_*`` / ``BASILICA_*`` automatically -- this
            parameter is for YOUR app's vars.
        pip_packages: Additional pip packages to install. Ignored when
            ``command`` is set (BYO image is expected to carry its own
            dependencies).
        ttl_seconds: Auto-delete after N seconds even if the workload is
            still running. Platform-side ceiling; set this for runaway
            protection.
        timeout: Seconds to wait for ``world_size.min`` ranks to be
            ready before returning from the deploy call. Default 600.
        enable_billing: Whether to bill for this deployment. Default
            ``True``.
        wait_for_bench: ``"never"`` (default) returns when workers
            Ready; the bench probe runs in the background and shows up
            on ``training.bench`` once the UD reaches a terminal state.
            ``"best_effort"`` waits for bench but swallows failures.
            ``"required"`` raises on a non-Succeeded bench terminal
            phase. Most users keep the default and read
            ``training.bench`` after ``wait_until_complete``.
        bench_timeout: Seconds budget for the
            ``wait_for_bench != "never"`` paths. Default 1500 (matches
            the operator's ``BENCH_ACTIVE_DEADLINE_SECONDS``).
        command: BYO-launcher entry. Setting this short-circuits the
            decorator path -- the call returns a
            :class:`DistributedTraining` immediately (factory mode).
            Mutually exclusive with the decorator shape. Inside the
            launcher, expand ``$BASILICA_RDZV_ENDPOINT`` /
            ``$BASILICA_WORLD_TARGET`` / ``$BASILICA_GPUS_PER_POD`` --
            the operator injects these into the worker pod environment.
            See basilica-backend issue 662 / SDK-S3.
        args: Positional args to pass after ``command`` in factory mode.
            Ignored unless ``command`` is set.
        client: Optional :class:`BasilicaClient` for factory mode.
            Ignored in decorator mode (the decorator's
            ``.deploy(client=)`` takes precedence). When ``command`` is
            set and ``client`` is None, a default ``BasilicaClient()``
            is constructed lazily.

    Returns:
        - Decorator mode (``command`` is None): a decorator closure that
          wraps the decorated function in a :class:`DistributedFunction`.
          Calling the wrapped function deploys and returns a
          :class:`DistributedTraining` context-manager.
        - Factory mode (``command`` is set): a
          :class:`DistributedTraining` ready to use under a ``with``
          block.
    """
    if world_size is None:
        raise ValueError("@distributed requires world_size")

    # Normalize provider_filter dict -> ProviderFilter.
    if isinstance(provider_filter, dict):
        provider_filter = ProviderFilter(
            include=provider_filter.get("include", []),
            exclude=provider_filter.get("exclude", []),
        )

    kwargs: Dict[str, Any] = {
        "name": name,
        "image": image,
        "port": port,
        "cpu": cpu,
        "memory": memory,
        "gpu_count": gpu_count,
        "gpu_models": gpu_models,
        "min_gpu_memory_gb": min_gpu_memory_gb,
        "world_size": world_size,
        "provider_filter": provider_filter,
        "topology_spread": topology_spread,
        "nccl_env": nccl_env,
        "bench": bench,
        "rendezvous_backend": rendezvous_backend,
        "env": env,
        "pip_packages": pip_packages,
        "ttl_seconds": ttl_seconds,
        "timeout": timeout,
        "enable_billing": enable_billing,
        "wait_for_bench": wait_for_bench,
        "bench_timeout": bench_timeout,
    }

    # basilica-backend issue 662 / SDK-S3: factory mode. When the caller
    # supplies a BYO launcher (``command=[...]``), short-circuit the
    # decorator path and deploy immediately. The decorator path bakes
    # ``_emit_deprecation=False`` into the inner deploy call so the
    # canonical surface stays quiet; the factory does the same.
    if command is not None:
        return _deploy_command_factory(
            kwargs=kwargs,
            command=command,
            args=args,
            client=client,
        )

    def decorator(func: Callable) -> DistributedFunction:
        return DistributedFunction(func, kwargs)

    return decorator


def _deploy_command_factory(
    kwargs: Dict[str, Any],
    command: List[str],
    args: Optional[List[str]],
    client: Optional[Any],
) -> DistributedTraining:
    """
    basilica-backend issue 662 / SDK-S3: BYO-launcher factory path.

    Invoked by :func:`distributed` when ``command`` is set. Materializes
    a :class:`BasilicaClient` if none was supplied, forwards ``command``
    / ``args`` plus the shared kwargs to the private
    :py:meth:`BasilicaClient._deploy_distributed_impl`, and returns the
    :class:`DistributedTraining` handle.

    Kept separate from :func:`distributed` so the decorator path's hot
    loop stays a closure constructor (no extra branch on every call).
    """
    from . import BasilicaClient  # local import to avoid circular dep

    bound_client = client if client is not None else BasilicaClient()
    return bound_client._deploy_distributed_impl(
        command=command,
        args=args,
        **kwargs,
    )
