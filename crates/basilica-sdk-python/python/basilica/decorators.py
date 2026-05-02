"""
Decorator-based deployment API.

Provides @deployment decorator for declarative function deployments and
@distributed decorator for distributed-training (NCCL collective) jobs.
"""
import functools
import inspect
import textwrap
from typing import Any, Callable, Dict, List, Optional, TYPE_CHECKING, Union

from .distributed import DistributedTraining, ProviderFilter, WorldSize
from .spec import DeploymentSpec
from .volume import Volume

if TYPE_CHECKING:
    from basilica._basilica import HealthCheckConfig


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
        """Extract function body as executable source code."""
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

        # Generate entry point that calls the function
        func_name = self._func.__name__
        return f'''{func_source}

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
                and `_async` counterparts.
        """
        from . import BasilicaClient

        client = client or BasilicaClient()
        source = self._extract_source()

        self._training = client.deploy_distributed(
            source=source,
            **self._kwargs,
        )
        return self._training

    def __call__(self, *args, **kwargs) -> DistributedTraining:
        """Calling the wrapped function deploys it. SDK arch § 5 example."""
        return self.deploy()

    def _extract_source(self) -> str:
        """
        Extract function body as executable source code, packaged as the
        per-rank entrypoint. Mirrors `DeployedFunction._extract_source`
        but emits a torchrun-friendly shape: the body runs once per
        rank, each rank's torchrun invocation provides
        `BASILICA_RANK`/`BASILICA_WORLD_*` env vars.
        """
        full_source = inspect.getsource(self._func)
        lines = full_source.split("\n")

        def_idx = 0
        for i, line in enumerate(lines):
            if line.lstrip().startswith("def "):
                def_idx = i
                break

        func_lines = lines[def_idx:]
        func_source = textwrap.dedent("\n".join(func_lines))
        func_name = self._func.__name__
        return f"""{func_source}

if __name__ == "__main__":
    {func_name}()
"""


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
    bench: str = "off",
    rendezvous_backend: str = "etcd-v2",
    env: Optional[Dict[str, str]] = None,
    pip_packages: Optional[List[str]] = None,
    ttl_seconds: Optional[int] = None,
    timeout: int = 600,
    enable_billing: bool = True,
) -> Callable[[Callable], DistributedFunction]:
    """
    Decorator marking a function as the per-rank entrypoint for a
    distributed-training UserDeployment. SDK arch § 5.

    The decorated function body is what each rank executes. The standard
    PyTorch idiom applies: `dist.init_process_group(backend="nccl")` then
    your training loop. torchrun + the operator handle fan-out; you do
    NOT branch on `rank == 0`.

    Args:
        name: Deployment name (DNS-safe).
        image: Container image (default: pytorch + cuda runtime).
        port: Worker container port.
        cpu, memory, gpu_count, gpu_models, min_gpu_memory_gb: Resources
            per rank pod.
        world_size: WorldSize(min, target, max). REQUIRED.
        provider_filter: ProviderFilter or `{"include": [...], "exclude": [...]}` dict.
        topology_spread: One of `pack | provider-aware | region-aware | none`.
        nccl_env: NCCL env vars merged on top of operator defaults.
        bench: `on-start` to schedule a 2-rank NCCL bench probe; `off` (default).
        rendezvous_backend: `etcd-v2` (default) | `c10d` | `static`.
        env: Environment variables passed to the worker pods.
        pip_packages: Additional pip packages to install.
        ttl_seconds: Auto-delete after N seconds.
        timeout: Seconds to wait for `min` ranks to be ready. Default 600.
        enable_billing: Whether to bill for this deployment.

    Returns:
        DistributedFunction wrapper. Calling it deploys; `.local()` runs
        in-process for single-rank testing; `.deploy(client=...)` for
        explicit-client usage.

    Example:
        >>> import basilica
        >>> from basilica import WorldSize
        >>>
        >>> @basilica.distributed(
        ...     name="dlc-llama-7b",
        ...     world_size=WorldSize(min=4, target=8, max=16),
        ...     gpu_count=1,
        ...     gpu_models=["H100"],
        ... )
        ... def train():
        ...     import os
        ...     import torch.distributed as dist
        ...     dist.init_process_group(backend="nccl")
        ...     rank = dist.get_rank()
        ...     # ... DiLoCo loop ...
        >>>
        >>> training = train()  # deploys, returns DistributedTraining
        >>> training.scale(target=12)
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
    }

    def decorator(func: Callable) -> DistributedFunction:
        return DistributedFunction(func, kwargs)

    return decorator
