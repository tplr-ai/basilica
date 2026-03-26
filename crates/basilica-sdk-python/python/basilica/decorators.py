"""
Decorator-based deployment API.

Provides @deployment decorator for declarative function deployments.
"""
import functools
import inspect
import textwrap
from typing import Any, Callable, Dict, List, Optional, TYPE_CHECKING, Union

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
