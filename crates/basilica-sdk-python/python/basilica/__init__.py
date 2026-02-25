"""
Basilica SDK for Python

Deploy and manage containerized applications on the Basilica GPU cloud.

Quick Start:
    >>> from basilica import BasilicaClient
    >>> client = BasilicaClient()
    >>>
    >>> # Deploy a Python app from a file
    >>> deployment = client.deploy("my-api", source="app.py", port=8000)
    >>> print(f"Live at: {deployment.url}")
    >>>
    >>> # Or deploy from inline code
    >>> deployment = client.deploy(
    ...     name="hello",
    ...     source="print('Hello, World!')",
    ... )

Authentication:
    Set the BASILICA_API_TOKEN environment variable:
        export BASILICA_API_TOKEN="basilica_..."

    Or pass directly:
        client = BasilicaClient(api_key="basilica_...")

    Create a token using: basilica tokens create
"""

import asyncio
import os
import re
import time
import warnings
from pathlib import Path
from typing import Any, Callable, Dict, List, Optional, Union

from basilica._basilica import (
    DEFAULT_API_URL,
    DEFAULT_CONTAINER_IMAGE,
    DEFAULT_CPU_CORES,
    DEFAULT_GPU_COUNT,
    DEFAULT_GPU_MIN_MEMORY_GB,
    DEFAULT_GPU_TYPE,
    DEFAULT_MEMORY_MB,
    DEFAULT_STORAGE_MB,
    DEFAULT_TIMEOUT_SECS,
    AvailabilityInfo,
    AvailableNode,
)
from basilica._basilica import (
    BasilicaClient as _BasilicaClient,
)  # Core client binding; Helper functions; Response types; Request types; Deployment types; Constants from Rust
from basilica._basilica import (
    CpuOffering,
    CpuRentalListItem,
    CpuRentalResponse,
    CpuSpec,
    CreateDeploymentRequest,
    DeleteDeploymentResponse,
    DeploymentListResponse,
    DeploymentResponse,
    DeploymentSummary,
    EnvVar,
    GpuOffering,
    GpuPriceQuery,
    GpuRequirements,
    GpuSpec,
    HealthCheckConfig,
    HealthCheckResponse,
    ListAvailableNodesQuery,
    ListCpuRentalsResponse,
    ListRentalsQuery,
    ListSecureCloudRentalsResponse,
    NodeDetails,
    PersistentStorageSpec,
    PodInfo,
    PortMappingRequest,
    ProbeConfig,
    RentalResponse,
    RentalStatus,
    RentalStatusWithSshResponse,
    ReplicaStatus,
    ResourceRequirements,
    ResourceRequirementsRequest,
    SecureCloudRentalListItem,
    SecureCloudRentalResponse,
    SpreadMode,
    SshAccess,
    SshKeyResponse,
    StartCpuRentalRequest,
    StartRentalApiRequest,
    StartSecureCloudRentalRequest,
    StopCpuRentalResponse,
    StopSecureCloudRentalResponse,
    StorageBackend,
    StorageSpec,
    TopologySpreadConfig,
    WebSocketConfig,
    VolumeMountRequest,
    EnrollMetadataResponse,
    PublicDeploymentMetadataResponse,
)

# GpuRequirementsSpec may not be available in older binaries
try:
    from basilica._basilica import GpuRequirementsSpec
except ImportError:
    # Fallback: define a compatible class
    from dataclasses import dataclass
    from typing import List, Optional

    @dataclass
    class GpuRequirementsSpec:
        """GPU requirements specification for deployments."""

        count: int
        model: Optional[List[str]] = None
        min_cuda_version: Optional[str] = None
        min_gpu_memory_gb: Optional[int] = None
        interconnect: Optional[str] = None
        geo: Optional[str] = None
        spot: Optional[bool] = None
        infiniband: Optional[bool] = None

        def __init__(
            self,
            count: int,
            model: Optional[List[str]] = None,
            min_cuda_version: Optional[str] = None,
            min_gpu_memory_gb: Optional[int] = None,
            interconnect: Optional[str] = None,
            geo: Optional[str] = None,
            spot: Optional[bool] = None,
            infiniband: Optional[bool] = None,
        ):
            self.count = count
            self.model = model or []
            self.min_cuda_version = min_cuda_version
            self.min_gpu_memory_gb = min_gpu_memory_gb
            self.interconnect = interconnect
            self.geo = geo
            self.spot = spot
            self.infiniband = infiniband


from .decorators import DeployedFunction, deployment

# Import new modules
from ._deployment import Deployment, DeploymentStatus, ProgressInfo
from .exceptions import (
    AuthenticationError,
    AuthorizationError,
    BasilicaError,
    DeploymentError,
    DeploymentFailed,
    DeploymentNotFound,
    DeploymentTimeout,
    NetworkError,
    RateLimitError,
    ResourceError,
    SourceError,
    StorageError,
    ValidationError,
)
from .source import SourcePackager
from .spec import DeploymentSpec
from .volume import Volume

# Default command is a list in Python
DEFAULT_COMMAND = ["/bin/bash"]

# Default Python image for source deployments
DEFAULT_PYTHON_IMAGE = "python:3.11-slim"


def _build_inference_health_check(port: int) -> HealthCheckConfig:
    """Build default health check config for inference servers (vLLM, SGLang).

    Matches the CLI template defaults:
    - Liveness: 60s initial delay, 30s period, 10s timeout, 3 failures
    - Readiness: 30s initial delay, 10s period, 5s timeout, 3 failures
    - Startup: 0s initial delay, 10s period, 5s timeout, 60 failures (10 min)
    """
    return HealthCheckConfig(
        liveness=ProbeConfig(
            path="/health",
            port=port,
            initial_delay_seconds=60,
            period_seconds=30,
            timeout_seconds=10,
            failure_threshold=3,
        ),
        readiness=ProbeConfig(
            path="/health",
            port=port,
            initial_delay_seconds=30,
            period_seconds=10,
            timeout_seconds=5,
            failure_threshold=3,
        ),
        startup=ProbeConfig(
            path="/health",
            port=port,
            initial_delay_seconds=0,
            period_seconds=10,
            timeout_seconds=5,
            failure_threshold=60,
        ),
    )


__version__ = "0.17.0"
__all__ = [
    # Main client
    "BasilicaClient",
    # Decorator API
    "deployment",
    "DeployedFunction",
    "Volume",
    "DeploymentSpec",
    # High-level types
    "Deployment",
    "DeploymentStatus",
    "ProgressInfo",
    "SourcePackager",
    # Exceptions
    "BasilicaError",
    "AuthenticationError",
    "AuthorizationError",
    "ValidationError",
    "DeploymentError",
    "DeploymentNotFound",
    "DeploymentTimeout",
    "DeploymentFailed",
    "ResourceError",
    "StorageError",
    "NetworkError",
    "RateLimitError",
    "SourceError",
    # Response types
    "HealthCheckResponse",
    "RentalResponse",
    "RentalStatusWithSshResponse",
    "RentalStatus",
    "SshAccess",
    "NodeDetails",
    "GpuSpec",
    "CpuSpec",
    "AvailableNode",
    "AvailabilityInfo",
    # Request types
    "StartRentalApiRequest",
    "GpuRequirements",
    "PortMappingRequest",
    "ListAvailableNodesQuery",
    "ListRentalsQuery",
    # Deployment types
    "EnvVar",
    "GpuRequirementsSpec",
    "ResourceRequirements",
    "ReplicaStatus",
    "PodInfo",
    "SpreadMode",
    "TopologySpreadConfig",
    "StorageBackend",
    "PersistentStorageSpec",
    "StorageSpec",
    "ProbeConfig",
    "HealthCheckConfig",
    "WebSocketConfig",
    "CreateDeploymentRequest",
    "DeploymentResponse",
    "DeploymentSummary",
    "DeploymentListResponse",
    "DeleteDeploymentResponse",
    # SSH Key types
    "SshKeyResponse",
    # CPU Rental types
    "CpuOffering",
    "StartCpuRentalRequest",
    "CpuRentalResponse",
    "StopCpuRentalResponse",
    "CpuRentalListItem",
    "ListCpuRentalsResponse",
    # GPU Rental types (secure cloud)
    "GpuPriceQuery",
    "GpuOffering",
    "SecureCloudRentalListItem",
    "SecureCloudRentalResponse",
    "StartSecureCloudRentalRequest",
    "StopSecureCloudRentalResponse",
    "ListSecureCloudRentalsResponse",
    # Public metadata types
    "EnrollMetadataResponse",
    "PublicDeploymentMetadataResponse",
]


class BasilicaClient:
    """
    Client for deploying and managing applications on Basilica.

    The BasilicaClient provides both high-level and low-level APIs for
    working with the Basilica GPU cloud platform.

    High-Level API (Recommended):
        Use deploy() for simple, one-line deployments:

        >>> client = BasilicaClient()
        >>> deployment = client.deploy("my-app", source="app.py", port=8000)
        >>> print(deployment.url)

    Low-Level API:
        Use create_deployment() for full control:

        >>> response = client.create_deployment(
        ...     instance_name="my-app",
        ...     image="python:3.11-slim",
        ...     command=["python", "app.py"],
        ...     port=8000,
        ... )

    Authentication (tried in order):
        1. Direct parameter: BasilicaClient(api_key="basilica_...")
        2. Environment variable: export BASILICA_API_TOKEN="basilica_..."
        3. CLI login tokens: run ``basilica login`` first

    Attributes:
        base_url: The API endpoint URL
    """

    def __init__(self, base_url: Optional[str] = None, api_key: Optional[str] = None):
        """
        Initialize the Basilica client.

        Args:
            base_url: API endpoint URL. Defaults to BASILICA_API_URL env var
                     or https://api.basilica.ai
            api_key: Authentication token. Defaults to BASILICA_API_TOKEN env var.
                    If neither is set, falls back to CLI login tokens.

        Raises:
            RuntimeError: If no authentication method is available

        Example:
            >>> # Auto-detect from environment
            >>> client = BasilicaClient()

            >>> # Explicit configuration
            >>> client = BasilicaClient(
            ...     base_url="https://api.basilica.ai",
            ...     api_key="basilica_..."
            ... )
        """
        if base_url is None:
            base_url = os.environ.get("BASILICA_API_URL", DEFAULT_API_URL)

        self._base_url = base_url
        self._client = _BasilicaClient(base_url, api_key)

    @property
    def base_url(self) -> str:
        """The API endpoint URL."""
        return self._base_url

    def _build_deploy_request(
        self,
        name: str,
        source: Optional[Union[str, Path, Callable]],
        image: str,
        port: int,
        env: Optional[Dict[str, str]],
        cpu: str,
        memory: str,
        storage: Union[bool, str],
        gpu_count: Optional[int],
        gpu_models: Optional[List[str]],
        min_cuda_version: Optional[str],
        min_gpu_memory_gb: Optional[int],
        replicas: int,
        ttl_seconds: Optional[int],
        public: bool,
        pip_packages: Optional[List[str]],
        topology_spread: Optional[TopologySpreadConfig] = None,
        health_check: Optional[HealthCheckConfig] = None,
        interconnect: Optional[str] = None,
        geo: Optional[str] = None,
        spot: Optional[bool] = None,
        infiniband: Optional[bool] = None,
    ) -> CreateDeploymentRequest:
        """Build CreateDeploymentRequest from deploy parameters."""
        command = None
        if source is not None:
            if callable(source):
                packager = SourcePackager.from_function(source)
            else:
                packager = SourcePackager(source)
            command = packager.build_command(pip_packages=pip_packages)

        storage_spec = None
        if storage:
            mount_path = storage if isinstance(storage, str) else "/data"
            storage_spec = StorageSpec(
                persistent=PersistentStorageSpec(
                    enabled=True,
                    backend=StorageBackend.R2,
                    bucket="",
                    credentials_secret=None,
                    sync_interval_ms=1000,
                    cache_size_mb=1024,
                    mount_path=mount_path,
                )
            )

        gpu_spec = None
        if gpu_count is not None:
            gpu_spec = GpuRequirementsSpec(
                count=gpu_count,
                model=gpu_models or [],
                min_cuda_version=min_cuda_version,
                min_gpu_memory_gb=min_gpu_memory_gb,
                interconnect=interconnect,
                geo=geo,
                spot=spot,
                infiniband=infiniband,
            )

        resources = ResourceRequirements(cpu=cpu, memory=memory, gpus=gpu_spec)

        return CreateDeploymentRequest(
            instance_name=name,
            image=image,
            replicas=replicas,
            port=port,
            command=command,
            args=None,
            env=env,
            resources=resources,
            ttl_seconds=ttl_seconds,
            public=public,
            storage=storage_spec,
            topology_spread=topology_spread,
            health_check=health_check,
        )

    def deploy(
        self,
        name: str,
        source: Optional[Union[str, Path, Callable]] = None,
        image: str = DEFAULT_PYTHON_IMAGE,
        port: int = 8000,
        env: Optional[Dict[str, str]] = None,
        cpu: str = "500m",
        memory: str = "512Mi",
        storage: Union[bool, str] = False,
        gpu_count: Optional[int] = None,
        gpu_models: Optional[List[str]] = None,
        min_cuda_version: Optional[str] = None,
        min_gpu_memory_gb: Optional[int] = None,
        replicas: int = 1,
        ttl_seconds: Optional[int] = None,
        public: bool = True,
        timeout: int = 300,
        pip_packages: Optional[List[str]] = None,
        topology_spread: Optional[TopologySpreadConfig] = None,
        health_check: Optional[HealthCheckConfig] = None,
        interconnect: Optional[str] = None,
        geo: Optional[str] = None,
        spot: Optional[bool] = None,
        infiniband: Optional[bool] = None,
    ) -> Deployment:
        """
        Deploy an application to Basilica.

        This is the recommended high-level method for deploying applications.
        It handles source code packaging, waits for the deployment to be ready,
        and returns a Deployment object with convenient methods.

        Args:
            name: Deployment name (DNS-safe: lowercase, numbers, hyphens).
                  Example: "my-api", "pytorch-trainer"
            source: Python source code to deploy. Can be:
                   - A file path: "app.py" or "/path/to/app.py"
                   - Inline code: "print('Hello!')"
                   - A callable: A Python function (source extracted via inspect)
                   - None: Just deploy the image without custom code
            image: Container image. Default: python:3.11-slim
                  For GPU: "pytorch/pytorch:2.1.0-cuda12.1-cudnn8-runtime"
            port: Port your application listens on. Default: 8000
            env: Environment variables as a dict.
                 Example: {"API_KEY": "secret", "DEBUG": "true"}
            cpu: CPU allocation. Default: "500m" (0.5 cores)
                 Examples: "1", "2", "500m", "2000m"
            memory: Memory allocation. Default: "512Mi"
                   Examples: "512Mi", "1Gi", "4Gi"
            storage: Persistent storage configuration:
                    - False: No storage (default)
                    - True: Enable storage at /data
                    - "/custom/path": Enable storage at custom path
            gpu_count: Number of GPUs (1-8). Enables GPU scheduling.
            gpu_models: Acceptable GPU models. Example: ["A100", "H100"]
            min_cuda_version: Minimum CUDA version. Example: "12.0"
            min_gpu_memory_gb: Minimum GPU VRAM in GB. Example: 40
            interconnect: GPU interconnect type. "SXM" or "PCIe"
            geo: Geographic region preference. "US", "EU", "CA", "APAC"
            spot: Spot instance preference. True=prefer spot, False=exclude spot
            infiniband: Require InfiniBand networking. True/False
            replicas: Number of instances. Default: 1
            ttl_seconds: Auto-delete after N seconds. None = never.
            public: Create public URL. Default: True
            timeout: Seconds to wait for deployment. Default: 300
            pip_packages: Additional pip packages to install.
                         Auto-detected for FastAPI apps if not specified.
            health_check: Custom health check configuration.
                         Use HealthCheckConfig(liveness=..., readiness=..., startup=...)
                         with ProbeConfig for each probe. Useful for GPU workloads
                         that need longer startup times (e.g. model downloading).

        Returns:
            Deployment: A deployment object with url, logs(), delete(), etc.

        Raises:
            ValidationError: If parameters are invalid
            DeploymentTimeout: If deployment doesn't become ready within timeout
            DeploymentFailed: If deployment fails to start
            SourceError: If source file cannot be read
            NetworkError: If API is unreachable

        Example:
            Deploy from a file:
            >>> deployment = client.deploy(
            ...     name="my-api",
            ...     source="api.py",
            ...     port=8000,
            ...     storage=True,
            ... )
            >>> print(f"Live at: {deployment.url}")

            Deploy inline code:
            >>> deployment = client.deploy(
            ...     name="hello",
            ...     source="from http.server import HTTPServer, BaseHTTPRequestHandler; HTTPServer(('', 8000), BaseHTTPRequestHandler).serve_forever()",
            ...     port=8000,
            ... )

            GPU deployment:
            >>> deployment = client.deploy(
            ...     name="pytorch-train",
            ...     source="train.py",
            ...     image="pytorch/pytorch:2.1.0-cuda12.1-cudnn8-runtime",
            ...     gpu_count=1,
            ...     gpu_models=["A100", "H100"],
            ...     memory="16Gi",
            ...     storage=True,
            ... )

            Deploy just an image (no source):
            >>> deployment = client.deploy(
            ...     name="nginx",
            ...     image="nginxinc/nginx-unprivileged:alpine",
            ...     port=8080,
            ... )
        """
        request = self._build_deploy_request(
            name=name,
            source=source,
            image=image,
            port=port,
            env=env,
            cpu=cpu,
            memory=memory,
            storage=storage,
            gpu_count=gpu_count,
            gpu_models=gpu_models,
            min_cuda_version=min_cuda_version,
            min_gpu_memory_gb=min_gpu_memory_gb,
            replicas=replicas,
            ttl_seconds=ttl_seconds,
            public=public,
            pip_packages=pip_packages,
            topology_spread=topology_spread,
            health_check=health_check,
            interconnect=interconnect,
            geo=geo,
            spot=spot,
            infiniband=infiniband,
        )

        response = self._client.create_deployment(request)

        # Create Deployment facade
        deployment = Deployment._from_response(self, response)

        # Wait for deployment to be ready
        deployment.wait_until_ready(timeout=timeout)

        # Refresh to get final URL and state
        deployment.refresh()

        return deployment

    def deploy_vllm(
        self,
        model: str = "Qwen/Qwen3-0.6B",
        name: Optional[str] = None,
        gpu_count: Optional[int] = None,
        gpu_models: Optional[List[str]] = None,
        memory: str = "16Gi",
        storage: bool = True,
        tensor_parallel_size: Optional[int] = None,
        max_model_len: Optional[int] = None,
        dtype: Optional[str] = None,
        quantization: Optional[str] = None,
        served_model_name: Optional[str] = None,
        api_key: Optional[str] = None,
        gpu_memory_utilization: Optional[float] = None,
        enforce_eager: bool = False,
        trust_remote_code: bool = False,
        env: Optional[Dict[str, str]] = None,
        ttl_seconds: Optional[int] = None,
        timeout: int = 600,
        health_check: Optional[HealthCheckConfig] = None,
        interconnect: Optional[str] = None,
        geo: Optional[str] = None,
        spot: Optional[bool] = None,
        infiniband: Optional[bool] = None,
    ) -> Deployment:
        """
        Deploy a vLLM inference server.

        Args:
            model: HuggingFace model ID (default: Qwen/Qwen3-0.6B)
            name: Deployment name (auto-generated if not specified)
            gpu_count: Number of GPUs (auto-detected based on model size if not specified)
            gpu_models: GPU model requirements (e.g., ["A100", "H100"])
            memory: Memory allocation (default: 16Gi)
            storage: Enable persistent storage for model cache (default: True)
            tensor_parallel_size: Number of GPUs for tensor parallelism
            max_model_len: Maximum sequence length
            dtype: Model dtype (auto, float16, bfloat16)
            quantization: Quantization method (awq, gptq, squeezellm, fp8)
            served_model_name: OpenAI API model name
            api_key: API key for vLLM authentication
            gpu_memory_utilization: Fraction of GPU memory to use (0.0-1.0)
            enforce_eager: Disable CUDA graphs
            trust_remote_code: Allow custom model code from HuggingFace
            env: Additional environment variables
            ttl_seconds: Auto-delete after this many seconds
            timeout: Wait timeout in seconds
            health_check: Custom health check configuration. If not provided,
                         uses sensible defaults for vLLM (10-minute startup tolerance).
            interconnect: GPU interconnect type. "SXM" or "PCIe"
            geo: Geographic region preference. "US", "EU", "CA", "APAC"
            spot: Spot instance preference. True=prefer spot, False=exclude spot
            infiniband: Require InfiniBand networking

        Returns:
            Deployment object with .url, .status(), .logs(), .delete() methods

        Example:
            >>> client = BasilicaClient()
            >>> deployment = client.deploy_vllm("meta-llama/Llama-2-7b")
            >>> print(f"OpenAI API: {deployment.url}/v1/chat/completions")
        """
        from .templates.model_size import estimate_gpu_requirements

        # Always estimate GPU requirements to get recommended GPU
        reqs = estimate_gpu_requirements(model)

        # Use user-specified GPU count or auto-detected
        if gpu_count is None:
            gpu_count = reqs.gpu_count

        # Generate name if not provided
        if name is None:
            import uuid
            model_part = model.split("/")[-1].lower()
            model_part = re.sub(r"[^a-z0-9-]", "-", model_part)[:40].strip("-")
            name = f"vllm-{model_part}-{str(uuid.uuid4())[:8]}"

        # Build vLLM command
        args = [
            "serve", model,
            "--host", "0.0.0.0",
            "--port", "8000",
        ]

        if tensor_parallel_size is not None:
            args.extend(["--tensor-parallel-size", str(tensor_parallel_size)])
        if max_model_len is not None:
            args.extend(["--max-model-len", str(max_model_len)])
        if dtype is not None:
            args.extend(["--dtype", dtype])
        if quantization is not None:
            args.extend(["--quantization", quantization])
        if served_model_name is not None:
            args.extend(["--served-model-name", served_model_name])
        if api_key is not None:
            args.extend(["--api-key", api_key])
        if gpu_memory_utilization is not None:
            args.extend(["--gpu-memory-utilization", str(gpu_memory_utilization)])
        if enforce_eager:
            args.append("--enforce-eager")
        if trust_remote_code:
            args.append("--trust-remote-code")

        # Build storage spec
        storage_spec = None
        if storage:
            storage_spec = StorageSpec(
                persistent=PersistentStorageSpec(
                    enabled=True,
                    backend=StorageBackend.R2,
                    bucket="",
                    credentials_secret="basilica-r2-credentials",
                    sync_interval_ms=1000,
                    cache_size_mb=4096,
                    mount_path="/root/.cache",
                )
            )

        # Build GPU spec - use min_gpu_memory_gb for scheduling, let API find suitable GPUs
        gpu_spec = GpuRequirementsSpec(
            count=gpu_count,
            model=gpu_models or [],
            min_cuda_version=None,
            min_gpu_memory_gb=reqs.memory_gb,
            interconnect=interconnect,
            geo=geo,
            spot=spot,
            infiniband=infiniband,
        )

        # Build resources
        resources = ResourceRequirements(
            cpu="4",
            memory=memory,
            gpus=gpu_spec,
        )

        # Apply default health check for vLLM if not provided
        effective_health_check = health_check or _build_inference_health_check(port=8000)

        # Create the deployment request
        request = CreateDeploymentRequest(
            instance_name=name,
            image="vllm/vllm-openai:latest",
            replicas=1,
            port=8000,
            command=["vllm"],
            args=args,
            env=env,
            resources=resources,
            ttl_seconds=ttl_seconds,
            public=True,
            storage=storage_spec,
            health_check=effective_health_check,
        )

        # Create deployment
        response = self._client.create_deployment(request)

        # Create Deployment facade
        deployment = Deployment._from_response(self, response)

        # Wait for deployment to be ready
        deployment.wait_until_ready(timeout=timeout)
        deployment.refresh()

        return deployment

    def deploy_sglang(
        self,
        model: str = "Qwen/Qwen2.5-0.5B-Instruct",
        name: Optional[str] = None,
        gpu_count: Optional[int] = None,
        gpu_models: Optional[List[str]] = None,
        memory: str = "16Gi",
        storage: bool = True,
        tensor_parallel_size: Optional[int] = None,
        context_length: Optional[int] = None,
        quantization: Optional[str] = None,
        mem_fraction_static: Optional[float] = None,
        trust_remote_code: bool = False,
        env: Optional[Dict[str, str]] = None,
        ttl_seconds: Optional[int] = None,
        timeout: int = 600,
        health_check: Optional[HealthCheckConfig] = None,
        interconnect: Optional[str] = None,
        geo: Optional[str] = None,
        spot: Optional[bool] = None,
        infiniband: Optional[bool] = None,
    ) -> Deployment:
        """
        Deploy an SGLang inference server.

        Args:
            model: HuggingFace model ID (default: Qwen/Qwen2.5-0.5B-Instruct)
            name: Deployment name (auto-generated if not specified)
            gpu_count: Number of GPUs (auto-detected based on model size if not specified)
            gpu_models: GPU model requirements (e.g., ["A100", "H100"])
            memory: Memory allocation (default: 16Gi)
            storage: Enable persistent storage for model cache (default: True)
            tensor_parallel_size: Number of GPUs for tensor parallelism
            context_length: Maximum context length
            quantization: Quantization method
            mem_fraction_static: Static memory fraction (0.0-1.0)
            trust_remote_code: Allow custom model code from HuggingFace
            env: Additional environment variables
            ttl_seconds: Auto-delete after this many seconds
            timeout: Wait timeout in seconds
            health_check: Custom health check configuration. If not provided,
                         uses sensible defaults for SGLang (10-minute startup tolerance).
            interconnect: GPU interconnect type. "SXM" or "PCIe"
            geo: Geographic region preference. "US", "EU", "CA", "APAC"
            spot: Spot instance preference. True=prefer spot, False=exclude spot
            infiniband: Require InfiniBand networking

        Returns:
            Deployment object with .url, .status(), .logs(), .delete() methods

        Example:
            >>> client = BasilicaClient()
            >>> deployment = client.deploy_sglang("Qwen/Qwen2.5-0.5B-Instruct")
            >>> print(deployment.url)
        """
        from .templates.model_size import estimate_gpu_requirements

        # Always estimate GPU requirements to get recommended GPU
        reqs = estimate_gpu_requirements(model)

        # Use user-specified GPU count or auto-detected
        if gpu_count is None:
            gpu_count = reqs.gpu_count

        # Generate name if not provided
        if name is None:
            import uuid
            model_part = model.split("/")[-1].lower()
            model_part = re.sub(r"[^a-z0-9-]", "-", model_part)[:40].strip("-")
            name = f"sglang-{model_part}-{str(uuid.uuid4())[:8]}"

        # Build SGLang command
        args = [
            "-m", "sglang.launch_server",
            "--model-path", model,
            "--host", "0.0.0.0",
            "--port", "30000",
        ]

        if tensor_parallel_size is not None:
            args.extend(["--tp", str(tensor_parallel_size)])
        if context_length is not None:
            args.extend(["--context-length", str(context_length)])
        if quantization is not None:
            args.extend(["--quantization", quantization])
        if mem_fraction_static is not None:
            args.extend(["--mem-fraction-static", str(mem_fraction_static)])
        if trust_remote_code:
            args.append("--trust-remote-code")

        # Build storage spec
        storage_spec = None
        if storage:
            storage_spec = StorageSpec(
                persistent=PersistentStorageSpec(
                    enabled=True,
                    backend=StorageBackend.R2,
                    bucket="",
                    credentials_secret="basilica-r2-credentials",
                    sync_interval_ms=1000,
                    cache_size_mb=4096,
                    mount_path="/root/.cache",
                )
            )

        # Build GPU spec - use min_gpu_memory_gb for scheduling, let API find suitable GPUs
        gpu_spec = GpuRequirementsSpec(
            count=gpu_count,
            model=gpu_models or [],
            min_cuda_version=None,
            min_gpu_memory_gb=reqs.memory_gb,
            interconnect=interconnect,
            geo=geo,
            spot=spot,
            infiniband=infiniband,
        )

        # Build resources
        resources = ResourceRequirements(
            cpu="4",
            memory=memory,
            gpus=gpu_spec,
        )

        # Apply default health check for SGLang if not provided
        effective_health_check = health_check or _build_inference_health_check(port=30000)

        # Create the deployment request
        request = CreateDeploymentRequest(
            instance_name=name,
            image="lmsysorg/sglang:latest",
            replicas=1,
            port=30000,
            command=["python"],
            args=args,
            env=env,
            resources=resources,
            ttl_seconds=ttl_seconds,
            public=True,
            storage=storage_spec,
            health_check=effective_health_check,
        )

        # Create deployment
        response = self._client.create_deployment(request)

        # Create Deployment facade
        deployment = Deployment._from_response(self, response)

        # Wait for deployment to be ready
        deployment.wait_until_ready(timeout=timeout)
        deployment.refresh()

        return deployment

    def get(self, name: str) -> Deployment:
        """
        Get an existing deployment by name.

        Args:
            name: The deployment instance name

        Returns:
            Deployment: A deployment object

        Raises:
            DeploymentNotFound: If deployment doesn't exist

        Example:
            >>> deployment = client.get("my-api")
            >>> print(deployment.url)
            >>> print(deployment.logs(tail=50))
        """
        try:
            response = self.get_deployment(name)
            return Deployment._from_response(self, response)
        except (KeyError, Exception) as e:
            error_msg = str(e)
            if "not found" in error_msg.lower() or "Not found" in error_msg:
                raise DeploymentNotFound(name) from None
            raise

    def list(self) -> List[Deployment]:
        """
        List all deployments.

        Returns:
            List of Deployment objects

        Example:
            >>> for deployment in client.list():
            ...     print(f"{deployment.name}: {deployment.state}")
        """
        response = self.list_deployments()
        deployments = []
        for summary in response.deployments:
            try:
                full_response = self.get_deployment(summary.instance_name)
                deployments.append(Deployment._from_response(self, full_response))
            except Exception as e:
                error_msg = str(e).lower()
                if "not found" in error_msg:
                    continue
                warnings.warn(
                    f"Failed to fetch deployment '{summary.instance_name}': {e}",
                    stacklevel=2,
                )
        return deployments

    # -------------------------------------------------------------------------
    # Low-Level API Methods (for advanced use cases)
    # -------------------------------------------------------------------------

    def health_check(self) -> HealthCheckResponse:
        """
        Check API health status.

        Returns:
            HealthCheckResponse with status, version, and validator info
        """
        return self._client.health_check()

    def list_nodes(
        self,
        available: Optional[bool] = None,
        gpu_type: Optional[str] = None,
        min_gpu_count: Optional[int] = None,
        min_gpu_memory: Optional[int] = None,
    ) -> List[AvailableNode]:
        """
        List available compute nodes.

        Args:
            available: Filter by availability
            gpu_type: Filter by GPU type (e.g., "A100", "H100")
            min_gpu_count: Minimum number of GPUs
            min_gpu_memory: Minimum GPU memory in GB

        Returns:
            List of AvailableNode objects
        """
        query = None
        if any(
            [
                available is not None,
                gpu_type is not None,
                min_gpu_count is not None,
                min_gpu_memory is not None,
            ]
        ):
            query = ListAvailableNodesQuery(
                available=available,
                gpu_type=gpu_type,
                min_gpu_count=min_gpu_count,
                min_gpu_memory=min_gpu_memory,
            )
        return self._client.list_nodes(query)

    def start_rental(
        self,
        container_image: Optional[str] = None,
        gpu_type: Optional[str] = None,
        max_hourly_rate: Optional[float] = None,
        ssh_pubkey_path: Optional[str] = None,
        environment: Optional[Dict[str, str]] = None,
        ports: Optional[List[Dict[str, Any]]] = None,
        command: Optional[List[str]] = None,
    ) -> RentalResponse:
        """
        Start a new GPU rental.

        For SSH access, ensure you have an SSH key at ~/.ssh/basilica_ed25519.pub

        Args:
            container_image: Docker image to run
            gpu_type: GPU type to request
            max_hourly_rate: Maximum USD per GPU-hour (rounded to nearest cent)
            ssh_pubkey_path: Path to SSH public key file
            environment: Environment variables
            ports: Port mappings
            command: Command to run

        Returns:
            RentalResponse with rental details
        """
        if container_image is None:
            container_image = DEFAULT_CONTAINER_IMAGE

        if gpu_type is None:
            gpu_type = DEFAULT_GPU_TYPE

        if max_hourly_rate is None:
            raise ValueError("max_hourly_rate is required and must be provided")

        ssh_public_key = None
        if ssh_pubkey_path is not None:
            ssh_key_path = os.path.expanduser(ssh_pubkey_path)
        else:
            ssh_key_path = os.path.expanduser("~/.ssh/basilica_ed25519.pub")

        if os.path.exists(ssh_key_path):
            with open(ssh_key_path) as f:
                ssh_public_key = f.read().strip()
        elif ssh_pubkey_path is not None:
            raise FileNotFoundError(
                f"SSH public key file not found: {ssh_key_path}"
            )

        resources = {
            "gpu_count": DEFAULT_GPU_COUNT,
            "gpu_types": [gpu_type] if gpu_type else [],
            "cpu_cores": DEFAULT_CPU_CORES,
            "memory_mb": DEFAULT_MEMORY_MB,
            "storage_mb": DEFAULT_STORAGE_MB,
        }

        port_mappings = []
        if ports:
            for port in ports:
                port_mappings.append(
                    PortMappingRequest(
                        container_port=port.get("container_port", 0),
                        host_port=port.get("host_port", 0),
                        protocol=port.get("protocol", "tcp"),
                    )
                )

        resource_req = ResourceRequirementsRequest(
            cpu_cores=resources.get("cpu_cores", DEFAULT_CPU_CORES),
            memory_mb=resources.get("memory_mb", DEFAULT_MEMORY_MB),
            storage_mb=resources.get("storage_mb", DEFAULT_STORAGE_MB),
            gpu_count=resources.get("gpu_count", DEFAULT_GPU_COUNT),
            gpu_types=resources.get("gpu_types", []),
        )

        request = StartRentalApiRequest(
            gpu_category=gpu_type,
            gpu_count=DEFAULT_GPU_COUNT,
            min_memory_gb=DEFAULT_GPU_MIN_MEMORY_GB,
            max_hourly_rate=max_hourly_rate,
            container_image=container_image,
            ssh_public_key=ssh_public_key if ssh_public_key else "",
            environment=environment or {},
            ports=port_mappings,
            resources=resource_req,
            command=command if command is not None else DEFAULT_COMMAND,
            volumes=[],
        )

        return self._client.start_rental(request)

    def get_rental(self, rental_id: str) -> RentalStatusWithSshResponse:
        """Get rental status by ID."""
        return self._client.get_rental(rental_id)

    def stop_rental(self, rental_id: str) -> None:
        """Stop a rental by ID."""
        self._client.stop_rental(rental_id)

    def list_rentals(
        self,
        status: Optional[str] = None,
        gpu_type: Optional[str] = None,
        min_gpu_count: Optional[int] = None,
    ) -> Dict[str, Any]:
        """List rentals with optional filters."""
        query = None
        if any([status is not None, gpu_type is not None, min_gpu_count is not None]):
            query = ListRentalsQuery(
                status=status, gpu_type=gpu_type, min_gpu_count=min_gpu_count
            )
        return self._client.list_rentals(query)

    def create_deployment(
        self,
        instance_name: str,
        image: str,
        replicas: int = 1,
        port: int = 80,
        command: Optional[List[str]] = None,
        args: Optional[List[str]] = None,
        env: Optional[Dict[str, str]] = None,
        cpu: str = "500m",
        memory: str = "512Mi",
        gpu_count: Optional[int] = None,
        gpu_models: Optional[List[str]] = None,
        min_cuda_version: Optional[str] = None,
        min_gpu_memory_gb: Optional[int] = None,
        interconnect: Optional[str] = None,
        geo: Optional[str] = None,
        spot: Optional[bool] = None,
        infiniband: Optional[bool] = None,
        ttl_seconds: Optional[int] = None,
        public: bool = True,
        storage: Optional[Union[str, StorageSpec]] = None,
        topology_spread: Optional[TopologySpreadConfig] = None,
        health_check: Optional[HealthCheckConfig] = None,
        websocket: Optional[WebSocketConfig] = None,
        public_metadata: bool = False,
    ) -> DeploymentResponse:
        """
        Create a deployment (low-level API).

        For most use cases, prefer the high-level deploy() method.

        Args:
            instance_name: Deployment name (DNS-safe)
            image: Container image
            replicas: Number of replicas
            port: Container port
            command: Container command
            args: Command arguments
            env: Environment variables
            cpu: CPU allocation
            memory: Memory allocation
            gpu_count: Number of GPUs
            gpu_models: Acceptable GPU models
            min_cuda_version: Minimum CUDA version
            min_gpu_memory_gb: Minimum GPU memory
            interconnect: GPU interconnect type. "SXM" or "PCIe"
            geo: Geographic region preference. "US", "EU", "CA", "APAC"
            spot: Spot instance preference. True=prefer spot, False=exclude spot
            infiniband: Require InfiniBand networking
            ttl_seconds: Auto-delete timeout
            public: Create public URL
            storage: Storage path or StorageSpec
            topology_spread: Topology spread configuration for pod distribution
            health_check: Custom health check configuration (HealthCheckConfig)
            websocket: WebSocket configuration (WebSocketConfig)
            public_metadata: Enable public metadata enrollment for validator verification

        Returns:
            DeploymentResponse with deployment details
        """
        # Build GPU spec if requested
        gpu_spec = None
        if gpu_count is not None:
            gpu_spec = GpuRequirementsSpec(
                count=gpu_count,
                model=gpu_models or [],
                min_cuda_version=min_cuda_version,
                min_gpu_memory_gb=min_gpu_memory_gb,
                interconnect=interconnect,
                geo=geo,
                spot=spot,
                infiniband=infiniband,
            )

        # Build resources
        resources = ResourceRequirements(cpu=cpu, memory=memory, gpus=gpu_spec)

        storage_spec = None
        if storage is not None:
            if isinstance(storage, str):
                storage_spec = StorageSpec(
                    persistent=PersistentStorageSpec(
                        enabled=True,
                        backend=StorageBackend.R2,
                        bucket="",
                        credentials_secret=None,
                        sync_interval_ms=1000,
                        cache_size_mb=1024,
                        mount_path=storage,
                    )
                )
            else:
                storage_spec = storage

        request = CreateDeploymentRequest(
            instance_name=instance_name,
            image=image,
            replicas=replicas,
            port=port,
            command=command,
            args=args,
            env=env,
            resources=resources,
            ttl_seconds=ttl_seconds,
            public=public,
            storage=storage_spec,
            topology_spread=topology_spread,
            health_check=health_check,
            websocket=websocket,
            public_metadata=public_metadata,
        )

        return self._client.create_deployment(request)

    def get_deployment(self, instance_name: str) -> DeploymentResponse:
        """Get deployment status by name."""
        return self._client.get_deployment(instance_name)

    def delete_deployment(self, instance_name: str) -> DeleteDeploymentResponse:
        """Delete a deployment by name."""
        return self._client.delete_deployment(instance_name)

    def list_deployments(self) -> DeploymentListResponse:
        """List all deployments."""
        return self._client.list_deployments()

    def enroll_metadata(
        self, instance_name: str, enabled: bool
    ) -> EnrollMetadataResponse:
        """Toggle public metadata enrollment for a deployment.

        Args:
            instance_name: Deployment name
            enabled: True to enroll, False to unenroll
        """
        return self._client.enroll_metadata(instance_name, enabled)

    def get_enrollment_status(self, instance_name: str) -> EnrollMetadataResponse:
        """Check public metadata enrollment status for a deployment.

        Args:
            instance_name: Deployment name
        """
        return self._client.get_enrollment_status(instance_name)

    def get_public_deployment_metadata(
        self, instance_name: str
    ) -> PublicDeploymentMetadataResponse:
        """Fetch public metadata for a deployment (no authentication required).

        Args:
            instance_name: Deployment name
        """
        return self._client.get_public_deployment_metadata(instance_name)

    def get_deployment_logs(
        self, instance_name: str, follow: bool = False, tail: Optional[int] = None
    ) -> str:
        """Get deployment logs."""
        return self._client.get_deployment_logs(instance_name, follow, tail)

    def get_balance(self) -> Dict[str, Any]:
        """Get account balance."""
        return self._client.get_balance()

    def list_usage_history(self, limit: int = 50, offset: int = 0) -> Dict[str, Any]:
        """Get usage history for billing."""
        return self._client.list_usage_history(limit, offset)

    # -------------------------------------------------------------------------
    # SSH Key Management Methods
    # -------------------------------------------------------------------------

    def register_ssh_key(
        self,
        name: str,
        public_key: Optional[str] = None,
        public_key_path: Optional[str] = None,
    ) -> SshKeyResponse:
        """
        Register an SSH key for secure cloud rentals.

        Only one SSH key per user is allowed. This key is required before
        starting CPU rentals.

        Args:
            name: A friendly name for the SSH key (e.g., "my-laptop")
            public_key: The SSH public key content directly
            public_key_path: Path to SSH public key file (default: ~/.ssh/id_ed25519.pub)

        Returns:
            SshKeyResponse with the registered key details including its ID

        Example:
            >>> key = client.register_ssh_key("my-laptop")
            >>> print(f"Key ID: {key.id}")
        """
        if public_key is None:
            if public_key_path is None:
                public_key_path = "~/.ssh/id_ed25519.pub"

            key_path = os.path.expanduser(public_key_path)
            if not os.path.exists(key_path):
                raise FileNotFoundError(f"SSH public key file not found: {key_path}")

            with open(key_path) as f:
                public_key = f.read().strip()

        return self._client.register_ssh_key(name, public_key)

    def get_ssh_key(self) -> Optional[SshKeyResponse]:
        """
        Get the authenticated user's registered SSH key.

        Returns:
            SshKeyResponse if a key is registered, None otherwise
        """
        return self._client.get_ssh_key()

    def delete_ssh_key(self) -> None:
        """Delete the authenticated user's SSH key."""
        self._client.delete_ssh_key()

    # -------------------------------------------------------------------------
    # Secure Cloud CPU Rental Methods
    # -------------------------------------------------------------------------

    def list_cpu_offerings(self) -> List[CpuOffering]:
        """
        List available CPU-only offerings from secure cloud providers.

        Returns:
            List of CpuOffering objects with specs and pricing

        Example:
            >>> offerings = client.list_cpu_offerings()
            >>> for o in offerings:
            ...     print(f"{o.id}: {o.vcpu_count} vCPUs @ ${o.hourly_rate}/hr")
        """
        return self._client.list_cpu_offerings()

    def start_cpu_rental(
        self,
        offering_id: str,
        ssh_public_key_id: Optional[str] = None,
    ) -> CpuRentalResponse:
        """
        Start a CPU-only rental.

        Args:
            offering_id: The offering ID from list_cpu_offerings()
            ssh_public_key_id: SSH key ID (auto-detected if not provided)

        Returns:
            CpuRentalResponse with rental details and SSH command

        Example:
            >>> offerings = client.list_cpu_offerings()
            >>> rental = client.start_cpu_rental(offerings[0].id)
            >>> print(f"SSH: {rental.ssh_command}")
        """
        # Auto-detect SSH key ID if not provided
        if ssh_public_key_id is None:
            key = self.get_ssh_key()
            if key is None:
                raise ValidationError(
                    "No SSH key registered. Use register_ssh_key() first."
                )
            ssh_public_key_id = key.id

        request = StartCpuRentalRequest(
            offering_id=offering_id,
            ssh_public_key_id=ssh_public_key_id,
        )

        return self._client.start_cpu_rental(request)

    def stop_cpu_rental(self, rental_id: str) -> StopCpuRentalResponse:
        """
        Stop a CPU rental.

        Args:
            rental_id: The rental ID to stop

        Returns:
            StopCpuRentalResponse with duration and total cost
        """
        return self._client.stop_cpu_rental(rental_id)

    def list_cpu_rentals(self) -> ListCpuRentalsResponse:
        """
        List all CPU rentals for the authenticated user.

        Returns:
            ListCpuRentalsResponse with rental list and total count
        """
        return self._client.list_cpu_rentals()

    # -------------------------------------------------------------------------
    # Secure Cloud GPU Rental Methods
    # -------------------------------------------------------------------------

    def list_secure_cloud_gpus(
        self, query: Optional[GpuPriceQuery] = None
    ) -> List[GpuOffering]:
        """
        List available GPU offerings from secure cloud providers.

        Returns GPU instances from datacenter providers like DataCrunch,
        Hyperstack, Lambda Labs, etc.

        Args:
            query: Optional GpuPriceQuery to filter by interconnect, region, spot.
                   Example: GpuPriceQuery(interconnect="SXM", region="US")

        Returns:
            List of GpuOffering objects with GPU specs and pricing

        Example:
            >>> offerings = client.list_secure_cloud_gpus()
            >>> for o in offerings:
            ...     print(f"{o.gpu_count}x {o.gpu_type} @ ${o.hourly_rate}/hr ({o.provider})")
            >>> sxm = client.list_secure_cloud_gpus(GpuPriceQuery(interconnect="SXM"))
        """
        return self._client.list_secure_cloud_gpus(query=query)

    def start_secure_cloud_rental(
        self,
        offering_id: str,
        ssh_public_key_id: Optional[str] = None,
    ) -> SecureCloudRentalResponse:
        """
        Start a secure cloud GPU rental from a datacenter provider.

        Args:
            offering_id: The offering ID from list_secure_cloud_gpus()
            ssh_public_key_id: SSH key ID (auto-detected if not provided)

        Returns:
            SecureCloudRentalResponse with rental details and SSH command

        Example:
            >>> offerings = client.list_secure_cloud_gpus()
            >>> rental = client.start_secure_cloud_rental(offerings[0].id)
            >>> print(f"SSH: {rental.ssh_command}")
        """
        # Auto-detect SSH key ID if not provided
        if ssh_public_key_id is None:
            key = self.get_ssh_key()
            if key is None:
                raise ValidationError(
                    "No SSH key registered. Use register_ssh_key() first."
                )
            ssh_public_key_id = key.id

        request = StartSecureCloudRentalRequest(
            offering_id=offering_id,
            ssh_public_key_id=ssh_public_key_id,
        )

        return self._client.start_secure_cloud_rental(request)

    def stop_secure_cloud_rental(self, rental_id: str) -> StopSecureCloudRentalResponse:
        """
        Stop a secure cloud GPU rental.

        Terminates the provider instance, finalizes billing, and returns total cost.

        Args:
            rental_id: The rental ID to stop

        Returns:
            StopSecureCloudRentalResponse with duration and total cost

        Example:
            >>> result = client.stop_secure_cloud_rental(rental_id)
            >>> print(f"Total cost: ${result.total_cost}")
        """
        return self._client.stop_secure_cloud_rental(rental_id)

    def list_secure_cloud_rentals(self) -> ListSecureCloudRentalsResponse:
        """
        List all secure cloud GPU rentals for the authenticated user.

        Returns all datacenter GPU rentals including their status, IP addresses,
        and cost information.

        Returns:
            ListSecureCloudRentalsResponse with rental list and total count

        Example:
            >>> rentals = client.list_secure_cloud_rentals()
            >>> print(f"Active rentals: {rentals.total_count}")
            >>> for r in rentals.rentals:
            ...     print(f"  {r.rental_id}: {r.gpu_count}x {r.gpu_type} - ${r.hourly_cost}/hr")
        """
        return self._client.list_secure_cloud_rentals()

    # -------------------------------------------------------------------------
    # Async API Methods
    # -------------------------------------------------------------------------

    async def deploy_async(
        self,
        name: str,
        source: Optional[Union[str, Path, Callable]] = None,
        image: str = DEFAULT_PYTHON_IMAGE,
        port: int = 8000,
        env: Optional[Dict[str, str]] = None,
        cpu: str = "500m",
        memory: str = "512Mi",
        storage: Union[bool, str] = False,
        gpu_count: Optional[int] = None,
        gpu_models: Optional[List[str]] = None,
        min_cuda_version: Optional[str] = None,
        min_gpu_memory_gb: Optional[int] = None,
        replicas: int = 1,
        ttl_seconds: Optional[int] = None,
        public: bool = True,
        timeout: int = 300,
        pip_packages: Optional[List[str]] = None,
        topology_spread: Optional[TopologySpreadConfig] = None,
        health_check: Optional[HealthCheckConfig] = None,
        interconnect: Optional[str] = None,
        geo: Optional[str] = None,
        spot: Optional[bool] = None,
        infiniband: Optional[bool] = None,
    ) -> Deployment:
        """
        Deploy an application asynchronously.

        This is the async version of deploy(). It uses asyncio.sleep() for
        waiting, allowing other coroutines to run concurrently.

        Args:
            name: Deployment name (DNS-safe: lowercase, numbers, hyphens).
            source: Python source code to deploy (file path, inline code, or callable).
            image: Container image. Default: python:3.11-slim
            port: Port your application listens on. Default: 8000
            env: Environment variables as a dict.
            cpu: CPU allocation. Default: "500m"
            memory: Memory allocation. Default: "512Mi"
            storage: Persistent storage configuration.
            gpu_count: Number of GPUs (1-8).
            gpu_models: Acceptable GPU models.
            min_cuda_version: Minimum CUDA version.
            min_gpu_memory_gb: Minimum GPU VRAM in GB.
            interconnect: GPU interconnect type. "SXM" or "PCIe"
            geo: Geographic region preference. "US", "EU", "CA", "APAC"
            spot: Spot instance preference. True=prefer spot, False=exclude spot
            infiniband: Require InfiniBand networking. True/False
            replicas: Number of instances. Default: 1
            ttl_seconds: Auto-delete after N seconds.
            public: Create public URL. Default: True
            timeout: Seconds to wait for deployment. Default: 300
            pip_packages: Additional pip packages to install.
            health_check: Custom health check configuration (HealthCheckConfig).

        Returns:
            Deployment: A deployment object with url, logs(), delete(), etc.

        Example:
            >>> async def main():
            ...     client = BasilicaClient()
            ...     deployment = await client.deploy_async("my-app", source="app.py")
            ...     print(deployment.url)
        """
        request = self._build_deploy_request(
            name=name,
            source=source,
            image=image,
            port=port,
            env=env,
            cpu=cpu,
            memory=memory,
            storage=storage,
            gpu_count=gpu_count,
            gpu_models=gpu_models,
            min_cuda_version=min_cuda_version,
            min_gpu_memory_gb=min_gpu_memory_gb,
            replicas=replicas,
            ttl_seconds=ttl_seconds,
            public=public,
            pip_packages=pip_packages,
            topology_spread=topology_spread,
            health_check=health_check,
            interconnect=interconnect,
            geo=geo,
            spot=spot,
            infiniband=infiniband,
        )

        loop = asyncio.get_running_loop()
        response = await loop.run_in_executor(
            None, self._client.create_deployment, request
        )

        deployment = Deployment._from_response(self, response)
        await deployment.wait_until_ready_async(timeout=timeout)
        await deployment.refresh_async()

        return deployment

    async def get_async(self, name: str) -> Deployment:
        """
        Get an existing deployment by name asynchronously.

        Args:
            name: The deployment instance name

        Returns:
            Deployment: A deployment object

        Raises:
            DeploymentNotFound: If deployment doesn't exist

        Example:
            >>> deployment = await client.get_async("my-api")
            >>> print(deployment.url)
        """
        try:
            response = await self.get_deployment_async(name)
            return Deployment._from_response(self, response)
        except (KeyError, Exception) as e:
            error_msg = str(e)
            if "not found" in error_msg.lower() or "Not found" in error_msg:
                raise DeploymentNotFound(name) from None
            raise

    async def list_async(self) -> List[Deployment]:
        """
        List all deployments asynchronously.

        Returns:
            List of Deployment objects

        Example:
            >>> deployments = await client.list_async()
            >>> for d in deployments:
            ...     print(f"{d.name}: {d.state}")
        """
        response = await self.list_deployments_async()
        deployments = []
        for summary in response.deployments:
            try:
                full_response = await self.get_deployment_async(summary.instance_name)
                deployments.append(Deployment._from_response(self, full_response))
            except Exception as e:
                error_msg = str(e).lower()
                if "not found" in error_msg:
                    continue
                warnings.warn(
                    f"Failed to fetch deployment '{summary.instance_name}': {e}",
                    stacklevel=2,
                )
        return deployments

    async def create_deployment_async(
        self,
        instance_name: str,
        image: str,
        replicas: int = 1,
        port: int = 80,
        command: Optional[List[str]] = None,
        args: Optional[List[str]] = None,
        env: Optional[Dict[str, str]] = None,
        cpu: str = "500m",
        memory: str = "512Mi",
        gpu_count: Optional[int] = None,
        gpu_models: Optional[List[str]] = None,
        min_cuda_version: Optional[str] = None,
        min_gpu_memory_gb: Optional[int] = None,
        interconnect: Optional[str] = None,
        geo: Optional[str] = None,
        spot: Optional[bool] = None,
        infiniband: Optional[bool] = None,
        ttl_seconds: Optional[int] = None,
        public: bool = True,
        storage: Optional[Union[str, StorageSpec]] = None,
        topology_spread: Optional[TopologySpreadConfig] = None,
        health_check: Optional[HealthCheckConfig] = None,
        websocket: Optional[WebSocketConfig] = None,
        public_metadata: bool = False,
    ) -> DeploymentResponse:
        """
        Create a deployment asynchronously (low-level API).

        For most use cases, prefer the high-level deploy_async() method.
        """
        gpu_spec = None
        if gpu_count is not None:
            gpu_spec = GpuRequirementsSpec(
                count=gpu_count,
                model=gpu_models or [],
                min_cuda_version=min_cuda_version,
                min_gpu_memory_gb=min_gpu_memory_gb,
                interconnect=interconnect,
                geo=geo,
                spot=spot,
                infiniband=infiniband,
            )

        resources = ResourceRequirements(cpu=cpu, memory=memory, gpus=gpu_spec)

        storage_spec = None
        if storage is not None:
            if isinstance(storage, str):
                storage_spec = StorageSpec(
                    persistent=PersistentStorageSpec(
                        enabled=True,
                        backend=StorageBackend.R2,
                        bucket="",
                        credentials_secret=None,
                        sync_interval_ms=1000,
                        cache_size_mb=1024,
                        mount_path=storage,
                    )
                )
            else:
                storage_spec = storage

        request = CreateDeploymentRequest(
            instance_name=instance_name,
            image=image,
            replicas=replicas,
            port=port,
            command=command,
            args=args,
            env=env,
            resources=resources,
            ttl_seconds=ttl_seconds,
            public=public,
            storage=storage_spec,
            topology_spread=topology_spread,
            health_check=health_check,
            websocket=websocket,
            public_metadata=public_metadata,
        )

        loop = asyncio.get_running_loop()
        return await loop.run_in_executor(
            None, self._client.create_deployment, request
        )

    async def get_deployment_async(self, instance_name: str) -> DeploymentResponse:
        """Get deployment status by name asynchronously."""
        loop = asyncio.get_running_loop()
        return await loop.run_in_executor(
            None, self._client.get_deployment, instance_name
        )

    async def delete_deployment_async(self, instance_name: str) -> DeleteDeploymentResponse:
        """Delete a deployment by name asynchronously."""
        loop = asyncio.get_running_loop()
        return await loop.run_in_executor(
            None, self._client.delete_deployment, instance_name
        )

    async def list_deployments_async(self) -> DeploymentListResponse:
        """List all deployments asynchronously."""
        loop = asyncio.get_running_loop()
        return await loop.run_in_executor(None, self._client.list_deployments)

    async def enroll_metadata_async(
        self, instance_name: str, enabled: bool
    ) -> EnrollMetadataResponse:
        """Toggle public metadata enrollment asynchronously."""
        loop = asyncio.get_running_loop()
        return await loop.run_in_executor(
            None, lambda: self._client.enroll_metadata(instance_name, enabled)
        )

    async def get_enrollment_status_async(
        self, instance_name: str
    ) -> EnrollMetadataResponse:
        """Check public metadata enrollment status asynchronously."""
        loop = asyncio.get_running_loop()
        return await loop.run_in_executor(
            None, self._client.get_enrollment_status, instance_name
        )

    async def get_public_deployment_metadata_async(
        self, instance_name: str
    ) -> PublicDeploymentMetadataResponse:
        """Fetch public metadata asynchronously (no authentication required)."""
        loop = asyncio.get_running_loop()
        return await loop.run_in_executor(
            None, self._client.get_public_deployment_metadata, instance_name
        )

    async def get_deployment_logs_async(
        self, instance_name: str, follow: bool = False, tail: Optional[int] = None
    ) -> str:
        """Get deployment logs asynchronously."""
        loop = asyncio.get_running_loop()
        return await loop.run_in_executor(
            None,
            lambda: self._client.get_deployment_logs(instance_name, follow, tail)
        )

    async def health_check_async(self) -> HealthCheckResponse:
        """Check API health status asynchronously."""
        loop = asyncio.get_running_loop()
        return await loop.run_in_executor(None, self._client.health_check)

    async def list_nodes_async(
        self,
        available: Optional[bool] = None,
        gpu_type: Optional[str] = None,
        min_gpu_count: Optional[int] = None,
        min_gpu_memory: Optional[int] = None,
    ) -> List[AvailableNode]:
        """List available compute nodes asynchronously."""
        query = None
        if any([
            available is not None,
            gpu_type is not None,
            min_gpu_count is not None,
            min_gpu_memory is not None,
        ]):
            query = ListAvailableNodesQuery(
                available=available,
                gpu_type=gpu_type,
                min_gpu_count=min_gpu_count,
                min_gpu_memory=min_gpu_memory,
            )
        loop = asyncio.get_running_loop()
        return await loop.run_in_executor(None, self._client.list_nodes, query)

    async def get_balance_async(self) -> Dict[str, Any]:
        """Get account balance asynchronously."""
        loop = asyncio.get_running_loop()
        return await loop.run_in_executor(None, self._client.get_balance)

    async def list_usage_history_async(
        self, limit: int = 50, offset: int = 0
    ) -> Dict[str, Any]:
        """Get usage history for billing asynchronously."""
        loop = asyncio.get_running_loop()
        return await loop.run_in_executor(
            None, lambda: self._client.list_usage_history(limit, offset)
        )
