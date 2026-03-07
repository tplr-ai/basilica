"""
Deployment specification dataclass.

Immutable configuration for deployments extracted from decorator parameters.
"""
from dataclasses import dataclass
from typing import Any, Dict, List, Optional, TYPE_CHECKING

if TYPE_CHECKING:
    from .volume import Volume
    from basilica._basilica import HealthCheckConfig


@dataclass(frozen=True)
class DeploymentSpec:
    """Immutable specification for a deployment."""

    name: str
    image: str = "python:3.11-slim"
    port: int = 8000
    cpu: str = "500m"
    memory: str = "512Mi"
    gpu: Optional[str] = None
    gpu_count: Optional[int] = None
    gpu_models: Optional[List[str]] = None
    min_cuda_version: Optional[str] = None
    min_gpu_memory_gb: Optional[int] = None
    interconnect: Optional[str] = None
    geo: Optional[str] = None
    spot: Optional[bool] = None
    infiniband: Optional[bool] = None
    volumes: Optional[Dict[str, "Volume"]] = None
    env: Optional[Dict[str, str]] = None
    pip_packages: Optional[List[str]] = None
    replicas: int = 1
    ttl_seconds: Optional[int] = None
    public: bool = True
    timeout: int = 300
    health_check: Optional["HealthCheckConfig"] = None
