"""
Basilica AFINE SDK - Agent Framework for Interactive Network Environments.

A Python SDK for defining, deploying, and interacting with containerized
multi-turn reinforcement learning environments on the Basilica protocol.
"""

from .api_client import BasilicaAPIClient
from .client import Client, create
from .docker_manager import DockerManager
from .models import (
    ErrorResponse,
    HealthResponse,
    RentalSecretInfo,
    RPCRequest,
    RPCResponse,
)
from .service import Service, serve
from .state import StatePersistence

__version__ = "0.1.0"

__all__ = [
    "Service",
    "serve",
    "Client",
    "create",
    "BasilicaAPIClient",
    "DockerManager",
    "StatePersistence",
    "RPCRequest",
    "RPCResponse",
    "ErrorResponse",
    "HealthResponse",
    "RentalSecretInfo",
]
