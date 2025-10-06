"""
Pydantic models and type definitions for Basilica AFINE SDK.
"""

from typing import Any, Dict, List
from pydantic import BaseModel, Field


class RPCRequest(BaseModel):
    """Request model for RPC calls."""
    args: List[Any] = Field(default_factory=list, description="Positional arguments")
    kwargs: Dict[str, Any] = Field(default_factory=dict, description="Keyword arguments")


class RPCResponse(BaseModel):
    """Response model for successful RPC calls."""
    result: Any = Field(description="Result of the RPC call")


class ErrorResponse(BaseModel):
    """Response model for RPC errors."""
    error: str = Field(description="Error type")
    detail: str = Field(description="Detailed error message")


class RentalSecretInfo(BaseModel):
    """Information about a rental including the generated secret."""
    rental_id: str = Field(description="Unique rental identifier")
    endpoint_url: str = Field(description="HTTP endpoint URL for the service")
    rental_secret: str = Field(description="Shared secret for authentication")
    ssh_credentials: str | None = Field(default=None, description="SSH credentials if available")
    container_info: Dict[str, Any] = Field(default_factory=dict, description="Container metadata")


class HealthResponse(BaseModel):
    """Health check response from service."""
    status: str = Field(description="Health status")
    service: str = Field(description="Service name")
    version: str = Field(description="Service version")
