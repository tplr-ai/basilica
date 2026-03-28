"""Sandbox SDK module.

Provides control-plane operations (create, list, get, delete) via the API,
and data-plane connectivity directly to sandbox domains.

Architecture:
    Control plane: SDK -> basilica-api -> BasilicaSandbox CRD
    Data plane:    SDK -> <sandbox-id>.sandboxes.basilica.ai (direct)

H1: The API is control-plane only. No exec/ws/file relay through the API.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any, Dict, List, Optional


@dataclass
class SandboxEnvVar:
    """An environment variable to set in the sandbox."""
    name: str
    value: str


@dataclass
class CreateSandboxRequest:
    """Request to create a new sandbox."""
    image: str
    cpu: Optional[str] = None
    memory: Optional[str] = None
    env: List[SandboxEnvVar] = field(default_factory=list)
    ttl_seconds: Optional[int] = None

    def to_dict(self) -> Dict[str, Any]:
        d: Dict[str, Any] = {"image": self.image}
        if self.cpu is not None:
            d["cpu"] = self.cpu
        if self.memory is not None:
            d["memory"] = self.memory
        if self.env:
            d["env"] = [{"name": e.name, "value": e.value} for e in self.env]
        if self.ttl_seconds is not None:
            d["ttlSeconds"] = self.ttl_seconds
        return d


@dataclass
class Sandbox:
    """A sandbox handle returned after creation.

    Provides the sandbox domain for direct data-plane access.
    Data-plane traffic goes directly to the sandbox domain, NOT through the API.
    """
    sandbox_id: str
    domain: str
    status: str

    @property
    def data_plane_url(self) -> str:
        """Base URL for data-plane operations."""
        return f"https://{self.domain}"

    @property
    def ws_url(self) -> str:
        """WebSocket URL for terminal access."""
        return f"wss://{self.domain}/ws"

    @property
    def exec_url(self) -> str:
        """Exec endpoint URL."""
        return f"https://{self.domain}/exec"

    @classmethod
    def from_response(cls, data: Dict[str, Any]) -> "Sandbox":
        return cls(
            sandbox_id=data["sandboxId"],
            domain=data["domain"],
            status=data.get("status", "unknown"),
        )


@dataclass
class SandboxSummary:
    """Summary of a sandbox from list operations."""
    sandbox_id: str
    image: str
    status: str
    domain: Optional[str] = None
    created_at: Optional[str] = None

    @classmethod
    def from_response(cls, data: Dict[str, Any]) -> "SandboxSummary":
        return cls(
            sandbox_id=data["sandboxId"],
            image=data["image"],
            status=data["status"],
            domain=data.get("domain"),
            created_at=data.get("createdAt"),
        )


@dataclass
class SandboxDetail:
    """Detailed sandbox info."""
    sandbox_id: str
    image: str
    cpu: str
    memory: str
    status: str
    domain: Optional[str] = None
    created_at: Optional[str] = None

    @classmethod
    def from_response(cls, data: Dict[str, Any]) -> "SandboxDetail":
        return cls(
            sandbox_id=data["sandboxId"],
            image=data["image"],
            cpu=data["cpu"],
            memory=data["memory"],
            status=data["status"],
            domain=data.get("domain"),
            created_at=data.get("createdAt"),
        )


class SandboxClient:
    """Client for sandbox control-plane operations.

    Uses the basilica-api for CRD lifecycle (create/list/get/delete).
    Data-plane operations go directly to sandbox domains.
    """

    def __init__(self, api_client: Any):
        """Initialize with an authenticated API client.

        Args:
            api_client: An authenticated BasilicaClient instance.
        """
        self._client = api_client

    def create(
        self,
        image: str,
        *,
        cpu: Optional[str] = None,
        memory: Optional[str] = None,
        env: Optional[List[SandboxEnvVar]] = None,
        ttl_seconds: Optional[int] = None,
    ) -> Sandbox:
        """Create a new sandbox.

        Args:
            image: Container image (must be in the server's allowlist).
            cpu: CPU resources (default: "1").
            memory: Memory resources (default: "2Gi").
            env: Environment variables to set.
            ttl_seconds: Optional TTL in seconds.

        Returns:
            A Sandbox handle with the domain for direct data-plane access.
        """
        request = CreateSandboxRequest(
            image=image,
            cpu=cpu,
            memory=memory,
            env=env or [],
            ttl_seconds=ttl_seconds,
        )
        response = self._client._post("/sandboxes", json=request.to_dict())
        return Sandbox.from_response(response)

    def list(self) -> List[SandboxSummary]:
        """List all sandboxes for the authenticated user."""
        response = self._client._get("/sandboxes")
        return [
            SandboxSummary.from_response(s)
            for s in response.get("sandboxes", [])
        ]

    def get(self, sandbox_id: str) -> SandboxDetail:
        """Get details of a specific sandbox."""
        response = self._client._get(f"/sandboxes/{sandbox_id}")
        return SandboxDetail.from_response(response)

    def delete(self, sandbox_id: str) -> None:
        """Delete a sandbox."""
        self._client._delete(f"/sandboxes/{sandbox_id}")
