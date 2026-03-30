"""Sandbox SDK module.

Provides control-plane operations (create, list, get, delete) via the API,
and data-plane connectivity directly to sandbox domains.

Architecture:
    Control plane: SDK -> basilica-api -> BasilicaSandbox CRD
    Data plane:    SDK -> <sandbox-id>.sandboxes.basilica.ai (direct)

H1: The API is control-plane only. No exec/ws/file relay through the API.
"""

from __future__ import annotations

import json
from dataclasses import dataclass, field
from typing import Any, Dict, List, Optional

try:
    import urllib.request
    import urllib.error

    _HAS_URLLIB = True
except ImportError:
    _HAS_URLLIB = False


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
    exec_agent_secret: Optional[str] = None
    _data_plane_base_url: Optional[str] = field(default=None, repr=False)

    def with_data_plane_url(self, url: str) -> "Sandbox":
        """Override the data-plane base URL (e.g. for K3d port-forward testing)."""
        self._data_plane_base_url = url
        return self

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

    def exec(self, command: List[str]) -> Dict[str, Any]:
        """Execute a command in the sandbox via data-plane.

        Args:
            command: Command and arguments to execute.

        Returns:
            dict with stdout, stderr, exitCode.
        """
        return self._data_plane_post("/exec", {"command": command})

    def run(self, code: str, language: Optional[str] = None) -> Dict[str, Any]:
        """Run code in the sandbox via data-plane.

        Args:
            code: Source code to execute.
            language: Optional language hint.

        Returns:
            dict with stdout, stderr, exitCode.
        """
        body: Dict[str, Any] = {"code": code}
        if language:
            body["language"] = language
        return self._data_plane_post("/run", body)

    @property
    def files(self) -> "SandboxFiles":
        """Get file operations handle."""
        return SandboxFiles(self)

    def _data_plane_post(self, path: str, body: Dict[str, Any]) -> Dict[str, Any]:
        """Make an authenticated POST to the sandbox data-plane."""
        if not self.exec_agent_secret:
            raise RuntimeError(
                "exec_agent_secret not available — sandbox was not created through this SDK"
            )

        base = self._data_plane_base_url or f"https://{self.domain}"
        url = f"{base}{path}"
        data = json.dumps(body).encode("utf-8")
        req = urllib.request.Request(
            url,
            data=data,
            headers={
                "Content-Type": "application/json",
                "Authorization": f"Bearer {self.exec_agent_secret}",
            },
            method="POST",
        )
        try:
            with urllib.request.urlopen(req) as resp:
                return json.loads(resp.read().decode("utf-8"))
        except urllib.error.HTTPError as e:
            body_text = e.read().decode("utf-8") if e.fp else ""
            raise RuntimeError(
                f"Data-plane request failed ({e.code}): {body_text}"
            ) from e

    @classmethod
    def from_response(cls, data: Dict[str, Any]) -> "Sandbox":
        return cls(
            sandbox_id=data["sandboxId"],
            domain=data["domain"],
            status=data.get("status", "unknown"),
            exec_agent_secret=data.get("execAgentSecret"),
        )


class SandboxFiles:
    """File operations on a sandbox."""

    def __init__(self, sandbox: Sandbox):
        self._sandbox = sandbox

    def write(self, path: str, content: str) -> Dict[str, Any]:
        """Write a file to the sandbox."""
        return self._sandbox._data_plane_post(
            "/files/write", {"path": path, "content": content}
        )

    def read(self, path: str) -> Dict[str, Any]:
        """Read a file from the sandbox."""
        return self._sandbox._data_plane_post("/files/read", {"path": path})

    def list(self, path: str) -> Dict[str, Any]:
        """List files in a directory in the sandbox."""
        return self._sandbox._data_plane_post("/files/list", {"path": path})


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
    Data-plane operations go directly to sandbox domains via the Sandbox handle.

    G5: Uses typed PyO3 methods on the BasilicaClient, not _post/_get/_delete.
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
        env_tuples = [(e.name, e.value) for e in (env or [])]
        response = self._client.create_sandbox(
            image=image,
            cpu=cpu,
            memory=memory,
            env=env_tuples if env_tuples else None,
            ttl_seconds=ttl_seconds,
        )
        return Sandbox.from_response(response)

    def list(self) -> List[SandboxSummary]:
        """List all sandboxes for the authenticated user."""
        response = self._client.list_sandboxes()
        return [
            SandboxSummary.from_response(s)
            for s in response.get("sandboxes", [])
        ]

    def get(self, sandbox_id: str) -> SandboxDetail:
        """Get details of a specific sandbox."""
        response = self._client.get_sandbox(sandbox_id)
        return SandboxDetail.from_response(response)

    def delete(self, sandbox_id: str) -> None:
        """Delete a sandbox."""
        self._client.delete_sandbox(sandbox_id)
