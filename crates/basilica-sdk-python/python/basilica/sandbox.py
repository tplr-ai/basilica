"""
Sandbox module for Basilica SDK

Provides Daytona-compatible API for running code in isolated sandboxes.

Usage:
    >>> from basilica.sandbox import Sandbox
    >>> 
    >>> # Create a Python sandbox
    >>> sandbox = Sandbox.create(language="python")
    >>> 
    >>> # Run code
    >>> result = sandbox.run("print('Hello, World!')")
    >>> print(result.stdout)
    Hello, World!
    >>> 
    >>> # Execute commands
    >>> result = sandbox.exec(["ls", "-la"])
    >>> print(result.stdout)
    >>> 
    >>> # File operations
    >>> sandbox.write_file("/workspace/app.py", "print('Hello')")
    >>> content = sandbox.read_file("/workspace/app.py")
    >>> 
    >>> # Cleanup
    >>> sandbox.delete()
"""

import os
import time
from dataclasses import dataclass, field
from enum import Enum
from typing import Any, Dict, List, Optional, Union

import requests


class SandboxState(Enum):
    """State of a sandbox."""

    CREATING = "Creating"
    INITIALIZING = "Initializing"
    READY = "Ready"
    EXECUTING = "Executing"
    SNAPSHOTTING = "Snapshotting"
    TERMINATING = "Terminating"
    TERMINATED = "Terminated"
    FAILED = "Failed"


class NetworkIsolation(Enum):
    """Network isolation mode."""

    NONE = "none"
    EGRESS = "egress"  # Allow outbound only
    FULL = "full"  # No network access


@dataclass
class GpuSpec:
    """GPU requirements for sandbox."""

    count: int
    model: List[str] = field(default_factory=list)
    min_cuda_version: Optional[str] = None
    min_gpu_memory_gb: Optional[int] = None


@dataclass
class ResourceSpec:
    """Resource requirements for sandbox."""

    cpu: str = "500m"
    memory: str = "512Mi"
    gpus: Optional[GpuSpec] = None


@dataclass
class ExecResult:
    """Result of executing a command in sandbox."""

    stdout: str
    stderr: str
    exit_code: int
    duration_ms: int = 0

    @property
    def success(self) -> bool:
        return self.exit_code == 0


@dataclass
class FileInfo:
    """Information about a file in the sandbox."""

    name: str
    path: str
    is_dir: bool
    size: int
    modified_at: Optional[str] = None


@dataclass
class Snapshot:
    """Sandbox snapshot."""

    snapshot_id: str
    sandbox_id: str
    name: Optional[str]
    created_at: str
    size_bytes: int


class SandboxError(Exception):
    """Base exception for sandbox errors."""

    pass


class SandboxNotFound(SandboxError):
    """Sandbox not found."""

    def __init__(self, sandbox_id: str):
        self.sandbox_id = sandbox_id
        super().__init__(f"Sandbox not found: {sandbox_id}")


class SandboxNotReady(SandboxError):
    """Sandbox is not ready for execution."""

    def __init__(self, sandbox_id: str, state: str):
        self.sandbox_id = sandbox_id
        self.state = state
        super().__init__(f"Sandbox {sandbox_id} is not ready (state: {state})")


class ExecutionError(SandboxError):
    """Command execution failed."""

    def __init__(self, message: str, exit_code: int, stderr: str):
        self.exit_code = exit_code
        self.stderr = stderr
        super().__init__(f"{message}: {stderr}")


class Sandbox:
    """
    A Basilica sandbox for running code in isolation.

    Sandboxes provide:
    - Isolated execution environments
    - File system access
    - Optional GPU support
    - Network isolation
    - Snapshot/restore functionality

    Example:
        >>> sandbox = Sandbox.create(language="python")
        >>> result = sandbox.run("print('Hello!')")
        >>> print(result.stdout)
        Hello!
        >>> sandbox.delete()
    """

    def __init__(
        self,
        sandbox_id: str,
        api_url: str,
        api_key: str,
        language: str = "python",
        state: str = "Creating",
    ):
        """Initialize a sandbox instance (use Sandbox.create() instead)."""
        self.sandbox_id = sandbox_id
        self.language = language
        self._state = state
        self._api_url = api_url.rstrip("/")
        self._api_key = api_key
        self._session = requests.Session()
        self._session.headers.update(
            {
                "Authorization": f"Bearer {api_key}",
                "Content-Type": "application/json",
            }
        )

    @property
    def state(self) -> SandboxState:
        """Current state of the sandbox."""
        try:
            return SandboxState(self._state)
        except ValueError:
            return SandboxState.FAILED

    @property
    def is_ready(self) -> bool:
        """Check if sandbox is ready for execution."""
        return self.state in (SandboxState.READY, SandboxState.EXECUTING)

    @classmethod
    def create(
        cls,
        language: str = "python",
        image: Optional[str] = None,
        cpu: str = "500m",
        memory: str = "512Mi",
        gpu_count: Optional[int] = None,
        gpu_models: Optional[List[str]] = None,
        env: Optional[Dict[str, str]] = None,
        timeout_seconds: int = 3600,
        idle_timeout_seconds: int = 600,
        auto_snapshot: bool = False,
        restore_from: Optional[str] = None,
        network_isolation: NetworkIsolation = NetworkIsolation.NONE,
        api_url: Optional[str] = None,
        api_key: Optional[str] = None,
        wait: bool = True,
        wait_timeout: int = 120,
    ) -> "Sandbox":
        """
        Create a new sandbox.

        Args:
            language: Programming language (python, javascript, bash, etc.)
            image: Custom container image (uses default for language if not specified)
            cpu: CPU allocation (e.g., "500m", "1", "2")
            memory: Memory allocation (e.g., "512Mi", "1Gi", "4Gi")
            gpu_count: Number of GPUs (optional)
            gpu_models: Acceptable GPU models (e.g., ["A100", "H100"])
            env: Environment variables
            timeout_seconds: Maximum lifetime in seconds (default: 1 hour)
            idle_timeout_seconds: Idle timeout (default: 10 minutes)
            auto_snapshot: Save snapshot on termination
            restore_from: Snapshot ID to restore from
            network_isolation: Network isolation mode
            api_url: Basilica API URL (defaults to BASILICA_API_URL env var)
            api_key: API key (defaults to BASILICA_API_TOKEN env var)
            wait: Wait for sandbox to be ready
            wait_timeout: Timeout for waiting (seconds)

        Returns:
            Sandbox: The created sandbox

        Raises:
            SandboxError: If creation fails
        """
        if api_url is None:
            api_url = os.environ.get("BASILICA_API_URL", "https://api.basilica.ai")
        if api_key is None:
            api_key = os.environ.get("BASILICA_API_TOKEN", "")
        if not api_key:
            raise SandboxError("API key is required. Set BASILICA_API_TOKEN env var.")

        # Build request
        request: Dict[str, Any] = {
            "language": language,
            "resources": {"cpu": cpu, "memory": memory},
            "env": [{"name": k, "value": v} for k, v in (env or {}).items()],
            "timeoutSeconds": timeout_seconds,
            "idleTimeoutSeconds": idle_timeout_seconds,
            "autoSnapshot": auto_snapshot,
            "networkIsolation": network_isolation.value,
        }

        if image:
            request["image"] = image
        if gpu_count:
            request["resources"]["gpus"] = {
                "count": gpu_count,
                "model": gpu_models or [],
            }
        if restore_from:
            request["restoreFrom"] = restore_from

        # Create sandbox
        url = f"{api_url}/api/v1/sandboxes"
        headers = {
            "Authorization": f"Bearer {api_key}",
            "Content-Type": "application/json",
        }

        try:
            response = requests.post(url, json=request, headers=headers, timeout=30)
            response.raise_for_status()
            data = response.json()
        except requests.RequestException as e:
            raise SandboxError(f"Failed to create sandbox: {e}") from e

        sandbox = cls(
            sandbox_id=data["sandboxId"],
            api_url=api_url,
            api_key=api_key,
            language=language,
            state=data.get("state", "Creating"),
        )

        if wait:
            sandbox.wait_until_ready(timeout=wait_timeout)

        return sandbox

    @classmethod
    def get(
        cls,
        sandbox_id: str,
        api_url: Optional[str] = None,
        api_key: Optional[str] = None,
    ) -> "Sandbox":
        """
        Get an existing sandbox by ID.

        Args:
            sandbox_id: The sandbox ID
            api_url: API URL
            api_key: API key

        Returns:
            Sandbox: The sandbox

        Raises:
            SandboxNotFound: If sandbox doesn't exist
        """
        if api_url is None:
            api_url = os.environ.get("BASILICA_API_URL", "https://api.basilica.ai")
        if api_key is None:
            api_key = os.environ.get("BASILICA_API_TOKEN", "")

        url = f"{api_url}/api/v1/sandboxes/{sandbox_id}"
        headers = {
            "Authorization": f"Bearer {api_key}",
            "Content-Type": "application/json",
        }

        try:
            response = requests.get(url, headers=headers, timeout=30)
            if response.status_code == 404:
                raise SandboxNotFound(sandbox_id)
            response.raise_for_status()
            data = response.json()
        except SandboxNotFound:
            raise
        except requests.RequestException as e:
            raise SandboxError(f"Failed to get sandbox: {e}") from e

        return cls(
            sandbox_id=sandbox_id,
            api_url=api_url,
            api_key=api_key,
            language=data.get("language", "python"),
            state=data.get("state", "Unknown"),
        )

    def refresh(self) -> None:
        """Refresh sandbox status from API."""
        url = f"{self._api_url}/api/v1/sandboxes/{self.sandbox_id}"
        try:
            response = self._session.get(url, timeout=30)
            if response.status_code == 404:
                raise SandboxNotFound(self.sandbox_id)
            response.raise_for_status()
            data = response.json()
            self._state = data.get("state", "Unknown")
        except SandboxNotFound:
            raise
        except requests.RequestException as e:
            raise SandboxError(f"Failed to refresh sandbox: {e}") from e

    def wait_until_ready(self, timeout: int = 120, poll_interval: float = 1.0) -> None:
        """
        Wait for sandbox to become ready.

        Args:
            timeout: Maximum time to wait in seconds
            poll_interval: Time between status checks

        Raises:
            SandboxError: If sandbox fails or times out
        """
        start = time.time()
        while time.time() - start < timeout:
            self.refresh()

            if self.state == SandboxState.READY:
                return
            if self.state == SandboxState.FAILED:
                raise SandboxError(f"Sandbox failed to start: {self.sandbox_id}")
            if self.state == SandboxState.TERMINATED:
                raise SandboxError(f"Sandbox was terminated: {self.sandbox_id}")

            time.sleep(poll_interval)

        raise SandboxError(f"Timeout waiting for sandbox to become ready")

    def run(
        self,
        code: str,
        entrypoint: Optional[str] = None,
        args: Optional[List[str]] = None,
        env: Optional[Dict[str, str]] = None,
        timeout_seconds: int = 300,
    ) -> ExecResult:
        """
        Run code in the sandbox.

        This is a convenience method that executes code using the sandbox's
        configured language runtime.

        Args:
            code: Code to execute
            entrypoint: Entry point file (optional)
            args: Additional arguments
            env: Environment variables for this execution
            timeout_seconds: Execution timeout

        Returns:
            ExecResult: The execution result

        Example:
            >>> result = sandbox.run("print('Hello!')")
            >>> print(result.stdout)
            Hello!
        """
        if not self.is_ready:
            self.refresh()
            if not self.is_ready:
                raise SandboxNotReady(self.sandbox_id, self._state)

        request = {
            "code": code,
            "args": args or [],
            "env": [{"name": k, "value": v} for k, v in (env or {}).items()],
            "timeoutSeconds": timeout_seconds,
        }
        if entrypoint:
            request["entrypoint"] = entrypoint

        url = f"{self._api_url}/api/v1/sandboxes/{self.sandbox_id}/run"
        try:
            response = self._session.post(url, json=request, timeout=timeout_seconds + 30)
            response.raise_for_status()
            data = response.json()
        except requests.RequestException as e:
            raise SandboxError(f"Failed to run code: {e}") from e

        return ExecResult(
            stdout=data.get("stdout", ""),
            stderr=data.get("stderr", ""),
            exit_code=data.get("exitCode", 0),
            duration_ms=data.get("durationMs", 0),
        )

    def exec(
        self,
        command: List[str],
        workdir: Optional[str] = None,
        stdin: Optional[str] = None,
        env: Optional[Dict[str, str]] = None,
        timeout_seconds: int = 300,
    ) -> ExecResult:
        """
        Execute a command in the sandbox.

        Args:
            command: Command and arguments
            workdir: Working directory
            stdin: Standard input
            env: Environment variables
            timeout_seconds: Timeout in seconds

        Returns:
            ExecResult: The execution result

        Example:
            >>> result = sandbox.exec(["ls", "-la", "/workspace"])
            >>> print(result.stdout)
        """
        if not self.is_ready:
            self.refresh()
            if not self.is_ready:
                raise SandboxNotReady(self.sandbox_id, self._state)

        request: Dict[str, Any] = {
            "command": command,
            "env": [{"name": k, "value": v} for k, v in (env or {}).items()],
            "timeoutSeconds": timeout_seconds,
        }
        if workdir:
            request["workdir"] = workdir
        if stdin:
            request["stdin"] = stdin

        url = f"{self._api_url}/api/v1/sandboxes/{self.sandbox_id}/exec"
        try:
            response = self._session.post(url, json=request, timeout=timeout_seconds + 30)
            response.raise_for_status()
            data = response.json()
        except requests.RequestException as e:
            raise SandboxError(f"Failed to execute command: {e}") from e

        return ExecResult(
            stdout=data.get("stdout", ""),
            stderr=data.get("stderr", ""),
            exit_code=data.get("exitCode", 0),
            duration_ms=data.get("durationMs", 0),
        )

    def read_file(self, path: str, encoding: str = "utf-8") -> str:
        """
        Read a file from the sandbox.

        Args:
            path: File path
            encoding: File encoding

        Returns:
            str: File contents

        Raises:
            SandboxError: If file cannot be read
        """
        request = {"path": path, "encoding": encoding}
        url = f"{self._api_url}/api/v1/sandboxes/{self.sandbox_id}/files/read"

        try:
            response = self._session.post(url, json=request, timeout=30)
            if response.status_code == 404:
                raise SandboxError(f"File not found: {path}")
            response.raise_for_status()
            data = response.json()
            return data.get("content", "")
        except SandboxError:
            raise
        except requests.RequestException as e:
            raise SandboxError(f"Failed to read file: {e}") from e

    def write_file(self, path: str, content: str, mode: Optional[str] = None) -> None:
        """
        Write a file to the sandbox.

        Args:
            path: File path
            content: File contents
            mode: File mode (e.g., "755")
        """
        request: Dict[str, Any] = {"path": path, "content": content}
        if mode:
            request["mode"] = mode

        url = f"{self._api_url}/api/v1/sandboxes/{self.sandbox_id}/files/write"
        try:
            response = self._session.post(url, json=request, timeout=30)
            response.raise_for_status()
        except requests.RequestException as e:
            raise SandboxError(f"Failed to write file: {e}") from e

    def list_files(self, path: str = "/workspace", recursive: bool = False) -> List[FileInfo]:
        """
        List files in the sandbox.

        Args:
            path: Directory path
            recursive: List recursively

        Returns:
            List[FileInfo]: List of files
        """
        request = {"path": path, "recursive": recursive}
        url = f"{self._api_url}/api/v1/sandboxes/{self.sandbox_id}/files/list"

        try:
            response = self._session.post(url, json=request, timeout=30)
            response.raise_for_status()
            data = response.json()
            return [
                FileInfo(
                    name=f.get("name", ""),
                    path=f.get("path", ""),
                    is_dir=f.get("isDir", False),
                    size=f.get("size", 0),
                    modified_at=f.get("modifiedAt"),
                )
                for f in data.get("files", [])
            ]
        except requests.RequestException as e:
            raise SandboxError(f"Failed to list files: {e}") from e

    def create_snapshot(self, name: Optional[str] = None) -> Snapshot:
        """
        Create a snapshot of the sandbox.

        Args:
            name: Optional snapshot name

        Returns:
            Snapshot: The created snapshot
        """
        request: Dict[str, Any] = {}
        if name:
            request["name"] = name

        url = f"{self._api_url}/api/v1/sandboxes/{self.sandbox_id}/snapshot"
        try:
            response = self._session.post(url, json=request, timeout=60)
            response.raise_for_status()
            data = response.json()
            return Snapshot(
                snapshot_id=data.get("snapshotId", ""),
                sandbox_id=self.sandbox_id,
                name=data.get("name"),
                created_at=data.get("createdAt", ""),
                size_bytes=data.get("sizeBytes", 0),
            )
        except requests.RequestException as e:
            raise SandboxError(f"Failed to create snapshot: {e}") from e

    def delete(self) -> Optional[str]:
        """
        Delete the sandbox.

        Returns:
            Optional[str]: Snapshot ID if auto-snapshot was enabled

        Note:
            After deletion, the sandbox object should not be used.
        """
        url = f"{self._api_url}/api/v1/sandboxes/{self.sandbox_id}"
        try:
            response = self._session.delete(url, timeout=30)
            response.raise_for_status()
            data = response.json()
            self._state = "Terminated"
            return data.get("snapshotId")
        except requests.RequestException as e:
            raise SandboxError(f"Failed to delete sandbox: {e}") from e

    def __enter__(self) -> "Sandbox":
        """Context manager entry."""
        if not self.is_ready:
            self.wait_until_ready()
        return self

    def __exit__(self, exc_type, exc_val, exc_tb) -> None:
        """Context manager exit - deletes the sandbox."""
        try:
            self.delete()
        except SandboxError:
            pass  # Ignore deletion errors

    def __repr__(self) -> str:
        return f"Sandbox(id={self.sandbox_id!r}, language={self.language!r}, state={self._state!r})"

