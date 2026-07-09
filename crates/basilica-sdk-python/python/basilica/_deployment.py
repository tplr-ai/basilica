"""
Deployment Facade

This module provides a high-level, user-friendly interface for managing deployments.
The Deployment class wraps the low-level API responses and provides convenient
methods for common operations.

Example:
    >>> deployment = client.deploy("my-app", source="app.py")
    >>> print(deployment.url)           # Public URL
    >>> print(deployment.logs())        # Get logs
    >>> deployment.delete()             # Clean up

Async Support:
    All methods have async variants with `_async` suffix:
    >>> status = await deployment.wait_until_ready_async()
    >>> await deployment.delete_async()
"""

import asyncio
import socket
import ssl
import time
import urllib.error
import urllib.request
from dataclasses import dataclass, field
from typing import TYPE_CHECKING, Any, Callable, Dict, List, Optional
from urllib.parse import urlparse

from basilica.exceptions import DeploymentFailed, DeploymentTimeout


# HTTP readiness verification settings
# Node scaling can take up to 10 minutes, so we need generous retry settings
HTTP_READY_TIMEOUT = 10.0  # seconds per attempt

if TYPE_CHECKING:
    from . import BasilicaClient


def _format_phase_message(phase: Optional[str]) -> str:
    """Format human-readable phase message for progress output."""
    messages = {
        "pending": "Waiting for scheduler...",
        "scheduling": "Finding suitable node...",
        "pulling": "Pulling container image...",
        "initializing": "Running init containers...",
        "storage_sync": "Syncing storage volume...",
        "starting": "Starting application...",
        "health_check": "Running health checks...",
        "ready": "Deployment ready!",
        "degraded": "Deployment degraded",
        "failed": "Deployment failed",
        "terminating": "Terminating...",
    }
    return messages.get(phase or "pending", f"Phase: {phase}")


@dataclass
class ProgressInfo:
    """
    Progress information for storage synchronization or other long-running operations.

    Attributes:
        bytes_synced: Number of bytes synchronized so far (None if unknown)
        bytes_total: Total bytes to synchronize (None if unknown)
        percentage: Completion percentage 0-100 (None if unknown)
        current_step: Human-readable description of current step
        started_at: ISO 8601 timestamp when the operation started
        elapsed_seconds: Seconds elapsed since the operation started
    """

    bytes_synced: Optional[int] = None
    bytes_total: Optional[int] = None
    percentage: Optional[float] = None
    current_step: str = ""
    started_at: str = ""
    elapsed_seconds: int = 0


@dataclass
class ContainerStatusInfo:
    """
    Per-container runtime snapshot surfaced from the operator (Phase 4: UD Status Honesty).

    Attributes:
        pod_name: Pod hosting the container
        container_name: Container name within the pod
        state: Lifecycle state (waiting, running, terminated)
        reason: Optional reason string from kubelet (e.g. ImagePullBackOff)
        message: Optional human-readable diagnostic message
        restart_count: Number of restarts observed for this container
    """

    pod_name: str
    container_name: str
    state: str
    restart_count: int = 0
    reason: Optional[str] = None
    message: Optional[str] = None


@dataclass
class DeploymentStatus:
    """
    Represents the current status of a deployment.

    Attributes:
        state: Current state (Pending, Active, Running, Failed, Terminating)
        replicas_ready: Number of replicas that are ready
        replicas_desired: Total number of desired replicas
        message: Optional status message or error description
        phase: Granular deployment phase (scheduling, pulling, starting, ready, etc.)
        progress: Progress information for long-running operations like storage sync
        container_statuses: Per-container snapshots from the operator (Phase 4)
        phase_progress: Sum of container `restartCount` across every pod (Phase 4)
    """

    state: str
    replicas_ready: int
    replicas_desired: int
    message: Optional[str] = None
    phase: Optional[str] = None
    progress: Optional[ProgressInfo] = None
    container_statuses: List[ContainerStatusInfo] = field(default_factory=list)
    phase_progress: int = 0

    @property
    def is_ready(self) -> bool:
        """Check if the deployment is fully ready."""
        return (
            self.state in ("Active", "Running")
            and self.replicas_ready == self.replicas_desired
            and self.replicas_ready > 0
        )

    @property
    def is_failed(self) -> bool:
        """Check if the deployment has failed."""
        return self.state == "Failed" or self.phase == "failed"

    @property
    def is_pending(self) -> bool:
        """Check if the deployment is still starting."""
        return self.state in ("Pending", "Provisioning") or self.phase in (
            "pending",
            "scheduling",
            "pulling",
            "initializing",
            "storage_sync",
            "starting",
            "health_check",
        )


class Deployment:
    """
    A facade for managing a Basilica deployment.

    This class provides a convenient, object-oriented interface for working
    with deployments. It wraps the low-level API client and provides methods
    for common operations like getting logs, checking status, and cleanup.

    Attributes:
        name: The deployment instance name
        url: The public URL for accessing the deployment
        namespace: The Kubernetes namespace
        user_id: The owner's user ID
        created_at: Timestamp when the deployment was created

    Example:
        >>> # Create a deployment (via client.deploy())
        >>> deployment = client.deploy("my-api", source="api.py", port=8000)

        >>> # Access deployment info
        >>> print(f"Live at: {deployment.url}")

        >>> # Get logs
        >>> print(deployment.logs(tail=50))

        >>> # Check current status
        >>> status = deployment.status()
        >>> print(f"State: {status.state}, Ready: {status.replicas_ready}/{status.replicas_desired}")

        >>> # Clean up
        >>> deployment.delete()
    """

    def __init__(
        self,
        client: "BasilicaClient",
        instance_name: str,
        url: str,
        namespace: str,
        user_id: str,
        state: str,
        created_at: str,
        replicas_ready: int = 0,
        replicas_desired: int = 1,
        updated_at: Optional[str] = None,
        friendly_name: str = "",
        image: str = "",
        phase: Optional[str] = None,
        message: Optional[str] = None,
        share_token: Optional[str] = None,
        share_url: Optional[str] = None,
        public: bool = True,
        public_metadata: bool = False,
        distributed: Optional[Dict[str, Any]] = None,
    ):
        """
        Initialize a Deployment instance.

        Note: Users should not create Deployment objects directly.
        Use client.deploy() or client.get_deployment() instead.

        Issues #453 / #454: prior to this fix the wrapper dropped
        `image`, `phase`, `distributed` (and several other fields) that
        the PyO3 layer exposes. The Python facade `_coerce_to_dict`
        consequently saw a 4-key dict on the wrapper path
        (`client.get(name)`) even after PR #451 fixed the PyO3 layer,
        so every `DistributedTraining.world` / `.ranks` / `.bench`
        read returned zeros. The wrapper now preserves the full
        attribute surface the binding exposes.
        """
        self._client = client
        self._name = instance_name
        self._friendly_name = friendly_name
        self._url = url
        self._namespace = namespace
        self._user_id = user_id
        self._state = state
        self._created_at = created_at
        self._updated_at = updated_at
        self._replicas_ready = replicas_ready
        self._replicas_desired = replicas_desired
        self._image = image
        self._phase = phase
        self._message = message
        self._share_token = share_token
        self._share_url = share_url
        self._public = public
        self._public_metadata = public_metadata
        self._distributed = distributed

    @property
    def name(self) -> str:
        """The deployment instance name (stable UUID identifier used in URLs and K8s)."""
        return self._name

    @property
    def friendly_name(self) -> str:
        """The user-supplied display name for the deployment."""
        return self._friendly_name

    @property
    def instance_name(self) -> str:
        """
        The deployment instance name (alias of `name`, matches the PyO3
        binding's snake_case attribute).

        Issues #453 / #454: `_coerce_to_dict` walks `obj.instance_name`
        (the binding-side name) so the wrapper must expose it for the
        camelCase `instanceName` key to survive the round-trip.
        """
        return self._name

    @property
    def url(self) -> str:
        """
        The public URL for accessing the deployment.

        Example:
            >>> print(deployment.url)
            'https://my-app.deployments.basilica.ai'
        """
        return self._url

    @property
    def namespace(self) -> str:
        """The Kubernetes namespace (e.g., 'u-userid123')."""
        return self._namespace

    @property
    def user_id(self) -> str:
        """The owner's user ID."""
        return self._user_id

    @property
    def created_at(self) -> str:
        """Timestamp when the deployment was created (ISO 8601 format)."""
        return self._created_at

    @property
    def state(self) -> str:
        """
        The last known deployment state.

        Note: This may be stale. Call status() for the latest state.
        """
        return self._state

    @property
    def updated_at(self) -> Optional[str]:
        """Timestamp when the deployment was last updated (ISO 8601)."""
        return self._updated_at

    @property
    def image(self) -> str:
        """Container image reference for this deployment."""
        return self._image

    @property
    def phase(self) -> Optional[str]:
        """Granular deployment phase (scheduling, pulling, ready, etc.)."""
        return self._phase

    @property
    def message(self) -> Optional[str]:
        """Human-readable status / error message, if any."""
        return self._message

    @property
    def share_token(self) -> Optional[str]:
        """Share token for private deployments (only set on creation)."""
        return self._share_token

    @property
    def share_url(self) -> Optional[str]:
        """Shareable URL with token query parameter for private deployments."""
        return self._share_url

    @property
    def public(self) -> bool:
        """Whether this deployment is publicly accessible without a share token."""
        return self._public

    @property
    def public_metadata(self) -> bool:
        """Whether deployment metadata is publicly readable."""
        return self._public_metadata

    @property
    def distributed(self) -> Optional[Dict[str, Any]]:
        """
        Operator-emitted `status.distributed` block for distributed-training
        UDs. `None` for single-replica deployments.

        The dict shape is the operator's camelCase wire format
        (`worldSize`, `ranks`, `bench`, `transport`). Researchers
        normally consume it via the `DistributedTraining` facade
        (`client.deploy_distributed(...)`) rather than reading this
        directly.

        Issues #453 / #454: this attribute was dropped by the
        `Deployment` wrapper before this fix, even though PR #451
        landed it on the PyO3 binding.
        """
        return self._distributed

    def _parse_status_response(self, response) -> DeploymentStatus:
        """Parse API response into DeploymentStatus and update cached state."""
        self._state = response.state
        self._replicas_ready = response.replicas.ready
        self._replicas_desired = response.replicas.desired

        progress = None
        if hasattr(response, "progress") and response.progress:
            progress = ProgressInfo(
                bytes_synced=getattr(response.progress, "bytes_synced", None),
                bytes_total=getattr(response.progress, "bytes_total", None),
                percentage=getattr(response.progress, "percentage", None),
                current_step=getattr(response.progress, "current_step", ""),
                started_at=getattr(response.progress, "started_at", ""),
                elapsed_seconds=getattr(response.progress, "elapsed_seconds", 0),
            )

        container_statuses = [
            ContainerStatusInfo(
                pod_name=getattr(snap, "pod_name", ""),
                container_name=getattr(snap, "container_name", ""),
                state=getattr(snap, "state", ""),
                restart_count=getattr(snap, "restart_count", 0),
                reason=getattr(snap, "reason", None),
                message=getattr(snap, "message", None),
            )
            for snap in getattr(response, "container_statuses", []) or []
        ]

        return DeploymentStatus(
            state=response.state,
            replicas_ready=response.replicas.ready,
            replicas_desired=response.replicas.desired,
            message=None,
            phase=getattr(response, "phase", None),
            progress=progress,
            container_statuses=container_statuses,
            phase_progress=getattr(response, "phase_progress", 0),
        )

    def status(self) -> DeploymentStatus:
        """
        Get the current deployment status from the API.

        Returns:
            DeploymentStatus with current state, replica counts, phase, and progress

        Raises:
            DeploymentNotFound: If the deployment no longer exists
            NetworkError: If the API is unreachable

        Example:
            >>> status = deployment.status()
            >>> if status.is_ready:
            ...     print("Deployment is healthy!")
            >>> elif status.is_failed:
            ...     print(f"Deployment failed: {status.message}")
            >>> elif status.phase:
            ...     print(f"Phase: {status.phase}")
            ...     if status.progress:
            ...         print(f"Progress: {status.progress.percentage}%")
        """
        response = self._client.get_deployment(self._name)
        return self._parse_status_response(response)

    def logs(self, tail: Optional[int] = None, follow: bool = False) -> str:
        """
        Get deployment logs.

        Args:
            tail: Number of lines from the end to return.
                  If None, returns all available logs.
            follow: If True, streams logs continuously (blocking).
                   Note: follow mode may not be fully supported yet.

        Returns:
            Log content as a string

        Raises:
            DeploymentNotFound: If the deployment doesn't exist
            NetworkError: If the API is unreachable

        Example:
            >>> # Get last 100 lines
            >>> recent_logs = deployment.logs(tail=100)

            >>> # Get all logs
            >>> all_logs = deployment.logs()
        """
        return self._client.get_deployment_logs(self._name, follow=follow, tail=tail)

    def wait_until_ready(
        self,
        timeout: int = 300,
        poll_interval: int = 5,
        raise_on_failure: bool = True,
        on_progress: Optional[Callable[[DeploymentStatus], None]] = None,
        silent: bool = False,
    ) -> DeploymentStatus:
        """
        Wait for the deployment to become ready.

        Polls the deployment status until it reaches a ready state,
        fails, or times out. By default, prints progress to stdout.
        Use silent=True to suppress output, or provide a custom on_progress callback.

        After the deployment is marked ready, also verifies DNS resolution and
        HTTP endpoint responsiveness before returning.

        Args:
            timeout: Maximum seconds to wait (default: 300)
            poll_interval: Seconds between status checks (default: 5)
            raise_on_failure: If True, raises DeploymentFailed on failure state
            on_progress: Optional callback invoked on each status poll.
                         Receives the current DeploymentStatus as argument.
                         If None and silent=False, uses default progress output.
            silent: If True, suppresses default progress output (default: False)

        Returns:
            Final DeploymentStatus when ready or failed

        Raises:
            DeploymentTimeout: If deployment doesn't become ready within timeout
            DeploymentFailed: If deployment enters Failed state (when raise_on_failure=True)
            DeploymentNotFound: If deployment is deleted during wait

        Example:
            >>> # Default: shows progress output
            >>> status = deployment.wait_until_ready(timeout=120)
            >>>
            >>> # Silent mode: no output
            >>> status = deployment.wait_until_ready(timeout=120, silent=True)
            >>>
            >>> # Custom callback
            >>> def show_progress(status):
            ...     print(f"Phase: {status.phase}")
            >>> status = deployment.wait_until_ready(on_progress=show_progress)
        """
        elapsed = 0
        last_status = None
        last_phase = None

        callback = on_progress
        if callback is None and not silent:
            callback = self._default_progress_callback

        while elapsed < timeout:
            last_status = self.status()

            if callback and last_status.phase != last_phase:
                callback(last_status)
                last_phase = last_status.phase

            if last_status.is_ready:
                if self._url:
                    hostname = urlparse(self._url).hostname
                    if hostname:
                        # Verify DNS resolution
                        if not self._is_dns_resolvable(hostname):
                            time.sleep(poll_interval)
                            elapsed += poll_interval
                            continue
                        # Verify HTTP endpoint is responding
                        if not self._is_http_ready(self._url):
                            time.sleep(poll_interval)
                            elapsed += poll_interval
                            continue
                if callback and not on_progress and not silent:
                    print(f"[{self._name}] Deployment ready!")
                return last_status

            if last_status.is_failed and raise_on_failure:
                raise DeploymentFailed(
                    instance_name=self._name,
                    reason=last_status.message,
                )

            time.sleep(poll_interval)
            elapsed += poll_interval

        raise DeploymentTimeout(
            instance_name=self._name,
            timeout_seconds=timeout,
            last_state=last_status.state if last_status else "Unknown",
            replicas_ready=last_status.replicas_ready if last_status else 0,
            replicas_desired=last_status.replicas_desired if last_status else 1,
        )

    def _default_progress_callback(self, status: DeploymentStatus) -> None:
        """Default progress callback that prints human-readable status."""
        phase_msg = _format_phase_message(status.phase)
        replica_info = f"{status.replicas_ready}/{status.replicas_desired}"
        print(f"[{self._name}] {phase_msg} (replicas: {replica_info})")

        if status.phase == "storage_sync" and status.progress:
            if status.progress.percentage is not None:
                print(f"  Storage sync: {status.progress.percentage:.1f}%")

    def _is_dns_resolvable(self, hostname: str) -> bool:
        """Check if hostname resolves to an IP address."""
        try:
            socket.gethostbyname(hostname)
            return True
        except socket.gaierror:
            return False

    async def _is_dns_resolvable_async(self, hostname: str) -> bool:
        """Check if hostname resolves to an IP address asynchronously."""
        loop = asyncio.get_running_loop()
        try:
            await loop.getaddrinfo(hostname, None)
            return True
        except socket.gaierror:
            return False

    def _is_http_ready(self, url: str, timeout: float = HTTP_READY_TIMEOUT) -> bool:
        """Check if HTTP endpoint is responding."""
        try:
            # Create SSL context that doesn't verify certificates for health checks
            ctx = ssl.create_default_context()
            ctx.check_hostname = False
            ctx.verify_mode = ssl.CERT_NONE

            req = urllib.request.Request(url, method="HEAD")
            with urllib.request.urlopen(req, timeout=timeout, context=ctx) as resp:
                # Any response (even 4xx/5xx) means the server is up
                return True
        except urllib.error.HTTPError:
            # HTTP errors (4xx, 5xx) mean server is responding
            return True
        except Exception:
            return False

    async def _is_http_ready_async(self, url: str, timeout: float = HTTP_READY_TIMEOUT) -> bool:
        """Check if HTTP endpoint is responding asynchronously."""
        loop = asyncio.get_running_loop()
        return await loop.run_in_executor(None, self._is_http_ready, url, timeout)


    async def status_async(self) -> DeploymentStatus:
        """
        Get the current deployment status from the API asynchronously.

        Returns:
            DeploymentStatus with current state, replica counts, phase, and progress

        Raises:
            DeploymentNotFound: If the deployment no longer exists
            NetworkError: If the API is unreachable

        Example:
            >>> status = await deployment.status_async()
            >>> if status.is_ready:
            ...     print("Deployment is healthy!")
        """
        loop = asyncio.get_running_loop()
        response = await loop.run_in_executor(
            None, self._client.get_deployment, self._name
        )
        return self._parse_status_response(response)

    async def wait_until_ready_async(
        self,
        timeout: int = 300,
        poll_interval: int = 5,
        raise_on_failure: bool = True,
        on_progress: Optional[Callable[[DeploymentStatus], None]] = None,
        silent: bool = False,
    ) -> DeploymentStatus:
        """
        Wait for the deployment to become ready asynchronously.

        This is the async version of wait_until_ready(). It uses asyncio.sleep()
        instead of time.sleep(), allowing other coroutines to run while waiting.

        After the deployment is marked ready, also verifies DNS resolution and
        HTTP endpoint responsiveness before returning.

        Args:
            timeout: Maximum seconds to wait (default: 300)
            poll_interval: Seconds between status checks (default: 5)
            raise_on_failure: If True, raises DeploymentFailed on failure state
            on_progress: Optional callback invoked on each phase change.
            silent: If True, suppresses default progress output (default: False)

        Returns:
            Final DeploymentStatus when ready or failed

        Raises:
            DeploymentTimeout: If deployment doesn't become ready within timeout
            DeploymentFailed: If deployment enters Failed state (when raise_on_failure=True)

        Example:
            >>> # Wait for multiple deployments concurrently
            >>> async def deploy_many():
            ...     deployments = [d1, d2, d3]
            ...     tasks = [d.wait_until_ready_async() for d in deployments]
            ...     await asyncio.gather(*tasks)
        """
        elapsed = 0
        last_status = None
        last_phase = None

        callback = on_progress
        if callback is None and not silent:
            callback = self._default_progress_callback

        while elapsed < timeout:
            last_status = await self.status_async()

            if callback and last_status.phase != last_phase:
                callback(last_status)
                last_phase = last_status.phase

            if last_status.is_ready:
                if self._url:
                    hostname = urlparse(self._url).hostname
                    if hostname:
                        # Verify DNS resolution
                        if not await self._is_dns_resolvable_async(hostname):
                            await asyncio.sleep(poll_interval)
                            elapsed += poll_interval
                            continue
                        # Verify HTTP endpoint is responding
                        if not await self._is_http_ready_async(self._url):
                            await asyncio.sleep(poll_interval)
                            elapsed += poll_interval
                            continue
                if callback and not on_progress and not silent:
                    print(f"[{self._name}] Deployment ready!")
                return last_status

            if last_status.is_failed and raise_on_failure:
                raise DeploymentFailed(
                    instance_name=self._name,
                    reason=last_status.message,
                )

            await asyncio.sleep(poll_interval)
            elapsed += poll_interval

        raise DeploymentTimeout(
            instance_name=self._name,
            timeout_seconds=timeout,
            last_state=last_status.state if last_status else "Unknown",
            replicas_ready=last_status.replicas_ready if last_status else 0,
            replicas_desired=last_status.replicas_desired if last_status else 1,
        )

    async def refresh_async(self) -> "Deployment":
        """
        Refresh the deployment data from the API asynchronously.

        Returns:
            Self, for method chaining

        Example:
            >>> await deployment.refresh_async()
            >>> print(f"Current state: {deployment.state}")
        """
        loop = asyncio.get_running_loop()
        response = await loop.run_in_executor(
            None, self._client.get_deployment, self._name
        )
        self._copy_from_response(response)
        return self

    async def delete_async(self) -> None:
        """
        Delete the deployment asynchronously.

        Raises:
            DeploymentNotFound: If the deployment doesn't exist
            NetworkError: If the API is unreachable

        Example:
            >>> await deployment.delete_async()
        """
        loop = asyncio.get_running_loop()
        await loop.run_in_executor(
            None, self._client.delete_deployment, self._name
        )
        self._state = "Deleted"

    async def logs_async(self, tail: Optional[int] = None, follow: bool = False) -> str:
        """
        Get deployment logs asynchronously.

        Args:
            tail: Number of lines from the end to return.
            follow: If True, streams logs continuously (blocking).

        Returns:
            Log content as a string

        Example:
            >>> logs = await deployment.logs_async(tail=100)
        """
        loop = asyncio.get_running_loop()
        return await loop.run_in_executor(
            None,
            lambda: self._client.get_deployment_logs(self._name, follow=follow, tail=tail)
        )

    def delete(self) -> None:
        """
        Delete the deployment.

        This permanently removes the deployment and all associated resources.
        The operation is asynchronous - the deployment may take a few seconds
        to fully terminate.

        Raises:
            DeploymentNotFound: If the deployment doesn't exist
            NetworkError: If the API is unreachable

        Example:
            >>> deployment.delete()
            >>> print(f"Deleted deployment: {deployment.name}")
        """
        self._client.delete_deployment(self._name)
        self._state = "Deleted"

    def refresh(self) -> "Deployment":
        """
        Refresh the deployment data from the API.

        Updates all cached properties with the latest values from the server.

        Returns:
            Self, for method chaining

        Example:
            >>> deployment.refresh()
            >>> print(f"Current state: {deployment.state}")
        """
        response = self._client.get_deployment(self._name)
        self._copy_from_response(response)
        return self

    def _copy_from_response(self, response) -> None:
        """Copy mutable fields from a fresh PyO3 response onto this wrapper.

        Shared between `refresh` and `refresh_async` so the two stay in lock
        step. Keeps `image`, `phase`, `distributed`, etc. fresh -- those
        fields drove issues #453 / #454 when the wrapper held stale (or
        missing) copies.
        """
        self._url = response.url
        self._state = response.state
        self._replicas_ready = response.replicas.ready
        self._replicas_desired = response.replicas.desired
        if response.updated_at:
            self._updated_at = response.updated_at
        self._image = getattr(response, "image", self._image) or self._image
        self._phase = getattr(response, "phase", self._phase)
        self._message = getattr(response, "message", self._message)
        share_token = getattr(response, "share_token", None)
        if share_token is not None:
            self._share_token = share_token
        share_url = getattr(response, "share_url", None)
        if share_url is not None:
            self._share_url = share_url
        self._public = bool(getattr(response, "public", self._public))
        self._public_metadata = bool(
            getattr(response, "public_metadata", self._public_metadata)
        )
        self._distributed = getattr(response, "distributed", self._distributed)

    def __repr__(self) -> str:
        return (
            f"Deployment(name={self._name!r}, state={self._state!r}, url={self._url!r})"
        )

    def __str__(self) -> str:
        return f"Deployment '{self._name}' ({self._state}) at {self._url}"

    @classmethod
    def _from_response(cls, client: "BasilicaClient", response) -> "Deployment":
        """
        Create a Deployment from an API response.

        Internal method used by BasilicaClient.

        Issues #453 / #454: this factory previously extracted only a
        narrow subset of the PyO3 `DeploymentResponse` fields, so
        downstream code paths -- notably `_coerce_to_dict` driving
        `DistributedTraining.world` / `.ranks` / `.bench` -- saw the
        wrapper as missing `image`, `phase`, and `distributed`
        regardless of what the binding had populated. Every PyO3
        attribute the user could plausibly want at runtime is now
        copied onto the wrapper. `getattr` with a default keeps the
        method robust to older binding builds that have not yet
        rolled to the post-#451 attribute set.
        """
        return cls(
            client=client,
            instance_name=response.instance_name,
            friendly_name=getattr(response, "friendly_name", "") or "",
            url=response.url,
            namespace=response.namespace,
            user_id=response.user_id,
            state=response.state,
            created_at=response.created_at,
            replicas_ready=response.replicas.ready,
            replicas_desired=response.replicas.desired,
            updated_at=response.updated_at,
            image=getattr(response, "image", "") or "",
            phase=getattr(response, "phase", None),
            message=getattr(response, "message", None),
            share_token=getattr(response, "share_token", None),
            share_url=getattr(response, "share_url", None),
            public=bool(getattr(response, "public", True)),
            public_metadata=bool(getattr(response, "public_metadata", False)),
            distributed=getattr(response, "distributed", None),
        )
