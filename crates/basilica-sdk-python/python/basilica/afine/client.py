"""
Client proxy and create() function for Basilica AFINE SDK.
"""

import atexit
import os
import tempfile
import time
from pathlib import Path
from typing import Any, Dict, List, Optional, Union

import httpx
from cryptography.hazmat.primitives import serialization
from cryptography.hazmat.primitives.asymmetric import ed25519
from tenacity import retry, retry_if_exception_type, stop_after_attempt, wait_exponential

from .api_client import BasilicaAPIClient
from .docker_manager import DockerManager


class Client:
    """
    Dynamic proxy for remote Service instances with authentication and resource management.

    Implements context manager protocol for proper cleanup.
    Automatically forwards method calls to the remote HTTP endpoint with authentication.

    Example:
        with bs.create("user/mathenv:latest") as client:
            obs = client.reset()
            obs, reward, terminated, truncated, info = client.step(3)
    """

    def __init__(
        self,
        rental_id: str,
        base_url: str,
        rental_secret: str,
        api_key: Optional[str] = None,
        timeout: float = 30.0
    ) -> None:
        """
        Initialize client proxy.

        Args:
            rental_id: ID of the rental
            base_url: Base URL of the remote service (e.g., "http://node-123.basilica.ai:34567")
            rental_secret: Shared secret for authenticating with service container
            api_key: Authentication key for Basilica API (separate from rental secret)
            timeout: Request timeout in seconds
        """
        self._rental_id = rental_id
        self._base_url = base_url.rstrip('/')
        self._rental_secret = rental_secret
        self._api_key = api_key
        self._timeout = timeout
        self._http_client = httpx.Client(timeout=timeout)
        self._closed = False

    def __enter__(self) -> 'Client':
        """Context manager entry."""
        return self

    def __exit__(self, exc_type, exc_val, exc_tb) -> None:
        """Context manager exit with automatic cleanup."""
        self.close()

    def __getattr__(self, name: str) -> Any:
        """
        Dynamically create method proxy for remote calls.

        This enables `client.reset()` to call the remote `reset()` endpoint.

        Args:
            name: Method name

        Returns:
            Callable that makes RPC call when invoked
        """
        if name.startswith('_'):
            raise AttributeError(f"'{type(self).__name__}' object has no attribute '{name}'")

        if self._closed:
            raise RuntimeError("Client is closed. Cannot make RPC calls.")

        @retry(
            stop=stop_after_attempt(3),
            wait=wait_exponential(multiplier=1, min=1, max=10),
            retry=retry_if_exception_type((httpx.TimeoutException, httpx.ConnectError)),
            reraise=True
        )
        def method_proxy(*args, **kwargs) -> Any:
            """Proxy function that makes the actual RPC call with retry logic."""
            try:
                response = self._http_client.post(
                    f"{self._base_url}/{name}",
                    json={"args": list(args), "kwargs": kwargs},
                    headers={"X-Basilica-Secret": self._rental_secret}
                )
                response.raise_for_status()
                return response.json()["result"]

            except httpx.HTTPStatusError as e:
                if e.response.status_code == 401:
                    raise RuntimeError(
                        "Authentication failed. Rental secret may be invalid or expired."
                    ) from e
                elif e.response.status_code == 404:
                    raise AttributeError(
                        f"Method '{name}' not found on remote service"
                    ) from e
                else:
                    raise RuntimeError(
                        f"RPC call to {name} failed: {e.response.status_code} {e.response.text}"
                    ) from e

        return method_proxy

    def close(self) -> None:
        """Close the HTTP client and release resources."""
        if not self._closed:
            self._http_client.close()
            self._closed = True

    def kill(self) -> None:
        """Terminate the rental and stop the container."""
        if self._closed:
            return

        api_client = BasilicaAPIClient(api_key=self._api_key)
        api_client.terminate_rental(self._rental_id, reason="User requested termination")

        self.close()

    def logs(self, follow: bool = False, tail: Optional[int] = None) -> None:
        """
        Stream logs from the container.

        Args:
            follow: Follow log output
            tail: Number of lines to show from end
        """
        api_client = BasilicaAPIClient(api_key=self._api_key)

        for log_line in api_client.stream_logs(self._rental_id, follow=follow, tail=tail):
            print(log_line, end='')

    def status(self) -> Dict[str, Any]:
        """
        Get rental and container status.

        Returns:
            Status information including container state and resource usage
        """
        api_client = BasilicaAPIClient(api_key=self._api_key)
        return api_client.get_rental_status(self._rental_id)

    def __del__(self) -> None:
        """Cleanup on garbage collection if not explicitly closed."""
        if not self._closed:
            self.close()


def create(
    image_or_path: Union[str, Path],
    api_key: Optional[str] = None,
    gpu_requirements: Optional[Dict[str, Any]] = None,
    node_id: Optional[str] = None,
    environment: Optional[Dict[str, str]] = None,
    ports: Optional[List[Dict[str, int]]] = None,
    timeout: float = 300.0
) -> Client:
    """
    Create a remote service instance with authentication.

    Args:
        image_or_path: Docker Hub image (e.g., "user/mathenv:latest") or local path
        api_key: Basilica API key (or read from env BASILICA_API_KEY)
        gpu_requirements: GPU requirements (e.g., {"gpu_count": 1, "min_memory_gb": 8})
        node_id: Specific node ID to use (optional)
        environment: Environment variables to pass to container
        ports: Port mappings (container_port -> host_port)
        timeout: Timeout for container to become ready (seconds)

    Returns:
        Client proxy instance with context manager support

    Example:
        with bs.create("user/mathenv:latest") as client:
            obs = client.reset()
            obs, reward, terminated, truncated, info = client.step(3)
    """
    if api_key is None:
        api_key = os.environ.get("BASILICA_API_KEY")
        if not api_key:
            raise ValueError(
                "API key not provided. Set BASILICA_API_KEY environment variable "
                "or pass api_key parameter."
            )

    path_obj = Path(image_or_path)
    is_local = path_obj.exists() and path_obj.is_dir()

    docker_manager = DockerManager()

    if is_local:
        service_file = "service.py"
        if not (path_obj / service_file).exists():
            raise FileNotFoundError(
                f"service.py not found in {path_obj}. "
                "Local paths must contain a service.py file."
            )

        tag = f"{os.environ.get('DOCKER_HUB_USERNAME', 'user')}/{path_obj.name}:latest"

        print(f"Building image: {tag}")
        docker_manager.build_image(path_obj, tag, service_file=service_file)

        try:
            docker_manager.login_registry()
            print(f"Pushing image: {tag}")
            docker_manager.push_image(tag)
        except ValueError:
            print("Warning: Docker credentials not set, skipping push. Image will only be available locally.")

        container_image = tag
    else:
        container_image = str(image_or_path)

        if "/" not in container_image:
            raise ValueError(
                f"Invalid image format: {container_image}. "
                "Expected format: 'user/image:tag' or local directory path."
            )

        print(f"Pulling image: {container_image}")
        try:
            docker_manager.pull_image(container_image)
        except Exception as e:
            print(f"Warning: Failed to pull image: {e}. Will attempt to use cached version or pull from remote during deployment.")

    ssh_public_key, ssh_private_key_path = generate_ssh_keypair()

    print("Starting rental on Basilica network...")
    api_client = BasilicaAPIClient(api_key=api_key)

    rental_info = api_client.start_rental(
        container_image=container_image,
        ssh_public_key=ssh_public_key,
        gpu_requirements=gpu_requirements,
        node_id=node_id,
        environment=environment,
        ports=ports
    )

    print(f"Rental created: {rental_info.rental_id}")
    print(f"Endpoint: {rental_info.endpoint_url}")

    print("Waiting for container to be ready...")
    ready = wait_for_health(
        rental_info.endpoint_url,
        timeout=timeout,
        check_interval=2.0
    )

    if not ready:
        raise TimeoutError(
            f"Container did not become healthy within {timeout} seconds. "
            f"Check logs with: basilica logs {rental_info.rental_id}"
        )

    print("Container ready!")

    return Client(
        rental_id=rental_info.rental_id,
        base_url=rental_info.endpoint_url,
        rental_secret=rental_info.rental_secret,
        api_key=api_key,
        timeout=30.0
    )


def wait_for_health(
    endpoint_url: str,
    timeout: float,
    check_interval: float = 2.0
) -> bool:
    """
    Poll health endpoint until container is ready.

    Args:
        endpoint_url: Base URL of service
        timeout: Maximum time to wait (seconds)
        check_interval: Time between checks (seconds)

    Returns:
        True if healthy, False if timeout
    """
    start_time = time.time()
    health_url = f"{endpoint_url}/health"

    while time.time() - start_time < timeout:
        try:
            response = httpx.get(health_url, timeout=5.0)
            if response.status_code == 200:
                data = response.json()
                if data.get("status") == "healthy":
                    return True
        except (httpx.RequestError, httpx.HTTPStatusError):
            pass

        time.sleep(check_interval)

    return False


def generate_ssh_keypair() -> tuple[str, str]:
    """
    Generate Ed25519 SSH key pair for container access.

    Returns:
        (public_key, private_key_path)
    """
    private_key = ed25519.Ed25519PrivateKey.generate()

    fd, private_key_path = tempfile.mkstemp(suffix=".key", prefix="basilica_")
    os.chmod(private_key_path, 0o600)

    with os.fdopen(fd, 'wb') as f:
        f.write(private_key.private_bytes(
            encoding=serialization.Encoding.PEM,
            format=serialization.PrivateFormat.OpenSSH,
            encryption_algorithm=serialization.NoEncryption()
        ))

    public_key = private_key.public_key()
    public_key_bytes = public_key.public_bytes(
        encoding=serialization.Encoding.OpenSSH,
        format=serialization.PublicFormat.OpenSSH
    )

    atexit.register(lambda: os.unlink(private_key_path) if os.path.exists(private_key_path) else None)

    return public_key_bytes.decode(), private_key_path
