"""
Basilica API client wrapper with secret generation for AFINE SDK.
"""

import os
import secrets
from typing import Any, Dict, Iterator, List, Optional

import httpx

from .models import RentalSecretInfo


class BasilicaAPIClient:
    """Client for Basilica API communication with secret management."""

    def __init__(
        self,
        base_url: str = "https://api.basilica.ai",
        api_key: Optional[str] = None
    ) -> None:
        """
        Initialize API client.

        Args:
            base_url: Basilica API base URL
            api_key: Authentication API key (or read from BASILICA_API_KEY env var)
        """
        self._base_url = base_url.rstrip('/')
        self._api_key = api_key or os.environ.get("BASILICA_API_KEY")

        if not self._api_key:
            raise ValueError(
                "API key not provided. "
                "Set BASILICA_API_KEY environment variable or pass api_key parameter."
            )

        self._http_client = httpx.Client(
            headers={"Authorization": f"Bearer {self._api_key}"},
            timeout=30.0
        )

    def start_rental(
        self,
        container_image: str,
        ssh_public_key: str,
        gpu_requirements: Optional[Dict[str, Any]] = None,
        node_id: Optional[str] = None,
        environment: Optional[Dict[str, str]] = None,
        ports: Optional[List[Dict[str, int]]] = None
    ) -> RentalSecretInfo:
        """
        Start a new rental with automatically generated secret.

        Args:
            container_image: Docker image to deploy
            ssh_public_key: SSH public key for container access
            gpu_requirements: GPU requirements (gpu_count, gpu_type, min_memory_gb)
            node_id: Specific node ID to use
            environment: Environment variables to pass to container
            ports: Port mappings

        Returns:
            Rental information including rental_id, endpoint_url, and rental_secret
        """
        rental_secret = secrets.token_urlsafe(32)

        env_vars = environment or {}
        env_vars["BASILICA_RENTAL_SECRET"] = rental_secret

        payload: Dict[str, Any] = {
            "container_image": container_image,
            "ssh_public_key": ssh_public_key,
            "environment": env_vars,
            "ports": ports or [{"container_port": 8000, "protocol": "tcp"}],
        }

        if gpu_requirements:
            payload["node_selection"] = {"exact_gpu_configuration": gpu_requirements}
        elif node_id:
            payload["node_selection"] = {"node_id": node_id}

        response = self._http_client.post(
            f"{self._base_url}/rentals/start",
            json=payload
        )
        response.raise_for_status()

        rental_data = response.json()

        endpoint_url = self._extract_endpoint_url(
            rental_data.get("ssh_credentials"),
            rental_data.get("container_info", {}).get("mapped_ports", [])
        )

        return RentalSecretInfo(
            rental_id=rental_data["rental_id"],
            endpoint_url=endpoint_url,
            rental_secret=rental_secret,
            ssh_credentials=rental_data.get("ssh_credentials"),
            container_info=rental_data.get("container_info", {})
        )

    def _extract_endpoint_url(
        self,
        ssh_credentials: Optional[str],
        mapped_ports: List[Dict[str, Any]]
    ) -> str:
        """
        Extract HTTP endpoint URL from rental response.

        Args:
            ssh_credentials: SSH credentials in format "user@host:port"
            mapped_ports: List of port mappings

        Returns:
            HTTP endpoint URL (e.g., "http://node-123.basilica.ai:34567")
        """
        port_mapping = next(
            (p for p in mapped_ports if p.get("container_port") == 8000),
            None
        )

        if not port_mapping:
            raise ValueError("Port 8000 not mapped in container")

        host_port = port_mapping["host_port"]

        if ssh_credentials and '@' in ssh_credentials:
            _, host_port_str = ssh_credentials.split('@', 1)
            host = host_port_str.split(':')[0] if ':' in host_port_str else host_port_str
        else:
            raise ValueError("Invalid SSH credentials format")

        return f"http://{host}:{host_port}"

    def get_rental_status(self, rental_id: str) -> Dict[str, Any]:
        """
        Get rental status.

        Args:
            rental_id: Rental ID

        Returns:
            Rental status information
        """
        response = self._http_client.get(f"{self._base_url}/rentals/{rental_id}/status")
        response.raise_for_status()
        return response.json()

    def terminate_rental(self, rental_id: str, reason: Optional[str] = None) -> None:
        """
        Terminate a rental.

        Args:
            rental_id: Rental ID
            reason: Optional termination reason
        """
        response = self._http_client.delete(
            f"{self._base_url}/rentals/{rental_id}",
            json={"reason": reason} if reason else None
        )
        response.raise_for_status()

    def stream_logs(
        self,
        rental_id: str,
        follow: bool = False,
        tail: Optional[int] = None
    ) -> Iterator[str]:
        """
        Stream logs from rental container.

        Args:
            rental_id: Rental ID
            follow: Follow log output
            tail: Number of lines from end

        Yields:
            Log lines
        """
        params: Dict[str, str] = {}
        if follow:
            params["follow"] = "true"
        if tail is not None:
            params["tail"] = str(tail)

        with self._http_client.stream(
            "GET",
            f"{self._base_url}/rentals/{rental_id}/logs",
            params=params
        ) as response:
            response.raise_for_status()
            for line in response.iter_lines():
                yield line

    def __del__(self) -> None:
        """Cleanup HTTP client."""
        self._http_client.close()
