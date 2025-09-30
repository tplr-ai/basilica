"""
Basilica SDK for Python

A Python SDK for interacting with the Basilica GPU rental network.
"""

import os
from typing import Optional, Dict, Any, List

from basilica._basilica import (
    BasilicaClient as _BasilicaClient,
    # Helper functions
    executor_by_id,
    executor_by_gpu,
    # Response types
    HealthCheckResponse,
    RentalResponse,
    RentalStatusWithSshResponse,
    RentalStatus,
    SshAccess,
    ExecutorDetails,
    GpuSpec,
    CpuSpec,
    AvailableExecutor,
    AvailabilityInfo,
    # Request types
    StartRentalApiRequest,
    ExecutorSelection,
    GpuRequirements,
    PortMappingRequest,
    ResourceRequirementsRequest,
    VolumeMountRequest,
    ListAvailableExecutorsQuery,
    ListRentalsQuery,
    # Constants from Rust
    DEFAULT_API_URL,
    DEFAULT_TIMEOUT_SECS,
    DEFAULT_CONTAINER_IMAGE,
    DEFAULT_GPU_TYPE,
    DEFAULT_GPU_COUNT,
    DEFAULT_GPU_MIN_MEMORY_GB,
    DEFAULT_CPU_CORES,
    DEFAULT_MEMORY_MB,
    DEFAULT_STORAGE_MB,
)

# Default command is a list in Python
DEFAULT_COMMAND = ["/bin/bash"]

__version__ = "0.1.0"
__all__ = [
    "BasilicaClient",
    # Helper functions
    "executor_by_id",
    "executor_by_gpu",
    # Response types
    "HealthCheckResponse",
    "RentalResponse",
    "RentalStatusWithSshResponse",
    "RentalStatus",
    "SshAccess",
    "ExecutorDetails",
    "GpuSpec",
    "CpuSpec",
    "AvailableExecutor",
    "AvailabilityInfo",
    # Request types
    "StartRentalApiRequest",
    "ExecutorSelection",
    "GpuRequirements",
    "PortMappingRequest",
    "ListAvailableExecutorsQuery",
    "ListRentalsQuery",
]


class BasilicaClient:
    """
    Client for interacting with the Basilica API.

    Example:
        >>> # Create token: basilica tokens create
        >>> client = BasilicaClient("https://api.basilica.ai", api_key="basilica_...")
        >>> health = client.health_check()
        >>> print(health["status"])
    """

    def __init__(
        self,
        base_url: Optional[str] = None,
        api_key: Optional[str] = None
    ):
        """
        Initialize a new Basilica client.

        Args:
            base_url: The base URL of the Basilica API (default: from BASILICA_API_URL env or DEFAULT_API_URL)
            api_key: Optional authentication token (default: from BASILICA_API_TOKEN env)
                Create token using: basilica tokens create
        """
        # Auto-detect base_url if not provided
        if base_url is None:
            base_url = os.environ.get("BASILICA_API_URL", DEFAULT_API_URL)

        # Pass api_key directly to Rust binding
        # The Rust binding will check BASILICA_API_TOKEN env var if api_key is None
        self._client = _BasilicaClient(base_url, api_key)
    
    def health_check(self) -> HealthCheckResponse:
        """
        Check the health of the API.
        
        Returns:
            HealthCheckResponse: Typed response with status, version, and validator info
        """
        return self._client.health_check()
    
    def list_executors(
        self,
        available: Optional[bool] = None,
        gpu_type: Optional[str] = None,
        min_gpu_count: Optional[int] = None,
        min_gpu_memory: Optional[int] = None
    ) -> List[AvailableExecutor]:
        """
        List available executors.
        
        Args:
            available: Filter by availability
            gpu_type: Filter by GPU type
            min_gpu_count: Filter by minimum GPU count
            min_gpu_memory: Filter by minimum GPU memory in GB
            
        Returns:
            List[AvailableExecutor]: List of typed executor objects with details
        """
        if any([available is not None, gpu_type is not None, min_gpu_count is not None, min_gpu_memory is not None]):
            query = ListAvailableExecutorsQuery(
                available=available,
                gpu_type=gpu_type,
                min_gpu_count=min_gpu_count,
                min_gpu_memory=min_gpu_memory
            )
            return self._client.list_executors(query)
        else:
            return self._client.list_executors(None)
    
    def start_rental(
        self,
        container_image: Optional[str] = None,
        executor_id: Optional[str] = None,
        gpu_type: Optional[str] = None,
        ssh_pubkey_path: Optional[str] = None,
        environment: Optional[Dict[str, str]] = None,
        ports: Optional[List[Dict[str, Any]]] = None,
        command: Optional[List[str]] = None,
        no_ssh: bool = False
    ) -> RentalResponse:
        """
        Start a new rental.
        
        Args:
            container_image: Docker image to run (default: DEFAULT_CONTAINER_IMAGE)
            executor_id: Optional specific executor to use
            gpu_type: GPU type to request (default: DEFAULT_GPU_TYPE)
            ssh_pubkey_path: Path to SSH public key file (e.g., "~/.ssh/id_rsa.pub").
                If None, defaults to ~/.ssh/basilica_ed25519.pub
            environment: Environment variables
            ports: Port mappings
            command: Command to run (default: ["/bin/bash"])
            no_ssh: Disable SSH access
            
        Returns:
            RentalResponse: Typed response with rental details
        """
        # Set defaults from constants
        if container_image is None:
            container_image = DEFAULT_CONTAINER_IMAGE
        
        if gpu_type is None:
            gpu_type = DEFAULT_GPU_TYPE
        
        ssh_public_key = None  # This will hold the actual key content
        if not no_ssh:
            # Determine which SSH key file to use
            if ssh_pubkey_path is not None:
                # User provided a custom path
                ssh_key_path = os.path.expanduser(ssh_pubkey_path)
            else:
                # Use default path
                ssh_key_path = os.path.expanduser("~/.ssh/basilica_ed25519.pub")
            
            # Read the SSH key from the file
            if os.path.exists(ssh_key_path):
                with open(ssh_key_path) as f:
                    ssh_public_key = f.read().strip()
            else:
                # If user specified a path that doesn't exist, raise an error
                if ssh_pubkey_path is not None:
                    raise FileNotFoundError(f"SSH public key file not found: {ssh_key_path}")
                # Otherwise, leave as None (no SSH key available)
        
        # Always use default resources internally
        resources = {
            "gpu_count": DEFAULT_GPU_COUNT,
            "gpu_types": [gpu_type] if gpu_type else [],  # Array of GPU types
            "cpu_cores": DEFAULT_CPU_CORES,
            "memory_mb": DEFAULT_MEMORY_MB,
            "storage_mb": DEFAULT_STORAGE_MB
        }
        
        # Build executor_selection based on whether executor_id is provided
        if executor_id:
            executor_selection = executor_by_id(executor_id)
        else:
            # Use GPU requirements for auto-selection with defaults
            gpu_count_val = DEFAULT_GPU_COUNT
            min_memory_gb_val = DEFAULT_GPU_MIN_MEMORY_GB
            
            # Get GPU type from gpu_types array if available
            gpu_types = resources.get("gpu_types", [])
            gpu_type_val = None
            if gpu_types and len(gpu_types) > 0:
                gpu_type_val = gpu_types[0]
            elif gpu_type:
                gpu_type_val = gpu_type
            
            gpu_requirements = GpuRequirements(
                gpu_count=gpu_count_val,
                min_memory_gb=min_memory_gb_val,
                gpu_type=gpu_type_val
            )
            executor_selection = executor_by_gpu(gpu_requirements)
        
        # Convert ports to PortMappingRequest objects
        port_mappings = []
        if ports:
            for port in ports:
                port_mappings.append(PortMappingRequest(
                    container_port=port.get("container_port", 0),
                    host_port=port.get("host_port", 0),
                    protocol=port.get("protocol", "tcp")
                ))
        
        # Create ResourceRequirementsRequest with defaults
        resource_req = ResourceRequirementsRequest(
            cpu_cores=resources.get("cpu_cores", DEFAULT_CPU_CORES),
            memory_mb=resources.get("memory_mb", DEFAULT_MEMORY_MB),
            storage_mb=resources.get("storage_mb", DEFAULT_STORAGE_MB),
            gpu_count=resources.get("gpu_count", DEFAULT_GPU_COUNT),
            gpu_types=resources.get("gpu_types", [])
        )
        
        # Volume mounts are always empty now
        volume_mounts = []
        
        request = StartRentalApiRequest(
            executor_selection=executor_selection,
            container_image=container_image,
            ssh_public_key=ssh_public_key if ssh_public_key else "",
            environment=environment or {},
            ports=port_mappings,
            resources=resource_req,
            command=command if command is not None else DEFAULT_COMMAND,
            volumes=volume_mounts,
            no_ssh=no_ssh
        )
            
        return self._client.start_rental(request)
    
    def get_rental(self, rental_id: str) -> RentalStatusWithSshResponse:
        """
        Get rental status.
        
        Args:
            rental_id: The rental ID
            
        Returns:
            RentalStatusWithSshResponse: Typed response with status and SSH details
        """
        return self._client.get_rental(rental_id)
    
    def stop_rental(self, rental_id: str) -> None:
        """
        Stop a rental.
        
        Args:
            rental_id: The rental ID
        """
        self._client.stop_rental(rental_id)
    
    def list_rentals(
        self,
        status: Optional[str] = None,
        gpu_type: Optional[str] = None,
        min_gpu_count: Optional[int] = None
    ) -> Dict[str, Any]:
        """
        List rentals.
        
        Args:
            status: Filter by status (e.g., "active", "provisioning")
            gpu_type: Filter by GPU type
            min_gpu_count: Filter by minimum GPU count
            
        Returns:
            List of rentals
        """
        if any([status is not None, gpu_type is not None, min_gpu_count is not None]):
            query = ListRentalsQuery(
                status=status,
                gpu_type=gpu_type,
                min_gpu_count=min_gpu_count
            )
            return self._client.list_rentals(query)
        else:
            return self._client.list_rentals(None)