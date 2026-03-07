# Basilica Python SDK

The official Python SDK for deploying containerized applications on the Basilica GPU cloud platform.

[![PyPI](https://img.shields.io/pypi/v/basilica-sdk)](https://pypi.org/project/basilica-sdk/)
[![Python](https://img.shields.io/pypi/pyversions/basilica-sdk)](https://pypi.org/project/basilica-sdk/)
[![License](https://img.shields.io/pypi/l/basilica-sdk)](https://github.com/one-covenant/basilica/blob/main/LICENSE)

## Installation

```bash
pip install basilica-sdk
```

**Requirements:** Python 3.10+

## Quick Start

### 1. Get an API Token

```bash
# Install the Basilica CLI
pip install basilica-cli

# Create an API token
basilica tokens create

# Set the environment variable
export BASILICA_API_TOKEN="basilica_..."
```

### 2. Deploy Your First App

```python
from basilica import BasilicaClient

client = BasilicaClient()

# Deploy inline Python code
deployment = client.deploy(
    name="hello",
    source="""
from http.server import HTTPServer, BaseHTTPRequestHandler

class Handler(BaseHTTPRequestHandler):
    def do_GET(self):
        self.send_response(200)
        self.end_headers()
        self.wfile.write(b'Hello from Basilica!')

HTTPServer(('', 8000), Handler).serve_forever()
""",
    port=8000,
    ttl_seconds=600,  # Auto-delete after 10 minutes
)

print(f"Live at: {deployment.url}")
```

### 3. Deploy a FastAPI Application

```python
from basilica import BasilicaClient

client = BasilicaClient()

deployment = client.deploy(
    name="my-api",
    source="""
from fastapi import FastAPI

app = FastAPI()

@app.get("/")
def root():
    return {"message": "Hello from FastAPI!"}

@app.get("/items/{item_id}")
def get_item(item_id: int):
    return {"item_id": item_id, "name": f"Item {item_id}"}

if __name__ == "__main__":
    import uvicorn
    uvicorn.run(app, host="0.0.0.0", port=8000)
""",
    port=8000,
    pip_packages=["fastapi", "uvicorn"],
    ttl_seconds=600,
)

print(f"API docs: {deployment.url}/docs")
```

## Features

### High-Level API

The SDK provides a simple `deploy()` method that handles:
- Source code packaging
- Container image selection
- Dependency installation
- Health checking and readiness waiting
- Public URL provisioning

```python
deployment = client.deploy(
    name="my-app",           # Deployment name
    source="app.py",         # File path or inline code
    port=8000,               # Application port
    pip_packages=["flask"],  # Dependencies
    storage=True,            # Persistent storage at /data
    ttl_seconds=3600,        # Auto-cleanup (optional)
)

print(deployment.url)        # Public URL
print(deployment.logs())     # Application logs
deployment.delete()          # Manual cleanup
```

### Decorator API

Define deployments as decorated functions:

```python
import basilica

@basilica.deployment(
    name="my-service",
    port=8000,
    pip_packages=["fastapi", "uvicorn"],
    ttl_seconds=600,
)
def serve():
    from fastapi import FastAPI
    import uvicorn

    app = FastAPI()

    @app.get("/")
    def root():
        return {"status": "running"}

    uvicorn.run(app, host="0.0.0.0", port=8000)

# Deploy by calling the function
deployment = serve()
print(f"Live at: {deployment.url}")
```

### GPU Deployments

Deploy applications with GPU access:

```python
deployment = client.deploy(
    name="pytorch-inference",
    source="inference.py",
    image="pytorch/pytorch:2.1.0-cuda12.1-cudnn8-runtime",
    port=8000,
    gpu_count=1,
    gpu_models=["NVIDIA-RTX-A4000"],  # Optional: specific GPU models
    memory="8Gi",
    timeout=300,
)
```

### GPU Flavour Preferences

Filter GPU offerings and control hardware placement with interconnect, region, and spot preferences.

#### Query GPU offerings

```python
from basilica import BasilicaClient, GpuPriceQuery

client = BasilicaClient()

# SXM interconnect in the US
gpus = client.list_secure_cloud_gpus(
    query=GpuPriceQuery(interconnect="SXM", region="US")
)

# Non-spot offerings only
gpus = client.list_secure_cloud_gpus(
    query=GpuPriceQuery(exclude_spot=True)
)

# Spot offerings only
gpus = client.list_secure_cloud_gpus(
    query=GpuPriceQuery(spot_only=True)
)
```

#### Deploy with flavour constraints

All deployment methods (`deploy`, `create_deployment`, `deploy_vllm`, `deploy_sglang`,
and the `@deployment` decorator) accept flavour parameters:

```python
# create_deployment with flavour
resp = client.create_deployment(
    instance_name="my-inference",
    image="nginx:alpine",
    port=80,
    gpu_count=1,
    gpu_models=["H100"],
    interconnect="SXM",       # SXM or PCIe
    geo="US",                 # US, EU, CA, APAC
    spot=True,                # True=prefer spot, False=exclude spot
    ttl_seconds=600,
)

# deploy() with flavour (waits until ready)
deployment = client.deploy(
    name="gpu-app",
    source="app.py",
    port=8000,
    gpu_count=1,
    gpu_models=["H100"],
    interconnect="SXM",
    geo="US",
    spot=False,
)

# deploy_vllm with flavour
deployment = client.deploy_vllm(
    model="Qwen/Qwen3-0.6B",
    interconnect="SXM",
    geo="US",
    spot=True,
)

# Decorator API with flavour
@basilica.deployment(
    name="my-service",
    port=8000,
    gpu_count=1,
    interconnect="SXM",
    geo="US",
    spot=True,
)
def serve():
    ...
```

| Parameter | Description |
|-----------|-------------|
| `interconnect` | GPU interconnect type: `"SXM"` or `"PCIe"` |
| `geo` | Geographic region: `"US"`, `"EU"`, `"CA"`, `"APAC"` |
| `spot` | `True` = prefer spot instances, `False` = exclude spot |
| `infiniband` | `True` = require InfiniBand networking |

### Persistent Storage

Enable persistent storage mounted at `/data`:

```python
# Simple: Enable storage at /data
deployment = client.deploy(
    name="stateful-app",
    source="app.py",
    port=8000,
    storage=True,
)

# Custom mount path
deployment = client.deploy(
    name="stateful-app",
    source="app.py",
    port=8000,
    storage="/custom/path",
)
```

Using volumes with the decorator API:

```python
import basilica

cache = basilica.Volume.from_name("my-cache", create_if_missing=True)

@basilica.deployment(
    name="app-with-storage",
    port=8000,
    volumes={"/data": cache},
)
def serve():
    # Your app can read/write to /data
    pass
```

### Health Checks

Large model deployments (vLLM, SGLang) can take minutes to download and load
into GPU memory. Configure custom health check probes to prevent Kubernetes
from killing pods before models are ready:

```python
from basilica import BasilicaClient, HealthCheckConfig, ProbeConfig

client = BasilicaClient()

# Deploy SGLang with a 20-minute startup tolerance
deployment = client.deploy_sglang(
    model="Qwen/Qwen2.5-3B-Instruct",
    health_check=HealthCheckConfig(
        startup=ProbeConfig(
            path="/health",
            port=30000,
            initial_delay_seconds=0,
            period_seconds=10,
            timeout_seconds=5,
            failure_threshold=120,  # 120 * 10s = 20 minutes
        ),
        liveness=ProbeConfig(
            path="/health",
            port=30000,
            initial_delay_seconds=120,
            period_seconds=30,
            timeout_seconds=10,
            failure_threshold=5,
        ),
        readiness=ProbeConfig(
            path="/health",
            port=30000,
            initial_delay_seconds=60,
            period_seconds=15,
            timeout_seconds=10,
            failure_threshold=5,
        ),
    ),
    timeout=1200,
)
```

`deploy_vllm()` and `deploy_sglang()` include sensible defaults (10-minute startup
tolerance) when no `health_check` is provided. For very large models, pass your own
`HealthCheckConfig` to extend the startup window.

Health checks work with any deployment method:

```python
# Generic deploy()
deployment = client.deploy(
    name="my-gpu-app",
    source="app.py",
    port=8000,
    gpu_count=1,
    health_check=HealthCheckConfig(
        startup=ProbeConfig(path="/ready", port=8000, failure_threshold=60),
    ),
)

# Decorator API
@basilica.deployment(
    name="my-service",
    port=8000,
    health_check=HealthCheckConfig(
        startup=ProbeConfig(path="/health", port=8000, failure_threshold=60),
    ),
)
def serve():
    ...
```

`ProbeConfig` fields:
- `path`: HTTP endpoint to probe (e.g. `"/health"`)
- `port`: Port to probe (defaults to container port if `None`)
- `initial_delay_seconds`: Seconds before first probe (default: 30)
- `period_seconds`: Interval between probes (default: 10)
- `timeout_seconds`: Probe timeout (default: 5)
- `failure_threshold`: Consecutive failures before action (default: 3)

### Pre-built Container Images

Deploy any Docker image:

```python
deployment = client.deploy(
    name="nginx",
    image="nginxinc/nginx-unprivileged:alpine",
    port=8080,
    replicas=1,
    cpu="250m",
    memory="256Mi",
)
```

## API Reference

### BasilicaClient

```python
class BasilicaClient:
    def __init__(
        self,
        base_url: str = None,  # Default: https://api.basilica.ai
        api_key: str = None,   # Default: BASILICA_API_TOKEN env var
    ): ...
```

### deploy()

The primary method for deploying applications:

```python
def deploy(
    name: str,                              # Deployment name (DNS-safe)
    source: Optional[str | Path] = None,    # File path or inline code
    image: str = "python:3.11-slim",        # Container image
    port: int = 8000,                       # Application port
    env: Optional[Dict[str, str]] = None,   # Environment variables
    cpu: str = "500m",                      # CPU allocation
    memory: str = "512Mi",                  # Memory allocation
    storage: Union[bool, str] = False,      # Persistent storage
    gpu_count: Optional[int] = None,        # Number of GPUs
    gpu_models: Optional[List[str]] = None, # GPU model requirements
    min_cuda_version: Optional[str] = None, # Minimum CUDA version
    min_gpu_memory_gb: Optional[int] = None,# Minimum GPU VRAM
    interconnect: Optional[str] = None,     # GPU interconnect (SXM, PCIe)
    geo: Optional[str] = None,              # Region (US, EU, CA, APAC)
    spot: Optional[bool] = None,            # Spot preference (True/False)
    infiniband: Optional[bool] = None,      # Require InfiniBand
    replicas: int = 1,                      # Number of instances
    ttl_seconds: Optional[int] = None,      # Auto-delete timeout
    public: bool = True,                    # Create public URL
    timeout: int = 300,                     # Deployment timeout
    pip_packages: Optional[List[str]] = None,  # pip dependencies
    health_check: Optional[HealthCheckConfig] = None,  # Custom health probes
) -> Deployment
```

### Deployment Object

```python
class Deployment:
    name: str                    # Deployment name
    url: str                     # Public URL
    namespace: str               # Kubernetes namespace
    user_id: str                 # Owner user ID
    state: str                   # Current state
    created_at: str              # Creation timestamp

    def status() -> DeploymentStatus     # Get detailed status
    def logs(tail=None) -> str           # Get application logs
    def wait_until_ready(timeout=300)    # Block until ready
    def delete() -> None                 # Delete deployment
    def refresh() -> Deployment          # Refresh state
```

### DeploymentStatus

```python
@dataclass
class DeploymentStatus:
    state: str                   # Pending, Active, Running, Failed, Terminating
    replicas_ready: int          # Ready replica count
    replicas_desired: int        # Desired replica count
    message: Optional[str]       # Status message
    phase: Optional[str]         # Detailed phase
    progress: Optional[ProgressInfo]  # Progress information

    @property
    def is_ready(self) -> bool   # Check if fully ready

    @property
    def is_failed(self) -> bool  # Check if failed

    @property
    def is_pending(self) -> bool # Check if still starting
```

## Exception Handling

The SDK provides a comprehensive exception hierarchy:

```python
from basilica import (
    BasilicaError,        # Base exception
    AuthenticationError,  # Invalid/missing token
    AuthorizationError,   # Permission denied
    ValidationError,      # Invalid parameters
    DeploymentError,      # Base deployment error
    DeploymentNotFound,   # Deployment doesn't exist
    DeploymentTimeout,    # Timeout waiting for ready
    DeploymentFailed,     # Deployment crashed
    ResourceError,        # Resource unavailable
    StorageError,         # Storage configuration error
    NetworkError,         # API communication error
    RateLimitError,       # Rate limit exceeded
    SourceError,          # Source file error
)

try:
    deployment = client.deploy(...)
except DeploymentTimeout:
    print("Deployment took too long to start")
except DeploymentFailed as e:
    print(f"Deployment failed: {e}")
except AuthenticationError:
    print("Invalid API token")
```

## Low-Level API

For advanced use cases, access the low-level API methods:

```python
# Create deployment with full control
response = client.create_deployment(
    instance_name="my-app",
    image="python:3.11-slim",
    command=["python", "-m", "http.server", "8000"],
    port=8000,
    cpu="1",
    memory="1Gi",
)

# Get deployment details
response = client.get_deployment("my-app")

# Delete deployment
client.delete_deployment("my-app")

# List all deployments
response = client.list_deployments()

# Get logs
logs = client.get_deployment_logs("my-app", tail=100)
```

### GPU Offerings (Secure Cloud)

```python
from basilica import GpuPriceQuery

# List all GPU offerings
gpus = client.list_secure_cloud_gpus()

# Filter by interconnect, region, spot
gpus = client.list_secure_cloud_gpus(
    query=GpuPriceQuery(interconnect="SXM", region="US", exclude_spot=True)
)
for g in gpus:
    print(f"{g.gpu_type} x{g.gpu_count}  {g.interconnect}  {g.region}  ${g.hourly_rate}/hr")

# Start a secure cloud GPU rental
rental = client.start_secure_cloud_rental(offering_id=gpus[0].id)
print(f"SSH: {rental.ssh_command}")

# Stop rental
client.stop_secure_cloud_rental(rental.rental_id)
```

### GPU Rentals (Legacy API)

For direct GPU node access via SSH:

```python
# List available nodes
nodes = client.list_nodes(gpu_type="A100", min_gpu_count=1)

# Start a rental
rental = client.start_rental(
    gpu_type="A100",
    container_image="pytorch/pytorch:latest",
)
print(f"Rental ID: {rental.rental_id}")

# Get SSH credentials
status = client.get_rental(rental.rental_id)
if status.ssh_credentials:
    print(f"SSH: {status.ssh_credentials.username}@{status.ssh_credentials.host}")

# Stop rental
client.stop_rental(rental.rental_id)
```

## Environment Variables

| Variable | Description | Default |
|----------|-------------|---------|
| `BASILICA_API_TOKEN` | API authentication token | Required |
| `BASILICA_API_URL` | API endpoint URL | `https://api.basilica.ai` |

## Examples

For complete working examples, see the [examples directory](https://github.com/one-covenant/basilica/tree/main/examples):

| Example | Description |
|---------|-------------|
| `01_hello_world.py` | Simplest deployment with inline code |
| `02_with_storage.py` | Persistent storage example |
| `03_fastapi.py` | Production FastAPI deployment |
| `04_gpu.py` | GPU/CUDA deployment with PyTorch |
| `05_decorator_*.py` | Decorator API patterns |
| `06_vllm_qwen.py` | LLM inference with vLLM |
| `07_sglang_model.py` | SGLang model serving |
| `08_external_file.py` | Deploy from external Python file |
| `09_container_image.py` | Pre-built Docker image deployment |
| `10_custom_docker/` | Multi-file project with Dockerfile |
| `11_agentgym.py` | RL environment deployment |
| `12_lobe_chat.py` | Self-hosted chat UI |
| `13_lobe_chat_vllm.py` | Full AI stack (LobeChat + vLLM) |
| `14_streamlit.py` | Interactive Streamlit app |
| `34_gpu_flavour_preferences.py` | Query GPU offerings with flavour filters |
| `35_deploy_with_flavour.py` | Deploy with interconnect, region, spot preferences |
| `36_gpu_flavour_cli/` | CLI usage with flavour flags |
| `deploy_sglang_health_check.py` | SGLang with custom health check probes |

## Development

### Building from Source

```bash
git clone https://github.com/one-covenant/basilica.git
cd basilica/crates/basilica-sdk-python

# Create virtual environment
python -m venv .venv
source .venv/bin/activate

# Install maturin and build
pip install maturin
maturin develop

# Test import
python -c "from basilica import BasilicaClient; print('OK')"
```

### Running Tests

```bash
pip install pytest
pytest tests/ -v
```

## Architecture

The SDK is built with:
- **PyO3**: Rust bindings for high-performance HTTP operations
- **Async runtime**: Tokio-based async operations exposed as sync Python calls
- **Type safety**: Full type hints with IDE support

For detailed architecture documentation, see [PYTHON-SDK-ARCHITECTURE.md](https://github.com/one-covenant/basilica/blob/main/docs/architecture/PYTHON-SDK-ARCHITECTURE.md).

## License

MIT OR Apache-2.0

## Links

- [Documentation](https://docs.basilica.ai)
- [PyPI Package](https://pypi.org/project/basilica-sdk/)
- [GitHub Repository](https://github.com/one-covenant/basilica)
- [Issue Tracker](https://github.com/one-covenant/basilica/issues)
- [Changelog](https://github.com/one-covenant/basilica/blob/main/crates/basilica-sdk-python/CHANGELOG.md)
