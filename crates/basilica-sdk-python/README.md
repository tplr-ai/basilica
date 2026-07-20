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

## Distributed Training (NCCL collectives)

For `torch.distributed` workloads (DDP, DiLoCo, FSDP, any NCCL-collective workload),
use the `@basilica.distributed` decorator. The platform provisions GPU spokes,
wires the WireGuard mesh, runs torchrun with the standard `RANK` / `WORLD_SIZE`
/ `MASTER_*` env-var contract, and tears down on context exit.

This is a separate surface from `@basilica.deployment` because the lifecycle
shape is different: a single function body fans out to N rank pods under
torchrun, the user reads `os.environ["RANK"]` instead of branching on rank in
SDK code, and the handle exposes `scale()`, `wait_until_*`, `bench`, and
`rank_exits` instead of `url` and `replicas`.

### Canonical surface: `@basilica.distributed` decorator

```python
import basilica
from basilica import ProviderFilter, WorldSize


@basilica.distributed(
    name="dlc-hello",
    image="ghcr.io/one-covenant/basilica/basilica-distributed-trainer:latest",
    world_size=WorldSize(min=2, target=2, max=2),
    gpu_count=1,
    gpu_models=["A100"],
    min_gpu_memory_gb=40,
    cpu="8",
    memory="32Gi",
    provider_filter=ProviderFilter(include=["cyan", "plum"]),
    topology_spread="pack",
    bench=True,
    nccl_env={"NCCL_DEBUG": "WARN"},
    ttl_seconds=900,
)
def train() -> None:
    import os

    import torch
    import torch.distributed as dist

    rank = int(os.environ["RANK"])
    world_size = int(os.environ["WORLD_SIZE"])
    local_rank = int(os.environ.get("LOCAL_RANK", 0))

    dist.init_process_group(backend="nccl")
    device = torch.device(f"cuda:{local_rank}")

    x = torch.ones(1024, device=device)
    dist.all_reduce(x)
    if rank == 0:
        print(f"world={world_size} sum={x.sum().item():.0f}", flush=True)

    dist.destroy_process_group()


if __name__ == "__main__":
    with train() as training:
        print(f"Deployed: {training.name}")
        training.wait_until_complete(timeout=1800)
        if training.bench is not None:
            print(f"Measured busbw: {training.bench.busbw_gbps_p50} Gbps")
```

Calling the decorated function returns a `DistributedTraining` context-manager.
Use it bare (`train()`) for fire-and-forget with auto-cleanup; use `with
train() as training:` for mid-run orchestration (`scale()`, `wait_until_*`,
`logs()`, `bench`).

### BYO launcher: `basilica.distributed(command=[...])` factory

When you want to drive torchrun (or mpirun, or accelerate) yourself, pass
`command=[...]` instead of decorating a function. The same `basilica.distributed`
symbol becomes a factory and returns a `DistributedTraining` directly:

```python
import basilica
from basilica import ProviderFilter, WorldSize

training = basilica.distributed(
    name="dlc-torchrun",
    image="ghcr.io/one-covenant/basilica/basilica-distributed-trainer:latest",
    command=[
        "torchrun",
        "--rdzv-backend=etcd",
        "--rdzv-endpoint=$BASILICA_RDZV_ENDPOINT",
        "--rdzv-id=$BASILICA_RDZV_ID",
        "--nnodes=$BASILICA_WORLD_TARGET",
        "--nproc-per-node=$BASILICA_GPUS_PER_POD",
        "/workspace/all_reduce_smoke.py",
    ],
    world_size=WorldSize(min=2, target=2, max=4),
    gpu_count=1,
    gpu_models=["A100"],
    provider_filter=ProviderFilter(include=["cyan", "plum"]),
    topology_spread="pack",
    ttl_seconds=900,
)

with training:
    training.scale(target=3)
    training.wait_until_target_world(timeout=300)
    print(training.logs(tail=30))
```

Same `DistributedTraining` handle as the decorator path. Same context-manager
cleanup. The only difference is `command=[...]` replaces the decorated function
body. Inside the launcher, expand `$BASILICA_RDZV_ENDPOINT` / `$BASILICA_RANK` /
`$BASILICA_WORLD_TARGET` / `$BASILICA_GPUS_PER_POD` — the operator injects
these into the worker pod environment.

### Lifecycle on the `DistributedTraining` handle

| Method | What it does |
|--------|--------------|
| `training.world` | `WorldStatus(ready, target, min, max, below_minimum)` snapshot |
| `training.phase` | Operator-driven phase: `pending` / `ready` / `succeeded` / `failed` / `cancelled` |
| `training.ranks` | Per-rank pod observations (`provider`, `region`, `phase`, `restarts`) |
| `training.rank_exits` | Per-rank exit diagnostics, populated after the UD reaches a terminal state |
| `training.bench` | `BenchResult \| None` — `None` means "no measurement", regardless of why |
| `training.bench_diagnostics` | Debug detail when `training.bench` is `None`; rarely needed |
| `training.scale(target=N)` | Patch `worldSize.target` mid-run; ranks join/drain asynchronously |
| `training.wait_until_min_world(timeout=...)` | Block until `ready >= min` or raise `BelowMinimumWorld` |
| `training.wait_until_target_world(timeout=...)` | Block until `ready >= target` |
| `training.wait_until_complete(timeout=...)` | Block until terminal phase (`succeeded` / `failed` / `cancelled`) |
| `training.logs(tail=..., follow=...)` | Stream merged-rank logs |
| `training.delete()` | Explicit teardown; `__exit__` runs this on scope exit |

Every method has an `_async` counterpart (`training.scale_async(...)`,
`async with training: ...`).

### `bench=True` — the diagnostic for "is my code slow or is the network slow?"

NCCL collective bandwidth varies with availability zone root, region, GPU model, and current
mesh load. Without a measurement, the user has to guess whether slow training
is their code or the network.

`bench=True` opts in to a 2-rank `all_reduce_perf` probe that runs alongside
your workers on the same availability zone root mesh with the same WireGuard transport. The
result lands on `training.bench` after the UD reaches a terminal state:

```python
import basilica
from basilica import ProviderFilter, WorldSize


@basilica.distributed(
    name="dlc-with-bench",
    image="ghcr.io/one-covenant/basilica/basilica-distributed-trainer:latest",
    world_size=WorldSize(min=2, target=2, max=2),
    gpu_count=1,
    gpu_models=["A100"],
    provider_filter=ProviderFilter(include=["cyan", "plum"]),
    topology_spread="pack",
    bench=True,
)
def train() -> None:
    import torch.distributed as dist

    dist.init_process_group(backend="nccl")
    dist.destroy_process_group()


with train() as training:
    training.wait_until_complete(timeout=1800)
    if training.bench is not None:
        print(f"busbw_gbps_p50:      {training.bench.busbw_gbps_p50}")
        print(f"latency_us_at_1mib:  {training.bench.latency_us_at_1mib}")
        print(
            f"probe pair:          {training.bench.probe_node_a} "
            f"<-> {training.bench.probe_node_b}"
        )
    else:
        # No measurement -- workers too short, probe couldn't co-schedule, etc.
        print(training.bench_diagnostics)
```

`None` means "no measurement" regardless of reason. `training.bench_diagnostics`
exists for the rare case where you want to know WHY the probe didn't land a
result. Default `bench=False` because the probe costs 2 extra GPU ranks for its
duration — users who don't need the measurement shouldn't pay for it.

### Running an external script

The canonical input is a Callable (what the decorator captures). For an
external `.py` file shipped in the trainer image, wrap it in a decorated
function:

```python
import basilica
from basilica import ProviderFilter, WorldSize


@basilica.distributed(
    name="dlc-script",
    image="ghcr.io/one-covenant/basilica/basilica-distributed-trainer:latest",
    world_size=WorldSize(min=2, target=2, max=2),
    gpu_count=1,
    gpu_models=["A100"],
    provider_filter=ProviderFilter(include=["cyan", "plum"]),
    topology_spread="pack",
)
def run_script() -> None:
    import runpy

    runpy.run_path("/workspace/my_training.py", run_name="__main__")
```

### Migration from the legacy surface

These surfaces emitted `DeprecationWarning` in 0.29.5-0.29.7 and were
REMOVED in 0.30.0 (SDK-S7, basilica-backend issue 666). If you upgraded
from 0.29.x and see `AttributeError` / `ImportError` / `ValidationError`
on a name below, apply the canonical mapping in the right column.

| Removed in 0.30.0 | Canonical replacement | Removed in | Plan ticket |
|-------------------|-----------------------|------------|-------------|
| `client.deploy_distributed(...)` | `@basilica.distributed(...)` on a function, call it | 0.30.0 | S1+S7 |
| `client.deploy_distributed_managed(...)` | `with train() as training:` on the decorated function | 0.30.0 | S1+S7 |
| `bench="on-start"` / `bench="off"` (str) | `bench=True` / `bench=False` (bool) | 0.30.0 | S2+S7 |
| `training.bench_status` (property) | `training.bench` (result) + `training.bench_diagnostics` (debug dict) | 0.30.0 | S2+S7 |
| `training.wait_until_bench_complete(...)` | `with train() as training:` (auto-blocks) + read `training.bench` | 0.30.0 | S2+S7 |
| `BenchStatus` re-export from `basilica` | `BenchResult` + `bench_diagnostics` dict | 0.30.0 | S2+S7 |
| `client.deploy_distributed_managed(command=[...])` | `basilica.distributed(command=[...])` factory | 0.30.0 | S3+S7 |
| `source=Path("./train.py")` / `source="<inline code>"` | decorate a function (Callable shape) | 0.30.0 | S4+S7 |
| `DistributedTrainingManaged` / `DistributedTrainingManagedAsync` | `DistributedTraining` (itself context-manager-able) | 0.30.0 | S1+S7 |

The decorator (`@basilica.distributed`) is the ONE canonical surface and
routes through the private `BasilicaClient._deploy_distributed_impl` /
`_deploy_distributed_impl_async` methods. There is no public
`deploy_distributed*` method on `BasilicaClient` in 0.30.0+.

### Worked examples in this repo

| Example | Pattern |
|---------|---------|
| `examples/20_distributed_diloco.py` | `@basilica.distributed` decorator + `bench=True` + DiLoCo (NCCL all-reduce in the outer step) |
| `examples/21_distributed_torchrun.py` | `basilica.distributed(command=[torchrun ...])` factory + mid-run `scale()` |
| `examples/22_distributed_with_bench.py` | `@basilica.distributed` decorator + bench-result inspection + JSON dump |

## Managed Inference

Basilica's inference gateway at `https://inference.basilica.ai` is OpenAI-compatible — it serves `POST /v1/chat/completions` and `POST /v1/completions` in the OpenAI wire format, authenticated with your `basilica_` API key as a Bearer token. The SDK's `client.inference` surface is the discovery/usage layer around it: model catalog, usage rollups, and the kwargs that point the stock `openai` SDK at the gateway.

Chat completions are deliberately **not** re-wrapped: the wire contract is OpenAI's, so the canonical client is the official `openai` package, which tracks that schema as it evolves. Wire it up with `openai_client_args()`:

```python
from openai import OpenAI
from basilica import BasilicaClient

client = BasilicaClient()
openai = OpenAI(**client.inference.openai_client_args())
# -> {"base_url": "https://inference.basilica.ai/v1", "api_key": "basilica_..."}

resp = openai.chat.completions.create(
    model="llama-3.1-70b-instruct",
    messages=[{"role": "user", "content": "hello"}],
)
```

The three surface methods:

```python
# Model catalog (GET /v1/models)
for model in client.inference.list_models():
    print(model.id, model.owned_by)

model = client.inference.get_model("llama-3.1-70b-instruct")  # GET /v1/models/{id}

# Usage rollups (GET {api_base}/v1/inference/usage/summary)
rows = client.inference.usage(
    from_date="2026-07-01",   # datetime.date also accepted
    to_date="2026-07-31",
    model="llama-3.1-70b-instruct",  # optional
    kid="key-abc",                  # optional, per-API-key filter
)
for row in rows:
    # charge_credits is Decimal (never float), date is datetime.date
    print(row.date, row.model, row.prompt_tokens, row.charge_credits)
```

Gateway errors raise typed exceptions mirroring its OpenAI error JSON — `InferenceAuthenticationError` (401), `InsufficientCreditsError` (402), `InferenceModelNotFoundError` (404), `InferenceUnavailableError` (503), and `InferenceQuotaExceededError` (429), the last exposing `cap` (`"rpm" | "tpm" | "concurrency" | "budget"`) and `retry_after` for operator automation:

```python
from basilica import InferenceQuotaExceededError

try:
    models = client.inference.list_models()
except InferenceQuotaExceededError as e:
    print(f"{e.cap} cap tripped; retry after {e.retry_after}s")
```

Async variants (`list_models_async`, `get_model_async`, `usage_async`) follow the same pattern as the rest of the client. Set `BASILICA_INFERENCE_ENDPOINT` to point the surface at a staging/local gateway.

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

### Distributed Training Types

The distributed-training surface lives in `basilica` (re-exported from
`basilica.distributed`). The canonical entry point is the
`@basilica.distributed` decorator; `basilica.distributed(command=[...])` is
the factory shape for BYO launchers.

```python
@dataclass(frozen=True)
class WorldSize:
    """Worker-rank triple. `1 <= min <= target <= max`."""
    min: int     # below this the run pauses (torchelastic MIN_NODES)
    target: int  # steady-state replica count
    max: int     # hard ceiling for scale()


@dataclass(frozen=True)
class ProviderFilter:
    """Inclusive/exclusive public availability zone root filter for worker scheduling.

    Match is against Basilica public availability zone root names (`cyan`,
    `plum`, `opal`). With `topology_spread="pack"`, the autoscaler picks ONE
    availability zone root from the include-list and all workers pack on it.
    """
    include: List[str] = []
    exclude: List[str] = []


@dataclass(frozen=True)
class BenchResult:
    """Per-UD NCCL probe measurement. Read via `training.bench`.

    Bandwidth fields are GB/s (1 GB = 10^9 bytes); latency is at the
    smallest swept message size. `None`-valued fields mean the probe
    could not measure them (partial sweep).
    """
    measured_at: datetime
    busbw_gbps_p10: Optional[float]
    busbw_gbps_p50: Optional[float]
    busbw_gbps_p90: Optional[float]
    algbw_gbps_p50: Optional[float]
    latency_us_at_1mib: Optional[float]
    size_bytes_swept: List[int]
    probe_node_a: str
    probe_node_b: str


class DistributedTraining:
    """Handle returned by `@basilica.distributed` and `basilica.distributed(command=...)`.

    Itself a context manager: `with train() as training:` blocks on the
    decorator call, yields the handle, and best-effort `delete()`s on
    scope exit (success OR exception).
    """
    name: str
    namespace: str
    rendezvous_endpoint: str

    # observation (lazy refresh on first access)
    world: WorldStatus
    phase: str            # operator-driven phase
    is_terminal: bool
    ranks: List[RankStatus]
    rank_exits: List[RankExit]
    bench: Optional[BenchResult]
    bench_diagnostics: Optional[Dict[str, Any]]

    # lifecycle
    def scale(target: int) -> WorldStatus: ...
    def wait_until_min_world(timeout: int = 300) -> None: ...
    def wait_until_target_world(timeout: int = 600) -> None: ...
    def wait_until_complete(timeout: int = 1800) -> WorldStatus: ...
    def logs(tail=None, follow=False, rank=None) -> str: ...
    def delete() -> None: ...

    # context manager
    def __enter__(self) -> "DistributedTraining": ...
    def __exit__(self, exc_type, exc_val, exc_tb) -> None: ...
    # Every method has an `_async` counterpart.
```

`WorldStatus`, `RankStatus`, `RankExit`, `DistributedMetrics`, and the
distributed-specific exceptions (`BelowMinimumWorld`, `WorldSizeOutOfBounds`,
`RendezvousUnavailable`, `UDTerminalState`, `QuotaExceeded`, `DistributedError`)
are exported from `basilica` directly.

## Exception Handling

The SDK provides a comprehensive exception hierarchy:

```python
from basilica import (
    BasilicaError,         # Base exception
    AuthenticationError,   # Invalid/missing token
    AuthorizationError,    # Permission denied
    ValidationError,       # Invalid parameters
    DeploymentError,       # Base deployment error
    DeploymentNotFound,    # Deployment doesn't exist
    DeploymentTimeout,     # Timeout waiting for ready
    DeploymentFailed,      # Deployment crashed
    ResourceError,         # Resource unavailable
    StorageError,          # Storage configuration error
    NetworkError,          # API communication error
    RateLimitError,        # Rate limit exceeded
    SourceError,           # Source file error
    # Distributed-training specific:
    DistributedError,      # Base distributed-training error
    BelowMinimumWorld,     # `ready < min` (timeout or terminal failure)
    WorldSizeOutOfBounds,  # WorldSize triple or scale() target invalid
    RendezvousUnavailable, # etcd rendezvous service unreachable
    UDTerminalState,       # UD is already terminal (e.g. scale on succeeded)
    QuotaExceeded,         # Namespace rank budget exceeded
    # Managed Inference (client.inference):
    InferenceError,                # Base inference-gateway error
    InferenceAuthenticationError,  # 401 bad/expired API key
    InsufficientCreditsError,      # 402 balance below floor
    InferenceModelNotFoundError,   # 404 unknown/unowned model
    InferenceQuotaExceededError,   # 429 quota cap (`.cap`, `.retry_after`)
    InferenceUnavailableError,     # 503 pool saturated/unavailable
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
| `BASILICA_INFERENCE_ENDPOINT` | Inference gateway origin (`client.inference`) | `https://inference.basilica.ai` |

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
| `20_distributed_diloco.py` | `@basilica.distributed` + DiLoCo NCCL training |
| `21_distributed_torchrun.py` | BYO `command=[torchrun ...]` factory + mid-run scale |
| `22_distributed_with_bench.py` | `bench=True` + reading `training.bench` |
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
