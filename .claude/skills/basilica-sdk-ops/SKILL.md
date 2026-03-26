---
name: basilica-sdk-ops
description: Use when the user wants to operate Basilica programmatically from Python, including deployments, rentals, inference templates, balance checks, and usage history.
---

# Basilica SDK Ops

Use this skill for Python-based Basilica automation:

- scripts
- notebooks
- CI jobs
- programmatic deploy/rental orchestration
- code generation that uses `BasilicaClient`

## Install And Authenticate

Install the SDK:

```bash
pip install basilica-sdk
```

Install the CLI if you need to mint an API token:

```bash
pip install basilica-cli
basilica tokens create my-sdk-token
export BASILICA_API_TOKEN="basilica_..."
```

Optional API override:

```bash
export BASILICA_API_URL="https://api.basilica.ai"
```

Basic client:

```python
from basilica import BasilicaClient

client = BasilicaClient()
```

## Safe First Call

```python
from basilica import BasilicaClient

client = BasilicaClient()
health = client.health_check()

print(health.status)
print(health.version)
```

## Balance And Usage

The public Python wrapper exposes balance and usage history:

```python
from basilica import BasilicaClient

client = BasilicaClient()

balance = client.get_balance()
usage = client.list_usage_history(limit=20, offset=0)

print(balance["balance"], balance["last_updated"])

for rental in usage["rentals"]:
    print(rental["rental_id"], rental["status"], rental["current_cost"])
```

If the user needs deposit-address creation or deposit-history inspection, prefer the CLI skill because the public Python wrapper does not expose that flow cleanly.

## High-Level Deploy

`deploy()` is the main SDK surface. It packages source, creates the deployment, waits for readiness, and refreshes the final URL.

```python
from basilica import BasilicaClient

client = BasilicaClient()

deployment = client.deploy(
    name="hello-api",
    source="""
from fastapi import FastAPI
import uvicorn

app = FastAPI()

@app.get("/")
def root():
    return {"ok": True}

uvicorn.run(app, host="0.0.0.0", port=8000)
""",
    port=8000,
    pip_packages=["fastapi", "uvicorn"],
    ttl_seconds=600,
)

print(deployment.url)
print(deployment.status().phase)
print(deployment.logs(tail=50))
```

Good default: set `ttl_seconds` unless the user explicitly wants the deploy to persist.

## Deploy A Prebuilt Image

```python
from basilica import BasilicaClient

client = BasilicaClient()

deployment = client.deploy(
    name="nginx-demo",
    image="nginxinc/nginx-unprivileged:alpine",
    port=8080,
    cpu="250m",
    memory="256Mi",
    ttl_seconds=300,
)

print(deployment.url)
```

## vLLM And SGLang Helpers

vLLM:

```python
from basilica import BasilicaClient, HealthCheckConfig

client = BasilicaClient()

deployment = client.deploy_vllm(
    model="Qwen/Qwen2.5-7B-Instruct",
    name="qwen-7b",
    ttl_seconds=3600,
    trust_remote_code=True,
    health_check=HealthCheckConfig.vllm_large_model(),
)

print(f"{deployment.url}/v1/chat/completions")
```

SGLang:

```python
from basilica import BasilicaClient, HealthCheckConfig

client = BasilicaClient()

deployment = client.deploy_sglang(
    model="Qwen/Qwen2.5-7B-Instruct",
    name="qwen-sg",
    ttl_seconds=3600,
    health_check=HealthCheckConfig.sglang_large_model(),
)

print(f"{deployment.url}/v1/chat/completions")
```

## Secure-Cloud Rentals

Preferred programmatic rental path:

```python
from basilica import BasilicaClient

client = BasilicaClient()

key = client.get_ssh_key()
if key is None:
    key = client.register_ssh_key("agent-key")

offering = sorted(client.list_secure_cloud_gpus(), key=lambda o: float(o.hourly_rate))[0]

rental = client.start_secure_cloud_rental(
    offering_id=offering.id,
    ssh_public_key_id=key.id,
)

print(rental.rental_id)
print(rental.status)
print(rental.ssh_command)
```

Teardown:

```python
result = client.stop_secure_cloud_rental(rental.rental_id)
print(result.total_cost)
```

CPU-only secure-cloud operations are also exposed through:

- `list_cpu_offerings()`
- `start_cpu_rental()`
- `stop_cpu_rental()`
- `list_cpu_rentals()`

## Important Data Model Notes

Deployment response/status fields include:

- `state`
- `phase`
- `message`
- `replicas_ready`
- `replicas_desired`
- `progress`
- `url`

Observed deployment phases include:

- `pending`
- `scheduling`
- `pulling`
- `initializing`
- `storage_sync`
- `starting`
- `health_check`
- `ready`
- `degraded`
- `failed`
- `terminating`

## Caveats Agents Must Know

### 1. `deploy()` blocks

`deploy()`, `deploy_vllm()`, and `deploy_sglang()` wait for readiness. They are not fire-and-forget. Use realistic `timeout` values for big models.

### 2. Large models need explicit health checks

For big models, custom health checks are often the difference between "works" and "killed during warmup".

### 3. Public is the default

The SDK deploy surface defaults to `public=True`.

### 4. Deposit flows are incomplete in the Python wrapper

Use the CLI for:

- deposit address creation
- deposit history

### 5. Legacy `start_rental()` is stale

The legacy GPU rental path exists, but the safer recommendation is secure-cloud rental APIs. If you do use `start_rental()`, the current wrapper requires `max_hourly_rate`.

### 6. `Deployment.status().message` is currently weak

If you need failure text, prefer a fresh low-level fetch:

```python
response = client.get_deployment("my-app")
print(response.message)
```

### 7. Source packaging has edge cases

- callable deployment sources depend on `inspect.getsource()`
- inline strings that end with `.py` may be treated as file paths
- decorator-based volume handling is less robust than full named-volume semantics

## Good Agent Defaults

- use `BasilicaClient()` with env-driven auth
- use `ttl_seconds` for any agent-created deployment
- use secure-cloud APIs for machine rentals
- use CLI for funding/deposit workflows
- use explicit health checks for anything bigger than a toy model

## File Pointers

- `crates/basilica-sdk-python/README.md`
- `crates/basilica-sdk-python/python/basilica/__init__.py`
- `crates/basilica-sdk-python/python/basilica/_deployment.py`
- `crates/basilica-sdk-python/python/basilica/source.py`
- `crates/basilica-sdk-python/python/basilica/decorators.py`
- `crates/basilica-sdk-python/examples/quickstart.py`
- `crates/basilica-sdk-python/examples/start_secure_cloud_gpu_rental.py`
- `examples/19_deploy_vllm_template.py`
- `examples/20_deploy_sglang_template.py`

## TODOs

- add a repo-local snippet for private deployments once the Python wrapper exposes that path cleanly
- add tested examples for usage-history parsing and cost summarization
