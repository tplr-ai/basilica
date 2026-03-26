# Basilica SDK Examples

Production-ready examples demonstrating deployment patterns on Basilica.

## Prerequisites

```bash
# 1. Get an API token
basilica tokens create my-token
export BASILICA_API_TOKEN="basilica_..."

# 2. Install Python SDK
pip install basilica-sdk
```

## Core Examples (01-04)

Simple, self-contained examples using `client.deploy()`:

| Example | Description | Run |
|---------|-------------|-----|
| `01_hello_world.py` | Basic HTTP server | `python3 01_hello_world.py` |
| `02_with_storage.py` | Persistent counter at /data | `python3 02_with_storage.py` |
| `03_fastapi.py` | FastAPI with pip packages | `python3 03_fastapi.py` |
| `04_gpu.py` | PyTorch + CUDA | `python3 04_gpu.py` |

## Decorator Examples (05)

Using `@basilica.deployment` decorator:

| Example | Description | Run |
|---------|-------------|-----|
| `05_decorator_hello.py` | Basic decorator usage | `python3 05_decorator_hello.py` |
| `05_decorator_storage.py` | With Volume mount | `python3 05_decorator_storage.py` |
| `05_decorator_fastapi.py` | FastAPI + uvicorn | `python3 05_decorator_fastapi.py` |
| `05_decorator_gpu.py` | GPU decorator | `python3 05_decorator_gpu.py` |

## Advanced Examples (06-23)

| Example | Description | Run |
|---------|-------------|-----|
| `06_vllm_qwen.py` | vLLM with Qwen model | `python3 06_vllm_qwen.py` |
| `07_sglang_model.py` | SGLang inference server | `python3 07_sglang_model.py` |
| `08_external_file.py` | Deploy from external .py file | `python3 08_external_file.py` |
| `09_container_image.py` | Deploy pre-built container (nginx) | `python3 09_container_image.py` |
| `10_custom_docker/` | Multi-file project with custom Docker | See directory README |
| `11_agentgym.py` | AgentGym RL evaluation environments | `python3 11_agentgym.py` |
| `12_lobe_chat.py` | LobeChat self-hosted AI interface | `python3 12_lobe_chat.py` |
| `13_lobe_chat_vllm.py` | LobeChat + vLLM (fully private AI stack) | `python3 13_lobe_chat_vllm.py` |
| `14_streamlit.py` | Streamlit interactive data app | `python3 14_streamlit.py` |
| `15_cli_deploy/` | CLI deploy walkthrough | See directory README |
| `16_progress_callback.py` | Custom deployment progress monitoring | `python3 16_progress_callback.py` |
| `17_agentgym_custom/` | Custom AgentGym environment | See directory README |
| `18_torchrun_ddp/` | PyTorch DDP training with torchrun | See directory README |
| `19_deploy_vllm_template.py` | vLLM using deploy_vllm() template | `python3 19_deploy_vllm_template.py` |
| `20_deploy_sglang_template.py` | SGLang using deploy_sglang() template | `python3 20_deploy_sglang_template.py` |
| `21_async_concurrent.py` | Async concurrent deployments | `python3 21_async_concurrent.py` |
| `22_vllm_embeddings.py` | vLLM Embeddings API (E5-Mistral) | `python3 22_vllm_embeddings.py` |
| `22_vllm_embeddings_llama.py` | vLLM Embeddings with Llama-3.1-8B | `python3 22_vllm_embeddings_llama.py` |
| `23_vllm_llama_with_embeddings.py` | Llama-3.1-8B + E5 Embeddings (RAG stack) | `python3 23_vllm_llama_with_embeddings.py` |
| `24_clawdbot.py` | Clawdbot AI agent platform | `python3 24_clawdbot.py` |
| `25_kimi_k2_5.py` | Kimi-K2-Instruct 1T MoE (8x H200) | `python3 25_kimi_k2_5.py` |
| `28_openclaw.py` | OpenClaw gateway | `python3 28_openclaw.py` |
| `29_deploy_sglang_health_check.py` | SGLang with custom health check probes | `python3 29_deploy_sglang_health_check.py` |
| `31_public_metadata.py` | Public metadata enrollment via Python SDK | `python3 31_public_metadata.py` |
| `31_public_metadata.sh` | Public metadata enrollment lifecycle via CLI | `./31_public_metadata.sh` |
| `32_public_metadata_cli/` | Public metadata CLI reference | See directory README |
| `33_websocket.py` | WebSocket deployment via Python SDK | `python3 33_websocket.py` |
| `33_websocket.sh` | WebSocket deployment via CLI | `./33_websocket.sh` |

## Large Model Deployment Notes

Models over 100B parameters (like Kimi-K2, DeepSeek-V3) require:
- **8x H200/H100 GPUs** for tensor parallelism
- **15-30 minutes** for model loading (500GB+ weights)
- **Extended health check timeout** - may require monitoring via logs

For extremely large models, consider using **GPU Rentals** (SSH access) instead:
```bash
basilica up h200 --gpu-count 8
basilica ssh <rental-id>
# Then run vLLM directly on the instance
```

## Deployment Options

### 1. Inline Source Code
Best for small scripts and quick prototypes.
```python
deployment = client.deploy(name="hello", source="print('Hello')", port=8000)
```

### 2. External File
Best for single-file applications.
```python
deployment = client.deploy(name="api", source="app.py", port=8000)
```

### 3. Pre-built Container Image
Best for existing Docker images (nginx, redis, etc.).
```python
deployment = client.deploy(name="nginx", image="nginxinc/nginx-unprivileged:alpine", port=8080)
```

### 4. Custom Docker Image (Multi-file Projects)
Best for complex applications with multiple files/modules.
```bash
# Build and push your image
docker build -t ghcr.io/user/my-api:latest .
docker push ghcr.io/user/my-api:latest
```
```python
deployment = client.deploy(name="my-api", image="ghcr.io/user/my-api:latest", port=8000)
```
See `10_custom_docker/` for complete example.

## API Patterns

### Basic Deploy
```python
from basilica import BasilicaClient

client = BasilicaClient()
deployment = client.deploy(
    name="hello",
    source="app.py",
    port=8000,
)
print(deployment.url)
```

### Decorator Deploy
```python
import basilica

@basilica.deployment(name="api", port=8000, pip_packages=["fastapi", "uvicorn"])
def serve():
    from fastapi import FastAPI
    import uvicorn
    app = FastAPI()
    @app.get("/")
    def root():
        return {"status": "ok"}
    uvicorn.run(app, host="0.0.0.0", port=8000)

deployment = serve()
print(deployment.url)
```

### With Volume
```python
import basilica

cache = basilica.Volume.from_name("my-cache", create_if_missing=True)

@basilica.deployment(name="app", volumes={"/cache": cache})
def serve():
    ...
```

### GPU Deployment
```python
@basilica.deployment(
    name="pytorch",
    image="pytorch/pytorch:2.1.0-cuda12.1-cudnn8-runtime",
    gpu="NVIDIA-RTX-A4000",
    gpu_count=1,
    memory="8Gi",
)
def serve():
    ...
```

### Progress Monitoring

By default, `client.deploy()` and `deployment.wait_until_ready()` show progress output:
```
[my-app] Waiting for scheduler... (replicas: 0/1)
[my-app] Pulling container image... (replicas: 0/1)
[my-app] Running health checks... (replicas: 0/1)
[my-app] Deployment ready!
```

**Silent mode** - suppress all output:
```python
deployment.wait_until_ready(timeout=120, silent=True)
```

**Custom callback** - for custom UIs or logging:
```python
def my_progress(status):
    print(f"Phase: {status.phase}, Replicas: {status.replicas_ready}/{status.replicas_desired}")
    if status.progress and status.progress.percentage:
        print(f"  Progress: {status.progress.percentage:.1f}%")

deployment.wait_until_ready(on_progress=my_progress)
```

See `16_progress_callback.py` for a complete example.

## Available GPUs

| Model | VRAM | CUDA | Use Case |
|-------|------|------|----------|
| NVIDIA RTX A4000 | 16GB | 12.8 | Small models (7B) |
| NVIDIA A100 | 40/80GB | 12.x | Medium models (70B) |
| NVIDIA H100 | 80GB | 12.x | Large models (70B+) |
| NVIDIA H200 | 141GB | 12.x | Massive MoE models (1T+) |

## Container Requirements

Basilica runs containers as non-root (UID 1000). When building custom images:
```dockerfile
RUN useradd -m -u 1000 appuser
USER appuser
```

## Troubleshooting

**Deployment pending**: Check image name, reduce resources, or verify GPU availability.

**502/503 errors**: Wait 10-15s for HTTP server startup, verify port matches.

**Storage not ready**: Check for `.fuse_ready` marker, wait 30-60s after deploy.

**GPU not detected**: Use CUDA image, verify `torch.cuda.is_available()`.

## Legacy Examples

Verbose examples with more detailed patterns are archived in `legacy/`.
