# Deploy LLM Inference on Basilica

Deploy OpenAI-compatible LLM endpoints with a single command.

## Quickstart

```bash
export BASILICA_API_TOKEN="your-token"
python3 quickstart.py
```

This deploys a small model and returns an endpoint you can use immediately:

```bash
curl https://your-deployment.deployments.basilica.ai/v1/chat/completions \
  -H "Content-Type: application/json" \
  -d '{"model": "Qwen/Qwen2.5-0.5B-Instruct", "messages": [{"role": "user", "content": "Hello!"}]}'
```

## CLI Alternative

```bash
basilica deploy vllm Qwen/Qwen2.5-0.5B-Instruct --name my-llm
```

## Configuration

The `@deployment` decorator accepts these parameters:

| Parameter | Description | Default |
|-----------|-------------|---------|
| `name` | Deployment name | Required |
| `image` | Container image | Required |
| `gpu` | GPU model (A100, H100) | A100 |
| `gpu_count` | Number of GPUs | 1 |
| `memory` | Memory allocation | 40Gi |
| `port` | Application port | 8000 |
| `ttl_seconds` | Auto-shutdown after idle | 3600 |
| `timeout` | Deploy timeout | 600 |

## Choosing a Model

Models are specified by their HuggingFace ID. GPU requirements depend on model size:

| Model Size | Example | GPUs | Memory |
|------------|---------|------|--------|
| < 1B | Qwen2.5-0.5B | 1 | 8GB |
| 7B | Llama-2-7B | 1 | 24GB |
| 13B | Llama-2-13B | 1 | 40GB |
| 70B | Llama-2-70B | 2-4 | 160GB |
| 200B+ | Large MoE models | 4-8 | 320GB+ |

## Inference Frameworks

### vLLM (Recommended)
```python
@deployment(image="vllm/vllm-openai:latest", port=8000)
def serve():
    subprocess.Popen(["vllm", "serve", "MODEL", "--host", "0.0.0.0", "--port", "8000"]).wait()
```

### SGLang
```python
@deployment(image="lmsysorg/sglang:latest", port=30000)
def serve():
    subprocess.Popen(["python3", "-m", "sglang.launch_server", "--model-path", "MODEL", "--port", "30000"]).wait()
```

## Management

```bash
basilica deploy ls                    # List deployments
basilica deploy status my-llm         # Check status
basilica deploy logs my-llm --follow  # Stream logs
basilica deploy delete my-llm         # Delete
```

## Reference Examples

See `reference/` for advanced configurations with multi-GPU setups and framework-specific options.
