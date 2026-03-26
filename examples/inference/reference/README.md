# Reference Configurations

Advanced deployment examples with framework-specific options.

## vLLM Examples

### Single GPU (7B model)
```python
@deployment(
    name="llama-7b",
    image="vllm/vllm-openai:latest",
    gpu="A100",
    gpu_count=1,
    memory="40Gi",
    port=8000,
)
def serve():
    subprocess.Popen([
        "vllm", "serve", "meta-llama/Llama-2-7b-chat-hf",
        "--host", "0.0.0.0",
        "--port", "8000",
        "--max-model-len", "4096",
    ]).wait()
```

### Multi-GPU with Tensor Parallelism (70B model)
```python
@deployment(
    name="llama-70b",
    image="vllm/vllm-openai:latest",
    gpu="H100",
    gpu_count=4,
    memory="320Gi",
    port=8000,
)
def serve():
    subprocess.Popen([
        "vllm", "serve", "meta-llama/Llama-2-70b-chat-hf",
        "--host", "0.0.0.0",
        "--port", "8000",
        "--tensor-parallel-size", "4",
        "--max-model-len", "4096",
    ]).wait()
```

## SGLang Examples

### Single GPU
```python
@deployment(
    name="qwen-7b",
    image="lmsysorg/sglang:latest",
    gpu="A100",
    gpu_count=1,
    memory="80Gi",
    port=30000,
)
def serve():
    subprocess.Popen([
        "python3", "-m", "sglang.launch_server",
        "--model-path", "Qwen/Qwen2.5-7B-Instruct",
        "--host", "0.0.0.0",
        "--port", "30000",
        "--mem-fraction-static", "0.85",
    ]).wait()
```

### Multi-GPU
```python
@deployment(
    name="large-model",
    image="lmsysorg/sglang:latest",
    gpu="H100",
    gpu_count=8,
    memory="640Gi",
    port=30000,
)
def serve():
    subprocess.Popen([
        "python3", "-m", "sglang.launch_server",
        "--model-path", "MODEL_ID",
        "--host", "0.0.0.0",
        "--port", "30000",
        "--tp", "8",
        "--trust-remote-code",
    ]).wait()
```

## Framework Options

### vLLM
| Option | Description |
|--------|-------------|
| `--tensor-parallel-size` | Split model across GPUs |
| `--max-model-len` | Maximum sequence length |
| `--gpu-memory-utilization` | GPU memory fraction (0.0-1.0) |
| `--quantization` | Quantization method (awq, gptq) |
| `--trust-remote-code` | Trust model's custom code |

### SGLang
| Option | Description |
|--------|-------------|
| `--tp` | Tensor parallelism size |
| `--mem-fraction-static` | Static memory fraction |
| `--context-length` | Maximum context length |
| `--quantization` | Quantization method (fp8) |

## Gated Models

Some HuggingFace models require authentication:

```python
@deployment(
    env={"HF_TOKEN": "hf_your_token"},
)
```

## Volumes for Model Caching

```python
import basilica

cache = basilica.Volume.from_name("model-cache", create_if_missing=True)

@deployment(
    volumes={"/root/.cache/huggingface": cache},
)
```
