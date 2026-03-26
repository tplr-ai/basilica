#!/usr/bin/env python3
"""
SGLang model deployment with custom HuggingFace model.

This example demonstrates deploying an SGLang inference server with:
- Custom HuggingFace model and revision
- GPU requirements (count, VRAM)
- Concurrency settings
- Auto-shutdown (TTL)

SGLang provides an OpenAI-compatible API at /v1/chat/completions.

Usage:
    export BASILICA_API_TOKEN="your-token"
    python3 07_sglang_model.py
"""
import basilica

compile_cache = basilica.Volume.from_name("sglang-cache", create_if_missing=True)


@basilica.deployment(
    name="sglang-qwen",
    image="lmsysorg/sglang:latest",
    port=30000,
    gpu_count=1,
    min_gpu_memory_gb=16,
    memory="16Gi",
    ttl_seconds=3600,
    timeout=900,
    volumes={"/root/.cache/sglang": compile_cache},
    env={
        "HF_HUB_ENABLE_HF_TRANSFER": "1",
    },
)
def serve():
    """Start SGLang server with Qwen model."""
    import subprocess

    cmd = [
        "python3", "-m", "sglang.launch_server",
        "--model-path", "Qwen/Qwen2.5-0.5B-Instruct",
        "--host", "0.0.0.0",
        "--port", "30000",
        "--mem-fraction-static", "0.8",
    ]
    subprocess.Popen(cmd).wait()


deployment = serve()

print(f"SGLang API: {deployment.url}")
print(f"OpenAI-compatible endpoint: {deployment.url}/v1/chat/completions")
print(f"Health check: {deployment.url}/health")
