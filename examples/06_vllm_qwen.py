#!/usr/bin/env python3
"""
vLLM deployment with Qwen model and persistent compilation cache.

The compilation cache stores compiled CUDA kernels, speeding up subsequent
container restarts. Model weights are downloaded fresh each time (FUSE storage
is too slow for large model files).

Usage:
    export BASILICA_API_TOKEN="your-token"
    python3 06_vllm_qwen.py
"""
import basilica

compile_cache = basilica.Volume.from_name("vllm-compile-cache", create_if_missing=True)


@basilica.deployment(
    name="vllm-qwen",
    image="vllm/vllm-openai:latest",
    port=8000,
    gpu_count=1,
    min_gpu_memory_gb=16,
    memory="16Gi",
    ttl_seconds=1800,
    timeout=900,
    volumes={"/root/.cache/vllm": compile_cache},
)
def serve():
    """Start vLLM server with Qwen model."""
    import subprocess

    cmd = "vllm serve Qwen/Qwen3-0.6B --host 0.0.0.0 --port 8000"
    subprocess.Popen(cmd, shell=True).wait()


deployment = serve()

print(f"vLLM API: {deployment.url}")
print(f"OpenAI-compatible endpoint: {deployment.url}/v1/chat/completions")
