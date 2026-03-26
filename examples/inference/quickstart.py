#!/usr/bin/env python3
"""
Quickstart: Deploy an LLM inference endpoint on Basilica.

This example deploys a small model and returns an OpenAI-compatible API endpoint.

Usage:
    export BASILICA_API_TOKEN="your-token"
    python3 quickstart.py
"""
from basilica.decorators import deployment


@deployment(
    name="my-llm",
    image="vllm/vllm-openai:latest",
    gpu="A100",
    gpu_count=1,
    memory="40Gi",
    port=8000,
    ttl_seconds=3600,
)
def serve():
    import subprocess
    subprocess.Popen([
        "vllm", "serve", "Qwen/Qwen2.5-0.5B-Instruct",
        "--host", "0.0.0.0",
        "--port", "8000",
    ]).wait()


if __name__ == "__main__":
    result = serve()
    print(f"\nDeployment ready!")
    print(f"  Endpoint: {result.url}/v1/chat/completions")
    print(f"\nTest with:")
    print(f'  curl {result.url}/v1/chat/completions \\')
    print(f'    -H "Content-Type: application/json" \\')
    print(f'    -d \'{{"model": "Qwen/Qwen2.5-0.5B-Instruct", "messages": [{{"role": "user", "content": "Hello!"}}]}}\'')
