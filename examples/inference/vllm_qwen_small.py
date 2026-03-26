#!/usr/bin/env python3
"""
Deploy Qwen2.5-0.5B-Instruct using vLLM.

This is a verified working example using a small model suitable for
quick testing and development. The model loads in under 2 minutes
and requires minimal GPU memory.

Model: Qwen/Qwen2.5-0.5B-Instruct (500M parameters)
Memory: ~2GB GPU RAM
Load time: ~1-2 minutes

Requirements:
- 1x A100 GPU (80GB VRAM)
- vLLM >= 0.6.0

Usage:
    export BASILICA_API_TOKEN="your-token"
    python3 vllm_qwen_small.py
"""
from basilica.decorators import deployment


@deployment(
    name="vllm-qwen-small",
    image="vllm/vllm-openai:latest",
    gpu="A100",
    port=8000,
    gpu_count=1,
    memory="40Gi",
    ttl_seconds=3600,
    timeout=600,
    env={
        "HF_HUB_ENABLE_HF_TRANSFER": "1",
    },
)
def serve():
    """Start vLLM with Qwen2.5-0.5B-Instruct model."""
    import subprocess

    cmd = [
        "vllm",
        "serve",
        "Qwen/Qwen2.5-0.5B-Instruct",
        "--host",
        "0.0.0.0",
        "--port",
        "8000",
        "--max-model-len",
        "8192",
    ]
    subprocess.Popen(cmd).wait()


if __name__ == "__main__":
    result = serve()

    print(f"vLLM Qwen2.5-0.5B deployment ready!")
    print(f"  URL:        {result.url}")
    print(f"  OpenAI API: {result.url}/v1/chat/completions")
    print(f"  Health:     {result.url}/health")
    print()
    print("Test with curl:")
    print(f"  curl -X POST {result.url}/v1/chat/completions \\")
    print('    -H "Content-Type: application/json" \\')
    print(
        '    -d \'{"model": "Qwen/Qwen2.5-0.5B-Instruct", "messages": [{"role": "user", "content": "Hello!"}]}\''
    )
