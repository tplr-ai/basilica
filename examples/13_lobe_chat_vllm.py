#!/usr/bin/env python3
"""
LobeChat + vLLM - Self-hosted AI chat with local LLM inference.

This example deploys a complete self-hosted AI stack:
  1. vLLM server running Qwen model on GPU
  2. LobeChat UI pre-configured to use the vLLM endpoint

The result is a fully private ChatGPT-like experience with no external API calls.

Usage:
    export BASILICA_API_TOKEN="your-token"
    python3 13_lobe_chat_vllm.py

Repository:
  - LobeChat: https://github.com/lobehub/lobe-chat
  - vLLM: https://github.com/vllm-project/vllm
"""
import basilica


def deploy_vllm(client: basilica.BasilicaClient) -> basilica.Deployment:
    """Deploy vLLM server with Qwen model."""
    print("Deploying vLLM server (this may take a few minutes)...")

    cache = basilica.Volume.from_name("vllm-cache", create_if_missing=True)

    @basilica.deployment(
        name="vllm-backend",
        image="vllm/vllm-openai:latest",
        port=8000,
        gpu_count=1,
        min_gpu_memory_gb=16,
        memory="16Gi",
        volumes={"/root/.cache/vllm": cache},
        ttl_seconds=3600,
        timeout=600,
    )
    def serve():
        import subprocess
        cmd = "vllm serve Qwen/Qwen3-0.6B --host 0.0.0.0 --port 8000"
        subprocess.Popen(cmd, shell=True).wait()

    return serve()


def deploy_lobechat(client: basilica.BasilicaClient, vllm_url: str) -> basilica.Deployment:
    """Deploy LobeChat configured to use vLLM backend."""
    print("Deploying LobeChat UI...")

    return client.deploy(
        name="lobe-chat-vllm",
        image="lobehub/lobe-chat:latest",
        port=3210,
        env={
            "OPENAI_API_KEY": "not-needed",
            "OPENAI_PROXY_URL": f"{vllm_url}/v1",
            "OPENAI_MODEL_LIST": "Qwen/Qwen3-0.6B",
            "ACCESS_CODE": "basilica",
        },
        cpu="500m",
        memory="1Gi",
        ttl_seconds=3600,
        timeout=180,
    )


def main():
    client = basilica.BasilicaClient()

    vllm = deploy_vllm(client)
    print(f"  vLLM API: {vllm.url}")

    lobechat = deploy_lobechat(client, vllm.url)

    print()
    print("=" * 60)
    print("Self-Hosted AI Stack Ready")
    print("=" * 60)
    print(f"LobeChat UI:  {lobechat.url}")
    print(f"vLLM Backend: {vllm.url}")
    print(f"Access Code:  basilica")
    print()
    print("Open LobeChat in your browser - it's pre-configured to use vLLM!")
    print()
    print("Available model: Qwen/Qwen3-0.6B")


if __name__ == "__main__":
    main()
