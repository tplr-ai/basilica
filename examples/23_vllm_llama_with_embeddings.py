#!/usr/bin/env python3
"""
Deploy an LLM with a dedicated Embeddings service.

This example deploys a complete LLM + Embeddings stack:
  1. Qwen2.5-7B-Instruct for chat/completions (vLLM)
  2. E5-Mistral-7B-Instruct for embeddings (vLLM auto-detects embedding mode)

This is the standard architecture for RAG applications:
  - Use the embedding model to vectorize documents and queries
  - Use the LLM to generate responses based on retrieved context

Requirements:
    - 2x GPUs with 24GB+ VRAM each (or autoscaling cluster)

Usage:
    export BASILICA_API_TOKEN="your-token"
    python3 23_vllm_llama_with_embeddings.py

Endpoints:
    LLM:        /v1/chat/completions, /v1/completions
    Embeddings: /v1/embeddings (OpenAI-compatible)
"""
import basilica


def deploy_embeddings(client: basilica.BasilicaClient) -> basilica.Deployment:
    """Deploy E5-Mistral-7B embedding model using vLLM."""
    print("Deploying intfloat/e5-mistral-7b-instruct embedding service (vLLM)...")

    @basilica.deployment(
        name="embed-7b",
        image="vllm/vllm-openai:v0.8.5.post1",
        port=8000,
        gpu_count=1,
        min_gpu_memory_gb=24,
        memory="32Gi",
        env={
            "HF_HUB_DISABLE_SYMLINKS_WARNING": "1",
            "HF_HUB_DISABLE_XET": "1",
        },
        ttl_seconds=3600,
        timeout=1800,
    )
    def serve():
        import subprocess
        # e5-mistral is a dedicated embedding model with pooling config
        # vLLM auto-detects and serves /v1/embeddings endpoint
        cmd = [
            "vllm", "serve", "intfloat/e5-mistral-7b-instruct",
            "--host", "0.0.0.0",
            "--port", "8000",
            "--max-model-len", "4096",
            "--trust-remote-code",
        ]
        subprocess.Popen(cmd).wait()

    return serve()


def deploy_llm(client: basilica.BasilicaClient) -> basilica.Deployment:
    """Deploy Qwen2.5-7B-Instruct for chat/completions."""
    print("Deploying Qwen/Qwen2.5-7B-Instruct inference service...")

    @basilica.deployment(
        name="llm-7b",
        image="vllm/vllm-openai:v0.8.5.post1",
        port=8000,
        gpu_count=1,
        min_gpu_memory_gb=24,
        memory="32Gi",
        env={
            "HF_HUB_DISABLE_SYMLINKS_WARNING": "1",
            "HF_HUB_DISABLE_XET": "1",
        },
        ttl_seconds=3600,
        timeout=1800,
    )
    def serve():
        import subprocess
        cmd = [
            "vllm", "serve", "Qwen/Qwen2.5-7B-Instruct",
            "--host", "0.0.0.0",
            "--port", "8000",
            "--max-model-len", "4096",
        ]
        subprocess.Popen(cmd).wait()

    return serve()


def main():
    client = basilica.BasilicaClient()

    # Deploy LLM first
    llm = deploy_llm(client)
    print(f"  LLM API: {llm.url}/v1/chat/completions")

    # Deploy embeddings
    embeddings = deploy_embeddings(client)
    print(f"  Embeddings API: {embeddings.url}/v1/embeddings")

    # Print summary
    print()
    print("=" * 60)
    print("LLM + Embeddings Stack Ready")
    print("=" * 60)
    print()
    print("Endpoints:")
    print(f"  Embeddings:  {embeddings.url}/v1/embeddings")
    print(f"  Chat:        {llm.url}/v1/chat/completions")
    print(f"  Completions: {llm.url}/v1/completions")
    print()
    print("Models:")
    print(f"  Embeddings: intfloat/e5-mistral-7b-instruct (vLLM)")
    print(f"  LLM:        Qwen/Qwen2.5-7B-Instruct (vLLM)")
    print()
    print("To delete deployments:")
    print("  embeddings.delete()")
    print("  llm.delete()")


if __name__ == "__main__":
    main()
