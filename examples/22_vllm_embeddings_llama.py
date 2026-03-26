#!/usr/bin/env python3
"""
Deploy Llama-3.1-8B-Instruct as an embedding model via vLLM.

IMPORTANT: Llama-3.1-8B-Instruct was trained for chat/instruction-following,
NOT for generating embeddings. While vLLM can serve any model as an embedding
model using --task embed, the quality of embeddings from chat models is
typically lower than purpose-built embedding models.

For production use cases requiring high-quality embeddings (semantic search,
RAG, clustering), consider using:
    - intfloat/e5-mistral-7b-instruct (see 22_vllm_embeddings.py)
    - BAAI/bge-large-en-v1.5
    - Alibaba-NLP/gte-Qwen2-7B-instruct

This example demonstrates how to use Llama-3.1-8B-Instruct for embeddings
when you specifically need Llama-based embeddings or want to experiment.

Requirements:
    - HuggingFace account with access to meta-llama/Llama-3.1-8B-Instruct
    - Set HF_TOKEN environment variable for gated model access

Usage:
    export BASILICA_API_TOKEN="your-token"
    export HF_TOKEN="your-huggingface-token"
    python3 22_vllm_embeddings_llama.py

API Endpoints:
    POST /v1/embeddings - OpenAI-compatible embeddings endpoint
    GET /health - Health check
"""
import os

import basilica

# Verify HF_TOKEN is set for gated model access
hf_token = os.environ.get("HF_TOKEN")
if not hf_token:
    print("Warning: HF_TOKEN not set. Llama models require HuggingFace authentication.")
    print("Set it with: export HF_TOKEN='your-huggingface-token'")
    print()

model_cache = basilica.Volume.from_name("vllm-llama-cache", create_if_missing=True)


@basilica.deployment(
    name="vllm-llama-embeddings",
    image="vllm/vllm-openai:v0.8.5.post1",
    port=8000,
    gpu_count=1,
    min_gpu_memory_gb=24,
    memory="32Gi",
    env={
        "HF_TOKEN": hf_token or "",
        "HF_HUB_ENABLE_HF_TRANSFER": "1",
    },
    ttl_seconds=3600,
    timeout=900,
    volumes={"/root/.cache/huggingface": model_cache},
)
def serve_llama_embeddings():
    """Deploy Llama-3.1-8B-Instruct as an embedding model.

    Uses vLLM's --task embed flag to convert the generative model
    into an embedding model that pools token representations.
    """
    import subprocess

    cmd = [
        "vllm", "serve", "meta-llama/Llama-3.1-8B-Instruct",
        "--task", "embed",
        "--host", "0.0.0.0",
        "--port", "8000",
        "--max-model-len", "4096",
        "--dtype", "bfloat16",
    ]
    subprocess.Popen(cmd).wait()


if __name__ == "__main__":
    print("Deploying Llama-3.1-8B-Instruct as embedding model...")
    print("Note: This model was not trained for embeddings - see docstring for alternatives.")
    print()

    deployment = serve_llama_embeddings()

    print(f"\nEmbeddings API ready!")
    print(f"  Base URL:       {deployment.url}")
    print(f"  Embeddings:     {deployment.url}/v1/embeddings")
    print(f"  Health:         {deployment.url}/health")
    print()
    print("Example usage with OpenAI client:")
    print(f'''
from openai import OpenAI

client = OpenAI(
    api_key="EMPTY",
    base_url="{deployment.url}/v1",
)

response = client.embeddings.create(
    input=["What is machine learning?", "Deep learning uses neural networks"],
    model="meta-llama/Llama-3.1-8B-Instruct",
)

for i, data in enumerate(response.data):
    print(f"Embedding {{i}}: {{len(data.embedding)}} dimensions")
''')
    print()
    print("Example with curl:")
    print(f'''
curl -X POST {deployment.url}/v1/embeddings \\
  -H "Content-Type: application/json" \\
  -d '{{
    "model": "meta-llama/Llama-3.1-8B-Instruct",
    "input": ["What is machine learning?", "Deep learning uses neural networks"]
  }}'
''')
    print("\nTo delete the deployment:")
    print("  deployment.delete()")
