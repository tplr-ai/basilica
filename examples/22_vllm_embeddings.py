#!/usr/bin/env python3
"""
Deploy vLLM with Embeddings API support.

vLLM can serve embedding models via the --task embed flag, exposing an
OpenAI-compatible /v1/embeddings endpoint.

Model Options:
    1. Purpose-built embedding models (recommended):
       - intfloat/e5-mistral-7b-instruct (7B, best quality)
       - BAAI/bge-large-en-v1.5 (335M, faster)
       - Alibaba-NLP/gte-Qwen2-7B-instruct (7B, multilingual)

    2. Generative models converted to embeddings:
       - meta-llama/Llama-3.1-8B-Instruct (works but not optimal for embeddings)
       - Any text generation model with --task embed

Usage:
    export BASILICA_API_TOKEN="your-token"
    python3 22_vllm_embeddings.py

API Endpoints:
    POST /v1/embeddings - OpenAI-compatible embeddings endpoint
    GET /health - Health check
"""
import basilica

# Create volume for model caching (speeds up subsequent deployments)
model_cache = basilica.Volume.from_name("vllm-embeddings-cache", create_if_missing=True)


@basilica.deployment(
    name="vllm-embeddings",
    image="vllm/vllm-openai:v0.8.5.post1",
    port=8000,
    gpu_count=1,
    min_gpu_memory_gb=24,
    memory="32Gi",
    ttl_seconds=3600,
    timeout=900,
    volumes={"/root/.cache/huggingface": model_cache},
)
def serve_embeddings():
    """Deploy E5-Mistral-7B as an embedding model.

    E5-Mistral-7B-Instruct is trained specifically for embeddings and provides
    high-quality sentence embeddings suitable for semantic search and RAG.
    vLLM auto-detects the model's pooling configuration.
    """
    import subprocess

    cmd = [
        "vllm", "serve", "intfloat/e5-mistral-7b-instruct",
        "--host", "0.0.0.0",
        "--port", "8000",
        "--max-model-len", "4096",
        "--trust-remote-code",
    ]
    subprocess.Popen(cmd).wait()


if __name__ == "__main__":
    print("Deploying vLLM with E5-Mistral-7B embedding model...")
    deployment = serve_embeddings()

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
    input=["Hello, world!", "Semantic search is powerful"],
    model="intfloat/e5-mistral-7b-instruct",
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
    "model": "intfloat/e5-mistral-7b-instruct",
    "input": ["Hello, world!", "Semantic search is powerful"]
  }}'
''')
    print("\nTo delete the deployment:")
    print("  deployment.delete()")
