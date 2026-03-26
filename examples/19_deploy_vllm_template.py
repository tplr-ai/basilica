#!/usr/bin/env python3
"""
Deploy vLLM using the template method.

This example demonstrates the new deploy_vllm() method which provides:
- Auto-detection of GPU requirements based on model size
- Automatic storage configuration for model caching
- Pre-configured health checks and timeouts
- OpenAI-compatible API endpoints

Usage:
    export BASILICA_API_TOKEN="your-token"
    python3 19_deploy_vllm_template.py
"""
import basilica

client = basilica.BasilicaClient()

# Deploy vLLM with minimal configuration - uses defaults
print("Deploying vLLM with Qwen/Qwen3-0.6B (default model)...")
deployment = client.deploy_vllm()

print(f"vLLM deployment ready!")
print(f"  URL:          {deployment.url}")
print(f"  OpenAI API:   {deployment.url}/v1/chat/completions")
print(f"  Health:       {deployment.url}/health")

# Example with custom model and options
print("\nDeploying vLLM with custom model...")
deployment2 = client.deploy_vllm(
    model="meta-llama/Llama-2-7b-hf",
    name="llama2-7b-server",
    gpu_count=1,
    memory="32Gi",
    dtype="float16",
    trust_remote_code=True,
    ttl_seconds=3600,
)

print(f"Llama-2-7B deployment ready!")
print(f"  URL:          {deployment2.url}")
print(f"  OpenAI API:   {deployment2.url}/v1/chat/completions")

# Clean up
print("\nCleaning up deployments...")
deployment.delete()
deployment2.delete()
print("Done!")
