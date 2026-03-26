#!/usr/bin/env python3
"""
Deploy SGLang using the template method.

This example demonstrates the new deploy_sglang() method which provides:
- Auto-detection of GPU requirements based on model size
- Automatic storage configuration for model caching
- Pre-configured health checks and timeouts
- Fast inference with RadixAttention

Usage:
    export BASILICA_API_TOKEN="your-token"
    python3 20_deploy_sglang_template.py
"""
import basilica

client = basilica.BasilicaClient()

# Deploy SGLang with minimal configuration - uses defaults
print("Deploying SGLang with Qwen/Qwen2.5-0.5B-Instruct (default model)...")
deployment = client.deploy_sglang()

print(f"SGLang deployment ready!")
print(f"  URL:          {deployment.url}")
print(f"  Generate API: {deployment.url}/generate")
print(f"  Chat API:     {deployment.url}/v1/chat/completions")
print(f"  Health:       {deployment.url}/health")

# Example with custom model and options
print("\nDeploying SGLang with custom model...")
deployment2 = client.deploy_sglang(
    model="Qwen/Qwen2.5-3B-Instruct",
    name="qwen-3b-server",
    gpu_count=1,
    context_length=8192,
    mem_fraction_static=0.85,
    trust_remote_code=True,
    ttl_seconds=3600,
)

print(f"Qwen 3B deployment ready!")
print(f"  URL:          {deployment2.url}")
print(f"  Generate API: {deployment2.url}/generate")

# Clean up
print("\nCleaning up deployments...")
deployment.delete()
deployment2.delete()
print("Done!")
