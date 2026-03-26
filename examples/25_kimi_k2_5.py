#!/usr/bin/env python3
"""
Kimi-K2-Instruct deployment - 1T parameter MoE model with tool-calling and reasoning.

Kimi-K2-Instruct is a text-only Mixture-of-Experts model from Moonshot AI:
  - Total Parameters: 1 Trillion
  - Activated Parameters: 32B per forward pass
  - Context Length: 128K tokens
  - Features: Tool calling, reasoning/thinking mode

Note: Kimi-K2.5 (multimodal) is not yet supported in vLLM.
      This example uses Kimi-K2-Instruct (text-only) instead.

Requirements:
  - 8x H200 GPUs (141GB VRAM each, 1.1TB total)
  - vLLM with kimi_k2 parser support
  - Model loading takes 15-20 minutes due to size (~500GB FP8 weights)

Usage:
    export BASILICA_API_TOKEN="your-token"
    python3 25_kimi_k2_5.py

Reference:
    https://huggingface.co/moonshotai/Kimi-K2-Instruct
    https://docs.vllm.ai/projects/recipes/en/latest/moonshotai/Kimi-K2.html
"""
import basilica
from basilica import (
    BasilicaClient,
    CreateDeploymentRequest,
    GpuRequirementsSpec,
    HealthCheckConfig,
    ProbeConfig,
    ResourceRequirements,
)


def deploy_kimi_k2():
    """Deploy Kimi-K2-Instruct with custom configuration for large model loading."""
    client = BasilicaClient()

    model = "moonshotai/Kimi-K2-Instruct"

    args = [
        "serve", model,
        "--host", "0.0.0.0",
        "--port", "8000",
        "--tensor-parallel-size", "8",
        "--trust-remote-code",
        "--tool-call-parser", "kimi_k2",
        "--enable-auto-tool-choice",
        "--max-model-len", "32768",
        "--gpu-memory-utilization", "0.95",
        "--enforce-eager",
    ]

    gpu_spec = GpuRequirementsSpec(
        count=8,
        model=["H200"],
        min_cuda_version=None,
        min_gpu_memory_gb=80,
    )

    resources = ResourceRequirements(
        cpu="32",
        memory="512Gi",
        gpus=gpu_spec,
    )

    # Extended health check timeout for 1T model loading (90 min initial delay)
    # Kimi-K2 needs ~33 min download + ~22 min shard loading = ~55 min total
    health_check = HealthCheckConfig(
        liveness=ProbeConfig(
            path="/health",
            port=8000,
            initial_delay_seconds=5400,  # 90 min delay for model loading
            period_seconds=30,
            timeout_seconds=10,
            failure_threshold=3,
        ),
        readiness=ProbeConfig(
            path="/health",
            port=8000,
            initial_delay_seconds=5400,  # 90 min delay for model loading
            period_seconds=10,
            timeout_seconds=5,
            failure_threshold=3,
        ),
    )

    # Increase HuggingFace download timeout to avoid ReadTimeoutError
    env = {
        "HF_HUB_DOWNLOAD_TIMEOUT": "3600",  # 1 hour timeout for large files
    }

    request = CreateDeploymentRequest(
        instance_name="kimi-k2-instruct",
        image="vllm/vllm-openai:latest",
        replicas=1,
        port=8000,
        command=["vllm"],
        args=args,
        env=env,
        resources=resources,
        ttl_seconds=7200,
        public=True,
        storage=None,
        health_check=health_check,
    )

    print(f"Creating Kimi-K2-Instruct deployment...")
    print(f"  Model: {model}")
    print(f"  GPUs:  8x H200")
    print(f"  This will take 15-20 minutes for model loading...")
    print()

    response = client._client.create_deployment(request)
    deployment = basilica.Deployment._from_response(client, response)

    print(f"Deployment created: {deployment.name}")
    print(f"URL: {deployment.url}")
    print()
    print("Waiting for model to load (this takes 15-20 minutes)...")
    print("Monitor with: basilica deploy logs kimi-k2-instruct --follow")
    print()

    try:
        deployment.wait_until_ready(timeout=2400, silent=False)
    except basilica.exceptions.DeploymentFailed:
        print()
        print("Deployment is still loading the model.")
        print("Large models like Kimi-K2 require extended startup time.")
        print()
        print("Check progress with:")
        print(f"  basilica deploy logs {deployment.name} --follow")
        print()
        print("When ready, the API will be available at:")
        print(f"  {deployment.url}/v1/chat/completions")
        return deployment

    return deployment


if __name__ == "__main__":
    deployment = deploy_kimi_k2()

    model = "moonshotai/Kimi-K2-Instruct"
    if deployment.state == "Ready":
        print()
        print(f"Kimi-K2-Instruct deployment ready!")
        print(f"  Name:    {deployment.name}")
        print(f"  URL:     {deployment.url}")
        print(f"  State:   {deployment.state}")
        print()
        print("OpenAI-compatible API endpoints:")
        print(f"  Chat:    {deployment.url}/v1/chat/completions")
        print(f"  Models:  {deployment.url}/v1/models")
        print(f"  Health:  {deployment.url}/health")
        print()
        print("Example usage (thinking mode enabled by default):")
        print(f'  curl {deployment.url}/v1/chat/completions \\')
        print('    -H "Content-Type: application/json" \\')
        print(f'    -d \'{{"model": "{model}", "messages": [{{"role": "user", "content": "Solve step by step: What is 25 * 37?"}}]}}\'')
