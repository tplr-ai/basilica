#!/usr/bin/env python3
"""
Kimi-K2.5 deployment - 1T parameter multimodal MoE model with thinking and agent capabilities.

Kimi-K2.5 is a native multimodal agentic model from Moonshot AI:
  - Total Parameters: 1 Trillion
  - Activated Parameters: 32B per forward pass
  - Context Length: 256K tokens (using 131K for memory safety)
  - Features: Thinking mode, tool calling, vision (experimental), agent swarm
  - Training: 15T visual-text tokens on top of Kimi-K2-Base

Differences from Kimi-K2-Instruct:
  - K2.5 is a THINKING model (requires --reasoning-parser kimi_k2)
  - K2.5 has native multimodal support (vision encoder)
  - K2.5 supports 256K context (vs 128K)
  - K2.5 has agent swarm capabilities

Requirements:
  - 8x H200 GPUs (141GB VRAM each, 1.1TB total)
  - vLLM nightly (KimiK25ForConditionalGeneration architecture not in stable releases)
  - Model loading takes ~60 minutes due to size (~500GB FP8 weights)

Usage:
    export BASILICA_API_TOKEN="your-token"
    python3 26_kimi_k2_5_multimodal.py

Reference:
    https://huggingface.co/moonshotai/Kimi-K2.5
    https://huggingface.co/moonshotai/Kimi-K2.5/blob/main/docs/deploy_guidance.md
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


def deploy_kimi_k2_5() -> basilica.Deployment:
    """Deploy Kimi-K2.5 with configuration per official Moonshot guidance."""
    client = BasilicaClient()

    model = "moonshotai/Kimi-K2.5"

    # Official K2.5 vLLM args per Moonshot deployment guidance:
    # https://huggingface.co/moonshotai/Kimi-K2.5/blob/main/docs/deploy_guidance.md
    args = [
        "serve", model,
        "--host", "0.0.0.0",
        "--port", "8000",
        # Tensor parallelism for 8 GPUs
        "--tensor-parallel-size", "8",
        # Required for custom model code
        "--trust-remote-code",
        # Required for K2.5 tool calling
        "--tool-call-parser", "kimi_k2",
        "--enable-auto-tool-choice",
        # Required for K2.5 thinking/reasoning mode
        "--reasoning-parser", "kimi_k2",
        # Context length: K2.5 supports 256K, using 131K for memory safety
        "--max-model-len", "131072",
        # GPU memory settings
        "--gpu-memory-utilization", "0.92",
    ]

    gpu_spec = GpuRequirementsSpec(
        count=8,
        model=["H200"],
        min_cuda_version=None,
        min_gpu_memory_gb=128,
    )

    resources = ResourceRequirements(
        cpu="32",
        memory="512Gi",
        gpus=gpu_spec,
    )

    # Extended health check for 1T model loading (~60 min total)
    # K2.5 needs ~35 min download + ~25 min shard loading
    health_check = HealthCheckConfig(
        liveness=ProbeConfig(
            path="/health",
            port=8000,
            initial_delay_seconds=5400,  # 90 min delay
            period_seconds=30,
            timeout_seconds=10,
            failure_threshold=3,
        ),
        readiness=ProbeConfig(
            path="/health",
            port=8000,
            initial_delay_seconds=5400,  # 90 min delay
            period_seconds=10,
            timeout_seconds=5,
            failure_threshold=3,
        ),
    )

    env = {
        "HF_HUB_DOWNLOAD_TIMEOUT": "3600",  # 1 hour timeout for large files
    }

    request = CreateDeploymentRequest(
        instance_name="kimi-k2-5",
        # K2.5 requires vLLM nightly for KimiK25ForConditionalGeneration architecture
        # Custom image built from examples/docker/kimi-k2.5-vllm/Dockerfile
        image="ghcr.io/one-covenant/kimi-k2.5-vllm:latest",
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

    print("Creating Kimi-K2.5 deployment...")
    print(f"  Model: {model}")
    print("  GPUs:  8x H200")
    print("  Context: 131K tokens")
    print("  Features: Thinking mode, tool calling")
    print()

    response = client._client.create_deployment(request)
    deployment = basilica.Deployment._from_response(client, response)

    print(f"Deployment created: {deployment.name}")
    print(f"URL: {deployment.url}")
    print()
    print("Waiting for model to load...")
    print(f"Monitor with: basilica deploy logs {deployment.name} --follow")
    print()

    try:
        deployment.wait_until_ready(timeout=5400, silent=False)
    except basilica.exceptions.DeploymentFailed:
        print()
        print("Deployment is still loading the model.")
        print("Large models like Kimi-K2.5 require extended startup time.")
        print()
        print("Check progress with:")
        print(f"  basilica deploy logs {deployment.name} --follow")
        print()
        print("When ready, the API will be available at:")
        print(f"  {deployment.url}/v1/chat/completions")
        return deployment

    return deployment


def print_usage(deployment: basilica.Deployment, model: str) -> None:
    """Print usage examples for the deployed model."""
    print()
    print("Kimi-K2.5 deployment ready!")
    print(f"  Name:    {deployment.name}")
    print(f"  URL:     {deployment.url}")
    print(f"  State:   {deployment.state}")
    print()
    print("API endpoints:")
    print(f"  Chat:    {deployment.url}/v1/chat/completions")
    print(f"  Models:  {deployment.url}/v1/models")
    print(f"  Health:  {deployment.url}/health")
    print()
    print("Example - Thinking mode (default):")
    print(f"  curl {deployment.url}/v1/chat/completions \\")
    print('    -H "Content-Type: application/json" \\')
    print(f"    -d '{{")
    print(f'      "model": "{model}",')
    print('      "messages": [{"role": "user", "content": "Solve: What is 123 * 456?"}],')
    print('      "temperature": 1.0')
    print("    }'")
    print()
    print("Example - Instant mode (disable thinking):")
    print(f"  curl {deployment.url}/v1/chat/completions \\")
    print('    -H "Content-Type: application/json" \\')
    print(f"    -d '{{")
    print(f'      "model": "{model}",')
    print('      "messages": [{"role": "user", "content": "Hello!"}],')
    print('      "temperature": 0.6,')
    print('      "extra_body": {"chat_template_kwargs": {"thinking": false}}')
    print("    }'")


if __name__ == "__main__":
    deployment = deploy_kimi_k2_5()
    model = "moonshotai/Kimi-K2.5"

    if deployment.state == "Ready":
        print_usage(deployment, model)
