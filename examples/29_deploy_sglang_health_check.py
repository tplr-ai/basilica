"""
Deploy SGLang with custom health checks and reliability mitigations.

Large language models can take several minutes to download and load into GPU
memory. This example demonstrates:
  1. Custom health probes to prevent premature pod kills
  2. Robust model download with retry logic
  3. Local disk caching (avoids FUSE reliability issues)

Why NOT use FUSE-based storage for model downloads:
  - FUSE mounts can disconnect mid-download (Errno 107: Transport endpoint not connected)
  - Network storage adds latency to large file operations
  - Local disk is more reliable for the download phase

Trade-off: Models re-download on pod restart, but downloads are reliable.

Usage:
    export BASILICA_API_TOKEN="basilica_..."
    python deploy_sglang_health_check.py

Real-world timing observations:
  - Model download:   ~10 minutes (Qwen2.5-3B on typical network)
  - SGLang loading:   ~5 minutes (weight loading, KV cache, CUDA graphs)
  - Total startup:    ~15 minutes

Kubernetes Health Probe Behavior:
  - startup:   Runs first. Liveness/readiness are DISABLED until startup succeeds.
  - liveness:  Runs AFTER startup succeeds. Detects crashes/hangs. Failure = restart.
  - readiness: Runs AFTER startup succeeds. Controls traffic routing. Failure = no traffic.

Key Formula:
  max_startup_time = initial_delay_seconds + (failure_threshold * period_seconds)
"""

from basilica import BasilicaClient, HealthCheckConfig, ProbeConfig

SGLANG_PORT = 8000


def build_sglang_health_check(startup_minutes: int = 30) -> HealthCheckConfig:
    """Build an SGLang health check config with a configurable startup window.

    Tested configuration values based on real-world deployment experience:
      - initial_delay_seconds=480 (8 min): Skips most of the download phase
      - period_seconds=120 (2 min): Reasonable probe interval
      - timeout_seconds=120 (2 min): Allows for slow /health responses under load
      - failure_threshold: Calculated to provide startup_minutes total window

    Args:
        startup_minutes: Maximum minutes to wait for model loading.
                        Recommended values by model size:
                          - 3B models: 20 minutes (warm) / 30 minutes (cold)
                          - 7B models: 25 minutes (warm) / 35 minutes (cold)
                          - 70B models: 45 minutes (warm) / 60 minutes (cold)
                        "Cold" = first deployment (download required)
                        "Warm" = model cached in persistent storage

    Returns:
        HealthCheckConfig with tested probe settings.

    Timing example (startup_minutes=30):
        initial_delay = 480s (8 min)
        period = 120s (2 min)
        failure_threshold = (30*60 - 480) / 120 = 11
        max_time = 480 + 11*120 = 1800s (30 min)
    """
    initial_delay = 480
    period = 120
    timeout = 120
    failure_threshold = max(1, (startup_minutes * 60 - initial_delay) // period)

    return HealthCheckConfig(
        startup=ProbeConfig(
            path="/health",
            port=SGLANG_PORT,
            initial_delay_seconds=initial_delay,
            period_seconds=period,
            timeout_seconds=timeout,
            failure_threshold=failure_threshold,
        ),
        liveness=ProbeConfig(
            path="/health",
            port=SGLANG_PORT,
            initial_delay_seconds=initial_delay,
            period_seconds=period,
            timeout_seconds=timeout,
            failure_threshold=5,
        ),
        readiness=ProbeConfig(
            path="/health",
            port=SGLANG_PORT,
            initial_delay_seconds=initial_delay,
            period_seconds=period,
            timeout_seconds=timeout,
            failure_threshold=5,
        ),
    )


def create_sglang_deployment_source(
    base_model: str,
    seed: int = 42,
    max_retries: int = 3,
) -> str:
    """Create source code for SGLang deployment with robust model download.

    Uses local disk for HuggingFace cache (not FUSE storage) for reliability.

    Args:
        base_model: HuggingFace model name or path
        seed: Random seed for deterministic inference
        max_retries: Number of download retry attempts (default: 3)
    """
    return f'''
import subprocess
import os
import time

MODEL = "{base_model}"
SEED = {seed}
MAX_RETRIES = {max_retries}

def setup_environment():
    """Configure environment for determinism."""
    os.environ["PYTHONHASHSEED"] = str(SEED)
    os.environ["CUBLAS_WORKSPACE_CONFIG"] = ":4096:8"
    os.environ["NVIDIA_TF32_OVERRIDE"] = "0"

def download_model_with_retry():
    """Download model with retry logic for reliability."""
    from huggingface_hub import snapshot_download

    for attempt in range(1, MAX_RETRIES + 1):
        try:
            print(f"Downloading model (attempt {{attempt}}/{{MAX_RETRIES}}): {{MODEL}}", flush=True)
            start = time.time()
            path = snapshot_download(MODEL, local_files_only=False)
            elapsed = time.time() - start
            print(f"Model downloaded in {{elapsed:.1f}}s: {{path}}", flush=True)
            return path
        except Exception as e:
            print(f"Download attempt {{attempt}} failed: {{e}}", flush=True)
            if attempt < MAX_RETRIES:
                wait = 30 * attempt
                print(f"Retrying in {{wait}}s...", flush=True)
                time.sleep(wait)
            else:
                print("All download attempts failed", flush=True)
                raise

def start_sglang_server():
    """Start SGLang server."""
    cmd = [
        "python3", "-m", "sglang.launch_server",
        "--model-path", MODEL,
        "--host", "0.0.0.0",
        "--port", "{SGLANG_PORT}",
        "--tensor-parallel-size", "1",
        "--dtype", "float16",
        "--enable-deterministic-inference",
        "--random-seed", str(SEED),
    ]
    print("Starting SGLang server...", flush=True)
    subprocess.run(cmd, check=True)

if __name__ == "__main__":
    setup_environment()
    download_model_with_retry()
    start_sglang_server()
'''


def main() -> None:
    client = BasilicaClient()

    health_check = build_sglang_health_check(startup_minutes=30)
    func_source = create_sglang_deployment_source(
        base_model="Qwen/Qwen2.5-3B-Instruct",
        max_retries=3,
    )

    startup_max = (
        health_check.startup.initial_delay_seconds
        + health_check.startup.failure_threshold * health_check.startup.period_seconds
    )

    print("Deploying SGLang with Qwen/Qwen2.5-3B-Instruct...")
    print()
    print("Mitigations enabled:")
    print("  - Model pre-download with retry (3 attempts, 30s backoff)")
    print("  - Local disk cache (no FUSE - more reliable)")
    print("  - 30-minute startup window for download + loading")
    print()
    print("Health probe configuration:")
    print(f"  Startup: {startup_max}s max ({startup_max // 60} min)")
    print(f"    initial_delay={health_check.startup.initial_delay_seconds}s, "
          f"period={health_check.startup.period_seconds}s, "
          f"failures={health_check.startup.failure_threshold}")
    print()

    deployment = client.deploy(
        name="sglang-qwen2.5-3b-instruct-health-check",
        source=func_source,
        image="lmsysorg/sglang:latest",
        port=SGLANG_PORT,
        health_check=health_check,
        timeout=startup_max + 60,
        ttl_seconds=1800,
        gpu_count=1,
        gpu_models=["A100"],
        min_gpu_memory_gb=80,
        cpu="2",
        memory="64Gi",
        env={
            "HF_HUB_DISABLE_SYMLINKS_WARNING": "1",
            "HF_HUB_DISABLE_XET": "1",
            "PYTHONHASHSEED": "42",
            "CUBLAS_WORKSPACE_CONFIG": ":4096:8",
            "NVIDIA_TF32_OVERRIDE": "0",
        },
    )

    print(f"Deployment ready: {deployment.name}")
    print(f"URL:              {deployment.url}")
    print(f"State:            {deployment.state}")
    print()
    print("Test with:")
    print(f"  curl {deployment.url}/v1/models")
    print()
    print(f"  curl {deployment.url}/v1/chat/completions \\")
    print('    -H "Content-Type: application/json" \\')
    print("    -d '{")
    print('      \"model\": \"Qwen/Qwen2.5-3B-Instruct\",')
    print('      \"messages\": [{\"role\": \"user\", \"content\": \"Hello!\"}]')
    print("    }'")


if __name__ == "__main__":
    main()
