"""
Distributed training example: BYO torchrun via client.deploy_distributed.

Demonstrates the lower-level path for users who want to control the
torchrun invocation explicitly (custom rdzv config, nproc-per-node tweaks,
or a non-standard launcher entirely).

Contrast with example 20 (decorator path): there, the operator's
`command="auto"` builds the torchrun command. Here, you pass `command=`
explicitly and the operator passes it through verbatim. Env vars
BASILICA_RDZV_ENDPOINT / BASILICA_WORLD_TARGET / BASILICA_RANK are still
injected by the init container so your launcher can consume them.

Prereqs:
- BASILICA_API_TOKEN set.
- A training script accessible to the trainer image (here we use the
  basilica-distributed-trainer image which ships an `all_reduce_smoke.py`
  fixture for exactly this purpose).
"""

import time

from basilica import BasilicaClient, ProviderFilter, WorldSize


def main() -> None:
    """
    Wrapped in main() so importing the module (e.g. from doc tooling or
    a test runner) does NOT spin up a paid distributed cluster. Per the
    coding guidelines, deploy/start_*_rental calls are cost-bearing and
    must be tied to an explicit script run.
    """
    client = BasilicaClient()

    # BYO-launcher: pass `command=` explicitly. The operator's CRD has
    # no "use image ENTRYPOINT, just pass args" mode -- distributed UDs
    # must specify either `source=` (auto-torchrun wrapping) or
    # `command=` (verbatim launcher). See operator's CRD § 4 / SDK
    # arch § 4 footnote on auto-torchrun wrapping. The basilica-
    # distributed-trainer image's `/workspace/all_reduce_smoke.py`
    # fixture is the smoke target.
    training = client.deploy_distributed(
        name="dlc-example-torchrun",
        image="ghcr.io/one-covenant/basilica/basilica-distributed-trainer:latest",
        command=[
            "torchrun",
            "--rdzv-backend=etcd-v2",
            "--rdzv-endpoint=$BASILICA_RDZV_ENDPOINT",
            "--rdzv-id=$BASILICA_RDZV_ID",
            "--nnodes=$BASILICA_WORLD_TARGET",
            "--nproc-per-node=$BASILICA_GPUS_PER_POD",
            "--max-restarts=10",
            "/workspace/all_reduce_smoke.py",
        ],
        world_size=WorldSize(min=2, target=2, max=4),
        gpu_count=1,
        gpu_models=["A100", "H100"],
        min_gpu_memory_gb=40,
        cpu="8",
        memory="32Gi",
        provider_filter=ProviderFilter(include=["verda"]),
        topology_spread="pack",
        nccl_env={"NCCL_DEBUG": "WARN"},
        ttl_seconds=600,
        timeout=900,
    )

    print(f"Deployed: {training.name}")
    print(f"Namespace: {training.namespace}")
    print(f"World: {training.world}")

    # Scale up by one rank, mid-run. Demonstrates Phase 2 elasticity:
    # torchelastic re-rendezvouses workers when the StatefulSet replica count
    # changes, and the new rank joins.
    time.sleep(30)
    new_world = training.scale(target=3)
    print(f"Scaled to target=3; world now: {new_world}")

    # Wait for the new rank to join.
    training.wait_until_target_world(timeout=300)
    print(f"All 3 ranks ready: {training.world}")

    # Tail logs briefly so the example surfaces "is it actually running".
    # Phase 5b: per-rank filtering not yet supported by the API; logs are
    # returned merged across ranks.
    print("--- merged logs ---")
    print(training.logs(tail=30))

    training.delete()
    print("Cleanup complete.")


if __name__ == "__main__":
    main()
