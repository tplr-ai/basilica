"""
Distributed training example: DiLoCo with @basilica.distributed.

Demonstrates the decorator-style entry point for distributed training:
- Decorate a function that becomes the per-rank entrypoint.
- Calling the wrapper deploys; the operator wraps with torchrun.
- Returns DistributedTraining with scale/wait/logs/bench.

Runs in well under 10 minutes on a 4-H100 cluster: 20 outer DiLoCo steps
on a tiny 6M-parameter MLP. The point is to verify the SDK -> API ->
operator -> rendezvous -> NCCL chain end-to-end, not to train anything
useful.

Prereqs:
- BASILICA_API_TOKEN set, account with H100 access.
- ~4 ranks of A100 or H100 available.

Cleanup:
- The function returns DistributedTraining; call .delete() when done
  (or rely on ttl_seconds=600 below).
"""

import os
import time

import basilica
from basilica import ProviderFilter, WorldSize


@basilica.distributed(
    name="dlc-example-diloco",
    # The basilica-distributed-trainer image is the canonical base for
    # distributed UDs: it ships torch + `python-etcd` (the latter is
    # required by torchrun's `--rdzv-backend=etcd` legacy backend that
    # the operator currently maps `etcd-v2` to; see operator
    # distributed.rs build_worker_command). A bare pytorch image does
    # not include `python-etcd` so torchrun would fail at rendezvous.
    image="ghcr.io/one-covenant/basilica/basilica-distributed-trainer:latest",
    world_size=WorldSize(min=2, target=4, max=4),
    gpu_count=1,
    gpu_models=["H100", "A100"],
    min_gpu_memory_gb=40,
    cpu="8",
    memory="32Gi",
    provider_filter=ProviderFilter(include=["hyperstack", "verda"]),
    topology_spread="provider-aware",
    bench="on-start",  # 2-rank NCCL bench probe; result on training.bench.
    nccl_env={"NCCL_DEBUG": "WARN"},
    ttl_seconds=900,
    timeout=900,
)
def train() -> None:
    """Per-rank DiLoCo step. Each rank executes this body once under torchrun."""
    import torch
    import torch.distributed as dist
    import torch.nn as nn

    dist.init_process_group(backend="nccl")
    rank = dist.get_rank()
    world_size = dist.get_world_size()
    local_rank = int(os.environ.get("LOCAL_RANK", 0))
    device = torch.device(f"cuda:{local_rank}")

    # 6M-param MLP; small enough that NCCL all_reduce is the bottleneck,
    # so an all-reduce-bound workload is what we exercise.
    model = nn.Sequential(
        nn.Linear(1024, 2048),
        nn.GELU(),
        nn.Linear(2048, 1024),
    ).to(device)

    inner = torch.optim.SGD(model.parameters(), lr=0.01)
    outer = torch.optim.SGD(model.parameters(), lr=0.7, momentum=0.9, nesterov=True)

    K_INNER = 50
    OUTER_STEPS = 20

    for outer_step in range(OUTER_STEPS):
        # Snapshot pre-DiLoCo state.
        anchor = [p.detach().clone() for p in model.parameters()]

        # K inner SGD steps (no communication).
        for _ in range(K_INNER):
            x = torch.randn(32, 1024, device=device)
            target = torch.randn(32, 1024, device=device)
            loss = nn.functional.mse_loss(model(x), target)
            inner.zero_grad()
            loss.backward()
            inner.step()

        # DiLoCo outer-step: average pseudo-gradients across ranks, then
        # apply outer optimizer. This is the only NCCL collective; the
        # all_reduce here is what we're stress-testing.
        with torch.no_grad():
            for p, a in zip(model.parameters(), anchor):
                pseudo_grad = a - p
                dist.all_reduce(pseudo_grad, op=dist.ReduceOp.AVG)
                p.copy_(a)
                if p.grad is None:
                    p.grad = torch.zeros_like(p)
                p.grad.copy_(pseudo_grad)
            outer.step()

        if rank == 0 and outer_step % 5 == 0:
            print(f"[rank 0] outer_step={outer_step}/{OUTER_STEPS} "
                  f"world_size={world_size}", flush=True)

    if rank == 0:
        print("[rank 0] training complete", flush=True)
    dist.destroy_process_group()


if __name__ == "__main__":
    # The decorator call deploys; the returned DistributedTraining lets
    # us follow + clean up.
    training = train()
    print(f"Deployed: {training.name}")
    print(f"Namespace: {training.namespace}")
    print(f"Rendezvous: {training.rendezvous_endpoint}")
    print(f"World: {training.world}")

    # Wait for the run to finish (heuristic: poll until ranks all exit
    # 'Running' state, or 30 minutes elapsed -- whichever first).
    deadline = time.time() + 1800
    while time.time() < deadline:
        training.refresh()
        ws = training.world
        if ws.ready == 0:
            print("All ranks finished; cleaning up.")
            break
        time.sleep(15)

    training.delete()
    print("Cleanup complete.")
