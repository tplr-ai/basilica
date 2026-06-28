"""
Distributed training example: per-UD NCCL bench probe (SDK arch § 7).

Demonstrates the per-UD bench surface via the canonical decorator.
``bench=True`` schedules a 2-rank NCCL ``all_reduce_perf`` Job in the
user's namespace alongside the workers. The probe runs in parallel; the
result lands on ``training.bench`` once the UD reaches a terminal state.

Why this matters: the platform-internal NCCL benchmark matrix
(architecture doc § 12) does NOT serve user-facing queries -- a
shared cluster-wide cache would violate the platform's tenancy
invariant (SDK arch § 1). User-visible bench data is per-UD, measured
on the user's actual nodes, billed against the user's GPU minutes,
scoped to the user's namespace.

There is intentionally NO ``client.preflight(...)`` and NO
``client.nccl_baseline(...)`` standalone helper. That surface implied
a shared cache. Use this per-UD pattern instead.

Bench is OPT-IN. ``bench=True`` schedules the probe; ``bench=False``
(default) skips it. The result reads back as ``training.bench``
(``BenchResult | None``); ``None`` means "no measurement" regardless
of why (workers too short, probe couldn't co-schedule, etc.). For
debug detail on a ``None`` result, inspect ``training.bench_diagnostics``.

Prereqs:
- ``BASILICA_API_TOKEN`` set.
- Account with access to the selected public AZ root (or adjust the provider filter).
- Namespace rank budget >= 4 (worker(2) + bench(2)).
"""

import json

import basilica
from basilica import ProviderFilter, WorldSize


@basilica.distributed(
    name="dlc-example-bench",
    image="ghcr.io/one-covenant/basilica/basilica-distributed-trainer:latest",
    world_size=WorldSize(min=2, target=2, max=2),
    gpu_count=1,
    gpu_models=["A100"],
    min_gpu_memory_gb=40,
    cpu="8",
    memory="32Gi",
    provider_filter=ProviderFilter(include=["cyan"]),
    topology_spread="pack",  # Pack for the smallest measured pair.
    bench=True,  # Schedule the 2-rank NCCL probe.
    nccl_env={"NCCL_DEBUG": "WARN"},
    ttl_seconds=900,
    timeout=900,
)
def workload() -> None:
    """
    Per-rank entrypoint. Each rank runs this body under torchrun;
    ``RANK`` / ``WORLD_SIZE`` / ``LOCAL_RANK`` / ``MASTER_*`` are wired
    by torchrun. The body just reads them from ``os.environ``.

    Contrast with example 21 (BYO ``command=`` for full launcher
    control); here the decorator hands the body to the operator's
    ``command="auto"`` torchrun wrapper.
    """
    import os
    import time

    import torch
    import torch.distributed as dist

    rank = int(os.environ["RANK"])
    world_size = int(os.environ["WORLD_SIZE"])
    local_rank = int(os.environ.get("LOCAL_RANK", 0))

    dist.init_process_group(backend="nccl")
    device = torch.device(f"cuda:{local_rank}")
    print(f"[rank {rank}] joined; world={world_size} device={device}", flush=True)

    # Brief NCCL all_reduce loop -- proves the workers rendezvoused and
    # the collective fabric is up. Each step sums a 1024-float tensor
    # across all ranks; expected sum at step k is world_size * 1024
    # (because we start from ones and re-fill each step).
    for step in range(5):
        x = torch.ones(1024, device=device)
        dist.all_reduce(x)
        torch.cuda.synchronize()
        print(
            f"[rank {rank}] step {step + 1}/5 sum={x.sum().item():.0f} "
            f"(expected {world_size * 1024})",
            flush=True,
        )
        time.sleep(2)

    dist.destroy_process_group()
    print(f"[rank {rank}] done", flush=True)


def main() -> None:
    """
    Deploy + collect the bench measurement. Wrapped in ``main()`` so
    importing the module does NOT spin up a paid distributed cluster.
    """
    # ``workload()`` deploys; the ``with`` block calls
    # ``training.delete()`` on scope exit (success OR exception), so
    # the UD does not leak when a mid-run wait raises.
    with workload() as training:
        print(f"Deployed: {training.name}")
        print("Bench probe scheduled (rank-budget cost: worker(2) + bench(2) = 4)")

        # Block until workers reach a terminal phase. Raises
        # ``BelowMinimumWorld`` on failure or timeout; the ``with``
        # block's ``__exit__`` cleans up either way.
        training.wait_until_complete(timeout=1800)
        print(f"Workers done: {training.world}")

        # ``training.bench`` is the canonical surface for the bench
        # measurement. ``None`` means "no measurement" regardless of
        # why -- workers too short, probe couldn't co-schedule, etc.
        # 99% of users only ever read this attribute.
        result = training.bench
        if result is None:
            print("Bench did not measure on this run.")
            # The rare debug case: inspect ``bench_diagnostics`` for
            # operator-side detail on why the probe didn't land a result.
            if training.bench_diagnostics is not None:
                print(f"Bench diagnostics: {training.bench_diagnostics}")
            return

        print("--- Bench result ---")
        print(f"  measured_at:        {result.measured_at}")
        print(f"  busbw_gbps_p10:     {result.busbw_gbps_p10}")
        print(f"  busbw_gbps_p50:     {result.busbw_gbps_p50}")
        print(f"  busbw_gbps_p90:     {result.busbw_gbps_p90}")
        print(f"  algbw_gbps_p50:     {result.algbw_gbps_p50}")
        print(f"  latency_us_at_1mib: {result.latency_us_at_1mib}")
        print(f"  size_bytes_swept:   {result.size_bytes_swept}")
        print(f"  probe_node_a:       {result.probe_node_a}")
        print(f"  probe_node_b:       {result.probe_node_b}")

        # Researchers writing papers can serialize this to JSON and
        # aggregate across many UDs offline -- each measurement is on
        # their own nodes, billed against their own usage, with no
        # cross-tenant shared cache.
        out = {
            "name": training.name,
            "world_size": {
                "min": training.world.min,
                "target": training.world.target,
                "max": training.world.max,
            },
            "bench": {
                "busbw_gbps_p50": result.busbw_gbps_p50,
                "latency_us_at_1mib": result.latency_us_at_1mib,
                "probe_node_a": result.probe_node_a,
                "probe_node_b": result.probe_node_b,
            },
        }
        with open(f"/tmp/{training.name}-bench.json", "w") as f:
            json.dump(out, f, indent=2, default=str)
        print(f"Saved bench result: /tmp/{training.name}-bench.json")


if __name__ == "__main__":
    main()
