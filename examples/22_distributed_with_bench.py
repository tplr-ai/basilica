"""
Distributed training example: per-UD NCCL bench probe (SDK arch § 7).

Demonstrates the per-UD bench surface. Launching a UD with
`bench="on-start"` schedules a 2-rank NCCL `all_reduce_perf` Job in the
user's namespace alongside the workers. The probe runs in parallel; the
result lands on `training.bench` once the probe Job completes.

Why this matters: the platform-internal NCCL benchmark matrix
(architecture doc § 12) does NOT serve user-facing queries -- a
shared cluster-wide cache would violate the platform's tenancy
invariant (SDK arch § 1). User-visible bench data is per-UD,
measured on the user's actual nodes, billed against the user's
GPU minutes, scoped to the user's namespace.

There is intentionally NO `client.preflight(...)` and NO
`client.nccl_baseline(...)` standalone helper. That surface implied a
shared cache. Use this per-UD pattern instead.

Bench wait is OPT-IN (refs #506): `deploy_distributed` returns as soon
as workers are Ready, regardless of bench probe state. This script
demonstrates the opt-in pattern via
`training.wait_until_bench_complete(...)`. A bench failure no longer
makes the deploy look failed -- the workers are healthy and you can
choose what to do with the measurement (or its absence).

Prereqs:
- BASILICA_API_TOKEN set.
- Account with verda A100 access (or adjust the provider filter).
- Namespace rank budget >= 4 (worker(2) + bench(2)).
"""

import json

from basilica import BasilicaClient, ProviderFilter, WorldSize


def main() -> None:
    """
    Wrapped in main() so importing the module does NOT spin up a paid
    distributed cluster. Per the coding guidelines, deploy/start_*_rental
    calls are cost-bearing and must be tied to an explicit script run.
    """
    client = BasilicaClient()

    # `source=` (string) is the recommended path for typical SDK users:
    # the SDK ships the source via base64 to /tmp/__basilica_source.py and
    # the operator's BYO renderer exec's torchrun on it. RANK / WORLD_SIZE
    # / MASTER_* are set by torchrun -- the user code just reads them.
    #
    # Contrast with `command=["python3", "/workspace/foo.py"]` which skips
    # torchrun entirely; ranks would crash with `RANK env var missing`.
    # For users who want full control over the launcher, see example 21
    # (BYO torchrun via `command=[...]`).
    training = client.deploy_distributed(
        name="dlc-example-bench",
        image="ghcr.io/one-covenant/basilica/basilica-distributed-trainer:latest",
        source="""\
import os
import time

import torch
import torch.distributed as dist


def main() -> None:
    rank = int(os.environ["RANK"])
    world_size = int(os.environ["WORLD_SIZE"])
    local_rank = int(os.environ.get("LOCAL_RANK", 0))

    dist.init_process_group(backend="nccl")
    device = torch.device(f"cuda:{local_rank}")
    print(f"[rank {rank}] joined; world={world_size} device={device}", flush=True)

    # Brief NCCL all_reduce loop -- proves the workers rendezvoused and
    # the collective fabric is up. Each step sums a 1024-float tensor
    # across all ranks; expected sum at step k is world_size (because we
    # start from ones and re-fill each step).
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


if __name__ == "__main__":
    main()
""",
        world_size=WorldSize(min=2, target=2, max=2),
        gpu_count=1,
        gpu_models=["A100"],
        min_gpu_memory_gb=40,
        cpu="8",
        memory="32Gi",
        provider_filter=ProviderFilter(include=["verda"]),
        topology_spread="pack",  # Pack for the smallest measured pair.
        bench="on-start",  # Schedule the 2-rank NCCL probe.
        nccl_env={"NCCL_DEBUG": "WARN"},
        ttl_seconds=900,
        timeout=900,
    )

    print(f"Deployed: {training.name}")
    print("Bench probe scheduled (rank-budget cost: worker(2) + bench(2) = 4)")

    # Refs #506: deploy_distributed now returns when workers are Ready;
    # the bench probe is decoupled. Use the opt-in waiter to block on it.
    # `bench_timeout=1200` (20 min) matches the prior implicit budget.
    print("Waiting for bench probe (opt-in via wait_until_bench_complete)...")
    bench_status = training.wait_until_bench_complete(timeout=1200)

    if bench_status is None:
        # mode=off — caller did not request a bench probe. Not reachable
        # in this example (we set bench="on-start" above), but kept for
        # symmetry with the API contract.
        print("Bench was not enabled on this UD.")
        training.delete()
        raise SystemExit(1)

    if bench_status.phase != "Succeeded":
        # Workers were Ready -- the deploy is fine -- but the bench
        # didn't measure. Print the lifecycle detail; do NOT exit non-zero
        # for a bench-only failure (the workload itself is healthy).
        print("Bench probe did NOT complete with a measurement.")
        print(f"  phase:              {bench_status.phase}")
        print(f"  message:            {bench_status.message}")
        print(f"  last_attempt_at:    {bench_status.last_attempt_at}")
        print(f"  last_attempt:       {bench_status.last_attempt_outcome}")
        print("Workers are still Ready; the UD itself is healthy.")
        print("Inspect via `kubectl get job -n <ns> -l basilica.ai/distributed-role=bench`.")
        training.delete()
        return

    # bench_status.phase == "Succeeded"; the result payload is on
    # `training.bench` (or `bench_status.result`).
    result = training.bench
    if result is None:
        # Defensive: phase=Succeeded but no result payload -- this would
        # be an operator-side invariant break. Surface it explicitly.
        print("Bench phase=Succeeded but no result payload available.")
        training.delete()
        raise SystemExit(1)
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

    # Researchers writing papers can serialize this to JSON and aggregate
    # across many UDs offline -- each measurement is on their own nodes,
    # billed against their own usage, with no cross-tenant shared cache.
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

    training.delete()
    print("Cleanup complete.")


if __name__ == "__main__":
    main()
