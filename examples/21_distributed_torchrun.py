"""
Distributed training example: BYO torchrun via ``basilica.distributed(command=...)``.

Demonstrates the canonical factory shape for users who want to control
the torchrun invocation explicitly (custom rdzv config, nproc-per-node
tweaks, or a non-standard launcher entirely).

Contrast with example 20 (decorator path): there, the operator's
``command="auto"`` builds the torchrun command around the decorated
function body. Here, you pass ``command=`` explicitly and the operator
passes it through verbatim. Env vars ``BASILICA_RDZV_ENDPOINT`` /
``BASILICA_WORLD_TARGET`` / ``BASILICA_RANK`` are still injected by the
init container so your launcher can consume them.

When ``command`` is set, ``basilica.distributed(...)`` short-circuits
the decorator path and returns a :class:`DistributedTraining` directly
(factory mode). Use it under a ``with`` block to get auto-cleanup.

Prereqs:
- ``BASILICA_API_TOKEN`` set.
- A training script accessible to the trainer image (here we use the
  basilica-distributed-trainer image which ships an
  ``all_reduce_smoke.py`` fixture for exactly this purpose).
"""

import time

import basilica
from basilica import ProviderFilter, WorldSize


def main() -> None:
    """
    Wrapped in ``main()`` so importing the module (e.g. from doc tooling
    or a test runner) does NOT spin up a paid distributed cluster. Per
    the coding guidelines, deploy/start_*_rental calls are cost-bearing
    and must be tied to an explicit script run.
    """
    # BYO launcher: pass ``command=`` to ``basilica.distributed(...)``.
    # The operator's CRD has no "use image ENTRYPOINT, just pass args"
    # mode -- distributed UDs must specify either an auto-torchrun-wrapped
    # function body (decorator path; see example 20) or ``command=``
    # (verbatim launcher). See operator's CRD § 4 / SDK arch § 4 footnote
    # on auto-torchrun wrapping. The basilica-distributed-trainer image's
    # ``/workspace/all_reduce_smoke.py`` fixture is the smoke target.
    #
    # NOTE on ``--rdzv-backend=etcd`` (not ``etcd-v2``): refs #368 / #490.
    # The torchelastic ``etcd-v2`` backend (DynamicRendezvousHandler over
    # etcd v3 gRPC) has an upstream regression in torch 2.5.0a0+nv24.10
    # that returns RendezvousClosedError on the FIRST connect to a fresh
    # etcd. The legacy ``etcd`` backend (EtcdRendezvousHandler over
    # python-etcd / v2 KV API) works against the same etcd Pod with
    # ``--enable-v2=true``. The operator's "auto" command-build path
    # (basilica-operator: ``distributed.rs::build_worker_command``) maps
    # the CRD value ``etcd-v2`` -> the working ``etcd`` torchrun arg
    # until upstream resolves it; BYO commands (this example) need to do
    # the same. When #368 closes, flip both back to ``etcd-v2``.
    #
    # NOTE on ``--rdzv-conf=timeout=1500``: refs #490. torchelastic's
    # rendezvous handlers default to a ~600s join timeout; on slow
    # cold-starts (image pull blip, transient registry slowness, GPU
    # node first boot) rank-0 can raise RendezvousClosedError before
    # rank-N joins. The operator's auto path injects this same value;
    # BYO commands need to mirror it.
    #
    # The ``with`` block calls ``training.delete()`` on scope exit
    # (success OR exception) -- so an intermediate wait such as
    # ``wait_until_target_world`` after ``scale()`` raising before the
    # explicit cleanup line does not leak the UD.
    training = basilica.distributed(
        name="dlc-example-torchrun",
        image="ghcr.io/one-covenant/basilica/basilica-distributed-trainer:latest",
        command=[
            "torchrun",
            "--rdzv-backend=etcd",
            "--rdzv-endpoint=$BASILICA_RDZV_ENDPOINT",
            "--rdzv-id=$BASILICA_RDZV_ID",
            "--rdzv-conf=timeout=1500",
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
        # Broadened availability zone root include-list: pinning to a single
        # availability zone root exposed
        # the example to localized capacity or provider-side transients.
        # Including several public availability zone roots lets the autoscaler
        # fall back when one root is unavailable. ``topology_spread="pack"``
        # below still keeps the workers on a single availability zone root per
        # UD; this list is a fallback set, not a spread directive.
        provider_filter=ProviderFilter(
            include=["cyan", "plum", "opal"]
        ),
        topology_spread="pack",
        nccl_env={"NCCL_DEBUG": "WARN"},
        ttl_seconds=600,
        timeout=900,
    )

    with training:
        print(f"Deployed: {training.name}")
        print(f"Namespace: {training.namespace}")
        print(f"World: {training.world}")

        # Scale up by one rank, mid-run. Demonstrates elastic scaling:
        # torchelastic re-rendezvouses workers when the StatefulSet
        # replica count changes, and the new rank joins.
        time.sleep(30)
        new_world = training.scale(target=3)
        print(f"Scaled to target=3; world now: {new_world}")

        # Wait for the new rank to join. If this raises (e.g. operator
        # leader transfer mid-rollout), the ``with`` block runs
        # ``delete()`` on its way out so the UD does not leak.
        training.wait_until_target_world(timeout=300)
        print(f"All 3 ranks ready: {training.world}")

        # Tail logs briefly so the example surfaces "is it actually
        # running". Per-rank filtering is not yet supported by the API;
        # logs are returned merged across ranks.
        print("--- merged logs ---")
        print(training.logs(tail=30))

    print("Cleanup complete.")


if __name__ == "__main__":
    main()
