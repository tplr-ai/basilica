# Python SDK contributor guidance

Read [the root policy](../../AGENTS.md) and
[the contributor guide](../../docs/DEVELOPMENT.md). This file owns the SDK-specific
invariants; CLI use belongs to the customer `use-basilica` skill.

## Source and checks

- Python public exports: `python/basilica/__init__.py`; Rust extension: `src/`.
- Deployment behavior: `python/basilica/_deployment.py` and `decorators.py`.
- Managed training: `python/basilica/distributed.py` and `decorators.py`.
- Preserve parity among the Python wrapper, `_basilica.pyi` type declarations,
  Rust bindings, and user-facing examples when changing a public interface.
- Keep `Cargo.toml` and `pyproject.toml` package versions aligned. Do not alter
  the version or dependency locks as a side effect of unrelated edits.
- Use the locked uv/maturin setup and exact live-test exclusions in the
  contributor guide. Build the actual extension before collecting tests;
  syntax-only or stub-only validation does not establish import behavior.

## Managed distributed training

For managed multi-rank PyTorch/NCCL, use the surface below. Rentals apply when
manual host/SSH control or a runtime unsupported by managed training is required.
This is contributor API guidance; running either example creates chargeable
resources and requires user authorization.


When the user wants multi-rank PyTorch training (DDP, DiLoCo, FSDP, any
NCCL-collective workload), the canonical SDK surface is the
`@basilica.distributed` decorator. ONE entry point handles both shapes:

```python
import basilica
from basilica import ProviderFilter, WorldSize


@basilica.distributed(
    name="dlc-...",
    image="ghcr.io/one-covenant/basilica/basilica-distributed-trainer:latest",
    world_size=WorldSize(min=2, target=4, max=4),
    gpu_count=1,
    gpu_models=["A100"],
    provider_filter=ProviderFilter(include=["hyperstack", "verda"]),
    topology_spread="pack",        # required for direct WG mesh on NCCL
    bench=True,                    # opt-in NCCL bandwidth probe
)
def train():
    import os, torch, torch.distributed as dist
    dist.init_process_group(backend="nccl")
    # ... uses os.environ['RANK'] / ['WORLD_SIZE'] / ['LOCAL_RANK'] ...
    dist.destroy_process_group()


with train() as training:               # auto-cleanup on scope exit
    training.wait_until_complete(timeout=1800)
    print(training.bench)               # BenchResult | None
```

For BYO launchers (torchrun, mpirun, accelerate), pass `command=[...]`
and `basilica.distributed(...)` returns a `DistributedTraining` directly
(factory mode):

```python
training = basilica.distributed(
    name="dlc-byo-launcher",
    image="ghcr.io/one-covenant/basilica/basilica-distributed-trainer:latest",
    command=[
        "torchrun",
        "--rdzv-backend=etcd",
        "--rdzv-endpoint=$BASILICA_RDZV_ENDPOINT",
        "--rdzv-id=$BASILICA_RDZV_ID",
        "--nnodes=$BASILICA_WORLD_TARGET",
        "--nproc-per-node=$BASILICA_GPUS_PER_POD",
        "/workspace/my_training.py",
    ],
    world_size=WorldSize(min=2, target=2, max=4),
    gpu_count=1,
    gpu_models=["A100"],
    provider_filter=ProviderFilter(include=["hyperstack", "verda"]),
    topology_spread="pack",
)
with training:
    training.scale(target=3)
```

Rules:

- ONE canonical surface: `@basilica.distributed` (decorator) or
  `basilica.distributed(command=[...])` (factory). Both return a
  `DistributedTraining` context manager.
- Use `with training:` for mid-run orchestration (`scale()`,
  `wait_until_*`, `logs()`, `bench`). Bare call is fire-and-forget with
  auto-cleanup.
- `bench=True` (bool) opts in to the per-UD NCCL bandwidth probe. Read
  via `training.bench` (`BenchResult | None`); `None` means "no
  measurement" regardless of why. Use `training.bench_diagnostics` only
  for the rare debug case.
- DO NOT use `client.deploy_distributed_managed(...)`, `bench="on-start"`,
  `training.bench_status`, or `training.wait_until_bench_complete()` --
  all four were REMOVED in 0.30.0 (SDK-S7) and now raise
  `AttributeError` / `ImportError` / `ValidationError`. The
  `@basilica.distributed` decorator is the ONE canonical surface; see
  the SDK README's "Migration from the legacy surface" table for the
  per-symbol mapping.
- Worked examples: `../../examples/20_distributed_diloco.py` (decorator +
  bench + DiLoCo), `../../examples/21_distributed_torchrun.py` (BYO command +
  mid-run scale), `../../examples/22_distributed_with_bench.py` (bench-result
  inspection + JSON dump).
- User-facing runbook: [distributed runbook](https://github.com/one-covenant/basilica-backend/blob/main/docs/runbooks/USER-RUNBOOK-DISTRIBUTED-NCCL.md)
  and [custom trainer image guide](https://github.com/one-covenant/basilica-backend/blob/main/docs/runbooks/BRINGING-YOUR-OWN-TRAINER-IMAGE.md)
  in the separate backend repository (access required).

