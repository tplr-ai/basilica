# AGENTS.md

This repository is the customer-facing surface for Basilica: the `basilica` CLI, the Python and Rust SDKs, and runnable examples. All agent work should target this surface.

Start here:

- CLI source: `crates/basilica-cli/src/`
- Python SDK: `crates/basilica-sdk-python/`
- user docs: `docs/GETTING-STARTED.md`, `docs/quickstart.md`, `examples/`

## What To Use

Paths below resolve from this repository root, regardless of the checkout name.
Use these repo-local skills for platform operations:

- `.claude/skills/basilica-account-ops/SKILL.md`
  - login, device-code auth, API tokens, balance, TAO funding, deposit tracking
- `.claude/skills/basilica-rentals-ops/SKILL.md`
  - machine discovery, SSH keys, direct rentals, volumes, copy/exec/ssh, teardown
- `.claude/skills/basilica-serverless-ops/SKILL.md`
  - `basilica deploy`, `summon`, vLLM, SGLang, OpenClaw, Tau, scale/logs/share-token
- `.claude/skills/basilica-sdk-ops/SKILL.md`
  - Python automation, SDK caveats, blocking deploy semantics, low-level API gaps

For a single end-to-end playbook with copyable flows, use:

- `docs/agent-cloud-ops.md`
  - auth, funding, rentals, deploys, inference, OpenClaw, Tau, SDK, cleanup

## Routing

If the request is about:

- account setup, credits, deposits, or top-up: use `basilica-account-ops`
- direct GPU/CPU machines, SSH access, training boxes, or persistent rented hosts: use `basilica-rentals-ops`
- HTTP services, inference endpoints, public URLs, or self-hosted apps: use `basilica-serverless-ops`
- Python code, notebooks, scripts, CI automation, or typed programmatic access: use `basilica-sdk-ops`
- distributed PyTorch / NCCL collective workloads (DDP, DiLoCo, FSDP), multi-rank training with `torch.distributed`: use the `@basilica.distributed` SDK surface (see "Distributed Training" below)

## Distributed Training

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
- Worked examples: `examples/20_distributed_diloco.py` (decorator +
  bench + DiLoCo), `examples/21_distributed_torchrun.py` (BYO command +
  mid-run scale), `examples/22_distributed_with_bench.py` (bench-result
  inspection + JSON dump).
- User-facing runbook: `docs/runbooks/USER-RUNBOOK-DISTRIBUTED-NCCL.md`
  on basilica-backend. Custom-image guide:
  `docs/runbooks/BRINGING-YOUR-OWN-TRAINER-IMAGE.md`.

## Canonical Agent Rules

### 1. Pick the right control plane

- Prefer the CLI for interactive operator workflows.
- Prefer the Python SDK for repeatable automation and code generation.
- Prefer direct rentals over serverless deploys when the workload needs SSH, custom system setup, distributed training, or huge models that may take too long to pass deployment health checks.

### 2. Use the real auth path

- For CLI usage, the canonical path is:

```bash
curl -sSL https://basilica.ai/install.sh | bash
basilica login
```

- For headless terminals, SSH boxes, or remote shells, use:

```bash
basilica login --device-code
```

- Use API tokens mainly for SDK/programmatic work:

```bash
basilica tokens create my-agent-token
export BASILICA_API_TOKEN="basilica_..."
```

### 3. Treat cost-bearing actions as explicit

Do not create chargeable resources unless the user asked for it. These usually incur cost:

- `basilica up ...`
- `basilica deploy ...`
- `basilica deploy vllm ...`
- `basilica deploy sglang ...`
- `basilica summon ...`
- SDK equivalents like `start_secure_cloud_rental()` and `deploy()`

Safe read-mostly operations include:

- `basilica balance`
- `basilica fund`
- `basilica fund list`
- `basilica ls`
- `basilica ps`
- `basilica status <id>`
- `basilica deploy ls`
- `basilica deploy status <name>`
- SDK health/list/get calls

### 4. Always include cleanup intent

When you create resources, prefer cleanup-friendly defaults:

- set `--ttl` / `ttl_seconds` for deployments unless the user explicitly wants persistence
- call out whether the resource will persist after the current task
- tear down rentals with `basilica down <rental-id>` when the task is finished unless the user asked to keep them

### 5. Use current command names

The current CLI uses:

- `basilica deploy delete ...`
- not `basilica deployments delete ...`

Be careful with stale examples in old docs or transcripts.

### 6. Understand default exposure

- deploys are public by default
- `--private` changes the deploy to share-token-gated access
- OpenClaw deployments are intentionally public and use their own gateway token flow

### 7. Know the current gaps

- CLI exposes balance + deposit flows, but not a rich spend-history surface
- SDK exposes balance + usage history, but does not cleanly expose the full deposit-account flow in the public Python wrapper
- for deposit address creation/history, prefer the CLI

## Repo Pointers

These files are the best source of truth for agent docs and command behavior:

- `README.md`
- `docs/README.md`
- `docs/GETTING-STARTED.md`
- `docs/quickstart.md`
- `examples/README.md`
- `examples/15_cli_deploy/README.md`
- `examples/inference/README.md`
- `crates/basilica-cli/src/cli/commands.rs`
- `crates/basilica-cli/src/cli/handlers/`
- `crates/basilica-sdk-python/README.md`
- `crates/basilica-sdk-python/python/basilica/__init__.py`
- `crates/basilica-sdk-python/python/basilica/_deployment.py`
- `crates/basilica-sdk-python/python/basilica/distributed.py` (distributed-training types + DistributedTraining facade)
- `crates/basilica-sdk-python/python/basilica/decorators.py` (`@basilica.distributed` + `@basilica.deployment` decorators)

## Fast Decision Table

- User wants a machine and shell access: rentals
- User wants a public API or app URL: serverless deploy
- User wants to top up credits: account ops
- User wants programmatic provisioning in Python: SDK ops
- User wants multi-rank distributed training (DDP, DiLoCo, FSDP, NCCL collectives): `@basilica.distributed` decorator (see "Distributed Training" above)
- User wants OpenClaw or Tau specifically: serverless deploy skill
- User wants huge multi-GPU model serving with lots of manual control: rentals first, deploy second

## TODOs For Future Agent Docs

- add a dedicated skill for billing/usage-history analysis once the CLI grows first-class spend commands
- add a repo-local troubleshooting skill once deployment failure patterns stabilize around phases and events
- add shell-tested end-to-end transcripts for account -> fund -> deploy -> cleanup flows
