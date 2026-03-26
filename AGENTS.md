# AGENTS.md

This repository has two very different surfaces:

- `validator` / `miner` / subnet infrastructure
- Basilica cloud customer operations via the `basilica` CLI and Python SDK

If the user is asking how to **use Basilica as a customer/operator** for compute, billing, rentals, deployments, inference, or OpenClaw-style apps, ignore most of the subnet docs and start here:

- CLI source: `crates/basilica-cli/src/`
- Python SDK: `crates/basilica-sdk-python/`
- user docs: `crates/basilica-sdk-python/README.md`, `examples/`, `docs/GETTING-STARTED.md`

## What To Use

Use these repo-local skills for agent work:

- `basilica/.claude/skills/basilica-account-ops/SKILL.md`
  - login, device-code auth, API tokens, balance, TAO funding, deposit tracking
- `basilica/.claude/skills/basilica-rentals-ops/SKILL.md`
  - machine discovery, SSH keys, direct rentals, volumes, copy/exec/ssh, teardown
- `basilica/.claude/skills/basilica-serverless-ops/SKILL.md`
  - `basilica deploy`, `summon`, vLLM, SGLang, OpenClaw, Tau, scale/logs/share-token
- `basilica/.claude/skills/basilica-sdk-ops/SKILL.md`
  - Python automation, SDK caveats, blocking deploy semantics, low-level API gaps

For a single end-to-end playbook with copyable flows, use:

- `basilica/docs/agent-cloud-ops.md`
  - auth, funding, rentals, deploys, inference, OpenClaw, Tau, SDK, cleanup

## Routing

If the request is about:

- account setup, credits, deposits, or top-up: use `basilica-account-ops`
- direct GPU/CPU machines, SSH access, training boxes, or persistent rented hosts: use `basilica-rentals-ops`
- HTTP services, inference endpoints, public URLs, or self-hosted apps: use `basilica-serverless-ops`
- Python code, notebooks, scripts, CI automation, or typed programmatic access: use `basilica-sdk-ops`

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
- `docs/GETTING-STARTED.md`
- `config/README.md`
- `examples/README.md`
- `examples/15_cli_deploy/README.md`
- `examples/inference/README.md`
- `crates/basilica-cli/src/cli/commands.rs`
- `crates/basilica-cli/src/cli/handlers/`
- `crates/basilica-sdk-python/README.md`
- `crates/basilica-sdk-python/python/basilica/__init__.py`
- `crates/basilica-sdk-python/python/basilica/_deployment.py`

## Fast Decision Table

- User wants a machine and shell access: rentals
- User wants a public API or app URL: serverless deploy
- User wants to top up credits: account ops
- User wants programmatic provisioning in Python: SDK ops
- User wants OpenClaw or Tau specifically: serverless deploy skill
- User wants huge multi-GPU model serving with lots of manual control: rentals first, deploy second

## TODOs For Future Agent Docs

- add a dedicated skill for billing/usage-history analysis once the CLI grows first-class spend commands
- add a repo-local troubleshooting skill once deployment failure patterns stabilize around phases and events
- add shell-tested end-to-end transcripts for account -> fund -> deploy -> cleanup flows
