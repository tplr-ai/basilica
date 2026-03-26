---
name: basilica-serverless-ops
description: Use when the user wants to deploy apps, APIs, inference endpoints, or OpenClaw/Tau-style services with the Basilica CLI.
---

# Basilica Serverless Ops

Use this skill for the managed deployment surface exposed by the CLI:

- generic `basilica deploy`
- inference templates like `vllm` and `sglang`
- `basilica summon openclaw`
- `basilica summon tau`
- deployment logs, status, scale, and delete
- share-token operations for private deployments

## Choose The Right Deployment Mode

Use serverless deploys when the user wants:

- a public URL
- an HTTP API
- an inference endpoint
- an app that should come up without managing a box manually

Prefer rentals instead when the user needs SSH, custom system control, distributed training, or very large models that may spend a long time loading.

## Canonical Preflight

Authenticate:

```bash
basilica login
```

Check spendability:

```bash
basilica balance
```

Before you create a deployment, decide:

- source form: inline code, Python file, or container image
- whether it should be public or `--private`
- whether it needs GPU
- whether it needs storage
- whether it should auto-expire via `--ttl`

## Generic Deploy Patterns

Inline Python:

```bash
basilica deploy 'print("hello")' --name hello --port 8000 --ttl 300
```

Python file:

```bash
basilica deploy my_api.py \
  --name my-api \
  --port 8000 \
  --pip fastapi uvicorn \
  --ttl 600
```

Container image:

```bash
basilica deploy nginxinc/nginx-unprivileged:alpine \
  --name nginx-demo \
  --port 8080 \
  --cpu 250m \
  --memory 256Mi \
  --ttl 300
```

GPU-backed deploy:

```bash
basilica deploy inference.py \
  --name gpu-model \
  --gpu 1 \
  --gpu-model H100 \
  --memory 32Gi \
  --pip torch
```

Persistent storage:

```bash
basilica deploy hello.py \
  --name stateful-app \
  --storage \
  --storage-path /data
```

## Deployment Management

List:

```bash
basilica deploy ls
basilica deploy ls --json
```

Status:

```bash
basilica deploy status my-app
basilica deploy status my-app --show-phases
```

Logs:

```bash
basilica deploy logs my-app
basilica deploy logs my-app --follow
basilica deploy logs my-app --tail 100
```

Scale:

```bash
basilica deploy scale my-app --replicas 3
```

Delete:

```bash
basilica deploy delete my-app --yes
```

Always use `deploy delete`, not stale `deployments delete` variants from older examples.

## Inference Templates

vLLM:

```bash
basilica deploy vllm Qwen/Qwen2.5-0.5B-Instruct --name my-llm
```

SGLang:

```bash
basilica deploy sglang Qwen/Qwen2.5-0.5B-Instruct --name my-sglang
```

For large or unusual models, expect to tune health checks, GPU count, memory, or move to rentals.

## OpenClaw And Tau

OpenClaw:

```bash
basilica summon openclaw --provider openai
basilica summon openclaw --provider anthropic
```

Tau:

```bash
basilica summon tau
```

OpenClaw-specific flags exposed by the current template include:

- `--provider openai|anthropic`
- `--backend-url`
- `--model-id`
- `--provider-id`
- `--provider-api`
- `--context-window`
- `--max-tokens`

OpenClaw template env requirements:

- `OPENAI_API_KEY` for OpenAI-backed deploys
- `ANTHROPIC_API_KEY` for Anthropic-backed deploys

Tau template env requirements:

- `TAU_BOT_TOKEN`
- `CHUTES_API_TOKEN`

Optional Tau env:

- `TAU_CHAT_MODEL`

## Private Deployments And Share Tokens

Deploy privately:

```bash
basilica deploy my_api.py --name private-app --port 8000 --private
```

Manage share tokens:

```bash
basilica deploy share-token status private-app
basilica deploy share-token regenerate private-app
basilica deploy share-token revoke private-app --yes
```

Current behavior:

- deployments are public by default
- `--private` makes the deploy share-token-gated
- OpenClaw is a special case and is intended to be public with its own gateway token behavior

## Topology And Placement Controls

Spread replicas across nodes:

```bash
basilica deploy hashicorp/http-echo:latest \
  --name spread-demo \
  --port 5678 \
  --replicas 2 \
  --unique-nodes
```

Use explicit spread policy:

```bash
basilica deploy image \
  --name app \
  --replicas 4 \
  --spread-mode required \
  --max-skew 1
```

## Operational Caveats

- Use `--ttl` by default unless the user explicitly wants a persistent service.
- Large models can spend 10-30 minutes loading. Watch `deploy status --show-phases` and `deploy logs --follow`.
- If a model is huge enough to need 8x H100/H200 or lots of manual tuning, prefer rentals first.
- Deploys are public unless you say otherwise.

## Good Agent Flow

1. check `basilica balance`
2. pick the deploy shape
3. deploy with `--ttl` unless persistence is required
4. watch status/logs until the URL is usable
5. return the URL and the cleanup plan
6. delete the deployment when done unless the user asked to keep it

## File Pointers

- `examples/15_cli_deploy/README.md`
- `examples/inference/README.md`
- `crates/basilica-cli/src/cli/handlers/deploy/mod.rs`
- `crates/basilica-cli/src/cli/handlers/deploy/templates/openclaw.rs`
- `crates/basilica-cli/src/cli/handlers/deploy/templates/vllm.rs`
- `crates/basilica-cli/src/cli/handlers/deploy/templates/sglang.rs`
- `crates/basilica-cli/src/cli/handlers/deploy/templates/tau.rs`

## TODOs

- add a repo-local end-to-end example for `summon openclaw` with provider env setup
- add a tested table mapping model size to recommended `deploy` vs `rentals` path
