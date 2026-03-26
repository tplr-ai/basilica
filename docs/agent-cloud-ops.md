# Basilica Agent Cloud Ops

This doc is for coding agents operating Basilica as a customer cloud platform, not for subnet/miner/validator work.

Use this when the user wants any of:

- login or API tokens
- balance or TAO funding
- GPU/CPU rentals
- serverless deploys
- inference endpoints
- OpenClaw or Tau
- Python SDK automation

## 0. Control Plane Decision

Pick the control plane first:

- needs shell access or a machine: rentals
- needs a public URL or HTTP API: `basilica deploy`
- needs Python automation: `BasilicaClient`
- needs credits or deposit address: account ops

## 1. Auth

Canonical CLI auth:

```bash
curl -sSL https://basilica.ai/install.sh | bash
basilica login
```

Headless auth:

```bash
basilica login --device-code
```

Programmatic auth:

```bash
basilica tokens create my-agent-token
export BASILICA_API_TOKEN="basilica_..."
```

## 2. Funding And Balance

Check credits:

```bash
basilica balance
```

Get or create the TAO deposit address:

```bash
basilica fund
```

Track deposits:

```bash
basilica fund list --limit 100 --offset 0
```

Current operator facts from the CLI:

- minimum deposit is `0.1 TAO`
- credits settle after `12` confirmations
- there is no separate `top-up` verb; `basilica fund` is the funding entrypoint

## 3. Rentals

Discover capacity:

```bash
basilica ls
basilica ls h100
basilica ls --price-max 5 --country US
basilica ls --compute secure-cloud
```

Ensure SSH key registration:

```bash
basilica ssh-keys list
basilica ssh-keys add
```

Create a rental:

```bash
basilica up h200 --gpu-count 8 --compute secure-cloud
```

Operate the machine:

```bash
basilica ps
basilica status <rental-id>
basilica ssh <rental-id>
basilica exec "nvidia-smi" --target <rental-id>
basilica cp ./local.txt <rental-id>:/workspace/local.txt
basilica restart <rental-id>
```

Teardown:

```bash
basilica down <rental-id>
basilica down --all
```

Persistent volume flow for secure-cloud:

```bash
basilica volumes create --name cache --size 100 --provider hyperstack --region US-1
basilica volumes attach cache --rental <rental-id>
basilica volumes list
basilica volumes detach cache --yes
basilica volumes delete cache --yes
```

## 4. Serverless Deploys

Inline code:

```bash
basilica deploy 'print("hello")' --name hello --port 8000 --ttl 300
```

Python file:

```bash
basilica deploy my_api.py --name my-api --port 8000 --pip fastapi uvicorn --ttl 600
```

Container image:

```bash
basilica deploy nginxinc/nginx-unprivileged:alpine --name nginx-demo --port 8080 --ttl 300
```

GPU deploy:

```bash
basilica deploy inference.py --name gpu-model --gpu 1 --gpu-model H100 --memory 32Gi --pip torch
```

Storage-backed deploy:

```bash
basilica deploy hello.py --name stateful-app --storage --storage-path /data
```

Manage deploys:

```bash
basilica deploy ls
basilica deploy status my-app --show-phases
basilica deploy logs my-app --follow
basilica deploy scale my-app --replicas 3
basilica deploy delete my-app --yes
```

Current command name is `basilica deploy delete`, not stale `basilica deployments delete`.

## 5. Inference Templates

vLLM:

```bash
basilica deploy vllm Qwen/Qwen2.5-0.5B-Instruct --name my-llm
```

SGLang:

```bash
basilica deploy sglang Qwen/Qwen2.5-0.5B-Instruct --name my-sglang
```

For very large models:

- expect longer startup and health-check tuning
- prefer rentals if the workload needs manual control or long warmup

## 6. OpenClaw And Tau

OpenClaw:

```bash
basilica summon openclaw --provider openai
basilica summon openclaw --provider anthropic
```

Tau:

```bash
basilica summon tau
```

OpenClaw notes:

- OpenClaw deploys are intentionally public
- access is controlled by the OpenClaw gateway token, not share-token auth
- the template supports provider/model/backend flags like `--backend-url`, `--model-id`, `--provider-id`, `--context-window`, and `--max-tokens`

## 7. Private Deployments

Create a private deployment:

```bash
basilica deploy my_api.py --name private-app --port 8000 --private
```

Manage share tokens:

```bash
basilica deploy share-token status private-app
basilica deploy share-token regenerate private-app
basilica deploy share-token revoke private-app --yes
```

## 8. Python SDK

Install:

```bash
pip install basilica-sdk
```

Basic client:

```python
from basilica import BasilicaClient

client = BasilicaClient()
health = client.health_check()
print(health.status)
```

High-level deploy:

```python
from basilica import BasilicaClient

client = BasilicaClient()
deployment = client.deploy(
    name="hello-api",
    source="app.py",
    port=8000,
    pip_packages=["fastapi", "uvicorn"],
    ttl_seconds=600,
)
print(deployment.url)
```

Secure-cloud rental:

```python
from basilica import BasilicaClient

client = BasilicaClient()
key = client.get_ssh_key() or client.register_ssh_key("agent-key")
offering = sorted(client.list_secure_cloud_gpus(), key=lambda o: float(o.hourly_rate))[0]
rental = client.start_secure_cloud_rental(offering_id=offering.id, ssh_public_key_id=key.id)
print(rental.ssh_command)
```

Balance and usage:

```python
balance = client.get_balance()
usage = client.list_usage_history(limit=20, offset=0)
```

SDK caveats:

- `deploy()` blocks until readiness
- public deployments are default
- deposit-address and deposit-history flows are better through the CLI
- for failure details, low-level `get_deployment(name).message` is more reliable than `deployment.status().message`

## 9. Cost Discipline

Treat these as cost-bearing actions:

- `basilica up ...`
- `basilica deploy ...`
- `basilica summon ...`
- SDK create/start methods

Agent defaults:

- check `basilica balance` first
- use TTLs for deploys unless persistence is explicitly requested
- tear down rentals when finished unless the user asked to keep them

## 10. Best Source Files

- `AGENTS.md`
- `README.md`
- `docs/GETTING-STARTED.md`
- `config/README.md`
- `examples/README.md`
- `examples/15_cli_deploy/README.md`
- `examples/inference/README.md`
- `crates/basilica-cli/src/cli/handlers/`
- `crates/basilica-sdk-python/README.md`
- `crates/basilica-sdk-python/python/basilica/__init__.py`

## TODO

- add shell-tested end-to-end examples for funding -> deploy -> cleanup
- add a spend/usage troubleshooting section once the CLI exposes richer billing history
