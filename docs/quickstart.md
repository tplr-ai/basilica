# Quickstart

Get from zero to a running GPU rental or deployed service in a few minutes.

## 1. Install the CLI

```bash
curl -sSL https://basilica.ai/install.sh | bash
```

Verify it's on your `$PATH`:

```bash
basilica --version
```

## 2. Log in

```bash
basilica login
```

For headless terminals (SSH sessions, remote boxes, CI):

```bash
basilica login --device-code
```

## 3. Fund your account (if needed)

Top up credits with TAO. The CLI will generate a deposit address for you:

```bash
basilica fund
basilica balance
basilica fund list
```

---

## Rent a GPU directly

Use this path when you need SSH access — training, custom environments, long-lived development boxes, or debugging.

### Browse available GPUs

```bash
basilica ls
```

Filter by GPU type, region, or spot:

```bash
basilica ls --gpu h100 --region US
basilica ls --gpu a100 --exclude-spot
```

### Start a rental

```bash
basilica up --gpu h100 --gpu-count 1
```

The CLI prompts for a name and your SSH key, then returns a rental ID and connection info.

### Work with your rental

```bash
basilica ps                    # list active rentals
basilica status <rental-id>    # details
basilica ssh <rental-id>       # interactive shell
basilica exec <rental-id> -- nvidia-smi
basilica cp ./local-file <rental-id>:/workspace/
```

### Tear it down

```bash
basilica down <rental-id>
```

---

## Deploy a service

Use this path when you want a public HTTP endpoint, a containerized app, or a hosted inference server.

### Basic deploy

```bash
basilica deploy my-app nginx:latest --port 80 --ttl 3600
basilica deploy status my-app
basilica deploy logs my-app
```

`--ttl` ensures the deployment auto-cleans; drop it for a persistent service.

### Inference servers

```bash
basilica deploy vllm my-llm --model meta-llama/Meta-Llama-3-8B-Instruct
basilica deploy sglang my-sglang --model Qwen/Qwen2.5-7B-Instruct
```

### Scale and share

```bash
basilica deploy scale my-app --replicas 3
basilica deploy share-token my-app          # generate token for private deploys
```

### Delete

```bash
basilica deploy delete my-app
```

---

## Drive it from Python

```bash
pip install basilica-sdk
basilica tokens create my-agent-token
export BASILICA_API_TOKEN="basilica_..."
```

```python
from basilica import BasilicaClient

client = BasilicaClient()

# Health
print(client.health_check().status)

# List GPU offerings
for node in client.list_nodes(available=True)[:5]:
    gpu = node.node.gpu_specs[0]
    print(f"{node.node.id}: {gpu.name} x{len(node.node.gpu_specs)}")
```

See [GETTING-STARTED.md](GETTING-STARTED.md) for the full SDK walkthrough.

---

## What's next

- [`examples/`](../examples/) — runnable end-to-end scripts for deployments, inference, GPU training, storage.
- [GETTING-STARTED.md](GETTING-STARTED.md) — the canonical SDK guide.
- [agent-cloud-ops.md](agent-cloud-ops.md) — copy-paste playbook for agents and automation.

## Troubleshooting

### CLI install didn't land in `$PATH`

The installer writes to `~/.basilica/bin`. Add it to your shell profile:

```bash
echo 'export PATH="$HOME/.basilica/bin:$PATH"' >> ~/.bashrc
source ~/.bashrc
```

### `basilica login` can't open a browser

Use device-code flow:

```bash
basilica login --device-code
```

### API token auth fails from Python

Confirm the env var is exported in the shell running your script:

```bash
echo $BASILICA_API_TOKEN
```

Create a fresh token if needed:

```bash
basilica tokens create fresh-token
```

### Something else

Check the [getting-started guide](GETTING-STARTED.md) troubleshooting section, or reach out on [Discord](https://discord.gg/Cy7c9vPsNK).
