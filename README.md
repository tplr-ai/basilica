# Basilica

On-demand access to the Basilica GPU cloud — via CLI, Python SDK, and programmatic APIs.

This repository is the public surface for Basilica users: the `basilica` CLI, the Python and Rust SDKs, and runnable examples.

---

**Quick links:**
[Getting Started](docs/GETTING-STARTED.md) · [Quickstart](docs/quickstart.md) · [Examples](examples/) · [Agent Cloud Ops](docs/agent-cloud-ops.md) · [Website](https://basilica.ai/) · [Discord](https://discord.gg/Cy7c9vPsNK)

## Install the CLI

```bash
curl -sSL https://basilica.ai/install.sh | bash
basilica login
```

For headless shells or remote servers:

```bash
basilica login --device-code
```

For coding agents:

```bash
basilica skills install
```

## What you can do

- **Rent GPUs directly** — SSH-accessible machines (secure cloud or community cloud) for training, inference, and interactive work.
- **Deploy services** — stand up public HTTP endpoints, inference servers (vLLM, SGLang), OpenClaw apps, or custom containers.
- **Drive it from Python** — provision, monitor, and tear down resources programmatically with the Python SDK.
- **Fund with TAO** — top up credits directly from a TAO wallet and track deposits.

## CLI at a glance

```bash
# Browse available GPUs
basilica ls

# Rent a machine
basilica up --gpu h100 --gpu-count 1

# List your active rentals
basilica ps

# SSH into a rental
basilica ssh <rental-id>

# Deploy an HTTP service
basilica deploy my-app nginx:latest --port 80 --ttl 3600

# Tear down
basilica down <rental-id>
basilica deploy delete my-app
```

See [`docs/quickstart.md`](docs/quickstart.md) for a full walkthrough.

## Python SDK

```bash
pip install basilica-sdk
export BASILICA_API_TOKEN="basilica_..."
```

```python
from basilica import BasilicaClient

client = BasilicaClient()
health = client.health_check()
print(health.status)
```

Full guide: [`docs/GETTING-STARTED.md`](docs/GETTING-STARTED.md). Working examples: [`examples/`](examples/).

## Repository layout

- `crates/basilica-cli/` — the `basilica` CLI
- `crates/basilica-sdk/` — Rust SDK for the public API
- `crates/basilica-sdk-python/` — Python bindings for the SDK
- `crates/basilica-common/` — shared types and utilities
- `examples/` — runnable end-to-end examples
- `docs/` — user guides

## License

Licensed under MIT. See [LICENSE](LICENSE).
