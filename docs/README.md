# Basilica Documentation

User guides for the Basilica CLI and Python SDK.

## Start here

- **[Quickstart](quickstart.md)** — install the CLI, log in, rent a GPU, and deploy a service.
- **[Getting Started (SDK)](GETTING-STARTED.md)** — generate an API token and drive Basilica from Python end-to-end.
- **[Agent Cloud Ops](agent-cloud-ops.md)** — canonical playbook for agents operating Basilica (auth, funding, rentals, deploys, cleanup).

## Examples

Runnable examples live in [`../examples/`](../examples/). They cover:

- Simple HTTP servers and FastAPI apps
- GPU workloads with PyTorch + CUDA
- Persistent storage, queues, and scale
- Inference servers (vLLM, SGLang)
- OpenClaw-style browser-native apps

## Choosing the right control plane

| If you want… | Use |
|---|---|
| A persistent GPU/CPU machine with SSH access | `basilica up` (rentals) |
| A public HTTP endpoint or containerized service | `basilica deploy` |
| Hosted inference with vLLM/SGLang | `basilica deploy vllm …` / `basilica deploy sglang …` |
| OpenClaw or Tau apps | `basilica summon …` / `basilica deploy …` |
| Programmatic automation | Python SDK (`basilica-sdk`) |

## Support

- **GitHub**: <https://github.com/one-covenant/basilica>
- **Discord**: <https://discord.gg/Cy7c9vPsNK>
- **Website**: <https://basilica.ai/>
