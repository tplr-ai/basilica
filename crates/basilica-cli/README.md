# basilica-cli

Unified CLI for Basilica GPU rental and network management.

[![Crates.io](https://img.shields.io/crates/v/basilica-cli.svg)](https://crates.io/crates/basilica-cli)
[![Documentation](https://docs.rs/basilica-cli/badge.svg)](https://docs.rs/basilica-cli)
[![License: MIT](https://img.shields.io/badge/License-MIT-yellow.svg)](https://opensource.org/licenses/MIT)

[Documentation](https://docs.rs/basilica-cli) | [Repository](https://github.com/one-covenant/basilica) | [Website](https://basilica.ai)

## Overview

`basilica-cli` provides a comprehensive command-line interface for interacting with the Basilica GPU rental network. Deploy workloads, manage rentals, and monitor resources all from your terminal.

## Installation

### Quick Install (Recommended)

```bash
curl -sSL https://basilica.ai/install.sh | bash
```

### From Cargo

```bash
cargo install basilica-cli
```

### From Source

```bash
git clone https://github.com/one-covenant/basilica
cd basilica
cargo build --release -p basilica-cli
```

## Quick Start

```bash
# Login to Basilica
basilica login

# List available GPU types
basilica gpus list

# Deploy a workload
basilica deploy --gpu h100 --image nvidia/cuda:12.0-base

# Check rental status
basilica rentals list

# Stream logs
basilica logs <rental-id> -f
```

## Commands

| Command | Description |
|---------|-------------|
| `login` | Authenticate with Basilica |
| `logout` | Clear authentication |
| `gpus` | List and filter available GPUs |
| `deploy` | Deploy a workload to the network |
| `rentals` | Manage active rentals |
| `logs` | View workload logs |
| `exec` | Execute commands in rentals |
| `ssh` | SSH into a rental |
| `billing` | View usage and billing |
| `config` | Manage CLI configuration |
| `completions` | Generate shell completions |
| `update` | Self-update to latest version |

## Examples

### Deploy a vLLM Model

```bash
basilica deploy \
  --gpu h100 \
  --image vllm/vllm-openai:latest \
  --env MODEL=meta-llama/Llama-2-7b-hf \
  --port 8000
```

### Deploy with Storage

```bash
basilica deploy \
  --gpu a100 \
  --image pytorch/pytorch:latest \
  --mount data:/data \
  --command "python train.py --data-dir /data"
```

### Interactive GPU Session

```bash
basilica deploy --gpu h100 --interactive
# Opens SSH session to GPU node
```

## Configuration

Configuration is stored in `~/.config/basilica/config.toml`:

```toml
[auth]
api_key = "your-api-key"

[defaults]
gpu_type = "h100"
region = "us-east"

[api]
base_url = "https://api.basilica.ai"
```

## Shell Completions

```bash
# Bash
basilica completions bash > ~/.local/share/bash-completion/completions/basilica

# Zsh
basilica completions zsh > ~/.zfunc/_basilica

# Fish
basilica completions fish > ~/.config/fish/completions/basilica.fish
```

## Environment Variables

| Variable | Description |
|----------|-------------|
| `BASILICA_API_KEY` | API key for authentication |
| `BASILICA_API_URL` | Custom API endpoint |
| `BASILICA_LOG_LEVEL` | Logging verbosity (debug, info, warn, error) |

## Related Crates

- [`basilica-sdk`](https://crates.io/crates/basilica-sdk) - Rust SDK for programmatic access
- [`basilica-api`](https://crates.io/crates/basilica-api) - API gateway

## License

MIT License - see [LICENSE](LICENSE) for details.

