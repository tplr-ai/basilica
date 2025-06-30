# Basilica Remote Executor Development Setup

This document describes how to use the new remote executor deployment system for local development.

## Overview

The remote executor system allows miners to deploy and manage executors on remote GPU machines via SSH. This is the production architecture where executors run directly on GPU machines, not in local Docker containers.

## Quick Start

```bash
# Start the development environment with one command
just dev
```

This command will:
1. Build the executor binary
2. Start the miner with Docker Compose
3. Attempt to deploy executors to configured remote machines

## Configuration

Edit `docker/configs/miner-local.toml` to configure your remote GPU machines:

```toml
[[remote_executor_deployment.remote_machines]]
id = "vast-gpu-1"
name = "Vast.ai GPU Instance"
executor_port = 50051

[remote_executor_deployment.remote_machines.ssh]
host = "YOUR_VAST_AI_HOST"  # Replace with actual host
port = 22222  # Replace with actual SSH port
username = "root"
private_key_path = "/root/.ssh/basilica_gpu_test"
# Example for vast.ai:
# ssh_options = [
#   "StrictHostKeyChecking=no",
#   "UserKnownHostsFile=/dev/null"
# ]
```

## SSH Key Setup

Generate an SSH key for your remote machines:

```bash
ssh-keygen -t ed25519 -f ~/.ssh/basilica_gpu_test -N ""
```

Then add the public key to your remote machine's `~/.ssh/authorized_keys`.

## Manual Operations

### Deploy executors manually
```bash
docker exec basilica-miner-dev miner -c /config/miner-local.toml deploy-executors
```

### Check executor status
```bash
docker exec basilica-miner-dev miner -c /config/miner-local.toml deploy-executors --status-only
```

### View logs
```bash
just dev-logs
```

### Stop everything
```bash
just dev-down
```

## Services

The development setup includes:
- **Miner**: Manages remote executors and serves validator requests
- **Prometheus**: Metrics collection (http://localhost:9090)
- **Grafana**: Metrics visualization (http://localhost:3000)
- **gRPC UI**: Debug gRPC endpoints (http://localhost:8081)

## Architecture

```
┌─────────────────┐         SSH          ┌──────────────────┐
│     Miner       │──────────────────────▶│ Remote GPU       │
│  (Docker)       │                       │ Machine          │
│                 │         gRPC          │                  │
│                 │◀──────────────────────│ - Executor       │
└─────────────────┘                       │ - GPUs           │
                                          └──────────────────┘
```

## Troubleshooting

### Miner container restarting
- Check logs: `docker logs basilica-miner-dev`
- Ensure SSH configuration is correct in `miner-local.toml`
- The system works fine even without configured remote machines

### SSH connection issues
- Verify SSH key permissions: `chmod 600 ~/.ssh/basilica_gpu_test`
- Test SSH manually: `ssh -i ~/.ssh/basilica_gpu_test -p PORT user@host`
- Check firewall rules on remote machine

### Executor deployment fails
- Ensure remote machine has required dependencies
- Check disk space on remote machine
- Verify executor binary was built: `ls target/release/executor`