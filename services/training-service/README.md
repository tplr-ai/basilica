# Basilica Training Service

GPU training service for fine-tuning LLMs with LoRA on Basilica.

## Quick Start (Python SDK)

```python
from basilica.training import Client

client = Client()
with client.training("facebook/opt-125m", rank=8, gpu_count=0) as session:
    loss = session.forward_backward([{"input_ids": [2, 133, 2119, 6219, 23602]}])
    session.optim_step()
    print(f"Loss: {loss:.4f}")
    print(session.sample("The quick brown"))
```

## Local Development

### 1. Start the cluster

```bash
./scripts/local-training-e2e.sh cluster-up
./scripts/local-training-e2e.sh deploy
./scripts/local-training-e2e.sh api      # Terminal 1
./scripts/local-training-e2e.sh gen-key  # Terminal 2
```

### 2. Run the example

```bash
export BASILICA_API_URL="http://localhost:8000"
python examples/training_example.py
```

### Example Output

```
=== Creating session for facebook/opt-125m ===
Session: ts-a1b2c3d4

=== Training (3 steps) ===
Step 1: loss=6.2612
Step 2: loss=5.7423
Step 3: loss=5.7501

Sample: The quick brown -> fox jumps over the lazy
```

## Overview

The training service provides a high-level API for:
- **LoRA fine-tuning** with HuggingFace + PEFT
- **Forward-backward passes** with gradient accumulation
- **Text generation** with the fine-tuned model
- **Checkpoint management** (save/load)

## Local E2E Testing (Mac)

The easiest way to test the training service is using the local e2e test script, which sets up a complete environment with k3d (k3s in Docker).

> **Note:** These instructions are for macOS. The e2e test runs in CPU-only mode, so no GPU is required.

### Prerequisites

- Docker Desktop
- ~8GB free RAM (for CPU-only mode)

### Install Dependencies (Mac)

```bash
# Install kubectl
brew install kubectl

# Install k3d (k3s in Docker)
brew install k3d

# Verify installation
k3d version
kubectl version --client
```

### Quick Start

```bash
# From repository root
cd /path/to/basilica

# 1. Create k3d cluster (one time)
./scripts/local-training-e2e.sh cluster-up

# Set KUBECONFIG for manual kubectl commands
export KUBECONFIG=$(pwd)/build/k3s-training.yaml

# 2. Deploy operator + gateway
./scripts/local-training-e2e.sh deploy

# 3. Run API (in one terminal) - starts Postgres
./scripts/local-training-e2e.sh api

# 4. Generate API key (in another terminal)
./scripts/local-training-e2e.sh gen-key

# 5. Run full training test
./scripts/local-training-e2e.sh test

# 6. Clean up when done
./scripts/local-training-e2e.sh cluster-down
```

### What the Test Does

1. Creates a TrainingSession via API (creates K8s CRD)
2. Waits for training pod to be ready
3. Port-forwards to training pod
4. Creates a training session in the backend
5. Loads `facebook/opt-125m` model with LoRA adapter
6. Runs 3 training steps (forward-backward + optim_step)
7. Tests text generation with the fine-tuned model
8. Saves/loads checkpoints (synced to R2 via FUSE storage)

### R2 Checkpoint Storage (FUSE)

The local e2e setup includes FUSE-based checkpoint storage that automatically syncs to Cloudflare R2. When you save a checkpoint, it's written to a FUSE mount that syncs to R2 in the background.

#### How It Works

```
Training Pod                    Storage Daemon (DaemonSet)           R2 Bucket
┌─────────────┐                ┌──────────────────────────┐         ┌─────────┐
│ /checkpoints│ ──hostPath──▶  │ FUSE Mount               │         │         │
│  (volume)   │                │ /var/lib/basilica/fuse/  │ ──sync──▶│ basilica│
└─────────────┘                │ u-testuser/              │         │         │
                               └──────────────────────────┘         └─────────┘
```

1. Training pod mounts `/checkpoints` from the host path `/var/lib/basilica/fuse/{namespace}/`
2. Storage daemon runs as a DaemonSet with a FUSE filesystem mounted at that path
3. When files are written, they're cached locally and synced to R2 in the background
4. Sync happens every 1 second with a 500ms quiet period (to coalesce writes)

#### Configure R2 Credentials

By default, the e2e script uses test credentials. To use your own R2 bucket:

```bash
# Set environment variables before running deploy
export R2_ENDPOINT="https://YOUR_ACCOUNT_ID.r2.cloudflarestorage.com"
export R2_ACCESS_KEY="your-access-key-id"
export R2_SECRET_KEY="your-secret-access-key"
export R2_BUCKET="your-bucket-name"

# Deploy with custom credentials
./scripts/local-training-e2e.sh deploy
```

Or update the secret directly:

```bash
export KUBECONFIG=$(pwd)/build/k3s-training.yaml

kubectl delete secret basilica-r2-credentials -n u-testuser
kubectl create secret generic basilica-r2-credentials \
    --namespace=u-testuser \
    --from-literal=endpoint=https://YOUR_ACCOUNT_ID.r2.cloudflarestorage.com \
    --from-literal=access_key_id=your-access-key-id \
    --from-literal=secret_access_key=your-secret-access-key \
    --from-literal=bucket=your-bucket-name \
    --from-literal=region=auto

# Restart storage daemon to pick up new credentials
kubectl delete pod -n basilica-storage -l app.kubernetes.io/component=fuse-daemon
```

#### Verify R2 Sync

Check storage daemon logs to see files being synced:

```bash
./scripts/local-training-e2e.sh logs storage

# Example output:
# Syncing 3 files
# Uploading 1779800 bytes to u-testuser/u-testuser/train-session-1/test-checkpoint-final/adapter_model.safetensors
# Successfully synced: /train-session-1/test-checkpoint-final/adapter_model.safetensors
```

Check mount status:

```bash
./scripts/local-training-e2e.sh status

# Shows active FUSE mounts under "=== Storage Mounts ==="
```

#### R2 Object Path

Checkpoints are stored in R2 with the following path structure:

```
{bucket}/{namespace}/{namespace}/{checkpoint_path}

Example:
basilica/u-testuser/u-testuser/train-session-1/test-checkpoint-final/adapter_model.safetensors
```

#### Troubleshooting FUSE Storage

**Mount not created:**
```bash
# Check storage daemon logs for credential errors
kubectl logs -n basilica-storage -l app.kubernetes.io/component=fuse-daemon | grep -i error

# Verify secret exists
kubectl get secret basilica-r2-credentials -n u-testuser

# Restart storage daemon after fixing credentials
kubectl delete pod -n basilica-storage -l app.kubernetes.io/component=fuse-daemon
```

**Files not syncing:**
```bash
# Check if mount is active
kubectl exec -n u-testuser <training-pod> -- ls -la /checkpoints/

# Should show .fuse_ready file if mount is working
```

**Verify pod has FUSE volume:**
```bash
kubectl get pod -n u-testuser <training-pod> -o jsonpath='{.spec.volumes}' | jq .

# Should include "basilica-checkpoint-storage" with hostPath
```

### Available Commands

| Command | Description |
|---------|-------------|
| `cluster-up` | Create k3d cluster |
| `deploy` | Install gateway + build images + deploy operator |
| `api` | Run Basilica API locally (starts Postgres) |
| `gen-key` | Generate API key and insert into Postgres |
| `reset-db` | Reset Postgres database (delete volume) |
| `test` | Run full E2E test with actual training steps |
| `cleanup` | Delete all existing training sessions |
| `status` | Show cluster status |
| `logs [operator\|training\|gateway]` | Show component logs |
| `cluster-down` | Delete the k3d cluster |

### Example Output

```
[STEP] Running training steps via port-forward...
[INFO] Training step 1/3
[INFO]   Loss: 6.26, Tokens: 10
[INFO]   Completed step: 1
[INFO] Training step 2/3
[INFO]   Loss: 5.74, Tokens: 10
[INFO]   Completed step: 2
[INFO] Training step 3/3
[INFO]   Loss: 5.75, Tokens: 10
[INFO]   Completed step: 3

[INFO] Training session status:
{
  "session_id": "train-session-1",
  "step_count": 3,
  "tokens_processed": 30,
  "lora_rank": 8
}
```

### KUBECONFIG

The e2e script creates a kubeconfig file at `build/k3s-training.yaml`. For manual kubectl commands, set:

```bash
export KUBECONFIG=$(pwd)/build/k3s-training.yaml

# Or use inline
KUBECONFIG=build/k3s-training.yaml kubectl get pods -A
```

The script commands (`test`, `status`, `logs`, etc.) automatically use this kubeconfig.

### CRD Status Reporting

The operator automatically polls the training service and updates the TrainingSession CRD status:

```bash
# Remember to set KUBECONFIG first
export KUBECONFIG=$(pwd)/build/k3s-training.yaml

kubectl get trainingsessions -A
# NAMESPACE    NAME          PHASE   STEPS   MODEL               AGE
# u-testuser   ts-06c46775   ready   3       facebook/opt-125m   5m
```

## API Endpoints

| Endpoint | Method | Description |
|----------|--------|-------------|
| `/health` | GET | Health check |
| `/sessions` | GET | List sessions |
| `/sessions` | POST | Create session |
| `/sessions/{id}` | GET | Get session status |
| `/sessions/{id}` | DELETE | Delete session |
| `/sessions/{id}/forward_backward` | POST | Compute gradients |
| `/sessions/{id}/optim_step` | POST | Apply gradients |
| `/sessions/{id}/sample` | POST | Generate text |
| `/sessions/{id}/save` | POST | Save checkpoint |
| `/sessions/{id}/load` | POST | Load checkpoint |

## Manual Local Development

### Run Standalone (without k3d)

```bash
# Install dependencies
cd services/training-service
pip install -e ".[dev]"

# Run locally (CPU mode)
MODEL_CACHE_DIR=/tmp/models \
CHECKPOINT_DIR=/tmp/checkpoints \
python -m uvicorn src.server:app --host 0.0.0.0 --port 8000
```

### Docker

```bash
# Build
docker build -t basilica-training:local -f services/training-service/Dockerfile services/training-service

# Run with GPU
docker run --gpus all -p 8000:8000 \
  -v /path/to/models:/models \
  -v /path/to/checkpoints:/checkpoints \
  basilica-training:local

# Run CPU-only (for testing)
docker run -p 8000:8000 \
  -v /path/to/models:/models \
  basilica-training:local
```

## Configuration

Environment variables:

| Variable | Default | Description |
|----------|---------|-------------|
| `MODEL_CACHE_DIR` | `/models` | HuggingFace model cache |
| `CHECKPOINT_DIR` | `/checkpoints` | Checkpoint storage |

## Example API Usage

### Create Session

```bash
curl -X POST http://localhost:8000/sessions \
  -H "Content-Type: application/json" \
  -d '{
    "session_id": "my-session",
    "base_model": "facebook/opt-125m",
    "lora_config": {"rank": 8, "alpha": 16},
    "optimizer_config": {"learning_rate": 0.0001}
  }'
```

### Forward-Backward

```bash
curl -X POST http://localhost:8000/sessions/my-session/forward_backward \
  -H "Content-Type: application/json" \
  -d '{
    "input_ids": [[2, 133, 2119, 6219, 23602, 13855, 81, 5, 22414, 2335]],
    "attention_mask": [[1, 1, 1, 1, 1, 1, 1, 1, 1, 1]],
    "labels": [[2, 133, 2119, 6219, 23602, 13855, 81, 5, 22414, 2335]]
  }'
```

### Optimizer Step

```bash
curl -X POST http://localhost:8000/sessions/my-session/optim_step
```

### Generate Sample

```bash
curl -X POST http://localhost:8000/sessions/my-session/sample \
  -H "Content-Type: application/json" \
  -d '{
    "prompt": "The quick brown",
    "max_tokens": 20,
    "temperature": 0.7
  }'
```

## Architecture

```
┌─────────────────────────────────────────────────────────────┐
│                    Basilica API                             │
│              (creates TrainingSession CRD)                  │
└─────────────────────┬───────────────────────────────────────┘
                      │
                      ▼
┌─────────────────────────────────────────────────────────────┐
│                 Basilica Operator                           │
│  • Reconciles TrainingSession CRD                           │
│  • Creates Pod + Service + NetworkPolicy                    │
│  • Polls training service for status updates                │
└─────────────────────┬───────────────────────────────────────┘
                      │
                      ▼
┌─────────────────────────────────────────────────────────────┐
│               Training Service Pod                          │
├─────────────────────────────────────────────────────────────┤
│              FastAPI Server (src/server.py)                 │
├─────────────────────────────────────────────────────────────┤
│              Training Backend (src/backend.py)              │
│  • Session management                                       │
│  • Forward-backward with gradient accumulation              │
│  • Optimizer step                                           │
│  • Text generation                                          │
│  • Checkpoint save/load                                     │
├─────────────────────────────────────────────────────────────┤
│           HuggingFace Transformers + PEFT                   │
│  • Model loading (OPT, LLaMA, etc.)                         │
│  • LoRA adapter (rank, alpha, dropout)                      │
│  • AdamW optimizer                                          │
└─────────────────────────────────────────────────────────────┘
```

## Troubleshooting

### Pod stuck in Pending
```bash
# Check events
kubectl describe pod -n u-testuser -l app=basilica-training
```

### Connection refused from operator
```bash
# Check NetworkPolicy exists
kubectl get networkpolicy -n u-testuser

# The operator creates allow-operator-training-{session} policy
```

### View training logs
```bash
./scripts/local-training-e2e.sh logs training
```

### Reset everything
```bash
./scripts/local-training-e2e.sh cleanup
./scripts/local-training-e2e.sh reset-db
./scripts/local-training-e2e.sh cluster-down
```

## License

Apache 2.0
