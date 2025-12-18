# Basilica Training Service

GPU training service for fine-tuning LLMs with LoRA on Basilica.

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
