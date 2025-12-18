# Basilica Training Service

GPU training service for fine-tuning LLMs with LoRA on Basilica.

## Overview

The training service provides a high-level API for:
- **LoRA fine-tuning** with HuggingFace + PEFT
- **Forward-backward passes** with gradient accumulation
- **Text generation** with the fine-tuned model
- **Checkpoint management** (save/load)

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

## Quick Start

### Local Development

```bash
# Install dependencies
cd services/training-service
pip install -e ".[dev]"

# Run locally (requires GPU)
MODEL_CACHE_DIR=/tmp/models \
CHECKPOINT_DIR=/tmp/checkpoints \
python -m uvicorn src.server:app --host 0.0.0.0 --port 8000
```

### Docker

```bash
# Build
docker build -t basilica/training:latest .

# Run with GPU
docker run --gpus all -p 8000:8000 \
  -v /path/to/models:/models \
  -v /path/to/checkpoints:/checkpoints \
  basilica/training:latest
```

### Docker Compose

```bash
# From repo root
docker-compose -f docker-compose.training.yml up -d
```

## Configuration

Environment variables:

| Variable | Default | Description |
|----------|---------|-------------|
| `MODEL_CACHE_DIR` | `/models` | HuggingFace model cache |
| `CHECKPOINT_DIR` | `/checkpoints` | Checkpoint storage |

## Example Usage

### Create Session

```bash
curl -X POST http://localhost:8000/sessions \
  -H "Content-Type: application/json" \
  -d '{
    "session_id": "my-session",
    "base_model": "meta-llama/Llama-3.1-8B-Instruct",
    "lora_config": {"rank": 32, "alpha": 64},
    "optimizer_config": {"learning_rate": 0.0001}
  }'
```

### Forward-Backward

```bash
curl -X POST http://localhost:8000/sessions/my-session/forward_backward \
  -H "Content-Type: application/json" \
  -d '{
    "input_ids": [[1, 2, 3, 4, 5]],
    "attention_mask": [[1, 1, 1, 1, 1]],
    "labels": [[1, 2, 3, 4, 5]]
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
    "prompt": "Hello, world!",
    "max_tokens": 50,
    "temperature": 0.7
  }'
```

### Save Checkpoint

```bash
curl -X POST http://localhost:8000/sessions/my-session/save \
  -H "Content-Type: application/json" \
  -d '{"checkpoint_name": "checkpoint-1000"}'
```

## Python SDK

```python
from basilica.training import TrainingClient, Datum

# Initialize client
client = TrainingClient(api_key="your-api-key")

# Create session
session = client.create_session(
    base_model="meta-llama/Llama-3.1-8B-Instruct",
    rank=32,
    learning_rate=1e-4,
)

# Training loop
for batch in dataloader:
    result = session.forward_backward(batch)
    print(f"Loss: {result.loss:.4f}")
    session.optim_step()

# Save checkpoint
session.save_state("checkpoint-final")

# Generate sample
sample = session.sample("Hello!", max_tokens=50)
print(sample.text)

# Cleanup
session.close()
```

## Testing

```bash
# Run tests
pytest tests/

# Run with GPU tests (requires GPU)
pytest tests/ --gpu
```

## Architecture

```
┌─────────────────────────────────────────┐
│           FastAPI Server                │
│         (src/server.py)                 │
├─────────────────────────────────────────┤
│         Training Backend                │
│         (src/backend.py)                │
│  • Session management                   │
│  • Forward-backward                     │
│  • Optimizer step                       │
│  • Text generation                      │
│  • Checkpoint save/load                 │
├─────────────────────────────────────────┤
│    HuggingFace Transformers + PEFT      │
│  • Model loading                        │
│  • LoRA adapter                         │
│  • AdamW optimizer                      │
└─────────────────────────────────────────┘
```

## License

Apache 2.0
