# Basilica Public API

HTTP gateway for accessing Basilica validators.

## Quick Start

```bash
# Build
cargo build -p public-api --release

# Generate config
./target/release/public-api --gen-config > config.toml

# Run
./target/release/public-api --config config.toml
```

## Testing

```bash
cd scripts/public-api
./test-run.sh  # Generates test wallet and runs the service
```

## Docker

```bash
# Build and extract binary
./scripts/public-api/build.sh

# Run
./public-api --config config.toml
```

## Configuration

Minimal `config.toml`:

```toml
[server]
bind_address = "0.0.0.0:8000"

[bittensor]
network = "finney"
netuid = 39

[auth]
allow_anonymous = true  # Set to false to require API keys
```

## API Endpoints

- `GET /health` - Health check
- `GET /docs` - Swagger UI
- `GET /api/v1/executors` - List available GPUs
- `POST /api/v1/rentals` - Rent GPU capacity
- `GET /api/v1/rentals/{id}` - Check rental status
- `GET /api/v1/rentals/{id}/logs` - Stream logs (SSE)

## Example Usage

```bash
# List executors
curl http://localhost:8000/api/v1/executors

# Rent GPU (with API key)
curl -X POST http://localhost:8000/api/v1/rentals \
  -H "X-API-Key: your-key" \
  -H "Content-Type: application/json" \
  -d '{
    "gpu_requirements": {"min_memory_gb": 40, "gpu_count": 1},
    "ssh_public_key": "ssh-rsa ...",
    "docker_image": "nvidia/cuda:12.0-base"
  }'
```

## Features

- Auto-discovers validators via Bittensor
- Load balances requests across validators
- Caches responses for better performance
- Rate limiting with API key tiers
- Real-time log streaming
