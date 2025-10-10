# Basilica API

HTTP gateway for accessing Basilica validators.

## Quick Start

```bash
# Build
cargo build -p basilica-api --release

# Generate config
./target/release/basilica-api --gen-config > config.toml

# Run
./target/release/basilica-api --config config.toml
```

## Testing

```bash
cd scripts/basilica-api
./test-run.sh  # Generates test wallet and runs the service
```

## Docker

```bash
# Build and extract binary
./scripts/basilica-api/build.sh

# Run
./basilica-api --config config.toml
```

## Configuration

Minimal `config.toml`:

```toml
[server]
bind_address = "0.0.0.0:8000"

[bittensor]
network = "finney"
netuid = 39

# Note: Auth0 JWT authentication is required for protected endpoints
```

## API Endpoints

- `GET /health` - Health check
- `GET /docs` - Swagger UI
- `GET /api/v1/nodes` - List available GPUs
- `POST /api/v1/rentals` - Rent GPU capacity
- `GET /api/v1/rentals/{id}` - Check rental status
- `GET /api/v1/rentals/{id}/logs` - Stream logs (SSE)

## Example Usage

```bash
# List nodes
curl http://localhost:8000/api/v1/nodes

# Rent GPU (requires Auth0 JWT token)
curl -X POST http://localhost:8000/api/v1/rentals \
  -H "Authorization: Bearer <YOUR_AUTH0_JWT_TOKEN>" \
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
- Auth0 JWT-based authentication
- Rate limiting per user
- Real-time log streaming
