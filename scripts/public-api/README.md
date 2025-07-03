# Basilica Public API

External HTTP API service for the Basilica network.

## Quick Start

```bash
# Build
./build.sh

# Deploy
./deploy.sh user@host port

# Run locally
docker-compose -f compose.dev.yml up
```

## Files

- `Dockerfile` - Container image definition
- `build.sh` - Build script for Docker image
- `deploy.sh` - Remote deployment script
- `compose.prod.yml` - Production docker-compose with Redis and watchtower
- `compose.dev.yml` - Development docker-compose with local build
- `.env.example` - Environment variables template

## Configuration

Copy and edit the configuration:
```bash
cp ../../config/public-api.toml.example ../../config/public-api.toml
```

Key settings:
- `[server]` - Port 8000 for HTTP API
- `[validator]` - gRPC endpoint for validator connection
- `[redis]` - Cache backend configuration
- `[rate_limiting]` - Request rate limits

## Ports

- `8000` - HTTP API
- `8080` - Metrics endpoint

## API Endpoints

- `GET /health` - Health check
- `POST /api/v1/compute/submit` - Submit compute request
- `GET /api/v1/compute/{id}/status` - Check request status
- `GET /api/v1/executors` - List available executors
- `GET /api/v1/validators` - List active validators

## Environment Variables

Create `.env` from `.env.example`:
```bash
RUST_LOG=info
RUST_BACKTRACE=1
```

## Dependencies

- Redis for caching and rate limiting
- Validator service for backend operations

## Deployment

```bash
# Deploy to production
./deploy.sh root@api.basilica.ai 22

# Check logs
ssh root@api.basilica.ai 'cd /opt/basilica && docker-compose logs -f public-api'
```

## Rate Limiting

Default limits:
- 100 requests/minute per IP
- 10 requests/minute for compute submissions
- Configurable per endpoint in `public-api.toml`