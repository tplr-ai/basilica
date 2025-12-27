# TUI Integration Testing

This directory contains tools for testing the Basilica TUI against a local API instance.

## Quick Start

### Option 1: TUI Dev Mode (No API required)

The simplest way to test the TUI is using its built-in dev mode with mock data:

```bash
cargo run -p basilica-tui -- --dev
```

This runs the TUI with mock data - no API connection required.

### Option 2: Test Against Local API

Start the API in dev mode (bypasses Bittensor/Validator):

```bash
# Start services
cd scripts/tui-test
docker compose up -d

# Wait for API to be ready
docker compose logs -f api

# In another terminal, run TUI against local API
BASILICA_API_URL=http://localhost:8000 cargo run -p basilica-tui
```

### Option 3: Run Integration Tests

```bash
# Start services
cd scripts/tui-test
docker compose up -d

# Run integration tests
cargo test -p basilica-tui --test integration
```

## Services

| Service  | Port | Description |
|----------|------|-------------|
| API      | 8000 | Basilica API Gateway (dev mode) |
| Postgres | 5432 | Database |
| Metrics  | 9401 | Prometheus metrics |

## Configuration

The API runs with `--dev` flag which:
- Bypasses Bittensor network connection
- Uses mock validator discovery
- Skips health check tasks that require a real validator
- Still uses a real PostgreSQL database for proper API behavior

## Stopping Services

```bash
docker compose down

# Remove volumes too:
docker compose down -v
```

## Troubleshooting

### API won't start
Check logs: `docker compose logs api`

### Database connection issues
Ensure postgres is healthy: `docker compose ps`

### TUI can't connect to API
- Verify API is running: `curl http://localhost:8000/health`
- Check `BASILICA_API_URL` environment variable

