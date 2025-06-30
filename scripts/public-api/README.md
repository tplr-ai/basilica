# Basilica Public API Gateway Build Scripts

This directory contains build and deployment scripts for the Basilica Public API Gateway.

## Overview

The Public API Gateway provides a unified HTTP interface to the Basilica validator network, offering:

- Smart load balancing across validators
- API key authentication and rate limiting
- Response caching
- OpenAPI documentation
- Health monitoring and telemetry

## Building

### Local Build

```bash
./build.sh
```

Build options:

- `PROFILE=debug` - Build debug version (default: release)
- `TARGET=x86_64-unknown-linux-musl` - Build for different target
- `FEATURES=redis` - Enable additional features

### Docker Build

```bash
docker build -f Dockerfile -t basilica/public-api:latest ../..
```

Or using docker-compose:

```bash
docker-compose build
```

## Running

### Local Execution

```bash
# Generate example configuration
./build/public-api --gen-config > public-api.toml

# Run with configuration
./build/public-api --config public-api.toml
```

### Docker Execution

```bash
# Start all services
docker-compose up -d

# View logs
docker-compose logs -f public-api

# Stop services
docker-compose down
```

## Configuration

The Public API Gateway supports configuration through:

1. TOML configuration file
2. Environment variables (prefixed with `PUBLIC_API__`)
3. Command line arguments

Key configuration sections:

- `server` - HTTP server settings
- `bittensor` - Network and validator discovery
- `load_balancer` - Load balancing strategy
- `cache` - Response caching settings
- `rate_limit` - API rate limiting
- `auth` - Authentication settings
- `telemetry` - Metrics and tracing

## API Endpoints

### Public Endpoints

- `GET /health` - Health check
- `GET /api-docs/openapi.json` - OpenAPI specification
- `GET /swagger-ui` - Interactive API documentation

### Authenticated Endpoints

- `POST /rentals` - Rent GPU capacity
- `GET /rentals/{id}` - Get rental status
- `POST /rentals/{id}/terminate` - Terminate rental
- `GET /rentals/{id}/logs` - Stream rental logs
- `GET /executors` - List available executors
- `GET /validators` - List active validators
- `GET /telemetry` - Export metrics

## Monitoring

### Prometheus Metrics

Metrics are exposed at `/metrics` endpoint:

- Request latencies
- Response status codes
- Cache hit rates
- Active connections
- Validator health status

### Health Checks

The `/health` endpoint provides:

- Service status
- Validator pool health
- Cache status
- Database connectivity

## Development

### Running Tests

```bash
cargo test -p public-api
```

### Local Development Setup

1. Start local Bittensor network
2. Configure validators
3. Run the public API:

   ```bash
   RUST_LOG=debug cargo run -p public-api -- --config dev.toml
   ```

## Production Deployment

### Kubernetes

See `deployment/k8s/public-api.yaml` for Kubernetes manifests.

### Systemd

```bash
sudo cp basilica-public-api.service /etc/systemd/system/
sudo systemctl enable basilica-public-api
sudo systemctl start basilica-public-api
```

## Security Considerations

1. Always use HTTPS in production
2. Rotate JWT secrets regularly
3. Configure rate limiting appropriately
4. Use strong API keys
5. Enable request logging for audit trails
6. Configure CORS origins restrictively

## Troubleshooting

### Common Issues

1. **No validators available**
   - Check Bittensor network connectivity
   - Verify subnet ID configuration
   - Check validator discovery logs

2. **High latency**
   - Enable caching
   - Check load balancer strategy
   - Monitor validator health

3. **Rate limit errors**
   - Adjust rate limit configuration
   - Use API key for higher limits
   - Enable Redis for distributed rate limiting

### Debug Mode

Enable detailed logging:

```bash
RUST_LOG=debug,public_api=trace ./public-api --config public-api.toml
```

## License

See the main project LICENSE file.
