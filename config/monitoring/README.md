# Shared Monitoring Configuration

This directory contains shared monitoring configurations used by both development (`docker/`) and testing (`scripts/localtest/`) environments.

## Files

- `prometheus.yml` - Prometheus scrape configuration
- `grafana-datasources.yml` - Grafana datasource configuration

## Usage

### In docker-compose files:

```yaml
services:
  prometheus:
    image: prom/prometheus:latest
    volumes:
      - ../../config/monitoring/prometheus.yml:/etc/prometheus/prometheus.yml:ro
```

### Environment Variables

The prometheus configuration supports environment variable substitution:

- `ENVIRONMENT` - Sets the environment label (default: development)

## Metrics Endpoints

| Service | Container Names | Port | Path |
|---------|----------------|------|------|
| Miner | miner, basilica-miner | 9091, 9092 | /metrics |
| Executor | executor, basilica-executor | 9090, 9092 | /metrics |
| Validator | validator, basilica-validator | 8081 | /metrics |
| GPU Exporter | executor, basilica-executor | 9835 | /metrics |

## Adding New Services

To add a new service to monitoring:

1. Add the scrape config to `prometheus.yml`
2. Use multiple target names to support different environments
3. Document the endpoint in this README