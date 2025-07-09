# Monitoring Guide

This guide covers setting up monitoring for your Basilica network deployment using the included Prometheus and Grafana configuration.

## Overview

Basilica services provide built-in metrics that can be monitored using Prometheus and Grafana. The project includes pre-configured monitoring setup for easy deployment.

## Available Monitoring Setup

The repository includes monitoring configurations in:
- `config/monitoring/prometheus.yml` - Prometheus configuration
- `config/monitoring/grafana-datasources.yml` - Grafana datasource configuration  
- `scripts/localtest/docker-compose-monitoring.yml` - Monitoring stack deployment

## Quick Start

### 1. Enable Metrics in Your Services

Ensure metrics are enabled in your configuration files:

```toml
# In validator.toml, miner.toml, executor.toml
[metrics]
enabled = true

[metrics.prometheus]
host = "0.0.0.0"
port = 9090
path = "/metrics"
```

### 2. Deploy Monitoring Stack

#### Option A: Using localtest environment

```bash
# Navigate to localtest directory
cd scripts/localtest

# Start services with monitoring
docker compose -f docker-compose.yml -f docker-compose-monitoring.yml up -d

# Access monitoring:
# - Grafana: http://localhost:3000 (admin/admin)
# - Prometheus: http://localhost:9090
```

#### Option B: Standalone monitoring

```bash
# Create a simple monitoring compose file
cat > docker-compose.monitoring.yml << 'EOF'
services:
  prometheus:
    image: prom/prometheus:latest
    container_name: basilica-prometheus
    ports:
      - "9090:9090"
    volumes:
      - ./config/monitoring/prometheus.yml:/etc/prometheus/prometheus.yml:ro
      - prometheus-data:/prometheus
    command:
      - '--config.file=/etc/prometheus/prometheus.yml'
      - '--storage.tsdb.path=/prometheus'
      - '--storage.tsdb.retention.time=30d'

  grafana:
    image: grafana/grafana:latest
    container_name: basilica-grafana
    ports:
      - "3000:3000"
    volumes:
      - ./config/monitoring/grafana-datasources.yml:/etc/grafana/provisioning/datasources/datasources.yml:ro
      - grafana-data:/var/lib/grafana
    environment:
      - GF_SECURITY_ADMIN_PASSWORD=admin

volumes:
  prometheus-data:
  grafana-data:
EOF

# Start monitoring
docker compose -f docker-compose.monitoring.yml up -d
```

### 3. Verify Metrics Collection

**Check individual service metrics:**
```bash
# Validator metrics (adjust port based on your config)
curl http://localhost:8080/metrics

# Miner metrics  
curl http://localhost:8080/metrics

# Executor metrics
curl http://localhost:8080/metrics
```

**Check Prometheus targets:**
- Open http://localhost:9090/targets
- Verify all services show as "UP"

**Access Grafana:**
- Open http://localhost:3000
- Login: admin/admin
- Prometheus should be pre-configured as a datasource

## Key Metrics to Monitor

### System Health
- `basilica_cpu_usage_percent` - CPU utilization
- `basilica_memory_usage_bytes` - Memory usage  
- `basilica_gpu_utilization_percent` - GPU utilization (executor only)

### Service Performance
- `basilica_validator_validations_total` - Total validations performed
- `basilica_miner_executor_health_checks_total` - Executor health checks
- `basilica_executor_grpc_requests_total` - gRPC requests handled

### Network Activity
- `basilica_validator_ssh_connections_total` - SSH connections
- `basilica_miner_ssh_sessions_active` - Active SSH sessions

## Production Monitoring

For production deployments, add monitoring to your existing service deployments:

```bash
# Use the production compose files with monitoring overlay
cd scripts/validator
docker compose -f compose.prod.yml -f ../../scripts/localtest/docker-compose-monitoring.yml up -d

# Or for miner
cd scripts/miner  
docker compose -f compose.prod.yml -f ../../scripts/localtest/docker-compose-monitoring.yml up -d
```

## Creating Simple Dashboards

In Grafana, create basic panels with these queries:

**System Overview:**
```promql
# CPU Usage
basilica_cpu_usage_percent

# Memory Usage  
basilica_memory_usage_bytes / 1024 / 1024 / 1024

# Service Uptime
up{job=~"basilica.*"}
```

**Validation Activity:**
```promql
# Validation Rate
rate(basilica_validator_validations_total[5m])

# SSH Connections
rate(basilica_validator_ssh_connections_total[5m])
```

## Troubleshooting

**No metrics showing up?**
- Check that `[metrics] enabled = true` in your service config
- Verify the service is running: `docker ps`
- Test metrics endpoint: `curl http://localhost:8080/metrics`
- Check Prometheus targets: http://localhost:9090/targets

**Grafana can't connect to Prometheus?**
- Verify both containers are running: `docker compose ps`
- Check Docker network connectivity
- Use the correct datasource URL: `http://prometheus:9090`

**Services not appearing in Prometheus?**
- Check the prometheus.yml configuration matches your service ports
- Verify services are exposing metrics on the configured ports
- Check Docker container network connectivity

## Configuration Reference

The monitoring setup uses these default ports:
- **Validator**: 8080 (metrics)
- **Miner**: 9090 (metrics) 
- **Executor**: 9090 (metrics)
- **Prometheus**: 9090 (web UI)
- **Grafana**: 3000 (web UI)

Adjust the `config/monitoring/prometheus.yml` file if your services use different ports.