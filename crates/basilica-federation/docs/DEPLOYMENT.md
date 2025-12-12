# Federation Deployment Guide

## Prerequisites

- Multiple K3s clusters configured
- Kubeconfig files for each cluster
- Network connectivity between clusters
- Sufficient resources (CPU, memory)

## Deployment Steps

### 1. Prepare Configuration

Create a `federation.toml` configuration file:

```toml
name = "basilica-federation"

[[clusters]]
id = "cluster-us-east"
name = "US East Cluster"
region = "us-east-1"
kubeconfig = "/path/to/kubeconfig-us-east"
api_server = "https://k3s-us-east.example.com:6443"
priority = 100
enabled = true
```

### 2. Deploy Using Ansible

```bash
ansible-playbook -i inventories/production.ini \
  playbooks/01-setup/federation.yml \
  -e federated_clusters='[{"id":"cluster1","kubeconfig":"/path/to/kubeconfig"}]'
```

### 3. Deploy Using Kubernetes

```bash
# Apply manifests
kubectl apply -f orchestrator/k8s/services/federation/deployment.yaml
kubectl apply -f orchestrator/k8s/services/federation/servicemonitor.yaml
kubectl apply -f orchestrator/k8s/services/federation/prometheusrule.yaml

# Verify deployment
kubectl get pods -n basilica-federation
kubectl get svc -n basilica-federation
```

### 4. Verify Deployment

```bash
# Check health
curl http://localhost:8080/health

# List clusters
curl http://localhost:8080/api/v1/clusters

# Check metrics
curl http://localhost:9090/metrics
```

## Configuration Options

### Gateway Configuration

- `listen_addr`: Gateway listen address (default: "0.0.0.0")
- `port`: Gateway port (default: 8080)
- `request_timeout`: Request timeout (default: 30s)
- `max_concurrent_requests`: Max concurrent requests (default: 1000)

### Discovery Configuration

- `refresh_interval`: Service discovery refresh interval (default: 30s)
- `cache_ttl`: Service cache TTL (default: 60s)
- `enable_cross_cluster`: Enable cross-cluster discovery (default: true)

### Health Configuration

- `check_interval`: Health check interval (default: 10s)
- `check_timeout`: Health check timeout (default: 5s)
- `failure_threshold`: Failure threshold (default: 3)
- `success_threshold`: Success threshold (default: 2)

### Load Balancer Configuration

- `algorithm`: Load balancing algorithm (RoundRobin, LeastConnections, WeightedRoundRobin, Random, Geographic)
- `health_aware`: Enable health-aware routing (default: true)
- `region_aware`: Enable region-aware routing (default: false)

## Troubleshooting

### Common Issues

1. **Clusters not discovered**: Check kubeconfig files and network connectivity
2. **Health checks failing**: Verify cluster API server accessibility
3. **Load balancing not working**: Check cluster health status
4. **High latency**: Review network configuration and cluster locations

### Debugging

Enable debug logging:

```bash
basilica-federation --log-level debug
```

Check logs:

```bash
kubectl logs -n basilica-federation -l app=basilica-federation
```

## Scaling

To scale the federation gateway:

```bash
kubectl scale deployment basilica-federation -n basilica-federation --replicas=3
```

## Upgrading

1. Update configuration if needed
2. Update deployment image
3. Rolling restart:

```bash
kubectl rollout restart deployment basilica-federation -n basilica-federation
```

