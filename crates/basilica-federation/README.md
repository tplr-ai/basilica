# Basilica Federation

Multi-cluster federation system for managing multiple K3s clusters with geographic distribution and high availability.

## Features

- **Multi-cluster API Gateway**: Unified API for accessing resources across clusters
- **Cross-cluster Service Discovery**: Automatic service discovery across federated clusters
- **Federated Resource Management**: Manage resources across multiple clusters
- **Cluster Health Aggregation**: Aggregate health status from all clusters
- **Cross-cluster Load Balancing**: Intelligent load balancing across clusters

## Architecture

```
┌─────────────────────────────────────────────────────────┐
│              Federation API Gateway                     │
│  ┌──────────┐  ┌──────────┐  ┌──────────┐             │
│  │Discovery │  │  Health  │  │   Load   │             │
│  │          │  │          │  │ Balancer │             │
│  └──────────┘  └──────────┘  └──────────┘             │
└─────────────────────────────────────────────────────────┘
         │              │              │
         └──────────────┼──────────────┘
                        │
        ┌───────────────┼───────────────┐
        │               │                 │
   ┌─────────┐    ┌─────────┐      ┌─────────┐
   │Cluster 1│    │Cluster 2│      │Cluster N│
   │(Region A)│    │(Region B)│      │(Region C)│
   └─────────┘    └─────────┘      └─────────┘
```

## Configuration

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

[clusters.tags]
environment = "production"
datacenter = "us-east"

[[clusters]]
id = "cluster-eu-west"
name = "EU West Cluster"
region = "eu-west-1"
kubeconfig = "/path/to/kubeconfig-eu-west"
api_server = "https://k3s-eu-west.example.com:6443"
priority = 90
enabled = true

[gateway]
listen_addr = "0.0.0.0"
port = 8080
request_timeout = "30s"
max_concurrent_requests = 1000
enable_logging = true

[gateway.rate_limit]
requests_per_second = 100
burst_size = 200

[discovery]
refresh_interval = "30s"
cache_ttl = "60s"
enable_cross_cluster = true

[health]
check_interval = "10s"
check_timeout = "5s"
failure_threshold = 3
success_threshold = 2
enable_metrics = true

[load_balancer]
algorithm = "RoundRobin"
health_aware = true
region_aware = false

[resource_manager]
sync_interval = "60s"
auto_distribute = false
distribution_policy = "Even"
enable_quotas = true
```

## Usage

### Running the Federation Gateway

```bash
# Using default config
basilica-federation

# Using custom config
basilica-federation --config /path/to/federation.toml

# With custom log level
basilica-federation --log-level debug
```

### API Endpoints

#### Health Check
```bash
curl http://localhost:8080/health
```

#### List Clusters
```bash
curl http://localhost:8080/clusters
```

#### Get Cluster Details
```bash
curl http://localhost:8080/clusters/cluster-us-east
```

#### List Services
```bash
curl http://localhost:8080/services
curl http://localhost:8080/services?namespace=default
```

#### Proxy Request
```bash
curl http://localhost:8080/proxy/api/v1/namespaces/default/pods
```

#### List Pods
```bash
curl http://localhost:8080/api/v1/namespaces/default/pods
```

## Deployment

### Using Ansible

```bash
ansible-playbook -i inventories/production.ini \
  playbooks/01-setup/federation.yml \
  -e federated_clusters='[{"id":"cluster1","kubeconfig":"/path/to/kubeconfig"}]'
```

### Using Kubernetes

```bash
kubectl apply -f orchestrator/k8s/services/federation/deployment.yaml
```

## Load Balancing Algorithms

- **RoundRobin**: Distribute requests evenly across clusters
- **LeastConnections**: Route to cluster with fewest active connections
- **WeightedRoundRobin**: Use cluster priority for weighted distribution
- **Random**: Random cluster selection
- **Geographic**: Route based on client geographic location

## Health Checks

The federation system continuously monitors cluster health:

- Node status (ready/not ready)
- API server availability
- Component health
- Resource availability

Clusters are automatically excluded from load balancing if they become unhealthy.

## Service Discovery

Services are automatically discovered across all federated clusters:

- Cross-cluster service lookup
- Namespace-aware filtering
- Label-based filtering
- Cached for performance

## Resource Management

Federated resource management provides:

- Cross-cluster resource visibility
- Resource distribution policies
- Quota management
- Automatic resource balancing

## Metrics

Prometheus metrics are exposed at `/metrics`:

- Request counts and latencies
- Cluster health status
- Load balancer metrics
- Service discovery metrics

## Security

- Kubeconfig-based authentication
- RBAC integration
- Secure inter-cluster communication
- Rate limiting and request validation

## Development

```bash
# Build
cargo build

# Run tests
cargo test

# Run with logging
RUST_LOG=debug cargo run
```

## License

MIT

