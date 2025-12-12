# Federation Architecture

## Overview

The Basilica Federation system provides multi-cluster management capabilities for K3s clusters, enabling geographic distribution and high availability.

## Components

### API Gateway

The federation API gateway provides a unified REST API for accessing resources across all federated clusters. It handles:

- Request routing and proxying
- Load balancing across clusters
- Authentication and authorization
- Rate limiting and throttling
- Request/response transformation

### Service Discovery

Automatic service discovery across federated clusters:

- Cross-cluster service lookup
- Namespace-aware filtering
- Label-based filtering
- Caching for performance
- Automatic refresh

### Health Aggregation

Real-time health monitoring and aggregation:

- Cluster health checks
- Node status monitoring
- Component health tracking
- Failure threshold management
- Health status aggregation

### Load Balancer

Intelligent load balancing across clusters:

- Multiple algorithms (RoundRobin, LeastConnections, WeightedRoundRobin, Random, Geographic)
- Health-aware routing
- Region-aware routing
- Sticky sessions support

### Resource Manager

Federated resource management:

- Cross-cluster resource visibility
- Resource synchronization
- Distribution policies
- Quota management

## Data Flow

```
Client Request
    ↓
Federation API Gateway
    ↓
Load Balancer (selects cluster)
    ↓
Service Discovery (finds service)
    ↓
Health Check (verifies cluster health)
    ↓
Proxy to Target Cluster
    ↓
Response
```

## Configuration

Federation is configured via TOML file with:

- Cluster definitions
- Gateway settings
- Discovery configuration
- Health check settings
- Load balancing policies
- Resource management settings

## Security

- Kubeconfig-based authentication
- RBAC integration
- Secure inter-cluster communication
- Rate limiting
- Request validation

## Monitoring

Prometheus metrics are exposed for:

- Request counts and latencies
- Cluster health status
- Load balancer metrics
- Service discovery metrics
- Health check metrics
- Resource sync metrics

## Events

Event system for:

- Cluster health changes
- Service discovery events
- Load balancer selections
- Resource synchronization
- Error events

