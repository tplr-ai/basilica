# basilica-operator

Basilica Kubernetes operator: CRDs and controllers for GPU workload orchestration.

[![Crates.io](https://img.shields.io/crates/v/basilica-operator.svg)](https://crates.io/crates/basilica-operator)
[![Documentation](https://docs.rs/basilica-operator/badge.svg)](https://docs.rs/basilica-operator)
[![License: MIT](https://img.shields.io/badge/License-MIT-yellow.svg)](https://opensource.org/licenses/MIT)

[Documentation](https://docs.rs/basilica-operator) | [Repository](https://github.com/one-covenant/basilica) | [Website](https://basilica.ai)

## Overview

`basilica-operator` is a Kubernetes operator that manages GPU workloads on K3s clusters. It provides Custom Resource Definitions (CRDs) for declarative workload management and handles the full lifecycle of GPU rentals.

## Installation

Add to your `Cargo.toml`:

```toml
[dependencies]
basilica-operator = "0.1"
```

Or deploy to your cluster:

```bash
kubectl apply -f https://basilica.ai/operator/install.yaml
```

## Features

- **Custom Resources**: `UserDeployment`, `GpuRental`, `GpuNode` CRDs
- **Lifecycle Management**: Full workload lifecycle from creation to cleanup
- **Node Onboarding**: Automatic GPU node discovery and registration
- **Resource Scheduling**: GPU-aware scheduling with affinity rules
- **Health Monitoring**: Continuous health checks and auto-recovery

## Custom Resource Definitions

### UserDeployment

```yaml
apiVersion: basilica.ai/v1
kind: UserDeployment
metadata:
  name: my-training-job
spec:
  image: pytorch/pytorch:latest
  gpuType: h100
  gpuCount: 4
  command: ["python", "train.py"]
  env:
    - name: MODEL_NAME
      value: "llama-7b"
  storage:
    - name: data
      size: 100Gi
      mountPath: /data
```

### GpuRental

```yaml
apiVersion: basilica.ai/v1
kind: GpuRental
metadata:
  name: rental-abc123
spec:
  userId: user_123
  gpuType: a100
  duration: 3600
  status: active
```

## Architecture

```
┌─────────────────┐
│  K8s API Server │
└────────┬────────┘
         │ Watch
         ▼
┌─────────────────────────────┐
│      Basilica Operator      │
│                             │
│  ┌─────────────────────┐   │
│  │ UserDeployment Ctrl │   │
│  └──────────┬──────────┘   │
│             │               │
│  ┌──────────▼──────────┐   │
│  │   GpuRental Ctrl    │   │
│  └──────────┬──────────┘   │
│             │               │
│  ┌──────────▼──────────┐   │
│  │   GpuNode Ctrl      │   │
│  └─────────────────────┘   │
└─────────────────────────────┘
         │
         ▼
┌─────────────────┐
│   GPU Nodes     │
│   (K3s Agents)  │
└─────────────────┘
```

## Example

```rust
use basilica_operator::{OperatorConfig, Operator};

#[tokio::main]
async fn main() -> Result<(), Box<dyn std::error::Error>> {
    // Initialize the operator
    let config = OperatorConfig::from_env()?;
    let operator = Operator::new(config).await?;
    
    // Start reconciliation loops
    operator.run().await?;
    
    Ok(())
}
```

## Configuration

```toml
[operator]
namespace = "basilica-system"
reconcile_interval_seconds = 30

[metrics]
port = 9090
enabled = true

[health]
port = 8080
```

## Metrics

Prometheus metrics exported:

- `basilica_operator_reconcile_total` - Reconciliation count by resource type
- `basilica_operator_reconcile_duration_seconds` - Reconciliation duration
- `basilica_operator_active_rentals` - Currently active GPU rentals
- `basilica_operator_gpu_nodes` - Registered GPU nodes by type

## Related Crates

- [`basilica-autoscaler`](https://crates.io/crates/basilica-autoscaler) - Node autoscaling
- [`basilica-api`](https://crates.io/crates/basilica-api) - API gateway
- [`basilica-common`](https://crates.io/crates/basilica-common) - Core types

## License

MIT License - see [LICENSE](LICENSE) for details.

