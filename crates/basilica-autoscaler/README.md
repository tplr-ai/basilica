# basilica-autoscaler

Basilica GPU node autoscaler: dynamic scaling of K3s GPU nodes.

[![Crates.io](https://img.shields.io/crates/v/basilica-autoscaler.svg)](https://crates.io/crates/basilica-autoscaler)
[![Documentation](https://docs.rs/basilica-autoscaler/badge.svg)](https://docs.rs/basilica-autoscaler)
[![License: MIT](https://img.shields.io/badge/License-MIT-yellow.svg)](https://opensource.org/licenses/MIT)

[Documentation](https://docs.rs/basilica-autoscaler) | [Repository](https://github.com/one-covenant/basilica) | [Website](https://basilica.ai)

## Overview

`basilica-autoscaler` provides automatic scaling of GPU nodes in K3s clusters based on demand. It monitors workload requirements and dynamically provisions or removes GPU nodes to optimize cost and availability.

## Installation

Add to your `Cargo.toml`:

```toml
[dependencies]
basilica-autoscaler = "0.1"
```

Or deploy as a Kubernetes controller.

## Features

- **Demand-Based Scaling**: Scale nodes based on pending workload requirements
- **GPU-Aware**: Understands GPU types and allocates appropriate nodes
- **Cost Optimization**: Minimize costs by scaling down idle resources
- **K3s Integration**: Native integration with K3s clusters
- **Custom Resources**: Define scaling policies via Kubernetes CRDs

## Architecture

```
┌─────────────────┐
│  K8s API Server │
│  (Watch Events) │
└────────┬────────┘
         │
         ▼
┌─────────────────┐     ┌─────────────────┐
│   Autoscaler    │────▶│  Cloud Provider │
│   Controller    │     │   (Provision)   │
└────────┬────────┘     └─────────────────┘
         │
         ▼
┌─────────────────┐
│   GPU Nodes     │
│   (K3s Agents)  │
└─────────────────┘
```

## Example

```rust
use basilica_autoscaler::{AutoscalerConfig, Autoscaler};

#[tokio::main]
async fn main() -> Result<(), Box<dyn std::error::Error>> {
    // Load configuration
    let config = AutoscalerConfig::from_env()?;
    
    // Create and start the autoscaler
    let autoscaler = Autoscaler::new(config).await?;
    autoscaler.run().await?;
    
    Ok(())
}
```

## Custom Resource Definition

```yaml
apiVersion: basilica.ai/v1
kind: GpuScalingPolicy
metadata:
  name: h100-scaling
spec:
  gpuType: h100
  minNodes: 1
  maxNodes: 10
  scaleUpThreshold: 0.8
  scaleDownThreshold: 0.2
  cooldownPeriodSeconds: 300
```

## Configuration

```toml
[autoscaler]
poll_interval_seconds = 30
scale_up_delay_seconds = 60
scale_down_delay_seconds = 300

[cluster]
kubeconfig_path = "~/.kube/config"
namespace = "basilica-system"

[provider]
type = "hyperstack"
api_key_env = "HYPERSTACK_API_KEY"
```

## Metrics

The autoscaler exports Prometheus metrics:

- `basilica_autoscaler_nodes_total` - Current node count by GPU type
- `basilica_autoscaler_scale_up_total` - Scale-up events
- `basilica_autoscaler_scale_down_total` - Scale-down events
- `basilica_autoscaler_pending_pods` - Pending GPU workloads

## Related Crates

- [`basilica-operator`](https://crates.io/crates/basilica-operator) - K8s operator
- [`basilica-common`](https://crates.io/crates/basilica-common) - Core types

## License

MIT License - see [LICENSE](LICENSE) for details.

