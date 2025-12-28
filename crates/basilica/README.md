# basilica

[![Crates.io](https://img.shields.io/crates/v/basilica.svg)](https://crates.io/crates/basilica)
[![Documentation](https://docs.rs/basilica/badge.svg)](https://docs.rs/basilica)
[![License: MIT](https://img.shields.io/badge/License-MIT-yellow.svg)](https://opensource.org/licenses/MIT)

**Decentralized GPU marketplace on the Bittensor network.**

Basilica enables GPU compute rental through a decentralized marketplace where miners provide hardware, validators verify availability, and users deploy workloads via a simple SDK.

## Quick Start

```toml
[dependencies]
basilica = "0.1"
```

```rust,ignore
use basilica::prelude::*;

#[tokio::main]
async fn main() -> Result<(), Box<dyn std::error::Error>> {
    let client = BasilicaClient::builder()
        .api_key("your-api-key")
        .build()
        .await?;

    let deployment = client.create_deployment(CreateDeploymentRequest {
        image: "pytorch/pytorch:2.0-cuda11.8".into(),
        gpu_count: Some(1),
        ..Default::default()
    }).await?;

    println!("Deployed: {}", deployment.id);
    Ok(())
}
```

## Features

| Feature | Description | Default |
|---------|-------------|---------|
| `sdk` | High-level client SDK | ✅ |
| `cli` | Command-line interface | ❌ |
| `validator` | Run a validator node | ❌ |
| `miner` | Run a miner node | ❌ |
| `api` | REST API server | ❌ |
| `full` | Everything | ❌ |

```toml
# Just the SDK (default)
basilica = "0.1"

# SDK + CLI
basilica = { version = "0.1", features = ["cli"] }

# Run a miner node
basilica = { version = "0.1", features = ["miner"] }

# Everything
basilica = { version = "0.1", features = ["full"] }
```

## Component Crates

This umbrella crate re-exports:

| Crate | Description |
|-------|-------------|
| [basilica-common](https://crates.io/crates/basilica-common) | Core types and utilities |
| [basilica-protocol](https://crates.io/crates/basilica-protocol) | gRPC protocol definitions |
| [basilica-sdk](https://crates.io/crates/basilica-sdk) | High-level client SDK |
| [basilica-cli](https://crates.io/crates/basilica-cli) | Command-line interface |
| [basilica-validator](https://crates.io/crates/basilica-validator) | Validator node |
| [basilica-miner](https://crates.io/crates/basilica-miner) | Miner node |
| [basilica-api](https://crates.io/crates/basilica-api) | REST API server |

## Links

- [Documentation](https://docs.rs/basilica)
- [GitHub](https://github.com/one-covenant/basilica)
- [Website](https://basilica.ai)
- [Bittensor](https://bittensor.com)

## License

MIT OR Apache-2.0

