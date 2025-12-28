# basilica-validator

Basilica Validator - Bittensor neuron for GPU hardware verification and miner scoring.

[![Crates.io](https://img.shields.io/crates/v/basilica-validator.svg)](https://crates.io/crates/basilica-validator)
[![Documentation](https://docs.rs/basilica-validator/badge.svg)](https://docs.rs/basilica-validator)
[![License: MIT](https://img.shields.io/badge/License-MIT-yellow.svg)](https://opensource.org/licenses/MIT)

[Documentation](https://docs.rs/basilica-validator) | [Repository](https://github.com/one-covenant/basilica) | [Website](https://basilica.ai)

## Overview

`basilica-validator` is a Bittensor subnet validator that verifies GPU hardware capabilities and scores miners on the Basilica network. It uses SSH-based direct verification where validators SSH directly to miners' GPU nodes, eliminating intermediary trust requirements.

## Installation

Add to your `Cargo.toml`:

```toml
[dependencies]
basilica-validator = "0.1"
```

Or install the binary:

```bash
curl -sSL https://basilica.ai/install.sh | bash
```

## Features

- **Hardware Verification**: Binary validation system for secure GPU verification
- **SSH-Based Verification**: Direct SSH access to miner nodes for trustless validation
- **Bittensor Integration**: Native participation in Bittensor consensus with weight allocation
- **GPU Profiling**: Automatic detection and profiling of GPU capabilities
- **REST API**: External access to validator data and status

## Feature Flags

- `client` - Enable HTTP client for external services (default)
- `test-utils` - Enable test utilities
- `cli` - Enable CLI support with clap derives

## Architecture

```
┌─────────────────┐     gRPC      ┌─────────────┐     SSH      ┌──────────┐
│    Validator    │──────────────▶│    Miner    │─────────────▶│ GPU Node │
│                 │◀──────────────│             │              │          │
└─────────────────┘               └─────────────┘              └──────────┘
        │
        ▼
┌─────────────────┐
│   Bittensor     │
│   (Weights)     │
└─────────────────┘
```

## Verification Flow

1. Validator queries Bittensor metagraph for miners
2. Validator authenticates with miner via gRPC + Sr25519 signature
3. Miner deploys validator's ephemeral SSH key to GPU nodes
4. Validator SSHs directly to nodes and uploads verification binary
5. Validator executes verification and downloads results
6. Validator stores scores and sets network weights

## Example

```rust
use basilica_validator::{ValidatorConfig, Validator};

#[tokio::main]
async fn main() -> Result<(), Box<dyn std::error::Error>> {
    // Load configuration
    let config = ValidatorConfig::load("validator.toml")?;
    
    // Create and start the validator
    let validator = Validator::new(config).await?;
    validator.run().await?;
    
    Ok(())
}
```

## Configuration

```toml
[validator]
hotkey = "your-hotkey-ss58"
coldkey = "your-coldkey-ss58"
netuid = 39

[database]
url = "postgres://localhost/basilica"

[api]
port = 8080
```

## Network Information

- **Mainnet**: Bittensor Finney, Subnet 39
- **Testnet**: Bittensor Test Network, Subnet 387
- **Chain Endpoint**: `wss://entrypoint-finney.opentensor.ai:443`

## Related Crates

- [`basilica-common`](https://crates.io/crates/basilica-common) - Core shared types
- [`basilica-protocol`](https://crates.io/crates/basilica-protocol) - gRPC definitions
- [`basilica-miner`](https://crates.io/crates/basilica-miner) - Miner implementation
- [`basilica-sdk`](https://crates.io/crates/basilica-sdk) - Client SDK

## License

MIT License - see [LICENSE](LICENSE) for details.

