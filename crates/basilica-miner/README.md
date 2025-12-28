# basilica-miner

Basilica Miner - Bittensor neuron that manages GPU node fleets.

[![Crates.io](https://img.shields.io/crates/v/basilica-miner.svg)](https://crates.io/crates/basilica-miner)
[![Documentation](https://docs.rs/basilica-miner/badge.svg)](https://docs.rs/basilica-miner)
[![License: MIT](https://img.shields.io/badge/License-MIT-yellow.svg)](https://opensource.org/licenses/MIT)

[Documentation](https://docs.rs/basilica-miner) | [Repository](https://github.com/one-covenant/basilica) | [Website](https://basilica.ai)

## Overview

`basilica-miner` is a Bittensor subnet miner that manages fleets of GPU nodes on the Basilica network. It handles validator authentication, SSH key deployment to nodes, and routes verification requests.

## Installation

Add to your `Cargo.toml`:

```toml
[dependencies]
basilica-miner = "0.1"
```

Or install the binary:

```bash
curl -sSL https://basilica.ai/install.sh | bash
```

## Features

- **Fleet Management**: Efficient orchestration of distributed GPU resources
- **Axon Server**: Bittensor discovery and communication endpoint
- **gRPC Server**: Validator authentication and node discovery
- **SSH Key Deployment**: Secure key management for validator access
- **Assignment Routing**: Smart routing of validator requests to appropriate nodes

## Architecture

```
┌─────────────────┐     Axon      ┌───────────────────┐
│   Bittensor     │◀─────────────▶│      Miner        │
│   (Discovery)   │               │                   │
└─────────────────┘               │  ┌─────────────┐  │
                                  │  │ Node Manager│  │
┌─────────────────┐     gRPC      │  └─────────────┘  │
│    Validator    │──────────────▶│         │        │
└─────────────────┘               │         ▼        │
                                  │  ┌─────────────┐  │
                                  │  │ SSH Key Mgr │  │
                                  │  └─────────────┘  │
                                  └───────────────────┘
                                           │
                                    SSH    │
                                           ▼
                                  ┌─────────────────┐
                                  │   GPU Node(s)   │
                                  │  - Docker       │
                                  │  - NVIDIA GPU   │
                                  └─────────────────┘
```

## GPU Node Requirements

GPU nodes managed by the miner need:

- Standard SSH server
- Docker with NVIDIA Container Toolkit
- CUDA drivers ≥12.8
- NVIDIA GPUs (A100, H100, B200, etc.)

## Example

```rust
use basilica_miner::{MinerConfig, Miner};

#[tokio::main]
async fn main() -> Result<(), Box<dyn std::error::Error>> {
    // Load configuration
    let config = MinerConfig::load("miner.toml")?;
    
    // Create and start the miner
    let miner = Miner::new(config).await?;
    miner.run().await?;
    
    Ok(())
}
```

## Configuration

```toml
[miner]
hotkey = "your-hotkey-ss58"
coldkey = "your-coldkey-ss58"
netuid = 39

[axon]
port = 8091
ip = "0.0.0.0"

[nodes]
# GPU nodes managed by this miner
[[nodes.gpu]]
host = "192.168.1.100"
ssh_user = "ubuntu"
ssh_key_path = "~/.ssh/node_key"
```

## Security Model

- **Ephemeral SSH keys**: Validators generate ed25519 keys per session
- **Key tagging**: Keys are tagged with validator hotkey for identification
- **Auto-cleanup**: Miner removes expired keys after session timeout (~1 hour)
- **Sr25519 signatures**: All validator requests are cryptographically signed

## Related Crates

- [`basilica-common`](https://crates.io/crates/basilica-common) - Core shared types
- [`basilica-protocol`](https://crates.io/crates/basilica-protocol) - gRPC definitions
- [`basilica-validator`](https://crates.io/crates/basilica-validator) - Validator implementation

## License

MIT License - see [LICENSE](LICENSE) for details.

