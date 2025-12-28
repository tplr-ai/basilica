# basilica-common

Core shared types, cryptographic utilities, and infrastructure for the Basilica GPU marketplace.

[![Crates.io](https://img.shields.io/crates/v/basilica-common.svg)](https://crates.io/crates/basilica-common)
[![Documentation](https://docs.rs/basilica-common/badge.svg)](https://docs.rs/basilica-common)
[![License: MIT](https://img.shields.io/badge/License-MIT-yellow.svg)](https://opensource.org/licenses/MIT)

[Documentation](https://docs.rs/basilica-common) | [Repository](https://github.com/one-covenant/basilica) | [Website](https://basilica.ai)

## Overview

`basilica-common` provides the foundational building blocks for all Basilica components. It is the base layer upon which the validator, miner, SDK, and other crates are built.

## Installation

Add to your `Cargo.toml`:

```toml
[dependencies]
basilica-common = "0.1"
```

## Features

- **Identity Types**: `Hotkey`, `NodeId`, `ValidatorUid`, `MinerUid` with SS58 validation
- **Cryptography**: Blake3 hashing, Ed25519/Sr25519 signature verification
- **Configuration**: Unified config loading with TOML files and environment overrides
- **Persistence**: Repository traits and database abstractions (SQLite/PostgreSQL)
- **Metrics**: Standardized metrics collection interfaces
- **SSH**: Trait abstractions for SSH key management

## Feature Flags

- `sqlite` - Enable SQLite persistence backend
- `postgres` - Enable PostgreSQL persistence backend
- `crypto-extra` - Additional cryptographic utilities

## Example

```rust
use basilica_common::{Hotkey, Config, CryptoProvider};

fn main() -> Result<(), Box<dyn std::error::Error>> {
    // Parse a Bittensor hotkey from SS58 format
    let hotkey = Hotkey::from_ss58("5GrwvaEF5zXb26Fz9rcQpDWS57CtERHpNehXCPcNoHGKutQY")?;
    
    // Load configuration from file with environment overrides
    let config = Config::builder()
        .file("config.toml")
        .env_prefix("BASILICA")
        .build()?;
    
    Ok(())
}
```

## Modules

| Module | Description |
|--------|-------------|
| `crypto` | Cryptographic primitives (Blake3, Ed25519, Sr25519, P256) |
| `identity` | Network identity types with validation |
| `config` | Configuration loading and management |
| `persistence` | Database traits and repository patterns |
| `ssh` | SSH key management abstractions |
| `metrics` | Metrics collection traits |
| `compute` | Compute resource definitions |
| `rental` | GPU rental types and states |

## Related Crates

- [`basilica-sdk`](https://crates.io/crates/basilica-sdk) - High-level client SDK
- [`basilica-protocol`](https://crates.io/crates/basilica-protocol) - gRPC protocol definitions
- [`basilica-validator`](https://crates.io/crates/basilica-validator) - Validator node implementation

## License

MIT License - see [LICENSE](LICENSE) for details.

