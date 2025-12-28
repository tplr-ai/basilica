# basilica-protocol

gRPC protocol definitions and message types for Basilica network communication.

[![Crates.io](https://img.shields.io/crates/v/basilica-protocol.svg)](https://crates.io/crates/basilica-protocol)
[![Documentation](https://docs.rs/basilica-protocol/badge.svg)](https://docs.rs/basilica-protocol)
[![License: MIT](https://img.shields.io/badge/License-MIT-yellow.svg)](https://opensource.org/licenses/MIT)

[Documentation](https://docs.rs/basilica-protocol) | [Repository](https://github.com/one-covenant/basilica) | [Website](https://basilica.ai)

## Overview

`basilica-protocol` contains the Protocol Buffer definitions and generated Rust code for communication between Basilica network components. It defines the gRPC services and message types used for validator-miner communication.

## Installation

Add to your `Cargo.toml`:

```toml
[dependencies]
basilica-protocol = "0.1"
```

## Features

- **MinerDiscovery Service**: Validator authentication and node discovery
- **Strongly Typed Messages**: All protocol messages are type-safe Rust structs
- **Tonic Integration**: Built on the tonic gRPC framework

## Feature Flags

- `client` - Enable client-side gRPC stubs
- `server` - Enable server-side gRPC stubs

## Protocol Services

### MinerDiscovery

```protobuf
service MinerDiscovery {
    // Validator authenticates with a miner
    rpc AuthenticateValidator(AuthRequest) returns (AuthResponse);
    
    // Stream available GPU nodes from the miner
    rpc DiscoverNodes(NodeDiscoveryRequest) returns (stream NodeInfo);
}
```

## Example

```rust
use basilica_protocol::miner_discovery_client::MinerDiscoveryClient;
use basilica_protocol::AuthRequest;

#[tokio::main]
async fn main() -> Result<(), Box<dyn std::error::Error>> {
    // Connect to a miner's gRPC endpoint
    let mut client = MinerDiscoveryClient::connect("http://miner:50051").await?;
    
    // Authenticate as a validator
    let request = AuthRequest {
        validator_hotkey: "5GrwvaEF...".to_string(),
        signature: vec![/* Sr25519 signature */],
        // ...
    };
    
    let response = client.authenticate_validator(request).await?;
    println!("Authenticated: {:?}", response);
    
    Ok(())
}
```

## Building from Proto

The protocol buffers are compiled at build time using `tonic-build`. The proto files are located in `proto/`:

```
proto/
├── common.proto        # Shared message types
├── miner.proto         # Miner service definitions
└── billing.proto       # Billing message types
```

## Related Crates

- [`basilica-common`](https://crates.io/crates/basilica-common) - Core shared types
- [`basilica-validator`](https://crates.io/crates/basilica-validator) - Validator implementation
- [`basilica-miner`](https://crates.io/crates/basilica-miner) - Miner implementation

## License

MIT License - see [LICENSE](LICENSE) for details.
