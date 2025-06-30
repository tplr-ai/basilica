# Protocol Crate

This crate provides the gRPC protocol definitions and generated Rust code for all inter-service communication in the Basilca network.

## Overview

The protocol crate defines three main services that support the validator-miner-executor interaction flow:

1. **MinerDiscovery** - Validator ↔ Miner coordination (steps 3-4 of interaction flow)
2. **ExecutorControl** - Direct Validator ↔ Executor communication (step 5)
3. **ValidatorExternalApi** - External services → Validator API for capacity rental

## Building

The protobuf files are automatically compiled during the build process:

```bash
cargo build -p protocol
```

Generated code is placed in `src/gen/` and included in the crate.

## Usage

### Client Example

```rust
use protocol::executor_control::executor_control_client::ExecutorControlClient;
use protocol::helpers;

let mut client = ExecutorControlClient::connect("http://[::1]:50051").await?;

// Add authentication
let request = helpers::add_auth_metadata(
    Request::new(request_data),
    "hotkey",
    "signature"
);

let response = client.health_check(request).await?;
```

### Server Example

```rust
use protocol::executor_control::executor_control_server::{ExecutorControl, ExecutorControlServer};

#[tonic::async_trait]
impl ExecutorControl for MyService {
    // Implement service methods
}

Server::builder()
    .add_service(ExecutorControlServer::new(MyService))
    .serve(addr)
    .await?;
```

## Helper Functions

The crate provides utility functions in the `utils` and `helpers` modules:

- `validate_gpu_spec()` - Validate GPU specifications
- `validate_container_spec()` - Validate container configurations
- `current_timestamp()` - Create protocol timestamps
- `extract_metadata()` - Extract gRPC metadata
- `add_auth_metadata()` - Add authentication to requests
- `verify_protocol_version()` - Check protocol compatibility

## Security

All gRPC communication should use mTLS in production. The `helpers::create_tls_config()` function provides a starting point for TLS configuration.

## Testing

Run tests with:

```bash
cargo test -p protocol
```

See `tests/integration_test.rs` for comprehensive examples of using the protocol types and services.
