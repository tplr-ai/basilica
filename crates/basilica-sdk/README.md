# basilica-sdk

Official Rust SDK for interacting with the Basilica GPU rental network.

[![Crates.io](https://img.shields.io/crates/v/basilica-sdk.svg)](https://crates.io/crates/basilica-sdk)
[![Documentation](https://docs.rs/basilica-sdk/badge.svg)](https://docs.rs/basilica-sdk)
[![License: MIT](https://img.shields.io/badge/License-MIT-yellow.svg)](https://opensource.org/licenses/MIT)

[Documentation](https://docs.rs/basilica-sdk) | [Repository](https://github.com/one-covenant/basilica) | [Website](https://basilica.ai)

## Overview

`basilica-sdk` provides a type-safe, async Rust client for the Basilica API. It enables programmatic access to GPU rentals, workload management, and billing on the Basilica network.

## Installation

Add to your `Cargo.toml`:

```toml
[dependencies]
basilica-sdk = "0.10"
tokio = { version = "1", features = ["full"] }
```

## Quick Start

```rust
use basilica_sdk::{BasilicaClient, ClientBuilder};

#[tokio::main]
async fn main() -> Result<(), Box<dyn std::error::Error>> {
    // Create a client with API key authentication
    let client = ClientBuilder::new()
        .api_key("your-api-key")
        .build()?;
    
    // List available GPUs
    let gpus = client.list_gpus().await?;
    for gpu in gpus {
        println!("{}: {} available", gpu.name, gpu.available_count);
    }
    
    // Create a GPU rental
    let rental = client.create_rental()
        .gpu_type("h100")
        .image("nvidia/cuda:12.0-base")
        .command(vec!["nvidia-smi"])
        .submit()
        .await?;
    
    println!("Rental created: {}", rental.id);
    
    // Wait for rental to be ready
    let rental = client.wait_for_rental(&rental.id).await?;
    
    // Stream logs
    let mut logs = client.stream_logs(&rental.id).await?;
    while let Some(line) = logs.next().await {
        println!("{}", line?);
    }
    
    Ok(())
}
```

## Features

- **Async/Await**: Built on Tokio for efficient async operations
- **Type Safety**: Strongly typed request/response models
- **Error Handling**: Comprehensive error types with retry hints
- **Authentication**: API key and OAuth2 authentication support
- **Streaming**: Real-time log streaming and event subscriptions
- **Configurable**: Timeouts, retries, connection pooling

## Feature Flags

- `client` - Enable full client functionality (default)

## API Overview

### Client

```rust
use basilica_sdk::{BasilicaClient, ClientBuilder};

// Build with API key
let client = ClientBuilder::new()
    .api_key("sk_...")
    .base_url("https://api.basilica.ai")
    .timeout(Duration::from_secs(30))
    .build()?;

// Or from environment
let client = BasilicaClient::from_env()?;
```

### GPU Operations

```rust
// List available GPU types
let gpus = client.list_gpus().await?;

// Get specific GPU info
let gpu = client.get_gpu("h100").await?;
```

### Rental Operations

```rust
// Create a rental
let rental = client.create_rental()
    .gpu_type("a100")
    .gpu_count(4)
    .image("pytorch/pytorch:latest")
    .env("MODEL", "llama-7b")
    .mount("data", "/data")
    .submit()
    .await?;

// Get rental status
let status = client.get_rental(&rental.id).await?;

// List user's rentals
let rentals = client.list_rentals().await?;

// Stop a rental
client.stop_rental(&rental.id).await?;
```

### Streaming

```rust
// Stream logs
let mut logs = client.stream_logs(&rental.id).await?;
while let Some(line) = logs.next().await {
    println!("{}", line?);
}

// Execute command
let output = client.exec(&rental.id, "nvidia-smi").await?;
```

### Billing

```rust
// Get account balance
let balance = client.get_balance().await?;

// Get usage history
let usage = client.get_usage()
    .start_date(start)
    .end_date(end)
    .fetch()
    .await?;
```

## Error Handling

```rust
use basilica_sdk::{ApiError, ErrorResponse};

match client.create_rental().gpu_type("h100").submit().await {
    Ok(rental) => println!("Created: {}", rental.id),
    Err(ApiError::RateLimited { retry_after }) => {
        println!("Rate limited, retry after {} seconds", retry_after);
    }
    Err(ApiError::InsufficientBalance) => {
        println!("Please add funds to your account");
    }
    Err(e) => return Err(e.into()),
}
```

## Environment Variables

| Variable | Description |
|----------|-------------|
| `BASILICA_API_KEY` | API key for authentication |
| `BASILICA_API_URL` | Custom API endpoint (default: `https://api.basilica.ai`) |

## Related Crates

- [`basilica-cli`](https://crates.io/crates/basilica-cli) - Command-line interface
- [`basilica-common`](https://crates.io/crates/basilica-common) - Core types
- [`basilica-api`](https://crates.io/crates/basilica-api) - API server

## Python SDK

For Python users, see the [`basilica` package on PyPI](https://pypi.org/project/basilica/).

## License

MIT License - see [LICENSE](LICENSE) for details.
