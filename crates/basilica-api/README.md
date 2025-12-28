# basilica-api

Smart HTTP gateway for Basilica validator network with load balancing and caching.

[![Crates.io](https://img.shields.io/crates/v/basilica-api.svg)](https://crates.io/crates/basilica-api)
[![Documentation](https://docs.rs/basilica-api/badge.svg)](https://docs.rs/basilica-api)
[![License: MIT](https://img.shields.io/badge/License-MIT-yellow.svg)](https://opensource.org/licenses/MIT)

[Documentation](https://docs.rs/basilica-api) | [Repository](https://github.com/one-covenant/basilica) | [Website](https://basilica.ai)

## Overview

`basilica-api` provides an HTTP gateway for the Basilica network, offering load-balanced access to validators with authentication, caching, and request aggregation.

## Installation

Add to your `Cargo.toml`:

```toml
[dependencies]
basilica-api = "0.1"
```

Or install the binary:

```bash
curl -sSL https://basilica.ai/install.sh | bash
```

## Features

- **Load Balancing**: Smart distribution of requests across validators
- **Request Aggregation**: Combine similar requests for efficiency
- **Authentication**: API key and JWT-based authentication
- **Rate Limiting**: Protect backends from overload
- **Caching**: Response caching for improved latency
- **WebSocket Support**: Real-time streaming capabilities

## Feature Flags

- `server` - Enable HTTP server functionality (default)
- `client` - Enable HTTP client functionality
- `utoipa` - Enable OpenAPI documentation generation
- `full` - Enable all features

## Architecture

```
┌─────────────────┐      HTTP       ┌───────────────────┐
│   API Clients   │────────────────▶│   Basilica API    │
│   (SDK, CLI)    │◀────────────────│                   │
└─────────────────┘                 │  ┌─────────────┐  │
                                    │  │Load Balancer│  │
                                    │  └──────┬──────┘  │
                                    │         │         │
                                    │  ┌──────▼──────┐  │
                                    │  │   Cache     │  │
                                    │  └──────┬──────┘  │
                                    └─────────┼─────────┘
                                              │
                              ┌───────────────┼───────────────┐
                              ▼               ▼               ▼
                       ┌──────────┐    ┌──────────┐    ┌──────────┐
                       │Validator │    │Validator │    │Validator │
                       │    1     │    │    2     │    │    N     │
                       └──────────┘    └──────────┘    └──────────┘
```

## Example

```rust
use basilica_api::{ApiConfig, ApiServer};

#[tokio::main]
async fn main() -> Result<(), Box<dyn std::error::Error>> {
    // Load configuration
    let config = ApiConfig::load("api.toml")?;
    
    // Create and start the API server
    let server = ApiServer::new(config).await?;
    server.run().await?;
    
    Ok(())
}
```

## Configuration

```toml
[api]
port = 8080
host = "0.0.0.0"

[auth]
enabled = true
jwt_secret = "your-secret"

[rate_limit]
requests_per_second = 100
burst_size = 200

[cache]
enabled = true
ttl_seconds = 60
```

## API Endpoints

| Endpoint | Method | Description |
|----------|--------|-------------|
| `/health` | GET | Health check |
| `/miners` | GET | List available miners |
| `/rentals` | POST | Create a GPU rental |
| `/rentals/{id}` | GET | Get rental status |
| `/ws` | WS | WebSocket for streaming |

## Related Crates

- [`basilica-sdk`](https://crates.io/crates/basilica-sdk) - Client SDK
- [`basilica-validator`](https://crates.io/crates/basilica-validator) - Validator implementation
- [`basilica-billing`](https://crates.io/crates/basilica-billing) - Billing service

## License

MIT License - see [LICENSE](LICENSE) for details.
