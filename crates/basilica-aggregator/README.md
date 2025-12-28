# basilica-aggregator

Price aggregation and billing utilities for Basilica GPU marketplace.

[![Crates.io](https://img.shields.io/crates/v/basilica-aggregator.svg)](https://crates.io/crates/basilica-aggregator)
[![Documentation](https://docs.rs/basilica-aggregator/badge.svg)](https://docs.rs/basilica-aggregator)
[![License: MIT](https://img.shields.io/badge/License-MIT-yellow.svg)](https://opensource.org/licenses/MIT)

[Documentation](https://docs.rs/basilica-aggregator) | [Repository](https://github.com/one-covenant/basilica) | [Website](https://basilica.ai)

## Overview

`basilica-aggregator` provides price aggregation, billing calculation, and inventory management utilities for the Basilica GPU marketplace. It fetches GPU pricing from various sources and calculates costs for compute usage.

## Installation

Add to your `Cargo.toml`:

```toml
[dependencies]
basilica-aggregator = "0.1"
```

## Features

- **Price Aggregation**: Collect and normalize GPU prices from multiple sources
- **Cost Calculation**: Calculate rental costs based on usage duration
- **Inventory Management**: Track GPU availability and allocation
- **VIP Machine Support**: Integration with VIP machine inventory from S3

## Example

```rust
use basilica_aggregator::{PriceAggregator, GpuType};
use rust_decimal::Decimal;

#[tokio::main]
async fn main() -> Result<(), Box<dyn std::error::Error>> {
    // Create an aggregator
    let aggregator = PriceAggregator::new().await?;
    
    // Get current price for H100
    let price = aggregator.get_price(GpuType::H100).await?;
    println!("H100 price: ${}/hour", price);
    
    // Calculate cost for 2 hours
    let duration_hours = Decimal::new(2, 0);
    let cost = aggregator.calculate_cost(GpuType::H100, duration_hours).await?;
    println!("Total cost: ${}", cost);
    
    Ok(())
}
```

## Pricing Sources

The aggregator can fetch pricing from:

- Basilica network validators
- External cloud provider APIs
- Static configuration files
- S3-hosted inventory files

## Configuration

```toml
[aggregator]
cache_ttl_seconds = 300
default_currency = "USD"

[sources.vip]
s3_bucket = "basilica-inventory"
s3_key = "vip-machines.csv"

[sources.cloud]
enabled = true
refresh_interval_seconds = 600
```

## Related Crates

- [`basilica-billing`](https://crates.io/crates/basilica-billing) - Billing service
- [`basilica-sdk`](https://crates.io/crates/basilica-sdk) - Client SDK
- [`basilica-common`](https://crates.io/crates/basilica-common) - Core types

## License

MIT License - see [LICENSE](LICENSE) for details.

