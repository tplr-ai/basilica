# basilica-billing

Billing service for Basilica compute subnet with usage tracking and invoicing.

[![Crates.io](https://img.shields.io/crates/v/basilica-billing.svg)](https://crates.io/crates/basilica-billing)
[![Documentation](https://docs.rs/basilica-billing/badge.svg)](https://docs.rs/basilica-billing)
[![License: MIT](https://img.shields.io/badge/License-MIT-yellow.svg)](https://opensource.org/licenses/MIT)

[Documentation](https://docs.rs/basilica-billing) | [Repository](https://github.com/one-covenant/basilica) | [Website](https://basilica.ai)

## Overview

`basilica-billing` provides a comprehensive billing service for the Basilica GPU marketplace. It tracks compute usage, calculates costs, manages account balances, and generates invoices.

## Installation

Add to your `Cargo.toml`:

```toml
[dependencies]
basilica-billing = "0.1"
```

## Features

- **Usage Tracking**: Real-time tracking of GPU compute usage
- **Cost Calculation**: Flexible pricing with per-second granularity
- **Account Management**: Balance tracking, credits, and top-ups
- **Invoice Generation**: Automated invoice generation and history
- **gRPC API**: High-performance gRPC interface
- **PostgreSQL**: Durable storage with full audit trail

## Feature Flags

- `with-billing-db-tests` - Enable database-backed integration tests

## Architecture

```
┌─────────────────┐     gRPC      ┌───────────────────┐
│   API / SDK     │──────────────▶│  Billing Service  │
└─────────────────┘               │                   │
                                  │  ┌─────────────┐  │
┌─────────────────┐               │  │Usage Tracker│  │
│   Validators    │──────────────▶│  └─────────────┘  │
│ (Usage Reports) │               │         │         │
└─────────────────┘               │  ┌──────▼──────┐  │
                                  │  │Cost Engine  │  │
                                  │  └──────┬──────┘  │
                                  │         │         │
                                  │  ┌──────▼──────┐  │
                                  │  │ PostgreSQL  │  │
                                  │  └─────────────┘  │
                                  └───────────────────┘
```

## Example

```rust
use basilica_billing::{BillingService, UsageRecord, AccountId};
use rust_decimal::Decimal;

#[tokio::main]
async fn main() -> Result<(), Box<dyn std::error::Error>> {
    // Connect to billing service
    let billing = BillingService::connect("postgres://localhost/billing").await?;
    
    // Record usage
    let usage = UsageRecord {
        account_id: AccountId::new("acc_123"),
        gpu_type: "h100".to_string(),
        duration_seconds: 3600,
        // ...
    };
    billing.record_usage(usage).await?;
    
    // Get account balance
    let balance = billing.get_balance(AccountId::new("acc_123")).await?;
    println!("Balance: ${}", balance);
    
    Ok(())
}
```

## Configuration

```toml
[billing]
database_url = "postgres://billing:password@localhost/basilica_billing"

[pricing]
default_currency = "USD"
pricing_model = "per_second"

[aws]
secrets_manager_enabled = true
region = "us-east-1"
```

## Database Schema

The billing service uses PostgreSQL with migrations for:

- `accounts` - User accounts and balances
- `usage_records` - Detailed usage tracking
- `invoices` - Generated invoices
- `payments` - Payment history
- `credits` - Applied credits and adjustments

## gRPC Services

```protobuf
service Billing {
    rpc RecordUsage(UsageRecord) returns (RecordResponse);
    rpc GetBalance(AccountId) returns (Balance);
    rpc GetInvoices(InvoiceQuery) returns (stream Invoice);
    rpc ApplyCredits(CreditRequest) returns (CreditResponse);
}
```

## Related Crates

- [`basilica-payments`](https://crates.io/crates/basilica-payments) - Payment processing
- [`basilica-aggregator`](https://crates.io/crates/basilica-aggregator) - Price aggregation
- [`basilica-common`](https://crates.io/crates/basilica-common) - Core types

## License

MIT License - see [LICENSE](LICENSE) for details.

