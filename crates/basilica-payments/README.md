# basilica-payments

Payment processing service for Basilica GPU marketplace with TAO integration.

[![Crates.io](https://img.shields.io/crates/v/basilica-payments.svg)](https://crates.io/crates/basilica-payments)
[![Documentation](https://docs.rs/basilica-payments/badge.svg)](https://docs.rs/basilica-payments)
[![License: MIT](https://img.shields.io/badge/License-MIT-yellow.svg)](https://opensource.org/licenses/MIT)

[Documentation](https://docs.rs/basilica-payments) | [Repository](https://github.com/one-covenant/basilica) | [Website](https://basilica.ai)

## Overview

`basilica-payments` handles cryptocurrency payment processing for the Basilica network. It integrates with the Bittensor blockchain for TAO deposits, manages user wallets, and coordinates with the billing service.

## Installation

Add to your `Cargo.toml`:

```toml
[dependencies]
basilica-payments = "0.1"
```

## Features

- **TAO Integration**: Native Bittensor TAO token support
- **Wallet Management**: Secure user wallet generation and management
- **Deposit Tracking**: Monitor and credit blockchain deposits
- **gRPC API**: High-performance payment processing API
- **PostgreSQL**: Durable transaction storage

## Architecture

```
┌─────────────────┐
│   Bittensor     │
│   Blockchain    │
└────────┬────────┘
         │ Watch Deposits
         ▼
┌─────────────────────────────┐
│     Payments Service        │
│                             │
│  ┌─────────────────────┐   │
│  │  Deposit Watcher    │   │
│  └──────────┬──────────┘   │
│             │               │
│  ┌──────────▼──────────┐   │
│  │  Wallet Manager     │   │
│  └──────────┬──────────┘   │
│             │               │
│  ┌──────────▼──────────┐   │
│  │    PostgreSQL       │   │
│  └─────────────────────┘   │
└─────────────────────────────┘
         │
         ▼ gRPC
┌─────────────────┐
│ Billing Service │
└─────────────────┘
```

## Example

```rust
use basilica_payments::{PaymentsService, WalletRequest};

#[tokio::main]
async fn main() -> Result<(), Box<dyn std::error::Error>> {
    // Connect to payments service
    let payments = PaymentsService::connect("postgres://localhost/payments").await?;
    
    // Create a deposit wallet for a user
    let wallet = payments.create_wallet(WalletRequest {
        user_id: "user_123".to_string(),
    }).await?;
    
    println!("Deposit to: {}", wallet.address);
    
    // Check for deposits
    let deposits = payments.list_deposits("user_123").await?;
    for deposit in deposits {
        println!("Received {} TAO", deposit.amount);
    }
    
    Ok(())
}
```

## Configuration

```toml
[payments]
database_url = "postgres://payments:password@localhost/basilica_payments"

[bittensor]
network = "finney"
chain_endpoint = "wss://entrypoint-finney.opentensor.ai:443"

[wallet]
derivation_path = "m/44'/501'/0'/0'"
```

## gRPC Services

```protobuf
service Payments {
    rpc CreateWallet(WalletRequest) returns (Wallet);
    rpc GetWallet(WalletId) returns (Wallet);
    rpc ListDeposits(DepositsQuery) returns (stream Deposit);
    rpc GetBalance(WalletId) returns (Balance);
}
```

## Command Line Tools

```bash
# Check wallet balance
paymentsctl wallet balance <wallet-id>

# List recent deposits
paymentsctl deposits list --user <user-id>

# Manual credit (admin)
paymentsctl credits apply --user <user-id> --amount 100
```

## Related Crates

- [`basilica-billing`](https://crates.io/crates/basilica-billing) - Billing and usage tracking
- [`basilica-common`](https://crates.io/crates/basilica-common) - Core types
- [`basilica-protocol`](https://crates.io/crates/basilica-protocol) - gRPC definitions

## License

MIT License - see [LICENSE](LICENSE) for details.

