//! # basilica-payments
//!
//! Payment processing service for Basilica GPU marketplace with TAO integration.
//!
//! This crate handles cryptocurrency payment processing for the Basilica network.
//! It integrates with the Bittensor blockchain for TAO deposits, manages user wallets,
//! and coordinates with the billing service.
//!
//! ## Overview
//!
//! `basilica-payments` provides:
//!
//! - **TAO Integration**: Native Bittensor TAO token support
//! - **Wallet Management**: Secure user wallet generation and management
//! - **Deposit Tracking**: Monitor and credit blockchain deposits
//! - **Price Oracle**: TAO/USD price conversion
//! - **gRPC API**: High-performance payment processing API
//!
//! ## Quick Start
//!
//! ```rust,ignore
//! use basilica_payments::{config::PaymentsConfig, client::PaymentsClient};
//!
//! #[tokio::main]
//! async fn main() -> Result<(), Box<dyn std::error::Error>> {
//!     let config = PaymentsConfig::load("payments.toml")?;
//!     let client = PaymentsClient::connect(&config).await?;
//!     
//!     // Create a deposit wallet for a user
//!     let wallet = client.create_wallet("user_123").await?;
//!     println!("Deposit to: {}", wallet.address);
//!     
//!     Ok(())
//! }
//! ```
//!
//! ## Related Crates
//!
//! - [`basilica-billing`](https://crates.io/crates/basilica-billing) - Billing and usage tracking
//! - [`basilica-common`](https://crates.io/crates/basilica-common) - Core types
//! - [`basilica-protocol`](https://crates.io/crates/basilica-protocol) - gRPC definitions

pub mod blockchain;
pub mod client;
pub mod config;
pub mod domain;
pub mod error;
pub mod grpc;
pub mod metrics;
pub mod price_oracle;
pub mod processor;
pub mod reconciliation;
pub mod server;
pub mod storage;
