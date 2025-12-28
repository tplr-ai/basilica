//! # basilica-billing
//!
//! Billing service for Basilica compute subnet with usage tracking and invoicing.
//!
//! This crate provides comprehensive billing functionality for the Basilica network,
//! including usage tracking, cost calculation, account management, and invoicing.
//!
//! ## Overview
//!
//! `basilica-billing` enables:
//!
//! - **Usage Tracking**: Real-time tracking of GPU compute usage
//! - **Cost Calculation**: Flexible pricing with per-second granularity
//! - **Account Management**: Balance tracking, credits, and top-ups
//! - **Invoice Generation**: Automated invoice generation and history
//! - **gRPC API**: High-performance gRPC interface for service integration
//!
//! ## Quick Start
//!
//! ```rust,ignore
//! use basilica_billing::{BillingConfig, BillingClient};
//!
//! #[tokio::main]
//! async fn main() -> Result<(), Box<dyn std::error::Error>> {
//!     let config = BillingConfig::load("billing.toml")?;
//!     let client = BillingClient::connect(&config.database_url).await?;
//!     
//!     // Get account balance
//!     let balance = client.get_balance("user_123").await?;
//!     println!("Balance: ${}", balance);
//!     
//!     Ok(())
//! }
//! ```
//!
//! ## Feature Flags
//!
//! - `with-billing-db-tests` - Enable database-backed integration tests
//!
//! ## Related Crates
//!
//! - [`basilica-payments`](https://crates.io/crates/basilica-payments) - Payment processing
//! - [`basilica-aggregator`](https://crates.io/crates/basilica-aggregator) - Price aggregation
//! - [`basilica-common`](https://crates.io/crates/basilica-common) - Core types

pub mod client;
pub mod config;
pub mod domain;
pub mod error;
pub mod grpc;
pub mod metrics;
pub mod server;
pub mod storage;
pub mod telemetry;

pub use client::BillingClient;
pub use config::BillingConfig;
pub use error::{BillingError, Result};
