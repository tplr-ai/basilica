//! # basilica-aggregator
//!
//! Price aggregation and billing utilities for Basilica GPU marketplace.
//!
//! This crate provides GPU price aggregation, cost calculation, and inventory
//! management for the Basilica network.
//!
//! ## Overview
//!
//! `basilica-aggregator` enables:
//!
//! - **Price Aggregation**: Collect and normalize GPU prices from multiple sources
//! - **Cost Calculation**: Calculate rental costs based on usage duration
//! - **Inventory Management**: Track GPU availability and allocation
//! - **VIP Machine Support**: Integration with VIP machine inventory from S3
//!
//! ## Quick Start
//!
//! ```rust,ignore
//! use basilica_aggregator::{AggregatorService, AggregatorConfig};
//!
//! #[tokio::main]
//! async fn main() -> Result<(), Box<dyn std::error::Error>> {
//!     let config = AggregatorConfig::load("aggregator.toml")?;
//!     let service = AggregatorService::new(config).await?;
//!     
//!     // Get current H100 price
//!     let offerings = service.get_offerings("h100").await?;
//!     for offering in offerings {
//!         println!("{}: ${}/hour", offering.provider, offering.price_per_hour);
//!     }
//!     
//!     Ok(())
//! }
//! ```
//!
//! ## Related Crates
//!
//! - [`basilica-billing`](https://crates.io/crates/basilica-billing) - Billing service
//! - [`basilica-common`](https://crates.io/crates/basilica-common) - Core types

pub mod background;
pub mod config;
pub mod db;
pub mod error;
pub mod models;
pub mod providers;
pub mod service;
pub mod vip;

// Re-export commonly used types for easy access
pub use config::Config as AggregatorConfig;
pub use db::Database;
pub use error::{AggregatorError, Result};
pub use models::{
    Deployment, DeploymentStatus, GpuOffering, Provider as ProviderEnum, ProviderHealth, SshKey,
};
pub use service::AggregatorService;
