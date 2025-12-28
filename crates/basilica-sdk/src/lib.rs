//! # basilica-sdk
//!
//! Official Rust SDK for interacting with the Basilica GPU rental network.
//!
//! This crate provides a type-safe, async client for the Basilica API, supporting
//! GPU rentals, workload management, and billing operations.
//!
//! ## Overview
//!
//! `basilica-sdk` enables programmatic access to:
//!
//! - **GPU Rentals**: Create, manage, and terminate GPU rentals
//! - **Workload Management**: Deploy containers, stream logs, execute commands
//! - **Billing**: Check balances, view usage, manage credits
//!
//! ## Quick Start
//!
//! ```rust,ignore
//! use basilica_sdk::{BasilicaClient, ClientBuilder};
//!
//! #[tokio::main]
//! async fn main() -> Result<(), Box<dyn std::error::Error>> {
//!     // Create a client with API key authentication
//!     let client = ClientBuilder::new()
//!         .api_key("your-api-key")
//!         .build()?;
//!     
//!     // List available GPUs
//!     let gpus = client.list_gpus().await?;
//!     for gpu in gpus {
//!         println!("{}: {} available", gpu.name, gpu.available_count);
//!     }
//!     
//!     // Create a GPU rental
//!     let rental = client.create_rental()
//!         .gpu_type("h100")
//!         .image("nvidia/cuda:12.0-base")
//!         .command(vec!["nvidia-smi"])
//!         .submit()
//!         .await?;
//!     
//!     println!("Rental created: {}", rental.id);
//!     Ok(())
//! }
//! ```
//!
//! ## Features
//!
//! - **Async/Await**: Built on Tokio for efficient async operations
//! - **Type Safety**: Strongly typed request/response models
//! - **Error Handling**: Comprehensive error types with retry hints
//! - **Authentication**: API key and OAuth2 support
//! - **Streaming**: Real-time log streaming and events
//!
//! ## Environment Variables
//!
//! | Variable | Description |
//! |----------|-------------|
//! | `BASILICA_API_KEY` | API key for authentication |
//! | `BASILICA_API_URL` | Custom API endpoint |
//!
//! ## Related Crates
//!
//! - [`basilica-cli`](https://crates.io/crates/basilica-cli) - Command-line interface
//! - [`basilica-common`](https://crates.io/crates/basilica-common) - Core types

pub mod auth;
pub mod client;
pub mod error;
pub mod jobs;
pub mod types;

// Re-export main types
pub use client::{BasilicaClient, ClientBuilder};
pub use error::{ApiError, ErrorResponse, Result};
pub use jobs::*;
pub use types::*;

/// SDK version
pub const VERSION: &str = env!("CARGO_PKG_VERSION");
