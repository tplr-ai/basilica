//! # basilica-cli
//!
//! Unified command-line interface for Basilica GPU rental and network management.
//!
//! This crate provides a comprehensive CLI for interacting with the Basilica network
//! from your terminal. Deploy workloads, manage rentals, and monitor resources with ease.
//!
//! ## Overview
//!
//! The CLI provides commands for:
//!
//! - **GPU Rentals**: `deploy`, `rentals`, `logs`, `exec`, `ssh`
//! - **Resource Discovery**: `gpus`, filtering by type and availability
//! - **Account Management**: `login`, `logout`, `billing`, `config`
//! - **Utilities**: `completions`, `update`
//!
//! ## Quick Start
//!
//! ```bash
//! # Login to Basilica
//! basilica login
//!
//! # List available GPU types
//! basilica gpus list
//!
//! # Deploy a workload
//! basilica deploy --gpu h100 --image nvidia/cuda:12.0-base
//!
//! # Check rental status
//! basilica rentals list
//!
//! # Stream logs
//! basilica logs <rental-id> -f
//! ```
//!
//! ## Installation
//!
//! ```bash
//! curl -sSL https://basilica.ai/install.sh | bash
//! ```
//!
//! Or via Cargo:
//!
//! ```bash
//! cargo install basilica-cli
//! ```
//!
//! ## Commands
//!
//! | Command | Description |
//! |---------|-------------|
//! | `login` | Authenticate with Basilica |
//! | `deploy` | Deploy a workload to GPU |
//! | `rentals` | Manage active rentals |
//! | `gpus` | List available GPUs |
//! | `logs` | View workload logs |
//! | `exec` | Execute commands in rental |
//! | `ssh` | SSH into a rental |
//! | `billing` | View usage and billing |
//! | `update` | Self-update to latest |
//!
//! ## Architecture
//!
//! The CLI follows the same patterns as other Basilica components:
//! - Clap-based argument parsing with derive macros
//! - Handler-based command processing
//! - Shared configuration and error handling
//! - Integration with basilica-sdk for API access
//!
//! ## Related Crates
//!
//! - [`basilica-sdk`](https://crates.io/crates/basilica-sdk) - Rust SDK for programmatic access
//! - [`basilica-common`](https://crates.io/crates/basilica-common) - Core types

pub mod auth;
pub mod cli;
pub mod client;
pub mod config;
pub mod error;
pub mod github_releases;
pub mod interactive;
pub mod output;
pub mod progress;
pub mod source;
pub mod ssh;
pub mod types;
pub mod update_check;

pub use cli::*;
pub use error::*;
