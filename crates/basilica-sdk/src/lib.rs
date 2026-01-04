//! # Basilica SDK
//!
//! Official SDK for interacting with the Basilica GPU rental network.
//!
//! This crate provides a type-safe client for the Basilica API, supporting
//! both authenticated and unauthenticated requests.
//!
//! ## Sandbox Support
//!
//! The SDK includes support for Daytona-compatible sandboxes for code execution:
//!
//! ```rust,no_run
//! use basilica_sdk::{BasilicaClient, ClientBuilder};
//! use basilica_sdk::sandbox::{Sandbox, SandboxConfig};
//!
//! # async fn example() -> basilica_sdk::Result<()> {
//! let client = ClientBuilder::default()
//!     .base_url("https://api.basilica.ai")
//!     .with_tokens("access_token", "refresh_token")
//!     .build()?;
//!
//! let sandbox = Sandbox::create(&client, SandboxConfig::new("python")).await?;
//! let result = sandbox.run("print('Hello!')").await?;
//! # Ok(())
//! # }
//! ```

pub mod auth;
pub mod client;
pub mod error;
pub mod jobs;
pub mod sandbox;
pub mod types;

// Re-export main types
pub use client::{BasilicaClient, ClientBuilder};
pub use error::{ApiError, ErrorResponse, Result};
pub use jobs::*;
pub use types::*;

// Re-export sandbox types under sandbox:: namespace
pub mod sandbox_types {
    pub use crate::sandbox::{
        EnvVar, ExecResult, FileInfo, GpuSpec, NetworkIsolation, ResourceSpec, Sandbox,
        SandboxConfig, SandboxState, SandboxStatus, SnapshotInfo,
    };
}

/// SDK version
pub const VERSION: &str = env!("CARGO_PKG_VERSION");
