//! Bootstrap Shell Commands
//!
//! This module contains the shell command strings used for remote TEE setup.
//! Commands are split into TDX and GPU categories for better organization.

pub mod gpu;
pub mod tdx;

// Re-export all commands for convenience
pub use gpu::*;
pub use tdx::*;
