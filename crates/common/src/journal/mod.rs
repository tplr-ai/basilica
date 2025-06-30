//! Simple journal system using tracing + journald
//!
//! This module provides a lightweight structured logging system that integrates
//! with systemd's journal for production deployments and falls back to console
//! logging for development.

pub mod init;
pub mod logging;
pub mod query;
pub mod types;

// Re-export public API
pub use init::init_journal;
pub use logging::*;
pub use query::query_logs;
pub use types::SecuritySeverity;
