//! Unified logging initialization for all Basilica binaries
//!
//! This module provides a standardized logging setup that respects the following priority order:
//! 1. CLI flags (`-v/-q`) - highest priority
//! 2. RUST_LOG environment variable
//! 3. Binary-specific defaults - lowest priority

use anyhow::Result;
use clap_verbosity_flag::{LogLevel, Verbosity};
use tracing_subscriber::{layer::SubscriberExt, util::SubscriberInitExt, EnvFilter};

/// Initialize logging with the specified verbosity level and default filter.
///
/// # Arguments
///
/// * `verbosity` - The verbosity flags from clap (-v/-q)
/// * `base_filter` - The base filter string to scope verbose flags to (e.g., "basilica_protocol=info,basilica_miner")
/// * `default_filter` - The default filter string if no CLI flags or RUST_LOG are set
///
/// # Example
///
/// ```no_run
/// use clap_verbosity_flag::{Verbosity, InfoLevel};
/// use basilica_common::logging;
///
/// // Example: initialize with default verbosity (as if no -v/-q provided)
/// let verbosity = Verbosity::<InfoLevel>::default();
/// // In doctests, CARGO_BIN_NAME may be unset; use a fallback.
/// let binary_name = std::env::var("CARGO_BIN_NAME").unwrap_or_else(|_| "basilica_binary".to_string());
/// logging::init_logging(&verbosity, &binary_name, "basilica_miner=info").unwrap();
/// ```
pub fn init_logging<L: LogLevel>(
    verbosity: &Verbosity<L>,
    base_filter: &str,
    default_filter: &str,
) -> Result<()> {
    // Check if verbosity flags were explicitly used
    let filter = if verbosity.is_present() {
        // CLI flags take priority - scope to specific binary
        EnvFilter::try_new(format!("{}={}", base_filter, verbosity.log_level_filter()))?
    } else {
        // Fall back to RUST_LOG, then default
        EnvFilter::try_from_default_env().unwrap_or_else(|_| EnvFilter::new(default_filter))
    };

    let fmt_layer = tracing_subscriber::fmt::layer()
        .with_target(true) // Show module path
        // .with_file(true) // Show source file
        // .with_line_number(true) // Show line number
        .compact(); // Use compact format

    let fmt_layer = match resolve_ansi_override() {
        Some(ansi) => fmt_layer.with_ansi(ansi),
        None => fmt_layer,
    };

    tracing_subscriber::registry()
        .with(filter)
        .with(fmt_layer)
        .init();

    Ok(())
}

fn resolve_ansi_override() -> Option<bool> {
    if std::env::var_os("NO_COLOR").is_some() {
        return Some(false);
    }

    None
}
