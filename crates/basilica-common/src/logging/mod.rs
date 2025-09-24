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
/// ```ignore
/// // Example usage with clap and clap-verbosity-flag in a binary:
/// // use clap::Parser;
/// // use clap_verbosity_flag::{Verbosity, InfoLevel};
/// // use basilica_common::logging;
/// //
/// // #[derive(Parser)]
/// // struct Args {
/// //     #[clap(flatten)]
/// //     verbosity: Verbosity<InfoLevel>,
/// // }
/// //
/// // let args = Args::parse();
/// // logging::init_logging(&args.verbosity, "basilica_miner", "basilica_miner=info").unwrap();
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

    tracing_subscriber::registry()
        .with(filter)
        .with(
            tracing_subscriber::fmt::layer()
                .with_target(true) // Show module path
                // .with_file(true) // Show source file
                // .with_line_number(true) // Show line number
                .compact(), // Use compact format
        )
        .init();

    Ok(())
}
