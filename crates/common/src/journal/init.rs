//! Journal initialization

use tracing::info;

/// Initialize the journal system with tracing + journald
pub fn init_journal() -> Result<(), Box<dyn std::error::Error>> {
    use tracing_subscriber::{fmt, layer::SubscriberExt, util::SubscriberInitExt, EnvFilter};

    let env_filter = EnvFilter::try_from_default_env().unwrap_or_else(|_| EnvFilter::new("info"));

    // Try to use journald, fall back to console if not available
    let journal_layer = tracing_journald::layer().ok();

    match journal_layer {
        Some(journal) => {
            // Production: use journald
            tracing_subscriber::registry()
                .with(env_filter)
                .with(fmt::layer().with_target(false))
                .with(journal)
                .init();
            info!("Journal initialized with systemd journald");
        }
        None => {
            // Development: use console
            tracing_subscriber::registry()
                .with(env_filter)
                .with(fmt::layer().with_target(false))
                .init();
            info!("Journal initialized with console logging (journald not available)");
        }
    }

    Ok(())
}
