//! Main entry point for the Basilica CLI

use basilica_cli::cli::commands::Commands;
use basilica_cli::cli::Args;
use basilica_cli::update_check;
use clap::{CommandFactory, Parser};
use clap_complete::env::CompleteEnv;
use clap_verbosity_flag::{LevelFilter, OffLevel, Verbosity};
use color_eyre::eyre::{eyre, Result};

fn main() -> Result<()> {
    // Handle shell completions first (must be before argument parsing)
    CompleteEnv::with_factory(Args::command).complete();

    // Parse args
    let args = Args::parse();

    // Handle upgrade command WITHOUT Tokio runtime to avoid conflicts with self_update
    if let Commands::Upgrade { version, dry_run } = &args.command {
        use basilica_cli::cli::handlers;

        // Configure color-eyre first
        color_eyre::config::HookBuilder::default()
            .display_location_section(false)
            .display_env_section(false)
            .install()?;

        return handlers::upgrade::handle_upgrade(version.clone(), *dry_run).map_err(|e| match e {
            basilica_cli::CliError::Internal(report) => report,
            other => eyre!("{}", other),
        });
    }

    // For all other commands, run in Tokio runtime
    run_async(args)
}

#[tokio::main]
async fn run_async(args: Args) -> Result<()> {
    // Check cache for updates and show notification if a new version is available
    // This must happen BEFORE running the command to ensure notification appears first
    update_check::check_cache_and_show_notification();

    // Start background check for updates (non-blocking, will update cache for next run)
    update_check::check_and_notify_update();

    // Configure color-eyre with custom settings
    // Disable location display (file paths and line numbers)
    color_eyre::config::HookBuilder::default()
        .display_location_section(false)
        .display_env_section(false)
        .install()?;

    // std::env::set_var("RUST_SPANTRACE", "0");

    match args.verbosity.log_level_filter() {
        LevelFilter::Off | LevelFilter::Error => {}
        _ => {
            std::env::set_var("RUST_LIB_BACKTRACE", "1");
        }
    }

    // Initialize logging here in the binary context where CARGO_BIN_NAME is available
    let binary_name = env!("CARGO_BIN_NAME").replace("-", "_");
    let default_filter = format!("{}=error", binary_name);
    basilica_common::logging::init_logging(
        // disable verbosity by default for CLI, if needed it needs to be enabled with RUST_LOG
        &Verbosity::<OffLevel>::default(),
        &binary_name,
        &default_filter,
    )
    .map_err(|e| eyre!("Failed to initialize logging: {}", e))?;

    // Capture render-affecting flags before `args` is moved into `run()`.
    let json = args.json;

    // Run and handle errors explicitly to show suggestions
    if let Err(err) = args.run().await {
        let exit_code = err.exit_code();

        if json {
            let _ = basilica_cli::output::render_error(
                &err,
                basilica_cli::output::RenderMode::Json,
                &mut std::io::stderr(),
            );
            std::process::exit(exit_code);
        }

        // CommandExit carries a child process's status and renders its own bare
        // message, so it keeps the plain renderer and its own exit code.
        // Everything else goes through the report, which carries suggestions.
        if matches!(err, basilica_cli::CliError::CommandExit { .. }) {
            let _ = basilica_cli::output::render_error(
                &err,
                basilica_cli::output::RenderMode::Human,
                &mut std::io::stderr(),
            );
            std::process::exit(exit_code);
        }

        return Err(basilica_cli::output::into_report(err));
    }

    Ok(())
}
