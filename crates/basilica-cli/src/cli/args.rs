use crate::auth::should_use_device_flow;
use crate::cli::{commands::Commands, handlers};
use crate::config::CliConfig;
use crate::error::CliError;
use clap::builder::styling::AnsiColor;
use clap::builder::Styles;
use clap::{Parser, ValueHint};
use clap_verbosity_flag::Verbosity;
use console::Term;
use etcetera::{choose_base_strategy, BaseStrategy};
use std::path::{Path, PathBuf};

// Styles are disabled by default in clap v4, this are styles used in clap v3
const USAGE_STYLES: Styles = Styles::styled()
    .header(AnsiColor::Yellow.on_default())
    .usage(AnsiColor::Green.on_default())
    .literal(AnsiColor::Green.on_default())
    .placeholder(AnsiColor::Green.on_default());

/// Basilica CLI - Unified GPU rental and network management
#[derive(Parser, Debug)]
#[clap(styles = USAGE_STYLES)]
#[command(
    name = "basilica",
    author = "Basilica Team",
    version,
    about = "Basilica CLI - Unified GPU rental and network management",
    long_about = "Unified command-line interface for Basilica GPU compute marketplace.

QUICK START:
  basilica login                    # Login and authentication  
  basilica up <spec>                # Start GPU rental with specification
  basilica exec <uid> \"python train.py\"  # Run your code
  basilica down <uid>               # Terminate specific rental

GPU RENTAL:
  basilica ls                       # List available GPUs with pricing
  basilica ps                       # List active rentals
  basilica status <uid>             # Check rental status
  basilica logs <uid>               # Stream logs
  basilica ssh <uid>                # SSH into instance
  basilica cp <src> <dst>           # Copy files

NETWORK COMPONENTS:
  basilica validator                # Run validator
  basilica miner                    # Run miner  
  basilica executor                 # Run executor

AUTHENTICATION:
  basilica login                    # Log in to Basilica
  basilica login --device-code      # Log in using device flow
  basilica logout                   # Log out of Basilica"
)]
pub struct Args {
    /// Configuration file path
    #[arg(short, long, global = true, value_hint = ValueHint::FilePath)]
    pub config: Option<PathBuf>,

    #[command(flatten)]
    pub verbosity: Verbosity,

    /// Output format as JSON
    #[arg(long, global = true)]
    pub json: bool,

    /// Subcommand to execute
    #[command(subcommand)]
    pub command: Commands,
}

impl Args {
    /// Execute the CLI command
    pub async fn run(self) -> Result<(), CliError> {
        // Load config using the common loader pattern
        let config = if let Some(path) = &self.config {
            let expanded_path = expand_tilde(path);
            CliConfig::load_from_file(&expanded_path)?
        } else {
            CliConfig::load()?
        };

        // Check if command requires authentication and handle auto-login if needed
        if self.command.requires_auth() {
            self.execute_with_auth_retry(&config).await
        } else {
            self.execute_command(&config).await
        }
    }

    /// Execute command with automatic login retry on authentication failure
    async fn execute_with_auth_retry(&self, config: &CliConfig) -> Result<(), CliError> {
        // First attempt to execute the command
        match self.execute_command(config).await {
            Err(err) => {
                // Check if this is specifically a login_required error
                if matches!(&err, CliError::Auth(_)) {
                    // Inform user we need to authenticate
                    println!("You need to authenticate to continue.");
                    println!();

                    // Determine whether to use device flow based on environment
                    let use_device_flow = should_use_device_flow();

                    // Attempt login without showing command suggestions
                    handlers::auth::handle_login_with_options(use_device_flow, config, false)
                        .await?;

                    // Clear the login output lines (approximately 8 lines without suggestions)
                    // Lines: "You need to auth" + empty + banner + empty + success + empty + SSH key messages
                    let term = Term::stdout();
                    let _ = term.clear_last_lines(6);

                    // After successful login, retry the original command
                    self.execute_command(config).await
                } else {
                    // Not a login_required error, propagate it
                    Err(err)
                }
            }
            Ok(result) => Ok(result),
        }
    }

    /// Execute the actual command
    async fn execute_command(&self, config: &CliConfig) -> Result<(), CliError> {
        match &self.command {
            Commands::Login { device_code } => {
                handlers::auth::handle_login(*device_code, config).await?;
            }
            Commands::Logout => handlers::auth::handle_logout(config).await?,
            #[cfg(debug_assertions)]
            Commands::TestAuth { api } => {
                if *api {
                    handlers::test_auth::handle_test_api_auth(config).await?;
                } else {
                    handlers::test_auth::handle_test_auth(config).await?;
                }
            }

            // GPU rental operations
            Commands::Ls { gpu_type, filters } => {
                handlers::gpu_rental::handle_ls(
                    gpu_type.clone(),
                    filters.clone(),
                    self.json,
                    config,
                )
                .await?;
            }
            Commands::Up { target, options } => {
                handlers::gpu_rental::handle_up(target.clone(), options.clone(), config).await?;
            }
            Commands::Ps { filters } => {
                handlers::gpu_rental::handle_ps(filters.clone(), self.json, config).await?;
            }
            Commands::Status { target } => {
                handlers::gpu_rental::handle_status(target.clone(), self.json, config).await?;
            }
            Commands::Logs { target, options } => {
                handlers::gpu_rental::handle_logs(target.clone(), options.clone(), config).await?;
            }
            Commands::Down { target, all } => {
                handlers::gpu_rental::handle_down(target.clone(), *all, config).await?;
            }
            Commands::Exec { command, target } => {
                handlers::gpu_rental::handle_exec(target.clone(), command.clone(), config).await?;
            }
            Commands::Ssh { target, options } => {
                handlers::gpu_rental::handle_ssh(target.clone(), options.clone(), config).await?;
            }
            Commands::Cp {
                source,
                destination,
            } => {
                handlers::gpu_rental::handle_cp(source.clone(), destination.clone(), config).await?
            }

            // Network component delegation
            Commands::Validator { args } => handlers::external::handle_validator(args.clone())?,
            Commands::Miner { args } => handlers::external::handle_miner(args.clone())?,
            Commands::Executor { args } => handlers::external::handle_executor(args.clone())?,

            // Token management
            Commands::Tokens { action } => {
                use crate::cli::commands::TokenAction;
                use crate::client::create_client;

                // Create client with file-based auth (JWT required for token management)
                let client = create_client(config).await?;

                match action {
                    TokenAction::Create { name } => {
                        handlers::tokens::handle_create_token(&client, name.clone()).await?;
                    }
                    TokenAction::List => {
                        handlers::tokens::handle_list_tokens(&client).await?;
                    }
                    TokenAction::Revoke { name, yes } => {
                        handlers::tokens::handle_revoke_token(&client, name.clone(), *yes).await?;
                    }
                }
            }
        }
        Ok(())
    }
}

/// Expand tilde (~) in file paths to home directory
fn expand_tilde(path: &Path) -> PathBuf {
    if let Some(path_str) = path.to_str() {
        if let Some(stripped) = path_str.strip_prefix("~/") {
            if let Ok(strategy) = choose_base_strategy() {
                return strategy.home_dir().join(stripped);
            }
        }
    }
    path.to_path_buf()
}
