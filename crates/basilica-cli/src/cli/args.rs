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

AUTHENTICATION:
  basilica login                    # Log in to Basilica
  basilica login --device-code      # Log in using device flow
  basilica logout                   # Log out of Basilica

AUTH TOKEN MANAGEMENT:
  basilica tokens create <name>     # Create API token
  basilica tokens list              # List API tokens
  basilica tokens revoke <name>     # Revoke API token

FUND MANAGEMENT:
  basilica fund                     # Show deposit address
  basilica fund list --limit 100    # List deposits
"
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
        // If BASILICA_API_TOKEN is set, skip the auth retry flow (dev mode / CI)
        let has_env_token = std::env::var("BASILICA_API_TOKEN")
            .map(|t| !t.is_empty())
            .unwrap_or(false);

        if self.command.requires_auth() && !has_env_token {
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
            Commands::Ls {
                gpu_type,
                filters,
                compute,
            } => {
                handlers::gpu_rental::handle_ls(
                    gpu_type.clone(),
                    filters.clone(),
                    *compute,
                    self.json,
                    config,
                )
                .await?;
            }
            Commands::Up {
                target,
                options,
                compute,
            } => {
                handlers::gpu_rental::handle_up(target.clone(), options.clone(), *compute, config)
                    .await?;
            }
            Commands::Ps { compute, filters } => {
                handlers::gpu_rental::handle_ps(filters.clone(), *compute, self.json, config)
                    .await?;
            }
            Commands::Status { target } => {
                handlers::gpu_rental::handle_status(target.clone(), self.json, config).await?;
            }
            Commands::Logs { target, options } => {
                handlers::gpu_rental::handle_logs(target.clone(), options.clone(), config).await?;
            }
            Commands::Down {
                target,
                compute,
                all,
            } => {
                handlers::gpu_rental::handle_down(target.clone(), *compute, *all, config).await?;
            }
            Commands::Restart { target } => {
                handlers::gpu_rental::handle_restart(target.clone(), config).await?;
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
                        handlers::tokens::handle_list_tokens(&client, self.json).await?;
                    }
                    TokenAction::Revoke { name, yes } => {
                        handlers::tokens::handle_revoke_token(&client, name.clone(), *yes).await?;
                    }
                }
            }

            // SSH key management
            Commands::SshKeys { action } => {
                use crate::cli::commands::SshKeyAction;
                use crate::client::create_client;

                // Create client with file-based auth (JWT required for SSH key management)
                let client = create_client(config).await?;

                match action {
                    SshKeyAction::Add { name, file } => {
                        handlers::ssh_keys::handle_add_ssh_key(&client, name.clone(), file.clone())
                            .await?;
                    }
                    SshKeyAction::List => {
                        handlers::ssh_keys::handle_list_ssh_keys(&client, self.json).await?;
                    }
                    SshKeyAction::Delete { yes } => {
                        handlers::ssh_keys::handle_delete_ssh_key(&client, *yes).await?;
                    }
                }
            }

            // Fund management
            Commands::Fund { action } => {
                use crate::cli::commands::FundAction;
                use crate::client::create_authenticated_client;

                // Create authenticated client
                let client = create_authenticated_client(config).await?;

                match action {
                    None => {
                        // Default action: show deposit address
                        handlers::fund::handle_show_deposit_address(&client, self.json).await?;
                    }
                    Some(FundAction::List { limit, offset }) => {
                        handlers::fund::handle_list_deposits(&client, *limit, *offset, self.json)
                            .await?;
                    }
                }
            }

            // Balance check
            Commands::Balance => {
                use crate::client::create_authenticated_client;

                // Create authenticated client
                let client = create_authenticated_client(config).await?;

                handlers::balance::handle_check_balance(&client, self.json).await?;
            }

            // Deploy command
            Commands::Deploy(cmd) => {
                handlers::deploy::handle_deploy(*cmd.clone(), config).await?;
            }

            // Volume management
            Commands::Volumes { action } => {
                use crate::cli::commands::VolumeAction;
                use crate::client::create_client;

                // Create client with file-based auth (JWT required for volume management)
                let client = create_client(config).await?;

                match action {
                    VolumeAction::Create {
                        name,
                        size,
                        provider,
                        region,
                        description,
                    } => {
                        handlers::volumes::handle_create_volume(
                            &client,
                            name.clone(),
                            *size,
                            provider.clone(),
                            region.clone(),
                            description.clone(),
                            self.json,
                        )
                        .await?;
                    }
                    VolumeAction::List => {
                        handlers::volumes::handle_list_volumes(&client, self.json).await?;
                    }
                    VolumeAction::Delete { volume, yes } => {
                        handlers::volumes::handle_delete_volume(
                            &client,
                            volume.clone(),
                            *yes,
                            self.json,
                        )
                        .await?;
                    }
                    VolumeAction::Attach { volume, rental } => {
                        handlers::volumes::handle_attach_volume(
                            &client,
                            volume.clone(),
                            rental.clone(),
                            self.json,
                        )
                        .await?;
                    }
                    VolumeAction::Detach { volume, yes } => {
                        handlers::volumes::handle_detach_volume(
                            &client,
                            volume.clone(),
                            *yes,
                            self.json,
                        )
                        .await?;
                    }
                }
            }

            // Upgrade command is handled in main.rs before entering async runtime
            Commands::Upgrade { .. } => {
                unreachable!("Upgrade command should be handled in main.rs")
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
