//! Authentication command handlers

use crate::auth::{should_use_device_flow, CallbackServer, DeviceFlow, OAuthFlow, TokenStore};
use crate::config::CliConfig;
use crate::error::CliError;
use crate::output::{banner, compress_path, print_success};
use crate::progress::{complete_spinner_and_clear, complete_spinner_error, create_spinner};
use color_eyre::eyre::{eyre, WrapErr};
use color_eyre::Section;
use tracing::{debug, warn};

/// Handle login command
pub async fn handle_login(device_code: bool, config: &CliConfig) -> Result<(), CliError> {
    handle_login_with_options(device_code, config, true).await
}

/// Handle login with configurable options
pub async fn handle_login_with_options(
    device_code: bool,
    config: &CliConfig,
    show_suggestions: bool,
) -> Result<(), CliError> {
    debug!("Starting login process, device_code: {}", device_code);

    println!(
        "{}",
        console::style("⛪ Basilica - Sacred Compute ⛪")
            .red()
            .bold()
    );
    println!();

    // Determine which flow to use
    let use_device_flow = device_code || should_use_device_flow();

    // Create authentication configuration with available port
    let auth_config = if use_device_flow {
        // Device flow doesn't need a callback server, so port doesn't matter
        crate::config::create_auth_config_with_port(0)
    } else {
        // Find available port for OAuth callback server
        let port = CallbackServer::find_available_port().map_err(|e| {
            eyre!(e)
                .wrap_err("Failed to find available port")
                .suggestion("Try 'basilica login --device-code' for device flow")
                .note("The standard OAuth flow requires an available local port")
        })?;
        crate::config::create_auth_config_with_port(port)
    };

    // Initialize token store
    let data_dir = CliConfig::data_dir().wrap_err("Failed to get data directory")?;
    let token_store = TokenStore::new(data_dir).wrap_err("Failed to initialize token store")?;

    let token_set = if use_device_flow {
        let spinner = create_spinner("Requesting device code...");
        let device_flow = DeviceFlow::new(auth_config);

        match device_flow.start_flow().await {
            Ok(tokens) => {
                complete_spinner_and_clear(spinner);
                tokens
            }
            Err(e) => {
                complete_spinner_error(spinner, "Authentication failed");
                return Err(e.into());
            }
        }
    } else {
        let mut oauth_flow = OAuthFlow::new(auth_config);

        match oauth_flow.start_flow().await {
            Ok(tokens) => tokens,
            Err(e) => {
                return Err(e.into());
            }
        }
    };

    let spinner = create_spinner("Storing authentication tokens...");

    // Store the tokens securely
    if let Err(e) = token_store.store_tokens(&token_set).await {
        complete_spinner_error(spinner, "Failed to store tokens");
        return Err(eyre!("Failed to store tokens: {}", e)
            .suggestion("Check that you have permission to access the CLI data directory and auth.json file")
            .note("The auth file is located at: ~/.local/share/basilica/auth.json. Ensure the directory exists and is writable.")
            .into());
    }

    complete_spinner_and_clear(spinner);

    print_success("⛪ Login successful!");
    println!();

    // Check and generate SSH keys if they don't exist
    let spinner = create_spinner("Checking SSH keys...");

    // First check if keys already exist
    if config.ssh.ssh_keys_exist() {
        complete_spinner_and_clear(spinner);
        debug!("SSH keys already exist");
    } else {
        // Keys don't exist, generate them
        spinner.set_message("Generating SSH keys for GPU rentals...");

        match crate::ssh::ensure_ssh_keys_exist(&config.ssh).await {
            Ok(()) => {
                complete_spinner_and_clear(spinner);
                // Show user where keys were generated
                print_success("🔑 SSH keys generated successfully!");
                println!("  Location: {}", compress_path(&config.ssh.key_path));
                println!();
            }
            Err(e) => {
                complete_spinner_error(spinner, "Failed to generate SSH keys");
                warn!("Failed to generate SSH keys: {}", e);
                println!();
                println!("⚠️  SSH keys could not be generated automatically.");
                println!("   You can generate them manually with:");
                println!("   ssh-keygen -t ed25519 -f ~/.ssh/basilica_ed25519");
                println!();
                // Don't fail the login, just warn
            }
        }
    }

    // Show helpful command suggestions only if requested
    if show_suggestions {
        banner::print_command_suggestions();
    }

    Ok(())
}

/// Handle logout command
pub async fn handle_logout(_config: &CliConfig) -> Result<(), CliError> {
    let spinner = create_spinner("Checking authentication status...");

    // Initialize token store
    let data_dir = CliConfig::data_dir().wrap_err("Failed to get data directory")?;
    let token_store = TokenStore::new(data_dir).wrap_err("Failed to initialize token store")?;

    // Check if user is currently logged in and get tokens for revocation
    let tokens = match token_store.get_tokens().await {
        Ok(Some(tokens)) => {
            complete_spinner_and_clear(spinner);
            Some(tokens)
        }
        Ok(None) => {
            complete_spinner_and_clear(spinner);
            println!("You are not currently logged in.");
            return Ok(());
        }
        Err(e) => {
            warn!("Failed to check login status: {}", e);
            complete_spinner_error(spinner, "Failed to check status");
            // Continue with logout anyway to ensure cleanup
            None
        }
    };

    // Attempt to revoke tokens with Auth0 before deleting locally
    if let Some(ref token_set) = tokens {
        let spinner = create_spinner("Revoking authentication tokens...");

        // Create auth config (port doesn't matter for revocation)
        let auth_config = crate::config::create_auth_config_with_port(0);
        let oauth_flow = OAuthFlow::new(auth_config);

        match oauth_flow.revoke_token(token_set).await {
            Ok(()) => {
                complete_spinner_and_clear(spinner);
            }
            Err(e) => {
                warn!("Failed to revoke tokens with Auth0: {}", e);
                complete_spinner_error(spinner, "Token revocation failed, continuing cleanup");
            }
        }
    }

    let spinner = create_spinner("Clearing local authentication data...");

    // Delete stored authentication tokens
    if let Err(e) = token_store.delete_tokens().await {
        complete_spinner_error(spinner, "Failed to clear tokens");
        return Err(eyre!("Failed to clear authentication data: {}", e)
            .note("The auth file is located at: ~/.local/share/basilica/auth.json. Ensure you have write permissions to delete the file.")
            .into());
    }

    complete_spinner_and_clear(spinner);
    print_success("⛪ Logout successful!");
    Ok(())
}
