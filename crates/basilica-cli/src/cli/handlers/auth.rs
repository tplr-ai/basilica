//! Authentication command handlers

use crate::auth::{should_use_device_flow, CallbackServer, DeviceFlow, OAuthFlow, TokenStore};
use crate::config::CliConfig;
use crate::error::CliError;
use crate::output::{banner, print_success};
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
    let token_store = TokenStore::new(data_dir)
        .await
        .wrap_err("Failed to initialize token store")?;

    let token_set = if use_device_flow {
        let spinner = create_spinner("Requesting device code...");
        let device_flow = DeviceFlow::new(auth_config);

        let (instructions, pending) = match device_flow.start_flow().await {
            Ok(result) => {
                complete_spinner_and_clear(spinner);
                result
            }
            Err(e) => {
                complete_spinner_error(spinner, "Authentication failed");
                return Err(e.into());
            }
        };

        // Display instructions to user
        println!(
            "1. Visit: {}",
            console::style(&instructions.verification_uri).dim()
        );
        println!(
            "2. Enter code: {}",
            console::style(&instructions.user_code).bold()
        );
        if let Some(ref complete_uri) = instructions.verification_uri_complete {
            println!(
                "\n   Or visit this direct link: {}",
                console::style(complete_uri).dim()
            );
        }
        println!();
        crate::output::print_info("Waiting for authentication...");
        crate::output::print_info("Press Ctrl+C to cancel");

        // Wait for user to complete authentication
        match pending.wait_for_completion().await {
            Ok(tokens) => {
                // Clear the instructions from terminal
                let term = console::Term::stdout();
                let lines_to_clear = if instructions.verification_uri_complete.is_some() {
                    8
                } else {
                    6
                };
                let _ = term.clear_last_lines(lines_to_clear);
                tokens
            }
            Err(e) => {
                return Err(e.into());
            }
        }
    } else {
        let mut oauth_flow = OAuthFlow::new(auth_config);

        // Print info before opening browser
        crate::output::print_info("Opening browser for sign in...");
        crate::output::print_info("Browser didn't open? Use the URL below to sign in:");
        let auth_url = oauth_flow.get_auth_url().map_err(|e| eyre!(e))?;
        println!("{}", console::style(&auth_url).dim());
        crate::output::print_info("Waiting for authentication...");

        match oauth_flow.start_flow().await {
            Ok(tokens) => {
                // Clear the browser instructions
                let term = console::Term::stdout();
                let _ = term.clear_last_lines(6);
                tokens
            }
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

    // Trigger billing service to create user entry
    // We don't need the result - the call itself ensures the user exists in billing
    if let Ok(client) = crate::client::create_authenticated_client(config).await {
        let _ = client.get_balance().await;
    }

    print_success("⛪ Login successful!");
    println!();

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
    let token_store = TokenStore::new(data_dir)
        .await
        .wrap_err("Failed to initialize token store")?;

    // Check if user is currently logged in and get tokens for revocation
    let tokens = match token_store.retrieve_tokens().await {
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
