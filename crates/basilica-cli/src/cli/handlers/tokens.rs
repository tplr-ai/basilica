//! Token management handlers for the Basilica CLI

use crate::error::CliError;
use crate::output::{print_success, table_output};
use basilica_common::{ApiKeyName, ApiKeyNameError};
use basilica_sdk::BasilicaClient;
use console::style;
use dialoguer::{theme::ColorfulTheme, Confirm, Input, Select};

/// Handle creating a new token
pub async fn handle_create_token(
    client: &BasilicaClient,
    name: Option<String>,
) -> Result<(), CliError> {
    // Get name interactively if not provided
    let name = match name {
        Some(n) => {
            // Validate provided name
            ApiKeyName::new(n.clone()).map_err(|e| {
                CliError::Internal(color_eyre::eyre::eyre!(
                    "Invalid API key name: {}",
                    match e {
                        ApiKeyNameError::Empty => "Name cannot be empty",
                        ApiKeyNameError::TooLong => "Name too long (max 100 characters)",
                        ApiKeyNameError::InvalidCharacters =>
                            "Only alphanumeric characters, hyphens, and underscores are allowed",
                    }
                ))
            })?;
            n
        }
        None => {
            let theme = ColorfulTheme::default();
            let input: String = Input::with_theme(&theme)
                .with_prompt("Enter a name for this API key (letters, numbers, -, _ only)")
                .interact_text()
                .map_err(|e| CliError::Internal(e.into()))?;

            // Validate the input
            ApiKeyName::new(input.clone()).map_err(|e| {
                CliError::Internal(color_eyre::eyre::eyre!(
                    "Invalid API key name: {}",
                    match e {
                        ApiKeyNameError::Empty => "Name cannot be empty",
                        ApiKeyNameError::TooLong => "Name too long (max 100 characters)",
                        ApiKeyNameError::InvalidCharacters =>
                            "Only alphanumeric characters, hyphens, and underscores are allowed",
                    }
                ))
            })?;

            input
        }
    };

    // Check if we're at the limit
    let existing_keys = client.list_api_keys().await.map_err(CliError::Api)?;
    if existing_keys.len() >= 10 {
        println!(
            "{}",
            style("⚠️  You have reached the maximum number of API keys (10). Please delete one first.").yellow()
        );
        return Ok(());
    } else if existing_keys.len() >= 8 {
        println!(
            "{}",
            style(format!(
                "⚠️  You have {} API keys. Maximum allowed is 10.",
                existing_keys.len()
            ))
            .yellow()
        );
    }

    // Create the new token
    let response = client.create_api_key(&name).await.map_err(CliError::Api)?;

    // Display the key with clear formatting
    print_success("Token created successfully!");
    println!();
    println!("Token: {}", style(&response.token).cyan());
    println!();
    println!(
        "{}",
        style("⚠️  Save this token - it won't be shown again!")
            .yellow()
            .bold()
    );
    println!();
    println!("To use this token:");
    println!("export BASILICA_API_TOKEN=\"{}\"", response.token);

    Ok(())
}

/// Handle listing all tokens
pub async fn handle_list_tokens(client: &BasilicaClient) -> Result<(), CliError> {
    let keys = client.list_api_keys().await.map_err(CliError::Api)?;

    if keys.is_empty() {
        println!("No API keys exist.");
        println!(
            "Create one with: {} tokens create [name]",
            style("basilica").cyan()
        );
    } else {
        table_output::display_api_keys(&keys).map_err(|e| {
            CliError::Internal(color_eyre::eyre::eyre!("Failed to display API keys: {}", e))
        })?;
    }

    Ok(())
}

/// Handle revoking a token
pub async fn handle_revoke_token(
    client: &BasilicaClient,
    name: Option<String>,
    skip_confirm: bool,
) -> Result<(), CliError> {
    // Get all keys
    let keys = client.list_api_keys().await.map_err(CliError::Api)?;

    if keys.is_empty() {
        println!("No API keys exist to revoke.");
        return Ok(());
    }

    // Determine which key to delete
    let key_name = match name {
        Some(n) => n,
        None => {
            // Interactive selection
            let options: Vec<String> = keys
                .iter()
                .map(|k| {
                    let last_used = k
                        .last_used_at
                        .map(|d| format!("last used: {}", d.format("%Y-%m-%d")))
                        .unwrap_or_else(|| "never used".to_string());
                    format!("{} ({})", k.name, last_used)
                })
                .collect();

            let theme = ColorfulTheme::default();
            let selection = Select::with_theme(&theme)
                .with_prompt("Select an API key to revoke")
                .items(&options)
                .default(0)
                .interact()
                .map_err(|e| CliError::Internal(e.into()))?;

            keys[selection].name.clone()
        }
    };

    // Confirm deletion if not skipped
    if !skip_confirm {
        let theme = ColorfulTheme::default();
        let confirmed = Confirm::with_theme(&theme)
            .with_prompt(format!("Are you sure you want to delete '{}'?", key_name))
            .default(false)
            .interact()
            .map_err(|e| CliError::Internal(e.into()))?;

        if !confirmed {
            println!("Deletion cancelled.");
            return Ok(());
        }
    }

    // Delete the specific token
    client
        .revoke_api_key(&key_name)
        .await
        .map_err(CliError::Api)?;

    println!(
        "{}",
        style(format!("✅ API key '{}' deleted successfully.", key_name)).green()
    );

    Ok(())
}
