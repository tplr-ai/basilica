//! Deploy command handlers

use crate::cli::commands::{DeployAction, DeployCommand};
use crate::client::create_authenticated_client;
use crate::config::CliConfig;
use crate::error::{CliError, DeployError};
use crate::output::{json_output, print_error, print_info, print_success};
use crate::progress::{complete_spinner_and_clear, create_spinner};
use basilica_sdk::ApiError;
use console::style;

mod create;
pub mod helpers;
mod share_token;
pub mod templates;
mod validation;

/// Handle all deploy subcommands (matches existing handler pattern)
pub async fn handle_deploy(cmd: DeployCommand, config: &CliConfig) -> Result<(), CliError> {
    // Validate request before doing anything
    if cmd.action.is_none() && cmd.source.is_some() {
        validation::validate_deployment_request(&cmd)?;
    }

    // Create authenticated client
    let client = create_authenticated_client(config).await?;

    // Global output flags from parent command
    let json = cmd.json;
    let show_phases = cmd.show_phases;

    match cmd.action {
        Some(DeployAction::List) => handle_list(&client, json).await,
        Some(DeployAction::Status { name, show_token }) => {
            let name = helpers::resolve_deployment_name(name, &client).await?;
            handle_status(&client, &name, json, show_phases, show_token).await
        }
        Some(DeployAction::Logs { name, follow, tail }) => {
            let name = helpers::resolve_deployment_name(name, &client).await?;
            handle_logs(&client, &name, follow, tail).await
        }
        Some(DeployAction::Delete { name, yes }) => {
            let name = helpers::resolve_deployment_name(name, &client).await?;
            handle_delete(&client, &name, yes).await
        }
        Some(DeployAction::Scale { name, replicas }) => {
            let name = helpers::resolve_deployment_name(name, &client).await?;
            handle_scale(&client, &name, replicas).await
        }
        Some(DeployAction::ShareToken { action }) => {
            share_token::handle_share_token(&client, action).await
        }
        Some(DeployAction::Vllm {
            model,
            common,
            vllm,
        }) => templates::handle_vllm_deploy(&client, model, common, vllm).await,
        Some(DeployAction::Sglang {
            model,
            common,
            sglang,
        }) => templates::handle_sglang_deploy(&client, model, common, sglang).await,
        Some(DeployAction::Openclaw { common, openclaw }) => {
            templates::handle_openclaw_deploy(&client, common, openclaw).await
        }
        Some(DeployAction::Tau { common, tau }) => {
            templates::handle_tau_deploy(&client, common, tau).await
        }
        None => {
            if let Some(source) = cmd.source.clone() {
                create::handle_create(&client, &source, cmd).await
            } else {
                print_error(
                    "No source specified. Use 'basilica summon <source>' or 'basilica summon ls'",
                );
                Ok(())
            }
        }
    }
}

/// List all deployments
async fn handle_list(client: &basilica_sdk::BasilicaClient, json: bool) -> Result<(), CliError> {
    let spinner = create_spinner("Fetching summons...");
    let result = client.list_deployments().await;
    complete_spinner_and_clear(spinner);
    let response = result.map_err(CliError::Api)?;

    if json {
        json_output(&response)?;
    } else {
        helpers::print_deployments_table(&response.deployments);
    }

    Ok(())
}

/// Get deployment status with phase tracking
async fn handle_status(
    client: &basilica_sdk::BasilicaClient,
    name: &str,
    json: bool,
    verbose: bool,
    show_token: bool,
) -> Result<(), CliError> {
    let spinner = create_spinner(&format!("Fetching summons '{}'...", name));
    let result = client.get_deployment(name).await;
    complete_spinner_and_clear(spinner);
    let response = result.map_err(|e| {
        if matches!(e, basilica_sdk::error::ApiError::NotFound { .. }) {
            CliError::Deploy(DeployError::NotFound {
                name: name.to_string(),
            })
        } else {
            CliError::Api(e)
        }
    })?;

    if json {
        json_output(&response)?;
    } else {
        helpers::print_deployment_details(&response, verbose);

        if show_token {
            handle_show_token_status(client, name).await?;
        }
    }

    Ok(())
}

/// Show token status for a deployment
async fn handle_show_token_status(
    client: &basilica_sdk::BasilicaClient,
    name: &str,
) -> Result<(), CliError> {
    match client.get_share_token_status(name).await {
        Ok(status) if status.exists => {
            println!();
            println!(
                "{}",
                style("Share token exists but cannot be retrieved.").yellow()
            );
            println!(
                "Use {} to generate a new token.",
                style("deploy share-token regenerate").cyan()
            );
        }
        Ok(_) => {
            println!();
            println!(
                "{}",
                style("No share token configured for this deployment.").dim()
            );
            println!(
                "Use {} to generate one.",
                style("deploy share-token regenerate").cyan()
            );
        }
        Err(ApiError::BadRequest { .. }) => {
            // Deployment is public, no token needed
            tracing::debug!("Deployment is public, share token not applicable");
        }
        Err(e) => {
            tracing::warn!("Could not fetch share token status: {}", e);
        }
    }
    Ok(())
}

/// Stream deployment logs
async fn handle_logs(
    client: &basilica_sdk::BasilicaClient,
    name: &str,
    follow: bool,
    tail: Option<u32>,
) -> Result<(), CliError> {
    let response = client
        .get_deployment_logs(name, follow, tail)
        .await
        .map_err(CliError::Api)?;

    helpers::stream_logs_to_stdout(response).await
}

/// Delete a deployment
async fn handle_delete(
    client: &basilica_sdk::BasilicaClient,
    name: &str,
    skip_confirm: bool,
) -> Result<(), CliError> {
    if !skip_confirm {
        use dialoguer::{theme::ColorfulTheme, Confirm};

        let confirm = Confirm::with_theme(&ColorfulTheme::default())
            .with_prompt(format!("Delete summons '{}'?", name))
            .default(false)
            .interact()
            .map_err(|e| {
                CliError::Internal(color_eyre::eyre::eyre!("Failed to get confirmation: {}", e))
            })?;

        if !confirm {
            print_info("Deletion cancelled");
            return Ok(());
        }
    }

    let spinner = create_spinner(&format!("Deleting summons '{}'...", name));
    let result = client.delete_deployment(name).await;
    complete_spinner_and_clear(spinner);
    result.map_err(CliError::Api)?;

    print_success(&format!("Summons '{}' deletion initiated", name));

    Ok(())
}

/// Scale deployment replicas
async fn handle_scale(
    client: &basilica_sdk::BasilicaClient,
    name: &str,
    replicas: u32,
) -> Result<(), CliError> {
    let spinner = create_spinner(&format!(
        "Scaling summons '{}' to {} replicas...",
        name, replicas
    ));

    // Verify deployment exists before scaling
    let verify_result = client.get_deployment(name).await;
    if let Err(e) = verify_result {
        complete_spinner_and_clear(spinner);
        return Err(
            if matches!(e, basilica_sdk::error::ApiError::NotFound { .. }) {
                CliError::Deploy(DeployError::NotFound {
                    name: name.to_string(),
                })
            } else {
                CliError::Api(e)
            },
        );
    }

    // Scale via dedicated endpoint
    let scale_result = client.scale_deployment(name, replicas).await;
    complete_spinner_and_clear(spinner);
    scale_result.map_err(CliError::Api)?;

    print_success(&format!(
        "Summons '{}' scaled to {} replicas",
        name, replicas
    ));

    Ok(())
}
