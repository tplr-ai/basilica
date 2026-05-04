//! Share token management handlers for private deployments.

use super::helpers::display_name;
use crate::cli::commands::ShareTokenAction;
use crate::error::{CliError, DeployError};
use crate::progress::{complete_spinner_and_clear, create_spinner};
use basilica_sdk::{ApiError, BasilicaClient};
use color_eyre::eyre::eyre;
use console::style;

/// Identifier pair used by share-token handlers: the UUID for API calls, the
/// friendly label for human output.
struct ResolvedShareTarget {
    instance_name: String,
    display_name: String,
}

/// Create error for when share token operations are attempted on public deployments.
fn public_deployment_error(name: &str) -> CliError {
    CliError::Deploy(DeployError::ShareTokenError {
        message: format!(
            "Deployment '{}' is public. Share tokens only apply to private deployments.",
            name
        ),
    })
}

/// Map API errors to CLI errors for share token operations.
fn map_share_token_error(e: ApiError, name: &str) -> CliError {
    match e {
        ApiError::BadRequest { .. } => public_deployment_error(name),
        ApiError::NotFound { .. } => CliError::Deploy(DeployError::NotFound {
            name: name.to_string(),
        }),
        other => CliError::Api(other),
    }
}

pub async fn handle_share_token(
    client: &BasilicaClient,
    action: ShareTokenAction,
) -> Result<(), CliError> {
    match action {
        ShareTokenAction::Regenerate { name } => handle_regenerate(client, name).await,
        ShareTokenAction::Status { name } => handle_token_status(client, name).await,
        ShareTokenAction::Revoke { name, yes } => handle_revoke(client, name, yes).await,
    }
}

async fn handle_regenerate(client: &BasilicaClient, name: Option<String>) -> Result<(), CliError> {
    let target = resolve_private_deployment_name(client, name).await?;

    let spinner = create_spinner(&format!(
        "Regenerating share token for deployment '{}'...",
        target.display_name
    ));
    let result = client.regenerate_share_token(&target.instance_name).await;
    complete_spinner_and_clear(spinner);

    let response = result.map_err(|e| map_share_token_error(e, &target.display_name))?;

    println!();
    println!(
        "{}",
        style("Share token regenerated successfully!")
            .green()
            .bold()
    );
    println!();
    println!(
        "{}",
        style("Share Token (save this - cannot be retrieved later):")
            .yellow()
            .bold()
    );
    println!("  Token:     {}", style(&response.token).cyan());
    println!("  Share URL: {}", style(&response.share_url).cyan());
    println!();
    println!(
        "{}",
        style("Note: Previous share token has been invalidated.").dim()
    );

    Ok(())
}

async fn handle_token_status(
    client: &BasilicaClient,
    name: Option<String>,
) -> Result<(), CliError> {
    let target = resolve_private_deployment_name(client, name).await?;

    let spinner = create_spinner(&format!(
        "Checking share token for '{}'...",
        target.display_name
    ));
    let result = client.get_share_token_status(&target.instance_name).await;
    complete_spinner_and_clear(spinner);

    let response = result.map_err(|e| map_share_token_error(e, &target.display_name))?;

    if response.exists {
        println!(
            "Deployment '{}' has an active share token.",
            style(&target.display_name).cyan()
        );
        println!(
            "Use {} to regenerate or {} to revoke.",
            style("share-token regenerate").yellow(),
            style("share-token revoke").yellow()
        );
    } else {
        println!(
            "Deployment '{}' does not have a share token.",
            style(&target.display_name).cyan()
        );
        println!(
            "Use {} to generate one.",
            style("share-token regenerate").yellow()
        );
    }

    Ok(())
}

async fn handle_revoke(
    client: &BasilicaClient,
    name: Option<String>,
    skip_confirmation: bool,
) -> Result<(), CliError> {
    let target = resolve_private_deployment_name(client, name).await?;
    let display = &target.display_name;

    if !skip_confirmation {
        use dialoguer::{theme::ColorfulTheme, Confirm};

        let confirm = Confirm::with_theme(&ColorfulTheme::default())
            .with_prompt(format!(
                "Revoke share token for '{}'? The deployment will no longer be accessible via share URL.",
                display
            ))
            .default(false)
            .interact()
            .map_err(|e| CliError::Internal(eyre!("Failed to get confirmation: {}", e)))?;

        if !confirm {
            println!("Cancelled.");
            return Ok(());
        }
    }

    let spinner = create_spinner(&format!("Revoking share token for '{}'...", display));
    let result = client.delete_share_token(&target.instance_name).await;
    complete_spinner_and_clear(spinner);

    let response = result.map_err(|e| map_share_token_error(e, display))?;

    if response.revoked {
        println!(
            "{} Share token revoked for deployment '{}'.",
            style("Success:").green().bold(),
            display
        );
    } else {
        println!("No share token existed for deployment '{}'.", display);
    }

    Ok(())
}

/// Resolve a user identifier (friendly name or UUID) to the UUID `instance_name`
/// plus a display label, restricted to private deployments. If `name` is `None`
/// the user picks from a list rendered by friendly name.
async fn resolve_private_deployment_name(
    client: &BasilicaClient,
    name: Option<String>,
) -> Result<ResolvedShareTarget, CliError> {
    let spinner = create_spinner("Fetching deployments...");
    let list = client.list_deployments().await.map_err(CliError::Api)?;
    complete_spinner_and_clear(spinner);

    let private: Vec<_> = list.deployments.iter().filter(|d| !d.public).collect();

    if private.is_empty() {
        return Err(CliError::Deploy(DeployError::NoPrivateDeployments));
    }

    if let Some(input) = name {
        if let Some(d) = private
            .iter()
            .find(|d| !d.friendly_name.is_empty() && d.friendly_name == input)
        {
            return Ok(ResolvedShareTarget {
                instance_name: d.instance_name.clone(),
                display_name: display_name(&d.friendly_name, &d.instance_name).to_string(),
            });
        }
        if let Some(d) = private.iter().find(|d| d.instance_name == input) {
            return Ok(ResolvedShareTarget {
                instance_name: d.instance_name.clone(),
                display_name: display_name(&d.friendly_name, &d.instance_name).to_string(),
            });
        }
        return Err(CliError::Deploy(DeployError::NotFound { name: input }));
    }

    let labels: Vec<String> = private
        .iter()
        .map(|d| display_name(&d.friendly_name, &d.instance_name).to_string())
        .collect();

    use dialoguer::{theme::ColorfulTheme, Select};

    let selection = Select::with_theme(&ColorfulTheme::default())
        .with_prompt("Select private deployment")
        .items(&labels)
        .default(0)
        .interact()
        .map_err(|e| CliError::Internal(eyre!("Selection failed: {}", e)))?;

    let chosen = private[selection];
    Ok(ResolvedShareTarget {
        instance_name: chosen.instance_name.clone(),
        display_name: display_name(&chosen.friendly_name, &chosen.instance_name).to_string(),
    })
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_public_deployment_error_message() {
        let error = public_deployment_error("my-app");
        match error {
            CliError::Deploy(DeployError::ShareTokenError { message }) => {
                assert!(message.contains("my-app"));
                assert!(message.contains("public"));
                assert!(message.contains("private deployments"));
            }
            _ => panic!("Expected ShareTokenError"),
        }
    }

    #[test]
    fn test_map_share_token_error_bad_request() {
        let api_error = ApiError::BadRequest {
            message: "Deployment is public".to_string(),
        };
        let error = map_share_token_error(api_error, "test-app");
        match error {
            CliError::Deploy(DeployError::ShareTokenError { message }) => {
                assert!(message.contains("test-app"));
            }
            _ => panic!("Expected ShareTokenError for BadRequest"),
        }
    }

    #[test]
    fn test_map_share_token_error_not_found() {
        let api_error = ApiError::NotFound {
            resource: "deployment".to_string(),
        };
        let error = map_share_token_error(api_error, "missing-app");
        match error {
            CliError::Deploy(DeployError::NotFound { name }) => {
                assert_eq!(name, "missing-app");
            }
            _ => panic!("Expected NotFound error"),
        }
    }

    #[test]
    fn test_map_share_token_error_other() {
        let api_error = ApiError::Internal {
            message: "Server error".to_string(),
        };
        let error = map_share_token_error(api_error, "some-app");
        match error {
            CliError::Api(ApiError::Internal { message }) => {
                assert_eq!(message, "Server error");
            }
            _ => panic!("Expected Api error pass-through"),
        }
    }
}
