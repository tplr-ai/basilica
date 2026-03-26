//! Share token management handlers for private deployments.

use crate::cli::commands::ShareTokenAction;
use crate::error::{CliError, DeployError};
use crate::progress::{complete_spinner_and_clear, create_spinner};
use basilica_sdk::{ApiError, BasilicaClient};
use color_eyre::eyre::eyre;
use console::style;

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
    let name = resolve_private_deployment_name(client, name).await?;

    let spinner = create_spinner(&format!(
        "Regenerating share token for deployment '{}'...",
        name
    ));
    let result = client.regenerate_share_token(&name).await;
    complete_spinner_and_clear(spinner);

    let response = result.map_err(|e| map_share_token_error(e, &name))?;

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
    let name = resolve_private_deployment_name(client, name).await?;

    let spinner = create_spinner(&format!("Checking share token for '{}'...", name));
    let result = client.get_share_token_status(&name).await;
    complete_spinner_and_clear(spinner);

    let response = result.map_err(|e| map_share_token_error(e, &name))?;

    if response.exists {
        println!(
            "Deployment '{}' has an active share token.",
            style(&name).cyan()
        );
        println!(
            "Use {} to regenerate or {} to revoke.",
            style("share-token regenerate").yellow(),
            style("share-token revoke").yellow()
        );
    } else {
        println!(
            "Deployment '{}' does not have a share token.",
            style(&name).cyan()
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
    let name = resolve_private_deployment_name(client, name).await?;

    if !skip_confirmation {
        use dialoguer::{theme::ColorfulTheme, Confirm};

        let confirm = Confirm::with_theme(&ColorfulTheme::default())
            .with_prompt(format!(
                "Revoke share token for '{}'? The deployment will no longer be accessible via share URL.",
                name
            ))
            .default(false)
            .interact()
            .map_err(|e| CliError::Internal(eyre!("Failed to get confirmation: {}", e)))?;

        if !confirm {
            println!("Cancelled.");
            return Ok(());
        }
    }

    let spinner = create_spinner(&format!("Revoking share token for '{}'...", name));
    let result = client.delete_share_token(&name).await;
    complete_spinner_and_clear(spinner);

    let response = result.map_err(|e| map_share_token_error(e, &name))?;

    if response.revoked {
        println!(
            "{} Share token revoked for deployment '{}'.",
            style("Success:").green().bold(),
            name
        );
    } else {
        println!("No share token existed for deployment '{}'.", name);
    }

    Ok(())
}

/// Resolve deployment name, with interactive selection filtering to private deployments only.
/// If name is explicitly provided, validates it belongs to a private deployment.
async fn resolve_private_deployment_name(
    client: &BasilicaClient,
    name: Option<String>,
) -> Result<String, CliError> {
    match name {
        Some(n) => {
            // Validate the deployment exists and is private by trying to get token status
            client
                .get_share_token_status(&n)
                .await
                .map(|_| n.clone())
                .map_err(|e| map_share_token_error(e, &n))
        }
        None => {
            // Interactive selection from private deployments only
            let spinner = create_spinner("Fetching deployments...");
            let result = client.list_deployments().await;
            complete_spinner_and_clear(spinner);

            let list = result.map_err(CliError::Api)?;
            let private_deployments: Vec<_> =
                list.deployments.iter().filter(|d| !d.public).collect();

            if private_deployments.is_empty() {
                return Err(CliError::Deploy(DeployError::NoPrivateDeployments));
            }

            let names: Vec<&str> = private_deployments
                .iter()
                .map(|d| d.instance_name.as_str())
                .collect();

            use dialoguer::{theme::ColorfulTheme, Select};

            let selection = Select::with_theme(&ColorfulTheme::default())
                .with_prompt("Select private deployment")
                .items(&names)
                .default(0)
                .interact()
                .map_err(|e| CliError::Internal(eyre!("Selection failed: {}", e)))?;

            Ok(names[selection].to_string())
        }
    }
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
