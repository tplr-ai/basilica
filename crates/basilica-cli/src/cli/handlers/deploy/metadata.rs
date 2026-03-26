//! Handlers for public deployment metadata enrollment

use crate::error::CliError;
use crate::output::{print_error, print_info, print_success};
use crate::progress::{complete_spinner_and_clear, create_spinner};
use basilica_sdk::error::ApiError;
use basilica_sdk::BasilicaClient;

/// Handle the enroll-metadata subcommand.
///
/// With no flags: display current enrollment status.
/// With --enable: enable metadata enrollment.
/// With --disable: disable metadata enrollment.
pub async fn handle_enroll_metadata(
    client: &BasilicaClient,
    name: &str,
    enable: bool,
    disable: bool,
) -> Result<(), CliError> {
    if !enable && !disable {
        return show_enrollment_status(client, name).await;
    }

    let enabled = enable;
    let action_word = if enabled { "Enabling" } else { "Disabling" };
    let spinner = create_spinner(&format!(
        "{} metadata enrollment for '{}'...",
        action_word, name
    ));

    let result = client.enroll_metadata(name, enabled).await;
    complete_spinner_and_clear(spinner);

    match result {
        Ok(response) => {
            if response.public_metadata {
                print_success(&format!(
                    "Public metadata enrollment enabled for '{}'",
                    name
                ));
                println!("  Validators can now verify this deployment via:");
                println!("  basilica deploy metadata {}", name);
            } else {
                print_success(&format!(
                    "Public metadata enrollment disabled for '{}'",
                    name
                ));
            }
            Ok(())
        }
        Err(ApiError::Conflict { message }) => {
            print_error(&format!("Conflict: {}", message));
            Ok(())
        }
        Err(e) => Err(CliError::Api(e)),
    }
}

/// Display current enrollment status.
async fn show_enrollment_status(client: &BasilicaClient, name: &str) -> Result<(), CliError> {
    let spinner = create_spinner(&format!("Checking enrollment status for '{}'...", name));
    let result = client.get_enrollment_status(name).await;
    complete_spinner_and_clear(spinner);

    let response = result.map_err(CliError::Api)?;

    if response.public_metadata {
        println!("Public Metadata: Enrolled");
        println!("  Metadata is publicly visible for validator verification.");
    } else {
        println!("Public Metadata: Not enrolled");
        println!("  Use --enable to opt-in to public metadata exposure.");
    }

    Ok(())
}

/// Handle the metadata subcommand (unauthenticated public lookup).
pub async fn handle_get_public_metadata(
    base_url: &str,
    name: &str,
    json: bool,
) -> Result<(), CliError> {
    let spinner = create_spinner(&format!("Fetching public metadata for '{}'...", name));

    let url = format!(
        "{}/public/deployments/{}/metadata",
        base_url,
        urlencoding::encode(name)
    );
    let http_client = reqwest::Client::new();
    let response =
        http_client.get(&url).send().await.map_err(|e| {
            CliError::Internal(color_eyre::eyre::eyre!("HTTP request failed: {}", e))
        })?;

    complete_spinner_and_clear(spinner);

    if !response.status().is_success() {
        if response.status() == reqwest::StatusCode::NOT_FOUND {
            print_info(&format!(
                "No public metadata found for '{}'. The deployment may not exist or metadata enrollment is not enabled.",
                name
            ));
            return Ok(());
        }
        let status = response.status();
        let body = response.text().await.unwrap_or_default();
        return Err(CliError::Internal(color_eyre::eyre::eyre!(
            "API returned {}: {}",
            status,
            body
        )));
    }

    let metadata: basilica_sdk::types::PublicDeploymentMetadataResponse =
        response.json().await.map_err(|e| {
            CliError::Internal(color_eyre::eyre::eyre!("Failed to parse response: {}", e))
        })?;

    if json {
        crate::output::json_output(&metadata)?;
    } else {
        println!("Public Deployment Metadata: {}", metadata.instance_name);
        println!();
        println!("  Image:    {}:{}", metadata.image, metadata.image_tag);
        println!("  ID:       {}", metadata.id);
        println!("  State:    {}", metadata.state);
        println!(
            "  Replicas: {}/{}",
            metadata.replicas.ready, metadata.replicas.desired
        );
        println!("  Uptime:   {}", format_uptime(metadata.uptime_seconds));
    }

    Ok(())
}

/// Format seconds into a human-readable duration string.
fn format_uptime(seconds: u64) -> String {
    let days = seconds / 86400;
    let hours = (seconds % 86400) / 3600;
    let minutes = (seconds % 3600) / 60;

    if days > 0 {
        format!("{}d {}h {}m", days, hours, minutes)
    } else if hours > 0 {
        format!("{}h {}m", hours, minutes)
    } else {
        format!("{}m", minutes)
    }
}
