use super::HandlerUtils;
use crate::cli::{commands::ValidatorCommands, CliContext};
use crate::validation_session::ValidatorId;
use anyhow::Result;
use common::network::get_public_ip;

pub async fn handle_validator_command(cmd: &ValidatorCommands, context: &CliContext) -> Result<()> {
    match cmd {
        ValidatorCommands::Grant {
            hotkey,
            access_type,
            duration,
            ssh_public_key,
        } => {
            grant_access(
                hotkey,
                access_type,
                *duration,
                ssh_public_key.as_deref(),
                context,
            )
            .await
        }
        ValidatorCommands::Revoke { hotkey } => revoke_access(hotkey, context).await,
        ValidatorCommands::List => list_access(context).await,
        ValidatorCommands::Logs { hotkey, limit } => {
            show_logs(hotkey.as_deref(), *limit, context).await
        }
    }
}

async fn grant_access(
    hotkey: &str,
    access_type_str: &str,
    _duration: u64,
    ssh_public_key: Option<&str>,
    context: &CliContext,
) -> Result<()> {
    HandlerUtils::print_info(&format!(
        "Granting {access_type_str} access to validator: {hotkey}"
    ));

    let config = HandlerUtils::load_config(&context.config_path)?;
    let state = HandlerUtils::init_executor_state(config).await?;

    // Check if validator service is enabled
    let validation_service = match &state.validation_service {
        Some(service) => service,
        None => {
            HandlerUtils::print_error("Validator service is not enabled in configuration");
            return Ok(());
        }
    };

    // We only support SSH access type now
    if access_type_str != "ssh" {
        HandlerUtils::print_error(&format!(
            "Only SSH access type is supported, got: {access_type_str}"
        ));
        return Ok(());
    }

    let validator_id = ValidatorId::new(hotkey.to_string());

    // Grant SSH access (simplified)
    match ssh_public_key {
        Some(pub_key) => {
            match validation_service
                .grant_ssh_access(&validator_id, pub_key)
                .await
            {
                Ok(()) => {
                    HandlerUtils::print_success("SSH access granted successfully");
                    let username = format!("validator_{}", validator_id.hotkey);
                    let public_ip = get_public_ip().await;
                    println!("SSH Details:");
                    println!("  Validator: {validator_id}");
                    println!("  SSH Username: {username}");
                    println!("  SSH Command: ssh {username}@{public_ip}");
                }
                Err(e) => {
                    HandlerUtils::print_error(&format!("Failed to grant SSH access: {e}"));
                }
            }
        }
        None => {
            HandlerUtils::print_error("SSH public key is required for simplified access");
        }
    }

    Ok(())
}

async fn revoke_access(hotkey: &str, context: &CliContext) -> Result<()> {
    HandlerUtils::print_info(&format!("Revoking access for validator: {hotkey}"));

    let config = HandlerUtils::load_config(&context.config_path)?;
    let state = HandlerUtils::init_executor_state(config).await?;

    // Check if validator service is enabled
    let validation_service = match &state.validation_service {
        Some(service) => service,
        None => {
            HandlerUtils::print_error("Validator service is not enabled in configuration");
            return Ok(());
        }
    };

    let validator_id = ValidatorId::new(hotkey.to_string());

    match validation_service.revoke_ssh_access(&validator_id).await {
        Ok(()) => {
            HandlerUtils::print_success(&format!("Access revoked for validator: {hotkey}"));
        }
        Err(e) => {
            HandlerUtils::print_error(&format!("Failed to revoke access: {e}"));
        }
    }

    Ok(())
}

async fn list_access(context: &CliContext) -> Result<()> {
    HandlerUtils::print_info("Listing active validator access...");

    let config = HandlerUtils::load_config(&context.config_path)?;
    let state = HandlerUtils::init_executor_state(config).await?;

    // Check if validator service is enabled
    let validation_service = match &state.validation_service {
        Some(service) => service,
        None => {
            HandlerUtils::print_error("Validator service is not enabled in configuration");
            return Ok(());
        }
    };

    match validation_service.list_active_access().await {
        Ok(access_list) => {
            if access_list.is_empty() {
                HandlerUtils::print_info("No active validator access found");
            } else {
                println!("Active Validator Access:");
                println!(
                    "{:<20} {:<10} {:<20} {:<20} {:<10}",
                    "Hotkey", "Type", "Granted", "Expires", "SSH"
                );
                println!("{}", "-".repeat(80));

                for access in access_list {
                    let granted = access
                        .granted_at
                        .duration_since(std::time::SystemTime::UNIX_EPOCH)
                        .unwrap()
                        .as_secs();
                    let expires = access
                        .expires_at
                        .duration_since(std::time::SystemTime::UNIX_EPOCH)
                        .unwrap()
                        .as_secs();

                    println!(
                        "{:<20} {:<10} {:<20} {:<20} {:<10}",
                        access.validator_id.hotkey,
                        "ssh", // Only SSH access type supported now
                        granted,
                        expires,
                        if access.has_ssh_access() { "Yes" } else { "No" }
                    );
                }
            }
        }
        Err(e) => {
            HandlerUtils::print_error(&format!("Failed to list access: {e}"));
        }
    }

    Ok(())
}

async fn show_logs(hotkey: Option<&str>, limit: u32, context: &CliContext) -> Result<()> {
    let filter_msg = if let Some(hk) = hotkey {
        format!("for validator: {hk}")
    } else {
        "for all validators".to_string()
    };

    HandlerUtils::print_info(&format!(
        "Showing {limit} access logs {filter_msg} (limit: {limit})"
    ));

    let config = HandlerUtils::load_config(&context.config_path)?;
    let state = HandlerUtils::init_executor_state(config).await?;

    // Check if validator service is enabled
    let _validation_service = match &state.validation_service {
        Some(service) => service,
        None => {
            HandlerUtils::print_error("Validator service is not enabled in configuration");
            return Ok(());
        }
    };

    // Use the simplified journal query
    let log_entries = common::journal::query_logs(
        hotkey,
        Some("1 hour ago"), // Show logs from last hour
        Some(limit as usize),
    );

    match log_entries {
        Ok(entries) => {
            if entries.is_empty() {
                HandlerUtils::print_info("No log entries found");
            } else {
                println!("Validator Access Logs:");
                println!("{}", "-".repeat(80));
                for entry in entries {
                    println!("{entry}");
                }
            }
        }
        Err(e) => {
            HandlerUtils::print_error(&format!("Failed to read logs: {e}"));
        }
    }

    Ok(())
}
