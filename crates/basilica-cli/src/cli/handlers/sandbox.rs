//! Sandbox command handlers.

use crate::cli::commands::SandboxAction;
use crate::config::CliConfig;
use crate::error::CliError;
use basilica_sdk::types::{CreateSandboxRequest, SandboxStatusResponse};
use console::style;
use dialoguer::Confirm;

pub async fn handle_sandbox(
    action: SandboxAction,
    json: bool,
    config: &CliConfig,
) -> Result<(), CliError> {
    let client = crate::client::create_client(config).await?;

    match action {
        SandboxAction::Create {
            image,
            ttl,
            idle_timeout,
        } => {
            let request = CreateSandboxRequest {
                image: image
                    .unwrap_or_else(|| "ghcr.io/one-covenant/sandbox-base:latest".to_string()),
                ttl_seconds: ttl.unwrap_or(3600),
                idle_timeout_seconds: idle_timeout.unwrap_or(1800),
            };

            println!(
                "{} Creating sandbox (image: {}, TTL: {}s)...",
                style("[sandbox]").cyan().bold(),
                request.image,
                request.ttl_seconds,
            );

            let resp = client.create_sandbox(request).await.map_err(|e| {
                CliError::Api(e)
            })?;

            if json {
                println!(
                    "{}",
                    serde_json::to_string_pretty(&resp)
                        .unwrap_or_else(|_| "{}".to_string())
                );
                return Ok(());
            }

            println!(
                "{} Sandbox created: {}",
                style("[sandbox]").cyan().bold(),
                style(&resp.sandbox_id).green().bold()
            );
            println!("  Domain:      {}", resp.domain);
            println!("  Status:      {}", resp.status);
            println!("  Rental ID:   {}", resp.rental_id);
            println!("  Hourly cost: {} credits", resp.hourly_cost);
            println!();

            // Poll until Running
            println!(
                "{} Waiting for sandbox to reach Running state...",
                style("[sandbox]").cyan().bold(),
            );

            match client
                .wait_for_sandbox_running(&resp.sandbox_id, 120, 2)
                .await
            {
                Ok(status) => {
                    println!(
                        "{} Sandbox is {}",
                        style("[sandbox]").cyan().bold(),
                        style("Running").green().bold()
                    );
                    println!();
                    println!(
                        "  Connect: wss://{}/ws",
                        status.domain
                    );
                    println!(
                        "  Header:  X-Exec-Secret: {}",
                        resp.exec_secret
                    );
                    println!();
                    println!(
                        "  {} The exec_secret is a bearer credential.",
                        style("Note:").yellow().bold(),
                    );
                    println!("  It is shown once. Do not persist or log it.");
                    println!();
                    println!(
                        "  Sandbox will expire in {}s (idle timeout: {}s).",
                        status.ttl_seconds, status.idle_timeout_seconds
                    );
                    println!("  Files in /workspace are ephemeral. Use R2 upload for persistence.");
                }
                Err(e) => {
                    eprintln!(
                        "{} Sandbox created but failed to reach Running: {}",
                        style("[sandbox]").red().bold(),
                        e
                    );
                    eprintln!("  Sandbox ID: {}", resp.sandbox_id);
                    eprintln!("  You can check status with: basilica sandbox status {}", resp.sandbox_id);
                }
            }
        }

        SandboxAction::List => {
            let resp = client.list_sandboxes().await.map_err(|e| {
                CliError::Api(e)
            })?;

            if json {
                println!(
                    "{}",
                    serde_json::to_string_pretty(&resp)
                        .unwrap_or_else(|_| "{}".to_string())
                );
                return Ok(());
            }

            if resp.sandboxes.is_empty() {
                println!("No sandboxes found.");
                return Ok(());
            }

            println!(
                "{:<38} {:<12} {:<44} {:<10} {:<8}",
                "ID", "STATUS", "DOMAIN", "TTL", "IMAGE"
            );
            println!("{}", "-".repeat(112));
            for sb in &resp.sandboxes {
                print_sandbox_row(sb);
            }
        }

        SandboxAction::Status { sandbox_id } => {
            let resp = client.get_sandbox(&sandbox_id).await.map_err(|e| {
                CliError::Api(e)
            })?;

            if json {
                println!(
                    "{}",
                    serde_json::to_string_pretty(&resp)
                        .unwrap_or_else(|_| "{}".to_string())
                );
                return Ok(());
            }

            println!(
                "{} Sandbox {}",
                style("[sandbox]").cyan().bold(),
                style(&resp.sandbox_id).green().bold()
            );
            println!("  Status:       {}", format_status(&resp.status));
            println!("  Domain:       {}", resp.domain);
            println!("  Image:        {}", resp.image);
            println!("  Created:      {}", resp.created_at);
            println!("  TTL:          {}s", resp.ttl_seconds);
            println!("  Idle timeout: {}s", resp.idle_timeout_seconds);
            if let Some(msg) = &resp.message {
                println!("  Message:      {}", msg);
            }
            if resp.status == "Running" {
                println!();
                println!("  Connect: wss://{}/ws", resp.domain);
            }
        }

        SandboxAction::Delete { sandbox_id, yes } => {
            if !yes {
                let confirmed = Confirm::new()
                    .with_prompt(format!("Delete sandbox {}?", sandbox_id))
                    .default(false)
                    .interact()
                    .unwrap_or(false);

                if !confirmed {
                    println!("Cancelled.");
                    return Ok(());
                }
            }

            let resp = client.delete_sandbox(&sandbox_id).await.map_err(|e| {
                CliError::Api(e)
            })?;

            if json {
                println!(
                    "{}",
                    serde_json::to_string_pretty(&resp)
                        .unwrap_or_else(|_| "{}".to_string())
                );
                return Ok(());
            }

            println!(
                "{} Sandbox {} — {}",
                style("[sandbox]").cyan().bold(),
                style(&resp.sandbox_id).yellow(),
                resp.message
            );
        }
    }

    Ok(())
}

fn print_sandbox_row(sb: &SandboxStatusResponse) {
    let image_short = sb
        .image
        .rsplit('/')
        .next()
        .unwrap_or(&sb.image);
    println!(
        "{:<38} {:<12} {:<44} {:<10} {:<8}",
        sb.sandbox_id,
        sb.status,
        sb.domain,
        format!("{}s", sb.ttl_seconds),
        image_short,
    );
}

fn format_status(status: &str) -> String {
    match status {
        "Running" => style(status).green().bold().to_string(),
        "Pending" => style(status).yellow().to_string(),
        "Terminating" => style(status).yellow().bold().to_string(),
        "Failed" => style(status).red().bold().to_string(),
        _ => status.to_string(),
    }
}
