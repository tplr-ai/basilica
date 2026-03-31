//! Sandbox management handlers for the CLI.

use basilica_sdk::{BasilicaClient, CreateSandboxRequest};
use color_eyre::eyre::eyre;

use crate::error::CliError;

pub async fn handle_create_sandbox(
    client: &BasilicaClient,
    image: String,
    cpu: String,
    memory: String,
    ttl: Option<u32>,
    network_isolation: String,
    json: bool,
) -> Result<(), CliError> {
    let request = CreateSandboxRequest {
        image: image.clone(),
        cpu: Some(cpu),
        memory: Some(memory),
        env: vec![],
        ttl_seconds: ttl,
        network_isolation: Some(network_isolation.clone()),
    };

    let sandbox = client.create_sandbox(request).await?;

    if json {
        let out = serde_json::json!({
            "sandboxId": sandbox.sandbox_id,
            "domain": sandbox.domain,
            "status": sandbox.status,
            "execAgentSecret": sandbox.exec_agent_secret(),
        });
        println!("{}", serde_json::to_string_pretty(&out).unwrap_or_default());
    } else {
        println!("Sandbox created successfully!");
        println!("  ID:     {}", sandbox.sandbox_id);
        println!("  Domain: {}", sandbox.domain);
        println!("  Status: {}", sandbox.status);
        println!("  Image:  {image}");
        println!("  Network Isolation: {network_isolation}");

        if let Some(secret) = sandbox.exec_agent_secret() {
            println!("  Exec Agent Secret: {secret}");
            println!("\n  Save the secret — it is shown only once.");
        }
    }

    Ok(())
}

pub async fn handle_list_sandboxes(
    client: &BasilicaClient,
    json: bool,
) -> Result<(), CliError> {
    let response = client.list_sandboxes().await?;

    if json {
        println!(
            "{}",
            serde_json::to_string_pretty(&response).unwrap_or_default()
        );
    } else {
        if response.sandboxes.is_empty() {
            println!("No active sandboxes.");
            return Ok(());
        }

        println!(
            "{:<14} {:<45} {:<10} {}",
            "ID", "IMAGE", "STATUS", "DOMAIN"
        );
        println!("{}", "-".repeat(90));

        for sb in &response.sandboxes {
            let domain = sb.domain.as_deref().unwrap_or("-");
            println!(
                "{:<14} {:<45} {:<10} {}",
                sb.sandbox_id, sb.image, sb.status, domain
            );
        }
    }

    Ok(())
}

pub async fn handle_get_sandbox(
    client: &BasilicaClient,
    sandbox_id: String,
    json: bool,
) -> Result<(), CliError> {
    let detail = client.get_sandbox(&sandbox_id).await?;

    if json {
        println!(
            "{}",
            serde_json::to_string_pretty(&detail).unwrap_or_default()
        );
    } else {
        let domain = detail.domain.as_deref().unwrap_or("-");
        println!("Sandbox: {}", detail.sandbox_id);
        println!("  Image:  {}", detail.image);
        println!("  CPU:    {}", detail.cpu);
        println!("  Memory: {}", detail.memory);
        println!("  Status: {}", detail.status);
        println!("  Domain: {domain}");
    }

    Ok(())
}

pub async fn handle_delete_sandbox(
    client: &BasilicaClient,
    sandbox_id: String,
) -> Result<(), CliError> {
    client.delete_sandbox(&sandbox_id).await?;
    println!("Sandbox {sandbox_id} is being deleted.");
    Ok(())
}

pub async fn handle_exec_sandbox(
    client: &BasilicaClient,
    sandbox_id: String,
    command: Vec<String>,
) -> Result<(), CliError> {
    let detail = client.get_sandbox(&sandbox_id).await?;

    let domain = detail
        .domain
        .as_deref()
        .ok_or_else(|| eyre!("Sandbox has no domain yet — it may still be starting"))?;

    // Exec requires the exec_agent_secret which is only returned at creation time.
    // The CLI user must have stored it. Check the BASILICA_SANDBOX_SECRET env var.
    let secret = std::env::var("BASILICA_SANDBOX_SECRET").map_err(|_| {
        eyre!(
            "BASILICA_SANDBOX_SECRET env var is required for exec. \
             This secret is returned when you create a sandbox — store it securely."
        )
    })?;

    let sandbox = basilica_sdk::Sandbox::from(basilica_sdk::sandbox::CreateSandboxResponse {
        sandbox_id: sandbox_id.clone(),
        domain: domain.to_string(),
        status: detail.status,
        exec_agent_secret: secret,
    });

    let result = sandbox.exec(command).await?;

    if !result.stdout.is_empty() {
        print!("{}", result.stdout);
    }
    if !result.stderr.is_empty() {
        eprint!("{}", result.stderr);
    }
    if result.exit_code != 0 {
        std::process::exit(result.exit_code);
    }

    Ok(())
}
