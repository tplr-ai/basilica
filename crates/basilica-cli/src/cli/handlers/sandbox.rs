//! Sandbox management handlers for the CLI.

use basilica_sdk::sandbox::SandboxEnvVar;
use basilica_sdk::{BasilicaClient, CreateSandboxRequest};
use color_eyre::eyre::eyre;
use std::path::PathBuf;

use crate::error::CliError;

fn parse_env_vars(raw: &[String]) -> Result<Vec<SandboxEnvVar>, CliError> {
    raw.iter()
        .map(|s| {
            let (name, value) = s.split_once('=').ok_or_else(|| {
                CliError::from(eyre!(
                    "invalid --env format: '{s}'. Expected KEY=VALUE"
                ))
            })?;
            Ok(SandboxEnvVar {
                name: name.to_string(),
                value: value.to_string(),
            })
        })
        .collect()
}

pub async fn handle_create_sandbox(
    client: &BasilicaClient,
    image: String,
    cpu: String,
    memory: String,
    ttl: Option<u32>,
    network_isolation: String,
    env_raw: Vec<String>,
    json: bool,
) -> Result<(), CliError> {
    let env = parse_env_vars(&env_raw)?;

    let request = CreateSandboxRequest {
        image: image.clone(),
        cpu: Some(cpu),
        memory: Some(memory),
        env,
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

fn sandbox_secret_from_env(action: &str) -> Result<String, CliError> {
    std::env::var("BASILICA_SANDBOX_SECRET").map_err(|_| {
        eyre!(
            "BASILICA_SANDBOX_SECRET env var is required for {action}. \
             This secret is returned when you create a sandbox — store it securely."
        )
        .into()
    })
}

async fn load_sandbox_handle(
    client: &BasilicaClient,
    sandbox_id: &str,
    action: &str,
) -> Result<basilica_sdk::Sandbox, CliError> {
    let detail = client.get_sandbox(sandbox_id).await?;
    let domain = detail
        .domain
        .as_deref()
        .ok_or_else(|| eyre!("Sandbox has no domain yet — it may still be starting"))?;
    let secret = sandbox_secret_from_env(action)?;

    Ok(basilica_sdk::Sandbox::from(
        basilica_sdk::sandbox::CreateSandboxResponse {
            sandbox_id: sandbox_id.to_string(),
            domain: domain.to_string(),
            status: detail.status,
            exec_agent_secret: secret,
        },
    ))
}

fn print_exec_response(
    result: &basilica_sdk::sandbox::ExecResponse,
    json: bool,
) -> Result<(), CliError> {
    if json {
        println!(
            "{}",
            serde_json::to_string_pretty(result)
                .map_err(|e| CliError::from(eyre!("failed to serialize response: {e}")))?,
        );
    } else {
        if !result.stdout.is_empty() {
            print!("{}", result.stdout);
        }
        if !result.stderr.is_empty() {
            eprint!("{}", result.stderr);
        }
    }

    if result.exit_code != 0 {
        std::process::exit(result.exit_code);
    }

    Ok(())
}

pub async fn handle_exec_sandbox(
    client: &BasilicaClient,
    sandbox_id: String,
    command: Vec<String>,
    json: bool,
) -> Result<(), CliError> {
    let sandbox = load_sandbox_handle(client, &sandbox_id, "exec").await?;
    let result = sandbox.exec(command).await?;
    print_exec_response(&result, json)
}

pub async fn handle_run_sandbox(
    client: &BasilicaClient,
    sandbox_id: String,
    file: Option<PathBuf>,
    code: Option<String>,
    json: bool,
) -> Result<(), CliError> {
    let sandbox = load_sandbox_handle(client, &sandbox_id, "run").await?;
    let code = match (file, code) {
        (Some(path), None) => std::fs::read_to_string(&path)
            .map_err(|e| CliError::from(eyre!("failed to read {}: {e}", path.display())))?,
        (None, Some(code)) => code,
        (Some(_), Some(_)) => {
            return Err(CliError::from(eyre!(
                "pass either inline CODE or --file, not both"
            )));
        }
        (None, None) => {
            return Err(CliError::from(eyre!(
                "missing code: pass inline CODE or --file"
            )));
        }
    };

    // TODO: add CLI flags for language/args once the SDK exposes a richer `run` API.
    let result = sandbox.run(&code).await?;
    print_exec_response(&result, json)
}
