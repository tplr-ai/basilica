//! Sandbox management handlers for the CLI.

use basilica_sdk::sandbox::SandboxEnvVar;
use basilica_sdk::{BasilicaClient, CreateSandboxRequest};
use color_eyre::eyre::eyre;
use serde::Serialize;
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

fn display_optional<T: ToString>(value: Option<T>, fallback: &str) -> String {
    value
        .map(|value| value.to_string())
        .unwrap_or_else(|| fallback.to_string())
}

fn yes_no(value: bool) -> &'static str {
    if value { "yes" } else { "no" }
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
        println!("created {}", sandbox.sandbox_id);
        println!("  domain: {}", sandbox.domain);
        println!("  status: {}", sandbox.status);
        println!("  image: {}", image);
        println!("  isolation: {}", network_isolation);
        if let Some(ttl) = ttl {
            println!("  ttl: {}s", ttl);
        }

        if let Some(secret) = sandbox.exec_agent_secret() {
            println!("  secret: {secret}");
            println!("\n  Save it now; it is shown only once.");
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
        println!("id: {}", detail.sandbox_id);
        println!("status: {}", detail.status);
        println!("domain: {domain}");
        println!("image: {}", detail.image);
        println!("cpu: {}", detail.cpu);
        println!("memory: {}", detail.memory);
        println!("isolation: {}", detail.network_isolation);
        println!("ttl: {}", display_optional(detail.ttl_seconds, "-"));
        println!("warm_pool: {}", yes_no(detail.from_warm_pool));
        if let Some(ready_at) = detail.ready_at {
            println!("ready_at: {}", ready_at);
        }
        if let Some(expires_at) = detail.expires_at {
            println!("expires_at: {}", expires_at);
        }
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

pub async fn handle_rotate_sandbox_secret(
    client: &BasilicaClient,
    sandbox_id: String,
    json: bool,
) -> Result<(), CliError> {
    let response = client.rotate_sandbox_secret(&sandbox_id).await?;

    if json {
        let out = serde_json::json!({
            "sandboxId": response.sandbox_id,
            "execAgentSecret": response.exec_agent_secret,
        });
        println!("{}", serde_json::to_string_pretty(&out).unwrap_or_default());
    } else {
        println!("rotated secret for {}", response.sandbox_id);
        println!("  secret: {}", response.exec_agent_secret);
        println!("\n  Update BASILICA_SANDBOX_SECRET before the next data-plane call.");
    }

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

fn print_json<T: Serialize>(value: &T) -> Result<(), CliError> {
    println!(
        "{}",
        serde_json::to_string_pretty(value)
            .map_err(|e| CliError::from(eyre!("failed to serialize response: {e}")))?,
    );
    Ok(())
}

fn load_inline_or_file(
    file: Option<PathBuf>,
    content: Option<String>,
    what: &str,
) -> Result<String, CliError> {
    match (file, content) {
        (Some(path), None) => std::fs::read_to_string(&path)
            .map_err(|e| CliError::from(eyre!("failed to read {}: {e}", path.display()))),
        (None, Some(content)) => Ok(content),
        (Some(_), Some(_)) => Err(CliError::from(eyre!(
            "pass either inline {what} or --file, not both"
        ))),
        (None, None) => Err(CliError::from(eyre!(
            "missing {what}: pass inline {what} or --file"
        ))),
    }
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
    let code = load_inline_or_file(file, code, "CODE")?;

    // TODO: add CLI flags for language/args once the SDK exposes a richer `run` API.
    let result = sandbox.run(&code).await?;
    print_exec_response(&result, json)
}

pub async fn handle_read_sandbox_file(
    client: &BasilicaClient,
    sandbox_id: String,
    path: String,
    json: bool,
) -> Result<(), CliError> {
    let sandbox = load_sandbox_handle(client, &sandbox_id, "read").await?;
    let response = sandbox.files().read(&path).await?;
    if json {
        print_json(&response)
    } else {
        print!("{}", response.content);
        Ok(())
    }
}

pub async fn handle_write_sandbox_file(
    client: &BasilicaClient,
    sandbox_id: String,
    path: String,
    file: Option<PathBuf>,
    content: Option<String>,
    json: bool,
) -> Result<(), CliError> {
    let sandbox = load_sandbox_handle(client, &sandbox_id, "write").await?;
    let content = load_inline_or_file(file, content, "CONTENT")?;
    let response = sandbox.files().write(&path, &content).await?;
    if json {
        print_json(&response)
    } else {
        println!("Wrote {}", response.path);
        Ok(())
    }
}

pub async fn handle_list_sandbox_files(
    client: &BasilicaClient,
    sandbox_id: String,
    path: String,
    json: bool,
) -> Result<(), CliError> {
    let sandbox = load_sandbox_handle(client, &sandbox_id, "files").await?;
    let response = sandbox.files().list(&path).await?;
    if json {
        print_json(&response)
    } else {
        if response.files.is_empty() {
            println!("No files under {}.", path);
            return Ok(());
        }
        for file in response.files {
            let kind = if file.is_dir { "dir " } else { "file" };
            let size = file
                .size
                .map(|value| value.to_string())
                .unwrap_or_else(|| "-".to_string());
            println!("{:<4} {:>10} {}", kind, size, file.name);
        }
        Ok(())
    }
}

pub async fn handle_delete_sandbox_file(
    client: &BasilicaClient,
    sandbox_id: String,
    path: String,
    json: bool,
) -> Result<(), CliError> {
    let sandbox = load_sandbox_handle(client, &sandbox_id, "remove").await?;
    sandbox.files().delete(&path).await?;
    if json {
        print_json(&serde_json::json!({ "path": path, "status": "deleted" }))
    } else {
        println!("Deleted {}", path);
        Ok(())
    }
}

pub async fn handle_mkdir_sandbox(
    client: &BasilicaClient,
    sandbox_id: String,
    path: String,
    recursive: bool,
    json: bool,
) -> Result<(), CliError> {
    let sandbox = load_sandbox_handle(client, &sandbox_id, "mkdir").await?;
    sandbox.files().mkdir(&path, recursive).await?;
    if json {
        print_json(&serde_json::json!({
            "path": path,
            "recursive": recursive,
            "status": "created",
        }))
    } else {
        println!("Created directory {}", path);
        Ok(())
    }
}

pub async fn handle_stat_sandbox_file(
    client: &BasilicaClient,
    sandbox_id: String,
    path: String,
    json: bool,
) -> Result<(), CliError> {
    let sandbox = load_sandbox_handle(client, &sandbox_id, "stat").await?;
    let response = sandbox.files().stat(&path).await?;
    if json {
        print_json(&response)
    } else {
        println!("Path: {}", path);
        println!("  Size: {}", response.size);
        println!("  File: {}", response.is_file);
        println!("  Dir:  {}", response.is_dir);
        if let Some(modified) = response.modified {
            println!("  Modified: {}", modified);
        }
        Ok(())
    }
}

pub async fn handle_snapshot_create_sandbox(
    client: &BasilicaClient,
    sandbox_id: String,
    path: Option<String>,
    json: bool,
) -> Result<(), CliError> {
    let sandbox = load_sandbox_handle(client, &sandbox_id, "snap-create").await?;
    let response = sandbox.snapshot_create(path).await?;
    if json {
        print_json(&response)
    } else {
        println!("Snapshot created: {}", response.snapshot_id);
        println!("  Status: {}", response.status);
        if let Some(size) = response.archive_size_bytes {
            println!("  Archive Size Bytes: {}", size);
        }
        Ok(())
    }
}

pub async fn handle_snapshot_upload_sandbox(
    client: &BasilicaClient,
    sandbox_id: String,
    snapshot_id: String,
    presigned_url: String,
    json: bool,
) -> Result<(), CliError> {
    let sandbox = load_sandbox_handle(client, &sandbox_id, "snap-upload").await?;
    let response = sandbox.snapshot_upload(snapshot_id.clone(), presigned_url).await?;
    if json {
        print_json(&response)
    } else {
        println!("Snapshot uploaded: {}", snapshot_id);
        println!("  Status: {}", response.status);
        println!("  Bytes Uploaded: {}", response.bytes_uploaded);
        Ok(())
    }
}

pub async fn handle_snapshot_status_sandbox(
    client: &BasilicaClient,
    sandbox_id: String,
    json: bool,
) -> Result<(), CliError> {
    let sandbox = load_sandbox_handle(client, &sandbox_id, "snap-status").await?;
    let response = sandbox.snapshot_status().await?;
    if json {
        print_json(&response)
    } else {
        println!("Snapshot Status: {}", response.status);
        if let Some(snapshot_id) = response.snapshot_id {
            println!("  Snapshot ID: {}", snapshot_id);
        }
        if let Some(message) = response.message {
            println!("  Message: {}", message);
        }
        if let Some(size) = response.archive_size_bytes {
            println!("  Archive Size Bytes: {}", size);
        }
        if let Some(bytes_uploaded) = response.bytes_uploaded {
            println!("  Bytes Uploaded: {}", bytes_uploaded);
        }
        if let Some(bytes_downloaded) = response.bytes_downloaded {
            println!("  Bytes Downloaded: {}", bytes_downloaded);
        }
        Ok(())
    }
}

pub async fn handle_snapshot_restore_sandbox(
    client: &BasilicaClient,
    sandbox_id: String,
    snapshot_id: String,
    presigned_url: String,
    path: Option<String>,
    json: bool,
) -> Result<(), CliError> {
    let sandbox = load_sandbox_handle(client, &sandbox_id, "snap-restore").await?;
    let response = sandbox
        .snapshot_restore(snapshot_id.clone(), presigned_url, path)
        .await?;
    if json {
        print_json(&response)
    } else {
        println!("Snapshot restored: {}", snapshot_id);
        println!("  Status: {}", response.status);
        println!("  Restored Path: {}", response.restored_path);
        println!("  Bytes Downloaded: {}", response.bytes_downloaded);
        Ok(())
    }
}
