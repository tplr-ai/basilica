//! Sandbox management handlers for the Basilica CLI
//!
//! Implements create, list, status, delete, and WebSocket connect flows
//! for ephemeral compute sandboxes.

use crate::error::CliError;
use crate::output::{json_output, print_info, print_success, print_warning};
use crate::progress::{complete_spinner_and_clear, complete_spinner_error, create_spinner};
use basilica_sdk::types::{
    CreateSandboxRequest, CreateSandboxResponse, SandboxEnvVar, SandboxListResponse, SandboxPhase,
    SandboxResponse, SandboxWsRequest, SandboxWsResponse,
};
use basilica_sdk::BasilicaClient;
use color_eyre::eyre::eyre;
use console::style;
use dialoguer::theme::ColorfulTheme;
use dialoguer::Confirm;
use futures_util::{SinkExt, StreamExt};
use std::time::Duration;
use tokio_tungstenite::tungstenite;

/// Maximum number of status poll attempts before giving up
const MAX_POLL_ATTEMPTS: u32 = 120;

/// Interval between status polls in milliseconds
const POLL_INTERVAL_MS: u64 = 500;

/// Handle sandbox create command
pub async fn handle_create(
    client: &BasilicaClient,
    image: Option<String>,
    cpu: String,
    memory: String,
    ttl: u32,
    env: Vec<String>,
    json: bool,
) -> Result<(), CliError> {
    // Validate TTL
    if ttl < 60 || ttl > 7200 {
        return Err(CliError::Internal(
            eyre!("TTL must be between 60 and 7200 seconds (got {})", ttl)
                .into(),
        ));
    }

    // Parse env vars
    let env_vars = if env.is_empty() {
        None
    } else {
        let parsed: Result<Vec<SandboxEnvVar>, _> = env
            .iter()
            .map(|e| {
                let parts: Vec<&str> = e.splitn(2, '=').collect();
                if parts.len() != 2 {
                    Err(eyre!("Invalid env var format '{}'. Expected KEY=VALUE", e))
                } else {
                    Ok(SandboxEnvVar {
                        name: parts[0].to_string(),
                        value: parts[1].to_string(),
                    })
                }
            })
            .collect();
        Some(parsed.map_err(CliError::Internal)?)
    };

    let request = CreateSandboxRequest {
        image,
        cpu: Some(cpu),
        memory: Some(memory),
        ttl_seconds: Some(ttl),
        env: env_vars,
    };

    let spinner = create_spinner("Creating sandbox...");
    let response: CreateSandboxResponse = client.create_sandbox(request).await.map_err(|e| {
        complete_spinner_error(spinner.clone(), "Failed to create sandbox");
        match &e {
            basilica_sdk::ApiError::ApiResponse { status, message } if *status == 402 => {
                CliError::Internal(
                    eyre!("Insufficient balance: {}", message)
                        .wrap_err("Top up your account with: basilica fund"),
                )
            }
            _ => CliError::Api(e),
        }
    })?;
    complete_spinner_and_clear(spinner);

    if json {
        json_output(&response)?;
        return Ok(());
    }

    print_success(&format!("Sandbox {} created", style(&response.sandbox_id).cyan()));
    println!(
        "  {}: {}",
        style("Domain").dim(),
        style(&response.domain).green()
    );
    println!(
        "  {}: {}",
        style("Phase").dim(),
        format_phase(&response.status.phase)
    );
    println!();
    println!(
        "  {}: {}",
        style("Exec Secret").dim(),
        style(&response.exec_secret).yellow()
    );
    println!();
    print_warning("Save the exec secret now. It will not be shown again.");
    print_info("This is a bearer credential. Do not persist or share it.");
    println!();
    println!(
        "Connect with: {}",
        style(format!(
            "basilica sandbox connect {} --secret <exec_secret>",
            response.sandbox_id
        ))
        .cyan()
    );
    println!();
    print_sandbox_help_text();

    Ok(())
}

/// Handle sandbox list command
pub async fn handle_list(client: &BasilicaClient, json: bool) -> Result<(), CliError> {
    let spinner = create_spinner("Fetching sandboxes...");
    let response: SandboxListResponse = client.list_sandboxes().await.map_err(|e| {
        complete_spinner_error(spinner.clone(), "Failed to list sandboxes");
        CliError::Api(e)
    })?;
    complete_spinner_and_clear(spinner);

    if json {
        json_output(&response)?;
        return Ok(());
    }

    if response.sandboxes.is_empty() {
        println!("No active sandboxes.");
        println!();
        println!(
            "Create one with: {}",
            style("basilica sandbox create").cyan()
        );
        return Ok(());
    }

    println!("{}", style("Active Sandboxes").bold());
    println!(
        "  {:<12} {:<40} {:<12} {}",
        style("ID").dim(),
        style("Domain").dim(),
        style("Phase").dim(),
        style("Started").dim(),
    );

    for sb in &response.sandboxes {
        let started = sb
            .status
            .started_at
            .as_deref()
            .unwrap_or("-");
        println!(
            "  {:<12} {:<40} {:<12} {}",
            style(&sb.sandbox_id).cyan(),
            &sb.domain,
            format_phase(&sb.status.phase),
            started,
        );
    }

    println!();
    println!("Total: {}", response.sandboxes.len());

    Ok(())
}

/// Handle sandbox status command
pub async fn handle_status(
    client: &BasilicaClient,
    sandbox_id: &str,
    json: bool,
) -> Result<(), CliError> {
    let spinner = create_spinner(&format!("Fetching sandbox {}...", sandbox_id));
    let response: SandboxResponse = client.get_sandbox(sandbox_id).await.map_err(|e| {
        complete_spinner_error(spinner.clone(), "Failed to get sandbox");
        CliError::Api(e)
    })?;
    complete_spinner_and_clear(spinner);

    if json {
        json_output(&response)?;
        return Ok(());
    }

    println!("{}", style("Sandbox Status").bold());
    println!("  {}: {}", style("ID").dim(), style(&response.sandbox_id).cyan());
    println!("  {}: {}", style("Domain").dim(), &response.domain);
    println!(
        "  {}: {}",
        style("Phase").dim(),
        format_phase(&response.status.phase)
    );
    if let Some(started) = &response.status.started_at {
        println!("  {}: {}", style("Started").dim(), started);
    }
    for cond in &response.status.conditions {
        println!(
            "  {}: {} = {}",
            style("Condition").dim(),
            cond.condition_type,
            cond.status
        );
    }

    Ok(())
}

/// Handle sandbox delete command
pub async fn handle_delete(
    client: &BasilicaClient,
    sandbox_id: &str,
    skip_confirm: bool,
    json: bool,
) -> Result<(), CliError> {
    if !skip_confirm {
        let id_owned = sandbox_id.to_string();
        let confirmed = tokio::task::spawn_blocking(move || {
            let theme = ColorfulTheme::default();
            Confirm::with_theme(&theme)
                .with_prompt(format!("Delete sandbox '{}'?", id_owned))
                .default(false)
                .interact()
        })
        .await
        .map_err(|e| CliError::Internal(eyre!("Task join error: {}", e)))?
        .map_err(|e| CliError::Internal(e.into()))?;

        if !confirmed {
            println!("Deletion cancelled.");
            return Ok(());
        }
    }

    let spinner = create_spinner(&format!("Deleting sandbox {}...", sandbox_id));
    client.delete_sandbox(sandbox_id).await.map_err(|e| {
        complete_spinner_error(spinner.clone(), "Failed to delete sandbox");
        CliError::Api(e)
    })?;
    complete_spinner_and_clear(spinner);

    if json {
        json_output(&serde_json::json!({
            "success": true,
            "sandbox_id": sandbox_id,
        }))?;
        return Ok(());
    }

    print_success(&format!("Sandbox {} deleted", sandbox_id));

    Ok(())
}

/// Handle sandbox connect command — WebSocket session to exec-agent
pub async fn handle_connect(
    client: &BasilicaClient,
    sandbox_id: &str,
    exec_secret: Option<String>,
    exec_cmd: Option<String>,
    json: bool,
) -> Result<(), CliError> {
    // Get sandbox status to obtain domain
    let spinner = create_spinner(&format!("Resolving sandbox {}...", sandbox_id));
    let sandbox: SandboxResponse = client.get_sandbox(sandbox_id).await.map_err(|e| {
        complete_spinner_error(spinner.clone(), "Failed to get sandbox");
        CliError::Api(e)
    })?;
    complete_spinner_and_clear(spinner);

    // If not yet running, poll until ready
    let domain = if sandbox.status.phase == SandboxPhase::Pending {
        poll_until_running(client, sandbox_id).await?
    } else if sandbox.status.phase == SandboxPhase::Running {
        sandbox.domain.clone()
    } else {
        return Err(CliError::Internal(eyre!(
            "Sandbox {} is in phase '{}'. Cannot connect.",
            sandbox_id,
            sandbox.status.phase
        )));
    };

    // Prompt for secret if not provided
    let secret = match exec_secret {
        Some(s) => s,
        None => {
            let s = tokio::task::spawn_blocking(|| {
                dialoguer::Password::with_theme(&ColorfulTheme::default())
                    .with_prompt("Exec secret")
                    .interact()
            })
            .await
            .map_err(|e| CliError::Internal(eyre!("Task join error: {}", e)))?
            .map_err(|e| CliError::Internal(e.into()))?;
            s
        }
    };

    if let Some(cmd) = exec_cmd {
        // Non-interactive: execute single command
        return exec_single_command(&domain, &secret, &cmd, json).await;
    }

    // Interactive REPL loop
    interactive_session(&domain, &secret).await
}

/// Poll sandbox status until Running or failure
async fn poll_until_running(client: &BasilicaClient, sandbox_id: &str) -> Result<String, CliError> {
    let spinner = create_spinner("Waiting for sandbox to start...");
    for _ in 0..MAX_POLL_ATTEMPTS {
        tokio::time::sleep(Duration::from_millis(POLL_INTERVAL_MS)).await;
        match client.get_sandbox(sandbox_id).await {
            Ok(sb) => match sb.status.phase {
                SandboxPhase::Running => {
                    complete_spinner_and_clear(spinner);
                    print_success("Sandbox is running");
                    return Ok(sb.domain);
                }
                SandboxPhase::Failed => {
                    complete_spinner_and_clear(spinner);
                    return Err(CliError::Internal(eyre!(
                        "Sandbox {} failed to start",
                        sandbox_id
                    )));
                }
                SandboxPhase::Terminating => {
                    complete_spinner_and_clear(spinner);
                    return Err(CliError::Internal(eyre!(
                        "Sandbox {} is terminating",
                        sandbox_id
                    )));
                }
                SandboxPhase::Pending => continue,
            },
            Err(e) => {
                // Transient errors during polling are tolerable
                tracing::debug!("Poll error: {}", e);
                continue;
            }
        }
    }
    complete_spinner_and_clear(spinner);
    Err(CliError::Internal(eyre!(
        "Timed out waiting for sandbox {} to start ({}s)",
        sandbox_id,
        MAX_POLL_ATTEMPTS as u64 * POLL_INTERVAL_MS / 1000
    )))
}

/// Execute a single command on a sandbox via WebSocket
async fn exec_single_command(
    domain: &str,
    secret: &str,
    command: &str,
    json: bool,
) -> Result<(), CliError> {
    let (mut ws, _) = connect_ws(domain, secret).await?;

    let op_id = uuid::Uuid::new_v4().to_string();
    let request = SandboxWsRequest {
        id: op_id.clone(),
        op: "exec".to_string(),
        args: Some(serde_json::json!({
            "command": command.split_whitespace().collect::<Vec<&str>>()
        })),
    };

    let msg = serde_json::to_string(&request)
        .map_err(|e| CliError::Internal(eyre!("Failed to serialize request: {}", e)))?;

    ws.send(tungstenite::Message::Text(msg))
        .await
        .map_err(|e| CliError::Internal(eyre!("WebSocket send error: {}", e)))?;

    let mut exit_code: Option<i32> = None;
    let mut stdout_buf = String::new();
    let mut stderr_buf = String::new();

    // Read frames until we get an exit or error for this op
    while let Some(frame) = ws.next().await {
        let frame = frame.map_err(|e| CliError::Internal(eyre!("WebSocket read error: {}", e)))?;

        match frame {
            tungstenite::Message::Text(text) => {
                let resp: SandboxWsResponse = serde_json::from_str(&text)
                    .map_err(|e| CliError::Internal(eyre!("Invalid response frame: {}", e)))?;

                if resp.id != op_id {
                    continue;
                }

                match resp.response_type.as_str() {
                    "stdout" => {
                        if let Some(data) = &resp.data {
                            let s = data.as_str().unwrap_or("");
                            if json {
                                stdout_buf.push_str(s);
                            } else {
                                print!("{}", s);
                            }
                        }
                    }
                    "stderr" => {
                        if let Some(data) = &resp.data {
                            let s = data.as_str().unwrap_or("");
                            if json {
                                stderr_buf.push_str(s);
                            } else {
                                eprint!("{}", s);
                            }
                        }
                    }
                    "exit" => {
                        exit_code = resp.code;
                        break;
                    }
                    "error" => {
                        let msg = resp.error.unwrap_or_else(|| "Unknown error".to_string());
                        let code = resp.error_code.unwrap_or_default();
                        return Err(CliError::Internal(eyre!(
                            "exec-agent error [{}]: {}",
                            code,
                            msg
                        )));
                    }
                    _ => {}
                }
            }
            tungstenite::Message::Close(_) => break,
            _ => {}
        }
    }

    // Close WebSocket
    let _ = ws.close(None).await;

    if json {
        json_output(&serde_json::json!({
            "stdout": stdout_buf,
            "stderr": stderr_buf,
            "exit_code": exit_code,
        }))?;
    }

    if let Some(code) = exit_code {
        if code != 0 {
            std::process::exit(code);
        }
    }

    Ok(())
}

/// Interactive sandbox session — read commands, send exec ops, display output
async fn interactive_session(domain: &str, secret: &str) -> Result<(), CliError> {
    let (mut ws, _) = connect_ws(domain, secret).await?;

    println!(
        "{} Connected to {}",
        style("●").green(),
        style(domain).cyan()
    );
    println!(
        "{}",
        style("Type commands to execute. Ctrl+C or 'exit' to disconnect.").dim()
    );
    println!(
        "{}",
        style("File ops: :read <path>, :write <path> <content>, :ls [path], :stat <path>, :mv <old> <new>").dim()
    );
    println!();

    let stdin = tokio::io::stdin();
    let reader = tokio::io::BufReader::new(stdin);
    let mut lines = tokio::io::AsyncBufReadExt::lines(reader);

    loop {
        // Print prompt
        eprint!("{} ", style("sandbox>").green().bold());

        let line = match lines.next_line().await {
            Ok(Some(line)) => line,
            Ok(None) => break, // EOF
            Err(_) => break,
        };

        let trimmed = line.trim();
        if trimmed.is_empty() {
            continue;
        }
        if trimmed == "exit" || trimmed == "quit" {
            break;
        }

        // Check for file op shortcuts
        let (op, args) = parse_interactive_command(trimmed);

        let op_id = uuid::Uuid::new_v4().to_string();
        let request = SandboxWsRequest {
            id: op_id.clone(),
            op,
            args: Some(args),
        };

        let msg = serde_json::to_string(&request)
            .map_err(|e| CliError::Internal(eyre!("Failed to serialize: {}", e)))?;

        if ws.send(tungstenite::Message::Text(msg)).await.is_err() {
            eprintln!("{} Connection lost. Attempting reconnect...", style("●").red());
            match connect_ws(domain, secret).await {
                Ok((new_ws, _)) => {
                    ws = new_ws;
                    println!("{} Reconnected", style("●").green());
                    continue;
                }
                Err(e) => {
                    eprintln!("Reconnect failed: {}", e);
                    eprintln!("Create a new sandbox with: basilica sandbox create");
                    return Err(e);
                }
            }
        }

        // Read response frames for this op
        let done = read_op_response(&mut ws, &op_id).await?;
        if done {
            break;
        }
    }

    let _ = ws.close(None).await;
    println!("Disconnected.");
    Ok(())
}

/// Parse interactive command into (op, args) pair
fn parse_interactive_command(input: &str) -> (String, serde_json::Value) {
    if let Some(rest) = input.strip_prefix(":read ") {
        return (
            "read_file".to_string(),
            serde_json::json!({ "path": rest.trim() }),
        );
    }
    if let Some(rest) = input.strip_prefix(":write ") {
        let parts: Vec<&str> = rest.splitn(2, ' ').collect();
        if parts.len() == 2 {
            return (
                "write_file".to_string(),
                serde_json::json!({ "path": parts[0], "content": parts[1] }),
            );
        }
        return (
            "write_file".to_string(),
            serde_json::json!({ "path": rest.trim(), "content": "" }),
        );
    }
    if let Some(rest) = input.strip_prefix(":ls") {
        let path = rest.trim();
        let path = if path.is_empty() { "." } else { path };
        return (
            "list_dir".to_string(),
            serde_json::json!({ "path": path }),
        );
    }
    if let Some(rest) = input.strip_prefix(":stat ") {
        return (
            "stat".to_string(),
            serde_json::json!({ "path": rest.trim() }),
        );
    }
    if let Some(rest) = input.strip_prefix(":mkdir ") {
        return (
            "mkdir".to_string(),
            serde_json::json!({ "path": rest.trim() }),
        );
    }
    if let Some(rest) = input.strip_prefix(":rm ") {
        return (
            "remove".to_string(),
            serde_json::json!({ "path": rest.trim() }),
        );
    }
    if let Some(rest) = input.strip_prefix(":upload ") {
        let parts: Vec<&str> = rest.splitn(2, ' ').collect();
        if parts.len() == 2 {
            return (
                "upload_r2".to_string(),
                serde_json::json!({ "local_path": parts[0], "key": parts[1] }),
            );
        }
    }
    if let Some(rest) = input.strip_prefix(":download ") {
        let parts: Vec<&str> = rest.splitn(2, ' ').collect();
        if parts.len() == 2 {
            return (
                "download_r2".to_string(),
                serde_json::json!({ "key": parts[0], "local_path": parts[1] }),
            );
        }
    }
    if let Some(rest) = input.strip_prefix(":rename ") {
        let parts: Vec<&str> = rest.splitn(2, ' ').collect();
        if parts.len() == 2 {
            return (
                "rename".to_string(),
                serde_json::json!({ "old_path": parts[0], "new_path": parts[1] }),
            );
        }
    }
    if let Some(rest) = input.strip_prefix(":mv ") {
        let parts: Vec<&str> = rest.splitn(2, ' ').collect();
        if parts.len() == 2 {
            return (
                "rename".to_string(),
                serde_json::json!({ "old_path": parts[0], "new_path": parts[1] }),
            );
        }
    }

    // Default: exec command
    (
        "exec".to_string(),
        serde_json::json!({ "command": input.split_whitespace().collect::<Vec<&str>>() }),
    )
}

/// Read all response frames for a given op ID. Returns true if connection closed.
async fn read_op_response(
    ws: &mut WsStream,
    op_id: &str,
) -> Result<bool, CliError> {
    loop {
        let frame = match tokio::time::timeout(Duration::from_secs(30), ws.next()).await {
            Ok(Some(Ok(frame))) => frame,
            Ok(Some(Err(e))) => {
                return Err(CliError::Internal(eyre!("WebSocket error: {}", e)));
            }
            Ok(None) => return Ok(true), // stream ended
            Err(_) => {
                // Timeout — for exec ops this is fine, just means long-running command
                continue;
            }
        };

        match frame {
            tungstenite::Message::Text(text) => {
                let resp: SandboxWsResponse = match serde_json::from_str(&text) {
                    Ok(r) => r,
                    Err(_) => continue,
                };

                if resp.id != op_id {
                    continue;
                }

                match resp.response_type.as_str() {
                    "stdout" => {
                        if let Some(data) = &resp.data {
                            print!("{}", data.as_str().unwrap_or(""));
                        }
                    }
                    "stderr" => {
                        if let Some(data) = &resp.data {
                            eprint!("{}", data.as_str().unwrap_or(""));
                        }
                    }
                    "exit" => {
                        if let Some(code) = resp.code {
                            if code != 0 {
                                eprintln!(
                                    "{}",
                                    style(format!("(exit code: {})", code)).dim()
                                );
                            }
                        }
                        return Ok(false);
                    }
                    "ok" => return Ok(false),
                    "file" => {
                        if let Some(data) = &resp.data {
                            println!("{}", serde_json::to_string_pretty(data).unwrap_or_default());
                        }
                        return Ok(false);
                    }
                    "stat" => {
                        if let Some(data) = &resp.data {
                            println!("{}", serde_json::to_string_pretty(data).unwrap_or_default());
                        }
                        return Ok(false);
                    }
                    "dir" => {
                        if let Some(data) = &resp.data {
                            if let Some(entries) = data.as_array() {
                                for entry in entries {
                                    println!("{}", entry.as_str().unwrap_or(""));
                                }
                            } else {
                                println!(
                                    "{}",
                                    serde_json::to_string_pretty(data).unwrap_or_default()
                                );
                            }
                        }
                        return Ok(false);
                    }
                    "pong" => return Ok(false),
                    "error" => {
                        let msg = resp.error.unwrap_or_else(|| "Unknown error".to_string());
                        let code = resp.error_code.unwrap_or_default();
                        eprintln!(
                            "{} [{}] {}",
                            style("error").red().bold(),
                            code,
                            msg
                        );
                        return Ok(false);
                    }
                    other => {
                        eprintln!("{}: {}", style("unknown response type").dim(), other);
                        return Ok(false);
                    }
                }
            }
            tungstenite::Message::Close(_) => return Ok(true),
            tungstenite::Message::Ping(data) => {
                let _ = ws.send(tungstenite::Message::Pong(data)).await;
            }
            _ => {}
        }
    }
}

/// WebSocket stream type alias for exec-agent connections
type WsStream = tokio_tungstenite::WebSocketStream<
    tokio_tungstenite::MaybeTlsStream<tokio::net::TcpStream>,
>;

/// Connect to exec-agent WebSocket endpoint
async fn connect_ws(
    domain: &str,
    secret: &str,
) -> Result<
    (
        tokio_tungstenite::WebSocketStream<
            tokio_tungstenite::MaybeTlsStream<tokio::net::TcpStream>,
        >,
        tungstenite::http::Response<Option<Vec<u8>>>,
    ),
    CliError,
> {
    let url = format!("wss://{}/ws", domain);

    let request = tungstenite::http::Request::builder()
        .uri(&url)
        .header("X-Exec-Secret", secret)
        .header("Host", domain)
        .header("Connection", "Upgrade")
        .header("Upgrade", "websocket")
        .header("Sec-WebSocket-Version", "13")
        .header(
            "Sec-WebSocket-Key",
            tungstenite::handshake::client::generate_key(),
        )
        .body(())
        .map_err(|e| CliError::Internal(eyre!("Failed to build WS request: {}", e)))?;

    tokio_tungstenite::connect_async(request)
        .await
        .map_err(|e| {
            CliError::Internal(eyre!(
                "WebSocket connection to {} failed: {}.\n\
                 If the sandbox pod is gone, create a new one with: basilica sandbox create",
                domain,
                e
            ))
        })
}

/// Format sandbox phase with color
fn format_phase(phase: &SandboxPhase) -> String {
    match phase {
        SandboxPhase::Pending => style("Pending").yellow().to_string(),
        SandboxPhase::Running => style("Running").green().to_string(),
        SandboxPhase::Terminating => style("Terminating").red().to_string(),
        SandboxPhase::Failed => style("Failed").red().bold().to_string(),
    }
}

/// Print help text about sandbox lifecycle
fn print_sandbox_help_text() {
    println!("{}", style("Sandbox Info:").bold().dim());
    println!(
        "  {} Sandboxes auto-terminate after the configured TTL (default: 30 min, max: 2 hrs).",
        style("·").dim()
    );
    println!(
        "  {} Idle sandboxes are terminated after 120s of no WebSocket activity.",
        style("·").dim()
    );
    println!(
        "  {} Files are ephemeral — stored on the container filesystem only.",
        style("·").dim()
    );
    println!(
        "  {} Use :upload / :download in a session to persist files to R2 object storage.",
        style("·").dim()
    );
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_parse_exec_command() {
        let (op, args) = parse_interactive_command("echo hello world");
        assert_eq!(op, "exec");
        assert_eq!(
            args,
            serde_json::json!({"command": ["echo", "hello", "world"]})
        );
    }

    #[test]
    fn test_parse_read_file() {
        let (op, args) = parse_interactive_command(":read /workspace/main.py");
        assert_eq!(op, "read_file");
        assert_eq!(args, serde_json::json!({"path": "/workspace/main.py"}));
    }

    #[test]
    fn test_parse_write_file() {
        let (op, args) = parse_interactive_command(":write /tmp/test.txt hello world");
        assert_eq!(op, "write_file");
        assert_eq!(
            args,
            serde_json::json!({"path": "/tmp/test.txt", "content": "hello world"})
        );
    }

    #[test]
    fn test_parse_list_dir_default() {
        let (op, args) = parse_interactive_command(":ls");
        assert_eq!(op, "list_dir");
        assert_eq!(args, serde_json::json!({"path": "."}));
    }

    #[test]
    fn test_parse_list_dir_path() {
        let (op, args) = parse_interactive_command(":ls /workspace");
        assert_eq!(op, "list_dir");
        assert_eq!(args, serde_json::json!({"path": "/workspace"}));
    }

    #[test]
    fn test_parse_stat() {
        let (op, args) = parse_interactive_command(":stat /workspace/file.py");
        assert_eq!(op, "stat");
        assert_eq!(args, serde_json::json!({"path": "/workspace/file.py"}));
    }

    #[test]
    fn test_parse_mkdir() {
        let (op, args) = parse_interactive_command(":mkdir /workspace/subdir");
        assert_eq!(op, "mkdir");
        assert_eq!(args, serde_json::json!({"path": "/workspace/subdir"}));
    }

    #[test]
    fn test_parse_rm() {
        let (op, args) = parse_interactive_command(":rm /tmp/junk");
        assert_eq!(op, "remove");
        assert_eq!(args, serde_json::json!({"path": "/tmp/junk"}));
    }

    #[test]
    fn test_parse_upload_r2() {
        let (op, args) = parse_interactive_command(":upload /local/file.tar.gz models/v1.tar.gz");
        assert_eq!(op, "upload_r2");
        assert_eq!(
            args,
            serde_json::json!({"local_path": "/local/file.tar.gz", "key": "models/v1.tar.gz"})
        );
    }

    #[test]
    fn test_parse_download_r2() {
        let (op, args) = parse_interactive_command(":download models/v1.tar.gz /local/file.tar.gz");
        assert_eq!(op, "download_r2");
        assert_eq!(
            args,
            serde_json::json!({"key": "models/v1.tar.gz", "local_path": "/local/file.tar.gz"})
        );
    }

    #[test]
    fn test_format_phase() {
        // Just verify no panics and correct string content
        assert!(format_phase(&SandboxPhase::Running).contains("Running"));
        assert!(format_phase(&SandboxPhase::Pending).contains("Pending"));
        assert!(format_phase(&SandboxPhase::Failed).contains("Failed"));
        assert!(format_phase(&SandboxPhase::Terminating).contains("Terminating"));
    }

    #[test]
    fn test_sandbox_ws_request_serialization() {
        let req = SandboxWsRequest {
            id: "test-123".to_string(),
            op: "exec".to_string(),
            args: Some(serde_json::json!({"command": ["echo", "hi"]})),
        };
        let json = serde_json::to_string(&req).unwrap();
        assert!(json.contains("\"op\":\"exec\""));
        assert!(json.contains("\"id\":\"test-123\""));

        let parsed: SandboxWsRequest = serde_json::from_str(&json).unwrap();
        assert_eq!(parsed.op, "exec");
        assert_eq!(parsed.id, "test-123");
    }

    #[test]
    fn test_sandbox_ws_response_deserialization() {
        let json = r#"{"id":"op-1","type":"stdout","data":"hello\n"}"#;
        let resp: SandboxWsResponse = serde_json::from_str(json).unwrap();
        assert_eq!(resp.id, "op-1");
        assert_eq!(resp.response_type, "stdout");
        assert_eq!(resp.data.unwrap().as_str().unwrap(), "hello\n");

        let json = r#"{"id":"op-1","type":"exit","code":0}"#;
        let resp: SandboxWsResponse = serde_json::from_str(json).unwrap();
        assert_eq!(resp.response_type, "exit");
        assert_eq!(resp.code, Some(0));

        let json = r#"{"id":"op-1","type":"error","error":"file not found","error_code":"NOT_FOUND"}"#;
        let resp: SandboxWsResponse = serde_json::from_str(json).unwrap();
        assert_eq!(resp.response_type, "error");
        assert_eq!(resp.error.unwrap(), "file not found");
        assert_eq!(resp.error_code.unwrap(), "NOT_FOUND");
    }

    #[test]
    fn test_sandbox_phase_display() {
        assert_eq!(SandboxPhase::Pending.to_string(), "Pending");
        assert_eq!(SandboxPhase::Running.to_string(), "Running");
        assert_eq!(SandboxPhase::Terminating.to_string(), "Terminating");
        assert_eq!(SandboxPhase::Failed.to_string(), "Failed");
    }

    #[test]
    fn test_sandbox_phase_serde_roundtrip() {
        for phase in [
            SandboxPhase::Pending,
            SandboxPhase::Running,
            SandboxPhase::Terminating,
            SandboxPhase::Failed,
        ] {
            let json = serde_json::to_string(&phase).unwrap();
            let parsed: SandboxPhase = serde_json::from_str(&json).unwrap();
            assert_eq!(parsed, phase);
        }
    }

    #[test]
    fn test_create_sandbox_request_serialization() {
        let req = CreateSandboxRequest {
            image: Some("registry.basilica.ai/sandbox/python:3.11".to_string()),
            cpu: Some("2".to_string()),
            memory: Some("4Gi".to_string()),
            ttl_seconds: Some(1800),
            env: Some(vec![SandboxEnvVar {
                name: "FOO".to_string(),
                value: "bar".to_string(),
            }]),
        };
        let json = serde_json::to_string(&req).unwrap();
        assert!(json.contains("\"ttl_seconds\":1800"));
        assert!(json.contains("\"cpu\":\"2\""));

        // Minimal request with None fields — they should be omitted
        let req = CreateSandboxRequest {
            image: None,
            cpu: None,
            memory: None,
            ttl_seconds: None,
            env: None,
        };
        let json = serde_json::to_string(&req).unwrap();
        assert!(!json.contains("image"));
        assert!(!json.contains("ttl_seconds"));
    }

    #[test]
    fn test_create_sandbox_response_deserialization() {
        let json = r#"{
            "sandbox_id": "ab12cd34",
            "domain": "sb-ab12cd34.sandboxes.basilica.ai",
            "exec_secret": "deadbeef0123456789abcdef0123456789abcdef0123456789abcdef01234567",
            "status": {
                "phase": "Pending",
                "conditions": []
            }
        }"#;
        let resp: CreateSandboxResponse = serde_json::from_str(json).unwrap();
        assert_eq!(resp.sandbox_id, "ab12cd34");
        assert_eq!(resp.domain, "sb-ab12cd34.sandboxes.basilica.ai");
        assert_eq!(resp.exec_secret.len(), 64);
        assert_eq!(resp.status.phase, SandboxPhase::Pending);
    }

    #[test]
    fn test_sandbox_list_response_deserialization() {
        let json = r#"{
            "sandboxes": [
                {
                    "sandbox_id": "ab12cd34",
                    "domain": "sb-ab12cd34.sandboxes.basilica.ai",
                    "status": {
                        "phase": "Running",
                        "started_at": "2026-04-06T12:00:00Z",
                        "conditions": [
                            {"type": "Ready", "status": "True"}
                        ]
                    }
                }
            ]
        }"#;
        let resp: SandboxListResponse = serde_json::from_str(json).unwrap();
        assert_eq!(resp.sandboxes.len(), 1);
        assert_eq!(resp.sandboxes[0].status.phase, SandboxPhase::Running);
        assert_eq!(
            resp.sandboxes[0].status.started_at.as_deref(),
            Some("2026-04-06T12:00:00Z")
        );
    }

    #[test]
    fn test_parse_rename() {
        let (op, args) = parse_interactive_command(":rename /old/path /new/path");
        assert_eq!(op, "rename");
        assert_eq!(
            args,
            serde_json::json!({"old_path": "/old/path", "new_path": "/new/path"})
        );
    }

    #[test]
    fn test_parse_mv_alias() {
        let (op, args) = parse_interactive_command(":mv /workspace/a.py /workspace/b.py");
        assert_eq!(op, "rename");
        assert_eq!(
            args,
            serde_json::json!({"old_path": "/workspace/a.py", "new_path": "/workspace/b.py"})
        );
    }

    #[test]
    fn test_parse_write_file_no_content() {
        let (op, args) = parse_interactive_command(":write /tmp/empty.txt");
        assert_eq!(op, "write_file");
        assert_eq!(
            args,
            serde_json::json!({"path": "/tmp/empty.txt", "content": ""})
        );
    }

    #[test]
    fn test_parse_single_word_exec() {
        let (op, args) = parse_interactive_command("ls");
        assert_eq!(op, "exec");
        assert_eq!(args, serde_json::json!({"command": ["ls"]}));
    }

    #[test]
    fn test_parse_upload_missing_key_falls_through_to_exec() {
        // :upload with only one arg doesn't match the two-arg pattern, falls to exec
        let (op, _args) = parse_interactive_command(":upload /local/only");
        assert_eq!(op, "exec");
    }

    #[test]
    fn test_parse_download_missing_path_falls_through_to_exec() {
        // :download with only one arg doesn't match the two-arg pattern, falls to exec
        let (op, _args) = parse_interactive_command(":download only-key");
        assert_eq!(op, "exec");
    }

    #[test]
    fn test_ws_request_without_args() {
        let req = SandboxWsRequest {
            id: "ping-1".to_string(),
            op: "ping".to_string(),
            args: None,
        };
        let json = serde_json::to_string(&req).unwrap();
        assert!(!json.contains("args"));

        let parsed: SandboxWsRequest = serde_json::from_str(&json).unwrap();
        assert!(parsed.args.is_none());
        assert_eq!(parsed.op, "ping");
    }

    #[test]
    fn test_ws_response_pong() {
        let json = r#"{"id":"ping-1","type":"pong"}"#;
        let resp: SandboxWsResponse = serde_json::from_str(json).unwrap();
        assert_eq!(resp.response_type, "pong");
        assert!(resp.data.is_none());
        assert!(resp.code.is_none());
        assert!(resp.error.is_none());
    }

    #[test]
    fn test_ws_response_dir_listing() {
        let json = r#"{"id":"op-1","type":"dir","data":["file1.py","file2.py","subdir/"]}"#;
        let resp: SandboxWsResponse = serde_json::from_str(json).unwrap();
        assert_eq!(resp.response_type, "dir");
        let entries = resp.data.unwrap();
        assert_eq!(entries.as_array().unwrap().len(), 3);
    }

    #[test]
    fn test_ws_response_stat() {
        let json = r#"{"id":"op-1","type":"stat","data":{"size":1024,"mtime":"2026-04-06T12:00:00Z","permissions":"0644"}}"#;
        let resp: SandboxWsResponse = serde_json::from_str(json).unwrap();
        assert_eq!(resp.response_type, "stat");
        let data = resp.data.unwrap();
        assert_eq!(data["size"], 1024);
    }

    #[test]
    fn test_sandbox_response_without_started_at() {
        let json = r#"{
            "sandbox_id": "ab12cd34",
            "domain": "sb-ab12cd34.sandboxes.basilica.ai",
            "status": {
                "phase": "Pending",
                "conditions": []
            }
        }"#;
        let resp: SandboxResponse = serde_json::from_str(json).unwrap();
        assert!(resp.status.started_at.is_none());
        assert_eq!(resp.status.phase, SandboxPhase::Pending);
    }

    #[test]
    fn test_sandbox_condition_deserialization() {
        let json = r#"{
            "sandbox_id": "ab12cd34",
            "domain": "sb-ab12cd34.sandboxes.basilica.ai",
            "status": {
                "phase": "Running",
                "started_at": "2026-04-06T12:00:00Z",
                "conditions": [
                    {"type": "Ready", "status": "True", "last_transition_time": "2026-04-06T12:00:01Z"},
                    {"type": "PodScheduled", "status": "True"}
                ]
            }
        }"#;
        let resp: SandboxResponse = serde_json::from_str(json).unwrap();
        assert_eq!(resp.status.conditions.len(), 2);
        assert_eq!(resp.status.conditions[0].condition_type, "Ready");
        assert_eq!(resp.status.conditions[0].status, "True");
        assert!(resp.status.conditions[0].last_transition_time.is_some());
        assert!(resp.status.conditions[1].last_transition_time.is_none());
    }

    #[test]
    fn test_empty_sandbox_list() {
        let json = r#"{"sandboxes": []}"#;
        let resp: SandboxListResponse = serde_json::from_str(json).unwrap();
        assert!(resp.sandboxes.is_empty());
    }

    #[test]
    fn test_all_wire_protocol_ops_parse() {
        // Verify all 12 wire protocol ops are reachable through interactive commands
        let ops: Vec<(&str, &str)> = vec![
            ("echo test", "exec"),
            (":read /path", "read_file"),
            (":write /path data", "write_file"),
            (":stat /path", "stat"),
            (":ls /path", "list_dir"),
            (":mkdir /path", "mkdir"),
            (":rm /path", "remove"),
            (":rename /a /b", "rename"),
            (":upload /a key", "upload_r2"),
            (":download key /a", "download_r2"),
        ];
        for (input, expected_op) in ops {
            let (op, _) = parse_interactive_command(input);
            assert_eq!(op, expected_op, "Input '{}' should produce op '{}'", input, expected_op);
        }
    }

    #[test]
    fn test_ws_response_all_error_codes() {
        let error_codes = [
            "AUTH_FAILED", "NOT_FOUND", "PERMISSION_DENIED", "TIMEOUT",
            "QUEUE_FULL", "IO_ERROR", "PROCESS_ERROR", "INVALID_REQUEST",
        ];
        for code in &error_codes {
            let json = format!(
                r#"{{"id":"op-1","type":"error","error":"test error","error_code":"{}"}}"#,
                code
            );
            let resp: SandboxWsResponse = serde_json::from_str(&json).unwrap();
            assert_eq!(resp.error_code.as_deref(), Some(*code));
        }
    }

    #[test]
    fn test_create_request_with_env_vars() {
        let req = CreateSandboxRequest {
            image: None,
            cpu: None,
            memory: None,
            ttl_seconds: None,
            env: Some(vec![
                SandboxEnvVar { name: "KEY1".to_string(), value: "val1".to_string() },
                SandboxEnvVar { name: "KEY2".to_string(), value: "val=with=equals".to_string() },
            ]),
        };
        let json = serde_json::to_string(&req).unwrap();
        let parsed: CreateSandboxRequest = serde_json::from_str(&json).unwrap();
        let env = parsed.env.unwrap();
        assert_eq!(env.len(), 2);
        assert_eq!(env[1].value, "val=with=equals");
    }

    #[test]
    fn test_exec_secret_not_in_get_response() {
        // GET /v1/sandboxes/{id} response does NOT contain exec_secret
        let json = r#"{
            "sandbox_id": "ab12cd34",
            "domain": "sb-ab12cd34.sandboxes.basilica.ai",
            "status": {"phase": "Running", "conditions": []}
        }"#;
        let resp: SandboxResponse = serde_json::from_str(json).unwrap();
        assert_eq!(resp.sandbox_id, "ab12cd34");
        // SandboxResponse has no exec_secret field — this is by design.
        // Only CreateSandboxResponse includes it.
    }
}
