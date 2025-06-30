//! Container management command handlers

use super::HandlerUtils;
use crate::cli::{commands::ContainerCommands, CliContext};
use crate::config::ContainerResourceLimits;
use anyhow::Result;

pub async fn handle_container_command(cmd: &ContainerCommands, context: &CliContext) -> Result<()> {
    match cmd {
        ContainerCommands::Create {
            image,
            command,
            memory,
            cpu,
            stream,
        } => {
            create_container(
                image,
                command,
                memory.as_ref(),
                cpu.as_ref(),
                *stream,
                context,
            )
            .await
        }
        ContainerCommands::Exec {
            container_id,
            command,
            timeout,
        } => exec_command(container_id, command, *timeout, context).await,
        ContainerCommands::Remove {
            container_id,
            force,
        } => remove_container(container_id, *force, context).await,
        ContainerCommands::List => list_containers(context).await,
        ContainerCommands::Logs {
            container_id,
            follow,
            tail,
        } => show_logs(container_id, *follow, *tail, context).await,
        ContainerCommands::Status { container_id } => show_status(container_id, context).await,
        ContainerCommands::Cleanup => cleanup_containers(context).await,
    }
}

async fn create_container(
    image: &str,
    command: &str,
    memory: Option<&u64>,
    cpu: Option<&f64>,
    stream: bool,
    context: &CliContext,
) -> Result<()> {
    HandlerUtils::print_info(&format!("Creating container with image: {image}"));

    let config = HandlerUtils::load_config(&context.config_path)?;
    let state = HandlerUtils::init_executor_state(config).await?;

    let limits = if memory.is_some() || cpu.is_some() {
        Some(ContainerResourceLimits {
            memory_bytes: memory
                .map(|m| m * 1024 * 1024)
                .unwrap_or(8 * 1024 * 1024 * 1024),
            cpu_cores: cpu.copied().unwrap_or(4.0),
            gpu_memory_bytes: None,
            disk_io_bps: None,
            network_bps: None,
        })
    } else {
        None
    };

    // Parse command string into shell arguments
    let command_args = vec!["/bin/sh".to_string(), "-c".to_string(), command.to_string()];

    let container_id = state
        .container_manager
        .create_container(image, &command_args, limits)
        .await?;

    if stream {
        // Stream output in real-time
        use futures_util::StreamExt;

        let log_stream = state
            .container_manager
            .stream_logs(&container_id, true, None)
            .await?;

        tokio::pin!(log_stream);

        while let Some(log_entry) = log_stream.next().await {
            // Extract just the message content, removing Docker timestamp if present
            let message = if let Some(pos) = log_entry.message.find(' ') {
                // Skip timestamp and get the actual message
                let potential_msg = &log_entry.message[pos + 1..];
                if potential_msg.ends_with('\n') {
                    potential_msg.to_string()
                } else {
                    format!("{potential_msg}\n")
                }
            } else {
                log_entry.message
            };

            print!("{message}");
            use std::io::Write;
            std::io::stdout().flush().unwrap();
        }
    } else {
        HandlerUtils::print_success(&format!("Container created: {container_id}"));
        println!("Container ID: {container_id}");
    }

    Ok(())
}

async fn exec_command(
    container_id: &str,
    command: &str,
    timeout: Option<u64>,
    context: &CliContext,
) -> Result<()> {
    HandlerUtils::print_info(&format!(
        "Executing command in container {container_id}: {command}"
    ));

    let config = HandlerUtils::load_config(&context.config_path)?;
    let state = HandlerUtils::init_executor_state(config).await?;

    let result = state
        .container_manager
        .execute_command(container_id, command, timeout)
        .await?;

    println!("Exit Code: {}", result.exit_code);
    if !result.stdout.is_empty() {
        println!("STDOUT:\n{}", result.stdout);
    }
    if !result.stderr.is_empty() {
        println!("STDERR:\n{}", result.stderr);
    }

    if result.exit_code == 0 {
        HandlerUtils::print_success(&format!(
            "Command executed successfully ({}ms)",
            result.duration_ms
        ));
    } else {
        HandlerUtils::print_error(&format!(
            "Command failed with exit code: {}",
            result.exit_code
        ));
    }

    Ok(())
}

async fn remove_container(container_id: &str, force: bool, context: &CliContext) -> Result<()> {
    HandlerUtils::print_info(&format!(
        "Removing container: {container_id} (force: {force})"
    ));

    let config = HandlerUtils::load_config(&context.config_path)?;
    let state = HandlerUtils::init_executor_state(config).await?;

    state
        .container_manager
        .destroy_container(container_id, force)
        .await?;
    HandlerUtils::print_success(&format!("Container {container_id} removed"));

    Ok(())
}

async fn list_containers(context: &CliContext) -> Result<()> {
    HandlerUtils::print_info("Listing containers...");

    let config = HandlerUtils::load_config(&context.config_path)?;
    let state = HandlerUtils::init_executor_state(config).await?;

    let containers = state.container_manager.list_containers().await?;

    if containers.is_empty() {
        HandlerUtils::print_info("No containers found");
        return Ok(());
    }

    let rows: Vec<Vec<String>> = containers
        .iter()
        .map(|c| {
            vec![
                c.id[..12].to_string(),
                c.name.clone(),
                c.image.clone(),
                c.state.clone(),
                c.created.to_string(),
            ]
        })
        .collect();

    let table = HandlerUtils::format_table(&["ID", "Name", "Image", "State", "Created"], &rows);

    println!("{table}");
    HandlerUtils::print_success(&format!("Found {} containers", containers.len()));

    Ok(())
}

async fn show_logs(
    container_id: &str,
    follow: bool,
    tail: Option<i32>,
    context: &CliContext,
) -> Result<()> {
    HandlerUtils::print_info(&format!("Showing logs for container: {container_id}"));

    let config = HandlerUtils::load_config(&context.config_path)?;
    let state = HandlerUtils::init_executor_state(config).await?;

    let mut log_stream = state
        .container_manager
        .stream_logs(container_id, follow, tail)
        .await?;

    use futures_util::StreamExt;
    while let Some(log_entry) = log_stream.next().await {
        println!("[{}] {}", log_entry.timestamp, log_entry.message);

        if !follow {
            break;
        }
    }

    if !follow {
        HandlerUtils::print_success("Logs retrieved");
    }

    Ok(())
}

async fn show_status(container_id: &str, context: &CliContext) -> Result<()> {
    HandlerUtils::print_info(&format!("Getting status for container: {container_id}"));

    let config = HandlerUtils::load_config(&context.config_path)?;
    let state = HandlerUtils::init_executor_state(config).await?;

    if let Some(status) = state
        .container_manager
        .get_container_status(container_id)
        .await?
    {
        println!("Container Status:");
        println!("  ID: {}", status.id);
        println!("  Name: {}", status.name);
        println!("  Image: {}", status.image);
        println!("  State: {}", status.state);
        println!("  Status: {}", status.status);
        println!("  Created: {}", status.created);

        if let Some(started) = status.started {
            println!("  Started: {started}");
        }

        if let Some(finished) = status.finished {
            println!("  Finished: {finished}");
        }

        if let Some(exit_code) = status.exit_code {
            println!("  Exit Code: {exit_code}");
        }

        HandlerUtils::print_success("Status retrieved");
    } else {
        HandlerUtils::print_error(&format!("Container not found: {container_id}"));
    }

    Ok(())
}

async fn cleanup_containers(context: &CliContext) -> Result<()> {
    HandlerUtils::print_info("Cleaning up inactive containers...");

    let config = HandlerUtils::load_config(&context.config_path)?;
    let state = HandlerUtils::init_executor_state(config).await?;

    state
        .container_manager
        .cleanup_inactive_containers()
        .await?;
    HandlerUtils::print_success("Container cleanup completed");

    Ok(())
}
