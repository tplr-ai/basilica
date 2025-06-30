use super::HandlerUtils;
use crate::cli::{commands::ServiceCommands, CliContext};
use anyhow::Result;
use common::config::ConfigValidation;

pub async fn handle_service_command(cmd: &ServiceCommands, context: &CliContext) -> Result<()> {
    match cmd {
        ServiceCommands::Start => start_service(context).await,
        ServiceCommands::Stop => stop_service(context).await,
        ServiceCommands::Restart => restart_service(context).await,
        ServiceCommands::Status => show_status(context).await,
        ServiceCommands::Health => run_health_check(context).await,
        ServiceCommands::Reload => reload_config(context).await,
        ServiceCommands::Logs { lines, follow } => show_logs(*lines, *follow, context).await,
    }
}

async fn start_service(context: &CliContext) -> Result<()> {
    HandlerUtils::print_info("Starting Basilica Executor service...");

    let config = HandlerUtils::load_config(&context.config_path)?;

    let addr = format!("{}:{}", config.server.host, config.server.port);
    if std::net::TcpStream::connect_timeout(&addr.parse()?, std::time::Duration::from_secs(2))
        .is_ok()
    {
        HandlerUtils::print_warning("Service appears to be already running");
        return Ok(());
    }

    // TODO: Implement service daemon start
    HandlerUtils::print_warning("Service start not implemented yet");
    Ok(())
}

async fn stop_service(context: &CliContext) -> Result<()> {
    HandlerUtils::print_info("Stopping Basilica Executor service...");

    let config = HandlerUtils::load_config(&context.config_path)?;
    let addr = format!("{}:{}", config.server.host, config.server.port);

    // Check if service is running
    if std::net::TcpStream::connect_timeout(&addr.parse()?, std::time::Duration::from_secs(2))
        .is_err()
    {
        HandlerUtils::print_info("Service is not currently running");
        return Ok(());
    }

    // TODO: Implement service shutdown signal
    HandlerUtils::print_warning("Service stop not implemented yet");
    Ok(())
}

async fn restart_service(context: &CliContext) -> Result<()> {
    HandlerUtils::print_info("Restarting Basilica Executor service...");

    // Stop service
    stop_service(context).await?;

    // Wait a moment
    HandlerUtils::print_info("Waiting for service to stop...");
    tokio::time::sleep(tokio::time::Duration::from_secs(2)).await;

    // Start service
    start_service(context).await?;

    HandlerUtils::print_success("Service restart completed");
    Ok(())
}

async fn show_status(context: &CliContext) -> Result<()> {
    HandlerUtils::print_info("Checking Basilica Executor service status...");

    let config = HandlerUtils::load_config(&context.config_path)?;
    let addr = format!("{}:{}", config.server.host, config.server.port);

    // Check if service is running by attempting connection
    let is_running =
        std::net::TcpStream::connect_timeout(&addr.parse()?, std::time::Duration::from_secs(5))
            .is_ok();

    println!("Service Status:");
    if is_running {
        println!("  Status: Running");
        println!("  Address: {addr}");
        HandlerUtils::print_success("Service is running");
    } else {
        println!("  Status: Stopped");
        println!("  Address: {addr} (not accessible)");
        HandlerUtils::print_info("Service is not running");
    }

    println!("\nConfiguration:");
    println!("  Config File: {}", context.config_path);
    println!("  Host: {}", config.server.host);
    println!("  Port: {}", config.server.port);
    println!("  Max Connections: {}", config.server.max_connections);
    println!("  TLS Enabled: {}", config.server.tls_enabled);

    println!("\nFeatures:");
    println!(
        "  Metrics: {}",
        if config.metrics.enabled {
            "Enabled"
        } else {
            "Disabled"
        }
    );

    Ok(())
}

async fn run_health_check(context: &CliContext) -> Result<()> {
    HandlerUtils::print_info("Running comprehensive health check...");

    let config = HandlerUtils::load_config(&context.config_path)?;

    // Try to initialize executor state to test all components
    match HandlerUtils::init_executor_state(config).await {
        Ok(state) => {
            HandlerUtils::print_info("Testing all components...");

            // Test each component
            let mut all_healthy = true;

            // System monitor health
            match state.system_monitor.health_check().await {
                Ok(_) => println!("  System Monitor: Healthy"),
                Err(e) => {
                    println!("  System Monitor: {e}");
                    all_healthy = false;
                }
            }

            // Container manager health
            match state.container_manager.health_check().await {
                Ok(_) => println!("  Container Manager: Healthy"),
                Err(e) => {
                    println!("  Container Manager: {e}");
                    all_healthy = false;
                }
            }

            // Note: Persistence layer removed

            // Overall health
            if all_healthy {
                HandlerUtils::print_success("All components are healthy");
                println!("\nExecutor ID: {}", state.id);

                // Get system info for additional health data
                if let Ok(system_info) = state.system_monitor.get_system_info().await {
                    println!("System Summary:");
                    println!("  CPU Usage: {:.1}%", system_info.cpu.usage_percent);
                    println!("  Memory Usage: {:.1}%", system_info.memory.usage_percent);
                    println!("  GPU Count: {}", system_info.gpu.len());
                    println!(
                        "  Active Containers: {}",
                        state
                            .container_manager
                            .list_containers()
                            .await
                            .unwrap_or_default()
                            .len()
                    );
                }
            } else {
                HandlerUtils::print_error("Some components failed health checks");
            }
        }
        Err(e) => {
            HandlerUtils::print_error(&format!("Failed to initialize executor: {e}"));
        }
    }

    Ok(())
}

async fn reload_config(context: &CliContext) -> Result<()> {
    HandlerUtils::print_info("Reloading configuration...");

    // Validate new configuration
    match HandlerUtils::load_config(&context.config_path) {
        Ok(config) => {
            // Validate configuration
            if let Err(e) = config.validate() {
                HandlerUtils::print_error(&format!("Configuration validation failed: {e}"));
                return Ok(());
            }

            // Show warnings if any
            let warnings = config.warnings();
            if !warnings.is_empty() {
                HandlerUtils::print_warning("Configuration warnings:");
                for warning in warnings {
                    println!("  - {warning}");
                }
            }

            HandlerUtils::print_success("Configuration validated successfully");
            HandlerUtils::print_info("In a running service, this would trigger a hot reload");
            HandlerUtils::print_warning(
                "Note: Some configuration changes require a service restart",
            );
        }
        Err(e) => {
            HandlerUtils::print_error(&format!("Failed to load configuration: {e}"));
        }
    }

    Ok(())
}

async fn show_logs(lines: u32, follow: bool, context: &CliContext) -> Result<()> {
    let follow_msg = if follow { " (following)" } else { "" };
    HandlerUtils::print_info(&format!(
        "Showing last {lines} service log lines{follow_msg}"
    ));

    let _config = HandlerUtils::load_config(&context.config_path)?;

    // TODO: Implement log file reading from journald or log files
    HandlerUtils::print_warning("Log reading not implemented yet");

    Ok(())
}
