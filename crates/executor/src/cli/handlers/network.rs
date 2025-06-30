use super::HandlerUtils;
use crate::cli::{commands::NetworkCommands, CliContext};
use anyhow::Result;

pub async fn handle_network_command(cmd: &NetworkCommands, context: &CliContext) -> Result<()> {
    match cmd {
        NetworkCommands::Status => show_status(context).await,
        NetworkCommands::Test { host } => test_connectivity(host.as_deref(), context).await,
        NetworkCommands::Grpc => show_grpc_status(context).await,
        NetworkCommands::Metrics => show_metrics_status(context).await,
        NetworkCommands::SpeedTest => run_speed_test(context).await,
    }
}

async fn show_status(context: &CliContext) -> Result<()> {
    HandlerUtils::print_info("Checking network status...");

    let config = HandlerUtils::load_config(&context.config_path)?;
    let state = HandlerUtils::init_executor_state(config).await?;

    let system_info = state.system_monitor.get_system_info().await?;

    println!("Network Status:");
    println!("  Interfaces: {}", system_info.network.interfaces.len());
    println!(
        "  Total Bytes Sent: {} MB",
        system_info.network.total_bytes_sent / (1024 * 1024)
    );
    println!(
        "  Total Bytes Received: {} MB",
        system_info.network.total_bytes_received / (1024 * 1024)
    );

    println!("\nInterface Details:");
    for interface in &system_info.network.interfaces {
        println!("  {}:", interface.name);
        println!(
            "    Status: {}",
            if interface.is_up { "UP" } else { "DOWN" }
        );
        println!(
            "    Bytes Sent: {} MB",
            interface.bytes_sent / (1024 * 1024)
        );
        println!(
            "    Bytes Received: {} MB",
            interface.bytes_received / (1024 * 1024)
        );
        println!("    Packets Sent: {}", interface.packets_sent);
        println!("    Packets Received: {}", interface.packets_received);

        if interface.errors_sent > 0 || interface.errors_received > 0 {
            HandlerUtils::print_warning(&format!(
                "  Errors - Sent: {}, Received: {}",
                interface.errors_sent, interface.errors_received
            ));
        }
    }

    HandlerUtils::print_success("Network status retrieved");
    Ok(())
}

async fn test_connectivity(host: Option<&str>, context: &CliContext) -> Result<()> {
    let target = host.unwrap_or("8.8.8.8");
    HandlerUtils::print_info(&format!("Testing connectivity to: {target}"));

    let config = HandlerUtils::load_config(&context.config_path)?;
    let _state = HandlerUtils::init_executor_state(config).await?;

    match std::process::Command::new("ping")
        .args(["-c", "3", target])
        .output()
    {
        Ok(output) => {
            if output.status.success() {
                let stdout = String::from_utf8_lossy(&output.stdout);
                println!("Ping Results:");
                for line in stdout.lines() {
                    if line.contains("time=") || line.contains("packet loss") {
                        println!("  {}", line.trim());
                    }
                }
                HandlerUtils::print_success(&format!("Connectivity to {target} is working"));
            } else {
                let stderr = String::from_utf8_lossy(&output.stderr);
                HandlerUtils::print_error(&format!("Ping failed: {stderr}"));
            }
        }
        Err(e) => {
            HandlerUtils::print_error(&format!("Failed to run ping: {e}"));
        }
    }

    HandlerUtils::print_info("Testing DNS resolution...");
    match std::process::Command::new("nslookup").arg(target).output() {
        Ok(output) => {
            if output.status.success() {
                HandlerUtils::print_success("DNS resolution working");
            } else {
                HandlerUtils::print_warning("DNS resolution issues detected");
            }
        }
        Err(_) => {
            HandlerUtils::print_info("nslookup not available, skipping DNS test");
        }
    }

    Ok(())
}

async fn show_grpc_status(context: &CliContext) -> Result<()> {
    HandlerUtils::print_info("Checking gRPC server status...");

    let config = HandlerUtils::load_config(&context.config_path)?;

    println!("gRPC Server Configuration:");
    println!("  Host: {}", config.server.host);
    println!("  Port: {}", config.server.port);
    println!("  Max Connections: {}", config.server.max_connections);
    println!("  TLS Enabled: {}", config.server.tls_enabled);
    println!(
        "  Request Timeout: {}s",
        config.server.request_timeout.as_secs()
    );
    println!(
        "  Keep Alive Timeout: {}s",
        config.server.keep_alive_timeout.as_secs()
    );

    let addr = format!("{}:{}", config.server.host, config.server.port);
    match std::net::TcpStream::connect_timeout(&addr.parse()?, std::time::Duration::from_secs(5)) {
        Ok(_) => {
            HandlerUtils::print_success(&format!("gRPC server is accessible on {addr}"));
        }
        Err(e) => {
            HandlerUtils::print_warning(&format!("gRPC server not accessible: {e}"));
            HandlerUtils::print_info("This is normal if the server is not currently running");
        }
    }

    Ok(())
}

async fn show_metrics_status(context: &CliContext) -> Result<()> {
    HandlerUtils::print_info("Checking metrics endpoint status...");

    let config = HandlerUtils::load_config(&context.config_path)?;

    println!("Metrics Configuration:");
    println!("  Enabled: {}", config.metrics.enabled);

    if config.metrics.enabled {
        if let Some(prometheus) = &config.metrics.prometheus {
            println!("  Host: {}", prometheus.host);
            println!("  Port: {}", prometheus.port);
            println!("  Path: {}", prometheus.path);
            println!(
                "  Collection Interval: {}s",
                config.metrics.collection_interval.as_secs()
            );

            let metrics_url = format!(
                "http://{}:{}{}",
                prometheus.host, prometheus.port, prometheus.path
            );

            // TODO: Implement metrics endpoint check using reqwest
            HandlerUtils::print_info(&format!("Metrics URL: {metrics_url}"));
            HandlerUtils::print_warning("Metrics endpoint check not implemented yet");
        } else {
            HandlerUtils::print_warning("Prometheus configuration not found");
        }
    } else {
        HandlerUtils::print_info("Metrics collection is disabled");
    }

    Ok(())
}

async fn run_speed_test(context: &CliContext) -> Result<()> {
    HandlerUtils::print_info("Running network speed test...");

    let config = HandlerUtils::load_config(&context.config_path)?;
    let _state = HandlerUtils::init_executor_state(config).await?;

    // TODO: Implement actual network speed test using system monitor capability
    HandlerUtils::print_warning("Speed test not implemented yet");

    Ok(())
}
