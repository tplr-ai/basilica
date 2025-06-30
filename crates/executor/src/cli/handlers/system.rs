use super::HandlerUtils;
use crate::cli::{commands::SystemCommands, CliContext};
use anyhow::Result;

pub async fn handle_system_command(cmd: &SystemCommands, context: &CliContext) -> Result<()> {
    match cmd {
        SystemCommands::Status => show_status(context).await,
        SystemCommands::Profile => run_profile(context).await,
        SystemCommands::Resources => show_resources(context).await,
        SystemCommands::Monitor { interval } => monitor_system(*interval, context).await,
    }
}

async fn show_status(context: &CliContext) -> Result<()> {
    HandlerUtils::print_info("Fetching system status...");

    let config = HandlerUtils::load_config(&context.config_path)?;
    let state = HandlerUtils::init_executor_state(config).await?;

    let system_info = state.system_monitor.get_system_info().await?;

    println!("System Status:");
    println!("  Hostname: {}", system_info.system.hostname);
    println!(
        "  OS: {} {}",
        system_info.system.os_name, system_info.system.os_version
    );
    println!("  Uptime: {} seconds", system_info.system.uptime_seconds);
    println!("  CPU Usage: {:.1}%", system_info.cpu.usage_percent);
    println!(
        "  Memory Usage: {:.1}% ({} MB / {} MB)",
        system_info.memory.usage_percent,
        system_info.memory.used_bytes / (1024 * 1024),
        system_info.memory.total_bytes / (1024 * 1024)
    );

    if !system_info.gpu.is_empty() {
        println!("  GPU Count: {}", system_info.gpu.len());
        for (i, gpu) in system_info.gpu.iter().enumerate() {
            println!(
                "    GPU {}: {} ({:.1}% usage)",
                i, gpu.name, gpu.utilization_percent
            );
        }
    }

    HandlerUtils::print_success("System status retrieved");
    Ok(())
}

async fn run_profile(context: &CliContext) -> Result<()> {
    HandlerUtils::print_info("Running system profile...");

    let config = HandlerUtils::load_config(&context.config_path)?;
    let state = HandlerUtils::init_executor_state(config).await?;

    // Run comprehensive system profiling
    tracing::info!("Gathering system information for profiling...");
    let system_info = state.system_monitor.get_system_info().await?;

    let profile_data = serde_json::json!({
        "timestamp": system_info.timestamp,
        "system": system_info.system,
        "cpu": system_info.cpu,
        "memory": system_info.memory,
        "gpu": system_info.gpu,
        "disk": system_info.disk,
        "network": system_info.network
    });

    println!("{}", HandlerUtils::format_json(&profile_data)?);
    HandlerUtils::print_success("System profile completed");

    Ok(())
}

async fn show_resources(context: &CliContext) -> Result<()> {
    HandlerUtils::print_info("Showing resource usage...");

    let config = HandlerUtils::load_config(&context.config_path)?;
    let state = HandlerUtils::init_executor_state(config).await?;

    let system_info = state.system_monitor.get_system_info().await?;

    let rows = vec![
        vec![
            "CPU".to_string(),
            format!("{:.1}%", system_info.cpu.usage_percent),
        ],
        vec![
            "Memory".to_string(),
            format!("{:.1}%", system_info.memory.usage_percent),
        ],
        vec![
            "Disk".to_string(),
            format!("{} disks", system_info.disk.len()),
        ],
        vec![
            "Network".to_string(),
            format!("{} interfaces", system_info.network.interfaces.len()),
        ],
    ];

    let table = HandlerUtils::format_table(&["Resource", "Usage"], &rows);
    println!("{table}");

    Ok(())
}

async fn monitor_system(interval: u64, context: &CliContext) -> Result<()> {
    HandlerUtils::print_info(&format!(
        "Starting system monitor (interval: {interval}s, press Ctrl+C to stop)..."
    ));

    let config = HandlerUtils::load_config(&context.config_path)?;
    let state = HandlerUtils::init_executor_state(config).await?;

    loop {
        let system_info = state.system_monitor.get_system_info().await?;

        print!("\x1B[2J\x1B[1;1H"); // Clear screen
        println!("=== System Monitor ===");
        println!("Time: {}", chrono::Utc::now().format("%Y-%m-%d %H:%M:%S"));
        println!("CPU:    {:.1}%", system_info.cpu.usage_percent);
        println!("Memory: {:.1}%", system_info.memory.usage_percent);

        if !system_info.gpu.is_empty() {
            for (i, gpu) in system_info.gpu.iter().enumerate() {
                println!("GPU {}:  {:.1}%", i, gpu.utilization_percent);
            }
        }

        tokio::time::sleep(tokio::time::Duration::from_secs(interval)).await;
    }
}
