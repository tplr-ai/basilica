use super::HandlerUtils;
use crate::service::ValidatorService;

use anyhow::Result;
use std::path::PathBuf;
use std::time::{Duration, SystemTime};

pub async fn handle_start(config_path: PathBuf) -> Result<()> {
    HandlerUtils::print_info("Starting Basilica Validator...");

    let config = HandlerUtils::load_config(config_path)?;

    HandlerUtils::validate_config(&config)?;

    let service = ValidatorService::new(config);
    service.start().await
}

pub async fn handle_stop() -> Result<()> {
    HandlerUtils::print_info("Stopping Basilica Validator...");

    let start_time = SystemTime::now();

    ValidatorService::stop().await?;

    let elapsed = start_time.elapsed().unwrap_or(Duration::from_secs(0));
    HandlerUtils::print_success(&format!("Shutdown completed in {}ms", elapsed.as_millis()));

    Ok(())
}

pub async fn handle_status(config_path: PathBuf) -> Result<()> {
    println!("=== Basilica Validator Status ===");
    println!("Version: {}", env!("CARGO_PKG_VERSION"));

    let start_time = SystemTime::now();

    // Load config to show actual configuration being used
    let config = HandlerUtils::load_config(config_path)?;

    println!("\nConfiguration:");
    println!("  Wallet: {}", config.bittensor.common.wallet_name);
    println!("  Hotkey: {}", config.bittensor.common.hotkey_name);
    println!("  Network: {}", config.bittensor.common.network);
    println!("  NetUID: {}", config.bittensor.common.netuid);

    // Get service status
    let service = ValidatorService::new(config);
    let status = service.status().await?;

    // Display process status
    println!("\nProcess Status:");
    match status.process {
        Some((pid, memory_mb, cpu_percent)) => {
            println!(
                "  Validator process running (PID: {pid}, Memory: {memory_mb}MB, CPU: {cpu_percent:.1}%)"
            );
        }
        None => {
            println!("  ERROR: No validator process found");
        }
    }

    // Display database status
    println!("\nDatabase Status:");
    if status.database_healthy {
        println!("  SQLite database connection successful");
    } else {
        println!("  ERROR: Database connection failed");
    }

    // Display API server status
    println!("\nAPI Server Status:");
    if let Some(response_time_ms) = status.api_response_time {
        println!("  API server healthy (response time: {response_time_ms}ms)");
    } else {
        println!("  ERROR: API server check failed");
    }

    // Display Bittensor network status
    println!("\nBittensor Network Status:");
    if let Some(block_number) = status.bittensor_block {
        println!("  Bittensor network connected (block: {block_number})");
    } else {
        println!("  ERROR: Bittensor network check failed");
    }

    // Display overall health summary
    let elapsed = start_time.elapsed().unwrap_or(Duration::from_secs(0));
    println!("\nOverall Status:");
    if status.is_healthy() {
        println!("  All systems operational");
    } else {
        println!("  ERROR: Some components have issues");
    }
    println!("  Status check completed in {}ms", elapsed.as_millis());

    if !status.is_healthy() {
        std::process::exit(1);
    }

    Ok(())
}

pub async fn handle_gen_config(output: PathBuf) -> Result<()> {
    let config = crate::config::ValidatorConfig::default();
    let toml_content = toml::to_string_pretty(&config)?;
    std::fs::write(&output, toml_content)?;
    HandlerUtils::print_success(&format!(
        "Generated configuration file: {}",
        output.display()
    ));
    Ok(())
}
