//! # Service Management Commands
//!
//! Handles service lifecycle operations for the miner including
//! start, stop, restart, status, and reload operations.

use anyhow::{anyhow, Result};
use std::process::Command;
use tracing::{error, info, warn};

use crate::config::MinerConfig;
use common::config::ConfigValidation;

/// Service management operations
#[derive(Debug, Clone)]
pub enum ServiceOperation {
    Start,
    Stop,
    Restart,
    Status,
    Reload,
}

/// Service status information
#[derive(Debug)]
pub struct ServiceStatus {
    pub is_running: bool,
    pub pid: Option<u32>,
    pub uptime: Option<String>,
    pub memory_usage: Option<String>,
    pub status_text: String,
}

/// Handle service management commands
pub async fn handle_service_command(
    operation: ServiceOperation,
    config: &MinerConfig,
) -> Result<()> {
    match operation {
        ServiceOperation::Start => start_service(config).await,
        ServiceOperation::Stop => stop_service().await,
        ServiceOperation::Restart => restart_service(config).await,
        ServiceOperation::Status => show_service_status().await,
        ServiceOperation::Reload => reload_service_config(config).await,
    }
}

/// Start the miner service
async fn start_service(config: &MinerConfig) -> Result<()> {
    info!("Starting miner service...");

    // Check if service is already running
    let status = get_service_status().await?;
    if status.is_running {
        warn!("Miner service is already running (PID: {:?})", status.pid);
        println!("✅ Miner service is already running");
        return Ok(());
    }

    // For systemd-based systems
    if is_systemd_available() {
        let output = Command::new("systemctl")
            .args(["start", "basilica-miner"])
            .output()
            .map_err(|e| anyhow!("Failed to execute systemctl: {}", e))?;

        if output.status.success() {
            println!("✅ Miner service started successfully");
            info!("Miner service started via systemctl");
        } else {
            let error_msg = String::from_utf8_lossy(&output.stderr);
            error!("Failed to start service: {}", error_msg);
            return Err(anyhow!("Failed to start service: {}", error_msg));
        }
    } else {
        // Fallback for non-systemd systems
        warn!("Systemd not available, manual service management required");
        println!("⚠️  Manual service management required on this system");
        println!(
            "   Run: ./target/release/miner --config {}",
            config.database.url
        );
    }

    Ok(())
}

/// Stop the miner service
async fn stop_service() -> Result<()> {
    info!("Stopping miner service...");

    // Check if service is running
    let status = get_service_status().await?;
    if !status.is_running {
        warn!("Miner service is not currently running");
        println!("⚠️  Miner service is not currently running");
        return Ok(());
    }

    // For systemd-based systems
    if is_systemd_available() {
        let output = Command::new("systemctl")
            .args(["stop", "basilica-miner"])
            .output()
            .map_err(|e| anyhow!("Failed to execute systemctl: {}", e))?;

        if output.status.success() {
            println!("✅ Miner service stopped successfully");
            info!("Miner service stopped via systemctl");
        } else {
            let error_msg = String::from_utf8_lossy(&output.stderr);
            error!("Failed to stop service: {}", error_msg);
            return Err(anyhow!("Failed to stop service: {}", error_msg));
        }
    } else {
        // Fallback for non-systemd systems
        if let Some(pid) = status.pid {
            let output = Command::new("kill")
                .args(["-TERM", &pid.to_string()])
                .output()
                .map_err(|e| anyhow!("Failed to kill process: {}", e))?;

            if output.status.success() {
                println!("✅ Miner service stopped successfully (PID: {pid})");
                info!("Miner service stopped via kill signal");
            } else {
                let error_msg = String::from_utf8_lossy(&output.stderr);
                error!("Failed to stop process: {}", error_msg);
                return Err(anyhow!("Failed to stop process: {}", error_msg));
            }
        } else {
            return Err(anyhow!("Could not determine process ID to stop"));
        }
    }

    Ok(())
}

/// Restart the miner service
async fn restart_service(config: &MinerConfig) -> Result<()> {
    info!("Restarting miner service...");
    println!("🔄 Restarting miner service...");

    // Stop the service first
    stop_service().await?;

    // Wait a moment for graceful shutdown
    tokio::time::sleep(tokio::time::Duration::from_secs(3)).await;

    // Start the service again
    start_service(config).await?;

    println!("✅ Miner service restarted successfully");
    Ok(())
}

/// Show service status
async fn show_service_status() -> Result<()> {
    let status = get_service_status().await?;

    println!("=== Miner Service Status ===");
    println!(
        "Running: {}",
        if status.is_running {
            "✅ Yes"
        } else {
            "❌ No"
        }
    );

    if let Some(pid) = status.pid {
        println!("PID: {pid}");
    }

    if let Some(uptime) = &status.uptime {
        println!("Uptime: {uptime}");
    }

    if let Some(memory) = &status.memory_usage {
        println!("Memory Usage: {memory}");
    }

    println!("Status: {}", status.status_text);

    // Show recent logs if available
    if status.is_running && is_systemd_available() {
        println!("\n=== Recent Logs ===");
        let log_output = Command::new("journalctl")
            .args(["-u", "basilica-miner", "--no-pager", "-n", "5"])
            .output();

        if let Ok(output) = log_output {
            let logs = String::from_utf8_lossy(&output.stdout);
            println!("{logs}");
        }
    }

    Ok(())
}

/// Reload service configuration
async fn reload_service_config(config: &MinerConfig) -> Result<()> {
    info!("Reloading miner service configuration...");
    println!("🔄 Reloading miner service configuration...");

    // Validate configuration first
    if let Err(e) = config.validate() {
        error!("Configuration validation failed: {}", e);
        return Err(anyhow!("Configuration validation failed: {}", e));
    }

    // Check if service is running
    let status = get_service_status().await?;
    if !status.is_running {
        warn!("Miner service is not running, cannot reload configuration");
        println!("⚠️  Miner service is not running");
        return Ok(());
    }

    // For systemd-based systems, send reload signal
    if is_systemd_available() {
        let output = Command::new("systemctl")
            .args(["reload-or-restart", "basilica-miner"])
            .output()
            .map_err(|e| anyhow!("Failed to execute systemctl: {}", e))?;

        if output.status.success() {
            println!("✅ Configuration reloaded successfully");
            info!("Miner service configuration reloaded");
        } else {
            let error_msg = String::from_utf8_lossy(&output.stderr);
            error!("Failed to reload configuration: {}", error_msg);
            return Err(anyhow!("Failed to reload configuration: {}", error_msg));
        }
    } else {
        // Fallback: send HUP signal to process
        if let Some(pid) = status.pid {
            let output = Command::new("kill")
                .args(["-HUP", &pid.to_string()])
                .output()
                .map_err(|e| anyhow!("Failed to send reload signal: {}", e))?;

            if output.status.success() {
                println!("✅ Reload signal sent successfully (PID: {pid})");
                info!("Reload signal sent to miner process");
            } else {
                let error_msg = String::from_utf8_lossy(&output.stderr);
                error!("Failed to send reload signal: {}", error_msg);
                return Err(anyhow!("Failed to send reload signal: {}", error_msg));
            }
        } else {
            return Err(anyhow!("Could not determine process ID for reload"));
        }
    }

    Ok(())
}

/// Get current service status
async fn get_service_status() -> Result<ServiceStatus> {
    // Try systemd first
    if is_systemd_available() {
        return get_systemd_status().await;
    }

    // Fallback to process detection
    get_process_status().await
}

/// Get service status from systemd
async fn get_systemd_status() -> Result<ServiceStatus> {
    let output = Command::new("systemctl")
        .args(["status", "basilica-miner", "--no-pager"])
        .output()
        .map_err(|e| anyhow!("Failed to execute systemctl: {}", e))?;

    let status_text = String::from_utf8_lossy(&output.stdout);
    let is_running = output.status.success() && status_text.contains("active (running)");

    // Extract PID if available
    let pid = if is_running {
        extract_pid_from_systemctl_output(&status_text)
    } else {
        None
    };

    // Extract uptime if available
    let uptime = if is_running {
        extract_uptime_from_systemctl_output(&status_text)
    } else {
        None
    };

    Ok(ServiceStatus {
        is_running,
        pid,
        uptime,
        memory_usage: None, // Would need additional commands for memory info
        status_text: status_text.lines().take(3).collect::<Vec<_>>().join("\n"),
    })
}

/// Get service status by process detection
async fn get_process_status() -> Result<ServiceStatus> {
    let output = Command::new("pgrep")
        .args(["-f", "basilica.*miner"])
        .output()
        .map_err(|e| anyhow!("Failed to execute pgrep: {}", e))?;

    let is_running = output.status.success();
    let pid = if is_running {
        String::from_utf8_lossy(&output.stdout)
            .trim()
            .parse::<u32>()
            .ok()
    } else {
        None
    };

    let status_text = if is_running {
        "Process running".to_string()
    } else {
        "Process not found".to_string()
    };

    Ok(ServiceStatus {
        is_running,
        pid,
        uptime: None,
        memory_usage: None,
        status_text,
    })
}

/// Check if systemd is available
fn is_systemd_available() -> bool {
    Command::new("systemctl")
        .args(["--version"])
        .output()
        .map(|output| output.status.success())
        .unwrap_or(false)
}

/// Extract PID from systemctl output
fn extract_pid_from_systemctl_output(output: &str) -> Option<u32> {
    output
        .lines()
        .find(|line| line.contains("Main PID:"))
        .and_then(|line| {
            line.split_whitespace()
                .find(|word| word.chars().all(|c| c.is_ascii_digit()))
                .and_then(|pid_str| pid_str.parse().ok())
        })
}

/// Extract uptime from systemctl output
fn extract_uptime_from_systemctl_output(output: &str) -> Option<String> {
    output
        .lines()
        .find(|line| line.contains("Active:"))
        .and_then(|line| {
            line.find("since ")
                .map(|start| line[start + 6..].trim().to_string())
        })
}
