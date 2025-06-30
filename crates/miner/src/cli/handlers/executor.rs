//! # Enhanced Executor Management Commands
//!
//! Provides advanced executor management operations including
//! restart, log viewing, direct connection, and diagnostics.

use anyhow::{anyhow, Result};
use std::process::Command;
use tokio::io::{AsyncBufReadExt, BufReader};
use tokio::process::Command as AsyncCommand;
use tracing::{error, info};

use crate::config::{MinerConfig, SshConnectionConfig as SshConfig};
use crate::persistence::RegistrationDb;

/// Enhanced executor operation types
#[derive(Debug, Clone)]
pub enum ExecutorOperation {
    Restart {
        executor_id: String,
    },
    Logs {
        executor_id: String,
        follow: bool,
        lines: Option<usize>,
    },
    Connect {
        executor_id: String,
    },
    Diagnostics {
        executor_id: String,
    },
    Ping {
        executor_id: String,
    },
}

/// Executor connection information
#[derive(Debug)]
pub struct ExecutorConnection {
    pub executor_id: String,
    pub grpc_address: String,
    pub is_available: bool,
    pub last_seen: Option<chrono::DateTime<chrono::Utc>>,
}

/// Handle enhanced executor management commands
pub async fn handle_enhanced_executor_command(
    operation: ExecutorOperation,
    config: &MinerConfig,
    db: RegistrationDb,
) -> Result<()> {
    match operation {
        ExecutorOperation::Restart { executor_id } => {
            restart_executor(&executor_id, config, db).await
        }
        ExecutorOperation::Logs {
            executor_id,
            follow,
            lines,
        } => view_executor_logs(&executor_id, config, follow, lines).await,
        ExecutorOperation::Connect { executor_id } => {
            connect_to_executor(&executor_id, config).await
        }
        ExecutorOperation::Diagnostics { executor_id } => {
            run_executor_diagnostics(&executor_id, config, db).await
        }
        ExecutorOperation::Ping { executor_id } => ping_executor(&executor_id, config).await,
    }
}

/// Restart a specific executor
async fn restart_executor(
    executor_id: &str,
    config: &MinerConfig,
    db: RegistrationDb,
) -> Result<()> {
    info!("Restarting executor: {}", executor_id);
    println!("🔄 Restarting executor: {executor_id}");

    // Find executor configuration
    let executor_config = config
        .executor_management
        .executors
        .iter()
        .find(|e| e.id == executor_id)
        .ok_or_else(|| anyhow!("Executor not found: {}", executor_id))?;

    // Check current health status
    let health_records = db.get_all_executor_health().await?;
    let current_health = health_records.iter().find(|h| h.executor_id == executor_id);

    if let Some(health) = current_health {
        println!(
            "   Current status: {}",
            if health.is_healthy {
                "Healthy"
            } else {
                "Unhealthy"
            }
        );
        if let Some(error) = &health.last_error {
            println!("   Last error: {error}");
        }
    }

    // Attempt to restart via different methods
    let restart_result = if is_remote_executor(&executor_config.grpc_address) {
        restart_remote_executor(executor_id, &executor_config.grpc_address).await
    } else {
        restart_local_executor(executor_id).await
    };

    match restart_result {
        Ok(_) => {
            println!("✅ Executor restart command sent successfully");

            // Wait a moment and check if it comes back online
            println!("   Waiting for executor to come back online...");
            tokio::time::sleep(tokio::time::Duration::from_secs(10)).await;

            // Check health again
            if let Ok(new_health) = check_executor_health(&executor_config.grpc_address).await {
                if new_health {
                    println!("✅ Executor is back online and healthy");
                } else {
                    println!("⚠️  Executor restart completed but health check failed");
                }
            } else {
                println!("⚠️  Could not verify executor health after restart");
            }
        }
        Err(e) => {
            error!("Failed to restart executor: {}", e);
            println!("❌ Failed to restart executor: {e}");
            return Err(e);
        }
    }

    Ok(())
}

/// View executor logs
async fn view_executor_logs(
    executor_id: &str,
    config: &MinerConfig,
    follow: bool,
    lines: Option<usize>,
) -> Result<()> {
    info!(
        "Viewing logs for executor: {} (follow: {})",
        executor_id, follow
    );
    println!("📋 Viewing logs for executor: {executor_id}");

    // Find executor configuration
    let executor_config = config
        .executor_management
        .executors
        .iter()
        .find(|e| e.id == executor_id)
        .ok_or_else(|| anyhow!("Executor not found: {}", executor_id))?;

    if is_remote_executor(&executor_config.grpc_address) {
        // For remote executors, try to get logs via SSH or systemd
        view_remote_executor_logs(executor_id, &executor_config.grpc_address, follow, lines).await
    } else {
        // For local executors, use systemd or direct log files
        view_local_executor_logs(executor_id, follow, lines).await
    }
}

/// Connect directly to an executor
async fn connect_to_executor(executor_id: &str, config: &MinerConfig) -> Result<()> {
    info!("Connecting to executor: {}", executor_id);
    println!("🔗 Connecting to executor: {executor_id}");

    // Find executor configuration
    let executor_config = config
        .executor_management
        .executors
        .iter()
        .find(|e| e.id == executor_id)
        .ok_or_else(|| anyhow!("Executor not found: {}", executor_id))?;

    println!("   Executor address: {}", executor_config.grpc_address);

    // Test gRPC connection
    println!("   Testing gRPC connection...");
    match test_grpc_connection(&executor_config.grpc_address).await {
        Ok(_) => {
            println!("✅ gRPC connection successful");
        }
        Err(e) => {
            println!("❌ gRPC connection failed: {e}");
        }
    }

    // If it's a remote executor, provide SSH connection info
    if is_remote_executor(&executor_config.grpc_address) {
        let host = extract_host_from_address(&executor_config.grpc_address)?;
        println!("\n📡 Remote Executor Connection Options:");
        println!("   SSH: ssh user@{host}");
        println!(
            "   gRPC: grpcurl -plaintext {} health.v1.Health/Check",
            executor_config.grpc_address
        );

        // Offer to establish SSH connection if possible
        if let Some(ssh_config) = get_ssh_config_for_host(&host, config) {
            println!("\n🔐 SSH Connection Available:");
            println!(
                "   Command: ssh -i {} {}@{}",
                ssh_config.private_key_path.display(),
                ssh_config.username,
                ssh_config.host
            );

            // Ask user if they want to connect
            println!("\nPress Enter to establish SSH connection, or Ctrl+C to cancel...");
            let mut input = String::new();
            std::io::stdin().read_line(&mut input).ok();

            // Execute SSH connection
            let ssh_result = establish_ssh_connection(&ssh_config).await;
            match ssh_result {
                Ok(_) => println!("✅ SSH connection established"),
                Err(e) => println!("❌ SSH connection failed: {e}"),
            }
        }
    } else {
        println!("\n💻 Local Executor Connection:");
        println!(
            "   gRPC: grpcurl -plaintext {} health.v1.Health/Check",
            executor_config.grpc_address
        );
        println!("   Logs: journalctl -u basilica-executor-{executor_id} -f");
    }

    Ok(())
}

/// Run comprehensive diagnostics on an executor
async fn run_executor_diagnostics(
    executor_id: &str,
    config: &MinerConfig,
    db: RegistrationDb,
) -> Result<()> {
    info!("Running diagnostics for executor: {}", executor_id);
    println!("🔍 Running diagnostics for executor: {executor_id}");

    // Find executor configuration
    let executor_config = config
        .executor_management
        .executors
        .iter()
        .find(|e| e.id == executor_id)
        .ok_or_else(|| anyhow!("Executor not found: {}", executor_id))?;

    println!("\n=== Executor Diagnostics Report ===");
    println!("Executor ID: {executor_id}");
    println!("Address: {}", executor_config.grpc_address);
    println!(
        "Type: {}",
        if is_remote_executor(&executor_config.grpc_address) {
            "Remote"
        } else {
            "Local"
        }
    );

    // Check database health records
    println!("\n--- Database Health Records ---");
    let health_records = db.get_all_executor_health().await?;
    if let Some(health) = health_records.iter().find(|h| h.executor_id == executor_id) {
        println!(
            "Status: {}",
            if health.is_healthy {
                "✅ Healthy"
            } else {
                "❌ Unhealthy"
            }
        );
        println!("Consecutive Failures: {}", health.consecutive_failures);
        if let Some(last_check) = health.last_health_check {
            println!(
                "Last Health Check: {}",
                last_check.format("%Y-%m-%d %H:%M:%S UTC")
            );
        }
        if let Some(error) = &health.last_error {
            println!("Last Error: {error}");
        }
    } else {
        println!("⚠️  No health records found");
    }

    // Test network connectivity
    println!("\n--- Network Connectivity ---");
    match test_network_connectivity(&executor_config.grpc_address).await {
        Ok(latency) => {
            println!("✅ Network reachable (latency: {latency}ms)");
        }
        Err(e) => {
            println!("❌ Network unreachable: {e}");
        }
    }

    // Test gRPC health
    println!("\n--- gRPC Health Check ---");
    match test_grpc_health_check(&executor_config.grpc_address).await {
        Ok(status) => {
            println!("✅ gRPC health check passed: {status}");
        }
        Err(e) => {
            println!("❌ gRPC health check failed: {e}");
        }
    }

    // Test executor-specific endpoints
    println!("\n--- Executor Service Check ---");
    match test_executor_endpoints(&executor_config.grpc_address).await {
        Ok(results) => {
            for (endpoint, success) in results {
                println!("{} {}", if success { "✅" } else { "❌" }, endpoint);
            }
        }
        Err(e) => {
            println!("❌ Executor endpoint tests failed: {e}");
        }
    }

    // If remote, check SSH connectivity
    if is_remote_executor(&executor_config.grpc_address) {
        println!("\n--- SSH Connectivity ---");
        let host = extract_host_from_address(&executor_config.grpc_address)?;
        if let Some(ssh_config) = get_ssh_config_for_host(&host, config) {
            match test_ssh_connectivity(&ssh_config).await {
                Ok(_) => {
                    println!("✅ SSH connection successful");
                }
                Err(e) => {
                    println!("❌ SSH connection failed: {e}");
                }
            }
        } else {
            println!("⚠️  No SSH configuration found");
        }
    }

    println!("\n=== Diagnostics Complete ===");
    Ok(())
}

/// Ping an executor to test basic connectivity
async fn ping_executor(executor_id: &str, config: &MinerConfig) -> Result<()> {
    println!("🏓 Pinging executor: {executor_id}");

    // Find executor configuration
    let executor_config = config
        .executor_management
        .executors
        .iter()
        .find(|e| e.id == executor_id)
        .ok_or_else(|| anyhow!("Executor not found: {}", executor_id))?;

    let start_time = std::time::Instant::now();

    match test_grpc_connection(&executor_config.grpc_address).await {
        Ok(_) => {
            let duration = start_time.elapsed();
            println!("✅ Ping successful: {}ms", duration.as_millis());
        }
        Err(e) => {
            let duration = start_time.elapsed();
            println!("❌ Ping failed after {}ms: {}", duration.as_millis(), e);
            return Err(e);
        }
    }

    Ok(())
}

/// Check if executor is remote based on its address
fn is_remote_executor(address: &str) -> bool {
    !address.starts_with("127.0.0.1")
        && !address.starts_with("localhost")
        && !address.starts_with("0.0.0.0")
}

/// Restart remote executor via gRPC or SSH
async fn restart_remote_executor(executor_id: &str, address: &str) -> Result<()> {
    // Try gRPC restart first (if the executor supports it)
    if send_grpc_restart_command(address).await.is_ok() {
        return Ok(());
    }

    // Fallback to SSH-based restart
    let host = extract_host_from_address(address)?;
    restart_executor_via_ssh(&host, executor_id).await
}

/// Restart local executor via systemd
async fn restart_local_executor(executor_id: &str) -> Result<()> {
    let service_name = format!("basilica-executor-{executor_id}");

    let output = Command::new("systemctl")
        .args(["restart", &service_name])
        .output()
        .map_err(|e| anyhow!("Failed to execute systemctl: {}", e))?;

    if output.status.success() {
        Ok(())
    } else {
        let error_msg = String::from_utf8_lossy(&output.stderr);
        Err(anyhow!("Failed to restart executor service: {}", error_msg))
    }
}

/// View logs for remote executor
async fn view_remote_executor_logs(
    executor_id: &str,
    address: &str,
    follow: bool,
    lines: Option<usize>,
) -> Result<()> {
    let host = extract_host_from_address(address)?;
    let service_name = format!("basilica-executor-{executor_id}");

    let mut ssh_args: Vec<String> = vec![
        host,
        "journalctl".to_string(),
        "-u".to_string(),
        service_name.clone(),
        "--no-pager".to_string(),
    ];

    if let Some(n) = lines {
        ssh_args.extend(["-n".to_string(), n.to_string()]);
    }

    if follow {
        ssh_args.push("-f".to_string());
    }

    let mut child = AsyncCommand::new("ssh")
        .args(ssh_args.iter().map(|s| s.as_str()))
        .stdout(std::process::Stdio::piped())
        .spawn()
        .map_err(|e| anyhow!("Failed to start SSH process: {}", e))?;

    if let Some(stdout) = child.stdout.take() {
        let reader = BufReader::new(stdout);
        let mut lines = reader.lines();

        while let Some(line) = lines.next_line().await? {
            println!("{line}");
        }
    }

    Ok(())
}

/// View logs for local executor
async fn view_local_executor_logs(
    executor_id: &str,
    follow: bool,
    lines: Option<usize>,
) -> Result<()> {
    let service_name = format!("basilica-executor-{executor_id}");

    let mut args: Vec<String> = vec![
        "journalctl".to_string(),
        "-u".to_string(),
        service_name.clone(),
        "--no-pager".to_string(),
    ];

    if let Some(n) = lines {
        args.extend(["-n".to_string(), n.to_string()]);
    }

    if follow {
        args.push("-f".to_string());
    }

    let mut child = AsyncCommand::new("sudo")
        .args(args.iter().map(|s| s.as_str()))
        .stdout(std::process::Stdio::piped())
        .spawn()
        .map_err(|e| anyhow!("Failed to start journalctl: {}", e))?;

    if let Some(stdout) = child.stdout.take() {
        let reader = BufReader::new(stdout);
        let mut lines = reader.lines();

        while let Some(line) = lines.next_line().await? {
            println!("{line}");
        }
    }

    Ok(())
}

/// Test gRPC connection to executor
async fn test_grpc_connection(address: &str) -> Result<()> {
    // Simple TCP connection test
    let socket_addr = address
        .parse::<std::net::SocketAddr>()
        .map_err(|e| anyhow!("Invalid address format: {}", e))?;

    match tokio::net::TcpStream::connect(socket_addr).await {
        Ok(_) => Ok(()),
        Err(e) => Err(anyhow!("TCP connection failed: {}", e)),
    }
}

/// Test gRPC health check
async fn test_grpc_health_check(address: &str) -> Result<String> {
    // Use grpcurl to test health check
    let output = Command::new("grpcurl")
        .args(["-plaintext", address, "health.v1.Health/Check"])
        .output()
        .map_err(|e| anyhow!("Failed to execute grpcurl: {}", e))?;

    if output.status.success() {
        Ok(String::from_utf8_lossy(&output.stdout).to_string())
    } else {
        Err(anyhow!(
            "Health check failed: {}",
            String::from_utf8_lossy(&output.stderr)
        ))
    }
}

/// Test network connectivity and measure latency
async fn test_network_connectivity(address: &str) -> Result<u64> {
    let host = extract_host_from_address(address)?;
    let start_time = std::time::Instant::now();

    let output = Command::new("ping")
        .args(["-c", "1", "-W", "5", &host])
        .output()
        .map_err(|e| anyhow!("Failed to execute ping: {}", e))?;

    let duration = start_time.elapsed();

    if output.status.success() {
        Ok(duration.as_millis() as u64)
    } else {
        Err(anyhow!("Ping failed"))
    }
}

/// Test executor-specific endpoints
async fn test_executor_endpoints(address: &str) -> Result<Vec<(String, bool)>> {
    let endpoints = vec![
        ("health.v1.Health/Check", "health check"),
        (
            "basilica.executor.v1.ExecutorControl/GetSystemProfile",
            "system profile",
        ),
        (
            "basilica.executor.v1.ExecutorControl/ListContainers",
            "list containers",
        ),
    ];

    let mut results = Vec::new();

    for (endpoint, description) in endpoints {
        let success = Command::new("grpcurl")
            .args(["-plaintext", address, endpoint])
            .output()
            .map(|output| output.status.success())
            .unwrap_or(false);

        results.push((description.to_string(), success));
    }

    Ok(results)
}

/// Check executor health
async fn check_executor_health(address: &str) -> Result<bool> {
    test_grpc_health_check(address).await.map(|_| true)
}

/// Extract host from gRPC address
fn extract_host_from_address(address: &str) -> Result<String> {
    let parts: Vec<&str> = address.split(':').collect();
    if !parts.is_empty() {
        Ok(parts[0].to_string())
    } else {
        Err(anyhow!("Invalid address format: {}", address))
    }
}

/// Get SSH configuration for a host
fn get_ssh_config_for_host(host: &str, config: &MinerConfig) -> Option<SshConfig> {
    config
        .remote_executor_deployment
        .as_ref()?
        .remote_machines
        .iter()
        .find(|machine| machine.ssh.host == host)
        .map(|machine| machine.ssh.clone())
}

/// Establish SSH connection
async fn establish_ssh_connection(ssh_config: &SshConfig) -> Result<()> {
    let mut ssh_args = vec![format!("{}@{}", ssh_config.username, ssh_config.host)];

    let key_path = ssh_config.private_key_path.to_string_lossy().to_string();
    ssh_args.extend(["-i".to_string(), key_path]);

    if ssh_config.port != 22 {
        let port_str = ssh_config.port.to_string();
        ssh_args.extend(["-p".to_string(), port_str]);
    }

    let mut child = AsyncCommand::new("ssh")
        .args(&ssh_args)
        .spawn()
        .map_err(|e| anyhow!("Failed to start SSH: {}", e))?;

    let status = child
        .wait()
        .await
        .map_err(|e| anyhow!("SSH process error: {}", e))?;

    if status.success() {
        Ok(())
    } else {
        Err(anyhow!("SSH connection failed"))
    }
}

/// Test SSH connectivity
async fn test_ssh_connectivity(ssh_config: &SshConfig) -> Result<()> {
    let mut ssh_args: Vec<String> = vec![
        "-o".to_string(),
        "ConnectTimeout=10".to_string(),
        "-o".to_string(),
        "BatchMode=yes".to_string(),
        format!("{}@{}", ssh_config.username, ssh_config.host),
        "echo".to_string(),
        "SSH test successful".to_string(),
    ];

    ssh_args.insert(0, ssh_config.private_key_path.to_string_lossy().to_string());
    ssh_args.insert(0, "-i".to_string());

    if ssh_config.port != 22 {
        ssh_args.insert(0, ssh_config.port.to_string());
        ssh_args.insert(0, "-p".to_string());
    }

    let output = Command::new("ssh")
        .args(ssh_args.iter().map(|s| s.as_str()))
        .output()
        .map_err(|e| anyhow!("Failed to execute SSH test: {}", e))?;

    if output.status.success() {
        Ok(())
    } else {
        Err(anyhow!(
            "SSH test failed: {}",
            String::from_utf8_lossy(&output.stderr)
        ))
    }
}

/// Send gRPC restart command to executor
async fn send_grpc_restart_command(_address: &str) -> Result<()> {
    // This would require implementing a restart endpoint in the executor
    // For now, return an error to indicate it's not supported
    Err(anyhow!("gRPC restart not implemented"))
}

/// Restart executor via SSH
async fn restart_executor_via_ssh(host: &str, executor_id: &str) -> Result<()> {
    let service_name = format!("basilica-executor-{executor_id}");
    let ssh_command = format!("sudo systemctl restart {service_name}");

    let output = Command::new("ssh")
        .args([host, &ssh_command])
        .output()
        .map_err(|e| anyhow!("Failed to execute SSH restart: {}", e))?;

    if output.status.success() {
        Ok(())
    } else {
        Err(anyhow!(
            "SSH restart failed: {}",
            String::from_utf8_lossy(&output.stderr)
        ))
    }
}
