use super::HandlerUtils;
use anyhow::Result;
use bittensor::Service as BittensorService;
use reqwest::Client;
use std::path::PathBuf;
use std::sync::Arc;
use std::time::{Duration, SystemTime};
use sysinfo::{Pid, System};
use tokio::signal;
use tracing::error;

pub async fn handle_start(config_path: Option<PathBuf>, local_test: bool) -> Result<()> {
    HandlerUtils::print_info("Starting Basilica Validator...");

    let config = match config_path.as_ref() {
        Some(path) => HandlerUtils::load_config(Some(path.to_str().unwrap()))?,
        None => HandlerUtils::load_config(None)?,
    };

    HandlerUtils::validate_config(&config)?;

    start_validator_services(config, local_test).await
}

pub async fn handle_stop() -> Result<()> {
    println!("🛑 Stopping Basilica Validator...");

    let start_time = SystemTime::now();

    // 1. Find running validator process(es)
    println!("\n🔍 Finding validator processes...");
    let processes = find_validator_processes()?;

    if processes.is_empty() {
        println!("  ℹ️  No validator processes found");
        return Ok(());
    }

    println!("  📋 Found {} validator process(es)", processes.len());
    for &pid in &processes {
        println!("    - PID: {pid}");
    }

    // 2. Send graceful shutdown signal (SIGTERM)
    println!("\n✋ Sending graceful shutdown signal (SIGTERM)...");
    let mut failed_graceful = Vec::new();

    for &pid in &processes {
        match send_signal_to_process(pid, Signal::Term) {
            Ok(()) => {
                println!("  ✅ SIGTERM sent to PID {pid}");
            }
            Err(e) => {
                println!("  ❌ Failed to send SIGTERM to PID {pid}: {e}");
                failed_graceful.push(pid);
            }
        }
    }

    // 3. Wait for clean shutdown with timeout
    println!("\n⏳ Waiting for graceful shutdown (30 seconds timeout)...");
    let shutdown_timeout = Duration::from_secs(30);
    let shutdown_start = SystemTime::now();

    let mut remaining_processes = processes.clone();

    while !remaining_processes.is_empty()
        && shutdown_start.elapsed().unwrap_or(Duration::from_secs(0)) < shutdown_timeout
    {
        tokio::time::sleep(Duration::from_millis(1000)).await;

        remaining_processes.retain(|&pid| {
            match is_process_running(pid) {
                Ok(true) => true, // Still running
                Ok(false) => {
                    println!("  ✅ Process {pid} shutdown gracefully");
                    false // Remove from list
                }
                Err(_) => {
                    println!("  ⚠️  Unable to check status of process {pid}");
                    false // Assume it's gone
                }
            }
        });
    }

    // 4. Force kill remaining processes if necessary
    if !remaining_processes.is_empty() {
        println!("\n💥 Force killing remaining processes (SIGKILL)...");

        for &pid in &remaining_processes {
            match send_signal_to_process(pid, Signal::Kill) {
                Ok(()) => {
                    println!("  🔥 SIGKILL sent to PID {pid}");

                    // Give it a moment to die
                    tokio::time::sleep(Duration::from_millis(500)).await;

                    match is_process_running(pid) {
                        Ok(false) => println!("  ✅ Process {pid} terminated"),
                        Ok(true) => println!("  ❌ Process {pid} still running after SIGKILL"),
                        Err(e) => {
                            println!("  ⚠️  Cannot verify termination of process {pid}: {e}")
                        }
                    }
                }
                Err(e) => {
                    println!("  ❌ Failed to send SIGKILL to PID {pid}: {e}");
                }
            }
        }
    }

    // 5. Final verification
    println!("\n🔍 Final verification...");
    let final_processes = find_validator_processes()?;

    let elapsed = start_time.elapsed().unwrap_or(Duration::from_secs(0));

    if final_processes.is_empty() {
        println!("  ✅ All validator processes terminated successfully");
        println!("  🕒 Shutdown completed in {}ms", elapsed.as_millis());
    } else {
        println!(
            "  ❌ {} validator process(es) still running:",
            final_processes.len()
        );
        for &pid in &final_processes {
            println!("    - PID: {pid}");
        }
        println!(
            "  🕒 Shutdown attempt completed in {}ms (with warnings)",
            elapsed.as_millis()
        );
        return Err(anyhow::anyhow!("Some processes could not be terminated"));
    }

    Ok(())
}

pub async fn handle_status() -> Result<()> {
    println!("=== Basilica Validator Status ===");
    println!("Version: {}", env!("CARGO_PKG_VERSION"));

    let start_time = SystemTime::now();
    let mut all_healthy = true;

    // 1. Check if validator process is running
    println!("\n🔍 Process Status:");
    match check_validator_process() {
        Ok(Some((pid, memory_mb, cpu_percent))) => {
            println!(
                "  ✅ Validator process running (PID: {pid}, Memory: {memory_mb}MB, CPU: {cpu_percent:.1}%)"
            );
        }
        Ok(None) => {
            println!("  ❌ No validator process found");
            all_healthy = false;
        }
        Err(e) => {
            println!("  ⚠️  Process check failed: {e}");
            all_healthy = false;
        }
    }

    // 2. Test database connectivity
    println!("\n💾 Database Status:");
    match test_database_connectivity().await {
        Ok(()) => {
            println!("  ✅ SQLite database connection successful");
        }
        Err(e) => {
            println!("  ❌ Database connection failed: {e}");
            all_healthy = false;
        }
    }

    // 3. Check API server health
    println!("\n🌐 API Server Status:");
    match test_api_health().await {
        Ok(response_time_ms) => {
            println!("  ✅ API server healthy (response time: {response_time_ms}ms)");
        }
        Err(e) => {
            println!("  ❌ API server check failed: {e}");
            all_healthy = false;
        }
    }

    // 4. Check Bittensor network connection
    println!("\n⛓️  Bittensor Network Status:");
    match test_bittensor_connectivity().await {
        Ok(block_number) => {
            println!("  ✅ Bittensor network connected (block: {block_number})");
        }
        Err(e) => {
            println!("  ❌ Bittensor network check failed: {e}");
            all_healthy = false;
        }
    }

    // 5. Display overall health summary
    let elapsed = start_time.elapsed().unwrap_or(Duration::from_secs(0));
    println!("\n📊 Overall Status:");
    if all_healthy {
        println!("  ✅ All systems operational");
    } else {
        println!("  ❌ Some components have issues");
    }
    println!("  🕒 Status check completed in {}ms", elapsed.as_millis());

    if !all_healthy {
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

async fn start_validator_services(
    config: crate::config::ValidatorConfig,
    local_test: bool,
) -> Result<()> {
    // Initialize metrics system if enabled
    let validator_metrics = if config.metrics.enabled {
        let metrics = crate::metrics::ValidatorMetrics::new(config.metrics.clone())?;
        metrics.start_server().await?;
        HandlerUtils::print_success("Validator metrics server started");
        Some(metrics)
    } else {
        None
    };

    let storage_path =
        std::path::PathBuf::from(&config.storage.data_dir).join("validator_storage.json");
    let storage = common::MemoryStorage::with_file(storage_path).await?;

    let db_path = std::path::PathBuf::from(&config.storage.data_dir).join("validator.db");
    let persistence = crate::persistence::SimplePersistence::new(
        db_path.to_str().unwrap(),
        config.bittensor.common.hotkey_name.clone(),
    )
    .await?;

    let persistence_arc = Arc::new(persistence);

    if local_test {
        HandlerUtils::print_info("Running in local test mode - Bittensor services disabled");
    }

    let (_bittensor_service, miner_prover_opt, weight_setter_opt) = if !local_test {
        let bittensor_service: Arc<BittensorService> =
            Arc::new(BittensorService::new(config.bittensor.common.clone()).await?);

        // Initialize chain registration and perform startup registration
        let chain_registration = crate::bittensor_core::ChainRegistration::new(
            &config,
            bittensor_service.clone(),
            local_test,
        )
        .await?;

        // Perform one-time startup registration
        chain_registration.register_startup().await?;
        HandlerUtils::print_success("Validator registered on chain with axon endpoint");

        let miner_prover = Some(crate::miner_prover::MinerProver::new(
            config.verification.clone(),
            bittensor_service.clone(),
        ));

        // Initialize weight setter with block-based timing
        let blocks_per_weight_set = 360; // Set weights every ~1 hour (360 blocks * 12s/block)
        let weight_setter = crate::bittensor_core::WeightSetter::new(
            config.bittensor.common.clone(),
            bittensor_service.clone(),
            storage.clone(),
            persistence_arc.clone(),
            config.verification.min_score_threshold,
            blocks_per_weight_set,
        )?;
        let weight_setter_arc = Arc::new(weight_setter);

        let weight_setter_opt = Some(weight_setter_arc);

        (Some(bittensor_service), miner_prover, weight_setter_opt)
    } else {
        (None, None, None)
    };

    let api_handler =
        crate::api::ApiHandler::new(config.api.clone(), persistence_arc.clone(), storage.clone());

    // Store metrics for cleanup (if needed)
    let _validator_metrics = validator_metrics;

    HandlerUtils::print_success("All components initialized successfully");

    // Start scoring update task if weight setter is available
    let scoring_task_handle = weight_setter_opt.as_ref().map(|weight_setter| {
        let weight_setter = weight_setter.clone();
        tokio::spawn(async move {
            let mut interval = tokio::time::interval(Duration::from_secs(300)); // Update scores every 5 minutes
            loop {
                interval.tick().await;
                if let Err(e) = weight_setter.update_all_miner_scores().await {
                    error!("Failed to update miner scores: {}", e);
                }
            }
        })
    });

    let weight_setter_handle = weight_setter_opt.map(|weight_setter| {
        let weight_setter = weight_setter.clone();
        tokio::spawn(async move {
            if let Err(e) = weight_setter.start().await {
                error!("Weight setter task failed: {}", e);
            }
        })
    });

    let miner_prover_handle = miner_prover_opt.map(|mut miner_prover| {
        tokio::spawn(async move {
            if let Err(e) = miner_prover.start().await {
                error!("Miner prover task failed: {}", e);
            }
        })
    });

    let api_handler_handle = tokio::spawn(async move {
        if let Err(e) = api_handler.start().await {
            error!("API handler task failed: {}", e);
        }
    });

    HandlerUtils::print_success("Validator started successfully - all services running");

    signal::ctrl_c().await?;
    HandlerUtils::print_info("Shutdown signal received, stopping validator...");

    if let Some(handle) = scoring_task_handle {
        handle.abort();
    }
    if let Some(handle) = weight_setter_handle {
        handle.abort();
    }
    if let Some(handle) = miner_prover_handle {
        handle.abort();
    }
    api_handler_handle.abort();

    // SQLite connections will be closed automatically when dropped
    HandlerUtils::print_success("Validator shutdown complete");

    Ok(())
}

#[derive(Debug, Clone, Copy)]
enum Signal {
    Term,
    Kill,
}

/// Check if validator process is currently running
fn check_validator_process() -> Result<Option<(u32, u64, f32)>> {
    let mut system = System::new_all();
    system.refresh_all();

    for (pid, process) in system.processes() {
        let name = process.name();
        let cmd = process.cmd();

        // Look for validator process by name or command line
        if name == "validator"
            || cmd
                .iter()
                .any(|arg| arg.contains("validator") && !arg.contains("cargo"))
        {
            let memory_mb = process.memory() / 1024 / 1024;
            let cpu_percent = process.cpu_usage();
            return Ok(Some((pid.as_u32(), memory_mb, cpu_percent)));
        }
    }

    Ok(None)
}

/// Test database connectivity
async fn test_database_connectivity() -> Result<()> {
    // Try to connect to the default SQLite database path
    let db_path = "./validator.db";
    let pool = sqlx::SqlitePool::connect(&format!("sqlite:{db_path}")).await?;

    // Execute a simple query to verify connectivity
    sqlx::query("SELECT 1").fetch_one(&pool).await?;

    pool.close().await;
    Ok(())
}

/// Test API server health
async fn test_api_health() -> Result<u64> {
    let client = Client::new();
    let start_time = SystemTime::now();

    // Try to connect to the default API endpoint
    let response = client
        .get("http://127.0.0.1:8080/api/v1/health")
        .timeout(Duration::from_secs(10))
        .send()
        .await?;

    let elapsed = start_time.elapsed().unwrap_or(Duration::from_secs(0));

    if response.status().is_success() {
        Ok(elapsed.as_millis() as u64)
    } else {
        Err(anyhow::anyhow!(
            "API server returned status: {}",
            response.status()
        ))
    }
}

/// Test Bittensor network connectivity
async fn test_bittensor_connectivity() -> Result<u64> {
    // Load default config to get Bittensor settings
    let config = crate::config::ValidatorConfig::default();

    // Create a temporary Bittensor service to test connectivity
    let service = bittensor::Service::new(config.bittensor.common)
        .await
        .map_err(|e| anyhow::anyhow!("Failed to create Bittensor service: {}", e))?;

    // Get current block number to verify connectivity
    let block_number = service
        .get_block_number()
        .await
        .map_err(|e| anyhow::anyhow!("Failed to get block number: {}", e))?;

    Ok(block_number)
}

/// Find all running validator processes
fn find_validator_processes() -> Result<Vec<u32>> {
    let mut system = System::new_all();
    system.refresh_all();

    let mut processes = Vec::new();

    for (pid, process) in system.processes() {
        let name = process.name();
        let cmd = process.cmd();

        // Look for validator process by name or command line
        if name == "validator"
            || cmd
                .iter()
                .any(|arg| arg.contains("validator") && !arg.contains("cargo"))
        {
            processes.push(pid.as_u32());
        }
    }

    Ok(processes)
}

/// Send signal to process
fn send_signal_to_process(pid: u32, signal: Signal) -> Result<()> {
    use std::process::Command;

    let signal_str = match signal {
        Signal::Term => "TERM",
        Signal::Kill => "KILL",
    };

    #[cfg(unix)]
    {
        let output = Command::new("kill")
            .arg(format!("-{signal_str}"))
            .arg(pid.to_string())
            .output()?;

        if !output.status.success() {
            let stderr = String::from_utf8_lossy(&output.stderr);
            return Err(anyhow::anyhow!(
                "Failed to send {} to PID {}: {}",
                signal_str,
                pid,
                stderr
            ));
        }
    }

    #[cfg(windows)]
    {
        match signal {
            Signal::Term => {
                // On Windows, use taskkill for graceful termination
                let output = Command::new("taskkill")
                    .args(["/PID", &pid.to_string()])
                    .output()?;

                if !output.status.success() {
                    let stderr = String::from_utf8_lossy(&output.stderr);
                    return Err(anyhow::anyhow!(
                        "Failed to terminate PID {}: {}",
                        pid,
                        stderr
                    ));
                }
            }
            Signal::Kill => {
                // Force kill on Windows
                let output = Command::new("taskkill")
                    .args(["/F", "/PID", &pid.to_string()])
                    .output()?;

                if !output.status.success() {
                    let stderr = String::from_utf8_lossy(&output.stderr);
                    return Err(anyhow::anyhow!(
                        "Failed to force kill PID {}: {}",
                        pid,
                        stderr
                    ));
                }
            }
        }
    }

    Ok(())
}

/// Check if process is still running
fn is_process_running(pid: u32) -> Result<bool> {
    let mut system = System::new();
    let pid_obj = Pid::from_u32(pid);
    system.refresh_process(pid_obj);

    Ok(system.process(pid_obj).is_some())
}
