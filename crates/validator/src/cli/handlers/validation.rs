//! Validation CLI Handler
//!
//! Provides command-line interface for executor validation functionality.
//! Handles both SSH connection testing and full hardware validation via remote execution.

use super::HandlerUtils;
use crate::persistence::SimplePersistence;
use crate::ssh::{ExecutorSshDetails, ValidatorSshClient};
use crate::validation::{
    AttestationResult, HardwareValidator, HardwareValidatorFactory, ValidationConfig,
};
use anyhow::{Context, Result};
use common::identity::ExecutorId;
use std::path::PathBuf;
use std::str::FromStr;
use std::sync::Arc;
use std::time::Duration;
use tokio::time::timeout;
use tracing::info;

/// SSH connection arguments
#[derive(Debug, Clone)]
pub struct SshConnectArgs {
    /// SSH hostname or IP address
    pub host: String,
    /// SSH username
    pub username: String,
    /// SSH port (default: 22)
    pub port: Option<u16>,
    /// Path to private key file
    pub private_key: Option<PathBuf>,
    /// Connection timeout in seconds
    pub timeout: Option<u64>,
    /// Executor ID (alternative to connection details)
    pub executor_id: Option<String>,
}

/// SSH verification arguments
#[derive(Debug, Clone)]
pub struct SshVerifyArgs {
    /// SSH connection details
    pub ssh_args: Option<SshConnectArgs>,
    /// Executor ID to verify
    pub executor_id: Option<String>,
    /// Miner UID to verify all executors
    pub miner_uid: Option<u16>,
    /// Path to gpu-attestor binary
    pub gpu_attestor_path: Option<PathBuf>,
    /// Remote working directory
    pub remote_work_dir: Option<String>,
    /// Execution timeout in seconds
    pub execution_timeout: Option<u64>,
    /// Skip cleanup after verification
    pub skip_cleanup: bool,
    /// Verbose output
    pub verbose: bool,
}

/// Handle SSH connect command
pub async fn handle_ssh_connect(args: SshConnectArgs) -> Result<()> {
    info!("Starting SSH connection test");

    if let Some(executor_id) = &args.executor_id {
        // Connect using stored executor details (requires persistence)
        let persistence = initialize_persistence().await?;
        connect_by_executor_id(executor_id, persistence).await
    } else {
        // Connect using provided SSH details (direct connection test)
        connect_by_ssh_details(&args).await
    }
}

/// Handle SSH verify command
pub async fn handle_ssh_verify(args: SshVerifyArgs) -> Result<()> {
    info!("Starting SSH verification");

    let persistence = initialize_persistence().await?;
    let config = create_validation_config(&args);
    let mut validator = create_hardware_validator(Some(config), persistence).await?;

    match (&args.ssh_args, &args.executor_id, &args.miner_uid) {
        // Verify using SSH connection details
        (Some(ssh_args), None, None) => {
            verify_by_ssh_details(&mut validator, ssh_args, &args).await
        }
        // Verify using executor ID
        (None, Some(executor_id), None) => {
            verify_by_executor_id(&mut validator, executor_id, &args).await
        }
        // Verify all executors for a miner
        (None, None, Some(miner_uid)) => {
            verify_by_miner_uid(&mut validator, *miner_uid, &args).await
        }
        _ => {
            HandlerUtils::print_error("Specify either SSH details, --executor-id, or --miner-uid");
            Err(anyhow::anyhow!("Invalid verification command arguments"))
        }
    }
}

/// Connect to executor using stored details
async fn connect_by_executor_id(
    executor_id: &str,
    _persistence: Arc<SimplePersistence>,
) -> Result<()> {
    println!("Looking up SSH details for executor: {executor_id}");

    // Parse executor ID
    let _exec_id = ExecutorId::from_str(executor_id).context("Invalid executor ID format")?;

    // TODO: This would be implemented when persistence lookup is ready
    println!("ERROR: Executor SSH details lookup not yet implemented");
    println!("NOTE: To connect, use explicit SSH details instead:");
    println!("   validator connect --host <HOST> --username <USER> --private-key <KEY>");

    Err(anyhow::anyhow!(
        "Executor SSH details lookup not implemented"
    ))
}

/// Connect using provided SSH details  
async fn connect_by_ssh_details(args: &SshConnectArgs) -> Result<()> {
    println!("Testing SSH connection to {}@{}", args.username, args.host);

    let ssh_details = create_ssh_details_from_args(args)?;

    // Create a timeout for the connection test
    let connection_timeout = Duration::from_secs(args.timeout.unwrap_or(30));

    match timeout(connection_timeout, test_ssh_connection(&ssh_details)).await {
        Ok(Ok(())) => {
            println!("SUCCESS: SSH connection successful!");
            println!("   Host: {}", ssh_details.connection().host);
            println!("   Port: {}", ssh_details.connection().port);
            println!("   Username: {}", ssh_details.connection().username);
            println!(
                "   Key: {}",
                ssh_details.connection().private_key_path.display()
            );
            Ok(())
        }
        Ok(Err(e)) => {
            println!("ERROR: SSH connection failed: {e}");
            println!("Troubleshooting tips:");
            println!("   - Verify the host is reachable");
            println!("   - Check SSH key permissions (should be 600)");
            println!("   - Ensure your public key is in the target's authorized_keys");
            println!(
                "   - Try connecting manually: ssh -i {} {}@{}",
                ssh_details.connection().private_key_path.display(),
                args.username,
                args.host
            );
            Err(e)
        }
        Err(_) => {
            let err = anyhow::anyhow!(
                "SSH connection timed out after {} seconds",
                connection_timeout.as_secs()
            );
            println!("TIMEOUT: {err}");
            Err(err)
        }
    }
}

/// Verify executor using SSH connection details
async fn verify_by_ssh_details(
    validator: &mut HardwareValidator,
    ssh_args: &SshConnectArgs,
    verify_args: &SshVerifyArgs,
) -> Result<()> {
    println!(
        "Starting hardware verification for {}@{}",
        ssh_args.username, ssh_args.host
    );

    let ssh_details = create_ssh_details_from_args(ssh_args)?;

    if verify_args.verbose {
        println!("Configuration:");
        println!(
            "   Connection timeout: {}s",
            ssh_details.connection().timeout.as_secs()
        );
        println!(
            "   Execution timeout: {}s",
            verify_args.execution_timeout.unwrap_or(300)
        );
        println!("   Skip cleanup: {}", verify_args.skip_cleanup);
    }

    let start_time = std::time::Instant::now();

    match validator.validate_executor(&ssh_details).await {
        Ok(result) => {
            let duration = start_time.elapsed();

            if result.is_valid {
                println!(
                    "SUCCESS: Hardware verification successful! (took {:.2}s)",
                    duration.as_secs_f64()
                );
                print_verification_results(&result, verify_args.verbose);
            } else {
                println!(
                    "FAILED: Hardware verification failed! (took {:.2}s)",
                    duration.as_secs_f64()
                );
                if let Some(error) = &result.error_message {
                    println!("   Error: {error}");
                }
            }

            Ok(())
        }
        Err(e) => {
            let duration = start_time.elapsed();
            println!(
                "ERROR: Verification error after {:.2}s: {}",
                duration.as_secs_f64(),
                e
            );

            if verify_args.verbose {
                println!("Error details: {e:?}");
            }

            Err(e.into())
        }
    }
}

/// Verify executor by ID
async fn verify_by_executor_id(
    _validator: &mut HardwareValidator,
    executor_id: &str,
    _args: &SshVerifyArgs,
) -> Result<()> {
    println!("Verifying executor: {executor_id}");

    // Parse executor ID
    let _exec_id = ExecutorId::from_str(executor_id).context("Invalid executor ID format")?;

    // TODO: This would use the persistence lookup when implemented
    println!("ERROR: Executor-by-ID verification not yet implemented");
    println!("NOTE: To verify, use explicit SSH details instead:");
    println!("   validator verify --host <HOST> --username <USER> --private-key <KEY>");

    Err(anyhow::anyhow!("Executor ID verification not implemented"))
}

/// Verify all executors for a miner
async fn verify_by_miner_uid(
    _validator: &mut HardwareValidator,
    miner_uid: u16,
    _args: &SshVerifyArgs,
) -> Result<()> {
    println!("Verifying all executors for miner UID: {miner_uid}");

    // TODO: This would:
    // 1. Query metagraph for miner details
    // 2. Connect to miner gRPC service
    // 3. Request list of available executors
    // 4. Verify each executor in parallel
    // 5. Aggregate results

    println!("ERROR: Miner cluster verification not yet implemented");
    println!("This feature requires:");
    println!("   - Metagraph integration for miner discovery");
    println!("   - Miner gRPC client for executor enumeration");
    println!("   - Parallel verification with rate limiting");
    println!("   - Result aggregation and reporting");

    Err(anyhow::anyhow!(
        "Miner cluster verification not implemented"
    ))
}

/// Initialize persistence layer
async fn initialize_persistence() -> Result<Arc<SimplePersistence>> {
    let db_path = std::env::current_dir()?.join("validator.db");
    let persistence = Arc::new(
        SimplePersistence::new(
            db_path.to_string_lossy().as_ref(),
            "validator_hotkey".to_string(),
        )
        .await
        .context("Failed to initialize persistence layer")?,
    );
    Ok(persistence)
}

/// Create hardware validator with optional custom config
async fn create_hardware_validator(
    config: Option<ValidationConfig>,
    persistence: Arc<SimplePersistence>,
) -> Result<HardwareValidator> {
    let validator = if let Some(config) = config {
        HardwareValidatorFactory::create_with_config(config, persistence).await
    } else {
        HardwareValidatorFactory::create_default(persistence).await
    };

    validator.context("Failed to create hardware validator")
}

/// Create validation configuration from arguments
fn create_validation_config(args: &SshVerifyArgs) -> ValidationConfig {
    let mut config = ValidationConfig::default();

    if let Some(gpu_attestor_path) = &args.gpu_attestor_path {
        config.gpu_attestor_binary_path = gpu_attestor_path.clone();
    }

    if let Some(remote_work_dir) = &args.remote_work_dir {
        config.remote_work_dir = remote_work_dir.clone();
    }

    if let Some(execution_timeout) = args.execution_timeout {
        config.execution_timeout = Duration::from_secs(execution_timeout);
    }

    config.cleanup_remote_files = !args.skip_cleanup;

    config
}

/// Create SSH details from connection arguments
fn create_ssh_details_from_args(args: &SshConnectArgs) -> Result<ExecutorSshDetails> {
    let private_key_path = args.private_key.clone().unwrap_or_else(|| {
        // Default SSH key locations
        let home = std::env::var("HOME").unwrap_or_else(|_| "/root".to_string());
        PathBuf::from(format!("{home}/.ssh/id_rsa"))
    });

    // Verify key file exists
    if !private_key_path.exists() {
        return Err(anyhow::anyhow!(
            "SSH private key not found at: {}. Use --private-key to specify location.",
            private_key_path.display()
        ));
    }

    let executor_id = if let Some(exec_id) = &args.executor_id {
        ExecutorId::from_str(exec_id).context("Invalid executor ID format")?
    } else {
        ExecutorId::new() // Generate random ID for connection testing
    };

    Ok(ExecutorSshDetails::new(
        executor_id,
        args.host.clone(),
        args.username.clone(),
        args.port,
        private_key_path,
        Some(Duration::from_secs(args.timeout.unwrap_or(30))),
    ))
}

/// Test SSH connection without full validation
async fn test_ssh_connection(ssh_details: &ExecutorSshDetails) -> Result<()> {
    let client = ValidatorSshClient::new();
    client
        .test_connection(ssh_details.connection())
        .await
        .context("SSH connection test failed")
}

/// Print detailed verification results
fn print_verification_results(result: &AttestationResult, verbose: bool) {
    if let Some(specs) = &result.hardware_specs {
        println!("Hardware Specifications:");

        // CPU Information
        println!(
            "   CPU: {} ({} cores, {} MHz, {})",
            specs.cpu.model, specs.cpu.cores, specs.cpu.frequency_mhz, specs.cpu.architecture
        );

        // GPU Information
        println!("   GPUs: {} detected", specs.gpu.len());
        for (i, gpu) in specs.gpu.iter().enumerate() {
            println!(
                "      GPU {}: {} {} ({} MB VRAM, driver {})",
                i + 1,
                gpu.vendor,
                gpu.model,
                gpu.vram_mb,
                gpu.driver_version
            );
            if let Some(utilization) = gpu.utilization_percent {
                println!("         Utilization: {utilization:.1}%");
            }
        }

        // Memory Information
        println!(
            "   Memory: {:.1} GB total, {:.1} GB available ({})",
            specs.memory.total_mb as f64 / 1024.0,
            specs.memory.available_mb as f64 / 1024.0,
            specs.memory.memory_type
        );

        // Storage Information
        println!(
            "   Storage: {:.1} GB total, {:.1} GB available ({})",
            specs.storage.total_gb, specs.storage.available_gb, specs.storage.storage_type
        );

        if let Some(io_perf) = &specs.storage.io_performance {
            println!(
                "      I/O: {:.1} MB/s read, {:.1} MB/s write, {} IOPS",
                io_perf.read_speed_mbps, io_perf.write_speed_mbps, io_perf.iops
            );
        }

        // Network Information
        println!(
            "   Network: {:.1} Mbps bandwidth, {:.1}ms latency, {:.2}% packet loss",
            specs.network.bandwidth_mbps,
            specs.network.latency_ms,
            specs.network.packet_loss_percent
        );

        // Docker Status
        println!(
            "   Docker: {} (version {})",
            if specs.docker_status.daemon_running {
                "Running"
            } else {
                "Not running"
            },
            specs.docker_status.version
        );

        if specs.docker_status.nvidia_runtime_available {
            println!("      NVIDIA runtime: Available");
        }

        if verbose {
            println!("   Validation Details:");
            println!("      Executor ID: {}", result.executor_id);
            println!("      Validated at: {:?}", result.validated_at);
            println!("      Duration: {:?}", result.validation_duration);

            if let Some(signature) = &result.signature {
                println!(
                    "      Signature: {}...",
                    &signature[..std::cmp::min(32, signature.len())]
                );
            }
        }
    }
}

/// Print command usage examples
pub fn print_usage_examples() {
    println!("SSH Validation Commands Usage Examples:");
    println!();

    println!("Testing SSH Connection:");
    println!("   validator connect --host 192.168.1.100 --username ubuntu");
    println!("   validator connect --host gpu-node-1 --username ubuntu --port 2222");
    println!("   validator connect --host 10.0.0.5 --username ubuntu --private-key ~/.ssh/gpu_key");
    println!();

    println!("Hardware Verification:");
    println!("   validator verify --host 192.168.1.100 --username ubuntu");
    println!("   validator verify --host gpu-node-1 --username ubuntu --verbose");
    println!("   validator verify --executor-id exec_123 --verbose");
    println!("   validator verify --miner-uid 42");
    println!();

    println!("Advanced Options:");
    println!("   validator verify --host 10.0.0.5 --username ubuntu \\");
    println!("                    --gpu-attestor-path ./custom-gpu-attestor \\");
    println!("                    --remote-work-dir /tmp/custom \\");
    println!("                    --execution-timeout 600 \\");
    println!("                    --skip-cleanup --verbose");
}
