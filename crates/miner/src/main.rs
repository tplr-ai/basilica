//! # Basilca Miner
//!
//! Bittensor neuron that manages a fleet of executors and serves
//! validator requests for GPU rental and computational challenges.

use anyhow::{Context, Result};
use clap::{Parser, Subcommand};
use common::identity::MinerUid;
use std::net::SocketAddr;
use std::path::PathBuf;
use std::sync::Arc;
use tokio::signal;
use tracing::{error, info, warn};
use tracing_subscriber::{layer::SubscriberExt, util::SubscriberInitExt, EnvFilter};

mod auth;
mod bittensor_core;
mod cli;
mod config;
// mod executor_manager; // Replaced by executors module
mod executors;
mod metrics;
mod persistence;
mod request_verification;
mod session_cleanup;
mod ssh;
mod validator_comms;
mod validator_discovery;

use auth::JwtAuthService;
use bittensor_core::ChainRegistration;
// use common::ssh::manager::DefaultSshService; // Not needed with current architecture
use config::MinerConfig;
use executors::{ExecutorConnectionManager, ExecutorInfo};
use executors::connection_manager::ExecutorConnectionConfig;
use persistence::RegistrationDb;
use session_cleanup::run_cleanup_service;
use ssh::{MinerSshConfig, SshSessionConfig, SshSessionOrchestrator};
use validator_comms::ValidatorCommsServer;

#[derive(Parser, Debug)]
#[command(author, version, about = "Basilca Miner - Bittensor neuron managing executor fleets", long_about = None)]
struct Args {
    /// Configuration file path
    #[arg(short, long, default_value = "miner.toml")]
    config: String,

    /// Log level (trace, debug, info, warn, error)
    #[arg(short, long, default_value = "info")]
    log_level: String,

    /// Enable prometheus metrics endpoint
    #[arg(long)]
    metrics: bool,

    /// Metrics server address
    #[arg(long, default_value = "0.0.0.0:9091")]
    metrics_addr: SocketAddr,

    /// Generate sample configuration file
    #[arg(long)]
    gen_config: bool,

    /// Subcommands for CLI operations
    #[command(subcommand)]
    command: Option<Commands>,
}

#[derive(Subcommand, Debug)]
enum Commands {
    /// Executor management commands
    Executor {
        #[command(subcommand)]
        executor_cmd: cli::ExecutorCommand,
    },
    /// Validator management commands
    Validator {
        #[command(subcommand)]
        validator_cmd: cli::ValidatorCommand,
    },
    /// Service management commands
    Service {
        #[command(subcommand)]
        service_cmd: cli::ServiceCommand,
    },
    /// Database management commands
    Database {
        #[command(subcommand)]
        database_cmd: cli::DatabaseCommand,
    },
    /// Configuration management commands
    Config {
        #[command(subcommand)]
        config_cmd: cli::ConfigCommand,
    },
    /// Show miner status and statistics
    Status,
    /// Run database migrations
    Migrate,
    /// Deploy executors to remote machines
    DeployExecutors {
        /// Only show what would be deployed without actually deploying
        #[arg(long)]
        dry_run: bool,
        /// Deploy to specific machine IDs only (comma-separated)
        #[arg(long)]
        only_machines: Option<String>,
        /// Skip deployment and only check status
        #[arg(long)]
        status_only: bool,
    },
}

/// Main miner state
pub struct MinerState {
    pub config: MinerConfig,
    pub miner_uid: MinerUid,
    pub chain_registration: ChainRegistration,
    pub validator_comms: ValidatorCommsServer,
    pub executor_manager: Arc<ExecutorConnectionManager>,
    pub registration_db: RegistrationDb,
    pub ssh_session_orchestrator: Arc<SshSessionOrchestrator>,
    pub jwt_service: std::sync::Arc<JwtAuthService>,
    pub metrics: Option<metrics::MinerMetrics>,
    pub validator_discovery: Option<std::sync::Arc<validator_discovery::ValidatorDiscovery>>,
}

impl MinerState {
    /// Initialize miner state
    pub async fn new(config: MinerConfig, enable_metrics: bool) -> Result<Self> {
        info!("Initializing miner...");

        // Initialize metrics system if enabled
        let metrics = if enable_metrics && config.metrics.enabled {
            let miner_metrics = metrics::MinerMetrics::new(config.metrics.clone())?;
            Some(miner_metrics)
        } else {
            None
        };

        // Initialize persistence layer
        let registration_db = RegistrationDb::new(&config.database).await?;

        // Initialize executor connection manager
        let executor_config = ExecutorConnectionConfig::default();
        let executor_manager = Arc::new(ExecutorConnectionManager::new(executor_config));
        
        // Register executors from config
        for executor in &config.executor_management.executors {
            let (host, _grpc_port) = if let Some(colon_pos) = executor.grpc_address.rfind(':') {
                let host = &executor.grpc_address[..colon_pos];
                let port = executor.grpc_address[colon_pos + 1..].parse().unwrap_or(50051);
                (host.to_string(), port)
            } else {
                (executor.grpc_address.clone(), 50051)
            };
            
            let info = ExecutorInfo {
                id: executor.id.parse().unwrap_or_else(|_| common::identity::ExecutorId::new()),
                host: host.clone(),
                ssh_port: 22,
                ssh_username: "executor".to_string(),
                grpc_endpoint: Some(executor.grpc_address.clone()),
                last_health_check: None,
                is_healthy: true,
                gpu_count: 1, // Default to 1 GPU
                resources: None,
            };
            executor_manager.register_executor(info).await?;
        }

        // Initialize SSH services
        let _ssh_config = MinerSshConfig::default();
        let ssh_session_config = SshSessionConfig::default();
        let ssh_session_orchestrator = Arc::new(
            SshSessionOrchestrator::new(executor_manager.clone(), ssh_session_config)
        );
        
        // Start SSH cleanup background task
        ssh_session_orchestrator.clone().start_cleanup_task().await;

        // Initialize Bittensor chain registration
        let chain_registration = ChainRegistration::new(config.bittensor.clone()).await?;

        // Initialize validator discovery (optional - only if assignment is configured)
        let validator_discovery = if config.bittensor.skip_registration {
            // Skip validator discovery in local testing mode
            None
        } else {
            // Create assignment strategy based on configuration
            // For now, using business logic assignment as default
            let strategy: Box<dyn validator_discovery::AssignmentStrategy> =
                Box::new(validator_discovery::BusinessLogicAssignment {
                    preferred_validators: vec![], // TODO: Load from config
                    max_executors_per_validator: 3,
                });

            let discovery = validator_discovery::ValidatorDiscovery::new(
                chain_registration.get_bittensor_service(),
                executor_manager.clone(),
                strategy,
                config.bittensor.common.netuid,
            );
            Some(std::sync::Arc::new(discovery))
        };

        // Initialize validator communications server
        let mut validator_comms = ValidatorCommsServer::new(
            config.validator_comms.clone(),
            config.security.clone(),
            executor_manager.clone(),
            registration_db.clone(),
            validator_discovery.clone(),
        )
        .await?;
        
        // Set the SSH session orchestrator
        validator_comms = validator_comms.with_ssh_session_orchestrator(ssh_session_orchestrator.clone());

        let jwt_service = validator_comms.jwt_service.clone();

        // Use a placeholder UID that will be updated after chain registration
        let miner_uid = MinerUid::from(0);

        Ok(Self {
            config,
            miner_uid,
            chain_registration,
            validator_comms,
            executor_manager,
            registration_db,
            ssh_session_orchestrator,
            jwt_service,
            metrics,
            validator_discovery,
        })
    }

    /// Run health check on all components
    pub async fn health_check(&self) -> Result<()> {
        info!("Running miner health check...");

        // Check database connection
        self.registration_db.health_check().await?;

        info!("Miner components healthy");
        Ok(())
    }

    /// Start all miner services
    pub async fn start_services(&self) -> Result<()> {
        info!("Starting miner services...");

        // Start metrics server if enabled
        if let Some(ref metrics) = self.metrics {
            metrics.start_server().await?;
            info!("Miner metrics server started");
        }

        // Perform one-time chain registration (Bittensor network presence)
        self.chain_registration.register_startup().await?;

        // Log the discovered UID
        if let Some(uid) = self.chain_registration.get_discovered_uid().await {
            info!("Miner registered with discovered UID: {}", uid);
        } else {
            warn!("No UID discovered - miner may not be registered on chain");
        }

        // Start validator communications server
        let validator_addr = format!("{}:{}", self.config.server.host, self.config.server.port)
            .parse()
            .expect("Invalid server address configuration");
        let validator_handle = {
            let validator_comms = self.validator_comms.clone();
            tokio::spawn(async move {
                if let Err(e) = validator_comms.serve(validator_addr).await {
                    error!("Validator comms server error: {}", e);
                }
            })
        };

        // Start executor health monitoring
        let executor_handle = {
            let executor_manager = self.executor_manager.clone();
            tokio::spawn(async move {
                if let Err(e) = executor_manager.start_monitoring().await {
                    error!("Executor monitoring error: {}", e);
                }
                // Keep the task alive
                loop {
                    tokio::time::sleep(tokio::time::Duration::from_secs(3600)).await;
                }
            })
        };

        // Start JWT session cleanup service
        let cleanup_handle = {
            let jwt_service = self.jwt_service.clone();
            let cleanup_interval = tokio::time::Duration::from_secs(300); // 5 minutes
            tokio::spawn(async move {
                if let Err(e) = run_cleanup_service(jwt_service, cleanup_interval).await {
                    error!("Session cleanup service error: {}", e);
                }
            })
        };

        // Start validator discovery service if enabled
        let discovery_handle = if let Some(ref discovery) = self.validator_discovery {
            let discovery = discovery.clone();
            let discovery_interval = tokio::time::Duration::from_secs(600); // 10 minutes
            Some(tokio::spawn(async move {
                info!("Starting validator discovery service");
                loop {
                    if let Err(e) = discovery.discover_and_assign().await {
                        error!("Validator discovery error: {}", e);
                    }
                    tokio::time::sleep(discovery_interval).await;
                }
            }))
        } else {
            info!("Validator discovery disabled (local testing mode)");
            None
        };

        info!("All miner services started successfully");

        // Wait for shutdown signal
        if let Some(discovery_handle) = discovery_handle {
            tokio::select! {
                _ = signal::ctrl_c() => {
                    info!("Received shutdown signal, stopping miner...");
                }
                _ = validator_handle => {
                    warn!("Validator comms server stopped unexpectedly");
                }
                _ = executor_handle => {
                    warn!("Executor monitoring stopped unexpectedly");
                }
                _ = cleanup_handle => {
                    warn!("Session cleanup service stopped unexpectedly");
                }
                _ = discovery_handle => {
                    warn!("Validator discovery service stopped unexpectedly");
                }
            }
        } else {
            tokio::select! {
                _ = signal::ctrl_c() => {
                    info!("Received shutdown signal, stopping miner...");
                }
                _ = validator_handle => {
                    warn!("Validator comms server stopped unexpectedly");
                }
                _ = executor_handle => {
                    warn!("Executor monitoring stopped unexpectedly");
                }
                _ = cleanup_handle => {
                    warn!("Session cleanup service stopped unexpectedly");
                }
            }
        }

        Ok(())
    }
}

#[tokio::main]
async fn main() -> Result<()> {
    let args = Args::parse();

    // Generate config file if requested
    if args.gen_config {
        let config = MinerConfig::default();
        let toml_content = toml::to_string_pretty(&config)?;
        std::fs::write(&args.config, toml_content)?;
        println!("Generated configuration file: {}", args.config);
        return Ok(());
    }

    // Initialize logging
    init_logging(&args.log_level)?;

    // Load configuration
    let config = load_config(&args.config)?;
    info!("Loaded configuration from: {}", args.config);

    // Handle CLI commands if provided
    if let Some(command) = args.command {
        return handle_cli_command(command, &config).await;
    }

    // Initialize miner state
    let state = MinerState::new(config, args.metrics).await?;

    // Run initial health check
    if let Err(e) = state.health_check().await {
        error!("Initial health check failed: {}", e);
        return Err(e);
    }

    info!("Starting Basilca Miner (UID: {})", state.miner_uid.as_u16());

    // Start all services
    state.start_services().await?;

    info!("Basilca Miner stopped");
    Ok(())
}

/// Handle CLI commands
async fn handle_cli_command(command: Commands, config: &MinerConfig) -> Result<()> {
    match command {
        Commands::Executor { executor_cmd } => {
            let db = RegistrationDb::new(&config.database).await?;
            cli::handle_executor_command(executor_cmd, config, db).await
        }
        Commands::Validator { validator_cmd } => {
            let db = RegistrationDb::new(&config.database).await?;
            cli::handle_validator_command(validator_cmd, db).await
        }
        Commands::Service { service_cmd } => cli::handle_service_command(service_cmd, config).await,
        Commands::Database { database_cmd } => {
            cli::handle_database_command(database_cmd, config).await
        }
        Commands::Config { config_cmd } => cli::handle_config_command(config_cmd, config).await,
        Commands::Status => cli::show_miner_status(config).await,
        Commands::Migrate => {
            let mut db_config = config.database.clone();
            db_config.run_migrations = true;
            let _db = RegistrationDb::new(&db_config).await?;
            println!("Database migrations completed successfully");
            Ok(())
        }
        Commands::DeployExecutors {
            dry_run,
            only_machines,
            status_only,
        } => handle_deploy_executors(config, dry_run, only_machines, status_only).await,
    }
}

/// Handle deploy executors command
async fn handle_deploy_executors(
    config: &MinerConfig,
    dry_run: bool,
    only_machines: Option<String>,
    status_only: bool,
) -> Result<()> {
    // Check if deployment is configured
    let deployment_config = config
        .remote_executor_deployment
        .as_ref()
        .ok_or_else(|| anyhow::anyhow!("No executor deployment configuration found"))?;

    // Initialize executor connection manager
    let executor_config = ExecutorConnectionConfig::default();
    let executor_manager = Arc::new(ExecutorConnectionManager::new(executor_config));

    // Status check only
    if status_only {
        info!("Checking status of configured executors...");
        
        // Register and check each configured machine
        for machine in &deployment_config.remote_machines {
            let executor_id = machine.id.parse().unwrap_or_else(|_| common::identity::ExecutorId::new());
            let info = ExecutorInfo {
                id: executor_id.clone(),
                host: machine.ssh.host.clone(),
                ssh_port: machine.ssh.port,
                ssh_username: machine.ssh.username.clone(),
                grpc_endpoint: Some(format!("{}:{}", machine.ssh.host, machine.executor_port)),
                last_health_check: None,
                is_healthy: false,
                gpu_count: machine.gpu_count.unwrap_or(1),
                resources: None,
            };
            
            if let Err(e) = executor_manager.register_executor(info).await {
                warn!("Failed to register executor {}: {}", machine.id, e);
                continue;
            }
            
            // Check health
            match executor_manager.health_check_executor(&executor_id).await {
                Ok(healthy) => {
                    if healthy {
                        info!("✓ {} ({}) - Available", machine.name, machine.id);
                        info!("  Host: {}:{}", machine.ssh.host, machine.executor_port);
                        info!("  GPUs: {}", machine.gpu_count.unwrap_or(0));
                    } else {
                        warn!("✗ {} ({}) - Unhealthy", machine.name, machine.id);
                    }
                }
                Err(e) => {
                    error!("✗ {} ({}) - Failed: {}", machine.name, machine.id, e);
                }
            }
        }
        return Ok(());
    }

    // Dry run
    if dry_run {
        info!("DRY RUN: Would deploy executors to the following machines:");
        for machine in &deployment_config.remote_machines {
            info!(
                "  - {} ({}) at {}:{}",
                machine.name, machine.id, machine.ssh.host, machine.ssh.port
            );
            info!("    SSH user: {}", machine.ssh.username);
            info!("    SSH key: {}", machine.ssh.private_key_path.display());
            info!("    GPU count: {}", machine.gpu_count.unwrap_or(0));
            info!("    Executor port: {}", machine.executor_port);
            info!("    Data dir: {}", machine.executor_data_dir.as_ref().map(|s| s.as_str()).unwrap_or("/opt/basilica"));
        }
        return Ok(());
    }

    // Filter machines if only specific ones requested
    let machines_to_deploy: Vec<_> = if let Some(only) = only_machines {
        let only_ids: std::collections::HashSet<_> = only.split(',').collect();
        deployment_config.remote_machines
            .iter()
            .filter(|m| only_ids.contains(m.id.as_str()))
            .collect()
    } else {
        deployment_config.remote_machines.iter().collect()
    };

    // Actual deployment
    info!("Deploying executors to {} machines...", machines_to_deploy.len());
    
    let mut results = Vec::new();
    
    for machine in machines_to_deploy {
        info!("\nDeploying to {} ({})...", machine.name, machine.id);
        
        match deploy_executor_to_machine(machine, &deployment_config.local_executor_binary).await {
            Ok(_) => {
                info!("✓ {} - Deployed successfully", machine.name);
                results.push((machine.name.clone(), true, None));
            }
            Err(e) => {
                error!("✗ {} - Failed: {}", machine.name, e);
                results.push((machine.name.clone(), false, Some(e.to_string())));
            }
        }
    }

    // Report summary
    let successful = results.iter().filter(|(_, success, _)| *success).count();
    let failed = results.len() - successful;

    info!("\nDeployment Summary:");
    info!("  Successful: {}", successful);
    info!("  Failed: {}", failed);

    if failed > 0 {
        error!("\nFailed deployments:");
        for (name, success, error) in &results {
            if !success {
                error!("  {} - {}", name, error.as_ref().unwrap());
            }
        }
    }

    Ok(())
}

/// Deploy executor to a specific machine
async fn deploy_executor_to_machine(
    machine: &config::RemoteMachineConfig,
    executor_binary_path: &std::path::Path,
) -> Result<()> {
    use common::ssh::{SshConnectionDetails, SshConnectionManager, SshFileTransferManager, StandardSshClient};
    
    let ssh_client = StandardSshClient::new();
    let connection = SshConnectionDetails {
        host: machine.ssh.host.clone(),
        port: machine.ssh.port,
        username: machine.ssh.username.clone(),
        private_key_path: machine.ssh.private_key_path.clone(),
        timeout: std::time::Duration::from_secs(30),
    };
    
    // Test SSH connection
    ssh_client.test_connection(&connection).await
        .context("Failed to establish SSH connection")?;
    
    let data_dir = machine.executor_data_dir.as_deref().unwrap_or("/opt/basilica");
    
    // Create directories
    let mkdir_cmd = format!("sudo mkdir -p {}/config {}/logs && sudo chown -R {} {}", 
        data_dir, data_dir, machine.ssh.username, data_dir);
    ssh_client.execute_command(&connection, &mkdir_cmd, true).await
        .context("Failed to create directories")?;
    
    // Copy executor binary
    let remote_executor = format!("{}/executor", data_dir);
    ssh_client.upload_file(&connection, executor_binary_path, &remote_executor).await
        .context("Failed to upload executor binary")?;
    
    // Make executable
    let chmod_cmd = format!("sudo chmod +x {}", remote_executor);
    ssh_client.execute_command(&connection, &chmod_cmd, true).await
        .context("Failed to make executor executable")?;
    
    // Generate and deploy configuration
    let config_content = generate_executor_config(machine);
    let temp_config = format!("/tmp/executor-{}.toml", machine.id);
    std::fs::write(&temp_config, &config_content)?;
    
    let remote_config = format!("{}/config/executor.toml", data_dir);
    ssh_client.upload_file(&connection, std::path::Path::new(&temp_config), &remote_config).await
        .context("Failed to upload configuration")?;
    std::fs::remove_file(&temp_config).ok();
    
    // Install systemd service
    let service_content = generate_systemd_service(machine, data_dir);
    let temp_service = format!("/tmp/basilica-executor-{}.service", machine.id);
    std::fs::write(&temp_service, &service_content)?;
    
    ssh_client.upload_file(&connection, std::path::Path::new(&temp_service), "/tmp/basilica-executor.service").await
        .context("Failed to upload systemd service")?;
    std::fs::remove_file(&temp_service).ok();
    
    // Install and start service
    let service_cmds = vec![
        "sudo mv /tmp/basilica-executor.service /etc/systemd/system/",
        "sudo systemctl daemon-reload",
        "sudo systemctl enable basilica-executor",
        "sudo systemctl restart basilica-executor",
    ];
    
    for cmd in service_cmds {
        ssh_client.execute_command(&connection, cmd, true).await
            .with_context(|| format!("Failed to execute: {}", cmd))?;
    }
    
    // Wait a moment for service to start
    tokio::time::sleep(std::time::Duration::from_secs(3)).await;
    
    // Check service status
    let status_output = ssh_client.execute_command(&connection, "sudo systemctl status basilica-executor", true).await?;
    if !status_output.contains("active (running)") {
        return Err(anyhow::anyhow!("Service failed to start. Status: {}", status_output));
    }
    
    info!("Executor deployed and running on {}", machine.name);
    Ok(())
}

/// Generate executor configuration file
fn generate_executor_config(machine: &config::RemoteMachineConfig) -> String {
    format!(r#"# Executor Configuration
# Generated for: {}

[executor]
id = "{}"
name = "{}"
grpc_port = {}
metrics_port = 9091

[logging]
level = "info"
format = "json"

[gpu]
count = {}

[resources]
max_containers = 10
max_memory_gb = 32
"#,
        machine.name,
        machine.id,
        machine.name,
        machine.executor_port,
        machine.gpu_count.unwrap_or(1)
    )
}

/// Generate systemd service file
fn generate_systemd_service(machine: &config::RemoteMachineConfig, data_dir: &str) -> String {
    format!(r#"[Unit]
Description=Basilica Executor Service for {}
After=network.target

[Service]
Type=simple
User={}
WorkingDirectory={}
ExecStart={}/executor --config {}/config/executor.toml
Restart=always
RestartSec=10
Environment="RUST_LOG=info"
Environment="RUST_BACKTRACE=1"

# Resource limits
LimitNOFILE=65536
LimitMEMLOCK=infinity

# Logging
StandardOutput=append:{}/logs/executor.log
StandardError=append:{}/logs/executor.log

[Install]
WantedBy=multi-user.target
"#,
        machine.name,
        machine.ssh.username,
        data_dir,
        data_dir,
        data_dir,
        data_dir,
        data_dir
    )
}

/// Initialize structured logging
fn init_logging(level: &str) -> Result<()> {
    let filter = EnvFilter::try_from_default_env().unwrap_or_else(|_| EnvFilter::new(level));

    tracing_subscriber::registry()
        .with(filter)
        .with(
            tracing_subscriber::fmt::layer()
                .with_target(false)
                .with_thread_ids(true)
                .with_file(true)
                .with_line_number(true)
                .compact(),
        )
        .init();

    Ok(())
}

/// Load configuration from file and environment
fn load_config(config_path: &str) -> Result<MinerConfig> {
    use common::config::ConfigValidation;

    let path = PathBuf::from(config_path);
    let config = if path.exists() {
        MinerConfig::load_from_file(&path)?
    } else {
        MinerConfig::load()?
    };

    // Validate configuration before proceeding
    config.validate()?;

    // Log any warnings
    let warnings = config.warnings();
    if !warnings.is_empty() {
        warn!("Configuration warnings:");
        for warning in warnings {
            warn!("  - {}", warning);
        }
    }

    Ok(config)
}

// TODO: Implement the following for production readiness:
// 1. Graceful shutdown with proper cleanup of all services
// 2. Configuration hot-reloading for dynamic updates
// 3. Service discovery and health monitoring
// 4. Backup and disaster recovery procedures
// 5. Performance monitoring and alerting
// 6. Load balancing across multiple validators
// 7. Fault tolerance and circuit breaker patterns
// 8. Distributed tracing and correlation IDs
// 9. Security hardening and rate limiting
// 10. Integration testing and end-to-end validation
