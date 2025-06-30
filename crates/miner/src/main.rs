//! # Basilca Miner
//!
//! Bittensor neuron that manages a fleet of executors and serves
//! validator requests for GPU rental and computational challenges.

use anyhow::Result;
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
mod executor_manager;
mod metrics;
mod persistence;
mod request_verification;
mod session_cleanup;
mod ssh;
mod validator_comms;
mod validator_discovery;

use auth::JwtAuthService;
use bittensor_core::ChainRegistration;
use common::ssh::manager::DefaultSshService;
use config::MinerConfig;
use executor_manager::ExecutorManager;
use persistence::RegistrationDb;
use session_cleanup::run_cleanup_service;
use ssh::{MinerSshConfig, SshCleanupService, ValidatorAccessService};
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
    pub executor_manager: Arc<ExecutorManager>,
    pub registration_db: RegistrationDb,
    pub ssh_access_service: ValidatorAccessService,
    pub jwt_service: std::sync::Arc<JwtAuthService>,
    pub metrics: Option<metrics::MinerMetrics>,
    pub validator_discovery: Option<std::sync::Arc<validator_discovery::ValidatorDiscovery>>,
}

impl MinerState {
    /// Initialize miner state
    pub async fn new(config: MinerConfig, enable_metrics: bool) -> Result<Self> {
        let miner_uid = config.bittensor.uid;
        info!("Initializing miner with UID: {}", miner_uid.as_u16());

        // Initialize metrics system if enabled
        let metrics = if enable_metrics && config.metrics.enabled {
            let miner_metrics = metrics::MinerMetrics::new(config.metrics.clone())?;
            Some(miner_metrics)
        } else {
            None
        };

        // Initialize persistence layer
        let registration_db = RegistrationDb::new(&config.database).await?;

        // Initialize executor manager
        let executor_manager =
            Arc::new(ExecutorManager::new(&config, registration_db.clone()).await?);

        // Initialize SSH services
        let ssh_config = MinerSshConfig::default();
        let ssh_service = std::sync::Arc::new(DefaultSshService::new(ssh_config.clone())?);
        let ssh_access_service = ValidatorAccessService::new(
            ssh_config.clone(),
            ssh_service,
            executor_manager.clone(),
            registration_db.clone(),
        )
        .await?;

        // Start SSH cleanup background task
        let cleanup_service = SshCleanupService::new(ssh_access_service.clone(), &ssh_config);
        cleanup_service.start_cleanup_task().await?;

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
        let validator_comms = ValidatorCommsServer::new(
            config.validator_comms.clone(),
            config.security.clone(),
            executor_manager.clone(),
            registration_db.clone(),
            ssh_access_service.clone(),
            validator_discovery.clone(),
        )
        .await?;

        let jwt_service = validator_comms.jwt_service.clone();

        Ok(Self {
            config,
            miner_uid,
            chain_registration,
            validator_comms,
            executor_manager,
            registration_db,
            ssh_access_service,
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
    _only_machines: Option<String>,
    status_only: bool,
) -> Result<()> {
    // Check if deployment is configured
    let deployment_config = config
        .remote_executor_deployment
        .as_ref()
        .ok_or_else(|| anyhow::anyhow!("No executor deployment configuration found"))?;

    // Initialize persistence
    let db = RegistrationDb::new(&config.database).await?;

    // Create executor manager
    let manager = ExecutorManager::new(config, db).await?;

    // Status check only
    if status_only {
        let available = manager.list_available().await?;
        info!("Checking status of executors...");

        if available.is_empty() {
            warn!("No healthy executors found");
        } else {
            for executor in available {
                info!("✓ {} ({}) - Available", executor.name, executor.id);
                info!(
                    "  GPUs: {}, Address: {}",
                    executor.gpu_count, executor.grpc_address
                );
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
            info!("    GPU count: {}", machine.gpu_count.unwrap_or(0));
            info!("    Executor port: {}", machine.executor_port);
        }
        return Ok(());
    }

    // Actual deployment
    info!("Deploying executors...");
    let results = manager.deploy_all().await?;

    // Report results
    let successful = results.iter().filter(|r| r.success).count();
    let failed = results.len() - successful;

    info!("\nDeployment Summary:");
    info!("  Successful: {}", successful);
    info!("  Failed: {}", failed);

    // Show details
    for result in &results {
        if result.success {
            info!("✓ {} - Deployed successfully", result.machine_name);
        } else {
            error!(
                "✗ {} - Failed: {}",
                result.machine_name,
                result.error.as_ref().unwrap()
            );
        }
    }

    Ok(())
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
    let path = PathBuf::from(config_path);
    let config = if path.exists() {
        MinerConfig::load_from_file(&path)?
    } else {
        MinerConfig::load()?
    };

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
