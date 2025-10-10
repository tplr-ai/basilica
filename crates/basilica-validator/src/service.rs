use crate::api::ApiHandler;
use crate::bittensor_core::{ChainRegistration, WeightSetter};
use crate::cli::handlers::rental::create_rental_manager;
use crate::collateral::collateral_scan::Collateral;
use crate::config::ValidatorConfig;
use crate::gpu::GpuScoringEngine;
use crate::metrics::ValidatorMetrics;
use crate::miner_prover::MinerProver;
use crate::persistence::cleanup_task::CleanupTask;
use crate::persistence::gpu_profile_repository::GpuProfileRepository;
use crate::persistence::SimplePersistence;

use anyhow::Result;
use basilica_common::MemoryStorage;
use bittensor::Service as BittensorService;
use reqwest::Client;
use std::path::PathBuf;
use std::sync::Arc;
use std::time::{Duration, SystemTime};
use sysinfo::{Pid, System};
use tokio::signal;
use tracing::{debug, error, info};

/// Main validator service that manages all validator components and their lifecycle
pub struct ValidatorService {
    config: ValidatorConfig,
}

impl ValidatorService {
    /// Create a new validator service instance
    pub fn new(config: ValidatorConfig) -> Self {
        Self { config }
    }

    /// Start the validator with all its components
    pub async fn start(&self) -> Result<()> {
        let storage_path =
            PathBuf::from(&self.config.storage.data_dir).join("validator_storage.json");
        let storage = MemoryStorage::with_file(storage_path).await?;

        // Extract database path from URL (remove sqlite: prefix if present)
        let db_url = &self.config.database.url;
        let db_path = if let Some(stripped) = db_url.strip_prefix("sqlite:") {
            stripped
        } else {
            db_url
        };

        debug!("Database URL: {}", db_url);
        debug!("Database path: {}", db_path);

        // Ensure the database directory exists
        if let Some(parent) = std::path::Path::new(db_path).parent() {
            debug!("Creating directory: {:?}", parent);
            std::fs::create_dir_all(parent)?;
        }

        let persistence =
            SimplePersistence::new(db_path, self.config.bittensor.common.hotkey_name.clone())
                .await?;

        persistence.run_migrations().await?;

        let persistence_arc = Arc::new(persistence);

        // Create GPU profile repository (needed for weight setter and cleanup task)
        let gpu_profile_repo = Arc::new(GpuProfileRepository::new(persistence_arc.pool().clone()));

        // Initialize metrics system if enabled
        let validator_metrics = if self.config.metrics.enabled {
            let metrics =
                ValidatorMetrics::new(self.config.metrics.clone(), persistence_arc.clone())?;
            metrics.start_server().await?;
            info!("Validator metrics server started with GPU metrics collection");
            Some(metrics)
        } else {
            None
        };

        let bittensor_service: Arc<BittensorService> =
            Arc::new(BittensorService::new(self.config.bittensor.common.clone()).await?);

        // Initialize chain registration and perform startup registration
        let chain_registration =
            ChainRegistration::new(&self.config, bittensor_service.clone()).await?;

        // Perform one-time startup registration
        chain_registration.register_startup().await?;
        info!("Validator registered on chain with axon endpoint");

        // Log the discovered UID
        if let Some(uid) = chain_registration.get_discovered_uid().await {
            info!("Validator registered with discovered UID: {uid}");
        } else {
            info!("No UID discovered - validator may not be registered on chain");
        }

        let miner_prover = MinerProver::new(
            self.config.verification.clone(),
            self.config.automatic_verification.clone(),
            self.config.ssh_session.clone(),
            bittensor_service.clone(),
            persistence_arc.clone(),
            validator_metrics.as_ref().map(|m| Arc::new(m.clone())),
        )?;

        // Initialize weight setter with block-based timing from emission config
        let blocks_per_weight_set = self.config.emission.weight_set_interval_blocks;

        // Create GPU scoring engine using the existing gpu_profile_repo
        let gpu_scoring_engine = if let Some(ref metrics) = validator_metrics {
            Arc::new(GpuScoringEngine::with_metrics(
                gpu_profile_repo.clone(),
                Arc::new(metrics.clone()),
                self.config.emission.clone(),
            ))
        } else {
            Arc::new(GpuScoringEngine::new(
                gpu_profile_repo.clone(),
                self.config.emission.clone(),
            ))
        };

        let weight_setter = WeightSetter::new(
            self.config.bittensor.common.clone(),
            bittensor_service.clone(),
            storage.clone(),
            persistence_arc.clone(),
            self.config.verification.min_score_threshold,
            blocks_per_weight_set,
            gpu_scoring_engine,
            self.config.emission.clone(),
            gpu_profile_repo.clone(),
            validator_metrics.as_ref().map(|m| Arc::new(m.clone())),
        )?;
        let weight_setter = Arc::new(weight_setter);

        // Create validator hotkey for API handler
        // Get the account ID from bittensor service and convert to SS58 string
        let account_id = bittensor_service.get_account_id();
        let ss58_address = format!("{account_id}");
        let validator_hotkey = basilica_common::identity::Hotkey::new(ss58_address)
            .map_err(|e| anyhow::anyhow!("Failed to create hotkey: {}", e))?;

        let mut api_handler = ApiHandler::new(
            self.config.api.clone(),
            persistence_arc.clone(),
            gpu_profile_repo.clone(),
            storage.clone(),
            self.config.clone(),
            validator_hotkey.clone(),
        );

        let rental_manager = if let Some(ref metrics) = validator_metrics {
            Some(
                create_rental_manager(
                    &self.config,
                    persistence_arc.clone(),
                    metrics.prometheus(), // Pass prometheus metrics
                )
                .await?,
            )
        } else {
            tracing::warn!("Rental manager disabled: metrics must be enabled for rentals");
            None
        };

        if let Ok(miner_client) = miner_prover
            .get_verification_engine()
            .create_authenticated_client()
        {
            api_handler = api_handler.with_miner_client(Arc::new(miner_client));
        }

        if let Some(rental_manager) = rental_manager {
            api_handler = api_handler.with_rental_manager(Arc::new(rental_manager));
        }

        // Store metrics for cleanup (if needed)
        let _validator_metrics = validator_metrics;

        info!("All components initialized successfully");

        // Start scoring update task
        let weight_setter_clone = weight_setter.clone();
        let scoring_task_handle = tokio::spawn(async move {
            let mut interval = tokio::time::interval(Duration::from_secs(300)); // Update scores every 5 minutes
            loop {
                interval.tick().await;
                if let Err(e) = weight_setter_clone.update_all_miner_scores().await {
                    error!("Failed to update miner scores: {}", e);
                }
            }
        });

        let weight_setter_clone = weight_setter.clone();
        let weight_setter_handle = tokio::spawn(async move {
            if let Err(e) = weight_setter_clone.start().await {
                error!("Weight setter task failed: {}", e);
            }
        });

        let miner_prover_handle = tokio::spawn(async move {
            if let Err(e) = miner_prover.start().await {
                error!("Miner prover task failed: {}", e);
            }
        });

        let api_handler_handle = tokio::spawn(async move {
            if let Err(e) = api_handler.start().await {
                error!("API handler task failed: {}", e);
            }
        });

        // Start cleanup task if enabled
        let cleanup_task_handle = if self.config.cleanup.enabled {
            let cleanup_config = self.config.cleanup.clone();
            let gpu_repo = gpu_profile_repo.clone();

            Some(tokio::spawn(async move {
                let cleanup_task = CleanupTask::new(cleanup_config, gpu_repo);
                if let Err(e) = cleanup_task.start().await {
                    error!("Database cleanup task failed: {}", e);
                }
            }))
        } else {
            info!("Database cleanup task is disabled");
            None
        };

        let mut collateral_scan =
            Collateral::new(self.config.verification.clone(), persistence_arc.clone());

        let collateral_scan_handle = tokio::spawn(async move {
            if let Err(e) = collateral_scan.start().await {
                error!("Collateral scan task failed: {}", e);
            }
        });

        info!("Validator started successfully - all services running");

        signal::ctrl_c().await?;
        info!("Shutdown signal received, stopping validator...");

        scoring_task_handle.abort();
        weight_setter_handle.abort();
        miner_prover_handle.abort();
        if let Some(handle) = cleanup_task_handle {
            handle.abort();
        }
        api_handler_handle.abort();

        collateral_scan_handle.abort();

        // SQLite connections will be closed automatically when dropped
        info!("Validator shutdown complete");

        Ok(())
    }

    /// Stop all running validator processes
    pub async fn stop() -> Result<()> {
        ProcessUtils::stop_all_processes().await
    }

    /// Check the status of the validator and its components
    pub async fn status(&self) -> Result<ServiceStatus> {
        let status = ServiceStatus {
            process: ProcessUtils::check_validator_process()?,
            database_healthy: self.test_database_connectivity().await.is_ok(),
            api_response_time: self.test_api_health().await.ok(),
            bittensor_block: self.test_bittensor_connectivity().await.ok(),
        };

        Ok(status)
    }

    /// Test database connectivity
    async fn test_database_connectivity(&self) -> Result<()> {
        let pool = sqlx::SqlitePool::connect(&self.config.database.url).await?;
        sqlx::query("SELECT 1").fetch_one(&pool).await?;
        pool.close().await;
        Ok(())
    }

    /// Test API server health
    async fn test_api_health(&self) -> Result<u64> {
        let client = Client::new();
        let start_time = SystemTime::now();

        let api_url = format!(
            "http://{}:{}/health",
            self.config.server.host, self.config.server.port
        );
        let response = client
            .get(&api_url)
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
    async fn test_bittensor_connectivity(&self) -> Result<u64> {
        let service = BittensorService::new(self.config.bittensor.common.clone())
            .await
            .map_err(|e| anyhow::anyhow!("Failed to create Bittensor service: {}", e))?;

        let block_number = service
            .get_block_number()
            .await
            .map_err(|e| anyhow::anyhow!("Failed to get block number: {}", e))?;

        Ok(block_number)
    }
}

/// Status information for the validator service
#[derive(Default)]
pub struct ServiceStatus {
    pub process: Option<(u32, u64, f32)>, // pid, memory_mb, cpu_percent
    pub database_healthy: bool,
    pub api_response_time: Option<u64>,
    pub bittensor_block: Option<u64>,
}

impl ServiceStatus {
    pub fn is_healthy(&self) -> bool {
        self.process.is_some()
            && self.database_healthy
            && self.api_response_time.is_some()
            && self.bittensor_block.is_some()
    }
}

/// Process management utilities
struct ProcessUtils;

impl ProcessUtils {
    /// Check if validator process is currently running
    fn check_validator_process() -> Result<Option<(u32, u64, f32)>> {
        let mut system = System::new_all();
        system.refresh_all();

        for (pid, process) in system.processes() {
            let name = process.name();
            let cmd = process.cmd();

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

    /// Find all running validator processes
    fn find_validator_processes() -> Result<Vec<u32>> {
        let mut system = System::new_all();
        system.refresh_all();

        let mut processes = Vec::new();

        for (pid, process) in system.processes() {
            let name = process.name();
            let cmd = process.cmd();

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

    /// Stop all validator processes with graceful shutdown and force kill if needed
    async fn stop_all_processes() -> Result<()> {
        use std::time::{Duration, SystemTime};

        let _start_time = SystemTime::now();

        let processes = Self::find_validator_processes()?;

        if processes.is_empty() {
            return Ok(());
        }

        // Send graceful shutdown signal (SIGTERM)
        for &pid in &processes {
            let _ = Self::send_signal_to_process(pid, Signal::Term);
        }

        // Wait for clean shutdown with timeout
        let shutdown_timeout = Duration::from_secs(30);
        let shutdown_start = SystemTime::now();

        let mut remaining_processes = processes.clone();

        while !remaining_processes.is_empty()
            && shutdown_start.elapsed().unwrap_or(Duration::from_secs(0)) < shutdown_timeout
        {
            tokio::time::sleep(Duration::from_millis(1000)).await;

            remaining_processes.retain(|&pid| matches!(Self::is_process_running(pid), Ok(true)));
        }

        // Force kill remaining processes if necessary
        if !remaining_processes.is_empty() {
            for &pid in &remaining_processes {
                let _ = Self::send_signal_to_process(pid, Signal::Kill);
                tokio::time::sleep(Duration::from_millis(500)).await;
            }
        }

        // Final verification
        let final_processes = Self::find_validator_processes()?;

        if !final_processes.is_empty() {
            return Err(anyhow::anyhow!("Some processes could not be terminated"));
        }

        Ok(())
    }
}

#[derive(Debug, Clone, Copy)]
enum Signal {
    Term,
    Kill,
}
