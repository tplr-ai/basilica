use crate::api::ApiHandler;
use crate::basilica_api::{BasilicaApiClient, ValidatorSigner};
use crate::bittensor_core::{ChainRegistration, WeightSetter};
use crate::collateral::collateral_scan::Collateral;
use crate::collateral::evaluator::CollateralEvaluator;
use crate::collateral::evidence::EvidenceStore;
use crate::collateral::grace_tracker::GracePeriodTracker;
use crate::collateral::manager::CollateralManager;
use crate::collateral::SlashExecutor;
use crate::config::ValidatorConfig;
use crate::gpu::GpuScoringEngine;
use crate::grpc::start_registration_server;
use crate::metrics::{ValidatorMetrics, ValidatorPrometheusMetrics};
use crate::miner_prover::{MinerProver, MinerProverParams};
use crate::persistence::cleanup_task::CleanupTask;
use crate::persistence::gpu_profile_repository::GpuProfileRepository;
use crate::persistence::SimplePersistence;

use anyhow::{Context, Result};
use basilica_common::MemoryStorage;
use bittensor::Service as BittensorService;
use reqwest::Client;
use std::path::PathBuf;
use std::sync::Arc;
use std::time::{Duration as StdDuration, SystemTime};
use sysinfo::{Pid, System};
use tokio::signal;
use tokio::task::JoinHandle;
use tracing::{debug, error, info, warn};

/// Main validator service that manages all validator components and their lifecycle
pub struct ValidatorService {
    config: ValidatorConfig,
}

struct RuntimeHandles {
    scoring_task: JoinHandle<()>,
    weight_setter_task: JoinHandle<()>,
    miner_prover_task: JoinHandle<()>,
    api_handler_task: JoinHandle<()>,
    registration_server_task: JoinHandle<()>,
    cleanup_task: Option<JoinHandle<()>>,
    collateral_scan_task: JoinHandle<()>,
}

struct TaskInputs {
    weight_setter: Arc<WeightSetter>,
    miner_prover: MinerProver,
    api_handler: ApiHandler,
    persistence: Arc<SimplePersistence>,
    collateral_manager: Option<Arc<CollateralManager>>,
    gpu_profile_repo: Arc<GpuProfileRepository>,
    validator_ssh_public_key: String,
    api_client: Arc<BasilicaApiClient>,
}

impl ValidatorService {
    /// Create a new validator service instance
    pub fn new(config: ValidatorConfig) -> Self {
        Self { config }
    }

    /// Start the validator with all its components
    pub async fn start(&self) -> Result<()> {
        let storage = self.init_storage().await?;
        let persistence_arc = self.init_persistence().await?;
        let gpu_profile_repo = Arc::new(GpuProfileRepository::new(persistence_arc.pool().clone()));
        let validator_metrics = self.init_metrics(persistence_arc.clone()).await?;
        let bittensor_service = self.init_bittensor_service().await?;

        let signer: Arc<dyn ValidatorSigner> = bittensor_service.clone();
        let api_client = Arc::new(BasilicaApiClient::new(
            self.config.api_endpoint.clone(),
            signer,
            self.config.billing.timeout_secs,
            StdDuration::from_secs(self.config.bidding.price_cache_ttl_secs),
            self.config.pricing.cache_ttl(),
        )?);

        let chain_registration = self
            .init_chain_registration(bittensor_service.clone())
            .await?;
        self.log_chain_registration(&chain_registration).await;

        let weight_setter = self.build_weight_setter(
            bittensor_service.clone(),
            storage.clone(),
            persistence_arc.clone(),
            gpu_profile_repo.clone(),
            api_client.clone(),
            validator_metrics.as_ref(),
        )?;
        let validator_hotkey = self.build_validator_hotkey(&bittensor_service)?;

        let mut api_handler = self.build_api_handler(
            persistence_arc.clone(),
            gpu_profile_repo.clone(),
            storage.clone(),
            validator_hotkey.clone(),
        );

        let collateral_metrics = validator_metrics.as_ref().map(|m| m.prometheus());
        let (collateral_manager, slash_executor) = self
            .init_collateral_components(
                persistence_arc.clone(),
                collateral_metrics.clone(),
                bittensor_service.clone(),
                api_client.clone(),
            )
            .await?;

        let miner_prover = self.init_miner_prover(
            bittensor_service.clone(),
            persistence_arc.clone(),
            validator_metrics.as_ref(),
        )?;

        let rental_manager = self
            .init_rental_manager(
                persistence_arc.clone(),
                validator_metrics.as_ref(),
                collateral_manager.clone(),
                slash_executor.clone(),
                &validator_hotkey,
            )
            .await?;

        // Extract SSH public key from rental manager (required for miner registration)
        let validator_ssh_public_key = rental_manager
            .as_ref()
            .map(|rm| rm.get_validator_ssh_public_key())
            .ok_or_else(|| {
                anyhow::anyhow!("Rental manager required for SSH key - metrics must be enabled")
            })?;

        info!("Validator SSH public key loaded for miner registration");

        if let Some(rental_manager) = rental_manager {
            api_handler = api_handler.with_rental_manager(Arc::new(rental_manager));
        }

        info!("All components initialized successfully");

        let handles = self
            .spawn_tasks(TaskInputs {
                weight_setter,
                miner_prover,
                api_handler,
                persistence: persistence_arc.clone(),
                collateral_manager: collateral_manager.clone(),
                gpu_profile_repo: gpu_profile_repo.clone(),
                validator_ssh_public_key,
                api_client,
            })
            .await;

        info!("Validator started successfully - all services running");
        signal::ctrl_c().await?;
        info!("Shutdown signal received, stopping validator...");

        self.shutdown(handles);

        info!("Validator shutdown complete");
        Ok(())
    }

    async fn init_storage(&self) -> Result<MemoryStorage> {
        let storage_path =
            PathBuf::from(&self.config.storage.data_dir).join("validator_storage.json");
        MemoryStorage::with_file(storage_path).await
    }

    fn resolve_db_path(&self) -> Result<String> {
        let db_url = &self.config.database.url;
        let db_path = if let Some(stripped) = db_url.strip_prefix("sqlite:") {
            stripped
        } else {
            db_url
        };
        debug!("Database URL: {}", db_url);
        debug!("Database path: {}", db_path);
        if let Some(parent) = std::path::Path::new(db_path).parent() {
            debug!("Creating directory: {:?}", parent);
            std::fs::create_dir_all(parent)?;
        }
        Ok(db_path.to_string())
    }

    async fn init_persistence(&self) -> Result<Arc<SimplePersistence>> {
        let db_path = self.resolve_db_path()?;
        let persistence =
            SimplePersistence::new(&db_path, self.config.bittensor.common.hotkey_name.clone())
                .await?;
        persistence.run_migrations().await?;
        Ok(Arc::new(persistence))
    }

    async fn init_metrics(
        &self,
        persistence: Arc<SimplePersistence>,
    ) -> Result<Option<ValidatorMetrics>> {
        if !self.config.metrics.enabled {
            return Ok(None);
        }
        let metrics = ValidatorMetrics::new(self.config.metrics.clone(), persistence)?;
        metrics.start_server().await?;
        info!("Validator metrics server started with GPU metrics collection");
        Ok(Some(metrics))
    }

    async fn init_bittensor_service(&self) -> Result<Arc<BittensorService>> {
        Ok(Arc::new(
            BittensorService::new(self.config.bittensor.common.clone()).await?,
        ))
    }

    async fn init_chain_registration(
        &self,
        bittensor_service: Arc<BittensorService>,
    ) -> Result<ChainRegistration> {
        let chain_registration = ChainRegistration::new(&self.config, bittensor_service).await?;
        chain_registration.register_startup().await?;
        Ok(chain_registration)
    }

    async fn log_chain_registration(&self, chain_registration: &ChainRegistration) {
        info!("Validator registered on chain with axon endpoint");
        if let Some(uid) = chain_registration.get_discovered_uid().await {
            info!("Validator registered with discovered UID: {uid}");
        } else {
            info!("No UID discovered - validator may not be registered on chain");
        }
    }

    fn build_weight_setter(
        &self,
        bittensor_service: Arc<BittensorService>,
        storage: MemoryStorage,
        persistence: Arc<SimplePersistence>,
        gpu_profile_repo: Arc<GpuProfileRepository>,
        api_client: Arc<BasilicaApiClient>,
        validator_metrics: Option<&ValidatorMetrics>,
    ) -> Result<Arc<WeightSetter>> {
        let gpu_scoring_engine = if let Some(metrics) = validator_metrics {
            Arc::new(GpuScoringEngine::with_metrics(
                gpu_profile_repo.clone(),
                persistence.clone(),
                Arc::new(metrics.clone()),
                self.config.emission.clone(),
            ))
        } else {
            Arc::new(GpuScoringEngine::new(
                gpu_profile_repo.clone(),
                persistence.clone(),
                self.config.emission.clone(),
            ))
        };

        let weight_setter = WeightSetter::new(
            self.config.bittensor.common.clone(),
            bittensor_service,
            storage,
            persistence,
            self.config.emission.weight_set_interval_blocks,
            gpu_scoring_engine,
            self.config.emission.clone(),
            api_client,
            gpu_profile_repo,
            validator_metrics.map(|m| Arc::new(m.clone())),
            self.config.collateral.as_ref().map(|c| c.grace_period()),
        )?;
        Ok(Arc::new(weight_setter))
    }

    fn build_validator_hotkey(
        &self,
        bittensor_service: &BittensorService,
    ) -> Result<basilica_common::identity::Hotkey> {
        let account_id = bittensor_service.get_account_id();
        let ss58_address = format!("{account_id}");
        basilica_common::identity::Hotkey::new(ss58_address)
            .map_err(|e| anyhow::anyhow!("Failed to create hotkey: {}", e))
    }

    fn build_api_handler(
        &self,
        persistence: Arc<SimplePersistence>,
        gpu_profile_repo: Arc<GpuProfileRepository>,
        storage: MemoryStorage,
        validator_hotkey: basilica_common::identity::Hotkey,
    ) -> ApiHandler {
        ApiHandler::new(
            self.config.api.clone(),
            persistence,
            gpu_profile_repo,
            storage,
            self.config.clone(),
            validator_hotkey,
        )
    }

    async fn init_collateral_components(
        &self,
        persistence: Arc<SimplePersistence>,
        collateral_metrics: Option<Arc<ValidatorPrometheusMetrics>>,
        signer: Arc<BittensorService>,
        api_client: Arc<BasilicaApiClient>,
    ) -> Result<(Option<Arc<CollateralManager>>, Option<Arc<SlashExecutor>>)> {
        let Some(collateral_config) = self.config.collateral.clone() else {
            return Ok((None, None));
        };
        if collateral_config.shadow_mode {
            warn!("Collateral shadow_mode is enabled; on-chain slashing is disabled");
        }
        let grace_tracker = Arc::new(GracePeriodTracker::new(
            persistence.clone(),
            collateral_config.grace_period(),
        ));
        let evaluator = Arc::new(CollateralEvaluator::new(
            collateral_config.clone(),
            grace_tracker.clone(),
        ));
        let collateral_manager = Arc::new(CollateralManager::new(
            persistence.clone(),
            api_client,
            evaluator,
            grace_tracker.clone(),
            self.config.bittensor.common.netuid,
            collateral_metrics.clone(),
        ));
        let evidence_store = EvidenceStore::from_config(&collateral_config).await?;
        let slash_executor = Arc::new(SlashExecutor::new(
            collateral_config,
            evidence_store,
            grace_tracker,
            collateral_metrics,
            Some(signer),
        ));
        Ok((Some(collateral_manager), Some(slash_executor)))
    }

    fn init_miner_prover(
        &self,
        bittensor_service: Arc<BittensorService>,
        persistence: Arc<SimplePersistence>,
        validator_metrics: Option<&ValidatorMetrics>,
    ) -> Result<MinerProver> {
        MinerProver::new(MinerProverParams {
            config: self.config.verification.clone(),
            automatic_config: self.config.automatic_verification.clone(),
            ssh_session_config: self.config.ssh_session.clone(),
            bittensor_service,
            persistence,
            metrics: validator_metrics.map(|m| Arc::new(m.clone())),
            netuid: self.config.bittensor.common.netuid,
        })
    }

    async fn init_rental_manager(
        &self,
        persistence: Arc<SimplePersistence>,
        validator_metrics: Option<&ValidatorMetrics>,
        collateral_manager: Option<Arc<CollateralManager>>,
        slash_executor: Option<Arc<SlashExecutor>>,
        validator_hotkey: &basilica_common::identity::Hotkey,
    ) -> Result<Option<crate::rental::RentalManager>> {
        let Some(metrics) = validator_metrics else {
            tracing::warn!("Rental manager disabled: metrics must be enabled for rentals");
            return Ok(None);
        };
        let manager = crate::rental::RentalManager::create(
            &self.config,
            persistence,
            metrics.prometheus(),
            collateral_manager,
            slash_executor,
            Some(validator_hotkey.as_str().to_string()),
        )
        .await?;

        manager.start();
        manager
            .initialize_rental_metrics()
            .await
            .context("Failed to initialize rental metrics")?;
        manager
            .initialize_node_metrics()
            .await
            .context("Failed to initialize node metrics")?;
        Ok(Some(manager))
    }

    async fn spawn_tasks(&self, inputs: TaskInputs) -> RuntimeHandles {
        let weight_setter_clone = inputs.weight_setter.clone();
        let scoring_task = tokio::spawn(async move {
            let mut interval = tokio::time::interval(StdDuration::from_secs(300));
            loop {
                interval.tick().await;
                if let Err(e) = weight_setter_clone.update_all_miner_scores().await {
                    error!("Failed to update miner scores: {}", e);
                }
            }
        });

        let weight_setter_task = tokio::spawn(async move {
            if let Err(e) = inputs.weight_setter.start().await {
                error!("Weight setter task failed: {}", e);
            }
        });

        let miner_prover_task = tokio::spawn(async move {
            if let Err(e) = inputs.miner_prover.start().await {
                error!("Miner prover task failed: {}", e);
            }
        });

        let api_handler_task = tokio::spawn(async move {
            if let Err(e) = inputs.api_handler.start().await {
                error!("API handler task failed: {}", e);
            }
        });

        let registration_grpc_config = self.config.bid_grpc.clone();
        let registration_persistence = inputs.persistence.clone();
        let registration_bidding_config = self.config.bidding.clone();
        let registration_collateral_manager = inputs.collateral_manager.clone();
        let validator_ssh_public_key = inputs.validator_ssh_public_key.clone();
        let registration_api_client = inputs.api_client.clone();
        let registration_server_task = tokio::spawn(async move {
            if let Err(e) = start_registration_server(
                registration_grpc_config,
                registration_persistence,
                registration_bidding_config,
                registration_collateral_manager,
                validator_ssh_public_key,
                Some(registration_api_client),
            )
            .await
            {
                error!("Registration gRPC server failed: {}", e);
            }
        });

        let cleanup_task = if self.config.cleanup.enabled {
            let cleanup_config = self.config.cleanup.clone();
            Some(tokio::spawn(async move {
                let cleanup_task = CleanupTask::new(cleanup_config, inputs.gpu_profile_repo);
                if let Err(e) = cleanup_task.start().await {
                    error!("Database cleanup task failed: {}", e);
                }
            }))
        } else {
            info!("Database cleanup task is disabled");
            None
        };

        let collateral_scan_task = if let Some(collateral_config) = self.config.collateral.clone() {
            let verification_config = self.config.verification.clone();
            let persistence = inputs.persistence.clone();
            tokio::spawn(async move {
                let mut collateral_scan =
                    Collateral::new(verification_config, collateral_config, persistence);
                if let Err(e) = collateral_scan.start().await {
                    error!("Collateral scan task failed: {}", e);
                }
            })
        } else {
            tokio::spawn(async { info!("Collateral scan disabled (no collateral config)") })
        };

        RuntimeHandles {
            scoring_task,
            weight_setter_task,
            miner_prover_task,
            api_handler_task,
            registration_server_task,
            cleanup_task,
            collateral_scan_task,
        }
    }

    fn shutdown(&self, handles: RuntimeHandles) {
        handles.scoring_task.abort();
        handles.weight_setter_task.abort();
        handles.miner_prover_task.abort();
        if let Some(handle) = handles.cleanup_task {
            handle.abort();
        }
        handles.api_handler_task.abort();
        handles.registration_server_task.abort();
        handles.collateral_scan_task.abort();
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
            .timeout(StdDuration::from_secs(10))
            .send()
            .await?;

        let elapsed = start_time.elapsed().unwrap_or(StdDuration::from_secs(0));

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
        let shutdown_timeout = StdDuration::from_secs(30);
        let shutdown_start = SystemTime::now();

        let mut remaining_processes = processes.clone();

        while !remaining_processes.is_empty()
            && shutdown_start
                .elapsed()
                .unwrap_or(StdDuration::from_secs(0))
                < shutdown_timeout
        {
            tokio::time::sleep(StdDuration::from_millis(1000)).await;

            remaining_processes.retain(|&pid| matches!(Self::is_process_running(pid), Ok(true)));
        }

        // Force kill remaining processes if necessary
        if !remaining_processes.is_empty() {
            for &pid in &remaining_processes {
                let _ = Self::send_signal_to_process(pid, Signal::Kill);
                tokio::time::sleep(StdDuration::from_millis(500)).await;
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
