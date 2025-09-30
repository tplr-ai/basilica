//! # Verification Scheduler
//!
//! Manages the scheduling and lifecycle of verification tasks.
//! Implements Single Responsibility Principle by focusing only on task scheduling.

use super::discovery::MinerDiscovery;
use super::types::{MinerInfo, ValidationType};
use super::verification::VerificationEngine;
use crate::config::VerificationConfig;
use anyhow::Result;
use futures::stream::{self, StreamExt};
use std::collections::HashMap;
use std::sync::Arc;
use std::time::Duration;
use tokio::sync::RwLock;
use tokio::task::JoinHandle;
use tokio::time::interval;
use tracing::{debug, error, info, warn};
use uuid::Uuid;

#[derive(Clone)]
struct SchedulerSharedState {
    verification_handles:
        Arc<RwLock<HashMap<Uuid, JoinHandle<Result<super::verification::VerificationResult>>>>>,
    active_full_tasks: Arc<RwLock<HashMap<Uuid, VerificationTask>>>,
    active_lightweight_tasks: Arc<RwLock<HashMap<Uuid, VerificationTask>>>,
}

impl SchedulerSharedState {
    fn new() -> Self {
        Self {
            verification_handles: Arc::new(RwLock::new(HashMap::new())),
            active_full_tasks: Arc::new(RwLock::new(HashMap::new())),
            active_lightweight_tasks: Arc::new(RwLock::new(HashMap::new())),
        }
    }
}

pub struct VerificationScheduler {
    config: VerificationConfig,
    shared_state: SchedulerSharedState,
}

impl VerificationScheduler {
    pub fn new(config: VerificationConfig) -> Self {
        info!("[EVAL_FLOW] Initializing VerificationScheduler");
        Self {
            config,
            shared_state: SchedulerSharedState::new(),
        }
    }

    /// Start the verification scheduling loop
    pub async fn start(
        self,
        discovery: MinerDiscovery,
        mut verification: VerificationEngine,
    ) -> Result<()> {
        let shared_state = self.shared_state.clone();
        let config = self.config.clone();
        let discovery = Arc::new(discovery);

        info!("Initializing validation binary server");
        verification
            .initialize_validation_server()
            .await
            .map_err(|e| {
                error!("Failed to initialize validation server: {}", e);
                e
            })?;
        info!("Validation binary server initialized successfully");

        // Initialize worker queue if enabled
        if config.enable_worker_queue {
            info!("Worker queue is enabled in config - initializing worker queue for decoupled execution");
            verification.init_worker_queue().await.map_err(|e| {
                error!("Failed to initialize worker queue: {}", e);
                e
            })?;
            info!("Worker queue initialized successfully");
        } else {
            info!("Worker queue is disabled in config - using direct execution mode");
        }

        let verification = Arc::new(verification);

        info!("Starting verification scheduler");
        info!(
            "Verification interval: {}s, Cleanup interval: {}s",
            config.verification_interval.as_secs(),
            900
        );

        // Full validation loop
        let full_shared_state = shared_state.clone();
        let full_config = config.clone();
        let full_discovery = discovery.clone();
        let full_verification = verification.clone();
        let full_loop = tokio::spawn(async move {
            let mut full_interval = interval(full_config.verification_interval);
            loop {
                tokio::select! {
                    _ = full_interval.tick() => {
                        if let Err(e) = run_full_validation(
                            &full_shared_state,
                            &full_config,
                            &full_discovery,
                            &full_verification,
                        ).await {
                            error!("Full validation cycle failed: {}", e);
                        }
                    }
                }
            }
        });

        // Lightweight validation loop
        let lightweight_shared_state = shared_state.clone();
        let lightweight_config = config.clone();
        let lightweight_discovery = discovery.clone();
        let lightweight_verification = verification.clone();
        let lightweight_loop = tokio::spawn(async move {
            let mut lightweight_interval = interval(lightweight_config.verification_interval);
            loop {
                tokio::select! {
                    _ = lightweight_interval.tick() => {
                        if let Err(e) = run_lightweight_validation(
                            &lightweight_shared_state,
                            &lightweight_config,
                            &lightweight_discovery,
                            &lightweight_verification,
                        ).await {
                            error!("Lightweight validation cycle failed: {}", e);
                        }
                    }
                }
            }
        });

        // Cleanup loop
        let cleanup_shared_state = shared_state.clone();
        let cleanup_verification = verification.clone();
        let cleanup_loop = tokio::spawn(async move {
            let mut cleanup_interval = tokio::time::interval(Duration::from_secs(900));
            loop {
                tokio::select! {
                    _ = cleanup_interval.tick() => {
                        cleanup_completed_verification_handles(&cleanup_shared_state).await;

                        info!("Running scheduled executor cleanup for failed executors");
                        match cleanup_verification.cleanup_failed_executors_after_failures(2).await {
                            Ok(()) => info!("Executor cleanup completed successfully"),
                            Err(e) => error!("Failed executor cleanup failed: {}", e),
                        }
                    }
                }
            }
        });

        // Run all loops concurrently
        let (full_result, lightweight_result, cleanup_result) =
            tokio::join!(full_loop, lightweight_loop, cleanup_loop);

        if let Err(e) = full_result {
            error!("Full validation loop panicked: {}", e);
        }
        if let Err(e) = lightweight_result {
            error!("Lightweight validation loop panicked: {}", e);
        }
        if let Err(e) = cleanup_result {
            error!("Cleanup loop panicked: {}", e);
        }

        Ok(())
    }
}

async fn run_full_validation(
    shared_state: &SchedulerSharedState,
    config: &VerificationConfig,
    discovery: &MinerDiscovery,
    verification: &VerificationEngine,
) -> Result<()> {
    info!("[EVAL_FLOW] Starting full validation cycle");
    let cycle_start = std::time::Instant::now();

    let discovered_miners = discovery.get_miners_for_verification().await?;
    if discovered_miners.is_empty() {
        return Ok(());
    }

    verification
        .sync_miners_from_metagraph(&discovered_miners)
        .await?;

    let schedulable_miners: Vec<MinerInfo> = discovered_miners
        .into_iter()
        .filter(|miner| can_schedule(shared_state, miner, ValidationType::Full))
        .collect();

    if schedulable_miners.is_empty() {
        return Ok(());
    }

    let full_tasks = spawn_validation_pipeline(
        shared_state,
        config,
        schedulable_miners,
        verification,
        ValidationType::Full,
    )
    .await?;

    info!(
        "[EVAL_FLOW] Full validation cycle completed in {:?}: {} tasks",
        cycle_start.elapsed(),
        full_tasks
    );

    Ok(())
}

async fn run_lightweight_validation(
    shared_state: &SchedulerSharedState,
    config: &VerificationConfig,
    discovery: &MinerDiscovery,
    verification: &VerificationEngine,
) -> Result<()> {
    info!("[EVAL_FLOW] Starting lightweight validation cycle");
    let cycle_start = std::time::Instant::now();

    let discovered_miners = discovery.get_miners_for_verification().await?;
    if discovered_miners.is_empty() {
        return Ok(());
    }

    verification
        .sync_miners_from_metagraph(&discovered_miners)
        .await?;

    let schedulable_miners: Vec<MinerInfo> = discovered_miners
        .into_iter()
        .filter(|miner| can_schedule(shared_state, miner, ValidationType::Lightweight))
        .collect();

    if schedulable_miners.is_empty() {
        return Ok(());
    }

    let lightweight_tasks = spawn_validation_pipeline(
        shared_state,
        config,
        schedulable_miners,
        verification,
        ValidationType::Lightweight,
    )
    .await?;

    info!(
        "[EVAL_FLOW] Lightweight validation cycle completed in {:?}: {} tasks",
        cycle_start.elapsed(),
        lightweight_tasks
    );

    Ok(())
}

async fn spawn_validation_pipeline(
    shared_state: &SchedulerSharedState,
    config: &VerificationConfig,
    miners: Vec<MinerInfo>,
    verification: &VerificationEngine,
    intended_strategy: ValidationType,
) -> Result<usize> {
    info!(
        intended_strategy = ?intended_strategy,
        "[EVAL_FLOW] Starting {:?} validation pipeline for {} miners",
        intended_strategy,
        miners.len()
    );

    let concurrency = config
        .max_concurrent_verifications
        .max(50)
        .min(miners.len());

    let results: Vec<_> = stream::iter(miners)
        .map(|miner| {
            let shared_state = shared_state.clone();
            let verification = verification.clone();
            let config = config.clone();

            async move {
                let miner_uid = miner.uid.as_u16();
                let verification_task = VerificationTask {
                    miner_uid,
                    miner_hotkey: miner.hotkey.to_string(),
                    miner_endpoint: miner.endpoint.clone(),
                    stake_tao: miner.stake_tao,
                    is_validator: miner.is_validator,
                    verification_type: VerificationType::AutomatedWithSsh,
                    intended_validation_strategy: intended_strategy,
                    created_at: chrono::Utc::now(),
                    timeout: config.challenge_timeout,
                };

                let result =
                    spawn_verification_task(&shared_state, verification_task, &verification).await;
                (miner_uid, intended_strategy, result)
            }
        })
        .buffer_unordered(concurrency)
        .collect()
        .await;

    // Count successful spawns and log failures
    let mut tasks_spawned = 0;
    for (miner_uid, strategy, result) in results {
        match result {
            Ok(_) => tasks_spawned += 1,
            Err(e) => warn!(
                miner_uid = miner_uid,
                intended_strategy = ?strategy,
                error = %e,
                "[EVAL_FLOW] Failed to spawn {:?} validation task",
                strategy
            ),
        }
    }

    info!(
        intended_strategy = ?intended_strategy,
        "[EVAL_FLOW] {:?} validation pipeline spawned {} tasks",
        intended_strategy, tasks_spawned
    );

    Ok(tasks_spawned)
}

async fn spawn_verification_task(
    shared_state: &SchedulerSharedState,
    task: VerificationTask,
    verification: &VerificationEngine,
) -> Result<()> {
    let verification_engine = verification.clone();
    let task_id = uuid::Uuid::new_v4();
    let has_worker_queue = verification_engine.has_worker_queue();
    let active_tasks = match task.intended_validation_strategy {
        ValidationType::Full => &shared_state.active_full_tasks,
        ValidationType::Lightweight => &shared_state.active_lightweight_tasks,
    };

    info!(
        miner_uid = task.miner_uid,
        task_id = %task_id,
        miner_endpoint = %task.miner_endpoint,
        verification_type = ?task.verification_type,
        intended_strategy = ?task.intended_validation_strategy,
        has_worker_queue = has_worker_queue,
        "[EVAL_FLOW] Preparing to spawn verification task"
    );

    {
        let mut tasks_map = active_tasks.write().await;
        tasks_map.insert(task_id, task.clone());
        info!(
            "[EVAL_FLOW] Active {:?} validation tasks count: {}",
            task.intended_validation_strategy,
            tasks_map.len()
        );
    }

    let verification_handle = tokio::spawn(async move {
        let workflow_start = std::time::Instant::now();
        info!(
            miner_uid = task.miner_uid,
            task_id = %task_id,
            miner_endpoint = %task.miner_endpoint,
            verification_type = ?task.verification_type,
            intended_strategy = ?task.intended_validation_strategy,
            workflow_start = ?workflow_start,
            has_worker_queue = has_worker_queue,
            "[EVAL_FLOW] Starting verification workflow",
        );

        let result = verification_engine
            .execute_verification_workflow(&task)
            .await;

        match result {
            Ok(verification_result) => {
                info!(
                    miner_uid = task.miner_uid,
                    task_id = %task_id,
                    "[EVAL_FLOW] Automated verification completed for miner {} in {:?}: score={:.2} (task: {})",
                    task.miner_uid, workflow_start.elapsed(), verification_result.overall_score, task_id
                );
                debug!(
                    miner_uid = task.miner_uid,
                    task_id = %task_id,
                    "[EVAL_FLOW] Verification steps completed: {}",
                    verification_result.verification_steps.len()
                );
                for step in &verification_result.verification_steps {
                    debug!(
                        miner_uid = task.miner_uid,
                        task_id = %task_id,
                        "[EVAL_FLOW]   Step: {} - {:?} - {}",
                        step.step_name, step.status, step.details
                    );
                }
                Ok(verification_result)
            }
            Err(e) => {
                error!(
                    miner_uid = task.miner_uid,
                    task_id = %task_id,
                    "[EVAL_FLOW] Automated verification failed for miner {} after {:?} (task: {}): {}",
                    task.miner_uid, workflow_start.elapsed(), task_id, e
                );
                Err(e)
            }
        }
    });

    {
        let mut verification_handles = shared_state.verification_handles.write().await;
        verification_handles.insert(task_id, verification_handle);
        debug!(
            task_id = %task_id,
            "[EVAL_FLOW] Tracking new verification handle for task {}",
            task_id
        );
        info!(
            "[EVAL_FLOW] Total verification handles tracked: {}",
            verification_handles.len()
        );
    }

    Ok(())
}

fn can_schedule(
    shared_state: &SchedulerSharedState,
    miner: &MinerInfo,
    strategy: ValidationType,
) -> bool {
    let miner_uid = miner.uid.as_u16();

    let active_tasks = match strategy {
        ValidationType::Full => &shared_state.active_full_tasks,
        ValidationType::Lightweight => &shared_state.active_lightweight_tasks,
    };

    if let Ok(tasks_map) = active_tasks.try_read() {
        if tasks_map.values().any(|t| t.miner_uid == miner_uid) {
            debug!(
                miner_uid = miner_uid,
                intended_strategy = ?strategy,
                "[EVAL_FLOW] Cannot schedule {:?} validation for miner {} - already active",
                strategy, miner_uid
            );
            return false;
        }
    }

    true
}

async fn cleanup_completed_verification_handles(shared_state: &SchedulerSharedState) {
    let mut to_remove: Vec<Uuid> = Vec::new();
    {
        let handles = shared_state.verification_handles.read().await;
        for (id, handle) in handles.iter() {
            if handle.is_finished() {
                to_remove.push(*id);
            }
        }
    }
    if !to_remove.is_empty() {
        let mut full_tasks = shared_state.active_full_tasks.write().await;
        let mut lightweight_tasks = shared_state.active_lightweight_tasks.write().await;
        let mut handles = shared_state.verification_handles.write().await;
        for id in &to_remove {
            if let Some(h) = handles.remove(id) {
                if let Err(e) = h.await {
                    error!(
                        task_id = %id,
                        "[EVAL_FLOW] Verification task {} panicked: {}", id, e
                    );
                }
            }
            full_tasks.remove(id);
            lightweight_tasks.remove(id);
            debug!(
                task_id = %id,
                "[EVAL_FLOW] Removed completed verification task {} from active tracking",
                id
            );
        }
        debug!(
            "[EVAL_FLOW] Cleaned up {} completed verification tasks",
            to_remove.len()
        );
    }
}

/// Enhanced verification task structure
#[derive(Debug, Clone)]
pub struct VerificationTask {
    pub miner_uid: u16,
    pub miner_hotkey: String,
    pub miner_endpoint: String,
    pub stake_tao: f64,
    pub is_validator: bool,
    pub verification_type: VerificationType,
    pub intended_validation_strategy: ValidationType,
    pub created_at: chrono::DateTime<chrono::Utc>,
    pub timeout: std::time::Duration,
}

/// Verification type specification
#[derive(Debug, Clone)]
pub enum VerificationType {
    Manual,
    AutomatedWithSsh,
    ScheduledRoutine,
}

#[cfg(test)]
mod tests {
    use super::*;
    use std::time::Duration;

    fn create_test_config(max_concurrent_full_validations: usize) -> VerificationConfig {
        VerificationConfig {
            verification_interval: Duration::from_secs(60),
            max_concurrent_verifications: 50,
            max_concurrent_full_validations,
            challenge_timeout: Duration::from_secs(120),
            min_score_threshold: 0.1,
            max_miners_per_round: 20,
            min_verification_interval: Duration::from_secs(1800),
            netuid: 39,
            use_dynamic_discovery: true,
            discovery_timeout: Duration::from_secs(30),
            fallback_to_static: true,
            cache_miner_info_ttl: Duration::from_secs(300),
            grpc_port_offset: None,
            binary_validation: crate::config::BinaryValidationConfig::default(),
            docker_validation: crate::config::DockerValidationConfig::default(),
            collateral_event_scan_interval: Duration::from_secs(12),
            executor_validation_interval: Duration::from_secs(6 * 3600),
            gpu_assignment_cleanup_ttl: Some(Duration::from_secs(120 * 60)),
            enable_worker_queue: false,
            storage_validation: crate::config::StorageValidationConfig::default(),
        }
    }

    #[tokio::test]
    async fn test_scheduler_shared_state_initialization() {
        let shared_state = SchedulerSharedState::new();

        // Verify components are initialized
        assert_eq!(shared_state.verification_handles.read().await.len(), 0);
        assert_eq!(shared_state.active_full_tasks.read().await.len(), 0);
        assert_eq!(shared_state.active_lightweight_tasks.read().await.len(), 0);
    }

    #[tokio::test]
    async fn test_verification_scheduler_initialization() {
        let config = create_test_config(2);
        let scheduler = VerificationScheduler::new(config.clone());

        // Verify scheduler initializes properly
        assert_eq!(scheduler.config.netuid, config.netuid);
    }
}
