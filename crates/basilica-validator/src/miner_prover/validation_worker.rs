use super::scheduler::VerificationTask;
use super::types::{NodeInfoDetailed, ValidationType};
use super::verification::VerificationEngine;
use anyhow::Result;
use basilica_common::identity::NodeId;
use lru::LruCache;
use std::cmp::Ordering;
use std::collections::{BTreeMap, HashMap};
use std::sync::atomic::{AtomicU64, AtomicUsize, Ordering as AtomicOrdering};
use std::sync::Arc;
use std::time::{Duration, Instant};
use tokio::sync::{Notify, RwLock};
use tokio::task::JoinHandle;
use tokio::time::{interval, timeout};
use tracing::{debug, error, info, warn};
use uuid::Uuid;

const DEFAULT_MAX_PENDING_ITEMS: usize = 1000;
const DEFAULT_MAX_PROCESSING_TIME_SECS: u64 = 2400; // 40 minutes
const DEFAULT_LIGHTWEIGHT_PROCESSING_TIME_SECS: u64 = 480; // 8 minutes
const DEFAULT_FULL_STALE_TIMEOUT_SECS: u64 = DEFAULT_MAX_PROCESSING_TIME_SECS + 600; // 50 minutes
const DEFAULT_LIGHTWEIGHT_STALE_TIMEOUT_SECS: u64 = DEFAULT_LIGHTWEIGHT_PROCESSING_TIME_SECS + 180; // 11 minutes
const DEFAULT_COMPLETED_ITEM_TTL_SECS: u64 = 3600; // 1 hour
const DEFAULT_RETRY_BACKOFF_BASE_SECS: u64 = 60;
const DEFAULT_MAX_RETRIES: u8 = 3;
const DEFAULT_HEALTH_CHECK_INTERVAL_SECS: u64 = 30;

#[derive(Debug, Clone)]
pub struct WorkerQueueConfig {
    pub max_pending_items: usize,
    pub max_processing_time_secs: u64,
    pub lightweight_processing_time_secs: u64,
    pub full_stale_timeout_secs: u64,
    pub lightweight_stale_timeout_secs: u64,
    pub completed_item_ttl_secs: u64,
    pub full_worker_count: usize,
    pub lightweight_worker_count: usize,
    pub retry_backoff_base_secs: u64,
    pub max_retries: u8,
    pub health_check_interval_secs: u64,
}

impl Default for WorkerQueueConfig {
    fn default() -> Self {
        Self {
            max_pending_items: DEFAULT_MAX_PENDING_ITEMS,
            max_processing_time_secs: DEFAULT_MAX_PROCESSING_TIME_SECS,
            lightweight_processing_time_secs: DEFAULT_LIGHTWEIGHT_PROCESSING_TIME_SECS,
            full_stale_timeout_secs: DEFAULT_FULL_STALE_TIMEOUT_SECS,
            lightweight_stale_timeout_secs: DEFAULT_LIGHTWEIGHT_STALE_TIMEOUT_SECS,
            completed_item_ttl_secs: DEFAULT_COMPLETED_ITEM_TTL_SECS,
            full_worker_count: 1,
            lightweight_worker_count: num_cpus::get().min(4),
            retry_backoff_base_secs: DEFAULT_RETRY_BACKOFF_BASE_SECS,
            max_retries: DEFAULT_MAX_RETRIES,
            health_check_interval_secs: DEFAULT_HEALTH_CHECK_INTERVAL_SECS,
        }
    }
}

#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum Priority {
    High = 0,
    Normal = 1,
    Low = 2,
}

impl PartialOrd for Priority {
    fn partial_cmp(&self, other: &Self) -> Option<Ordering> {
        Some(self.cmp(other))
    }
}

impl Ord for Priority {
    fn cmp(&self, other: &Self) -> Ordering {
        (*self as u8).cmp(&(*other as u8))
    }
}

#[derive(Debug, Clone)]
pub struct WorkItem {
    pub id: Uuid,
    pub miner_uid: u16,
    pub node_info: NodeInfoDetailed,
    pub task: VerificationTask,
    pub miner_hotkey: String,
    pub miner_endpoint: String,
    pub created_at: Instant,
    pub priority: Priority,
    pub retry_count: u8,
}

#[derive(Debug, Clone)]
struct ProcessingEntry {
    started_at: Instant,
    work_item: WorkItem,
}

#[derive(Debug, Clone)]
enum CompletionStatus {
    Success {
        completed_at: Instant,
    },
    Failed {
        completed_at: Instant,
        error: String,
    },
}

#[derive(Debug, Clone, PartialEq, Eq, Hash)]
struct NodeKey {
    miner_uid: u16,
    node_id: NodeId,
}

impl NodeKey {
    fn new(miner_uid: u16, node_id: NodeId) -> Self {
        Self { miner_uid, node_id }
    }
}

#[derive(Debug, Clone, PartialEq, Eq)]
struct QueueKey {
    priority: Priority,
    created_at: Instant,
    id: Uuid,
}

impl PartialOrd for QueueKey {
    fn partial_cmp(&self, other: &Self) -> Option<Ordering> {
        Some(self.cmp(other))
    }
}

impl Ord for QueueKey {
    fn cmp(&self, other: &Self) -> Ordering {
        self.priority
            .cmp(&other.priority)
            .then_with(|| self.created_at.cmp(&other.created_at))
            .then_with(|| self.id.cmp(&other.id))
    }
}

struct QueueState {
    pending: BTreeMap<QueueKey, WorkItem>,
    processing: HashMap<NodeKey, ProcessingEntry>,
    completed: LruCache<NodeKey, CompletionStatus>,
    node_index: HashMap<NodeKey, QueueKey>,
}

impl QueueState {
    fn new(completed_capacity: usize) -> Self {
        Self {
            pending: BTreeMap::new(),
            processing: HashMap::new(),
            completed: LruCache::new(std::num::NonZeroUsize::new(completed_capacity).unwrap()),
            node_index: HashMap::new(),
        }
    }

    fn is_duplicate(&self, key: &NodeKey) -> bool {
        self.node_index.contains_key(key) || self.processing.contains_key(key)
    }

    fn can_requeue(&self, key: &NodeKey, cooldown: Duration) -> bool {
        if let Some(status) = self.completed.peek(key) {
            match status {
                CompletionStatus::Success { completed_at, .. }
                | CompletionStatus::Failed { completed_at, .. } => {
                    completed_at.elapsed() > cooldown
                }
            }
        } else {
            true
        }
    }

    fn add_pending(&mut self, item: WorkItem) -> Result<()> {
        let node_key = NodeKey::new(item.miner_uid, item.node_info.id.clone());
        let queue_key = QueueKey {
            priority: item.priority,
            created_at: item.created_at,
            id: item.id,
        };

        if self.is_duplicate(&node_key) {
            return Err(anyhow::anyhow!("Duplicate work item"));
        }

        self.node_index.insert(node_key, queue_key.clone());
        self.pending.insert(queue_key, item);
        Ok(())
    }

    fn pop_next(&mut self) -> Option<WorkItem> {
        if let Some((_queue_key, work_item)) = self.pending.pop_first() {
            let node_key = NodeKey::new(work_item.miner_uid, work_item.node_info.id.clone());
            self.node_index.remove(&node_key);
            Some(work_item)
        } else {
            None
        }
    }

    fn mark_processing(&mut self, node_key: NodeKey, work_item: WorkItem) {
        self.processing.insert(
            node_key,
            ProcessingEntry {
                started_at: Instant::now(),
                work_item,
            },
        );
    }

    fn complete(&mut self, key: NodeKey, status: CompletionStatus) -> Option<ProcessingEntry> {
        let entry = self.processing.remove(&key);

        // Log failed completions with their error for diagnostics
        if let CompletionStatus::Failed { ref error, .. } = &status {
            debug!(
                miner_uid = key.miner_uid,
                node_id = %key.node_id,
                error = %error,
                "Task for node {} (miner {}) failed: {}",
                key.node_id,
                key.miner_uid,
                error
            );
        }

        self.completed.put(key, status);
        entry
    }

    fn rollback(&mut self, key: NodeKey) -> Option<ProcessingEntry> {
        self.processing.remove(&key)
    }

    fn rollback_for_retry(&mut self, key: NodeKey) -> Option<WorkItem> {
        self.processing.remove(&key).map(|entry| entry.work_item)
    }
}

pub struct QueueMetrics {
    pending_full: AtomicU64,
    pending_lightweight: AtomicU64,
    processing_full: AtomicU64,
    processing_lightweight: AtomicU64,
    completed_total: AtomicU64,
    failed_total: AtomicU64,
    retried_total: AtomicU64,
    avg_processing_time_ms: AtomicU64,
}

impl QueueMetrics {
    fn new() -> Self {
        Self {
            pending_full: AtomicU64::new(0),
            pending_lightweight: AtomicU64::new(0),
            processing_full: AtomicU64::new(0),
            processing_lightweight: AtomicU64::new(0),
            completed_total: AtomicU64::new(0),
            failed_total: AtomicU64::new(0),
            retried_total: AtomicU64::new(0),
            avg_processing_time_ms: AtomicU64::new(0),
        }
    }

    pub fn report(&self) {
        info!(
            pending_full = self.pending_full.load(AtomicOrdering::Relaxed),
            pending_lightweight = self.pending_lightweight.load(AtomicOrdering::Relaxed),
            processing_full = self.processing_full.load(AtomicOrdering::Relaxed),
            processing_lightweight = self.processing_lightweight.load(AtomicOrdering::Relaxed),
            completed_total = self.completed_total.load(AtomicOrdering::Relaxed),
            failed_total = self.failed_total.load(AtomicOrdering::Relaxed),
            retried_total = self.retried_total.load(AtomicOrdering::Relaxed),
            avg_processing_time_ms = self.avg_processing_time_ms.load(AtomicOrdering::Relaxed),
            "Queue metrics report"
        );
    }
}

pub struct ValidationWorkerQueue {
    config: WorkerQueueConfig,
    full_queue: Arc<RwLock<QueueState>>,
    lightweight_queue: Arc<RwLock<QueueState>>,
    full_workers: Arc<RwLock<Vec<JoinHandle<()>>>>,
    lightweight_workers: Arc<RwLock<Vec<JoinHandle<()>>>>,
    verification_engine: Arc<VerificationEngine>,
    metrics: Arc<QueueMetrics>,
    shutdown: Arc<Notify>,
    running: Arc<AtomicUsize>,
}

impl ValidationWorkerQueue {
    pub fn new(config: WorkerQueueConfig, verification_engine: Arc<VerificationEngine>) -> Self {
        let completed_capacity = 1000;

        Self {
            config,
            full_queue: Arc::new(RwLock::new(QueueState::new(completed_capacity))),
            lightweight_queue: Arc::new(RwLock::new(QueueState::new(completed_capacity))),
            full_workers: Arc::new(RwLock::new(Vec::new())),
            lightweight_workers: Arc::new(RwLock::new(Vec::new())),
            verification_engine,
            metrics: Arc::new(QueueMetrics::new()),
            shutdown: Arc::new(Notify::new()),
            running: Arc::new(AtomicUsize::new(0)),
        }
    }

    /// Helper method to create MinerInfo from WorkItem
    fn create_miner_info(
        item: &WorkItem,
        verification_score: f64,
    ) -> Result<super::types::MinerInfo> {
        let hotkey = basilica_common::identity::Hotkey::new(item.miner_hotkey.clone())
            .map_err(|e| anyhow::anyhow!("Invalid hotkey: {}", e))?;
        Ok(super::types::MinerInfo {
            uid: basilica_common::identity::MinerUid::new(item.miner_uid),
            hotkey,
            endpoint: item.miner_endpoint.clone(),
            stake_tao: item.task.stake_tao,
            is_validator: item.task.is_validator,
            last_verified: None,
            verification_score,
        })
    }

    /// Helper method to store a successful validation result
    async fn store_successful_validation(
        verification_engine: &Arc<VerificationEngine>,
        item: &WorkItem,
        node_key: &NodeKey,
        verification_result: &super::types::NodeVerificationResult,
    ) {
        match Self::create_miner_info(item, verification_result.verification_score) {
            Ok(miner_info) => {
                if let Err(e) = verification_engine
                    .store_node_verification_result_with_miner_info(
                        item.miner_uid,
                        verification_result,
                        &miner_info,
                    )
                    .await
                {
                    error!(
                        "Failed to store verification result for node {} (miner {}): {}",
                        node_key.node_id, node_key.miner_uid, e
                    );
                }
            }
            Err(e) => {
                error!(
                    "Invalid miner hotkey '{}' for miner {}: {}",
                    item.miner_hotkey, item.miner_uid, e
                );
            }
        }
    }

    /// Helper method to create and store a failed validation result
    async fn store_failed_validation(
        verification_engine: &Arc<VerificationEngine>,
        item: &WorkItem,
        node_key: &NodeKey,
        error_message: String,
        execution_time: Duration,
        validation_type: ValidationType,
    ) {
        // Create a failed verification result
        let failed_result = super::types::NodeVerificationResult {
            node_id: item.node_info.id.clone(),
            node_ssh_endpoint: item.node_info.node_ssh_endpoint.clone(),
            verification_score: 0.0,
            ssh_connection_successful: false,
            binary_validation_successful: false,
            node_result: None,
            error: Some(error_message.clone()),
            execution_time,
            validation_details: super::types::ValidationDetails {
                ssh_test_duration: Duration::from_secs(0),
                binary_upload_duration: Duration::from_secs(0),
                binary_execution_duration: Duration::from_secs(0),
                total_validation_duration: execution_time,
                ssh_score: 0.0,
                binary_score: 0.0,
                combined_score: 0.0,
            },
            gpu_count: 0,
            validation_type,
        };

        // Attempt to store the failed result
        match Self::create_miner_info(item, 0.0) {
            Ok(miner_info) => {
                if let Err(e) = verification_engine
                    .store_node_verification_result_with_miner_info(
                        item.miner_uid,
                        &failed_result,
                        &miner_info,
                    )
                    .await
                {
                    error!(
                        "Failed to store failed validation result for node {} (miner {}): {}",
                        node_key.node_id, node_key.miner_uid, e
                    );
                }
            }
            Err(e) => {
                error!(
                    "Invalid miner hotkey '{}' for miner {} (failed validation): {}",
                    item.miner_hotkey, item.miner_uid, e
                );
            }
        }
    }

    pub async fn start(&self) -> Result<()> {
        match self
            .running
            .compare_exchange(0, 1, AtomicOrdering::SeqCst, AtomicOrdering::SeqCst)
        {
            Ok(_) => {}
            Err(_) => return Err(anyhow::anyhow!("Worker queue already running")),
        }

        info!("Starting validation worker queue");

        self.start_full_workers().await?;
        self.start_lightweight_workers().await?;
        self.start_health_monitor().await?;

        info!(
            "Worker queue started with {} full workers and {} lightweight workers",
            self.config.full_worker_count, self.config.lightweight_worker_count
        );

        Ok(())
    }

    async fn start_full_workers(&self) -> Result<()> {
        let mut workers = self.full_workers.write().await;

        for worker_id in 0..self.config.full_worker_count {
            let worker = self.spawn_full_worker(worker_id).await?;
            workers.push(worker);
        }

        Ok(())
    }

    async fn start_lightweight_workers(&self) -> Result<()> {
        let mut workers = self.lightweight_workers.write().await;

        for worker_id in 0..self.config.lightweight_worker_count {
            let worker = self.spawn_lightweight_worker(worker_id).await?;
            workers.push(worker);
        }

        Ok(())
    }

    async fn spawn_full_worker(&self, worker_id: usize) -> Result<JoinHandle<()>> {
        let queue = self.full_queue.clone();
        let verification_engine = self.verification_engine.clone();
        let metrics = self.metrics.clone();
        let shutdown = self.shutdown.clone();
        let config = self.config.clone();
        let max_processing_time = Duration::from_secs(self.config.max_processing_time_secs);
        let worker_uuid = Uuid::new_v4();

        let handle = tokio::spawn(async move {
            info!("Full validation worker {} started", worker_id);

            loop {
                tokio::select! {
                    _ = shutdown.notified() => {
                        info!("Full validation worker {} shutting down", worker_id);
                        break;
                    }
                    _ = Self::process_full_validation_item(
                        queue.clone(),
                        verification_engine.clone(),
                        metrics.clone(),
                        worker_uuid,
                        max_processing_time,
                        config.clone(),
                    ) => {}
                }

                tokio::time::sleep(Duration::from_millis(100)).await;
            }
        });

        Ok(handle)
    }

    async fn spawn_lightweight_worker(&self, worker_id: usize) -> Result<JoinHandle<()>> {
        let queue = self.lightweight_queue.clone();
        let verification_engine = self.verification_engine.clone();
        let metrics = self.metrics.clone();
        let shutdown = self.shutdown.clone();
        let config = self.config.clone();
        let max_processing_time = Duration::from_secs(self.config.lightweight_processing_time_secs);
        let worker_uuid = Uuid::new_v4();

        let handle = tokio::spawn(async move {
            info!("Lightweight validation worker {} started", worker_id);

            loop {
                tokio::select! {
                    _ = shutdown.notified() => {
                        info!("Lightweight validation worker {} shutting down", worker_id);
                        break;
                    }
                    _ = Self::process_lightweight_validation_item(
                        queue.clone(),
                        verification_engine.clone(),
                        metrics.clone(),
                        worker_uuid,
                        max_processing_time,
                        config.clone(),
                    ) => {}
                }

                tokio::time::sleep(Duration::from_millis(50)).await;
            }
        });

        Ok(handle)
    }

    async fn process_full_validation_item(
        queue: Arc<RwLock<QueueState>>,
        verification_engine: Arc<VerificationEngine>,
        metrics: Arc<QueueMetrics>,
        worker_id: Uuid,
        max_processing_time: Duration,
        config: WorkerQueueConfig,
    ) {
        let work_item = {
            let mut state = queue.write().await;
            state.pop_next()
        };

        if let Some(item) = work_item {
            let node_key = NodeKey::new(item.miner_uid, item.node_info.id.clone());

            metrics
                .processing_full
                .fetch_add(1, AtomicOrdering::Relaxed);
            metrics.pending_full.fetch_sub(1, AtomicOrdering::Relaxed);

            debug!(
                miner_uid = node_key.miner_uid,
                node_id = %node_key.node_id,
                worker_id = %worker_id,
                "Full worker {:?} processing node {} for miner {}",
                worker_id, node_key.node_id, node_key.miner_uid
            );

            {
                let mut state = queue.write().await;
                state.mark_processing(node_key.clone(), item.clone());
            }

            // Clear in-queue state now that processing has started
            // The validation strategy will set appropriate states during verification
            if let Some(ref metrics) = verification_engine.get_metrics().await {
                metrics.prometheus().clear_node_validation_states(
                    &node_key.node_id.to_string(),
                    node_key.miner_uid,
                    ValidationType::Full,
                );
            }

            let start_time = Instant::now();
            let result = timeout(
                max_processing_time,
                verification_engine.verify_node(
                    &item.task.miner_endpoint,
                    &item.node_info,
                    item.miner_uid,
                    &item.task.miner_hotkey,
                    item.task.intended_validation_strategy,
                ),
            )
            .await;

            let mut state = queue.write().await;

            match result {
                Ok(Ok(verification_result)) => {
                    let status = CompletionStatus::Success {
                        completed_at: Instant::now(),
                    };
                    state.complete(node_key.clone(), status);

                    metrics
                        .completed_total
                        .fetch_add(1, AtomicOrdering::Relaxed);
                    metrics
                        .processing_full
                        .fetch_sub(1, AtomicOrdering::Relaxed);

                    let duration_ms = start_time.elapsed().as_millis() as u64;
                    metrics
                        .avg_processing_time_ms
                        .store(duration_ms, AtomicOrdering::Relaxed);

                    info!(
                        miner_uid = node_key.miner_uid,
                        node_id = %node_key.node_id,
                        worker_id = %worker_id,
                        "Full validation completed for node {} (miner {}): score={:.2}",
                        node_key.node_id,
                        node_key.miner_uid,
                        verification_result.verification_score
                    );

                    // Store verification result to database (release lock before await)
                    drop(state);
                    Self::store_successful_validation(
                        &verification_engine,
                        &item,
                        &node_key,
                        &verification_result,
                    )
                    .await;
                }
                Ok(Err(e)) => {
                    // Use rollback_for_retry to get the work item with its current retry count
                    if let Some(mut retry_item) = state.rollback_for_retry(node_key.clone()) {
                        metrics
                            .processing_full
                            .fetch_sub(1, AtomicOrdering::Relaxed);

                        if retry_item.retry_count < config.max_retries {
                            retry_item.retry_count += 1;
                            retry_item.priority = Priority::Low;
                            drop(state);

                            let attempt = retry_item.retry_count as u32;
                            let factor = 1u64 << attempt.saturating_sub(1).min(63);
                            let mut delay_seconds =
                                config.retry_backoff_base_secs.saturating_mul(factor);
                            let cap = config.max_processing_time_secs / 2;
                            if cap > 0 {
                                delay_seconds = delay_seconds.min(cap);
                            }
                            tokio::time::sleep(Duration::from_secs(delay_seconds)).await;
                            Self::requeue_item(queue, retry_item, &metrics, ValidationType::Full)
                                .await;
                            metrics.retried_total.fetch_add(1, AtomicOrdering::Relaxed);
                        } else {
                            let error_msg = format!(
                                "Verification failed after {} retries: {}",
                                config.max_retries, e
                            );
                            let status = CompletionStatus::Failed {
                                completed_at: Instant::now(),
                                error: error_msg.clone(),
                            };
                            state.complete(node_key.clone(), status);
                            metrics.failed_total.fetch_add(1, AtomicOrdering::Relaxed);

                            // Store failed validation result (release lock before await)
                            drop(state);
                            Self::store_failed_validation(
                                &verification_engine,
                                &retry_item,
                                &node_key,
                                error_msg,
                                start_time.elapsed(),
                                ValidationType::Full,
                            )
                            .await;
                        }
                    } else {
                        // Should not happen but handle gracefully
                        state.rollback(node_key.clone());
                        metrics
                            .processing_full
                            .fetch_sub(1, AtomicOrdering::Relaxed);
                    }
                }
                Err(_) => {
                    // Timeout case - use rollback_for_retry to get the work item
                    if let Some(mut retry_item) = state.rollback_for_retry(node_key.clone()) {
                        metrics
                            .processing_full
                            .fetch_sub(1, AtomicOrdering::Relaxed);

                        if retry_item.retry_count < config.max_retries {
                            retry_item.retry_count += 1;
                            retry_item.priority = Priority::Low;
                            drop(state);

                            let attempt = retry_item.retry_count as u32;
                            let factor = 1u64 << attempt.saturating_sub(1).min(63);
                            let mut delay_seconds =
                                config.retry_backoff_base_secs.saturating_mul(factor);
                            let cap = config.max_processing_time_secs / 2;
                            if cap > 0 {
                                delay_seconds = delay_seconds.min(cap);
                            }
                            tokio::time::sleep(Duration::from_secs(delay_seconds)).await;
                            Self::requeue_item(queue, retry_item, &metrics, ValidationType::Full)
                                .await;
                            metrics.retried_total.fetch_add(1, AtomicOrdering::Relaxed);
                        } else {
                            let error_msg = format!(
                                "Verification timed out after {}s ({} retries)",
                                config.max_processing_time_secs, config.max_retries
                            );
                            let status = CompletionStatus::Failed {
                                completed_at: Instant::now(),
                                error: error_msg.clone(),
                            };
                            state.complete(node_key.clone(), status);
                            metrics.failed_total.fetch_add(1, AtomicOrdering::Relaxed);

                            // Store failed validation result (release lock before await)
                            drop(state);
                            Self::store_failed_validation(
                                &verification_engine,
                                &retry_item,
                                &node_key,
                                error_msg,
                                start_time.elapsed(),
                                ValidationType::Full,
                            )
                            .await;
                        }
                    } else {
                        // Should not happen but handle gracefully
                        state.rollback(node_key.clone());
                        metrics
                            .processing_full
                            .fetch_sub(1, AtomicOrdering::Relaxed);
                    }
                }
            }
        }
    }

    async fn process_lightweight_validation_item(
        queue: Arc<RwLock<QueueState>>,
        verification_engine: Arc<VerificationEngine>,
        metrics: Arc<QueueMetrics>,
        worker_id: Uuid,
        max_processing_time: Duration,
        config: WorkerQueueConfig,
    ) {
        let work_item = {
            let mut state = queue.write().await;
            state.pop_next()
        };

        if let Some(item) = work_item {
            let node_key = NodeKey::new(item.miner_uid, item.node_info.id.clone());

            metrics
                .processing_lightweight
                .fetch_add(1, AtomicOrdering::Relaxed);
            metrics
                .pending_lightweight
                .fetch_sub(1, AtomicOrdering::Relaxed);

            debug!(
                miner_uid = node_key.miner_uid,
                node_id = %node_key.node_id,
                worker_id = %worker_id,
                "Lightweight worker {:?} processing node {} for miner {}",
                worker_id, node_key.node_id, node_key.miner_uid
            );

            {
                let mut state = queue.write().await;
                state.mark_processing(node_key.clone(), item.clone());
            }

            // Clear in-queue state now that processing has started
            // The validation strategy will set appropriate states during verification
            if let Some(ref metrics) = verification_engine.get_metrics().await {
                metrics.prometheus().clear_node_validation_states(
                    &node_key.node_id.to_string(),
                    node_key.miner_uid,
                    ValidationType::Lightweight,
                );
            }

            let start_time = Instant::now();
            let result = timeout(
                max_processing_time,
                verification_engine.verify_node(
                    &item.task.miner_endpoint,
                    &item.node_info,
                    item.miner_uid,
                    &item.task.miner_hotkey,
                    item.task.intended_validation_strategy,
                ),
            )
            .await;

            let mut state = queue.write().await;

            match result {
                Ok(Ok(verification_result)) => {
                    let status = CompletionStatus::Success {
                        completed_at: Instant::now(),
                    };
                    state.complete(node_key.clone(), status);

                    metrics
                        .completed_total
                        .fetch_add(1, AtomicOrdering::Relaxed);
                    metrics
                        .processing_lightweight
                        .fetch_sub(1, AtomicOrdering::Relaxed);

                    let duration_ms = start_time.elapsed().as_millis() as u64;
                    metrics
                        .avg_processing_time_ms
                        .store(duration_ms, AtomicOrdering::Relaxed);

                    info!(
                        miner_uid = node_key.miner_uid,
                        node_id = %node_key.node_id,
                        worker_id = %worker_id,
                        "Lightweight validation completed for node {} (miner {}): score={:.2}",
                        node_key.node_id,
                        node_key.miner_uid,
                        verification_result.verification_score
                    );

                    // Store verification result to database (release lock before await)
                    drop(state);
                    Self::store_successful_validation(
                        &verification_engine,
                        &item,
                        &node_key,
                        &verification_result,
                    )
                    .await;
                }
                Ok(Err(e)) => {
                    // Use rollback_for_retry to get the work item with current retry count
                    if let Some(mut retry_item) = state.rollback_for_retry(node_key.clone()) {
                        metrics
                            .processing_lightweight
                            .fetch_sub(1, AtomicOrdering::Relaxed);

                        if retry_item.retry_count < config.max_retries {
                            retry_item.retry_count += 1;
                            drop(state);

                            let attempt = retry_item.retry_count as u32;
                            let factor = 1u64 << attempt.saturating_sub(1).min(63);
                            let mut delay_seconds =
                                config.retry_backoff_base_secs.saturating_mul(factor);
                            let cap = config.lightweight_processing_time_secs / 2;
                            if cap > 0 {
                                delay_seconds = delay_seconds.min(cap);
                            }
                            tokio::time::sleep(Duration::from_secs(delay_seconds)).await;
                            Self::requeue_item(
                                queue,
                                retry_item,
                                &metrics,
                                ValidationType::Lightweight,
                            )
                            .await;
                            metrics.retried_total.fetch_add(1, AtomicOrdering::Relaxed);
                        } else {
                            let error_msg = format!(
                                "Lightweight verification failed after {} retries: {}",
                                config.max_retries, e
                            );
                            let status = CompletionStatus::Failed {
                                completed_at: Instant::now(),
                                error: error_msg.clone(),
                            };
                            state.complete(node_key.clone(), status);
                            metrics.failed_total.fetch_add(1, AtomicOrdering::Relaxed);

                            // Store failed validation result (release lock before await)
                            drop(state);
                            Self::store_failed_validation(
                                &verification_engine,
                                &retry_item,
                                &node_key,
                                error_msg,
                                start_time.elapsed(),
                                ValidationType::Lightweight,
                            )
                            .await;
                        }
                    } else {
                        // Should not happen but handle gracefully
                        state.rollback(node_key.clone());
                        metrics
                            .processing_lightweight
                            .fetch_sub(1, AtomicOrdering::Relaxed);
                    }
                }
                Err(_) => {
                    // Timeout case - use rollback_for_retry to get the work item
                    if let Some(mut retry_item) = state.rollback_for_retry(node_key.clone()) {
                        metrics
                            .processing_lightweight
                            .fetch_sub(1, AtomicOrdering::Relaxed);

                        if retry_item.retry_count < config.max_retries {
                            retry_item.retry_count += 1;
                            drop(state);

                            let attempt = retry_item.retry_count as u32;
                            let factor = 1u64 << attempt.saturating_sub(1).min(63);
                            let mut delay_seconds =
                                config.retry_backoff_base_secs.saturating_mul(factor);
                            let cap = config.lightweight_processing_time_secs / 2;
                            if cap > 0 {
                                delay_seconds = delay_seconds.min(cap);
                            }
                            tokio::time::sleep(Duration::from_secs(delay_seconds)).await;
                            Self::requeue_item(
                                queue,
                                retry_item,
                                &metrics,
                                ValidationType::Lightweight,
                            )
                            .await;
                            metrics.retried_total.fetch_add(1, AtomicOrdering::Relaxed);
                        } else {
                            let error_msg = format!(
                                "Lightweight verification timed out after {}s ({} retries)",
                                config.lightweight_processing_time_secs, config.max_retries
                            );
                            let status = CompletionStatus::Failed {
                                completed_at: Instant::now(),
                                error: error_msg.clone(),
                            };
                            state.complete(node_key.clone(), status);
                            metrics.failed_total.fetch_add(1, AtomicOrdering::Relaxed);

                            // Store failed validation result (release lock before await)
                            drop(state);
                            Self::store_failed_validation(
                                &verification_engine,
                                &retry_item,
                                &node_key,
                                error_msg,
                                start_time.elapsed(),
                                ValidationType::Lightweight,
                            )
                            .await;
                        }
                    } else {
                        // Should not happen but handle gracefully
                        state.rollback(node_key.clone());
                        metrics
                            .processing_lightweight
                            .fetch_sub(1, AtomicOrdering::Relaxed);
                    }
                }
            }
        }
    }

    async fn requeue_item(
        queue: Arc<RwLock<QueueState>>,
        item: WorkItem,
        metrics: &QueueMetrics,
        validation_type: ValidationType,
    ) {
        let mut state = queue.write().await;
        if let Err(e) = state.add_pending(item) {
            warn!("Failed to requeue item: {}", e);
        } else {
            match validation_type {
                ValidationType::Full => metrics.pending_full.fetch_add(1, AtomicOrdering::Relaxed),
                ValidationType::Lightweight => metrics
                    .pending_lightweight
                    .fetch_add(1, AtomicOrdering::Relaxed),
            };
        }
    }

    async fn start_health_monitor(&self) -> Result<()> {
        let metrics = self.metrics.clone();
        let interval_secs = self.config.health_check_interval_secs;
        let full_stale_timeout_secs = self.config.full_stale_timeout_secs;
        let lightweight_stale_timeout_secs = self.config.lightweight_stale_timeout_secs;
        let full_queue = self.full_queue.clone();
        let lightweight_queue = self.lightweight_queue.clone();
        let shutdown = self.shutdown.clone();
        let verification_engine = self.verification_engine.clone();

        tokio::spawn(async move {
            let mut health_interval = interval(Duration::from_secs(interval_secs));

            loop {
                tokio::select! {
                    _ = shutdown.notified() => {
                        info!("Health monitor shutting down");
                        break;
                    }
                    _ = health_interval.tick() => {
                        Self::check_stale_items(
                            full_queue.clone(),
                            Duration::from_secs(full_stale_timeout_secs),
                            metrics.clone(),
                            ValidationType::Full,
                            verification_engine.clone()
                        ).await;
                        Self::check_stale_items(
                            lightweight_queue.clone(),
                            Duration::from_secs(lightweight_stale_timeout_secs),
                            metrics.clone(),
                            ValidationType::Lightweight,
                            verification_engine.clone()
                        ).await;
                        metrics.report();
                    }
                }
            }
        });

        Ok(())
    }

    async fn check_stale_items(
        queue: Arc<RwLock<QueueState>>,
        max_age: Duration,
        metrics: Arc<QueueMetrics>,
        validation_type: ValidationType,
        verification_engine: Arc<VerificationEngine>,
    ) {
        let now = Instant::now();

        // Collect stale items with their data
        let stale_items: Vec<(NodeKey, WorkItem, Duration)> = {
            let state = queue.read().await;
            state
                .processing
                .iter()
                .filter(|(_, entry)| now.duration_since(entry.started_at) > max_age)
                .map(|(key, entry)| {
                    (
                        key.clone(),
                        entry.work_item.clone(),
                        now.duration_since(entry.started_at),
                    )
                })
                .collect()
        };

        // Process each stale item
        for (key, work_item, elapsed) in stale_items {
            warn!("Removing stale processing item: {:?}", key);

            let error_msg = format!(
                "Processing stale after {}s (exceeded {}s timeout)",
                elapsed.as_secs(),
                max_age.as_secs()
            );

            // Update state
            {
                let mut state = queue.write().await;
                if let Some(_entry) = state.rollback(key.clone()) {
                    let status = CompletionStatus::Failed {
                        completed_at: now,
                        error: error_msg.clone(),
                    };
                    state.completed.put(key.clone(), status);

                    match validation_type {
                        ValidationType::Full => {
                            metrics
                                .processing_full
                                .fetch_sub(1, AtomicOrdering::Relaxed);
                        }
                        ValidationType::Lightweight => {
                            metrics
                                .processing_lightweight
                                .fetch_sub(1, AtomicOrdering::Relaxed);
                        }
                    }
                    metrics.failed_total.fetch_add(1, AtomicOrdering::Relaxed);
                }
            }

            // Store failed validation result (outside lock)
            Self::store_failed_validation(
                &verification_engine,
                &work_item,
                &key,
                error_msg,
                elapsed,
                validation_type,
            )
            .await;
        }
    }

    pub async fn publish(&self, node: NodeInfoDetailed, task: VerificationTask) -> Result<()> {
        let node_key = NodeKey::new(task.miner_uid, node.id.clone());
        let validation_type = task.intended_validation_strategy;

        let queue = match validation_type {
            ValidationType::Full => &self.full_queue,
            ValidationType::Lightweight => &self.lightweight_queue,
        };

        let mut state = queue.write().await;

        if state.is_duplicate(&node_key) {
            debug!(
                "Node {} for miner {} already queued, skipping",
                node_key.node_id, node_key.miner_uid
            );
            return Ok(());
        }

        let cooldown = Duration::from_secs(self.config.completed_item_ttl_secs);
        if !state.can_requeue(&node_key, cooldown) {
            debug!(
                "Node {} for miner {} in cooldown period, skipping",
                node_key.node_id, node_key.miner_uid
            );
            return Ok(());
        }

        if state.pending.len() >= self.config.max_pending_items {
            return Err(anyhow::anyhow!("Queue is full"));
        }

        let priority = match validation_type {
            ValidationType::Lightweight => Priority::High,
            ValidationType::Full => Priority::Normal,
        };

        let work_item = WorkItem {
            id: Uuid::new_v4(),
            miner_uid: task.miner_uid,
            node_info: node,
            miner_hotkey: task.miner_hotkey.clone(),
            miner_endpoint: task.miner_endpoint.clone(),
            task,
            created_at: Instant::now(),
            priority,
            retry_count: 0,
        };

        state.add_pending(work_item)?;

        match validation_type {
            ValidationType::Full => self
                .metrics
                .pending_full
                .fetch_add(1, AtomicOrdering::Relaxed),
            ValidationType::Lightweight => self
                .metrics
                .pending_lightweight
                .fetch_add(1, AtomicOrdering::Relaxed),
        };

        Ok(())
    }

    pub async fn shutdown(&self) -> Result<()> {
        if self.running.load(AtomicOrdering::SeqCst) == 0 {
            return Ok(());
        }

        info!("Shutting down validation worker queue");
        self.running.store(0, AtomicOrdering::SeqCst);
        self.shutdown.notify_waiters();

        let full_workers = self.full_workers.read().await;
        let lightweight_workers = self.lightweight_workers.read().await;

        for worker in full_workers.iter() {
            worker.abort();
        }

        for worker in lightweight_workers.iter() {
            worker.abort();
        }

        info!("Worker queue shutdown complete");
        Ok(())
    }

    pub fn metrics(&self) -> &QueueMetrics {
        &self.metrics
    }

    pub async fn pending_count(&self) -> (usize, usize) {
        let full = self.full_queue.read().await.pending.len();
        let lightweight = self.lightweight_queue.read().await.pending.len();
        (full, lightweight)
    }

    pub async fn processing_count(&self) -> (usize, usize) {
        let full = self.full_queue.read().await.processing.len();
        let lightweight = self.lightweight_queue.read().await.processing.len();
        (full, lightweight)
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::config::VerificationConfig;
    use crate::miner_prover::miner_client::MinerClientConfig;
    use crate::miner_prover::verification_engine_builder::VerificationEngineBuilder;
    use basilica_common::identity::Hotkey;

    fn create_test_node_info(id: &str) -> NodeInfoDetailed {
        NodeInfoDetailed {
            id: NodeId::new("test-node").unwrap(),
            miner_uid: basilica_common::identity::MinerUid::new(1),
            status: "online".to_string(),
            capabilities: vec!["gpu".to_string()],
            node_ssh_endpoint: format!("http://node-{}.test:8080", id),
        }
    }

    fn create_test_task(miner_uid: u16, validation_type: ValidationType) -> VerificationTask {
        VerificationTask {
            miner_uid,
            miner_hotkey: "5GrwvaEF5zXb26Fz9rcQpDWS57CtERHpNehXCPcNoHGKutQY".to_string(),
            miner_endpoint: format!("http://miner-{}.test:8091", miner_uid),
            stake_tao: 100.0,
            is_validator: false,
            verification_type: super::super::scheduler::VerificationType::AutomatedWithSsh,
            intended_validation_strategy: validation_type,
            created_at: chrono::Utc::now(),
            timeout: Duration::from_secs(300),
        }
    }

    #[tokio::test]
    async fn test_queue_deduplication() {
        use crate::config::{AutomaticVerificationConfig, SshSessionConfig};
        use crate::persistence::SimplePersistence;

        let config = WorkerQueueConfig::default();
        let verification_config = VerificationConfig::test_default();
        let automatic_config = AutomaticVerificationConfig::test_default();
        let ssh_config = SshSessionConfig::test_default();
        let _miner_client_config = MinerClientConfig::default();
        let validator_hotkey =
            Hotkey::new("5GrwvaEF5zXb26Fz9rcQpDWS57CtERHpNehXCPcNoHGKutQY".to_string()).unwrap();
        let persistence = Arc::new(SimplePersistence::for_testing().await.unwrap());

        let verification_engine = VerificationEngineBuilder::new(
            verification_config,
            automatic_config,
            ssh_config,
            validator_hotkey,
            persistence,
            None,
        )
        .build_for_testing()
        .await
        .unwrap();

        let queue = ValidationWorkerQueue::new(config, Arc::new(verification_engine));

        let node = create_test_node_info("exec1");
        let task = create_test_task(1, ValidationType::Lightweight);

        assert!(queue.publish(node.clone(), task.clone()).await.is_ok());

        assert!(queue.publish(node.clone(), task.clone()).await.is_ok());

        let (_, lightweight_pending) = queue.pending_count().await;
        assert_eq!(lightweight_pending, 1);
    }

    #[tokio::test]
    async fn test_priority_ordering() {
        let _config = WorkerQueueConfig::default();
        let completed_capacity = 100;
        let mut queue_state = QueueState::new(completed_capacity);

        let high_priority_item = WorkItem {
            id: Uuid::new_v4(),
            miner_uid: 1,
            node_info: create_test_node_info("high"),
            task: create_test_task(1, ValidationType::Lightweight),
            miner_hotkey: "5GrwvaEF5zXb26Fz9rcQpDWS57CtERHpNehXCPcNoHGKutQY".to_string(),
            miner_endpoint: "http://miner-1.test:8091".to_string(),
            created_at: Instant::now(),
            priority: Priority::High,
            retry_count: 0,
        };

        let normal_priority_item = WorkItem {
            id: Uuid::new_v4(),
            miner_uid: 2,
            node_info: create_test_node_info("normal"),
            task: create_test_task(2, ValidationType::Full),
            miner_hotkey: "5GrwvaEF5zXb26Fz9rcQpDWS57CtERHpNehXCPcNoHGKutQY".to_string(),
            miner_endpoint: "http://miner-2.test:8091".to_string(),
            created_at: Instant::now(),
            priority: Priority::Normal,
            retry_count: 0,
        };

        let low_priority_item = WorkItem {
            id: Uuid::new_v4(),
            miner_uid: 3,
            node_info: create_test_node_info("low"),
            task: create_test_task(3, ValidationType::Full),
            miner_hotkey: "5GrwvaEF5zXb26Fz9rcQpDWS57CtERHpNehXCPcNoHGKutQY".to_string(),
            miner_endpoint: "http://miner-3.test:8091".to_string(),
            created_at: Instant::now(),
            priority: Priority::Low,
            retry_count: 1,
        };

        queue_state.add_pending(low_priority_item).unwrap();
        queue_state.add_pending(normal_priority_item).unwrap();
        queue_state.add_pending(high_priority_item).unwrap();

        let first = queue_state.pop_next().unwrap();
        assert_eq!(first.priority, Priority::High);

        let second = queue_state.pop_next().unwrap();
        assert_eq!(second.priority, Priority::Normal);

        let third = queue_state.pop_next().unwrap();
        assert_eq!(third.priority, Priority::Low);
    }

    #[tokio::test]
    async fn test_completion_status_tracking() {
        let completed_capacity = 10;
        let mut queue_state = QueueState::new(completed_capacity);

        let node_key = NodeKey::new(1, NodeId::new("test-node").unwrap());

        let status = CompletionStatus::Success {
            completed_at: Instant::now(),
        };

        queue_state.completed.put(node_key.clone(), status);

        assert!(queue_state.completed.contains(&node_key));

        let cooldown = Duration::from_millis(1);
        tokio::time::sleep(Duration::from_millis(2)).await;
        assert!(queue_state.can_requeue(&node_key, cooldown));
    }

    #[tokio::test]
    async fn test_metrics_tracking() {
        let metrics = QueueMetrics::new();

        metrics.pending_full.store(5, AtomicOrdering::Relaxed);
        metrics
            .pending_lightweight
            .store(10, AtomicOrdering::Relaxed);
        metrics.completed_total.store(100, AtomicOrdering::Relaxed);

        assert_eq!(metrics.pending_full.load(AtomicOrdering::Relaxed), 5);
        assert_eq!(
            metrics.pending_lightweight.load(AtomicOrdering::Relaxed),
            10
        );
        assert_eq!(metrics.completed_total.load(AtomicOrdering::Relaxed), 100);
    }
}
