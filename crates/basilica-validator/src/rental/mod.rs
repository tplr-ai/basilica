//! Rental module for container deployment and management
//!
//! This module provides functionality for validators to rent GPU resources
//! and deploy containers on node machines.

use anyhow::Result;
use std::sync::Arc;
use uuid::Uuid;

pub mod billing;
pub mod container_client;
pub mod deployment;
pub mod monitoring;
pub mod types;

pub use billing::RentalBillingMonitor;
pub use container_client::ContainerClient;
pub use deployment::DeploymentManager;
pub use monitoring::{DatabaseHealthMonitor, LogStreamer};
pub use types::*;

use crate::ban_system::BanManager;
use crate::billing::BillingClient;
use crate::collateral::{CollateralManager, CollateralPreference};
use crate::metrics::ValidatorPrometheusMetrics;
use crate::persistence::entities::MisbehaviourType;
use crate::persistence::miner_nodes::NodeBidCandidate;
use crate::persistence::{SimplePersistence, ValidatorPersistence};
use crate::ssh::ValidatorSshKeyManager;

/// Rental manager for coordinating container deployments
pub struct RentalManager {
    /// Persistence layer
    persistence: Arc<SimplePersistence>,
    /// Deployment manager
    deployment_manager: Arc<DeploymentManager>,
    /// Log streamer
    log_streamer: Arc<LogStreamer>,
    /// Health monitor
    health_monitor: Arc<DatabaseHealthMonitor>,
    /// Billing telemetry monitor (optional)
    billing: Option<Arc<RentalBillingMonitor>>,
    /// SSH key manager for validator keys
    ssh_key_manager: Option<Arc<ValidatorSshKeyManager>>,
    /// Metrics for tracking rental status (required)
    metrics: Arc<ValidatorPrometheusMetrics>,
    /// Ban manager for logging misbehaviours
    ban_manager: Arc<BanManager>,
    /// Collateral manager for bid selection and eligibility
    collateral_manager: Option<Arc<CollateralManager>>,
    /// Max age for full validation before allowing a rental
    pre_rental_full_validation_max_age: std::time::Duration,
}

// /// Parse SSH host from credentials string format "user@host:port"
// fn parse_ssh_host(credentials: &str) -> Result<&str> {
//     let (_, host_port) = credentials
//         .split_once('@')
//         .context("Invalid SSH credentials format: missing '@' separator")?;

//     let host = host_port
//         .split(':')
//         .next()
//         .filter(|h| !h.is_empty())
//         .context("Invalid SSH credentials format: empty host")?;

//     Ok(host)
// }

/// Extract miner UID from miner_id format: "miner_{uid}"
pub(crate) fn extract_miner_uid(miner_id: &str) -> Option<u16> {
    if let Some(uid_str) = miner_id.strip_prefix("miner_") {
        return uid_str.parse().ok();
    }
    None
}

/// Get normalized GPU type from node details
pub(crate) fn get_gpu_type(node_details: &crate::api::types::NodeDetails) -> String {
    use basilica_common::types::GpuCategory;
    use std::str::FromStr;

    node_details
        .gpu_specs
        .first()
        .map(|gpu| {
            let category = GpuCategory::from_str(&gpu.name).unwrap();
            category.to_string()
        })
        .unwrap_or_else(|| "unknown".to_string())
}

struct RentalSelection {
    // Filled by bid selection
    node_id: String,
    miner_id: String,
    miner_uid: Option<u16>,
}

impl RentalManager {
    /// Helper function to create a ContainerClient with SSH credentials
    fn create_container_client(&self, ssh_credentials: &str) -> Result<ContainerClient> {
        let private_key_path = self
            .ssh_key_manager
            .as_ref()
            .and_then(|km| km.get_persistent_key())
            .map(|(_, path)| path.clone());

        ContainerClient::new(ssh_credentials.to_string(), private_key_path)
    }

    /// Cleanup container on rental setup failure
    async fn cleanup_container_on_failure(
        &self,
        ssh_credentials: &str,
        container_id: &str,
        node_id: &str,
        rental_id: &str,
    ) {
        tracing::warn!(
            node_id = %node_id,
            rental_id = %rental_id,
            container_id = %container_id,
            "Cleaning up container due to rental setup failure"
        );

        match self.create_container_client(ssh_credentials) {
            Ok(client) => {
                if let Err(e) = self
                    .deployment_manager
                    .stop_container(&client, container_id, true)
                    .await
                {
                    tracing::error!(
                        node_id = %node_id,
                        rental_id = %rental_id,
                        container_id = %container_id,
                        "Failed to cleanup container: {}",
                        e
                    );
                }
            }
            Err(e) => {
                tracing::error!(
                    node_id = %node_id,
                    rental_id = %rental_id,
                    "Failed to create SSH client for cleanup: {}",
                    e
                );
            }
        }
    }

    /// Create a new rental manager with SSH key manager
    pub fn new(
        persistence: Arc<SimplePersistence>,
        ssh_key_manager: Arc<ValidatorSshKeyManager>,
        metrics: Arc<ValidatorPrometheusMetrics>,
    ) -> Self {
        let deployment_manager = Arc::new(DeploymentManager::new());
        let log_streamer = Arc::new(LogStreamer::new());

        // Create ban manager
        let ban_manager = Arc::new(BanManager::new(
            persistence.clone(),
            Some(metrics.clone()),
            None,
            None,
        ));

        // Create health monitor with SSH key manager, metrics, and ban manager
        let health_monitor = Arc::new(DatabaseHealthMonitor::new(
            persistence.clone(),
            ssh_key_manager.clone(),
            metrics.clone(),
            ban_manager.clone(),
        ));

        Self {
            persistence,
            deployment_manager: deployment_manager.clone(),
            log_streamer: log_streamer.clone(),
            health_monitor,
            billing: None,
            ssh_key_manager: Some(ssh_key_manager),
            metrics,
            ban_manager,
            collateral_manager: None,
            // TODO: Wire this from config for callers using `new`.
            pre_rental_full_validation_max_age: std::time::Duration::from_secs(12 * 60 * 60),
        }
    }

    /// Create rental manager with all components (SSH, billing if enabled)
    /// Does NOT start monitoring loops - call start() separately
    pub async fn create(
        config: &crate::config::ValidatorConfig,
        persistence: Arc<SimplePersistence>,
        metrics: Arc<ValidatorPrometheusMetrics>,
        collateral_manager: Option<Arc<CollateralManager>>,
        slash_executor: Option<Arc<crate::collateral::SlashExecutor>>,
        validator_hotkey: Option<String>,
    ) -> Result<Self> {
        // Create SSH key manager
        let ssh_key_dir = config.ssh_session.ssh_key_directory.clone();
        let mut ssh_key_manager = ValidatorSshKeyManager::new(ssh_key_dir).await?;
        ssh_key_manager
            .load_or_generate_persistent_key(None)
            .await?;
        let ssh_key_manager = Arc::new(ssh_key_manager);

        // Create ban manager
        let ban_manager = Arc::new(BanManager::new(
            persistence.clone(),
            Some(metrics.clone()),
            slash_executor,
            validator_hotkey,
        ));

        // Create health monitor
        let health_monitor = Arc::new(DatabaseHealthMonitor::new(
            persistence.clone(),
            ssh_key_manager.clone(),
            metrics.clone(),
            ban_manager.clone(),
        ));

        // Create billing monitor if enabled
        let billing = if config.billing.enabled {
            let billing_client = Arc::new(
                BillingClient::new_with_metrics(config.billing.clone(), Some(metrics.clone()))
                    .await?,
            );
            billing_client.clone().start_streaming_task().await;

            Some(Arc::new(RentalBillingMonitor::new(
                persistence.clone(),
                ssh_key_manager.clone(),
                billing_client,
                &config.billing,
            )))
        } else {
            None
        };

        Ok(Self {
            persistence,
            deployment_manager: Arc::new(DeploymentManager::new()),
            log_streamer: Arc::new(LogStreamer::new()),
            health_monitor,
            billing,
            ssh_key_manager: Some(ssh_key_manager),
            metrics,
            ban_manager,
            collateral_manager,
            pre_rental_full_validation_max_age: config.verification.node_validation_interval,
        })
    }

    /// Start all monitoring loops (health + billing)
    pub fn start(&self) {
        self.health_monitor.start_monitoring_loop();
        if let Some(ref billing) = self.billing {
            billing.start();
        }
    }

    /// Get the validator's SSH public key for sharing with miners during registration.
    /// Panics if called before SSH key manager is initialized.
    pub fn get_validator_ssh_public_key(&self) -> String {
        self.ssh_key_manager
            .as_ref()
            .expect("SSH key manager not initialized")
            .get_ssh_public_key()
            .expect("SSH public key not generated")
    }

    /// Initialize metrics for all existing rentals on startup
    pub async fn initialize_rental_metrics(&self) -> Result<()> {
        // Query all non-terminal rentals from persistence
        let rentals = self.persistence.query_non_terminated_rentals().await?;

        let rental_count = rentals.len();

        for rental in rentals {
            let miner_uid = extract_miner_uid(&rental.miner_id);

            if let Some(miner_uid) = miner_uid {
                let gpu_type = get_gpu_type(&rental.node_details);

                // Set metric based on rental state
                let is_rented = matches!(
                    rental.state,
                    RentalState::Active | RentalState::Provisioning | RentalState::Stopping
                );

                self.metrics.record_node_rental_status(
                    &rental.node_id,
                    miner_uid,
                    &gpu_type,
                    is_rented,
                );

                tracing::info!(
                    "Initialized rental metric for node {} (state: {:?}, is_rented: {})",
                    rental.node_id,
                    rental.state,
                    is_rented
                );
            }
        }

        tracing::info!("Initialized metrics for {} existing rentals", rental_count);
        Ok(())
    }

    /// Initialize metrics for all nodes on startup
    pub async fn initialize_node_metrics(&self) -> Result<()> {
        use basilica_common::types::GpuCategory;
        use std::str::FromStr;

        // Get all nodes with their GPU and rental data in a single query
        let node_metrics = self.persistence.get_all_nodes_for_metrics().await?;

        let node_count = node_metrics.len();
        tracing::info!("Initializing metrics for {} nodes", node_count);

        for metric_data in node_metrics {
            // Convert GPU name to category
            let gpu_type = metric_data
                .gpu_name
                .and_then(|name| GpuCategory::from_str(&name).ok())
                .map(|category| category.to_string())
                .unwrap_or_else(|| "unknown".to_string());

            self.metrics.record_node_rental_status(
                &metric_data.node_id,
                metric_data.miner_uid,
                &gpu_type,
                metric_data.has_active_rental,
            );

            let _ = self
                .ban_manager
                .get_ban_expiry(metric_data.miner_uid, &metric_data.node_id)
                .await?;

            tracing::debug!(
                "Initialized node metric: node={}, miner_uid={}, gpu_type={}, is_rented={}",
                metric_data.node_id,
                metric_data.miner_uid,
                gpu_type,
                metric_data.has_active_rental
            );
        }

        tracing::info!("Successfully initialized metrics for {} nodes", node_count);
        Ok(())
    }

    /// Start a new rental
    pub async fn start_rental(&self, request: RentalRequest) -> Result<RentalResponse> {
        let rental_id = format!("rental-{}", Uuid::new_v4());

        // 1. Find winning bid (required - no fallback)
        let selection = self.select_node_from_bids(&request, &rental_id).await?;

        // 2. Validate node is not banned
        self.ensure_not_banned(&selection, &rental_id).await?;

        // 3. Ensure node has recent validation (PR #331 requirement)
        self.ensure_recent_validation(&selection, &rental_id)
            .await?;

        // 4. Get SSH endpoint and node details
        let ssh_endpoint = self.require_ssh_endpoint(&selection, &rental_id).await?;
        let node_details = self.require_node_details(&selection, &rental_id).await?;

        tracing::info!(
            node_id = %selection.node_id,
            rental_id = %rental_id,
            miner_uid = selection.miner_uid,
            "Starting rental {} on node {} (miner: {})",
            rental_id,
            selection.node_id,
            selection.miner_id
        );

        let ssh_credentials = self.build_ssh_credentials(&ssh_endpoint);
        let container_client = self.create_container_client(&ssh_credentials)?;

        // 5. Deploy container
        let container_info = self
            .deploy_container_or_log_failure(
                &container_client,
                &request,
                &rental_id,
                &selection,
                &ssh_credentials,
            )
            .await?;

        let end_user_ssh_credentials =
            self.resolve_end_user_ssh_credentials(&ssh_endpoint, &container_info);

        // 6. Finalize rental — on success, active_rental_id stays set for the
        // lifetime of the rental. On failure, release the node claim.
        let finalize_result = self
            .finalize_rental(
                &request,
                &rental_id,
                &selection,
                &ssh_credentials,
                &node_details,
                &container_info,
            )
            .await;

        let rental_info = match finalize_result {
            Ok(info) => info,
            Err(e) => {
                self.release_node(&selection, &rental_id).await;
                return Err(e);
            }
        };

        self.record_rental_metrics(&selection, &rental_info);

        tracing::info!(
            node_id = %selection.node_id,
            rental_id = %rental_id,
            miner_uid = selection.miner_uid,
            "Successfully started rental {} on node {} (miner: {})",
            rental_id,
            selection.node_id,
            selection.miner_id
        );

        Ok(RentalResponse {
            rental_id,
            ssh_credentials: end_user_ssh_credentials,
            container_info,
        })
    }

    /// Select a node from bids matching the request criteria
    async fn select_node_from_bids(
        &self,
        request: &RentalRequest,
        rental_id: &str,
    ) -> Result<RentalSelection> {
        // Query candidates directly from miner_nodes (no epoch lookup needed)
        let candidates = self
            .persistence
            .get_node_bid_candidates(
                &request.gpu_category,
                request.gpu_count,
                request.max_hourly_rate_cents,
                50,
            )
            .await?;

        // Apply min_memory_gb filter if specified
        let candidates = self
            .filter_by_min_memory(candidates, request.min_memory_gb)
            .await;

        // Rank by collateral preference
        let ordered = self
            .rank_bid_candidates(candidates, &request.gpu_category, request.gpu_count)
            .await;

        if ordered.is_empty() {
            return Err(anyhow::anyhow!(
                "No available nodes matching criteria: gpu_category={}, gpu_count={}",
                request.gpu_category,
                request.gpu_count
            ));
        }

        // Try to claim nodes (atomic: sets active_rental_id where NULL)
        const MAX_CLAIM_ATTEMPTS: usize = 3;
        for candidate in ordered.into_iter().take(MAX_CLAIM_ATTEMPTS) {
            if self
                .persistence
                .claim_node(&candidate.node_id, &candidate.miner_id, rental_id)
                .await?
            {
                return Ok(RentalSelection {
                    node_id: candidate.node_id,
                    miner_id: candidate.miner_id,
                    miner_uid: Some(candidate.miner_uid as u16),
                });
            }
        }

        Err(anyhow::anyhow!(
            "Failed to claim any matching node after {} attempts",
            MAX_CLAIM_ATTEMPTS
        ))
    }

    /// Release the node claim for the given selection.
    ///
    /// INVARIANT: This MUST be called on every error/termination path.
    /// See persistence::miner_nodes for the full invariant comment.
    async fn release_node(&self, selection: &RentalSelection, rental_id: &str) {
        let _ = self
            .persistence
            .release_node(&selection.node_id, &selection.miner_id, rental_id)
            .await;
    }

    /// Filter candidates by minimum GPU memory requirement
    async fn filter_by_min_memory(
        &self,
        candidates: Vec<NodeBidCandidate>,
        min_memory_gb: Option<u32>,
    ) -> Vec<NodeBidCandidate> {
        let Some(min_memory) = min_memory_gb else {
            return candidates;
        };

        if min_memory == 0 {
            return candidates;
        }

        let mut filtered = Vec::new();
        for candidate in candidates {
            match self
                .persistence
                .get_node_first_gpu_memory_gb(&candidate.miner_id, &candidate.node_id)
                .await
            {
                Ok(memory_gb) if memory_gb >= min_memory as f64 => {
                    filtered.push(candidate);
                }
                Ok(memory_gb) => {
                    tracing::debug!(
                        node_id = %candidate.node_id,
                        miner_uid = candidate.miner_uid,
                        gpu_memory_gb = memory_gb,
                        min_memory_gb = min_memory,
                        "Skipping node - GPU memory below minimum requirement"
                    );
                }
                Err(e) => {
                    tracing::warn!(
                        node_id = %candidate.node_id,
                        miner_uid = candidate.miner_uid,
                        error = %e,
                        "Failed to get GPU memory for node, excluding from candidates"
                    );
                }
            }
        }
        filtered
    }

    async fn rank_bid_candidates(
        &self,
        candidates: Vec<NodeBidCandidate>,
        gpu_category: &str,
        gpu_count: u32,
    ) -> Vec<NodeBidCandidate> {
        let mut preferred = Vec::new();
        let mut fallback = Vec::new();

        for candidate in candidates {
            let preference = match &self.collateral_manager {
                Some(collateral) => {
                    collateral
                        .get_preference(
                            &candidate.miner_hotkey,
                            &candidate.node_id,
                            gpu_category,
                            gpu_count,
                        )
                        .await
                }
                None => CollateralPreference::Fallback,
            };
            match preference {
                CollateralPreference::Preferred => preferred.push(candidate),
                CollateralPreference::Fallback => fallback.push(candidate),
                CollateralPreference::Excluded => {
                    tracing::info!(
                        node_id = %candidate.node_id,
                        miner_uid = candidate.miner_uid,
                        "Skipping excluded node due to insufficient collateral"
                    );
                }
            }
        }

        let mut ordered = preferred;
        ordered.extend(fallback);
        ordered
    }

    async fn ensure_not_banned(&self, selection: &RentalSelection, rental_id: &str) -> Result<()> {
        if let Some(miner_uid) = selection.miner_uid {
            if let Some(ban_expiry) = self
                .ban_manager
                .get_ban_expiry(miner_uid, &selection.node_id)
                .await?
            {
                self.release_node(selection, rental_id).await;
                tracing::warn!(
                    node_id = %selection.node_id,
                    miner_id = %selection.miner_id,
                    miner_uid = miner_uid,
                    ban_expiry = %ban_expiry,
                    "Attempted rental on a banned node; rejecting request"
                );
                return Err(anyhow::anyhow!(
                    "Node {} is currently banned. Ban expires at: {:?}",
                    selection.node_id,
                    ban_expiry
                ));
            }
        }
        Ok(())
    }

    async fn require_ssh_endpoint(
        &self,
        selection: &RentalSelection,
        rental_id: &str,
    ) -> Result<String> {
        let ssh_endpoint = self
            .persistence
            .get_node_ssh_endpoint(&selection.node_id, &selection.miner_id)
            .await?;
        match ssh_endpoint {
            Some(endpoint) => Ok(endpoint),
            None => {
                self.release_node(selection, rental_id).await;
                Err(anyhow::anyhow!(
                    "SSH endpoint not found for node {} (miner: {})",
                    selection.node_id,
                    selection.miner_id
                ))
            }
        }
    }

    async fn require_node_details(
        &self,
        selection: &RentalSelection,
        rental_id: &str,
    ) -> Result<crate::api::types::NodeDetails> {
        let node_details = self
            .persistence
            .get_node_details(&selection.node_id, &selection.miner_id)
            .await?;
        match node_details {
            Some(details) => Ok(details),
            None => {
                self.release_node(selection, rental_id).await;
                tracing::warn!(
                    node_id = %selection.node_id,
                    miner_uid = selection.miner_uid,
                    rental_id = %rental_id,
                    "Node details not found for node {} (miner: {})",
                    selection.node_id,
                    selection.miner_id
                );
                Err(anyhow::anyhow!(
                    "Node details not found for node {} (miner: {})",
                    selection.node_id,
                    selection.miner_id
                ))
            }
        }
    }

    async fn ensure_recent_validation(
        &self,
        selection: &RentalSelection,
        rental_id: &str,
    ) -> Result<()> {
        let Some(miner_uid) = selection.miner_uid else {
            return Ok(());
        };

        let last_full_validation = self
            .persistence
            .get_last_full_validation_timestamp(&selection.node_id, &selection.miner_id)
            .await
            .unwrap_or_else(|e| {
                tracing::warn!(
                    node_id = %selection.node_id,
                    miner_id = %selection.miner_id,
                    error = %e,
                    "Failed to read last full validation timestamp"
                );
                None
            });

        let is_stale = last_full_validation
            .map(|ts| {
                chrono::Utc::now() - ts
                    > chrono::Duration::from_std(self.pre_rental_full_validation_max_age)
                        .unwrap_or(chrono::Duration::hours(12))
            })
            .unwrap_or(true);

        if is_stale {
            self.release_node(selection, rental_id).await;
            // TODO: Consider auto-triggering a full validation on stale rentals instead of rejecting.
            return Err(anyhow::anyhow!(
                "Node {} requires a recent full validation before rental (miner_uid: {})",
                selection.node_id,
                miner_uid
            ));
        }

        Ok(())
    }

    fn build_ssh_credentials(&self, ssh_endpoint: &str) -> String {
        if ssh_endpoint.contains('@') {
            ssh_endpoint.to_string()
        } else {
            format!("root@{}", ssh_endpoint)
        }
    }

    fn resolve_end_user_ssh_credentials(
        &self,
        ssh_endpoint: &str,
        container_info: &ContainerInfo,
    ) -> Option<String> {
        container_info
            .mapped_ports
            .iter()
            .find(|p| p.container_port == 22)
            .map(|ssh_mapping| {
                let host = if ssh_endpoint.contains('@') {
                    ssh_endpoint
                        .split('@')
                        .nth(1)
                        .and_then(|hp| hp.split(':').next())
                        .unwrap_or("localhost")
                } else {
                    ssh_endpoint.split(':').next().unwrap_or("localhost")
                };
                format!("root@{}:{}", host, ssh_mapping.host_port)
            })
    }

    async fn deploy_container_or_log_failure(
        &self,
        container_client: &ContainerClient,
        request: &RentalRequest,
        rental_id: &str,
        selection: &RentalSelection,
        ssh_credentials: &str,
    ) -> Result<ContainerInfo> {
        match self
            .deployment_manager
            .deploy_container(
                container_client,
                &request.container_spec,
                rental_id,
                &request.ssh_public_key,
            )
            .await
        {
            Ok(info) => Ok(info),
            Err(e) => {
                self.release_node(selection, rental_id).await;
                tracing::error!(
                    node_id = %selection.node_id,
                    rental_id = %rental_id,
                    miner_uid = selection.miner_uid,
                    "[RENTAL_FLOW] Failed to deploy container on node {}: {}",
                    selection.node_id,
                    e
                );

                if let Some(miner_uid) = selection.miner_uid {
                    let details = BanManager::create_rental_failure_details(
                        rental_id,
                        &selection.node_id,
                        &e.to_string(),
                        Some(ssh_credentials),
                    );

                    if let Err(log_err) = tokio::task::block_in_place(|| {
                        tokio::runtime::Handle::current().block_on(async {
                            self.ban_manager
                                .log_misbehaviour(
                                    miner_uid,
                                    &selection.node_id,
                                    MisbehaviourType::DeploymentFailed,
                                    &details,
                                )
                                .await
                        })
                    }) {
                        tracing::warn!(
                            "Failed to log misbehaviour for node {}: {}",
                            selection.node_id,
                            log_err
                        );
                    }
                }

                Err(e)
            }
        }
    }

    async fn finalize_rental(
        &self,
        request: &RentalRequest,
        rental_id: &str,
        selection: &RentalSelection,
        ssh_credentials: &str,
        node_details: &crate::api::types::NodeDetails,
        container_info: &ContainerInfo,
    ) -> Result<RentalInfo> {
        let container_id = container_info.container_id.clone();
        let metadata = request.metadata.clone();

        let now = chrono::Utc::now();
        let rental_info = RentalInfo {
            rental_id: rental_id.to_string(),
            validator_hotkey: request.validator_hotkey.clone(),
            node_id: selection.node_id.clone(),
            container_id: container_id.clone(),
            ssh_session_id: format!("direct-{}", rental_id),
            ssh_credentials: ssh_credentials.to_string(),
            state: RentalState::Active,
            created_at: now,
            updated_at: now,
            container_spec: request.container_spec.clone(),
            miner_id: selection.miner_id.clone(),
            node_details: node_details.clone(),
            metadata,
        };

        if let Err(e) = self.persistence.save_rental(&rental_info).await {
            tracing::error!(
                node_id = %selection.node_id,
                rental_id = %rental_id,
                miner_uid = selection.miner_uid,
                container_id = %container_id,
                "[RENTAL_FLOW] Failed to finalize rental setup: {}",
                e
            );
            self.cleanup_container_on_failure(
                ssh_credentials,
                &container_id,
                &selection.node_id,
                rental_id,
            )
            .await;
            return Err(e);
        }

        Ok(rental_info)
    }

    fn record_rental_metrics(&self, selection: &RentalSelection, rental_info: &RentalInfo) {
        if let Some(miner_uid) = selection.miner_uid {
            let gpu_type = get_gpu_type(&rental_info.node_details);
            self.metrics
                .record_node_rental_status(&selection.node_id, miner_uid, &gpu_type, true);
            self.metrics.record_rental_created(miner_uid, &gpu_type);
            tracing::debug!(
                node_id = %selection.node_id,
                rental_id = %rental_info.rental_id,
                miner_uid = miner_uid,
                "Recorded rental start for node {} (miner_uid: {}, gpu_type: {})",
                selection.node_id,
                miner_uid,
                gpu_type
            );
        }
    }

    /// Get rental status
    pub async fn get_rental_status(&self, rental_id: &str) -> Result<RentalStatus> {
        let rental_info = self
            .persistence
            .load_rental(rental_id)
            .await?
            .ok_or_else(|| anyhow::anyhow!("Rental not found"))?;

        // For terminal states, return DB state without Docker queries
        if matches!(
            rental_info.state,
            RentalState::Stopped | RentalState::Failed
        ) {
            return Ok(RentalStatus {
                rental_id: rental_id.to_string(),
                state: rental_info.state.clone(),
                container_status: ContainerStatus {
                    container_id: rental_info.container_id.clone(),
                    state: if rental_info.state == RentalState::Stopped {
                        "stopped".to_string()
                    } else {
                        "exited".to_string()
                    },
                    exit_code: None,
                    health: "none".to_string(),
                    started_at: None,
                    finished_at: None,
                },
                created_at: rental_info.created_at,
                resource_usage: ResourceUsage::default(),
            });
        }

        // Get container status using validator SSH credentials
        let container_client = self.create_container_client(&rental_info.ssh_credentials)?;

        let container_status = container_client
            .get_container_status(&rental_info.container_id)
            .await?;

        // Get resource usage
        let resource_usage = container_client
            .get_resource_usage(&rental_info.container_id)
            .await?;

        Ok(RentalStatus {
            rental_id: rental_id.to_string(),
            state: rental_info.state.clone(),
            container_status,
            created_at: rental_info.created_at,
            resource_usage,
        })
    }

    /// Stop a rental
    pub async fn stop_rental(&self, rental_id: &str, force: bool) -> Result<()> {
        let rental_info = self
            .persistence
            .load_rental(rental_id)
            .await?
            .ok_or_else(|| anyhow::anyhow!("Rental not found"))?;

        // Already in terminal state — nothing to do
        if matches!(
            rental_info.state,
            RentalState::Stopped | RentalState::Failed
        ) {
            return Ok(());
        }

        // Stop container using validator SSH credentials
        let container_client = self.create_container_client(&rental_info.ssh_credentials)?;

        self.deployment_manager
            .stop_container(&container_client, &rental_info.container_id, force)
            .await?;

        // Update rental state
        let mut updated_rental = rental_info.clone();
        updated_rental.state = RentalState::Stopped;
        updated_rental.updated_at = chrono::Utc::now();
        self.persistence.save_rental(&updated_rental).await?;

        // INVARIANT: Release node claim so it becomes available again.
        let _ = self
            .persistence
            .release_node(&rental_info.node_id, &rental_info.miner_id, rental_id)
            .await;

        // Clear rental metric
        let miner_uid = extract_miner_uid(&rental_info.miner_id);

        if let Some(miner_uid) = miner_uid {
            let gpu_type = get_gpu_type(&rental_info.node_details);
            self.metrics.record_node_rental_status(
                &rental_info.node_id,
                miner_uid,
                &gpu_type,
                false, // is_rented = false
            );
            tracing::debug!(
                "Cleared rental metric for node {} (miner_uid: {}, gpu_type: {})",
                rental_info.node_id,
                miner_uid,
                gpu_type
            );
        }

        Ok(())
    }

    /// Restart a rental's container
    pub async fn restart_rental(&self, rental_id: &str) -> Result<RentalRestartResponse> {
        let start_time = std::time::Instant::now();

        // Load rental info
        let rental_info = self
            .persistence
            .load_rental(rental_id)
            .await?
            .ok_or_else(|| anyhow::anyhow!("Rental not found: {}", rental_id))?;

        // Validate state - only Active rentals can be restarted
        if rental_info.state != RentalState::Active {
            return Err(anyhow::anyhow!(
                "Cannot restart rental in {} state. Only Active rentals can be restarted.",
                rental_info.state
            ));
        }

        tracing::info!(
            rental_id = %rental_id,
            node_id = %rental_info.node_id,
            container_id = %rental_info.container_id,
            "Restarting rental"
        );

        // Update state to Restarting
        let mut updated_rental = rental_info.clone();
        updated_rental.state = RentalState::Restarting;
        updated_rental.updated_at = chrono::Utc::now();
        self.persistence.save_rental(&updated_rental).await?;

        // Create container client and perform restart
        let container_client = match self.create_container_client(&rental_info.ssh_credentials) {
            Ok(client) => client,
            Err(e) => {
                tracing::error!(
                    rental_id = %rental_id,
                    error = %e,
                    "Failed to create container client for restart"
                );
                updated_rental.state = RentalState::Failed;
                updated_rental.updated_at = chrono::Utc::now();
                self.persistence.save_rental(&updated_rental).await?;
                // INVARIANT: Release node claim on restart failure.
                let _ = self
                    .persistence
                    .release_node(&rental_info.node_id, &rental_info.miner_id, rental_id)
                    .await;
                return Err(e);
            }
        };
        let restart_result = container_client
            .restart_container(&rental_info.container_id, 10)
            .await;

        // Update state based on result
        let final_state = match restart_result {
            Ok(_) => {
                tracing::info!(rental_id = %rental_id, "Successfully restarted rental");
                RentalState::Active
            }
            Err(ref e) => {
                tracing::error!(
                    rental_id = %rental_id,
                    error = %e,
                    "Failed to restart rental"
                );
                RentalState::Failed
            }
        };

        updated_rental.state = final_state.clone();
        updated_rental.updated_at = chrono::Utc::now();
        self.persistence.save_rental(&updated_rental).await?;

        // INVARIANT: Release node claim if restart failed and state is terminal.
        if matches!(final_state, RentalState::Failed) {
            let _ = self
                .persistence
                .release_node(&rental_info.node_id, &rental_info.miner_id, rental_id)
                .await;
        }

        // Return error if restart failed
        restart_result?;

        Ok(RentalRestartResponse {
            rental_id: rental_id.to_string(),
            status: final_state,
            message: "Container restarted successfully".to_string(),
            operation_duration_ms: start_time.elapsed().as_millis() as u64,
        })
    }

    /// Stream container logs
    pub async fn stream_logs(
        &self,
        rental_id: &str,
        follow: bool,
        tail_lines: Option<u32>,
    ) -> Result<tokio::sync::mpsc::Receiver<LogEntry>> {
        let rental_info = self
            .persistence
            .load_rental(rental_id)
            .await?
            .ok_or_else(|| anyhow::anyhow!("Rental not found"))?;

        let container_client = self.create_container_client(&rental_info.ssh_credentials)?;

        self.log_streamer
            .stream_logs(
                &container_client,
                &rental_info.container_id,
                follow,
                tail_lines,
            )
            .await
    }

    pub async fn list_rentals(&self, validator_hotkey: &str) -> Result<Vec<RentalInfo>> {
        let all_rentals = self
            .persistence
            .list_validator_rentals(validator_hotkey)
            .await?;

        // Filter out rentals from banned executors
        let mut available_rentals = Vec::new();
        for rental in all_rentals {
            // Extract miner_uid from miner_id
            let miner_uid = extract_miner_uid(&rental.miner_id);

            if let Some(miner_uid) = miner_uid {
                // Check if node is banned
                if self
                    .ban_manager
                    .is_executor_banned(miner_uid, &rental.node_id)
                    .await
                    .unwrap_or(false)
                {
                    tracing::debug!(
                        "Filtering out rental {} from banned node {} (miner_uid: {})",
                        rental.rental_id,
                        rental.node_id,
                        miner_uid
                    );
                    continue;
                }
            }

            available_rentals.push(rental);
        }

        Ok(available_rentals)
    }
}

impl Drop for RentalManager {
    fn drop(&mut self) {
        self.health_monitor.stop();
        if let Some(ref billing) = self.billing {
            billing.stop();
        }
        tracing::debug!("Stopped monitors for RentalManager");
    }
}

// #[cfg(test)]
// mod tests {
//     use super::*;

//     #[test]
//     fn test_parse_ssh_host() {
//         // Valid formats
//         assert_eq!(
//             parse_ssh_host("user@example.com:22").unwrap(),
//             "example.com"
//         );
//         assert_eq!(
//             parse_ssh_host("root@192.168.1.1:2222").unwrap(),
//             "192.168.1.1"
//         );
//         assert_eq!(parse_ssh_host("admin@host").unwrap(), "host");

//         // Invalid formats should return errors
//         assert!(parse_ssh_host("no-at-sign").is_err());
//         assert!(parse_ssh_host("@:22").is_err());
//         assert!(parse_ssh_host("user@").is_err());
//         assert!(parse_ssh_host("user@:22").is_err());
//         assert!(parse_ssh_host("").is_err());
//     }
// }
