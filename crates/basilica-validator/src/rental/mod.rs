//! Rental module for container deployment and management
//!
//! This module provides functionality for validators to rent GPU resources
//! and deploy containers on executor machines.

use anyhow::{Context, Result};
use std::sync::Arc;
use uuid::Uuid;

pub mod container_client;
pub mod deployment;
pub mod monitoring;
pub mod types;

pub use container_client::ContainerClient;
pub use deployment::DeploymentManager;
pub use monitoring::{DatabaseHealthMonitor, LogStreamer};
pub use types::*;

use crate::metrics::ValidatorPrometheusMetrics;
use crate::miner_prover::miner_client::{AuthenticatedMinerConnection, MinerClient};
use crate::persistence::{SimplePersistence, ValidatorPersistence};
use crate::ssh::ValidatorSshKeyManager;
use basilica_protocol::basilca::miner::v1::CloseSshSessionRequest;

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
    /// Miner client for reconnections
    miner_client: Arc<MinerClient>,
    /// SSH key manager for validator keys
    ssh_key_manager: Option<Arc<ValidatorSshKeyManager>>,
    /// Metrics for tracking rental status (required)
    metrics: Arc<ValidatorPrometheusMetrics>,
}

/// Parse SSH host from credentials string format "user@host:port"
fn parse_ssh_host(credentials: &str) -> Result<&str> {
    let (_, host_port) = credentials
        .split_once('@')
        .context("Invalid SSH credentials format: missing '@' separator")?;

    let host = host_port
        .split(':')
        .next()
        .filter(|h| !h.is_empty())
        .context("Invalid SSH credentials format: empty host")?;

    Ok(host)
}

/// Extract miner UID from miner_id format: "miner_{uid}"
pub(crate) fn extract_miner_uid(miner_id: &str) -> Option<u16> {
    if let Some(uid_str) = miner_id.strip_prefix("miner_") {
        return uid_str.parse().ok();
    }
    None
}

/// Get normalized GPU type from executor details
pub(crate) fn get_gpu_type(executor_details: &crate::api::types::ExecutorDetails) -> String {
    use crate::gpu::categorization::GpuCategory;
    use std::str::FromStr;

    executor_details
        .gpu_specs
        .first()
        .map(|gpu| {
            let category = GpuCategory::from_str(&gpu.name).unwrap();
            category.to_string()
        })
        .unwrap_or_else(|| "unknown".to_string())
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

    /// Create a new rental manager with SSH key manager
    pub fn new(
        miner_client: Arc<MinerClient>,
        persistence: Arc<SimplePersistence>,
        ssh_key_manager: Arc<ValidatorSshKeyManager>,
        metrics: Arc<ValidatorPrometheusMetrics>,
    ) -> Self {
        let deployment_manager = Arc::new(DeploymentManager::new());
        let log_streamer = Arc::new(LogStreamer::new());

        // Create health monitor with SSH key manager and metrics
        let health_monitor = Arc::new(DatabaseHealthMonitor::new(
            persistence.clone(),
            ssh_key_manager.clone(),
            metrics.clone(),
        ));

        Self {
            persistence,
            deployment_manager: deployment_manager.clone(),
            log_streamer: log_streamer.clone(),
            health_monitor,
            miner_client,
            ssh_key_manager: Some(ssh_key_manager),
            metrics,
        }
    }

    // Start the monitoring loop
    pub fn start_monitor(&self) {
        self.health_monitor.start_monitoring_loop();
    }

    /// Initialize metrics for all existing rentals on startup
    pub async fn initialize_rental_metrics(&self) -> Result<()> {
        // Query all non-terminal rentals from persistence
        let rentals = self.persistence.query_non_terminated_rentals().await?;

        let rental_count = rentals.len();

        for rental in rentals {
            let miner_uid = extract_miner_uid(&rental.miner_id);

            if let Some(miner_uid) = miner_uid {
                let gpu_type = get_gpu_type(&rental.executor_details);

                // Set metric based on rental state
                let is_rented = matches!(
                    rental.state,
                    RentalState::Active | RentalState::Provisioning | RentalState::Stopping
                );

                self.metrics.record_executor_rental_status(
                    &rental.executor_id,
                    miner_uid,
                    &gpu_type,
                    is_rented,
                );

                tracing::info!(
                    "Initialized rental metric for executor {} (state: {:?}, is_rented: {})",
                    rental.executor_id,
                    rental.state,
                    is_rented
                );
            }
        }

        tracing::info!("Initialized metrics for {} existing rentals", rental_count);
        Ok(())
    }

    /// Initialize metrics for all executors on startup
    pub async fn initialize_executor_metrics(&self) -> Result<()> {
        use crate::gpu::categorization::GpuCategory;
        use std::str::FromStr;

        // Get all executors with their GPU and rental data in a single query
        let executor_metrics = self.persistence.get_all_executors_for_metrics().await?;

        let executor_count = executor_metrics.len();
        tracing::info!("Initializing metrics for {} executors", executor_count);

        for metric_data in executor_metrics {
            // Convert GPU name to category
            let gpu_type = metric_data
                .gpu_name
                .and_then(|name| GpuCategory::from_str(&name).ok())
                .map(|category| category.to_string())
                .unwrap_or_else(|| "unknown".to_string());

            self.metrics.record_executor_rental_status(
                &metric_data.executor_id,
                metric_data.miner_uid,
                &gpu_type,
                metric_data.has_active_rental,
            );

            tracing::debug!(
                "Initialized executor metric: executor={}, miner_uid={}, gpu_type={}, is_rented={}",
                metric_data.executor_id,
                metric_data.miner_uid,
                gpu_type,
                metric_data.has_active_rental
            );
        }

        tracing::info!(
            "Successfully initialized metrics for {} executors",
            executor_count
        );
        Ok(())
    }

    /// Start a new rental
    pub async fn start_rental(
        &self,
        request: RentalRequest,
        miner_connection: &mut AuthenticatedMinerConnection,
    ) -> Result<RentalResponse> {
        // Generate rental ID
        let rental_id = format!("rental-{}", Uuid::new_v4());

        let (validator_public_key, _validator_private_key_path) = self
            .ssh_key_manager
            .as_ref()
            .ok_or_else(|| anyhow::anyhow!("SSH key manager is required for rentals"))?
            .get_persistent_key()
            .ok_or_else(|| anyhow::anyhow!("No persistent validator SSH key available"))?
            .clone();

        // Get rental session duration from miner client config
        let session_duration = self.miner_client.get_rental_session_duration();

        // Request SSH session from miner with rental mode
        let ssh_session = miner_connection
            .initiate_rental_ssh_session(
                &request.executor_id,
                &request.validator_hotkey,
                &validator_public_key,
                &rental_id,
                session_duration,
            )
            .await?;

        let container_client = self.create_container_client(&ssh_session.access_credentials)?;

        // Deploy container with end-user's SSH public key
        let container_info = match self
            .deployment_manager
            .deploy_container(
                &container_client,
                &request.container_spec,
                &rental_id,
                &request.ssh_public_key,
            )
            .await
        {
            Ok(info) => info,
            Err(e) => {
                let close_request = CloseSshSessionRequest {
                    session_id: ssh_session.session_id.clone(),
                    validator_hotkey: request.validator_hotkey.clone(),
                    reason: "Deployment failed".to_string(),
                };
                if let Err(cleanup_err) = miner_connection.close_ssh_session(close_request).await {
                    tracing::error!(
                        "Failed to cleanup SSH session after deployment failure: {}",
                        cleanup_err
                    );
                }
                return Err(e);
            }
        };

        // Check if SSH port is mapped and construct proper SSH credentials for end-user
        let ssh_credentials = container_info
            .mapped_ports
            .iter()
            .find(|p| p.container_port == 22)
            .map(|ssh_mapping| {
                // Parse host from original credentials (format: "user@host:port")
                let host = parse_ssh_host(&ssh_session.access_credentials).unwrap_or_else(|e| {
                    tracing::warn!("Failed to parse SSH host from credentials: {}", e);
                    "localhost"
                });
                // Always use root as username for containers with the mapped port
                format!("root@{}:{}", host, ssh_mapping.host_port)
            });

        // Fetch executor details from persistence
        let executor_details = match self
            .persistence
            .get_executor_details(&request.executor_id, &request.miner_id)
            .await
        {
            Ok(Some(details)) => details,
            Ok(None) => {
                tracing::warn!(
                    "Executor details not found for executor_id: {}, using defaults",
                    request.executor_id
                );
                // Provide default executor details
                crate::api::types::ExecutorDetails {
                    id: request.executor_id.clone(),
                    gpu_specs: vec![],
                    cpu_specs: crate::api::types::CpuSpec {
                        cores: 0,
                        model: "Unknown".to_string(),
                        memory_gb: 0,
                    },
                    location: None,
                    network_speed: None,
                }
            }
            Err(e) => {
                tracing::error!(
                    "Failed to fetch executor details for executor_id {}: {}",
                    request.executor_id,
                    e
                );
                return Err(anyhow::anyhow!("Failed to fetch executor details: {}", e));
            }
        };

        // Store rental info
        let rental_info = RentalInfo {
            rental_id: rental_id.clone(),
            validator_hotkey: request.validator_hotkey.clone(),
            executor_id: request.executor_id.clone(),
            container_id: container_info.container_id.clone(),
            ssh_session_id: ssh_session.session_id.clone(),
            ssh_credentials: ssh_session.access_credentials.clone(), // Store validator's SSH credentials for operations
            state: RentalState::Active,
            created_at: chrono::Utc::now(),
            container_spec: request.container_spec.clone(),
            miner_id: request.miner_id.clone(),
            executor_details,
        };

        // Save to persistence
        self.persistence.save_rental(&rental_info).await?;

        // Record rental metrics
        let miner_uid = extract_miner_uid(&rental_info.miner_id);

        if let Some(miner_uid) = miner_uid {
            let gpu_type = get_gpu_type(&rental_info.executor_details);

            // Record rental status
            self.metrics.record_executor_rental_status(
                &request.executor_id,
                miner_uid,
                &gpu_type,
                true, // is_rented = true
            );

            // Record rental creation
            self.metrics.record_rental_created(miner_uid, &gpu_type);

            tracing::debug!(
                "Recorded rental start for executor {} (miner_uid: {}, gpu_type: {})",
                request.executor_id,
                miner_uid,
                gpu_type
            );
        }

        // Health monitoring happens automatically via the database monitor loop

        Ok(RentalResponse {
            rental_id,
            ssh_credentials,
            container_info,
        })
    }

    /// Get rental status
    pub async fn get_rental_status(&self, rental_id: &str) -> Result<RentalStatus> {
        let rental_info = self
            .persistence
            .load_rental(rental_id)
            .await?
            .ok_or_else(|| anyhow::anyhow!("Rental not found"))?;

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

        // Stop container using validator SSH credentials
        let container_client = self.create_container_client(&rental_info.ssh_credentials)?;

        self.deployment_manager
            .stop_container(&container_client, &rental_info.container_id, force)
            .await?;

        // Close SSH session through miner connection
        if let Err(e) = self.close_ssh_session(&rental_info).await {
            tracing::error!(
                "Failed to close SSH session {} for rental {}: {}",
                rental_info.ssh_session_id,
                rental_id,
                e
            );
            // Continue with cleanup even if SSH session closure fails
        }

        // Update rental state
        let mut updated_rental = rental_info.clone();
        updated_rental.state = RentalState::Stopped;
        self.persistence.save_rental(&updated_rental).await?;

        // Clear rental metric
        let miner_uid = extract_miner_uid(&rental_info.miner_id);

        if let Some(miner_uid) = miner_uid {
            let gpu_type = get_gpu_type(&rental_info.executor_details);
            self.metrics.record_executor_rental_status(
                &rental_info.executor_id,
                miner_uid,
                &gpu_type,
                false, // is_rented = false
            );
            tracing::debug!(
                "Cleared rental metric for executor {} (miner_uid: {}, gpu_type: {})",
                rental_info.executor_id,
                miner_uid,
                gpu_type
            );
        }

        Ok(())
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

    /// Close SSH session for a rental
    async fn close_ssh_session(&self, rental_info: &RentalInfo) -> Result<()> {
        let miner_data = self
            .persistence
            .get_miner_by_id(&rental_info.miner_id)
            .await?
            .ok_or_else(|| {
                anyhow::anyhow!("Miner {} not found in database", rental_info.miner_id)
            })?;

        let mut miner_connection = self
            .miner_client
            .connect_and_authenticate(&miner_data.endpoint, &miner_data.hotkey)
            .await
            .map_err(|e| anyhow::anyhow!("Failed to reconnect to miner: {}", e))?;

        // Close the SSH session
        miner_connection
            .close_ssh_session_by_id(
                &rental_info.ssh_session_id,
                &rental_info.validator_hotkey,
                "rental_stopped",
            )
            .await?;

        tracing::info!(
            "Successfully closed SSH session {} for rental {}",
            rental_info.ssh_session_id,
            rental_info.rental_id
        );

        Ok(())
    }

    pub async fn list_rentals(&self, validator_hotkey: &str) -> Result<Vec<RentalInfo>> {
        self.persistence
            .list_validator_rentals(validator_hotkey)
            .await
    }
}

impl Drop for RentalManager {
    fn drop(&mut self) {
        self.health_monitor.stop();
        tracing::debug!("Stopped health monitor for RentalManager");
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_parse_ssh_host() {
        // Valid formats
        assert_eq!(
            parse_ssh_host("user@example.com:22").unwrap(),
            "example.com"
        );
        assert_eq!(
            parse_ssh_host("root@192.168.1.1:2222").unwrap(),
            "192.168.1.1"
        );
        assert_eq!(parse_ssh_host("admin@host").unwrap(), "host");

        // Invalid formats should return errors
        assert!(parse_ssh_host("no-at-sign").is_err());
        assert!(parse_ssh_host("@:22").is_err());
        assert!(parse_ssh_host("user@").is_err());
        assert!(parse_ssh_host("user@:22").is_err());
        assert!(parse_ssh_host("").is_err());
    }
}
