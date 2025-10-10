//! Container monitoring and log streaming
//!
//! This module provides health monitoring and log streaming capabilities
//! for deployed containers.

use anyhow::{Context, Result};
use chrono::Utc;
use std::sync::Arc;
use tokio::io::{AsyncBufReadExt, BufReader};
use tokio::sync::mpsc;
use tokio::time::{interval, Duration};
use tokio_util::sync::CancellationToken;
use tracing::{debug, error, info, warn};

use super::container_client::ContainerClient;
use super::types::{LogEntry, RentalInfo, RentalState};
use crate::metrics::ValidatorPrometheusMetrics;
use crate::persistence::{SimplePersistence, ValidatorPersistence};
use crate::ssh::ValidatorSshKeyManager;

/// Database-driven health monitor for containers
#[derive(Clone)]
pub struct DatabaseHealthMonitor {
    /// Persistence layer for database operations
    persistence: Arc<SimplePersistence>,
    /// SSH key manager for validator keys
    ssh_key_manager: Arc<ValidatorSshKeyManager>,
    /// Metrics for tracking rental status (required)
    metrics: Arc<ValidatorPrometheusMetrics>,
    /// Health check configuration
    config: HealthCheckConfig,
    /// Cancellation token for the monitoring loop
    cancellation_token: CancellationToken,
}

/// Health check configuration
#[derive(Debug, Clone)]
pub struct HealthCheckConfig {
    /// Health check interval
    pub check_interval: Duration,
    /// Timeout for health check commands
    pub check_timeout: Duration,
}

impl Default for HealthCheckConfig {
    fn default() -> Self {
        Self {
            check_interval: Duration::from_secs(30),
            check_timeout: Duration::from_secs(10),
        }
    }
}

impl DatabaseHealthMonitor {
    /// Create a new database-driven health monitor
    pub fn new(
        persistence: Arc<SimplePersistence>,
        ssh_key_manager: Arc<ValidatorSshKeyManager>,
        metrics: Arc<ValidatorPrometheusMetrics>,
    ) -> Self {
        Self {
            persistence,
            ssh_key_manager,
            metrics,
            config: HealthCheckConfig::default(),
            cancellation_token: CancellationToken::new(),
        }
    }

    /// Create with custom configuration
    pub fn with_config(
        persistence: Arc<SimplePersistence>,
        ssh_key_manager: Arc<ValidatorSshKeyManager>,
        metrics: Arc<ValidatorPrometheusMetrics>,
        config: HealthCheckConfig,
    ) -> Self {
        Self {
            persistence,
            ssh_key_manager,
            metrics,
            config,
            cancellation_token: CancellationToken::new(),
        }
    }

    /// Start the monitoring loop
    pub fn start_monitoring_loop(&self) {
        let monitor = self.clone();
        tokio::spawn(async move {
            monitor.monitoring_loop().await;
        });
    }

    /// Stop the monitoring loop
    pub fn stop(&self) {
        self.cancellation_token.cancel();
    }

    /// Main monitoring loop
    async fn monitoring_loop(&self) {
        let mut check_interval = interval(self.config.check_interval);
        info!("Database health monitor started");

        loop {
            tokio::select! {
                _ = self.cancellation_token.cancelled() => {
                    info!("Database health monitor stopped");
                    break;
                }
                _ = check_interval.tick() => {
                    if let Err(e) = self.check_all_rentals().await {
                        error!("Error checking rental health: {}", e);
                    }
                }
            }
        }
    }

    /// Check health status of all non-terminal rentals
    async fn check_all_rentals(&self) -> Result<()> {
        // Query all rentals that are not in terminal states
        let rentals = self
            .persistence
            .query_non_terminated_rentals()
            .await
            .context("Failed to query non-terminal rentals")?;

        debug!("Checking health for {} rentals", rentals.len());

        // TODO: this can be done in parallel
        for rental in rentals {
            if let Err(e) = self.check_rental_health(&rental).await {
                error!(
                    "Failed to check health for rental {}: {}",
                    rental.rental_id, e
                );
                // Continue checking other rentals even if one fails
            }
        }

        Ok(())
    }

    /// Check health of a single rental
    async fn check_rental_health(&self, rental: &RentalInfo) -> Result<()> {
        debug!("Checking health for rental {}", rental.rental_id);

        // Get validator's private key path
        let validator_private_key_path = self
            .ssh_key_manager
            .get_persistent_key()
            .ok_or_else(|| anyhow::anyhow!("No persistent validator SSH key available"))?
            .1
            .clone();

        // Create container client with SSH credentials
        let container_client = ContainerClient::new(
            rental.ssh_credentials.clone(),
            Some(validator_private_key_path),
        )?;

        // Perform health check
        let health_result = tokio::time::timeout(
            self.config.check_timeout,
            Self::perform_health_check(&container_client, &rental.container_id),
        )
        .await;

        // Determine new state based on current state and health result
        let new_state = match (rental.state.clone(), health_result) {
            // Timeout or error during health check
            (_, Err(_)) => {
                warn!(
                    "Health check timeout for rental {} in state {:?}",
                    rental.rental_id, rental.state
                );
                Some(RentalState::Failed)
            }
            // Health check returned an error
            (current_state, Ok(Err(e))) => {
                error!(
                    "Health check error for rental {} in state {:?}: {}",
                    rental.rental_id, current_state, e
                );
                match current_state {
                    RentalState::Provisioning => Some(RentalState::Failed),
                    RentalState::Active => Some(RentalState::Stopped),
                    RentalState::Stopping => Some(RentalState::Stopped),
                    _ => None,
                }
            }
            // Health check succeeded
            (current_state, Ok(Ok(healthy))) => {
                if healthy {
                    debug!("Rental {} is healthy", rental.rental_id);
                    None // No state change needed
                } else {
                    warn!(
                        "Rental {} is unhealthy in state {:?}",
                        rental.rental_id, current_state
                    );
                    match current_state {
                        RentalState::Provisioning => Some(RentalState::Failed),
                        RentalState::Active => Some(RentalState::Stopped),
                        RentalState::Stopping => Some(RentalState::Stopped),
                        _ => None,
                    }
                }
            }
        };

        // Update rental state if needed
        if let Some(new_state) = new_state {
            info!(
                "Updating rental {} state from {:?} to {:?}",
                rental.rental_id, rental.state, new_state
            );

            let mut updated_rental = rental.clone();
            updated_rental.state = new_state.clone();

            self.persistence
                .save_rental(&updated_rental)
                .await
                .context("Failed to update rental state")?;

            // Update metrics when state changes to terminal states
            if matches!(new_state, RentalState::Stopped | RentalState::Failed) {
                let miner_uid = super::extract_miner_uid(&rental.miner_id);

                if let Some(miner_uid) = miner_uid {
                    let gpu_type = super::get_gpu_type(&rental.node_details);
                    self.metrics.record_node_rental_status(
                        &rental.node_id,
                        miner_uid,
                        &gpu_type,
                        false, // is_rented = false for stopped/failed states
                    );
                    debug!(
                        "Health monitor cleared rental metric for node {} (state: {:?}, miner_uid: {}, gpu_type: {})",
                        rental.node_id,
                        new_state,
                        miner_uid,
                        gpu_type
                    );
                }
            }
        }

        Ok(())
    }

    /// Perform a health check on a container
    async fn perform_health_check(client: &ContainerClient, container_id: &str) -> Result<bool> {
        // Get container status
        let status = client.get_container_status(container_id).await?;

        // Check if container is running
        if status.state != "running" {
            return Ok(false);
        }

        // Check container health status if available
        if status.health != "none" {
            return Ok(status.health == "healthy");
        }

        // Container is running and no specific health check configured
        Ok(true)
    }
}

/// Log streamer for containers
pub struct LogStreamer {
    /// Configuration
    config: LogStreamConfig,
}

/// Log streaming configuration
#[derive(Debug, Clone)]
pub struct LogStreamConfig {
    /// Buffer size for log channels
    pub buffer_size: usize,
    /// Maximum line length
    pub max_line_length: usize,
}

impl Default for LogStreamConfig {
    fn default() -> Self {
        Self {
            buffer_size: 1000,
            max_line_length: 4096,
        }
    }
}

impl Default for LogStreamer {
    fn default() -> Self {
        Self::new()
    }
}

impl LogStreamer {
    /// Create a new log streamer
    pub fn new() -> Self {
        Self {
            config: LogStreamConfig::default(),
        }
    }

    /// Create with custom configuration
    pub fn with_config(config: LogStreamConfig) -> Self {
        Self { config }
    }

    /// Stream logs from a container
    pub async fn stream_logs(
        &self,
        client: &ContainerClient,
        container_id: &str,
        follow: bool,
        tail_lines: Option<u32>,
    ) -> Result<mpsc::Receiver<LogEntry>> {
        let (tx, rx) = mpsc::channel(self.config.buffer_size);

        let container_id = container_id.to_string();
        let max_line_length = self.config.max_line_length;

        // Start log streaming process
        let mut child = client
            .stream_logs(&container_id, follow, tail_lines)
            .await
            .context("Failed to start log streaming")?;

        // Spawn task to read logs
        tokio::spawn(async move {
            // Read stdout
            if let Some(stdout) = child.stdout.take() {
                let tx_stdout = tx.clone();
                let container_id_stdout = container_id.clone();

                tokio::spawn(async move {
                    let reader = BufReader::new(stdout);
                    let mut lines = reader.lines();

                    while let Ok(Some(line)) = lines.next_line().await {
                        let log_entry = Self::parse_log_line(
                            &line,
                            "stdout",
                            &container_id_stdout,
                            max_line_length,
                        );

                        if tx_stdout.send(log_entry).await.is_err() {
                            break;
                        }
                    }
                });
            }

            // Read stderr
            if let Some(stderr) = child.stderr.take() {
                let tx_stderr = tx;
                let container_id_stderr = container_id.clone();

                tokio::spawn(async move {
                    let reader = BufReader::new(stderr);
                    let mut lines = reader.lines();

                    while let Ok(Some(line)) = lines.next_line().await {
                        let log_entry = Self::parse_log_line(
                            &line,
                            "stderr",
                            &container_id_stderr,
                            max_line_length,
                        );

                        if tx_stderr.send(log_entry).await.is_err() {
                            break;
                        }
                    }
                });
            }

            // Wait for process to complete
            let _ = child.wait().await;
        });

        Ok(rx)
    }

    /// Parse a log line into a LogEntry
    fn parse_log_line(line: &str, stream: &str, container_id: &str, max_length: usize) -> LogEntry {
        // Docker logs with timestamps format: "2024-01-01T00:00:00.000000000Z message"
        let (timestamp, message) = if let Some(space_idx) = line.find(' ') {
            let (ts_str, msg) = line.split_at(space_idx);

            match chrono::DateTime::parse_from_rfc3339(ts_str) {
                Ok(ts) => (ts.with_timezone(&chrono::Utc), msg.trim_start().to_string()),
                Err(_) => (Utc::now(), line.to_string()),
            }
        } else {
            (Utc::now(), line.to_string())
        };

        // Truncate message if too long
        let message = if message.len() > max_length {
            format!("{}... (truncated)", &message[..max_length])
        } else {
            message
        };

        LogEntry {
            timestamp,
            stream: stream.to_string(),
            message,
            container_id: container_id.to_string(),
        }
    }
}
