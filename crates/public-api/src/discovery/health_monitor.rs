//! Health monitoring for discovered validators

use super::ValidatorInfo;
use crate::config::Config;
use dashmap::DashMap;
use std::sync::Arc;
use tokio::time::{interval, Duration};
use tracing::{debug, error, info, warn};

/// Health monitor for validators
pub struct HealthMonitor {
    /// Validators to monitor
    validators: Arc<DashMap<u16, ValidatorInfo>>,

    /// Configuration
    config: Arc<Config>,

    /// HTTP client for health checks
    client: reqwest::Client,
}

impl HealthMonitor {
    /// Create a new health monitor
    pub fn new(validators: Arc<DashMap<u16, ValidatorInfo>>, config: Arc<Config>) -> Self {
        let client = reqwest::Client::builder()
            .timeout(Duration::from_secs(config.discovery.validator_timeout))
            .build()
            .unwrap_or_default();

        Self {
            validators,
            config,
            client,
        }
    }

    /// Start the health monitoring loop
    pub async fn start_monitoring(&self) {
        info!("Starting validator health monitoring");

        let mut check_interval = interval(self.config.health_check_interval());

        loop {
            check_interval.tick().await;

            let validators_to_check: Vec<(u16, String)> = self
                .validators
                .iter()
                .map(|entry| (*entry.key(), entry.endpoint.clone()))
                .collect();

            debug!(
                "Checking health of {} validators",
                validators_to_check.len()
            );

            // Check validators concurrently
            let mut tasks = Vec::new();
            for (uid, endpoint) in validators_to_check {
                let client = self.client.clone();
                let validators = self.validators.clone();
                let failover_threshold = self.config.discovery.failover_threshold;

                tasks.push(tokio::spawn(async move {
                    let is_healthy = check_validator_health(&client, &endpoint).await;

                    // Update validator health status
                    if let Some(mut validator) = validators.get_mut(&uid) {
                        validator.is_healthy = is_healthy;
                        validator.last_health_check = Some(chrono::Utc::now());

                        if !is_healthy {
                            validator.failure_count += 1;
                            if validator.failure_count >= failover_threshold {
                                warn!(
                                    "Validator {} has failed {} times, marking as unhealthy",
                                    uid, validator.failure_count
                                );
                            }
                        } else {
                            validator.failure_count = 0;
                        }
                    }
                }));
            }

            // Wait for all health checks to complete
            for task in tasks {
                if let Err(e) = task.await {
                    error!("Health check task failed: {}", e);
                }
            }

            let healthy_count = self
                .validators
                .iter()
                .filter(|entry| entry.is_healthy)
                .count();

            info!(
                "Health check complete: {}/{} validators healthy",
                healthy_count,
                self.validators.len()
            );
        }
    }
}

/// Check the health of a single validator
async fn check_validator_health(client: &reqwest::Client, endpoint: &str) -> bool {
    let health_url = format!("{endpoint}/health");

    match client.get(&health_url).send().await {
        Ok(response) => {
            if response.status().is_success() {
                debug!("Validator {} is healthy", endpoint);
                true
            } else {
                warn!(
                    "Validator {} returned unhealthy status: {}",
                    endpoint,
                    response.status()
                );
                false
            }
        }
        Err(e) => {
            debug!("Failed to check validator {} health: {}", endpoint, e);
            false
        }
    }
}
