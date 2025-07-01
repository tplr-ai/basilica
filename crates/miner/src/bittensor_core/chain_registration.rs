//! # Chain Registration
//!
//! Miner-specific chain registration wrapper around the common bittensor registration module.

use anyhow::Result;
use std::sync::Arc;
use tracing::info;

use crate::config::MinerBittensorConfig;

/// Chain registration service for miners
#[derive(Clone)]
pub struct ChainRegistration {
    inner: bittensor::ChainRegistration,
    bittensor_service: Arc<bittensor::Service>,
}

impl ChainRegistration {
    /// Create a new chain registration service
    pub async fn new(config: MinerBittensorConfig) -> Result<Self> {
        info!(
            "Initializing chain registration for network: {}",
            config.common.network
        );

        // Initialize the bittensor service
        let bittensor_service = Arc::new(
            bittensor::Service::new(config.common.clone())
                .await
                .map_err(|e| anyhow::anyhow!("Failed to initialize bittensor service: {}", e))?,
        );

        // Build registration config for miner
        let reg_config = bittensor::RegistrationConfigBuilder::new(
            config.common.netuid,
            config.common.network.clone(),
            config.axon_port,
        )
        .external_ip(config.external_ip.clone())
        .skip_registration(config.skip_registration)
        .local_spoofed_ip("10.0.0.1".to_string())
        .neuron_type("miner".to_string())
        .build();

        let inner = bittensor::ChainRegistration::new(reg_config, bittensor_service.clone());

        Ok(Self {
            inner,
            bittensor_service,
        })
    }

    /// Perform one-time startup registration
    pub async fn register_startup(&self) -> Result<()> {
        self.inner.register_startup().await
    }

    /// Get current registration state
    pub async fn get_state(&self) -> RegistrationStateSnapshot {
        let state = self.inner.get_state().await;
        RegistrationStateSnapshot {
            is_registered: state.is_registered,
            registration_time: state.registration_time,
            discovered_uid: state.discovered_uid,
        }
    }

    /// Get discovered UID
    pub async fn get_discovered_uid(&self) -> Option<u16> {
        self.inner.get_discovered_uid().await
    }

    /// Health check for registration service
    pub async fn health_check(&self) -> Result<()> {
        self.inner.health_check().await
    }

    /// Get the bittensor service
    pub fn get_bittensor_service(&self) -> Arc<bittensor::Service> {
        self.bittensor_service.clone()
    }
}

/// Snapshot of the current registration state
#[derive(Debug, Clone)]
pub struct RegistrationStateSnapshot {
    pub is_registered: bool,
    pub registration_time: Option<chrono::DateTime<chrono::Utc>>,
    pub discovered_uid: Option<u16>,
}
