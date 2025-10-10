//! # Chain Registration
//!
//! Validator-specific chain registration wrapper around the common bittensor registration module.

use anyhow::Result;
use std::sync::Arc;
use tracing::info;

use crate::config::ValidatorConfig;

/// Chain registration service for validators
#[derive(Clone)]
pub struct ChainRegistration {
    inner: bittensor::ChainRegistration,
}

impl ChainRegistration {
    /// Create a new chain registration service
    pub async fn new(
        config: &ValidatorConfig,
        bittensor_service: Arc<bittensor::Service>,
    ) -> Result<Self> {
        info!(
            "Initializing chain registration for validator on netuid: {}",
            config.bittensor.common.netuid
        );

        // Build registration config for validator
        let reg_config = bittensor::RegistrationConfigBuilder::new(
            config.bittensor.common.netuid,
            config.bittensor.common.network.clone(),
            config.bittensor.axon_port,
        )
        .external_ip(config.bittensor.external_ip.clone())
        .local_spoofed_ip("10.0.0.2".to_string())
        .neuron_type("validator".to_string())
        .build();

        let inner = bittensor::ChainRegistration::new(reg_config, bittensor_service);

        Ok(Self { inner })
    }

    /// Perform one-time startup registration
    pub async fn register_startup(&self) -> Result<()> {
        self.inner.register_startup().await
    }

    /// Get discovered UID
    pub async fn get_discovered_uid(&self) -> Option<u16> {
        self.inner.get_discovered_uid().await
    }
}
