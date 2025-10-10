//! # Miner Prover Module
//!
//! Manages the lifecycle of verifying selected miners from the metagraph.
//! This module is organized following SOLID principles with clear separation of concerns.

pub mod discovery;
pub mod miner_client;
pub mod scheduler;
pub mod types;
pub mod validation_binary;
pub mod validation_docker;
pub mod validation_hardware;
pub mod validation_nat;
pub mod validation_network;
pub mod validation_speedtest;
pub mod validation_states;
pub mod validation_storage;
pub mod validation_strategy;
pub mod validation_worker;
pub mod verification;
pub mod verification_engine_builder;

#[cfg(test)]
mod tests;

#[cfg(test)]
mod test_discovery;

pub use discovery::MinerDiscovery;
pub use scheduler::VerificationScheduler;
pub use verification::VerificationEngine;

use crate::config::VerificationConfig;
use crate::metrics::ValidatorMetrics;
use crate::persistence::SimplePersistence;
use crate::ssh::ValidatorSshClient;
use anyhow::Result;
use bittensor::Service as BittensorService;
use std::sync::Arc;
use tracing::info;

/// Main orchestrator for miner verification process
pub struct MinerProver {
    discovery: MinerDiscovery,
    scheduler: VerificationScheduler,
    verification: VerificationEngine,
}

impl MinerProver {
    /// Create a new MinerProver instance
    pub fn new(
        config: VerificationConfig,
        automatic_config: crate::config::AutomaticVerificationConfig,
        ssh_session_config: crate::config::SshSessionConfig,
        bittensor_service: Arc<BittensorService>,
        persistence: Arc<SimplePersistence>,
        metrics: Option<Arc<ValidatorMetrics>>,
    ) -> Result<Self> {
        let mut discovery = MinerDiscovery::new(bittensor_service.clone(), config.clone());

        // Add metrics if available
        if let Some(metrics_ref) = &metrics {
            discovery = discovery.with_metrics(metrics_ref.clone());
        }

        // Get validator hotkey from bittensor service
        let validator_hotkey = bittensor::account_id_to_hotkey(bittensor_service.get_account_id())
            .map_err(|e| anyhow::anyhow!("Failed to convert account ID to hotkey: {}", e))?;

        // Use VerificationEngineBuilder to properly initialize SSH key manager
        let verification_engine_builder =
            verification_engine_builder::VerificationEngineBuilder::new(
                config.clone(),
                automatic_config.clone(),
                ssh_session_config.clone(),
                validator_hotkey,
                persistence,
                metrics,
            )
            .with_bittensor_service(bittensor_service.clone())
            .with_ssh_client(Arc::new(ValidatorSshClient::new()));

        // Build verification engine with proper SSH key manager
        let verification = tokio::task::block_in_place(|| {
            tokio::runtime::Handle::current()
                .block_on(async { verification_engine_builder.build().await })
        })
        .map_err(|e| {
            anyhow::anyhow!(
                "Failed to build verification engine with SSH automation: {}",
                e
            )
        })?;

        // Create scheduler with automatic verification configuration
        let scheduler = VerificationScheduler::new(config.clone());

        Ok(Self {
            discovery,
            scheduler,
            verification,
        })
    }

    /// Start the miner verification loop
    pub async fn start(self) -> Result<()> {
        info!("Starting miner prover with automatic SSH session management");
        self.scheduler
            .start(self.discovery, self.verification)
            .await
    }

    pub fn get_public_key(&self) -> Option<String> {
        self.verification.get_ssh_public_key()
    }

    pub fn get_verification_engine(&self) -> &VerificationEngine {
        &self.verification
    }
}
