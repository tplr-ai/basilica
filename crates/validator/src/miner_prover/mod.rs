//! # Miner Prover Module
//!
//! Manages the lifecycle of verifying selected miners from the metagraph.
//! This module is organized following SOLID principles with clear separation of concerns.

pub mod discovery;
pub mod miner_client;
pub mod scheduler;
pub mod types;
pub mod verification;
pub mod verification_engine_builder;

#[cfg(test)]
mod tests;

#[cfg(test)]
mod test_discovery;

pub use discovery::MinerDiscovery;
pub use scheduler::VerificationScheduler;
pub use types::VerificationStats;
pub use verification::VerificationEngine;

use crate::config::VerificationConfig;
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
    ) -> Self {
        let discovery = MinerDiscovery::new(bittensor_service.clone(), config.clone());

        // Get validator hotkey from bittensor service
        let validator_hotkey = bittensor::account_id_to_hotkey(bittensor_service.get_account_id())
            .expect("Failed to convert account ID to hotkey");

        // Use VerificationEngineBuilder to properly initialize SSH key manager
        let verification_engine_builder =
            verification_engine_builder::VerificationEngineBuilder::new(
                config.clone(),
                automatic_config.clone(),
                ssh_session_config.clone(),
                validator_hotkey,
            )
            .with_bittensor_service(bittensor_service.clone());

        // Build verification engine with proper SSH key manager
        let verification = tokio::task::block_in_place(|| {
            tokio::runtime::Handle::current()
                .block_on(async { verification_engine_builder.build().await })
        })
        .expect("Failed to build verification engine with SSH automation");

        // Create scheduler with automatic verification configuration
        let scheduler = VerificationScheduler::new(config.clone());

        Self {
            discovery,
            scheduler,
            verification,
        }
    }

    /// Start the miner verification loop
    pub async fn start(&mut self) -> Result<()> {
        info!("Starting miner prover with automatic SSH session management");
        self.scheduler
            .start(self.discovery.clone(), self.verification.clone())
            .await
    }

    /// Get current verification statistics
    pub fn get_verification_stats(&self) -> VerificationStats {
        self.scheduler.get_stats()
    }
}
