//! # Miner Prover Module
//!
//! Manages the lifecycle of verifying selected miners from the metagraph.
//! This module is organized following SOLID principles with clear separation of concerns.

pub mod discovery;
pub mod miner_client;
pub mod scheduler;
pub mod types;
pub mod verification;

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
        _automatic_config: crate::config::AutomaticVerificationConfig,
        bittensor_service: Arc<BittensorService>,
    ) -> Self {
        let discovery = MinerDiscovery::new(bittensor_service.clone(), config.clone());

        // Create SSH client and hardware validator (optional)
        let ssh_client = Arc::new(crate::ssh::ValidatorSshClient::new());
        let hardware_validator = None; // Can be configured later if needed
        let ssh_key_path = None; // Can be configured later if needed

        // Use with_bittensor_service to properly load the validator's hotkey
        let verification = VerificationEngine::with_bittensor_service(
            config.clone(),
            bittensor_service.clone(),
            ssh_client,
            hardware_validator,
            ssh_key_path,
        );

        // Create shutdown channel for the scheduler
        let (_shutdown_tx, _shutdown_rx) = tokio::sync::broadcast::channel::<()>(1);

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
