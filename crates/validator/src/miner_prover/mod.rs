//! # Miner Prover Module
//!
//! Manages the lifecycle of verifying selected miners from the metagraph.
//! This module is organized following SOLID principles with clear separation of concerns.

pub mod discovery;
pub mod scheduler;
pub mod types;
pub mod verification;

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
    pub fn new(config: VerificationConfig, bittensor_service: Arc<BittensorService>) -> Self {
        let discovery = MinerDiscovery::new(bittensor_service, config.clone());
        let scheduler = VerificationScheduler::new(config.clone());
        let verification = VerificationEngine::new(config);

        Self {
            discovery,
            scheduler,
            verification,
        }
    }

    /// Start the miner verification loop
    pub async fn start(&mut self) -> Result<()> {
        info!("Starting miner prover");
        self.scheduler
            .start(self.discovery.clone(), self.verification.clone())
            .await
    }

    /// Get current verification statistics
    pub fn get_verification_stats(&self) -> VerificationStats {
        self.scheduler.get_stats()
    }
}
