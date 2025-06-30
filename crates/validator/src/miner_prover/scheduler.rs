//! # Verification Scheduler
//!
//! Manages the scheduling and lifecycle of verification tasks.
//! Implements Single Responsibility Principle by focusing only on task scheduling.

use super::discovery::MinerDiscovery;
use super::types::{MinerInfo, VerificationStats};
use super::verification::VerificationEngine;
use crate::config::VerificationConfig;
use anyhow::Result;
use common::identity::MinerUid;
use std::collections::HashMap;
use tokio::time::interval;
use tracing::{debug, error, info, warn};

pub struct VerificationScheduler {
    config: VerificationConfig,
    active_verifications: HashMap<MinerUid, tokio::task::JoinHandle<()>>,
}

impl VerificationScheduler {
    pub fn new(config: VerificationConfig) -> Self {
        Self {
            config,
            active_verifications: HashMap::new(),
        }
    }

    /// Start the verification scheduling loop
    pub async fn start(
        &mut self,
        discovery: MinerDiscovery,
        verification: VerificationEngine,
    ) -> Result<()> {
        let mut interval = interval(self.config.verification_interval);

        info!(
            "Starting verification scheduler with interval: {}s",
            self.config.verification_interval.as_secs()
        );

        loop {
            interval.tick().await;

            if let Err(e) = self.schedule_verifications(&discovery, &verification).await {
                error!("Failed to schedule verifications: {}", e);
            }

            self.cleanup_completed_verifications().await;
        }
    }

    async fn schedule_verifications(
        &mut self,
        discovery: &MinerDiscovery,
        verification: &VerificationEngine,
    ) -> Result<()> {
        let miners = discovery.get_miners_for_verification().await?;
        info!("Selected {} miners for verification", miners.len());

        for miner in miners {
            if !self.can_schedule_verification(&miner) {
                continue;
            }

            let miner_uid = miner.uid;
            let handle = self
                .spawn_verification_task(miner, verification.clone())
                .await?;
            self.active_verifications.insert(miner_uid, handle);
        }

        Ok(())
    }

    fn can_schedule_verification(&self, miner: &MinerInfo) -> bool {
        if self.active_verifications.len() >= self.config.max_concurrent_verifications {
            warn!(
                "Maximum concurrent verifications reached, skipping miner {}",
                miner.uid.as_u16()
            );
            return false;
        }

        if self.active_verifications.contains_key(&miner.uid) {
            debug!("Miner {} already being verified", miner.uid.as_u16());
            return false;
        }

        true
    }

    async fn spawn_verification_task(
        &self,
        miner: MinerInfo,
        verification: VerificationEngine,
    ) -> Result<tokio::task::JoinHandle<()>> {
        let handle = tokio::spawn(async move {
            info!("Starting verification for miner {}", miner.uid.as_u16());

            match verification.verify_miner(miner.clone()).await {
                Ok(score) => {
                    info!(
                        "Miner {} verification completed with score: {:.4}",
                        miner.uid.as_u16(),
                        score
                    );
                }
                Err(e) => {
                    error!("Miner {} verification failed: {}", miner.uid.as_u16(), e);
                }
            }
        });

        Ok(handle)
    }

    async fn cleanup_completed_verifications(&mut self) {
        let completed: Vec<MinerUid> = self
            .active_verifications
            .iter()
            .filter_map(|(uid, handle)| {
                if handle.is_finished() {
                    Some(*uid)
                } else {
                    None
                }
            })
            .collect();

        let num_completed = completed.len();

        for uid in completed {
            if let Some(handle) = self.active_verifications.remove(&uid) {
                if let Err(e) = handle.await {
                    error!(
                        "Verification task for miner {} panicked: {}",
                        uid.as_u16(),
                        e
                    );
                }
            }
        }

        if num_completed > 0 {
            debug!("Cleaned up {} completed verification tasks", num_completed);
        }
    }

    /// Get current verification statistics
    pub fn get_stats(&self) -> VerificationStats {
        VerificationStats {
            active_verifications: self.active_verifications.len(),
            max_concurrent: self.config.max_concurrent_verifications,
        }
    }
}
