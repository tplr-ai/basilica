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

/// Batch verification task handle
struct BatchVerificationHandle {
    miners: Vec<MinerInfo>,
    task_handle: tokio::task::JoinHandle<()>,
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

        // Filter miners that can be scheduled
        let schedulable_miners: Vec<MinerInfo> = miners
            .into_iter()
            .filter(|miner| self.can_schedule_verification(miner))
            .collect();

        if schedulable_miners.is_empty() {
            debug!("No miners available for scheduling verification");
            return Ok(());
        }

        // Check if automated verification is available and batch process if possible
        if schedulable_miners.len() > 1 && verification.supports_batch_processing() {
            info!(
                "Using batch automated verification for {} miners",
                schedulable_miners.len()
            );
            let handle = self
                .spawn_batch_verification_task(schedulable_miners, verification.clone())
                .await?;

            // Track first miner with the handle (batch processing)
            if let Some(first_miner) = handle.miners.first() {
                self.active_verifications
                    .insert(first_miner.uid, handle.task_handle);
            }
        } else {
            // Process individual miners
            for miner in schedulable_miners {
                let miner_uid = miner.uid;
                let handle = self
                    .spawn_verification_task(miner, verification.clone())
                    .await?;
                self.active_verifications.insert(miner_uid, handle);
            }
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
            info!(
                "Starting automated verification workflow for miner {}",
                miner.uid.as_u16()
            );

            // Use automated verification workflow with SSH session management
            match verification
                .execute_automated_verification_workflow(&[miner.clone()])
                .await
            {
                Ok(results) => {
                    if let Some(score) = results.get(&miner.uid) {
                        info!(
                            "Miner {} automated verification completed with score: {:.4}",
                            miner.uid.as_u16(),
                            score
                        );
                    } else {
                        warn!(
                            "No score returned for miner {} from automated verification",
                            miner.uid.as_u16()
                        );
                    }
                }
                Err(e) => {
                    error!(
                        "Miner {} automated verification failed: {}",
                        miner.uid.as_u16(),
                        e
                    );

                    // Fallback to basic verification if automated workflow fails
                    info!(
                        "Attempting fallback verification for miner {}",
                        miner.uid.as_u16()
                    );
                    match verification.verify_miner(miner.clone()).await {
                        Ok(score) => {
                            info!(
                                "Miner {} fallback verification completed with score: {:.4}",
                                miner.uid.as_u16(),
                                score
                            );
                        }
                        Err(fallback_err) => {
                            error!(
                                "Miner {} fallback verification also failed: {}",
                                miner.uid.as_u16(),
                                fallback_err
                            );
                        }
                    }
                }
            }
        });

        Ok(handle)
    }

    async fn spawn_batch_verification_task(
        &self,
        miners: Vec<MinerInfo>,
        verification: VerificationEngine,
    ) -> Result<BatchVerificationHandle> {
        let miners_clone = miners.clone();
        let handle = tokio::spawn(async move {
            info!(
                "Starting batch automated verification workflow for {} miners",
                miners.len()
            );

            match verification
                .execute_automated_verification_workflow(&miners)
                .await
            {
                Ok(results) => {
                    for miner in &miners {
                        if let Some(score) = results.get(&miner.uid) {
                            info!(
                                "Miner {} batch verification completed with score: {:.4}",
                                miner.uid.as_u16(),
                                score
                            );
                        } else {
                            warn!(
                                "No score returned for miner {} from batch verification",
                                miner.uid.as_u16()
                            );
                        }
                    }
                }
                Err(e) => {
                    error!(
                        "Batch automated verification failed for {} miners: {}",
                        miners.len(),
                        e
                    );

                    // Fallback to individual verification for failed batch
                    for miner in &miners {
                        info!(
                            "Attempting individual fallback verification for miner {}",
                            miner.uid.as_u16()
                        );
                        match verification.verify_miner(miner.clone()).await {
                            Ok(score) => {
                                info!(
                                    "Miner {} individual fallback completed with score: {:.4}",
                                    miner.uid.as_u16(),
                                    score
                                );
                            }
                            Err(fallback_err) => {
                                error!(
                                    "Miner {} individual fallback also failed: {}",
                                    miner.uid.as_u16(),
                                    fallback_err
                                );
                            }
                        }
                    }
                }
            }
        });

        Ok(BatchVerificationHandle {
            miners: miners_clone,
            task_handle: handle,
        })
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
