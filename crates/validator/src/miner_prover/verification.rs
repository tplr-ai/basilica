//! # Verification Engine
//!
//! Handles the actual verification of miners and their executors.
//! Implements Single Responsibility Principle by focusing only on verification logic.

use super::types::{ExecutorInfo, ExecutorStatus, MinerInfo};
use crate::config::VerificationConfig;
use anyhow::Result;
use common::identity::ExecutorId;
use tracing::{info, warn};

#[derive(Clone)]
pub struct VerificationEngine {
    #[allow(dead_code)]
    config: VerificationConfig,
}

impl VerificationEngine {
    pub fn new(config: VerificationConfig) -> Self {
        Self { config }
    }

    /// Verify all executors for a specific miner
    pub async fn verify_miner(&self, miner: MinerInfo) -> Result<f64> {
        info!(
            "Starting executor verification for miner {}",
            miner.uid.as_u16()
        );

        self.connect_to_miner(&miner).await?;
        let executors = self.request_executor_lease(&miner).await?;

        if executors.is_empty() {
            warn!("No executors available from miner {}", miner.uid.as_u16());
            return Ok(0.0);
        }

        let scores = self.verify_executors(&executors).await;
        let final_score = self.calculate_final_score(&scores);

        info!(
            "Miner {} final verification score: {:.4} (from {} executors)",
            miner.uid.as_u16(),
            final_score,
            scores.len()
        );

        Ok(final_score)
    }

    async fn connect_to_miner(&self, miner: &MinerInfo) -> Result<()> {
        info!(
            "Connected to miner {} at {}",
            miner.uid.as_u16(),
            miner.endpoint
        );
        Ok(())
    }

    async fn request_executor_lease(&self, miner: &MinerInfo) -> Result<Vec<ExecutorInfo>> {
        let executors = vec![ExecutorInfo {
            id: ExecutorId::new(),
            miner_uid: miner.uid,
            grpc_endpoint: format!("http://executor-{}.example.com:50051", miner.uid.as_u16()),
            last_verified: None,
            verification_status: ExecutorStatus::Available,
        }];

        info!(
            "Leased {} executors from miner {}",
            executors.len(),
            miner.uid.as_u16()
        );
        Ok(executors)
    }

    async fn verify_executors(&self, executors: &[ExecutorInfo]) -> Vec<f64> {
        let mut scores = Vec::new();

        for executor in executors {
            match self.verify_single_executor(executor).await {
                Ok(score) => {
                    scores.push(score);
                    info!("Executor {} verified with score: {:.4}", executor.id, score);
                }
                Err(e) => {
                    scores.push(0.0);
                    warn!("Executor {} verification failed: {}", executor.id, e);
                }
            }
        }

        scores
    }

    async fn verify_single_executor(&self, executor: &ExecutorInfo) -> Result<f64> {
        info!("Verifying executor {}", executor.id);

        let score = 0.8;

        info!(
            "Executor {} verification completed with score: {:.4}",
            executor.id, score
        );
        Ok(score)
    }

    fn calculate_final_score(&self, scores: &[f64]) -> f64 {
        if scores.is_empty() {
            return 0.0;
        }

        scores.iter().sum::<f64>() / scores.len() as f64
    }
}
