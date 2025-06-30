//! # Types for Miner Verification
//!
//! Shared data structures used across the miner verification system.

use common::identity::{ExecutorId, Hotkey, MinerUid};

/// Information about a miner being verified
#[derive(Debug, Clone)]
pub struct MinerInfo {
    pub uid: MinerUid,
    pub hotkey: Hotkey,
    pub endpoint: String,
    pub is_validator: bool,
    pub stake_tao: f64,
    pub last_verified: Option<chrono::DateTime<chrono::Utc>>,
    pub verification_score: f64,
}

impl MinerInfo {
    /// Check if this miner needs verification based on time and score
    pub fn needs_verification(
        &self,
        min_interval: chrono::Duration,
        now: chrono::DateTime<chrono::Utc>,
    ) -> bool {
        if self.last_verified.is_none() {
            return true;
        }

        if let Some(last_verified) = self.last_verified {
            let time_since = now.signed_duration_since(last_verified);
            if time_since < min_interval {
                return false;
            }
        }

        if self.verification_score < 0.5 {
            return true;
        }

        use rand::Rng;
        let mut rng = rand::thread_rng();
        let random_factor: f64 = rng.gen();
        let verification_probability = 1.0 - (self.verification_score * 0.7);

        random_factor < verification_probability
    }
}

/// Information about an executor available for verification
#[derive(Debug, Clone)]
pub struct ExecutorInfo {
    pub id: ExecutorId,
    pub miner_uid: MinerUid,
    pub grpc_endpoint: String,
    pub last_verified: Option<chrono::DateTime<chrono::Utc>>,
    pub verification_status: ExecutorStatus,
}

#[derive(Debug, Clone)]
pub enum ExecutorStatus {
    Available,
    Verifying,
    Failed,
    Verified,
}

#[derive(Debug, Clone)]
pub struct VerificationStats {
    pub active_verifications: usize,
    pub max_concurrent: usize,
}
