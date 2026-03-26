use anyhow::{Context, Result};
use chrono::{DateTime, Utc};
use serde::{Deserialize, Serialize};
use std::sync::Arc;
use tracing::{debug, error, info, warn};

use crate::ban_system::BanManager;
use crate::metrics::ValidatorPrometheusMetrics;
use crate::persistence::SimplePersistence;

/// Misbehaviour validation profile
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct MisbehaviourProfile {
    pub is_banned: bool,
    pub ban_expiry: Option<DateTime<Utc>>,
    pub check_timestamp: DateTime<Utc>,
    pub miner_uid: u16,
    pub executor_id: String,
    pub recent_misbehaviours: usize,
    pub full_json: String,
}

/// Collector for checking executor ban status
#[derive(Clone)]
pub struct Misbehaviour {
    ban_manager: Arc<BanManager>,
    persistence: Arc<SimplePersistence>,
}

impl Misbehaviour {
    pub fn new(
        persistence: Arc<SimplePersistence>,
        metrics: Option<Arc<ValidatorPrometheusMetrics>>,
    ) -> Self {
        Self {
            ban_manager: Arc::new(BanManager::new(persistence.clone(), metrics, None, None)),
            persistence,
        }
    }

    /// Collect ban status for an executor
    pub async fn collect(&self, executor_id: &str, miner_uid: u16) -> Result<MisbehaviourProfile> {
        info!(
            miner_uid = miner_uid,
            executor_id = executor_id,
            "[MISBEHAVIOUR] Checking ban status for executor"
        );

        // Check if executor is banned
        let ban_expiry = self
            .ban_manager
            .get_ban_expiry(miner_uid, executor_id)
            .await
            .context("Failed to determine ban expiry")?;
        let is_banned = ban_expiry.is_some();

        // Get recent misbehaviours count (last 7 days)
        let recent_logs = self
            .persistence
            .get_misbehaviour_logs(miner_uid, executor_id, chrono::Duration::days(7))
            .await
            .unwrap_or_default();
        let recent_misbehaviours = recent_logs.len();

        let check_timestamp = Utc::now();

        // Create full JSON for storage
        let full_json = serde_json::json!({
            "is_banned": is_banned,
            "ban_expiry": ban_expiry.map(|dt| dt.to_rfc3339()),
            "check_timestamp": check_timestamp.to_rfc3339(),
            "miner_uid": miner_uid,
            "executor_id": executor_id,
            "recent_misbehaviours": recent_misbehaviours,
            "recent_misbehaviour_types": recent_logs.iter()
                .map(|log| log.type_of_misbehaviour.as_str())
                .collect::<Vec<_>>(),
        })
        .to_string();

        let profile = MisbehaviourProfile {
            is_banned,
            ban_expiry,
            check_timestamp,
            miner_uid,
            executor_id: executor_id.to_string(),
            recent_misbehaviours,
            full_json,
        };

        if is_banned {
            warn!(
                miner_uid = miner_uid,
                executor_id = executor_id,
                ban_expiry = ?ban_expiry,
                recent_misbehaviours = recent_misbehaviours,
                "[MISBEHAVIOUR] Executor is currently banned"
            );
        } else {
            debug!(
                miner_uid = miner_uid,
                executor_id = executor_id,
                recent_misbehaviours = recent_misbehaviours,
                "[MISBEHAVIOUR] Executor is not banned"
            );
        }

        Ok(profile)
    }

    /// this is for consistency, we don't store during the validation process
    pub async fn store(
        &self,
        _miner_uid: u16,
        _executor_id: &str,
        _profile: &MisbehaviourProfile,
    ) -> Result<()> {
        Ok(())
    }

    /// Collect and store ban status
    pub async fn collect_and_store(
        &self,
        executor_id: &str,
        miner_uid: u16,
    ) -> Result<MisbehaviourProfile> {
        let profile = self.collect(executor_id, miner_uid).await?;
        self.store(miner_uid, executor_id, &profile).await?;
        Ok(profile)
    }

    /// Collect with fallback to unbanned profile on error
    pub async fn collect_with_fallback(
        &self,
        executor_id: &str,
        miner_uid: u16,
    ) -> Option<MisbehaviourProfile> {
        match self.collect_and_store(executor_id, miner_uid).await {
            Ok(profile) => {
                info!(
                    miner_uid = miner_uid,
                    executor_id = executor_id,
                    is_banned = profile.is_banned,
                    "[MISBEHAVIOUR] Ban status check completed successfully"
                );
                Some(profile)
            }
            Err(e) => {
                error!(
                    miner_uid = miner_uid,
                    executor_id = executor_id,
                    error = %e,
                    "[MISBEHAVIOUR] Ban status check failed, returning fail-closed profile"
                );
                Some(MisbehaviourProfile {
                    // Fail closed: treat unknown as banned to avoid spoofing.
                    // TODO: Consider a softer penalty path if availability issues become frequent.
                    is_banned: true,
                    ban_expiry: None,
                    check_timestamp: Utc::now(),
                    miner_uid,
                    executor_id: executor_id.to_string(),
                    recent_misbehaviours: 0,
                    full_json: serde_json::json!({
                        "error": e.to_string(),
                        "fallback": true
                    })
                    .to_string(),
                })
            }
        }
    }

    /// Retrieve misbehaviour profile if exists
    pub async fn retrieve(
        &self,
        miner_uid: u16,
        executor_id: &str,
    ) -> Result<Option<MisbehaviourProfile>> {
        match self.collect(executor_id, miner_uid).await {
            Ok(profile) => Ok(Some(profile)),
            Err(e) => {
                debug!(
                    miner_uid = miner_uid,
                    executor_id = executor_id,
                    error = %e,
                    "[MISBEHAVIOUR] No ban profile available"
                );
                Ok(None)
            }
        }
    }

    /// Get the ban manager for logging misbehaviours
    pub fn ban_manager(&self) -> Arc<BanManager> {
        self.ban_manager.clone()
    }
}
