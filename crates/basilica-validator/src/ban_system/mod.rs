use anyhow::Result;
use chrono::{DateTime, Duration, Utc};
use serde_json::json;
use std::sync::Arc;
use tracing::{debug, info, warn};

use crate::collateral::SlashExecutor;
use crate::metrics::ValidatorPrometheusMetrics;
use crate::persistence::entities::{MisbehaviourLog, MisbehaviourType};
use crate::persistence::SimplePersistence;

/// Ban manager for handling executor misbehaviour and ban status
pub struct BanManager {
    persistence: Arc<SimplePersistence>,
    metrics: Option<Arc<ValidatorPrometheusMetrics>>,
    slash_executor: Option<Arc<SlashExecutor>>,
    validator_hotkey: Option<String>,
}

impl BanManager {
    /// Create a new ban manager
    pub fn new(
        persistence: Arc<SimplePersistence>,
        metrics: Option<Arc<ValidatorPrometheusMetrics>>,
        slash_executor: Option<Arc<SlashExecutor>>,
        validator_hotkey: Option<String>,
    ) -> Self {
        Self {
            persistence,
            metrics,
            slash_executor,
            validator_hotkey,
        }
    }

    /// Log a misbehaviour for an executor
    ///
    /// This function:
    /// 1. Records the misbehaviour
    /// 2. Checks if a ban should be triggered
    pub async fn log_misbehaviour(
        &self,
        miner_uid: u16,
        node_id: &str,
        type_of_misbehaviour: MisbehaviourType,
        details: &str,
    ) -> Result<()> {
        // Convert miner_uid to miner_id format
        let miner_id = format!("miner_{}", miner_uid);

        // Get node endpoint
        let endpoint = self
            .persistence
            .get_executor_endpoint(&miner_id, node_id)
            .await?
            .unwrap_or_else(|| "unknown".to_string());

        // Create misbehaviour log
        let log = MisbehaviourLog::new(
            miner_uid,
            node_id.to_string(),
            endpoint,
            type_of_misbehaviour,
            details.to_string(),
        );

        // Insert the log into database
        self.persistence.insert_misbehaviour_log(&log).await?;

        info!(
            miner_uid = miner_uid,
            node_id = node_id,
            misbehaviour_type = ?type_of_misbehaviour,
            "Misbehaviour logged for node"
        );

        // Check if ban should be triggered
        let should_ban = self
            .check_ban_trigger(miner_uid, node_id)
            .await
            .unwrap_or(false);

        if should_ban {
            warn!(
                miner_uid = miner_uid,
                node_id = node_id,
                "Node has triggered ban conditions"
            );
        }

        let should_slash = should_ban;

        if should_slash {
            if let Err(err) = self
                .try_execute_slash(miner_uid, node_id, type_of_misbehaviour, details)
                .await
            {
                warn!(
                    miner_uid = miner_uid,
                    node_id = node_id,
                    error = %err,
                    "Failed to execute collateral slash"
                );
            }
        }

        // Refresh ban metric after recording misbehaviour
        if let Err(err) = self
            .compute_current_ban(miner_uid, node_id)
            .await
            .map(|_| ())
        {
            warn!(
                miner_uid = miner_uid,
                node_id = node_id,
                error = %err,
                "Failed to refresh ban metric after logging misbehaviour"
            );
        }

        Ok(())
    }

    async fn try_execute_slash(
        &self,
        miner_uid: u16,
        node_id: &str,
        type_of_misbehaviour: MisbehaviourType,
        details: &str,
    ) -> Result<()> {
        let slash_executor = match &self.slash_executor {
            Some(exec) => exec.clone(),
            None => return Ok(()),
        };
        let miner_hotkey = self
            .persistence
            .get_miner_hotkey_by_uid(miner_uid)
            .await?
            .ok_or_else(|| anyhow::anyhow!("miner hotkey not found"))?;
        let rental_id = extract_rental_id(details).unwrap_or_else(|| "unknown".to_string());
        let validator_hotkey = self
            .validator_hotkey
            .clone()
            .unwrap_or_else(|| "unknown-validator".to_string());

        slash_executor
            .execute_slash(
                &miner_hotkey,
                node_id,
                type_of_misbehaviour.as_str(),
                details,
                &validator_hotkey,
                &rental_id,
            )
            .await?;

        if slash_executor.is_shadow_mode() {
            return Ok(());
        }

        info!(miner_uid, node_id, "Slash executed for miner");

        Ok(())
    }

    /// Check if a node is currently banned
    pub async fn is_executor_banned(&self, miner_uid: u16, node_id: &str) -> Result<bool> {
        let status = self.compute_current_ban(miner_uid, node_id).await?;

        if let (Some(ban_expiry), Some(ban_trigger)) = (&status.ban_expiry, &status.ban_trigger) {
            debug!(
                miner_uid = miner_uid,
                node_id = node_id,
                ban_trigger = %ban_trigger,
                ban_expiry = %ban_expiry,
                offense_count = status.offense_count,
                "Node is currently banned"
            );
        }

        Ok(status.ban_expiry.is_some())
    }

    /// Get ban expiry time for a node
    pub async fn get_ban_expiry(
        &self,
        miner_uid: u16,
        node_id: &str,
    ) -> Result<Option<DateTime<Utc>>> {
        let status = self.compute_current_ban(miner_uid, node_id).await?;
        Ok(status.ban_expiry)
    }

    async fn compute_current_ban(&self, miner_uid: u16, node_id: &str) -> Result<BanComputation> {
        let logs = self
            .get_recent_misbehaviours(miner_uid, node_id, Duration::days(7))
            .await?;

        let offense_count = logs.len();

        if offense_count == 0 {
            self.record_ban_metric(miner_uid, node_id, None);
            return Ok(BanComputation {
                ban_expiry: None,
                ban_trigger: None,
                offense_count,
            });
        }

        let ban_trigger = self.find_ban_trigger_timestamp(&logs);

        let ban_expiry = ban_trigger.and_then(|trigger_time| {
            let ban_duration = self.calculate_ban_duration(offense_count);
            let expiry = trigger_time + ban_duration;

            if Utc::now() < expiry {
                Some(expiry)
            } else {
                None
            }
        });

        self.record_ban_metric(miner_uid, node_id, ban_expiry);

        Ok(BanComputation {
            ban_expiry,
            ban_trigger,
            offense_count,
        })
    }

    fn record_ban_metric(&self, miner_uid: u16, node_id: &str, ban_expiry: Option<DateTime<Utc>>) {
        if let Some(metrics) = &self.metrics {
            metrics.record_node_ban_till(node_id, miner_uid, ban_expiry);
        }
    }

    /// Check if ban should be triggered based on recent misbehaviours
    async fn check_ban_trigger(&self, miner_uid: u16, node_id: &str) -> Result<bool> {
        let logs = self
            .persistence
            .get_misbehaviour_logs(miner_uid, node_id, Duration::days(1))
            .await?;

        // Trigger ban if 3 or more misbehaviours within 1 day
        Ok(logs.len() >= 3)
    }

    /// Get recent misbehaviours within a time window
    async fn get_recent_misbehaviours(
        &self,
        miner_uid: u16,
        node_id: &str,
        window: Duration,
    ) -> Result<Vec<MisbehaviourLog>> {
        self.persistence
            .get_misbehaviour_logs(miner_uid, node_id, window)
            .await
    }

    /// Find the latest timestamp where a ban was triggered
    ///
    /// A ban is triggered when there are 3+ misbehaviours within any 1-day sliding window.
    /// Returns the timestamp of the later misbehaviour that triggered the ban.
    fn find_ban_trigger_timestamp(&self, logs: &[MisbehaviourLog]) -> Option<DateTime<Utc>> {
        if logs.len() < 3 {
            return None;
        }

        // Sort logs by timestamp (oldest to newest)
        let mut sorted_logs: Vec<&MisbehaviourLog> = logs.iter().collect();
        sorted_logs.sort_by_key(|log| log.recorded_at);

        let mut latest_trigger: Option<DateTime<Utc>> = None;

        // Check each log as a potential trigger point
        for i in 2..sorted_logs.len() {
            let current_log = sorted_logs[i];
            let one_day_before = current_log.recorded_at - Duration::days(1);

            // Count how many logs fall within the 1-day window before this log
            let failures_in_window = sorted_logs
                .iter()
                .filter(|log| {
                    log.recorded_at >= one_day_before && log.recorded_at <= current_log.recorded_at
                })
                .count();

            // If this timestamp triggers a ban (3+ failures in window), update latest trigger
            if failures_in_window >= 3 {
                latest_trigger = Some(current_log.recorded_at);
            }
        }

        latest_trigger
    }

    /// Calculate ban duration based on offense count within 7 days
    ///
    /// Ban duration progression:
    /// - 1st offense: 1 hour
    /// - 2nd offense: 2 hours
    /// - 3rd offense: 4 hours
    /// - 4th offense: 8 hours
    /// - 5th+ offense: 24 hours (max)
    fn calculate_ban_duration(&self, offense_count: usize) -> Duration {
        match offense_count {
            0 => Duration::hours(0),
            1 => Duration::hours(1),
            2 => Duration::hours(2),
            3 => Duration::hours(4),
            4 => Duration::hours(8),
            _ => Duration::hours(24), // Max ban duration
        }
    }

    /// Create a JSON string with rental failure details
    pub fn create_rental_failure_details(
        rental_id: &str,
        executor_id: &str,
        error: &str,
        ssh_details: Option<&str>,
    ) -> String {
        json!({
            "rental_id": rental_id,
            "executor_id": executor_id,
            "error": error,
            "ssh_details": ssh_details,
            "timestamp": Utc::now().to_rfc3339(),
        })
        .to_string()
    }

    /// Create a JSON string with health check failure details
    pub fn create_health_failure_details(
        rental_id: &str,
        executor_id: &str,
        container_id: &str,
        error: &str,
    ) -> String {
        json!({
            "rental_id": rental_id,
            "executor_id": executor_id,
            "container_id": container_id,
            "error": error,
            "timestamp": Utc::now().to_rfc3339(),
        })
        .to_string()
    }

    /// Create a JSON string for deployment failure details
    pub fn create_deployment_failure_details(error_message: &str) -> String {
        json!({
            "reason": "deployment_failed",
            "error": error_message,
            "timestamp": Utc::now().to_rfc3339(),
        })
        .to_string()
    }

    /// Create a JSON string for rejection details
    pub fn create_rejection_details(rejection_reason: &str) -> String {
        json!({
            "reason": "rental_rejected",
            "rejection_reason": rejection_reason,
            "timestamp": Utc::now().to_rfc3339(),
        })
        .to_string()
    }

    /// Create a JSON string for health check failure details
    pub fn create_health_check_failure_details(
        container_id: &str,
        rental_state: &str,
        error_message: &str,
    ) -> String {
        json!({
            "reason": "health_check_failed",
            "container_id": container_id,
            "rental_state": rental_state,
            "error": error_message,
            "timestamp": Utc::now().to_rfc3339(),
        })
        .to_string()
    }
}

fn extract_rental_id(details: &str) -> Option<String> {
    // TODO: Fall back to other detail fields if rental_id format changes.
    let parsed: serde_json::Value = serde_json::from_str(details).ok()?;
    parsed
        .get("rental_id")
        .and_then(|value| value.as_str())
        .map(|value| value.to_string())
}

struct BanComputation {
    ban_expiry: Option<DateTime<Utc>>,
    ban_trigger: Option<DateTime<Utc>>,
    offense_count: usize,
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::persistence::entities::{MisbehaviourLog, MisbehaviourType};
    use chrono::{Duration, Utc};

    fn create_test_log(
        miner_uid: u16,
        node_id: &str,
        recorded_at: DateTime<Utc>,
    ) -> MisbehaviourLog {
        let now = Utc::now();
        MisbehaviourLog {
            miner_uid,
            node_id: node_id.to_string(),
            endpoint_executor: "test-endpoint".to_string(),
            type_of_misbehaviour: MisbehaviourType::DeploymentFailed,
            details: "test details".to_string(),
            recorded_at,
            created_at: now,
            updated_at: now,
        }
    }

    // Helper function to test ban trigger logic without needing persistence
    fn test_find_ban_trigger(logs: &[MisbehaviourLog]) -> Option<DateTime<Utc>> {
        if logs.len() < 3 {
            return None;
        }

        let mut sorted_logs: Vec<&MisbehaviourLog> = logs.iter().collect();
        sorted_logs.sort_by_key(|log| log.recorded_at);

        let mut latest_trigger: Option<DateTime<Utc>> = None;

        for i in 2..sorted_logs.len() {
            let current_log = sorted_logs[i];
            let one_day_before = current_log.recorded_at - Duration::days(1);

            let failures_in_window = sorted_logs
                .iter()
                .filter(|log| {
                    log.recorded_at >= one_day_before && log.recorded_at <= current_log.recorded_at
                })
                .count();

            if failures_in_window >= 3 {
                latest_trigger = Some(current_log.recorded_at);
            }
        }

        latest_trigger
    }

    // Helper function to test ban duration calculation
    fn test_calculate_duration(offense_count: usize) -> Duration {
        match offense_count {
            0 => Duration::hours(0),
            1 => Duration::hours(1),
            2 => Duration::hours(2),
            3 => Duration::hours(4),
            4 => Duration::hours(8),
            _ => Duration::hours(24), // Max ban duration
        }
    }

    #[test]
    fn test_find_ban_trigger_no_logs() {
        let logs = vec![];
        assert_eq!(test_find_ban_trigger(&logs), None);
    }

    #[test]
    fn test_find_ban_trigger_single_log() {
        let now = Utc::now();
        let logs = vec![create_test_log(1, "node1", now)];
        assert_eq!(test_find_ban_trigger(&logs), None);
    }

    #[test]
    fn test_find_ban_trigger_two_logs_within_day() {
        let now = Utc::now();
        let logs = vec![
            create_test_log(1, "node1", now - Duration::hours(6)),
            create_test_log(1, "node1", now),
        ];
        // Should NOT trigger ban — need 3+ within 1 day
        assert_eq!(test_find_ban_trigger(&logs), None);
    }

    #[test]
    fn test_find_ban_trigger_three_logs_within_day() {
        let now = Utc::now();
        let logs = vec![
            create_test_log(1, "node1", now - Duration::hours(12)),
            create_test_log(1, "node1", now - Duration::hours(6)),
            create_test_log(1, "node1", now),
        ];
        // Should trigger ban at the third log timestamp
        assert_eq!(test_find_ban_trigger(&logs), Some(now));
    }

    #[test]
    fn test_find_ban_trigger_three_logs_outside_day() {
        let now = Utc::now();
        let logs = vec![
            create_test_log(1, "node1", now - Duration::days(3)),
            create_test_log(1, "node1", now - Duration::days(2)),
            create_test_log(1, "node1", now),
        ];
        // Should not trigger ban as only 1 log in the 1-day window ending at `now`
        assert_eq!(test_find_ban_trigger(&logs), None);
    }

    #[test]
    fn test_find_ban_trigger_multiple_triggers() {
        let now = Utc::now();
        let logs = vec![
            create_test_log(1, "node1", now - Duration::hours(20)),
            create_test_log(1, "node1", now - Duration::hours(10)),
            create_test_log(1, "node1", now - Duration::hours(5)),
            create_test_log(1, "node1", now - Duration::hours(1)),
        ];
        // Should return the latest trigger (now - 1 hour)
        assert_eq!(test_find_ban_trigger(&logs), Some(now - Duration::hours(1)));
    }

    #[test]
    fn test_find_ban_trigger_sliding_window() {
        let now = Utc::now();
        // Four logs spread across time; last three within 1 day
        let logs = vec![
            create_test_log(1, "node1", now - Duration::days(2)),
            create_test_log(1, "node1", now - Duration::hours(20)),
            create_test_log(1, "node1", now - Duration::hours(10)),
            create_test_log(1, "node1", now),
        ];
        // The last three logs form a trigger (all within 1 day)
        assert_eq!(test_find_ban_trigger(&logs), Some(now));
    }

    #[test]
    fn test_calculate_ban_duration_progression() {
        assert_eq!(test_calculate_duration(0), Duration::hours(0));
        assert_eq!(test_calculate_duration(1), Duration::hours(1));
        assert_eq!(test_calculate_duration(2), Duration::hours(2));
        assert_eq!(test_calculate_duration(3), Duration::hours(4));
        assert_eq!(test_calculate_duration(4), Duration::hours(8));
        assert_eq!(test_calculate_duration(5), Duration::hours(24));
        assert_eq!(test_calculate_duration(10), Duration::hours(24)); // Max
    }

    #[tokio::test]
    async fn test_ban_persists_after_day_window() {
        // This test verifies the ban persists even after the 1-day window
        // no longer contains 3 failures.

        use crate::persistence::SimplePersistence;
        use std::sync::Arc;

        // Create in-memory database for testing
        let persistence = Arc::new(
            SimplePersistence::new(":memory:", "test_hotkey".to_string())
                .await
                .unwrap(),
        );
        persistence.run_migrations().await.unwrap();

        let ban_manager = BanManager::new(persistence.clone(), None, None, None);
        let miner_uid = 1;
        let node_id = "node1";

        // Insert three misbehaviours within 1 day (triggers ban)
        let now = Utc::now();
        let log1 = MisbehaviourLog {
            miner_uid,
            node_id: node_id.to_string(),
            endpoint_executor: "endpoint1".to_string(),
            type_of_misbehaviour: MisbehaviourType::DeploymentFailed,
            details: "failure 1".to_string(),
            recorded_at: now - Duration::hours(20),
            created_at: now,
            updated_at: now,
        };
        let log2 = MisbehaviourLog {
            miner_uid,
            node_id: node_id.to_string(),
            endpoint_executor: "endpoint1".to_string(),
            type_of_misbehaviour: MisbehaviourType::DeploymentFailed,
            details: "failure 2".to_string(),
            recorded_at: now - Duration::hours(10),
            created_at: now,
            updated_at: now,
        };
        let log3 = MisbehaviourLog {
            miner_uid,
            node_id: node_id.to_string(),
            endpoint_executor: "endpoint1".to_string(),
            type_of_misbehaviour: MisbehaviourType::DeploymentFailed,
            details: "failure 3".to_string(),
            recorded_at: now - Duration::hours(2),
            created_at: now,
            updated_at: now,
        };

        persistence.insert_misbehaviour_log(&log1).await.unwrap();
        persistence.insert_misbehaviour_log(&log2).await.unwrap();
        persistence.insert_misbehaviour_log(&log3).await.unwrap();

        // Since we have 3 offenses, ban duration is 4 hours
        // Ban was triggered at log3.recorded_at (2 hours ago)
        // Ban should expire at log3.recorded_at + 4 hours = now + 2 hours
        // Ban should still be active

        let is_banned = ban_manager
            .is_executor_banned(miner_uid, node_id)
            .await
            .unwrap();

        assert!(
            is_banned,
            "Node should still be banned (ban duration is 4 hours, only 2 hours elapsed)"
        );

        // Check ban expiry is correct
        let ban_expiry = ban_manager
            .get_ban_expiry(miner_uid, node_id)
            .await
            .unwrap();
        assert!(ban_expiry.is_some(), "Ban expiry should be set");

        let expected_expiry = log3.recorded_at + Duration::hours(4);
        let actual_expiry = ban_expiry.unwrap();

        // Allow small time difference for test execution
        let diff = (expected_expiry - actual_expiry).num_seconds().abs();
        assert!(
            diff < 5,
            "Ban expiry should be approximately 4 hours from trigger time"
        );
    }

    #[tokio::test]
    async fn test_ban_expires_after_duration() {
        use crate::persistence::SimplePersistence;
        use std::sync::Arc;

        let persistence = Arc::new(
            SimplePersistence::new(":memory:", "test_hotkey".to_string())
                .await
                .unwrap(),
        );
        persistence.run_migrations().await.unwrap();

        let ban_manager = BanManager::new(persistence.clone(), None, None, None);
        let miner_uid = 1;
        let node_id = "node1";

        // Insert three old misbehaviours that should have expired
        let now = Utc::now();
        let log1 = MisbehaviourLog {
            miner_uid,
            node_id: node_id.to_string(),
            endpoint_executor: "endpoint1".to_string(),
            type_of_misbehaviour: MisbehaviourType::DeploymentFailed,
            details: "failure 1".to_string(),
            recorded_at: now - Duration::hours(6),
            created_at: now,
            updated_at: now,
        };
        let log2 = MisbehaviourLog {
            miner_uid,
            node_id: node_id.to_string(),
            endpoint_executor: "endpoint1".to_string(),
            type_of_misbehaviour: MisbehaviourType::DeploymentFailed,
            details: "failure 2".to_string(),
            recorded_at: now - Duration::hours(5) + Duration::minutes(30),
            created_at: now,
            updated_at: now,
        };
        let log3 = MisbehaviourLog {
            miner_uid,
            node_id: node_id.to_string(),
            endpoint_executor: "endpoint1".to_string(),
            type_of_misbehaviour: MisbehaviourType::DeploymentFailed,
            details: "failure 3".to_string(),
            recorded_at: now - Duration::hours(5),
            created_at: now,
            updated_at: now,
        };

        persistence.insert_misbehaviour_log(&log1).await.unwrap();
        persistence.insert_misbehaviour_log(&log2).await.unwrap();
        persistence.insert_misbehaviour_log(&log3).await.unwrap();

        // Ban duration for 3 offenses is 4 hours
        // Ban was triggered 5 hours ago, so it should have expired

        let is_banned = ban_manager
            .is_executor_banned(miner_uid, node_id)
            .await
            .unwrap();
        assert!(!is_banned, "Ban should have expired after 4 hours");

        let ban_expiry = ban_manager
            .get_ban_expiry(miner_uid, node_id)
            .await
            .unwrap();
        assert!(
            ban_expiry.is_none(),
            "No ban expiry should be returned for expired ban"
        );
    }

    #[tokio::test]
    async fn test_progressive_ban_durations() {
        use crate::persistence::SimplePersistence;
        use std::sync::Arc;

        let persistence = Arc::new(
            SimplePersistence::new(":memory:", "test_hotkey".to_string())
                .await
                .unwrap(),
        );
        persistence.run_migrations().await.unwrap();

        let ban_manager = BanManager::new(persistence.clone(), None, None, None);
        let miner_uid = 1;
        let node_id = "node1";

        // Insert 5 misbehaviours over 7 days, but no 3 in a single 1-day window yet
        let now = Utc::now();
        let logs = vec![
            create_test_log(miner_uid, node_id, now - Duration::days(6)),
            create_test_log(miner_uid, node_id, now - Duration::days(5)),
            create_test_log(miner_uid, node_id, now - Duration::days(3)),
            create_test_log(miner_uid, node_id, now - Duration::days(2)),
            // Recent log, but only 1 within the last day
            create_test_log(miner_uid, node_id, now - Duration::minutes(30)),
        ];

        for log in &logs {
            persistence.insert_misbehaviour_log(log).await.unwrap();
        }

        // With 5 offenses but none forming 3-in-1-day, should not be banned
        let is_banned = ban_manager
            .is_executor_banned(miner_uid, node_id)
            .await
            .unwrap();
        assert!(!is_banned, "Should not be banned as no 3-in-1-day trigger");

        // Add two more logs within 1 day to trigger ban (3 in last day)
        let trigger_log1 = create_test_log(miner_uid, node_id, now - Duration::minutes(10));
        let trigger_log2 = create_test_log(miner_uid, node_id, now);
        persistence
            .insert_misbehaviour_log(&trigger_log1)
            .await
            .unwrap();
        persistence
            .insert_misbehaviour_log(&trigger_log2)
            .await
            .unwrap();

        let is_banned = ban_manager
            .is_executor_banned(miner_uid, node_id)
            .await
            .unwrap();
        assert!(
            is_banned,
            "Should be banned with 7 offenses and recent trigger"
        );

        let ban_expiry = ban_manager
            .get_ban_expiry(miner_uid, node_id)
            .await
            .unwrap();
        assert!(ban_expiry.is_some());

        // With 7 offenses, duration should be 24 hours (max)
        let expected_expiry = now + Duration::hours(24);
        let actual_expiry = ban_expiry.unwrap();
        let diff = (expected_expiry - actual_expiry).num_seconds().abs();
        assert!(diff < 5, "Ban should be 24 hours for 7 offenses");
    }
}
