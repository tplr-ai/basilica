use crate::persistence::SimplePersistence;
use anyhow::Result;
use chrono::{DateTime, Duration, Utc};
use std::sync::Arc;

#[derive(Clone)]
pub struct GracePeriodTracker {
    persistence: Arc<SimplePersistence>,
    grace_period: Duration,
}

impl GracePeriodTracker {
    pub fn new(persistence: Arc<SimplePersistence>, grace_period: Duration) -> Self {
        Self {
            persistence,
            grace_period,
        }
    }

    pub async fn mark_undercollateralized(&self, hotkey: &str, node_id: &str) -> Result<()> {
        let now = Utc::now();
        self.persistence
            .mark_undercollateralized(hotkey, node_id, now)
            .await
    }

    pub async fn clear_undercollateralized(&self, hotkey: &str, node_id: &str) -> Result<()> {
        self.persistence
            .clear_undercollateralized(hotkey, node_id)
            .await
    }

    pub async fn get_grace_remaining(
        &self,
        hotkey: &str,
        node_id: &str,
    ) -> Result<Option<Duration>> {
        let since = self
            .persistence
            .get_undercollateralized_since(hotkey, node_id)
            .await?;
        let since = match since {
            Some(since) => since,
            None => return Ok(None),
        };
        let elapsed = Utc::now() - since;
        let remaining = self.grace_period - elapsed;
        if remaining <= Duration::zero() {
            Ok(Some(Duration::zero()))
        } else {
            Ok(Some(remaining))
        }
    }

    pub async fn is_grace_expired(&self, hotkey: &str, node_id: &str) -> Result<bool> {
        let remaining = self.get_grace_remaining(hotkey, node_id).await?;
        Ok(matches!(remaining, Some(r) if r <= Duration::zero()))
    }

    pub async fn force_exclude(&self, hotkey: &str, node_id: &str) -> Result<()> {
        let since = Utc::now() - self.grace_period - Duration::seconds(1);
        self.persistence
            .mark_undercollateralized(hotkey, node_id, since)
            .await
    }

    pub async fn get_since(&self, hotkey: &str, node_id: &str) -> Result<Option<DateTime<Utc>>> {
        self.persistence
            .get_undercollateralized_since(hotkey, node_id)
            .await
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[tokio::test]
    async fn test_grace_period_remaining() {
        let persistence = Arc::new(SimplePersistence::for_testing().await.unwrap());
        let tracker = GracePeriodTracker::new(persistence.clone(), Duration::hours(24));
        let hotkey = "hk";
        let node_id = "node";

        tracker
            .mark_undercollateralized(hotkey, node_id)
            .await
            .unwrap();
        let remaining = tracker.get_grace_remaining(hotkey, node_id).await.unwrap();
        assert!(remaining.is_some());
        assert!(remaining.unwrap() <= Duration::hours(24));
    }

    #[tokio::test]
    async fn test_force_exclude_expired() {
        let persistence = Arc::new(SimplePersistence::for_testing().await.unwrap());
        let tracker = GracePeriodTracker::new(persistence.clone(), Duration::hours(1));
        let hotkey = "hk";
        let node_id = "node";

        tracker.force_exclude(hotkey, node_id).await.unwrap();
        let expired = tracker.is_grace_expired(hotkey, node_id).await.unwrap();
        assert!(expired);
    }
}
