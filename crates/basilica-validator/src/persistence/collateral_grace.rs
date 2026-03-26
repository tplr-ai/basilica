use crate::persistence::SimplePersistence;
use anyhow::Result;
use chrono::{DateTime, Duration, Utc};
use sqlx::Row;
use std::collections::HashSet;

impl SimplePersistence {
    pub async fn get_undercollateralized_since(
        &self,
        hotkey: &str,
        node_id: &str,
    ) -> Result<Option<DateTime<Utc>>> {
        let row = sqlx::query(
            "SELECT undercollateralized_since FROM collateral_grace_periods WHERE hotkey = ? AND node_id = ?",
        )
        .bind(hotkey)
        .bind(node_id)
        .fetch_optional(self.pool())
        .await?;

        if let Some(row) = row {
            let since: String = row.get(0);
            let since = DateTime::parse_from_rfc3339(&since)
                .map_err(|e| anyhow::anyhow!("Invalid undercollateralized_since: {e}"))?
                .with_timezone(&Utc);
            Ok(Some(since))
        } else {
            Ok(None)
        }
    }

    pub async fn mark_undercollateralized(
        &self,
        hotkey: &str,
        node_id: &str,
        since: DateTime<Utc>,
    ) -> Result<()> {
        let now = Utc::now().to_rfc3339();
        sqlx::query(
            r#"
            INSERT INTO collateral_grace_periods (hotkey, node_id, undercollateralized_since, updated_at)
            VALUES (?, ?, ?, ?)
            ON CONFLICT(hotkey, node_id)
            DO UPDATE SET undercollateralized_since = excluded.undercollateralized_since, updated_at = excluded.updated_at
            "#,
        )
        .bind(hotkey)
        .bind(node_id)
        .bind(since.to_rfc3339())
        .bind(now)
        .execute(self.pool())
        .await?;
        Ok(())
    }

    pub async fn clear_undercollateralized(&self, hotkey: &str, node_id: &str) -> Result<()> {
        sqlx::query("DELETE FROM collateral_grace_periods WHERE hotkey = ? AND node_id = ?")
            .bind(hotkey)
            .bind(node_id)
            .execute(self.pool())
            .await?;
        Ok(())
    }

    pub async fn get_excluded_nodes(
        &self,
        grace_period: Duration,
    ) -> Result<HashSet<(String, String)>> {
        let rows = sqlx::query(
            "SELECT hotkey, node_id, undercollateralized_since FROM collateral_grace_periods",
        )
        .fetch_all(self.pool())
        .await?;

        let cutoff = Utc::now() - grace_period;
        let mut excluded = HashSet::new();
        for row in rows {
            let hotkey: String = row.get(0);
            let node_id: String = row.get(1);
            let since_raw: String = row.get(2);
            let since = DateTime::parse_from_rfc3339(&since_raw)
                .map(|dt| dt.with_timezone(&Utc))
                .ok();
            if let Some(since) = since {
                if since <= cutoff {
                    excluded.insert((hotkey, node_id));
                }
            }
        }

        Ok(excluded)
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::persistence::SimplePersistence;
    use chrono::Duration;

    #[tokio::test]
    async fn test_grace_period_roundtrip() {
        let persistence = SimplePersistence::for_testing().await.unwrap();
        let hotkey = "hotkey-1";
        let node_id = "node-1";
        let since = Utc::now() - Duration::minutes(5);

        persistence
            .mark_undercollateralized(hotkey, node_id, since)
            .await
            .unwrap();

        let fetched = persistence
            .get_undercollateralized_since(hotkey, node_id)
            .await
            .unwrap();
        assert!(fetched.is_some());
        assert_eq!(fetched.unwrap().timestamp(), since.timestamp());

        persistence
            .clear_undercollateralized(hotkey, node_id)
            .await
            .unwrap();
        let cleared = persistence
            .get_undercollateralized_since(hotkey, node_id)
            .await
            .unwrap();
        assert!(cleared.is_none());
    }

    #[tokio::test]
    async fn test_get_excluded_nodes_filters_by_grace_period() {
        let persistence = SimplePersistence::for_testing().await.unwrap();
        let now = Utc::now();

        let expired_hotkey = "hk-expired";
        let expired_node = "node-expired";
        persistence
            .mark_undercollateralized(expired_hotkey, expired_node, now - Duration::hours(25))
            .await
            .unwrap();

        let active_hotkey = "hk-active";
        let active_node = "node-active";
        persistence
            .mark_undercollateralized(active_hotkey, active_node, now - Duration::hours(2))
            .await
            .unwrap();

        let excluded = persistence
            .get_excluded_nodes(Duration::hours(24))
            .await
            .unwrap();

        assert!(excluded.contains(&(expired_hotkey.to_string(), expired_node.to_string())));
        assert!(!excluded.contains(&(active_hotkey.to_string(), active_node.to_string())));
    }
}
