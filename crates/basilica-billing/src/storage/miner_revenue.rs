use crate::error::{BillingError, Result};
use crate::storage::rds::RdsConnection;
use async_trait::async_trait;
use chrono::{DateTime, Utc};
use rust_decimal::Decimal;
use sqlx::Row;
use std::sync::Arc;
use uuid::Uuid;

/// Domain type representing a miner revenue summary snapshot
#[derive(Debug, Clone)]
pub struct MinerRevenueSummary {
    pub id: Uuid,
    pub node_id: String,
    pub validator_id: Option<String>,
    /// Bittensor miner UID for payment reconciliation (part of grouping key with miner_hotkey)
    pub miner_uid: Option<i32>,
    /// Bittensor miner hotkey for payment reconciliation (part of grouping key with miner_uid)
    pub miner_hotkey: String,

    // Time period
    pub period_start: DateTime<Utc>,
    pub period_end: DateTime<Utc>,

    // Aggregated metrics
    pub total_rentals: i32,
    pub completed_rentals: i32,
    pub failed_rentals: i32,
    pub total_revenue: Decimal,
    pub total_hours: Decimal,

    // Computed metrics
    pub avg_hourly_rate: Option<Decimal>,
    pub avg_rental_duration_hours: Option<Decimal>,

    // Audit fields
    pub computed_at: DateTime<Utc>,
    pub computation_version: i32,
    pub created_at: DateTime<Utc>,
}

/// Filter criteria for querying miner revenue summaries
#[derive(Debug, Clone, Default)]
pub struct MinerRevenueSummaryFilter {
    pub node_ids: Option<Vec<String>>,
    pub validator_ids: Option<Vec<String>>,
    /// Filter by Bittensor miner UIDs
    pub miner_uids: Option<Vec<i32>>,
    /// Filter by Bittensor miner hotkeys
    pub miner_hotkeys: Option<Vec<String>>,
    pub period_start: Option<DateTime<Utc>>,
    pub period_end: Option<DateTime<Utc>>,
    pub computed_at: Option<DateTime<Utc>>,
    pub latest_only: bool,
    pub limit: Option<i64>,
    pub offset: Option<i64>,
}

#[async_trait]
pub trait MinerRevenueRepository: Send + Sync {
    /// Refresh miner revenue summaries for a given time period
    /// Returns the number of summary rows created
    async fn refresh_summary_for_period(
        &self,
        period_start: DateTime<Utc>,
        period_end: DateTime<Utc>,
        computation_version: i32,
    ) -> Result<i32>;

    /// Get miner revenue summaries with optional filtering
    async fn get_summaries(
        &self,
        filter: &MinerRevenueSummaryFilter,
    ) -> Result<Vec<MinerRevenueSummary>>;

    /// Get total count of summaries matching filter (for pagination)
    async fn count_summaries(&self, filter: &MinerRevenueSummaryFilter) -> Result<i64>;
}

pub struct SqlMinerRevenueRepository {
    connection: Arc<RdsConnection>,
}

impl SqlMinerRevenueRepository {
    pub fn new(connection: Arc<RdsConnection>) -> Self {
        Self { connection }
    }

    pub fn pool(&self) -> &sqlx::PgPool {
        self.connection.pool()
    }

    fn summary_from_row(row: &sqlx::postgres::PgRow) -> MinerRevenueSummary {
        MinerRevenueSummary {
            id: row.get("id"),
            node_id: row.get("node_id"),
            validator_id: row.get("validator_id"),
            miner_uid: row.get("miner_uid"),
            miner_hotkey: row.get("miner_hotkey"),
            period_start: row.get("period_start"),
            period_end: row.get("period_end"),
            total_rentals: row.get("total_rentals"),
            completed_rentals: row.get("completed_rentals"),
            failed_rentals: row.get("failed_rentals"),
            total_revenue: row.get("total_revenue"),
            total_hours: row.get("total_hours"),
            avg_hourly_rate: row.get("avg_hourly_rate"),
            avg_rental_duration_hours: row.get("avg_rental_duration_hours"),
            computed_at: row.get("computed_at"),
            computation_version: row.get("computation_version"),
            created_at: row.get("created_at"),
        }
    }
}

#[async_trait]
impl MinerRevenueRepository for SqlMinerRevenueRepository {
    async fn refresh_summary_for_period(
        &self,
        period_start: DateTime<Utc>,
        period_end: DateTime<Utc>,
        computation_version: i32,
    ) -> Result<i32> {
        let result = sqlx::query_scalar::<_, i32>(
            "SELECT billing.refresh_miner_revenue_summary($1, $2, $3)",
        )
        .bind(period_start)
        .bind(period_end)
        .bind(computation_version)
        .fetch_one(self.pool())
        .await
        .map_err(|e| BillingError::DatabaseError {
            operation: "refresh_miner_revenue_summary".to_string(),
            source: Box::new(e),
        })?;

        Ok(result)
    }

    async fn get_summaries(
        &self,
        filter: &MinerRevenueSummaryFilter,
    ) -> Result<Vec<MinerRevenueSummary>> {
        let mut query = String::from(
            r#"
            SELECT
                id, node_id, validator_id, miner_uid, miner_hotkey, period_start, period_end,
                total_rentals, completed_rentals, failed_rentals,
                total_revenue, total_hours, avg_hourly_rate,
                avg_rental_duration_hours, computed_at, computation_version, created_at
            FROM billing.miner_revenue_summary
            WHERE 1=1
            "#,
        );

        let mut conditions = Vec::new();

        // Build WHERE conditions
        if let Some(ref node_ids) = filter.node_ids {
            if !node_ids.is_empty() {
                conditions.push(format!("node_id = ANY(${})", conditions.len() + 1));
            }
        }

        if let Some(ref validator_ids) = filter.validator_ids {
            if !validator_ids.is_empty() {
                conditions.push(format!("validator_id = ANY(${})", conditions.len() + 1));
            }
        }

        if let Some(ref miner_uids) = filter.miner_uids {
            if !miner_uids.is_empty() {
                conditions.push(format!("miner_uid = ANY(${})", conditions.len() + 1));
            }
        }

        if let Some(ref miner_hotkeys) = filter.miner_hotkeys {
            if !miner_hotkeys.is_empty() {
                conditions.push(format!("miner_hotkey = ANY(${})", conditions.len() + 1));
            }
        }

        if filter.period_start.is_some() {
            conditions.push(format!("period_start >= ${}", conditions.len() + 1));
        }

        if filter.period_end.is_some() {
            conditions.push(format!("period_end <= ${}", conditions.len() + 1));
        }

        if filter.computed_at.is_some() {
            conditions.push(format!("computed_at = ${}", conditions.len() + 1));
        }

        if !conditions.is_empty() {
            query.push_str(" AND ");
            query.push_str(&conditions.join(" AND "));
        }

        // Handle latest_only flag
        if filter.latest_only {
            query.push_str(
                r#"
                AND computed_at = (
                    SELECT MAX(computed_at)
                    FROM billing.miner_revenue_summary s2
                    WHERE s2.node_id = miner_revenue_summary.node_id
                      AND (s2.validator_id = miner_revenue_summary.validator_id
                           OR (s2.validator_id IS NULL AND miner_revenue_summary.validator_id IS NULL))
                      AND s2.miner_hotkey = miner_revenue_summary.miner_hotkey
                      AND s2.miner_uid = miner_revenue_summary.miner_uid
                      AND s2.period_start = miner_revenue_summary.period_start
                      AND s2.period_end = miner_revenue_summary.period_end
                )
                "#,
            );
        }

        query.push_str(" ORDER BY computed_at DESC, period_start DESC");

        if let Some(limit) = filter.limit {
            query.push_str(&format!(" LIMIT {}", limit));
        }

        if let Some(offset) = filter.offset {
            query.push_str(&format!(" OFFSET {}", offset));
        }

        // Build and execute query
        let mut sql_query = sqlx::query(&query);

        if let Some(ref node_ids) = filter.node_ids {
            if !node_ids.is_empty() {
                sql_query = sql_query.bind(node_ids);
            }
        }

        if let Some(ref validator_ids) = filter.validator_ids {
            if !validator_ids.is_empty() {
                sql_query = sql_query.bind(validator_ids);
            }
        }

        if let Some(ref miner_uids) = filter.miner_uids {
            if !miner_uids.is_empty() {
                sql_query = sql_query.bind(miner_uids);
            }
        }

        if let Some(ref miner_hotkeys) = filter.miner_hotkeys {
            if !miner_hotkeys.is_empty() {
                sql_query = sql_query.bind(miner_hotkeys);
            }
        }

        if let Some(period_start) = filter.period_start {
            sql_query = sql_query.bind(period_start);
        }

        if let Some(period_end) = filter.period_end {
            sql_query = sql_query.bind(period_end);
        }

        if let Some(computed_at) = filter.computed_at {
            sql_query = sql_query.bind(computed_at);
        }

        let rows =
            sql_query
                .fetch_all(self.pool())
                .await
                .map_err(|e| BillingError::DatabaseError {
                    operation: "get_miner_revenue_summaries".to_string(),
                    source: Box::new(e),
                })?;

        Ok(rows.iter().map(Self::summary_from_row).collect())
    }

    async fn count_summaries(&self, filter: &MinerRevenueSummaryFilter) -> Result<i64> {
        let mut query = String::from(
            r#"
            SELECT COUNT(*)
            FROM billing.miner_revenue_summary
            WHERE 1=1
            "#,
        );

        let mut conditions = Vec::new();

        // Build WHERE conditions (same as get_summaries)
        if let Some(ref node_ids) = filter.node_ids {
            if !node_ids.is_empty() {
                conditions.push(format!("node_id = ANY(${})", conditions.len() + 1));
            }
        }

        if let Some(ref validator_ids) = filter.validator_ids {
            if !validator_ids.is_empty() {
                conditions.push(format!("validator_id = ANY(${})", conditions.len() + 1));
            }
        }

        if let Some(ref miner_uids) = filter.miner_uids {
            if !miner_uids.is_empty() {
                conditions.push(format!("miner_uid = ANY(${})", conditions.len() + 1));
            }
        }

        if let Some(ref miner_hotkeys) = filter.miner_hotkeys {
            if !miner_hotkeys.is_empty() {
                conditions.push(format!("miner_hotkey = ANY(${})", conditions.len() + 1));
            }
        }

        if filter.period_start.is_some() {
            conditions.push(format!("period_start >= ${}", conditions.len() + 1));
        }

        if filter.period_end.is_some() {
            conditions.push(format!("period_end <= ${}", conditions.len() + 1));
        }

        if filter.computed_at.is_some() {
            conditions.push(format!("computed_at = ${}", conditions.len() + 1));
        }

        if !conditions.is_empty() {
            query.push_str(" AND ");
            query.push_str(&conditions.join(" AND "));
        }

        // Handle latest_only flag
        if filter.latest_only {
            query.push_str(
                r#"
                AND computed_at = (
                    SELECT MAX(computed_at)
                    FROM billing.miner_revenue_summary s2
                    WHERE s2.node_id = miner_revenue_summary.node_id
                      AND (s2.validator_id = miner_revenue_summary.validator_id
                           OR (s2.validator_id IS NULL AND miner_revenue_summary.validator_id IS NULL))
                      AND s2.miner_hotkey = miner_revenue_summary.miner_hotkey
                      AND s2.miner_uid = miner_revenue_summary.miner_uid
                      AND s2.period_start = miner_revenue_summary.period_start
                      AND s2.period_end = miner_revenue_summary.period_end
                )
                "#,
            );
        }

        // Build and execute query
        let mut sql_query = sqlx::query_scalar(&query);

        if let Some(ref node_ids) = filter.node_ids {
            if !node_ids.is_empty() {
                sql_query = sql_query.bind(node_ids);
            }
        }

        if let Some(ref validator_ids) = filter.validator_ids {
            if !validator_ids.is_empty() {
                sql_query = sql_query.bind(validator_ids);
            }
        }

        if let Some(ref miner_uids) = filter.miner_uids {
            if !miner_uids.is_empty() {
                sql_query = sql_query.bind(miner_uids);
            }
        }

        if let Some(ref miner_hotkeys) = filter.miner_hotkeys {
            if !miner_hotkeys.is_empty() {
                sql_query = sql_query.bind(miner_hotkeys);
            }
        }

        if let Some(period_start) = filter.period_start {
            sql_query = sql_query.bind(period_start);
        }

        if let Some(period_end) = filter.period_end {
            sql_query = sql_query.bind(period_end);
        }

        if let Some(computed_at) = filter.computed_at {
            sql_query = sql_query.bind(computed_at);
        }

        let count =
            sql_query
                .fetch_one(self.pool())
                .await
                .map_err(|e| BillingError::DatabaseError {
                    operation: "count_miner_revenue_summaries".to_string(),
                    source: Box::new(e),
                })?;

        Ok(count)
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_miner_revenue_summary_filter_default() {
        let filter = MinerRevenueSummaryFilter::default();
        assert!(filter.node_ids.is_none());
        assert!(filter.validator_ids.is_none());
        assert!(filter.period_start.is_none());
        assert!(filter.period_end.is_none());
        assert!(filter.computed_at.is_none());
        assert!(!filter.latest_only);
        assert!(filter.limit.is_none());
        assert!(filter.offset.is_none());
    }

    #[test]
    fn test_miner_revenue_summary_clone() {
        let summary = MinerRevenueSummary {
            id: Uuid::new_v4(),
            node_id: "node1".to_string(),
            validator_id: Some("val1".to_string()),
            miner_uid: Some(42),
            miner_hotkey: "5GrwvaEF5zXb26Fz9rcQpDWS57CtERHpNehXCPcNoHGKutQY".to_string(),
            period_start: Utc::now(),
            period_end: Utc::now(),
            total_rentals: 10,
            completed_rentals: 8,
            failed_rentals: 2,
            total_revenue: Decimal::new(100, 0),
            total_hours: Decimal::new(50, 0),
            avg_hourly_rate: Some(Decimal::new(2, 0)),
            avg_rental_duration_hours: Some(Decimal::new(5, 0)),
            computed_at: Utc::now(),
            computation_version: 1,
            created_at: Utc::now(),
        };

        let cloned = summary.clone();
        assert_eq!(summary.id, cloned.id);
        assert_eq!(summary.node_id, cloned.node_id);
        assert_eq!(summary.miner_hotkey, cloned.miner_hotkey);
    }
}
