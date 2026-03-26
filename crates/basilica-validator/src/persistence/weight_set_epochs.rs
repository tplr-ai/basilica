use crate::persistence::SimplePersistence;
use anyhow::{anyhow, Result};
use chrono::{DateTime, Utc};
use sqlx::Row;
use std::sync::Arc;

#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum WeightSetEpochStatus {
    Pending,
    Success,
}

impl WeightSetEpochStatus {
    fn from_str(value: &str) -> Option<Self> {
        match value {
            "pending" => Some(Self::Pending),
            "success" => Some(Self::Success),
            _ => None,
        }
    }
}

#[derive(Debug, Clone)]
pub struct WeightSetEpoch {
    pub id: i64,
    pub netuid: u16,
    pub period_start: DateTime<Utc>,
    pub period_end: DateTime<Utc>,
    pub status: WeightSetEpochStatus,
    pub attempts: i64,
    pub created_at: DateTime<Utc>,
    pub updated_at: DateTime<Utc>,
}

#[derive(Clone)]
pub struct WeightSetEpochRepository {
    persistence: Arc<SimplePersistence>,
}

impl WeightSetEpochRepository {
    pub fn new(persistence: Arc<SimplePersistence>) -> Self {
        Self { persistence }
    }

    fn row_to_epoch(row: &sqlx::sqlite::SqliteRow) -> Result<WeightSetEpoch> {
        let status_raw: String = row.get("status");
        let status = WeightSetEpochStatus::from_str(&status_raw)
            .ok_or_else(|| anyhow!("Invalid weight_set_epochs status value: {}", status_raw))?;

        let period_start_ts: i64 = row.get("period_start");
        let period_end_ts: i64 = row.get("period_end");
        let created_at_ts: i64 = row.get("created_at");
        let updated_at_ts: i64 = row.get("updated_at");

        let period_start = DateTime::<Utc>::from_timestamp(period_start_ts, 0)
            .ok_or_else(|| anyhow!("Invalid period_start timestamp: {}", period_start_ts))?;
        let period_end = DateTime::<Utc>::from_timestamp(period_end_ts, 0)
            .ok_or_else(|| anyhow!("Invalid period_end timestamp: {}", period_end_ts))?;
        let created_at = DateTime::<Utc>::from_timestamp(created_at_ts, 0)
            .ok_or_else(|| anyhow!("Invalid created_at timestamp: {}", created_at_ts))?;
        let updated_at = DateTime::<Utc>::from_timestamp(updated_at_ts, 0)
            .ok_or_else(|| anyhow!("Invalid updated_at timestamp: {}", updated_at_ts))?;

        Ok(WeightSetEpoch {
            id: row.get("id"),
            netuid: row.get::<i64, _>("netuid") as u16,
            period_start,
            period_end,
            status,
            attempts: row.get("attempts"),
            created_at,
            updated_at,
        })
    }

    pub async fn get_pending_epoch(&self, netuid: u16) -> Result<Option<WeightSetEpoch>> {
        let row = sqlx::query(
            r#"
            SELECT id, netuid, period_start, period_end, status, attempts, created_at, updated_at
            FROM weight_set_epochs
            WHERE netuid = ? AND status = 'pending'
            ORDER BY id DESC
            LIMIT 1
            "#,
        )
        .bind(netuid as i64)
        .fetch_optional(self.persistence.pool())
        .await?;

        match row {
            Some(row) => Ok(Some(Self::row_to_epoch(&row)?)),
            None => Ok(None),
        }
    }

    pub async fn get_last_success_end(&self, netuid: u16) -> Result<Option<DateTime<Utc>>> {
        let row = sqlx::query(
            r#"
            SELECT period_end
            FROM weight_set_epochs
            WHERE netuid = ? AND status = 'success'
            ORDER BY period_end DESC
            LIMIT 1
            "#,
        )
        .bind(netuid as i64)
        .fetch_optional(self.persistence.pool())
        .await?;

        let Some(row) = row else {
            return Ok(None);
        };

        let period_end_ts: i64 = row.get("period_end");
        let period_end = DateTime::<Utc>::from_timestamp(period_end_ts, 0)
            .ok_or_else(|| anyhow!("Invalid period_end timestamp: {}", period_end_ts))?;

        Ok(Some(period_end))
    }

    pub async fn create_epoch(
        &self,
        netuid: u16,
        period_start: DateTime<Utc>,
        period_end: DateTime<Utc>,
    ) -> Result<WeightSetEpoch> {
        let now = Utc::now().timestamp();
        let result = sqlx::query(
            r#"
            INSERT INTO weight_set_epochs (
                netuid, period_start, period_end, status, attempts, created_at, updated_at
            ) VALUES (?, ?, ?, 'pending', 0, ?, ?)
            "#,
        )
        .bind(netuid as i64)
        .bind(period_start.timestamp())
        .bind(period_end.timestamp())
        .bind(now)
        .bind(now)
        .execute(self.persistence.pool())
        .await?;

        let id = result.last_insert_rowid();
        self.get_epoch_by_id(id)
            .await?
            .ok_or_else(|| anyhow!("Failed to fetch created weight_set_epoch with id {}", id))
    }

    pub async fn increment_attempts(&self, id: i64) -> Result<()> {
        let now = Utc::now().timestamp();
        sqlx::query(
            r#"
            UPDATE weight_set_epochs
            SET attempts = attempts + 1, updated_at = ?
            WHERE id = ?
            "#,
        )
        .bind(now)
        .bind(id)
        .execute(self.persistence.pool())
        .await?;
        Ok(())
    }

    pub async fn mark_success(&self, id: i64) -> Result<()> {
        let now = Utc::now().timestamp();
        sqlx::query(
            r#"
            UPDATE weight_set_epochs
            SET status = 'success', updated_at = ?
            WHERE id = ?
            "#,
        )
        .bind(now)
        .bind(id)
        .execute(self.persistence.pool())
        .await?;
        Ok(())
    }

    async fn get_epoch_by_id(&self, id: i64) -> Result<Option<WeightSetEpoch>> {
        let row = sqlx::query(
            r#"
            SELECT id, netuid, period_start, period_end, status, attempts, created_at, updated_at
            FROM weight_set_epochs
            WHERE id = ?
            "#,
        )
        .bind(id)
        .fetch_optional(self.persistence.pool())
        .await?;

        match row {
            Some(row) => Ok(Some(Self::row_to_epoch(&row)?)),
            None => Ok(None),
        }
    }
}
